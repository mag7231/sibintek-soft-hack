# src/statistics.py
"""
StatisticsCollector — модуль, который помогает заполнять sqlite базу во время обработки видео.

Как пользоваться (пример в processing loop):

    db = DatabaseManager(db_path='data/app.db')
    session = db.create_video_session(video_path, start_time, fps, total_frames)

    stats = StatisticsCollector(db, session.id, fps)

    for frame_id, frame in enumerate(frames):
        # получить detections и tracks от детектора/тракера/классификатора
        detections = detector.get_detections(results)
        tracks = tracker.update(detections)  # трековые объекты со 'track_id', 'bbox', 'class_name', 'worker_class'

        timestamp = session.start_time + timedelta(seconds=frame_id / fps)
        stats.process_frame(frame_id, timestamp, detections, tracks)

    stats.finish_session(end_time)

Описание функционала:
- Для каждого кадра обновляет WorkerActivity (create_or_update_worker_activity)
- Регистрирует TrainEvent (arrival/departure) по простой стабильности
- Генерирует AttentionEvent по правилам:
    * worker (mechanic/worker/driver) вне своей рабочей зоны > idle_threshold -> low severity
    * cleaner in RepairZone > wrong_zone (medium)
    * unknown in RepairZone > unknown_in_repair (high)

- Поддерживает порогов в секундах и может сохранять bbox в файлы (и помечать в БД) при событии внимания.

"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import os

from src.database import DatabaseManager


class StatisticsCollector:
    def __init__(
        self,
        db: DatabaseManager,
        session_id: int,
        fps: float,
        idle_threshold: float = 10.0,  # seconds out-of-zone to consider idle
        wrong_zone_threshold: float = 10.0,  # seconds to trigger wrong_zone for cleaners
        unknown_in_repair_threshold: float = 10.0,  # seconds
        train_stable_frames: int = 5,
        save_crops_dir: Optional[str] = None
    ):
        self.db = db
        self.session_id = session_id
        self.fps = float(fps)
        self.frame_time = 1.0 / self.fps

        # thresholds
        self.idle_threshold = idle_threshold
        self.wrong_zone_threshold = wrong_zone_threshold
        self.unknown_in_repair_threshold = unknown_in_repair_threshold
        self.train_stable_frames = train_stable_frames

        # runtime per-track accumulators
        # track_id -> { 'last_zone': str, 'time_in_work_zone': float, 'time_out': float, 'first_frame': int, 'last_frame': int, 'worker_class': str }
        self.tracks_state: Dict[int, Dict] = {}

        # Attention state: to avoid duplicates, we remember when attention already emitted
        # keys: (track_id, event_type) -> last_emitted_frame
        self.attention_emitted: Dict[Tuple[int, str], int] = {}

        # Train state
        self.current_train_event_id: Optional[int] = None
        self.train_present_counter = 0

        # Where to save crops for later fine-tuning
        self.save_crops_dir = save_crops_dir
        if self.save_crops_dir:
            os.makedirs(self.save_crops_dir, exist_ok=True)

    def _frame_seconds(self, frame_idx: int) -> float:
        return frame_idx / self.fps

    def process_frame(self, frame_idx: int, timestamp: datetime, detections: List[Dict], tracks: List[Dict]):
        """
        Обработка одного кадра: обновление worker activities, проверки attention и поездов.

        `detections` - список словарей с keys: bbox, conf, class_name
        `tracks` - список словарей с keys: track_id, bbox, class_name (person), worker_class (может быть 'unknown'), zone_name
        """
        # 1) Обновляем train logic
        self._process_trains(frame_idx, timestamp, detections)

        # 2) Обновляем tracks (работники)
        active_track_ids = []
        for tr in tracks:
            if tr.get('class_name') != 'person':
                continue
            tid = int(tr['track_id'])
            active_track_ids.append(tid)

            worker_class = tr.get('worker_class', 'unknown')
            zone_name = tr.get('zone_name', 'out_of_zones')

            # update db record
            self.db.create_or_update_worker_activity(
                self.session_id,
                tid,
                worker_class,
                timestamp,
                frame_idx,
                zone_name,
                self.frame_time
            )

            # update local state
            state = self.tracks_state.get(tid, {
                'worker_class': worker_class,
                'first_frame': frame_idx,
                'last_frame': frame_idx,
                'time_in_work_zone': 0.0,
                'time_in_other_zones': 0.0,
                'last_zone': zone_name,
                'acc_out_of_zone': 0.0,  # accumulated out-of-zone seconds
                'acc_in_wrong_zone': 0.0
            })

            # accumulate zone times
            work_zones = self.db._get_work_zones_for_class(worker_class)
            if zone_name in work_zones:
                state['time_in_work_zone'] += self.frame_time
                state['acc_out_of_zone'] = 0.0
            else:
                state['time_in_other_zones'] += self.frame_time
                state['acc_out_of_zone'] += self.frame_time

            # special case: cleaner in RepairZone
            if worker_class == 'cleaner' and zone_name == 'RepairZone':
                state['acc_in_wrong_zone'] = state.get('acc_in_wrong_zone', 0.0) + self.frame_time
            else:
                state['acc_in_wrong_zone'] = 0.0

            # unknown in repair
            if worker_class == 'unknown' and zone_name == 'RepairZone':
                state['acc_unknown_in_repair'] = state.get('acc_unknown_in_repair', 0.0) + self.frame_time
            else:
                state['acc_unknown_in_repair'] = 0.0

            state['last_zone'] = zone_name
            state['last_frame'] = frame_idx
            state['worker_class'] = worker_class

            self.tracks_state[tid] = state

            # emit attention events if thresholds exceeded
            self._maybe_emit_attention(tid, state, timestamp, frame_idx, tr)

        # cleanup lost tracks in both DB and local state
        self._cleanup_lost_tracks(active_track_ids)

    # -------------------- Trains --------------------
    def _process_trains(self, frame_idx: int, timestamp: datetime, detections: List[Dict]):
        # consider any detection with class_name == 'train' as presence
        train_detected = any(d['class_name'] == 'train' for d in detections)

        if train_detected:
            self.train_present_counter += 1
        else:
            # if no train in this frame, decrement (but not below 0)
            self.train_present_counter = max(0, self.train_present_counter - 1)

        # if train just appeared
        if self.current_train_event_id is None and self.train_present_counter >= self.train_stable_frames:
            # create train arrival event (train_number unknown for now)
            ev = self.db.create_train_arrival(
                self.session_id,
                train_number='unknown',
                arrival_time=timestamp,
                arrival_frame=frame_idx
            )
            self.current_train_event_id = ev.id

        # if train disappeared for a while, close event
        if self.current_train_event_id is not None and self.train_present_counter == 0:
            # close event
            self.db.update_train_departure(self.current_train_event_id, timestamp, frame_idx)
            self.current_train_event_id = None

    # -------------------- Attention --------------------
    def _maybe_emit_attention(self, track_id: int, state: Dict, timestamp: datetime, frame_idx: int, track_obj: Dict):
        """
        Проверяем пороги и создаём AttentionEvent при необходимости.
        """
        worker_class = state.get('worker_class', 'unknown')

        # 1) worker outside work zone too long
        out_seconds = state.get('acc_out_of_zone', 0.0)
        if worker_class in ['mechanic', 'worker', 'driver'] and out_seconds >= self.idle_threshold:
            key = (track_id, 'idle')
            last_emit = self.attention_emitted.get(key, -999999)
            if frame_idx - last_emit > int(self.idle_threshold * self.fps):
                desc = f"{worker_class} (track {track_id}) out of work zone for {out_seconds:.1f}s"
                self.db.create_attention_event(
                    self.session_id,
                    event_type='idle',
                    severity='low',
                    track_id=track_id,
                    worker_class=worker_class,
                    timestamp=timestamp,
                    frame_number=frame_idx,
                    zone_name=state.get('last_zone', ''),
                    description=desc,
                    bbox=track_obj.get('bbox')
                )
                self.attention_emitted[key] = frame_idx

        # 2) cleaner in repair zone
        wrong_seconds = state.get('acc_in_wrong_zone', 0.0)
        if worker_class == 'cleaner' and wrong_seconds >= self.wrong_zone_threshold:
            key = (track_id, 'wrong_zone')
            last_emit = self.attention_emitted.get(key, -999999)
            if frame_idx - last_emit > int(self.wrong_zone_threshold * self.fps):
                desc = f"Cleaner (track {track_id}) in RepairZone for {wrong_seconds:.1f}s"
                self.db.create_attention_event(
                    self.session_id,
                    event_type='wrong_zone',
                    severity='medium',
                    track_id=track_id,
                    worker_class=worker_class,
                    timestamp=timestamp,
                    frame_number=frame_idx,
                    zone_name=state.get('last_zone', ''),
                    description=desc,
                    bbox=track_obj.get('bbox')
                )
                self.attention_emitted[key] = frame_idx

        # 3) unknown in repair
        unk_seconds = state.get('acc_unknown_in_repair', 0.0)
        if worker_class == 'unknown' and unk_seconds >= self.unknown_in_repair_threshold:
            key = (track_id, 'unknown_in_repair')
            last_emit = self.attention_emitted.get(key, -999999)
            if frame_idx - last_emit > int(self.unknown_in_repair_threshold * self.fps):
                desc = f"Unknown person (track {track_id}) in RepairZone for {unk_seconds:.1f}s"
                self.db.create_attention_event(
                    self.session_id,
                    event_type='unknown_in_repair',
                    severity='high',
                    track_id=track_id,
                    worker_class='unknown',
                    timestamp=timestamp,
                    frame_number=frame_idx,
                    zone_name=state.get('last_zone', ''),
                    description=desc,
                    bbox=track_obj.get('bbox')
                )
                self.attention_emitted[key] = frame_idx

    # -------------------- Cleanup --------------------
    def _cleanup_lost_tracks(self, active_track_ids: List[int]):
        # remove local state for tracks not active anymore
        to_remove = [tid for tid in list(self.tracks_state.keys()) if tid not in active_track_ids]
        for tid in to_remove:
            # keep DB records (we don't close worker activity rows explicitly)
            self.tracks_state.pop(tid, None)
            # also remove vote/attention memory
            keys_to_remove = [k for k in self.attention_emitted.keys() if k[0] == tid]
            for k in keys_to_remove:
                self.attention_emitted.pop(k, None)

    def finish_session(self, end_time: datetime):
        # Update session end time in DB
        self.db.update_video_session_end_time(self.session_id, end_time)

    # -------------------- Utility exports --------------------
    def export_session_worker_summary(self) -> List[Dict]:
        """
        Возвращает агрегированную сводку по работникам для текущей сессии.
        Каждая запись: { track_id, worker_class, total_time, time_in_work_zone, time_in_other_zones }
        """
        session_workers = self.db.session.query(self.db.session.mapper_registry.mapped[0].__class__).all()
        # NOTE: we will instead query WorkerActivity table directly
        rows = self.db.session.query(self.db.WorkerActivity).filter_by(session_id=self.session_id).all()
        summary = []
        for r in rows:
            summary.append({
                'track_id': r.track_id,
                'worker_class': r.worker_class,
                'total_time': r.total_time,
                'time_in_work_zone': r.time_in_work_zone,
                'time_in_other_zones': r.time_in_other_zones,
                'zones_visited': json.loads(r.zones_visited or '{}')
            })
        return summary
