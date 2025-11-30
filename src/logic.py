# src/logic.py

from shapely.geometry import Point, Polygon
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict


class ZoneManager:
    """Менеджер зон."""
    
    def __init__(self, zones_config, frame_width, frame_height):
        """
        Инициализирует полигоны, масштабируя их под размер кадра.
        """
        self.polygons = {}
        # Коэффициенты масштабирования
        self.scale_x = frame_width / 1000
        self.scale_y = frame_height / 1000
        
        for name, data in zones_config.items():
            # Масштабируем нормализованные координаты (0-1000) к реальным пикселям
            scaled_coords = [
                (p[0] * self.scale_x, p[1] * self.scale_y) 
                for p in data['coords']
            ]
            self.polygons[name] = Polygon(scaled_coords)

    def get_zone(self, bbox: list) -> str:
        """
        Определяет, в какой зоне находится объект.
        Используем точку "ног" человека для определения зоны.
        """
        x1, y1, x2, y2 = bbox
        
        # Точка "ног" человека (середина нижней границы BBox)
        feet_point = Point((x1 + x2) / 2, y2)
        
        for zone_name, polygon in self.polygons.items():
            if polygon.contains(feet_point):
                return zone_name
        
        return "out_of_zones"
    
    def get_polygons_for_drawing(self):
        """
        Возвращает словарь с именем зоны и координатами в формате numpy.
        """
        drawable_polygons = {}
        for name, poly in self.polygons.items():
            points = np.array(poly.exterior.coords, dtype=np.int32)
            drawable_polygons[name] = points
        return drawable_polygons


class ActivityAnalyzer:
    """
    Анализатор активности работников на основе зон и времени.
    """
    
    def __init__(self, config: dict, db_manager, fps: int):
        """
        Инициализация анализатора.
        
        Args:
            config: конфиг с правилами активности
            db_manager: DatabaseManager для записи событий
            fps: FPS видео
        """
        self.config = config
        self.db = db_manager
        self.fps = fps
        
        # Правила
        self.work_zones = config.get('work_zones', {})
        self.wrong_zone_threshold = config.get('wrong_zone_threshold', 10.0)
        self.unknown_threshold = config.get('unknown_in_repair_threshold', 15.0)
        
        # Трекинг времени в зонах
        self.track_zone_timers = defaultdict(lambda: {
            'zone': None,
            'time_in_zone': 0.0,
            'last_frame': 0
        })
        
        # Attention события (чтобы не дублировать)
        self.attention_triggered = defaultdict(set)
        
        print(f"✅ ActivityAnalyzer инициализирован")
    
    def analyze_track(
        self,
        session_id: int,
        track: dict,
        zone_name: str,
        current_time: datetime,
        frame_number: int
    ):
        """
        Анализ трека работника.
        
        Args:
            session_id: ID видео сессии
            track: информация о треке
            zone_name: текущая зона
            current_time: текущее время
            frame_number: номер кадра
        """
        track_id = track['track_id']
        worker_class = track.get('worker_class', 'unknown')
        
        # Время с последнего кадра
        timer = self.track_zone_timers[track_id]
        
        if timer['last_frame'] == 0:
            # Первый раз видим этот трек
            timer['zone'] = zone_name
            timer['last_frame'] = frame_number
            return
        
        # Вычисляем время с последнего кадра
        frames_passed = frame_number - timer['last_frame']
        time_delta = frames_passed / self.fps
        
        # Если зона не изменилась
        if timer['zone'] == zone_name:
            timer['time_in_zone'] += time_delta
        else:
            # Зона изменилась - сбрасываем таймер
            timer['zone'] = zone_name
            timer['time_in_zone'] = time_delta
        
        timer['last_frame'] = frame_number
        
        # Обновляем активность в БД
        self.db.create_or_update_worker_activity(
            session_id=session_id,
            track_id=track_id,
            worker_class=worker_class,
            current_time=current_time,
            current_frame=frame_number,
            zone_name=zone_name,
            time_delta=time_delta
        )
        
        # Проверяем attention события
        self._check_attention_events(
            session_id=session_id,
            track=track,
            zone_name=zone_name,
            time_in_zone=timer['time_in_zone'],
            current_time=current_time,
            frame_number=frame_number
        )
    
    def _check_attention_events(
        self,
        session_id: int,
        track: dict,
        zone_name: str,
        time_in_zone: float,
        current_time: datetime,
        frame_number: int
    ):
        """
        Проверка на события требующие внимания.
        """
        track_id = track['track_id']
        worker_class = track.get('worker_class', 'unknown')
        
        # Event key для предотвращения дублирования
        event_key = f"{track_id}_{worker_class}_{zone_name}"
        
        # 1. Работник в неправильной зоне
        if worker_class in self.work_zones:
            work_zones = self.work_zones[worker_class]
            
            if zone_name not in work_zones and zone_name != "out_of_zones":
                if time_in_zone >= self.wrong_zone_threshold:
                    if 'wrong_zone' not in self.attention_triggered[event_key]:
                        self.db.create_attention_event(
                            session_id=session_id,
                            event_type='wrong_zone',
                            severity='medium',
                            track_id=track_id,
                            worker_class=worker_class,
                            timestamp=current_time,
                            frame_number=frame_number,
                            zone_name=zone_name,
                            description=f"{worker_class.capitalize()} находится {time_in_zone:.1f}с в зоне {zone_name}",
                            bbox=track['bbox']
                        )
                        self.attention_triggered[event_key].add('wrong_zone')
        
        # 2. Unknown работник в RepairZone
        if worker_class == 'unknown' and zone_name == 'RepairZone':
            if time_in_zone >= self.unknown_threshold:
                if 'unknown_in_repair' not in self.attention_triggered[event_key]:
                    self.db.create_attention_event(
                        session_id=session_id,
                        event_type='unknown_in_repair',
                        severity='high',
                        track_id=track_id,
                        worker_class=worker_class,
                        timestamp=current_time,
                        frame_number=frame_number,
                        zone_name=zone_name,
                        description=f"Неопознанный человек в зоне ремонта {time_in_zone:.1f}с. Возможно работник без спецодежды.",
                        bbox=track['bbox']
                    )
                    self.attention_triggered[event_key].add('unknown_in_repair')
        
        # 3. Cleaner в RepairZone
        if worker_class == 'cleaner' and zone_name == 'RepairZone':
            if time_in_zone >= self.wrong_zone_threshold:
                if 'cleaner_in_repair' not in self.attention_triggered[event_key]:
                    self.db.create_attention_event(
                        session_id=session_id,
                        event_type='cleaner_in_repair',
                        severity='medium',
                        track_id=track_id,
                        worker_class=worker_class,
                        timestamp=current_time,
                        frame_number=frame_number,
                        zone_name=zone_name,
                        description=f"Уборщик находится в зоне ремонта {time_in_zone:.1f}с",
                        bbox=track['bbox']
                    )
                    self.attention_triggered[event_key].add('cleaner_in_repair')
    
    def reset_track_timer(self, track_id: int):
        """Сброс таймера для трека (когда трек потерян)."""
        if track_id in self.track_zone_timers:
            del self.track_zone_timers[track_id]


class TrainTracker:
    """
    Отслеживание поездов и их стабильной детекции.
    """
    
    def __init__(self, stable_frames_threshold: int = 10):
        """
        Инициализация.
        
        Args:
            stable_frames_threshold: количество кадров для стабильной детекции
        """
        self.stable_threshold = stable_frames_threshold
        self.consecutive_detections = 0
        self.train_present = False
        self.current_train_number = None
        self.last_train_event_id = None
        
        print(f"✅ TrainTracker инициализирован (threshold={stable_frames_threshold})")
    
    def update(
        self,
        train_detected: bool,
        train_number: Optional[str] = None
    ) -> Dict:
        """
        Обновление состояния поезда.
        
        Args:
            train_detected: детектирован ли поезд на кадре
            train_number: номер поезда (если распознан)
            
        Returns:
            dict с событиями: {'arrival': bool, 'departure': bool, 'stable': bool}
        """
        events = {
            'arrival': False,
            'departure': False,
            'stable': False
        }
        
        if train_detected:
            self.consecutive_detections += 1
            
            # Стабильная детекция
            if self.consecutive_detections >= self.stable_threshold and not self.train_present:
                self.train_present = True
                events['arrival'] = True
                events['stable'] = True
                
                if train_number:
                    self.current_train_number = train_number
            
            # Обновляем номер если распознан
            if train_number and not self.current_train_number:
                self.current_train_number = train_number
        
        else:
            # Поезд не детектирован
            if self.consecutive_detections > 0:
                self.consecutive_detections = 0
            
            # Если был present - отбытие
            if self.train_present:
                self.train_present = False
                events['departure'] = True
                self.current_train_number = None
        
        return events
    
    def get_current_train_number(self) -> Optional[str]:
        """Получение текущего номера поезда."""
        return self.current_train_number