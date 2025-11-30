# src/database.py

"""
База данных для хранения событий и статистики.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import List, Dict, Optional
import json

Base = declarative_base()


class VideoSession(Base):
    """Сессия обработки видео."""
    __tablename__ = 'video_sessions'
    
    id = Column(Integer, primary_key=True)
    video_path = Column(String(500))
    start_time = Column(DateTime)  # Время начала видео (из OCR)
    end_time = Column(DateTime)
    fps = Column(Float)
    total_frames = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    train_events = relationship('TrainEvent', back_populates='session')
    worker_activities = relationship('WorkerActivity', back_populates='session')
    attention_events = relationship('AttentionEvent', back_populates='session')


class TrainEvent(Base):
    """События связанные с поездами."""
    __tablename__ = 'train_events'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('video_sessions.id'))
    
    train_number = Column(String(50))
    arrival_time = Column(DateTime)
    departure_time = Column(DateTime, nullable=True)
    
    arrival_frame = Column(Integer)
    departure_frame = Column(Integer, nullable=True)
    
    stable_detection = Column(Boolean, default=False)  # Стабильно детектирован
    
    # Relationships
    session = relationship('VideoSession', back_populates='train_events')


class WorkerActivity(Base):
    """Активность работников."""
    __tablename__ = 'worker_activities'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('video_sessions.id'))
    
    track_id = Column(Integer)
    worker_class = Column(String(50))  # mechanic, worker, cleaner, driver, unknown
    
    # Временные метки
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    
    # Кадры
    first_frame = Column(Integer)
    last_frame = Column(Integer)
    
    # Статистика по зонам
    time_in_work_zone = Column(Float, default=0.0)  # секунды
    time_in_other_zones = Column(Float, default=0.0)  # секунды
    
    total_time = Column(Float, default=0.0)  # секунды
    
    # Зоны посещения (JSON)
    zones_visited = Column(Text)  # {"RepairZone": 120.5, "PlatformZone": 45.2, ...}
    
    # Relationships
    session = relationship('VideoSession', back_populates='worker_activities')


class AttentionEvent(Base):
    """События требующие внимания."""
    __tablename__ = 'attention_events'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('video_sessions.id'))
    
    event_type = Column(String(100))  # wrong_zone, unknown_in_repair, no_uniform, etc.
    severity = Column(String(20))  # low, medium, high
    
    track_id = Column(Integer)
    worker_class = Column(String(50))
    
    timestamp = Column(DateTime)
    frame_number = Column(Integer)
    
    zone_name = Column(String(100))
    description = Column(Text)
    
    # Для сохранения bbox в S3 (для дообучения)
    bbox = Column(Text)  # JSON: [x1, y1, x2, y2]
    crop_saved = Column(Boolean, default=False)
    s3_path = Column(String(500), nullable=True)
    
    # Статус разрешения
    resolved = Column(Boolean, default=False)
    resolution = Column(String(100), nullable=True)  # ok, no_uniform, new_uniform, etc.
    resolved_at = Column(DateTime, nullable=True)
    
    # Relationships
    session = relationship('VideoSession', back_populates='attention_events')


class DatabaseManager:
    """
    Менеджер базы данных.
    """
    
    def __init__(self, db_path: str = 'data/app.db'):
        """
        Инициализация базы данных.
        
        Args:
            db_path: путь к SQLite базе
        """
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        
        # Создание таблиц
        Base.metadata.create_all(self.engine)
        
        # Сессия
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        print(f"✅ Database инициализирована: {db_path}")
    
    # ============= VIDEO SESSION =============
    
    def create_video_session(
        self,
        video_path: str,
        start_time: datetime,
        fps: float,
        total_frames: int
    ) -> VideoSession:
        """Создание новой сессии видео."""
        session = VideoSession(
            video_path=video_path,
            start_time=start_time,
            fps=fps,
            total_frames=total_frames
        )
        self.session.add(session)
        self.session.commit()
        
        print(f"📹 Создана video session ID={session.id}")
        return session
    
    def update_video_session_end_time(self, session_id: int, end_time: datetime):
        """Обновление времени окончания видео."""
        session = self.session.query(VideoSession).filter_by(id=session_id).first()
        if session:
            session.end_time = end_time
            self.session.commit()
    
    # ============= TRAIN EVENTS =============
    
    def create_train_arrival(
        self,
        session_id: int,
        train_number: str,
        arrival_time: datetime,
        arrival_frame: int
    ) -> TrainEvent:
        """Регистрация прибытия поезда."""
        event = TrainEvent(
            session_id=session_id,
            train_number=train_number,
            arrival_time=arrival_time,
            arrival_frame=arrival_frame,
            stable_detection=True
        )
        self.session.add(event)
        self.session.commit()
        
        print(f"🚂 Поезд прибыл: {train_number} в {arrival_time.strftime('%H:%M:%S')}")
        return event
    
    def update_train_departure(
        self,
        train_event_id: int,
        departure_time: datetime,
        departure_frame: int
    ):
        """Обновление отбытия поезда."""
        event = self.session.query(TrainEvent).filter_by(id=train_event_id).first()
        if event:
            event.departure_time = departure_time
            event.departure_frame = departure_frame
            self.session.commit()
            
            print(f"🚂 Поезд отбыл: {event.train_number} в {departure_time.strftime('%H:%M:%S')}")
    
    # ============= WORKER ACTIVITIES =============
    
    def create_or_update_worker_activity(
        self,
        session_id: int,
        track_id: int,
        worker_class: str,
        current_time: datetime,
        current_frame: int,
        zone_name: str,
        time_delta: float
    ):
        """Создание или обновление активности работника."""
        # Ищем существующую запись
        activity = self.session.query(WorkerActivity).filter_by(
            session_id=session_id,
            track_id=track_id
        ).first()
        
        if not activity:
            # Создаем новую
            activity = WorkerActivity(
                session_id=session_id,
                track_id=track_id,
                worker_class=worker_class,
                first_seen=current_time,
                first_frame=current_frame,
                zones_visited=json.dumps({}),
                time_in_work_zone=0.0,
                time_in_other_zones=0.0,
                total_time=0.0
            )
            self.session.add(activity)
        
        # Обновляем (с проверкой на None)
        activity.last_seen = current_time
        activity.last_frame = current_frame
        activity.total_time = (activity.total_time or 0.0) + time_delta
        
        # Обновляем зоны
        zones_dict = json.loads(activity.zones_visited)
        zones_dict[zone_name] = zones_dict.get(zone_name, 0.0) + time_delta
        activity.zones_visited = json.dumps(zones_dict)
        
        # Определяем рабочую зону
        work_zones = self._get_work_zones_for_class(worker_class)
        
        if zone_name in work_zones:
            activity.time_in_work_zone = (activity.time_in_work_zone or 0.0) + time_delta
        else:
            activity.time_in_other_zones = (activity.time_in_other_zones or 0.0) + time_delta
        
        self.session.commit()
    
    def _get_work_zones_for_class(self, worker_class: str) -> List[str]:
        """Получение рабочих зон для класса работника."""
        work_zones_map = {
            'mechanic': ['RepairZone'],
            'worker': ['RepairZone'],
            'driver': ['RepairZone', 'PlatformZone'],
            'cleaner': ['PlatformZone', 'CrossingZone']
        }
        return work_zones_map.get(worker_class, [])
    
    # ============= ATTENTION EVENTS =============
    
    def create_attention_event(
        self,
        session_id: int,
        event_type: str,
        severity: str,
        track_id: int,
        worker_class: str,
        timestamp: datetime,
        frame_number: int,
        zone_name: str,
        description: str,
        bbox: List[float] = None
    ) -> AttentionEvent:
        """Создание события требующего внимания."""
        event = AttentionEvent(
            session_id=session_id,
            event_type=event_type,
            severity=severity,
            track_id=track_id,
            worker_class=worker_class,
            timestamp=timestamp,
            frame_number=frame_number,
            zone_name=zone_name,
            description=description,
            bbox=json.dumps(bbox) if bbox else None
        )
        self.session.add(event)
        self.session.commit()
        
        print(f"⚠️ Attention: {event_type} - {description}")
        return event
    
    def resolve_attention_event(
        self,
        event_id: int,
        resolution: str
    ):
        """Разрешение attention события."""
        event = self.session.query(AttentionEvent).filter_by(id=event_id).first()
        if event:
            event.resolved = True
            event.resolution = resolution
            event.resolved_at = datetime.now()
            self.session.commit()
    
    # ============= QUERIES =============
    
    def get_session_statistics(self, session_id: int) -> Dict:
        """Получение статистики по сессии."""
        session = self.session.query(VideoSession).filter_by(id=session_id).first()
        
        if not session:
            return {}
        
        # Поезда
        trains = self.session.query(TrainEvent).filter_by(session_id=session_id).all()
        
        # Работники
        workers = self.session.query(WorkerActivity).filter_by(session_id=session_id).all()
        
        # Attention события
        attentions = self.session.query(AttentionEvent).filter_by(
            session_id=session_id,
            resolved=False
        ).all()
        
        # Статистика по классам
        worker_stats = {}
        for worker in workers:
            cls = worker.worker_class
            if cls not in worker_stats:
                worker_stats[cls] = {
                    'count': 0,
                    'total_time': 0.0,
                    'work_time': 0.0,
                    'idle_time': 0.0
                }
            
            worker_stats[cls]['count'] += 1
            worker_stats[cls]['total_time'] += worker.total_time
            worker_stats[cls]['work_time'] += worker.time_in_work_zone
            worker_stats[cls]['idle_time'] += worker.time_in_other_zones
        
        return {
            'session': session,
            'trains': trains,
            'workers': workers,
            'worker_stats': worker_stats,
            'attentions': attentions
        }
    
    def close(self):
        """Закрытие сессии."""
        self.session.close()