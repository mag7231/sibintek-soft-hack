# src/tracker.py

import numpy as np
import cv2
from collections import defaultdict
from typing import List, Dict, Tuple
import torch


class ReIDFeatureExtractor:
    """
    Извлечение appearance features для ReID.
    Использует простую CNN или предобученную модель для извлечения эмбеддингов одежды.
    """
    
    def __init__(self, model_type='resnet50'):
        """
        Инициализация экстрактора признаков.
        
        Args:
            model_type: тип модели ('simple', 'resnet50', 'osnet')
        """
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Для простоты используем color histogram + HOG
        # В продакшене можно заменить на OSNet или ResNet50
        print(f"ReID Feature Extractor инициализирован: {model_type} на {self.device}")
    
    def extract_features(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Извлечение appearance features из bbox.
        
        Args:
            frame: полный кадр
            bbox: [x1, y1, x2, y2]
            
        Returns:
            feature_vector: numpy array с признаками
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop person region
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return np.zeros(256)  # Пустой вектор если crop невалидный
        
        # Resize to fixed size
        person_crop = cv2.resize(person_crop, (64, 128))
        
        # Extract color histogram (RGB, 8 bins per channel = 512 dims)
        color_hist = self._extract_color_histogram(person_crop)
        
        # Extract HOG features (можно добавить для более точного matching)
        # hog_features = self._extract_hog(person_crop)
        
        # Комбинируем признаки
        features = color_hist
        
        # Нормализация
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def _extract_color_histogram(self, image: np.ndarray, bins=8) -> np.ndarray:
        """
        Извлечение цветовой гистограммы.
        """
        hist_features = []
        
        for channel in range(3):  # BGR
            hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
            hist = hist.flatten()
            hist_features.extend(hist)
        
        return np.array(hist_features)
    
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Вычисление similarity между двумя feature vectors.
        
        Args:
            feat1, feat2: feature vectors
            
        Returns:
            similarity score (0-1, где 1 = идентичны)
        """
        # Cosine similarity
        similarity = np.dot(feat1, feat2)
        return max(0.0, similarity)  # Clip to [0, 1]


class AdvancedTracker:
    """
    Продвинутый трекер с ReID для долгосрочного отслеживания.
    Основан на BoT-SORT с custom ReID logic.
    """
    
    def __init__(self, config: dict):
        """
        Инициализация трекера.
        
        Args:
            config: словарь с настройками трекинга
        """
        self.config = config
        
        # ReID экстрактор
        self.reid_extractor = ReIDFeatureExtractor()
        
        # Активные треки
        self.active_tracks = {}  # track_id -> track_info
        
        # Потерянные треки (для long-term ReID)
        self.lost_tracks = {}  # track_id -> track_info
        
        # Счетчик ID
        self.next_id = 1
        
        # Параметры из конфига
        self.track_buffer = config.get('track_buffer', 120)
        self.long_term_buffer = config.get('long_term_buffer', 9000)
        self.appearance_thresh = config.get('appearance_thresh', 0.25)
        self.match_thresh = config.get('match_thresh', 0.8)
        self.reid_confidence = config.get('reid_confidence', 0.7)
        
        print(f"AdvancedTracker инициализирован:")
        print(f"  - Track buffer: {self.track_buffer} frames")
        print(f"  - Long-term buffer: {self.long_term_buffer} frames")
        print(f"  - ReID threshold: {self.appearance_thresh}")
    
    def update(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Обновление треков на основе новых детекций.
        
        Args:
            frame: текущий кадр
            detections: список детекций [{bbox, conf, class_id, class_name}, ...]
            
        Returns:
            tracks: список треков с назначенными ID
        """
        # Фильтруем только людей
        person_detections = [d for d in detections if d['class_name'] == 'person']
        
        if len(person_detections) == 0:
            # Обновляем lost треки
            self._update_lost_tracks()
            return []
        
        # Извлекаем ReID features для всех детекций
        detection_features = []
        for det in person_detections:
            features = self.reid_extractor.extract_features(frame, det['bbox'])
            detection_features.append(features)
        
        # Matching с активными треками
        matched_tracks, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(
            person_detections, detection_features
        )
        
        # Обновляем matched треки
        for track_id, det_idx in matched_tracks:
            det = person_detections[det_idx]
            features = detection_features[det_idx]
            self._update_track(track_id, det, features, frame)
        
        # Создаем новые треки для unmatched detections
        # Сначала пытаемся match с lost tracks (ReID)
        for det_idx in unmatched_detections:
            det = person_detections[det_idx]
            features = detection_features[det_idx]
            
            # Пытаемся найти в lost tracks
            matched_lost_id = self._match_with_lost_tracks(features)
            
            if matched_lost_id is not None:
                # Восстанавливаем трек из lost
                self._reactivate_track(matched_lost_id, det, features, frame)
            else:
                # Создаем новый трек
                self._create_new_track(det, features, frame)
        
        # Переводим unmatched active треки в lost
        for track_id in unmatched_tracks:
            self._move_to_lost(track_id)
        
        # Очистка old lost tracks
        self._cleanup_lost_tracks()
        
        # Возвращаем активные треки
        return self._get_active_tracks_output()
    
    def _match_detections_to_tracks(
        self, 
        detections: List[Dict], 
        features: List[np.ndarray]
    ) -> Tuple[List, List, List]:
        """
        Matching детекций с активными треками.
        Использует IoU + ReID appearance similarity.
        """
        if len(self.active_tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Cost matrix: [num_tracks, num_detections]
        track_ids = list(self.active_tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self.active_tracks[track_id]
            track_bbox = track['bbox']
            track_features = track['features']
            
            for j, det in enumerate(detections):
                det_bbox = det['bbox']
                det_features = features[j]
                
                # IoU cost
                iou = self._calculate_iou(track_bbox, det_bbox)
                
                # Appearance cost (ReID)
                appearance_sim = self.reid_extractor.compute_similarity(
                    track_features, det_features
                )
                
                # Combined cost (lower is better)
                # Инвертируем, чтобы высокий IoU и similarity давали низкую стоимость
                cost = 1.0 - (0.5 * iou + 0.5 * appearance_sim)
                cost_matrix[i, j] = cost
        
        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        
        matched_indices = linear_sum_assignment(cost_matrix)
        matched_tracks = []
        
        matched_det_indices = set()
        matched_track_indices = set()
        
        for track_idx, det_idx in zip(*matched_indices):
            cost = cost_matrix[track_idx, det_idx]
            
            # Проверяем threshold
            if cost < (1.0 - self.match_thresh):
                track_id = track_ids[track_idx]
                matched_tracks.append((track_id, det_idx))
                matched_det_indices.add(det_idx)
                matched_track_indices.add(track_idx)
        
        # Unmatched
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_indices]
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in matched_track_indices]
        
        return matched_tracks, unmatched_detections, unmatched_tracks
    
    def _match_with_lost_tracks(self, features: np.ndarray) -> int:
        """
        Matching с lost tracks используя только ReID.
        """
        best_match_id = None
        best_similarity = 0.0
        
        for track_id, track in self.lost_tracks.items():
            track_features = track['features']
            similarity = self.reid_extractor.compute_similarity(features, track_features)
            
            if similarity > best_similarity and similarity > self.reid_confidence:
                best_similarity = similarity
                best_match_id = track_id
        
        return best_match_id
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Вычисление IoU между двумя bbox.
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _create_new_track(self, detection: Dict, features: np.ndarray, frame: np.ndarray):
        """Создание нового трека."""
        track_id = self.next_id
        self.next_id += 1
        
        self.active_tracks[track_id] = {
            'id': track_id,
            'bbox': detection['bbox'],
            'conf': detection['conf'],
            'features': features,
            'age': 1,
            'hits': 1,
            'time_since_update': 0,
            'history': [detection['bbox']],
            'first_seen_frame': frame.copy()  # Для визуализации
        }
    
    def _update_track(self, track_id: int, detection: Dict, features: np.ndarray, frame: np.ndarray):
        """Обновление существующего трека."""
        track = self.active_tracks[track_id]
        track['bbox'] = detection['bbox']
        track['conf'] = detection['conf']
        
        # Обновляем features (EMA для сглаживания)
        alpha = 0.9
        track['features'] = alpha * track['features'] + (1 - alpha) * features
        track['features'] = track['features'] / (np.linalg.norm(track['features']) + 1e-6)
        
        track['hits'] += 1
        track['time_since_update'] = 0
        track['history'].append(detection['bbox'])
        
        # Ограничиваем историю
        if len(track['history']) > 30:
            track['history'] = track['history'][-30:]
    
    def _reactivate_track(self, track_id: int, detection: Dict, features: np.ndarray, frame: np.ndarray):
        """Реактивация трека из lost."""
        track = self.lost_tracks.pop(track_id)
        track['bbox'] = detection['bbox']
        track['conf'] = detection['conf']
        track['time_since_update'] = 0
        
        # Обновляем features
        alpha = 0.8
        track['features'] = alpha * track['features'] + (1 - alpha) * features
        track['features'] = track['features'] / (np.linalg.norm(track['features']) + 1e-6)
        
        self.active_tracks[track_id] = track
        
        print(f"Трек #{track_id} реактивирован через ReID!")
    
    def _move_to_lost(self, track_id: int):
        """Перевод трека в lost."""
        track = self.active_tracks.pop(track_id)
        track['time_since_update'] = 0
        track['lost_frame_count'] = 0
        self.lost_tracks[track_id] = track
    
    def _update_lost_tracks(self):
        """Обновление счетчиков для lost треков."""
        for track in self.lost_tracks.values():
            track['lost_frame_count'] += 1
    
    def _cleanup_lost_tracks(self):
        """Удаление старых lost треков."""
        to_remove = []
        
        for track_id, track in self.lost_tracks.items():
            if track['lost_frame_count'] > self.long_term_buffer:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            self.lost_tracks.pop(track_id)
            print(f"  🗑️ Трек #{track_id} удален (превышен long-term buffer)")
    
    def _get_active_tracks_output(self) -> List[Dict]:
        """Получение списка активных треков для вывода."""
        output = []
        
        for track_id, track in self.active_tracks.items():
            track['time_since_update'] += 1
            
            output.append({
                'track_id': track_id,
                'bbox': track['bbox'],
                'conf': track['conf'],
                'class_name': 'person',
                'hits': track['hits'],
                'age': track.get('age', 1)
            })
        
        return output