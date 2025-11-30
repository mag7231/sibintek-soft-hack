# Modified src/detector.py with static train bbox smoothing

import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO
import numpy as np


class Detector:
    """
    Обертка над YOLO11 для детекции людей и поездов.
    Добавлен механизм стабилизации bbox для поезда, чтобы стоящий поезд
    не "дребезжал" и не создавал новые боксы при небольших изменениях.
    """

    def __init__(self, config_path: str = 'configs/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        weights_path = Path('weights') / self.config['model']['weights']
        self.conf_threshold = self.config['model']['conf_threshold']

        print(f"Загрузка модели: {weights_path}")
        self.model = YOLO(str(weights_path))

        self.target_classes = self.config['target_classes']
        self.class_ids = list(self.target_classes.values())

        # --- память для стабилизации bbox для поездов ---
        self.prev_train_bbox = None
        self.iou_threshold = 0.97  # если новый bbox почти совпадает — считаем поезд неподвижным

        print(f"Детектор инициализирован. Классы: {self.target_classes}")

    def detect(self, frame: np.ndarray):
        results = self.model(
            frame,
            conf=self.conf_threshold,
            classes=self.class_ids,
            verbose=False
        )
        return results[0]

    def _iou(self, boxA, boxB):
        # Intersection
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _stable_train_bbox(self, new_bbox):
        """Стабилизация bbox для поезда"""
        if self.prev_train_bbox is None:
            self.prev_train_bbox = new_bbox
            return new_bbox

        iou = self._iou(self.prev_train_bbox, new_bbox)
        if iou > self.iou_threshold:
            # поезд стоит — возвращаем старый бокс
            return self.prev_train_bbox
        else:
            # поезд реально сместился
            self.prev_train_bbox = new_bbox
            return new_bbox

    def get_detections(self, results):
        detections = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(boxes, confs, class_ids):
                class_name = self.get_class_name(class_id)

                # стабилизация только для поездов
                if class_name == 'train':
                    box = self._stable_train_bbox(box.tolist())
                else:
                    box = box.tolist()

                detections.append({
                    'bbox': box,
                    'conf': float(conf),
                    'class_id': int(class_id),
                    'class_name': class_name
                })

        return detections

    def get_class_name(self, class_id: int) -> str:
        for name, cid in self.target_classes.items():
            if cid == class_id:
                return name
        return 'unknown'

    def get_zones(self) -> dict:
        return self.config.get('zones', {})

    def create_zone_manager(self, frame_width: int, frame_height: int):
        from src.logic import ZoneManager
        zones_config = self.get_zones()
        return ZoneManager(zones_config, frame_width, frame_height)
