# # main.py

# import cv2
# import argparse
# from pathlib import Path
# from tqdm import tqdm
# import numpy as np

# from src.detector import Detector
# from src.tracker import AdvancedTracker
# from src.classifier import WorkerClassifier


# def draw_detections(frame: np.ndarray, tracks: list, detections: list, zone_manager=None) -> np.ndarray:
#     """
#     Отрисовка треков, детекций и зон на кадре.
    
#     Args:
#         frame: исходный кадр
#         tracks: список треков людей с track_id и классификацией
#         detections: все детекции (включая поезда)
#         zone_manager: менеджер зон (опционально)
        
#     Returns:
#         frame с отрисовкой
#     """
#     overlay = frame.copy()
    
#     # Отрисовка зон (полупрозрачные полигоны) используя ZoneManager
#     if zone_manager:
#         zone_colors = [
#             (255, 0, 0),    # Красный
#             (0, 255, 0),    # Зеленый
#             (0, 0, 255),    # Синий
#             (255, 255, 0),  # Желтый
#             (255, 0, 255),  # Фиолетовый
#         ]
        
#         drawable_zones = zone_manager.get_polygons_for_drawing()
        
#         for idx, (zone_name, pts) in enumerate(drawable_zones.items()):
#             zone_color = zone_colors[idx % len(zone_colors)]
            
#             # Полупрозрачный полигон (более прозрачный)
#             cv2.fillPoly(overlay, [pts], zone_color)
            
#             # Граница зоны
#             cv2.polylines(frame, [pts], True, zone_color, 2)
            
#             # Название зоны
#             centroid = pts.mean(axis=0).astype(int)
#             cv2.putText(frame, zone_name, tuple(centroid), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # Смешиваем оверлей с исходным кадром (более прозрачно: 0.15 вместо 0.3)
#         cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    
#     # Отрисовка поездов (из detections, не треков)
#     for det in detections:
#         if det.get('class_name') == 'train':
#             x1, y1, x2, y2 = map(int, det['bbox'])
#             conf = det['conf']
            
#             # Синий цвет для поездов
#             train_color = (255, 0, 0)  # BGR: синий
            
#             # Бокс
#             cv2.rectangle(frame, (x1, y1), (x2, y2), train_color, 3)
            
#             # Label
#             label = f"Train: {conf:.2f}"
#             (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#             cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), train_color, -1)
#             cv2.putText(frame, label, (x1, y1 - 5), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
#     # Отрисовка треков людей
#     for track in tracks:
#         x1, y1, x2, y2 = map(int, track['bbox'])
#         track_id = track['track_id']
#         conf = track['conf']
        
#         # Используем цвет типа работника, если есть
#         if 'worker_color' in track:
#             track_color = track['worker_color']
#         else:
#             # Генерируем уникальный цвет для каждого track_id
#             np.random.seed(track_id)
#             track_color = tuple(np.random.randint(50, 255, 3).tolist())
        
#         # Бокс
#         cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 3)
        
#         # Определяем зону, если есть ZoneManager
#         zone_name = "N/A"
#         if zone_manager:
#             zone_name = zone_manager.get_zone(track['bbox'])
        
#         # Формируем label с типом работника
#         worker_name = track.get('worker_name', 'Unknown')
#         worker_conf = track.get('worker_confidence', 0.0)
        
#         label = f"ID:{track_id} | {worker_name} ({worker_conf:.2f})"
#         if zone_name != "N/A":
#             label += f" | {zone_name}"
        
#         # Фон для текста
#         (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), track_color, -1)
        
#         # Текст
#         cv2.putText(frame, label, (x1, y1 - 5), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#         # Рисуем точку "ног"
#         feet_x = int((x1 + x2) / 2)
#         feet_y = int(y2)
#         cv2.circle(frame, (feet_x, feet_y), 5, track_color, -1)
        
#         # Показываем ID и тип работника крупно над головой
#         cv2.putText(frame, f"#{track_id} {worker_name}", (x1, y1 - 25), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, track_color, 2)
    
#     return frame


# def process_video(input_path: str, output_path: str, config_path: str = 'configs/config.yaml'):
#     """
#     Обработка видео с детекцией и сохранение результата.
    
#     Args:
#         input_path: путь к входному видео
#         output_path: путь для сохранения результата
#         config_path: путь к конфигу
#     """
#     # Инициализация детектора
#     detector = Detector(config_path)
    
#     # Открытие видео
#     cap = cv2.VideoCapture(input_path)
    
#     if not cap.isOpened():
#         raise ValueError(f"Не удалось открыть видео: {input_path}")
    
#     # Параметры видео
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     print(f"Видео: {width}x{height}, {fps} FPS, {total_frames} кадров")
    
#     # Создание ZoneManager после получения размеров видео
#     zone_manager = detector.create_zone_manager(width, height)
#     print(f"ZoneManager создан для разрешения {width}x{height}")
    
#     # Создание трекера
#     tracker_config = detector.config.get('tracker', {})
#     tracker = AdvancedTracker(tracker_config)
#     print(f"Трекер инициализирован")
    
#     # Создание классификатора работников
#     classifier_config = detector.config.get('classifier', {})
#     classifier = WorkerClassifier(
#         model_name=classifier_config.get('model_name', 'hf-hub:Marqo/marqo-fashionCLIP'),
#         use_fine_tuned=classifier_config.get('use_fine_tuned', False),
#         fine_tuned_path=classifier_config.get('fine_tuned_path')
#     )
#     print(f"Классификатор инициализирован")
    
#     # Создание writer для сохранения
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     # Статистика
#     total_unique_tracks = 0
#     classify_every_n = classifier_config.get('classify_every_n_frames', 5)
    
#     # Обработка кадров
#     frame_count = 0
    
#     with tqdm(total=total_frames, desc="Обработка видео") as pbar:
#         while True:
#             ret, frame = cap.read()
            
#             if not ret:
#                 break
            
#             # Детекция
#             results = detector.detect(frame)
#             detections = detector.get_detections(results)
            
#             # Трекинг (передаем кадр для ReID)
#             tracks = tracker.update(frame, detections)
            
#             # Классификация работников (не каждый кадр для оптимизации)
#             should_classify = (frame_count % classify_every_n == 0)
#             tracks = classifier.classify_batch(frame, tracks, force_classify=should_classify)
            
#             # Обновляем статистику
#             if len(tracks) > 0:
#                 max_track_id = max([t['track_id'] for t in tracks])
#                 total_unique_tracks = max(total_unique_tracks, max_track_id)
            
#             # Отрисовка (передаем и треки и все детекции для поездов)
#             frame_vis = draw_detections(frame.copy(), tracks, detections, zone_manager)
            
#             # Информация о кадре
#             num_trains = len([d for d in detections if d.get('class_name') == 'train'])
#             info_text = f"Frame: {frame_count} | People: {len(tracks)} | Trains: {num_trains} | Total IDs: {total_unique_tracks}"
#             cv2.putText(frame_vis, info_text, (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
#             # Сохранение
#             out.write(frame_vis)
            
#             frame_count += 1
#             pbar.update(1)
    
#     # Освобождение ресурсов
#     cap.release()
#     out.release()
    
#     print(f"\n✅ Обработка завершена!")
#     print(f"📁 Результат сохранен: {output_path}")
#     print(f"📊 Обработано кадров: {frame_count}")
#     print(f"👥 Всего уникальных людей: {total_unique_tracks}")


# def main():
#     parser = argparse.ArgumentParser(description='YOLO11 детекция на видео')
#     parser.add_argument('--input', type=str, default='data/input/test4.mp4',
#                        help='Путь к входному видео')
#     parser.add_argument('--output', type=str, default='data/output/result7.mp4',
#                        help='Путь для сохранения результата')
#     parser.add_argument('--config', type=str, default='configs/config.yaml',
#                        help='Путь к конфигу')
    
#     args = parser.parse_args()
    
#     # Создание output директории если не существует
#     Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
#     # Проверка существования входного файла
#     if not Path(args.input).exists():
#         print(f"❌ Ошибка: Файл не найден: {args.input}")
#         return
    
#     # Обработка видео
#     try:
#         process_video(args.input, args.output, args.config)
#     except Exception as e:
#         print(f"❌ Ошибка при обработке: {e}")
#         raise


# if __name__ == '__main__':
#     main()
# main.py
# main.py
# main.py

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta

from src.detector import Detector
from src.tracker import AdvancedTracker
from src.classifier import WorkerClassifier
from src.database import DatabaseManager
from src.statistics import StatisticsCollector  # <-- Импортируем новый класс

def draw_detections(frame: np.ndarray, tracks: list, detections: list, zone_manager=None) -> np.ndarray:
    """
    Отрисовка треков, детекций и зон на кадре.
    
    Args:
        frame: исходный кадр
        tracks: список треков людей с track_id и классификацией
        detections: все детекции (включая поезда)
        zone_manager: менеджер зон (опционально)
        
    Returns:
        frame с отрисовкой
    """
    overlay = frame.copy()
    
    # Отрисовка зон (полупрозрачные полигоны) используя ZoneManager
    if zone_manager:
        zone_colors = [
            (255, 0, 0),    # Красный
            (0, 255, 0),    # Зеленый
            (0, 0, 255),    # Синий
            (255, 255, 0),  # Желтый
            (255, 0, 255),  # Фиолетовый
        ]
        
        drawable_zones = zone_manager.get_polygons_for_drawing()
        
        for idx, (zone_name, pts) in enumerate(drawable_zones.items()):
            zone_color = zone_colors[idx % len(zone_colors)]
            
            # Полупрозрачный полигон (более прозрачный)
            cv2.fillPoly(overlay, [pts], zone_color)
            
            # Граница зоны
            cv2.polylines(frame, [pts], True, zone_color, 2)
            
            # Название зоны
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(frame, zone_name, tuple(centroid), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Смешиваем оверлей с исходным кадром (более прозрачно: 0.15 вместо 0.3)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    
    # Отрисовка поездов (из detections, не треков)
    for det in detections:
        if det.get('class_name') == 'train':
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['conf']
            
            # Синий цвет для поездов
            train_color = (255, 0, 0)  # BGR: синий
            
            # Бокс
            cv2.rectangle(frame, (x1, y1), (x2, y2), train_color, 3)
            
            # Label
            label = f"Train: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), train_color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Отрисовка треков людей
    for track in tracks:
        x1, y1, x2, y2 = map(int, track['bbox'])
        track_id = track['track_id']
        conf = track['conf']
        
        # Используем цвет типа работника, если есть
        if 'worker_color' in track:
            track_color = track['worker_color']
        else:
            # Генерируем уникальный цвет для каждого track_id
            np.random.seed(track_id)
            track_color = tuple(np.random.randint(50, 255, 3).tolist())
        
        # Бокс
        cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 3)
        
        # Используем зону, которая была добавлена в `process_video`
        zone_name = track.get('zone_name', 'N/A')
        
        # Формируем label с типом работника
        worker_name = track.get('worker_name', 'Unknown')
        worker_conf = track.get('worker_confidence', 0.0)
        
        label = f"ID:{track_id} | {worker_name} ({worker_conf:.2f})"
        if zone_name != "N/A":
            label += f" | {zone_name}"
        
        # Фон для текста
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), track_color, -1)
        
        # Текст
        cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Рисуем точку "ног"
        feet_x = int((x1 + x2) / 2)
        feet_y = int(y2)
        cv2.circle(frame, (feet_x, feet_y), 5, track_color, -1)
        
        # Показываем ID и тип работника крупно над головой
        cv2.putText(frame, f"#{track_id} {worker_name}", (x1, y1 - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, track_color, 2)
    
    return frame


def process_video(
    input_path: str,
    output_path: str,
    config_path: str,
    video_start_time: datetime,
    db_path: str
):
    """
    Обработка видео с детекцией, сохранением результата и логированием в БД.
    
    Args:
        input_path: путь к входному видео
        output_path: путь для сохранения результата
        config_path: путь к конфигу
        video_start_time: время начала видео (datetime)
        db_path: путь к файлу БД
    """
    # Инициализация менеджера БД
    db_manager = DatabaseManager(db_path=db_path)
    
    # Инициализация детектора
    detector = Detector(config_path)
    
    # Открытие видео
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {input_path}")
    
    # Параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Видео: {width}x{height}, {fps} FPS, {total_frames} кадров")
    
    # Создание сессии в БД
    video_session = db_manager.create_video_session(
        video_path=input_path,
        start_time=video_start_time,
        fps=fps,
        total_frames=total_frames
    )
    session_id = video_session.id
    
    # Создание ZoneManager после получения размеров видео
    zone_manager = detector.create_zone_manager(width, height)
    print(f"ZoneManager создан для разрешения {width}x{height}")
    
    # Создание трекера
    tracker_config = detector.config.get('tracker', {})
    tracker = AdvancedTracker(tracker_config)
    print(f"Трекер инициализирован")
    
    # Создание классификатора работников
    classifier_config = detector.config.get('classifier', {})
    classifier = WorkerClassifier(
        model_name=classifier_config.get('model_name', 'hf-hub:Marqo/marqo-fashionCLIP'),
        use_fine_tuned=classifier_config.get('use_fine_tuned', False),
        fine_tuned_path=classifier_config.get('fine_tuned_path')
    )
    print(f"Классификатор инициализирован")
    
    # --- НОВОЕ: Инициализация StatisticsCollector ---
    # Он будет управлять всей логикой БД внутри цикла
    stats_config = detector.config.get('statistics', {})
    stats_collector = StatisticsCollector(
        db=db_manager,
        session_id=session_id,
        fps=fps,
        idle_threshold=stats_config.get('idle_threshold_sec', 10.0),
        wrong_zone_threshold=stats_config.get('wrong_zone_threshold_sec', 10.0),
        unknown_in_repair_threshold=stats_config.get('unknown_in_repair_threshold_sec', 5.0),
        train_stable_frames=stats_config.get('train_stable_frames', 5)
    )
    print("StatisticsCollector инициализирован.")
    # --- КОНЕЦ НОВОГО ---
    
    # Создание writer для сохранения
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Статистика
    total_unique_tracks = 0
    classify_every_n = classifier_config.get('classify_every_n_frames', 5)

    # Обработка кадров
    frame_count = 0
    
    with tqdm(total=total_frames, desc="Обработка видео") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Текущее время в видео
            current_time = video_start_time + timedelta(seconds=frame_count / fps)

            # Детекция
            results = detector.detect(frame)
            detections = detector.get_detections(results) # Все детекции (люди + поезда)
            
            # Фильтруем детекции людей для трекера
            person_detections = [d for d in detections if d.get('class_name') == 'person']
            
            # Трекинг (передаем кадр для ReID и только детекции людей)
            tracks = tracker.update(frame, person_detections)
            
            # Классификация работников (не каждый кадр для оптимизации)
            should_classify = (frame_count % classify_every_n == 0)
            tracks = classifier.classify_batch(frame, tracks, force_classify=should_classify)
            
            # --- Логика для Базы Данных (теперь через StatisticsCollector) ---
            
            # 1. Добавляем 'zone_name' к трекам, т.к. stats_collector его ожидает
            for track in tracks:
                zone_name = zone_manager.get_zone(track['bbox'])
                track['zone_name'] = zone_name  # Сохраняем для отрисовки И для stats_collector
            
            # 2. Вызываем StatisticsCollector
            # Он сам обновит WorkerActivity, проверит поезда и создаст AttentionEvents
            stats_collector.process_frame(
                frame_idx=frame_count,
                timestamp=current_time,
                detections=detections,  # Передаем ВСЕ детекции (для поездов)
                tracks=tracks          # Передаем треки людей
            )
            
            # --- Вся ручная логика БД отсюда УДАЛЕНА ---
            
            # --------------------------------

            # Обновляем статистику для отображения
            if len(tracks) > 0:
                max_track_id = max([t['track_id'] for t in tracks])
                total_unique_tracks = max(total_unique_tracks, max_track_id)
            
            # Отрисовка (передаем и треки и *все* детекции для поездов)
            frame_vis = draw_detections(frame.copy(), tracks, detections, zone_manager)
            
            # Информация о кадре
            num_trains = len([d for d in detections if d.get('class_name') == 'train'])
            info_text = f"Frame: {frame_count} | Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
            cv2.putText(frame_vis, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            info_text_2 = f"People: {len(tracks)} | Trains: {num_trains} | Total IDs: {total_unique_tracks}"
            cv2.putText(frame_vis, info_text_2, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Сохранение
            out.write(frame_vis)
            
            frame_count += 1
            pbar.update(1)
    
    # --- Завершение обработки ---
    
    # Рассчитываем время окончания
    video_end_time = video_start_time + timedelta(seconds=frame_count / fps)
    
    # --- НОВОЕ: Завершаем сессию через StatisticsCollector ---
    # Он сам обновит время окончания сессии и закроет незакрытые события поезда
    stats_collector.finish_session(video_end_time)
    # --- КОНЕЦ НОВОГО ---

    # --- Старая ручная логика УДАЛЕНА ---
    
    # Освобождение ресурсов
    cap.release()
    out.release()
    
    print(f"\n✅ Обработка завершена!")
    print(f"📁 Результат сохранен: {output_path}")
    print(f"📊 Обработано кадров: {frame_count}")
    
    # Печать статистики из БД (эта часть остается, она просто читает данные)
    print("\n--- 📊 Статистика из Базы Данных ---")
    try:
        stats = db_manager.get_session_statistics(session_id)
        
        print(f"Сессия: {stats['session'].id} (Начало: {stats['session'].start_time})")
        
        print(f"\n🚂 События поездов ({len(stats['trains'])}):")
        for train in stats['trains']:
            print(f"  - Поезд {train.train_number}: "
                  f"Прибыл {train.arrival_time.strftime('%H:%M:%S')}, "
                  f"Отбыл {train.departure_time.strftime('%H:%M:%S') if train.departure_time else 'N/A'}")
        
        print(f"\n👥 Статистика по работникам ({len(stats['worker_stats'])}):")
        for cls, data in stats['worker_stats'].items():
            print(f"  - Класс: {cls.upper()}")
            print(f"    - Кол-во: {data['count']}")
            print(f"    - В рабочей зоне: {data['work_time']:.1f} сек.")
            print(f"    - Вне зоны: {data['idle_time']:.1f} сек.")
            print(f"    - Всего: {data['total_time']:.1f} сек.")

        print(f"\n⚠️ События (неразрешенные): {len(stats['attentions'])}")
        for event in stats['attentions']:
             print(f"  - {event.timestamp.strftime('%H:%M:%S')}: {event.event_type} (ID: {event.track_id}) - {event.description}")

    except Exception as e:
        print(f"❌ Не удалось получить статистику из БД: {e}")

    # Закрытие сессии БД
    db_manager.close()
    print("------------------------------------")


def main():
    parser = argparse.ArgumentParser(description='YOLO11 детекция на видео с логированием в БД')
    
    # --- Стандартные аргументы ---
    parser.add_argument('--input', type=str, default='data/input/test4.mp4',
                        help='Путь к входному видео')
    parser.add_argument('--output', type=str, default='data/output/result7.mp4',
                        help='Путь для сохранения результата')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Путь к конфигу')
    
    # --- Аргументы для БД ---
    parser.add_argument('--db-path', type=str, default='data/app.db',
                        help='Путь к файлу базы данных SQLite')
    
    default_start_time = datetime.now().replace(microsecond=0).isoformat()
    parser.add_argument('--start-time', type=str, default=default_start_time,
                        help=f'Время начала видео в ISO формате (YYYY-MM-DDTHH:MM:SS). '
                             f'По умолчанию: {default_start_time}')
    
    args = parser.parse_args()
    
    # Создание output директории если не существует
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Проверка существования входного файла
    if not Path(args.input).exists():
        print(f"❌ Ошибка: Файл не найден: {args.input}")
        return
        
    # Парсинг времени начала
    try:
        video_start_time = datetime.fromisoformat(args.start_time)
    except ValueError:
        print(f"❌ Ошибка: Неверный формат --start-time. Ожидается YYYY-MM-DDTHH:MM:SS")
        return

    print(f"--- 🚀 Запуск обработки ---")
    print(f"Видео: {args.input}")
    print(f"Время старта: {video_start_time}")
    print(f"База данных: {args.db_path}")
    print(f"--------------------------")

    # Обработка видео
    try:
        process_video(
            args.input,
            args.output,
            args.config,
            video_start_time,
            args.db_path
        )
    except Exception as e:
        print(f"❌ Ошибка при обработке: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()