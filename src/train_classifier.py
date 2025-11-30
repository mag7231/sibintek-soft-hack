# train_classifier.py

"""
Скрипт для дообучения MobileCLIP на кастомном датасете работников железной дороги.

Структура датасета:
data/workers/
├── train/
│   ├── mechanic/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── worker/
│   ├── cleaner/
│   ├── driver/
│   └── unknown/
└── val/
    ├── mechanic/
    ├── worker/
    ├── cleaner/
    ├── driver/
    └── unknown/
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import argparse
import yaml

from src.classifier import WorkerClassifier, WorkerClassifierFineTuner


class WorkerDataset(Dataset):
    """
    Dataset для обучения классификатора работников.
    """
    
    def __init__(self, root_dir: str, transform, tokenizer, classes: list):
        """
        Args:
            root_dir: путь к директории с классами
            transform: трансформации для изображений
            tokenizer: токенизатор для текста
            classes: список классов
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Собираем все изображения
        self.samples = []
        for class_name in classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"⚠️ Директория {class_dir} не найдена, пропускаем...")
                continue
            
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, class_name))
            for img_path in class_dir.glob('*.png'):
                self.samples.append((img_path, class_name))
        
        print(f"📁 Загружено {len(self.samples)} изображений из {root_dir}")
        
        # Статистика по классам
        class_counts = {}
        for _, class_name in self.samples:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"   - {class_name}: {count} изображений")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        
        # Загрузка и преобразование изображения
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Текстовое описание класса
        class_descriptions = {
            'mechanic': 'person wearing white work uniform and safety vest',
            'worker': 'person wearing grey work overalls and hard hat',
            'cleaner': 'person wearing dark blue uniform with orange horizontal stripe on shoulders',
            'driver': 'train driver wearing dark blue professional uniform',
            'unknown': 'person wearing casual clothes'
        }
        
        text = f"a photo of {class_descriptions[class_name]}"
        text_tokens = self.tokenizer([text])[0]
        
        # Label
        label = self.class_to_idx[class_name]
        
        return image, text_tokens, label


def main():
    parser = argparse.ArgumentParser(description='Дообучение MobileCLIP для классификации работников')
    parser.add_argument('--data-root', type=str, default='data/workers',
                       help='Путь к корневой директории датасета')
    parser.add_argument('--model-name', type=str, default='hf-hub:Marqo/marqo-fashionCLIP',
                       help='Название модели CLIP')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Размер batch')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Количество эпох')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--save-path', type=str, default='weights/fashionclip_workers_finetuned.pt',
                       help='Путь для сохранения весов')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Путь к конфигу')
    
    args = parser.parse_args()
    
    # Загрузка конфига
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    classifier_config = config.get('classifier', {})
    classes = list(classifier_config.get('classes', {}).keys())
    
    print(f"🎓 Начинаем дообучение MobileCLIP")
    print(f"   Модель: {args.model_name}")
    print(f"   Датасет: {args.data_root}")
    print(f"   Классы: {classes}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    
    # Инициализация базовой модели
    base_classifier = WorkerClassifier(
        model_name=args.model_name,
        use_fine_tuned=False
    )
    
    # Создание datasets
    train_dataset = WorkerDataset(
        root_dir=Path(args.data_root) / 'train',
        transform=base_classifier.preprocess,
        tokenizer=base_classifier.tokenizer,
        classes=classes
    )
    
    val_dataset = WorkerDataset(
        root_dir=Path(args.data_root) / 'val',
        transform=base_classifier.preprocess,
        tokenizer=base_classifier.tokenizer,
        classes=classes
    )
    
    if len(train_dataset) == 0:
        print("❌ Ошибка: тренировочный датасет пуст!")
        print(f"   Убедитесь, что структура датасета правильная:")
        print(f"   {args.data_root}/train/<class_name>/*.jpg")
        return
    
    # Создание fine-tuner
    fine_tuner = WorkerClassifierFineTuner(
        base_model=base_classifier,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    # Обучение
    fine_tuner.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset if len(val_dataset) > 0 else None,
        save_path=args.save_path
    )
    
    print(f"\n✅ Дообучение завершено!")
    print(f"   Модель сохранена: {args.save_path}")
    print(f"\nДля использования дообученной модели обновите config.yaml:")
    print(f"   classifier:")
    print(f"     use_fine_tuned: true")
    print(f"     fine_tuned_path: '{args.save_path}'")
    print(f"     model_name: '{args.model_name}'")


if __name__ == '__main__':
    main()