# src/ocr.py

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import re
from datetime import datetime, timedelta
from paddleocr import PaddleOCR


class TimeOCR:
    """Recognize timestamp from video frame (top-left corner)."""
    
    def __init__(self):
        # PaddleOCR with English + digits
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', device='cpu')
        print("TimeOCR initialized (PaddleOCR)")
    
    def extract_timestamp(self, frame: np.ndarray, roi: Tuple[int, int, int, int] = None) -> Optional[datetime]:
        """
        Extract timestamp from frame.
        
        Args:
            frame: BGR image
            roi: region of interest (x1, y1, x2, y2), default top-left 400x80
            
        Returns:
            datetime object or None
        """
        if roi is None:
            h, w = frame.shape[:2]
            roi = (0, 0, min(400, w), min(80, h))
        
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        
        # Preprocess: grayscale + increase contrast
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        
        # OCR
        result = self.ocr.ocr(gray, cls=True)
        
        if not result or not result[0]:
            return None
        
        # Extract text
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            
            if confidence > 0.6:
                timestamp = self._parse_timestamp(text)
                if timestamp:
                    return timestamp
        
        return None
    
    def _parse_timestamp(self, text: str) -> Optional[datetime]:
        """
        Parse text to datetime.
        Expected format: "2024-11-30 14:23:45"
        """
        text = text.strip()
        
        # Pattern: YYYY-MM-DD HH:MM:SS
        pattern = r'(\d{4}[-/]\d{2}[-/]\d{2})\s+(\d{2}:\d{2}:\d{2})'
        match = re.search(pattern, text)
        
        if match:
            try:
                date_str = match.group(1).replace('/', '-')
                time_str = match.group(2)
                timestamp_str = f"{date_str} {time_str}"
                return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        
        return None


class TrainNumberOCR:
    """Recognize train numbers from OCR zone."""
    
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', device='cpu')
        
        # Buffer for stable detection
        self.detection_buffer = []
        self.buffer_size = 5
        
        print("TrainNumberOCR initialized (PaddleOCR)")
    
    def extract_train_number(
        self,
        frame: np.ndarray,
        zone_coords: list,
        confidence_threshold: float = 0.7
    ) -> Optional[str]:
        """
        Extract train number from OCR zone.
        
        Args:
            frame: full BGR frame
            zone_coords: zone polygon coordinates [[x1,y1], [x2,y2], ...]
            confidence_threshold: minimum OCR confidence
            
        Returns:
            train number (string) or None
        """
        # Get bounding box of zone
        zone_array = np.array(zone_coords, dtype=np.int32)
        x1, y1 = zone_array.min(axis=0)
        x2, y2 = zone_array.max(axis=0)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        # Preprocess for better OCR
        crop = self._preprocess(crop)
        
        # OCR
        result = self.ocr.ocr(crop, cls=True)
        
        if not result or not result[0]:
            return None
        
        # Find train number
        candidates = []
        for line in result[0]:
            text = line[1][0]
            conf = line[1][1]
            
            if conf > confidence_threshold:
                train_num = self._clean_train_number(text)
                if train_num:
                    candidates.append((train_num, conf))
        
        if not candidates:
            return None
        
        # Pick best candidate
        best = max(candidates, key=lambda x: x[1])
        train_number = best[0]
        
        # Add to buffer for stability
        self.detection_buffer.append(train_number)
        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer.pop(0)
        
        # Return most common from buffer
        if len(self.detection_buffer) >= 3:
            most_common = max(set(self.detection_buffer), key=self.detection_buffer.count)
            return most_common
        
        return train_number
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess for train number recognition."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Denoise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def _clean_train_number(self, text: str) -> Optional[str]:
        """
        Clean and validate train number.
        Expected formats: "12345", "ЭД-123", "РЖД 456"
        """
        # Remove non-alphanumeric except hyphen
        text = re.sub(r'[^\w\d-]', '', text.upper())
        
        # Must have at least 2 chars and 1 digit
        if len(text) < 2 or not re.search(r'\d', text):
            return None
        
        return text
    
    def reset_buffer(self):
        """Reset buffer when train leaves."""
        self.detection_buffer = []


class ProgrammaticClock:
    """Programmatic clock for video timeline."""
    
    def __init__(self, start_time: datetime, fps: int):
        self.start_time = start_time
        self.fps = fps
        self.frame_count = 0
        
        print(f"Clock initialized: {start_time.strftime('%Y-%m-%d %H:%M:%S')} @ {fps} FPS")
    
    def get_current_time(self) -> datetime:
        """Get current time based on frame count."""
        elapsed_seconds = self.frame_count / self.fps
        return self.start_time + timedelta(seconds=elapsed_seconds)
    
    def increment(self):
        """Increment frame counter."""
        self.frame_count += 1
    
    def get_elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return self.frame_count / self.fps
    
    def get_time_string(self) -> str:
        """Get current time as string."""
        return self.get_current_time().strftime('%Y-%m-%d %H:%M:%S')


class OCRManager:
    """Manager for all OCR operations."""
    
    def __init__(self, fps: int):
        self.fps = fps
        self.time_ocr = TimeOCR()
        self.train_ocr = TrainNumberOCR()
        self.clock = None
        self.initial_timestamp_detected = False
        
        print("OCRManager initialized")
    
    def initialize_clock_from_frame(self, frame: np.ndarray) -> bool:
        """
        Initialize clock from first frame timestamp.
        
        Returns:
            True if successfully initialized
        """
        if self.initial_timestamp_detected:
            return True
        
        timestamp = self.time_ocr.extract_timestamp(frame)
        
        if timestamp:
            self.clock = ProgrammaticClock(timestamp, self.fps)
            self.initial_timestamp_detected = True
            print(f"Clock initialized from frame: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        
        return False
    
    def get_current_time(self) -> Optional[datetime]:
        """Get current video time."""
        return self.clock.get_current_time() if self.clock else None
    
    def increment_time(self):
        """Increment clock (call every frame)."""
        if self.clock:
            self.clock.increment()
    
    def detect_train_number(
        self,
        frame: np.ndarray,
        zone_coords: list,
        train_detected: bool
    ) -> Optional[str]:
        """
        Detect train number from OCR zone.
        
        Args:
            frame: current frame
            zone_coords: OCR zone coordinates
            train_detected: is train present in this frame
            
        Returns:
            train number or None
        """
        if not train_detected:
            self.train_ocr.reset_buffer()
            return None
        
        return self.train_ocr.extract_train_number(frame, zone_coords)