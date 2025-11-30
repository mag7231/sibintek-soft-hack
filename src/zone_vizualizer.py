import cv2
import numpy as np
import yaml
from logic import ZoneManager

CONFIG_PATH = 'configs/config.yaml'
INPUT_IMAGE_PATH = 'data/input/test_frame.jpg'
OUTPUT_IMAGE_PATH = 'data/output/test_frame_with_zones.jpg'

def visualize_zones_on_frame(frame: np.ndarray, zone_manager: ZoneManager):
    frame_copy = frame.copy()
    
    colors = {
        "RepairZone": (0, 0, 255),
        "FrontCrossingZone": (0, 165, 255),
        "PlatformZone": (255, 0, 0),
        "SideZone": (0, 255, 255),
        "TrainNumberOCR": (255, 255, 0)
    }
    
    zone_polygons = zone_manager.get_polygons_for_drawing()
    
    for name_cfg, points in zone_polygons.items():
        name_display = name_cfg
        color = colors.get(name_display, (200, 200, 200))
        
        overlay = frame_copy.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.3, frame_copy, 0.7, 0, frame_copy)
        cv2.polylines(frame_copy, [points], True, color, 2, cv2.LINE_AA)
        
        center_x = int(points[:, 0].mean())
        center_y = int(points[:, 1].min() + 15)
        cv2.putText(frame_copy, name_display, (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return frame_copy

def process_image():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    frame = cv2.imread(INPUT_IMAGE_PATH)
    H, W = frame.shape[:2]
    
    zone_manager = ZoneManager(cfg['zones'], W, H)
    result_frame = visualize_zones_on_frame(frame, zone_manager)
    
    cv2.imwrite(OUTPUT_IMAGE_PATH, result_frame)
    print(f"Saved: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    process_image()