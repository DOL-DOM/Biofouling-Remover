import cv2
import numpy as np
from ultralytics import YOLO

class BarnacleDetector:
    def __init__(self, model_path="best.pt", conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.min_confidence_limit = 0.35

    def infer_frame(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=640,
            verbose=False
        )

        detections = []
        low_confidence_flag = False
        total_dirty_area = 0

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                conf = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                area = int((x2 - x1) * (y2 - y1))
                total_dirty_area += area

                detections.append({
                    "center": (cx, cy),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "area": area
                })

                if conf < self.min_confidence_limit:
                    low_confidence_flag = True

        status = "SUCCESS"
        if low_confidence_flag or (len(detections) > 0 and np.mean([d["confidence"] for d in detections]) < self.min_confidence_limit):
            status = "LOW_CONFIDENCE"

        return detections, total_dirty_area, status