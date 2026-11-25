import os
import cv2
import numpy as np
from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model_path=None, confidence=0.35):

        if model_path is None:
            if os.path.exists("models/yolov8n.pt"):
                model_path = "models/yolov8n.pt"
            else:
                model_path = "yolov8n.pt"

        print("Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        self.confidence = confidence

        print("Model loaded successfully!")

    def get_features(self, frame, bbox):

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros(256, dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        histogram = cv2.calcHist(
            [hsv], [0, 1], None,
            [16, 16], [0, 180, 0, 256]
        )

        cv2.normalize(histogram, histogram)

        return histogram.flatten().astype(np.float32)

    def detect(self, frame, track=True):

        if track:
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.confidence,
                verbose=False
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.confidence,
                verbose=False
            )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        coordinates = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        if track and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            ids = [None] * len(coordinates)

        names = self.model.names

        for box, score, class_id, raw_id in zip(
            coordinates, scores, class_ids, ids
        ):

            x1, y1, x2, y2 = map(int, box)
            bbox = (x1, y1, x2, y2)

            class_name = names[int(class_id)]

            # Appearance descriptors are used for people only.
            if class_id == 0:
                features = self.get_features(frame, bbox)
            else:
                features = None

            detections.append({
                "bbox": bbox,
                "confidence": float(score),
                "raw_id": raw_id,
                "class_id": int(class_id),
                "class_name": class_name,
                "features": features
            })

        return detections
