from __future__ import annotations
from dataclasses import dataclass
import cv2
from config import Config
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    label: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    center: tuple[float, float]


class PieceDetector:
    """Wykrywa figury modelu YOLO i zwraca prostokąty wraz z centrem."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or Config.MODEL_PATH
        self.model = YOLO(self.model_path)

    def detect(self, frame):
        results = self.model(
            frame,
            conf=Config.CONF,
            iou=Config.IOU,               # niższy próg = agresywniej usuwa nakładające się ramki (domyślnie 0.7)
            agnostic_nms=True,     # porównuje ramki między klasami, zostaje ta z większą pewnością
            verbose=False,
        )[0]
        detections: list[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls_id]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            detections.append(
                Detection(
                    class_id=cls_id,
                    label=label,
                    conf=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    center=center,
                )
            )
        return detections

    def draw(self, frame, detections):
        for detection in detections:
            cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{detection.label} {detection.conf:.2f}",
                (detection.x1, max(detection.y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        return frame
