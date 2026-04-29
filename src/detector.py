from typing import Dict, List, Optional, Tuple

from ultralytics import YOLO

from src.config import AppConfig

class YOLODetector:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = YOLO(config.model_path)
        self.names = self.model.names

    def track(self, frame) -> Tuple[List[Dict], Optional[object]]:
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.config.confidence,
            iou=self.config.iou,
            device=self.config.device,
            imgsz=self.config.imgsz,
            verbose=False,
        )

        if not results:
            return [], None

        result = results[0]
        detections: List[Dict] = []

        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return detections, result.plot()

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)

        for i, box in enumerate(xyxy):
            cls_id = int(clss[i]) if len(clss) > i else -1
            raw_name = self.names.get(cls_id, "other") if isinstance(self.names, dict) else str(cls_id)
            class_name = raw_name.lower()
            track_id = int(ids[i]) if len(ids) > i else -1
            x1, y1, x2, y2 = map(int, box.tolist())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append(
                {
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "raw_name": raw_name,
                    "conf": float(confs[i]) if len(confs) > i else 0.0,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                }
            )

        return detections, None
