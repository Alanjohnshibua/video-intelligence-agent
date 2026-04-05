from __future__ import annotations

from typing import TYPE_CHECKING

from video_intelligence_agent.cctv_pipeline.models import Detection
from video_intelligence_agent.cctv_pipeline.utils.error_handler import DetectionError
from video_intelligence_agent.cctv_pipeline.utils.logger import get_pipeline_logger

if TYPE_CHECKING:
    import numpy as np


class YOLOPersonDetector:
    """YOLOv8 Nano person detector with graceful startup failure handling."""

    def __init__(
        self,
        *,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.logger = get_pipeline_logger("detector")
        self._model = None
        self._load_error: Exception | None = None
        self._load_model()

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Exception | None:
        return self._load_error

    def detect(self, frame: "np.ndarray", *, frame_index: int) -> list[Detection]:
        if self._model is None:
            raise DetectionError(
                "YOLOv8 Nano model is unavailable.",
                module="detector",
                frame_index=frame_index,
                cause=self._load_error,
                context={"model_path": self.model_path},
            )

        try:
            results = self._model.predict(
                source=frame,
                verbose=False,
                conf=self.confidence_threshold,
                device=self.device,
                classes=[0],
            )
        except Exception as exc:
            raise DetectionError(
                "YOLO inference failed.",
                module="detector",
                frame_index=frame_index,
                cause=exc,
                context={"model_path": self.model_path},
            ) from exc

        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        detections: list[Detection] = []
        xyxy_values = boxes.xyxy.tolist() if getattr(boxes, "xyxy", None) is not None else []
        confidence_values = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
        for bbox_values, confidence in zip(xyxy_values, confidence_values):
            x1, y1, x2, y2 = bbox_values
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(confidence),
                    label="person",
                )
            )
        return detections

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on local environment
            self._load_error = exc
            return

        try:
            self._model = YOLO(self.model_path)
        except Exception as exc:  # pragma: no cover - model file availability varies
            self._load_error = exc
