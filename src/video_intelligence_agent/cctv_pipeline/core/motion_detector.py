from __future__ import annotations

from typing import TYPE_CHECKING

from video_intelligence_agent.cctv_pipeline.models import MotionResult
from video_intelligence_agent.cctv_pipeline.utils.error_handler import MotionDetectionError
from video_intelligence_agent.cctv_pipeline.utils.logger import get_pipeline_logger

if TYPE_CHECKING:
    import numpy as np


class MotionDetector:
    """OpenCV-based frame differencing motion detector."""

    def __init__(self, *, motion_threshold: float, min_motion_area: int) -> None:
        self.motion_threshold = motion_threshold
        self.min_motion_area = min_motion_area
        self.logger = get_pipeline_logger("motion_detector")
        self._cv2 = self._load_cv2()
        self._previous_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_gray = None

    def analyze(self, frame: "np.ndarray", *, frame_index: int) -> MotionResult:
        if self._cv2 is None:
            raise MotionDetectionError(
                "OpenCV is required for motion detection.",
                module="motion_detector",
                frame_index=frame_index,
            )

        try:
            gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
            gray = self._cv2.GaussianBlur(gray, (21, 21), 0)
        except Exception as exc:
            raise MotionDetectionError(
                "Unable to preprocess frame for motion detection.",
                module="motion_detector",
                frame_index=frame_index,
                cause=exc,
            ) from exc

        if self._previous_gray is None:
            self._previous_gray = gray
            return MotionResult(active=False, score=0.0, boxes=[])

        try:
            delta = self._cv2.absdiff(self._previous_gray, gray)
            _, threshold_frame = self._cv2.threshold(delta, 25, 255, self._cv2.THRESH_BINARY)
            dilated = self._cv2.dilate(threshold_frame, None, iterations=2)
            contours, _ = self._cv2.findContours(
                dilated,
                self._cv2.RETR_EXTERNAL,
                self._cv2.CHAIN_APPROX_SIMPLE,
            )
        except Exception as exc:
            raise MotionDetectionError(
                "OpenCV failed while computing motion contours.",
                module="motion_detector",
                frame_index=frame_index,
                cause=exc,
            ) from exc
        finally:
            self._previous_gray = gray

        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            area = self._cv2.contourArea(contour)
            if area < self.min_motion_area:
                continue
            x, y, w, h = self._cv2.boundingRect(contour)
            boxes.append((int(x), int(y), int(x + w), int(y + h)))

        score = (
            float(threshold_frame.sum()) / float(threshold_frame.size * 255)
            if threshold_frame.size
            else 0.0
        )
        return MotionResult(
            active=bool(boxes or score >= self.motion_threshold),
            score=score,
            boxes=boxes,
        )

    @staticmethod
    def _load_cv2():
        try:
            import cv2
        except ImportError:  # pragma: no cover - depends on local environment
            return None
        return cv2
