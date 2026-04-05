from __future__ import annotations

from video_intelligence_agent.cctv_pipeline.core.motion_detector import MotionDetector
from video_intelligence_agent.cctv_pipeline.models import MotionResult


class MotionPreprocessingService:
    """Preprocessing stage that filters inactive frames before heavier inference."""

    def __init__(self, *, motion_threshold: float, min_motion_area: int) -> None:
        self._detector = MotionDetector(
            motion_threshold=motion_threshold,
            min_motion_area=min_motion_area,
        )

    def reset(self) -> None:
        self._detector.reset()

    def analyze(self, frame, *, frame_index: int) -> MotionResult:
        return self._detector.analyze(frame, frame_index=frame_index)
