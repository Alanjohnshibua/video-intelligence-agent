from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast

from video_intelligence_agent.cctv.config import CCTVAgentConfig
from video_intelligence_agent.cctv.models import MotionAnalysis
from video_intelligence_agent.image_io import GenericImageArray
from video_intelligence_agent.models import BoundingBox


class OpenCVModuleProtocol(Protocol):
    COLOR_BGR2GRAY: int
    THRESH_BINARY: int
    RETR_EXTERNAL: int
    CHAIN_APPROX_SIMPLE: int

    def cvtColor(self, src: GenericImageArray, code: int) -> GenericImageArray: ...

    def GaussianBlur(
        self,
        src: GenericImageArray,
        ksize: tuple[int, int],
        sigmaX: float,
    ) -> GenericImageArray: ...

    def absdiff(self, src1: GenericImageArray, src2: GenericImageArray) -> GenericImageArray: ...

    def threshold(
        self,
        src: GenericImageArray,
        thresh: float,
        maxval: float,
        type: int,
    ) -> tuple[float, GenericImageArray]: ...

    def dilate(
        self,
        src: GenericImageArray,
        kernel: object | None,
        iterations: int,
    ) -> GenericImageArray: ...

    def findContours(
        self,
        image: GenericImageArray,
        mode: int,
        method: int,
    ) -> tuple[list[object], object]: ...

    def contourArea(self, contour: object) -> float: ...

    def boundingRect(self, contour: object) -> tuple[int, int, int, int]: ...


def _load_cv2() -> OpenCVModuleProtocol | None:
    try:
        module = import_module("cv2")
    except ImportError:  # pragma: no cover - depends on local environment
        return None
    return cast(OpenCVModuleProtocol, module)


class MotionDetector:
    """Lightweight frame-difference motion detector for offline CCTV filtering."""

    def __init__(self, config: CCTVAgentConfig) -> None:
        self.config = config
        self._previous_gray: GenericImageArray | None = None

    def reset(self) -> None:
        self._previous_gray = None

    def analyze(self, frame: GenericImageArray) -> MotionAnalysis:
        cv2_module = _load_cv2()
        if cv2_module is None:
            raise RuntimeError("opencv-python is required for motion detection.")

        gray = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2GRAY)
        blurred = cv2_module.GaussianBlur(gray, (21, 21), 0)

        if self._previous_gray is None:
            self._previous_gray = blurred
            return MotionAnalysis(active=False, motion_score=0.0)

        delta = cv2_module.absdiff(self._previous_gray, blurred)
        _threshold, thresh = cv2_module.threshold(delta, 25, 255, cv2_module.THRESH_BINARY)
        thresh = cv2_module.dilate(thresh, None, iterations=2)
        contours, _hierarchy = cv2_module.findContours(
            thresh,
            cv2_module.RETR_EXTERNAL,
            cv2_module.CHAIN_APPROX_SIMPLE,
        )

        boxes: list[BoundingBox] = []
        total_area = 0.0
        for contour in contours:
            area = float(cv2_module.contourArea(contour))
            if area < float(self.config.min_motion_area):
                continue
            x, y, w, h = cv2_module.boundingRect(contour)
            boxes.append(BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h)))
            total_area += area

        frame_area = float(frame.shape[0] * frame.shape[1]) if frame.size else 1.0
        motion_score = total_area / frame_area
        self._previous_gray = blurred
        return MotionAnalysis(
            active=bool(boxes) and motion_score >= self.config.motion_threshold,
            motion_score=motion_score,
            boxes=boxes,
            total_area=total_area,
        )

