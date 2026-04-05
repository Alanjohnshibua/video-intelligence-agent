from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from video_intelligence_agent.cctv.models import FramePacket, VideoMetadata
from video_intelligence_agent.image_io import GenericImageArray


class VideoIngestionError(Exception):
    pass


class VideoCaptureProtocol(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, GenericImageArray | None]: ...

    def get(self, prop_id: int) -> float: ...

    def release(self) -> None: ...


class OpenCVModuleProtocol(Protocol):
    CAP_PROP_FRAME_COUNT: int
    CAP_PROP_FRAME_WIDTH: int
    CAP_PROP_FRAME_HEIGHT: int
    CAP_PROP_FPS: int
    CAP_PROP_POS_MSEC: int

    def VideoCapture(self, filename: str) -> VideoCaptureProtocol: ...


def _load_cv2() -> OpenCVModuleProtocol | None:
    try:
        module = import_module("cv2")
    except ImportError:  # pragma: no cover - depends on local environment
        return None
    return cast(OpenCVModuleProtocol, module)


@dataclass(slots=True)
class VideoReader:
    video_path: str

    def metadata(self) -> VideoMetadata:
        cv2_module = _load_cv2()
        if cv2_module is None:
            raise VideoIngestionError("opencv-python is required for video ingestion.")

        capture = cv2_module.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")

        try:
            total_frames = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(capture.get(cv2_module.CAP_PROP_FPS) or 0.0)
            duration = total_frames / fps if fps > 0 else 0.0
            recorded_at = datetime.fromtimestamp(
                Path(self.video_path).stat().st_mtime
            ).isoformat()
            return VideoMetadata(
                video_path=self.video_path,
                fps=fps,
                total_frames=total_frames,
                width=width,
                height=height,
                duration_seconds=duration,
                recorded_at=recorded_at,
            )
        finally:
            capture.release()

    def iter_frames(self, *, frame_step: int = 1) -> list[FramePacket] | Any:
        cv2_module = _load_cv2()
        if cv2_module is None:
            raise VideoIngestionError("opencv-python is required for video ingestion.")

        capture = cv2_module.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")

        frame_index = 0
        step = max(frame_step, 1)
        fps = float(capture.get(cv2_module.CAP_PROP_FPS) or 0.0)

        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if frame_index % step == 0:
                    timestamp = (
                        frame_index / fps
                        if fps > 0
                        else float(capture.get(cv2_module.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                    )
                    yield FramePacket(
                        frame_index=frame_index,
                        timestamp_seconds=timestamp,
                        frame=frame,
                    )
                frame_index += 1
        finally:
            capture.release()

