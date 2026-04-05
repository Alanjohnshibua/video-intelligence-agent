from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from video_intelligence_agent.cctv_pipeline.utils.error_handler import EventStorageError
from video_intelligence_agent.cctv_pipeline.utils.logger import get_pipeline_logger

if TYPE_CHECKING:
    import numpy as np


class ClipManager:
    """Keeps a rolling frame buffer and writes clips or debug frames on demand."""

    def __init__(
        self,
        *,
        clips_dir: Path | str,
        debug_dir: Path | str,
        codec: str,
        fps: float,
        pre_event_seconds: float = 2.0,
    ) -> None:
        self.clips_dir = Path(clips_dir)
        self.debug_dir = Path(debug_dir)
        self.codec = codec
        self.fps = max(fps, 1.0)
        self.logger = get_pipeline_logger("clip_manager")
        self._cv2 = self._load_cv2()
        self._buffer: deque[np.ndarray] = deque(maxlen=max(int(self.fps * pre_event_seconds), 1))

        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def buffer_frame(self, frame: "np.ndarray") -> None:
        self._buffer.append(frame.copy())

    def save_clip_from_buffer(self, clip_name: str) -> str | None:
        if not self._buffer:
            return None
        return self.save_clip(clip_name=clip_name, frames=list(self._buffer))

    def save_clip(self, *, clip_name: str, frames: list["np.ndarray"]) -> str | None:
        if not frames:
            return None
        if self._cv2 is None:
            self.logger.warning("OpenCV is unavailable, clip %s was not written.", clip_name)
            return None

        height, width = frames[0].shape[:2]
        path = self.clips_dir / f"{clip_name}.mp4"
        writer = self._cv2.VideoWriter(
            str(path),
            self._cv2.VideoWriter_fourcc(*self.codec),
            self.fps,
            (width, height),
        )

        if not writer.isOpened():
            raise EventStorageError(
                f"Unable to create video writer for {path}",
                module="clip_manager",
            )

        try:
            for frame in frames:
                writer.write(frame)
        except Exception as exc:  # pragma: no cover - depends on local OpenCV backend
            raise EventStorageError(
                f"Failed while writing clip {path}",
                module="clip_manager",
                cause=exc,
            ) from exc
        finally:
            writer.release()
        return str(path)

    def save_debug_frame(self, frame: "np.ndarray", *, frame_index: int) -> str | None:
        if self._cv2 is None:
            return None
        path = self.debug_dir / f"frame-{frame_index:06d}.jpg"
        written = self._cv2.imwrite(str(path), frame)
        if not written:
            raise EventStorageError(
                f"Failed to save debug frame to {path}",
                module="clip_manager",
                frame_index=frame_index,
            )
        return str(path)

    @staticmethod
    def _load_cv2():
        try:
            import cv2
        except ImportError:  # pragma: no cover - environment dependent
            return None
        return cv2
