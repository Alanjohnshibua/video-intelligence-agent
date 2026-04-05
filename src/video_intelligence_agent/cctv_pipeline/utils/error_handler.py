from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any


class BaseAppError(Exception):
    """Base application error with pipeline context for cleaner logs."""

    def __init__(
        self,
        message: str,
        *,
        module: str,
        frame_index: int | None = None,
        track_id: int | None = None,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.module = module
        self.frame_index = frame_index
        self.track_id = track_id
        self.cause = cause
        self.context = context or {}
        super().__init__(self._build_message(message))

    def _build_message(self, message: str) -> str:
        parts = [message]
        if self.frame_index is not None:
            parts.append(f"frame={self.frame_index}")
        if self.track_id is not None:
            parts.append(f"track={self.track_id}")
        for key, value in self.context.items():
            parts.append(f"{key}={value}")
        return " | ".join(parts)


class ConfigurationError(BaseAppError):
    """Raised when runtime configuration is invalid."""


class VideoInputError(BaseAppError):
    """Raised when the video file cannot be opened or read."""


class MotionDetectionError(BaseAppError):
    """Raised when motion analysis fails for a frame."""


class DetectionError(BaseAppError):
    """Raised when person detection fails."""


class TrackingError(BaseAppError):
    """Raised when multi-object tracking fails."""


class RecognitionError(BaseAppError):
    """Raised when face identification fails."""


class EventStorageError(BaseAppError):
    """Raised when event data or clips cannot be written."""


@dataclass(slots=True)
class ErrorTracker:
    """Collects non-fatal errors so the processor can summarize them later."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.entries)

    def add(
        self,
        *,
        module: str,
        message: str,
        frame_index: int | None = None,
        level: str = "ERROR",
    ) -> None:
        self.entries.append(
            {
                "module": module,
                "message": message,
                "frame_index": frame_index,
                "level": level,
            }
        )

    def add_exception(self, exc: BaseAppError | Exception, *, module: str) -> None:
        frame_index = exc.frame_index if isinstance(exc, BaseAppError) else None
        self.add(module=module, message=str(exc), frame_index=frame_index)

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.entries)


def log_exception(
    logger: logging.Logger,
    exc: BaseAppError | Exception,
    *,
    error_tracker: ErrorTracker | None = None,
    level: int = logging.ERROR,
    module: str | None = None,
) -> None:
    """Send exceptions through the logging system instead of printing raw tracebacks."""

    target_module = module or getattr(exc, "module", logger.name.rsplit(".", 1)[-1])
    logger.log(level, str(exc))
    if error_tracker is not None:
        error_tracker.add_exception(exc, module=target_module)
