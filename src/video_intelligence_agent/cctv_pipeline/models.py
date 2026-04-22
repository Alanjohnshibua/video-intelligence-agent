from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_intelligence_agent.image_io import GenericImageArray


def _metadata_factory() -> dict[str, Any]:
    return {}


@dataclass(slots=True)
class MotionResult:
    """Result returned by the motion detector for a single frame."""

    active: bool
    score: float
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "score": round(float(self.score), 6),
            "boxes": [list(box) for box in self.boxes],
        }


@dataclass(slots=True)
class Detection:
    """Normalized detection produced by the person detector."""

    bbox: tuple[int, int, int, int]
    confidence: float
    label: str = "person"
    metadata: dict[str, Any] = field(default_factory=_metadata_factory)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TrackedPerson:
    """Track state exposed by the tracker to downstream modules."""

    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    first_seen_seconds: float
    last_seen_seconds: float
    lost_frames: int = 0
    identity: str = "unknown"
    identity_confidence: float = 0.0
    known: bool = False
    history: list[tuple[float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=_metadata_factory)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def duration_seconds(self) -> float:
        return max(self.last_seen_seconds - self.first_seen_seconds, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "first_seen_seconds": round(float(self.first_seen_seconds), 3),
            "last_seen_seconds": round(float(self.last_seen_seconds), 3),
            "lost_frames": self.lost_frames,
            "identity": self.identity,
            "identity_confidence": round(float(self.identity_confidence), 4),
            "known": self.known,
            "history": [[round(x, 2), round(y, 2)] for x, y in self.history],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TrackingUpdate:
    """Visible and lost tracks emitted by the tracker per frame."""

    visible_tracks: list[TrackedPerson] = field(default_factory=list)
    lost_tracks: list[TrackedPerson] = field(default_factory=list)


@dataclass(slots=True)
class EventRecord:
    """Structured event written to disk and returned to callers."""

    event_id: str
    person_id: str
    action: str
    start_time: str
    end_time: str
    duration_seconds: float
    frame_index: int
    track_id: int | None = None
    clip_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=_metadata_factory)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "person_id": self.person_id,
            "action": self.action,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(float(self.duration_seconds), 3),
            "frame_index": self.frame_index,
            "metadata": self.metadata,
        }
        if self.track_id is not None:
            payload["track_id"] = self.track_id
        if self.clip_path is not None:
            payload["clip_path"] = self.clip_path
        return payload


@dataclass(slots=True)
class VideoMetadata:
    """Metadata for the input video source."""

    video_path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "fps": round(float(self.fps), 3),
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration_seconds": round(float(self.duration_seconds), 3),
        }


@dataclass(slots=True)
class FramePacket:
    """Frame container used by the processor and tests."""

    frame_index: int
    timestamp_seconds: float
    frame: GenericImageArray


@dataclass(slots=True)
class PipelineStats:
    """Aggregated counters shown in the terminal summary."""

    total_frames_read: int = 0
    processed_frames: int = 0
    motion_frames: int = 0
    skipped_frames: int = 0
    detections: int = 0
    tracked_objects: int = 0
    events_detected: int = 0
    corrupted_frames: int = 0
    errors_encountered: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_frames_read": self.total_frames_read,
            "processed_frames": self.processed_frames,
            "motion_frames": self.motion_frames,
            "skipped_frames": self.skipped_frames,
            "detections": self.detections,
            "tracked_objects": self.tracked_objects,
            "events_detected": self.events_detected,
            "corrupted_frames": self.corrupted_frames,
            "errors_encountered": self.errors_encountered,
        }


@dataclass(slots=True)
class PipelineArtifacts:
    """Filesystem artifacts produced by a pipeline run."""

    events_path: Path
    clips_dir: Path
    analysis_path: Path | None = None
    summary_path: Path | None = None
    debug_dir: Path | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "events_path": str(self.events_path),
            "clips_dir": str(self.clips_dir),
            "analysis_path": str(self.analysis_path) if self.analysis_path is not None else None,
            "summary_path": str(self.summary_path) if self.summary_path is not None else None,
            "debug_dir": str(self.debug_dir) if self.debug_dir is not None else None,
        }
