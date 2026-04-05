from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from video_intelligence_agent.image_io import GenericImageArray
from video_intelligence_agent.models import BoundingBox


def _metadata_factory() -> dict[str, Any]:
    return {}


@dataclass(slots=True)
class VideoMetadata:
    video_path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_seconds: float
    recorded_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "video_path": self.video_path,
            "fps": round(self.fps, 4),
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration_seconds": round(self.duration_seconds, 4),
            "recorded_at": self.recorded_at,
        }


@dataclass(slots=True)
class FramePacket:
    frame_index: int
    timestamp_seconds: float
    frame: GenericImageArray


@dataclass(slots=True)
class MotionAnalysis:
    active: bool
    motion_score: float
    boxes: list[BoundingBox] = field(default_factory=list)
    total_area: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "motion_score": round(self.motion_score, 6),
            "total_area": round(self.total_area, 2),
            "boxes": [box.to_dict() for box in self.boxes],
        }


@dataclass(slots=True)
class PersonObservation:
    name: str
    confidence: float
    known: bool
    bbox: BoundingBox | None = None
    track_id: str | None = None
    tracked_duration_seconds: float = 0.0
    snapshot_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=_metadata_factory)
    face_crop: GenericImageArray | None = None

    def display_name(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "known": self.known,
            "track_id": self.track_id,
            "tracked_duration_seconds": round(self.tracked_duration_seconds, 2),
        }
        if self.bbox is not None:
            payload["bbox"] = self.bbox.to_dict()
        if self.snapshot_path is not None:
            payload["snapshot_path"] = self.snapshot_path
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(slots=True)
class ActivityRecord:
    event_id: str
    start_time_seconds: float
    end_time_seconds: float
    start_frame_index: int
    end_frame_index: int
    motion_score_mean: float
    people: list[PersonObservation] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    event_category: str = "routine_motion"
    alert_level: str = "info"
    alert_reasons: list[str] = field(default_factory=list)
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    clip_path: str | None = None
    unknown_snapshot_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=_metadata_factory)

    @property
    def duration_seconds(self) -> float:
        return max(self.end_time_seconds - self.start_time_seconds, 0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "start_time_seconds": round(self.start_time_seconds, 2),
            "end_time_seconds": round(self.end_time_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "motion_score_mean": round(self.motion_score_mean, 6),
            "people": [person.to_dict() for person in self.people],
            "actions": self.actions,
            "event_category": self.event_category,
            "alert_level": self.alert_level,
            "alert_reasons": self.alert_reasons,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "clip_path": self.clip_path,
            "unknown_snapshot_paths": self.unknown_snapshot_paths,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class DailySummary:
    generated_at: str
    total_activity_duration: float
    total_events: int
    total_people_detected: int
    known_people: list[str] = field(default_factory=list)
    unknown_people_count: int = 0
    key_events: list[str] = field(default_factory=list)
    summary_text: str = ""

    @classmethod
    def empty(cls) -> "DailySummary":
        return cls(
            generated_at=datetime.now().isoformat(),
            total_activity_duration=0.0,
            total_events=0,
            total_people_detected=0,
            summary_text="No relevant CCTV activity was detected.",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "total_activity_duration": round(self.total_activity_duration, 2),
            "total_events": self.total_events,
            "total_people_detected": self.total_people_detected,
            "known_people": self.known_people,
            "unknown_people_count": self.unknown_people_count,
            "key_events": self.key_events,
            "summary_text": self.summary_text,
        }


@dataclass(slots=True)
class VideoAnalysisResult:
    metadata: VideoMetadata
    activities: list[ActivityRecord]
    summary: DailySummary
    storage_reduction_ratio: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata": self.metadata.to_dict(),
            "activities": [activity.to_dict() for activity in self.activities],
            "summary": self.summary.to_dict(),
            "storage_reduction_ratio": round(self.storage_reduction_ratio, 4),
        }

