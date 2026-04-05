from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from video_intelligence_agent.image_io import GenericImageArray
from video_intelligence_agent.types import EmbeddingVector


def _metadata_factory() -> dict[str, Any]:
    return {}


@dataclass(slots=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass(slots=True)
class DetectedFace:
    bbox: BoundingBox
    crop: GenericImageArray
    detection_confidence: float = 0.0


@dataclass(slots=True)
class MatchResult:
    name: str
    confidence: float
    bbox: BoundingBox | None = None
    timestamp: str | None = None
    saved_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=_metadata_factory)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "confidence": round(float(self.confidence), 4),
        }
        if self.bbox is not None:
            payload["bbox"] = self.bbox.to_dict()
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        if self.saved_path is not None:
            payload["saved_path"] = self.saved_path
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(slots=True)
class StoredEmbedding:
    name: str
    embedding: EmbeddingVector
    source_image: str | None = None
    metadata: dict[str, Any] = field(default_factory=_metadata_factory)

