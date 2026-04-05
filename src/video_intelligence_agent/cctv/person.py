from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from video_intelligence_agent.core import FaceIdentifier
from video_intelligence_agent.cctv.models import PersonObservation
from video_intelligence_agent.image_io import GenericImageArray
from video_intelligence_agent.models import BoundingBox


class PersonRecognizerProtocol(Protocol):
    def recognize(self, frame: GenericImageArray) -> list[PersonObservation]: ...


class FaceIdentifierRecognizer:
    """Adapter that reuses the existing FaceIdentifier for CCTV person recognition."""

    def __init__(
        self,
        identifier: FaceIdentifier,
        *,
        unknown_person_label: str = "Unknown Person",
    ) -> None:
        self.identifier = identifier
        self.unknown_person_label = unknown_person_label

    def recognize(self, frame: GenericImageArray) -> list[PersonObservation]:
        observations: list[PersonObservation] = []
        for detected in self.identifier.detect_faces(frame):
            embedding = self.identifier.get_embedding(detected.crop)
            matched = self.identifier.match_face(embedding)
            known = matched.name != "Unknown"
            observations.append(
                PersonObservation(
                    name=matched.name if known else self.unknown_person_label,
                    confidence=float(matched.confidence),
                    known=known,
                    bbox=detected.bbox,
                    metadata={"source": "video_intelligence_agent"},
                    face_crop=detected.crop,
                )
            )
        return observations


@dataclass(slots=True)
class _TrackState:
    track_id: str
    name: str
    known: bool
    first_seen: float
    last_seen: float
    last_bbox: BoundingBox | None
    max_displacement: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


class SimpleTrackManager:
    """A lightweight IoU-based tracker suitable for low-resource demos."""

    def __init__(self, *, iou_threshold: float = 0.2) -> None:
        self.iou_threshold = iou_threshold
        self._tracks: dict[str, _TrackState] = {}
        self._next_track_index = 1

    def reset(self) -> None:
        self._tracks.clear()

    def update(
        self,
        timestamp_seconds: float,
        observations: list[PersonObservation],
    ) -> list[PersonObservation]:
        remaining_track_ids = set(self._tracks.keys())
        for observation in observations:
            matched_id = self._match_track(observation, remaining_track_ids)
            if matched_id is None:
                matched_id = f"track-{self._next_track_index:04d}"
                self._next_track_index += 1
                self._tracks[matched_id] = _TrackState(
                    track_id=matched_id,
                    name=observation.name,
                    known=observation.known,
                    first_seen=timestamp_seconds,
                    last_seen=timestamp_seconds,
                    last_bbox=observation.bbox,
                    metadata={"is_new_track": True},
                )
            track = self._tracks[matched_id]
            movement_px = self._center_distance(track.last_bbox, observation.bbox)
            track.max_displacement = max(track.max_displacement, movement_px)
            track.last_bbox = observation.bbox
            track.last_seen = timestamp_seconds
            track.name = observation.name
            track.known = observation.known
            remaining_track_ids.discard(matched_id)

            observation.track_id = matched_id
            observation.tracked_duration_seconds = max(
                timestamp_seconds - track.first_seen,
                0.0,
            )
            observation.metadata = {
                **observation.metadata,
                "movement_px": round(track.max_displacement, 2),
                "is_new_track": bool(track.metadata.pop("is_new_track", False)),
            }

        return observations

    def _match_track(
        self,
        observation: PersonObservation,
        candidate_ids: set[str],
    ) -> str | None:
        best_track_id: str | None = None
        best_score = 0.0
        for track_id in candidate_ids:
            track = self._tracks[track_id]
            if track.known != observation.known:
                continue
            if (observation.known or track.known) and track.name != observation.name:
                continue
            iou_score = self._iou(track.last_bbox, observation.bbox)
            if iou_score >= self.iou_threshold and iou_score > best_score:
                best_track_id = track_id
                best_score = iou_score
                continue

            distance = self._center_distance(track.last_bbox, observation.bbox)
            if distance <= self._max_match_distance(track.last_bbox, observation.bbox):
                distance_score = 1.0 / (1.0 + distance)
                if distance_score > best_score:
                    best_track_id = track_id
                    best_score = distance_score
        return best_track_id

    @staticmethod
    def _iou(a: BoundingBox | None, b: BoundingBox | None) -> float:
        if a is None or b is None:
            return 0.0

        ax2 = a.x + a.w
        ay2 = a.y + a.h
        bx2 = b.x + b.w
        by2 = b.y + b.h

        inter_left = max(a.x, b.x)
        inter_top = max(a.y, b.y)
        inter_right = min(ax2, bx2)
        inter_bottom = min(ay2, by2)
        if inter_right <= inter_left or inter_bottom <= inter_top:
            return 0.0

        inter_area = float((inter_right - inter_left) * (inter_bottom - inter_top))
        union_area = float(a.w * a.h + b.w * b.h) - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _center_distance(a: BoundingBox | None, b: BoundingBox | None) -> float:
        if a is None or b is None:
            return 0.0
        ax = a.x + a.w / 2.0
        ay = a.y + a.h / 2.0
        bx = b.x + b.w / 2.0
        by = b.y + b.h / 2.0
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    @staticmethod
    def _max_match_distance(a: BoundingBox | None, b: BoundingBox | None) -> float:
        if a is None or b is None:
            return 0.0
        reference = max(a.w, a.h, b.w, b.h)
        return max(reference * 4.0, 64.0)

