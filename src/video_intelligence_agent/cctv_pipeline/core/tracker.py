from __future__ import annotations

from dataclasses import dataclass, field

from video_intelligence_agent.cctv_pipeline.models import Detection, TrackedPerson, TrackingUpdate
from video_intelligence_agent.cctv_pipeline.utils.error_handler import TrackingError
from video_intelligence_agent.cctv_pipeline.utils.logger import get_pipeline_logger


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    first_seen_seconds: float
    last_seen_seconds: float
    lost_frames: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)
    identity: str = "unknown"
    identity_confidence: float = 0.0
    known: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_public(self) -> TrackedPerson:
        return TrackedPerson(
            track_id=self.track_id,
            bbox=self.bbox,
            confidence=self.confidence,
            first_seen_frame=self.first_seen_frame,
            last_seen_frame=self.last_seen_frame,
            first_seen_seconds=self.first_seen_seconds,
            last_seen_seconds=self.last_seen_seconds,
            lost_frames=self.lost_frames,
            identity=self.identity,
            identity_confidence=self.identity_confidence,
            known=self.known,
            history=list(self.history),
            metadata=dict(self.metadata),
        )


class MultiObjectTracker:
    """ByteTrack-style tracker that falls back to a built-in IoU tracker."""

    def __init__(
        self,
        *,
        backend: str = "bytetrack",
        max_lost: int = 20,
        iou_threshold: float = 0.3,
    ) -> None:
        self.backend = backend
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.logger = get_pipeline_logger("tracker")
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1
        self._warned_fallback = False

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
        timestamp_seconds: float,
    ) -> TrackingUpdate:
        try:
            return self._update_iou_fallback(
                detections,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )
        except Exception as exc:
            raise TrackingError(
                "Tracker update failed.",
                module="tracker",
                frame_index=frame_index,
                cause=exc,
            ) from exc

    def apply_identity_updates(self, tracks: list[TrackedPerson]) -> None:
        for track in tracks:
            state = self._tracks.get(track.track_id)
            if state is None:
                continue
            state.identity = track.identity
            state.identity_confidence = track.identity_confidence
            state.known = track.known
            state.metadata = dict(track.metadata)

    def _update_iou_fallback(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
        timestamp_seconds: float,
    ) -> TrackingUpdate:
        if self.backend == "bytetrack" and not self._warned_fallback:
            self.logger.warning(
                "ByteTrack backend not found. Falling back to the built-in IoU tracker."
            )
            self._warned_fallback = True

        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()
        visible_tracks: list[TrackedPerson] = []
        lost_tracks: list[TrackedPerson] = []

        candidate_pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                score = _iou(track.bbox, detection.bbox)
                if score >= self.iou_threshold:
                    candidate_pairs.append((score, track_id, detection_index))
        candidate_pairs.sort(reverse=True)

        for _, track_id, detection_index in candidate_pairs:
            if track_id in matched_track_ids or detection_index in matched_detection_indices:
                continue
            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)
            detection = detections[detection_index]
            state = self._tracks[track_id]
            state.bbox = detection.bbox
            state.confidence = detection.confidence
            state.last_seen_frame = frame_index
            state.last_seen_seconds = timestamp_seconds
            state.lost_frames = 0
            state.history.append(state.center)
            visible_tracks.append(state.to_public())

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            state = _TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                first_seen_frame=frame_index,
                last_seen_frame=frame_index,
                first_seen_seconds=timestamp_seconds,
                last_seen_seconds=timestamp_seconds,
            )
            state.history.append(state.center)
            self._tracks[track_id] = state
            visible_tracks.append(state.to_public())

        expired_track_ids: list[int] = []
        for track_id, state in self._tracks.items():
            if track_id in matched_track_ids:
                continue
            state.lost_frames += 1
            if state.lost_frames > self.max_lost:
                lost_tracks.append(state.to_public())
                expired_track_ids.append(track_id)

        for track_id in expired_track_ids:
            self._tracks.pop(track_id, None)

        return TrackingUpdate(visible_tracks=visible_tracks, lost_tracks=lost_tracks)


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(inter_x2 - inter_x1, 0)
    inter_height = max(inter_y2 - inter_y1, 0)
    intersection = inter_width * inter_height
    if intersection == 0:
        return 0.0

    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union
