from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

from video_intelligence_agent.cctv_pipeline.models import TrackedPerson
from video_intelligence_agent.cctv_pipeline.utils.error_handler import RecognitionError
from video_intelligence_agent.cctv_pipeline.utils.logger import get_pipeline_logger

if TYPE_CHECKING:
    import numpy as np


class FaceIdentifierProtocol(Protocol):
    """Minimal interface expected from a face identifier implementation."""

    def identify_face(self, image: "np.ndarray") -> Any: ...


class FaceRecognitionService:
    """Assigns stable known or unknown identities to tracked people."""

    def __init__(
        self,
        *,
        identifier: FaceIdentifierProtocol | None = None,
        unknown_label_prefix: str = "unknown",
    ) -> None:
        self.identifier = identifier
        self.unknown_label_prefix = unknown_label_prefix
        self.logger = get_pipeline_logger("recognition")
        self._unknown_index = 0
        self._unknown_by_track: dict[int, str] = {}

    def identify_tracks(
        self,
        frame: "np.ndarray",
        tracks: list[TrackedPerson],
        *,
        frame_index: int,
    ) -> list[TrackedPerson]:
        identified: list[TrackedPerson] = []
        for track in tracks:
            try:
                identified.append(self._identify_single(frame, track, frame_index=frame_index))
            except RecognitionError as exc:
                self.logger.warning(str(exc))
                track.identity = self._unknown_label_for_track(track.track_id)
                track.identity_confidence = 0.0
                track.known = False
                track.metadata["recognition_error"] = str(exc)
                identified.append(track)
        return identified

    def _identify_single(
        self,
        frame: "np.ndarray",
        track: TrackedPerson,
        *,
        frame_index: int,
    ) -> TrackedPerson:
        if self.identifier is None:
            track.identity = self._unknown_label_for_track(track.track_id)
            track.identity_confidence = 0.0
            track.known = False
            return track

        crop = _crop_frame(frame, track.bbox)
        if crop is None:
            raise RecognitionError(
                "Invalid crop bounds for face identification.",
                module="recognition",
                frame_index=frame_index,
                track_id=track.track_id,
            )

        try:
            result = self.identifier.identify_face(crop)
        except Exception as exc:
            raise RecognitionError(
                "Face recognition failed. Marking person as unknown.",
                module="recognition",
                frame_index=frame_index,
                track_id=track.track_id,
                cause=exc,
            ) from exc

        name = str(getattr(result, "name", "Unknown"))
        confidence = float(getattr(result, "confidence", 0.0))
        known = name.lower() != "unknown"
        if known:
            track.identity = name
            track.identity_confidence = confidence
            track.known = True
        else:
            track.identity = self._unknown_label_for_track(track.track_id)
            track.identity_confidence = confidence
            track.known = False
        return track

    def _unknown_label_for_track(self, track_id: int) -> str:
        label = self._unknown_by_track.get(track_id)
        if label is not None:
            return label
        self._unknown_index += 1
        label = f"{self.unknown_label_prefix}_{self._unknown_index}"
        self._unknown_by_track[track_id] = label
        return label


def _crop_frame(
    frame: "np.ndarray",
    bbox: tuple[int, int, int, int],
) -> "np.ndarray | None":
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop
