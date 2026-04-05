from __future__ import annotations

from video_intelligence_agent.cctv_pipeline.core.event_logic import RuleBasedEventDetector
from video_intelligence_agent.cctv_pipeline.models import EventRecord, TrackedPerson


class RuleEventEngine:
    """Rule-based event engine for entry, exit, loitering, and repeated-presence logic."""

    def __init__(
        self,
        *,
        loitering_seconds: float,
        loitering_radius_px: float,
        border_margin_ratio: float,
    ) -> None:
        self._engine = RuleBasedEventDetector(
            loitering_seconds=loitering_seconds,
            loitering_radius_px=loitering_radius_px,
            border_margin_ratio=border_margin_ratio,
        )

    def reset(self) -> None:
        self._engine.reset()

    def update(
        self,
        *,
        visible_tracks: list[TrackedPerson],
        lost_tracks: list[TrackedPerson],
        frame_shape: tuple[int, ...],
    ) -> list[EventRecord]:
        return self._engine.update(
            visible_tracks=visible_tracks,
            lost_tracks=lost_tracks,
            frame_shape=frame_shape,
        )

    def flush(self) -> list[EventRecord]:
        return self._engine.flush()
