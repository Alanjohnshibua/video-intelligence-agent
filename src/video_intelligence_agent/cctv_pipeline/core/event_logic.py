from __future__ import annotations

from dataclasses import dataclass

from video_intelligence_agent.cctv_pipeline.models import EventRecord, TrackedPerson


@dataclass(slots=True)
class _TrackEventState:
    track_id: int
    person_id: str
    first_seen_seconds: float
    last_seen_seconds: float
    first_seen_frame: int
    last_seen_frame: int
    first_center: tuple[float, float]
    last_center: tuple[float, float]
    loiter_anchor: tuple[float, float]
    loiter_start_seconds: float
    start_zone: str
    last_zone: str
    known: bool = False
    identity_confidence: float = 0.0
    entered_center: bool = False
    entering_emitted: bool = False
    loitering_emitted: bool = False


class RuleBasedEventDetector:
    """Generates entering, exiting, loitering, and movement events from tracks."""

    def __init__(
        self,
        *,
        loitering_seconds: float,
        loitering_radius_px: float,
        movement_min_distance_px: float,
        border_margin_ratio: float,
    ) -> None:
        self.loitering_seconds = loitering_seconds
        self.loitering_radius_px = loitering_radius_px
        self.movement_min_distance_px = movement_min_distance_px
        self.border_margin_ratio = border_margin_ratio
        self._states: dict[int, _TrackEventState] = {}
        self._event_index = 0

    def reset(self) -> None:
        self._states.clear()
        self._event_index = 0

    def update(
        self,
        *,
        visible_tracks: list[TrackedPerson],
        lost_tracks: list[TrackedPerson],
        frame_shape: tuple[int, ...],
    ) -> list[EventRecord]:
        height, width = frame_shape[:2]
        events: list[EventRecord] = []

        for track in visible_tracks:
            zone = _zone_for_point(
                track.center,
                width=width,
                height=height,
                margin_ratio=self.border_margin_ratio,
            )
            state = self._states.get(track.track_id)
            if state is None:
                state = _TrackEventState(
                    track_id=track.track_id,
                    person_id=track.identity,
                    first_seen_seconds=track.first_seen_seconds,
                    last_seen_seconds=track.last_seen_seconds,
                    first_seen_frame=track.first_seen_frame,
                    last_seen_frame=track.last_seen_frame,
                    first_center=track.center,
                    last_center=track.center,
                    loiter_anchor=track.center,
                    loiter_start_seconds=track.first_seen_seconds,
                    start_zone=zone,
                    last_zone=zone,
                    known=track.known,
                    identity_confidence=track.identity_confidence,
                    entered_center=(zone == "center"),
                )
                self._states[track.track_id] = state
            else:
                state.person_id = track.identity
                state.last_seen_seconds = track.last_seen_seconds
                state.last_seen_frame = track.last_seen_frame
                state.last_center = track.center
                state.last_zone = zone
                state.known = track.known
                state.identity_confidence = track.identity_confidence

            if zone == "center" and not state.entered_center:
                state.entered_center = True
                state.loiter_anchor = track.center
                state.loiter_start_seconds = track.last_seen_seconds

            if state.start_zone == "edge" and zone == "center" and not state.entering_emitted:
                events.append(
                    self._build_event(
                        action="entering",
                        track=track,
                        start_seconds=state.first_seen_seconds,
                        end_seconds=track.last_seen_seconds,
                        frame_index=track.last_seen_frame,
                        metadata={"start_zone": state.start_zone, "end_zone": zone},
                    )
                )
                state.entering_emitted = True

            if (
                not state.loitering_emitted
                and (track.last_seen_seconds - state.loiter_start_seconds) >= self.loitering_seconds
                and _distance(state.loiter_anchor, track.center) <= self.loitering_radius_px
            ):
                events.append(
                    self._build_event(
                        action="loitering",
                        track=track,
                        start_seconds=state.loiter_start_seconds,
                        end_seconds=track.last_seen_seconds,
                        frame_index=track.last_seen_frame,
                        metadata={
                            "radius_px": round(_distance(state.loiter_anchor, track.center), 2)
                        },
                    )
                )
                state.loitering_emitted = True

        for track in lost_tracks:
            state = self._states.pop(track.track_id, None)
            if state is None:
                continue
            events.extend(self._build_terminal_events(track=track, state=state))

        return events

    def flush(self) -> list[EventRecord]:
        events: list[EventRecord] = []
        for track_id, state in list(self._states.items()):
            track = TrackedPerson(
                track_id=state.track_id,
                bbox=(0, 0, 0, 0),
                confidence=0.0,
                first_seen_frame=state.first_seen_frame,
                last_seen_frame=state.last_seen_frame,
                first_seen_seconds=state.first_seen_seconds,
                last_seen_seconds=state.last_seen_seconds,
                identity=state.person_id,
                identity_confidence=state.identity_confidence,
                known=state.known,
                history=[state.first_center, state.last_center],
            )
            events.extend(self._build_terminal_events(track=track, state=state))
            self._states.pop(track_id, None)
        self._states.clear()
        return events

    def _build_event(
        self,
        *,
        action: str,
        track: TrackedPerson,
        start_seconds: float,
        end_seconds: float,
        frame_index: int,
        metadata: dict[str, object],
    ) -> EventRecord:
        self._event_index += 1
        return EventRecord(
            event_id=f"event-{self._event_index:05d}",
            person_id=track.identity,
            action=action,
            start_time=_format_offset(start_seconds),
            end_time=_format_offset(end_seconds),
            duration_seconds=max(end_seconds - start_seconds, 0.0),
            frame_index=frame_index,
            track_id=track.track_id,
            metadata={
                "known": track.known,
                "identity_confidence": round(float(track.identity_confidence), 4),
                "start_seconds": round(float(start_seconds), 3),
                "end_seconds": round(float(end_seconds), 3),
                **metadata,
            },
        )

    def _build_terminal_events(self, *, track: TrackedPerson, state: _TrackEventState) -> list[EventRecord]:
        events: list[EventRecord] = []
        if state.entered_center and state.last_zone == "edge":
            events.append(
                self._build_event(
                    action="exiting",
                    track=track,
                    start_seconds=state.first_seen_seconds,
                    end_seconds=state.last_seen_seconds,
                    frame_index=state.last_seen_frame,
                    metadata={"start_zone": state.start_zone, "end_zone": state.last_zone},
                )
            )

        displacement = _distance(state.first_center, state.last_center)
        if displacement >= self.movement_min_distance_px:
            events.append(
                self._build_event(
                    action="movement",
                    track=track,
                    start_seconds=state.first_seen_seconds,
                    end_seconds=state.last_seen_seconds,
                    frame_index=state.last_seen_frame,
                    metadata={
                        "start_zone": state.start_zone,
                        "end_zone": state.last_zone,
                        "start_center": [round(state.first_center[0], 2), round(state.first_center[1], 2)],
                        "end_center": [round(state.last_center[0], 2), round(state.last_center[1], 2)],
                        "displacement_px": round(displacement, 2),
                    },
                )
            )
        return events


def _zone_for_point(
    point: tuple[float, float],
    *,
    width: int,
    height: int,
    margin_ratio: float,
) -> str:
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    x, y = point
    if x <= margin_x or x >= (width - margin_x) or y <= margin_y or y >= (height - margin_y):
        return "edge"
    return "center"


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx * dx + dy * dy) ** 0.5


def _format_offset(seconds: float) -> str:
    total_milliseconds = int(round(max(seconds, 0.0) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
