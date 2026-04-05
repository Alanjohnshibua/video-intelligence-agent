from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from video_intelligence_agent.cctv.models import ActivityRecord, VideoMetadata


@dataclass(slots=True)
class EventDecision:
    event_category: str
    alert_level: str
    alert_reasons: list[str] = field(default_factory=list)
    start_timestamp: str | None = None
    end_timestamp: str | None = None


class EventDecisionEngine:
    """Assigns an event category, severity, and human-readable timestamps."""

    def classify(
        self,
        metadata: VideoMetadata,
        record: ActivityRecord,
    ) -> EventDecision:
        actions = [action.lower() for action in record.actions]
        has_unknown_person = any(not person.known for person in record.people)
        has_known_person = any(person.known for person in record.people)
        is_loitering = any("loitering" in action for action in actions)
        is_entry = any("entering" in action for action in actions)
        is_exit = any("exiting" in action for action in actions)
        is_interaction = any("interacting" in action for action in actions)

        reasons: list[str] = []
        category = "routine_motion"
        alert_level = "info"

        if has_unknown_person:
            reasons.append("unknown person detected")
        if is_loitering:
            reasons.append("loitering behavior detected")
        if is_interaction:
            reasons.append("multi-person interaction detected")
        if is_entry:
            reasons.append("entry event detected")
        if is_exit:
            reasons.append("exit event detected")

        if has_unknown_person and is_loitering:
            category = "suspicious_presence"
            alert_level = "high"
        elif has_unknown_person:
            category = "unknown_person_detected"
            alert_level = "medium"
        elif is_loitering:
            category = "loitering"
            alert_level = "medium"
        elif is_interaction:
            category = "interaction"
            alert_level = "medium"
        elif is_entry and is_exit:
            category = "transit"
            alert_level = "low"
        elif is_entry:
            category = "entry"
            alert_level = "low"
        elif is_exit:
            category = "exit"
            alert_level = "low"
        elif has_known_person:
            category = "known_person_activity"

        return EventDecision(
            event_category=category,
            alert_level=alert_level,
            alert_reasons=reasons,
            start_timestamp=self._event_timestamp(metadata.recorded_at, record.start_time_seconds),
            end_timestamp=self._event_timestamp(metadata.recorded_at, record.end_time_seconds),
        )

    @staticmethod
    def _event_timestamp(recorded_at: str | None, seconds_offset: float) -> str | None:
        if recorded_at:
            try:
                base_time = datetime.fromisoformat(recorded_at)
            except ValueError:
                base_time = None
            else:
                return (base_time + timedelta(seconds=max(seconds_offset, 0.0))).isoformat(
                    timespec="seconds"
                )

        return f"T+{max(seconds_offset, 0.0):.1f}s"
