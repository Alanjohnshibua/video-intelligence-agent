from __future__ import annotations

from datetime import datetime

from video_intelligence_agent.cctv.models import ActivityRecord, DailySummary, VideoMetadata


class DailySummaryGenerator:
    """Builds concise human-readable summaries from extracted CCTV events."""

    def generate(
        self,
        metadata: VideoMetadata,
        activities: list[ActivityRecord],
    ) -> DailySummary:
        if not activities:
            return DailySummary.empty()

        total_activity_duration = sum(activity.duration_seconds for activity in activities)
        known_people = sorted(
            {
                person.name
                for activity in activities
                for person in activity.people
                if person.known
            }
        )
        unknown_people = {
            person.track_id or person.name
            for activity in activities
            for person in activity.people
            if not person.known
        }
        all_people = {
            (person.track_id or person.name)
            for activity in activities
            for person in activity.people
        }

        key_events = [self._describe_activity(activity) for activity in activities[:5]]
        high_alert_count = sum(1 for activity in activities if activity.alert_level == "high")
        medium_alert_count = sum(1 for activity in activities if activity.alert_level == "medium")
        summary_text = self._build_summary_text(
            metadata=metadata,
            activities=activities,
            total_activity_duration=total_activity_duration,
            known_people=known_people,
            unknown_people_count=len(unknown_people),
            key_events=key_events,
            high_alert_count=high_alert_count,
            medium_alert_count=medium_alert_count,
        )
        return DailySummary(
            generated_at=datetime.now().isoformat(),
            total_activity_duration=total_activity_duration,
            total_events=len(activities),
            total_people_detected=len(all_people),
            known_people=known_people,
            unknown_people_count=len(unknown_people),
            key_events=key_events,
            summary_text=summary_text,
        )

    def _describe_activity(self, activity: ActivityRecord) -> str:
        if activity.people:
            people_text = ", ".join(person.display_name() for person in activity.people)
        else:
            people_text = "motion only"

        actions_text = ", ".join(activity.actions) if activity.actions else "movement detected"
        time_window = (
            f"{activity.start_timestamp} to {activity.end_timestamp}"
            if activity.start_timestamp and activity.end_timestamp
            else f"{activity.start_time_seconds:.1f}s to {activity.end_time_seconds:.1f}s"
        )
        return (
            f"{activity.event_id} | {time_window} | Category: {activity.event_category} | "
            f"Alert: {activity.alert_level} | People: {people_text} | Actions: {actions_text}"
        )

    def _build_summary_text(
        self,
        *,
        metadata: VideoMetadata,
        activities: list[ActivityRecord],
        total_activity_duration: float,
        known_people: list[str],
        unknown_people_count: int,
        key_events: list[str],
        high_alert_count: int,
        medium_alert_count: int,
    ) -> str:
        lines = [
            f"CCTV Summary for {metadata.video_path}",
            f"Total activity duration: {total_activity_duration:.1f}s across {len(activities)} event segments.",
            (
                f"People detected: {len(known_people) + unknown_people_count} unique "
                f"({len(known_people)} known, {unknown_people_count} unknown)."
            ),
            (
                f"Alerts raised: {high_alert_count} high, {medium_alert_count} medium, "
                f"{max(len(activities) - high_alert_count - medium_alert_count, 0)} low/info."
            ),
        ]
        if known_people:
            lines.append("Known individuals: " + ", ".join(known_people))
        else:
            lines.append("Known individuals: none detected")

        lines.append("Key events:")
        for index, event in enumerate(key_events, start=1):
            lines.append(f"{index}. {event}")
        return "\n".join(lines)

