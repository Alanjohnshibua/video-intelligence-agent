from __future__ import annotations

from typing import Protocol

from video_intelligence_agent.cctv.models import ActivityRecord, VideoAnalysisResult


class ChatResponderProtocol(Protocol):
    def generate(self, *, question: str, context: str) -> str: ...


class FootageQueryAgent:
    """
    Lightweight conversational layer over extracted events.

    It works locally with rule-based retrieval, and can optionally delegate
    final phrasing to a Gemini-style external responder through a simple protocol.
    """

    def __init__(
        self,
        result: VideoAnalysisResult,
        *,
        responder: ChatResponderProtocol | None = None,
    ) -> None:
        self.result = result
        self.responder = responder

    def ask(self, question: str) -> str:
        matched_events = self._match_events(question)
        context = self._build_context(question, matched_events)
        if self.responder is not None:
            return self.responder.generate(question=question, context=context)
        return self._fallback_answer(question, matched_events, context)

    def _match_events(self, question: str) -> list[ActivityRecord]:
        text = question.lower()
        events = self.result.activities

        if "unknown" in text:
            events = [
                event
                for event in events
                if any(not person.known for person in event.people)
            ]

        for keyword in ("loiter", "walk", "enter", "exit", "interact", "unknown", "suspicious"):
            if keyword in text:
                events = [
                    event
                    for event in events
                    if any(keyword in action.lower() for action in event.actions)
                    or keyword in event.event_category.lower()
                    or any(keyword in reason.lower() for reason in event.alert_reasons)
                ]
                break

        return events

    def _build_context(self, question: str, events: list[ActivityRecord]) -> str:
        lines = [
            "CCTV analysis context",
            f"Question: {question}",
            self.result.summary.summary_text,
        ]
        if events:
            lines.append("Matched events:")
            for event in events[:10]:
                lines.append(self._describe_event(event))
        else:
            lines.append("Matched events: none")
        return "\n".join(lines)

    def _fallback_answer(
        self,
        question: str,
        events: list[ActivityRecord],
        context: str,
    ) -> str:
        text = question.lower()
        if "summary" in text or "overview" in text:
            return self.result.summary.summary_text

        if not events:
            return "No matching CCTV events were found for that question."

        lines = [f"I found {len(events)} matching event(s)."]
        for event in events[:5]:
            lines.append(self._describe_event(event))
        return "\n".join(lines)

    def _describe_event(self, event: ActivityRecord) -> str:
        people = ", ".join(person.display_name() for person in event.people) or "motion only"
        actions = ", ".join(event.actions) or "movement detected"
        clip = event.clip_path or "no clip saved"
        time_window = (
            f"{event.start_timestamp} to {event.end_timestamp}"
            if event.start_timestamp and event.end_timestamp
            else f"{event.start_time_seconds:.1f}s-{event.end_time_seconds:.1f}s"
        )
        alert_text = f"category={event.event_category}, alert={event.alert_level}"
        if event.unknown_snapshot_paths:
            snapshots = "; ".join(event.unknown_snapshot_paths)
            return (
                f"{event.event_id}: {time_window}, people={people}, actions={actions}, "
                f"{alert_text}, clip={clip}, snapshots={snapshots}"
            )
        return (
            f"{event.event_id}: {time_window}, people={people}, actions={actions}, "
            f"{alert_text}, clip={clip}"
        )

