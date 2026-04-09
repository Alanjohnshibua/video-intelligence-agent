"""
Orchestrates the full user-query to structured-response pipeline:

1. Parse the user's natural-language query
2. Resolve the time window
3. Filter the event log locally
4. Skip Sarvam if no events matched
5. Call Sarvam with a structured prompt when reasoning is needed
6. Return a human-readable response

The controller also maintains a small in-memory cache so that repeated queries
do not repeatedly consume API quota.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_intelligence_agent.agent.event_retriever import EventRetriever
from video_intelligence_agent.agent.query_parser import QueryIntent, parse_query
from video_intelligence_agent.agent.sarvam_client import SarvamClient, SarvamClientError
from video_intelligence_agent.agent.time_filter import resolve_window
from video_intelligence_agent.cctv_pipeline.utils.logger import get_pipeline_logger

logger = get_pipeline_logger("agent.controller")

_CACHE_SIZE = 64


@dataclass
class AgentResponse:
    """Structured result returned by :meth:`AgentController.ask`."""

    answer: str
    intent: QueryIntent
    matched_events: list[dict[str, Any]] = field(default_factory=list)
    sarvam_called: bool = False
    from_cache: bool = False
    error: str = ""


class AgentController:
    """High-level entry point for conversational CCTV event queries."""

    def __init__(
        self,
        events_path: Path | str,
        *,
        sarvam_client: SarvamClient | None = None,
        max_events_per_query: int = 50,
        enable_cache: bool = True,
    ) -> None:
        self._retriever = EventRetriever(events_path, max_events=max_events_per_query)
        self._sarvam = sarvam_client
        self._enable_cache = enable_cache
        self._cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}

    def ask(self, query: str) -> AgentResponse:
        """Process a natural-language CCTV query end-to-end."""
        logger.info("Agent query received | query=%r", query)

        intent = parse_query(query)

        if self._enable_cache:
            cache_key = _cache_key(intent)
            cached_item = self._cache.get(cache_key)
            if cached_item:
                cached_answer, cached_events = cached_item
                logger.info("Cache hit | key=%s", cache_key)
                return AgentResponse(
                    answer=cached_answer,
                    intent=intent,
                    matched_events=cached_events,
                    from_cache=True,
                )

        window = resolve_window(intent)

        try:
            matched = self._retriever.filter(intent, window)
        except (ValueError, OSError) as exc:
            error_msg = f"Could not load event data: {exc}"
            logger.error(error_msg)
            return AgentResponse(answer=f"[warning] {error_msg}", intent=intent, error=error_msg)

        if not matched:
            answer = _no_events_response(intent, window)
            logger.info("No events matched - returning direct response.")
            return AgentResponse(answer=answer, intent=intent, matched_events=[])

        if self._sarvam is None:
            answer = _format_events_locally(matched, query, intent)
            logger.info("Offline mode - Sarvam not configured.")
            return AgentResponse(answer=answer, intent=intent, matched_events=matched)

        events_json = json.dumps(matched, indent=2, ensure_ascii=False)
        prompt = SarvamClient.build_cctv_prompt(
            query,
            events_json,
            extra_context=_prompt_context(intent),
        )

        try:
            answer = self._sarvam.generate(prompt=prompt)
            logger.info("Sarvam response received | chars=%d", len(answer))
        except SarvamClientError as exc:
            error_msg = str(exc)
            logger.error("Sarvam call failed: %s", error_msg)
            answer = (
                f"[warning] Sarvam API error: {error_msg}\n\n"
                f"Here is the raw event data instead:\n\n"
                f"{_format_events_locally(matched, query, intent)}"
            )
            return AgentResponse(
                answer=answer,
                intent=intent,
                matched_events=matched,
                sarvam_called=True,
                error=error_msg,
            )

        if self._enable_cache:
            self._cache[cache_key] = (answer, matched)
            if len(self._cache) > _CACHE_SIZE:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

        return AgentResponse(
            answer=answer,
            intent=intent,
            matched_events=matched,
            sarvam_called=True,
        )

    def reload_events(self) -> None:
        """Force a reload of the event log from disk and clear the cache."""
        self._retriever.invalidate_cache()
        self._cache.clear()
        logger.info("Event cache and query cache cleared.")

    @property
    def total_events(self) -> int:
        """Return the number of cached events currently loaded."""
        return len(self._retriever.all_events())


def _cache_key(intent: QueryIntent) -> str:
    """Produce a canonical cache key from the parsed intent fields."""
    normalized_query = " ".join(intent.raw_query.lower().split())
    return (
        f"{normalized_query}|{intent.summary_requested}|{intent.clip_requested}|"
        f"{intent.date_hint}|{intent.time_of_day}|"
        f"{intent.start_time_hint}|{intent.end_time_hint}|"
        f"{intent.person_type_filter}|{intent.action_filter}|"
        f"{getattr(intent, 'person_id_filter', '')}"
    )


def _no_events_response(intent: QueryIntent, window: Any) -> str:
    """Build a human-readable message when no events match the query."""
    parts = []
    if intent.person_type_filter:
        parts.append(intent.person_type_filter + " people")
    if intent.action_filter:
        parts.append(f"performing '{intent.action_filter}'")
    description = " ".join(parts) if parts else "matching events"

    time_str = ""
    if intent.start_time_hint or intent.time_of_day:
        time_str = f" ({window.start:%H:%M} - {window.end:%H:%M})"

    return f"No {description} were found for {intent.date_hint}{time_str}."


def _format_events_locally(events: list[dict[str, Any]], query: str, intent: QueryIntent) -> str:
    """Format a compact event summary without calling Sarvam."""
    if intent.summary_requested:
        return _summarise_events_locally(events, query)

    lines = [f"Found {len(events)} event(s) matching your query: \"{query}\"\n"]
    for i, event in enumerate(events, 1):
        pid = event.get("person_id", "unknown")
        action = event.get("action", "unknown")
        start = event.get("start_time", "--")
        end = event.get("end_time", "--")
        duration = event.get("duration_seconds", event.get("duration", "--"))
        clip = event.get("clip_path")
        line = f"  {i}. [{start} -> {end}] {pid} - {action}"
        if isinstance(duration, (int, float)):
            line += f" ({duration:.1f}s)"
        if clip:
            line += f"\n     Clip: {clip}"
        lines.append(line)
    return "\n".join(lines)


def _summarise_events_locally(events: list[dict[str, Any]], query: str) -> str:
    """Generate a concise overview for broad 'what happened' style questions."""
    actions: dict[str, int] = {}
    people: set[str] = set()
    start_times: list[str] = []
    end_times: list[str] = []

    for event in events:
        action = str(event.get("action", "unknown"))
        actions[action] = actions.get(action, 0) + 1
        people.add(str(event.get("person_id", "unknown")))
        if event.get("start_time"):
            start_times.append(str(event["start_time"]))
        if event.get("end_time"):
            end_times.append(str(event["end_time"]))

    ordered_actions = ", ".join(f"{count} {action}" for action, count in sorted(actions.items()))
    first_seen = min(start_times) if start_times else "--"
    last_seen = max(end_times) if end_times else "--"
    ordered_people = ", ".join(sorted(people))

    lines = [
        f"Here is a summary for \"{query}\":",
        f"- Total events: {len(events)}",
        f"- People involved: {ordered_people}",
        f"- Activity breakdown: {ordered_actions}",
        f"- Time span covered: {first_seen} to {last_seen}",
    ]

    if events:
        highlighted = events[: min(3, len(events))]
        lines.append("- Key moments:")
        for event in highlighted:
            lines.append(
                "  "
                f"[{event.get('start_time', '--')}] "
                f"{event.get('person_id', 'unknown')} "
                f"{event.get('action', 'unknown')}"
            )

    return "\n".join(lines)


def _prompt_context(intent: QueryIntent) -> str:
    """Add lightweight guidance so the reasoning model answers in the right mode."""
    if intent.summary_requested and not intent.clip_requested:
        return (
            "The user wants a high-level explanation of what happened. "
            "Prioritize a concise narrative summary and do not lead with clip availability."
        )
    if intent.clip_requested and not intent.summary_requested:
        return (
            "The user is asking for evidence they can watch. "
            "Mention the most relevant events and reference clip paths when available."
        )
    return ""
