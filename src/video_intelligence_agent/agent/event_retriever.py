"""
Loads the JSON event log written by the CCTV pipeline and applies structured
filters derived from a QueryIntent and TimeWindow.

All filtering happens locally before any Sarvam API call is made, so the
system sends only the minimal relevant dataset to the reasoning layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from video_intelligence_agent.cctv_pipeline.services.clip_manager import ClipManager
from video_intelligence_agent.cctv_pipeline.utils.logger import get_pipeline_logger

if TYPE_CHECKING:
    from video_intelligence_agent.agent.query_parser import QueryIntent
    from video_intelligence_agent.agent.time_filter import TimeWindow

logger = get_pipeline_logger("agent.event_retriever")

_MISSING = object()
_DEFAULT_MAX_EVENTS = 50


class EventRetriever:
    """Load and filter the pipeline event log with a small in-memory cache."""

    def __init__(
        self,
        events_path: Path | str,
        *,
        max_events: int = _DEFAULT_MAX_EVENTS,
    ) -> None:
        self._events_path = Path(events_path)
        self._max_events = max_events
        self._cache: list[dict[str, Any]] | None = None

    def load(self, *, force: bool = False) -> list[dict[str, Any]]:
        """Load or reload the event log from disk."""
        if self._cache is not None and not force:
            return self._cache

        if not self._events_path.exists():
            logger.warning("Event log not found: %s - returning empty dataset.", self._events_path)
            self._cache = []
            return self._cache

        try:
            raw = self._events_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to read event log %s: %s", self._events_path, exc)
            raise ValueError(f"Could not parse event log at {self._events_path}: {exc}") from exc

        if not isinstance(payload, list):
            logger.warning(
                "Event log is not a JSON array (type=%s). Treating as empty.",
                type(payload).__name__,
            )
            self._cache = []
            return self._cache

        self._cache = _deduplicate(payload)
        if len(self._cache) < len(payload):
            logger.info(
                "Deduplicated events: %d -> %d (removed %d duplicates)",
                len(payload),
                len(self._cache),
                len(payload) - len(self._cache),
            )
        else:
            logger.info("Loaded %d events from %s", len(self._cache), self._events_path)
        return self._cache

    def invalidate_cache(self) -> None:
        """Discard the in-memory event cache."""
        self._cache = None

    def filter(self, intent: "QueryIntent", window: "TimeWindow") -> list[dict[str, Any]]:
        """Apply all active query filters and return a bounded result set."""
        all_events = self.load()
        results: list[dict[str, Any]] = []

        for event in all_events:
            if not _passes_time_filter(event, window):
                continue
            if not _passes_person_type_filter(event, intent.person_type_filter):
                continue
            if not _passes_person_id_filter(event, getattr(intent, "person_id_filter", None)):
                continue
            if not _passes_action_filter(event, intent.action_filter):
                continue
            results.append(event)
            if len(results) >= self._max_events:
                break

        logger.info(
            "Filtered %d / %d events | time_window=%s person_type=%s action=%s",
            len(results),
            len(all_events),
            window,
            intent.person_type_filter,
            intent.action_filter,
        )
        return results

    def all_events(self) -> list[dict[str, Any]]:
        """Return the full cached event list."""
        return self.load()

    def ensure_clips(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Backfill missing clip files for matched events when enough metadata is available."""
        if not events:
            return events

        clip_manager: ClipManager | None = None
        updated = False
        for event in events:
            if event.get("clip_path"):
                continue
            metadata = event.get("metadata", {}) or {}
            source_video_path = metadata.get("source_video_path")
            start_seconds = metadata.get("start_seconds")
            end_seconds = metadata.get("end_seconds")
            if not isinstance(source_video_path, str):
                continue
            if not isinstance(start_seconds, (int, float)) or not isinstance(end_seconds, (int, float)):
                continue

            if clip_manager is None:
                clip_manager = ClipManager(
                    clips_dir=self._events_path.parent / "clips",
                    debug_dir=self._events_path.parent / "debug",
                    codec="mp4v",
                    fps=1.0,
                    pre_event_seconds=2.0,
                )

            clip_basename = _clip_basename_for_event(event)
            try:
                clip_path = clip_manager.save_clip_for_time_range(
                    video_path=source_video_path,
                    clip_name=clip_basename,
                    start_seconds=float(start_seconds),
                    end_seconds=float(end_seconds),
                    seconds_before=2.0,
                    seconds_after=2.0,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to backfill clip for event %s: %s",
                    event.get("event_id", "unknown"),
                    exc,
                )
                continue

            if clip_path:
                event["clip_path"] = clip_path
                updated = True

        if updated:
            self._flush_cache()
        return events

    def _flush_cache(self) -> None:
        """Persist the cached event list after in-memory enrichment."""
        if self._cache is None:
            return
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        self._events_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _passes_time_filter(event: dict[str, Any], window: "TimeWindow") -> bool:
    """Return True when the event start time falls within the resolved window."""
    start_time = event.get("start_time")
    if not start_time or not isinstance(start_time, str):
        return True
    return window.contains_iso(start_time)


def _passes_person_type_filter(event: dict[str, Any], person_type: str | None) -> bool:
    """Apply the known or unknown person filter from the parsed intent."""
    if person_type is None:
        return True

    metadata: dict[str, Any] = event.get("metadata", {}) or {}
    known_flag = metadata.get("known", _MISSING)

    if known_flag is not _MISSING:
        is_known = bool(known_flag)
        return is_known if person_type == "known" else not is_known

    type_field = str(event.get("type", "")).lower()
    person_id = str(event.get("person_id", "")).lower()

    if person_type == "unknown":
        return "unknown" in type_field or "unknown" in person_id
    if person_type == "known":
        return "unknown" not in type_field and "unknown" not in person_id
    return True


def _passes_person_id_filter(event: dict[str, Any], person_id: str | None) -> bool:
    """Return True when the event belongs to the requested person ID."""
    if not person_id:
        return True
    event_pid = str(event.get("person_id", "")).lower()
    return event_pid == person_id.lower()


def _passes_action_filter(event: dict[str, Any], action: str | None) -> bool:
    """Return True when the event action matches the requested action."""
    if action is None:
        return True
    event_action = str(event.get("action", "")).lower()
    return action.lower() in event_action


def _deduplicate(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicates while preserving the first occurrence of each event."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for event in events:
        event_id = event.get("event_id")
        if event_id:
            key = str(event_id)
        else:
            key = "|".join(
                [
                    str(event.get("person_id", "")),
                    str(event.get("action", "")),
                    str(event.get("start_time", "")),
                    str(event.get("track_id", "")),
                ]
            )
        if key not in seen:
            seen.add(key)
            unique.append(event)
    return unique


def _clip_basename_for_event(event: dict[str, Any]) -> str:
    """Reconstruct a stable clip filename for an event when backfilling on demand."""
    metadata = event.get("metadata", {}) or {}
    source_label = str(metadata.get("source_video_label", "video"))
    event_id = str(event.get("event_id", "event"))
    track_id = int(event.get("track_id", 0) or 0)
    return f"{source_label}__{event_id}_track-{track_id:04d}"
