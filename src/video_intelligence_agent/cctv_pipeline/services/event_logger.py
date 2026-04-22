from __future__ import annotations

import json
from pathlib import Path

from video_intelligence_agent.cctv_pipeline.models import EventRecord
from video_intelligence_agent.cctv_pipeline.utils.error_handler import EventStorageError
from video_intelligence_agent.cctv_pipeline.utils.logger import get_pipeline_logger


class EventLoggerService:
    """Writes structured events to JSON and supports simple queries."""

    def __init__(self, output_path: Path | str, *, load_existing: bool = False) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_pipeline_logger("event_logger")
        self._events: list[EventRecord] = []
        if load_existing:
            self._load_existing_events()

    def append(self, event: EventRecord) -> None:
        self._events.append(event)
        self.flush()

    def extend(self, events: list[EventRecord]) -> None:
        if not events:
            return
        self._events.extend(events)
        self.flush()

    def flush(self) -> None:
        try:
            payload = [event.to_dict() for event in self._events]
            self.output_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise EventStorageError(
                f"Failed to write events to {self.output_path}",
                module="event_logger",
                cause=exc,
            ) from exc

    def events(self) -> list[EventRecord]:
        return list(self._events)

    def query_events(
        self,
        *,
        person_id: str | None = None,
        action: str | None = None,
        track_id: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for event in self._events:
            if person_id is not None and event.person_id != person_id:
                continue
            if action is not None and event.action != action:
                continue
            if track_id is not None and event.track_id != track_id:
                continue
            results.append(event.to_dict())
            if limit is not None and len(results) >= limit:
                break
        return results

    def _load_existing_events(self) -> None:
        if not self.output_path.exists():
            return

        try:
            payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EventStorageError(
                f"Failed to read existing event log from {self.output_path}",
                module="event_logger",
                cause=exc,
            ) from exc

        if not isinstance(payload, list):
            self.logger.warning("Event log is not a JSON array. Starting a fresh in-memory log.")
            return

        for item in payload:
            if not isinstance(item, dict):
                continue
            self._events.append(
                EventRecord(
                    event_id=str(item.get("event_id", "")),
                    person_id=str(item.get("person_id", "unknown")),
                    action=str(item.get("action", "unknown")),
                    start_time=str(item.get("start_time", "")),
                    end_time=str(item.get("end_time", "")),
                    duration_seconds=float(item.get("duration_seconds", 0.0)),
                    frame_index=int(item.get("frame_index", 0)),
                    track_id=_coerce_track_id(item.get("track_id")),
                    clip_path=_coerce_optional_str(item.get("clip_path")),
                    metadata=_coerce_metadata(item.get("metadata")),
                )
            )


def _coerce_track_id(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _coerce_optional_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _coerce_metadata(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {}
