"""
time_filter.py
--------------
Resolves human-readable time expressions from a ``QueryIntent`` into concrete
``datetime`` windows that ``EventRetriever`` can use for fast local filtering.

Supported patterns
------------------
* Named dates   :  "today", "yesterday", ISO "YYYY-MM-DD"
* Named slots   :  "morning" 06-12, "afternoon" 12-17, "evening" 17-21, "night" 21-06
* Explicit HH:MM:  any two "HH:MM" hints produced by ``query_parser``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

from video_intelligence_agent.cctv_pipeline.utils.logger import get_pipeline_logger

if TYPE_CHECKING:
    from video_intelligence_agent.agent.query_parser import QueryIntent

logger = get_pipeline_logger("agent.time_filter")


# ---------------------------------------------------------------------------
# Named time-of-day windows  (start_hour, end_hour)  – 24-hour, exclusive end
# ---------------------------------------------------------------------------

_TIME_OF_DAY_WINDOWS: dict[str, tuple[int, int]] = {
    "morning": (6, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 24),  # midnight handled specially below
}


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class TimeWindow:
    """
    A resolved, half-open time window ``[start, end)``.

    Both timestamps are timezone-naive ``datetime`` objects sharing the same
    local calendar date unless the slot crosses midnight.
    """

    start: datetime
    end: datetime

    def __str__(self) -> str:   # pragma: no cover
        return f"TimeWindow({self.start:%Y-%m-%d %H:%M} -> {self.end:%Y-%m-%d %H:%M})"

    def contains_iso(self, iso_string: str) -> bool:
        """
        Return ``True`` when *iso_string* falls inside ``[start, end)``.

        Accepts both full ISO-8601 strings (``2026-04-08T17:32:00``) and the
        pipeline's HH:MM:SS.mmm offset format (``00:10:34.200``).  Strings
        that cannot be parsed are treated as outside the window.
        """
        dt = _parse_flexible_datetime(iso_string, reference_date=self.start.date())
        if dt is None:
            return True  # can't filter → include by default
        return self.start <= dt < self.end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_window(intent: "QueryIntent") -> TimeWindow:
    """
    Convert the date/time fields of a :class:`~query_parser.QueryIntent` into
    a concrete :class:`TimeWindow`.

    Resolution order
    ----------------
    1. Resolve the calendar date from ``intent.date_hint``.
    2. If explicit ``start_time_hint`` **and** ``end_time_hint`` are present,
       use them verbatim.
    3. Otherwise fall back to the named ``time_of_day`` slot.
    4. If no time restriction is present, the whole calendar day is returned.

    Parameters
    ----------
    intent:
        Parsed user intent produced by ``query_parser.parse_query``.

    Returns
    -------
    TimeWindow
        Resolved half-open time window.
    """
    base_date = _resolve_date(intent.date_hint)

    # Explicit clock range takes priority
    if intent.start_time_hint and intent.end_time_hint:
        window = _window_from_explicit_times(
            base_date,
            intent.start_time_hint,
            intent.end_time_hint,
        )
        logger.info("Time window resolved (explicit) | %s", window)
        return window

    # Named time-of-day slot
    if intent.time_of_day:
        window = _window_from_named_slot(base_date, intent.time_of_day)
        logger.info("Time window resolved (named slot '%s') | %s", intent.time_of_day, window)
        return window

    # Single explicit start without end → rest of the day
    if intent.start_time_hint:
        start_dt = _make_datetime(base_date, intent.start_time_hint)
        end_dt = datetime.combine(base_date, time(23, 59, 59))
        window = TimeWindow(start=start_dt, end=end_dt)
        logger.info("Time window resolved (start only) | %s", window)
        return window

    # No time restriction → full calendar day
    window = _full_day_window(base_date)
    logger.info("Time window resolved (full day) | %s", window)
    return window


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_date(date_hint: str) -> date:
    """Map a date hint string to a ``datetime.date`` object."""
    today = date.today()
    if date_hint == "today":
        return today
    if date_hint == "yesterday":
        return today - timedelta(days=1)
    # Try ISO format
    try:
        return date.fromisoformat(date_hint)
    except (ValueError, AttributeError):
        logger.warning("Unrecognised date hint %r – defaulting to today.", date_hint)
        return today


def _window_from_explicit_times(
    base_date: date,
    start_hint: str,
    end_hint: str,
) -> TimeWindow:
    """Build a TimeWindow from two 'HH:MM' strings and a base date."""
    start_dt = _make_datetime(base_date, start_hint)
    end_dt = _make_datetime(base_date, end_hint)
    # Handle midnight-crossing (e.g. 23:00 → 01:00)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return TimeWindow(start=start_dt, end=end_dt)


def _window_from_named_slot(base_date: date, slot: str) -> TimeWindow:
    """Build a TimeWindow from a named time-of-day slot."""
    start_hour, end_hour = _TIME_OF_DAY_WINDOWS.get(slot, (0, 24))
    start_dt = datetime.combine(base_date, time(start_hour % 24, 0))
    if end_hour == 24:
        end_dt = datetime.combine(base_date + timedelta(days=1), time(0, 0))
    else:
        end_dt = datetime.combine(base_date, time(end_hour, 0))
    return TimeWindow(start=start_dt, end=end_dt)


def _full_day_window(base_date: date) -> TimeWindow:
    """Return a window covering the entire calendar day."""
    return TimeWindow(
        start=datetime.combine(base_date, time(0, 0)),
        end=datetime.combine(base_date + timedelta(days=1), time(0, 0)),
    )


def _make_datetime(base_date: date, hhmm: str) -> datetime:
    """
    Combine *base_date* with an 'HH:MM' string to produce a ``datetime``.
    Falls back to midnight on parse failure.
    """
    try:
        t = time.fromisoformat(hhmm if len(hhmm) > 5 else hhmm + ":00")
        return datetime.combine(base_date, t)
    except ValueError:
        logger.warning("Could not parse time string %r – using 00:00.", hhmm)
        return datetime.combine(base_date, time(0, 0))


def _parse_flexible_datetime(value: str, *, reference_date: date) -> datetime | None:
    """
    Try to parse *value* as either an ISO-8601 datetime or a pipeline
    HH:MM:SS.mmm offset timestamp, returning ``None`` on failure.

    ISO-8601 example : ``2026-04-08T17:32:00``
    Offset example   : ``00:10:34.200``
    """
    # Full ISO datetime
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    # Pipeline offset  HH:MM:SS  or  HH:MM:SS.mmm
    offset_match = _OFFSET_RE.match(value)
    if offset_match:
        h = int(offset_match.group(1))
        m = int(offset_match.group(2))
        s = int(offset_match.group(3))
        base = datetime.combine(reference_date, time(0, 0))
        return base + timedelta(hours=h, minutes=m, seconds=s)

    return None


import re as _re  # noqa: E402 – placed after function bodies for readability
_OFFSET_RE = _re.compile(r"^(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?$")
