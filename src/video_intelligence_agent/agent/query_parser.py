"""
query_parser.py
---------------
Parses free-form user queries into a structured ``QueryIntent`` that downstream
modules (time_filter, event_retriever, agent_controller) can act on without
re-parsing the raw string themselves.

The parser is intentionally rule-based and keyword-driven so that it works
without an external model and does not add latency to the critical path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from video_intelligence_agent.cctv_pipeline.utils.logger import get_pipeline_logger

logger = get_pipeline_logger("agent.query_parser")


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class QueryIntent:
    """Structured representation of what the user is asking for."""

    raw_query: str
    """The original, unmodified user query string."""

    summary_requested: bool = False
    """Whether the user is asking for an overview/summary of activity."""

    clip_requested: bool = False
    """Whether the user is explicitly asking to see clips/footage."""

    date_hint: str = "today"
    """Date expression extracted from the query (today / yesterday / YYYY-MM-DD)."""

    time_of_day: str | None = None
    """Named time-of-day slot: morning | afternoon | evening | night | None."""

    start_time_hint: str | None = None
    """Explicit start time extracted as a string, e.g. '14:00' or '3 PM'."""

    end_time_hint: str | None = None
    """Explicit end time extracted as a string, e.g. '18:00' or '6 PM'."""

    person_type_filter: str | None = None
    """'known' | 'unknown' | None – restricts results to a person category."""

    person_id_filter: str | None = None
    """Exact person ID, e.g. 'unknown_3' | None."""

    action_filter: str | None = None
    """Event action filter, e.g. 'loitering' | 'entering' | 'exiting' | None."""

    raw_filters: dict[str, str] = field(default_factory=dict)
    """Any additional key/value pairs found in the query for future extensibility."""

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"QueryIntent("
            f"summary_requested={self.summary_requested!r}, "
            f"clip_requested={self.clip_requested!r}, "
            f"date={self.date_hint!r}, "
            f"time_of_day={self.time_of_day!r}, "
            f"start={self.start_time_hint!r}, "
            f"end={self.end_time_hint!r}, "
            f"person_type={self.person_type_filter!r}, "
            f"person_id={self.person_id_filter!r}, "
            f"action={self.action_filter!r})"
        )


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

# Named time-of-day slots
_TIME_OF_DAY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmorning\b", re.IGNORECASE), "morning"),
    (re.compile(r"\bafternoon\b", re.IGNORECASE), "afternoon"),
    (re.compile(r"\bevening\b", re.IGNORECASE), "evening"),
    (re.compile(r"\bnight\b", re.IGNORECASE), "night"),
]

# Explicit 12-hour clock with optional space before AM/PM
_CLOCK_12H_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
    re.IGNORECASE,
)

# Explicit 24-hour clock, e.g. 14:30
_CLOCK_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")

# Action keywords → canonical event action strings
_ACTION_MAP: dict[str, str] = {
    "loiter": "loitering",
    "loitering": "loitering",
    "enter": "entering",
    "entering": "entering",
    "entered": "entering",
    "exit": "exiting",
    "exiting": "exiting",
    "exited": "exiting",
    "leave": "exiting",
    "leaving": "exiting",
    "movement": "movement",
    "moving": "movement",
    "move": "movement",
}

# Person type keywords
_PERSON_TYPE_MAP: dict[str, str] = {
    "unknown": "unknown",
    "stranger": "unknown",
    "unrecognised": "unknown",
    "unrecognized": "unknown",
    "known": "known",
    "recognised": "known",
    "recognized": "known",
    "employee": "known",
    "staff": "known",
}

# Date keywords
_DATE_MAP: dict[str, str] = {
    "today": "today",
    "yesterday": "yesterday",
}

_SUMMARY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsummary\b", re.IGNORECASE),
    re.compile(r"\bsummarise\b", re.IGNORECASE),
    re.compile(r"\bsummarize\b", re.IGNORECASE),
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\bwhat happened\b", re.IGNORECASE),
    re.compile(r"\bwhat happening\b", re.IGNORECASE),
    re.compile(r"\bwhat is happening\b", re.IGNORECASE),
    re.compile(r"\bwhat's happening\b", re.IGNORECASE),
    re.compile(r"\btell me what happened\b", re.IGNORECASE),
    re.compile(r"\btell me what happening\b", re.IGNORECASE),
    re.compile(r"\btell me what is happening\b", re.IGNORECASE),
    re.compile(r"\btell me what's happening\b", re.IGNORECASE),
    re.compile(r"\bdescribe\b", re.IGNORECASE),
]

_CLIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bclip\b", re.IGNORECASE),
    re.compile(r"\bclips\b", re.IGNORECASE),
    re.compile(r"\bfootage\b", re.IGNORECASE),
    re.compile(r"\bshow me\b", re.IGNORECASE),
    re.compile(r"\bplay\b", re.IGNORECASE),
    re.compile(r"\bcan i get\b.*\bclips?\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_query(query: str) -> QueryIntent:
    """
    Parse a natural-language CCTV query into a :class:`QueryIntent`.

    Parameters
    ----------
    query:
        Raw user query string.

    Returns
    -------
    QueryIntent
        A structured intent object.  Fields that cannot be extracted remain
        at their default (``None`` / ``'today'``).

    Examples
    --------
    >>> parse_query("Show me unknown people yesterday evening")
    QueryIntent(date='yesterday', time_of_day='evening', ..., person_type='unknown', ...)

    >>> parse_query("Who was loitering between 3 PM and 5 PM?")
    QueryIntent(date='today', start='15:00', end='17:00', action='loitering')
    """
    intent = QueryIntent(raw_query=query)

    _extract_request_mode(query, intent)
    _extract_date(query, intent)
    _extract_time_of_day(query, intent)
    _extract_explicit_time_range(query, intent)
    _extract_person_type(query, intent)
    _extract_person_id(query, intent)
    _extract_action(query, intent)

    logger.info(
        "Query parsed | summary=%s clip=%s date=%s time_of_day=%s start=%s end=%s person_type=%s person_id=%s action=%s",
        intent.summary_requested,
        intent.clip_requested,
        intent.date_hint,
        intent.time_of_day,
        intent.start_time_hint,
        intent.end_time_hint,
        intent.person_type_filter,
        intent.person_id_filter,
        intent.action_filter,
    )
    return intent


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------

def _extract_request_mode(query: str, intent: QueryIntent) -> None:
    """Infer whether the user wants an overview or direct video evidence."""
    intent.summary_requested = any(pattern.search(query) for pattern in _SUMMARY_PATTERNS)
    intent.clip_requested = any(pattern.search(query) for pattern in _CLIP_PATTERNS)

def _extract_date(query: str, intent: QueryIntent) -> None:
    """Look for date keywords or ISO date strings and populate ``intent.date_hint``."""
    lower = query.lower()

    for keyword, value in _DATE_MAP.items():
        if re.search(rf"\b{keyword}\b", lower):
            intent.date_hint = value
            return

    # ISO date: 2026-04-08
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
    if iso_match:
        intent.date_hint = iso_match.group(1)


def _extract_time_of_day(query: str, intent: QueryIntent) -> None:
    """Detect a named time-of-day slot if no explicit clock times follow."""
    for pattern, label in _TIME_OF_DAY_PATTERNS:
        if pattern.search(query):
            intent.time_of_day = label
            return


def _extract_explicit_time_range(query: str, intent: QueryIntent) -> None:
    """
    Extract an explicit time range such as "3 PM to 6 PM" or "14:00 to 17:00".
    Sets ``intent.start_time_hint`` and/or ``intent.end_time_hint``.
    """
    # First try 12-hour format matches
    matches_12h = list(_CLOCK_12H_RE.finditer(query))
    if len(matches_12h) >= 2:
        intent.start_time_hint = _normalise_12h(matches_12h[0])
        intent.end_time_hint = _normalise_12h(matches_12h[1])
        return
    if len(matches_12h) == 1:
        intent.start_time_hint = _normalise_12h(matches_12h[0])

    # Then try 24-hour format matches
    matches_24h = list(_CLOCK_24H_RE.finditer(query))
    if len(matches_24h) >= 2:
        intent.start_time_hint = f"{matches_24h[0].group(1).zfill(2)}:{matches_24h[0].group(2)}"
        intent.end_time_hint = f"{matches_24h[1].group(1).zfill(2)}:{matches_24h[1].group(2)}"
    elif len(matches_24h) == 1 and not intent.start_time_hint:
        intent.start_time_hint = f"{matches_24h[0].group(1).zfill(2)}:{matches_24h[0].group(2)}"


def _extract_person_type(query: str, intent: QueryIntent) -> None:
    """Set ``intent.person_type_filter`` from recognised person-type keywords."""
    lower = query.lower()
    for keyword, value in _PERSON_TYPE_MAP.items():
        if re.search(rf"\b{keyword}\b", lower):
            intent.person_type_filter = value
            return


def _extract_person_id(query: str, intent: QueryIntent) -> None:
    """Set ``intent.person_id_filter`` from exact ID patterns like 'unknown_3'."""
    lower = query.lower()
    match = re.search(r"\b(unknown_\d+|known_\d+)\b", lower)
    if match:
        intent.person_id_filter = match.group(1)
        intent.person_type_filter = "unknown" if "unknown" in match.group(1) else "known"


def _extract_action(query: str, intent: QueryIntent) -> None:
    """Set ``intent.action_filter`` from recognised event action keywords."""
    lower = query.lower()
    for keyword, canonical in _ACTION_MAP.items():
        if re.search(rf"\b{keyword}\b", lower):
            intent.action_filter = canonical
            return


def _normalise_12h(match: re.Match[str]) -> str:
    """Convert a 12-hour clock regex match to 'HH:MM' 24-hour format string."""
    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    meridiem = match.group(3).lower()
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"
