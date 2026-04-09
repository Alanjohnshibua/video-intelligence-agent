from __future__ import annotations

import json
from pathlib import Path

from video_intelligence_agent.agent import AgentController
from video_intelligence_agent.agent.query_parser import parse_query


def _write_events(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "event_id": "event-00001",
                    "person_id": "unknown_3",
                    "action": "entering",
                    "start_time": "00:00:03.600",
                    "end_time": "00:00:07.400",
                    "duration_seconds": 3.8,
                    "clip_path": "outputs/lobby_demo/clips/event-00001_track-0003.mp4",
                },
                {
                    "event_id": "event-00002",
                    "person_id": "unknown_3",
                    "action": "exiting",
                    "start_time": "00:00:03.600",
                    "end_time": "00:00:10.600",
                    "duration_seconds": 7.0,
                    "clip_path": "outputs/lobby_demo/clips/event-00002_track-0003.mp4",
                },
                {
                    "event_id": "event-00003",
                    "person_id": "unknown_1",
                    "action": "loitering",
                    "start_time": "00:00:00.200",
                    "end_time": "00:00:15.600",
                    "duration_seconds": 15.4,
                    "clip_path": "outputs/lobby_demo/clips/event-00003_track-0001.mp4",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def test_parse_query_detects_summary_and_clip_intents() -> None:
    clip_intent = parse_query("Can I get clips for unknown people yesterday evening?")
    summary_intent = parse_query("Can you tell me what happening in this video?")

    assert clip_intent.clip_requested is True
    assert clip_intent.summary_requested is False
    assert summary_intent.summary_requested is True
    assert summary_intent.clip_requested is False


def test_agent_controller_does_not_reuse_clip_answer_for_summary_query(tmp_path: Path) -> None:
    events_path = tmp_path / "events.json"
    _write_events(events_path)

    controller = AgentController(events_path=events_path, sarvam_client=None)

    clip_response = controller.ask("Can I get clips?")
    summary_response = controller.ask("Can you tell me what happening in this video?")

    assert clip_response.from_cache is False
    assert summary_response.from_cache is False
    assert "Clip:" in clip_response.answer
    assert "Here is a summary" in summary_response.answer
    assert "Clip:" not in summary_response.answer
