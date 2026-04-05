from __future__ import annotations

from pathlib import Path

import numpy as np

from video_intelligence_agent.cctv import CCTVAgentConfig, CCTVAnalysisPipeline, FootageQueryAgent
from video_intelligence_agent.cctv.models import (
    FramePacket,
    MotionAnalysis,
    PersonObservation,
    VideoMetadata,
)
from video_intelligence_agent.models import BoundingBox


class StubVideoReader:
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self._frames = [
            FramePacket(0, 0.0, np.zeros((100, 100, 3), dtype=np.uint8)),
            FramePacket(5, 1.0, np.ones((100, 100, 3), dtype=np.uint8)),
            FramePacket(10, 2.0, np.ones((100, 100, 3), dtype=np.uint8) * 2),
            FramePacket(15, 3.0, np.zeros((100, 100, 3), dtype=np.uint8)),
            FramePacket(20, 4.0, np.zeros((100, 100, 3), dtype=np.uint8)),
        ]

    def metadata(self) -> VideoMetadata:
        return VideoMetadata(
            video_path=self.video_path,
            fps=5.0,
            total_frames=25,
            width=100,
            height=100,
            duration_seconds=5.0,
            recorded_at="2026-04-04T12:00:00",
        )

    def iter_frames(self, *, frame_step: int = 1) -> list[FramePacket]:
        return self._frames


class StubMotionDetector:
    def __init__(self) -> None:
        self._items = [
            MotionAnalysis(active=False, motion_score=0.0),
            MotionAnalysis(active=True, motion_score=0.03),
            MotionAnalysis(active=True, motion_score=0.04),
            MotionAnalysis(active=False, motion_score=0.0),
            MotionAnalysis(active=False, motion_score=0.0),
        ]
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def analyze(self, frame: np.ndarray) -> MotionAnalysis:
        item = self._items[self._index]
        self._index += 1
        return item


class StubRecognizer:
    def __init__(self) -> None:
        self._index = 0

    def recognize(self, frame: np.ndarray) -> list[PersonObservation]:
        payloads = [
            [
                PersonObservation(
                    name="Alan",
                    confidence=0.95,
                    known=True,
                    bbox=BoundingBox(x=0, y=10, w=10, h=10),
                ),
                PersonObservation(
                    name="Unknown Person",
                    confidence=0.51,
                    known=False,
                    bbox=BoundingBox(x=18, y=10, w=10, h=10),
                ),
            ],
            [
                PersonObservation(
                    name="Alan",
                    confidence=0.97,
                    known=True,
                    bbox=BoundingBox(x=40, y=10, w=10, h=10),
                ),
                PersonObservation(
                    name="Unknown Person",
                    confidence=0.55,
                    known=False,
                    bbox=BoundingBox(x=55, y=10, w=10, h=10),
                ),
            ],
        ]
        result = payloads[self._index]
        self._index += 1
        return result


def test_cctv_pipeline_creates_motion_filtered_summary(tmp_path: Path) -> None:
    config = CCTVAgentConfig(
        output_dir=tmp_path / "cctv",
        inactivity_tolerance_frames=1,
        walking_distance_px=10.0,
        save_event_clips=False,
        save_unknown_snapshots=False,
    )
    pipeline = CCTVAnalysisPipeline(
        config=config,
        motion_detector=StubMotionDetector(),
        person_recognizer=StubRecognizer(),
        video_reader_factory=StubVideoReader,
    )

    result = pipeline.process_video("sample.mp4")

    assert len(result.activities) == 1
    assert result.summary.total_events == 1
    assert result.summary.total_people_detected == 2
    assert result.summary.known_people == ["Alan"]
    assert result.summary.unknown_people_count == 1
    assert result.storage_reduction_ratio > 0.0
    assert any("entering" in action for action in result.activities[0].actions)
    assert any("walking" in action for action in result.activities[0].actions)
    assert result.activities[0].event_category == "unknown_person_detected"
    assert result.activities[0].alert_level == "medium"
    assert result.activities[0].start_timestamp == "2026-04-04T12:00:01"
    assert "Alerts raised" in result.summary.summary_text


def test_footage_query_agent_returns_unknown_event_details(tmp_path: Path) -> None:
    config = CCTVAgentConfig(
        output_dir=tmp_path / "cctv",
        inactivity_tolerance_frames=1,
        walking_distance_px=10.0,
        save_event_clips=False,
        save_unknown_snapshots=False,
    )
    pipeline = CCTVAnalysisPipeline(
        config=config,
        motion_detector=StubMotionDetector(),
        person_recognizer=StubRecognizer(),
        video_reader_factory=StubVideoReader,
    )
    result = pipeline.process_video("sample.mp4")

    agent = FootageQueryAgent(result)
    answer = agent.ask("Show me unknown persons from yesterday")

    assert "matching event" in answer.lower()
    assert "Unknown Person" in answer

