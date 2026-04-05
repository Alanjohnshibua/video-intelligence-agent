from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from video_intelligence_agent.cctv_pipeline.core.event_logic import RuleBasedEventDetector
from video_intelligence_agent.cctv_pipeline.core.video_processor import VideoProcessor
from video_intelligence_agent.cctv_pipeline.models import (
    Detection,
    EventRecord,
    FramePacket,
    TrackingUpdate,
    TrackedPerson,
    VideoMetadata,
)
from video_intelligence_agent.cctv_pipeline.services.event_logger import EventLoggerService
from video_intelligence_agent.cctv_pipeline.utils.config import (
    PipelineConfig,
    StorageConfig,
    load_pipeline_config,
)
from video_intelligence_agent.cctv_pipeline.utils.error_handler import DetectionError


class StubVideoSource:
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self._frames = [
            FramePacket(frame_index=0, timestamp_seconds=0.0, frame=np.zeros((80, 120, 3), dtype=np.uint8)),
            FramePacket(frame_index=1, timestamp_seconds=1.0, frame=np.ones((80, 120, 3), dtype=np.uint8)),
            FramePacket(frame_index=2, timestamp_seconds=2.0, frame=np.ones((80, 120, 3), dtype=np.uint8) * 2),
        ]

    def metadata(self) -> VideoMetadata:
        return VideoMetadata(
            video_path=self.video_path,
            fps=1.0,
            total_frames=len(self._frames),
            width=120,
            height=80,
            duration_seconds=3.0,
        )

    def iter_frames(self, *, frame_step: int = 1):
        return list(self._frames)

    def close(self) -> None:
        return None


class StubMotionDetector:
    def __init__(self) -> None:
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def analyze(self, frame: np.ndarray, *, frame_index: int):
        active = [False, True, True][self._index]
        self._index += 1
        return type("Motion", (), {"active": active, "boxes": [], "score": 0.1 if active else 0.0})()


class StubDetector:
    ready = True
    load_error = None

    def __init__(self) -> None:
        self._index = 0

    def detect(self, frame: np.ndarray, *, frame_index: int) -> list[Detection]:
        self._index += 1
        if frame_index == 1:
            raise DetectionError("YOLO inference failed.", module="detector", frame_index=frame_index)
        return [Detection(bbox=(40, 10, 70, 70), confidence=0.91)]


class StubTracker:
    def reset(self) -> None:
        return None

    def update(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
        timestamp_seconds: float,
    ) -> TrackingUpdate:
        if not detections:
            return TrackingUpdate(visible_tracks=[], lost_tracks=[])
        track = TrackedPerson(
            track_id=1,
            bbox=detections[0].bbox,
            confidence=detections[0].confidence,
            first_seen_frame=frame_index,
            last_seen_frame=frame_index,
            first_seen_seconds=timestamp_seconds,
            last_seen_seconds=timestamp_seconds,
        )
        return TrackingUpdate(visible_tracks=[track], lost_tracks=[])

    def apply_identity_updates(self, tracks: list[TrackedPerson]) -> None:
        return None


class StubRecognizer:
    def identify_tracks(
        self,
        frame: np.ndarray,
        tracks: list[TrackedPerson],
        *,
        frame_index: int,
    ) -> list[TrackedPerson]:
        for track in tracks:
            track.identity = "alice"
            track.identity_confidence = 0.93
            track.known = True
        return tracks


class StubEventLogic:
    def reset(self) -> None:
        return None

    def update(
        self,
        *,
        visible_tracks: list[TrackedPerson],
        lost_tracks: list[TrackedPerson],
        frame_shape: tuple[int, ...],
    ) -> list[EventRecord]:
        if not visible_tracks:
            return []
        track = visible_tracks[0]
        return [
            EventRecord(
                event_id="event-00001",
                person_id=track.identity,
                action="entering",
                start_time="00:00:02.000",
                end_time="00:00:02.000",
                duration_seconds=0.0,
                frame_index=track.last_seen_frame,
                track_id=track.track_id,
                metadata={"known": True},
            )
        ]

    def flush(self) -> list[EventRecord]:
        return []


class StubClipManager:
    def __init__(self) -> None:
        self.saved_clips: list[str] = []

    def buffer_frame(self, frame: np.ndarray) -> None:
        return None

    def save_clip_from_buffer(self, clip_name: str) -> str:
        self.saved_clips.append(clip_name)
        return f"memory://{clip_name}"

    def save_debug_frame(self, frame: np.ndarray, *, frame_index: int) -> str:
        return f"memory://frame-{frame_index}"


def test_load_pipeline_config_reads_face_and_debug_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "video_path: sample.mp4",
                "database_path: data/embeddings.pkl",
                "unknown_dir: outputs/unknown",
                "similarity_threshold: 0.55",
                "detection_confidence: 0.33",
                "debug_enabled: true",
                "save_debug_frames: true",
                "save_unknown_clips: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)

    assert config.video_path == "sample.mp4"
    assert str(config.database_path) == "data/embeddings.pkl"
    assert str(config.unknown_dir) == "outputs/unknown"
    assert config.similarity_threshold == 0.55
    assert config.detection_confidence == 0.33
    assert config.debug.enabled is True
    assert config.debug.save_frames is True
    assert config.storage.save_unknown_clips is False


def test_rule_based_event_detector_emits_entering_loitering_and_exiting() -> None:
    detector = RuleBasedEventDetector(
        loitering_seconds=2.0,
        loitering_radius_px=20.0,
        border_margin_ratio=0.1,
    )

    frame_shape = (100, 100, 3)
    events = detector.update(
        visible_tracks=[
            TrackedPerson(
                track_id=1,
                bbox=(0, 30, 10, 70),
                confidence=0.9,
                first_seen_frame=0,
                last_seen_frame=0,
                first_seen_seconds=0.0,
                last_seen_seconds=0.0,
                identity="unknown_1",
            )
        ],
        lost_tracks=[],
        frame_shape=frame_shape,
    )
    assert events == []

    events = detector.update(
        visible_tracks=[
            TrackedPerson(
                track_id=1,
                bbox=(35, 30, 55, 70),
                confidence=0.9,
                first_seen_frame=0,
                last_seen_frame=1,
                first_seen_seconds=0.0,
                last_seen_seconds=1.0,
                identity="unknown_1",
            )
        ],
        lost_tracks=[],
        frame_shape=frame_shape,
    )
    assert [event.action for event in events] == ["entering"]

    events = detector.update(
        visible_tracks=[
            TrackedPerson(
                track_id=1,
                bbox=(38, 32, 58, 72),
                confidence=0.9,
                first_seen_frame=0,
                last_seen_frame=3,
                first_seen_seconds=0.0,
                last_seen_seconds=3.0,
                identity="unknown_1",
            )
        ],
        lost_tracks=[],
        frame_shape=frame_shape,
    )
    assert [event.action for event in events] == ["loitering"]

    events = detector.update(
        visible_tracks=[
            TrackedPerson(
                track_id=1,
                bbox=(90, 30, 99, 70),
                confidence=0.9,
                first_seen_frame=0,
                last_seen_frame=4,
                first_seen_seconds=0.0,
                last_seen_seconds=4.0,
                identity="unknown_1",
            )
        ],
        lost_tracks=[],
        frame_shape=frame_shape,
    )
    assert events == []

    events = detector.update(
        visible_tracks=[],
        lost_tracks=[
            TrackedPerson(
                track_id=1,
                bbox=(90, 30, 99, 70),
                confidence=0.9,
                first_seen_frame=0,
                last_seen_frame=4,
                first_seen_seconds=0.0,
                last_seen_seconds=4.0,
                identity="unknown_1",
            )
        ],
        frame_shape=frame_shape,
    )
    assert [event.action for event in events] == ["exiting"]


def test_video_processor_logs_detector_failures_and_keeps_running(tmp_path: Path) -> None:
    config = PipelineConfig(
        video_path="sample.mp4",
        storage=StorageConfig(
            output_dir=tmp_path,
            event_filename="events.json",
            save_event_clips=True,
            save_unknown_clips=True,
            clip_seconds_before=1.0,
            clip_seconds_after=1.0,
            clip_codec="mp4v",
        ),
    )

    event_logger = EventLoggerService(tmp_path / "events.json")
    processor = VideoProcessor(
        config=config,
        motion_detector=StubMotionDetector(),
        detector=StubDetector(),
        tracker=StubTracker(),
        recognizer=StubRecognizer(),
        event_logic=StubEventLogic(),
        event_logger=event_logger,
        clip_manager=StubClipManager(),
        video_source_cls=StubVideoSource,
    )

    result = processor.process_video()
    events_on_disk = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))

    assert result.stats.errors_encountered == 1
    assert result.stats.events_detected == 1
    assert result.stats.processed_frames == 1
    assert events_on_disk[0]["action"] == "entering"
    assert processor.query_events(action="entering")[0]["person_id"] == "alice"
