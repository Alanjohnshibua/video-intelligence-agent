from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from video_intelligence_agent.cctv.actions import ActionAnalyzer
from video_intelligence_agent.cctv.config import CCTVAgentConfig
from video_intelligence_agent.cctv.events import EventDecisionEngine
from video_intelligence_agent.cctv.ingestion import VideoReader
from video_intelligence_agent.cctv.models import (
    ActivityRecord,
    FramePacket,
    MotionAnalysis,
    PersonObservation,
    VideoAnalysisResult,
    VideoMetadata,
)
from video_intelligence_agent.cctv.motion import MotionDetector, MotionDetectorProtocol
from video_intelligence_agent.cctv.person import PersonRecognizerProtocol, SimpleTrackManager
from video_intelligence_agent.cctv.storage import EventStorageManager
from video_intelligence_agent.cctv.summary import DailySummaryGenerator
from video_intelligence_agent.image_io import GenericImageArray


@dataclass(slots=True)
class _ActiveEvent:
    event_id: str
    start_time_seconds: float
    start_frame_index: int
    end_time_seconds: float
    end_frame_index: int
    frames: list[GenericImageArray] = field(default_factory=list)
    motion_scores: list[float] = field(default_factory=list)
    people_by_track: dict[str, PersonObservation] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    unknown_snapshot_paths: list[str] = field(default_factory=list)
    unknown_snapshot_keys: set[str] = field(default_factory=set)


class _NullPersonRecognizer:
    def recognize(self, frame: GenericImageArray) -> list[PersonObservation]:
        return []


class CCTVAnalysisPipeline:
    """Modular pre-recorded CCTV analysis pipeline with motion-first filtering."""

    def __init__(
        self,
        config: CCTVAgentConfig | None = None,
        *,
        motion_detector: MotionDetectorProtocol | None = None,
        person_recognizer: PersonRecognizerProtocol | None = None,
        tracker: SimpleTrackManager | None = None,
        action_analyzer: ActionAnalyzer | None = None,
        event_decision_engine: EventDecisionEngine | None = None,
        storage_manager: EventStorageManager | None = None,
        summary_generator: DailySummaryGenerator | None = None,
        video_reader_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config or CCTVAgentConfig()
        self.motion_detector = motion_detector or MotionDetector(self.config)
        self.person_recognizer = person_recognizer or _NullPersonRecognizer()
        self.tracker = tracker or SimpleTrackManager()
        self.action_analyzer = action_analyzer or ActionAnalyzer(self.config)
        self.event_decision_engine = event_decision_engine or EventDecisionEngine()
        self.storage_manager = storage_manager or EventStorageManager(self.config)
        self.summary_generator = summary_generator or DailySummaryGenerator()
        self.video_reader_factory = video_reader_factory or VideoReader
        self._event_counter = 0

    def process_video(self, video_path: str) -> VideoAnalysisResult:
        reader = self.video_reader_factory(video_path)
        metadata = reader.metadata()
        active_event: _ActiveEvent | None = None
        inactive_streak = 0
        activities: list[ActivityRecord] = []
        active_frame_count = 0

        self.motion_detector.reset()
        self.tracker.reset()

        for packet in reader.iter_frames(frame_step=self.config.frame_step):
            motion = self.motion_detector.analyze(packet.frame)
            if not motion.active:
                if active_event is not None:
                    inactive_streak += 1
                    if inactive_streak > self.config.inactivity_tolerance_frames:
                        activities.append(self._finalize_event(active_event, metadata))
                        active_event = None
                        inactive_streak = 0
                        self.tracker.reset()
                continue

            inactive_streak = 0
            active_frame_count += 1
            if active_event is None:
                active_event = self._start_event(packet)

            self._update_event(active_event, packet, metadata, motion)

        if active_event is not None:
            activities.append(self._finalize_event(active_event, metadata))

        summary = self.summary_generator.generate(metadata, activities)
        storage_reduction_ratio = self._calculate_storage_reduction_ratio(
            metadata.total_frames,
            active_frame_count,
        )
        return VideoAnalysisResult(
            metadata=metadata,
            activities=activities,
            summary=summary,
            storage_reduction_ratio=storage_reduction_ratio,
        )

    def _start_event(self, packet: FramePacket) -> _ActiveEvent:
        self._event_counter += 1
        event_id = f"event-{self._event_counter:04d}"
        return _ActiveEvent(
            event_id=event_id,
            start_time_seconds=packet.timestamp_seconds,
            start_frame_index=packet.frame_index,
            end_time_seconds=packet.timestamp_seconds,
            end_frame_index=packet.frame_index,
        )

    def _update_event(
        self,
        event: _ActiveEvent,
        packet: FramePacket,
        metadata: VideoMetadata,
        motion: MotionAnalysis,
    ) -> None:
        event.end_time_seconds = packet.timestamp_seconds
        event.end_frame_index = packet.frame_index
        event.motion_scores.append(motion.motion_score)

        if self.config.save_event_clips:
            event.frames.append(packet.frame.copy())

        people = self.person_recognizer.recognize(packet.frame)
        tracked_people = self.tracker.update(packet.timestamp_seconds, people)
        actions = self.action_analyzer.infer(packet.frame.shape, tracked_people)
        for action in actions:
            if action not in event.actions:
                event.actions.append(action)

        for person in tracked_people:
            key = person.track_id or person.name
            existing = event.people_by_track.get(key)
            if existing is None or person.confidence >= existing.confidence:
                event.people_by_track[key] = self._clone_person(person)

            if (
                not person.known
                and self.config.save_unknown_snapshots
                and person.face_crop is not None
                and key not in event.unknown_snapshot_keys
            ):
                snapshot_path = self.storage_manager.save_unknown_snapshot(
                    person.face_crop,
                    event_id=event.event_id,
                    track_id=person.track_id,
                    timestamp_seconds=packet.timestamp_seconds,
                )
                event.unknown_snapshot_paths.append(snapshot_path)
                event.unknown_snapshot_keys.add(key)
                stored = event.people_by_track[key]
                stored.snapshot_path = snapshot_path

    def _finalize_event(
        self,
        event: _ActiveEvent,
        metadata: VideoMetadata,
    ) -> ActivityRecord:
        clip_fps = self.config.clip_fps or (metadata.fps / max(self.config.frame_step, 1))
        clip_path = self.storage_manager.save_event_clip(
            event.event_id,
            event.frames,
            fps=max(clip_fps, 1.0),
        )
        record = ActivityRecord(
            event_id=event.event_id,
            start_time_seconds=event.start_time_seconds,
            end_time_seconds=event.end_time_seconds,
            start_frame_index=event.start_frame_index,
            end_frame_index=event.end_frame_index,
            motion_score_mean=(
                sum(event.motion_scores) / len(event.motion_scores)
                if event.motion_scores
                else 0.0
            ),
            people=list(event.people_by_track.values()),
            actions=event.actions or ["motion detected"],
            clip_path=clip_path,
            unknown_snapshot_paths=event.unknown_snapshot_paths,
            metadata={"frame_count": len(event.frames)},
        )
        decision = self.event_decision_engine.classify(metadata, record)
        record.event_category = decision.event_category
        record.alert_level = decision.alert_level
        record.alert_reasons = decision.alert_reasons
        record.start_timestamp = decision.start_timestamp
        record.end_timestamp = decision.end_timestamp
        self.storage_manager.append_event_record(record)
        return record

    @staticmethod
    def _calculate_storage_reduction_ratio(total_frames: int, active_frames: int) -> float:
        if total_frames <= 0:
            return 0.0
        inactive_frames = max(total_frames - active_frames, 0)
        return inactive_frames / total_frames

    @staticmethod
    def _clone_person(person: PersonObservation) -> PersonObservation:
        return PersonObservation(
            name=person.name,
            confidence=person.confidence,
            known=person.known,
            bbox=person.bbox,
            track_id=person.track_id,
            tracked_duration_seconds=person.tracked_duration_seconds,
            snapshot_path=person.snapshot_path,
            metadata=dict(person.metadata),
        )

