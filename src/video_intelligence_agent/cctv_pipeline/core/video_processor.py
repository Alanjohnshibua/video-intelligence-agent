from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from video_intelligence_agent.cctv_pipeline.core.detector import YOLOPersonDetector
from video_intelligence_agent.cctv_pipeline.core.event_logic import RuleBasedEventDetector
from video_intelligence_agent.cctv_pipeline.core.motion_detector import MotionDetector
from video_intelligence_agent.cctv_pipeline.core.recognition import (
    FaceIdentifierProtocol,
    FaceRecognitionService,
)
from video_intelligence_agent.cctv_pipeline.core.tracker import MultiObjectTracker
from video_intelligence_agent.cctv_pipeline.models import (
    EventRecord,
    FramePacket,
    PipelineArtifacts,
    PipelineStats,
    TrackedPerson,
    VideoMetadata,
)
from video_intelligence_agent.cctv_pipeline.services.clip_manager import ClipManager
from video_intelligence_agent.cctv_pipeline.services.event_logger import EventLoggerService
from video_intelligence_agent.cctv_pipeline.utils.config import PipelineConfig
from video_intelligence_agent.cctv_pipeline.utils.error_handler import (
    BaseAppError,
    DetectionError,
    ErrorTracker,
    EventStorageError,
    VideoInputError,
    log_exception,
)
from video_intelligence_agent.cctv_pipeline.utils.logger import configure_logging, get_pipeline_logger

if TYPE_CHECKING:
    import numpy as np


@dataclass(slots=True)
class PipelineRunResult:
    """Return object for a complete pipeline execution."""

    metadata: VideoMetadata
    artifacts: PipelineArtifacts
    events: list[EventRecord]
    stats: PipelineStats
    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "stats": self.stats.to_dict(),
            "errors": self.errors,
            "events": [event.to_dict() for event in self.events],
        }


class OpenCVVideoSource:
    """File-based frame reader used by the production pipeline."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        self._cv2 = self._load_cv2()
        if self._cv2 is None:
            raise VideoInputError(
                "OpenCV is required to read video files.",
                module="video_processor",
                context={"video_path": str(self.video_path)},
            )
        if not self.video_path.exists():
            raise VideoInputError(
                f"Video path does not exist: {self.video_path}",
                module="video_processor",
                context={"video_path": str(self.video_path)},
            )

        self._capture = self._cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            raise VideoInputError(
                f"Unable to open video file: {self.video_path}",
                module="video_processor",
                context={"video_path": str(self.video_path)},
            )

    def metadata(self) -> VideoMetadata:
        fps = float(self._capture.get(self._cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(self._capture.get(self._cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(self._capture.get(self._cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_seconds = (total_frames / fps) if fps > 0 else 0.0
        return VideoMetadata(
            video_path=str(self.video_path),
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
        )

    def iter_frames(self, *, frame_step: int = 1):
        frame_index = 0
        stride = max(frame_step, 1)
        fps = float(self._capture.get(self._cv2.CAP_PROP_FPS) or 0.0)

        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            if frame is None or getattr(frame, "size", 0) == 0:
                yield FramePacket(frame_index=frame_index, timestamp_seconds=0.0, frame=frame)
                frame_index += 1
                continue
            if frame_index % stride == 0:
                timestamp = (frame_index / fps) if fps > 0 else float(frame_index)
                yield FramePacket(frame_index=frame_index, timestamp_seconds=timestamp, frame=frame)
            frame_index += 1

    def close(self) -> None:
        self._capture.release()

    @staticmethod
    def _load_cv2():
        try:
            import cv2
        except ImportError:  # pragma: no cover - depends on local environment
            return None
        return cv2


class VideoProcessor:
    """Orchestrates the full CCTV analysis pipeline with resilient error handling."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        face_identifier: FaceIdentifierProtocol | None = None,
        motion_detector: MotionDetector | None = None,
        detector: YOLOPersonDetector | None = None,
        tracker: MultiObjectTracker | None = None,
        recognizer: FaceRecognitionService | None = None,
        event_logic: RuleBasedEventDetector | None = None,
        event_logger: EventLoggerService | None = None,
        clip_manager: ClipManager | None = None,
        video_source_cls: type[OpenCVVideoSource] = OpenCVVideoSource,
    ) -> None:
        self.config = config
        self.config.validate()
        configure_logging(debug=config.debug.enabled)
        self.logger = get_pipeline_logger("video_processor")
        self.error_tracker = ErrorTracker()
        self.video_source_cls = video_source_cls

        self.motion_detector = motion_detector or MotionDetector(
            motion_threshold=config.motion_threshold,
            min_motion_area=config.min_motion_area,
        )
        self.detector = detector or YOLOPersonDetector(
            model_path=config.yolo_model_path,
            confidence_threshold=config.detection_confidence,
            device=config.yolo_device,
        )
        self.tracker = tracker or MultiObjectTracker(
            backend=config.tracker_backend,
            max_lost=config.tracker_max_lost,
            iou_threshold=config.tracker_iou_threshold,
        )
        self.recognizer = recognizer or FaceRecognitionService(
            identifier=face_identifier,
            unknown_label_prefix=config.unknown_label_prefix,
        )
        self.event_logic = event_logic or RuleBasedEventDetector(
            loitering_seconds=config.loitering_seconds,
            loitering_radius_px=config.loitering_radius_px,
            border_margin_ratio=config.border_margin_ratio,
        )
        self.event_logger = event_logger or EventLoggerService(config.resolved_events_path())
        self.clip_manager = clip_manager

    def process_video(self, video_path: str | None = None) -> PipelineRunResult:
        subject = video_path or self.config.video_path
        if not subject:
            raise VideoInputError(
                "No input video path was provided.",
                module="video_processor",
            )

        source = self.video_source_cls(subject)
        metadata = source.metadata()
        if metadata.total_frames == 0:
            source.close()
            raise VideoInputError(
                "The input video appears to be empty.",
                module="video_processor",
                context={"video_path": subject},
            )

        self.logger.info("Starting CCTV analysis for %s", metadata.video_path)
        if not self.detector.ready and self.detector.load_error is not None:
            log_exception(
                self.logger,
                DetectionError(
                    "YOLO model initialization failed. Motion analysis will continue, but person detection is disabled.",
                    module="detector",
                    cause=self.detector.load_error,
                ),
                error_tracker=self.error_tracker,
            )

        self.tracker.reset()
        self.motion_detector.reset()
        self.event_logic.reset()

        clip_manager = self.clip_manager or ClipManager(
            clips_dir=self.config.resolved_clips_dir(),
            debug_dir=self.config.resolved_debug_dir(),
            codec=self.config.storage.clip_codec,
            fps=metadata.fps if metadata.fps > 0 else 1.0,
            pre_event_seconds=self.config.storage.clip_seconds_before,
        )
        self.clip_manager = clip_manager

        stats = PipelineStats()
        try:
            for packet in source.iter_frames(frame_step=self.config.frame_step):
                stats.total_frames_read += 1
                self._process_packet(packet, stats=stats)
        finally:
            source.close()

        try:
            trailing_events = self.event_logic.flush()
            if trailing_events:
                self.event_logger.extend(trailing_events)
                stats.events_detected += len(trailing_events)
        except BaseAppError as exc:
            log_exception(self.logger, exc, error_tracker=self.error_tracker)

        stats.errors_encountered = self.error_tracker.count
        result = PipelineRunResult(
            metadata=metadata,
            artifacts=PipelineArtifacts(
                events_path=self.config.resolved_events_path(),
                clips_dir=self.config.resolved_clips_dir(),
                debug_dir=self.config.resolved_debug_dir() if self.config.debug.enabled else None,
            ),
            events=self.event_logger.events(),
            stats=stats,
            errors=self.error_tracker.to_list(),
        )
        self.logger.info(
            "Processing complete | frames=%s | events=%s | errors=%s",
            stats.processed_frames,
            stats.events_detected,
            stats.errors_encountered,
        )
        return result

    def query_events(
        self,
        *,
        person_id: str | None = None,
        action: str | None = None,
        track_id: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        return self.event_logger.query_events(
            person_id=person_id,
            action=action,
            track_id=track_id,
            limit=limit,
        )

    def _process_packet(self, packet: FramePacket, *, stats: PipelineStats) -> None:
        frame_index = packet.frame_index
        frame = packet.frame
        if frame is None or getattr(frame, "size", 0) == 0:
            stats.corrupted_frames += 1
            self.error_tracker.add(
                module="video_processor",
                message="Corrupted or empty frame encountered.",
                frame_index=frame_index,
                level="WARNING",
            )
            self.logger.warning("Corrupted frame skipped | frame=%s", frame_index)
            return

        self.clip_manager.buffer_frame(frame)

        try:
            motion = self.motion_detector.analyze(frame, frame_index=frame_index)
        except BaseAppError as exc:
            log_exception(self.logger, exc, error_tracker=self.error_tracker)
            return

        if not motion.active:
            stats.skipped_frames += 1
            try:
                lost_update = self.tracker.update(
                    [],
                    frame_index=frame_index,
                    timestamp_seconds=packet.timestamp_seconds,
                )
            except BaseAppError as exc:
                log_exception(self.logger, exc, error_tracker=self.error_tracker)
                return
            self._handle_events(
                visible_tracks=[],
                lost_tracks=lost_update.lost_tracks,
                frame_shape=frame.shape,
                frame_index=frame_index,
                stats=stats,
            )
            self._maybe_write_debug_frame(
                frame=frame,
                frame_index=frame_index,
                motion_boxes=motion.boxes,
                detections=[],
                tracks=[],
            )
            return

        stats.motion_frames += 1

        try:
            detections = self.detector.detect(frame, frame_index=frame_index) if self.detector.ready else []
        except BaseAppError as exc:
            log_exception(self.logger, exc, error_tracker=self.error_tracker)
            return

        stats.detections += len(detections)

        try:
            tracking_update = self.tracker.update(
                detections,
                frame_index=frame_index,
                timestamp_seconds=packet.timestamp_seconds,
            )
        except BaseAppError as exc:
            log_exception(self.logger, exc, error_tracker=self.error_tracker)
            return

        identified_tracks = self.recognizer.identify_tracks(
            frame,
            tracking_update.visible_tracks,
            frame_index=frame_index,
        )
        self.tracker.apply_identity_updates(identified_tracks)
        stats.tracked_objects += len(identified_tracks)
        stats.processed_frames += 1

        self._handle_events(
            visible_tracks=identified_tracks,
            lost_tracks=tracking_update.lost_tracks,
            frame_shape=frame.shape,
            frame_index=frame_index,
            stats=stats,
        )
        self._maybe_write_debug_frame(
            frame=frame,
            frame_index=frame_index,
            motion_boxes=motion.boxes,
            detections=[item.bbox for item in detections],
            tracks=identified_tracks,
        )
        self.logger.info("Frame %s processed successfully", frame_index)

    def _handle_events(
        self,
        *,
        visible_tracks: list[TrackedPerson],
        lost_tracks: list[TrackedPerson],
        frame_shape: tuple[int, ...],
        frame_index: int,
        stats: PipelineStats,
    ) -> None:
        try:
            events = self.event_logic.update(
                visible_tracks=visible_tracks,
                lost_tracks=lost_tracks,
                frame_shape=frame_shape,
            )
            if not events:
                return
            for event in events:
                if (
                    event.person_id.startswith(f"{self.config.unknown_label_prefix}_")
                    and self.config.storage.save_unknown_clips
                ) or self.config.storage.save_event_clips:
                    clip_path = self.clip_manager.save_clip_from_buffer(
                        f"{event.event_id}_track-{event.track_id or 0:04d}"
                    )
                    if clip_path is not None:
                        event.clip_path = clip_path
            self.event_logger.extend(events)
            stats.events_detected += len(events)
            for event in events:
                self.logger.info(
                    "Event detected | frame=%s | action=%s | person=%s",
                    frame_index,
                    event.action,
                    event.person_id,
                )
        except BaseAppError as exc:
            log_exception(self.logger, exc, error_tracker=self.error_tracker)
        except Exception as exc:
            wrapped = EventStorageError(
                "Unexpected failure while handling events.",
                module="event_logger",
                frame_index=frame_index,
                cause=exc,
            )
            log_exception(self.logger, wrapped, error_tracker=self.error_tracker)

    def _maybe_write_debug_frame(
        self,
        *,
        frame: "np.ndarray",
        frame_index: int,
        motion_boxes: list[tuple[int, int, int, int]],
        detections: list[tuple[int, int, int, int]],
        tracks: list[TrackedPerson],
    ) -> None:
        if not self.config.debug.enabled or not self.config.debug.save_frames:
            return

        try:
            annotated = frame.copy()
            if self.config.debug.draw_boxes:
                annotated = self._annotate_frame(
                    annotated,
                    motion_boxes=motion_boxes,
                    detections=detections,
                    tracks=tracks,
                )
            self.clip_manager.save_debug_frame(annotated, frame_index=frame_index)
        except BaseAppError as exc:
            log_exception(self.logger, exc, error_tracker=self.error_tracker)

    def _annotate_frame(
        self,
        frame: "np.ndarray",
        *,
        motion_boxes: list[tuple[int, int, int, int]],
        detections: list[tuple[int, int, int, int]],
        tracks: list[TrackedPerson],
    ) -> "np.ndarray":
        try:
            import cv2
        except ImportError:  # pragma: no cover - depends on local environment
            return frame

        for x1, y1, x2, y2 in motion_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                "motion",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
        for x1, y1, x2, y2 in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                "person",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            label = f"{track.identity}#{track.track_id}"
            color = (0, 200, 0) if track.known else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return frame
