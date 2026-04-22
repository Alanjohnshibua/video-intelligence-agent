from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from video_intelligence_agent.cctv_pipeline.utils.error_handler import ConfigurationError

_T = TypeVar("_T")


@dataclass(slots=True)
class DebugConfig:
    """Runtime switches that make the pipeline easier to inspect."""

    enabled: bool = False
    save_frames: bool = False
    draw_boxes: bool = True
    live_preview: bool = False


@dataclass(slots=True)
class StorageConfig:
    """Filesystem output configuration."""

    output_dir: Path | str = Path("outputs/cctv_pipeline")
    event_filename: str = "events.json"
    save_event_clips: bool = True
    save_unknown_clips: bool = True
    clip_seconds_before: float = 2.0
    clip_seconds_after: float = 2.0
    clip_codec: str = "avc1"


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for the CCTV analysis pipeline."""

    video_path: str = ""
    video_library_dir: Path | str = Path("data/clip_library/uploads")
    library_output_dir: Path | str = Path("outputs/clip_library")
    yolo_model_path: str = "yolov8n.pt"
    yolo_device: str = "cpu"
    database_path: Path | str = Path("data/embeddings.pkl")
    unknown_dir: Path | str = Path("outputs/unknown")
    similarity_threshold: float = 0.4
    detector_backend: str = "retinaface"
    embedding_model: str = "ArcFace"
    prefer_gpu: bool = True
    tensorflow_memory_growth: bool = True
    frame_step: int = 1
    detection_confidence: float = 0.25
    motion_threshold: float = 0.01
    min_motion_area: int = 500
    tracker_backend: str = "bytetrack"
    tracker_max_lost: int = 20
    tracker_iou_threshold: float = 0.3
    loitering_seconds: float = 15.0
    loitering_radius_px: float = 35.0
    movement_min_distance_px: float = 24.0
    border_margin_ratio: float = 0.12
    unknown_label_prefix: str = "unknown"
    debug: DebugConfig = field(default_factory=DebugConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def resolved_output_dir(self) -> Path:
        return Path(self.storage.output_dir)

    def resolved_video_library_dir(self) -> Path:
        return Path(self.video_library_dir)

    def resolved_library_output_dir(self) -> Path:
        return Path(self.library_output_dir)

    def resolved_events_path(self) -> Path:
        return self.resolved_output_dir() / self.storage.event_filename

    def resolved_analysis_path(self) -> Path:
        return self.resolved_output_dir() / "latest_analysis.json"

    def resolved_summary_path(self) -> Path:
        return self.resolved_output_dir() / "daily_summary.txt"

    def resolved_clips_dir(self) -> Path:
        return self.resolved_output_dir() / "clips"

    def resolved_debug_dir(self) -> Path:
        return self.resolved_output_dir() / "debug"

    def validate(self) -> None:
        if self.frame_step <= 0:
            raise ConfigurationError("frame_step must be greater than 0.", module="config")
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ConfigurationError(
                "detection_confidence must be between 0 and 1.",
                module="config",
            )
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ConfigurationError(
                "similarity_threshold must be between 0 and 1.",
                module="config",
            )
        if self.motion_threshold < 0.0:
            raise ConfigurationError(
                "motion_threshold must be greater than or equal to 0.",
                module="config",
            )
        if self.min_motion_area < 0:
            raise ConfigurationError(
                "min_motion_area must be greater than or equal to 0.",
                module="config",
            )
        if self.tracker_max_lost < 0:
            raise ConfigurationError(
                "tracker_max_lost must be greater than or equal to 0.",
                module="config",
            )
        if not 0.0 < self.tracker_iou_threshold <= 1.0:
            raise ConfigurationError(
                "tracker_iou_threshold must be between 0 and 1.",
                module="config",
            )
        if self.loitering_seconds < 0.0:
            raise ConfigurationError(
                "loitering_seconds must be greater than or equal to 0.",
                module="config",
            )
        if self.loitering_radius_px < 0.0:
            raise ConfigurationError(
                "loitering_radius_px must be greater than or equal to 0.",
                module="config",
            )
        if self.movement_min_distance_px < 0.0:
            raise ConfigurationError(
                "movement_min_distance_px must be greater than or equal to 0.",
                module="config",
            )
        if not 0.0 <= self.border_margin_ratio < 0.5:
            raise ConfigurationError(
                "border_margin_ratio must be between 0 and 0.5.",
                module="config",
            )
        if self.storage.clip_seconds_before < 0.0 or self.storage.clip_seconds_after < 0.0:
            raise ConfigurationError(
                "clip_seconds_before and clip_seconds_after must be non-negative.",
                module="config",
            )

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "PipelineConfig":
        default = cls()
        debug = DebugConfig(
            enabled=_get_bool(data, "debug_enabled", default.debug.enabled),
            save_frames=_get_bool(data, "save_debug_frames", default.debug.save_frames),
            draw_boxes=_get_bool(data, "draw_debug_boxes", default.debug.draw_boxes),
            live_preview=_get_bool(data, "live_debug_preview", default.debug.live_preview),
        )
        storage = StorageConfig(
            output_dir=_get_path_or_str(data, "output_dir", default.storage.output_dir),
            event_filename=_get_str(data, "event_filename", default.storage.event_filename),
            save_event_clips=_get_bool(
                data,
                "save_event_clips",
                default.storage.save_event_clips,
            ),
            save_unknown_clips=_get_bool(
                data,
                "save_unknown_clips",
                default.storage.save_unknown_clips,
            ),
            clip_seconds_before=_get_float(
                data,
                "clip_seconds_before",
                default.storage.clip_seconds_before,
            ),
            clip_seconds_after=_get_float(
                data,
                "clip_seconds_after",
                default.storage.clip_seconds_after,
            ),
            clip_codec=_get_str(data, "clip_codec", default.storage.clip_codec),
        )
        config = cls(
            video_path=_get_str(data, "video_path", default.video_path),
            video_library_dir=_get_path_or_str(data, "video_library_dir", default.video_library_dir),
            library_output_dir=_get_path_or_str(
                data,
                "library_output_dir",
                default.library_output_dir,
            ),
            yolo_model_path=_get_str(data, "yolo_model_path", default.yolo_model_path),
            yolo_device=_get_str(data, "yolo_device", default.yolo_device),
            database_path=_get_path_or_str(data, "database_path", default.database_path),
            unknown_dir=_get_path_or_str(data, "unknown_dir", default.unknown_dir),
            similarity_threshold=_get_float(
                data,
                "similarity_threshold",
                default.similarity_threshold,
            ),
            detector_backend=_get_str(data, "detector_backend", default.detector_backend),
            embedding_model=_get_str(data, "embedding_model", default.embedding_model),
            prefer_gpu=_get_bool(data, "prefer_gpu", default.prefer_gpu),
            tensorflow_memory_growth=_get_bool(
                data,
                "tensorflow_memory_growth",
                default.tensorflow_memory_growth,
            ),
            frame_step=_get_int(data, "frame_step", default.frame_step),
            detection_confidence=_get_float(
                data,
                "detection_confidence",
                default.detection_confidence,
            ),
            motion_threshold=_get_float(data, "motion_threshold", default.motion_threshold),
            min_motion_area=_get_int(data, "min_motion_area", default.min_motion_area),
            tracker_backend=_get_str(data, "tracker_backend", default.tracker_backend),
            tracker_max_lost=_get_int(data, "tracker_max_lost", default.tracker_max_lost),
            tracker_iou_threshold=_get_float(
                data,
                "tracker_iou_threshold",
                default.tracker_iou_threshold,
            ),
            loitering_seconds=_get_float_with_aliases(
                data,
                ["loitering_seconds", "loitering_threshold_sec"],
                default.loitering_seconds,
            ),
            loitering_radius_px=_get_float(
                data,
                "loitering_radius_px",
                default.loitering_radius_px,
            ),
            movement_min_distance_px=_get_float_with_aliases(
                data,
                ["movement_min_distance_px", "walking_distance_px"],
                default.movement_min_distance_px,
            ),
            border_margin_ratio=_get_float(
                data,
                "border_margin_ratio",
                default.border_margin_ratio,
            ),
            unknown_label_prefix=_get_str(
                data,
                "unknown_label_prefix",
                default.unknown_label_prefix,
            ),
            debug=debug,
            storage=storage,
        )
        config.validate()
        return config


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load a simple flat YAML-like config file without extra dependencies."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(
            f"Config file not found: {config_path}",
            module="config",
        )

    payload: dict[str, object] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        payload[key.strip()] = _parse_scalar(raw_value.strip())
    return PipelineConfig.from_mapping(payload)


def _parse_scalar(raw_value: str) -> object:
    if raw_value == "":
        return ""
    if raw_value.startswith(("'", '"')) and raw_value.endswith(("'", '"')):
        return raw_value[1:-1]

    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    try:
        if "." in raw_value:
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def _get_value(data: dict[str, object], key: str, default: _T) -> object | _T:
    return data.get(key, default)


def _get_str(data: dict[str, object], key: str, default: str) -> str:
    value = _get_value(data, key, default)
    return value if isinstance(value, str) else default


def _get_path_or_str(data: dict[str, object], key: str, default: Path | str) -> Path | str:
    value = _get_value(data, key, default)
    if isinstance(value, (Path, str)):
        return value
    return default


def _get_int(data: dict[str, object], key: str, default: int) -> int:
    value = _get_value(data, key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _get_float(data: dict[str, object], key: str, default: float) -> float:
    value = _get_value(data, key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _get_bool(data: dict[str, object], key: str, default: bool) -> bool:
    value = _get_value(data, key, default)
    return value if isinstance(value, bool) else default


def _get_float_with_aliases(data: dict[str, object], keys: list[str], default: float) -> float:
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return default
