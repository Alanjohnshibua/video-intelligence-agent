from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from video_intelligence_agent.config import FaceIdentifierConfig
from video_intelligence_agent.cctv.config import CCTVAgentConfig

_T = TypeVar("_T")


@dataclass(slots=True)
class SurveillanceRuntimeConfig:
    project_name: str = "Video Intelligence Agent"
    camera_id: str = "camera-01"
    video_path: str = ""
    output_dir: Path | str = Path("outputs")
    database_path: Path | str = Path("data/embeddings.pkl")
    unknown_dir: Path | str = Path("outputs/unknown")
    similarity_threshold: float = 0.4
    detector_backend: str = "retinaface"
    embedding_model: str = "ArcFace"
    prefer_gpu: bool = True
    tensorflow_memory_growth: bool = True
    frame_step: int = 5
    motion_threshold: float = 0.002
    min_motion_area: int = 500
    inactivity_tolerance_frames: int = 2
    save_event_clips: bool = True
    save_unknown_snapshots: bool = True
    clip_fps: float | None = None
    loitering_threshold_sec: float = 15.0
    walking_distance_px: float = 24.0
    border_margin_ratio: float = 0.12
    interaction_distance_px: float = 120.0
    unknown_person_label: str = "Unknown Person"

    def build_face_identifier_config(self) -> FaceIdentifierConfig:
        return FaceIdentifierConfig(
            database_path=self.database_path,
            unknown_dir=self.unknown_dir,
            similarity_threshold=self.similarity_threshold,
            detector_backend=self.detector_backend,
            embedding_model=self.embedding_model,
            prefer_gpu=self.prefer_gpu,
            tensorflow_memory_growth=self.tensorflow_memory_growth,
        )

    def build_cctv_config(self) -> CCTVAgentConfig:
        return CCTVAgentConfig(
            output_dir=self.output_dir,
            frame_step=self.frame_step,
            motion_threshold=self.motion_threshold,
            min_motion_area=self.min_motion_area,
            inactivity_tolerance_frames=self.inactivity_tolerance_frames,
            save_event_clips=self.save_event_clips,
            save_unknown_snapshots=self.save_unknown_snapshots,
            clip_fps=self.clip_fps,
            loitering_threshold_sec=self.loitering_threshold_sec,
            walking_distance_px=self.walking_distance_px,
            border_margin_ratio=self.border_margin_ratio,
            interaction_distance_px=self.interaction_distance_px,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "SurveillanceRuntimeConfig":
        default = cls()
        return cls(
            project_name=_get_str(data, "project_name", default.project_name),
            camera_id=_get_str(data, "camera_id", default.camera_id),
            video_path=_get_str(data, "video_path", default.video_path),
            output_dir=_get_path_or_str(data, "output_dir", default.output_dir),
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
            motion_threshold=_get_float(data, "motion_threshold", default.motion_threshold),
            min_motion_area=_get_int(data, "min_motion_area", default.min_motion_area),
            inactivity_tolerance_frames=_get_int(
                data,
                "inactivity_tolerance_frames",
                default.inactivity_tolerance_frames,
            ),
            save_event_clips=_get_bool(data, "save_event_clips", default.save_event_clips),
            save_unknown_snapshots=_get_bool(
                data,
                "save_unknown_snapshots",
                default.save_unknown_snapshots,
            ),
            clip_fps=_get_optional_float(data, "clip_fps", default.clip_fps),
            loitering_threshold_sec=_get_float(
                data,
                "loitering_threshold_sec",
                default.loitering_threshold_sec,
            ),
            walking_distance_px=_get_float(
                data,
                "walking_distance_px",
                default.walking_distance_px,
            ),
            border_margin_ratio=_get_float(
                data,
                "border_margin_ratio",
                default.border_margin_ratio,
            ),
            interaction_distance_px=_get_float(
                data,
                "interaction_distance_px",
                default.interaction_distance_px,
            ),
            unknown_person_label=_get_str(
                data,
                "unknown_person_label",
                default.unknown_person_label,
            ),
        )


def load_runtime_config(path: str | Path) -> SurveillanceRuntimeConfig:
    config_path = Path(path)
    payload: dict[str, object] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        payload[key.strip()] = _parse_scalar(raw_value.strip())
    return SurveillanceRuntimeConfig.from_mapping(payload)


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


def _get_optional_float(
    data: dict[str, object],
    key: str,
    default: float | None,
) -> float | None:
    value = _get_value(data, key, default)
    if value is None:
        return None
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _get_bool(data: dict[str, object], key: str, default: bool) -> bool:
    value = _get_value(data, key, default)
    return value if isinstance(value, bool) else default
