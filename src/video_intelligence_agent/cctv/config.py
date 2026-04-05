from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CCTVAgentConfig:
    output_dir: Path | str = Path("data/cctv_agent")
    frame_step: int = 5
    motion_threshold: float = 0.002
    min_motion_area: int = 500
    inactivity_tolerance_frames: int = 2
    save_event_clips: bool = True
    save_unknown_snapshots: bool = True
    clip_codec: str = "mp4v"
    clip_fps: float | None = None
    loitering_threshold_sec: float = 15.0
    walking_distance_px: float = 24.0
    border_margin_ratio: float = 0.12
    interaction_distance_px: float = 120.0

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)

    def resolved_clips_dir(self) -> Path:
        return self.resolved_output_dir() / "clips"

    def resolved_snapshots_dir(self) -> Path:
        return self.resolved_output_dir() / "snapshots"

    def resolved_manifests_dir(self) -> Path:
        return self.resolved_output_dir() / "manifests"

    def resolved_events_manifest_path(self) -> Path:
        return self.resolved_manifests_dir() / "events.jsonl"
