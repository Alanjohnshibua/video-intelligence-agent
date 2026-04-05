from __future__ import annotations

import json
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from video_intelligence_agent.cctv.config import CCTVAgentConfig
from video_intelligence_agent.cctv.models import ActivityRecord
from video_intelligence_agent.image_io import GenericImageArray, save_image


class OpenCVWriterProtocol(Protocol):
    def write(self, image: GenericImageArray) -> None: ...

    def release(self) -> None: ...


class OpenCVModuleProtocol(Protocol):
    def VideoWriter_fourcc(self, *codes: str) -> int: ...

    def VideoWriter(
        self,
        filename: str,
        fourcc: int,
        fps: float,
        frameSize: tuple[int, int],
    ) -> OpenCVWriterProtocol: ...


def _load_cv2() -> OpenCVModuleProtocol | None:
    try:
        module = import_module("cv2")
    except ImportError:  # pragma: no cover - depends on local environment
        return None
    return cast(OpenCVModuleProtocol, module)


class EventStorageManager:
    def __init__(self, config: CCTVAgentConfig) -> None:
        self.config = config
        self.config.resolved_clips_dir().mkdir(parents=True, exist_ok=True)
        self.config.resolved_snapshots_dir().mkdir(parents=True, exist_ok=True)
        self.config.resolved_manifests_dir().mkdir(parents=True, exist_ok=True)

    def save_unknown_snapshot(
        self,
        face_crop: GenericImageArray,
        *,
        event_id: str,
        track_id: str | None,
        timestamp_seconds: float,
    ) -> str:
        filename = (
            f"{event_id}_{track_id or 'track'}_{int(timestamp_seconds * 1000):08d}.jpg"
        )
        path = self.config.resolved_snapshots_dir() / filename
        save_image(path, face_crop)
        return str(path)

    def save_event_clip(
        self,
        event_id: str,
        frames: list[GenericImageArray],
        *,
        fps: float,
    ) -> str | None:
        if not self.config.save_event_clips or not frames:
            return None

        cv2_module = _load_cv2()
        if cv2_module is None:
            return None

        path = self.config.resolved_clips_dir() / f"{event_id}.mp4"
        height, width = frames[0].shape[:2]
        writer = cv2_module.VideoWriter(
            str(path),
            cv2_module.VideoWriter_fourcc(*self.config.clip_codec),
            fps,
            (width, height),
        )
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return str(path)

    def append_event_record(self, record: ActivityRecord) -> None:
        payload = record.to_dict()
        payload["logged_at"] = datetime.now().isoformat()
        manifest_path = self.config.resolved_events_manifest_path()
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

