from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeAlias

from video_intelligence_agent.image_io import GenericImageArray, save_image

ImageWriter: TypeAlias = Callable[[str | Path, GenericImageArray], None]


class UnknownFaceLogger:
    def __init__(
        self,
        unknown_dir: str | Path,
        image_writer: ImageWriter | None = None,
    ) -> None:
        self.unknown_dir = Path(unknown_dir)
        self.unknown_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.unknown_dir / "unknown_log.jsonl"
        self._image_writer: ImageWriter = image_writer or save_image

    def log(self, image: GenericImageArray) -> dict[str, str]:
        now = datetime.now().astimezone()
        timestamp = now.isoformat()
        safe_stamp = now.strftime("%Y%m%dT%H%M%S_%f")
        file_path = self.unknown_dir / f"unknown_{safe_stamp}.jpg"

        self._image_writer(file_path, image)
        entry = {"timestamp": timestamp, "saved_path": str(file_path)}

        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

        return entry

