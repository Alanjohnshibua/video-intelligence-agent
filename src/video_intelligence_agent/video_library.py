from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass(slots=True)
class VideoRecord:
    """Discovered video plus the output folder reserved for its analysis artifacts."""

    video_path: Path
    relative_path: Path
    slug: str
    output_dir: Path

    @property
    def display_name(self) -> str:
        return self.relative_path.as_posix()


def is_video_file(path: Path) -> bool:
    """Return True when *path* looks like a supported video file."""
    return path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES


def discover_video_records(library_dir: Path | str, output_root: Path | str) -> list[VideoRecord]:
    """Recursively discover uploaded videos and map each one to a stable output directory."""
    library_path = Path(library_dir)
    output_root_path = Path(output_root)
    if not library_path.exists():
        return []

    records: list[VideoRecord] = []
    for video_path in sorted(path for path in library_path.rglob("*") if is_video_file(path)):
        relative_path = video_path.relative_to(library_path)
        slug = slugify_video_path(relative_path)
        records.append(
            VideoRecord(
                video_path=video_path,
                relative_path=relative_path,
                slug=slug,
                output_dir=output_root_path / slug,
            )
        )
    return records


def slugify_video_path(video_path: Path | str) -> str:
    """Build a filesystem-safe, human-readable slug from a relative video path."""
    path = Path(video_path)
    parts = [part for part in path.with_suffix("").parts if part not in {".", ""}]
    if not parts:
        return "video"
    joined = "__".join(parts)
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", joined).strip("-._")
    return sanitized or "video"


def output_dir_for_video(
    video_path: Path | str,
    *,
    library_dir: Path | str,
    output_root: Path | str,
) -> Path:
    """Return the dedicated output directory for a video inside the configured library."""
    video_path = Path(video_path).resolve()
    library_path = Path(library_dir).resolve()
    output_root_path = Path(output_root)
    relative_path = video_path.relative_to(library_path)
    return output_root_path / slugify_video_path(relative_path)
