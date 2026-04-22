from __future__ import annotations

from pathlib import Path

from video_intelligence_agent.video_library import discover_video_records, output_dir_for_video, slugify_video_path


def test_slugify_video_path_preserves_structure() -> None:
    slug = slugify_video_path(Path("example-street/example-street-20260416-120744.mp4"))
    assert slug == "example-street__example-street-20260416-120744"


def test_discover_video_records_maps_videos_to_dedicated_output_dirs(tmp_path: Path) -> None:
    uploads_dir = tmp_path / "uploads"
    output_root = tmp_path / "outputs"
    nested_dir = uploads_dir / "camera-a"
    nested_dir.mkdir(parents=True)
    video_path = nested_dir / "clip-001.mp4"
    video_path.write_bytes(b"demo")

    records = discover_video_records(uploads_dir, output_root)

    assert len(records) == 1
    assert records[0].relative_path.as_posix() == "camera-a/clip-001.mp4"
    assert records[0].output_dir == output_root / "camera-a__clip-001"
    assert output_dir_for_video(video_path, library_dir=uploads_dir, output_root=output_root) == records[0].output_dir
