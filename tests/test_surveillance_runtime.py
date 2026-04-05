from __future__ import annotations

from pathlib import Path

from video_intelligence_agent.surveillance import load_runtime_config


def test_load_runtime_config_reads_flat_yaml_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: Video Intelligence Agent",
                "camera_id: entrance-cam",
                "video_path: demo.mp4",
                "frame_step: 7",
                "save_event_clips: false",
                "clip_fps: 4.5",
                "prefer_gpu: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.project_name == "Video Intelligence Agent"
    assert config.camera_id == "entrance-cam"
    assert config.video_path == "demo.mp4"
    assert config.frame_step == 7
    assert config.save_event_clips is False
    assert config.clip_fps == 4.5
    assert config.prefer_gpu is False
