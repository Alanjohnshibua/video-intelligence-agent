from __future__ import annotations

from pathlib import Path

from video_intelligence_agent.cctv_pipeline.core.video_processor import OpenCVVideoSource
from video_intelligence_agent.cctv_pipeline.models import FramePacket, VideoMetadata


class VideoIngestionService:
    """File-based video ingestion service for the production pipeline."""

    def __init__(self, video_path: str | Path) -> None:
        self._source = OpenCVVideoSource(video_path)

    def metadata(self) -> VideoMetadata:
        return self._source.metadata()

    def iter_frames(self, *, frame_step: int = 1):
        return self._source.iter_frames(frame_step=frame_step)

    def close(self) -> None:
        self._source.close()


__all__ = ["FramePacket", "VideoIngestionService", "VideoMetadata"]
