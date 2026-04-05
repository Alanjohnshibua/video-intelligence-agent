from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from video_intelligence_agent.core import FaceIdentifier
from video_intelligence_agent.cctv import CCTVAnalysisPipeline, FaceIdentifierRecognizer
from video_intelligence_agent.cctv.models import VideoAnalysisResult
from video_intelligence_agent.surveillance.config import SurveillanceRuntimeConfig


@dataclass(slots=True)
class SurveillanceRunResult:
    analysis: VideoAnalysisResult
    analysis_path: Path
    summary_path: Path


class FaceSurveillanceAgent:
    """High-level runner for the Video Intelligence Agent CCTV workflow."""

    def __init__(
        self,
        config: SurveillanceRuntimeConfig,
        *,
        identifier: FaceIdentifier | None = None,
        pipeline: CCTVAnalysisPipeline | None = None,
    ) -> None:
        self.config = config
        self.identifier = identifier or FaceIdentifier(config.build_face_identifier_config())
        self.pipeline = pipeline or CCTVAnalysisPipeline(
            config=config.build_cctv_config(),
            person_recognizer=FaceIdentifierRecognizer(
                self.identifier,
                unknown_person_label=config.unknown_person_label,
            ),
        )

    def analyze_video(self, video_path: str | None = None) -> VideoAnalysisResult:
        subject = video_path or self.config.video_path
        if not subject:
            raise ValueError("A video_path must be provided in config.yaml or at runtime.")
        return self.pipeline.process_video(subject)

    def run(self, video_path: str | None = None) -> SurveillanceRunResult:
        result = self.analyze_video(video_path=video_path)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis_path = output_dir / "latest_analysis.json"
        summary_path = output_dir / "daily_summary.txt"

        analysis_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary_path.write_text(result.summary.summary_text + "\n", encoding="utf-8")

        return SurveillanceRunResult(
            analysis=result,
            analysis_path=analysis_path,
            summary_path=summary_path,
        )
