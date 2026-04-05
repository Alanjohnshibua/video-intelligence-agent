from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from video_intelligence_agent.cctv_pipeline import BaseAppError, PipelineConfig, VideoProcessor, load_pipeline_config
from video_intelligence_agent.cctv_pipeline.utils.logger import configure_logging, get_pipeline_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production CCTV video analysis runner")
    parser.add_argument("--config", default="config.yaml", help="Path to the flat config file.")
    parser.add_argument("--video", default="", help="Optional video path override.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and frame dumps.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color in terminal logs.")
    parser.add_argument("--query-person", default="", help="Optional person_id filter for event queries.")
    parser.add_argument("--query-action", default="", help="Optional action filter for event queries.")
    parser.add_argument("--query-limit", type=int, default=20, help="Maximum number of queried events to return.")
    return parser


def build_face_identifier(config: PipelineConfig, *, logger):
    try:
        from video_intelligence_agent.config import FaceIdentifierConfig
        from video_intelligence_agent.core import FaceIdentifier
    except Exception:
        logger.warning(
            "Face recognition dependencies are unavailable. People will be labeled as unknown."
        )
        return None

    try:
        return FaceIdentifier(
            FaceIdentifierConfig(
                database_path=config.database_path,
                unknown_dir=config.unknown_dir,
                similarity_threshold=config.similarity_threshold,
                detector_backend=config.detector_backend,
                embedding_model=config.embedding_model,
                prefer_gpu=config.prefer_gpu,
                tensorflow_memory_growth=config.tensorflow_memory_growth,
            )
        )
    except Exception as exc:
        logger.warning(
            "Face recognition failed to initialize. People will be labeled as unknown. reason=%s",
            exc,
        )
        return None


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(debug=args.debug, use_color=not args.no_color)
    logger = get_pipeline_logger("main")

    try:
        config = load_pipeline_config(args.config)
        if args.video:
            config = replace(config, video_path=args.video)
        if args.debug:
            config = replace(
                config,
                debug=replace(config.debug, enabled=True, save_frames=True, draw_boxes=True),
            )

        processor = VideoProcessor(
            config=config,
            face_identifier=build_face_identifier(config, logger=logger),
        )
        result = processor.process_video()

        if args.query_person or args.query_action:
            query_results = processor.query_events(
                person_id=args.query_person or None,
                action=args.query_action or None,
                limit=args.query_limit,
            )
            print(json.dumps(query_results, indent=2))
        else:
            print(json.dumps(result.to_dict(), indent=2))
        return 0
    except BaseAppError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - fatal fallback
        logger.error("Fatal pipeline error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
