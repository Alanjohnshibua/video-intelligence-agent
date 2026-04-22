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
from video_intelligence_agent.video_library import discover_video_records, output_dir_for_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production CCTV video analysis runner")
    parser.add_argument("--config", default="config.yaml", help="Path to the flat config file.")
    parser.add_argument("--video", default="", help="Optional video path override.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and frame dumps.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color in terminal logs.")
    parser.add_argument(
        "--analyze-all-uploads",
        action="store_true",
        help="Process every video found under the configured upload library directory.",
    )
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
        if args.analyze_all_uploads:
            run_results: list[dict[str, object]] = []
            records = discover_video_records(
                config.resolved_video_library_dir(),
                config.resolved_library_output_dir(),
            )
            if not records:
                logger.error("No uploaded videos were found in %s", config.resolved_video_library_dir())
                return 1
            for record in records:
                run_config = _prepare_run_config(
                    config,
                    video_path=str(record.video_path),
                    debug=args.debug,
                    output_dir=record.output_dir,
                )
                processor = VideoProcessor(
                    config=run_config,
                    face_identifier=build_face_identifier(run_config, logger=logger),
                )
                result = processor.process_video()
                run_results.append(
                    {
                        "video_path": str(record.video_path),
                        "output_dir": str(record.output_dir),
                        "events_detected": result.stats.events_detected,
                        "events_path": str(result.artifacts.events_path),
                        "analysis_path": str(result.artifacts.analysis_path) if result.artifacts.analysis_path else None,
                    }
                )
            print(json.dumps(run_results, indent=2))
        else:
            effective_video = args.video or config.video_path
            output_dir = _resolve_output_dir_for_video(config, effective_video)
            run_config = _prepare_run_config(
                config,
                video_path=args.video or None,
                debug=args.debug,
                output_dir=output_dir,
            )

            processor = VideoProcessor(
                config=run_config,
                face_identifier=build_face_identifier(run_config, logger=logger),
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


def _prepare_run_config(
    config: PipelineConfig,
    *,
    video_path: str | None,
    debug: bool,
    output_dir: Path | None,
) -> PipelineConfig:
    updated = config
    if video_path:
        updated = replace(updated, video_path=video_path)
    if output_dir is not None:
        updated = replace(
            updated,
            storage=replace(updated.storage, output_dir=output_dir),
        )
    if debug:
        updated = replace(
            updated,
            debug=replace(updated.debug, enabled=True, save_frames=True, draw_boxes=True),
        )
    return updated


def _resolve_output_dir_for_video(config: PipelineConfig, video_path: str) -> Path | None:
    if not video_path:
        return None

    video = Path(video_path).resolve()
    library_dir = config.resolved_video_library_dir().resolve()
    try:
        video.relative_to(library_dir)
    except ValueError:
        return None

    return output_dir_for_video(
        video,
        library_dir=library_dir,
        output_root=config.resolved_library_output_dir(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
