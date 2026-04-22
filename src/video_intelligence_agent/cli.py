from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from video_intelligence_agent.config import FaceIdentifierConfig
from video_intelligence_agent.core import FaceIdentifier
from video_intelligence_agent.image_io import GenericImageArray, load_image
from video_intelligence_agent.video_library import discover_video_records, output_dir_for_video
from video_intelligence_agent.video_summarizer import DEFAULT_LIGHTWEIGHT_MODEL_PATH


class VideoCaptureProtocol(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, GenericImageArray | None]: ...

    def release(self) -> None: ...


class OpenCVModuleProtocol(Protocol):
    def VideoCapture(self, filename: str) -> VideoCaptureProtocol: ...


def _load_cv2() -> OpenCVModuleProtocol | None:
    try:
        module = import_module("cv2")
    except ImportError:  # pragma: no cover - depends on local environment
        return None
    return cast(OpenCVModuleProtocol, module)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video Intelligence Agent (VIA) CLI")
    parser.add_argument("--db", default="data/embeddings.pkl", help="Path to the embedding database.")
    parser.add_argument("--unknown-dir", default="data/unknown", help="Directory for unknown face logs.")
    parser.add_argument("--threshold", type=float, default=0.4, help="Cosine similarity threshold.")
    parser.add_argument("--verbose", action="store_true", help="Enable info-level logging.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_person = subparsers.add_parser("add-person", help="Enroll a person from an image.")
    add_person.add_argument("--name", required=True, help="Person name.")
    add_person.add_argument("--image", required=True, help="Image path.")

    identify = subparsers.add_parser("identify-image", help="Identify faces in an image.")
    identify.add_argument("--image", required=True, help="Image path.")

    batch = subparsers.add_parser("batch-identify", help="Identify faces in multiple images.")
    batch.add_argument("--input-dir", required=True, help="Directory containing images.")

    process_video = subparsers.add_parser("process-video", help="Process a video file frame-by-frame.")
    process_video.add_argument("--video", required=True, help="Video path.")
    process_video.add_argument("--frame-step", type=int, default=1, help="Analyze every Nth frame.")

    analyze_cctv = subparsers.add_parser(
        "analyze-cctv",
        help="Run the modular CCTV analysis pipeline on a pre-recorded video.",
    )
    analyze_cctv.add_argument("--config", default="config.yaml", help="Path to the flat config file.")
    analyze_cctv.add_argument("--video", default="", help="Optional video path override.")
    analyze_cctv.add_argument("--debug", action="store_true", help="Enable debug logs and saved debug frames.")
    analyze_cctv.add_argument(
        "--analyze-all-uploads",
        action="store_true",
        help="Process every uploaded video under the configured library directory.",
    )
    analyze_cctv.add_argument("--query-person", default="", help="Optional person_id filter for event queries.")
    analyze_cctv.add_argument("--query-action", default="", help="Optional action filter for event queries.")
    analyze_cctv.add_argument(
        "--query-limit",
        type=int,
        default=20,
        help="Maximum number of queried events to return.",
    )

    summarize_video = subparsers.add_parser(
        "summarize-video",
        help="Generate a multimodal summary for a video using transcript and optional scene cues.",
    )
    summarize_video.add_argument("--video", required=True, help="Video path.")
    summarize_video.add_argument(
        "--llm-backend",
        default="transformers",
        choices=["transformers", "openai"],
        help="Use a local Transformers model or an OpenAI-compatible API backend.",
    )
    summarize_video.add_argument(
        "--llm-base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible base URL, for example Ollama or another local server.",
    )
    summarize_video.add_argument("--llm-api-key", default=None, help="API key for the LLM endpoint.")
    summarize_video.add_argument("--llm-model", default="qwen2.5:2b", help="LLM model name.")
    summarize_video.add_argument(
        "--llm-model-path",
        default=DEFAULT_LIGHTWEIGHT_MODEL_PATH,
        help="Local Hugging Face model path used when --llm-backend transformers.",
    )
    summarize_video.add_argument("--whisper-model", default="base", help="Faster-Whisper model name.")
    summarize_video.add_argument(
        "--whisper-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used by Faster-Whisper.",
    )
    summarize_video.add_argument(
        "--language",
        default="en",
        help="Transcription language code. Use Python if you need full auto-detection control.",
    )
    summarize_video.add_argument(
        "--summary-tokens",
        type=int,
        default=512,
        help="Maximum number of output tokens for the summary.",
    )
    summarize_video.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the LLM.",
    )
    summarize_video.add_argument(
        "--module-context",
        default="",
        help="Optional extra instructions or module notes to inject into the prompt.",
    )
    summarize_video.add_argument(
        "--keep-temp-audio",
        action="store_true",
        help="Keep the extracted WAV file instead of deleting it after processing.",
    )
    summarize_video.add_argument(
        "--no-person-id",
        action="store_true",
        help="Skip person identification and summarize from transcript only.",
    )
    summarize_video.add_argument(
        "--scene-analysis",
        action="store_true",
        help="Run YOLOv8 Nano scene analysis on sampled frames and include it in the summary.",
    )
    summarize_video.add_argument(
        "--scene-model-repo",
        default="Ultralytics/YOLOv8",
        help="Hugging Face repo that hosts the YOLOv8 Nano checkpoint.",
    )
    summarize_video.add_argument(
        "--scene-model-file",
        default="yolov8n.pt",
        help="Checkpoint filename downloaded from the Hugging Face repo.",
    )
    summarize_video.add_argument(
        "--scene-frame-step",
        type=int,
        default=48,
        help="Analyze every Nth frame for visual scene understanding.",
    )
    summarize_video.add_argument(
        "--scene-max-samples",
        type=int,
        default=20,
        help="Maximum number of sampled frames sent through YOLOv8 Nano.",
    )
    summarize_video.add_argument(
        "--scene-confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLOv8 Nano detections.",
    )
    summarize_video.add_argument(
        "--scene-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used by YOLOv8 Nano scene analysis.",
    )
    summarize_video.add_argument(
        "--person-frame-step",
        type=int,
        default=30,
        help="Analyze every Nth video frame when person identification is enabled.",
    )
    summarize_video.add_argument(
        "--person-min-confidence",
        type=float,
        default=0.0,
        help="Ignore identified people below this confidence score.",
    )
    summarize_video.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit loading for the local lightweight Transformers model.",
    )

    return parser


def build_config(args: argparse.Namespace) -> FaceIdentifierConfig:
    return FaceIdentifierConfig(
        database_path=args.db,
        unknown_dir=args.unknown_dir,
        similarity_threshold=args.threshold,
    )


def build_identifier(args: argparse.Namespace) -> FaceIdentifier:
    return FaceIdentifier(config=build_config(args))


def run_add_person(identifier: FaceIdentifier, args: argparse.Namespace) -> int:
    image = load_image(args.image)
    result = identifier.add_person(args.name, image, source_image=args.image)
    print(json.dumps(result, indent=2))
    return 0


def run_identify_image(identifier: FaceIdentifier, args: argparse.Namespace) -> int:
    image = load_image(args.image)
    results = [item.to_dict() for item in identifier.process_frame(image)]
    print(json.dumps(results, indent=2))
    return 0


def run_batch_identify(identifier: FaceIdentifier, args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    results: dict[str, list[dict[str, object]]] = {}
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for image_path in sorted(input_dir.glob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in allowed_suffixes:
            continue
        results[str(image_path)] = [
            item.to_dict() for item in identifier.process_frame(load_image(image_path))
        ]
    print(json.dumps(results, indent=2))
    return 0


def run_process_video(identifier: FaceIdentifier, args: argparse.Namespace) -> int:
    cv2_module = _load_cv2()
    if cv2_module is None:
        raise RuntimeError("opencv-python is required for video processing.")

    capture = cv2_module.VideoCapture(args.video)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {args.video}")

    frame_index = 0
    output: list[dict[str, object]] = []
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if frame_index % max(args.frame_step, 1) == 0:
            output.append(
                {
                    "frame_index": frame_index,
                    "results": [item.to_dict() for item in identifier.process_frame(frame)],
                }
            )
        frame_index += 1

    capture.release()
    print(json.dumps(output, indent=2))
    return 0


def run_analyze_cctv(args: argparse.Namespace) -> int:
    from video_intelligence_agent.cctv_pipeline import BaseAppError, VideoProcessor, load_pipeline_config
    from video_intelligence_agent.cctv_pipeline.utils.logger import configure_logging

    configure_logging(debug=args.debug)
    pipeline_logger = logging.getLogger("video_intelligence_agent.cctv_pipeline.cli")

    try:
        config = load_pipeline_config(args.config)
        if args.analyze_all_uploads:
            records = discover_video_records(
                config.resolved_video_library_dir(),
                config.resolved_library_output_dir(),
            )
            if not records:
                pipeline_logger.error(
                    "No uploaded videos were found in %s",
                    config.resolved_video_library_dir(),
                )
                return 1

            run_results: list[dict[str, object]] = []
            for record in records:
                run_config = _prepare_cctv_run_config(
                    config,
                    video_path=str(record.video_path),
                    debug=args.debug,
                    output_dir=record.output_dir,
                )
                try:
                    face_identifier = build_identifier(args)
                except Exception as exc:
                    pipeline_logger.warning(
                        "Face recognition failed to initialize. People will be labeled as unknown. reason=%s",
                        exc,
                    )
                    face_identifier = None
                processor = VideoProcessor(config=run_config, face_identifier=face_identifier)
                result = processor.process_video()
                run_results.append(
                    {
                        "video_path": str(record.video_path),
                        "output_dir": str(record.output_dir),
                        "events_path": str(result.artifacts.events_path),
                        "analysis_path": str(result.artifacts.analysis_path) if result.artifacts.analysis_path else None,
                        "events_detected": result.stats.events_detected,
                    }
                )
            print(json.dumps(run_results, indent=2))
        else:
            output_dir = _resolve_output_dir_for_video(config, args.video or config.video_path)
            run_config = _prepare_cctv_run_config(
                config,
                video_path=args.video or None,
                debug=args.debug,
                output_dir=output_dir,
            )

            try:
                face_identifier = build_identifier(args)
            except Exception as exc:
                pipeline_logger.warning(
                    "Face recognition failed to initialize. People will be labeled as unknown. reason=%s",
                    exc,
                )
                face_identifier = None

            processor = VideoProcessor(config=run_config, face_identifier=face_identifier)
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
        pipeline_logger.error(str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - fatal fallback
        pipeline_logger.error("Fatal CCTV pipeline error: %s", exc)
        return 1


def _prepare_cctv_run_config(config, *, video_path: str | None, debug: bool, output_dir: Path | None):
    updated = config
    if video_path:
        updated = replace(updated, video_path=video_path)
    if output_dir is not None:
        updated = replace(updated, storage=replace(updated.storage, output_dir=output_dir))
    if debug:
        updated = replace(
            updated,
            debug=replace(updated.debug, enabled=True, save_frames=True, draw_boxes=True),
        )
    return updated


def _resolve_output_dir_for_video(config, video_path: str) -> Path | None:
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


def run_summarize_video(args: argparse.Namespace) -> int:
    from video_intelligence_agent.video_summarizer import (
        VideoIntelligencePersonIdentifier,
        VideoSummarizer,
    )
    from video_intelligence_agent.video_scene_analyzer import VideoSceneAnalyzer

    person_module = None
    dataset_path = None
    if not args.no_person_id:
        config = build_config(args)
        person_module = VideoIntelligencePersonIdentifier(
            config=config,
            frame_step=args.person_frame_step,
            min_confidence=args.person_min_confidence,
        )
        dataset_path = str(config.resolved_database_path())

    scene_module = None
    if args.scene_analysis:
        scene_module = VideoSceneAnalyzer(
            model_repo=args.scene_model_repo,
            model_filename=args.scene_model_file,
            frame_step=args.scene_frame_step,
            max_samples=args.scene_max_samples,
            confidence_threshold=args.scene_confidence,
            device=args.scene_device,
        )

    summarizer = VideoSummarizer(
        llm_backend=args.llm_backend,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_model_path=args.llm_model_path,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        transcription_language=args.language or None,
        person_id_module=person_module,
        scene_analysis_module=scene_module,
        dataset_path=dataset_path,
        module_context=args.module_context,
        unknown_person_label="Unknown Person",
        local_llm_use_4bit=not args.disable_4bit,
        log_level=logging.INFO if args.verbose else logging.WARNING,
    )
    summary = summarizer.summarize_video(
        args.video,
        keep_temp_audio=args.keep_temp_audio,
        max_summary_tokens=args.summary_tokens,
        temperature=args.temperature,
    )
    print(summary)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.command == "add-person":
        return run_add_person(build_identifier(args), args)
    if args.command == "identify-image":
        return run_identify_image(build_identifier(args), args)
    if args.command == "batch-identify":
        return run_batch_identify(build_identifier(args), args)
    if args.command == "process-video":
        return run_process_video(build_identifier(args), args)
    if args.command == "analyze-cctv":
        return run_analyze_cctv(args)
    if args.command == "summarize-video":
        return run_summarize_video(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

