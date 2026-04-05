from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from video_intelligence_agent.config import FaceIdentifierConfig
from video_intelligence_agent.core import FaceIdentifier
from video_intelligence_agent.hardware import detect_hardware
from video_intelligence_agent.video_scene_analyzer import SceneAnalysisProtocol

logger = logging.getLogger(__name__)

DEFAULT_LIGHTWEIGHT_MODEL_PATH = (
    r"C:\Users\acer\.cache\huggingface\hub\models--Qwen--Qwen3.5-2B-Base"
    r"\snapshots\982d64a16433b4ece3b33ee53b24a0b416bd979a"
)


class VideoSummarizerError(Exception):
    """Custom exception that keeps track of the failed pipeline stage."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "unknown",
        original_error: Exception | None = None,
    ) -> None:
        self.stage = stage
        self.original_error = original_error
        super().__init__(message)

    def __str__(self) -> str:
        message = f"[Video Summarizer] {super().__str__()}"
        if logger.level <= logging.DEBUG and self.original_error is not None:
            message += (
                f"\n[Stage: {self.stage}] Caused by: "
                f"{type(self.original_error).__name__}: {self.original_error}"
            )
        return message


class PersonIdentifierProtocol(Protocol):
    def identify(
        self,
        video_path: str,
        dataset_path: str | None = None,
    ) -> list[dict[str, object]]: ...


class VideoCaptureProtocol(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Any | None]: ...

    def get(self, prop_id: int) -> float: ...

    def release(self) -> None: ...


class OpenCVVideoModuleProtocol(Protocol):
    CAP_PROP_FPS: int
    CAP_PROP_POS_MSEC: int

    def VideoCapture(self, filename: str) -> VideoCaptureProtocol: ...


class OpenAIClientProtocol(Protocol):
    chat: Any


class OpenAIClientFactoryProtocol(Protocol):
    def __call__(self, *, api_key: str, base_url: str) -> OpenAIClientProtocol: ...


class WhisperModelFactoryProtocol(Protocol):
    def __call__(
        self,
        model_size_or_path: str,
        *,
        device: str,
        compute_type: str,
    ) -> Any: ...


def _load_cv2() -> OpenCVVideoModuleProtocol | None:
    try:
        module = import_module("cv2")
    except ImportError:  # pragma: no cover - depends on local environment
        return None
    return cast(OpenCVVideoModuleProtocol, module)


def _build_openai_client(base_url: str, api_key: str | None) -> Any:
    try:
        module = import_module("openai")
        openai_client_factory = cast(
            OpenAIClientFactoryProtocol,
            getattr(module, "OpenAI"),
        )
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise VideoSummarizerError(
            "The 'openai' package is required for LLM summarization.",
            stage="client_initialization",
            original_error=exc,
        ) from exc

    return openai_client_factory(
        api_key=api_key or "dummy",
        base_url=base_url,
    )


class VideoIntelligencePersonIdentifier:
    """
    Adapter that samples a video and reuses Video Intelligence Agent (VIA) to identify people.

    This gives VideoSummarizer a package-native person identification module
    without forcing callers to build their own adapter.
    """

    def __init__(
        self,
        *,
        identifier: FaceIdentifier | None = None,
        config: FaceIdentifierConfig | None = None,
        frame_step: int = 30,
        min_confidence: float = 0.0,
        max_results_per_person: int = 3,
        unknown_person_label: str = "Unknown Person",
    ) -> None:
        if identifier is None and config is None:
            config = FaceIdentifierConfig()

        self.identifier = identifier
        self.config = config
        self.frame_step = max(frame_step, 1)
        self.min_confidence = min_confidence
        self.max_results_per_person = max_results_per_person
        self.unknown_person_label = unknown_person_label

    def _resolve_identifier(self, dataset_path: str | None) -> FaceIdentifier:
        if dataset_path is None:
            if self.identifier is not None:
                return self.identifier
            return FaceIdentifier(config=self.config)

        requested_path = Path(dataset_path)

        if self.identifier is not None and self.config is not None:
            current_path = Path(self.config.database_path)
            if current_path == requested_path:
                return self.identifier

        if self.config is None:
            if self.identifier is not None:
                return self.identifier
            return FaceIdentifier(config=FaceIdentifierConfig(database_path=requested_path))

        updated_config = replace(self.config, database_path=requested_path)
        return FaceIdentifier(config=updated_config)

    def identify(
        self,
        video_path: str,
        dataset_path: str | None = None,
    ) -> list[dict[str, object]]:
        cv2_module = _load_cv2()
        if cv2_module is None:
            raise VideoSummarizerError(
                "opencv-python is required for video-based person identification.",
                stage="person_identification",
            )

        capture = cv2_module.VideoCapture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        identifier = self._resolve_identifier(dataset_path)
        fps = float(capture.get(cv2_module.CAP_PROP_FPS) or 0.0)

        results: list[dict[str, object]] = []
        seen_counts: dict[str, int] = {}
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                if frame_index % self.frame_step == 0:
                    timestamp = self._timestamp_seconds(
                        frame_index=frame_index,
                        fps=fps,
                        capture=capture,
                        position_prop=cv2_module.CAP_PROP_POS_MSEC,
                    )
                    for match in identifier.process_frame(frame):
                        if match.confidence < self.min_confidence:
                            continue

                        name = match.name
                        if name == "Unknown":
                            name = self.unknown_person_label

                        if (
                            self.max_results_per_person > 0
                            and seen_counts.get(name, 0) >= self.max_results_per_person
                        ):
                            continue

                        results.append(
                            {
                                "name": name,
                                "timestamp": round(timestamp, 2),
                                "confidence": round(float(match.confidence), 4),
                                "frame_index": frame_index,
                            }
                        )
                        seen_counts[name] = seen_counts.get(name, 0) + 1

                frame_index += 1
        finally:
            capture.release()

        return results

    @staticmethod
    def _timestamp_seconds(
        *,
        frame_index: int,
        fps: float,
        capture: VideoCaptureProtocol,
        position_prop: int,
    ) -> float:
        if fps > 0:
            return frame_index / fps

        position_ms = float(capture.get(position_prop) or 0.0)
        if position_ms > 0:
            return position_ms / 1000.0

        return 0.0


# Backward-compatible alias for older imports.
FaceIDLitePersonIdentifier = VideoIntelligencePersonIdentifier


class VideoSummarizer:
    """
    Lightweight pipeline for generating a text summary from a video.

    Steps:
    1. extract audio with FFmpeg
    2. transcribe speech with Faster-Whisper
    3. optionally identify people in the video
    4. optionally analyze sampled frames with YOLOv8 Nano
    5. send the transcript and visual/person context to an LLM
    """

    def __init__(
        self,
        *,
        llm_mode: str | None = None,
        llm_backend: str = "transformers",
        llm_base_url: str = "http://localhost:11434/v1",
        llm_api_key: str | None = None,
        llm_model_name: str | None = None,
        llm_model: str = "qwen2.5:2b",
        model_path: str | None = None,
        llm_model_path: str | None = DEFAULT_LIGHTWEIGHT_MODEL_PATH,
        whisper_model: str = "base",
        whisper_device: str = "auto",
        transcription_language: str | None = "en",
        person_id_module: PersonIdentifierProtocol | None = None,
        scene_analysis_module: SceneAnalysisProtocol | None = None,
        dataset_path: str | None = None,
        unknown_person_label: str = "Unknown Person",
        module_context: str | None = None,
        local_llm_use_4bit: bool = True,
        log_level: int = logging.INFO,
        llm_client: Any | None = None,
        whisper_instance: Any | None = None,
    ) -> None:
        self.llm_backend = self._normalize_llm_backend(llm_backend, llm_mode)
        self.llm_model = llm_model_name or llm_model
        self.llm_model_path = model_path or llm_model_path
        self.whisper_model = whisper_model
        self.whisper_device = self._resolve_whisper_device(whisper_device)
        self.transcription_language = transcription_language
        self.person_id_module = person_id_module
        self.scene_analysis_module = scene_analysis_module
        self.dataset_path = dataset_path
        self.unknown_person_label = unknown_person_label
        self.module_context = module_context or ""
        self.local_llm_use_4bit = local_llm_use_4bit

        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

        self.llm_client = llm_client
        self._local_llm_model: Any | None = None
        self._local_llm_tokenizer: Any | None = None

        if self.llm_client is None:
            if self.llm_backend == "openai":
                self.llm_client = _build_openai_client(llm_base_url, llm_api_key)
            elif self.llm_backend == "transformers":
                self._load_local_llm()
            else:
                raise VideoSummarizerError(
                    "Unsupported llm_backend. Use 'transformers' or 'openai'.",
                    stage="client_initialization",
                )
        self.whisper = whisper_instance or self._load_whisper()

    def _load_whisper(self) -> Any:
        try:
            module = import_module("faster_whisper")
            whisper_model_factory = cast(
                WhisperModelFactoryProtocol,
                getattr(module, "WhisperModel"),
            )
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise VideoSummarizerError(
                "The 'faster-whisper' package is required for transcription.",
                stage="model_loading",
                original_error=exc,
            ) from exc

        try:
            self.logger.info("Loading Faster-Whisper model: %s", self.whisper_model)
            whisper = whisper_model_factory(
                self.whisper_model,
                device=self.whisper_device,
                compute_type="int8" if self.whisper_device == "cpu" else "float16",
            )
            self.logger.info("Whisper model loaded successfully.")
            return whisper
        except Exception as exc:
            raise VideoSummarizerError(
                "Failed to initialize the speech-to-text model.",
                stage="model_loading",
                original_error=exc,
            ) from exc

    def _extract_audio(self, video_path: str, output_path: str) -> None:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ]
        try:
            self.logger.info("Extracting audio with FFmpeg...")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Extracted audio file is missing or empty.")
            self.logger.info("Audio extraction successful.")
        except FileNotFoundError as exc:
            raise VideoSummarizerError(
                "FFmpeg was not found. Make sure it is installed and available in PATH.",
                stage="audio_extraction",
                original_error=exc,
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise VideoSummarizerError(
                "FFmpeg failed during audio extraction.",
                stage="audio_extraction",
                original_error=exc,
            ) from exc
        except Exception as exc:
            raise VideoSummarizerError(
                "Unexpected error during audio extraction.",
                stage="audio_extraction",
                original_error=exc,
            ) from exc

    def _transcribe_audio(self, audio_path: str) -> str:
        try:
            self.logger.info("Transcribing audio...")
            options: dict[str, object] = {"beam_size": 5}
            if self.transcription_language is not None:
                options["language"] = self.transcription_language

            segments, _info = self.whisper.transcribe(audio_path, **options)
            transcription = " ".join(
                segment.text.strip()
                for segment in segments
                if getattr(segment, "text", "").strip()
            ).strip()
            if not transcription:
                self.logger.info(
                    "No spoken dialogue detected. Continuing with visual context only."
                )
                return "No spoken dialogue was detected in the audio track."
            self.logger.info("Transcription complete (%d characters).", len(transcription))
            return transcription
        except Exception as exc:
            raise VideoSummarizerError(
                "Failed to transcribe audio.",
                stage="transcription",
                original_error=exc,
            ) from exc

    def _identify_persons(self, video_path: str) -> str:
        if self.person_id_module is None:
            self.logger.debug("Person identification skipped: no module configured.")
            return "Person identification data not available."

        try:
            self.logger.info("Running person identification...")
            results = self.person_id_module.identify(video_path, self.dataset_path)
            if not results:
                return "No persons identified in the video."

            formatted_lines: list[str] = []
            for entry in results:
                name = str(entry.get("name") or self.unknown_person_label)
                if name == "Unknown":
                    name = self.unknown_person_label
                timestamp = self._coerce_float(entry.get("timestamp"), default=0.0)
                confidence = self._coerce_float(entry.get("confidence"), default=0.0)
                formatted_lines.append(
                    f"- {name} at {timestamp:.1f}s (confidence: {confidence:.2f})"
                )

            self.logger.info(
                "Person identification complete (%d matches).", len(formatted_lines)
            )
            return "\n".join(formatted_lines)
        except AttributeError as exc:
            raise VideoSummarizerError(
                "Person ID module is missing the expected 'identify' method.",
                stage="person_identification",
                original_error=exc,
            ) from exc
        except Exception as exc:
            raise VideoSummarizerError(
                "Failed during person identification.",
                stage="person_identification",
                original_error=exc,
            ) from exc

    def _describe_visual_activity(self, video_path: str) -> str:
        if self.scene_analysis_module is None:
            self.logger.debug("Visual scene analysis skipped: no module configured.")
            return "Visual scene analysis not available."

        try:
            self.logger.info("Running visual scene analysis...")
            description = self.scene_analysis_module.describe(video_path).strip()
            if not description:
                return "No strong visual scene cues were detected in the sampled frames."
            self.logger.info("Visual scene analysis complete.")
            return description
        except AttributeError as exc:
            raise VideoSummarizerError(
                "Scene-analysis module is missing the expected 'describe' method.",
                stage="scene_analysis",
                original_error=exc,
            ) from exc
        except Exception as exc:
            raise VideoSummarizerError(
                "Failed during visual scene analysis.",
                stage="scene_analysis",
                original_error=exc,
            ) from exc

    def _load_local_llm(self) -> None:
        model_path = self.llm_model_path or DEFAULT_LIGHTWEIGHT_MODEL_PATH
        if not os.path.isdir(model_path):
            raise VideoSummarizerError(
                f"Local lightweight model path not found: {model_path}",
                stage="client_initialization",
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise VideoSummarizerError(
                "The 'transformers' package is required for the local lightweight model backend.",
                stage="client_initialization",
                original_error=exc,
            ) from exc

        self.logger.info("Loading local lightweight LLM from: %s", model_path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left",
                use_fast=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs: dict[str, object] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if device == "cuda":
                model_kwargs["device_map"] = "auto"

            if device == "cuda" and self.local_llm_use_4bit:
                try:
                    transformers_module = import_module("transformers")
                    bits_and_bytes_config = getattr(transformers_module, "BitsAndBytesConfig")
                except (ImportError, AttributeError):
                    BitsAndBytesConfig = None
                else:
                    BitsAndBytesConfig = bits_and_bytes_config

                if BitsAndBytesConfig is not None:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    model_kwargs["dtype"] = torch.float16
            else:
                model_kwargs["dtype"] = torch.float16 if device == "cuda" else torch.float32

            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            model.eval()

            self._local_llm_tokenizer = tokenizer
            self._local_llm_model = model
            self.logger.info("Local lightweight LLM loaded successfully.")
        except Exception as exc:
            raise VideoSummarizerError(
                "Failed to initialize the local lightweight model.",
                stage="client_initialization",
                original_error=exc,
            ) from exc

    def _build_prompt(
        self,
        transcription: str,
        person_data: str,
        visual_data: str,
        max_tokens: int,
    ) -> str:
        sections = [
            (
                f"You are an expert video summarizer. Write a clear, structured summary "
                f"in under {max_tokens} tokens."
            ),
            "Use the transcript and visual scene analysis together as the main source of truth.",
            (
                "Use the person-identification results to mention who appears and when, "
                "but do not invent names, timestamps, or events."
            ),
            (
                "Use the visual scene analysis to explain what is happening in the video, "
                "but keep claims coarse and grounded in visible evidence."
            ),
            (
                f"If a person is not recognized, refer to them as "
                f"'{self.unknown_person_label}'."
            ),
        ]

        if self.module_context.strip():
            sections.append(
                "Only use the module context below when it is relevant, and do not invent "
                "APIs or function signatures."
            )

        prompt_parts = [
            "\n".join(sections),
            "PERSON IDENTIFICATION RESULTS\n" + person_data,
            "VISUAL SCENE ANALYSIS\n" + visual_data,
        ]

        if self.module_context.strip():
            prompt_parts.append("CUSTOM MODULE CONTEXT\n" + self.module_context.strip())

        prompt_parts.append("AUDIO TRANSCRIPT\n" + transcription)
        prompt_parts.append(
            "OUTPUT FORMAT\n"
            "1. Short Summary\n"
            "2. Key Moments\n"
            "3. People Mentioned"
        )
        return "\n\n".join(prompt_parts)

    def _prepare_local_prompt(self, prompt: str) -> str:
        tokenizer = self._local_llm_tokenizer
        if tokenizer is None:
            return prompt

        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                return str(
                    apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception as exc:
                self.logger.debug(
                    "Falling back to raw prompt after chat-template failure: %s",
                    exc,
                )

        return prompt

    def _summarize_text(
        self,
        text: str,
        person_data: str,
        visual_data: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        if self.llm_client is not None:
            return self._summarize_with_client(
                text,
                person_data,
                visual_data,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        if self.llm_backend == "transformers":
            return self._summarize_with_local_model(
                text,
                person_data,
                visual_data,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        raise VideoSummarizerError(
            "No LLM backend is available for summarization.",
            stage="llm_summarization",
        )

    def _summarize_with_client(
        self,
        text: str,
        person_data: str,
        visual_data: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        prompt = self._build_prompt(text, person_data, visual_data, max_tokens)
        client = self.llm_client
        if client is None:
            raise VideoSummarizerError(
                "No OpenAI-compatible client is available for summarization.",
                stage="llm_summarization",
            )
        try:
            self.logger.info("Generating summary with LLM: %s", self.llm_model)
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            summary = response.choices[0].message.content
            if not isinstance(summary, str) or not summary.strip():
                raise RuntimeError("LLM returned an empty response.")
            self.logger.info("Summary generation successful.")
            return summary.strip()
        except Exception as exc:
            raise VideoSummarizerError(
                "Failed to generate summary with the LLM.",
                stage="llm_summarization",
                original_error=exc,
            ) from exc

    def _summarize_with_local_model(
        self,
        text: str,
        person_data: str,
        visual_data: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        if self._local_llm_model is None or self._local_llm_tokenizer is None:
            self._load_local_llm()

        prompt = self._prepare_local_prompt(
            self._build_prompt(text, person_data, visual_data, max_tokens)
        )

        try:
            import torch
            transformers_module = import_module("transformers")
            generation_config_type = getattr(transformers_module, "GenerationConfig")
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise VideoSummarizerError(
                "The local lightweight model backend requires 'torch' and 'transformers'.",
                stage="llm_summarization",
                original_error=exc,
            ) from exc

        try:
            self.logger.info(
                "Generating summary with local lightweight model: %s",
                self.llm_model_path or DEFAULT_LIGHTWEIGHT_MODEL_PATH,
            )
            tokenizer = self._local_llm_tokenizer
            model = self._local_llm_model
            if tokenizer is None or model is None:
                raise VideoSummarizerError(
                    "Local lightweight model artifacts are not initialized.",
                    stage="llm_summarization",
                )
            inputs = tokenizer(prompt, return_tensors="pt")
            device = self._resolve_model_device(model)
            inputs = {name: value.to(device) for name, value in inputs.items()}

            do_sample = temperature > 0
            generation_config = generation_config_type(
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=0.9,
                do_sample=do_sample,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            with torch.inference_mode():
                outputs = model.generate(**inputs, generation_config=generation_config)

            prompt_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][prompt_length:]
            summary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            if not summary:
                raise RuntimeError("Local model returned an empty response.")
            self.logger.info("Summary generation successful.")
            return summary
        except Exception as exc:
            raise VideoSummarizerError(
                "Failed to generate summary with the local lightweight model.",
                stage="llm_summarization",
                original_error=exc,
            ) from exc

    def summarize_video(
        self,
        video_path: str,
        *,
        temp_audio_path: str | None = None,
        keep_temp_audio: bool = False,
        max_summary_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        self.logger.info("Starting video summarization pipeline for: %s", video_path)

        created_temp_file = False
        if temp_audio_path is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_path = temp_file.name
            created_temp_file = True
        else:
            audio_path = temp_audio_path

        try:
            self._extract_audio(video_path, audio_path)
            transcription = self._transcribe_audio(audio_path)
            person_data = self._identify_persons(video_path)
            visual_data = self._describe_visual_activity(video_path)
            return self._summarize_text(
                transcription,
                person_data,
                visual_data,
                max_tokens=max_summary_tokens,
                temperature=temperature,
            )
        except VideoSummarizerError:
            raise
        except Exception as exc:
            raise VideoSummarizerError(
                "An unexpected error occurred during video processing.",
                stage="pipeline",
                original_error=exc,
            ) from exc
        finally:
            should_delete = not keep_temp_audio and (created_temp_file or temp_audio_path is not None)
            if should_delete and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    self.logger.debug("Removed temporary audio file.")
                except OSError as exc:
                    self.logger.warning(
                        "Failed to remove temporary audio file: %s",
                        exc,
                    )

    @staticmethod
    def _coerce_float(value: object, *, default: float) -> float:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float, str)):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def _resolve_model_device(model: Any) -> Any:
        try:
            return next(model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            try:
                return model.device
            except AttributeError:
                return "cpu"

    @staticmethod
    def _normalize_llm_backend(llm_backend: str, llm_mode: str | None) -> str:
        selected = (llm_mode or llm_backend).strip().lower()
        aliases = {
            "transformers": "transformers",
            "local": "transformers",
            "openai": "openai",
            "api": "openai",
        }
        try:
            return aliases[selected]
        except KeyError as exc:
            raise VideoSummarizerError(
                "Unsupported llm backend/mode. Use transformers/local or openai/api.",
                stage="client_initialization",
                original_error=ValueError(selected),
            ) from exc

    @staticmethod
    def _resolve_whisper_device(requested_device: str) -> str:
        selected = requested_device.strip().lower()
        if selected in {"cpu", "cuda"}:
            return selected
        if selected != "auto":
            raise VideoSummarizerError(
                "Unsupported whisper device. Use auto, cpu, or cuda.",
                stage="client_initialization",
                original_error=ValueError(selected),
            )
        return detect_hardware().preferred_torch_device


__all__ = [
    "DEFAULT_LIGHTWEIGHT_MODEL_PATH",
    "FaceIDLitePersonIdentifier",
    "PersonIdentifierProtocol",
    "VideoIntelligencePersonIdentifier",
    "VideoSummarizer",
    "VideoSummarizerError",
]

