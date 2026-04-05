from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from video_intelligence_agent.hardware import detect_hardware


class VideoSceneAnalyzerError(Exception):
    """Raised when visual scene analysis cannot be completed."""


class VideoCaptureProtocol(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Any | None]: ...

    def get(self, prop_id: int) -> float: ...

    def release(self) -> None: ...


class OpenCVVideoModuleProtocol(Protocol):
    CAP_PROP_FPS: int
    CAP_PROP_POS_MSEC: int

    def VideoCapture(self, filename: str) -> VideoCaptureProtocol: ...


class YOLOFactoryProtocol(Protocol):
    def __call__(self, model: str | Path) -> Any: ...


class HfHubDownloadProtocol(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        cache_dir: str | None = None,
    ) -> str: ...


@dataclass(slots=True)
class SceneObservation:
    timestamp_seconds: float
    labels: list[str]


class SceneAnalysisProtocol(Protocol):
    def describe(self, video_path: str) -> str: ...


def _load_cv2() -> OpenCVVideoModuleProtocol | None:
    try:
        module = import_module("cv2")
    except ImportError:  # pragma: no cover - depends on local environment
        return None
    return cast(OpenCVVideoModuleProtocol, module)


class VideoSceneAnalyzer:
    """
    Samples frames and uses YOLOv8 Nano to generate coarse visual activity cues.

    The model is downloaded from Hugging Face and executed through Ultralytics.
    """

    def __init__(
        self,
        *,
        model_repo: str = "Ultralytics/YOLOv8",
        model_filename: str = "yolov8n.pt",
        cache_dir: str | None = None,
        frame_step: int = 48,
        max_samples: int = 20,
        confidence_threshold: float = 0.25,
        max_labels_per_frame: int = 4,
        device: str = "auto",
    ) -> None:
        self.model_repo = model_repo
        self.model_filename = model_filename
        self.cache_dir = cache_dir
        self.frame_step = max(frame_step, 1)
        self.max_samples = max(max_samples, 1)
        self.confidence_threshold = confidence_threshold
        self.max_labels_per_frame = max(max_labels_per_frame, 1)
        self.device = device.strip().lower()
        self._model: Any | None = None

    def describe(self, video_path: str) -> str:
        observations = self.analyze(video_path)
        if not observations:
            return "No strong visual scene cues were detected in the sampled frames."
        return self._format_report(observations)

    def analyze(self, video_path: str) -> list[SceneObservation]:
        cv2_module = _load_cv2()
        if cv2_module is None:
            raise VideoSceneAnalyzerError(
                "opencv-python is required for visual scene analysis."
            )

        capture = cv2_module.VideoCapture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        fps = float(capture.get(cv2_module.CAP_PROP_FPS) or 0.0)
        observations: list[SceneObservation] = []
        frame_index = 0
        sampled_count = 0

        try:
            while sampled_count < self.max_samples:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                if frame_index % self.frame_step == 0:
                    labels = self._predict_labels(frame)
                    if labels:
                        timestamp = self._timestamp_seconds(
                            frame_index=frame_index,
                            fps=fps,
                            capture=capture,
                            position_prop=cv2_module.CAP_PROP_POS_MSEC,
                        )
                        observations.append(
                            SceneObservation(
                                timestamp_seconds=round(timestamp, 2),
                                labels=labels,
                            )
                        )
                    sampled_count += 1

                frame_index += 1
        finally:
            capture.release()

        return observations

    def _predict_labels(self, frame: Any) -> list[str]:
        model = self._load_model()
        try:
            results = model.predict(
                source=frame,
                conf=self.confidence_threshold,
                verbose=False,
                device=self._resolve_inference_device(),
            )
        except TypeError:
            results = model.predict(
                source=frame,
                conf=self.confidence_threshold,
                verbose=False,
            )

        label_counter: Counter[str] = Counter()
        for result in results:
            names = self._normalize_names(getattr(result, "names", {}))
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            class_ids = self._to_list(getattr(boxes, "cls", []))
            confidences = self._to_list(getattr(boxes, "conf", []))

            for index, raw_class_id in enumerate(class_ids):
                label = names.get(int(raw_class_id))
                if label is None:
                    continue
                confidence = 1.0
                if index < len(confidences):
                    confidence = float(confidences[index])
                if confidence < self.confidence_threshold:
                    continue
                label_counter[str(label)] += 1

        return [
            label
            for label, _count in label_counter.most_common(self.max_labels_per_frame)
        ]

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            huggingface_hub = import_module("huggingface_hub")
            ultralytics_module = import_module("ultralytics")
            hf_hub_download = cast(
                HfHubDownloadProtocol,
                getattr(huggingface_hub, "hf_hub_download"),
            )
            yolo_factory = cast(YOLOFactoryProtocol, getattr(ultralytics_module, "YOLO"))
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise VideoSceneAnalyzerError(
                "The 'ultralytics' and 'huggingface-hub' packages are required for "
                "YOLOv8 scene analysis."
            ) from exc
        except AttributeError as exc:  # pragma: no cover - defensive
            raise VideoSceneAnalyzerError(
                "Unable to initialize the Ultralytics YOLO loader."
            ) from exc

        model_path = hf_hub_download(
            repo_id=self.model_repo,
            filename=self.model_filename,
            cache_dir=self.cache_dir,
        )
        self._model = yolo_factory(model_path)
        return self._model

    def _format_report(self, observations: list[SceneObservation]) -> str:
        object_counter: Counter[str] = Counter()
        person_frames = 0
        vehicle_frames = 0

        vehicle_labels = {"car", "truck", "bus", "motorcycle", "bicycle"}
        device_labels = {"cell phone", "laptop", "tv", "monitor"}

        for observation in observations:
            labels = observation.labels
            object_counter.update(labels)
            if "person" in labels:
                person_frames += 1
            if any(label in vehicle_labels for label in labels):
                vehicle_frames += 1

        activity_cues: list[str] = []
        if person_frames >= max(2, len(observations) // 2):
            activity_cues.append("sustained human presence across the clip")
        if vehicle_frames >= 2 and person_frames >= 1:
            activity_cues.append("people and vehicles appear together, suggesting movement or arrival")
        elif vehicle_frames >= 2:
            activity_cues.append("vehicle movement or parked-vehicle activity is visible")
        if any(label in object_counter for label in device_labels):
            activity_cues.append("indoor device-oriented activity is visible in parts of the clip")
        if not activity_cues and object_counter:
            dominant_label = object_counter.most_common(1)[0][0]
            activity_cues.append(
                f"the scene is primarily centered around visible '{dominant_label}' activity"
            )

        top_objects = ", ".join(
            f"{label} ({count})" for label, count in object_counter.most_common(5)
        )
        timeline = "\n".join(
            f"- {observation.timestamp_seconds:.1f}s: {', '.join(observation.labels)}"
            for observation in observations[:8]
        )
        cue_lines = "\n".join(f"- {cue}" for cue in activity_cues)

        return (
            "Visual scene analysis generated from sampled frames with YOLOv8 Nano.\n"
            f"Likely activity cues:\n{cue_lines}\n"
            f"Dominant detected objects: {top_objects}\n"
            "Sampled timeline:\n"
            f"{timeline}"
        )

    def _resolve_inference_device(self) -> str | int:
        if self.device == "cpu":
            return "cpu"
        if self.device == "cuda":
            return 0
        if self.device != "auto":
            raise VideoSceneAnalyzerError(
                "Unsupported scene-analysis device. Use auto, cpu, or cuda."
            )
        return 0 if detect_hardware().torch_cuda_available else "cpu"

    @staticmethod
    def _normalize_names(raw_names: object) -> dict[int, str]:
        if isinstance(raw_names, dict):
            return {int(key): str(value) for key, value in raw_names.items()}
        if isinstance(raw_names, list):
            return {index: str(value) for index, value in enumerate(raw_names)}
        return {}

    @staticmethod
    def _to_list(value: object) -> list[float]:
        if value is None:
            return []
        if isinstance(value, list):
            return [float(item) for item in value]
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            raw_items = tolist()
            if isinstance(raw_items, list):
                return [float(item) for item in raw_items]
        return []

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


__all__ = [
    "SceneAnalysisProtocol",
    "SceneObservation",
    "VideoSceneAnalyzer",
    "VideoSceneAnalyzerError",
]
