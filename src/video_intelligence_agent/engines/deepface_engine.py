from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Protocol, Sequence, TypedDict, cast

import numpy as np

from video_intelligence_agent.config import FaceIdentifierConfig
from video_intelligence_agent.engines.base import FaceEngine
from video_intelligence_agent.image_io import GenericImageArray, crop_image
from video_intelligence_agent.models import BoundingBox, DetectedFace
from video_intelligence_agent.types import EmbeddingVector


class FacialAreaPayload(TypedDict, total=False):
    x: int | float
    y: int | float
    w: int | float
    h: int | float


class ExtractedFacePayload(TypedDict, total=False):
    facial_area: FacialAreaPayload
    confidence: float


class RepresentedFacePayload(TypedDict, total=False):
    embedding: Sequence[float]


class DeepFaceAPI(Protocol):
    @staticmethod
    def extract_faces(
        *,
        img_path: GenericImageArray,
        detector_backend: str,
        enforce_detection: bool,
        align: bool,
    ) -> list[ExtractedFacePayload]: ...

    @staticmethod
    def represent(
        *,
        img_path: GenericImageArray,
        model_name: str,
        detector_backend: str,
        enforce_detection: bool,
        align: bool,
        normalization: str,
    ) -> list[RepresentedFacePayload] | RepresentedFacePayload: ...


class DeepFaceLoadError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_deepface() -> DeepFaceAPI | None:
    try:
        module = import_module("deepface")
    except ImportError:  # pragma: no cover - depends on local environment
        return None

    deepface_api = getattr(module, "DeepFace", None)
    if deepface_api is not None:
        return cast(DeepFaceAPI, deepface_api)

    try:
        submodule = import_module("deepface.DeepFace")
    except ImportError:  # pragma: no cover - depends on local environment
        return None
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise DeepFaceLoadError(
            f"DeepFace is installed but failed to initialize: {type(exc).__name__}: {exc}"
        ) from exc

    deepface_api = getattr(submodule, "DeepFace", None)
    if deepface_api is not None:
        return cast(DeepFaceAPI, deepface_api)

    if callable(getattr(submodule, "extract_faces", None)) and callable(
        getattr(submodule, "represent", None)
    ):
        return cast(DeepFaceAPI, submodule)

    return None


class DeepFaceEngine(FaceEngine):
    def __init__(self, config: FaceIdentifierConfig) -> None:
        self.config = config

    def detect_faces(self, image: GenericImageArray) -> list[DetectedFace]:
        try:
            deepface_api = _load_deepface()
        except DeepFaceLoadError as exc:
            raise RuntimeError(str(exc)) from exc
        if deepface_api is None:
            raise RuntimeError("deepface is required for RetinaFace-based face detection.")

        try:
            extracted = deepface_api.extract_faces(
                img_path=image,
                detector_backend=self.config.detector_backend,
                enforce_detection=self.config.enforce_detection,
                align=self.config.align,
            )
        except OSError as exc:
            raise RuntimeError(self._explain_weight_file_error(exc, "retinaface.h5")) from exc
        except ValueError as exc:
            raise RuntimeError(self._explain_detector_backend_error(exc)) from exc

        detections: list[DetectedFace] = []
        for item in extracted:
            area = item.get("facial_area", {})
            bbox = BoundingBox(
                x=int(area.get("x", 0)),
                y=int(area.get("y", 0)),
                w=int(area.get("w", 0)),
                h=int(area.get("h", 0)),
            )
            crop = crop_image(image, bbox)
            if crop.size == 0:
                continue
            detections.append(
                DetectedFace(
                    bbox=bbox,
                    crop=crop,
                    detection_confidence=float(item.get("confidence", 0.0)),
                )
            )
        return detections

    def get_embedding(self, face_image: GenericImageArray) -> EmbeddingVector:
        try:
            deepface_api = _load_deepface()
        except DeepFaceLoadError as exc:
            raise RuntimeError(str(exc)) from exc
        if deepface_api is None:
            raise RuntimeError("deepface is required for ArcFace embeddings.")

        try:
            represented = deepface_api.represent(
                img_path=face_image,
                model_name=self.config.embedding_model,
                detector_backend="skip",
                enforce_detection=False,
                align=False,
                normalization=self.config.normalization,
            )
        except OSError as exc:
            raise RuntimeError(
                self._explain_weight_file_error(exc, "the configured DeepFace model weights")
            ) from exc
        payload = self._normalize_representation_payload(represented)
        embedding = payload.get("embedding")
        if embedding is None:
            raise RuntimeError("DeepFace did not return an embedding for the provided face image.")
        return np.asarray(embedding, dtype=np.float32)

    @staticmethod
    def _normalize_representation_payload(
        represented: list[RepresentedFacePayload] | RepresentedFacePayload,
    ) -> RepresentedFacePayload:
        if isinstance(represented, list):
            if not represented:
                raise RuntimeError("DeepFace returned an empty embedding payload.")
            return represented[0]
        return represented

    @staticmethod
    def _explain_weight_file_error(exc: OSError, model_file_hint: str) -> str:
        message = str(exc)
        if "truncated file" in message.lower():
            weights_dir = Path.home() / ".deepface" / "weights"
            return (
                f"DeepFace model weights appear to be corrupted ({model_file_hint}). "
                f"Delete the cached file in {weights_dir} and rerun so it can be re-downloaded. "
                f"Original error: {message}"
            )
        return message

    @staticmethod
    def _explain_detector_backend_error(exc: ValueError) -> str:
        message = str(exc)
        if "KerasTensor cannot be used as input to a TensorFlow function" in message:
            return (
                "The selected DeepFace detector backend is not compatible with your current "
                "TensorFlow/Keras runtime on native Windows. Switch config.yaml "
                "detector_backend to 'opencv' or 'mtcnn' for local Windows runs, or use "
                "WSL2/Linux if you want RetinaFace with a modern TensorFlow stack. "
                f"Original error: {message}"
            )
        return message

