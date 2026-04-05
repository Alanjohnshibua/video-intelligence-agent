from __future__ import annotations

import logging
from pathlib import Path

from video_intelligence_agent.config import FaceIdentifierConfig
from video_intelligence_agent.engines.base import FaceEngine
from video_intelligence_agent.engines.deepface_engine import DeepFaceEngine
from video_intelligence_agent.hardware import configure_tensorflow_runtime
from video_intelligence_agent.image_io import GenericImageArray
from video_intelligence_agent.matcher import FaceMatcher
from video_intelligence_agent.models import DetectedFace, MatchResult
from video_intelligence_agent.storage import EmbeddingStore
from video_intelligence_agent.types import EmbeddingVector
from video_intelligence_agent.unknowns import UnknownFaceLogger

logger = logging.getLogger(__name__)


class FaceIdentifier:
    def __init__(
        self,
        config: FaceIdentifierConfig | None = None,
        engine: FaceEngine | None = None,
        store: EmbeddingStore | None = None,
        matcher: FaceMatcher | None = None,
        unknown_logger: UnknownFaceLogger | None = None,
    ) -> None:
        self.config = config or FaceIdentifierConfig()
        configure_tensorflow_runtime(
            prefer_gpu=self.config.prefer_gpu,
            enable_memory_growth=self.config.tensorflow_memory_growth,
        )
        self.engine = engine or DeepFaceEngine(self.config)
        self.store = store or EmbeddingStore(self.config.resolved_database_path())
        self.matcher = matcher or FaceMatcher(self.config.similarity_threshold)
        self.unknown_logger = unknown_logger or UnknownFaceLogger(
            self.config.resolved_unknown_dir()
        )

    def detect_faces(self, frame: GenericImageArray) -> list[DetectedFace]:
        return self.engine.detect_faces(frame)

    def get_embedding(self, image: GenericImageArray) -> EmbeddingVector:
        return self.engine.get_embedding(image)

    def match_face(self, embedding: EmbeddingVector) -> MatchResult:
        record, confidence = self.matcher.match(embedding, self.store.records)
        if record is None:
            return MatchResult(name="Unknown", confidence=confidence)
        return MatchResult(name=record.name, confidence=confidence, metadata=record.metadata)

    def identify_face(self, image: GenericImageArray) -> MatchResult:
        embedding = self.get_embedding(image)
        return self.match_face(embedding)

    def add_person(
        self,
        name: str,
        image: GenericImageArray,
        source_image: str | Path | None = None,
    ) -> dict[str, object]:
        detections = self.detect_faces(image)
        if not detections:
            raise ValueError("No face detected while adding a person to the database.")

        # For enrollment we choose the largest face, which is usually the subject.
        target = max(detections, key=lambda item: item.bbox.w * item.bbox.h)
        embedding = self.get_embedding(target.crop)
        record = self.store.add_embedding(
            name=name,
            embedding=embedding,
            source_image=str(source_image) if source_image is not None else None,
            metadata={"enrolled_bbox": target.bbox.to_dict()},
        )
        logger.info("Added %s to the embedding store.", name)
        return {
            "name": record.name,
            "source_image": record.source_image,
            "bbox": target.bbox.to_dict(),
        }

    def process_frame(self, frame: GenericImageArray) -> list[MatchResult]:
        detections = self.detect_faces(frame)
        results: list[MatchResult] = []

        for detected in detections:
            embedding = self.get_embedding(detected.crop)
            matched = self.match_face(embedding)
            matched.bbox = detected.bbox

            if matched.name == "Unknown":
                unknown_entry = self.unknown_logger.log(detected.crop)
                matched.saved_path = unknown_entry["saved_path"]
                matched.timestamp = unknown_entry["timestamp"]

            results.append(matched)

        return results

