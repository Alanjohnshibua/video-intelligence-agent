from __future__ import annotations

from video_intelligence_agent.config import FaceIdentifierConfig
from video_intelligence_agent.core import FaceIdentifier
from video_intelligence_agent.engines.base import FaceEngine
from video_intelligence_agent.image_io import GenericImageArray
from video_intelligence_agent.matcher import FaceMatcher
from video_intelligence_agent.models import DetectedFace, MatchResult
from video_intelligence_agent.storage import EmbeddingStore
from video_intelligence_agent.types import EmbeddingVector


def _build_identifier(
    config: FaceIdentifierConfig | None = None,
    engine: FaceEngine | None = None,
    store: EmbeddingStore | None = None,
) -> FaceIdentifier:
    return FaceIdentifier(
        config=config,
        engine=engine,
        store=store,
        matcher=FaceMatcher((config or FaceIdentifierConfig()).similarity_threshold),
    )


def detect_faces(
    image: GenericImageArray,
    *,
    config: FaceIdentifierConfig | None = None,
    engine: FaceEngine | None = None,
) -> list[DetectedFace]:
    return _build_identifier(config=config, engine=engine).detect_faces(image)


def get_embedding(
    image: GenericImageArray,
    *,
    config: FaceIdentifierConfig | None = None,
    engine: FaceEngine | None = None,
) -> EmbeddingVector:
    return _build_identifier(config=config, engine=engine).get_embedding(image)


def match_face(
    embedding: EmbeddingVector,
    *,
    config: FaceIdentifierConfig | None = None,
    store: EmbeddingStore | None = None,
) -> MatchResult:
    return _build_identifier(config=config, store=store).match_face(embedding)


def process_frame(
    frame: GenericImageArray,
    *,
    identifier: FaceIdentifier | None = None,
    config: FaceIdentifierConfig | None = None,
) -> list[MatchResult]:
    subject = identifier or FaceIdentifier(config=config)
    return subject.process_frame(frame)

