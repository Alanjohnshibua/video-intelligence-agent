from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from video_intelligence_agent.config import FaceIdentifierConfig
from video_intelligence_agent.core import FaceIdentifier
from video_intelligence_agent.engines.base import FaceEngine
from video_intelligence_agent.image_io import GenericImageArray
from video_intelligence_agent.matcher import FaceMatcher, cosine_similarity
from video_intelligence_agent.models import BoundingBox, DetectedFace
from video_intelligence_agent.storage import EmbeddingStore
from video_intelligence_agent.types import EmbeddingVector
from video_intelligence_agent.unknowns import UnknownFaceLogger


class StubEngine(FaceEngine):
    def __init__(self, detection_embeddings: list[EmbeddingVector]) -> None:
        self.detection_embeddings: list[EmbeddingVector] = [
            np.asarray(item, dtype=np.float32) for item in detection_embeddings
        ]
        self._next_embedding = 0

    def detect_faces(self, image: GenericImageArray) -> list[DetectedFace]:
        detections: list[DetectedFace] = []
        for index, _embedding in enumerate(self.detection_embeddings):
            detections.append(
                DetectedFace(
                    bbox=BoundingBox(x=index * 10, y=0, w=8, h=8),
                    crop=image.copy(),
                    detection_confidence=0.99,
                )
            )
        return detections

    def get_embedding(self, face_image: GenericImageArray) -> EmbeddingVector:
        embedding = self.detection_embeddings[self._next_embedding]
        self._next_embedding += 1
        return embedding


def fake_image_writer(path: str | Path, image: GenericImageArray) -> None:
    Path(path).write_bytes(b"image")


def test_add_person_and_process_known_face(tmp_path: Path) -> None:
    config = FaceIdentifierConfig(
        database_path=tmp_path / "embeddings.pkl",
        unknown_dir=tmp_path / "unknown",
        similarity_threshold=0.4,
    )
    image = np.ones((10, 10, 3), dtype=np.uint8)
    enrollment_engine = StubEngine([np.array([1.0, 0.0], dtype=np.float32)])
    identifier = FaceIdentifier(config=config, engine=enrollment_engine)

    enrolled = identifier.add_person("Alan", image, source_image="alan.jpg")
    assert enrolled["name"] == "Alan"

    inference_engine = StubEngine([np.array([1.0, 0.0], dtype=np.float32)])
    inference_identifier = FaceIdentifier(
        config=config,
        engine=inference_engine,
        store=EmbeddingStore(config.resolved_database_path()),
    )

    results = inference_identifier.process_frame(image)
    assert len(results) == 1
    assert results[0].name == "Alan"
    assert results[0].confidence == 1.0


def test_process_unknown_face_logs_image(tmp_path: Path) -> None:
    config = FaceIdentifierConfig(
        database_path=tmp_path / "embeddings.pkl",
        unknown_dir=tmp_path / "unknown",
        similarity_threshold=0.8,
    )
    store = EmbeddingStore(config.resolved_database_path())
    store.add_embedding("Alan", np.array([1.0, 0.0], dtype=np.float32))

    identifier = FaceIdentifier(
        config=config,
        engine=StubEngine([np.array([0.0, 1.0], dtype=np.float32)]),
        store=store,
        unknown_logger=UnknownFaceLogger(config.resolved_unknown_dir(), image_writer=fake_image_writer),
    )

    results = identifier.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert len(results) == 1
    assert results[0].name == "Unknown"
    assert results[0].saved_path is not None
    assert Path(results[0].saved_path).exists()

    log_lines = (config.resolved_unknown_dir() / "unknown_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    payload = json.loads(log_lines[0])
    assert payload["saved_path"] == results[0].saved_path


def test_cosine_similarity_and_threshold(tmp_path: Path) -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0

    matcher = FaceMatcher(threshold=0.5)
    store = EmbeddingStore(tmp_path / "embeddings.pkl")
    store.add_embedding("Alan", np.array([1.0, 0.0], dtype=np.float32))
    matched, score = matcher.match(np.array([0.9, 0.1], dtype=np.float32), store.records)

    assert matched is not None
    assert matched.name == "Alan"
    assert score > 0.5

