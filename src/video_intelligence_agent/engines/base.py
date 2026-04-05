from __future__ import annotations

from abc import ABC, abstractmethod

from video_intelligence_agent.image_io import GenericImageArray
from video_intelligence_agent.models import DetectedFace
from video_intelligence_agent.types import EmbeddingVector


class FaceEngine(ABC):
    @abstractmethod
    def detect_faces(self, image: GenericImageArray) -> list[DetectedFace]:
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, face_image: GenericImageArray) -> EmbeddingVector:
        raise NotImplementedError

