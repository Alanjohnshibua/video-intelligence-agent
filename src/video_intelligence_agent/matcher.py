from __future__ import annotations

import numpy as np

from video_intelligence_agent.models import StoredEmbedding
from video_intelligence_agent.types import EmbeddingVector


def cosine_similarity(left: EmbeddingVector, right: EmbeddingVector) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


class FaceMatcher:
    def __init__(self, threshold: float = 0.4) -> None:
        self.threshold = threshold

    def match(
        self,
        embedding: EmbeddingVector,
        records: list[StoredEmbedding],
    ) -> tuple[StoredEmbedding | None, float]:
        best_record: StoredEmbedding | None = None
        best_score = -1.0

        for record in records:
            score = cosine_similarity(embedding, record.embedding)
            if score > best_score:
                best_record = record
                best_score = score

        if best_record is None or best_score < self.threshold:
            return None, max(best_score, 0.0)

        return best_record, best_score

