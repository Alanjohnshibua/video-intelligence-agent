from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, TypedDict, cast

import numpy as np

from video_intelligence_agent.models import StoredEmbedding
from video_intelligence_agent.types import EmbeddingVector


class SerializedEmbeddingRecord(TypedDict):
    name: str
    embedding: list[float]
    source_image: str | None
    metadata: dict[str, Any]


class EmbeddingStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._records: list[StoredEmbedding] = []
        self.load()

    @property
    def records(self) -> list[StoredEmbedding]:
        return list(self._records)

    def load(self) -> None:
        if not self.database_path.exists():
            self._records = []
            return

        with self.database_path.open("rb") as handle:
            payload = cast(list[SerializedEmbeddingRecord], pickle.load(handle))

        records: list[StoredEmbedding] = []
        for item in payload:
            records.append(
                StoredEmbedding(
                    name=item["name"],
                    embedding=np.asarray(item["embedding"], dtype=np.float32),
                    source_image=item.get("source_image"),
                    metadata=item.get("metadata", {}),
                )
            )
        self._records = records

    def save(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        payload: list[SerializedEmbeddingRecord] = [
            {
                "name": record.name,
                "embedding": record.embedding.astype(np.float32).tolist(),
                "source_image": record.source_image,
                "metadata": record.metadata,
            }
            for record in self._records
        ]
        with self.database_path.open("wb") as handle:
            pickle.dump(payload, handle)

    def add_embedding(
        self,
        name: str,
        embedding: EmbeddingVector,
        source_image: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredEmbedding:
        record = StoredEmbedding(
            name=name,
            embedding=np.asarray(embedding, dtype=np.float32),
            source_image=source_image,
            metadata=metadata or {},
        )
        self._records.append(record)
        self.save()
        return record

    def extend(self, records: Iterable[StoredEmbedding]) -> None:
        self._records.extend(records)
        self.save()

