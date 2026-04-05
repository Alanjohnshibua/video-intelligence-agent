from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FaceIdentifierConfig:
    database_path: Path | str = Path("data/embeddings.pkl")
    unknown_dir: Path | str = Path("data/unknown")
    similarity_threshold: float = 0.4
    detector_backend: str = "retinaface"
    embedding_model: str = "ArcFace"
    normalization: str = "ArcFace"
    align: bool = True
    enforce_detection: bool = False
    prefer_gpu: bool = True
    tensorflow_memory_growth: bool = True

    def resolved_database_path(self) -> Path:
        return Path(self.database_path)

    def resolved_unknown_dir(self) -> Path:
        return Path(self.unknown_dir)
