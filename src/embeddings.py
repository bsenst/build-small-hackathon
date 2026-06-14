from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class EmbeddingConfig:
    model_name: str = DEFAULT_EMBEDDING_MODEL


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            # SentenceTransformer respects TRANSFORMERS_OFFLINE env var internally
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True
            )
        return self._model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
