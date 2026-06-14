from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np

from src.chunking import EbmDocument, dataframe_to_documents, document_to_search_text, document_to_structured_dict
from src.embeddings import EmbeddingModel, DEFAULT_EMBEDDING_MODEL


@dataclass
class RetrievalResult:
    code: str
    title: str
    score: float
    text: str
    structured: dict[str, Any]


class EbmVectorStore:
    def __init__(self, index: faiss.Index | None, documents: list[dict[str, Any]], embedding_model_name: str):
        self.index = index
        self.documents = documents
        self.embedding_model_name = embedding_model_name

    @classmethod
    def build(
        cls,
        documents: Iterable[EbmDocument],
        embedding_model: EmbeddingModel | None = None,
    ) -> tuple["EbmVectorStore", np.ndarray]:
        embedding_model = embedding_model or EmbeddingModel()
        docs = [
            document_to_structured_dict(doc) if hasattr(doc, "__dataclass_fields__") else dict(doc)
            for doc in documents
        ]
        texts = [
            document_to_search_text(EbmDocument(**{k: v for k, v in doc.items() if k != "search_text"}))
            for doc in docs
        ]
        if not texts:
            raise ValueError(
                "Cannot build vector store: no documents available. "
                "Check that data/ebm.xml contains Fachgruppe 001 entries or remove the Fachgruppe-001 filter."
            )
        embeddings = embedding_model.encode(texts)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(
                "Embedding model returned invalid embeddings. "
                "Expected a 2D array with one embedding per document."
            )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        store = cls(index=index, documents=docs, embedding_model_name=embedding_model.model_name)
        return store, embeddings

    @classmethod
    def from_dataframe(cls, df, embedding_model: EmbeddingModel | None = None) -> tuple["EbmVectorStore", np.ndarray]:
        return cls.build(dataframe_to_documents(df), embedding_model=embedding_model)

    def save(self, directory: str | Path, embeddings: np.ndarray | None = None) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise ValueError("Cannot save a store without an index.")

        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "metadata.jsonl").write_text(
            "\n".join(json.dumps(doc, ensure_ascii=False) for doc in self.documents),
            encoding="utf-8",
        )
        (path / "config.json").write_text(
            json.dumps({"embedding_model_name": self.embedding_model_name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if embeddings is not None:
            np.save(path / "embeddings.npy", embeddings)

    @classmethod
    def load(cls, directory: str | Path) -> "EbmVectorStore":
        path = Path(directory)
        index = faiss.read_index(str(path / "index.faiss"))
        metadata_path = path / "metadata.jsonl"
        documents = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        config_path = path / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            embedding_model_name = config.get("embedding_model_name", DEFAULT_EMBEDDING_MODEL)
        else:
            embedding_model_name = DEFAULT_EMBEDDING_MODEL
        return cls(index=index, documents=documents, embedding_model_name=embedding_model_name)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        if self.index is None:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        scores, indices = self.index.search(query, top_k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            structured = dict(doc)
            search_text = structured.get("search_text")
            if not search_text:
                search_text = document_to_search_text(
                    EbmDocument(**{k: v for k, v in structured.items() if k != "search_text"})
                )
            results.append(
                RetrievalResult(
                    code=str(doc.get("code") or ""),
                    title=str(doc.get("title") or doc.get("short_text") or doc.get("code") or ""),
                    score=float(score),
                    text=str(search_text or ""),
                    structured=structured,
                )
            )
        return results
