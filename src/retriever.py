from __future__ import annotations

import re
from typing import Any

from src.embeddings import EmbeddingModel
from src.vector_store import EbmVectorStore, RetrievalResult


CODE_PATTERN = re.compile(r"\b\d{5}\b")


class EbmRetriever:
    def __init__(self, store: EbmVectorStore, embedding_model: EmbeddingModel | None = None):
        self.store = store
        self.embedding_model = embedding_model or EmbeddingModel(store.embedding_model_name)

    def retrieve(self, query: str, top_k: int = 5, chapter: str | None = None) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        embeddings = self.embedding_model.encode([query])
        results = self.store.search(embeddings, top_k=top_k * 3 if chapter and chapter != "All" else top_k)

        payloads = [self._to_payload(result) for result in results]
        if chapter and chapter != "All":
            payloads = [item for item in payloads if item.get("chapter_name") == chapter]
        return payloads[:top_k]

    def get_by_code(self, code: str) -> dict[str, Any] | None:
        code = code.strip()
        for doc in self.store.documents:
            if str(doc.get("code") or "") == code:
                return dict(doc)
        return None

    def random_document(self) -> dict[str, Any]:
        import random

        if not self.store.documents:
            raise ValueError("No documents available.")
        return dict(random.choice(self.store.documents))

    def list_chapters(self) -> list[str]:
        chapters = sorted(
            {
                str(doc.get("chapter_name"))
                for doc in self.store.documents
                if doc.get("chapter_name")
            }
        )
        return chapters

    def search(self, query: str, top_k: int = 10, chapter: str | None = None) -> list[dict[str, Any]]:
        return self.retrieve(query=query, top_k=top_k, chapter=chapter)

    def code_from_text(self, text: str) -> str | None:
        match = CODE_PATTERN.search(text or "")
        return match.group(0) if match else None

    @staticmethod
    def _to_payload(result: RetrievalResult) -> dict[str, Any]:
        payload = dict(result.structured)
        payload["score"] = result.score
        payload["title"] = result.title
        payload["text"] = result.text
        payload["confidence"] = max(0.0, min(1.0, (result.score + 1.0) / 2.0))
        payload["exclusions_text"] = [
            item.get("code")
            for item in payload.get("exclusions", [])
            if isinstance(item, dict) and item.get("code")
        ]
        return payload
