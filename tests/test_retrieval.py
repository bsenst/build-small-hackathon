from __future__ import annotations

import numpy as np

from src.chunking import EbmDocument
from src.vector_store import EbmVectorStore
from src.retriever import EbmRetriever


class DummyEmbeddingModel:
    model_name = "dummy"

    def encode(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                np.array(
                    [
                        1.0 if "01100" in lower else 0.0,
                        1.0 if "inanspruchnahme" in lower else 0.0,
                        1.0 if "vorsorge" in lower else 0.0,
                    ],
                    dtype=np.float32,
                )
            )
        arr = np.vstack(vectors)
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return arr / norm


def test_retrieval_ranks_relevant_code() -> None:
    docs = [
        EbmDocument(
            code="01100",
            title="Unvorhergesehene Inanspruchnahme I",
            short_text="Unvorhergesehene Inanspruchnahme I",
            receipt_text=None,
            long_text="Notfallversorgung",
            chapter_code=None,
            chapter_name="Kapitel A",
            bereich=None,
            kapitel=None,
            abschnitt=None,
            notes=[],
            points=196,
            fachgruppen=[],
            exclusions=[],
            gkv_account_types=[],
        ),
        EbmDocument(
            code="01732",
            title="Vorsorge",
            short_text="Vorsorge",
            receipt_text=None,
            long_text="Vorsorgeleistung",
            chapter_code=None,
            chapter_name="Kapitel B",
            bereich=None,
            kapitel=None,
            abschnitt=None,
            notes=[],
            points=100,
            fachgruppen=[],
            exclusions=[],
            gkv_account_types=[],
        ),
    ]

    store, _ = EbmVectorStore.build(docs, embedding_model=DummyEmbeddingModel())
    retriever = EbmRetriever(store, embedding_model=DummyEmbeddingModel())
    results = retriever.retrieve("Was bedeutet 01100?", top_k=1)
    assert results[0]["code"] == "01100"

