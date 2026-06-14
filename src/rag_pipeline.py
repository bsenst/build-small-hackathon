from __future__ import annotations

from pathlib import Path
from typing import Any

from src.chunking import dataframe_to_documents
from src.embeddings import EmbeddingModel
from src.model import load_generation_pipeline
from src.prompts import ANSWER_PROMPT, CODE_EXPLANATION_PROMPT, NO_ANSWER_TEXT
from src.parser import parse_ebm_xml_to_dataframe
from src.retriever import EbmRetriever
from src.vector_store import EbmVectorStore


def _format_context(results: list[dict[str, Any]]) -> str:
    blocks = []
    for item in results:
        notes = "\n".join(f"- {note}" for note in item.get("notes", [])) or "Keine."
        exclusions = "\n".join(
            f"- {ex['code']}: {ex.get('description') or ''}".strip()
            for ex in item.get("exclusions", [])
            if ex.get("code")
        ) or "Keine."
        blocks.append(
            "\n".join(
                [
                    f"EBM Code: {item.get('code')}",
                    f"Title: {item.get('title') or ''}",
                    f"Points: {item.get('points') if item.get('points') is not None else 'Nicht angegeben'}",
                    f"Chapter: {item.get('chapter_name') or ''}",
                    f"Description: {item.get('long_text') or item.get('short_text') or ''}",
                    f"Notes:\n{notes}",
                    f"Exclusions:\n{exclusions}",
                    f"Fachgruppen: {', '.join(item.get('fachgruppen', [])) or 'Nicht angegeben'}",
                    f"GKV account types: {', '.join(item.get('gkv_account_types', [])) or 'Nicht angegeben'}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _extract_citations(results: list[dict[str, Any]]) -> list[str]:
    citations = []
    for item in results:
        code = item.get("code")
        title = item.get("title")
        if code:
            citations.append(f"{code} - {title}".strip())
    return citations


class EbmRAGPipeline:
    def __init__(self, retriever: EbmRetriever, generator=None):
        self.retriever = retriever
        self._generator = generator

    @property
    def generator(self):
        if self._generator is None:
            self._generator = load_generation_pipeline()
        return self._generator

    def answer(self, question: str, top_k: int = 5, chapter: str | None = None) -> dict[str, Any]:
        retrieved = self.retriever.retrieve(question, top_k=top_k, chapter=chapter)
        if not retrieved:
            return {
                "answer": NO_ANSWER_TEXT,
                "retrieved_documents": [],
                "confidence": 0.0,
                "citations": [],
            }

        context = _format_context(retrieved)
        prompt = ANSWER_PROMPT.format(retrieved_documents=context, user_question=question)
        try:
            generated = self.generator(prompt)[0]["generated_text"].strip()
        except Exception:
            generated = ""
        answer = generated or NO_ANSWER_TEXT
        confidence = max(0.0, min(1.0, float(retrieved[0].get("confidence", 0.0))))
        return {
            "answer": answer,
            "retrieved_documents": retrieved,
            "confidence": confidence,
            "citations": _extract_citations(retrieved),
        }

    def explain_code(self, code: str) -> dict[str, Any]:
        code = code.strip()
        document = self.retriever.get_by_code(code)
        if not document:
            return {
                "answer": NO_ANSWER_TEXT,
                "retrieved_documents": [],
                "confidence": 0.0,
                "citations": [],
            }

        retrieved = [dict(document)]
        context = _format_context(retrieved)
        prompt = CODE_EXPLANATION_PROMPT.format(retrieved_documents=context, user_question=f"Explain EBM code {code}.")
        try:
            generated = self.generator(prompt)[0]["generated_text"].strip()
        except Exception:
            generated = ""
        confidence = 1.0
        return {
            "answer": generated or NO_ANSWER_TEXT,
            "retrieved_documents": retrieved,
            "confidence": confidence,
            "citations": _extract_citations(retrieved),
        }

    def search(self, query: str, top_k: int = 10, chapter: str | None = None) -> list[dict[str, Any]]:
        return self.retriever.search(query=query, top_k=top_k, chapter=chapter)

    def random_document(self):
        from types import SimpleNamespace

        doc = self.retriever.random_document()
        return SimpleNamespace(**doc)

    def list_chapters(self) -> list[str]:
        return self.retriever.list_chapters()


def build_pipeline_from_paths(xml_path: str | Path, store_dir: str | Path, embedding_model: EmbeddingModel | None = None) -> EbmRAGPipeline:
    xml_path = Path(xml_path)
    store_dir = Path(store_dir)
    embedding_model = embedding_model or EmbeddingModel()

    if store_dir.exists() and (store_dir / "index.faiss").exists() and (store_dir / "metadata.jsonl").exists():
        store = EbmVectorStore.load(store_dir)
    else:
        df = parse_ebm_xml_to_dataframe(str(xml_path))
        documents = dataframe_to_documents(df)
        store, embeddings = EbmVectorStore.build(documents, embedding_model=embedding_model)
        store.save(store_dir, embeddings=embeddings)

    retriever = EbmRetriever(store, embedding_model=embedding_model)
    return EbmRAGPipeline(retriever=retriever)
