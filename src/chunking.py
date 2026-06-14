from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EbmDocument:
    code: str
    title: str
    short_text: str | None
    receipt_text: str | None
    long_text: str | None
    chapter_code: str | None
    chapter_name: str | None
    bereich: str | None
    kapitel: str | None
    abschnitt: str | None
    notes: list[str]
    points: int | None
    fachgruppen: list[str]
    exclusions: list[dict[str, str | None]]
    gkv_account_types: list[str]
    raw: dict[str, Any] | None = None


def _coerce_points(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def dataframe_to_documents(df: pd.DataFrame) -> list[EbmDocument]:
    documents: list[EbmDocument] = []
    for _, row in df.iterrows():
        data = row.to_dict()
        title = data.get("short_text") or data.get("receipt_text") or data.get("code") or ""
        documents.append(
            EbmDocument(
                code=str(data.get("code") or ""),
                title=str(title),
                short_text=data.get("short_text"),
                receipt_text=data.get("receipt_text"),
                long_text=data.get("long_text"),
                chapter_code=data.get("chapter_code"),
                chapter_name=data.get("chapter_name"),
                bereich=data.get("bereich"),
                kapitel=data.get("kapitel"),
                abschnitt=data.get("abschnitt"),
                notes=[str(item) for item in _safe_list(data.get("notes")) if item],
                points=_coerce_points(data.get("points")),
                fachgruppen=[str(item) for item in _safe_list(data.get("fachgruppen")) if item],
                exclusions=[
                    {
                        "code": item.get("code"),
                        "description": item.get("description"),
                    }
                    for item in _safe_list(data.get("exclusions"))
                    if isinstance(item, dict)
                ],
                gkv_account_types=[str(item) for item in _safe_list(data.get("gkv_account_types")) if item],
                raw=data,
            )
        )
    return documents


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "Nicht angegeben."


def _format_exclusions(items: list[dict[str, str | None]]) -> str:
    if not items:
        return "Keine Ausschlüsse angegeben."
    formatted = []
    for item in items:
        code = item.get("code") or ""
        description = item.get("description") or ""
        if description:
            formatted.append(f"- {code}: {description}")
        else:
            formatted.append(f"- {code}")
    return "\n".join(formatted)


def document_to_search_text(doc: EbmDocument) -> str:
    parts = [
        f"EBM Code: {doc.code}",
        f"Title: {doc.title}",
    ]
    if doc.short_text:
        parts.append(f"Short text: {doc.short_text}")
    if doc.receipt_text:
        parts.append(f"Receipt text: {doc.receipt_text}")
    if doc.long_text:
        parts.append(f"Description: {doc.long_text}")
    if doc.points is not None:
        parts.append(f"Points: {doc.points}")
    if doc.notes:
        parts.append("Notes:\n" + _format_bullets(doc.notes))
    if doc.exclusions:
        parts.append("Exclusions:\n" + _format_exclusions(doc.exclusions))
    if doc.fachgruppen:
        parts.append("Fachgruppen:\n" + _format_bullets(doc.fachgruppen))
    if doc.gkv_account_types:
        parts.append("GKV account types:\n" + _format_bullets(doc.gkv_account_types))
    if doc.chapter_name:
        parts.append(f"Chapter: {doc.chapter_name}")
    if doc.kapitel:
        parts.append(f"Kapitel: {doc.kapitel}")
    if doc.abschnitt:
        parts.append(f"Abschnitt: {doc.abschnitt}")
    return "\n\n".join(parts)


def document_to_structured_dict(doc: EbmDocument) -> dict[str, Any]:
    payload = asdict(doc)
    payload["search_text"] = document_to_search_text(doc)
    return payload


def dataframe_to_search_corpus(df: pd.DataFrame) -> list[dict[str, Any]]:
    docs = dataframe_to_documents(df)
    return [document_to_structured_dict(doc) for doc in docs]

