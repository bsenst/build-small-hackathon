from __future__ import annotations

import pandas as pd

from src.chunking import dataframe_to_documents, document_to_search_text


def test_dataframe_to_documents_and_search_text() -> None:
    df = pd.DataFrame(
        [
            {
                "code": "01100",
                "short_text": "Unvorhergesehene Inanspruchnahme I",
                "receipt_text": "Receipt",
                "long_text": "Beschreibung.",
                "chapter_code": "01",
                "chapter_name": "Kapitel 01",
                "bereich": "Bereich A",
                "kapitel": "Kapitelname",
                "abschnitt": "Abschnittname",
                "notes": ["Note 1", "Note 2"],
                "points": "196",
                "fachgruppen": ["A", "B"],
                "exclusions": [{"code": "01101", "description": "Ausschluss"}],
                "gkv_account_types": ["A"],
            }
        ]
    )

    docs = dataframe_to_documents(df)
    assert len(docs) == 1
    text = document_to_search_text(docs[0])
    assert "EBM Code: 01100" in text
    assert "Points: 196" in text
    assert "Exclusions:" in text

