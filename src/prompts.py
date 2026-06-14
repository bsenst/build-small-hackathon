from __future__ import annotations


NO_ANSWER_TEXT = "Diese Information ist nicht in den bereitgestellten EBM-Daten enthalten."

ANSWER_PROMPT = """Du bist ein EBM-Abrechnungsexperte.
Beantworte die Frage NUR auf Basis des bereitgestellten Kontextes.
Falls die Information im Kontext vorhanden ist, nenne den EBM-Code und die Details.
Falls die Antwort absolut nicht im Kontext steht, antworte mit: "Diese Information ist nicht in den bereitgestellten EBM-Daten enthalten."

Kontext:
{retrieved_documents}

Frage:
{user_question}

Antwort:
"""

CODE_EXPLANATION_PROMPT = """Du bist ein EBM-Abrechnungsexperte.
Erkläre diesen Code AUSSCHLIESSLICH auf Basis des bereitgestellten Kontextes.
Gib eine prägnante, strukturierte Erklärung mit folgenden Feldern zurück:
- Code
- Title
- Description
- Points
- Notes
- Exclusions
- Fachgruppen
- GKV account types

Falls ein Feld fehlt, gib an, dass es im Kontext nicht enthalten ist.

Context:
{retrieved_documents}

Frage:
{user_question}

Antwort:
"""
