---
title: EBM Mentor
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# EBM Mentor

EBM Mentor is a Retrieval-Augmented Generation assistant for German EBM billing research. It helps physicians, coders, and practice staff search and understand EBM code descriptions, points, exclusions, notes, and eligibility details.

The system is RAG-only:

- `CohereLabs/tiny-aya-water` is used only for answer generation.
- All EBM knowledge comes from retrieval over locally indexed EBM XML data.
- If the answer is not in the retrieved context, the app must say:
  `Diese Information ist nicht in den bereitgestellten EBM-Daten enthalten.`

## Architecture

```mermaid
flowchart LR
    A[Official EBM XML] --> B[src/parser.py]
    B --> C[Structured records]
    C --> D[src/chunking.py]
    D --> E[Searchable documents]
    E --> F[src/embeddings.py]
    F --> G[FAISS index]
    G --> H[src/retriever.py]
    H --> I[src/prompts.py]
    I --> J[src/model.py]
    J --> K[Gradio app]
```

## Features

- Free-form EBM question answering
- Code explanation with structured metadata
- Random quiz mode
- Search and chapter browsing
- Source citations and retrieved document viewer
- Confidence score for retrieval results

## Repository Layout

```text
ebm-rag-trainer/
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── data/
│   └── ebm.xml
├── src/
│   ├── parser.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── rag_pipeline.py
│   └── model.py
├── scripts/
│   ├── build_database.py
│   └── ingest_ebm.py
├── tests/
└── .github/
```

## Setup

### 1. Create an environment

```bash
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add or download the official EBM XML

Place the official XML file at:

```text
data/ebm.xml
```

Or download the full KBV EBM source and extract the XML automatically:

```bash
python scripts/download_full_ebm.py
```

This will write the extracted XML to `data/ebm.xml` and keep the archive in `data/SDEBM_V1.61.zip`.

Die Daten stammen vom offiziellen KBV-Update unter https://update.kbv.de/ita-update/Stammdateien/SDEBM/ und werden hier lokal verarbeitet. Die App liest die XML-Datei ein, wandelt sie in strukturierte Datensätze um und erstellt daraus einen lokalen FAISS-Vektorstore. Anschließend nutzt die Anwendung diese lokal gespeicherten EBM-Daten für die Suche und Retrieval-basierte Antwortgenerierung. Weiterführende Informationen zum EBM finden Sie unter https://ebm.kbv.de/.

Hinweis: Diese Funktion ist experimentell und es besteht keine Gewährleistung für Vollständigkeit oder Richtigkeit der Ausgaben. Die Anwendung ersetzt keine offizielle Abrechnungsauskunft.

The repository includes a tiny demo XML so the codebase is runnable out of the box, but production use should replace it with the official current EBM source.

### 4. Build the local database

```bash
python scripts/ingest_ebm.py --xml data/ebm.xml --output data/processed
python scripts/build_database.py --xml data/ebm.xml --store data/vector_store
```

### 5. Run the app

```bash
python app.py
```

## Hugging Face Spaces Deployment

1. Create a new Hugging Face Space.
2. Choose `Gradio` as the SDK.
3. Upload this repository.
4. Ensure `data/ebm.xml` is included or mounted in the Space.
5. Let the Space build the local FAISS index on first launch or prebuild it with the scripts above.

### Space Notes

- SDK: `gradio`
- Hardware: `CPU Basic / Zero`
- No external database
- No paid APIs
- Local FAISS index and local metadata files only

## Example Screenshots

Add screenshots here after deployment:

```text
docs/screenshots/ask-ebm.png
docs/screenshots/explain-code.png
docs/screenshots/quiz-mode.png
docs/screenshots/explore-ebm.png
```

## Sample Questions

- Was bedeutet EBM-Code 01100?
- Welche Punkte hat die Leistung?
- Gibt es Ausschlüsse für diesen Code?
- Welche Fachgruppen sind berechtigt?
- Welche Hinweise stehen in den Anmerkungen?

## Testing

Run the local test suite:

```bash
pytest
```

The tests cover:

- XML parsing
- document chunking
- retrieval

## Safety and Scope

This project is designed as a research assistant, not a legal authority. Always verify billing decisions against the official EBM source material and current practice guidance.
