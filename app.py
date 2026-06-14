from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd
import subprocess
import sys
import os
import traceback

from src.parser import filter_df_by_fachgruppe, parse_ebm_xml_to_dataframe
from src.rag_pipeline import EbmRAGPipeline, build_pipeline_from_paths


ROOT = Path(__file__).resolve().parent
DATA_XML = ROOT / "data" / "ebm.xml"
STORE_DIR = ROOT / "data" / "vector_store"


PIPELINE: EbmRAGPipeline | None = None
DATA_SOURCE_STATUS: str = "unknown"


def get_pipeline() -> EbmRAGPipeline:
    global PIPELINE
    if PIPELINE is None:
        # Ensure vector store exists. Try to download full EBM and build; on failure, fall back to demo XML.
        try:
            ensure_vector_store()
        except Exception:
            # Log and continue; build_pipeline_from_paths will attempt to load or build and may raise a clearer error
            print("Warning: ensure_vector_store failed:\n" + traceback.format_exc())
        PIPELINE = build_pipeline_from_paths(DATA_XML, STORE_DIR)
    return PIPELINE


def ensure_vector_store() -> str:
    """Try to prepare the FAISS store before the app starts.

    Steps:
    - If the store already exists, do nothing.
    - Attempt to download the full EBM ZIP (scripts/download_full_ebm.py).
    - Attempt to build the FAISS store (scripts/build_database.py).
    - If building fails due to missing Fachgruppe 001 data, fall back to using the local demo XML by
      disabling the Fachgruppe filter via environment variable `SKIP_FG_FILTER=1`.
    """
    global DATA_SOURCE_STATUS
    root = Path(__file__).resolve().parent
    store_dir = STORE_DIR
    if store_dir.exists() and (store_dir / "index.faiss").exists() and (store_dir / "metadata.jsonl").exists():
        # Existing store present
        DATA_SOURCE_STATUS = "store"
        return DATA_SOURCE_STATUS

    # Try to download full EBM
    download_script = root / "scripts" / "download_full_ebm.py"
    build_script = root / "scripts" / "build_database.py"

    if download_script.exists():
        try:
            print("Attempting to download full KBV EBM archive...")
            subprocess.run([sys.executable, str(download_script)], check=True, timeout=600)
        except Exception:
            print("Download script failed:\n" + traceback.format_exc())

    # Try to build the FAISS store
    if build_script.exists():
        try:
            print("Attempting to build FAISS store from XML...")
            subprocess.run([sys.executable, str(build_script), "--xml", str(DATA_XML), "--store", str(STORE_DIR)], check=True, timeout=600)
            DATA_SOURCE_STATUS = "full"
            return DATA_SOURCE_STATUS
        except subprocess.CalledProcessError as e:
            print("Build script failed with CalledProcessError:\n" + traceback.format_exc())
            # If the build failed due to empty Fachgruppe 001, allow fallback to demo XML
            # The build scripts raise ValueError in that case; detect it by inspecting output/exception
            # Fallback: disable fachgruppe filter so demo XML (with 2 entries) can be used
            print("Falling back to demo XML: disabling Fachgruppe-001 filter for this run.")
            os.environ["SKIP_FG_FILTER"] = "1"
            DATA_SOURCE_STATUS = "demo"
            return DATA_SOURCE_STATUS
        except Exception:
            print("Build script failed:\n" + traceback.format_exc())
            print("Falling back to demo XML: disabling Fachgruppe-001 filter for this run.")
            os.environ["SKIP_FG_FILTER"] = "1"
            DATA_SOURCE_STATUS = "demo"
            return DATA_SOURCE_STATUS
    else:
        print("No build script found; continuing and relying on existing data/ebm.xml")
        DATA_SOURCE_STATUS = "xml"
        return DATA_SOURCE_STATUS

    # Default
    DATA_SOURCE_STATUS = "unknown"
    return DATA_SOURCE_STATUS


def format_retrieved(results: list[dict]) -> str:
    if not results:
        return "No retrieved documents."
    lines = []
    for item in results:
        lines.append(
            f"### {item['code']} - {item.get('title') or 'Unbenannt'}\n"
            f"Score: {item['score']:.3f}\n\n"
            f"{item['text']}"
        )
    return "\n\n".join(lines)


def ask_ebm(question: str) -> tuple[str, str, float]:
    pipeline = get_pipeline()
    result = pipeline.answer(question)
    return result["answer"], format_retrieved(result["retrieved_documents"]), result["confidence"]


def explain_code(code: str) -> tuple[str, str]:
    pipeline = get_pipeline()
    result = pipeline.explain_code(code)
    return result["answer"], format_retrieved(result["retrieved_documents"])


def quiz_me() -> tuple[str, str, str]:
    pipeline = get_pipeline()
    doc = pipeline.random_document()
    prompt = f"What does this code describe?\n\nEBM code: {doc.code}"
    return prompt, gr.update(value="", visible=False), doc.code


def reveal_quiz_answer(code: str) -> tuple[str, str]:
    pipeline = get_pipeline()
    if not code:
        return gr.update(value="No code selected.", visible=True), ""
    result = pipeline.explain_code(code)
    return gr.update(value=result["answer"], visible=True), format_retrieved(result["retrieved_documents"])


def explore_ebm(query: str, chapter: str) -> pd.DataFrame:
    pipeline = get_pipeline()
    results = pipeline.search(query=query, chapter=chapter, top_k=20)
    if not results:
        return pd.DataFrame(columns=["code", "title", "points", "exclusions", "notes"])

    rows = []
    for item in results[:10]:
        rows.append(
            {
                "code": item["code"],
                "title": item.get("title") or "",
                "points": item.get("points") or "",
                "exclusions": ", ".join(item.get("exclusions_text", [])),
                "notes": " | ".join(item.get("notes", [])),
            }
        )

    return pd.DataFrame(rows)


def browse_chapters() -> list[str]:
    if STORE_DIR.exists() and (STORE_DIR / "metadata.jsonl").exists():
        pipeline = get_pipeline()
        chapters = pipeline.list_chapters()
    elif DATA_XML.exists():
        df = parse_ebm_xml_to_dataframe(str(DATA_XML))
        df = filter_df_by_fachgruppe(df)
        chapters = sorted(
            {
                str(value)
                for value in df.get("chapter_name", pd.Series(dtype=str)).dropna().tolist()
                if str(value).strip()
            }
        )
    else:
        chapters = []
    return ["All"] + chapters

def build_app() -> gr.Blocks:
    # Ensure the vector store/download step has been attempted so browse_chapters sees correct data
    try:
        ensure_vector_store()
    except Exception:
        print("ensure_vector_store during UI build failed:\n" + traceback.format_exc())
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
    ) as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>EBM Mentor</h1>
              <p>Erkunde die deutsche EBM interaktiv.</p>
              <p>Die Daten basieren auf dem offiziellen KBV EBM-Update und werden lokal als XML verarbeitet. Die App erstellt aus der XML-Datei einen Suchindex und nutzt Retrieval, um Antworten auf Basis der lokalen EBM-Daten zu liefern.</p>
              <p style="font-size:0.95rem; opacity:0.85;">Hinweis: Diese Funktion ist experimentell. Es gibt keine Gewährleistung für die Richtigkeit der Ergebnisse, und die Anwendung ersetzt keine offizielle Abrechnungsauskunft.</p>
            </div>
            """
        )

        # Data source status indicator
        status_text = "Unbekannt"
        if DATA_SOURCE_STATUS == "full":
            status_text = "Datenquelle: Vollständiges KBV EBM (heruntergeladen und indexiert)."
        elif DATA_SOURCE_STATUS == "store":
            status_text = "Datenquelle: Vorhandener Vektor-Store (lokal)."
        elif DATA_SOURCE_STATUS == "xml":
            status_text = "Datenquelle: Lokale XML-Datei (data/ebm.xml)."
        elif DATA_SOURCE_STATUS == "demo":
            status_text = "Datenquelle: Demo-XML verwendet (nur Beispiel-Einträge)."
        else:
            status_text = "Datenquelle: Unbekannt."

        gr.Markdown(f"**Status:** {status_text}")

        chapter_choices = browse_chapters()

        with gr.Row():
            search_query = gr.Textbox(label="Search", placeholder="points, exclusions, title, notes...")
            chapter = gr.Dropdown(
                label="Chapter",
                choices=chapter_choices,
                value="All",
                interactive=True,
            )
        search_btn = gr.Button("Search", variant="primary")
        table = gr.Dataframe(
            headers=["code", "title", "points", "exclusions", "notes"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            row_count=(5, "dynamic"),
        )
        search_btn.click(
            explore_ebm,
            inputs=[search_query, chapter],
            outputs=[table],
        )

    return demo


app = build_app()


if __name__ == "__main__":
    app.launch()
