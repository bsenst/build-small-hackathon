from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd
import subprocess
import sys
import traceback

from src.parser import parse_ebm_xml_to_dataframe
from src.rag_pipeline import EbmRAGPipeline, build_pipeline_from_paths


ROOT = Path(__file__).resolve().parent
DATA_XML = ROOT / "data" / "ebm.xml"
STORE_DIR = ROOT / "data" / "vector_store"


PIPELINE: EbmRAGPipeline | None = None
DATA_SOURCE_STATUS: str = "unknown"


def get_pipeline() -> EbmRAGPipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = build_pipeline_from_paths(DATA_XML, STORE_DIR)
    return PIPELINE


def ensure_vector_store() -> str:
    """Ensure the FAISS vector store is ready.
    
    Steps:
    1. If store already exists, use it.
    2. Try to download full EBM and build the store from it with Fachgruppe 001 filter.
    3. If download/build fails, fall back to dummy XML.
    
    Returns:
    - Status string: "full" (built from full EBM), "store" (reused), or "demo" (fallback)
    """
    global DATA_SOURCE_STATUS
    store_dir = STORE_DIR
    
    # Check if store already exists
    if store_dir.exists() and (store_dir / "index.faiss").exists() and (store_dir / "metadata.jsonl").exists():
        print("✓ Vector store found, using it.")
        DATA_SOURCE_STATUS = "store"
        return DATA_SOURCE_STATUS

    root = Path(__file__).resolve().parent
    download_script = root / "scripts" / "download_full_ebm.py"
    build_script = root / "scripts" / "build_database.py"
    
    # Path to extracted full EBM (if download succeeds)
    extracted_xml_path = root / "data" / "sdebm_extracted" / "XML" / "850_01.61_74_tf2017q4_nr1.xml"
    dummy_xml_path = DATA_XML

    # Try to download full EBM
    download_success = False
    if download_script.exists():
        try:
            print("📥 Downloading full KBV EBM archive...")
            result = subprocess.run([sys.executable, str(download_script)], capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(result.stdout)
                download_success = True
            else:
                print(f"⚠️  Download warning:\n{result.stderr}")
        except Exception as e:
            print(f"⚠️  Download failed: {e}")

    # Choose XML source: extracted full EBM with Fachgruppe 001 filter, or dummy XML as fallback
    xml_to_use = dummy_xml_path
    use_fachgruppe_filter = False
    
    if download_success and extracted_xml_path.exists():
        xml_to_use = extracted_xml_path
        use_fachgruppe_filter = True
        print(f"✓ Using downloaded full EBM: {extracted_xml_path}")
        print("  Applying Fachgruppe 001 filter...")
    else:
        print(f"⚠️  Using fallback dummy XML: {dummy_xml_path}")

    # Try to build FAISS store
    if build_script.exists():
        try:
            print("🔨 Building FAISS vector store...")
            cmd = [sys.executable, str(build_script), "--xml", str(xml_to_use), "--store", str(STORE_DIR)]
            if use_fachgruppe_filter:
                cmd.append("--fachgruppe-filter")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            if result.returncode == 0:
                print(result.stdout)
                data_source = "full" if use_fachgruppe_filter else "demo"
                DATA_SOURCE_STATUS = data_source
                print(f"✓ Vector store built successfully from {'full EBM (Fachgruppe 001)' if use_fachgruppe_filter else 'demo data'}.")
                return DATA_SOURCE_STATUS
            else:
                print(f"⚠️  Build failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Build failed: {e}")

    # Final fallback: use dummy XML without filter
    print("⚠️  Using demo/fallback XML (sample data)...")
    DATA_SOURCE_STATUS = "demo"
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
        # Use all documents from the full EBM
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
    # Initialize EBM data and vector store at startup
    print("\n" + "="*70)
    print("STARTUP: Initializing EBM data and vector store...")
    print("="*70)
    try:
        ensure_vector_store()
    except Exception as e:
        print(f"Warning: {traceback.format_exc()}")
    
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
            status_text = "✓ Vollständiges KBV EBM (heruntergeladen und indexiert)."
        elif DATA_SOURCE_STATUS == "store":
            status_text = "✓ Vektor-Store wiederverwendet (aus vorherigem Durchlauf)."
        elif DATA_SOURCE_STATUS == "demo":
            status_text = "⚠️  Demo-Daten (Beispiel-EBM-Einträge)."
        
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
