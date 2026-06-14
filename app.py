from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from src.parser import parse_ebm_xml_to_dataframe
from src.rag_pipeline import EbmRAGPipeline, build_pipeline_from_paths


ROOT = Path(__file__).resolve().parent
DATA_XML = ROOT / "data" / "ebm.xml"
STORE_DIR = ROOT / "data" / "vector_store"


PIPELINE: EbmRAGPipeline | None = None


def get_pipeline() -> EbmRAGPipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = build_pipeline_from_paths(DATA_XML, STORE_DIR)
    return PIPELINE


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


def explore_ebm(query: str, chapter: str) -> tuple[str, str, str]:
    pipeline = get_pipeline()
    results = pipeline.search(query=query, chapter=chapter, top_k=10)
    if not results:
        empty = pd.DataFrame(columns=["code", "title", "points", "exclusions", "notes"])
        return empty, "No matches found.", ""

    rows = []
    for item in results:
        rows.append(
            {
                "code": item["code"],
                "title": item.get("title") or "",
                "points": item.get("points") or "",
                "exclusions": ", ".join(item.get("exclusions_text", [])),
                "notes": " | ".join(item.get("notes", [])),
            }
        )

    table = pd.DataFrame(rows)
    details = format_retrieved(results)
    chapters = "\n".join(sorted({item.get("chapter_name") or "" for item in results if item.get("chapter_name")}))
    return table, details, chapters


def browse_chapters() -> list[str]:
    if STORE_DIR.exists() and (STORE_DIR / "metadata.jsonl").exists():
        pipeline = get_pipeline()
        chapters = pipeline.list_chapters()
    elif DATA_XML.exists():
        df = parse_ebm_xml_to_dataframe(str(DATA_XML))
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


CSS = """
:root {
  --bg: #f4f7f6;
  --panel: rgba(255,255,255,0.85);
  --ink: #183a37;
  --muted: #4e6964;
  --accent: #2e7d6b;
  --accent-2: #165d50;
  --border: rgba(24,58,55,0.12);
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(46,125,107,0.12), transparent 32%),
    radial-gradient(circle at top right, rgba(24,58,55,0.08), transparent 28%),
    linear-gradient(180deg, #f7fbfa 0%, #eef4f2 100%);
  color: var(--ink);
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}

.gradio-container,
.gradio-container * {
  color: var(--ink);
}

.gradio-container .prose,
.gradio-container .prose *,
.gradio-container .markdown,
.gradio-container .markdown * {
  color: var(--ink) !important;
}

.hero {
  background: linear-gradient(135deg, rgba(24,58,55,0.95), rgba(46,125,107,0.92));
  color: white;
  border-radius: 24px;
  padding: 28px;
  box-shadow: 0 20px 60px rgba(24,58,55,0.18);
}

.hero h1 {
  margin: 0;
  font-size: 2.4rem;
  letter-spacing: -0.03em;
}

.hero p {
  margin: 8px 0 0 0;
  opacity: 0.92;
  font-size: 1.05rem;
}

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 20px;
  backdrop-filter: blur(10px);
  box-shadow: 0 10px 30px rgba(24,58,55,0.08);
}

.card * {
  color: var(--ink);
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap,
.gradio-container .block,
.gradio-container .table,
.gradio-container .table *,
.gradio-container .dataframe,
.gradio-container .dataframe *,
.gradio-container .tab-nav button {
  color: var(--ink) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: rgba(255, 255, 255, 0.98) !important;
  border-color: rgba(24, 58, 55, 0.18) !important;
}

.gradio-container .tab-nav button.selected,
.gradio-container .tab-nav button[aria-selected="true"] {
  color: white !important;
}

.gr-button-primary {
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  border: none !important;
  color: white !important;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        css=CSS,
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
              <p>Learn and explore the German EBM interactively.</p>
            </div>
            """
        )

        chapter_choices = browse_chapters()

        with gr.Tabs():
            with gr.Tab("Ask EBM"):
                with gr.Row():
                    question = gr.Textbox(
                        label="Ask a question",
                        placeholder="Was bedeutet EBM-Code 01100?",
                        lines=3,
                    )
                ask_btn = gr.Button("Ask EBM", variant="primary")
                answer = gr.Markdown()
                confidence = gr.Slider(0, 1, value=0, label="Retrieval confidence", interactive=False)
                ask_sources = gr.Markdown()
                ask_btn.click(ask_ebm, inputs=[question], outputs=[answer, ask_sources, confidence])

            with gr.Tab("Explain a Code"):
                code_input = gr.Textbox(label="EBM code", placeholder="01100")
                explain_btn = gr.Button("Explain", variant="primary")
                explanation = gr.Markdown()
                explanation_sources = gr.Markdown()
                explain_btn.click(explain_code, inputs=[code_input], outputs=[explanation, explanation_sources])

            with gr.Tab("Quiz Me"):
                quiz_prompt = gr.Markdown()
                quiz_answer = gr.Markdown(visible=False)
                quiz_sources = gr.Markdown()
                quiz_code = gr.State("")
                quiz_btn = gr.Button("Random code", variant="primary")
                reveal_btn = gr.Button("Reveal answer")
                quiz_btn.click(
                    quiz_me,
                    inputs=None,
                    outputs=[quiz_prompt, quiz_answer, quiz_code],
                )
                reveal_btn.click(
                    reveal_quiz_answer,
                    inputs=[quiz_code],
                    outputs=[quiz_answer, quiz_sources],
                )

            with gr.Tab("Explore EBM"):
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
                explore_sources = gr.Markdown()
                explore_chapters = gr.Markdown()
                search_btn.click(
                    explore_ebm,
                    inputs=[search_query, chapter],
                    outputs=[table, explore_sources, explore_chapters],
                )

    return demo


app = build_app()


if __name__ == "__main__":
    app.launch()
