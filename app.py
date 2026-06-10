#!/usr/bin/env python3
"""
Gradio web interface — Yale Residential College Unofficial Guide
Run: python app.py
Open: http://localhost:7860
"""

from pathlib import Path

import chromadb
import gradio as gr
from sentence_transformers import SentenceTransformer

from query import ask

CHROMA_DIR = Path("documents/chroma")
COLLECTION = "yale_guide"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load embedding model and vector store once at startup
_model = SentenceTransformer(MODEL_NAME)
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_collection(COLLECTION)


def handle_query(question: str):
    if not question.strip():
        return "Please enter a question.", ""
    result = ask(question, model=_model, collection=_collection)
    sources_text = "\n".join(f"• {s}" for s in result["sources"])
    return result["answer"], sources_text


with gr.Blocks(title="Yale Residential College Unofficial Guide") as demo:
    gr.Markdown("# Yale Residential College Unofficial Guide")
    gr.Markdown(
        "Ask questions about Yale's 14 residential colleges — dining, housing, "
        "social life, butteries, housing lottery, transfer requests, and more. "
        "All answers are grounded in student journalism, reviews, and forums. "
        "If the sources don't cover your question, the system will say so."
    )

    with gr.Row():
        inp = gr.Textbox(
            label="Your question",
            placeholder="e.g. Which college has the best dining hall?",
            lines=2,
            scale=4,
        )

    btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        answer_box = gr.Textbox(label="Answer", lines=10, scale=3)
        sources_box = gr.Textbox(label="Retrieved from", lines=10, scale=2)

    gr.Examples(
        examples=[
            ["Which Yale residential college dining hall ranked last in the 2025 Yale Daily News dining study?"],
            ["What is a buttery and which colleges have the most popular ones?"],
            ["How many students requested residential college transfers in 2025?"],
            ["What is the most common reason students give for requesting a transfer?"],
            ["What is Yale's average SAT score for admitted students?"],
        ],
        inputs=inp,
    )

    btn.click(handle_query, inputs=inp, outputs=[answer_box, sources_box])
    inp.submit(handle_query, inputs=inp, outputs=[answer_box, sources_box])

if __name__ == "__main__":
    demo.launch()
