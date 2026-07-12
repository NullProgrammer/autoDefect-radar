from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import gradio as gr
from langchain_core.documents import Document

from src import config
from src.answer import AnswerGenerationError, answer_question, format_context


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AutoDefect Radar") as demo:
        gr.Markdown(
            "# AutoDefect Radar\n"
            "Ask against the local NHTSA vector index and inspect the exact request, "
            "answer, retrieved sources, and retrieved context."
        )

        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="Request",
                    lines=4,
                    placeholder="Are there ABS module brake and traction control light complaints?",
                )
                with gr.Row():
                    k = gr.Slider(
                        label="Retrieved chunks",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=config.DEFAULT_RETRIEVAL_K,
                    )
                    llm_model = gr.Textbox(
                        label="Ollama model",
                        value=config.LLM_MODEL,
                    )
                ask = gr.Button("Ask", variant="primary")

            with gr.Column(scale=3):
                answer = gr.Markdown(label="Response")

        with gr.Accordion("Request payload", open=True):
            request_payload = gr.JSON(label="Request")

        with gr.Accordion("Retrieved sources", open=True):
            sources = gr.Dataframe(
                headers=["Type", "Evidence ID", "Source"],
                datatype=["str", "str", "str"],
                interactive=False,
                label="Sources",
            )

        with gr.Accordion("Retrieved context", open=False):
            context = gr.Textbox(
                label="Context",
                lines=16,
            )

        inputs = [question, k, llm_model]
        outputs = [request_payload, answer, sources, context]
        ask.click(fn=run_question, inputs=inputs, outputs=outputs)
        question.submit(fn=run_question, inputs=inputs, outputs=outputs)

        gr.Examples(
            examples=[
                ["What airbag issues appear?"],
                ["Are there ABS module complaints?"],
                ["Are there brake and traction control light complaints?"],
            ],
            inputs=question,
        )

    return demo


def run_question(
    question: str,
    k: int | float,
    llm_model: str,
) -> tuple[dict[str, Any], str, list[list[str]], str]:
    cleaned_question = question.strip()
    cleaned_model = llm_model.strip() or config.LLM_MODEL
    retrieval_k = int(k)
    request_payload = {
        "question": cleaned_question,
        "k": retrieval_k,
        "llm_model": cleaned_model,
    }

    if not cleaned_question:
        return request_payload, "Question must not be empty.", [], ""

    try:
        answer, docs = answer_question(
            cleaned_question,
            k=retrieval_k,
            model=cleaned_model,
        )
    except (AnswerGenerationError, FileNotFoundError, ValueError) as exc:
        return request_payload, f"error: {exc}", [], ""

    return request_payload, answer, source_rows(docs), format_context(docs)


def source_rows(docs: list[Document]) -> list[list[str]]:
    rows = []
    seen: set[tuple[str, str, str]] = set()
    for doc in docs:
        source_type = str(doc.metadata.get("source_type", "unknown"))
        evidence_id = str(doc.metadata.get("evidence_id", "unknown"))
        source = str(doc.metadata.get("source", "unknown"))
        key = (source_type, evidence_id, source)
        if key in seen:
            continue
        seen.add(key)
        rows.append([source_type, evidence_id, source])
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autodefect-radar-ui",
        description="Launch the AutoDefect Radar Gradio UI.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
