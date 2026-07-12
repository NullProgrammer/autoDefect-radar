from langchain_core.documents import Document

from src import app


def test_build_demo_creates_gradio_blocks():
    demo = app.build_demo()

    assert demo.title == "AutoDefect Radar"


def test_run_question_returns_request_answer_sources_and_context(monkeypatch):
    docs = [
        Document(
            page_content="Complaint summary",
            metadata={
                "source_type": "complaint",
                "evidence_id": "11681044",
                "source": "complaint_11681044.md",
            },
        )
    ]
    calls = []

    def fake_answer_question(*args, **kwargs):
        calls.append((args, kwargs))
        return "answer", docs

    monkeypatch.setattr(app, "answer_question", fake_answer_question)

    request, answer, sources, context = app.run_question(
        " Are there ABS complaints? ",
        3,
        "test-model",
    )

    assert request == {
        "question": "Are there ABS complaints?",
        "k": 3,
        "llm_model": "test-model",
    }
    assert answer == "answer"
    assert sources == [["complaint", "11681044", "complaint_11681044.md"]]
    assert "Context Chunk 1" in context
    assert calls == [
        (
            ("Are there ABS complaints?",),
            {"k": 3, "model": "test-model"},
        )
    ]


def test_run_question_requires_nonempty_question(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("answer_question should not be called")

    monkeypatch.setattr(app, "answer_question", fail_if_called)

    request, answer, sources, context = app.run_question("  ", 2, "")

    assert request == {
        "question": "",
        "k": 2,
        "llm_model": app.config.LLM_MODEL,
    }
    assert answer == "Question must not be empty."
    assert sources == []
    assert context == ""
