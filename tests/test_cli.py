from types import SimpleNamespace

import pytest

from src import cli
from src.models import VehicleQuery, VehicleRecords


@pytest.mark.parametrize(
    "argv",
    [
        ["ask", "question", "--k", "0"],
        ["ingest", "--year", "1800"],
        ["ingest", "--complaints-limit", "-2"],
        ["ask", "   "],
    ],
)
def test_invalid_arguments_exit_with_usage_error(argv):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)

    assert exc_info.value.code == 2


def test_missing_index_has_actionable_error(monkeypatch, capsys):
    def missing_index(*args, **kwargs):
        raise FileNotFoundError("Vector DB not found. Run the ingest CLI first.")

    monkeypatch.setattr(cli, "answer_question", missing_index)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["ask", "What airbag issues appear?"])

    assert exc_info.value.code == 1
    assert "Run the ingest CLI first" in capsys.readouterr().err


def test_unavailable_ollama_has_actionable_error(monkeypatch, capsys):
    def unavailable_model(*args, **kwargs):
        raise cli.AnswerGenerationError(
            "Could not generate an answer. Make sure Ollama is running."
        )

    monkeypatch.setattr(cli, "answer_question", unavailable_model)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["ask", "What airbag issues appear?"])

    assert exc_info.value.code == 1
    assert "Make sure Ollama is running" in capsys.readouterr().err


def test_ask_passes_llm_model_override(monkeypatch, capsys):
    calls = []

    def fake_answer_question(*args, **kwargs):
        calls.append((args, kwargs))
        return "answer", []

    monkeypatch.setattr(cli, "answer_question", fake_answer_question)

    exit_code = cli.main(
        ["ask", "What brake complaints exist?", "--llm-model", "qwen2.5:7b"]
    )

    assert exit_code == 0
    assert calls == [
        (
            ("What brake complaints exist?",),
            {"k": cli.config.DEFAULT_RETRIEVAL_K, "model": "qwen2.5:7b"},
        )
    ]
    assert "answer" in capsys.readouterr().out


def test_ingest_reports_counts(monkeypatch, capsys):
    records = VehicleRecords(
        query=VehicleQuery(),
        recalls=[{"NHTSACampaignNumber": "18V661000"}],
        complaints=[{"odiNumber": 11745320}],
    )
    monkeypatch.setattr(cli, "fetch_vehicle_records", lambda *args, **kwargs: records)
    monkeypatch.setattr(cli, "build_knowledge_base", lambda *args, **kwargs: ["a", "b"])
    monkeypatch.setattr(
        cli,
        "ingest_knowledge_base",
        lambda: SimpleNamespace(document_count=2, chunk_count=4),
    )

    exit_code = cli.main(["ingest"])

    assert exit_code == 0
    assert "2 documents, and 4 chunks" in capsys.readouterr().out
