from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date

from src import config
from src.answer import AnswerGenerationError, answer_question
from src.ingest import ingest_knowledge_base
from src.kb import build_knowledge_base
from src.models import VehicleQuery
from src.nhtsa import NHTSAError, fetch_vehicle_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autodefect-radar",
        description="Explore public NHTSA recall and complaint data with local RAG.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Fetch NHTSA data and build the local vector index"
    )
    ingest_parser.add_argument("--make", default=config.DEFAULT_MAKE)
    ingest_parser.add_argument("--model", default=config.DEFAULT_MODEL)
    ingest_parser.add_argument(
        "--year", type=_model_year, default=config.DEFAULT_MODEL_YEAR
    )
    ingest_parser.add_argument(
        "--complaints-limit",
        type=_complaints_limit,
        default=config.DEFAULT_COMPLAINTS_LIMIT,
        help="Number of recent complaints to index; use -1 for all complaints",
    )
    ingest_parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached API JSON"
    )

    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question against the local vector index"
    )
    ask_parser.add_argument("question", type=_nonempty_text)
    ask_parser.add_argument("--k", type=_positive_int, default=config.DEFAULT_RETRIEVAL_K)

    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "ingest":
        query = VehicleQuery(make=args.make, model=args.model, model_year=args.year)
        limit = None if args.complaints_limit == -1 else args.complaints_limit
        records = fetch_vehicle_records(
            query, complaints_limit=limit, refresh=args.refresh
        )
        paths = build_knowledge_base(records)
        result = ingest_knowledge_base()
        print(
            f"Indexed {len(records.recalls)} recalls, "
            f"{len(records.complaints)} complaints, {len(paths)} documents, "
            f"and {result.chunk_count} chunks."
        )
        return 0

    answer, docs = answer_question(args.question, k=args.k)
    print("\nAnswer\n======")
    print(answer)
    print("\nRetrieved Sources\n=================")
    seen: set[tuple[object, object, object]] = set()
    for doc in docs:
        source_type = doc.metadata.get("source_type", "unknown")
        evidence_id = doc.metadata.get("evidence_id", "unknown")
        source = doc.metadata.get("source", "unknown")
        key = (source_type, evidence_id, source)
        if key in seen:
            continue
        seen.add(key)
        print(f"- {source_type}: {evidence_id} ({source})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (AnswerGenerationError, FileNotFoundError, NHTSAError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _model_year(value: str) -> int:
    parsed = int(value)
    if parsed < 1886 or parsed > date.today().year + 1:
        raise argparse.ArgumentTypeError(
            f"must be between 1886 and {date.today().year + 1}"
        )
    return parsed


def _complaints_limit(value: str) -> int:
    parsed = int(value)
    if parsed < -1:
        raise argparse.ArgumentTypeError("must be -1, zero, or a positive integer")
    return parsed


def _nonempty_text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
