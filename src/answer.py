from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_ollama import ChatOllama

from src import config
from src.embeddings import build_embedding_model


STRICT_SYSTEM_PROMPT = """
You are AutoDefect Radar, answering questions about indexed NHTSA records.
Answer only from the retrieved context below.
If the context does not answer the question, say: "The indexed NHTSA data does not support an answer to that question."
State what the records report; do not diagnose, infer root cause, or give repair, legal, or safety advice.
Include evidence IDs for factual claims, such as ODI 11745320 or NHTSA Campaign 18V661000.

Context:
{context}

Answer the user's question accurately and concisely.
"""


RELAXED_SYSTEM_PROMPT = """
You are AutoDefect Radar, answering questions about indexed NHTSA records.
Use the retrieved context below as the evidence base.
Component labels, titles, summaries, and metadata can all be evidence.
If one retrieved complaint or recall matches the issue asked about, say it exists.
If the context is not enough, say what evidence is missing.
State what the records report; do not diagnose, infer root cause, or give repair, legal, or safety advice.
Include evidence IDs for factual claims, such as ODI 11745320 or NHTSA Campaign 18V661000.

Context:
{context}

Answer the user's question accurately and concisely.
"""


class AnswerGenerationError(RuntimeError):
    """Raised when the local language model cannot generate an answer."""


def get_vectorstore(
    vector_db_dir: Path = config.VECTOR_DB_DIR,
    embeddings: Embeddings | None = None,
) -> Chroma:
    if not vector_db_dir.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {vector_db_dir}. Run the ingest CLI first."
        )
    return Chroma(
        persist_directory=str(vector_db_dir),
        embedding_function=embeddings or build_embedding_model(),
        collection_name=config.COLLECTION_NAME,
    )


def fetch_context(
    question: str,
    k: int = config.DEFAULT_RETRIEVAL_K,
    vector_db_dir: Path = config.VECTOR_DB_DIR,
    embeddings: Embeddings | None = None,
) -> list[Document]:
    vectorstore = get_vectorstore(vector_db_dir=vector_db_dir, embeddings=embeddings)
    return vectorstore.similarity_search(question, k=k)


def answer_question(
    question: str,
    history: list[dict] | None = None,
    k: int = config.DEFAULT_RETRIEVAL_K,
    vector_db_dir: Path = config.VECTOR_DB_DIR,
    model: str = config.LLM_MODEL,
    embeddings: Embeddings | None = None,
    strict_prompt: bool | None = None,
) -> tuple[str, list[Document]]:
    docs = fetch_context(question, k=k, vector_db_dir=vector_db_dir, embeddings=embeddings)
    print(f"Retrieved {len(docs)} context chunks for question: {question!r}")
    if not docs:
        return (
            "The indexed NHTSA data does not support an answer to that question.",
            docs,
        )

    context = format_context(docs)
    print(f"Retrieved context")
    messages = [SystemMessage(content=build_system_prompt(context, strict_prompt))]
    if history:
        messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    print(f"LLM messages")
    llm = ChatOllama(temperature=0, model=model)
    try:
        print(f"Invoking Ollama model {model!r}...")
        response = llm.invoke(messages)
        print(f"LLM response: {response.content}")
    except Exception as exc:
        raise AnswerGenerationError(
            f"Could not generate an answer with Ollama model {model!r}. "
            "Make sure Ollama is running and the model is installed."
        ) from exc
    return ensure_citations(response.content, docs), docs


def build_system_prompt(context: str, strict_prompt: bool | None = None) -> str:
    if strict_prompt is None:
        strict_prompt = config.STRICT_ANSWER_PROMPT
    prompt = STRICT_SYSTEM_PROMPT if strict_prompt else RELAXED_SYSTEM_PROMPT
    return prompt.format(context=context)


def format_context(docs: list[Document]) -> str:
    chunks = []
    for index, doc in enumerate(docs, start=1):
        source_type = doc.metadata.get("source_type", "unknown")
        evidence_id = doc.metadata.get("evidence_id", "unknown")
        source = doc.metadata.get("source", "unknown")
        chunks.append(
            "\n".join(
                [
                    f"Context Chunk {index}",
                    f"Source Type: {source_type}",
                    f"Evidence ID: {evidence_id}",
                    f"Source File: {source}",
                    doc.page_content,
                ]
            )
        )
    return "\n\n---\n\n".join(chunks)


def ensure_citations(answer: str, docs: list[Document]) -> str:
    citations = []
    for doc in docs:
        evidence_id = str(doc.metadata.get("evidence_id", "")).strip()
        source_type = str(doc.metadata.get("source_type", "")).strip()
        if not evidence_id:
            continue
        if source_type == "recall":
            citation = f"NHTSA Campaign {evidence_id}"
        elif source_type == "complaint":
            citation = f"ODI {evidence_id}"
        else:
            citation = evidence_id
        if citation not in citations:
            citations.append(citation)

    if (
        not citations
        or _is_unsupported_answer(answer)
        or any(citation in answer for citation in citations)
    ):
        return answer
    return answer.rstrip() + "\n\nEvidence used: " + "; ".join(citations) + "."


def _is_unsupported_answer(answer: str) -> bool:
    lowered = answer.lower()
    unsupported_phrases = [
        "does not support an answer",
        "does not contain any information",
        "cannot provide a response",
        "cannot answer",
        "don't know",
        "do not know",
    ]
    return any(phrase in lowered for phrase in unsupported_phrases)
