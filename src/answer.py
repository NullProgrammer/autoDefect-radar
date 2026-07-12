from __future__ import annotations

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_ollama import ChatOllama

from src import config
from src.embeddings import build_embedding_model


SYSTEM_PROMPT = """
You are AutoDefect Radar, a careful assistant for learning RAG over NHTSA data.
You answer only from the retrieved context.
Every factual claim about a defect, recall, complaint, consequence, or remedy must cite the evidence ID from the context, such as NHTSA Campaign 18V661000 or ODI 11745320.
If the retrieved context does not support the answer, say: "The indexed NHTSA data does not support an answer to that question."
Do not provide safety, legal, repair, or diagnostic advice. Suggest checking official NHTSA/OEM sources for action.

Retrieved context:
{context}
"""

STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "and",
    "answer",
    "appear",
    "from",
    "have",
    "into",
    "issue",
    "issues",
    "more",
    "that",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}


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
) -> tuple[str, list[Document]]:
    docs = fetch_context(question, k=k, vector_db_dir=vector_db_dir, embeddings=embeddings)

    if not should_answer_from_context(question, docs):
        return (
            "The indexed NHTSA data does not support an answer to that question.",
            docs,
        )

    context = format_context(docs)
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
    if history:
        messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))

    llm = ChatOllama(temperature=0, model=model)
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        raise AnswerGenerationError(
            f"Could not generate an answer with Ollama model {model!r}. "
            "Make sure Ollama is running and the model is installed."
        ) from exc
    return ensure_citations(response.content, docs), docs


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


def should_answer_from_context(question: str, docs: list[Document]) -> bool:
    if not docs:
        return False
    question_terms = _important_terms(question)
    if not question_terms:
        return False
    context_text = " ".join(doc.page_content for doc in docs).lower()
    context_terms = _important_terms(context_text)
    return bool(question_terms & context_terms)


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

    if not citations or any(citation in answer for citation in citations):
        return answer
    return answer.rstrip() + "\n\nEvidence used: " + "; ".join(citations) + "."


def _important_terms(text: str) -> set[str]:
    normalized = text.lower().replace("airbag", "air bag")
    terms = set(re.findall(r"[a-z0-9]+", normalized))
    return {term for term in terms if len(term) >= 3 and term not in STOPWORDS}
