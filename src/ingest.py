from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.embeddings import build_embedding_model


@dataclass(frozen=True)
class IngestionResult:
    vectorstore: Chroma
    document_count: int
    chunk_count: int


def load_documents(knowledge_base_dir: Path = config.KNOWLEDGE_BASE_DIR) -> list[Document]:
    if not knowledge_base_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {knowledge_base_dir}. Run the ingest CLI first."
        )

    documents: list[Document] = []
    for folder in sorted(path for path in knowledge_base_dir.iterdir() if path.is_dir()):
        for source_path in sorted(folder.rglob("*.md")):
            documents.append(
                Document(
                    page_content=source_path.read_text(encoding="utf-8"),
                    metadata={
                        "source": str(source_path),
                        "source_type": folder.name,
                        "evidence_id": _evidence_id_from_path(source_path),
                    },
                )
            )

    if not documents:
        raise ValueError(f"No markdown documents found under {knowledge_base_dir}")
    return documents


def create_chunks(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def create_embeddings(
    chunks: list[Document],
    vector_db_dir: Path = config.VECTOR_DB_DIR,
    embeddings: Embeddings | None = None,
) -> Chroma:
    if vector_db_dir.exists():
        shutil.rmtree(vector_db_dir)
    vector_db_dir.mkdir(parents=True, exist_ok=True)

    embedding_function = embeddings or build_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=str(vector_db_dir),
        collection_name=config.COLLECTION_NAME,
    )
    return vectorstore


def ingest_knowledge_base(
    knowledge_base_dir: Path = config.KNOWLEDGE_BASE_DIR,
    vector_db_dir: Path = config.VECTOR_DB_DIR,
    embeddings: Embeddings | None = None,
) -> IngestionResult:
    documents = load_documents(knowledge_base_dir)
    chunks = create_chunks(documents)
    vectorstore = create_embeddings(chunks, vector_db_dir=vector_db_dir, embeddings=embeddings)
    return IngestionResult(
        vectorstore=vectorstore,
        document_count=len(documents),
        chunk_count=len(chunks),
    )


def _evidence_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("recall_"):
        return stem.removeprefix("recall_").upper()
    if stem.startswith("complaint_"):
        return stem.removeprefix("complaint_")
    return stem
