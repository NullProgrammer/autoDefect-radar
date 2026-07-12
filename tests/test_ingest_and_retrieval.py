from pathlib import Path

from langchain_core.embeddings import Embeddings

from langchain_core.documents import Document

from src.answer import ensure_citations, fetch_context, should_answer_from_context
from src.ingest import ingest_knowledge_base
from src.kb import build_knowledge_base
from src.models import VehicleQuery, VehicleRecords


class KeywordEmbeddings(Embeddings):
    vocabulary = [
        "air",
        "bag",
        "airbag",
        "inflator",
        "passenger",
        "abs",
        "module",
        "brake",
        "traction",
        "pizza",
    ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower().replace("airbag", "air bag airbag")
        return [float(lowered.count(term)) for term in self.vocabulary]


def test_ingestion_preserves_metadata_and_retrieves_expected_sources(tmp_path: Path):
    records = VehicleRecords(
        query=VehicleQuery(),
        recalls=[
            {
                "NHTSACampaignNumber": "18V661000",
                "Component": "AIR BAGS:FRONTAL:PASSENGER SIDE:INFLATOR MODULE",
                "ReportReceivedDate": "27/09/2018",
                "Manufacturer": "Honda",
                "Summary": "The passenger frontal air bag inflator may explode.",
                "Consequence": "Metal fragments may strike occupants.",
                "Remedy": "Dealers will replace the passenger frontal air bag inflator.",
                "Notes": "Owner notification details.",
                "parkIt": False,
                "parkOutSide": False,
                "overTheAirUpdate": False,
            }
        ],
        complaints=[
            {
                "odiNumber": 11681044,
                "manufacturer": "Honda",
                "crash": False,
                "fire": False,
                "numberOfInjuries": 0,
                "numberOfDeaths": 0,
                "dateOfIncident": "06/27/2025",
                "dateComplaintFiled": "08/16/2025",
                "vin": "1HGCP3F89CA123456",
                "components": "SERVICE BRAKES",
                "summary": "The VSA ABS module failed and the brake and traction control lights stayed on.",
                "products": [],
            }
        ],
    )

    kb_dir = tmp_path / "knowledge-base"
    vector_dir = tmp_path / "vector_db"
    paths = build_knowledge_base(records, output_dir=kb_dir)
    result = ingest_knowledge_base(
        knowledge_base_dir=kb_dir,
        vector_db_dir=vector_dir,
        embeddings=KeywordEmbeddings(),
    )

    assert len(paths) == 2
    assert result.document_count == 2
    assert result.chunk_count > 0

    airbag_docs = fetch_context(
        "passenger frontal air bag inflator",
        k=2,
        vector_db_dir=vector_dir,
        embeddings=KeywordEmbeddings(),
    )
    assert any(
        doc.metadata["source_type"] == "recall" and doc.metadata["evidence_id"] == "18V661000"
        for doc in airbag_docs
    )

    abs_docs = fetch_context(
        "ABS module brake and traction control lights",
        k=2,
        vector_db_dir=vector_dir,
        embeddings=KeywordEmbeddings(),
    )
    assert any(
        doc.metadata["source_type"] == "complaint" and doc.metadata["evidence_id"] == "11681044"
        for doc in abs_docs
    )

    assert should_answer_from_context("best pizza in town", abs_docs) is False


def test_ensure_citations_appends_missing_evidence_ids():
    docs = [
        Document(page_content="Recall text", metadata={"source_type": "recall", "evidence_id": "18V661000"}),
        Document(page_content="Complaint text", metadata={"source_type": "complaint", "evidence_id": "11681044"}),
    ]

    answer = ensure_citations("The data mentions airbag and brake issues.", docs)

    assert "NHTSA Campaign 18V661000" in answer
    assert "ODI 11681044" in answer
