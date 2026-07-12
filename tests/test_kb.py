from src.kb import build_knowledge_base
from src.models import VehicleQuery, VehicleRecords


def test_build_knowledge_base_writes_markdown_without_vin(tmp_path):
    records = VehicleRecords(
        query=VehicleQuery(),
        recalls=[],
        complaints=[
            {
                "odiNumber": 11745320,
                "manufacturer": "Honda",
                "crash": False,
                "fire": False,
                "numberOfInjuries": 0,
                "numberOfDeaths": 0,
                "dateOfIncident": "06/01/2026",
                "dateComplaintFiled": "06/19/2026",
                "vin": "1HGCP2F34CA123456",
                "components": "TIRES",
                "summary": "VIN 1HGCP2F34CA123456 developed steering wheel vibration.",
                "products": [],
            }
        ],
    )

    paths = build_knowledge_base(records, output_dir=tmp_path / "kb")

    assert len(paths) == 1
    text = paths[0].read_text(encoding="utf-8")
    assert "ODI 11745320" in text
    assert "1HGCP2F34CA123456" not in text
    assert "VIN Present In Source: Yes, redacted" in text
