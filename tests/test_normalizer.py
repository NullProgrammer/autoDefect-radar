from src.models import VehicleQuery
from src.normalizer import normalize_complaint, normalize_recall


def test_normalize_recall_preserves_campaign_and_metadata():
    query = VehicleQuery()
    raw = {
        "NHTSACampaignNumber": "18V661000",
        "Component": "AIR BAGS:FRONTAL:PASSENGER SIDE:INFLATOR MODULE",
        "ReportReceivedDate": "27/09/2018",
        "Manufacturer": "Honda (American Honda Motor Co.)",
        "Summary": "Passenger frontal air bag inflator may explode.",
        "Consequence": "Sharp fragments may strike occupants.",
        "Remedy": "Dealers will replace the inflator.",
        "Notes": "Owners may contact NHTSA.",
        "parkIt": False,
        "parkOutSide": False,
        "overTheAirUpdate": False,
    }

    document = normalize_recall(raw, query)

    assert document.source_type == "recall"
    assert document.evidence_id == "18V661000"
    assert document.metadata["source_type"] == "recall"
    assert document.metadata["component"] == raw["Component"]
    assert "NHTSA Campaign 18V661000" in document.content
    assert "Passenger frontal air bag inflator" in document.content


def test_normalize_complaint_redacts_vin_but_keeps_odi():
    query = VehicleQuery()
    raw = {
        "odiNumber": 11745320,
        "manufacturer": "Honda (American Honda Motor Co.)",
        "crash": False,
        "fire": False,
        "numberOfInjuries": 0,
        "numberOfDeaths": 0,
        "dateOfIncident": "06/01/2026",
        "dateComplaintFiled": "06/19/2026",
        "vin": "1HGCP2F34CA123456",
        "components": "TIRES",
        "summary": "VIN 1HGCP2F34CA123456 developed vibration at highway speeds.",
        "products": [
            {
                "type": "Vehicle",
                "productYear": "2012",
                "productMake": "HONDA",
                "productModel": "ACCORD",
            }
        ],
    }

    document = normalize_complaint(raw, query)

    assert document.source_type == "complaint"
    assert document.evidence_id == "11745320"
    assert document.metadata["source_type"] == "complaint"
    assert "ODI 11745320" in document.content
    assert "1HGCP2F34CA123456" not in document.content
    assert "1HGCP2F34CA123456" not in str(document.metadata)
    assert "[VIN REDACTED]" in document.content
