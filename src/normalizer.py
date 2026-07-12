from __future__ import annotations

import re
from typing import Any

from src.models import NormalizedDocument, VehicleQuery, VehicleRecords


VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE)


def normalize_records(records: VehicleRecords) -> list[NormalizedDocument]:
    documents: list[NormalizedDocument] = []
    documents.extend(normalize_recall(item, records.query) for item in records.recalls)
    documents.extend(normalize_complaint(item, records.query) for item in records.complaints)
    return documents


def normalize_recall(raw: dict[str, Any], query: VehicleQuery) -> NormalizedDocument:
    campaign = str(raw.get("NHTSACampaignNumber") or "unknown_campaign")
    component = _clean(raw.get("Component"))
    title = f"NHTSA Recall {campaign}: {component or 'Unknown component'}"
    content = "\n".join(
        [
            f"# {title}",
            "",
            f"Vehicle: {query.label}",
            "Source Type: Recall",
            f"Evidence ID: NHTSA Campaign {campaign}",
            f"Component: {component or 'Unknown'}",
            f"Report Received Date: {_clean(raw.get('ReportReceivedDate')) or 'Unknown'}",
            f"Manufacturer: {_clean(raw.get('Manufacturer')) or 'Unknown'}",
            f"Park It Advisory: {_bool_text(raw.get('parkIt'))}",
            f"Park Outside Advisory: {_bool_text(raw.get('parkOutSide'))}",
            f"Over The Air Update: {_bool_text(raw.get('overTheAirUpdate'))}",
            "",
            "## Summary",
            _sanitize_text(raw.get("Summary")),
            "",
            "## Consequence",
            _sanitize_text(raw.get("Consequence")),
            "",
            "## Remedy",
            _sanitize_text(raw.get("Remedy")),
            "",
            "## Notes",
            _sanitize_text(raw.get("Notes")),
        ]
    )
    return NormalizedDocument(
        source_type="recall",
        evidence_id=campaign,
        title=title,
        content=content,
        metadata={
            "source_type": "recall",
            "evidence_id": campaign,
            "component": component,
            "vehicle": query.label,
        },
    )


def normalize_complaint(raw: dict[str, Any], query: VehicleQuery) -> NormalizedDocument:
    odi = str(raw.get("odiNumber") or "unknown_odi")
    components = _clean(raw.get("components"))
    title = f"NHTSA Complaint ODI {odi}: {components or 'Unknown component'}"
    products = _format_products(raw.get("products", []))
    vin_present = "Yes, redacted" if raw.get("vin") else "No"
    content = "\n".join(
        [
            f"# {title}",
            "",
            f"Vehicle: {query.label}",
            "Source Type: Complaint",
            f"Evidence ID: ODI {odi}",
            f"Components: {components or 'Unknown'}",
            f"Incident Date: {_clean(raw.get('dateOfIncident')) or 'Unknown'}",
            f"Complaint Filed Date: {_clean(raw.get('dateComplaintFiled')) or 'Unknown'}",
            f"Manufacturer: {_clean(raw.get('manufacturer')) or 'Unknown'}",
            f"Crash Reported: {_bool_text(raw.get('crash'))}",
            f"Fire Reported: {_bool_text(raw.get('fire'))}",
            f"Number Of Injuries: {raw.get('numberOfInjuries', 0)}",
            f"Number Of Deaths: {raw.get('numberOfDeaths', 0)}",
            f"VIN Present In Source: {vin_present}",
            "",
            "## Complaint Summary",
            _sanitize_text(raw.get("summary"), redactions=[str(raw.get("vin") or "")]),
            "",
            "## Products",
            products,
        ]
    )
    return NormalizedDocument(
        source_type="complaint",
        evidence_id=odi,
        title=title,
        content=content,
        metadata={
            "source_type": "complaint",
            "evidence_id": odi,
            "component": components,
            "vehicle": query.label,
        },
    )


def _format_products(products: list[dict[str, Any]]) -> str:
    if not products:
        return "No product details provided."
    lines = []
    for product in products:
        fields = [
            _clean(product.get("type")),
            _clean(product.get("productYear")),
            _clean(product.get("productMake")),
            _clean(product.get("productModel")),
            _clean(product.get("size")),
        ]
        lines.append("- " + " ".join(field for field in fields if field))
    return "\n".join(lines)


def _sanitize_text(value: Any, redactions: list[str] | None = None) -> str:
    text = _clean(value) or "Not provided."
    for redaction in redactions or []:
        if redaction:
            text = text.replace(redaction, "[VIN REDACTED]")
    return VIN_RE.sub("[VIN REDACTED]", text)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").strip()


def _bool_text(value: Any) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"
