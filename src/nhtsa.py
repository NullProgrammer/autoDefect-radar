from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from src import config
from src.models import VehicleQuery, VehicleRecords


NHTSA_BASE_URL = "https://api.nhtsa.gov"


class NHTSAError(RuntimeError):
    pass


class NHTSAClient:
    def __init__(self, cache_dir: Path = config.RAW_CACHE_DIR, timeout: int = 30):
        self.cache_dir = cache_dir
        self.timeout = timeout

    def fetch_recalls(self, query: VehicleQuery, refresh: bool = False) -> list[dict[str, Any]]:
        payload = self._get_json(
            endpoint="/recalls/recallsByVehicle",
            params=query.as_api_params(),
            cache_name=f"{query.slug}_recalls.json",
            refresh=refresh,
        )
        return _extract_results(payload, "recall")

    def fetch_complaints(
        self,
        query: VehicleQuery,
        complaints_limit: int | None = config.DEFAULT_COMPLAINTS_LIMIT,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            endpoint="/complaints/complaintsByVehicle",
            params=query.as_api_params(),
            cache_name=f"{query.slug}_complaints.json",
            refresh=refresh,
        )
        complaints = _extract_results(payload, "complaint")
        complaints = sorted(complaints, key=_complaint_sort_key, reverse=True)
        if complaints_limit is None:
            return complaints
        return complaints[:complaints_limit]

    def _get_json(
        self,
        endpoint: str,
        params: dict[str, str | int],
        cache_name: str,
        refresh: bool,
    ) -> dict[str, Any]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / cache_name

        if cache_path.exists() and not refresh:
            return _read_cache(cache_path)

        url = f"{NHTSA_BASE_URL}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            if cache_path.exists():
                return _read_cache(cache_path)
            raise NHTSAError(
                f"Could not fetch {url} and no cache exists at {cache_path}"
            ) from exc

        if not isinstance(payload, dict):
            raise NHTSAError(f"NHTSA returned an invalid response for {url}")

        try:
            cache_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError as exc:
            raise NHTSAError(f"Could not write NHTSA cache at {cache_path}") from exc
        return payload


def fetch_vehicle_records(
    vehicle_query: VehicleQuery | None = None,
    complaints_limit: int | None = config.DEFAULT_COMPLAINTS_LIMIT,
    refresh: bool = False,
    cache_dir: Path = config.RAW_CACHE_DIR,
) -> VehicleRecords:
    query = vehicle_query or VehicleQuery()
    client = NHTSAClient(cache_dir=cache_dir)
    recalls = client.fetch_recalls(query, refresh=refresh)
    complaints = client.fetch_complaints(
        query,
        complaints_limit=complaints_limit,
        refresh=refresh,
    )
    return VehicleRecords(query=query, recalls=recalls, complaints=complaints)


def _complaint_sort_key(complaint: dict[str, Any]) -> datetime:
    date_text = complaint.get("dateComplaintFiled") or complaint.get("dateOfIncident") or ""
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_text, fmt)
        except ValueError:
            continue
    return datetime.min


def _extract_results(payload: dict[str, Any], record_type: str) -> list[dict[str, Any]]:
    results = payload.get("results", [])
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise NHTSAError(f"NHTSA returned invalid {record_type} records")
    return results


def _read_cache(cache_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NHTSAError(f"Could not read NHTSA cache at {cache_path}") from exc
    if not isinstance(payload, dict):
        raise NHTSAError(f"NHTSA cache at {cache_path} contains an invalid response")
    return payload
