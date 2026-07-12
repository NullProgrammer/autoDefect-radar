import json

import pytest
import requests

from src.models import VehicleQuery
from src.nhtsa import NHTSAClient, NHTSAError


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_fetch_recalls_uses_existing_cache(tmp_path, monkeypatch):
    query = VehicleQuery()
    cache_path = tmp_path / f"{query.slug}_recalls.json"
    cache_path.write_text(
        json.dumps({"results": [{"NHTSACampaignNumber": "18V661000"}]}),
        encoding="utf-8",
    )

    def unexpected_request(*args, **kwargs):
        pytest.fail("cache hit should not make an HTTP request")

    monkeypatch.setattr(requests, "get", unexpected_request)

    records = NHTSAClient(cache_dir=tmp_path).fetch_recalls(query)

    assert records[0]["NHTSACampaignNumber"] == "18V661000"


def test_fetch_without_cache_reports_http_failure(tmp_path, monkeypatch):
    def failed_request(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", failed_request)

    with pytest.raises(NHTSAError, match="no cache exists"):
        NHTSAClient(cache_dir=tmp_path).fetch_recalls(VehicleQuery())


def test_refresh_falls_back_to_existing_cache(tmp_path, monkeypatch):
    query = VehicleQuery()
    cache_path = tmp_path / f"{query.slug}_complaints.json"
    cache_path.write_text(
        json.dumps({"results": [{"odiNumber": 11745320}]}), encoding="utf-8"
    )

    def failed_request(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", failed_request)

    records = NHTSAClient(cache_dir=tmp_path).fetch_complaints(
        query, refresh=True
    )

    assert records[0]["odiNumber"] == 11745320


def test_malformed_api_response_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse([]))

    with pytest.raises(NHTSAError, match="invalid response"):
        NHTSAClient(cache_dir=tmp_path).fetch_recalls(VehicleQuery(), refresh=True)


def test_malformed_results_are_rejected(tmp_path, monkeypatch):
    response = FakeResponse({"results": "not-a-list"})
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(NHTSAError, match="invalid recall records"):
        NHTSAClient(cache_dir=tmp_path).fetch_recalls(VehicleQuery(), refresh=True)
