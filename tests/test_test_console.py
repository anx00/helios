from pathlib import Path

import httpx
import pytest

import web_server


ROOT = Path(__file__).resolve().parents[1]
TEST_TEMPLATE = ROOT / "templates" / "test.html"
BASE_TEMPLATE = ROOT / "templates" / "base.html"


@pytest.mark.asyncio
async def test_test_console_html_route_is_available():
    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/test")

    assert response.status_code == 200
    assert "Test Console" in response.text
    assert "Official, pressure, market and signal" in response.text


@pytest.mark.asyncio
async def test_test_console_api_route_returns_payload(monkeypatch):
    async def fake_payload(station_id, *, target_day=0, target_date=None, depth=5):
        assert station_id == "KLGA"
        assert target_day == 0
        assert target_date is None
        assert depth == 5
        return {
            "station": {"station_id": station_id, "target_day": target_day, "market_unit": "F"},
            "meta": {"status": "ok", "missing_sections": [], "errors": {}},
            "summary": {"official_available": True, "pressure_available": True},
            "official": {"temp_f": 46.0},
            "pressure": {"station_count": 2},
            "market": {"top_bracket": "46 F"},
            "signal": {"recommendation": "LEAN_YES"},
            "settlement_review": {"status": "verified"},
            "forecast": {"available": True},
            "provenance": {"generated_at_utc": "2026-03-20T10:00:00+00:00"},
        }

    monkeypatch.setattr(web_server, "_build_test_console_payload", fake_payload)

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/test/KLGA", params={"target_day": 0, "depth": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["station"]["station_id"] == "KLGA"
    assert payload["market"]["top_bracket"] == "46 F"
    assert payload["signal"]["recommendation"] == "LEAN_YES"


def test_test_template_and_sidebar_are_wired():
    template_html = TEST_TEMPLATE.read_text(encoding="utf-8")
    base_html = BASE_TEMPLATE.read_text(encoding="utf-8")

    assert "/api/test/" in template_html
    assert 'id="test-shell"' in template_html
    assert "Settlement Review" in template_html
    assert 'href="/test"' in base_html
    assert "<span>Test</span>" in base_html
