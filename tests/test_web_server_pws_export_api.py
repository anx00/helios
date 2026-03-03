import httpx
import pytest

from core import pws_browser
import web_server


@pytest.mark.asyncio
async def test_export_pws_city_api_returns_filtered_day_exports(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        pws_browser,
        "list_pws_browser_dates",
        lambda station_id: ["2026-02-10", "2026-02-08", "2026-02-09"],
    )
    monkeypatch.setattr(
        pws_browser,
        "build_pws_browser_payload",
        lambda station_id, date: {
            "date": date,
            "market_overview": {"raw_updates_total": 1, "table_rows_total": 1},
            "pws_stations": [],
        },
    )
    monkeypatch.setattr(
        pws_browser,
        "build_pws_audit_day_payload",
        lambda station_id, date: {
            "date": date,
            "row_count": 0,
            "stations": [],
        },
    )

    def fake_build_summary(station_id, day_exports, *, available_dates=None, selected_dates=None):
        captured["station_id"] = station_id
        captured["day_exports"] = day_exports
        captured["available_dates"] = available_dates
        captured["selected_dates"] = selected_dates
        return {
            "station_id": station_id,
            "market_station": {"station_id": station_id, "city": "New York"},
            "available_dates": list(available_dates or []),
            "selected_dates": list(selected_dates or []),
            "date_from": selected_dates[0] if selected_dates else None,
            "date_to": selected_dates[-1] if selected_dates else None,
            "summary": {"day_count": len(day_exports)},
            "stations": [],
        }

    monkeypatch.setattr(pws_browser, "build_pws_city_export_summary", fake_build_summary)

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(
            "/api/pws/export/city",
            params={
                "station_id": "KLGA",
                "date_from": "2026-02-09",
                "date_to": "2026-02-10",
                "hydrate_metar": "false",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "station_id": "KLGA",
        "market_station": {"station_id": "KLGA", "city": "New York"},
        "available_dates": ["2026-02-10", "2026-02-08", "2026-02-09"],
        "selected_dates": ["2026-02-09", "2026-02-10"],
        "date_from": "2026-02-09",
        "date_to": "2026-02-10",
        "summary": {"day_count": 2},
        "stations": [],
        "days": [
            {
                "date": "2026-02-09",
                "browser": {
                    "date": "2026-02-09",
                    "market_overview": {"raw_updates_total": 1, "table_rows_total": 1},
                    "pws_stations": [],
                },
                "audit": {
                    "date": "2026-02-09",
                    "row_count": 0,
                    "stations": [],
                },
            },
            {
                "date": "2026-02-10",
                "browser": {
                    "date": "2026-02-10",
                    "market_overview": {"raw_updates_total": 1, "table_rows_total": 1},
                    "pws_stations": [],
                },
                "audit": {
                    "date": "2026-02-10",
                    "row_count": 0,
                    "stations": [],
                },
            },
        ],
    }
    assert captured == {
        "station_id": "KLGA",
        "day_exports": [
            {
                "date": "2026-02-09",
                "browser": {
                    "date": "2026-02-09",
                    "market_overview": {"raw_updates_total": 1, "table_rows_total": 1},
                    "pws_stations": [],
                },
                "audit": {"date": "2026-02-09", "row_count": 0, "stations": []},
            },
            {
                "date": "2026-02-10",
                "browser": {
                    "date": "2026-02-10",
                    "market_overview": {"raw_updates_total": 1, "table_rows_total": 1},
                    "pws_stations": [],
                },
                "audit": {"date": "2026-02-10", "row_count": 0, "stations": []},
            },
        ],
        "available_dates": ["2026-02-10", "2026-02-08", "2026-02-09"],
        "selected_dates": ["2026-02-09", "2026-02-10"],
    }


@pytest.mark.asyncio
async def test_export_pws_city_api_returns_422_for_invalid_date_range(monkeypatch):
    monkeypatch.setattr(pws_browser, "list_pws_browser_dates", lambda station_id: ["2026-02-10"])

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(
            "/api/pws/export/city",
            params={
                "station_id": "KLGA",
                "date_from": "2026-02-11",
                "date_to": "2026-02-10",
                "hydrate_metar": "false",
            },
        )

    assert response.status_code == 422
    assert response.json() == {
        "error": "date_from must be less than or equal to date_to",
        "station_id": "KLGA",
        "date_from": "2026-02-11",
        "date_to": "2026-02-10",
        "available_dates": ["2026-02-10"],
    }
