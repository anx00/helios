from datetime import date

import httpx
import pytest
import web_server


@pytest.mark.asyncio
async def test_get_ensemble_edge_api_returns_edges(monkeypatch):
    captured = {}

    def fake_get_latest_prediction(station_id):
        assert station_id == "EGLC"
        return {
            "market": {
                "buckets": [
                    {"label": "11 C", "best_ask": 0.12},
                    {"label": "12 C", "price": 0.31},
                    {"label": "13 C", "yes_price": 0.44},
                    {"label": "skip-zero", "best_ask": 0.0},
                    {"label": "", "best_ask": 0.22},
                ]
            }
        }

    async def fake_fetch_ensemble_edge(station_id, market_buckets, target_date, min_edge_pct):
        captured["station_id"] = station_id
        captured["market_buckets"] = market_buckets
        captured["target_date"] = target_date
        captured["min_edge_pct"] = min_edge_pct
        return [
            {
                "label": "12°C",
                "market_price": 0.31,
                "ensemble_prob": 0.47,
                "edge_pct": 16.0,
                "signal": "BUY_YES",
            }
        ]

    monkeypatch.setattr(web_server, "get_latest_prediction", fake_get_latest_prediction)
    monkeypatch.setattr("collector.ensemble_fetcher.fetch_ensemble_edge", fake_fetch_ensemble_edge)

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(
            "/api/ensemble/edge/EGLC",
            params={"target_date": "2026-03-03", "min_edge": 9.5},
        )

    assert response.status_code == 200
    assert response.json() == {
        "station_id": "EGLC",
        "target_date": "2026-03-03",
        "min_edge_pct": 9.5,
        "market_buckets_count": 3,
        "edges": [
            {
                "label": "12°C",
                "market_price": 0.31,
                "ensemble_prob": 0.47,
                "edge_pct": 16.0,
                "signal": "BUY_YES",
            }
        ],
    }
    assert captured == {
        "station_id": "EGLC",
        "market_buckets": [
            {"label": "11 C", "price": 0.12},
            {"label": "12 C", "price": 0.31},
            {"label": "13 C", "price": 0.44},
        ],
        "target_date": date(2026, 3, 3),
        "min_edge_pct": 9.5,
    }


@pytest.mark.asyncio
async def test_get_ensemble_edge_api_returns_404_without_market(monkeypatch):
    monkeypatch.setattr(web_server, "get_latest_prediction", lambda station_id: None)

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/ensemble/edge/EGLC")

    assert response.status_code == 404
    assert response.json() == {
        "error": "No market data available",
        "station_id": "EGLC",
    }
