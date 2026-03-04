import httpx
import pytest

import database
import web_server


@pytest.mark.asyncio
async def test_probability_lab_board_api_is_stable_when_one_station_fails(monkeypatch):
    async def fake_station_payload(station_id, *, target_day, lookback_hours=36, include_detail=True):
        if station_id == "KATL":
            raise RuntimeError("boom")
        return {
            "card": {
                "station_id": station_id,
                "station_name": station_id,
                "target_date": "2026-03-05",
                "horizon": "DAY_AHEAD",
                "market": {"event_title": "Test", "top_label": "48-49 F", "top_price": 0.31, "volume": 2000.0, "status": "OPEN"},
                "model": {"source": "DAYAHEAD_FUSION", "mean": 48.9, "sigma": 1.6, "confidence": 0.79, "top_label": "48-49 F", "top_probability": 0.34, "source_count": 4, "source_spread": 1.1, "notes": []},
                "actionable": {"available": station_id == "KLGA", "label": "48-49 F", "side": "YES", "recommendation": "LEAN_YES", "edge_points": 5.2, "entry_price": 0.28, "fair_prob": 0.332, "reason": "ok"},
                "source_strip": [{"source": "WUNDERGROUND", "status": "ok", "forecast_high_market": 48.0}],
                "reality": {},
                "pws": {"top_profiles": []},
                "_sort": {"actionable_rank": 2, "actionable_edge": 5.2, "confidence": 0.79, "volume": 2000.0},
            }
        }

    monkeypatch.setattr(web_server, "_build_probability_lab_station_payload", fake_station_payload)
    monkeypatch.setattr(web_server, "get_active_stations", lambda: {"KLGA": object(), "KATL": object()})

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/probability-lab/board", params={"target_day": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_day"] == 1
    assert len(payload["cards"]) == 2
    assert payload["cards"][0]["station_id"] == "KLGA"
    assert payload["cards"][1]["market"]["status"] == "ERROR"


@pytest.mark.asyncio
async def test_probability_lab_station_api_returns_detail_payload(monkeypatch):
    async def fake_station_payload(station_id, *, target_day, lookback_hours=36, include_detail=True):
        return {
            "detail": {
                "station_id": station_id,
                "target_day": target_day,
                "target_date": "2026-03-05",
                "summary": {"station_name": "New York LaGuardia", "horizon": "DAY_AHEAD", "model": {"source": "DAYAHEAD_FUSION"}},
                "source_detail": [],
                "source_history": {"lookback_hours": lookback_hours, "history_warming_up": True, "series": []},
                "bracket_ladder": [],
                "pws_profiles": [],
                "tactical": None,
                "reality": {},
                "calibration": {"available": False},
            }
        }

    monkeypatch.setattr(web_server, "_build_probability_lab_station_payload", fake_station_payload)

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/probability-lab/station/KLGA", params={"target_day": 1, "lookback_hours": 48})

    assert response.status_code == 200
    payload = response.json()
    assert payload["station_id"] == "KLGA"
    assert payload["source_history"]["lookback_hours"] == 48
    assert payload["source_history"]["history_warming_up"] is True
    assert payload["calibration"]["available"] is False


@pytest.mark.asyncio
async def test_probability_lab_html_route_marks_active_page():
    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/probability-lab")

    assert response.status_code == 200
    assert "Probability Lab" in response.text
    assert "probability-lab" in response.text


@pytest.mark.asyncio
async def test_build_probability_lab_calibration_uses_persisted_state(monkeypatch):
    persisted = {
        "station_id": "KLGA",
        "computed_at_utc": "2026-03-04T20:00:00+00:00",
        "available": True,
        "warming_up": False,
        "samples": 4,
        "mae_market": 1.1,
        "bias_market": -0.2,
        "rows": [{"target_date": "2026-03-03"}],
        "sources": {"WUNDERGROUND": {"samples": 4, "mae_market": 0.7, "bias_market": -0.1}},
    }

    monkeypatch.setattr(web_server, "_PROBABILITY_LAB_CALIBRATION_CACHE", {})
    monkeypatch.setattr(database, "get_probability_lab_calibration_state", lambda station_id: dict(persisted))
    monkeypatch.setattr(
        database,
        "get_recent_forecast_target_dates",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not rebuild from history")),
    )

    payload = await web_server._build_probability_lab_calibration("KLGA", use_cache=False, cache_only=True)

    assert payload["available"] is True
    assert payload["samples"] == 4
    assert payload["mae_market"] == 1.1
    assert payload["sources"]["WUNDERGROUND"]["mae_market"] == 0.7


@pytest.mark.asyncio
async def test_startup_event_registers_future_snapshot_loop(monkeypatch):
    created = []

    def fake_init_database():
        return None

    async def fake_pws_loop():
        return None

    async def fake_prediction_loop():
        return None

    async def fake_snapshot_loop():
        return None

    async def fake_websocket_loop():
        return None

    async def fake_nowcast_loop():
        return None

    async def fake_recorder_loop():
        return None

    async def fake_compactor_loop():
        return None

    async def fake_future_forecast_snapshot_loop():
        return None

    async def fake_probability_lab_calibration_loop():
        return None

    def fake_create_task(coro):
        created.append(coro.cr_code.co_name)
        coro.close()
        return None

    monkeypatch.setattr(database, "init_database", fake_init_database)
    monkeypatch.setattr("core.forecast_snapshot_service.future_forecast_snapshot_loop", fake_future_forecast_snapshot_loop)
    monkeypatch.setattr("core.telegram_bot.build_telegram_bot_from_env", lambda: None)
    monkeypatch.setattr(web_server.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(web_server, "probability_lab_calibration_loop", fake_probability_lab_calibration_loop)
    monkeypatch.setattr(web_server, "pws_loop", fake_pws_loop)
    monkeypatch.setattr(web_server, "prediction_loop", fake_prediction_loop)
    monkeypatch.setattr(web_server, "snapshot_loop", fake_snapshot_loop)
    monkeypatch.setattr(web_server, "websocket_loop", fake_websocket_loop)
    monkeypatch.setattr(web_server, "nowcast_loop", fake_nowcast_loop)
    monkeypatch.setattr(web_server, "recorder_loop", fake_recorder_loop)
    monkeypatch.setattr(web_server, "compactor_loop", fake_compactor_loop)

    await web_server.startup_event()

    assert "fake_future_forecast_snapshot_loop" in created
    assert "fake_probability_lab_calibration_loop" in created
    assert len(created) >= 8
