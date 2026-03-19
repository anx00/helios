from types import SimpleNamespace
import copy

import httpx
import pytest

import web_server


class _FakeDistribution:
    def __init__(self, payload):
        self._payload = dict(payload)

    def to_dict(self):
        return copy.deepcopy(self._payload)


class _FakeEngine:
    def __init__(self, distribution=None, state=None):
        self._distribution = distribution
        self._state = state or {}

    def generate_distribution(self):
        return self._distribution

    def get_state_snapshot(self):
        return copy.deepcopy(self._state)


class _FakeIntegration:
    def __init__(self, engine, distribution=None):
        self._engine = engine
        self._distribution = distribution

    def get_engine(self, station_id):
        return self._engine

    def get_distribution(self, station_id):
        return self._distribution


class _FakeWorld:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def get_snapshot(self):
        return copy.deepcopy(self._snapshot)


def _build_baseline_payload():
    official = {
        "station_id": "KLGA",
        "temp_f": 46.0,
        "temp_f_raw": 46.2,
        "temp_c": 7.8,
        "daily_max_f": 47.0,
        "obs_time_utc": "2026-03-19T12:00:00+00:00",
        "source_age_s": 12.3,
        "report_type": "METAR",
        "is_speci": False,
        "qc_passed": True,
    }
    trading = {
        "available": True,
        "station_id": "KLGA",
        "target_day": 0,
        "target_date": "2026-03-19",
        "market_unit": "F",
        "horizon": "INTRADAY",
        "model": {
            "source": "NOWCAST",
            "mean": 46.2,
            "sigma": 1.1,
            "confidence": 0.81,
            "top_label": "46 F",
            "top_label_probability": 0.34,
        },
        "policy": {"mode": "AUTO_GUARDRAILS", "headline": "1 ideas pass live policy"},
        "best_terminal_trade": {
            "label": "46 F",
            "best_side": "YES",
            "best_edge": 0.08,
            "recommendation": "LEAN_YES",
            "selected_fair": 0.34,
            "selected_entry": 0.26,
        },
        "best_tactical_trade": {
            "label": "46 F",
            "best_side": "YES",
            "best_edge": 0.07,
            "recommendation": "LEAN_YES",
            "selected_fair": 0.34,
            "selected_entry": 0.27,
        },
        "forecast_winner": {
            "label": "46 F",
            "model_probability": 0.34,
            "market_probability": 0.26,
            "edge_points": 8.0,
        },
        "terminal_opportunities": [],
        "tactical_context": {"enabled": True},
    }
    return {
        "station_id": "KLGA",
        "target_day": 0,
        "target_date": "2026-03-19",
        "event_title": "Highest temperature in New York City on March 19?",
        "event_slug": "highest-temperature-in-new-york-city-on-march-19",
        "total_volume": 12345.0,
        "market_status": {
            "is_mature": False,
            "winning_outcome": None,
            "max_probability": 0.42,
            "event_closed": False,
            "event_active": True,
        },
        "ws_connected": True,
        "ws_metrics": {"reconnects": 0},
        "brackets": [
            {
                "name": "46 F",
                "yes_price": 0.28,
                "ws_best_bid": 0.26,
                "ws_best_ask": 0.29,
                "ws_mid": 0.275,
                "volume": 3000.0,
            },
            {
                "name": "47 F",
                "yes_price": 0.18,
                "ws_best_bid": 0.17,
                "ws_best_ask": 0.19,
                "ws_mid": 0.18,
                "volume": 2500.0,
            },
        ],
        "sentiment": "neutral",
        "shifts": [],
        "trading": trading,
        "intraday_context": {
            "official": official,
            "pws_detail_count": 2,
            "has_pws_metrics": True,
        },
        "autotrader": {"running": False},
        "papertrader": {"running": False},
        "server_ts": 1710840000.0,
    }


def _patch_happy_path(monkeypatch):
    baseline = _build_baseline_payload()
    world_snapshot = {
        "official": {
            "KLGA": baseline["intraday_context"]["official"],
        },
        "aux": {
            "KLGA": {
                "station_id": "KLGA",
                "temp_f": 45.5,
                "temp_c": 7.5,
                "support": 6,
                "drift": -0.5,
                "source": "PWS_CLUSTER",
                "obs_time_utc": "2026-03-19T12:10:00+00:00",
                "source_age_s": 18.0,
            }
        },
        "pws_details": {
            "KLGA": [
                {"station_id": "PWS1", "temp_f": 45.0, "valid": True},
                {"station_id": "PWS2", "temp_f": 46.0, "valid": True},
            ]
        },
        "pws_metrics": {
            "KLGA": {"support": 2, "mad": 0.5},
        },
    }
    nowcast_distribution = _FakeDistribution(
        {
            "station_id": "KLGA",
            "target_date": "2026-03-19",
            "tmax_mean_f": 46.2,
            "tmax_sigma_f": 1.1,
            "confidence": 0.81,
        }
    )
    nowcast_state = {
        "station_id": "KLGA",
        "target_date": "2026-03-19",
        "max_so_far_f": 47.0,
        "sigma_f": 1.1,
        "last_update_utc": "2026-03-19T12:15:00+00:00",
    }
    review = SimpleNamespace(
        verification_status="verified",
        source="wunderground",
        high_temp_f=47.0,
        high_temp_time="14:00",
        current_temp_f=46.0,
        forecast_high_f=48.0,
        forecast_peak_hour_local=15,
        forecast_peak_temp_f=48.0,
        confidence="high",
        notes="ok",
        metar_max_f=47.0,
        wunderground_max_f=47.0,
        wunderground_current_f=46.0,
    )

    async def fake_dashboard_data(station_id, target_day=0, target_date=None, depth=5):
        assert station_id == "KLGA"
        assert target_date is None
        assert depth == 5
        return copy.deepcopy(baseline)

    def fake_build_trading_signal(**kwargs):
        return copy.deepcopy(baseline["trading"])

    monkeypatch.setattr(web_server, "get_polymarket_dashboard_data", fake_dashboard_data)
    async def fake_observed_max(station_id, target_date=None):
        return review

    monkeypatch.setattr(web_server, "build_trading_signal", fake_build_trading_signal)
    monkeypatch.setattr(web_server, "get_world", lambda: _FakeWorld(world_snapshot))
    monkeypatch.setattr(
        web_server,
        "get_nowcast_integration",
        lambda: _FakeIntegration(_FakeEngine(state=nowcast_state), nowcast_distribution),
    )
    monkeypatch.setattr(web_server, "get_observed_max", fake_observed_max)
    return baseline


@pytest.mark.asyncio
async def test_experimental_live_station_api_happy_path(monkeypatch):
    baseline = _patch_happy_path(monkeypatch)

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/experimental/live-station/KLGA", params={"target_day": 0, "depth": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["station_id"] == "KLGA"
    assert payload["scope"] == "experimental_live_station"
    assert payload["status"] == "ok"
    assert payload["experimental"]["settlement_review"]["status"] == "verified"
    assert payload["comparison"]["official"]["status"] == "match"
    assert payload["comparison"]["market"]["status"] == "match"
    assert payload["comparison"]["signal"]["status"] == "match"
    assert payload["comparison"]["provenance"]["status"] == "match"
    assert payload["production"]["station_id"] == baseline["station_id"]
    assert payload["experimental"]["official"]["station_id"] == "KLGA"


@pytest.mark.asyncio
async def test_experimental_live_station_api_rejects_invalid_station():
    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/experimental/live-station/ZZZZ")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid station"


@pytest.mark.asyncio
async def test_experimental_live_station_api_is_klga_only():
    other_station = next(station_id for station_id in web_server.STATIONS if station_id != "KLGA")
    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(f"/api/experimental/live-station/{other_station}")

    assert response.status_code == 404
    assert "KLGA-only" in response.json()["detail"]


@pytest.mark.asyncio
async def test_experimental_live_station_api_marks_partial_when_wu_and_nowcast_missing(monkeypatch):
    baseline = _build_baseline_payload()
    world_snapshot = {
        "official": {"KLGA": baseline["intraday_context"]["official"]},
        "aux": {"KLGA": None},
        "pws_details": {"KLGA": []},
        "pws_metrics": {"KLGA": {}},
    }

    async def fake_dashboard_data(station_id, target_day=0, target_date=None, depth=5):
        return copy.deepcopy(baseline)

    def fake_build_trading_signal(**kwargs):
        return {"available": False, "station_id": "KLGA", "target_day": 0, "target_date": "2026-03-19"}

    monkeypatch.setattr(web_server, "get_polymarket_dashboard_data", fake_dashboard_data)
    async def fake_observed_max(station_id, target_date=None):
        return None

    monkeypatch.setattr(web_server, "build_trading_signal", fake_build_trading_signal)
    monkeypatch.setattr(web_server, "get_world", lambda: _FakeWorld(world_snapshot))
    monkeypatch.setattr(
        web_server,
        "get_nowcast_integration",
        lambda: _FakeIntegration(_FakeEngine(state={"station_id": "KLGA", "last_update_utc": "2026-03-19T12:15:00+00:00"}), None),
    )
    monkeypatch.setattr(web_server, "get_observed_max", fake_observed_max)

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/experimental/live-station/KLGA", params={"target_day": 0, "depth": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "partial"
    assert "nowcast" in payload["missing_sections"]
    assert "wunderground" in payload["missing_sections"]
    assert payload["experimental"]["settlement_review"]["status"] == "review_pending"


@pytest.mark.asyncio
async def test_experimental_live_station_api_reuses_production_baseline_once(monkeypatch):
    baseline = _build_baseline_payload()
    call_count = {"count": 0}

    async def fake_dashboard_data(station_id, target_day=0, target_date=None, depth=5):
        call_count["count"] += 1
        return copy.deepcopy(baseline)

    def fake_build_trading_signal(**kwargs):
        return copy.deepcopy(baseline["trading"])

    world_snapshot = {
        "official": {"KLGA": baseline["intraday_context"]["official"]},
        "aux": {"KLGA": {}},
        "pws_details": {"KLGA": []},
        "pws_metrics": {"KLGA": {}},
    }
    review = SimpleNamespace(
        verification_status="verified",
        source="wunderground",
        high_temp_f=47.0,
        high_temp_time="14:00",
        current_temp_f=46.0,
        forecast_high_f=48.0,
        forecast_peak_hour_local=15,
        forecast_peak_temp_f=48.0,
        confidence="high",
        notes="ok",
        metar_max_f=47.0,
        wunderground_max_f=47.0,
        wunderground_current_f=46.0,
    )

    monkeypatch.setattr(web_server, "get_polymarket_dashboard_data", fake_dashboard_data)
    async def fake_observed_max(station_id, target_date=None):
        return review

    monkeypatch.setattr(web_server, "build_trading_signal", fake_build_trading_signal)
    monkeypatch.setattr(web_server, "get_world", lambda: _FakeWorld(world_snapshot))
    monkeypatch.setattr(
        web_server,
        "get_nowcast_integration",
        lambda: _FakeIntegration(_FakeEngine(state={"station_id": "KLGA", "last_update_utc": "2026-03-19T12:15:00+00:00"}), _FakeDistribution({"station_id": "KLGA"})),
    )
    monkeypatch.setattr(web_server, "get_observed_max", fake_observed_max)

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/experimental/live-station/KLGA", params={"target_day": 0, "depth": 5})

    assert response.status_code == 200
    assert call_count["count"] == 1
