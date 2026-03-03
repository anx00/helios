from datetime import date

import httpx
import pytest

import database
import web_server


class _FakeWorld:
    def get_snapshot(self):
        return {
            "official": {
                "LFPG": {
                    "temp_c": 9.8,
                    "obs_time_utc": "2026-03-04T09:00:00+00:00",
                    "report_type": "METAR",
                }
            },
            "pws_details": {
                "LFPG": [
                    {
                        "station_id": "IPARIS1",
                        "station_name": "Paris Stable",
                        "source": "SYNOPTIC",
                        "temp_c": 10.0,
                        "valid": True,
                        "age_minutes": 5.0,
                        "learning_weight_predictive": 1.5,
                        "distance_km": 1.1,
                    }
                ]
            },
            "pws_metrics": {"LFPG": {"weighted_support": 10.5}},
        }


@pytest.mark.asyncio
async def test_prediction_api_returns_probability_payload(monkeypatch):
    monkeypatch.setattr(
        web_server,
        "_resolve_station_market_date",
        lambda station_id, target_day=0, target_date=None: (date(2026, 3, 4), 0),
    )
    monkeypatch.setattr(web_server, "_prediction_observed_max_f", lambda station_id, target_date_str: 54.5)
    monkeypatch.setattr(
        web_server,
        "_prediction_feature_summary",
        lambda station_id: {"available": True, "env_key_count": 0, "env_keys": [], "empty_env": True, "qc_present": False, "staleness_key_count": 0},
    )

    async def fake_external_sources(_station_id: str):
        return {
            "station_id": "LFPG",
            "station_name": "Paris Charles de Gaulle",
            "timezone": "Europe/Paris",
            "preferred_unit": "C",
            "generated_at_utc": "2026-03-04T09:00:00+00:00",
            "generated_at_local": "2026-03-04T10:00:00+01:00",
            "sources": [
                {
                    "source": "wunderground",
                    "display_name": "Wunderground",
                    "status": "ok",
                    "historical_day": {"confirmed": True, "kind": "observed", "max_temp_c": 8.4},
                    "realtime": {"temp_c": 8.8, "as_of_local": "2026-03-04T10:00:00+01:00"},
                    "forecast_hourly": {"points": [
                        {"timestamp_local": "2026-03-04T12:00:00+01:00", "temp_c": 10.4},
                        {"timestamp_local": "2026-03-04T15:00:00+01:00", "temp_c": 11.2},
                    ]},
                    "notes": [],
                    "errors": [],
                }
            ],
        }

    async def fake_market_payload(station_id: str, target_day: int = 0, target_date: str | None = None, depth: int = 3):
        return {
            "station_id": station_id,
            "target_date": target_date,
            "event_title": "Paris market",
            "event_slug": "paris-market",
            "total_volume": 2500.0,
            "ws_connected": True,
            "market_status": {"event_active": True},
            "brackets": [{"name": "11 C", "yes_price": 0.31, "volume": 900.0}],
            "trading": {
                "model": {
                    "source": "NOWCAST",
                    "mean": 11.1,
                    "sigma": 0.7,
                    "confidence": 0.72,
                    "top_label": "11 C",
                    "top_label_probability": 0.33,
                    "top_labels": [{"label": "11 C", "probability": 0.33}],
                },
                "policy": {"headline": "1 idea passes live policy"},
                "best_terminal_trade": {
                    "label": "11 C",
                    "best_side": "YES",
                    "selected_entry": 0.31,
                    "selected_fair": 0.39,
                    "best_edge": 0.08,
                    "edge_points": 8.0,
                    "recommendation": "BUY_YES",
                    "policy_reason": "Top bucket is underpriced.",
                },
                "best_tactical_trade": {
                    "label": "11 C",
                    "best_side": "YES",
                    "selected_entry": 0.31,
                    "selected_fair": 0.39,
                    "best_edge": 0.08,
                    "edge_points": 8.0,
                    "recommendation": "BUY_YES",
                    "policy_reason": "Top bucket is underpriced.",
                },
                "all_terminal_opportunities": [
                    {
                        "label": "11 C",
                        "fair_yes": 0.39,
                        "yes_entry": 0.31,
                        "best_side": "YES",
                        "recommendation": "BUY_YES",
                    }
                ],
                "modules": {"next_official": {"available": True, "direction": "UP", "expected_market": 10.8, "minutes_to_next": 28.0}},
            },
        }

    async def fake_pws_history(_station_id: str, window_days: int = 14):
        return {
            "station_id": "LFPG",
            "available_dates": ["2026-03-01", "2026-03-02", "2026-03-03"],
            "selected_dates": ["2026-03-01", "2026-03-02", "2026-03-03"],
            "window_mode": "all_available",
            "window_day_count": 3,
            "summary": {"day_count": 3, "unique_station_count": 1, "audit_row_total": 9},
            "stations": [
                {
                    "station_id": "IPARIS1",
                    "station_name": "Paris Stable",
                    "source": "SYNOPTIC",
                    "distance_km": 1.1,
                    "raw_updates_total": 160,
                    "next_metar_total": 9,
                    "next_metar_hit_rate": 0.78,
                    "next_metar_mae_c": 0.41,
                    "current_metar_mae_c": 0.66,
                    "audit_hit_rate_within_1_0_c": 0.8,
                    "browser_day_count": 3,
                    "audit_day_count": 3,
                }
            ],
        }

    monkeypatch.setattr("collector.external_sources_fetcher.fetch_external_sources_snapshot", fake_external_sources)
    monkeypatch.setattr(web_server, "get_polymarket_dashboard_data", fake_market_payload)
    monkeypatch.setattr(web_server, "_prediction_pws_history_summary", fake_pws_history)
    monkeypatch.setattr("core.world.get_world", lambda: _FakeWorld())
    monkeypatch.setattr(
        database,
        "get_latest_prediction_for_date",
        lambda station_id, target_date: {
            "timestamp": "2026-03-04T08:55:00+00:00",
            "final_prediction_f": 52.3,
            "hrrr_max_raw_f": 51.0,
            "physics_adjustment_f": 1.2,
            "delta_weight": 0.85,
            "physics_reason": "Advection +1.2, Cloud -0.4",
        },
    )

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/prediction/LFPG", params={"target_day": 0})

    assert response.status_code == 200
    payload = response.json()
    assert payload["station_id"] == "LFPG"
    assert payload["prior_model"]["consensus_max"] == pytest.approx(11.2, rel=1e-6)
    assert payload["market_model"]["best_terminal_trade"]["label"] == "11 C"
    assert payload["intraday_model"]["next_official"]["direction"] == "UP"
    assert any(item["title"] == "Historical forecast snapshots" for item in payload["data_inventory"]["missing"])
    assert payload["feature_summary"]["available"] is True


@pytest.mark.asyncio
async def test_prediction_api_gracefully_surfaces_missing_market(monkeypatch):
    async def fake_pws_history_empty(station_id: str, window_days: int = 14):
        return {
            "station_id": station_id,
            "available_dates": [],
            "selected_dates": [],
            "summary": {},
            "stations": [],
            "window_mode": "none",
            "window_day_count": 0,
        }

    monkeypatch.setattr(
        web_server,
        "_resolve_station_market_date",
        lambda station_id, target_day=0, target_date=None: (date(2026, 3, 6), 2),
    )
    monkeypatch.setattr(web_server, "_prediction_feature_summary", lambda station_id: {"available": False, "env_key_count": 0, "env_keys": [], "empty_env": True, "qc_present": False, "staleness_key_count": 0})
    monkeypatch.setattr(web_server, "_prediction_pws_history_summary", fake_pws_history_empty)
    monkeypatch.setattr("collector.external_sources_fetcher.fetch_external_sources_snapshot", lambda station_id: fake_external_sources_for_missing_market())
    monkeypatch.setattr(web_server, "get_polymarket_dashboard_data", lambda station_id, target_day=0, target_date=None, depth=3: fake_market_missing())
    monkeypatch.setattr(database, "get_latest_prediction_for_date", lambda station_id, target_date: None)
    monkeypatch.setattr(web_server, "get_latest_prediction", lambda station_id: None)
    monkeypatch.setattr("core.world.get_world", lambda: _FakeWorld())

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/prediction/LFPG", params={"target_day": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_model"]["available"] is False
    assert payload["intraday_model"]["available"] is False
    assert payload["intraday_model"]["reason"] == "The intraday layer only activates for today."


async def fake_external_sources_for_missing_market():
    return {
        "station_id": "LFPG",
        "station_name": "Paris Charles de Gaulle",
        "timezone": "Europe/Paris",
        "preferred_unit": "C",
        "generated_at_utc": "2026-03-04T09:00:00+00:00",
        "generated_at_local": "2026-03-04T10:00:00+01:00",
        "sources": [
            {
                "source": "wunderground",
                "display_name": "Wunderground",
                "status": "ok",
                "historical_day": {"confirmed": True, "kind": "observed", "max_temp_c": 8.4},
                "realtime": {"temp_c": 8.8, "as_of_local": "2026-03-04T10:00:00+01:00"},
                "forecast_hourly": {"points": [{"timestamp_local": "2026-03-06T15:00:00+01:00", "temp_c": 12.4}]},
                "notes": [],
                "errors": [],
            }
        ],
    }


async def fake_market_missing():
    return {"error": "Market not found", "event_slug": "missing-market"}
