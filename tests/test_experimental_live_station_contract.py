import httpx
import pytest

import web_server


def _baseline_payload(station_id: str, target_day: int = 0, target_date=None, depth: int = 5):
    return {
        "station_id": station_id,
        "target_day": target_day,
        "target_date": "2026-03-19",
        "event_title": "Highest temperature in Madrid on March 19?",
        "event_slug": "highest-temperature-in-madrid-on-march-19",
        "total_volume": 1234.5,
        "market_status": {
            "is_mature": False,
            "winning_outcome": None,
            "max_probability": 0.41,
            "event_closed": False,
            "event_active": True,
        },
        "ws_connected": True,
        "ws_metrics": {"connected": True},
        "brackets": [
            {
                "name": "24-25F",
                "yes_price": 0.41,
                "no_price": 0.59,
                "ws_best_bid": 0.39,
                "ws_best_ask": 0.42,
                "ws_mid": 0.405,
                "ws_bid_depth": 10,
                "ws_ask_depth": 12,
            }
        ],
        "sentiment": "steady",
        "shifts": [],
        "trading": {
            "available": True,
            "station_id": station_id,
            "target_day": target_day,
            "target_date": "2026-03-19",
            "market_unit": "F",
            "model": {"source": "NOWCAST", "mean": 55.0, "sigma": 1.2, "confidence": 0.9},
            "best_terminal_trade": {
                "label": "24-25F",
                "best_side": "YES",
                "best_edge": 0.05,
                "selected_fair": 0.55,
                "selected_entry": 0.50,
                "recommendation": "LEAN_YES",
                "policy_reason": "ok",
            },
            "tactical_context": {"enabled": False},
            "terminal_opportunities": [],
        },
        "intraday_context": {
            "official": {
                "station_id": station_id,
                "temp_f": 54.0,
                "daily_max_f": 57.0,
                "obs_time_utc": "2026-03-19T10:00:00+00:00",
            }
        },
        "autotrader": {},
        "papertrader": {},
        "server_ts": 1.0,
    }


def _experimental_payload(station_id: str, target_day: int = 0):
    return {
        "station_id": station_id,
        "scope": "experimental",
        "target_day": target_day,
        "target_date": "2026-03-19",
        "generated_at_utc": "2026-03-19T10:00:00+00:00",
        "status": "ok",
        "missing_sections": [],
        "errors": {},
        "production": {
            "station_id": station_id,
            "target_day": target_day,
            "target_date": "2026-03-19",
        },
        "experimental": {
            "official": {
                "contract_role": "official",
                "station_id": station_id,
                "temp_f": 54.0,
                "daily_max_f": 57.0,
                "obs_time_utc": "2026-03-19T10:00:00+00:00",
                "source_age_s": 120.0,
                "staleness_s": 120.0,
            },
            "auxiliary": {
                "contract_role": "auxiliary",
                "pws_details": [],
                "pws_metrics": {"support": 3},
                "source_age_s": 90.0,
                "staleness_s": 90.0,
            },
            "forecast": {
                "contract_role": "forecast",
                "nowcast_distribution": {"tmax_mean_f": 55.0, "tmax_sigma_f": 1.2},
                "nowcast_state": {"sigma_f": 1.2, "max_so_far_f": 54.0},
                "source_age_s": 30.0,
                "staleness_s": 30.0,
            },
            "market": {
                "contract_role": "market",
                "event_title": "Highest temperature in Madrid on March 19?",
                "event_slug": "highest-temperature-in-madrid-on-march-19",
                "market_status": {"event_active": True, "event_closed": False},
                "ws_connected": True,
                "ws_metrics": {"connected": True},
                "brackets": [{"name": "24-25F", "yes_price": 0.41}],
                "source_age_s": 10.0,
                "staleness_s": 10.0,
            },
            "signal": {
                "contract_role": "signal",
                "available": True,
                "recommendation": "LEAN_YES",
                "fair_probability": 0.55,
                "edge_points": 5.0,
                "policy_reason": "ok",
                "source_age_s": 15.0,
                "staleness_s": 15.0,
            },
            "settlement_review": {
                "contract_role": "settlement_review",
                "status": "review_pending",
                "observed_max_f": None,
                "source_age_s": None,
                "staleness_s": None,
            },
            "provenance": {
                "contract_role": "provenance",
                "generated_at_utc": "2026-03-19T10:00:00+00:00",
                "source_age_s": 20.0,
                "staleness_s": 20.0,
            },
        },
        "comparison": {
            "official": "match",
            "market": "match",
            "signal": "partial",
            "provenance": "match",
        },
    }


@pytest.mark.asyncio
async def test_experimental_live_station_contract_exposes_role_separated_payload(monkeypatch):
    baseline_calls = []

    async def fake_baseline(station_id, target_day=0, target_date=None, depth=5):
        baseline_calls.append((station_id, target_day, target_date, depth))
        return _baseline_payload(station_id, target_day=target_day, target_date=target_date, depth=depth)

    async def fake_experimental(station_id, target_day=0, **kwargs):
        return _experimental_payload(station_id, target_day=target_day)

    monkeypatch.setattr(web_server, "get_polymarket_dashboard_data", fake_baseline)
    monkeypatch.setattr(web_server, "_build_experimental_live_station_payload", fake_experimental, raising=False)

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/experimental/live-station/KLGA", params={"target_day": 0})

    assert response.status_code == 200
    payload = response.json()

    assert payload["station_id"] == "KLGA"
    assert payload["status"] == "ok"
    assert payload["missing_sections"] == []
    assert payload["production"]["station_id"] == "KLGA"
    assert payload["experimental"]["official"]["contract_role"] == "official"
    assert payload["experimental"]["auxiliary"]["contract_role"] == "auxiliary"
    assert payload["experimental"]["forecast"]["contract_role"] == "forecast"
    assert payload["experimental"]["market"]["contract_role"] == "market"
    assert payload["experimental"]["signal"]["contract_role"] == "signal"
    assert payload["experimental"]["settlement_review"]["contract_role"] == "settlement_review"
    assert payload["experimental"]["provenance"]["contract_role"] == "provenance"
    assert payload["experimental"]["official"]["temp_f"] == 54.0
    assert payload["experimental"]["settlement_review"]["status"] == "review_pending"
    assert payload["experimental"]["official"]["temp_f"] != payload["experimental"]["settlement_review"]["observed_max_f"]
    assert payload["experimental"]["provenance"]["source_age_s"] == 20.0
    assert payload["experimental"]["provenance"]["staleness_s"] == 20.0
    assert payload["comparison"] == {
        "official": "match",
        "market": "match",
        "signal": "partial",
        "provenance": "match",
    }
    assert baseline_calls == [("KLGA", 0, None, 5)]
