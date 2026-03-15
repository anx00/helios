import pytest

import web_server


@pytest.mark.asyncio
async def test_get_market_data_reuses_hot_cache(monkeypatch):
    web_server._HOT_API_CACHE.clear()
    web_server._HOT_API_INFLIGHT.clear()

    calls = {"fetch_event_for_station_date": 0}

    async def fake_fetch_event_for_station_date(station_id, target_date):
        calls["fetch_event_for_station_date"] += 1
        return {
            "title": f"{station_id} event",
            "markets": [
                {
                    "groupItemTitle": "80+",
                    "volume": "123.4",
                    "outcomePrices": ["0.61", "0.39"],
                }
            ],
        }

    async def fake_fetch_event_data(_slug):
        raise AssertionError("fallback fetch should not run when cached event is available")

    monkeypatch.setattr(web_server, "fetch_event_for_station_date", fake_fetch_event_for_station_date)
    monkeypatch.setattr(web_server, "fetch_event_data", fake_fetch_event_data)

    first = await web_server.get_market_data("KLGA", target_day=0)
    second = await web_server.get_market_data("KLGA", target_day=0)

    assert first == second
    assert calls["fetch_event_for_station_date"] == 1
    assert first["event_title"] == "KLGA event"
    assert first["outcomes"][0]["probability"] == 0.61


@pytest.mark.asyncio
async def test_get_polymarket_dashboard_data_reuses_hot_cache(monkeypatch):
    web_server._HOT_API_CACHE.clear()
    web_server._HOT_API_INFLIGHT.clear()

    calls = {"build": 0}

    async def fake_build_polymarket_dashboard_data(station_id, target_day=0, target_date=None, depth=5):
        calls["build"] += 1
        return {
            "station_id": station_id,
            "target_day": target_day,
            "target_date": target_date,
            "depth": depth,
            "server_ts": 123.0,
            "brackets": [],
        }

    monkeypatch.setattr(web_server, "_build_polymarket_dashboard_data", fake_build_polymarket_dashboard_data)

    first = await web_server.get_polymarket_dashboard_data("KLGA", target_day=0, target_date=None, depth=5)
    second = await web_server.get_polymarket_dashboard_data("KLGA", target_day=0, target_date=None, depth=5)

    assert first == second
    assert calls["build"] == 1
