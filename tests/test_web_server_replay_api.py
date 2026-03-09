import httpx
import pytest

import web_server


class _FakeReplaySession:
    def __init__(self):
        self.calls = []

    def get_snapshot(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "state": {
                "session_id": "abc123",
                "date": "2026-02-06",
                "station_id": "KLGA",
                "mode": "light",
                "state": "paused",
                "clock": {"current_time": "2026-02-06T12:00:00+00:00"},
            },
            "events": {"before": [], "current": None, "after": []},
        }


class _FakeReplayEngine:
    def __init__(self, session):
        self._session = session

    def get_session(self, session_id):
        if session_id == "abc123":
            return self._session
        return None


def test_replay_market_panel_helper_maps_trading_payload():
    payload = {
        "brackets": [
            {
                "name": "40-41F",
                "yes_price": 0.44,
                "ws_mid": 0.46,
                "ws_best_bid": 0.45,
                "ws_best_ask": 0.47,
                "ws_bid_depth": 20,
                "ws_ask_depth": 18,
            }
        ]
    }

    panel = web_server._replay_market_panel_from_trading_payload(payload)

    assert panel == {
        "40-41F": {
            "mid": 0.46,
            "best_bid": 0.45,
            "best_ask": 0.47,
            "spread": pytest.approx(0.02),
            "bid_depth": 20,
            "ask_depth": 18,
        }
    }


@pytest.mark.asyncio
async def test_replay_snapshot_endpoint_returns_consolidated_payload(monkeypatch):
    from core import replay_engine

    fake_session = _FakeReplaySession()
    fake_engine = _FakeReplayEngine(fake_session)
    monkeypatch.setattr(replay_engine, "get_replay_engine", lambda: fake_engine)

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(
            "/api/v4/replay/session/abc123/snapshot",
            params={
                "heavy": 1,
                "events_n": 12,
                "max_points": 90,
                "top_n": 7,
                "trend_points": 300,
                "trend_mode": "5min",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"]["session_id"] == "abc123"
    assert "events" in payload
    assert fake_session.calls == [
        {
            "heavy": True,
            "events_n": 12,
            "max_points": 90,
            "top_n": 7,
            "trend_points": 300,
            "trend_mode": "5min",
        }
    ]


@pytest.mark.asyncio
async def test_replay_live_endpoint_returns_payload(monkeypatch):
    async def fake_payload(station_id):
        return {
            "station_id": station_id,
            "market_date": "2026-03-09",
            "target_date": "2026-03-09",
            "updated_at_utc": "2026-03-09T12:00:00+00:00",
            "market_panel": {"40-41F": {"mid": 0.44}},
        }

    monkeypatch.setattr(web_server, "_build_replay_live_payload", fake_payload)

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v4/replay/live/KLGA")

    assert response.status_code == 200
    assert response.json() == {
        "station_id": "KLGA",
        "market_date": "2026-03-09",
        "target_date": "2026-03-09",
        "updated_at_utc": "2026-03-09T12:00:00+00:00",
        "market_panel": {"40-41F": {"mid": 0.44}},
    }
