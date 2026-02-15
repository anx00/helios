import asyncio
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import market.polymarket_checker as checker


NYC = ZoneInfo("America/New_York")


def _event(
    *,
    closed: bool,
    active: bool,
    end_date: str,
    yes_prob: float = 0.99,
):
    return {
        "closed": closed,
        "active": active,
        "endDate": end_date,
        "markets": [
            {
                "groupItemTitle": "48-49°F",
                "outcomePrices": f"[\"{yes_prob}\", \"{1.0 - yes_prob}\"]",
            }
        ],
    }


def test_get_target_date_keeps_today_when_open_even_if_probability_is_high(monkeypatch):
    today = date(2026, 2, 10)
    tomorrow = today + timedelta(days=1)
    event_map = {
        today: _event(
            closed=False,
            active=True,
            end_date="2026-02-11T05:00:00Z",
            yes_prob=0.995,
        ),
        tomorrow: _event(
            closed=False,
            active=True,
            end_date="2026-02-12T05:00:00Z",
            yes_prob=0.60,
        ),
    }

    async def fake_fetch_event_for_station_date(_station: str, target: date):
        return event_map.get(target)

    async def fake_fetch_event_data(_slug: str):
        return None

    monkeypatch.setattr(checker, "fetch_event_for_station_date", fake_fetch_event_for_station_date)
    monkeypatch.setattr(checker, "fetch_event_data", fake_fetch_event_data)

    now_local = datetime(2026, 2, 10, 18, 0, tzinfo=NYC)
    resolved = asyncio.run(checker.get_target_date("KLGA", now_local))
    assert resolved == today


def test_get_target_date_rolls_to_tomorrow_when_today_market_is_closed(monkeypatch):
    today = date(2026, 2, 10)
    tomorrow = today + timedelta(days=1)
    event_map = {
        today: _event(
            closed=True,
            active=True,
            end_date="2026-02-10T05:00:00Z",
            yes_prob=0.999,
        ),
        tomorrow: _event(
            closed=False,
            active=True,
            end_date="2026-02-12T05:00:00Z",
            yes_prob=0.55,
        ),
    }

    async def fake_fetch_event_for_station_date(_station: str, target: date):
        return event_map.get(target)

    async def fake_fetch_event_data(_slug: str):
        return None

    monkeypatch.setattr(checker, "fetch_event_for_station_date", fake_fetch_event_for_station_date)
    monkeypatch.setattr(checker, "fetch_event_data", fake_fetch_event_data)

    now_local = datetime(2026, 2, 10, 23, 0, tzinfo=NYC)
    resolved = asyncio.run(checker.get_target_date("KLGA", now_local))
    assert resolved == tomorrow


def test_get_target_date_keeps_today_when_discovery_fails_for_both_days(monkeypatch):
    today = date(2026, 2, 10)

    async def fake_fetch_event_for_station_date(_station: str, _target: date):
        return None

    async def fake_fetch_event_data(_slug: str):
        return None

    monkeypatch.setattr(checker, "fetch_event_for_station_date", fake_fetch_event_for_station_date)
    monkeypatch.setattr(checker, "fetch_event_data", fake_fetch_event_data)

    now_local = datetime(2026, 2, 10, 12, 0, tzinfo=NYC)
    resolved = asyncio.run(checker.get_target_date("KLGA", now_local))
    assert resolved == today
