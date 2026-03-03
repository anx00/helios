from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import collector.external_sources_fetcher as fetcher


def test_extract_longest_json_array_prefers_longest_match():
    html = """
    {"temperature":[1,2]}
    {"temperature":[3,4,5,6]}
    """

    assert fetcher._extract_longest_json_array(html, "temperature") == [3, 4, 5, 6]


def test_build_wunderground_forecast_points_filters_and_converts_metric_rows():
    html = """
    {"validTimeLocal":["2026-03-02T16:00:00-0500","2026-03-02T18:00:00-0500","2026-03-04T23:00:00-0500","2026-03-05T00:00:00-0500"],
     "temperature":[10,11,12,13],
     "wxPhraseLong":["Past","Next","Edge","Too far"]}
    """
    station_tz = ZoneInfo("America/New_York")
    now_local = datetime(2026, 3, 2, 17, 30, tzinfo=station_tz)

    rows = fetcher._build_wunderground_forecast_points(
        html=html,
        station_tz=station_tz,
        now_local=now_local,
        default_use_metric=True,
    )

    assert [row["time_local"] for row in rows] == [
        "2026-03-02 18:00",
        "2026-03-04 23:00",
    ]
    assert rows[0]["temp_f"] == 51.8
    assert rows[0]["conditions"] == "Next"


def test_build_open_meteo_source_from_payload_splits_history_and_forecast():
    payload = {
        "timezone": "America/New_York",
        "current": {
            "time": "2026-03-02T17:15",
            "temperature_2m": 5.0,
        },
        "hourly": {
            "time": [
                "2026-03-02T16:00",
                "2026-03-02T17:00",
                "2026-03-02T18:00",
                "2026-03-03T00:00",
            ],
            "temperature_2m": [4.0, 6.0, 8.0, 10.0],
        },
    }

    source = fetcher._build_open_meteo_source_from_payload(
        "KLGA",
        payload,
        now_local=datetime(2026, 3, 2, 17, 15, tzinfo=ZoneInfo("America/New_York")),
    )

    assert source["source"] == "open_meteo"
    assert source["historical_day"]["confirmed"] is False
    assert [row["time_local"] for row in source["historical_day"]["points"]] == [
        "2026-03-02 16:00",
        "2026-03-02 17:00",
    ]
    assert [row["time_local"] for row in source["forecast_hourly"]["points"]] == [
        "2026-03-02 18:00",
        "2026-03-03 00:00",
    ]
    assert source["historical_day"]["max_temp_f"] == 42.8
    assert source["historical_day"]["max_temp_time_local"] == "2026-03-02T17:00:00-05:00"
    assert source["realtime"]["temp_f"] == 41.0


def test_build_noaa_forecast_points_filters_horizon():
    station_tz = ZoneInfo("America/New_York")
    now_local = datetime(2026, 3, 2, 17, 30, tzinfo=station_tz)
    periods = [
        {"startTime": "2026-03-02T17:00:00-05:00", "temperature": 31, "temperatureUnit": "F", "shortForecast": "Past"},
        {"startTime": "2026-03-02T18:00:00-05:00", "temperature": 32, "temperatureUnit": "F", "shortForecast": "Soon"},
        {"startTime": "2026-03-04T23:00:00-05:00", "temperature": 5, "temperatureUnit": "C", "shortForecast": "Later"},
        {"startTime": "2026-03-05T01:00:00-05:00", "temperature": 33, "temperatureUnit": "F", "shortForecast": "Too far"},
    ]

    rows = fetcher._build_noaa_forecast_points(
        periods,
        station_tz=station_tz,
        now_local=now_local,
    )

    assert [row["time_local"] for row in rows] == [
        "2026-03-02 18:00",
        "2026-03-04 23:00",
    ]
    assert rows[0]["temp_f"] == 32.0
    assert rows[1]["temp_f"] == 41.0


@pytest.mark.asyncio
async def test_fetch_accuweather_source_returns_disabled_scaffold(monkeypatch):
    monkeypatch.setattr(fetcher, "ACCUWEATHER_ENABLED", True)
    monkeypatch.setattr(fetcher, "ACCUWEATHER_API_KEY", "")

    source = await fetcher.fetch_accuweather_source("KLGA")

    assert source["source"] == "accuweather"
    assert source["status"] == "disabled"
    assert source["setup"]["requires_env"] == "ACCUWEATHER_API_KEY"
    assert any("Scaffold only" in note for note in source["notes"])


@pytest.mark.asyncio
async def test_fetch_open_meteo_source_returns_friendly_429_error(monkeypatch):
    fetcher._OPEN_METEO_SOURCE_CACHE.clear()

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.request = httpx.Request("GET", fetcher.OPEN_METEO_URL)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None):
            return httpx.Response(
                429,
                request=self.request,
                headers={"Retry-After": "0"},
            )

    async def fake_sleep(_delay: float):
        return None

    monkeypatch.setattr(fetcher.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(fetcher.asyncio, "sleep", fake_sleep)

    with pytest.raises(RuntimeError) as excinfo:
        await fetcher.fetch_open_meteo_source("LFPG")

    assert str(excinfo.value) == "Open-Meteo rate limited the request (HTTP 429)."
    assert "api.open-meteo.com" not in str(excinfo.value)
    assert "Too Many Requests" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_fetch_external_sources_snapshot_returns_partial_error_source(monkeypatch):
    async def fake_wunderground_source(_station_id: str):
        raise RuntimeError("WU down")

    async def fake_open_meteo_source(_station_id: str):
        return {
            "source": "open_meteo",
            "display_name": "Open-Meteo",
            "status": "ok",
            "historical_day": {"kind": "modeled", "confirmed": False, "published_through_local": None, "max_temp_c": None, "max_temp_f": None, "max_temp_time_local": None, "points": []},
            "realtime": {"kind": "modeled_current", "as_of_local": None, "temp_c": 1.0, "temp_f": 33.8},
            "forecast_hourly": {"kind": "forecast", "horizon_days": 3, "points": []},
            "notes": [],
            "errors": [],
        }

    async def fake_noaa_source(_station_id: str):
        return {
            "source": "noaa_nws",
            "display_name": "NOAA/NWS",
            "status": "ok",
            "historical_day": {"kind": "observed", "confirmed": True, "published_through_local": None, "max_temp_c": None, "max_temp_f": None, "max_temp_time_local": None, "points": []},
            "realtime": {"kind": "observed_current", "as_of_local": None, "temp_c": 1.0, "temp_f": 33.8},
            "forecast_hourly": {"kind": "forecast", "horizon_days": 3, "points": []},
            "notes": [],
            "errors": [],
        }

    async def fake_accuweather_source(_station_id: str):
        return {
            "source": "accuweather",
            "display_name": "AccuWeather",
            "status": "disabled",
            "historical_day": {"kind": None, "confirmed": False, "published_through_local": None, "max_temp_c": None, "max_temp_f": None, "max_temp_time_local": None, "points": []},
            "realtime": {"kind": None, "as_of_local": None, "temp_c": None, "temp_f": None},
            "forecast_hourly": {"kind": None, "horizon_days": 3, "points": []},
            "notes": ["Scaffold only"],
            "errors": ["Missing ACCUWEATHER_API_KEY."],
            "setup": {"requires_env": "ACCUWEATHER_API_KEY"},
        }

    monkeypatch.setattr(fetcher, "fetch_wunderground_source", fake_wunderground_source)
    monkeypatch.setattr(fetcher, "fetch_open_meteo_source", fake_open_meteo_source)
    monkeypatch.setattr(fetcher, "fetch_noaa_source", fake_noaa_source)
    monkeypatch.setattr(fetcher, "fetch_accuweather_source", fake_accuweather_source)

    snapshot = await fetcher.fetch_external_sources_snapshot("KLGA")

    assert snapshot["station_id"] == "KLGA"
    assert snapshot["preferred_unit"] == "F"
    assert [source["source"] for source in snapshot["sources"]] == ["wunderground", "open_meteo", "noaa_nws", "accuweather"]
    assert snapshot["sources"][0]["status"] == "error"
    assert snapshot["sources"][0]["errors"] == ["WU down"]
    assert snapshot["sources"][1]["status"] == "ok"
    assert snapshot["sources"][2]["status"] == "ok"
    assert snapshot["sources"][3]["status"] == "disabled"


@pytest.mark.asyncio
async def test_fetch_external_sources_snapshot_uses_station_native_unit(monkeypatch):
    async def fake_wunderground_source(_station_id: str):
        return {
            "source": "wunderground",
            "display_name": "Wunderground",
            "status": "ok",
            "historical_day": {"kind": "observed", "confirmed": True, "published_through_local": None, "max_temp_c": 8.0, "max_temp_f": 46.4, "max_temp_time_local": None, "points": []},
            "realtime": {"kind": "observed_current", "as_of_local": None, "temp_c": 7.0, "temp_f": 44.6},
            "forecast_hourly": {"kind": "forecast", "horizon_days": 3, "points": []},
            "notes": [],
            "errors": [],
        }

    async def fake_open_meteo_source(_station_id: str):
        return {
            "source": "open_meteo",
            "display_name": "Open-Meteo",
            "status": "ok",
            "historical_day": {"kind": "modeled", "confirmed": False, "published_through_local": None, "max_temp_c": 8.0, "max_temp_f": 46.4, "max_temp_time_local": None, "points": []},
            "realtime": {"kind": "modeled_current", "as_of_local": None, "temp_c": 7.0, "temp_f": 44.6},
            "forecast_hourly": {"kind": "forecast", "horizon_days": 3, "points": []},
            "notes": [],
            "errors": [],
        }

    async def fake_accuweather_source(_station_id: str):
        return {
            "source": "accuweather",
            "display_name": "AccuWeather",
            "status": "disabled",
            "historical_day": {"kind": None, "confirmed": False, "published_through_local": None, "max_temp_c": None, "max_temp_f": None, "max_temp_time_local": None, "points": []},
            "realtime": {"kind": None, "as_of_local": None, "temp_c": None, "temp_f": None},
            "forecast_hourly": {"kind": None, "horizon_days": 3, "points": []},
            "notes": [],
            "errors": ["Missing ACCUWEATHER_API_KEY."],
            "setup": {"requires_env": "ACCUWEATHER_API_KEY"},
        }

    monkeypatch.setattr(fetcher, "fetch_wunderground_source", fake_wunderground_source)
    monkeypatch.setattr(fetcher, "fetch_open_meteo_source", fake_open_meteo_source)
    monkeypatch.setattr(fetcher, "fetch_accuweather_source", fake_accuweather_source)

    snapshot = await fetcher.fetch_external_sources_snapshot("LFPG")

    assert snapshot["station_id"] == "LFPG"
    assert snapshot["preferred_unit"] == "C"
    assert [source["source"] for source in snapshot["sources"]] == ["wunderground", "open_meteo", "accuweather"]
