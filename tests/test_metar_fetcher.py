import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collector.metar_fetcher as metar_fetcher


def test_fetch_metar_waits_for_fresher_tds_when_json_is_stale(monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)

    async def fake_json(_station_id: str):
        return {
            "station_id": "EGLC",
            "obs_time": now - timedelta(minutes=45),
            "temp": 10.0,
            "dewp": 4.0,
            "wdir": 280,
            "wspd": 13,
            "clouds": [],
            "raw_ob": "METAR EGLC stale json",
            "report_type": "METAR",
            "source": "NOAA_JSON_API",
        }

    async def fake_tds(_station_id: str):
        await asyncio.sleep(0.02)
        return {
            "station_id": "EGLC",
            "obs_time": now,
            "temp": 11.0,
            "dewp": 5.0,
            "wdir": 270,
            "wspd": 13,
            "clouds": [],
            "raw_ob": "METAR EGLC fresh tds",
            "report_type": "METAR",
            "source": "NOAA_TDS_XML",
        }

    async def fake_tgftp(_station_id: str):
        await asyncio.sleep(0.02)
        return None

    async def fake_history(_station_id: str, hours: int = 1):
        return []

    monkeypatch.setattr(metar_fetcher, "fetch_metar_json", fake_json)
    monkeypatch.setattr(metar_fetcher, "fetch_metar_tds", fake_tds)
    monkeypatch.setattr(metar_fetcher, "fetch_metar_tgftp", fake_tgftp)
    monkeypatch.setattr(metar_fetcher, "fetch_metar_history", fake_history)
    monkeypatch.setattr(metar_fetcher, "_METAR_RACE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(metar_fetcher, "_METAR_RACE_STALE_JSON_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(metar_fetcher, "_METAR_JSON_STALE_AGE_MINUTES", 5.0)

    result = asyncio.run(metar_fetcher.fetch_metar("EGLC"))

    assert result is not None
    assert result.temp_c == 11.0
    assert result.observation_time == now
    assert result.racing_results["NOAA_JSON_API"] == 50.0
    assert result.racing_results["NOAA_TDS_XML"] == 51.8
