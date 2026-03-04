from datetime import date
from pathlib import Path

import pytest

import database
from collector.wunderground_fetcher import WundergroundObservation
from core.forecast_snapshot_service import (
    capture_station_future_snapshots,
    normalize_source_daily_forecast,
)


def _init_tmp_db(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "forecast_snapshot.sqlite"
    monkeypatch.setattr(database, "DATABASE_PATH", str(db_path))
    database.init_database()


def test_normalize_source_daily_forecast_open_meteo_extracts_target_date_max():
    payload = {
        "status": "ok",
        "realtime": {"as_of_local": "2026-03-04T20:00:00+00:00"},
        "forecast_hourly": {
            "points": [
                {"timestamp_local": "2026-03-05T09:00:00+00:00", "temp_f": 50.0, "temp_c": 10.0},
                {"timestamp_local": "2026-03-05T14:00:00+00:00", "temp_f": 57.4, "temp_c": 14.1},
                {"timestamp_local": "2026-03-05T16:00:00+00:00", "temp_f": 56.0, "temp_c": 13.3},
                {"timestamp_local": "2026-03-06T14:00:00+00:00", "temp_f": 61.0, "temp_c": 16.1},
            ]
        },
        "notes": ["modeled"],
    }

    normalized = normalize_source_daily_forecast(
        station_id="KLGA",
        source="OPEN_METEO",
        target_date="2026-03-05",
        target_day=1,
        payload=payload,
    )

    assert normalized["status"] == "ok"
    assert normalized["forecast_high_f"] == 57.4
    assert normalized["forecast_high_market"] == 57.4
    assert normalized["peak_hour_local"] == 14
    assert normalized["notes"] == ["modeled"]


@pytest.mark.asyncio
async def test_capture_station_future_snapshots_persists_wunderground_snapshot(monkeypatch, tmp_path):
    _init_tmp_db(monkeypatch, tmp_path)

    async def fake_wu(station_id, target_date=None):
        return WundergroundObservation(
            station_id=station_id,
            date=target_date or date(2026, 3, 5),
            high_temp_f=61.0,
            high_temp_time="14:00",
            current_temp_f=58.0,
            forecast_high_f=67.8,
            forecast_peak_hour_local=15,
        )

    async def fake_open_meteo(station_id):
        return {
            "status": "ok",
            "realtime": {"as_of_local": "2026-03-04T18:00:00-05:00"},
            "forecast_hourly": {"points": [{"timestamp_local": "2026-03-05T14:00:00-05:00", "temp_f": 66.2, "temp_c": 19.0}]},
            "notes": [],
        }

    async def fake_nbm(station_id, target_date=None):
        class Dummy:
            max_temp_f = 66.9
            min_temp_f = 50.0
            fetch_time = "2026-03-04T23:00:00+00:00"
            source = "nbm"
        return Dummy()

    async def fake_lamp(station_id):
        class Dummy:
            max_temp_f = 68.1
            fetch_time = "2026-03-04T23:05:00+00:00"
            valid_hours = 25
            source = "lamp"
            confidence = 0.7
        return Dummy()

    monkeypatch.setattr("core.forecast_snapshot_service.fetch_wunderground_max", fake_wu)
    monkeypatch.setattr("core.forecast_snapshot_service.fetch_open_meteo_source", fake_open_meteo)
    monkeypatch.setattr("core.forecast_snapshot_service.fetch_nbm", fake_nbm)
    monkeypatch.setattr("core.forecast_snapshot_service.fetch_lamp", fake_lamp)

    payload = await capture_station_future_snapshots("KLGA", target_days=(1,), persist=True)
    wu_row = next(row for row in payload["snapshots"] if row["source"] == "WUNDERGROUND")

    latest = database.get_latest_forecast_source_snapshot("KLGA", wu_row["target_date"], "WUNDERGROUND")
    assert latest is not None
    assert latest["forecast_high_f"] == 67.8
    assert latest["peak_hour_local"] == 15
