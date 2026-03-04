from pathlib import Path

import database


def _init_tmp_db(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "probability_lab.sqlite"
    monkeypatch.setattr(database, "DATABASE_PATH", str(db_path))
    database.init_database()
    return db_path


def test_forecast_source_snapshot_helpers_insert_latest_history_and_dedupe(monkeypatch, tmp_path):
    _init_tmp_db(monkeypatch, tmp_path)

    inserted = database.insert_forecast_source_snapshot(
        station_id="KLGA",
        target_date="2026-03-05",
        target_day=1,
        source="WUNDERGROUND",
        status="ok",
        market_unit="F",
        forecast_high_market=64.2,
        forecast_high_f=64.2,
        forecast_high_c=17.9,
        peak_hour_local=15,
        provider_updated_local="2026-03-04T18:00:00-05:00",
        notes=["fresh"],
        meta={"provider": "wu"},
        captured_at_utc="2026-03-04T22:00:00+00:00",
    )
    assert inserted is not None

    skipped = database.insert_forecast_source_snapshot(
        station_id="KLGA",
        target_date="2026-03-05",
        target_day=1,
        source="WUNDERGROUND",
        status="ok",
        market_unit="F",
        forecast_high_market=64.24,
        forecast_high_f=64.2,
        forecast_high_c=17.9,
        peak_hour_local=15,
        provider_updated_local="2026-03-04T18:05:00-05:00",
        notes=["same"],
        meta={"provider": "wu"},
        captured_at_utc="2026-03-04T22:10:00+00:00",
    )
    assert skipped is None

    changed = database.insert_forecast_source_snapshot(
        station_id="KLGA",
        target_date="2026-03-05",
        target_day=1,
        source="OPEN_METEO",
        status="ok",
        market_unit="F",
        forecast_high_market=63.0,
        forecast_high_f=63.0,
        forecast_high_c=17.2,
        peak_hour_local=14,
        provider_updated_local="2026-03-04T18:07:00-05:00",
        notes=["modeled"],
        meta={"provider": "om"},
        captured_at_utc="2026-03-04T22:12:00+00:00",
    )
    assert changed is not None

    latest_wu = database.get_latest_forecast_source_snapshot("KLGA", "2026-03-05", "WUNDERGROUND")
    assert latest_wu["forecast_high_market"] == 64.2
    assert latest_wu["peak_hour_local"] == 15
    assert latest_wu["notes"] == ["fresh"]
    assert latest_wu["meta"] == {"provider": "wu"}

    latest_all = database.get_latest_forecast_source_snapshots("KLGA", "2026-03-05")
    assert {row["source"] for row in latest_all} == {"WUNDERGROUND", "OPEN_METEO"}

    history = database.get_forecast_source_history("KLGA", "2026-03-05", lookback_hours=48)
    assert len(history) == 2
    assert history[0]["source"] == "WUNDERGROUND"
    assert history[1]["source"] == "OPEN_METEO"

    dates = database.get_recent_forecast_target_dates("KLGA", before_date="2026-03-06", limit=5)
    assert dates == ["2026-03-05"]


def test_probability_lab_calibration_state_helpers_roundtrip(monkeypatch, tmp_path):
    _init_tmp_db(monkeypatch, tmp_path)

    database.upsert_probability_lab_calibration_state(
        "KLGA",
        {
            "available": True,
            "warming_up": False,
            "samples": 3,
            "mae_market": 1.25,
            "bias_market": -0.4,
            "rows": [{"target_date": "2026-03-03", "model_error_market": -0.2}],
            "sources": {"WUNDERGROUND": {"samples": 3, "mae_market": 0.6, "bias_market": -0.1}},
        },
        computed_at_utc="2026-03-04T20:00:00+00:00",
    )

    state = database.get_probability_lab_calibration_state("KLGA")
    assert state is not None
    assert state["station_id"] == "KLGA"
    assert state["available"] is True
    assert state["warming_up"] is False
    assert state["samples"] == 3
    assert state["mae_market"] == 1.25
    assert state["bias_market"] == -0.4
    assert state["rows"] == [{"target_date": "2026-03-03", "model_error_market": -0.2}]
    assert state["sources"] == {"WUNDERGROUND": {"samples": 3, "mae_market": 0.6, "bias_market": -0.1}}

    database.upsert_probability_lab_calibration_state(
        "KLGA",
        {
            "available": False,
            "warming_up": True,
            "samples": 0,
            "mae_market": None,
            "bias_market": None,
            "rows": [],
            "sources": {},
        },
        computed_at_utc="2026-03-04T21:00:00+00:00",
    )

    updated = database.get_probability_lab_calibration_state("KLGA")
    assert updated is not None
    assert updated["computed_at_utc"] == "2026-03-04T21:00:00+00:00"
    assert updated["available"] is False
    assert updated["warming_up"] is True
    assert updated["samples"] == 0
