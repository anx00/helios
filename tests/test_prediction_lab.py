from datetime import date

import pytest

from core.prediction_lab import build_intraday_model, build_prior_model


def test_build_prior_model_filters_target_day_and_computes_consensus():
    payload = {
        "sources": [
            {
                "source": "wunderground",
                "display_name": "Wunderground",
                "status": "ok",
                "historical_day": {"confirmed": True, "kind": "observed", "max_temp_c": 9.0},
                "realtime": {"temp_c": 8.5, "as_of_local": "2026-03-04T08:00:00+01:00"},
                "forecast_hourly": {
                    "points": [
                        {"timestamp_local": "2026-03-05T12:00:00+01:00", "temp_c": 11.2},
                        {"timestamp_local": "2026-03-05T15:00:00+01:00", "temp_c": 12.0},
                        {"timestamp_local": "2026-03-06T15:00:00+01:00", "temp_c": 13.3},
                    ]
                },
                "notes": [],
                "errors": [],
            },
            {
                "source": "open_meteo",
                "display_name": "Open-Meteo",
                "status": "ok",
                "historical_day": {"confirmed": False, "kind": "modeled", "max_temp_c": 8.7},
                "realtime": {"temp_c": 8.1, "as_of_local": "2026-03-04T08:00:00+01:00"},
                "forecast_hourly": {
                    "points": [
                        {"timestamp_local": "2026-03-05T13:00:00+01:00", "temp_c": 10.8},
                        {"timestamp_local": "2026-03-05T16:00:00+01:00", "temp_c": 11.0},
                    ]
                },
                "notes": [],
                "errors": [],
            },
        ]
    }

    model = build_prior_model(
        station_id="LFPG",
        target_date=date(2026, 3, 5),
        external_sources=payload,
        legacy_prediction={"final_prediction_f": 53.6},
    )

    assert model["available"] is True
    assert model["source_count"] == 2
    assert model["consensus_max"] == pytest.approx(11.5, rel=1e-6)
    assert model["source_rows"][0]["target_hour_count"] == 2
    assert model["source_rows"][0]["target_day_max"] == pytest.approx(12.0, rel=1e-6)
    assert model["source_rows"][1]["target_day_max"] == pytest.approx(11.0, rel=1e-6)
    assert "forecast snapshots" in " ".join(model["notes"]).lower()


def test_build_intraday_model_dedupes_current_pws_and_uses_history_classification():
    pws_history_summary = {
        "window_mode": "all_available",
        "window_day_count": 6,
        "available_dates": ["2026-03-01", "2026-03-02", "2026-03-03"],
        "summary": {"day_count": 6, "unique_station_count": 2, "audit_row_total": 18},
        "stations": [
            {
                "station_id": "IPARIS1",
                "station_name": "Paris Stable",
                "source": "SYNOPTIC",
                "distance_km": 1.2,
                "raw_updates_total": 280,
                "next_metar_total": 12,
                "next_metar_hit_rate": 0.75,
                "next_metar_mae_c": 0.42,
                "current_metar_mae_c": 0.71,
                "audit_hit_rate_within_1_0_c": 0.78,
                "browser_day_count": 6,
                "audit_day_count": 6,
            },
            {
                "station_id": "IPARIS2",
                "station_name": "Paris Delay",
                "source": "WUNDERGROUND",
                "distance_km": 2.4,
                "raw_updates_total": 240,
                "next_metar_total": 11,
                "next_metar_hit_rate": 0.61,
                "next_metar_mae_c": 0.58,
                "current_metar_mae_c": 1.02,
                "audit_hit_rate_within_1_0_c": 0.63,
                "browser_day_count": 6,
                "audit_day_count": 6,
            },
        ],
    }

    intraday = build_intraday_model(
        station_id="LFPG",
        target_day=0,
        official={"temp_c": 10.0, "obs_time_utc": "2026-03-04T09:00:00+00:00", "report_type": "METAR"},
        pws_details=[
            {"station_id": "IPARIS1", "station_name": "Paris Stable", "source": "WUNDERGROUND", "temp_c": 10.1, "valid": True, "age_minutes": 9.0, "learning_weight_predictive": 1.1, "distance_km": 1.2},
            {"station_id": "IPARIS1", "station_name": "Paris Stable", "source": "SYNOPTIC", "temp_c": 10.2, "valid": True, "age_minutes": 4.0, "learning_weight_predictive": 1.6, "distance_km": 1.2},
            {"station_id": "IPARIS2", "station_name": "Paris Delay", "source": "WUNDERGROUND", "temp_c": 10.8, "valid": True, "age_minutes": 6.0, "learning_weight_predictive": 1.2, "distance_km": 2.4},
        ],
        pws_metrics={"weighted_support": 12.4},
        pws_history_summary=pws_history_summary,
        observed_max_f=54.5,
        trading_signal={"modules": {"next_official": {"available": True, "direction": "UP", "expected_market": 11.1, "minutes_to_next": 31.0}}},
    )

    assert intraday["available"] is True
    assert intraday["pws_now"]["station_count"] == 2
    assert intraday["pws_now"]["top_current_stations"][0]["station_id"] == "IPARIS1"
    assert intraday["pws_now"]["top_current_stations"][0]["classification"]["code"] == "stable"
    assert intraday["next_official"]["direction"] == "UP"
    assert intraday["historical_ranking"]["top_stations"][1]["classification"]["code"] == "delayed_but_useful"
