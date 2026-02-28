from datetime import datetime
from zoneinfo import ZoneInfo

from core import pws_browser


UTC = ZoneInfo("UTC")


class _FakeLearningStore:
    lead_min_minutes = 5
    lead_max_minutes = 45

    def get_station_profile(self, market_station_id: str, pws_station_id: str, source: str = "UNKNOWN") -> dict:
        return {
            "station_id": pws_station_id,
            "source": source,
            "next_metar_score": 77.0,
            "next_metar_prob_within_1_0_c": 0.75,
            "lead_avg_minutes": 24.0,
            "lead_bucket_summaries": {
                "15-30": {
                    "bucket": "15-30",
                    "count": 4,
                    "avg_lead_minutes": 22.0,
                    "bias_c": 0.1,
                    "mae_c": 0.4,
                    "rmse_c": 0.5,
                    "sigma_c": 0.3,
                    "score": 77.0,
                    "prob_within_0_5_c": 0.75,
                    "prob_within_1_0_c": 1.0,
                }
            },
        }

    def get_predictive_summary(self, market_station_id: str, limit: int = 20, eligible_only: bool = False) -> dict:
        return {
            "rows": [
                {
                    "station_id": "KNYC001",
                    "source": "WUNDERGROUND",
                    "next_metar_score": 77.0,
                    "next_metar_mae_c": 0.4,
                    "next_metar_bias_c": 0.1,
                    "next_metar_prob_within_0_5_c": 0.75,
                    "next_metar_prob_within_1_0_c": 1.0,
                    "lead_avg_minutes": 24.0,
                    "lead_samples": 4,
                    "predictive_score": 77.0,
                    "weight": 1.2,
                    "weight_predictive": 1.1,
                    "rank_eligible": True,
                    "learning_phase": "READY",
                    "quality_band": "GOOD",
                    "lead_bucket_summaries": {
                        "15-30": {
                            "bucket": "15-30",
                            "count": 4,
                            "avg_lead_minutes": 22.0,
                            "bias_c": 0.1,
                            "mae_c": 0.4,
                            "rmse_c": 0.5,
                            "sigma_c": 0.3,
                            "score": 77.0,
                            "prob_within_0_5_c": 0.75,
                            "prob_within_1_0_c": 1.0,
                        }
                    },
                }
            ]
        }

    def list_audit_dates(self, market_station_id: str) -> list[str]:
        return []


def _world_event(station_id: str, obs_iso: str, temp_f: float) -> dict:
    return {
        "ch": "world",
        "station_id": station_id,
        "obs_time_utc": obs_iso,
        "ts_ingest_utc": obs_iso,
        "data": {
            "src": "METAR",
            "temp_f": temp_f,
            "temp_c": round((temp_f - 32.0) * (5.0 / 9.0), 3),
            "report_type": "METAR",
            "is_speci": False,
        },
    }


def _pws_event(station_id: str, ingest_iso: str, readings: list[dict], pws_ids: list[str] | None = None) -> dict:
    data = {
        "support": len(readings),
        "pws_ids": pws_ids or [row.get("station_id", "") for row in readings],
    }
    if readings:
        data["pws_readings"] = readings
    return {
        "ch": "pws",
        "station_id": station_id,
        "obs_time_utc": ingest_iso,
        "ts_ingest_utc": ingest_iso,
        "data": data,
    }


def test_build_payload_matches_current_and_next_metar(monkeypatch):
    station_id = "KLGA"
    events = [
        _world_event(station_id, "2026-02-10T14:00:00+00:00", 30.0),
        _pws_event(
            station_id,
            "2026-02-10T14:01:00+00:00",
            [
                {
                    "station_id": "KNYC001",
                    "station_name": "Astoria roof",
                    "source": "WUNDERGROUND",
                    "distance_km": 1.2,
                    "obs_time_utc": "2026-02-10T14:02:00+00:00",
                    "temp_f": 31.0,
                    "valid": True,
                },
                {
                    "station_id": "KNYC001",
                    "station_name": "Astoria roof",
                    "source": "WUNDERGROUND",
                    "distance_km": 1.2,
                    "obs_time_utc": "2026-02-10T14:06:00+00:00",
                    "temp_f": 31.4,
                    "valid": True,
                },
                {
                    "station_id": "KNYC001",
                    "station_name": "Astoria roof",
                    "source": "WUNDERGROUND",
                    "distance_km": 1.2,
                    "obs_time_utc": "2026-02-10T14:12:00+00:00",
                    "temp_f": 32.2,
                    "valid": True,
                },
            ],
        ),
        _world_event(station_id, "2026-02-10T14:15:00+00:00", 32.0),
    ]

    monkeypatch.setattr(pws_browser, "_load_station_day_events", lambda *_args, **_kwargs: events)
    monkeypatch.setattr(pws_browser, "get_pws_learning_store", lambda: _FakeLearningStore())

    payload = pws_browser.build_pws_browser_payload(station_id, "2026-02-10")

    assert payload["detail_available"] is True
    assert payload["detail_source"] == "recorded_pws_readings"
    assert payload["stats"]["metar_count"] == 2

    station = payload["pws_stations"][0]
    assert station["station_id"] == "KNYC001"
    assert station["raw_records_count"] == 3
    assert station["table_records_count"] == 2

    first_row, last_row = station["table_rows"]
    assert first_row["current_metar"]["temp_f"] == 30.0
    assert first_row["next_metar"]["temp_f"] == 32.0
    assert last_row["current_metar"]["temp_f"] == 30.0
    assert last_row["next_metar"]["temp_f"] == 32.0
    assert station["summary"]["current_metar_matches"] == 2
    assert station["summary"]["next_metar_matches"] == 2


def test_build_payload_exposes_legacy_station_ids_when_detail_missing(monkeypatch):
    station_id = "KLGA"
    events = [
        _world_event(station_id, "2026-02-10T14:00:00+00:00", 30.0),
        _pws_event(
            station_id,
            "2026-02-10T14:05:00+00:00",
            [],
            pws_ids=["KNYLEGACY1", "KNYLEGACY2", "KNYLEGACY1"],
        ),
    ]

    monkeypatch.setattr(pws_browser, "_load_station_day_events", lambda *_args, **_kwargs: events)
    monkeypatch.setattr(pws_browser, "get_pws_learning_store", lambda: _FakeLearningStore())

    payload = pws_browser.build_pws_browser_payload(station_id, "2026-02-10")

    assert payload["detail_available"] is False
    assert payload["detail_source"] == "none"
    assert payload["legacy_station_ids"][0]["station_id"] == "KNYLEGACY1"
    assert payload["legacy_station_ids"][0]["occurrences"] == 2
    assert "legacy" in payload["message"].lower()
