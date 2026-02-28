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

    def load_audit_rows(self, market_station_id: str, date_str: str) -> list[dict]:
        return []


class _FakeAuditLearningStore(_FakeLearningStore):
    def list_audit_dates(self, market_station_id: str) -> list[str]:
        return ["2026-02-10"]

    def load_audit_rows(self, market_station_id: str, date_str: str) -> list[dict]:
        return [
            {
                "station_id": "KNYAUD1",
                "station_name": "Queens terrace",
                "source": "WUNDERGROUND",
                "distance_km": 2.1,
                "kind": "LEAD",
                "lead_bucket": "15-30",
                "lead_minutes": 20.0,
                "sample_obs_time_utc": "2026-02-10T14:00:00+00:00",
                "sample_temp_c": 0.0,
                "sample_temp_f": 32.0,
                "official_obs_time_utc": "2026-02-10T14:20:00+00:00",
                "official_temp_c": 0.4,
                "official_temp_f": 32.72,
                "abs_error_c": 0.4,
            }
        ]


class _HistoryObs:
    def __init__(self, observation_time: str, temp_c: float, report_type: str = "METAR", is_speci: bool = False):
        self.observation_time = datetime.fromisoformat(observation_time)
        self.temp_c = temp_c
        self.temp_f = (temp_c * 9.0 / 5.0) + 32.0
        self.report_type = report_type
        self.is_speci = is_speci


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
    overview = payload["market_overview"]

    assert payload["detail_available"] is True
    assert payload["detail_source"] == "recorded_pws_readings"
    assert payload["stats"]["metar_count"] == 2
    assert overview["detailed_station_count"] == 1
    assert overview["overall_next_total"] == 2
    assert overview["overall_next_hits"] == 1
    assert overview["best_day_station"]["station_id"] == "KNYC001"

    station = payload["pws_stations"][0]
    assert station["station_id"] == "KNYC001"
    assert station["raw_records_count"] == 3
    assert station["table_records_count"] == 2

    first_row, last_row = station["table_rows"]
    assert first_row["current_metar"]["temp_f"] == 30.0
    assert first_row["next_metar"]["temp_f"] == 32.0
    assert first_row["next_metar_verdict"]["hit"] is False
    assert last_row["current_metar"]["temp_f"] == 30.0
    assert last_row["next_metar"]["temp_f"] == 32.0
    assert last_row["next_metar_verdict"]["hit"] is True
    assert station["summary"]["current_metar_matches"] == 2
    assert station["summary"]["next_metar_matches"] == 2


def test_build_payload_includes_station_level_summary_and_prediction(monkeypatch):
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
    summary = payload["pws_stations"][0]["summary"]

    assert round(summary["mean_temp_f"], 1) == 31.5
    assert round(summary["median_temp_f"], 1) == 31.4
    assert round(summary["min_temp_f"], 1) == 31.0
    assert round(summary["max_temp_f"], 1) == 32.2
    assert summary["predicted_next_metar"]["bucket"] == "15-30"


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


def test_build_payload_uses_learning_audit_when_replay_detail_missing(monkeypatch):
    station_id = "KLGA"
    events = [
        _world_event(station_id, "2026-02-10T13:50:00+00:00", 31.0),
        _world_event(station_id, "2026-02-10T14:20:00+00:00", 32.7),
        _pws_event(
            station_id,
            "2026-02-10T14:05:00+00:00",
            [],
            pws_ids=["KNYAUD1"],
        ),
    ]

    monkeypatch.setattr(pws_browser, "_load_station_day_events", lambda *_args, **_kwargs: events)
    monkeypatch.setattr(pws_browser, "get_pws_learning_store", lambda: _FakeAuditLearningStore())

    payload = pws_browser.build_pws_browser_payload(station_id, "2026-02-10")

    assert payload["detail_available"] is True
    assert payload["detail_source"] == "learning_audit"
    station = payload["pws_stations"][0]
    assert station["station_id"] == "KNYAUD1"
    assert station["table_rows"][0]["current_metar"]["temp_f"] == 31.0
    assert station["table_rows"][0]["next_metar"]["temp_f"] == 32.72
    assert station["table_rows"][0]["predicted_next_metar"]["bucket"] == "15-30"


def test_hydrate_payload_with_metar_history_fills_missing_current_and_next(monkeypatch):
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
                }
            ],
        ),
    ]

    monkeypatch.setattr(pws_browser, "_load_station_day_events", lambda *_args, **_kwargs: events)
    monkeypatch.setattr(pws_browser, "get_pws_learning_store", lambda: _FakeLearningStore())

    payload = pws_browser.build_pws_browser_payload(station_id, "2026-02-10")
    station = payload["pws_stations"][0]
    assert pws_browser.pws_payload_needs_metar_backfill(payload) is True
    assert station["table_rows"][0]["current_metar"]["temp_f"] == 30.0
    assert station["table_rows"][0]["next_metar"] is None

    history_rows = pws_browser.build_metar_series_from_history(
        station_id,
        "2026-02-10",
        [_HistoryObs("2026-02-10T14:20:00+00:00", 0.0)],
    )
    hydrated = pws_browser.hydrate_pws_browser_payload_with_metar_series(station_id, payload, history_rows)

    hydrated_station = hydrated["pws_stations"][0]
    assert hydrated_station["table_rows"][0]["next_metar"]["temp_f"] == 32.0
    assert hydrated_station["summary"]["next_metar_matches"] == 1


def test_build_metar_series_from_history_keeps_boundary_metars():
    rows = pws_browser.build_metar_series_from_history(
        "KLGA",
        "2026-02-10",
        [
            _HistoryObs("2026-02-10T04:51:00+00:00", 1.0),
            _HistoryObs("2026-02-10T05:51:00+00:00", 2.0),
            _HistoryObs("2026-02-11T05:51:00+00:00", 3.0),
        ],
    )

    assert len(rows) == 3
    assert rows[0]["station_time"] == "2026-02-09 23:51"
    assert rows[1]["station_time"] == "2026-02-10 00:51"
    assert rows[2]["station_time"] == "2026-02-11 00:51"


def test_prediction_verdict_uses_half_up_rounding(monkeypatch):
    station_id = "KLGA"
    events = [
        _world_event(station_id, "2026-02-10T14:00:00+00:00", 34.0),
        _pws_event(
            station_id,
            "2026-02-10T14:01:00+00:00",
            [
                {
                    "station_id": "KNYC001",
                    "station_name": "Astoria roof",
                    "source": "WUNDERGROUND",
                    "distance_km": 1.2,
                    "obs_time_utc": "2026-02-10T14:05:00+00:00",
                    "temp_f": 35.5,
                    "valid": True,
                }
            ],
        ),
        _world_event(station_id, "2026-02-10T15:00:00+00:00", 36.0),
    ]

    monkeypatch.setattr(pws_browser, "_load_station_day_events", lambda *_args, **_kwargs: events)
    monkeypatch.setattr(pws_browser, "get_pws_learning_store", lambda: _FakeLearningStore())

    payload = pws_browser.build_pws_browser_payload(station_id, "2026-02-10")
    verdict = payload["pws_stations"][0]["table_rows"][0]["next_metar_verdict"]

    assert verdict["hit"] is True
    assert verdict["pws_settlement"] == 36.0
    assert verdict["next_metar_settlement"] == 36.0
