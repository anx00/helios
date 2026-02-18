import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


UTC = ZoneInfo("UTC")


class _StubReader:
    def __init__(self, events):
        self._events = events

    def get_events_sorted(self, date_str, station_id=None, channels=None):
        return list(self._events)


class _DateAwareStubReader:
    def __init__(self, by_date):
        self._by_date = by_date

    def get_events_sorted(self, date_str, station_id=None, channels=None):
        events = list(self._by_date.get(date_str, []))
        if channels is not None:
            allowed = set(channels)
            events = [e for e in events if e.get("ch") in allowed]
        if station_id:
            events = [e for e in events if e.get("station_id") in (None, station_id)]
        return events

    def list_channels_for_date(self, date_str, station_id=None):
        events = self.get_events_sorted(date_str, station_id=station_id, channels=None)
        return sorted({str(e.get("ch")) for e in events if e.get("ch")})


class _TimestampLike:
    def __init__(self, dt: datetime):
        self._dt = dt

    def to_pydatetime(self):
        return self._dt


def test_replay_load_initializes_clock_with_datetime_timestamps():
    from core.replay_engine import ReplaySession

    start = datetime(2026, 2, 6, 12, 0, 0, tzinfo=UTC)
    mid = start + timedelta(minutes=30)
    end = start + timedelta(hours=1)

    events = [
        {"ch": "health", "data": {"ok": True}},  # no timestamp (ignored for bounds)
        {"ch": "world", "ts_ingest_utc": start, "data": {"src": "METAR"}},
        {"ch": "nowcast", "ts_ingest_utc": _TimestampLike(mid), "data": {"confidence": 0.7}},
        {"ch": "l2_snap", "ts_ingest_utc": end, "data": {"__meta__": {}}},
    ]

    session = ReplaySession("s1", "2026-02-06", "KLGA")
    session._reader = _StubReader(events)

    ok = asyncio.run(session.load())
    assert ok

    state = session.get_state()
    assert state["clock"]["session_start"] is not None
    assert state["clock"]["session_end"] is not None
    assert state["clock"]["progress_percent"] == 0.0

    session.seek_percent(50)
    mid_progress = session.clock.get_progress_percent()
    assert 49.0 <= mid_progress <= 51.0


def test_replay_pws_learning_summary_tracks_leader_changes():
    from core.replay_engine import ReplaySession

    start = datetime(2026, 2, 6, 12, 0, 0, tzinfo=UTC)
    e1 = {
        "ch": "pws",
        "ts_ingest_utc": start,
        "ts_nyc": "2026-02-05 19:00:00",
        "data": {
            "median_f": 70.0,
            "support": 12,
            "weighted_support": 10.5,
            "station_learning": {
                "A": {"station_id": "A", "weight": 1.20, "now_score": 80.0, "lead_score": 60.0, "now_samples": 10, "lead_samples": 5},
                "B": {"station_id": "B", "weight": 1.05, "now_score": 74.0, "lead_score": 64.0, "now_samples": 8, "lead_samples": 4},
            },
        },
    }
    e2 = {
        "ch": "pws",
        "ts_ingest_utc": start + timedelta(minutes=30),
        "ts_nyc": "2026-02-05 19:30:00",
        "data": {
            "median_f": 71.0,
            "support": 11,
            "weighted_support": 10.2,
            "station_learning": {
                "A": {"station_id": "A", "weight": 1.11, "now_score": 75.0, "lead_score": 62.0, "now_samples": 12, "lead_samples": 7},
                "B": {"station_id": "B", "weight": 1.24, "now_score": 83.0, "lead_score": 67.0, "now_samples": 11, "lead_samples": 7},
            },
        },
    }
    events = [
        {"ch": "world", "ts_ingest_utc": start - timedelta(minutes=10), "data": {"src": "METAR"}},
        e1,
        e2,
    ]

    session = ReplaySession("s2", "2026-02-06", "KLGA")
    session._reader = _StubReader(events)
    ok = asyncio.run(session.load())
    assert ok

    # Move to end so summary includes both pws events
    session.seek_percent(100)

    summary = session.get_pws_learning_summary(max_points=20, top_n=3)
    assert summary["total_pws_events"] == 2
    assert summary["current"] is not None
    assert summary["current"]["top"][0]["station_id"] == "B"
    assert len(summary["leader_changes"]) >= 2
    assert len(summary["timeline"]) >= 2


def test_replay_pws_learning_summary_warmup_when_no_rank_eligible_station():
    from core.replay_engine import ReplaySession

    start = datetime(2026, 2, 6, 12, 0, 0, tzinfo=UTC)
    events = [
        {
            "ch": "pws",
            "ts_ingest_utc": start,
            "ts_nyc": "2026-02-05 19:00:00",
            "data": {
                "median_f": 70.0,
                "support": 10,
                "weighted_support": 8.0,
                "station_learning": {
                    "A": {
                        "station_id": "A",
                        "weight": 1.20,
                        "now_score": 78.0,
                        "lead_score": 62.0,
                        "now_samples": 1,
                        "lead_samples": 0,
                        "rank_eligible": False,
                        "rank_min_now_samples": 2,
                        "rank_min_lead_samples": 1,
                    }
                },
            },
        }
    ]

    session = ReplaySession("s3", "2026-02-06", "KLGA")
    session._reader = _StubReader(events)
    ok = asyncio.run(session.load())
    assert ok
    session.seek_percent(100)

    summary = session.get_pws_learning_summary(max_points=20, top_n=3)
    assert summary["total_pws_events"] == 1
    assert summary["ranked_pws_events"] == 0
    assert summary["current"] is not None
    assert summary["current"]["status"] == "WARMUP"
    assert summary["current"]["top"] == []


def test_replay_pws_learning_summary_includes_consensus_snapshot_rows():
    from core.replay_engine import ReplaySession

    start = datetime(2026, 2, 6, 12, 0, 0, tzinfo=UTC)
    events = [
        {
            "ch": "pws",
            "station_id": "KLGA",
            "ts_ingest_utc": start,
            "ts_nyc": "2026-02-05 19:00:00",
            "obs_time_utc": start - timedelta(minutes=2),
            "data": {
                "median_f": 70.1,
                "mad_f": 1.2,
                "support": 2,
                "weighted_support": 1.9,
                "qc": "OK",
                "pws_readings": [
                    {
                        "station_id": "A",
                        "station_name": "Station A",
                        "source": "SYNOPTIC",
                        "temp_f": 69.8,
                        "temp_c": 21.0,
                        "age_minutes": 5.0,
                        "distance_km": 1.2,
                        "valid": True,
                        "weight": 1.25,
                        "now_score": 84.0,
                        "lead_score": 73.0,
                        "now_samples": 12,
                        "lead_samples": 8,
                        "rank_eligible": True,
                    }
                ],
                "pws_outliers": [
                    {
                        "station_id": "B",
                        "station_name": "Station B",
                        "source": "WUNDERGROUND",
                        "temp_f": 75.0,
                        "temp_c": 23.9,
                        "age_minutes": 7.0,
                        "distance_km": 2.5,
                        "valid": False,
                        "qc_flag": "SPATIAL_OUTLIER_MAD",
                    }
                ],
                "station_learning": {
                    "A": {
                        "station_id": "A",
                        "weight": 1.25,
                        "predictive_score": 80.5,
                        "samples_total": 20,
                        "now_ema_abs_error_c": 0.35,
                        "lead_ema_abs_error_c": 0.58,
                        "rank_eligible": True,
                    },
                    "B": {
                        "station_id": "B",
                        "weight": 0.92,
                        "predictive_score": 89.0,
                        "samples_total": 45,
                        "now_score": 69.0,
                        "lead_score": 90.0,
                        "now_samples": 21,
                        "lead_samples": 24,
                        "rank_eligible": True,
                    },
                },
            },
        }
    ]

    session = ReplaySession("s3b", "2026-02-06", "KLGA")
    session._reader = _StubReader(events)
    ok = asyncio.run(session.load())
    assert ok
    session.seek_percent(100)

    summary = session.get_pws_learning_summary(max_points=20, top_n=3)
    snapshot = summary.get("consensus_snapshot")
    assert isinstance(snapshot, dict)
    assert snapshot["market_station_id"] == "KLGA"
    assert snapshot["in_consensus_count"] == 1
    assert snapshot["excluded_count"] == 1
    assert snapshot["top_historical_station_id"] == "B"

    rows = snapshot.get("rows", [])
    assert len(rows) == 2

    row_a = next(r for r in rows if r["station_id"] == "A")
    assert row_a["in_consensus"] is True
    assert row_a["temp_f"] == 69.8
    assert row_a["predictive_score"] == 80.5
    assert row_a["now_ema_abs_error_c"] == 0.35
    assert row_a["now_samples_session"] is None
    assert row_a["lead_samples_session"] is None

    row_b = next(r for r in rows if r["station_id"] == "B")
    assert row_b["in_consensus"] is False
    assert row_b["excluded_reason"] == "SPATIAL_OUTLIER_MAD"
    assert row_b["predictive_score"] == 89.0


def test_replay_category_summary_maps_market_alias_to_l2_snap():
    from core.replay_engine import ReplaySession

    start = datetime(2026, 2, 6, 12, 0, 0, tzinfo=UTC)
    events = [
        {"ch": "market", "ts_ingest_utc": start, "ts_nyc": "2026-02-05 19:00:00", "data": {"34-35F": {"mid": 0.4}}},
        {"ch": "l2_snap_1s", "ts_ingest_utc": start + timedelta(seconds=1), "ts_nyc": "2026-02-05 19:00:01", "data": {"36+F": {"mid": 0.6}}},
    ]

    session = ReplaySession("s4", "2026-02-06", "KLGA")
    session._reader = _StubReader(events)
    ok = asyncio.run(session.load())
    assert ok
    session.seek_percent(100)

    cats = session.get_category_summary()
    assert cats["l2_snap"]["count"] == 2
    assert cats["l2_snap"]["latest"] is not None
    assert cats["l2_snap"]["latest"]["ch"] == "l2_snap_1s"


def test_replay_state_exposes_loaded_channels():
    from core.replay_engine import ReplaySession

    start = datetime(2026, 2, 6, 12, 0, 0, tzinfo=UTC)
    events = [
        {"ch": "world", "ts_ingest_utc": start, "data": {"src": "METAR"}},
        {"ch": "market", "ts_ingest_utc": start + timedelta(seconds=1), "data": {"x": {}}},
    ]

    session = ReplaySession("s5", "2026-02-06", "KLGA")
    session._reader = _StubReader(events)
    ok = asyncio.run(session.load())
    assert ok

    state = session.get_state()
    assert "channels_loaded" in state
    assert state["channels_loaded"] == ["market", "world"]


def test_replay_helpers_map_london_market_day_to_nyc_partitions():
    from core.replay_engine import (
        _map_partition_date_to_station_market_dates,
        _source_partition_dates_for_local_day,
    )

    london = ZoneInfo("Europe/London")

    # One NYC partition can touch two London market dates.
    local_dates = _map_partition_date_to_station_market_dates("2026-02-17", london)
    assert local_dates == ["2026-02-17", "2026-02-18"]

    # A London market day needs two NYC source partitions.
    source_dates = _source_partition_dates_for_local_day("2026-02-18", london)
    assert source_dates == ["2026-02-17", "2026-02-18"]


def test_replay_load_filters_to_station_local_market_day():
    from core.replay_engine import ReplaySession

    # London (UTC in February): local market day 2026-02-18 means UTC [00:00, 24:00).
    # Storage partitions are still NYC-based, so replay must read both 2026-02-17 and 2026-02-18.
    prev_partition_events = [
        {"ch": "world", "station_id": "EGLC", "ts_ingest_utc": datetime(2026, 2, 17, 23, 50, tzinfo=UTC), "data": {"src": "METAR"}},
        {"ch": "world", "station_id": "EGLC", "ts_ingest_utc": datetime(2026, 2, 18, 0, 10, tzinfo=UTC), "data": {"src": "METAR"}},
    ]
    next_partition_events = [
        {"ch": "world", "station_id": "EGLC", "ts_ingest_utc": datetime(2026, 2, 18, 15, 0, tzinfo=UTC), "data": {"src": "METAR"}},
        {"ch": "world", "station_id": "EGLC", "ts_ingest_utc": datetime(2026, 2, 19, 0, 5, tzinfo=UTC), "data": {"src": "METAR"}},
    ]

    session = ReplaySession("s6", "2026-02-18", "EGLC")
    session._reader = _DateAwareStubReader(
        {
            "2026-02-17": prev_partition_events,
            "2026-02-18": next_partition_events,
        }
    )

    ok = asyncio.run(session.load())
    assert ok
    assert len(session._events) == 2
    assert session._events[0]["ts_ingest_utc"] == datetime(2026, 2, 18, 0, 10, tzinfo=UTC)
    assert session._events[1]["ts_ingest_utc"] == datetime(2026, 2, 18, 15, 0, tzinfo=UTC)

    state = session.get_state()
    assert state["date_reference"] == "Europe/London market date"
    assert state["source_dates"] == ["2026-02-17", "2026-02-18"]
