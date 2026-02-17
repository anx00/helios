import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


UTC = ZoneInfo("UTC")


class _StubReader:
    def __init__(self, events):
        self._events = events

    def get_events_sorted(self, date_str, station_id=None, channels=None):
        return list(self._events)


class _TimestampLike:
    def __init__(self, dt: datetime):
        self._dt = dt

    def to_pydatetime(self):
        return self._dt


def test_replay_load_initializes_clock_with_datetime_timestamps():
    from core.replay_engine import ReplaySession

    start = datetime(2026, 2, 6, 0, 0, 0, tzinfo=UTC)
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

    start = datetime(2026, 2, 6, 0, 0, 0, tzinfo=UTC)
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
