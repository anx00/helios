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

