"""
Replay Engine - Phase 4 (Section 4.8)

Virtual clock-based replay of recorded sessions.
Emits events via the same WebSocket channels as live.

Features:
- Virtual clock with speed control (1x, 5x, 10x)
- Jump to specific events (METAR, windows, shocks)
- Timeline scrubbing
- Same channel/payload format as live
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import datetime, timedelta, date as date_cls, time as time_cls
from enum import Enum
from typing import Optional, Dict, List, Callable, Any
from zoneinfo import ZoneInfo

from core.compactor import get_hybrid_reader, HybridReader
from core.market_pricing import normalize_probability_pct, select_probability_pct_from_quotes
from config import STATIONS

logger = logging.getLogger("replay_engine")

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


def _parse_iso_date(value: str) -> Optional[date_cls]:
    try:
        return date_cls.fromisoformat(str(value))
    except Exception:
        return None


def _resolve_station_timezone(station_id: Optional[str]) -> ZoneInfo:
    sid = str(station_id or "").upper()
    if sid and sid in STATIONS:
        tz_name = str(getattr(STATIONS[sid], "timezone", "") or "")
        if tz_name:
            try:
                return ZoneInfo(tz_name)
            except Exception:
                logger.warning(
                    "Replay timezone fallback: invalid tz for station=%s (%s)",
                    sid,
                    tz_name,
                )
    return NYC


def _local_day_window_utc(
    date_str: str,
    station_tz: ZoneInfo,
) -> Optional[tuple[datetime, datetime]]:
    target_date = _parse_iso_date(date_str)
    if target_date is None:
        return None
    start_local = datetime.combine(target_date, time_cls.min, tzinfo=station_tz)
    end_local = start_local + timedelta(days=1)
    return (start_local.astimezone(UTC), end_local.astimezone(UTC))


def _source_partition_dates_for_local_day(
    date_str: str,
    station_tz: ZoneInfo,
) -> List[str]:
    """
    Resolve underlying storage partition dates for a station-local market day.

    Recordings are currently partitioned by NYC ingest date. Non-NYC market days
    can span two NYC partitions, so replay must read both and then filter by local
    day window.
    """
    window = _local_day_window_utc(date_str, station_tz)
    if window is None:
        return [date_str]

    start_utc, end_utc = window
    start_nyc = start_utc.astimezone(NYC).date()
    end_nyc = (end_utc - timedelta(microseconds=1)).astimezone(NYC).date()

    dates = {str(date_str)}
    cur = start_nyc
    while cur <= end_nyc:
        dates.add(cur.isoformat())
        cur += timedelta(days=1)
    return sorted(dates)


def _map_partition_date_to_station_market_dates(
    raw_date_str: str,
    station_tz: ZoneInfo,
) -> List[str]:
    """
    Map a raw NYC partition date into one or two station-local market dates.
    """
    raw_date = _parse_iso_date(raw_date_str)
    if raw_date is None:
        return []

    start_nyc = datetime.combine(raw_date, time_cls.min, tzinfo=NYC)
    end_nyc = start_nyc + timedelta(days=1)

    local_start = start_nyc.astimezone(station_tz).date()
    local_end = (end_nyc - timedelta(microseconds=1)).astimezone(station_tz).date()

    out = set()
    cur = local_start
    while cur <= local_end:
        out.add(cur.isoformat())
        cur += timedelta(days=1)
    return sorted(out)


class ReplayState(Enum):
    """Replay session state."""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"


class ReplaySpeed(Enum):
    """Playback speed multipliers."""
    REALTIME = 1.0
    FAST_2X = 2.0
    FAST_5X = 5.0
    FAST_10X = 10.0
    FAST_50X = 50.0
    INSTANT = 0.0  # No delay between events


class VirtualClock:
    """
    Virtual clock for replay.

    Maps real elapsed time to simulated session time.
    """

    def __init__(self):
        self._session_start: Optional[datetime] = None
        self._session_end: Optional[datetime] = None
        self._current_time: Optional[datetime] = None
        self._speed: float = 1.0
        self._real_start: Optional[datetime] = None
        self._paused_at: Optional[datetime] = None
        self._is_paused: bool = True

    def initialize(self, start_time: datetime, end_time: datetime):
        """Initialize clock with session bounds."""
        self._session_start = start_time
        self._session_end = end_time
        self._current_time = start_time
        self._is_paused = True

    def play(self, speed: float = 1.0):
        """Start/resume playback."""
        self._speed = speed
        self._real_start = datetime.now(UTC)
        if self._paused_at:
            # Resume from paused position
            self._current_time = self._paused_at
        self._paused_at = None
        self._is_paused = False

    def set_speed(self, speed: float):
        """Change speed without pausing/resuming. Snapshots current virtual time first."""
        if not self._is_paused:
            self._current_time = self.now()
            self._real_start = datetime.now(UTC)
        self._speed = speed

    def pause(self):
        """Pause playback."""
        self._paused_at = self.now()
        self._is_paused = True

    def seek(self, target_time: datetime):
        """Jump to specific time."""
        if self._session_start and self._session_end:
            # Clamp to session bounds
            target_time = max(self._session_start, min(target_time, self._session_end))
        self._current_time = target_time
        self._paused_at = target_time
        self._real_start = datetime.now(UTC)

    def seek_percent(self, percent: float):
        """Jump to percentage through session (0-100)."""
        if not self._session_start or not self._session_end:
            return

        duration = (self._session_end - self._session_start).total_seconds()
        offset = duration * (percent / 100.0)
        target = self._session_start + timedelta(seconds=offset)
        self.seek(target)

    def now(self) -> Optional[datetime]:
        """Get current virtual time."""
        if self._is_paused:
            return self._paused_at or self._current_time

        if not self._real_start or not self._current_time:
            return self._current_time

        # Calculate elapsed real time
        real_elapsed = (datetime.now(UTC) - self._real_start).total_seconds()

        # Apply speed multiplier
        virtual_elapsed = real_elapsed * self._speed

        # Calculate virtual time
        virtual_time = self._current_time + timedelta(seconds=virtual_elapsed)

        # Clamp to session end
        if self._session_end and virtual_time > self._session_end:
            return self._session_end

        return virtual_time

    def is_finished(self) -> bool:
        """Check if reached end of session."""
        current = self.now()
        if not current or not self._session_end:
            return False
        return current >= self._session_end

    def get_progress_percent(self) -> float:
        """Get progress through session (0-100)."""
        if not self._session_start or not self._session_end:
            return 0.0

        current = self.now()
        if not current:
            return 0.0

        duration = (self._session_end - self._session_start).total_seconds()
        if duration <= 0:
            return 0.0

        elapsed = (current - self._session_start).total_seconds()
        return max(0.0, min(100.0, (elapsed / duration) * 100.0))

    def get_state(self) -> Dict:
        """Get clock state for UI."""
        return {
            "session_start": self._session_start.isoformat() if self._session_start else None,
            "session_end": self._session_end.isoformat() if self._session_end else None,
            "current_time": self.now().isoformat() if self.now() else None,
            "speed": self._speed,
            "is_paused": self._is_paused,
            "progress_percent": self.get_progress_percent(),
            "is_finished": self.is_finished()
        }


class ReplaySession:
    """
    A single replay session.

    Loads recorded data and plays it back through callbacks.
    """

    def __init__(
        self,
        session_id: str,
        date_str: str,
        station_id: Optional[str] = None,
        channels: Optional[List[str]] = None
    ):
        self.session_id = session_id
        self.date_str = date_str
        self.station_id = station_id
        self.channels = channels
        self._station_tz = _resolve_station_timezone(station_id)
        self._date_reference = (
            f"{self._station_tz.key} market date"
            if station_id
            else "NYC market date"
        )
        self._source_dates: List[str] = [date_str]
        self._local_day_start_utc: Optional[datetime] = None
        self._local_day_end_utc: Optional[datetime] = None

        self.state = ReplayState.IDLE
        self.clock = VirtualClock()

        # Data
        self._events: List[Dict] = []
        self._event_index: int = 0
        self._metar_indices: List[int] = []
        self._window_indices: List[int] = []
        self._pws_learning_cache_key: Optional[tuple] = None
        self._pws_learning_cache_value: Optional[Dict[str, Any]] = None

        # Playback
        self._playback_task: Optional[asyncio.Task] = None
        self._event_callback: Optional[Callable[[Dict], None]] = None

        # Reader
        self._reader = get_hybrid_reader()

    async def load(self) -> bool:
        """Load events from Parquet."""
        self.state = ReplayState.LOADING
        logger.info(f"Loading replay session: {self.date_str} / {self.station_id}")

        try:
            local_window_utc: Optional[tuple[datetime, datetime]] = None
            if self.station_id:
                local_window_utc = _local_day_window_utc(self.date_str, self._station_tz)
                self._source_dates = _source_partition_dates_for_local_day(
                    self.date_str,
                    self._station_tz,
                )
            else:
                self._source_dates = [self.date_str]

            if local_window_utc:
                self._local_day_start_utc, self._local_day_end_utc = local_window_utc
            else:
                self._local_day_start_utc = None
                self._local_day_end_utc = None

            channels = self.channels
            # Default channel set optimized for UI/replay latency.
            # Prefer lower-volume market channel when multiple variants exist.
            if channels is None and hasattr(self._reader, "list_channels_for_date"):
                available = set()
                for source_date in self._source_dates:
                    available.update(self._reader.list_channels_for_date(source_date, self.station_id))
                channels = ["world", "pws", "nowcast", "features", "health", "event_window"]
                if "market" in available:
                    channels.append("market")
                elif "l2_snap" in available:
                    channels.append("l2_snap")
                elif "l2_snap_1s" in available:
                    channels.append("l2_snap_1s")

            # Read all events from source partitions, then filter to the requested
            # station-local market day window.
            all_events: List[Dict[str, Any]] = []
            for source_date in self._source_dates:
                source_events = self._reader.get_events_sorted(
                    source_date,
                    self.station_id,
                    channels,
                )
                if source_events:
                    all_events.extend(source_events)

            if local_window_utc:
                start_utc, end_utc = local_window_utc
                filtered_events: List[Dict[str, Any]] = []
                for ev in all_events:
                    ev_ts = self._parse_timestamp(ev)
                    if ev_ts is None:
                        continue
                    if start_utc <= ev_ts < end_utc:
                        filtered_events.append(ev)
                all_events = filtered_events

            def _sort_key(ev: Dict[str, Any]):
                ev_ts = self._parse_timestamp(ev)
                if ev_ts is not None:
                    return (0, ev_ts)
                return (1, str(ev.get("ts_ingest_utc", "")))

            all_events.sort(key=_sort_key)
            self._events = all_events

            if not self._events:
                # Diagnostic logging
                try:
                    available_channels = set()
                    for source_date in self._source_dates:
                        available_channels.update(self._reader.list_channels_for_date(source_date))
                    available_dates = self._reader.list_available_dates(self.station_id)
                    logger.warning(
                        f"No events found for date={self.date_str}, station={self.station_id}, "
                        f"sources={self._source_dates}. "
                        f"Channels on source dates: {sorted(available_channels)}. "
                        f"Recent dates for station: {available_dates[:5]}"
                    )
                    if not available_channels:
                        logger.warning(
                            f"No NDJSON or Parquet data exists for source dates {self._source_dates}. "
                            f"Is the recorder running?"
                        )
                    elif self.station_id:
                        # Data exists but not for this station
                        all_events = []
                        for source_date in self._source_dates:
                            all_events.extend(
                                self._reader.get_events_sorted(source_date, None, self.channels)
                            )
                        stations_found = set()
                        for ev in all_events[:200]:
                            sid = ev.get("station_id") or ev.get("data", {}).get("station_id")
                            if sid:
                                stations_found.add(sid)
                        if stations_found:
                            logger.warning(
                                f"Data exists for stations: {stations_found}, "
                                f"but not for requested station={self.station_id}"
                            )
                except Exception as diag_err:
                    logger.warning(f"Diagnostics failed: {diag_err}")
                self.state = ReplayState.IDLE
                return False

            # Build index of special events
            self._build_indices()

            # Initialize clock from event timestamps
            first_ts, last_ts = self._find_session_bounds()

            if first_ts and last_ts:
                self.clock.initialize(first_ts, last_ts)
            else:
                logger.warning(
                    "Replay session loaded without parseable timestamps; "
                    "clock/progress controls will be limited"
                )

            self.state = ReplayState.READY
            self._event_index = 0

            logger.info(
                f"Loaded {len(self._events)} events, "
                f"{len(self._metar_indices)} METARs, "
                f"{len(self._window_indices)} windows"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load replay: {e}")
            self.state = ReplayState.IDLE
            return False

    def _find_session_bounds(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """Find first/last parseable timestamps in the loaded event list."""
        first_ts: Optional[datetime] = None
        last_ts: Optional[datetime] = None

        for event in self._events:
            ts = self._parse_timestamp(event)
            if ts is not None:
                first_ts = ts
                break

        for event in reversed(self._events):
            ts = self._parse_timestamp(event)
            if ts is not None:
                last_ts = ts
                break

        return first_ts, last_ts

    def _build_indices(self):
        """Build indices for special events."""
        self._metar_indices = []
        self._window_indices = []

        for i, event in enumerate(self._events):
            ch = event.get("ch", "")
            data = event.get("data", {})

            # METAR events
            if ch == "world" and data.get("src") == "METAR":
                self._metar_indices.append(i)

            # Event windows
            if ch == "event_window":
                self._window_indices.append(i)

    def _parse_timestamp(self, event: Dict) -> Optional[datetime]:
        """Parse timestamp from event."""
        ts_val = event.get("ts_ingest_utc") or event.get("obs_time_utc")
        if ts_val is None:
            return None

        try:
            # Native datetime (including from parquet/pandas conversion).
            if isinstance(ts_val, datetime):
                return ts_val if ts_val.tzinfo else ts_val.replace(tzinfo=UTC)

            # pandas.Timestamp-like object
            to_py = getattr(ts_val, "to_pydatetime", None)
            if callable(to_py):
                py_dt = to_py()
                if isinstance(py_dt, datetime):
                    return py_dt if py_dt.tzinfo else py_dt.replace(tzinfo=UTC)

            # Numeric epoch seconds
            if isinstance(ts_val, (int, float)):
                return datetime.fromtimestamp(ts_val, tz=UTC)

            # String encodings
            if isinstance(ts_val, str):
                ts_str = ts_val.strip()
                if not ts_str:
                    return None
                return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            return None
        return None

    def set_event_callback(self, callback: Callable[[Dict], None]):
        """Set callback for event emission."""
        self._event_callback = callback

    async def play(self, speed: float = 1.0):
        """Start/resume playback."""
        if self.state == ReplayState.FINISHED:
            # Restart from beginning when user presses Play after finish.
            self._event_index = 0
            if self.clock._session_start:
                self.clock.seek(self.clock._session_start)
            self.state = ReplayState.READY

        if self.state not in [ReplayState.READY, ReplayState.PAUSED]:
            return

        self.clock.play(speed)
        self.state = ReplayState.PLAYING

        # Start playback task
        if self._playback_task is None or self._playback_task.done():
            self._playback_task = asyncio.create_task(self._playback_loop())

    def set_speed(self, speed: float):
        """Change playback speed without stopping."""
        self.clock.set_speed(speed)

    async def pause(self):
        """Pause playback."""
        self.clock.pause()
        self.state = ReplayState.PAUSED

    async def stop(self):
        """Stop playback and reset."""
        self.state = ReplayState.READY
        self.clock.pause()

        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
            self._playback_task = None

        self._event_index = 0
        self.clock.seek(self.clock._session_start)

    def seek_time(self, target_time: datetime):
        """Seek to specific time."""
        self.clock.seek(target_time)

        # Find event index at this time
        target_str = target_time.isoformat()
        for i, event in enumerate(self._events):
            ts = event.get("ts_ingest_utc", "")
            if ts >= target_str:
                self._event_index = i
                break

    def seek_percent(self, percent: float):
        """Seek to percentage position."""
        self.clock.seek_percent(percent)

        # Calculate event index
        total = len(self._events)
        self._event_index = int(total * (percent / 100.0))
        self._event_index = max(0, min(total - 1, self._event_index))

    def jump_to_next_metar(self) -> bool:
        """Jump to next METAR event."""
        for idx in self._metar_indices:
            if idx > self._event_index:
                self._event_index = idx
                event_ts = self._parse_timestamp(self._events[idx])
                if event_ts:
                    self.clock.seek(event_ts)
                return True
        return False

    def jump_to_prev_metar(self) -> bool:
        """Jump to previous METAR event."""
        for idx in reversed(self._metar_indices):
            if idx < self._event_index:
                self._event_index = idx
                event_ts = self._parse_timestamp(self._events[idx])
                if event_ts:
                    self.clock.seek(event_ts)
                return True
        return False

    def jump_to_next_window(self) -> bool:
        """Jump to next event window."""
        for idx in self._window_indices:
            if idx > self._event_index:
                self._event_index = idx
                event_ts = self._parse_timestamp(self._events[idx])
                if event_ts:
                    self.clock.seek(event_ts)
                return True
        return False

    async def _playback_loop(self):
        """Main playback loop - emits events at correct virtual times.
        At high speeds, batch-processes all events up to clock.now()."""
        logger.info("Playback started")

        try:
            while self.state == ReplayState.PLAYING:
                if self._event_index >= len(self._events):
                    self.state = ReplayState.FINISHED
                    logger.info("Playback finished")
                    break

                # Check if next event is in the future
                event = self._events[self._event_index]
                event_ts = self._parse_timestamp(event)

                if event_ts:
                    current = self.clock.now()
                    if current and event_ts > current:
                        # Wait (real time) until clock catches up
                        virtual_gap = (event_ts - current).total_seconds()
                        speed = self.clock._speed
                        wait_seconds = virtual_gap / speed if speed > 0 else 0
                        wait_seconds = min(wait_seconds, 0.5)

                        if wait_seconds > 0.01:
                            await asyncio.sleep(wait_seconds)

                        if self.state != ReplayState.PLAYING:
                            break

                # Batch-emit all events whose timestamp <= clock.now()
                current = self.clock.now()
                batch_count = 0
                while self._event_index < len(self._events) and batch_count < 50:
                    ev = self._events[self._event_index]
                    ev_ts = self._parse_timestamp(ev)

                    if ev_ts and current and ev_ts > current:
                        break  # Next event is in the future

                    if self._event_callback:
                        try:
                            self._event_callback(ev)
                        except Exception as e:
                            logger.warning(f"Event callback error: {e}")

                    self._event_index += 1
                    batch_count += 1

                await asyncio.sleep(0)  # Yield to event loop

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Playback error: {e}")

    def get_state(self) -> Dict:
        """Get session state for UI."""
        channels_loaded = sorted({
            str(ev.get("ch"))
            for ev in self._events
            if isinstance(ev, dict) and ev.get("ch")
        })
        return {
            "session_id": self.session_id,
            "date": self.date_str,
            "station_id": self.station_id,
            "station_timezone": self._station_tz.key,
            "date_reference": self._date_reference,
            "source_dates": list(self._source_dates),
            "local_day_start_utc": self._local_day_start_utc.isoformat() if self._local_day_start_utc else None,
            "local_day_end_utc": self._local_day_end_utc.isoformat() if self._local_day_end_utc else None,
            "state": self.state.value,
            "total_events": len(self._events),
            "current_index": self._event_index,
            "metar_count": len(self._metar_indices),
            "window_count": len(self._window_indices),
            "channels_loaded": channels_loaded,
            "clock": self.clock.get_state()
        }

    def get_category_summary(self) -> Dict:
        """Get latest event and count per channel for category cards."""
        channels = ["world", "nowcast", "pws", "features", "event_window", "health", "l2_snap"]
        categories = {ch: {"count": 0, "latest": None, "latest_time": None} for ch in channels}
        l2_aliases = {"l2_snap", "l2_snap_1s", "market"}

        for i in range(min(self._event_index + 1, len(self._events))):
            event = self._events[i]
            ch = event.get("ch", "")
            key = "l2_snap" if ch in l2_aliases else ch
            if key in categories:
                categories[key]["count"] += 1
                categories[key]["latest"] = event
                categories[key]["latest_time"] = event.get("ts_nyc")

        return categories

    @staticmethod
    def _is_rank_eligible(row: Dict[str, Any]) -> bool:
        flag = row.get("rank_eligible")
        if isinstance(flag, bool):
            return flag

        def _as_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except Exception:
                return default

        now_samples = _as_int(row.get("now_samples"), 0)
        lead_samples = _as_int(row.get("lead_samples"), 0)
        min_now = max(0, _as_int(row.get("rank_min_now_samples"), 2))
        min_lead = max(0, _as_int(row.get("rank_min_lead_samples"), 1))
        return now_samples >= min_now and lead_samples >= min_lead

    @staticmethod
    def _extract_station_learning_rows(
        data: Dict[str, Any],
        eligible_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract sortable learning rows from a PWS event payload.

        Supports both:
        - data.station_learning (rich profiles)
        - data.station_weights (legacy weight-only payload)
        """
        rows: List[Dict[str, Any]] = []
        learning = data.get("station_learning")
        if isinstance(learning, dict) and learning:
            for sid, raw in learning.items():
                if isinstance(raw, dict):
                    row = dict(raw)
                    row.setdefault("station_id", sid)
                    rows.append(row)

        if not rows:
            weights = data.get("station_weights")
            if isinstance(weights, dict):
                for sid, w in weights.items():
                    try:
                        wv = float(w)
                    except Exception:
                        continue
                    rows.append({
                        "station_id": sid,
                        "weight": wv,
                        "weight_now": wv,
                        "weight_predictive": wv,
                        "now_score": None,
                        "lead_score": None,
                        "predictive_score": None,
                        "now_samples": None,
                        "lead_samples": None,
                        "quality_band": "LEGACY",
                        "rank_eligible": False,
                        "learning_phase": "LEGACY",
                        "rank_min_now_samples": 2,
                        "rank_min_lead_samples": 1,
                    })

        if eligible_only:
            rows = [r for r in rows if ReplaySession._is_rank_eligible(r)]

        def _sort_weight(row: Dict[str, Any]) -> float:
            try:
                return float(row.get("weight", 0.0) or 0.0)
            except Exception:
                return 0.0

        rows.sort(key=_sort_weight, reverse=True)
        return rows

    @staticmethod
    def _decimate_points(points: List[Dict[str, Any]], max_points: int) -> List[Dict[str, Any]]:
        if max_points <= 0 or len(points) <= max_points:
            return points
        step = max(1, int(math.ceil(len(points) / float(max_points))))
        sampled = points[::step]
        if sampled and sampled[-1] is not points[-1]:
            sampled.append(points[-1])
        return sampled

    @staticmethod
    def _normalize_probability_pct(value: Any) -> Optional[float]:
        return normalize_probability_pct(value)

    @staticmethod
    def _market_label_midpoint_f(label: Any) -> Optional[float]:
        if label is None:
            return None
        txt = str(label).strip()
        if not txt:
            return None
        lower = txt.lower()
        cleaned = txt.replace("Â", "").replace("°", " ").replace("º", " ")
        cleaned_lower = cleaned.lower()
        unit_c = bool(
            re.search(r"\d+\s*c\b", cleaned_lower)
            or re.search(r"\bcelsius\b", cleaned_lower)
        )
        unit_f = bool(
            re.search(r"\d+\s*f\b", cleaned_lower)
            or re.search(r"\bfahrenheit\b", cleaned_lower)
        )

        def _to_f(v: float) -> float:
            if unit_c and not unit_f:
                return (v * 9.0 / 5.0) + 32.0
            return v

        m_range = re.search(r"(-?\d+(?:\.\d+)?)\s*[-–—]\s*(-?\d+(?:\.\d+)?)", cleaned)
        if m_range:
            try:
                low_v = float(m_range.group(1))
                high_v = float(m_range.group(2))
                midpoint = (low_v + high_v) / 2.0
                return round(_to_f(midpoint), 4)
            except Exception:
                pass

        m_num = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if not m_num:
            return None
        try:
            n = float(m_num.group(0))
        except Exception:
            return None

        # Open-ended and single buckets use the threshold as representative point.
        return round(_to_f(n), 4)

    @staticmethod
    def _extract_market_top3(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(data, dict):
            return []
        rows: List[Dict[str, Any]] = []
        for label, payload in data.items():
            if label == "__meta__":
                continue
            if not isinstance(payload, dict):
                continue
            bid_pct_direct = ReplaySession._normalize_probability_pct(payload.get("best_bid"))
            ask_pct_direct = ReplaySession._normalize_probability_pct(payload.get("best_ask"))
            mid_pct = ReplaySession._normalize_probability_pct(payload.get("mid"))
            no_bid_pct = ReplaySession._normalize_probability_pct(payload.get("no_best_bid"))
            no_ask_pct = ReplaySession._normalize_probability_pct(payload.get("no_best_ask"))

            implied_yes_bid_from_no_pct = (100.0 - no_ask_pct) if no_ask_pct is not None else None
            implied_yes_ask_from_no_pct = (100.0 - no_bid_pct) if no_bid_pct is not None else None

            bid_candidates = [v for v in [bid_pct_direct, implied_yes_bid_from_no_pct] if v is not None]
            ask_candidates = [v for v in [ask_pct_direct, implied_yes_ask_from_no_pct] if v is not None]
            bid_pct = max(bid_candidates) if bid_candidates else None
            ask_pct = min(ask_candidates) if ask_candidates else None

            ref_pct = ReplaySession._normalize_probability_pct(
                payload.get("yes_price") if payload.get("yes_price") is not None else payload.get("reference")
            )
            selected_pct = select_probability_pct_from_quotes(
                best_bid_pct=bid_pct,
                best_ask_pct=ask_pct,
                mid_pct=mid_pct,
                reference_pct=ref_pct,
            )
            if selected_pct is None:
                continue
            rows.append({
                "label": str(label),
                # Backward-compatible field name used by UI payload.
                "mid_pct": float(selected_pct),
                "midpoint_f": ReplaySession._market_label_midpoint_f(label),
            })
        rows.sort(key=lambda r: float(r.get("mid_pct", 0.0)), reverse=True)
        return rows[:3]

    @staticmethod
    def _extract_pws_top_weight_station_temp_f(
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {
                "station_id": None,
                "temp_f": None,
                "weight": None,
                "source": None,
            }

        ranked = ReplaySession._extract_station_learning_rows(data, eligible_only=True)
        if not ranked:
            ranked = ReplaySession._extract_station_learning_rows(data, eligible_only=False)
        if not ranked:
            return {
                "station_id": None,
                "temp_f": None,
                "weight": None,
                "source": None,
            }

        leader = ranked[0]
        sid = str(leader.get("station_id") or "").upper()
        if not sid:
            return {
                "station_id": None,
                "temp_f": None,
                "weight": None,
                "source": None,
            }

        reading_row = None
        for raw in data.get("pws_readings") or []:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("station_id") or raw.get("label") or "").upper() == sid:
                reading_row = raw
                break
        if reading_row is None:
            for raw in data.get("pws_outliers") or []:
                if not isinstance(raw, dict):
                    continue
                if str(raw.get("station_id") or raw.get("label") or "").upper() == sid:
                    reading_row = raw
                    break

        temp_f = None
        if isinstance(reading_row, dict):
            try:
                tf = float(reading_row.get("temp_f"))
                if math.isfinite(tf):
                    temp_f = tf
            except Exception:
                temp_f = None
            if temp_f is None:
                try:
                    tc = float(reading_row.get("temp_c"))
                    if math.isfinite(tc):
                        temp_f = (tc * 9.0 / 5.0) + 32.0
                except Exception:
                    temp_f = None

        return {
            "station_id": sid,
            "temp_f": round(float(temp_f), 4) if temp_f is not None and math.isfinite(float(temp_f)) else None,
            "weight": leader.get("weight"),
            "source": (reading_row or {}).get("source"),
        }

    @staticmethod
    def _build_pws_consensus_snapshot(
        event: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
        ranked_rows: Optional[List[Dict[str, Any]]] = None,
        all_rows: Optional[List[Dict[str, Any]]] = None,
        sample_baseline_by_station: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a station-level PWS snapshot for the current replay tick.

        Snapshot includes per-station temperatures (when recorded) and merged
        long-horizon learning fields (scores/samples/EMA errors).
        """
        if not isinstance(event, dict) or not isinstance(data, dict):
            return None

        def _as_float(value: Any) -> Optional[float]:
            try:
                n = float(value)
            except Exception:
                return None
            if not math.isfinite(n):
                return None
            return n

        def _as_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except Exception:
                return None

        def _as_iso(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.isoformat()
            txt = str(value).strip()
            return txt or None

        learning_by_station: Dict[str, Dict[str, Any]] = {}
        for row in (all_rows or []):
            if not isinstance(row, dict):
                continue
            sid = str(row.get("station_id") or "").upper()
            if sid:
                learning_by_station[sid] = row

        leader_row: Optional[Dict[str, Any]] = None
        if ranked_rows:
            leader_row = ranked_rows[0]
        elif all_rows:
            def _weight_value(row: Dict[str, Any]) -> float:
                try:
                    return float(row.get("weight", 0.0) or 0.0)
                except Exception:
                    return 0.0

            sorted_all = sorted(
                [r for r in all_rows if isinstance(r, dict)],
                key=_weight_value,
                reverse=True,
            )
            leader_row = sorted_all[0] if sorted_all else None

        leader_station_id = str((leader_row or {}).get("station_id") or "").upper() or None

        def _merge_reading_row(raw_row: Dict[str, Any], in_consensus: bool) -> Optional[Dict[str, Any]]:
            sid = str(raw_row.get("station_id") or raw_row.get("label") or "").upper()
            if not sid:
                return None
            learning = learning_by_station.get(sid, {})

            def _pick(key: str) -> Any:
                value = raw_row.get(key)
                if value is None and isinstance(learning, dict):
                    value = learning.get(key)
                return value

            temp_f = _as_float(_pick("temp_f"))
            temp_c = _as_float(_pick("temp_c"))
            if temp_f is None and temp_c is not None:
                temp_f = round((temp_c * 9.0 / 5.0) + 32.0, 3)
            if temp_c is None and temp_f is not None:
                temp_c = round((temp_f - 32.0) * (5.0 / 9.0), 3)

            valid_raw = _pick("valid")
            if isinstance(valid_raw, bool):
                valid = valid_raw
            elif valid_raw is None:
                valid = bool(in_consensus)
            else:
                valid = bool(valid_raw)

            now_samples = _as_int(_pick("now_samples"))
            lead_samples = _as_int(_pick("lead_samples"))
            samples_total = _as_int(_pick("samples_total"))
            if samples_total is None and (now_samples is not None or lead_samples is not None):
                samples_total = int(max(0, now_samples or 0) + max(0, lead_samples or 0))

            baseline = (sample_baseline_by_station or {}).get(sid, {})
            baseline_now = _as_int(baseline.get("now_samples"))
            baseline_lead = _as_int(baseline.get("lead_samples"))
            now_samples_session = None
            lead_samples_session = None
            if now_samples is not None and baseline_now is not None:
                now_samples_session = max(0, int(now_samples) - max(0, int(baseline_now or 0)))
            if lead_samples is not None and baseline_lead is not None:
                lead_samples_session = max(0, int(lead_samples) - max(0, int(baseline_lead or 0)))

            row = {
                "station_id": sid,
                "station_name": _pick("station_name"),
                "source": _pick("source"),
                "in_consensus": bool(in_consensus),
                "valid": bool(valid),
                "qc_flag": _pick("qc_flag"),
                "excluded_reason": None if in_consensus else (_pick("qc_flag") or "EXCLUDED"),
                "obs_time_utc": _as_iso(_pick("obs_time_utc")),
                "distance_km": _as_float(_pick("distance_km")),
                "age_minutes": _as_float(_pick("age_minutes")),
                "temp_c": temp_c,
                "temp_f": temp_f,
                "weight": _as_float(_pick("weight")),
                "weight_now": _as_float(_pick("weight_now")),
                "weight_predictive": _as_float(_pick("weight_predictive")),
                "now_score": _as_float(_pick("now_score")),
                "lead_score": _as_float(_pick("lead_score")),
                "predictive_score": _as_float(_pick("predictive_score")),
                "now_samples": now_samples,
                "lead_samples": lead_samples,
                "samples_total": samples_total,
                "now_samples_session": now_samples_session,
                "lead_samples_session": lead_samples_session,
                "now_ema_abs_error_c": _as_float(_pick("now_ema_abs_error_c")),
                "lead_ema_abs_error_c": _as_float(_pick("lead_ema_abs_error_c")),
                "quality_band": _pick("quality_band"),
                "learning_phase": _pick("learning_phase"),
                "rank_eligible": ReplaySession._is_rank_eligible(
                    {
                        "rank_eligible": _pick("rank_eligible"),
                        "now_samples": now_samples,
                        "lead_samples": lead_samples,
                        "rank_min_now_samples": _pick("rank_min_now_samples"),
                        "rank_min_lead_samples": _pick("rank_min_lead_samples"),
                    }
                ),
            }
            return row

        merged_rows: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for raw in data.get("pws_readings") or []:
            if not isinstance(raw, dict):
                continue
            row = _merge_reading_row(raw, in_consensus=True)
            if not row:
                continue
            sid = str(row.get("station_id") or "")
            if sid in seen:
                continue
            seen.add(sid)
            merged_rows.append(row)

        for raw in data.get("pws_outliers") or []:
            if not isinstance(raw, dict):
                continue
            row = _merge_reading_row(raw, in_consensus=False)
            if not row:
                continue
            sid = str(row.get("station_id") or "")
            if sid in seen:
                continue
            seen.add(sid)
            merged_rows.append(row)

        # Legacy fallback: if replay data predates pws_readings serialization,
        # show station list from learning rows (no per-tick temperature available).
        if not merged_rows and learning_by_station:
            for sid, lrow in learning_by_station.items():
                row = _merge_reading_row({"station_id": sid, "source": lrow.get("source")}, in_consensus=False)
                if row:
                    row["excluded_reason"] = "NO_READING_AT_TICK"
                    merged_rows.append(row)

        def _sort_key(row: Dict[str, Any]) -> tuple:
            sid = str(row.get("station_id") or "")
            in_consensus_rank = 0 if bool(row.get("in_consensus")) else 1
            leader_rank = 0 if (leader_station_id and sid == leader_station_id) else 1
            weight = _as_float(row.get("weight"))
            hist = _as_float(row.get("predictive_score"))
            return (
                in_consensus_rank,
                leader_rank,
                -(weight if weight is not None else -1.0),
                -(hist if hist is not None else -1.0),
                sid,
            )

        merged_rows.sort(key=_sort_key)

        hist_row: Optional[Dict[str, Any]] = None
        for row in merged_rows:
            score = _as_float(row.get("predictive_score"))
            if score is None:
                continue
            if hist_row is None:
                hist_row = row
                continue
            cur_best = _as_float(hist_row.get("predictive_score"))
            if cur_best is None:
                hist_row = row
                continue
            if score > cur_best:
                hist_row = row
            elif score == cur_best:
                cur_n = _as_int(hist_row.get("samples_total")) or 0
                row_n = _as_int(row.get("samples_total")) or 0
                if row_n > cur_n:
                    hist_row = row

        in_consensus_count = sum(1 for r in merged_rows if bool(r.get("in_consensus")))
        excluded_count = max(0, len(merged_rows) - in_consensus_count)

        market_station_id = str(event.get("station_id") or "").upper() or None
        return {
            "event_index": event.get("event_index"),
            "ts_nyc": event.get("ts_nyc"),
            "ts_ingest_utc": _as_iso(event.get("ts_ingest_utc")),
            "obs_time_utc": _as_iso(event.get("obs_time_utc")),
            "sample_scope": "historical_cumulative",
            "market_station_id": market_station_id,
            "median_f": _as_float(data.get("median_f")),
            "mad_f": _as_float(data.get("mad_f")),
            "support": _as_int(data.get("support")),
            "weighted_support": _as_float(data.get("weighted_support")),
            "qc": data.get("qc"),
            "leader_station_id": leader_station_id,
            "leader_weight": _as_float((leader_row or {}).get("weight")),
            "leader_now_score": _as_float((leader_row or {}).get("now_score")),
            "leader_lead_score": _as_float((leader_row or {}).get("lead_score")),
            "top_historical_station_id": str((hist_row or {}).get("station_id") or "").upper() or None,
            "top_historical_score": _as_float((hist_row or {}).get("predictive_score")),
            "top_historical_samples_total": _as_int((hist_row or {}).get("samples_total")),
            "in_consensus_count": in_consensus_count,
            "excluded_count": excluded_count,
            "rows": merged_rows,
        }

    def get_pws_learning_summary(
        self,
        max_points: int = 80,
        top_n: int = 6,
        trend_points: int = 240,
        trend_mode: str = "hourly",
    ) -> Dict[str, Any]:
        """
        Build replay-time PWS learning diagnostics up to current playback index.

        Returns:
        - current: latest top ranking snapshot
        - timeline: decimated leader evolution over the day
        - leader_changes: compact list when leader station switches
        - trend: multi-source series for replay chart overlay
        """
        def _as_float(value: Any) -> Optional[float]:
            try:
                n = float(value)
            except Exception:
                return None
            if not math.isfinite(n):
                return None
            return n

        current_limit = min(self._event_index + 1, len(self._events))
        trend_mode_norm = str(trend_mode or "hourly").strip().lower()
        if trend_mode_norm in {"5m", "5min", "5_min", "five_min"}:
            trend_mode_norm = "5min"
        if trend_mode_norm not in {"hourly", "event", "5min"}:
            trend_mode_norm = "hourly"
        cache_key = (current_limit, int(max_points), int(top_n), int(trend_points), trend_mode_norm)
        if self._pws_learning_cache_key == cache_key and self._pws_learning_cache_value is not None:
            return self._pws_learning_cache_value
        pws_count = 0
        points: List[Dict[str, Any]] = []
        leader_changes: List[Dict[str, Any]] = []
        latest_world_by_station: Dict[str, Dict[str, Any]] = {}
        sample_baseline_by_station: Dict[str, Dict[str, int]] = {}

        latest_any_rows: List[Dict[str, Any]] = []
        latest_any_data: Optional[Dict[str, Any]] = None
        latest_any_event: Optional[Dict[str, Any]] = None
        latest_any_idx: Optional[int] = None

        latest_rows: List[Dict[str, Any]] = []
        latest_data: Optional[Dict[str, Any]] = None
        latest_event: Optional[Dict[str, Any]] = None
        prev_leader: Optional[str] = None
        l2_aliases = {"l2_snap", "l2_snap_1s", "market"}
        trend_points_raw: List[Dict[str, Any]] = []
        trend_state: Dict[str, Any] = {
            "pm_top1_prob_pct": None,
            "pm_top2_prob_pct": None,
            "pm_top3_prob_pct": None,
            "pm_top1_label": None,
            "pm_top2_label": None,
            "pm_top3_label": None,
            "pm_top1_midpoint_f": None,
            "pm_top2_midpoint_f": None,
            "pm_top3_midpoint_f": None,
            "metar_temp_f": None,
            "pws_consensus_f": None,
            "pws_top_weight_station_id": None,
            "pws_top_weight_temp_f": None,
            "pws_top_weight": None,
            "helios_prediction_f": None,
        }

        def _append_trend_point(idx: int, event: Dict[str, Any]) -> None:
            point = {
                "event_index": idx,
                "ts_nyc": event.get("ts_nyc"),
                "ts_ingest_utc": event.get("ts_ingest_utc"),
                **trend_state,
            }
            trend_points_raw.append(point)

        def _set_trend(key: str, value: Any) -> bool:
            if trend_state.get(key) == value:
                return False
            trend_state[key] = value
            return True

        def _build_metar_context(market_station_id: Optional[str], median_f: Any) -> Dict[str, Any]:
            sid = str(market_station_id or "").upper()
            world = latest_world_by_station.get(sid, {})
            metar_temp = _as_float(world.get("temp_f"))
            pws_median = _as_float(median_f)
            delta = None
            if metar_temp is not None and pws_median is not None:
                delta = round(pws_median - metar_temp, 3)

            return {
                "market_station_id": sid or None,
                "metar_temp_f": metar_temp,
                "metar_aligned_f": _as_float(world.get("temp_aligned_f")),
                "metar_ts_nyc": world.get("ts_nyc"),
                "metar_source": world.get("source"),
                "metar_report_type": world.get("report_type"),
                "metar_is_speci": bool(world.get("is_speci")),
                "pws_vs_metar_f": delta,
            }

        for idx in range(current_limit):
            event = self._events[idx]
            ch = event.get("ch")
            data = event.get("data", {})
            if not isinstance(data, dict):
                data = {}

            trend_changed = False

            if ch == "world":
                station_id = str(event.get("station_id") or "").upper()
                if station_id:
                    latest_world_by_station[station_id] = {
                        "ts_nyc": event.get("ts_nyc"),
                        "temp_f": data.get("temp_f"),
                        "temp_aligned_f": data.get("temp_aligned"),
                        "source": data.get("src"),
                        "report_type": data.get("report_type"),
                        "is_speci": data.get("is_speci"),
                    }
                    src = str(data.get("src") or "").upper()
                    if src == "METAR":
                        metar_temp = _as_float(data.get("temp_f"))
                        if metar_temp is not None:
                            trend_changed = _set_trend("metar_temp_f", metar_temp) or trend_changed

            if ch in l2_aliases:
                top3 = self._extract_market_top3(data)
                vals = [None, None, None]
                labels = [None, None, None]
                mids = [None, None, None]
                for i, row in enumerate(top3[:3]):
                    vals[i] = row.get("mid_pct")
                    labels[i] = row.get("label")
                    mids[i] = row.get("midpoint_f")
                trend_changed = _set_trend("pm_top1_prob_pct", vals[0]) or trend_changed
                trend_changed = _set_trend("pm_top2_prob_pct", vals[1]) or trend_changed
                trend_changed = _set_trend("pm_top3_prob_pct", vals[2]) or trend_changed
                trend_changed = _set_trend("pm_top1_label", labels[0]) or trend_changed
                trend_changed = _set_trend("pm_top2_label", labels[1]) or trend_changed
                trend_changed = _set_trend("pm_top3_label", labels[2]) or trend_changed
                trend_changed = _set_trend("pm_top1_midpoint_f", _as_float(mids[0])) or trend_changed
                trend_changed = _set_trend("pm_top2_midpoint_f", _as_float(mids[1])) or trend_changed
                trend_changed = _set_trend("pm_top3_midpoint_f", _as_float(mids[2])) or trend_changed

            if ch == "nowcast":
                pred = _as_float(data.get("tmax_mean_f"))
                if pred is not None:
                    trend_changed = _set_trend("helios_prediction_f", pred) or trend_changed

            if ch == "pws":
                pws_median = _as_float(data.get("median_f"))
                if pws_median is not None:
                    trend_changed = _set_trend("pws_consensus_f", pws_median) or trend_changed

                top_weight = self._extract_pws_top_weight_station_temp_f(data)
                trend_changed = _set_trend(
                    "pws_top_weight_station_id",
                    top_weight.get("station_id"),
                ) or trend_changed
                trend_changed = _set_trend(
                    "pws_top_weight_temp_f",
                    _as_float(top_weight.get("temp_f")),
                ) or trend_changed
                trend_changed = _set_trend(
                    "pws_top_weight",
                    _as_float(top_weight.get("weight")),
                ) or trend_changed

                all_rows = self._extract_station_learning_rows(data, eligible_only=False)
                if all_rows:
                    for row in all_rows:
                        if not isinstance(row, dict):
                            continue
                        sid = str(row.get("station_id") or "").upper()
                        if not sid:
                            continue
                        entry = sample_baseline_by_station.get(sid, {})
                        try:
                            now_n = int(row.get("now_samples"))
                        except Exception:
                            now_n = None
                        try:
                            lead_n = int(row.get("lead_samples"))
                        except Exception:
                            lead_n = None
                        if now_n is not None and "now_samples" not in entry:
                            entry["now_samples"] = max(0, now_n)
                        if lead_n is not None and "lead_samples" not in entry:
                            entry["lead_samples"] = max(0, lead_n)
                        if entry:
                            sample_baseline_by_station[sid] = entry

                    pws_count += 1
                    latest_any_rows = all_rows
                    latest_any_data = data
                    latest_any_event = event
                    latest_any_idx = idx

                    rows = self._extract_station_learning_rows(data, eligible_only=True)
                    if rows:
                        leader = rows[0]
                        leader_id = str(leader.get("station_id") or "")
                        market_station_id = str(event.get("station_id") or "").upper()
                        point = {
                            "event_index": idx,
                            "ts_nyc": event.get("ts_nyc"),
                            "ts_ingest_utc": event.get("ts_ingest_utc"),
                            "market_station_id": market_station_id or None,
                            "leader_station_id": leader_id,
                            "leader_weight": leader.get("weight"),
                            "leader_now_score": leader.get("now_score"),
                            "leader_lead_score": leader.get("lead_score"),
                            "weighted_support": data.get("weighted_support"),
                            "support": data.get("support"),
                            "median_f": data.get("median_f"),
                            **_build_metar_context(market_station_id, data.get("median_f")),
                        }
                        points.append(point)

                        if leader_id and leader_id != prev_leader:
                            leader_changes.append(point)
                            prev_leader = leader_id

                        latest_rows = rows
                        latest_data = data
                        latest_event = event

            if trend_changed:
                _append_trend_point(idx, event)

        def _as_dt_utc(value: Any) -> Optional[datetime]:
            if value is None:
                return None
            if isinstance(value, datetime):
                dt = value if value.tzinfo else value.replace(tzinfo=UTC)
                return dt.astimezone(UTC)
            txt = str(value).strip()
            if not txt:
                return None
            try:
                dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt.astimezone(UTC)
            except Exception:
                return None

        trend_series_points = list(trend_points_raw)
        trend_bin_tz = self._station_tz if self.station_id else NYC
        if trend_mode_norm in {"hourly", "5min"} and trend_points_raw:
            bucket_order: List[str] = []
            bucket_last: Dict[str, Dict[str, Any]] = {}
            for p in trend_points_raw:
                point_ts_utc = _as_dt_utc(p.get("ts_ingest_utc"))
                if point_ts_utc is None:
                    continue
                local_ts = point_ts_utc.astimezone(trend_bin_tz)
                if trend_mode_norm == "5min":
                    bucket_minute = (local_ts.minute // 5) * 5
                    local_bucket = local_ts.replace(minute=bucket_minute, second=0, microsecond=0)
                else:
                    local_bucket = local_ts.replace(minute=0, second=0, microsecond=0)
                bucket_key = local_bucket.isoformat()
                if bucket_key not in bucket_last:
                    bucket_order.append(bucket_key)

                row = dict(p)
                row["bucket_local_hour"] = local_bucket.strftime("%Y-%m-%d %H:00")
                row["bucket_local_5m"] = local_bucket.strftime("%Y-%m-%d %H:%M")
                row["bucket_timezone"] = trend_bin_tz.key
                row["ts_nyc"] = local_bucket.astimezone(NYC).strftime("%Y-%m-%d %H:%M:%S")
                row["ts_ingest_utc"] = local_bucket.astimezone(UTC).isoformat()
                bucket_last[bucket_key] = row

            trend_series_points = [bucket_last[k] for k in bucket_order if k in bucket_last]

        trend_points_final = self._decimate_points(
            trend_series_points,
            max(40, int(trend_points)),
        )
        trend_payload = {
            "series_version": "replay_market_pws_v1",
            "resolution": trend_mode_norm,
            "bin_timezone": trend_bin_tz.key,
            "total_points_raw": len(trend_points_raw),
            "total_points": len(trend_series_points),
            "points": trend_points_final,
            "latest": trend_series_points[-1] if trend_series_points else None,
        }

        if pws_count == 0:
            result = {
                "total_pws_events": 0,
                "ranked_pws_events": 0,
                "current": None,
                "consensus_snapshot": None,
                "trend": trend_payload,
                "timeline": [],
                "leader_changes": [],
            }
            self._pws_learning_cache_key = cache_key
            self._pws_learning_cache_value = result
            return result

        latest_any_event_with_index = dict(latest_any_event or {})
        latest_any_event_with_index["event_index"] = latest_any_idx
        consensus_snapshot = self._build_pws_consensus_snapshot(
            latest_any_event_with_index if latest_any_event else None,
            latest_any_data,
            ranked_rows=latest_rows,
            all_rows=latest_any_rows,
            sample_baseline_by_station=sample_baseline_by_station,
        )

        if not latest_rows:
            rank_min_now = 2
            rank_min_lead = 1
            if latest_any_rows:
                rank_min_now = max(0, int(latest_any_rows[0].get("rank_min_now_samples") or 2))
                rank_min_lead = max(0, int(latest_any_rows[0].get("rank_min_lead_samples") or 1))

            current_payload = {
                "event_index": None,
                "ts_nyc": latest_any_event.get("ts_nyc") if latest_any_event else None,
                "weighted_support": latest_any_data.get("weighted_support") if latest_any_data else None,
                "support": latest_any_data.get("support") if latest_any_data else None,
                "median_f": latest_any_data.get("median_f") if latest_any_data else None,
                **_build_metar_context(
                    latest_any_event.get("station_id") if latest_any_event else None,
                    latest_any_data.get("median_f") if latest_any_data else None,
                ),
                "status": "WARMUP",
                "warmup_reason": (
                    f"Ranking hidden until station has now>={rank_min_now} and lead>={rank_min_lead} samples."
                ),
                "rank_min_now_samples": rank_min_now,
                "rank_min_lead_samples": rank_min_lead,
                "top": [],
            }
            result = {
                "total_pws_events": pws_count,
                "ranked_pws_events": 0,
                "current": current_payload,
                "consensus_snapshot": consensus_snapshot,
                "trend": trend_payload,
                "timeline": [],
                "leader_changes": [],
            }
            self._pws_learning_cache_key = cache_key
            self._pws_learning_cache_value = result
            return result

        current_top = latest_rows[: max(1, int(top_n))]
        current_payload = {
            "event_index": points[-1]["event_index"] if points else None,
            "ts_nyc": latest_event.get("ts_nyc") if latest_event else None,
            "weighted_support": latest_data.get("weighted_support") if latest_data else None,
            "support": latest_data.get("support") if latest_data else None,
            "median_f": latest_data.get("median_f") if latest_data else None,
            **_build_metar_context(
                latest_event.get("station_id") if latest_event else None,
                latest_data.get("median_f") if latest_data else None,
            ),
            "status": "READY",
            "warmup_reason": "",
            "top": current_top,
        }

        result = {
            "total_pws_events": pws_count,
            "ranked_pws_events": len(points),
            "current": current_payload,
            "consensus_snapshot": consensus_snapshot,
            "trend": trend_payload,
            "timeline": self._decimate_points(points, max(10, int(max_points))),
            "leader_changes": leader_changes[-20:],
        }
        self._pws_learning_cache_key = cache_key
        self._pws_learning_cache_value = result
        return result

    def get_current_event(self) -> Optional[Dict]:
        """Get the current event."""
        if 0 <= self._event_index < len(self._events):
            return self._events[self._event_index]
        return None

    def get_events_around(self, n: int = 5) -> Dict:
        """Get events around current position."""
        start = max(0, self._event_index - n)
        end = min(len(self._events), self._event_index + n + 1)

        return {
            "before": self._events[start:self._event_index],
            "current": self.get_current_event(),
            "after": self._events[self._event_index + 1:end] if self._event_index + 1 < len(self._events) else []
        }


class ReplayEngine:
    """
    Main replay engine manager.

    Manages multiple replay sessions and provides
    the same event protocol as live feeds.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ReplayEngine, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        self._sessions: Dict[str, ReplaySession] = {}
        self._active_session_id: Optional[str] = None
        self._reader = get_hybrid_reader()

        # Global event callback (for broadcasting to WS clients)
        self._broadcast_callback: Optional[Callable[[Dict], None]] = None

        self.initialized = True
        logger.info("ReplayEngine initialized")

    def set_broadcast_callback(self, callback: Callable[[Dict], None]):
        """Set callback for broadcasting events to all clients."""
        self._broadcast_callback = callback

    def list_available_dates(self, station_id: Optional[str] = None) -> List[str]:
        """List dates with recorded data."""
        raw_dates = self._reader.list_available_dates(station_id)
        if not station_id:
            return raw_dates

        station_tz = _resolve_station_timezone(station_id)
        local_dates = set()
        for raw_date in raw_dates:
            local_dates.update(_map_partition_date_to_station_market_dates(raw_date, station_tz))
        return sorted(local_dates, reverse=True)

    def list_channels_for_date(
        self,
        date_str: str,
        station_id: Optional[str] = None
    ) -> List[str]:
        """List channels available for a date."""
        if not station_id:
            return self._reader.list_channels_for_date(date_str, station_id)

        station_tz = _resolve_station_timezone(station_id)
        source_dates = _source_partition_dates_for_local_day(date_str, station_tz)
        channels = set()
        for source_date in source_dates:
            channels.update(self._reader.list_channels_for_date(source_date, station_id))
        return list(channels)

    async def create_session(
        self,
        date_str: str,
        station_id: Optional[str] = None,
        channels: Optional[List[str]] = None
    ) -> Optional[ReplaySession]:
        """Create and load a new replay session."""
        import uuid
        session_id = str(uuid.uuid4())[:8]

        session = ReplaySession(
            session_id=session_id,
            date_str=date_str,
            station_id=station_id,
            channels=channels
        )

        # Set up event routing
        session.set_event_callback(self._on_event)

        # Load data
        if not await session.load():
            return None

        self._sessions[session_id] = session
        self._active_session_id = session_id

        return session

    def get_session(self, session_id: str) -> Optional[ReplaySession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_active_session(self) -> Optional[ReplaySession]:
        """Get the active session."""
        if self._active_session_id:
            return self._sessions.get(self._active_session_id)
        return None

    async def close_session(self, session_id: str):
        """Close and remove a session."""
        session = self._sessions.get(session_id)
        if session:
            await session.stop()
            del self._sessions[session_id]

            if self._active_session_id == session_id:
                self._active_session_id = None

    def _on_event(self, event: Dict):
        """Handle event from session playback."""
        if self._broadcast_callback:
            # Transform to WS-compatible format
            ws_event = {
                "type": event.get("ch", "unknown"),
                "replay": True,
                "data": event
            }
            self._broadcast_callback(ws_event)

    def get_state(self) -> Dict:
        """Get engine state."""
        return {
            "sessions": {
                sid: s.get_state() for sid, s in self._sessions.items()
            },
            "active_session_id": self._active_session_id,
            "available_dates": self.list_available_dates()
        }

    # =========================================================================
    # Convenience methods for active session
    # =========================================================================

    async def play(self, speed: float = 1.0):
        """Play active session."""
        session = self.get_active_session()
        if session:
            await session.play(speed)

    async def pause(self):
        """Pause active session."""
        session = self.get_active_session()
        if session:
            await session.pause()

    async def stop(self):
        """Stop active session."""
        session = self.get_active_session()
        if session:
            await session.stop()

    def seek_percent(self, percent: float):
        """Seek active session to percentage."""
        session = self.get_active_session()
        if session:
            session.seek_percent(percent)

    def jump_next_metar(self) -> bool:
        """Jump to next METAR in active session."""
        session = self.get_active_session()
        if session:
            return session.jump_to_next_metar()
        return False

    def jump_prev_metar(self) -> bool:
        """Jump to previous METAR in active session."""
        session = self.get_active_session()
        if session:
            return session.jump_to_prev_metar()
        return False

    def jump_next_window(self) -> bool:
        """Jump to next event window in active session."""
        session = self.get_active_session()
        if session:
            return session.jump_to_next_window()
        return False


# =============================================================================
# Global accessor
# =============================================================================

_engine: Optional[ReplayEngine] = None


def get_replay_engine() -> ReplayEngine:
    """Get the global ReplayEngine instance."""
    global _engine
    if _engine is None:
        _engine = ReplayEngine()
    return _engine
