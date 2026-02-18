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
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Callable, Any
from zoneinfo import ZoneInfo

from core.compactor import get_hybrid_reader, HybridReader

logger = logging.getLogger("replay_engine")

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


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
            channels = self.channels
            # Default channel set optimized for UI/replay latency.
            # Prefer lower-volume market channel when multiple variants exist.
            if channels is None and hasattr(self._reader, "list_channels_for_date"):
                available = set(self._reader.list_channels_for_date(self.date_str, self.station_id))
                channels = ["world", "pws", "nowcast", "features", "health", "event_window"]
                if "l2_snap" in available:
                    channels.append("l2_snap")
                elif "market" in available:
                    channels.append("market")
                elif "l2_snap_1s" in available:
                    channels.append("l2_snap_1s")

            # Read all events sorted by timestamp
            self._events = self._reader.get_events_sorted(
                self.date_str,
                self.station_id,
                channels
            )

            if not self._events:
                # Diagnostic logging
                try:
                    available_channels = self._reader.list_channels_for_date(self.date_str)
                    available_dates = self._reader.list_available_dates(self.station_id)
                    logger.warning(
                        f"No events found for date={self.date_str}, station={self.station_id}. "
                        f"Channels on date: {available_channels}. "
                        f"Recent dates for station: {available_dates[:5]}"
                    )
                    if not available_channels:
                        logger.warning(
                            f"No NDJSON or Parquet data exists for {self.date_str}. "
                            f"Is the recorder running?"
                        )
                    elif self.station_id:
                        # Data exists but not for this station
                        all_events = self._reader.get_events_sorted(self.date_str, None, self.channels)
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

    def get_pws_learning_summary(self, max_points: int = 80, top_n: int = 6) -> Dict[str, Any]:
        """
        Build replay-time PWS learning diagnostics up to current playback index.

        Returns:
        - current: latest top ranking snapshot
        - timeline: decimated leader evolution over the day
        - leader_changes: compact list when leader station switches
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
        cache_key = (current_limit, int(max_points), int(top_n))
        if self._pws_learning_cache_key == cache_key and self._pws_learning_cache_value is not None:
            return self._pws_learning_cache_value
        pws_count = 0
        points: List[Dict[str, Any]] = []
        leader_changes: List[Dict[str, Any]] = []
        latest_world_by_station: Dict[str, Dict[str, Any]] = {}

        latest_any_rows: List[Dict[str, Any]] = []
        latest_any_data: Optional[Dict[str, Any]] = None
        latest_any_event: Optional[Dict[str, Any]] = None

        latest_rows: List[Dict[str, Any]] = []
        latest_data: Optional[Dict[str, Any]] = None
        latest_event: Optional[Dict[str, Any]] = None
        prev_leader: Optional[str] = None

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
                "pws_vs_metar_f": delta,
            }

        for idx in range(current_limit):
            event = self._events[idx]
            ch = event.get("ch")
            if ch == "world":
                station_id = str(event.get("station_id") or "").upper()
                data = event.get("data", {})
                if station_id and isinstance(data, dict):
                    latest_world_by_station[station_id] = {
                        "ts_nyc": event.get("ts_nyc"),
                        "temp_f": data.get("temp_f"),
                        "temp_aligned_f": data.get("temp_aligned"),
                        "source": data.get("src"),
                    }
                continue

            if ch != "pws":
                continue

            data = event.get("data", {})
            if not isinstance(data, dict):
                continue

            all_rows = self._extract_station_learning_rows(data, eligible_only=False)
            if not all_rows:
                continue

            pws_count += 1
            latest_any_rows = all_rows
            latest_any_data = data
            latest_any_event = event

            rows = self._extract_station_learning_rows(data, eligible_only=True)
            if not rows:
                continue

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

        if pws_count == 0:
            result = {
                "total_pws_events": 0,
                "ranked_pws_events": 0,
                "current": None,
                "timeline": [],
                "leader_changes": [],
            }
            self._pws_learning_cache_key = cache_key
            self._pws_learning_cache_value = result
            return result

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
        return self._reader.list_available_dates(station_id)

    def list_channels_for_date(
        self,
        date_str: str,
        station_id: Optional[str] = None
    ) -> List[str]:
        """List channels available for a date."""
        return self._reader.list_channels_for_date(date_str, station_id)

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
