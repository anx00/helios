"""
Recorder - Phase 4 (Section 4.5)

NDJSON append-only recording for all HELIOS events.
Does NOT block the live path - writes are batched and async.

Channels:
- world: METAR observations
- pws: PWS cluster consensus
- features: Environment features
- nowcast: Model outputs
- health: System health metrics
- l2_snap: Market snapshots (when available)
- event_window: Event window markers
"""

import asyncio
import logging
import os
import uuid
from collections import deque
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Deque
from zoneinfo import ZoneInfo

try:
    import orjson
    def json_dumps(obj):
        return orjson.dumps(obj).decode('utf-8')
except ImportError:
    import json
    def json_dumps(obj):
        return json.dumps(obj, default=str)

logger = logging.getLogger("recorder")

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")
MAD = ZoneInfo("Europe/Madrid")

# Schema version for all events
SCHEMA_VERSION = 1
DEFAULT_CHANNELS = [
    "world",
    "pws",
    "features",
    "nowcast",
    "l2_snap",
    "health",
    "event_window",
]
DEFAULT_PERSIST_CHANNELS = [ch for ch in DEFAULT_CHANNELS if ch != "l2_snap"]


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except ValueError:
        return int(default)


def _env_channel_list(name: str, default: List[str]) -> List[str]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return list(default)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {ch for ch in DEFAULT_CHANNELS}
    selected = [ch for ch in requested if ch in allowed]
    return selected or list(default)


def _compact_l2_market_state(market_state: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for bracket, payload in (market_state or {}).items():
        if bracket == "__meta__":
            compact[bracket] = dict(payload or {})
            continue
        if not isinstance(payload, dict):
            compact[bracket] = payload
            continue
        reduced: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in {"bids", "asks", "yes_bids", "yes_asks", "no_bids", "no_asks"}:
                continue
            reduced[key] = value
        compact[bracket] = reduced
    return compact


class RingBuffer:
    """
    In-memory ring buffer for recent events.
    Used for live UI without hitting disk.
    """
    def __init__(self, max_size: int = 1000):
        self._buffer: Deque[Dict] = deque(maxlen=max_size)
        self._max_size = max_size

    def append(self, event: Dict):
        self._buffer.append(event)

    def get_recent(self, n: int = 100) -> List[Dict]:
        """Get last N events."""
        return list(self._buffer)[-n:]

    def get_since(self, ts_utc: datetime) -> List[Dict]:
        """Get events since timestamp."""
        result = []
        for event in self._buffer:
            event_ts = event.get("ts_ingest_utc")
            if event_ts and event_ts >= ts_utc.isoformat():
                result.append(event)
        return result

    def clear(self):
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


class NDJSONWriter:
    """
    Async NDJSON file writer with batching.
    Writes are batched and flushed periodically to avoid blocking.
    """
    def __init__(
        self,
        base_path: Path,
        channel: str,
        flush_interval_seconds: float = 5.0,
        max_batch_size: int = 100
    ):
        self.base_path = base_path
        self.channel = channel
        self.flush_interval = flush_interval_seconds
        self.max_batch_size = max_batch_size

        self._batch: List[str] = []
        self._current_file: Optional[Path] = None
        self._current_date: Optional[date] = None
        self._file_handle = None
        self._lock = asyncio.Lock()
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None

        # Stats
        self._events_written = 0
        self._bytes_written = 0

    def _get_file_path(self, target_date: date) -> Path:
        """Get file path for a given date."""
        date_str = target_date.strftime("%Y-%m-%d")
        dir_path = self.base_path / f"date={date_str}" / f"ch={self.channel}"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / "events.ndjson"

    async def _ensure_file(self, target_date: date):
        """Ensure correct file is open for the date."""
        if self._current_date != target_date:
            await self._close_file()
            self._current_date = target_date
            self._current_file = self._get_file_path(target_date)
            self._file_handle = open(self._current_file, "a", encoding="utf-8")
            logger.debug(f"Opened file: {self._current_file}")

    async def _close_file(self):
        """Close current file."""
        if self._file_handle:
            try:
                self._file_handle.flush()
                self._file_handle.close()
            except Exception as e:
                logger.warning(f"Error closing file: {e}")
            self._file_handle = None

    async def write(self, event: Dict):
        """Queue an event for writing."""
        line = json_dumps(event) + "\n"
        async with self._lock:
            self._batch.append(line)

            # Flush if batch is full
            if len(self._batch) >= self.max_batch_size:
                await self._flush_batch()

    async def _flush_batch(self):
        """Flush current batch to disk."""
        if not self._batch:
            return

        # Get date from first event or use today
        today = datetime.now(NYC).date()
        await self._ensure_file(today)

        try:
            for line in self._batch:
                self._file_handle.write(line)
                self._bytes_written += len(line)
                self._events_written += 1

            self._file_handle.flush()
            self._batch.clear()

        except Exception as e:
            logger.error(f"Error flushing batch for {self.channel}: {e}")

    async def _flush_loop(self):
        """Background task for periodic flushing."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                async with self._lock:
                    await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush loop error: {e}")

    async def start(self):
        """Start the background flush task."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        """Stop and flush remaining data."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        async with self._lock:
            await self._flush_batch()
            await self._close_file()

    def get_stats(self) -> Dict:
        """Get writer statistics."""
        return {
            "channel": self.channel,
            "events_written": self._events_written,
            "bytes_written": self._bytes_written,
            "batch_pending": len(self._batch),
            "current_file": str(self._current_file) if self._current_file else None
        }


class Recorder:
    """
    Main recorder for HELIOS events.

    Maintains:
    - Ring buffers for live UI (in-memory)
    - NDJSON writers for persistence (async, non-blocking)

    Channels:
    - world: METAR observations
    - pws: PWS cluster data
    - features: Environment features
    - nowcast: Model outputs
    - l2_snap: Market L2 snapshots
    - health: System health
    - event_window: Event window markers
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Recorder, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, base_path: Optional[str] = None):
        if self.initialized:
            return

        self.base_path = Path(base_path or "data/recordings")
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Channel definitions
        self._channels = list(DEFAULT_CHANNELS)
        self._persist_channels = set(_env_channel_list("HELIOS_RECORDER_PERSIST_CHANNELS", DEFAULT_PERSIST_CHANNELS))
        default_ring_size = _env_int("HELIOS_RECORDER_RING_SIZE", 1000)
        l2_ring_size = _env_int("HELIOS_RECORDER_L2_RING_SIZE", 25)

        # Ring buffers (in-memory) for live UI
        self._ring_buffers: Dict[str, RingBuffer] = {
            ch: RingBuffer(max_size=(l2_ring_size if ch == "l2_snap" else default_ring_size))
            for ch in self._channels
        }

        # NDJSON writers
        self._writers: Dict[str, NDJSONWriter] = {
            ch: NDJSONWriter(self.base_path, ch) for ch in self._channels if ch in self._persist_channels
        }

        # Correlation tracking
        self._current_correlation_id: Optional[str] = None

        # Stats
        self._total_events = 0
        self._start_time = datetime.now(UTC)

        self._running = False
        self.initialized = True
        logger.info(
            "Recorder initialized at %s | persist_channels=%s",
            self.base_path,
            ",".join(sorted(self._persist_channels)),
        )

    def _make_event(
        self,
        channel: str,
        data: Dict,
        obs_time_utc: Optional[datetime] = None,
        station_id: Optional[str] = None,
        market_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        window_id: Optional[str] = None
    ) -> Dict:
        """
        Create a standardized event with all required fields.

        Required fields (per Section 4.6):
        - schema_version
        - event_id (uuid)
        - correlation_id
        - ts_ingest_utc, ts_nyc, ts_es
        - obs_time_utc (if applicable)
        """
        now_utc = datetime.now(UTC)
        now_nyc = now_utc.astimezone(NYC)
        now_es = now_utc.astimezone(MAD)

        event = {
            "schema_version": SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            "correlation_id": correlation_id or self._current_correlation_id or str(uuid.uuid4()),
            "ch": channel,
            "ts_ingest_utc": now_utc.isoformat(),
            "ts_nyc": now_nyc.strftime("%Y-%m-%d %H:%M:%S"),
            "ts_es": now_es.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if obs_time_utc:
            event["obs_time_utc"] = obs_time_utc.isoformat()

        if station_id:
            event["station_id"] = station_id

        if market_id:
            event["market_id"] = market_id

        if window_id:
            event["window_id"] = window_id

        # Merge data
        event["data"] = data

        return event

    async def record(
        self,
        channel: str,
        data: Dict,
        obs_time_utc: Optional[datetime] = None,
        station_id: Optional[str] = None,
        market_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        window_id: Optional[str] = None
    ):
        """
        Record an event to a channel.

        Non-blocking - writes to ring buffer immediately,
        queues for async NDJSON write.
        """
        if channel not in self._channels:
            logger.warning(f"Unknown channel: {channel}")
            return

        event = self._make_event(
            channel=channel,
            data=data,
            obs_time_utc=obs_time_utc,
            station_id=station_id,
            market_id=market_id,
            correlation_id=correlation_id,
            window_id=window_id
        )

        # Write to ring buffer (immediate, for live UI)
        self._ring_buffers[channel].append(event)

        # Queue for NDJSON write (async, non-blocking)
        if self._running:
            writer = self._writers.get(channel)
            if writer is not None:
                await writer.write(event)

        self._total_events += 1

    # =========================================================================
    # Convenience methods for specific channels
    # =========================================================================

    async def record_metar(
        self,
        station_id: str,
        raw: str,
        temp_c: float,
        temp_f: float,
        temp_aligned: float,
        obs_time_utc: datetime,
        dewpoint_c: Optional[float] = None,
        wind_dir: Optional[int] = None,
        wind_speed: Optional[float] = None,
        sky_condition: Optional[str] = None,
        qc_state: str = "OK",
        source: str = "METAR",
        report_type: str = "METAR",
        is_speci: bool = False,
        correlation_id: Optional[str] = None
    ):
        """Record a METAR observation (Section 4.3 B.1)."""
        data = {
            "src": source,
            "raw": raw,
            "temp_c": temp_c,
            "temp_f": temp_f,
            "temp_aligned": temp_aligned,
            "dewpoint_c": dewpoint_c,
            "wind_dir": wind_dir,
            "wind_speed": wind_speed,
            "sky": sky_condition,
            "qc": qc_state,
            "report_type": report_type,
            "is_speci": bool(is_speci),
        }
        await self.record(
            channel="world",
            data=data,
            obs_time_utc=obs_time_utc,
            station_id=station_id,
            correlation_id=correlation_id
        )

    async def record_pws(
        self,
        station_id: str,
        median_f: float,
        mad_f: float,
        support: int,
        pws_ids: List[str],
        qc_state: str,
        obs_time_utc: datetime,
        pws_all_ids: Optional[List[str]] = None,
        pws_readings: Optional[List[Dict[str, Any]]] = None,
        pws_outliers: Optional[List[Dict[str, Any]]] = None,
        station_weights: Optional[Dict[str, float]] = None,
        station_learning: Optional[Dict[str, Dict[str, Any]]] = None,
        weighted_support: Optional[float] = None,
        correlation_id: Optional[str] = None
    ):
        """Record PWS cluster consensus (Section 4.3 B.2)."""
        data = {
            "median_f": median_f,
            "mad_f": mad_f,
            "support": support,
            "pws_ids": pws_ids[:10],  # Limit to 10 IDs
            "qc": qc_state
        }
        if pws_all_ids is not None:
            data["pws_all_ids"] = list(pws_all_ids)
        if pws_readings is not None:
            data["pws_readings"] = list(pws_readings)
        if pws_outliers is not None:
            data["pws_outliers"] = list(pws_outliers)
        if station_weights is not None:
            data["station_weights"] = dict(station_weights)
        if station_learning is not None:
            data["station_learning"] = dict(station_learning)
        if weighted_support is not None:
            data["weighted_support"] = float(weighted_support)
        await self.record(
            channel="pws",
            data=data,
            obs_time_utc=obs_time_utc,
            station_id=station_id,
            correlation_id=correlation_id
        )

    async def record_features(
        self,
        station_id: str,
        features: Dict[str, Any],
        staleness: Dict[str, float],
        correlation_id: Optional[str] = None
    ):
        """Record feature vector (Section 4.3 C)."""
        data = {
            "features": features,
            "staleness": staleness
        }
        await self.record(
            channel="features",
            data=data,
            station_id=station_id,
            correlation_id=correlation_id
        )

    async def record_nowcast(
        self,
        station_id: str,
        tmax_mean_f: float,
        tmax_sigma_f: float,
        p_bucket: List[Dict],
        t_peak_bins: List[Dict],
        confidence: float,
        qc_state: str,
        bias_f: float,
        correlation_id: Optional[str] = None
    ):
        """Record nowcast output (Section 4.3 D)."""
        data = {
            "tmax_mean_f": tmax_mean_f,
            "tmax_sigma_f": tmax_sigma_f,
            "p_bucket": p_bucket,
            "t_peak_bins": t_peak_bins,
            "confidence": confidence,
            "qc": qc_state,
            "bias_f": bias_f
        }
        await self.record(
            channel="nowcast",
            data=data,
            station_id=station_id,
            correlation_id=correlation_id
        )

    async def record_l2_snap(
        self,
        station_id: str,
        market_state: Dict[str, Any],
        market_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ):
        """Record L2 market snapshot (Section 4.3 F)."""
        await self.record(
            channel="l2_snap",
            data=_compact_l2_market_state(market_state),
            station_id=station_id,
            market_id=market_id,
            correlation_id=correlation_id
        )

    async def record_health(
        self,
        station_id: str,
        latencies: Dict[str, float],
        sources_status: Dict[str, str],
        reconnects: int = 0,
        gaps: int = 0
    ):
        """Record health metrics (Section 4.3 E)."""
        data = {
            "latencies": latencies,
            "sources": sources_status,
            "reconnects": reconnects,
            "gaps": gaps
        }
        await self.record(
            channel="health",
            data=data,
            station_id=station_id
        )

    async def record_event_window(
        self,
        window_id: str,
        action: str,  # "START" or "END"
        reason: str,
        station_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ):
        """Record event window marker (Section 4.4)."""
        data = {
            "window_id": window_id,
            "action": action,
            "reason": reason
        }
        await self.record(
            channel="event_window",
            data=data,
            station_id=station_id,
            correlation_id=correlation_id,
            window_id=window_id
        )

    # =========================================================================
    # Correlation management
    # =========================================================================

    def start_correlation(self, reason: str = "") -> str:
        """Start a new correlation context. Returns correlation_id."""
        self._current_correlation_id = str(uuid.uuid4())
        logger.debug(f"Started correlation {self._current_correlation_id}: {reason}")
        return self._current_correlation_id

    def end_correlation(self):
        """End current correlation context."""
        self._current_correlation_id = None

    # =========================================================================
    # Query methods (for live UI)
    # =========================================================================

    def get_recent(self, channel: str, n: int = 100) -> List[Dict]:
        """Get recent events from ring buffer."""
        if channel not in self._ring_buffers:
            return []
        return self._ring_buffers[channel].get_recent(n)

    def get_all_recent(self, n: int = 50) -> Dict[str, List[Dict]]:
        """Get recent events from all channels."""
        return {
            ch: self._ring_buffers[ch].get_recent(n)
            for ch in self._channels
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self):
        """Start all writers."""
        self._running = True
        for writer in self._writers.values():
            await writer.start()
        logger.info("Recorder started")

    async def stop(self):
        """Stop all writers and flush."""
        self._running = False
        for writer in self._writers.values():
            await writer.stop()
        logger.info("Recorder stopped")

    def get_stats(self) -> Dict:
        """Get recorder statistics."""
        uptime = (datetime.now(UTC) - self._start_time).total_seconds()
        return {
            "running": self._running,
            "base_path": str(self.base_path),
            "persist_channels": sorted(self._persist_channels),
            "total_events": self._total_events,
            "uptime_seconds": uptime,
            "events_per_second": self._total_events / uptime if uptime > 0 else 0,
            "ring_buffer_sizes": {
                ch: len(buf) for ch, buf in self._ring_buffers.items()
            },
            "writers": {
                ch: writer.get_stats() for ch, writer in self._writers.items()
            }
        }

    def list_available_dates(self) -> List[str]:
        """List all dates with recorded data."""
        dates = set()
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name.startswith("date="):
                dates.add(item.name.replace("date=", ""))
        return sorted(dates)

    def list_available_channels(self, target_date: str) -> List[str]:
        """List channels available for a date."""
        date_path = self.base_path / f"date={target_date}"
        if not date_path.exists():
            return []

        channels = []
        for item in date_path.iterdir():
            if item.is_dir() and item.name.startswith("ch="):
                channels.append(item.name.replace("ch=", ""))
        return channels


# =============================================================================
# Global accessor
# =============================================================================

_recorder: Optional[Recorder] = None


def get_recorder(base_path: Optional[str] = None) -> Recorder:
    """Get the global Recorder instance."""
    global _recorder
    if _recorder is None:
        _recorder = Recorder(base_path)
    return _recorder


async def initialize_recorder(base_path: Optional[str] = None) -> Recorder:
    """Initialize and start the recorder."""
    recorder = get_recorder(base_path)
    await recorder.start()
    return recorder
