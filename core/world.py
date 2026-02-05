import asyncio
import os
from typing import Dict, Optional, List, Deque, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import logging
import json

from .models import OfficialObs, AuxObs, EnvFeature, ObservationEvent, NYC, MAD, UTC

logger = logging.getLogger("world_state")

# NDJSON log directory (Phase 2, section 2.6/2.7)
NDJSON_LOG_DIR = Path(__file__).parent.parent / "logs"


@dataclass
class SourceHealth:
    """Tracks health/staleness for a single data source."""
    source_name: str
    last_seen_utc: Optional[datetime] = None
    last_obs_time_utc: Optional[datetime] = None
    update_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    last_latency_s: float = 0.0

    @property
    def staleness_s(self) -> float:
        """Seconds since last data received."""
        if not self.last_seen_utc:
            return float("inf")
        return (datetime.now(UTC) - self.last_seen_utc).total_seconds()

    @property
    def source_age_s(self) -> float:
        """Seconds between obs_time and ingest_time (how old the data was at arrival)."""
        return self.last_latency_s

    @property
    def status(self) -> str:
        staleness = self.staleness_s
        if staleness == float("inf"):
            return "NO_DATA"
        if staleness < 120:
            return "LIVE"
        if staleness < 600:
            return "OK"
        if staleness < 1800:
            return "STALE"
        return "DEAD"

    def to_dict(self) -> Dict:
        return {
            "source": self.source_name,
            "status": self.status,
            "staleness_s": round(self.staleness_s, 1) if self.staleness_s != float("inf") else None,
            "source_age_s": round(self.source_age_s, 1),
            "last_seen_utc": self.last_seen_utc.isoformat() if self.last_seen_utc else None,
            "last_obs_time_utc": self.last_obs_time_utc.isoformat() if self.last_obs_time_utc else None,
            "update_count": self.update_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


@dataclass
class QCState:
    """Global QC state summary."""
    status: str = "OK"  # OK, UNCERTAIN, OUTLIER, SEVERE
    flags: List[str] = field(default_factory=list)
    last_check_utc: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "flags": self.flags,
            "last_check_utc": self.last_check_utc.isoformat() if self.last_check_utc else None,
        }


class WorldState:
    """
    Singleton In-Memory State Logic (The 'World Tape').
    Phase 2: Full event-driven state with health tracking, QC, and SSE support.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WorldState, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        # Current State (The 'Head')
        self.latest_official: Dict[str, OfficialObs] = {}  # station_id -> Obs
        self.latest_aux: Dict[str, AuxObs] = {}            # station_id -> Obs
        self.env_features: Dict[str, EnvFeature] = {}      # type -> Feature

        # PWS individual station details (for UI breakdown)
        self.pws_details: Dict[str, List[Dict]] = {}       # station_id -> list of readings

        # The Tape (Ring Buffer for History/Replay)
        self.tape: Deque[ObservationEvent] = deque(maxlen=1000)

        # Health tracking per source
        self.source_health: Dict[str, SourceHealth] = {}

        # QC state per station
        self.qc_states: Dict[str, QCState] = {}

        # SSE Subscribers (asyncio.Queue per subscriber)
        self._sse_queues: List[asyncio.Queue] = []

        # NDJSON persistent log (Phase 2, section 2.6)
        self._ndjson_file = None
        self._init_ndjson_log()

        self.initialized = True
        logger.info("WorldState Initialized (Phase 2)")

    def publish(self, event: ObservationEvent):
        """
        Main entry point for ANY new data.
        Updates state, tracks health, and notifies SSE subscribers.
        """
        # 1. Archive to Tape
        self.tape.append(event)

        # 2. Update Head State
        if isinstance(event, OfficialObs):
            self.latest_official[event.station_id] = event
            source_key = f"METAR_{event.station_id}"
        elif isinstance(event, AuxObs):
            self.latest_aux[event.station_id] = event
            source_key = f"AUX_{event.station_id}"
        elif isinstance(event, EnvFeature):
            self.env_features[event.feature_type] = event
            source_key = f"ENV_{event.feature_type}"
        else:
            source_key = f"UNKNOWN_{event.source}"

        # 3. Update health metrics
        if source_key not in self.source_health:
            self.source_health[source_key] = SourceHealth(source_name=source_key)
        health = self.source_health[source_key]
        health.last_seen_utc = event.ingest_time_utc
        health.last_obs_time_utc = event.obs_time_utc
        health.update_count += 1
        health.last_latency_s = event.latency_seconds

        # 4. Persist to NDJSON log
        self._write_ndjson(event)

        # 5. Notify SSE subscribers
        self._notify_sse(event)

    # --- NDJSON Persistent Log ---
    def _init_ndjson_log(self):
        """Open (or create) today's NDJSON log file."""
        try:
            NDJSON_LOG_DIR.mkdir(parents=True, exist_ok=True)
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            log_path = NDJSON_LOG_DIR / f"world_tape_{today}.ndjson"
            self._ndjson_file = open(log_path, "a", encoding="utf-8")
            logger.info(f"NDJSON log opened: {log_path}")
        except Exception as e:
            logger.warning(f"Failed to open NDJSON log: {e}")
            self._ndjson_file = None

    def _write_ndjson(self, event: ObservationEvent):
        """Append a serialized event as one JSON line."""
        if not self._ndjson_file:
            return
        try:
            payload = self._serialize(event)
            line = json.dumps(payload, default=str)
            self._ndjson_file.write(line + "\n")
            self._ndjson_file.flush()
        except Exception:
            pass  # Never let logging break the pipeline

    def record_error(self, source_key: str, error_msg: str):
        """Record an error for a source."""
        if source_key not in self.source_health:
            self.source_health[source_key] = SourceHealth(source_name=source_key)
        self.source_health[source_key].error_count += 1
        self.source_health[source_key].last_error = error_msg

    def set_qc_state(self, station_id: str, status: str, flags: List[str]):
        """Update QC state for a station."""
        self.qc_states[station_id] = QCState(
            status=status,
            flags=flags,
            last_check_utc=datetime.now(UTC),
        )

    def set_pws_details(self, station_id: str, readings: List[Dict]):
        """Store individual PWS station readings for UI breakdown."""
        self.pws_details[station_id] = readings

    # --- SSE Support ---
    def subscribe_sse(self) -> asyncio.Queue:
        """Create a new SSE subscriber queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._sse_queues.append(q)
        return q

    def unsubscribe_sse(self, q: asyncio.Queue):
        """Remove an SSE subscriber."""
        if q in self._sse_queues:
            self._sse_queues.remove(q)

    def _notify_sse(self, event: ObservationEvent):
        """Push event to all SSE subscribers (non-blocking)."""
        payload = self._serialize(event)
        for q in self._sse_queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Drop oldest if full
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except:
                    pass

    # --- Snapshot ---
    def get_snapshot(self) -> Dict:
        """Returns JSON-ready Full World State Snapshot (Phase 2)."""
        now = datetime.now(UTC)
        now_nyc = now.astimezone(NYC)
        now_mad = now.astimezone(MAD)

        return {
            "ts_utc": now.isoformat(),
            "ts_nyc": now_nyc.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "ts_madrid": now_mad.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "official": {k: self._serialize_official(v) for k, v in self.latest_official.items()},
            "aux": {k: self._serialize_aux(v) for k, v in self.latest_aux.items()},
            "env": {k: self._serialize_env(v) for k, v in self.env_features.items()},
            "qc": {k: v.to_dict() for k, v in self.qc_states.items()},
            "health": {k: v.to_dict() for k, v in self.source_health.items()},
            "pws_details": self.pws_details,  # Individual PWS station readings
            "tape_size": len(self.tape),
            "sse_subscribers": len(self._sse_queues),
        }

    def _serialize_official(self, obs: OfficialObs) -> Dict:
        """Rich serialization for OfficialObs with dual timestamps."""
        return {
            "station_id": obs.station_id,
            "temp_c": obs.temp_c,
            "temp_f": obs.temp_f,  # Rounded for settlement
            "temp_f_raw": obs.temp_f_raw,  # With decimals for display
            "dewpoint_c": obs.dewpoint_c,
            "wind_dir": obs.wind_dir,
            "wind_speed": obs.wind_speed,
            "sky_condition": obs.sky_condition,
            "source": obs.source,
            "obs_time_utc": obs.obs_time_utc.isoformat(),
            "obs_time_nyc": obs.obs_time_nyc.strftime("%Y-%m-%d %H:%M:%S"),
            "obs_time_madrid": obs.obs_time_madrid.strftime("%Y-%m-%d %H:%M:%S"),
            "ingest_time_utc": obs.ingest_time_utc.isoformat(),
            "source_age_s": round(obs.latency_seconds, 1),
            "qc_passed": obs.qc_passed,
            "qc_flags": obs.qc_flags,
        }

    def _serialize_aux(self, obs: AuxObs) -> Dict:
        """Rich serialization for AuxObs."""
        return {
            "station_id": obs.station_id,
            "temp_c": obs.temp_c,
            "temp_f": obs.temp_f,
            "humidity": obs.humidity,
            "is_aggregate": obs.is_aggregate,
            "support": obs.support,
            "drift": obs.drift,
            "source": obs.source,
            "obs_time_utc": obs.obs_time_utc.isoformat(),
            "obs_time_nyc": obs.obs_time_nyc.strftime("%Y-%m-%d %H:%M:%S"),
            "obs_time_madrid": obs.obs_time_madrid.strftime("%Y-%m-%d %H:%M:%S"),
            "ingest_time_utc": obs.ingest_time_utc.isoformat(),
            "source_age_s": round(obs.latency_seconds, 1),
        }

    def _serialize_env(self, feat: EnvFeature) -> Dict:
        """Rich serialization for EnvFeature."""
        return {
            "feature_type": feat.feature_type,
            "location_ref": feat.location_ref,
            "value": feat.value,
            "unit": feat.unit,
            "status": feat.status,
            "source": feat.source,
            "obs_time_utc": feat.obs_time_utc.isoformat(),
            "obs_time_nyc": feat.obs_time_nyc.strftime("%Y-%m-%d %H:%M:%S"),
            "obs_time_madrid": feat.obs_time_madrid.strftime("%Y-%m-%d %H:%M:%S"),
            "ingest_time_utc": feat.ingest_time_utc.isoformat(),
            "source_age_s": round(feat.latency_seconds, 1),
        }

    def _serialize(self, obj) -> Dict:
        """Generic serialization helper."""
        if isinstance(obj, OfficialObs):
            return {"type": "official", "data": self._serialize_official(obj)}
        elif isinstance(obj, AuxObs):
            return {"type": "aux", "data": self._serialize_aux(obj)}
        elif isinstance(obj, EnvFeature):
            return {"type": "env", "data": self._serialize_env(obj)}
        elif hasattr(obj, "__dict__"):
            d = obj.__dict__.copy()
            for k, v in d.items():
                if isinstance(v, datetime):
                    d[k] = v.isoformat()
            return d
        return {"raw": str(obj)}


# Global Instance Accessor
def get_world() -> WorldState:
    return WorldState()
