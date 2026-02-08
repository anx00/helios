"""
Dataset Module - Backtest Dataset Construction

Builds unified datasets for backtesting by joining:
- world_obs: METAR, PWS consensus, upstream observations
- features: HRRR/GFS/NBM aggregates, SST/AOD, derived features
- nowcast: Model outputs (tmax_mean, sigma, P(bucket), etc.)
- market: L2 snapshots, bid/ask, spread, depth
- events: Event windows, resyncs, gaps, staleness
- labels: Ground truth for the day

Uses "timeline join" (asof join) to reconstruct state at any timestamp.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any, Iterator, Tuple
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import os
from collections import OrderedDict
from threading import RLock
from time import perf_counter

from core.polymarket_labels import (
    normalize_label,
    normalize_p_bucket,
    normalize_market_snapshot,
    sort_labels,
)
logger = logging.getLogger("backtest.dataset")

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


@dataclass
class TimelineState:
    """
    State of the world at a specific timestamp.

    This is what the model would have "seen" at time t.
    """
    timestamp_utc: datetime
    timestamp_nyc: datetime

    # World observations (last valid)
    last_metar: Optional[Dict] = None
    metar_age_seconds: float = float('inf')

    last_pws: Optional[Dict] = None
    pws_age_seconds: float = float('inf')

    last_upstream: Optional[Dict] = None
    upstream_age_seconds: float = float('inf')

    # Current max observed so far
    max_observed_f: Optional[float] = None
    max_observed_at: Optional[datetime] = None

    # Features (last valid)
    features: Optional[Dict] = None
    features_age_seconds: float = float('inf')

    # Nowcast output (last valid)
    nowcast: Optional[Dict] = None
    nowcast_age_seconds: float = float('inf')

    # Market state (last valid)
    market: Optional[Dict] = None
    market_age_seconds: float = float('inf')

    # QC state
    qc_flags: List[str] = field(default_factory=list)

    # Event window (if any active)
    active_window: Optional[Dict] = None

    def is_stale(self, max_age_seconds: float = 600) -> bool:
        """Check if critical data is stale."""
        return (
            self.metar_age_seconds > max_age_seconds or
            self.nowcast_age_seconds > max_age_seconds
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "timestamp_nyc": self.timestamp_nyc.isoformat(),
            "last_metar": self.last_metar,
            "metar_age_seconds": self.metar_age_seconds,
            "last_pws": self.last_pws,
            "pws_age_seconds": self.pws_age_seconds,
            "max_observed_f": self.max_observed_f,
            "nowcast": self.nowcast,
            "nowcast_age_seconds": self.nowcast_age_seconds,
            "market": self.market,
            "market_age_seconds": self.market_age_seconds,
            "qc_flags": self.qc_flags,
            "is_stale": self.is_stale(),
        }


@dataclass
class BacktestDataset:
    """
    A complete dataset for one market day.

    Contains all events sorted by timestamp, plus labels.
    """
    station_id: str
    market_date: date
    market_id: Optional[str] = None

    # Events by channel
    world_events: List[Dict] = field(default_factory=list)
    pws_events: List[Dict] = field(default_factory=list)
    features_events: List[Dict] = field(default_factory=list)
    nowcast_events: List[Dict] = field(default_factory=list)
    market_events: List[Dict] = field(default_factory=list)
    event_windows: List[Dict] = field(default_factory=list)

    # All events merged and sorted
    all_events: List[Dict] = field(default_factory=list)

    # Labels (ground truth)
    label: Optional["DayLabel"] = None

    # Metadata
    total_events: int = 0
    duration_seconds: float = 0
    start_time_utc: Optional[datetime] = None
    end_time_utc: Optional[datetime] = None

    def get_events_in_range(
        self,
        start_utc: datetime,
        end_utc: datetime
    ) -> List[Dict]:
        """Get events within a time range."""
        return [
            e for e in self.all_events
            if start_utc <= self._parse_ts(e) < end_utc
        ]

    def _parse_ts(self, event: Dict) -> datetime:
        """Parse timestamp from event."""
        ts_str = event.get("ts_ingest_utc") or event.get("ts_utc")
        if isinstance(ts_str, str):
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return ts_str

    def iterate_timeline(
        self,
        interval_seconds: float = 60
    ) -> Iterator[TimelineState]:
        """
        Iterate through the day at regular intervals.

        Yields TimelineState at each interval, with "as-of" data.
        """
        if not self.start_time_utc or not self.end_time_utc:
            return

        current = self.start_time_utc
        state = TimelineState(
            timestamp_utc=current,
            timestamp_nyc=current.astimezone(NYC)
        )

        # Track last values for each channel
        last_metar = None
        last_metar_ts = None
        last_pws = None
        last_pws_ts = None
        last_nowcast = None
        last_nowcast_ts = None
        last_market = None
        last_market_ts = None
        max_temp = None
        max_temp_ts = None

        event_idx = 0

        while current <= self.end_time_utc:
            # Process all events up to current time
            while event_idx < len(self.all_events):
                event = self.all_events[event_idx]
                event_ts = self._parse_ts(event)

                if event_ts > current:
                    break

                ch = event.get("ch", "")
                data = event.get("data", event)

                if ch == "world":
                    last_metar = data
                    last_metar_ts = event_ts
                    # Update max observed
                    temp = data.get("temp_f") or data.get("temp_aligned")
                    if temp is not None:
                        if max_temp is None or temp > max_temp:
                            max_temp = temp
                            max_temp_ts = event_ts

                elif ch == "pws":
                    last_pws = data
                    last_pws_ts = event_ts

                elif ch == "nowcast":
                    last_nowcast = data
                    last_nowcast_ts = event_ts

                elif ch in ("market", "l2_snap", "l2_snap_1s"):
                    last_market = data
                    last_market_ts = event_ts

                event_idx += 1

            # Build state
            state = TimelineState(
                timestamp_utc=current,
                timestamp_nyc=current.astimezone(NYC),
                last_metar=last_metar,
                metar_age_seconds=(current - last_metar_ts).total_seconds() if last_metar_ts else float('inf'),
                last_pws=last_pws,
                pws_age_seconds=(current - last_pws_ts).total_seconds() if last_pws_ts else float('inf'),
                nowcast=last_nowcast,
                nowcast_age_seconds=(current - last_nowcast_ts).total_seconds() if last_nowcast_ts else float('inf'),
                market=last_market,
                market_age_seconds=(current - last_market_ts).total_seconds() if last_market_ts else float('inf'),
                max_observed_f=max_temp,
                max_observed_at=max_temp_ts,
            )

            yield state

            current += timedelta(seconds=interval_seconds)

    def get_nowcast_at(self, timestamp_utc: datetime) -> Optional[Dict]:
        """Get the nowcast that was valid at a given time."""
        last_valid = None

        for event in self.nowcast_events:
            event_ts = self._parse_ts(event)
            if event_ts <= timestamp_utc:
                last_valid = event.get("data", event)
            else:
                break

        return last_valid

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "station_id": self.station_id,
            "market_date": self.market_date.isoformat(),
            "market_id": self.market_id,
            "total_events": self.total_events,
            "events_by_channel": {
                "world": len(self.world_events),
                "pws": len(self.pws_events),
                "features": len(self.features_events),
                "nowcast": len(self.nowcast_events),
                "market": len(self.market_events),
                "event_windows": len(self.event_windows),
            },
            "duration_seconds": self.duration_seconds,
            "start_time_utc": self.start_time_utc.isoformat() if self.start_time_utc else None,
            "end_time_utc": self.end_time_utc.isoformat() if self.end_time_utc else None,
            "has_label": self.label is not None,
        }


class DatasetBuilder:
    """
    Builds BacktestDataset from recorded data.

    Uses HybridReader (from Phase 4) to read NDJSON/Parquet.
    """

    def __init__(
        self,
        recordings_dir: str = "data/recordings",
        parquet_dir: str = "data/parquet"
    ):
        self.recordings_dir = Path(recordings_dir)
        self.parquet_dir = Path(parquet_dir)

        # Import reader lazily
        self._reader = None
        # LRU cache for expensive per-day dataset builds.
        try:
            self._cache_size = int(os.getenv("HELIOS_BACKTEST_DATASET_CACHE_SIZE", "16"))
        except (TypeError, ValueError):
            self._cache_size = 16
        self._dataset_cache: "OrderedDict[Tuple[str, str, Optional[str]], BacktestDataset]" = OrderedDict()
        self._cache_lock = RLock()

    def _get_reader(self):
        """Get HybridReader instance."""
        if self._reader is None:
            from core.compactor import get_hybrid_reader
            self._reader = get_hybrid_reader(
                str(self.recordings_dir),
                str(self.parquet_dir)
            )
        return self._reader

    def build_dataset(
        self,
        station_id: str,
        market_date: date,
        market_id: Optional[str] = None
    ) -> BacktestDataset:
        """
        Build a complete backtest dataset for a market day.

        Args:
            station_id: Station identifier (e.g., "KLGA")
            market_date: Market date (NYC timezone)
            market_id: Optional market identifier

        Returns:
            BacktestDataset with all events and labels
        """
        reader = self._get_reader()
        date_str = market_date.isoformat()
        cache_key = (station_id, date_str, market_id)

        logger.info(f"Building dataset for {station_id}/{date_str}")

        # Fast path: return cached dataset for repeated suite runs.
        if self._cache_size > 0:
            with self._cache_lock:
                if cache_key in self._dataset_cache:
                    self._dataset_cache.move_to_end(cache_key)
                    logger.debug("Dataset cache hit for %s/%s", station_id, date_str)
                    return self._dataset_cache[cache_key]

        # Create dataset
        dataset = BacktestDataset(
            station_id=station_id,
            market_date=market_date,
            market_id=market_id
        )

        # Read only channels relevant to backtest runtime.
        # Station-scoped lookup avoids pulling giant channels from other stations.
        available_channels = set(reader.list_channels_for_date(date_str, station_id))
        channels_env = os.getenv("HELIOS_BACKTEST_CHANNELS", "").strip()
        if channels_env:
            requested_channels = [c.strip() for c in channels_env.split(",") if c.strip()]
        else:
            requested_channels = ["world", "pws", "nowcast", "event_window"]
            # Prefer lower-volume market channel when both exist.
            if "market" in available_channels:
                requested_channels.append("market")
            elif "l2_snap" in available_channels:
                requested_channels.append("l2_snap")
            elif "l2_snap_1s" in available_channels:
                requested_channels.append("l2_snap_1s")

        read_t0 = perf_counter()
        all_events = reader.get_events_sorted(
            date_str,
            station_id,
            channels=requested_channels if requested_channels else None,
        )
        logger.info(
            "Loaded raw events for %s/%s: %s events in %.2fs (channels=%s)",
            station_id,
            date_str,
            len(all_events),
            perf_counter() - read_t0,
            requested_channels,
        )

        if not all_events:
            logger.warning(f"No events found for {date_str}")
            return dataset

        # Categorize by channel
        for event in all_events:
            ch = event.get("ch", "")

            if ch == "world":
                dataset.world_events.append(event)
            elif ch == "pws":
                dataset.pws_events.append(event)
            elif ch == "features":
                dataset.features_events.append(event)
            elif ch == "nowcast":
                dataset.nowcast_events.append(event)
            elif ch in ("market", "l2_snap", "l2_snap_1s"):
                dataset.market_events.append(event)
            elif ch == "event_window":
                dataset.event_windows.append(event)

        # Diagnostic logging: per-channel event counts
        logger.debug(
            "Dataset %s/%s channel counts: world=%d pws=%d features=%d "
            "nowcast=%d market=%d event_window=%d (total=%d)",
            station_id, date_str,
            len(dataset.world_events), len(dataset.pws_events),
            len(dataset.features_events), len(dataset.nowcast_events),
            len(dataset.market_events), len(dataset.event_windows),
            len(all_events),
        )
        if not dataset.market_events:
            logger.warning(
                "No market events found for %s/%s. "
                "Channels seen: %s",
                station_id, date_str,
                sorted(set(e.get("ch", "?") for e in all_events)),
            )

        # Normalize nowcast and market events to Polymarket labels
        for event in dataset.nowcast_events:
            data = event.get("data", event)
            if isinstance(data, dict) and "p_bucket" in data:
                data["p_bucket"] = normalize_p_bucket(data.get("p_bucket", []))
                if "top_bucket" in data:
                    data["top_bucket"] = normalize_label(data.get("top_bucket"))
                event["data"] = data

        for event in dataset.market_events:
            data = event.get("data", event)
            if isinstance(data, dict):
                normalized = normalize_market_snapshot(data)
                # Keep only execution-relevant fields in-memory.
                for bucket, payload in list(normalized.items()):
                    if not isinstance(payload, dict):
                        continue
                    if "bids" in payload or "asks" in payload:
                        slim = dict(payload)
                        slim.pop("bids", None)
                        slim.pop("asks", None)
                        normalized[bucket] = slim
                event["data"] = normalized

        # Store all events sorted
        dataset.all_events = all_events
        dataset.total_events = len(all_events)

        # Calculate time bounds
        if all_events:
            first_ts = self._parse_ts(all_events[0])
            last_ts = self._parse_ts(all_events[-1])

            dataset.start_time_utc = first_ts
            dataset.end_time_utc = last_ts
            dataset.duration_seconds = (last_ts - first_ts).total_seconds()

        # Extract bucket labels from market data when available
        bucket_labels = None
        for event in dataset.market_events:
            data = event.get("data", event)
            if isinstance(data, dict):
                labels = [k for k in data.keys() if isinstance(k, str) and not k.startswith("__")]
                if labels:
                    bucket_labels = sort_labels(labels)
                    break

        # Try to load or derive labels
        dataset.label = self._get_or_derive_label(
            station_id, market_date, dataset.world_events, market_id, bucket_labels
        )

        logger.info(
            f"Built dataset: {dataset.total_events} events, "
            f"{dataset.duration_seconds:.0f}s duration"
        )

        # Store in LRU cache.
        if self._cache_size > 0:
            with self._cache_lock:
                self._dataset_cache[cache_key] = dataset
                self._dataset_cache.move_to_end(cache_key)
                while len(self._dataset_cache) > self._cache_size:
                    self._dataset_cache.popitem(last=False)

        return dataset

    def _parse_ts(self, event: Dict) -> datetime:
        """Parse timestamp from event."""
        ts_str = event.get("ts_ingest_utc") or event.get("ts_utc")
        if isinstance(ts_str, str):
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return ts_str

    def _get_or_derive_label(
        self,
        station_id: str,
        market_date: date,
        world_events: List[Dict],
        market_id: Optional[str],
        bucket_labels: Optional[List[str]] = None
    ) -> Optional["DayLabel"]:
        """Get existing label or derive from observations."""
        from .labels import get_label_manager

        label_mgr = get_label_manager()

        # Try to get existing label
        label = label_mgr.get_label(station_id, market_date)
        if label:
            if bucket_labels and label.y_tmax_aligned is not None:
                bucket_label, bucket_idx = label_mgr.get_bucket_for_temp(
                    label.y_tmax_aligned,
                    bucket_labels=bucket_labels
                )
                label.y_bucket_winner = normalize_label(bucket_label) if bucket_label else label.y_bucket_winner
                label.y_bucket_index = bucket_idx if bucket_idx is not None else label.y_bucket_index
            return label

        # Derive from observations
        observations = []
        for event in world_events:
            data = event.get("data", event)
            ts = event.get("ts_ingest_utc") or event.get("ts_utc")
            if ts and data.get("temp_f") is not None:
                observations.append({
                    "ts_utc": ts,
                    "temp_f": data["temp_f"]
                })

        if observations:
            label = label_mgr.derive_label_from_observations(
                station_id,
                market_date,
                observations,
                market_id,
                bucket_labels=bucket_labels
            )
            # Save derived label
            label_mgr.set_label(label, save=True)
            return label

        return None

    def list_available_dates(
        self,
        station_id: Optional[str] = None
    ) -> List[date]:
        """List dates with available data."""
        reader = self._get_reader()
        date_strs = reader.list_available_dates(station_id)
        return [date.fromisoformat(d) for d in date_strs]

    def build_datasets_for_range(
        self,
        station_id: str,
        start_date: date,
        end_date: date
    ) -> List[BacktestDataset]:
        """Build datasets for a date range."""
        datasets = []
        available = set(self.list_available_dates(station_id))

        current = start_date
        while current <= end_date:
            if current in available:
                try:
                    ds = self.build_dataset(station_id, current)
                    datasets.append(ds)
                except Exception as e:
                    logger.error(f"Failed to build dataset for {current}: {e}")
            current += timedelta(days=1)

        return datasets


# =============================================================================
# Global accessor
# =============================================================================

_dataset_builder: Optional[DatasetBuilder] = None


def get_dataset_builder(
    recordings_dir: str = "data/recordings",
    parquet_dir: str = "data/parquet"
) -> DatasetBuilder:
    """Get the global DatasetBuilder instance."""
    global _dataset_builder
    if _dataset_builder is None:
        _dataset_builder = DatasetBuilder(recordings_dir, parquet_dir)
    return _dataset_builder
