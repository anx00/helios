"""
Compactor - Phase 4 (Section 4.5, 4.7)

Converts NDJSON recordings to Parquet for efficient storage and querying.
Runs as a batch job (outside the hot path).

Parquet layout (per Section 4.7):
- data/parquet/station=KLGA/date=2026-01-29/ch=world/part-0000.parquet
- data/parquet/station=KLGA/date=2026-01-29/ch=nowcast/part-0000.parquet
"""

import asyncio
import logging
import json
import os
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Generator
from zoneinfo import ZoneInfo
from time import perf_counter

logger = logging.getLogger("compactor")

UTC = ZoneInfo("UTC")

# Try to import pyarrow for Parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("pyarrow not installed - Parquet compaction disabled")
_PARQUET_READ_WARNED = False


def read_ndjson_file(file_path: Path) -> Generator[Dict, None, None]:
    """Read NDJSON file line by line."""
    if not file_path.exists():
        return

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {file_path}: {e}")


def flatten_event(event: Dict) -> Dict:
    """
    Flatten nested event structure for Parquet.
    Moves 'data' fields to top level with 'd_' prefix.
    """
    flat = {}

    # Copy top-level fields
    for key, value in event.items():
        if key == "data":
            # Flatten data with prefix
            if isinstance(value, dict):
                for dk, dv in value.items():
                    # Convert complex types to JSON strings
                    if isinstance(dv, (dict, list)):
                        flat[f"d_{dk}"] = json.dumps(dv)
                    else:
                        flat[f"d_{dk}"] = dv
        else:
            flat[key] = value

    return flat


def events_to_table(events: List[Dict]) -> Optional["pa.Table"]:
    """Convert list of events to PyArrow Table."""
    if not PARQUET_AVAILABLE:
        return None

    if not events:
        return None

    # Flatten all events
    flattened = [flatten_event(e) for e in events]

    # Get all unique columns
    all_columns = set()
    for e in flattened:
        all_columns.update(e.keys())

    # Build column arrays
    columns = {}
    for col in all_columns:
        values = [e.get(col) for e in flattened]
        columns[col] = values

    try:
        return pa.table(columns)
    except Exception as e:
        logger.error(f"Failed to create table: {e}")
        return None


class Compactor:
    """
    Compacts NDJSON recordings to Parquet.

    Can run as:
    - One-shot: compact a specific date
    - Batch: compact all pending dates
    - Scheduled: run periodically via APScheduler
    """

    def __init__(
        self,
        ndjson_base: str = "data/recordings",
        parquet_base: str = "data/parquet",
        delete_after_compact: bool = False,
        retention_days: int = 14
    ):
        self.ndjson_base = Path(ndjson_base)
        self.parquet_base = Path(parquet_base)
        self.delete_after_compact = delete_after_compact
        self.retention_days = retention_days

        self.parquet_base.mkdir(parents=True, exist_ok=True)

        # Stats
        self._compaction_runs = 0
        self._files_compacted = 0
        self._events_compacted = 0
        self._bytes_written = 0

        logger.info(f"Compactor initialized: {self.ndjson_base} -> {self.parquet_base}")

    def list_ndjson_dates(self) -> List[str]:
        """List all dates with NDJSON data."""
        dates = set()
        if not self.ndjson_base.exists():
            return []

        for item in self.ndjson_base.iterdir():
            if item.is_dir() and item.name.startswith("date="):
                dates.add(item.name.replace("date=", ""))

        return sorted(dates)

    def list_channels_for_date(self, date_str: str) -> List[str]:
        """List channels available for a date."""
        date_path = self.ndjson_base / f"date={date_str}"
        if not date_path.exists():
            return []

        channels = []
        for item in date_path.iterdir():
            if item.is_dir() and item.name.startswith("ch="):
                channels.append(item.name.replace("ch=", ""))

        return channels

    def get_ndjson_path(self, date_str: str, channel: str) -> Path:
        """Get NDJSON file path for date and channel."""
        return self.ndjson_base / f"date={date_str}" / f"ch={channel}" / "events.ndjson"

    def get_parquet_path(
        self,
        date_str: str,
        channel: str,
        station_id: Optional[str] = None
    ) -> Path:
        """
        Get Parquet output path.

        Layout: data/parquet/station=KLGA/date=2026-01-29/ch=world/part-0000.parquet
        """
        if station_id:
            return (
                self.parquet_base /
                f"station={station_id}" /
                f"date={date_str}" /
                f"ch={channel}" /
                "part-0000.parquet"
            )
        else:
            return (
                self.parquet_base /
                f"date={date_str}" /
                f"ch={channel}" /
                "part-0000.parquet"
            )

    def compact_channel(
        self,
        date_str: str,
        channel: str
    ) -> Dict[str, int]:
        """
        Compact a single channel for a date.

        Returns stats dict with events_read, files_written, bytes_written.
        """
        if not PARQUET_AVAILABLE:
            logger.warning("Parquet not available - skipping compaction")
            return {"events_read": 0, "files_written": 0, "bytes_written": 0}

        ndjson_path = self.get_ndjson_path(date_str, channel)
        if not ndjson_path.exists():
            logger.debug(f"No NDJSON file: {ndjson_path}")
            return {"events_read": 0, "files_written": 0, "bytes_written": 0}

        # Read all events
        events = list(read_ndjson_file(ndjson_path))
        if not events:
            return {"events_read": 0, "files_written": 0, "bytes_written": 0}

        # Group by station_id
        events_by_station: Dict[str, List[Dict]] = {}
        for event in events:
            station = event.get("station_id", "ALL")
            if station not in events_by_station:
                events_by_station[station] = []
            events_by_station[station].append(event)

        total_bytes = 0
        files_written = 0

        # Write Parquet for each station
        for station_id, station_events in events_by_station.items():
            table = events_to_table(station_events)
            if table is None:
                continue

            parquet_path = self.get_parquet_path(
                date_str,
                channel,
                station_id if station_id != "ALL" else None
            )
            parquet_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                pq.write_table(table, parquet_path, compression="snappy")
                file_size = parquet_path.stat().st_size
                total_bytes += file_size
                files_written += 1

                logger.info(
                    f"Compacted {channel} for {date_str} "
                    f"(station={station_id}): {len(station_events)} events, "
                    f"{file_size / 1024:.1f} KB"
                )
            except Exception as e:
                logger.error(f"Failed to write Parquet: {e}")

        # Optionally delete NDJSON after successful compaction
        if self.delete_after_compact and files_written > 0:
            try:
                ndjson_path.unlink()
                logger.debug(f"Deleted NDJSON: {ndjson_path}")
            except Exception as e:
                logger.warning(f"Failed to delete NDJSON: {e}")

        self._events_compacted += len(events)
        self._bytes_written += total_bytes

        return {
            "events_read": len(events),
            "files_written": files_written,
            "bytes_written": total_bytes
        }

    def compact_date(self, date_str: str) -> Dict[str, Any]:
        """
        Compact all channels for a date.

        Returns summary stats.
        """
        channels = self.list_channels_for_date(date_str)
        if not channels:
            return {"date": date_str, "channels": [], "total_events": 0}

        results = {}
        total_events = 0

        for channel in channels:
            stats = self.compact_channel(date_str, channel)
            results[channel] = stats
            total_events += stats["events_read"]

        self._compaction_runs += 1
        self._files_compacted += len(channels)

        return {
            "date": date_str,
            "channels": results,
            "total_events": total_events
        }

    def compact_all_pending(self) -> Dict[str, Any]:
        """
        Compact all dates that haven't been compacted yet.

        Skips today (still being written).
        """
        today = datetime.now(UTC).date().isoformat()
        dates = self.list_ndjson_dates()

        # Skip today
        dates = [d for d in dates if d < today]

        results = {}
        for date_str in dates:
            # Check if already compacted (any parquet exists)
            if self._is_date_compacted(date_str):
                logger.debug(f"Skipping already compacted: {date_str}")
                continue

            results[date_str] = self.compact_date(date_str)

        return {
            "compacted_dates": list(results.keys()),
            "results": results,
            "total_events": sum(r["total_events"] for r in results.values())
        }

    def _is_date_compacted(self, date_str: str) -> bool:
        """Check if any Parquet exists for a date."""
        # Check in all possible station directories
        for item in self.parquet_base.iterdir():
            if item.is_dir():
                date_path = item / f"date={date_str}"
                if date_path.exists() and any(date_path.iterdir()):
                    return True

        # Also check dateless path
        date_path = self.parquet_base / f"date={date_str}"
        if date_path.exists() and any(date_path.iterdir()):
            return True

        return False

    def cleanup_old_ndjson(self) -> int:
        """
        Delete NDJSON files older than retention_days.

        Returns number of files deleted.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=self.retention_days)).date()
        deleted = 0

        for date_str in self.list_ndjson_dates():
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if file_date < cutoff:
                    date_path = self.ndjson_base / f"date={date_str}"
                    if date_path.exists():
                        import shutil
                        shutil.rmtree(date_path)
                        deleted += 1
                        logger.info(f"Deleted old NDJSON: {date_path}")
            except ValueError:
                continue

        return deleted

    def get_stats(self) -> Dict:
        """Get compactor statistics."""
        return {
            "ndjson_base": str(self.ndjson_base),
            "parquet_base": str(self.parquet_base),
            "parquet_available": PARQUET_AVAILABLE,
            "compaction_runs": self._compaction_runs,
            "files_compacted": self._files_compacted,
            "events_compacted": self._events_compacted,
            "bytes_written": self._bytes_written,
            "retention_days": self.retention_days,
            "available_dates": self.list_ndjson_dates()
        }


# =============================================================================
# Parquet Reader (for Replay)
# =============================================================================

class ParquetReader:
    """
    Reads Parquet files for replay.

    Provides iterator over events in timestamp order.
    """

    def __init__(self, parquet_base: str = "data/parquet"):
        self.parquet_base = Path(parquet_base)

    def list_available_dates(self, station_id: Optional[str] = None) -> List[str]:
        """List dates with Parquet data."""
        dates = set()

        if not self.parquet_base.exists():
            return []

        if station_id:
            station_path = self.parquet_base / f"station={station_id}"
            if station_path.exists():
                for item in station_path.iterdir():
                    if item.is_dir() and item.name.startswith("date="):
                        dates.add(item.name.replace("date=", ""))
        else:
            # Check all paths
            for item in self.parquet_base.iterdir():
                if item.is_dir():
                    if item.name.startswith("date="):
                        dates.add(item.name.replace("date=", ""))
                    elif item.name.startswith("station="):
                        for sub in item.iterdir():
                            if sub.is_dir() and sub.name.startswith("date="):
                                dates.add(sub.name.replace("date=", ""))

        return sorted(dates)

    def list_channels_for_date(
        self,
        date_str: str,
        station_id: Optional[str] = None
    ) -> List[str]:
        """List channels available for a date."""
        channels = set()

        if not self.parquet_base.exists():
            return []

        if station_id:
            date_path = (
                self.parquet_base /
                f"station={station_id}" /
                f"date={date_str}"
            )
            if date_path.exists():
                for item in date_path.iterdir():
                    if item.is_dir() and item.name.startswith("ch="):
                        channels.add(item.name.replace("ch=", ""))
        else:
            # Check dateless path
            date_path = self.parquet_base / f"date={date_str}"
            if date_path.exists():
                for item in date_path.iterdir():
                    if item.is_dir() and item.name.startswith("ch="):
                        channels.add(item.name.replace("ch=", ""))

            # Check station paths
            for station_dir in self.parquet_base.iterdir():
                if station_dir.is_dir() and station_dir.name.startswith("station="):
                    date_path = station_dir / f"date={date_str}"
                    if date_path.exists():
                        for item in date_path.iterdir():
                            if item.is_dir() and item.name.startswith("ch="):
                                channels.add(item.name.replace("ch=", ""))

        return list(channels)

    def read_channel(
        self,
        date_str: str,
        channel: str,
        station_id: Optional[str] = None
    ) -> List[Dict]:
        """Read all events from a channel."""
        global _PARQUET_READ_WARNED
        if not PARQUET_AVAILABLE:
            if not _PARQUET_READ_WARNED:
                logger.warning("Parquet not available")
                _PARQUET_READ_WARNED = True
            return []

        # Find parquet file
        if station_id:
            parquet_path = (
                self.parquet_base /
                f"station={station_id}" /
                f"date={date_str}" /
                f"ch={channel}" /
                "part-0000.parquet"
            )
        else:
            parquet_path = (
                self.parquet_base /
                f"date={date_str}" /
                f"ch={channel}" /
                "part-0000.parquet"
            )

        if not parquet_path.exists():
            return []

        try:
            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            # Convert to list of dicts
            events = df.to_dict(orient="records")

            # Unflatten data fields
            for event in events:
                data = {}
                keys_to_remove = []
                for key, value in event.items():
                    if key.startswith("d_"):
                        data_key = key[2:]
                        # Try to parse JSON strings
                        if isinstance(value, str):
                            try:
                                data[data_key] = json.loads(value)
                            except:
                                data[data_key] = value
                        else:
                            data[data_key] = value
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del event[key]

                if data:
                    event["data"] = data

            return events

        except Exception as e:
            logger.error(f"Failed to read Parquet: {e}")
            return []

    def read_all_channels(
        self,
        date_str: str,
        station_id: Optional[str] = None,
        channels: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """Read all channels for a date."""
        available_channels = self.list_channels_for_date(date_str, station_id)
        channel_filter = set(channels) if channels else None
        result = {}

        for channel in available_channels:
            if channel_filter is not None and channel not in channel_filter:
                continue
            result[channel] = self.read_channel(date_str, channel, station_id)

        return result

    def get_events_sorted(
        self,
        date_str: str,
        station_id: Optional[str] = None,
        channels: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get all events for a date, sorted by timestamp.

        This is the main method for replay.
        """
        all_data = self.read_all_channels(date_str, station_id, channels=channels)

        # Flatten to single list
        all_events = []
        for channel, events in all_data.items():
            all_events.extend(events)

        # Sort by ingest timestamp
        def get_timestamp(e):
            ts = e.get("ts_ingest_utc", "")
            return ts if ts else "9999"

        all_events.sort(key=get_timestamp)

        return all_events


# =============================================================================
# Hybrid Reader - NDJSON + Parquet
# =============================================================================

class HybridReader:
    """
    Reads from both NDJSON (fresh) and Parquet (compacted) sources.

    Prefers NDJSON for recent data, falls back to Parquet for compacted data.
    """

    def __init__(
        self,
        ndjson_base: str = "data/recordings",
        parquet_base: str = "data/parquet"
    ):
        self.ndjson_base = Path(ndjson_base)
        self.parquet_base = Path(parquet_base)
        self._parquet_reader = ParquetReader(parquet_base)
        # Cache for expensive "does this file contain station X?" scans.
        self._station_presence_cache: Dict[tuple, bool] = {}
        try:
            self._station_presence_cache_max = int(
                os.getenv("HELIOS_STATION_PRESENCE_CACHE_MAX", "2048")
            )
        except (TypeError, ValueError):
            self._station_presence_cache_max = 2048

    def _station_cache_key(self, event_file: Path, station_id: str) -> tuple:
        """Build a cache key that invalidates when file changes."""
        try:
            st = event_file.stat()
            return (str(event_file), station_id, st.st_mtime_ns, st.st_size)
        except OSError:
            return (str(event_file), station_id, 0, 0)

    @staticmethod
    def _top_level_prefix(line: str) -> str:
        """
        Return the top-level JSON prefix before the 'data' payload.
        This lets us inspect station_id without parsing large nested books.
        """
        idx = line.find('"data"')
        if idx <= 0:
            return line
        return line[:idx]

    def _line_matches_station(self, line: str, station_id: str) -> bool:
        """
        Fast station filter using only top-level JSON text.
        Accepts:
        - exact station_id
        - global events with no station_id
        - explicit null station_id
        """
        prefix = self._top_level_prefix(line)
        if '"station_id"' not in prefix:
            return True
        if f'"station_id":"{station_id}"' in prefix:
            return True
        if f'"station_id": "{station_id}"' in prefix:
            return True
        if '"station_id":null' in prefix or '"station_id": null' in prefix:
            return True
        return False

    def _iter_ndjson_event_files(self, date_str: str) -> Generator[Path, None, None]:
        """Iterate NDJSON event files for a given date."""
        date_path = self.ndjson_base / f"date={date_str}"
        if not date_path.exists():
            return

        for channel_dir in date_path.iterdir():
            if not (channel_dir.is_dir() and channel_dir.name.startswith("ch=")):
                continue
            event_file = channel_dir / "events.ndjson"
            if event_file.exists():
                yield event_file

    def _file_has_any_event(self, event_file: Path) -> bool:
        """Return True if file has at least one non-empty line."""
        try:
            with open(event_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    return True
        except Exception:
            return False
        return False

    def _file_has_station_event(self, event_file: Path, station_id: str) -> bool:
        """Return True if file has at least one event for station_id (or global)."""
        cache_key = self._station_cache_key(event_file, station_id)
        cached = self._station_presence_cache.get(cache_key)
        if cached is not None:
            return cached

        has_station = False
        try:
            with open(event_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if self._line_matches_station(line, station_id):
                        has_station = True
                        break
        except Exception:
            return False

        if len(self._station_presence_cache) >= self._station_presence_cache_max:
            self._station_presence_cache.clear()
        self._station_presence_cache[cache_key] = has_station
        return has_station

    def _date_has_events_ndjson(self, date_str: str, station_id: Optional[str]) -> bool:
        """Return True when date has at least one usable NDJSON event."""
        for event_file in self._iter_ndjson_event_files(date_str):
            if station_id:
                if self._file_has_station_event(event_file, station_id):
                    return True
            elif self._file_has_any_event(event_file):
                return True
        return False

    def _channel_has_events_ndjson(
        self,
        date_str: str,
        channel: str,
        station_id: Optional[str],
    ) -> bool:
        """Return True when channel/date has events for station_id (or any station)."""
        event_file = (
            self.ndjson_base /
            f"date={date_str}" /
            f"ch={channel}" /
            "events.ndjson"
        )
        if not event_file.exists():
            return False

        if station_id:
            return self._file_has_station_event(event_file, station_id)
        return self._file_has_any_event(event_file)

    def list_available_dates(self, station_id: Optional[str] = None) -> List[str]:
        """List dates with recorded data (NDJSON or Parquet)."""
        dates = set()

        # Check NDJSON recordings
        if self.ndjson_base.exists():
            for item in self.ndjson_base.iterdir():
                if item.is_dir() and item.name.startswith("date="):
                    date_str = item.name.replace("date=", "")
                    if self._date_has_events_ndjson(date_str, station_id):
                        dates.add(date_str)

        # Also check Parquet
        parquet_dates = self._parquet_reader.list_available_dates(station_id)
        dates.update(parquet_dates)

        return sorted(dates, reverse=True)  # Most recent first

    def list_channels_for_date(
        self,
        date_str: str,
        station_id: Optional[str] = None
    ) -> List[str]:
        """List channels available for a date."""
        channels = set()

        # Check NDJSON
        ndjson_date_path = self.ndjson_base / f"date={date_str}"
        if ndjson_date_path.exists():
            for item in ndjson_date_path.iterdir():
                if item.is_dir() and item.name.startswith("ch="):
                    channel = item.name.replace("ch=", "")
                    if self._channel_has_events_ndjson(date_str, channel, station_id):
                        channels.add(channel)

        # Also check Parquet
        parquet_channels = self._parquet_reader.list_channels_for_date(
            date_str, station_id
        )
        channels.update(parquet_channels)

        return list(channels)

    def read_channel_ndjson(
        self,
        date_str: str,
        channel: str,
        station_id: Optional[str] = None
    ) -> List[Dict]:
        """Read events from NDJSON file, optionally filtering by station."""
        ndjson_path = (
            self.ndjson_base /
            f"date={date_str}" /
            f"ch={channel}" /
            "events.ndjson"
        )

        if not ndjson_path.exists():
            return []

        events = []
        l2_channels = {"l2_snap", "l2_snap_1s"}
        sample_every = 1
        if channel in l2_channels:
            # Optional explicit sampling knob for very large L2 files.
            try:
                sample_every = max(1, int(os.getenv("HELIOS_L2_READ_SAMPLE_EVERY", "1")))
            except (TypeError, ValueError):
                sample_every = 1
            if sample_every <= 1:
                # Auto-sample only for very large files to bound latency.
                try:
                    target_mb = float(os.getenv("HELIOS_L2_AUTO_SAMPLE_TARGET_MB", "96"))
                except (TypeError, ValueError):
                    target_mb = 96.0
                if target_mb > 0:
                    file_mb = ndjson_path.stat().st_size / (1024 * 1024)
                    if file_mb > target_mb:
                        sample_every = max(1, int(file_mb // target_mb))

        t0 = perf_counter()
        total_lines = 0
        parsed_lines = 0
        try:
            with open(ndjson_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue

                    # Cheap pre-filter: skip lines that clearly belong to another station
                    # without paying JSON decoding cost for huge nested payloads.
                    if station_id and not self._line_matches_station(line, station_id):
                        continue

                    # For high-volume L2 channels, optionally sample lines before decoding.
                    if sample_every > 1 and (line_no % sample_every) != 1:
                        continue

                    try:
                        event = json.loads(line)
                        parsed_lines += 1
                    except json.JSONDecodeError:
                        continue

                    # Filter by station early to keep memory flat.
                    if station_id:
                        ev_station = event.get("station_id")
                        if ev_station not in (station_id, None):
                            continue
                        # Backward compatibility: old events may carry station_id in data.
                        if ev_station is None:
                            data_station = None
                            data = event.get("data")
                            if isinstance(data, dict):
                                data_station = data.get("station_id")
                            if data_station not in (None, station_id):
                                continue

                    # Drop heavy depth arrays for replay/backtest paths.
                    # Simulator/policy only require best bid/ask + depth, not full ladders.
                    if channel in l2_channels:
                        data = event.get("data")
                        if isinstance(data, dict):
                            for bucket, payload in list(data.items()):
                                if not isinstance(payload, dict):
                                    continue
                                if "bids" in payload or "asks" in payload:
                                    slim = dict(payload)
                                    slim.pop("bids", None)
                                    slim.pop("asks", None)
                                    data[bucket] = slim
                            event["data"] = data

                    # Ensure channel is set.
                    event["ch"] = channel
                    events.append(event)
        except Exception as e:
            logger.error(f"Failed to read NDJSON {ndjson_path}: {e}")

        elapsed = perf_counter() - t0
        logger.debug(
            "NDJSON read date=%s ch=%s station=%s sample=%s lines=%s parsed=%s kept=%s elapsed=%.2fs",
            date_str,
            channel,
            station_id,
            sample_every,
            total_lines,
            parsed_lines,
            len(events),
            elapsed,
        )
        if elapsed >= 5.0:
            logger.info(
                "Slow NDJSON read date=%s ch=%s station=%s sample=%s lines=%s kept=%s elapsed=%.2fs",
                date_str,
                channel,
                station_id,
                sample_every,
                total_lines,
                len(events),
                elapsed,
            )
        return events

    def read_channel(
        self,
        date_str: str,
        channel: str,
        station_id: Optional[str] = None
    ) -> List[Dict]:
        """Read channel from NDJSON first, then Parquet."""
        # Try NDJSON first (fresher data)
        events = self.read_channel_ndjson(date_str, channel, station_id=station_id)
        if events:
            return events

        # Fall back to Parquet
        return self._parquet_reader.read_channel(date_str, channel, station_id)

    def read_all_channels(
        self,
        date_str: str,
        station_id: Optional[str] = None,
        channels: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Read all channels for a date.

        Optimized path: avoid expensive station pre-scans in list_channels_for_date
        and let read_channel() do station filtering in one pass.
        """
        available_channels = set()
        ndjson_date_path = self.ndjson_base / f"date={date_str}"
        if ndjson_date_path.exists():
            for item in ndjson_date_path.iterdir():
                if item.is_dir() and item.name.startswith("ch="):
                    available_channels.add(item.name.replace("ch=", ""))

        # Include compacted channels as fallback/merge source.
        available_channels.update(self._parquet_reader.list_channels_for_date(date_str, station_id))
        channel_filter = set(channels) if channels else None
        result = {}

        for channel in available_channels:
            if channel_filter is not None and channel not in channel_filter:
                continue
            events = self.read_channel(date_str, channel, station_id)
            if events:
                result[channel] = events

        return result

    def get_events_sorted(
        self,
        date_str: str,
        station_id: Optional[str] = None,
        channels: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get all events for a date, sorted by timestamp.

        This is the main method for replay.
        """
        all_data = self.read_all_channels(date_str, station_id, channels=channels)

        # Flatten to single list
        all_events = []
        for channel, events in all_data.items():
            all_events.extend(events)

        # Filter by station_id if specified (NDJSON doesn't filter by station)
        if station_id:
            all_events = [
                e for e in all_events
                if e.get("station_id") == station_id or e.get("station_id") is None
            ]

        # Sort by ingest timestamp
        def get_timestamp(e):
            ts = e.get("ts_ingest_utc", "")
            return ts if ts else "9999"

        all_events.sort(key=get_timestamp)

        return all_events


# =============================================================================
# Global accessors
# =============================================================================

_compactor: Optional[Compactor] = None
_reader: Optional[ParquetReader] = None
_hybrid_reader: Optional[HybridReader] = None


def get_compactor(
    ndjson_base: str = "data/recordings",
    parquet_base: str = "data/parquet"
) -> Compactor:
    """Get the global Compactor instance."""
    global _compactor
    if _compactor is None:
        _compactor = Compactor(ndjson_base, parquet_base)
    return _compactor


def get_parquet_reader(parquet_base: str = "data/parquet") -> ParquetReader:
    """Get the global ParquetReader instance."""
    global _reader
    if _reader is None:
        _reader = ParquetReader(parquet_base)
    return _reader


def get_hybrid_reader(
    ndjson_base: str = "data/recordings",
    parquet_base: str = "data/parquet"
) -> HybridReader:
    """Get the global HybridReader instance."""
    global _hybrid_reader
    if _hybrid_reader is None:
        _hybrid_reader = HybridReader(ndjson_base, parquet_base)
    return _hybrid_reader
