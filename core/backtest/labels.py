"""
Labels Module - Ground Truth and Target Management

Handles the definition and retrieval of "truth" for backtesting:
- y_bucket_winner: winning bucket according to market/judge rules
- y_tmax_aligned: Tmax aligned to judge (contractual rounding)
- y_t_peak_bin: hour bin where maximum occurred

Important: All timestamps use America/New_York for market day alignment.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any, Tuple
from zoneinfo import ZoneInfo
from pathlib import Path
import json

from core.polymarket_labels import (
    normalize_label,
    format_range_label,
    format_below_label,
    format_above_label,
    label_for_temp,
)

logger = logging.getLogger("backtest.labels")

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


@dataclass
class DayLabel:
    """
    Ground truth labels for a single market day.

    Contains both "judge truth" (what matters for PnL) and
    "physical truth" (for diagnostics).
    """
    # Identity
    station_id: str
    market_date: date  # NYC date
    market_id: Optional[str] = None

    # Judge truth (what matters for PnL)
    y_bucket_winner: Optional[str] = None  # e.g., "21-22°F"
    y_tmax_aligned: Optional[float] = None  # Tmax per judge rules (°F)
    y_bucket_index: Optional[int] = None  # 0-indexed bucket position

    # Physical truth (diagnostics)
    y_tmax_physical: Optional[float] = None  # Raw observed max
    y_t_peak_hour_nyc: Optional[int] = None  # Hour (0-23) when max occurred
    y_t_peak_bin: Optional[str] = None  # Bin label, e.g., "14-16"

    # Metadata
    source: str = "unknown"  # Where label came from
    confidence: float = 1.0  # Confidence in label (1.0 = certain)
    notes: str = ""

    # Observations used for labeling
    observations_count: int = 0
    first_obs_utc: Optional[datetime] = None
    last_obs_utc: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "station_id": self.station_id,
            "market_date": self.market_date.isoformat(),
            "market_id": self.market_id,
            "y_bucket_winner": self.y_bucket_winner,
            "y_tmax_aligned": self.y_tmax_aligned,
            "y_bucket_index": self.y_bucket_index,
            "y_tmax_physical": self.y_tmax_physical,
            "y_t_peak_hour_nyc": self.y_t_peak_hour_nyc,
            "y_t_peak_bin": self.y_t_peak_bin,
            "source": self.source,
            "confidence": self.confidence,
            "notes": self.notes,
            "observations_count": self.observations_count,
            "first_obs_utc": self.first_obs_utc.isoformat() if self.first_obs_utc else None,
            "last_obs_utc": self.last_obs_utc.isoformat() if self.last_obs_utc else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DayLabel":
        """Create from dictionary."""
        return cls(
            station_id=data["station_id"],
            market_date=date.fromisoformat(data["market_date"]),
            market_id=data.get("market_id"),
            y_bucket_winner=data.get("y_bucket_winner"),
            y_tmax_aligned=data.get("y_tmax_aligned"),
            y_bucket_index=data.get("y_bucket_index"),
            y_tmax_physical=data.get("y_tmax_physical"),
            y_t_peak_hour_nyc=data.get("y_t_peak_hour_nyc"),
            y_t_peak_bin=data.get("y_t_peak_bin"),
            source=data.get("source", "unknown"),
            confidence=data.get("confidence", 1.0),
            notes=data.get("notes", ""),
            observations_count=data.get("observations_count", 0),
            first_obs_utc=datetime.fromisoformat(data["first_obs_utc"]) if data.get("first_obs_utc") else None,
            last_obs_utc=datetime.fromisoformat(data["last_obs_utc"]) if data.get("last_obs_utc") else None,
        )


class LabelManager:
    """
    Manages ground truth labels for backtesting.

    Responsibilities:
    - Load/save labels from storage
    - Derive labels from observations
    - Apply judge alignment rules
    - Track label provenance
    """

    # Standard bucket ranges for NYC temperature markets
    DEFAULT_BUCKETS = [
        (float('-inf'), 18),
        (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
        (23, 24), (24, 25), (25, 26), (26, 27), (27, 28),
        (28, 29), (29, 30), (30, 31), (31, 32), (32, 33),
        (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
        (38, 39), (39, 40), (40, float('inf'))
    ]

    # T-peak bins (2-hour windows, NYC time)
    TPEAK_BINS = [
        (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12),
        (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 24)
    ]

    def __init__(
        self,
        labels_dir: str = "data/labels",
        judge_rounding: str = "round_half_up"
    ):
        """
        Initialize label manager.

        Args:
            labels_dir: Directory for label storage
            judge_rounding: Rounding method for judge alignment
                           ("round_half_up", "floor", "ceil")
        """
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.judge_rounding = judge_rounding

        # In-memory cache
        self._labels: Dict[Tuple[str, date], DayLabel] = {}

        # Load existing labels
        self._load_all_labels()

        logger.info(f"LabelManager initialized with {len(self._labels)} labels")

    def _load_all_labels(self):
        """Load all labels from storage."""
        for label_file in self.labels_dir.glob("*.json"):
            try:
                with open(label_file, "r") as f:
                    data = json.load(f)
                    label = DayLabel.from_dict(data)
                    if label.y_bucket_winner:
                        label.y_bucket_winner = normalize_label(label.y_bucket_winner)
                    key = (label.station_id, label.market_date)
                    self._labels[key] = label
            except Exception as e:
                logger.warning(f"Failed to load label {label_file}: {e}")

    def _save_label(self, label: DayLabel):
        """Save label to storage."""
        filename = f"{label.station_id}_{label.market_date.isoformat()}.json"
        filepath = self.labels_dir / filename

        with open(filepath, "w") as f:
            json.dump(label.to_dict(), f, indent=2)

    def align_to_judge(self, temp_f: float) -> float:
        """
        Align temperature to judge rules.

        Polymarket typically uses integer °F with specific rounding.
        """
        if self.judge_rounding == "round_half_up":
            # Standard rounding: 0.5 rounds up
            return round(temp_f)
        elif self.judge_rounding == "floor":
            import math
            return math.floor(temp_f)
        elif self.judge_rounding == "ceil":
            import math
            return math.ceil(temp_f)
        else:
            return round(temp_f)

    def get_bucket_for_temp(
        self,
        temp_f: float,
        bucket_labels: Optional[List[str]] = None
    ) -> Tuple[str, int]:
        """
        Get bucket label and index for a temperature.

        Returns:
            (bucket_label, bucket_index)
        """
        aligned = self.align_to_judge(temp_f)

        if bucket_labels:
            label, idx = label_for_temp(aligned, bucket_labels)
            if label is not None and idx is not None:
                return label, idx

        for idx, (low, high) in enumerate(self.DEFAULT_BUCKETS):
            if low <= aligned < high:
                if low == float("-inf"):
                    return format_below_label(int(high)), idx
                if high == float("inf"):
                    return format_above_label(int(low)), idx
                return format_range_label(int(low), int(high)), idx

        return "unknown", -1

    def get_tpeak_bin(self, hour_nyc: int) -> str:
        """Get t-peak bin label for an hour."""
        for start, end in self.TPEAK_BINS:
            if start <= hour_nyc < end:
                return f"{start:02d}-{end:02d}"
        return "unknown"

    def derive_label_from_observations(
        self,
        station_id: str,
        market_date: date,
        observations: List[Dict[str, Any]],
        market_id: Optional[str] = None,
        bucket_labels: Optional[List[str]] = None
    ) -> DayLabel:
        """
        Derive ground truth label from observations.

        Args:
            station_id: Station identifier
            market_date: Market date (NYC)
            observations: List of observation dicts with:
                - ts_utc or obs_time_utc: timestamp
                - temp_f or temperature: temperature in °F
            market_id: Optional market identifier

        Returns:
            DayLabel with derived truth values
        """
        if not observations:
            return DayLabel(
                station_id=station_id,
                market_date=market_date,
                market_id=market_id,
                source="no_observations",
                confidence=0.0,
                notes="No observations available"
            )

        # Filter observations to market day (NYC time)
        day_start_nyc = datetime.combine(market_date, datetime.min.time()).replace(tzinfo=NYC)
        day_end_nyc = day_start_nyc + timedelta(days=1)

        valid_obs = []
        for obs in observations:
            # Parse timestamp
            ts_str = obs.get("ts_utc") or obs.get("obs_time_utc") or obs.get("ts_ingest_utc")
            if not ts_str:
                continue

            try:
                if isinstance(ts_str, str):
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    ts = ts_str

                # Ensure timezone
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)

                # Convert to NYC
                ts_nyc = ts.astimezone(NYC)

                # Check if within market day
                if day_start_nyc <= ts_nyc < day_end_nyc:
                    temp = obs.get("temp_f") or obs.get("temperature") or obs.get("temp_aligned")
                    if temp is not None:
                        valid_obs.append({
                            "ts_utc": ts,
                            "ts_nyc": ts_nyc,
                            "temp_f": float(temp)
                        })
            except Exception as e:
                logger.debug(f"Failed to parse observation: {e}")
                continue

        if not valid_obs:
            return DayLabel(
                station_id=station_id,
                market_date=market_date,
                market_id=market_id,
                source="no_valid_observations",
                confidence=0.0,
                notes=f"No valid observations for {market_date}"
            )

        # Sort by timestamp
        valid_obs.sort(key=lambda x: x["ts_utc"])

        # Find maximum
        max_obs = max(valid_obs, key=lambda x: x["temp_f"])
        tmax_physical = max_obs["temp_f"]
        tmax_aligned = self.align_to_judge(tmax_physical)

        # Get bucket
        bucket_label, bucket_idx = self.get_bucket_for_temp(
            tmax_aligned,
            bucket_labels=bucket_labels
        )

        # Get t-peak info
        peak_hour_nyc = max_obs["ts_nyc"].hour
        tpeak_bin = self.get_tpeak_bin(peak_hour_nyc)

        # Calculate confidence based on observation count and coverage
        hours_covered = len(set(o["ts_nyc"].hour for o in valid_obs))
        confidence = min(1.0, hours_covered / 12.0)  # Want at least 12 hours

        label = DayLabel(
            station_id=station_id,
            market_date=market_date,
            market_id=market_id,
            y_bucket_winner=normalize_label(bucket_label) if bucket_label else None,
            y_tmax_aligned=tmax_aligned,
            y_bucket_index=bucket_idx,
            y_tmax_physical=tmax_physical,
            y_t_peak_hour_nyc=peak_hour_nyc,
            y_t_peak_bin=tpeak_bin,
            source="derived_from_observations",
            confidence=confidence,
            observations_count=len(valid_obs),
            first_obs_utc=valid_obs[0]["ts_utc"],
            last_obs_utc=valid_obs[-1]["ts_utc"],
        )

        return label

    def set_label(self, label: DayLabel, save: bool = True):
        """Store a label."""
        if label.y_bucket_winner:
            label.y_bucket_winner = normalize_label(label.y_bucket_winner)
        key = (label.station_id, label.market_date)
        self._labels[key] = label

        if save:
            self._save_label(label)

        logger.debug(f"Set label for {label.station_id}/{label.market_date}")

    def get_label(
        self,
        station_id: str,
        market_date: date
    ) -> Optional[DayLabel]:
        """Get label for a station/date."""
        key = (station_id, market_date)
        return self._labels.get(key)

    def get_labels_for_range(
        self,
        station_id: str,
        start_date: date,
        end_date: date
    ) -> List[DayLabel]:
        """Get all labels for a date range."""
        labels = []
        current = start_date

        while current <= end_date:
            label = self.get_label(station_id, current)
            if label:
                labels.append(label)
            current += timedelta(days=1)

        return labels

    def list_available_dates(self, station_id: Optional[str] = None) -> List[date]:
        """List all dates with labels."""
        dates = []
        for (sid, dt), label in self._labels.items():
            if station_id is None or sid == station_id:
                dates.append(dt)
        return sorted(set(dates))

    def get_stats(self) -> Dict[str, Any]:
        """Get label statistics."""
        if not self._labels:
            return {
                "total_labels": 0,
                "stations": [],
                "date_range": None
            }

        stations = set()
        dates = []

        for (sid, dt) in self._labels.keys():
            stations.add(sid)
            dates.append(dt)

        return {
            "total_labels": len(self._labels),
            "stations": list(stations),
            "date_range": {
                "start": min(dates).isoformat(),
                "end": max(dates).isoformat()
            } if dates else None,
            "by_station": {
                sid: len([k for k in self._labels.keys() if k[0] == sid])
                for sid in stations
            }
        }


# =============================================================================
# Global accessor
# =============================================================================

_label_manager: Optional[LabelManager] = None


def get_label_manager(labels_dir: str = "data/labels") -> LabelManager:
    """Get the global LabelManager instance."""
    global _label_manager
    if _label_manager is None:
        _label_manager = LabelManager(labels_dir)
    return _label_manager
