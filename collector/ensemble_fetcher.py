"""
Ensemble Fetcher — Open-Meteo Ensemble API (ECMWF + GFS)

Fetches individual ensemble member forecasts and computes empirical
probability distributions over Polymarket temperature buckets.

This is the core "ensemble counting" approach used by profitable
weather traders: count how many members fall in each bucket to get
model-implied probabilities, then compare against market prices.

API: https://ensemble-api.open-meteo.com/v1/ensemble
  - ECMWF IFS (51 members)
  - GFS Seamless (31 members)
  - Total: 82 independent forecasts
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx

from config import STATIONS, get_polymarket_temp_unit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENSEMBLE_API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Models to fetch — each returns separate member arrays
ENSEMBLE_MODELS = ["ecmwf_ifs025", "gfs_seamless"]

# Cache TTL — ensemble data updates every 6h, 10-min cache is safe
_CACHE_TTL_SECONDS = 600
_cache: Dict[str, Dict[str, Any]] = {}

# Polymarket bucket width (°F for US, °C for EU)
_US_BUCKET_WIDTH = 2  # 2°F wide buckets (e.g., 40-41, 42-43)
_EU_BUCKET_WIDTH = 1  # 1°C wide buckets


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EnsembleBucketProb:
    """Probability for a single temperature bucket from ensemble counting."""
    bucket_low: float       # Lower bound (inclusive)
    bucket_high: float      # Upper bound (inclusive)
    unit: str               # "F" or "C"
    probability: float      # Fraction of members in this bucket
    member_count: int       # Number of members in this bucket
    total_members: int      # Total ensemble members

    @property
    def label(self) -> str:
        if self.unit == "F":
            return f"{int(self.bucket_low)}-{int(self.bucket_high)}°F"
        return f"{int(self.bucket_low)}-{int(self.bucket_high)}°C"


@dataclass
class EnsembleSnapshot:
    """Full ensemble analysis for a station+date."""
    station_id: str
    target_date: str                    # YYYY-MM-DD
    unit: str                           # "F" or "C"
    generated_utc: str
    total_members: int
    member_tmax_values: List[float]     # Raw Tmax from each member
    ensemble_mean: float
    ensemble_median: float
    ensemble_std: float
    ensemble_min: float
    ensemble_max: float
    bucket_probabilities: List[EnsembleBucketProb]
    models_used: List[str]
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def get_bucket_prob(self, label: str) -> Optional[float]:
        """Get probability for a bucket by its label."""
        for bp in self.bucket_probabilities:
            if bp.label == label:
                return bp.probability
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "target_date": self.target_date,
            "unit": self.unit,
            "generated_utc": self.generated_utc,
            "total_members": self.total_members,
            "ensemble_mean": round(self.ensemble_mean, 2),
            "ensemble_median": round(self.ensemble_median, 2),
            "ensemble_std": round(self.ensemble_std, 2),
            "ensemble_min": round(self.ensemble_min, 2),
            "ensemble_max": round(self.ensemble_max, 2),
            "member_tmax_values": [round(v, 1) for v in self.member_tmax_values],
            "bucket_probabilities": [
                {
                    "label": bp.label,
                    "bucket_low": bp.bucket_low,
                    "bucket_high": bp.bucket_high,
                    "probability": round(bp.probability, 4),
                    "member_count": bp.member_count,
                    "total_members": bp.total_members,
                }
                for bp in self.bucket_probabilities
            ],
            "models_used": self.models_used,
            "errors": self.errors,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _c_to_f(c: float) -> float:
    return (c * 9.0 / 5.0) + 32.0


def _extract_member_series(hourly: Dict[str, Any], model: str) -> List[List[float]]:
    """Extract all member temperature series from the API response.

    Returns list of lists — each inner list is the hourly temps for one member.
    """
    members: List[List[float]] = []

    # Primary member (control run)
    key_base = "temperature_2m"
    if key_base in hourly:
        series = hourly[key_base]
        if isinstance(series, list) and series:
            members.append([v for v in series if v is not None])

    # Numbered members: temperature_2m_member01 ... temperature_2m_memberNN
    idx = 1
    while True:
        key = f"temperature_2m_member{idx:02d}"
        if key not in hourly:
            break
        series = hourly[key]
        if isinstance(series, list) and series:
            members.append([v for v in series if v is not None])
        idx += 1

    return members


def _compute_daily_tmax_per_member(
    times: List[str],
    member_series: List[List[float]],
    target_date: date,
    station_tz: ZoneInfo,
) -> List[float]:
    """For each ensemble member, find the max temperature on target_date.

    The API returns temps in °C. We filter hours that fall on target_date
    in the station's local timezone, then take the max for each member.
    """
    # Find indices that correspond to target_date in local time
    target_indices: List[int] = []
    for i, ts_str in enumerate(times):
        try:
            # Open-Meteo returns ISO timestamps in the requested timezone
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=station_tz)
            if dt.date() == target_date:
                target_indices.append(i)
        except Exception:
            continue

    if not target_indices:
        return []

    tmax_values: List[float] = []
    for series in member_series:
        valid_temps = []
        for idx in target_indices:
            if idx < len(series) and series[idx] is not None:
                valid_temps.append(series[idx])
        if valid_temps:
            tmax_values.append(max(valid_temps))

    return tmax_values


def _build_bucket_probabilities(
    tmax_values: List[float],
    unit: str,
) -> List[EnsembleBucketProb]:
    """Count ensemble members in each Polymarket bucket.

    For °F markets: buckets are 2°F wide, aligned to even numbers (40-41, 42-43, ...).
    For °C markets: buckets are 1°C wide (20, 21, 22, ...).
    """
    if not tmax_values:
        return []

    total = len(tmax_values)

    if unit == "F":
        # Convert all to °F
        values_f = [_c_to_f(v) for v in tmax_values]
        # Polymarket °F buckets: 2°F wide, even-aligned
        # Round each value to nearest integer, then assign to bucket
        rounded = [round(v) for v in values_f]
        min_val = min(rounded)
        max_val = max(rounded)

        # Align to even bucket start
        bucket_start = min_val - (min_val % 2)
        bucket_end = max_val + (2 - max_val % 2) if max_val % 2 != 1 else max_val

        buckets: List[EnsembleBucketProb] = []
        low = bucket_start
        while low <= bucket_end:
            high = low + 1
            count = sum(1 for v in rounded if low <= v <= high)
            buckets.append(EnsembleBucketProb(
                bucket_low=low,
                bucket_high=high,
                unit="F",
                probability=count / total,
                member_count=count,
                total_members=total,
            ))
            low += 2
    else:
        # °C markets: 1°C wide buckets
        values_c = list(tmax_values)
        rounded = [round(v) for v in values_c]
        min_val = min(rounded)
        max_val = max(rounded)

        buckets = []
        for temp in range(min_val, max_val + 1):
            count = sum(1 for v in rounded if v == temp)
            buckets.append(EnsembleBucketProb(
                bucket_low=temp,
                bucket_high=temp,
                unit="C",
                probability=count / total,
                member_count=count,
                total_members=total,
            ))

    return buckets


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

async def fetch_ensemble_snapshot(
    station_id: str,
    target_date: Optional[date] = None,
    *,
    models: Optional[List[str]] = None,
) -> EnsembleSnapshot:
    """Fetch ensemble forecasts and compute bucket probabilities.

    Args:
        station_id: ICAO station code (e.g., "KLGA")
        target_date: Date to analyze (default: today in station TZ)
        models: Override ensemble models (default: ECMWF + GFS)

    Returns:
        EnsembleSnapshot with member Tmax values and bucket probabilities
    """
    station = STATIONS.get(station_id)
    if not station:
        raise ValueError(f"Unknown station: {station_id}")

    station_tz = ZoneInfo(station.timezone)
    now_utc = datetime.now(timezone.utc)

    if target_date is None:
        target_date = now_utc.astimezone(station_tz).date()

    unit = get_polymarket_temp_unit(station_id)
    models_to_use = models or list(ENSEMBLE_MODELS)

    # Check cache
    cache_key = f"{station_id}:{target_date.isoformat()}:{','.join(models_to_use)}"
    cached = _cache.get(cache_key)
    if cached and (time.monotonic() - cached["ts"]) < _CACHE_TTL_SECONDS:
        snapshot = cached["snapshot"]
        snapshot.notes = [f"Served from cache ({int(time.monotonic() - cached['ts'])}s old)"] + snapshot.notes
        return snapshot

    # Determine forecast_days needed
    days_ahead = (target_date - now_utc.astimezone(station_tz).date()).days
    forecast_days = max(2, days_ahead + 1)

    all_tmax_values: List[float] = []
    models_fetched: List[str] = []
    errors: List[str] = []
    notes: List[str] = []

    async with httpx.AsyncClient(timeout=25.0) as client:
        for model in models_to_use:
            params = {
                "latitude": station.latitude,
                "longitude": station.longitude,
                "hourly": "temperature_2m",
                "models": model,
                "timezone": station.timezone,
                "forecast_days": forecast_days,
            }

            try:
                resp = await client.get(ENSEMBLE_API_URL, params=params)
                if resp.status_code == 429:
                    errors.append(f"{model}: rate limited (HTTP 429)")
                    continue
                resp.raise_for_status()
                data = resp.json()

                hourly = data.get("hourly", {})
                times = hourly.get("time", [])
                member_series = _extract_member_series(hourly, model)

                tmax_vals = _compute_daily_tmax_per_member(
                    times, member_series, target_date, station_tz,
                )

                if tmax_vals:
                    all_tmax_values.extend(tmax_vals)
                    models_fetched.append(model)
                    notes.append(f"{model}: {len(tmax_vals)} members")
                else:
                    errors.append(f"{model}: no data for {target_date}")

            except httpx.HTTPStatusError as exc:
                errors.append(f"{model}: HTTP {exc.response.status_code}")
            except Exception as exc:
                errors.append(f"{model}: {type(exc).__name__}: {exc}")

    # Build snapshot
    if all_tmax_values:
        sorted_vals = sorted(all_tmax_values)
        n = len(sorted_vals)
        median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        mean = sum(sorted_vals) / n
        variance = sum((v - mean) ** 2 for v in sorted_vals) / n
        std = math.sqrt(variance)

        bucket_probs = _build_bucket_probabilities(all_tmax_values, unit)
    else:
        mean = median = std = 0.0
        sorted_vals = []
        bucket_probs = []
        errors.append("No ensemble data available")

    snapshot = EnsembleSnapshot(
        station_id=station_id,
        target_date=target_date.isoformat(),
        unit=unit,
        generated_utc=now_utc.isoformat(),
        total_members=len(all_tmax_values),
        member_tmax_values=all_tmax_values,
        ensemble_mean=mean,
        ensemble_median=median,
        ensemble_std=std,
        ensemble_min=min(all_tmax_values) if all_tmax_values else 0.0,
        ensemble_max=max(all_tmax_values) if all_tmax_values else 0.0,
        bucket_probabilities=bucket_probs,
        models_used=models_fetched,
        errors=errors,
        notes=notes,
    )

    # Cache result
    _cache[cache_key] = {"ts": time.monotonic(), "snapshot": snapshot}

    logger.info(
        "Ensemble %s %s: %d members, mean=%.1f°C, std=%.2f, %d buckets",
        station_id, target_date, len(all_tmax_values), mean, std, len(bucket_probs),
    )

    return snapshot


async def fetch_ensemble_edge(
    station_id: str,
    market_buckets: List[Dict[str, Any]],
    target_date: Optional[date] = None,
    *,
    min_edge_pct: float = 8.0,
) -> List[Dict[str, Any]]:
    """Compare ensemble probabilities against market prices to find edges.

    Args:
        station_id: ICAO station code
        market_buckets: List of dicts with keys: label, price (0-1)
        target_date: Date to analyze
        min_edge_pct: Minimum edge in percentage points to flag

    Returns:
        List of dicts with: label, ensemble_prob, market_price, edge_pct, signal
    """
    snapshot = await fetch_ensemble_snapshot(station_id, target_date)
    min_edge = min_edge_pct / 100.0

    edges: List[Dict[str, Any]] = []
    for market in market_buckets:
        label = market.get("label", "")
        price = float(market.get("price", 0))

        ensemble_prob = snapshot.get_bucket_prob(label)
        if ensemble_prob is None:
            # Bucket not covered by ensemble → model says ~0% probability
            if price > min_edge:
                edges.append({
                    "label": label,
                    "ensemble_prob": 0.0,
                    "market_price": price,
                    "edge_pct": round(-price * 100, 1),
                    "signal": "SELL" if price > 0.15 else "SKIP",
                    "reason": "Ensemble assigns ~0% probability",
                })
            continue

        edge = ensemble_prob - price

        if abs(edge) >= min_edge:
            signal = "BUY_YES" if edge > 0 else "BUY_NO"
            edges.append({
                "label": label,
                "ensemble_prob": round(ensemble_prob, 4),
                "market_price": round(price, 4),
                "edge_pct": round(edge * 100, 1),
                "signal": signal,
                "reason": f"Ensemble {ensemble_prob:.1%} vs market {price:.1%}",
            })

    # Sort by absolute edge descending
    edges.sort(key=lambda x: abs(x["edge_pct"]), reverse=True)
    return edges
