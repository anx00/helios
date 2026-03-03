"""
Ensemble Fetcher - Open-Meteo Ensemble API (ECMWF + GFS).

Fetches individual ensemble member forecasts and computes empirical
probability distributions over Polymarket temperature buckets.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx

from config import STATIONS, get_polymarket_temp_unit
from core.polymarket_labels import normalize_label, parse_label

logger = logging.getLogger(__name__)

DEGREE = "\N{DEGREE SIGN}"
ENSEMBLE_API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENSEMBLE_MODELS = ["ecmwf_ifs025", "gfs_seamless"]
_CACHE_TTL_SECONDS = 600
_cache: Dict[str, Dict[str, Any]] = {}


@dataclass
class EnsembleBucketProb:
    """Probability for a single temperature bucket from ensemble counting."""

    bucket_low: float
    bucket_high: float
    unit: str
    probability: float
    member_count: int
    total_members: int

    @property
    def label(self) -> str:
        low = int(self.bucket_low)
        high = int(self.bucket_high)
        suffix = f"{DEGREE}{self.unit}"
        if low == high:
            return f"{low}{suffix}"
        return f"{low}-{high}{suffix}"


@dataclass
class EnsembleSnapshot:
    """Full ensemble analysis for a station+date."""

    station_id: str
    target_date: str
    unit: str
    generated_utc: str
    total_members: int
    member_tmax_values: List[float]
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
        """Get probability for a bucket by Polymarket label semantics."""
        normalized = normalize_label(label)
        for bucket in self.bucket_probabilities:
            if normalize_label(bucket.label) == normalized:
                return bucket.probability

        kind, low, high = parse_label(normalized)
        rounded_values = _rounded_member_temperatures(self.member_tmax_values, self.unit)
        if not rounded_values or kind == "unknown":
            return None

        if kind == "below" and high is not None:
            count = sum(1 for value in rounded_values if value <= high)
        elif kind == "above" and low is not None:
            count = sum(1 for value in rounded_values if value >= low)
        elif kind == "range" and low is not None and high is not None:
            count = sum(1 for value in rounded_values if low <= value <= high)
        elif kind == "single" and low is not None:
            count = sum(1 for value in rounded_values if value == low)
        else:
            return None

        return count / len(rounded_values)

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
            "member_tmax_values": [round(value, 1) for value in self.member_tmax_values],
            "bucket_probabilities": [
                {
                    "label": bucket.label,
                    "bucket_low": bucket.bucket_low,
                    "bucket_high": bucket.bucket_high,
                    "probability": round(bucket.probability, 4),
                    "member_count": bucket.member_count,
                    "total_members": bucket.total_members,
                }
                for bucket in self.bucket_probabilities
            ],
            "models_used": self.models_used,
            "errors": self.errors,
            "notes": self.notes,
        }


def _c_to_f(temp_c: float) -> float:
    return (temp_c * 9.0 / 5.0) + 32.0


def _round_half_up_int(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _rounded_member_temperatures(tmax_values: List[float], unit: str) -> List[int]:
    if unit == "F":
        return [_round_half_up_int(_c_to_f(value)) for value in tmax_values]
    return [_round_half_up_int(value) for value in tmax_values]


def _extract_member_series(hourly: Dict[str, Any]) -> List[List[float]]:
    """Extract all member temperature series from the API response."""
    members: List[List[float]] = []

    control_series = hourly.get("temperature_2m")
    if isinstance(control_series, list) and control_series:
        members.append([value for value in control_series if value is not None])

    idx = 1
    while True:
        key = f"temperature_2m_member{idx:02d}"
        if key not in hourly:
            break
        series = hourly.get(key)
        if isinstance(series, list) and series:
            members.append([value for value in series if value is not None])
        idx += 1

    return members


def _compute_daily_tmax_per_member(
    times: List[str],
    member_series: List[List[float]],
    target_date: date,
    station_tz: ZoneInfo,
) -> List[float]:
    """For each ensemble member, find the max temperature on target_date."""
    target_indices: List[int] = []
    for idx, ts_str in enumerate(times):
        try:
            dt = datetime.fromisoformat(ts_str)
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=station_tz)
        if dt.date() == target_date:
            target_indices.append(idx)

    if not target_indices:
        return []

    tmax_values: List[float] = []
    for series in member_series:
        temps = [series[idx] for idx in target_indices if idx < len(series) and series[idx] is not None]
        if temps:
            tmax_values.append(max(temps))
    return tmax_values


def _build_bucket_probabilities(tmax_values: List[float], unit: str) -> List[EnsembleBucketProb]:
    """Count ensemble members in canonical Polymarket-aligned buckets."""
    rounded_values = _rounded_member_temperatures(tmax_values, unit)
    if not rounded_values:
        return []

    total = len(rounded_values)

    if unit == "F":
        min_val = min(rounded_values)
        max_val = max(rounded_values)
        bucket_start = min_val - (min_val % 2)
        bucket_end = max_val - (max_val % 2)

        buckets: List[EnsembleBucketProb] = []
        low = bucket_start
        while low <= bucket_end:
            high = low + 1
            count = sum(1 for value in rounded_values if low <= value <= high)
            buckets.append(
                EnsembleBucketProb(
                    bucket_low=low,
                    bucket_high=high,
                    unit="F",
                    probability=count / total,
                    member_count=count,
                    total_members=total,
                )
            )
            low += 2
        return buckets

    min_val = min(rounded_values)
    max_val = max(rounded_values)
    return [
        EnsembleBucketProb(
            bucket_low=temp,
            bucket_high=temp,
            unit="C",
            probability=sum(1 for value in rounded_values if value == temp) / total,
            member_count=sum(1 for value in rounded_values if value == temp),
            total_members=total,
        )
        for temp in range(min_val, max_val + 1)
    ]


async def fetch_ensemble_snapshot(
    station_id: str,
    target_date: Optional[date] = None,
    *,
    models: Optional[List[str]] = None,
) -> EnsembleSnapshot:
    """Fetch ensemble forecasts and compute bucket probabilities."""
    station = STATIONS.get(station_id)
    if not station:
        raise ValueError(f"Unknown station: {station_id}")

    station_tz = ZoneInfo(station.timezone)
    now_utc = datetime.now(timezone.utc)
    if target_date is None:
        target_date = now_utc.astimezone(station_tz).date()

    unit = get_polymarket_temp_unit(station_id)
    models_to_use = models or list(ENSEMBLE_MODELS)
    cache_key = f"{station_id}:{target_date.isoformat()}:{','.join(models_to_use)}"
    cached = _cache.get(cache_key)
    if cached and (time.monotonic() - cached["ts"]) < _CACHE_TTL_SECONDS:
        snapshot = cached["snapshot"]
        snapshot.notes = [f"Served from cache ({int(time.monotonic() - cached['ts'])}s old)"] + snapshot.notes
        return snapshot

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
                response = await client.get(ENSEMBLE_API_URL, params=params)
                if response.status_code == 429:
                    errors.append(f"{model}: rate limited (HTTP 429)")
                    continue
                response.raise_for_status()
                data = response.json()
                hourly = data.get("hourly", {})
                times = hourly.get("time", [])
                member_series = _extract_member_series(hourly)
                tmax_values = _compute_daily_tmax_per_member(times, member_series, target_date, station_tz)
                if tmax_values:
                    all_tmax_values.extend(tmax_values)
                    models_fetched.append(model)
                    notes.append(f"{model}: {len(tmax_values)} members")
                else:
                    errors.append(f"{model}: no data for {target_date}")
            except httpx.HTTPStatusError as exc:
                errors.append(f"{model}: HTTP {exc.response.status_code}")
            except Exception as exc:
                errors.append(f"{model}: {type(exc).__name__}: {exc}")

    if all_tmax_values:
        sorted_values = sorted(all_tmax_values)
        count = len(sorted_values)
        mean = sum(sorted_values) / count
        if count % 2 == 1:
            median = sorted_values[count // 2]
        else:
            median = (sorted_values[count // 2 - 1] + sorted_values[count // 2]) / 2
        variance = sum((value - mean) ** 2 for value in sorted_values) / count
        std = math.sqrt(variance)
        bucket_probabilities = _build_bucket_probabilities(all_tmax_values, unit)
    else:
        mean = 0.0
        median = 0.0
        std = 0.0
        bucket_probabilities = []
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
        bucket_probabilities=bucket_probabilities,
        models_used=models_fetched,
        errors=errors,
        notes=notes,
    )

    _cache[cache_key] = {"ts": time.monotonic(), "snapshot": snapshot}
    logger.info(
        "Ensemble %s %s: %d members, mean=%.1fC, std=%.2f, %d buckets",
        station_id,
        target_date,
        len(all_tmax_values),
        mean,
        std,
        len(bucket_probabilities),
    )
    return snapshot


async def fetch_ensemble_edge(
    station_id: str,
    market_buckets: List[Dict[str, Any]],
    target_date: Optional[date] = None,
    *,
    min_edge_pct: float = 8.0,
) -> List[Dict[str, Any]]:
    """Compare ensemble probabilities against market prices to find edges."""
    snapshot = await fetch_ensemble_snapshot(station_id, target_date)
    min_edge = min_edge_pct / 100.0
    edges: List[Dict[str, Any]] = []

    for market in market_buckets:
        label = normalize_label(str(market.get("label") or ""))
        price = float(market.get("price", 0))
        if not label:
            continue

        ensemble_prob = snapshot.get_bucket_prob(label)
        if ensemble_prob is None:
            if price > min_edge:
                edges.append(
                    {
                        "label": label,
                        "ensemble_prob": 0.0,
                        "market_price": price,
                        "edge_pct": round(-price * 100, 1),
                        "signal": "SELL" if price > 0.15 else "SKIP",
                        "reason": "Ensemble assigns ~0% probability",
                    }
                )
            continue

        edge = ensemble_prob - price
        if abs(edge) < min_edge:
            continue

        edges.append(
            {
                "label": label,
                "ensemble_prob": round(ensemble_prob, 4),
                "market_price": round(price, 4),
                "edge_pct": round(edge * 100, 1),
                "signal": "BUY_YES" if edge > 0 else "BUY_NO",
                "reason": f"Ensemble {ensemble_prob:.1%} vs market {price:.1%}",
            }
        )

    edges.sort(key=lambda item: abs(item["edge_pct"]), reverse=True)
    return edges
