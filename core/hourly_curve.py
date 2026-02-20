"""
Utilities for intraday hourly temperature curve alignment.

These helpers are used to:
- infer a more realistic fallback peak hour when the model curve is missing,
- shift an hourly curve toward a target peak time while preserving shape,
- nudge the curve so its daily max matches an external high-temperature target.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo


def clamp_hour(hour: Optional[int], default: int = 14) -> int:
    """Clamp an hour-like value to [0, 23]."""
    try:
        value = int(hour)
    except Exception:
        value = int(default)
    return max(0, min(23, value))


def infer_synthetic_peak_hour(
    now_utc: datetime,
    station_tz: ZoneInfo,
    current_temp_f: Optional[float],
    tmax_f: float,
    hinted_peak_hour: Optional[int] = None,
) -> int:
    """
    Infer a fallback peak hour for synthetic curves.

    Priority:
    1) Explicit hint (for example WU hourly peak) when available.
    2) Gap-to-target heuristic using local hour and (tmax - current).
    """
    if hinted_peak_hour is not None:
        return clamp_hour(hinted_peak_hour, default=15)

    now_local = now_utc.astimezone(station_tz)
    current_hour = now_local.hour + (now_local.minute / 60.0)

    try:
        target = float(tmax_f)
    except Exception:
        return clamp_hour(int(round(current_hour + 2.0)), default=15)

    obs = None
    if current_temp_f is not None:
        try:
            obs = float(current_temp_f)
        except Exception:
            obs = None

    if obs is None:
        if current_hour < 10.0:
            candidate = 14.5
        elif current_hour < 14.0:
            candidate = current_hour + 2.0
        elif current_hour < 18.0:
            candidate = current_hour + 1.5
        else:
            candidate = current_hour + 1.0
        return clamp_hour(int(round(candidate)), default=15)

    gap = max(0.0, target - obs)
    if current_hour < 9.0:
        candidate = 14.0 + min(2.0, gap / 4.0)
    elif current_hour < 13.0:
        candidate = current_hour + max(1.5, min(5.0, gap * 0.85))
    elif current_hour < 17.0:
        candidate = current_hour + max(1.0, min(4.0, gap * 0.75))
    else:
        candidate = current_hour + max(1.0, min(3.0, gap * 0.50))

    if obs >= (target - 0.2):
        candidate = min(candidate, current_hour + 1.0)

    return clamp_hour(int(round(candidate)), default=15)


def _sample_linear(curve: List[float], x: float) -> float:
    """Linear interpolation with edge clamping."""
    if not curve:
        return 0.0
    if x <= 0.0:
        return float(curve[0])
    last = len(curve) - 1
    if x >= float(last):
        return float(curve[last])
    lo = int(x)
    hi = min(last, lo + 1)
    frac = x - lo
    return float(curve[lo]) * (1.0 - frac) + float(curve[hi]) * frac


def shift_curve_toward_peak(
    hourly_temps_f: List[float],
    target_peak_hour: int,
    strength: float = 0.75,
    influence_span_hours: float = 12.0,
) -> List[float]:
    """
    Shift an hourly curve toward a target peak hour while preserving shape.

    The curve is time-shifted with interpolation and blended back to avoid
    discontinuities or unrealistic flat segments.
    """
    if not hourly_temps_f:
        return []

    curve = [float(t) for t in hourly_temps_f]
    n = len(curve)
    if n < 2:
        return [round(v, 1) for v in curve]

    current_peak = int(max(range(n), key=lambda i: curve[i]))
    target = max(0, min(n - 1, int(target_peak_hour)))
    shift_hours = float(target - current_peak)
    if abs(shift_hours) < 0.25:
        return [round(v, 1) for v in curve]

    blend = max(0.0, min(1.0, float(strength)))
    span = max(1.0, float(influence_span_hours))
    shifted = [_sample_linear(curve, h - shift_hours) for h in range(n)]

    aligned: List[float] = []
    for h in range(n):
        influence = max(0.15, 1.0 - (abs(h - target) / span))
        w = blend * influence
        value = curve[h] + ((shifted[h] - curve[h]) * w)
        aligned.append(round(float(value), 1))
    return aligned


def nudge_curve_to_target_max(
    hourly_temps_f: List[float],
    target_tmax_f: float,
    anchor_peak_hour: Optional[int] = None,
    spread_hours: float = 10.0,
    max_delta_f: float = 12.0,
) -> List[float]:
    """
    Nudge a curve so its maximum tracks a target max, preserving shape.
    """
    if not hourly_temps_f:
        return []

    curve = [float(t) for t in hourly_temps_f]
    try:
        target = float(target_tmax_f)
    except Exception:
        return [round(v, 1) for v in curve]

    current_max = max(curve)
    delta = target - current_max
    delta = max(-abs(max_delta_f), min(abs(max_delta_f), delta))
    if abs(delta) < 0.05:
        return [round(v, 1) for v in curve]

    n = len(curve)
    if n < 1:
        return []
    if anchor_peak_hour is None:
        anchor = int(max(range(n), key=lambda i: curve[i]))
    else:
        anchor = max(0, min(n - 1, int(anchor_peak_hour)))

    spread = max(1.0, float(spread_hours))
    adjusted: List[float] = []
    for h, temp in enumerate(curve):
        weight = max(0.0, 1.0 - (abs(h - anchor) / spread))
        adjusted.append(round(float(temp + (delta * weight)), 1))
    return adjusted
