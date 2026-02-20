import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hourly_curve import (
    infer_synthetic_peak_hour,
    nudge_curve_to_target_max,
    shift_curve_toward_peak,
)


UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


def _make_base_curve(peak_hour: int = 14) -> list[float]:
    curve = []
    for h in range(24):
        # Simple bell-like daily profile centered on peak_hour.
        curve.append(round(28.0 + max(0.0, 12.0 - abs(h - peak_hour)) * 0.8, 1))
    return curve


def test_shift_curve_toward_peak_moves_peak_later():
    base = _make_base_curve(peak_hour=14)
    shifted = shift_curve_toward_peak(base, target_peak_hour=18, strength=0.8)

    base_peak = max(range(len(base)), key=lambda i: base[i])
    shifted_peak = max(range(len(shifted)), key=lambda i: shifted[i])

    assert base_peak == 14
    assert shifted_peak >= 16
    assert abs(max(shifted) - max(base)) <= 1.0


def test_nudge_curve_to_target_max_tracks_target():
    base = _make_base_curve(peak_hour=15)
    adjusted = nudge_curve_to_target_max(
        base,
        target_tmax_f=42.0,
        anchor_peak_hour=17,
        spread_hours=10.0,
        max_delta_f=12.0,
    )
    assert max(adjusted) > max(base)
    assert abs(max(adjusted) - 42.0) <= 1.2


def test_infer_synthetic_peak_hour_prefers_hint():
    now_utc = datetime(2026, 2, 20, 17, 0, tzinfo=UTC)  # 12:00 NYC (EST)
    peak = infer_synthetic_peak_hour(
        now_utc=now_utc,
        station_tz=NYC,
        current_temp_f=34.0,
        tmax_f=39.0,
        hinted_peak_hour=18,
    )
    assert peak == 18


def test_infer_synthetic_peak_hour_moves_later_with_large_gap():
    now_utc = datetime(2026, 2, 20, 17, 0, tzinfo=UTC)  # 12:00 NYC (EST)
    peak = infer_synthetic_peak_hour(
        now_utc=now_utc,
        station_tz=NYC,
        current_temp_f=31.0,
        tmax_f=40.0,
    )
    assert peak >= 15
