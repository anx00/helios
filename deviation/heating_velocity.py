"""
Heating Velocity Tracker - Momentum térmico

Mide cuán rápido se está calentando la estación comparado
con la tasa que HRRR esperaba, para capturar "momentum".
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List


def make_tz_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def calculate_heating_velocity(
    station_id: str,
    current_temp_c: float,
    time_now: datetime,
    trajectory: List[Tuple[datetime, float]]
) -> Tuple[float, str]:
    """
    Calculate heating velocity ratio (actual/expected).
    
    Args:
        station_id: Station identifier
        current_temp_c: Current temperature in Celsius
        time_now: Current time (must be timezone-aware)
        trajectory: List of (datetime, temp_c) from HRRR morning capture
        
    Returns:
        (velocity_ratio, explanation)
        - ratio > 1.2: Heating faster than expected
        - ratio < 0.8: Heating slower than expected
        - ratio ~1.0: On track
    """
    try:
        # Ensure time_now is timezone-aware
        time_now = make_tz_aware(time_now)
        
        # Get temperature from 1 hour ago from trajectory
        one_hour_ago = time_now - timedelta(hours=1)
        
        # Find closest trajectory point to 1 hour ago
        temp_1h_ago = None
        min_diff = timedelta(hours=999)
        
        for traj_time, traj_temp in trajectory:
            # Make trajectory time timezone-aware
            traj_time_aware = make_tz_aware(traj_time)
            diff = abs(traj_time_aware - one_hour_ago)
            if diff < min_diff:
                min_diff = diff
                temp_1h_ago = traj_temp
        
        if temp_1h_ago is None:
            return 1.0, "No trajectory data"
        
        # Actual heating rate (°C/hour)
        actual_rate = current_temp_c - temp_1h_ago
        
        # Expected heating rate from HRRR trajectory
        temp_now_expected = None
        min_diff = timedelta(hours=999)
        
        for traj_time, traj_temp in trajectory:
            traj_time_aware = make_tz_aware(traj_time)
            diff = abs(traj_time_aware - time_now)
            if diff < min_diff:
                min_diff = diff
                temp_now_expected = traj_temp
        
        if temp_now_expected is None:
            return 1.0, "No current trajectory point"
        
        expected_rate = temp_now_expected - temp_1h_ago
        
        # Calculate velocity ratio
        if abs(expected_rate) < 0.1:
            # Minimal expected change, ratio not meaningful
            return 1.0, "Minimal expected change"
        
        velocity_ratio = actual_rate / expected_rate
        
        # Cap extreme values
        velocity_ratio = max(0.5, min(2.0, velocity_ratio))
        
        # Build explanation
        if velocity_ratio > 1.2:
            explanation = f"Heating {velocity_ratio:.1f}x faster (momentum +)"
        elif velocity_ratio < 0.8:
            explanation = f"Heating {velocity_ratio:.1f}x slower (momentum -)"
        else:
            explanation = f"On track ({velocity_ratio:.1f}x)"
        
        return velocity_ratio, explanation
        
    except Exception as e:
        # Safe fallback
        return 1.0, f"Error: {str(e)[:30]}"


def get_velocity_adjustment(velocity_ratio: float) -> Tuple[float, str]:
    """
    Get temperature adjustment based on heating velocity.
    
    Rules:
    - ratio > 1.2: Add +1.0°F (heating faster, expect higher max)
    - ratio < 0.8: Subtract -0.5°F (heating slower, expect lower max)
    
    Args:
        velocity_ratio: Heating velocity ratio from calculate_heating_velocity
        
    Returns:
        (adjustment_f, reason_string)
    """
    if velocity_ratio > 1.2:
        return 1.0, "FastHeating+1F"
    elif velocity_ratio < 0.8:
        return -0.5, "SlowHeating-0.5F"
    else:
        return 0.0, ""
