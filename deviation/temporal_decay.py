"""
Temporal Delta Decay - Advanced deviation tracking

This module enhances the basic deviation tracking with:
1. Time-of-day weighting: Delta has less impact after peak solar hours
2. Multi-day decay: Delta weakens for predictions further in future
3. Historical persistence: Average of last 3 days for long-term bias

CRITICAL FIX: After 15:00 local time, delta weight drops to near-zero
because cooling after peak is NOT predictive of the day's maximum.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from zoneinfo import ZoneInfo
from .db_helper import get_last_n_prediction_errors


def calculate_time_of_day_weight(local_hour: int) -> float:
    """
    Calculate delta weight based on time of day.
    
    The current deviation from HRRR is most relevant in the morning
    when we're predicting forward to the peak. After the peak,
    current temperature is NOT indicative of the day's maximum.
    
    Args:
        local_hour: Current hour in local timezone (0-23)
        
    Returns:
        Weight factor: 1.0 (morning) to 0.0 (post-peak)
        
    Logic:
        - Before 12:00: Weight 1.0 (full projection to future peak)
        - 12:00 - 15:00: Weight 0.5 (approaching peak)
        - After 15:00: Weight 0.0 (peak has passed, delta is noise)
    """
    if local_hour < 12:
        # Morning - delta is highly relevant
        return 1.0
    elif local_hour < 15:
        # Early afternoon - approaching peak
        return 0.5
    else:
        # Post-peak - delta should NOT affect max prediction
        return 0.0


def calculate_multi_day_decay(hours_until_target: int) -> float:
    """
    Calculate decay factor based on time distance to target.
    
    For predictions beyond today, current conditions matter less.
    
    Args:
        hours_until_target: Hours between now and prediction target
        
    Returns:
        Decay factor: 1.0 (same day) to 0.3 (next day+)
    """
    if hours_until_target <= 0:
        return 1.0  # Current or past
    elif hours_until_target <= 12:
        return 1.0  # Same day
    elif hours_until_target <= 24:
        return 0.5  # Tomorrow
    else:
        return 0.3  # Day after tomorrow+


def get_3day_trend(station_id: str) -> float:
    """
    Get average prediction error from last 3 days.
    
    This captures the persistent bias: if HRRR consistently
    over-predicts by +2°F, we should account for that.
    
    Args:
        station_id: Station identifier (e.g., 'KLGA')
        
    Returns:
        Average delta from last 3 days, or 0.0 if insufficient data
    """
    try:
        errors = get_last_n_prediction_errors(station_id, n=3)
        
        if not errors or len(errors) < 2:
            return 0.0  # Need at least 2 days
        
        # Average the errors
        avg_error = sum(errors) / len(errors)
        
        # Clamp to reasonable range
        return max(-3.0, min(3.0, avg_error))
        
    except Exception:
        return 0.0


def calculate_adjusted_delta(
    current_delta: float,
    hours_until_target: int,
    station_id: str,
    local_hour: Optional[int] = None,
    station_timezone: str = "America/New_York"
) -> Tuple[float, str]:
    """
    Calculate delta with time-of-day weight and multi-day decay.
    
    CRITICAL: After 15:00 local, the delta weight is ZERO.
    This prevents late-afternoon cooling from lowering the max prediction.
    
    Formula:
        time_weight = get_time_of_day_weight(local_hour)
        day_decay = get_multi_day_decay(hours_until_target)
        
        effective_weight = time_weight * day_decay
        adjusted_delta = (current_delta * effective_weight) + (trend * (1 - effective_weight))
    
    Args:
        current_delta: Current deviation (actual - hrrr_predicted) in °F
        hours_until_target: Hours until prediction target
        station_id: Station identifier
        local_hour: Current hour in local time (0-23). If None, calculated automatically.
        station_timezone: Timezone for the station
        
    Returns:
        (adjusted_delta, explanation_string)
    """
    # Get local hour if not provided
    if local_hour is None:
        try:
            local_tz = ZoneInfo(station_timezone)
            local_hour = datetime.now(local_tz).hour
        except Exception:
            local_hour = datetime.now(timezone.utc).hour
    
    # Calculate weights
    time_weight = calculate_time_of_day_weight(local_hour)
    day_decay = calculate_multi_day_decay(hours_until_target)
    
    # Combined effective weight
    effective_weight = time_weight * day_decay
    
    # Get historical trend
    trend = get_3day_trend(station_id)
    
    # Apply weighted combination
    if effective_weight <= 0.01:
        # Weight is essentially zero - use only trend (or 0 if no trend)
        adjusted = trend if trend != 0 else 0.0
        explanation = f"Post-peak: Delta ignorado (hora={local_hour}:00)"
    else:
        adjusted = (current_delta * effective_weight) + (trend * (1.0 - effective_weight))
        explanation = f"Peso: {effective_weight*100:.0f}% delta, {(1-effective_weight)*100:.0f}% trend"
    
    return round(adjusted, 1), explanation
