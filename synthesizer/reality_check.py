"""
Reality Check Module - Prevents predictions below observed maximum

This module ensures that predictions never violate physical causality:
- The predicted maximum cannot be lower than what has already been observed
- Implements "floor logic" based on actual temperature history
"""
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from collector.metar_fetcher import fetch_metar_history


async def get_max_observed_today(
    station_id: str,
    station_timezone: str = "America/New_York"
) -> Tuple[Optional[float], Optional[datetime]]:
    """
    Get the maximum temperature observed today so far.
    
    Args:
        station_id: ICAO station identifier
        station_timezone: Timezone for the station
        
    Returns:
        Tuple of (max_temp_c, time_of_max) or (None, None) if no data
    """
    try:
        # Get 24h of history
        history = await fetch_metar_history(station_id, hours=24)
        
        if not history:
            return None, None
        
        # Get local timezone
        local_tz = ZoneInfo(station_timezone)
        now_local = datetime.now(local_tz)
        today_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Filter to today's observations only
        todays_obs = []
        for obs in history:
            if obs.temp_c is None:
                continue
            
            # Convert observation time to local timezone
            obs_local = obs.observation_time.astimezone(local_tz)
            
            if obs_local >= today_start:
                todays_obs.append((obs.temp_c, obs.observation_time))
        
        if not todays_obs:
            return None, None
        
        # Find maximum
        max_temp, max_time = max(todays_obs, key=lambda x: x[0])
        
        return max_temp, max_time
        
    except Exception as e:
        print(f"[REALITY CHECK] Error getting max observed: {e}")
        return None, None


def apply_floor_logic(
    calculated_prediction_f: float,
    max_observed_f: Optional[float],
    max_observed_time: Optional[datetime]
) -> Tuple[float, str]:
    """
    Apply floor logic: prediction cannot be lower than observed maximum.
    
    Args:
        calculated_prediction_f: The calculated prediction in Fahrenheit
        max_observed_f: Maximum observed temperature today in Fahrenheit
        max_observed_time: When the maximum was observed
        
    Returns:
        Tuple of (final_prediction_f, explanation)
    """
    if max_observed_f is None:
        return calculated_prediction_f, "No floor (no history)"
    
    if calculated_prediction_f < max_observed_f:
        # Prediction would violate physics - use floor
        time_str = max_observed_time.strftime("%H:%M") if max_observed_time else "today"
        explanation = f"Floor applied: {max_observed_f:.1f}°F @ {time_str}"
        return max_observed_f, explanation
    else:
        return calculated_prediction_f, "Above floor"


def celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (temp_c * 9/5) + 32
