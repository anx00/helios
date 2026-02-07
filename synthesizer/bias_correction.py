"""
HELIOS Weather Lab - 7-Day Bias Correction Module
Calculates rolling bias for HRRR predictions to correct systematic errors.
"""

from typing import Optional
from datetime import datetime, timedelta
from database import get_connection


def get_7day_bias(station_id: str) -> float:
    """
    Calculate the 7-day rolling bias for a station.
    
    Bias = mean(predicted_max - actual_max) over last 7 days
    Positive bias means HRRR overestimates (common in winter)
    
    Args:
        station_id: ICAO station code

    Returns:
        Bias value in Fahrenheit (to be subtracted from HRRR base).
        Returns 0.0 if insufficient history (<3 verified predictions).
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Get verified predictions from last 7 days
            seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
            
            cursor.execute("""
                SELECT predicted_max_f, real_max_f
                FROM predictions
                WHERE station_id = ?
                  AND verified_at IS NOT NULL
                  AND real_max_f IS NOT NULL
                  AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 14  -- Up to 2 per day (today/tomorrow predictions)
            """, (station_id, seven_days_ago))
            
            rows = cursor.fetchall()
            
            if len(rows) < 3:
                # Not enough data for reliable bias calculation
                return 0.0
            
            # Calculate mean bias (predicted - actual)
            biases = [row[0] - row[1] for row in rows if row[0] and row[1]]
            
            if not biases:
                return 0.0
            
            mean_bias = sum(biases) / len(biases)
            
            # Cap bias correction to +/- 3°F to avoid overcorrection
            mean_bias = max(-3.0, min(3.0, mean_bias))
            
            return round(mean_bias, 1)
            
    except Exception as e:
        # Fail silently - don't break predictions if DB query fails
        return 0.0


def apply_bias_correction(hrrr_max_f: float, bias: float) -> float:
    """
    Apply bias correction to HRRR base forecast.
    
    Args:
        hrrr_max_f: Raw HRRR maximum temperature forecast (°F)
        bias: Calculated 7-day bias (°F)
        
    Returns:
        Corrected HRRR base temperature (°F)
    """
    # Subtract positive bias (if HRRR overestimates, lower the base)
    corrected = hrrr_max_f - bias
    return round(corrected, 1)
