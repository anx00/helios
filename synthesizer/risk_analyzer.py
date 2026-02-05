"""
Risk Analyzer - HELIOS v2.1
Quantifies confidence and rounding risks for final predictions.
"""
from typing import List, Optional, Tuple
from datetime import datetime

def celsius_to_fahrenheit(c: float) -> float:
    return (c * 9/5) + 32

def analyze_rounding_risk(prediction_f: float) -> Tuple[str, float, str]:
    """
    Analyze risk of rounding to integer.
    
    Safe: .00-.20, .80-.99
    Critical: .40-.60 (Close to .5 flip point)
    """
    decimal_part = prediction_f % 1
    dist_to_half = abs(decimal_part - 0.5)
    
    # If x.45 -> dist is 0.05. If x.55 -> dist is 0.05.
    
    if dist_to_half <= 0.05: # .45 to .55
        risk = "CRITICAL"
        # Calculate nearest ints
        target = int(round(prediction_f))
        msg = f"CRÍTICO (A {dist_to_half:.2f}°F del cambio de entero)"
    elif dist_to_half <= 0.15: # .35-.45 or .55-.65
        risk = "WARNING"
        msg = f"ALTO ({prediction_f:.2f}°F)"
    else:
        risk = "SAFE"
        msg = "Zona Segura"

    return risk, dist_to_half, msg

def analyze_night_risk(
    current_temp_f: float,
    hourly_temps_c: List[float],
    hourly_times: List[str],
    current_local_hour: int
) -> Tuple[bool, str, float]:
    """
    Analyze if temperature might rise before midnight.
    Only active if current_hour > 16.
    
    Returns:
        is_active (bool)
        risk_level (str): LOW/HIGH
        future_max_f (float): Expected max in remaining hours
    """
    if current_local_hour <= 16:
        return False, "N/A", 0.0
        
    future_max_c = -999.0
    found_future = False
    
    # Get today's date string prefix from the first timestamp or assumption
    # OpenMeteo returns ISO strings "YYYY-MM-DDTHH:MM"
    # We assume the list starts with today. We iterate until hour resets to 0 (tomorrow)
    
    if not hourly_times or not hourly_temps_c:
         return False, "N/A", 0.0

    today_date = hourly_times[0].split('T')[0]

    for t_str, temp_c in zip(hourly_times, hourly_temps_c):
        if not temp_c: continue
        
        parts = t_str.split('T')
        date_part = parts[0]
        time_part = parts[1]
        
        # Stop if we hit tomorrow
        if date_part != today_date:
            break
            
        try:
            h = int(time_part.split(':')[0])
            
            if h > current_local_hour:
                 if temp_c > future_max_c:
                     future_max_c = temp_c
                 found_future = True
        except:
             continue

    if not found_future:
        # No future hours left in today (e.g. it's 23:00)
        return True, "LOW", current_temp_f

    future_max_f = celsius_to_fahrenheit(future_max_c)
    
    # Threshold: If future max is notably higher than current
    if future_max_f > (current_temp_f + 1.0):
        return True, "HIGH", future_max_f
    else:
        return True, "LOW", future_max_f
