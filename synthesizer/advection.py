"""
HELIOS v3.0 - Thermal Advection Engine
Calculates the rate of temperature change due to horizontal air transport.
Equation: dT/dt = -V * ∇T
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from collector.advection_fetcher import UpstreamData

@dataclass
class AdvectionReport:
    rate_c_per_hour: float
    upstream_temp_c: float
    wind_speed_kmh: float
    distance_km: float
    description: str

def calculate_advection(
    current_temp_c: float,
    upstream_data: UpstreamData,
    local_wind_kmh: float
) -> AdvectionReport:
    """
    Calculate thermal advection in degrees Celsius per hour.
    
    Advection = V * (T_upstream - T_current) / Distance
    
    Args:
        current_temp_c: Current station temperature
        upstream_data: Data from the upstream point
        local_wind_kmh: Wind speed at the station (used to average transport speed)
        
    Returns:
        AdvectionReport with rate and details.
    """
    
    # 1. Average Wind Speed (Transport Velocity)
    # Use average of local and upstream wind to represent the flow field
    avg_speed_kmh = (local_wind_kmh + upstream_data.wind_speed_kmh) / 2.0
    
    # Safety: If wind is calm (< 5 km/h), advection is negligible/unreliable
    if avg_speed_kmh < 5.0:
        return AdvectionReport(0.0, upstream_data.temperature_c, avg_speed_kmh, upstream_data.distance_km, "Calm winds (No advection)")
        
    # 2. Temperature Gradient
    temp_diff_c = upstream_data.temperature_c - current_temp_c
    
    # 3. Calculate Rate (deg C / hour)
    # Rate = Speed (km/h) * Gradient (deg/km)
    # Gradient = Diff / Distance
    
    if upstream_data.distance_km < 1.0:
        return AdvectionReport(0.0, upstream_data.temperature_c, avg_speed_kmh, upstream_data.distance_km, "Distance too small")
        
    gradient_c_km = temp_diff_c / upstream_data.distance_km
    advection_rate_c_h = avg_speed_kmh * gradient_c_km
    
    # 4. Generate Description
    trend = "Heating" if advection_rate_c_h > 0.1 else "Cooling" if advection_rate_c_h < -0.1 else "Neutral"
    desc = f"{trend}: {advection_rate_c_h:+.1f}C/h (Upstream {upstream_data.temperature_c:.1f}C at {upstream_data.distance_km}km)"
    
    return AdvectionReport(
        rate_c_per_hour=advection_rate_c_h,
        upstream_temp_c=upstream_data.temperature_c,
        wind_speed_kmh=avg_speed_kmh,
        distance_km=upstream_data.distance_km,
        description=desc
    )
