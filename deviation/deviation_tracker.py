"""
Helios Weather Lab - Deviation Tracker (Module E)
Captures model trajectory and calculates real-time deviations.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List
from zoneinfo import ZoneInfo

from database import store_model_path, get_predicted_temp_for_hour
from config import STATIONS


@dataclass
class DeviationData:
    """Real-time deviation data for AI analysis."""
    station_id: str
    current_hour: int
    predicted_temp_c: float
    actual_temp_c: float
    delta_c: float                   # Actual - Predicted (positive = warmer than expected)
    delta_f: float
    predicted_sky: Optional[str]     # What model expected (if available)
    actual_sky: str                  # What METAR reports
    deviation_trend: str             # "warming", "cooling", "on_track"
    has_trajectory: bool             # Whether we have a stored trajectory for today


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return round((celsius * 9/5) + 32, 1)


async def capture_morning_trajectory(station_id: str, forecast_data) -> int:
    """
    Capture the HRRR hourly trajectory for the day.
    Should be called at 07:00 AM local time.
    
    Args:
        station_id: ICAO station identifier
        forecast_data: MultiModelData from the HRRR fetcher
        
    Returns:
        Number of hourly records stored
    """
    if not forecast_data or not forecast_data.hrrr:
        print(f"  ⚠ No HRRR data to capture trajectory for {station_id}")
        return 0
    
    # Get today's date in local station time
    station = STATIONS.get(station_id)
    tz_name = station.timezone if station else "UTC"
    local_now = datetime.now(timezone.utc).astimezone(ZoneInfo(tz_name))
    forecast_date = local_now.strftime("%Y-%m-%d")
    
    # Get hourly data directly from MultiModelData
    hourly_temps = forecast_data.hourly_temps_c or []
    hourly_times = forecast_data.hourly_times or []
    
    if hourly_temps and hourly_times:
        print(f"  [INFO] Found {len(hourly_temps)} hourly temps, {len(hourly_times)} times")
        inserted = store_model_path(
            station_id=station_id,
            forecast_date=forecast_date,
            hourly_temps=hourly_temps,
            hourly_times=hourly_times,
        )
        print(f"  [OK] Captured trajectory: {inserted} hours stored for {station_id}")
        return inserted
    else:
        print(f"  [WARN] No hourly data in forecast (temps={len(hourly_temps)}, times={len(hourly_times)})")
        return 0


def calculate_deviation(
    station_id: str,
    actual_temp_c: float,
    actual_sky: str,
    observation_time: datetime,
) -> Optional[DeviationData]:
    """
    Calculate the deviation between model prediction and actual observation.
    
    Args:
        station_id: ICAO station identifier
        actual_temp_c: Current METAR temperature
        actual_sky: Current METAR sky condition
        observation_time: Time of observation (UTC)
        
    Returns:
        DeviationData object with deviation analysis, or None if no trajectory
    """
    # Get local time for the station
    station = STATIONS.get(station_id)
    tz_name = station.timezone if station else "UTC"
    local_time = observation_time.astimezone(ZoneInfo(tz_name))
    
    forecast_date = local_time.strftime("%Y-%m-%d")
    current_hour = local_time.hour
    
    # Get the predicted temperature for this hour
    prediction = get_predicted_temp_for_hour(station_id, forecast_date, current_hour)
    
    if not prediction:
        # No trajectory stored yet (before 07:00 AM or first run)
        return DeviationData(
            station_id=station_id,
            current_hour=current_hour,
            predicted_temp_c=0.0,
            actual_temp_c=actual_temp_c,
            delta_c=0.0,
            delta_f=0.0,
            predicted_sky=None,
            actual_sky=actual_sky,
            deviation_trend="unknown",
            has_trajectory=False,
        )
    
    predicted_temp_c = prediction['predicted_temp_c']
    delta_c = actual_temp_c - predicted_temp_c
    delta_f = celsius_to_fahrenheit(actual_temp_c) - celsius_to_fahrenheit(predicted_temp_c)
    
    # Determine deviation trend
    if abs(delta_c) < 1.0:
        trend = "on_track"
    elif delta_c > 0:
        trend = "warming"  # Warmer than model expected
    else:
        trend = "cooling"  # Cooler than model expected
    
    return DeviationData(
        station_id=station_id,
        current_hour=current_hour,
        predicted_temp_c=predicted_temp_c,
        actual_temp_c=actual_temp_c,
        delta_c=round(delta_c, 1),
        delta_f=round(delta_f, 1),
        predicted_sky=None,  # We don't store sky predictions yet
        actual_sky=actual_sky,
        deviation_trend=trend,
        has_trajectory=True,
    )


def format_deviation_for_prompt(deviation: DeviationData) -> str:
    """
    Format deviation data as a string for the Gemini prompt.
    
    Args:
        deviation: DeviationData object
        
    Returns:
        Formatted string for AI prompt
    """
    if not deviation.has_trajectory:
        return "DESVIACIÓN DEL MODELO: Sin trayectoria capturada aún (ejecutar captura a las 07:00 AM)."
    
    # Create deviation description
    if deviation.deviation_trend == "on_track":
        trend_desc = "El modelo va bien encaminado."
        emoji = "✓"
    elif deviation.deviation_trend == "warming":
        trend_desc = f"La realidad está +{deviation.delta_c:.1f}°C MÁS CALIENTE de lo previsto."
        emoji = "🔥"
    else:
        trend_desc = f"La realidad está {deviation.delta_c:.1f}°C MÁS FRÍA de lo previsto."
        emoji = "❄️"
    
    return f"""DESVIACIÓN DEL MODELO (Hora {deviation.current_hour}:00):
- {emoji} {trend_desc}
- Modelo preveía: {deviation.predicted_temp_c}°C
- METAR informa: {deviation.actual_temp_c}°C
- Delta: {deviation.delta_c:+.1f}°C ({deviation.delta_f:+.1f}°F)
- Cielo actual: {deviation.actual_sky}

PREGUNTA CRÍTICA: ¿Es este desvío un "ruido momentáneo" o una "tendencia del día" que afectará la máxima final?"""
