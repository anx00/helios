"""
Helios Weather Lab - HRRR/GFS Fetcher (Protocol FORTRESS)
Fetches multi-model forecasts from Open-Meteo API.
"""

import httpx
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from config import OPEN_METEO_URL

logger = logging.getLogger("hrrr_fetcher")


@dataclass
class ModelForecast:
    """Forecast data from a single model."""
    model_name: str
    max_forecast_c: float
    max_forecast_f: float
    current_temp_c: Optional[float]
    soil_moisture: Optional[float]  # m³/m³ (0-10cm)
    shortwave_radiation: Optional[float]  # W/m²
    cloud_cover: Optional[float]  # %


@dataclass
class MultiModelData:
    """Combined forecast data from multiple models."""
    latitude: float
    longitude: float
    fetch_time: datetime
    hrrr: Optional[ModelForecast]
    gfs: Optional[ModelForecast]
    # Consensus values (average when both available)
    consensus_max_c: float
    consensus_max_f: float
    model_divergence_c: float  # Absolute difference between models
    # Hourly data for trajectory capture
    hourly_temps_c: Optional[List[float]] = None
    hourly_times: Optional[List[str]] = None


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return round((celsius * 9/5) + 32, 1)


async def fetch_model_forecast(
    latitude: float,
    longitude: float,
    model: str,
    target_days_ahead: int = 0
) -> Optional[ModelForecast]:
    """
    Fetch forecast from a specific model.
    
    Args:
        latitude: Station latitude
        longitude: Station longitude
        model: Model name ('hrrr_conus', 'gfs_seamless', or None for default)
        target_days_ahead: Days ahead to forecast (0=today, 1=tomorrow, etc.)
        
    Returns:
        ModelForecast object or None on error
    """
    forecast_days = target_days_ahead + 1  # API needs total days, not offset
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,soil_moisture_0_to_10cm,shortwave_radiation,cloud_cover",
        "forecast_days": forecast_days,
        "timezone": "auto",
    }
    
    # Add model parameter if specified
    if model:
        params["models"] = model
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(OPEN_METEO_URL, params=params)
            
            # If model-specific request fails, try without model param
            if response.status_code == 400 and model:
                del params["models"]
                response = await client.get(OPEN_METEO_URL, params=params)
            
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get("hourly", {})
            temps = hourly.get("temperature_2m", [])
            soil = hourly.get("soil_moisture_0_to_10cm", [])
            radiation = hourly.get("shortwave_radiation", [])
            clouds = hourly.get("cloud_cover", [])
            times = hourly.get("time", [])
            
            if not temps or not times:
                return None
            
            # Filter day temps
            # CRITICAL FIX: For the daily Maximum, we must consider the FULL 24-hour cycle,
            # even if those hours have already passed. The "High" is a daily statistic.
            tz_name = data.get("timezone") or "UTC"
            try:
                local_tz = ZoneInfo(tz_name)
            except Exception:
                local_tz = timezone.utc
            now_local = datetime.now(local_tz)
            target_date_local = now_local.date() + timedelta(days=target_days_ahead)

            # Build records with local datetime
            records = []
            for idx, ts in enumerate(times):
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=local_tz)
                    else:
                        dt = dt.astimezone(local_tz)
                except Exception:
                    continue
                temp = temps[idx] if idx < len(temps) else None
                soil_v = soil[idx] if idx < len(soil) else None
                rad_v = radiation[idx] if idx < len(radiation) else None
                cloud_v = clouds[idx] if idx < len(clouds) else None
                records.append((dt, temp, soil_v, rad_v, cloud_v))

            day_records = [r for r in records if r[0].date() == target_date_local and r[1] is not None]
            full_day_temps = [r[1] for r in day_records]

            if not full_day_temps:
                return None

            # Calculate max from the full 24h cycle
            max_temp_c = max(full_day_temps)
            
            # Get current hour values (for initial conditions)
            current_temp = None
            current_soil = None
            current_radiation = None
            current_clouds = None
            if day_records:
                if target_days_ahead == 0:
                    # Match current local hour
                    hour = now_local.hour
                    match = next((r for r in day_records if r[0].hour == hour), None)
                    if match:
                        _, current_temp, current_soil, current_radiation, current_clouds = match
                    else:
                        _, current_temp, current_soil, current_radiation, current_clouds = day_records[0]
                else:
                    _, current_temp, current_soil, current_radiation, current_clouds = day_records[0]
            
            model_name = model.upper().replace("_CONUS", "").replace("_SEAMLESS", "") if model else "DEFAULT"
            
            return ModelForecast(
                model_name=model_name,
                max_forecast_c=max_temp_c,
                max_forecast_f=celsius_to_fahrenheit(max_temp_c),
                current_temp_c=current_temp,
                soil_moisture=current_soil,
                shortwave_radiation=current_radiation,
                cloud_cover=current_clouds,
            )
            
    except Exception as e:
        # Keep logs ASCII-safe; Windows consoles may not support all Unicode symbols.
        logger.warning("[%s] model error: %s", model or "default", e)
        return None


async def fetch_hrrr(latitude: float, longitude: float, target_days_ahead: int = 0) -> Optional[MultiModelData]:
    """
    Fetch multi-model forecast data from Open-Meteo.
    Protocol FORTRESS: Dual-model consensus approach.
    
    - For US coords: Uses HRRR + GFS
    - For UK coords: Uses UKV (ukmo_seamless) + ECMWF
    
    Args:
        latitude: Station latitude
        longitude: Station longitude
        target_days_ahead: Days ahead to forecast (0=today, 1=tomorrow, etc.)
        
    Returns:
        MultiModelData object with both model forecasts, or None on error
    """
    now = datetime.now(timezone.utc)
    
    # Detect region and select appropriate models
    # UK region: 49°N < lat < 61°N and -10°W < lon < 2°E
    is_uk = 49 < latitude < 61 and -10 < longitude < 2
    
    if is_uk:
        primary_model = "ukmo_seamless"  # UK Met Office model
        secondary_model = "ecmwf_ifs04"  # ECMWF as fallback
    else:
        primary_model = None  # Default (includes HRRR for US)
        secondary_model = "gfs_seamless"
    
    # Fetch from both models in parallel
    try:
        primary_task = asyncio.create_task(fetch_model_forecast(latitude, longitude, primary_model, target_days_ahead))
        secondary_task = asyncio.create_task(fetch_model_forecast(latitude, longitude, secondary_model, target_days_ahead))
        
        hrrr_forecast, gfs_forecast = await asyncio.gather(primary_task, secondary_task)
    except Exception as e:
        logger.warning("Error fetching models for (%.4f, %.4f): %s", latitude, longitude, e)
        hrrr_forecast, gfs_forecast = None, None

    # Fetch hourly data for trajectory capture (and BaseForecast ingestion).
    hourly_temps = None
    hourly_times = None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m",
                "forecast_days": 2,  # Today + tomorrow
                "timezone": "auto",
            }
            response = await client.get(OPEN_METEO_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                hourly = data.get("hourly", {})
                hourly_temps = hourly.get("temperature_2m", [])
                hourly_times = hourly.get("time", [])
    except Exception as e:
        logger.warning("Failed to fetch hourly temperature curve for (%.4f, %.4f): %s", latitude, longitude, e)

    if not hrrr_forecast and not gfs_forecast and not hourly_temps:
        logger.error("No forecast data available for (%.4f, %.4f)", latitude, longitude)
        return None

    # Calculate consensus values.
    # If both model forecasts are missing but we do have an hourly curve, derive a max from it.
    if hrrr_forecast and gfs_forecast:
        consensus_max_c = (hrrr_forecast.max_forecast_c + gfs_forecast.max_forecast_c) / 2
        divergence = abs(hrrr_forecast.max_forecast_c - gfs_forecast.max_forecast_c)
    elif hrrr_forecast:
        consensus_max_c = hrrr_forecast.max_forecast_c
        divergence = 0.0
    elif gfs_forecast:
        consensus_max_c = gfs_forecast.max_forecast_c
        divergence = 0.0
    else:
        # Hourly curve is in Celsius from Open-Meteo. Use a conservative max over the first 24h window.
        try:
            temps = [t for t in (hourly_temps or []) if t is not None]
            consensus_max_c = max(temps[:24]) if temps else 0.0
        except Exception:
            consensus_max_c = 0.0
        divergence = 0.0
    
    return MultiModelData(
        latitude=latitude,
        longitude=longitude,
        fetch_time=now,
        hrrr=hrrr_forecast,
        gfs=gfs_forecast,
        consensus_max_c=consensus_max_c,
        consensus_max_f=celsius_to_fahrenheit(consensus_max_c),
        model_divergence_c=divergence,
        hourly_temps_c=hourly_temps,
        hourly_times=hourly_times,
    )


# Backward compatibility alias
HRRRData = MultiModelData
