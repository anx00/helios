"""
Helios Weather Lab - NBM Fetcher (National Blend of Models)
Fetches temperature forecasts from NOAA NBM text bulletins.
v10.0: Multi-model consensus integration.
"""

import httpx
import asyncio
import re
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, timezone, date, timedelta

# NOAA MDL NBM Text Products (more reliable than AWC)
# These are the raw text bulletins with station forecasts
NBM_TEXT_URL = "https://digital.mdl.nws.noaa.gov/newbulletin/nbm/nbhtx"

# Open-Meteo as fallback (provides NBM-like ensemble data)
OPEN_METEO_ENSEMBLE_URL = "https://api.open-meteo.com/v1/forecast"

# NOAA NOMADS for bulk data (backup)
NBM_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"

# Station coordinates for Open-Meteo fallback
STATION_META = {
    "KLGA": (40.7769, -73.8740, "America/New_York"),
    "KATL": (33.6407, -84.4277, "America/New_York"),
    "KORD": (41.9602, -87.9316, "America/Chicago"),
    "KMIA": (25.7881, -80.3169, "America/New_York"),
    "KDAL": (32.8384, -96.8358, "America/Chicago"),
    "EGLC": (51.5048, 0.0495, "Europe/London"),
    "LTAC": (40.1281, 32.9951, "Europe/Istanbul"),
}


@dataclass
class NBMForecast:
    """NBM forecast data for a station."""
    station_id: str
    max_temp_f: float
    min_temp_f: Optional[float] = None
    fetch_time: datetime = None
    source: str = "open_meteo"
    raw_data: Optional[str] = None
    
    def __post_init__(self):
        if self.fetch_time is None:
            self.fetch_time = datetime.now(timezone.utc)


async def fetch_nbm_from_open_meteo(station_id: str, target_date: date = None) -> Optional[NBMForecast]:
    """
    Fetch NBM-equivalent ensemble forecast from Open-Meteo.
    
    Open-Meteo provides multi-model ensemble data that's similar to NBM,
    including 30+ weather models blended together.
    
    Args:
        station_id: ICAO station ID (e.g., "KLGA")
        target_date: Target forecast date (default: today)
        
    Returns:
        NBMForecast object or None on error
    """
    if station_id not in STATION_META:
        return None
    
    lat, lon, tz_name = STATION_META[station_id]
    
    if target_date is None:
        target_date = date.today()
    
    # Calculate days ahead
    days_ahead = (target_date - date.today()).days
    if days_ahead < 0:
        days_ahead = 0
    
    # Use Open-Meteo ensemble model (similar to NBM approach)
    # The "best_match" model uses ensemble averaging
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone": tz_name,
        "forecast_days": max(2, days_ahead + 1),
        "models": "best_match"  # Uses ensemble/blend approach
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(OPEN_METEO_ENSEMBLE_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            daily = data.get("daily", {})
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            
            if max_temps and len(max_temps) > days_ahead:
                max_temp_f = max_temps[days_ahead]
                min_temp_f = min_temps[days_ahead] if min_temps and len(min_temps) > days_ahead else None
                
                return NBMForecast(
                    station_id=station_id,
                    max_temp_f=round(max_temp_f, 1),
                    min_temp_f=round(min_temp_f, 1) if min_temp_f else None,
                    source="open_meteo_ensemble",
                    raw_data=str(data.get("daily", {}))[:200]
                )
                
    except httpx.HTTPError as e:
        print(f"[NBM] HTTP error for {station_id}: {e}")
    except Exception as e:
        print(f"[NBM] Error fetching {station_id}: {e}")
    
    return None


async def fetch_nbm_from_nomads(station_id: str, target_date: date = None) -> Optional[NBMForecast]:
    """
    Fetch NBM from NOAA NOMADS text bulletins.
    
    Fallback method if AWC is unavailable.
    
    Args:
        station_id: ICAO station ID
        target_date: Target forecast date
        
    Returns:
        NBMForecast object or None on error
    """
    if target_date is None:
        target_date = date.today()
    
    date_str = target_date.strftime("%Y%m%d")
    
    # Try different cycles (00z, 06z, 12z, 18z)
    cycles = ["12", "06", "00", "18"]
    
    for cycle in cycles:
        url = f"{NBM_BASE_URL}/blend.{date_str}/{cycle}/text/blend_nbstx.t{cycle}z"
        
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    text = response.text
                    
                    # Parse station-specific data
                    max_temp_f = parse_nomads_bulletin(text, station_id)
                    
                    if max_temp_f is not None:
                        return NBMForecast(
                            station_id=station_id,
                            max_temp_f=max_temp_f,
                            source=f"nomads_{cycle}z",
                            raw_data=text[:500]
                        )
        except Exception as e:
            continue  # Try next cycle
    
    return None


def parse_nbm_max_temp(text: str, target_date: date = None) -> Optional[float]:
    """
    Parse NBM decoded output for maximum temperature.
    
    NBM hourly format typically includes TMP (temperature) rows.
    We look for the maximum value in the forecast period.
    
    Args:
        text: Raw NBM bulletin text
        target_date: Target date for max temp
        
    Returns:
        Maximum temperature in Fahrenheit or None
    """
    if not text:
        return None
    
    try:
        lines = text.strip().split('\n')
        
        # Look for temperature line (TMP or T)
        temps = []
        
        for line in lines:
            # Pattern 1: "TMP" followed by values
            if line.strip().startswith('TMP') or line.strip().startswith('T  '):
                # Extract numeric values
                values = re.findall(r'(\d+)', line)
                temps.extend([int(v) for v in values if 10 < int(v) < 120])
            
            # Pattern 2: Look for "MaxT" or "MxT" row
            if 'MaxT' in line or 'MxT' in line or 'MAX' in line:
                values = re.findall(r'(\d+)', line)
                if values:
                    # Take first reasonable value as max temp
                    for v in values:
                        temp = int(v)
                        if 10 < temp < 120:
                            return float(temp)
        
        # If we found temperature values, return the max
        if temps:
            return float(max(temps))
            
    except Exception as e:
        print(f"[NBM] Parse error: {e}")
    
    return None


def parse_nomads_bulletin(text: str, station_id: str) -> Optional[float]:
    """
    Parse NOMADS NBM text bulletin for station max temperature.
    
    Args:
        text: Full bulletin text
        station_id: ICAO station ID to find
        
    Returns:
        Maximum temperature in Fahrenheit or None
    """
    if not text or station_id not in text:
        return None
    
    try:
        # Find station block
        lines = text.split('\n')
        station_found = False
        
        for i, line in enumerate(lines):
            if station_id in line:
                station_found = True
                
            if station_found:
                # Look for temperature data in following lines
                if 'X/' in line or 'MX' in line:
                    values = re.findall(r'(\d+)', line)
                    for v in values:
                        temp = int(v)
                        if 20 < temp < 110:
                            return float(temp)
                
                # Stop after ~10 lines past station
                if i > 10 and not any(c.isdigit() for c in line):
                    break
                    
    except Exception as e:
        print(f"[NBM] NOMADS parse error: {e}")
    
    return None


async def fetch_nbm(station_id: str, target_date: date = None) -> Optional[NBMForecast]:
    """
    Main entry point: Fetch NBM forecast with fallback.
    
    Uses Open-Meteo ensemble (similar to NBM) as primary source.
    
    Args:
        station_id: ICAO station ID
        target_date: Target forecast date (default: today)
        
    Returns:
        NBMForecast object or None
    """
    if target_date is None:
        target_date = date.today()
    
    # Try Open-Meteo ensemble first (most reliable)
    result = await fetch_nbm_from_open_meteo(station_id, target_date)
    
    if result is None:
        # Fallback to NOMADS bulletins
        result = await fetch_nbm_from_nomads(station_id, target_date)
    
    if result:
        print(f"[NBM] {station_id}: Max={result.max_temp_f:.1f}°F (Source: {result.source})")
    else:
        print(f"[NBM] {station_id}: No data available")
    
    return result


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("Testing NBM Fetcher...")
        print("=" * 50)
        
        for station in ["KLGA", "KATL", "KORD", "KMIA", "KDAL", "EGLC", "LTAC"]:
            result = await fetch_nbm(station)
            if result:
                print(f"  {station}: {result.max_temp_f}°F from {result.source}")
            else:
                print(f"  {station}: No data")
        
        print("=" * 50)
        print("NBM Fetcher test complete.")
    
    asyncio.run(test())
