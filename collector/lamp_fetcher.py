"""
Helios Weather Lab - LAMP Fetcher (Localized Aviation MOS Program)
Fetches hourly temperature forecasts from NOAA LAMP MOS.
v10.0: Multi-model consensus integration.
"""

import httpx
import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime, timezone, timedelta

# Aviation Weather Center API v4 (correct endpoint)
AWC_API_URL = "https://aviationweather.gov/api/data/dataserver"

# Direct LAMP text from NWS
NWS_LAMP_URL = "https://forecast.weather.gov/product.php"

# Fallback: Open-Meteo hourly for temperature trajectory
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Station coordinates for fallback
STATION_COORDS = {
    "KLGA": (40.7769, -73.8740, "America/New_York"),
    "KATL": (33.6407, -84.4277, "America/New_York"),
    "EGLC": (51.5048, 0.0495, "Europe/London"),
}


@dataclass
class LAMPForecast:
    """LAMP forecast data for a station."""
    station_id: str
    max_temp_f: float
    hourly_temps_f: List[float] = field(default_factory=list)
    fetch_time: datetime = None
    valid_hours: int = 25  # LAMP provides 25 hours ahead
    source: str = "open_meteo_hourly"
    confidence: float = 0.5  # Will be updated based on 7-day rolling accuracy
    raw_data: Optional[str] = None
    
    def __post_init__(self):
        if self.fetch_time is None:
            self.fetch_time = datetime.now(timezone.utc)


async def fetch_lamp_from_open_meteo(station_id: str) -> Optional[LAMPForecast]:
    """
    Fetch LAMP-equivalent hourly forecast from Open-Meteo.
    
    Since AWC LAMP endpoint has access issues, we use Open-Meteo hourly
    forecast which provides similar granular temperature data.
    
    Args:
        station_id: ICAO station ID (e.g., "KLGA")
        
    Returns:
        LAMPForecast object or None on error
    """
    if station_id not in STATION_COORDS:
        return None
    
    lat, lon, tz = STATION_COORDS[station_id]
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": tz,
        "forecast_hours": 25
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            hourly = data.get("hourly", {})
            temps = hourly.get("temperature_2m", [])
            
            if temps:
                # Filter to today's daytime hours for max temp
                hourly_temps_f = [round(t, 1) for t in temps[:25] if t is not None]
                max_temp_f = max(hourly_temps_f) if hourly_temps_f else None
                
                if max_temp_f is not None:
                    return LAMPForecast(
                        station_id=station_id,
                        max_temp_f=round(max_temp_f, 1),
                        hourly_temps_f=hourly_temps_f,
                        source="open_meteo_hourly",
                        raw_data=str(temps[:6])
                    )
                
    except httpx.HTTPError as e:
        print(f"[LAMP] HTTP error for {station_id}: {e}")
    except Exception as e:
        print(f"[LAMP] Error fetching {station_id}: {e}")
    
    return None


async def fetch_lamp_from_tgftp(station_id: str) -> Optional[LAMPForecast]:
    """
    Fetch LAMP from NOAA TGFTP raw bulletins.
    
    Fallback method if AWC is unavailable.
    
    Args:
        station_id: ICAO station ID
        
    Returns:
        LAMPForecast object or None on error
    """
    if station_id not in LAMP_STATION_IDS or LAMP_STATION_IDS[station_id] is None:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(TGFTP_LAMP_URL)
            
            if response.status_code == 200:
                text = response.text
                
                # Extract station-specific section
                temps_f = parse_tgftp_lamp(text, station_id)
                
                if temps_f:
                    max_temp_f = max(temps_f)
                    return LAMPForecast(
                        station_id=station_id,
                        max_temp_f=max_temp_f,
                        hourly_temps_f=temps_f,
                        source="tgftp_raw",
                        raw_data=text[:500]
                    )
    except Exception as e:
        print(f"[LAMP] TGFTP error: {e}")
    
    return None


def parse_lamp_temperatures(text: str) -> List[float]:
    """
    Parse LAMP decoded output for hourly temperatures.
    
    LAMP format includes TMP row with hourly values.
    Values are in Fahrenheit for US stations.
    
    Args:
        text: Raw LAMP bulletin text
        
    Returns:
        List of hourly temperatures in Fahrenheit
    """
    temps = []
    
    if not text:
        return temps
    
    try:
        lines = text.strip().split('\n')
        
        for line in lines:
            line_upper = line.upper().strip()
            
            # Pattern 1: TMP row (most common)
            if line_upper.startswith('TMP') or line_upper.startswith('T  '):
                # Extract all numeric values
                values = re.findall(r'(-?\d+)', line)
                for v in values:
                    temp = int(v)
                    # Reasonable temperature range (°F)
                    if -20 < temp < 120:
                        temps.append(float(temp))
            
            # Pattern 2: Temperature line with F or empty (degrees F assumed)
            if 'TEMP' in line_upper or 'TMP' in line_upper:
                values = re.findall(r'(-?\d+)', line)
                for v in values:
                    temp = int(v)
                    if -20 < temp < 120 and temp not in temps:
                        temps.append(float(temp))
        
        # If we still have no data, try broader extraction
        if not temps:
            # Look for any line with multiple temperature-like values
            for line in lines:
                # Skip header lines
                if any(x in line.upper() for x in ['LAMP', 'UTC', 'VALID']):
                    continue
                    
                values = re.findall(r'\b(\d{2})\b', line)
                for v in values:
                    temp = int(v)
                    if 20 < temp < 100:
                        temps.append(float(temp))
                        
    except Exception as e:
        print(f"[LAMP] Parse error: {e}")
    
    return temps


def parse_tgftp_lamp(text: str, station_id: str) -> List[float]:
    """
    Parse TGFTP raw LAMP bulletin for station temperatures.
    
    Args:
        text: Full bulletin text
        station_id: ICAO station ID to find
        
    Returns:
        List of hourly temperatures in Fahrenheit
    """
    temps = []
    
    if not text or station_id not in text:
        return temps
    
    try:
        lines = text.split('\n')
        station_found = False
        temp_line_found = False
        
        for line in lines:
            if station_id in line:
                station_found = True
                continue
                
            if station_found:
                # Look for temperature line
                if line.strip().startswith('TMP') or line.strip().startswith('T'):
                    temp_line_found = True
                    values = re.findall(r'(-?\d+)', line)
                    for v in values:
                        temp = int(v)
                        if -20 < temp < 120:
                            temps.append(float(temp))
                
                # Stop at next station
                if temp_line_found and (not line.strip() or re.match(r'^[A-Z]{4}\s', line)):
                    break
                    
    except Exception as e:
        print(f"[LAMP] TGFTP parse error: {e}")
    
    return temps


def calculate_lamp_confidence(station_id: str) -> float:
    """
    Calculate LAMP confidence based on 7-day rolling accuracy.
    
    This connects to the database to check historical LAMP vs actual.
    
    Args:
        station_id: Station ID
        
    Returns:
        Confidence score (0.0 to 1.0)
    """
    try:
        from database import get_connection
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Get last 7 days of LAMP predictions vs actuals
            cursor.execute("""
                SELECT lamp_max_f, wu_final
                FROM performance_logs
                WHERE station_id = ?
                  AND lamp_max_f IS NOT NULL
                  AND wu_final IS NOT NULL
                  AND timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC
                LIMIT 50
            """, (station_id,))
            
            rows = cursor.fetchall()
            
            if len(rows) < 5:
                return 0.5  # Not enough data, use default
            
            # Calculate mean absolute error
            errors = [abs(row[0] - row[1]) for row in rows]
            mae = sum(errors) / len(errors)
            
            # Convert MAE to confidence (lower MAE = higher confidence)
            # MAE of 0°F = confidence 1.0
            # MAE of 3°F = confidence 0.5
            # MAE of 6°F+ = confidence 0.0
            confidence = max(0.0, 1.0 - (mae / 6.0))
            
            return round(confidence, 2)
            
    except Exception as e:
        # If database not available (e.g., first run), return default
        return 0.5


async def fetch_lamp(station_id: str) -> Optional[LAMPForecast]:
    """
    Main entry point: Fetch LAMP forecast with fallback.
    
    Uses Open-Meteo hourly as primary (NOAA endpoints have access issues).
    Also calculates 7-day rolling confidence.
    
    Args:
        station_id: ICAO station ID
        
    Returns:
        LAMPForecast object or None
    """
    # Check if station is supported
    if station_id not in STATION_COORDS:
        return None
    
    # Use Open-Meteo hourly as primary source
    result = await fetch_lamp_from_open_meteo(station_id)
    
    if result:
        # Calculate 7-day rolling confidence
        result.confidence = calculate_lamp_confidence(station_id)
        
        print(f"[LAMP] {station_id}: Max={result.max_temp_f:.1f}°F, "
              f"Hours={len(result.hourly_temps_f)}, "
              f"Confidence={result.confidence:.0%} (Source: {result.source})")
    else:
        print(f"[LAMP] {station_id}: No data available")
    
    return result


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("Testing LAMP Fetcher...")
        print("=" * 50)
        
        for station in ["KLGA", "KATL", "EGLC"]:
            result = await fetch_lamp(station)
            if result:
                print(f"  {station}: Max={result.max_temp_f}°F, "
                      f"Confidence={result.confidence:.0%}")
                if result.hourly_temps_f:
                    print(f"    Hourly temps: {result.hourly_temps_f[:6]}...")
            else:
                print(f"  {station}: No data (expected for non-US)")
        
        print("=" * 50)
        print("LAMP Fetcher test complete.")
    
    asyncio.run(test())
