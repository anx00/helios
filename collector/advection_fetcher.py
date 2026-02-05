"""
HELIOS v3.0 - Advection Data Fetcher
Fetches temperature data from 'upstream' locations to detect incoming air masses.
Uses Open-Meteo Current Weather API to probe virtual stations upwind.
"""

import httpx
import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvectionFetcher")

@dataclass
class UpstreamData:
    temperature_c: float
    wind_speed_kmh: float
    wind_dir: int
    distance_km: float
    bearing_degrees: float
    is_real_station: bool = False

async def get_upstream_coords(
    lat: float, 
    lon: float, 
    wind_dir_degrees: float, 
    distance_km: float = 50.0
) -> Tuple[float, float]:
    """
    Calculate coordinates of a point 'distance_km' away, UPWIND from the wind direction.
    Upwind = (Wind Direction + 180) % 360.
    """
    # Calculate bearing (direction TO the upstream point)
    # If wind comes FROM 270 (West), upstream is AT 270 (West).
    # Checking nomenclature: Wind Direction 270 means FROM West.
    # So we want to check the point to the West (270).
    bearing = wind_dir_degrees
    
    R = 6371.0 # Earth radius in km
    
    # Convert to radians
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing)
    
    # Formula for destination point
    lat2 = math.asin(math.sin(lat1) * math.cos(distance_km / R) +
                     math.cos(lat1) * math.sin(distance_km / R) * math.cos(brng))
                     
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(distance_km / R) * math.cos(lat1),
                             math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2))
                             
    return (math.degrees(lat2), math.degrees(lon2))

async def fetch_upstream_weather(
    base_lat: float, 
    base_lon: float, 
    wind_dir_degrees: int,
    distance_km: float = 60.0
) -> Optional[UpstreamData]:
    """
    Fetch weather from a virtual upstream station using Open-Meteo.
    """
    try:
        # 1. Calculate target coordinates
        target_lat, target_lon = await get_upstream_coords(base_lat, base_lon, wind_dir_degrees, distance_km)
        
        # 2. Query Open-Meteo
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": target_lat,
            "longitude": target_lon,
            "current": "temperature_2m,wind_speed_10m,wind_direction_10m",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh"
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            
            current = data.get("current", {})
            
            return UpstreamData(
                temperature_c=current.get("temperature_2m"),
                wind_speed_kmh=current.get("wind_speed_10m"),
                wind_dir=current.get("wind_direction_10m"),
                distance_km=distance_km,
                bearing_degrees=wind_dir_degrees,
                is_real_station=False
            )
            
    except Exception as e:
        logger.error(f"Failed to fetch upstream data: {e}")
        return None
