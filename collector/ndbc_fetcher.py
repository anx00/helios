"""
NOAA NDBC Buoy Fetcher - Sea Surface Temperature

Fetches real-time SST from NOAA National Data Buoy Center
to correct sea breeze temperature calculations.

Impact: ±2°F for coastal stations
"""
import httpx
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import asyncio


# Buoy station mappings for our weather stations
BUOY_STATIONS = {
    "KLGA": "44065",  # NY Harbor Entrance, 15 NM SE Breezy Point
    "KATL": "41008",  # Grays Reef, 40 NM SE Savannah (closest to Atlanta coast)
}

# NDBC Real-time data endpoint
NDBC_BASE_URL = "https://www.ndbc.noaa.gov/data/realtime2"


@dataclass
class BuoySST:
    """Sea Surface Temperature from buoy."""
    buoy_id: str
    station_id: str  # Associated weather station
    water_temp_c: float
    water_temp_f: float
    observation_time: datetime
    wave_height_m: Optional[float]
    wind_dir: Optional[int]
    wind_speed_kt: Optional[float]
    
    @property
    def is_cold(self) -> bool:
        """Check if water is notably cold (< 10°C / 50°F)."""
        return self.water_temp_c < 10.0
    
    def calculate_sst_adjustment(
        self,
        expected_sst_c: float,
        wind_is_onshore: bool
    ) -> tuple[float, str]:
        """
        Calculate temperature adjustment based on SST delta.
        
        Args:
            expected_sst_c: What the model expected
            wind_is_onshore: Is wind blowing from ocean to land?
            
        Returns:
            (adjustment_f, reason_string)
        """
        if not wind_is_onshore:
            return 0.0, ""
        
        delta_c = expected_sst_c - self.water_temp_c  # Positive = water colder than expected
        
        if delta_c > 2.0:
            # Water is significantly colder than model expects
            adjustment_f = -delta_c * 0.5  # ~0.5°F per 1°C delta
            reason = f"ColdSST{adjustment_f:.1f}F"
            return round(adjustment_f, 1), reason
        elif delta_c < -2.0:
            # Water is warmer than expected
            adjustment_f = -delta_c * 0.3  # Smaller effect for warm water
            reason = f"WarmSST+{abs(adjustment_f):.1f}F"
            return round(adjustment_f, 1), reason
        
        return 0.0, ""


def parse_ndbc_realtime(text: str, buoy_id: str, station_id: str) -> Optional[BuoySST]:
    """
    Parse NDBC realtime2 text format.
    
    Format:
    #YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
    #yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft
    2024 01 06 22 00 200  5.0  6.0   0.9   6.7   5.2 180 1020.0  10.5   8.2   7.1   MM   MM    MM
    """
    try:
        lines = text.strip().split('\n')
        
        # Find the first data line (skip headers starting with #)
        data_line = None
        for line in lines:
            if not line.startswith('#') and line.strip():
                data_line = line
                break
        
        if not data_line:
            return None
        
        parts = data_line.split()
        
        if len(parts) < 15:
            return None
        
        # Parse fields
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        hour = int(parts[3])
        minute = int(parts[4])
        
        obs_time = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        
        # Wind direction (index 5)
        wind_dir = int(parts[5]) if parts[5] != 'MM' else None
        
        # Wind speed in m/s (index 6)
        wind_speed_ms = float(parts[6]) if parts[6] != 'MM' else None
        wind_speed_kt = wind_speed_ms * 1.944 if wind_speed_ms else None
        
        # Wave height (index 8)
        wave_height = float(parts[8]) if parts[8] != 'MM' else None
        
        # Water temperature (index 14) - WTMP
        wtmp_str = parts[14] if len(parts) > 14 else 'MM'
        water_temp_c = float(wtmp_str) if wtmp_str != 'MM' else None
        
        if water_temp_c is None:
            return None
        
        water_temp_f = (water_temp_c * 9/5) + 32
        
        return BuoySST(
            buoy_id=buoy_id,
            station_id=station_id,
            water_temp_c=water_temp_c,
            water_temp_f=round(water_temp_f, 1),
            observation_time=obs_time,
            wave_height_m=wave_height,
            wind_dir=wind_dir,
            wind_speed_kt=round(wind_speed_kt, 1) if wind_speed_kt else None
        )
        
    except Exception as e:
        print(f"[NDBC] Parse error: {e}")
        return None


async def fetch_buoy_sst(station_id: str) -> Optional[BuoySST]:
    """
    Fetch real-time SST from NDBC buoy.
    
    Args:
        station_id: Weather station ID (KLGA, KATL, etc.)
        
    Returns:
        BuoySST data or None if unavailable
    """
    buoy_id = BUOY_STATIONS.get(station_id)
    
    if not buoy_id:
        # No buoy mapped for this station
        return None
    
    url = f"{NDBC_BASE_URL}/{buoy_id}.txt"
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            return parse_ndbc_realtime(response.text, buoy_id, station_id)
            
    except httpx.HTTPStatusError as e:
        print(f"[NDBC] HTTP error for {buoy_id}: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"[NDBC] Error fetching {buoy_id}: {e}")
        return None


def is_wind_onshore(station_id: str, wind_dir: Optional[int]) -> bool:
    """
    Check if wind is blowing from ocean to land.
    
    Args:
        station_id: Weather station ID
        wind_dir: Wind direction in degrees
        
    Returns:
        True if wind is onshore
    """
    if wind_dir is None:
        return False
    
    # Define onshore wind directions for each station
    onshore_ranges = {
        "KLGA": [(0, 90), (270, 360)],  # N to E, and W to N (from ocean)
        "KATL": [],  # Atlanta is inland, no sea breeze
    }
    
    ranges = onshore_ranges.get(station_id, [])
    
    for low, high in ranges:
        if low <= wind_dir <= high:
            return True
    
    return False


if __name__ == "__main__":
    # Test
    async def test():
        sst = await fetch_buoy_sst("KLGA")
        if sst:
            print(f"Buoy: {sst.buoy_id}")
            print(f"Water Temp: {sst.water_temp_f}°F ({sst.water_temp_c}°C)")
            print(f"Obs Time: {sst.observation_time}")
            print(f"Wind: {sst.wind_dir}° @ {sst.wind_speed_kt}kt")
        else:
            print("No SST data available")
    
    asyncio.run(test())
