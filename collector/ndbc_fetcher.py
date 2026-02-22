"""
NOAA NDBC Buoy Fetcher - Sea Surface Temperature

Fetches real-time SST from NOAA National Data Buoy Center
to correct sea-breeze temperature calculations.

Impact: +/-2F for coastal stations.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx


# Buoy station mappings for supported weather stations.
# Lists are ordered by preference (first usable WTMP wins).
BUOY_STATIONS = {
    "KLGA": ["44065"],  # NY Harbor Entrance, 15 NM SE Breezy Point
    "KATL": ["41008"],  # Grays Reef, 40 NM SE Savannah (legacy; inland impact near zero)
    # KMIA: use fresh coastal stations first, keep offshore fallback.
    "KMIA": ["VAKF1", "41122"],
}

# NDBC Real-time data endpoint.
NDBC_BASE_URL = "https://www.ndbc.noaa.gov/data/realtime2"
NDBC_MAX_AGE_HOURS = 12.0


@dataclass
class BuoySST:
    """Sea surface temperature observation from NDBC."""

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
        """Check if water is notably cold (< 10C / 50F)."""
        return self.water_temp_c < 10.0

    def calculate_sst_adjustment(
        self,
        expected_sst_c: float,
        wind_is_onshore: bool,
    ) -> tuple[float, str]:
        """
        Calculate temperature adjustment based on SST delta.

        Args:
            expected_sst_c: What the model expected.
            wind_is_onshore: Is wind blowing from ocean to land?

        Returns:
            (adjustment_f, reason_string)
        """
        if not wind_is_onshore:
            return 0.0, ""

        # Positive means water is colder than expected.
        delta_c = expected_sst_c - self.water_temp_c
        if delta_c > 2.0:
            # Water is significantly colder than model expects.
            adjustment_f = -delta_c * 0.5  # ~0.5F per 1C delta
            return round(adjustment_f, 1), f"ColdSST{adjustment_f:.1f}F"
        if delta_c < -2.0:
            # Water is warmer than expected.
            adjustment_f = -delta_c * 0.3  # Smaller effect for warm water
            return round(adjustment_f, 1), f"WarmSST+{abs(adjustment_f):.1f}F"

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
        lines = text.strip().split("\n")
        now_utc = datetime.now(timezone.utc)
        # Prefer the newest line with valid WTMP (some stations publish WTMP=MM in latest row).
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            if parts[14] == "MM":
                continue

            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            obs_time = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            age_hours = (now_utc - obs_time).total_seconds() / 3600.0
            if age_hours < -0.5 or age_hours > NDBC_MAX_AGE_HOURS:
                continue

            wind_dir = int(parts[5]) if parts[5] != "MM" else None
            wind_speed_ms = float(parts[6]) if parts[6] != "MM" else None
            wind_speed_kt = wind_speed_ms * 1.944 if wind_speed_ms is not None else None
            wave_height = float(parts[8]) if parts[8] != "MM" else None

            water_temp_c = float(parts[14])
            water_temp_f = (water_temp_c * 9 / 5) + 32

            return BuoySST(
                buoy_id=buoy_id,
                station_id=station_id,
                water_temp_c=water_temp_c,
                water_temp_f=round(water_temp_f, 1),
                observation_time=obs_time,
                wave_height_m=wave_height,
                wind_dir=wind_dir,
                wind_speed_kt=round(wind_speed_kt, 1) if wind_speed_kt is not None else None,
            )
    except Exception as e:
        print(f"[NDBC] Parse error: {e}")

    return None


async def fetch_buoy_sst(station_id: str) -> Optional[BuoySST]:
    """
    Fetch real-time SST from NDBC buoy.

    Args:
        station_id: Weather station ID (KLGA, KATL, KMIA, etc.)

    Returns:
        BuoySST data or None if unavailable.
    """
    buoy_ids = BUOY_STATIONS.get(station_id) or []
    if not buoy_ids:
        # No buoy mapped for this station.
        return None

    async with httpx.AsyncClient(timeout=15.0) as client:
        for buoy_id in buoy_ids:
            url = f"{NDBC_BASE_URL}/{buoy_id}.txt"
            try:
                response = await client.get(url)
                response.raise_for_status()
                parsed = parse_ndbc_realtime(response.text, buoy_id, station_id)
                if parsed is not None:
                    return parsed
            except httpx.HTTPStatusError as e:
                print(f"[NDBC] HTTP error for {buoy_id}: {e.response.status_code}")
            except Exception as e:
                print(f"[NDBC] Error fetching {buoy_id}: {e}")
                continue
    return None


def is_wind_onshore(station_id: str, wind_dir: Optional[int]) -> bool:
    """
    Check if wind is blowing from water to land.

    Args:
        station_id: Weather station ID.
        wind_dir: Wind direction in degrees.

    Returns:
        True if wind is onshore.
    """
    if wind_dir is None:
        return False

    onshore_ranges = {
        "KLGA": [(0, 90), (270, 360)],  # N->E and W->N sectors
        "KATL": [],  # Atlanta is inland
        "KMIA": [(40, 170)],  # Atlantic/Biscayne flow into Miami area
    }
    ranges = onshore_ranges.get(station_id, [])
    for low, high in ranges:
        if low <= wind_dir <= high:
            return True
    return False


if __name__ == "__main__":
    async def test() -> None:
        for station in ("KLGA", "KMIA"):
            sst = await fetch_buoy_sst(station)
            if sst:
                print(f"[{station}] Buoy: {sst.buoy_id}")
                print(f"[{station}] Water Temp: {sst.water_temp_f}F ({sst.water_temp_c}C)")
                print(f"[{station}] Obs Time: {sst.observation_time}")
                print(f"[{station}] Wind: {sst.wind_dir}deg @ {sst.wind_speed_kt}kt")
            else:
                print(f"[{station}] No SST data available")

    asyncio.run(test())
