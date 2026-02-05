"""
Helios Weather Lab - Collector Module (Protocol FORTRESS)
Unified data collection from METAR and multi-model sources.
"""

from .metar_fetcher import fetch_metar, MetarData
from .hrrr_fetcher import fetch_hrrr, MultiModelData, ModelForecast

__all__ = [
    "fetch_metar", "fetch_hrrr",
    "MetarData", "MultiModelData", "ModelForecast",
    "collect_all_data"
]


async def collect_all_data(station_id: str, target_days_ahead: int = 0) -> dict:
    """
    Collect all weather data for a station (FORTRESS enhanced).
    
    Args:
        station_id: ICAO station identifier
        target_days_ahead: Days ahead to forecast (0=today, 1=tomorrow)
        
    Returns:
        Dictionary with combined METAR and multi-model forecast data
    """
    from config import STATIONS
    
    station = STATIONS.get(station_id)
    if not station:
        raise ValueError(f"Unknown station: {station_id}")
    
    # Fetch both sources
    metar_data = await fetch_metar(station_id)
    forecast_data = await fetch_hrrr(station.latitude, station.longitude, target_days_ahead)
    
    return {
        "station_id": station_id,
        "metar": metar_data,
        "hrrr": forecast_data,  # Now contains MultiModelData with HRRR and GFS
    }
