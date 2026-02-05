"""
Helios Weather Lab - Configuration
Central configuration for weather stations and API endpoints.
"""

from dataclasses import dataclass
from typing import Dict
import sys

# ============================================================================
# HELIOS v3.0: SYSTEM PATCHES 
# Force ECMWF/Copernicus libraries to avoid the "500 retries" hang
# ============================================================================
try:
    import multiurl.http
    import multiurl.downloader
    
    # 1. Patch the robust function directly
    if not hasattr(multiurl.http, 'original_robust'):
        multiurl.http.original_robust = multiurl.http.robust
        
        def aggressive_robust(call, *args, **kwargs):
            """Override retry parameters strictly."""
            new_args = list(args)
            if len(new_args) >= 1: new_args[0] = 3 # maximum_tries
            else: kwargs['maximum_tries'] = 3
            if len(new_args) >= 2: new_args[1] = 5 # retry_after
            else: kwargs['retry_after'] = 5
            return multiurl.http.original_robust(call, *new_args, **kwargs)
            
        multiurl.http.robust = aggressive_robust
        multiurl.downloader.robust = aggressive_robust
        if hasattr(multiurl, 'robust'):
            multiurl.robust = aggressive_robust
            
    # 2. Patch the class defaults to be absolutely sure
    original_downloader_init = multiurl.http.HTTPDownloaderBase.__init__
    def patched_downloader_init(self, *args, **kwargs):
        original_downloader_init(self, *args, **kwargs)
        self.maximum_retries = 3
        self.retry_after = 5
    
    multiurl.http.HTTPDownloaderBase.__init__ = patched_downloader_init
    # print("[HELIOS] Multiurl aggressive survival patch applied.")
except Exception:
    pass

# ============================================================================
# STATION CONFIGURATION
# ============================================================================

@dataclass
class StationConfig:
    """Weather station configuration with metadata."""
    icao_id: str
    name: str
    latitude: float
    longitude: float
    timezone: str
    characteristics: str  # Local microclimate notes
    enabled: bool = True  # Whether station is active


STATIONS: Dict[str, StationConfig] = {
    "KLGA": StationConfig(
        icao_id="KLGA",
        name="New York LaGuardia",
        latitude=40.7769,
        longitude=-73.8740,
        timezone="America/New_York",
        characteristics="Coastal station - high sensitivity to marine wind cooling (sea breeze)",
        enabled=True,
    ),
    "KATL": StationConfig(
        icao_id="KATL",
        name="Hartsfield-Jackson Atlanta",
        latitude=33.6407,
        longitude=-84.4277,
        timezone="America/New_York",
        characteristics="Continental - watch for afternoon storm cooling effects",
        enabled=False,  # DISABLED - focusing on KLGA only
    ),
    "EGLC": StationConfig(
        icao_id="EGLC",
        name="London City Airport",
        latitude=51.5048,
        longitude=0.0495,
        timezone="Europe/London",
        characteristics="Urban riverside - Thames river effect, Docklands urban heat",
        enabled=False,  # DISABLED - focusing on KLGA only
    ),
}

def get_active_stations() -> Dict[str, StationConfig]:
    """Return only enabled stations."""
    return {k: v for k, v in STATIONS.items() if v.enabled}

# ============================================================================
# API ENDPOINTS
# ============================================================================

# NOAA Aviation Weather Center
NOAA_METAR_URL = "https://aviationweather.gov/api/data/metar"

# Open-Meteo (HRRR Model)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# ============================================================================
# SCHEDULER CONFIGURATION
# ============================================================================

# Data collection interval (minutes)
COLLECTION_INTERVAL_MINUTES = 30

# Daily audit time (24h format)
AUDIT_HOUR = 1  # 01:00 AM
AUDIT_MINUTE = 0

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DATABASE_PATH = "helios_weather.db"

# ============================================================================
# GEMINI CONFIGURATION
# ============================================================================

# Fallback list for Gemini models
GEMINI_MODELS = [
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite",
    "models/gemini-1.5-flash",
    "models/gemini-2.0-pro-exp",
    "models/gemini-1.5-pro"
]

# ============================================================================
# SKY CONDITION MAPPINGS
# ============================================================================

SKY_CONDITIONS = {
    "CLR": "Clear skies",
    "SKC": "Sky clear",
    "FEW": "Few clouds (1-2 oktas)",
    "SCT": "Scattered clouds (3-4 oktas)",
    "BKN": "Broken clouds (5-7 oktas)",
    "OVC": "Overcast (8 oktas)",
    "VV": "Vertical visibility (obscured sky)",
}

# Cloud conditions that significantly reduce solar heating
CLOUDY_CONDITIONS = {"BKN", "OVC", "VV"}

# ============================================================================
# POLYMARKET CONFIGURATION
# ============================================================================

# Polymarket Gamma API endpoint (events, not markets)
POLYMARKET_EVENTS_URL = "https://gamma-api.polymarket.com/events"

# Probability threshold to consider market "resolved" (98% = $0.98)
MARKET_MATURE_PROBABILITY = 0.98

# City slug mappings for Polymarket event URLs
POLYMARKET_CITY_SLUGS = {
    "KLGA": "nyc",
    "KATL": "atlanta",
    "EGLC": "london",
}

# Spanish city names for logging
POLYMARKET_CITY_NAMES_ES = {
    "KLGA": "Nueva York",
    "KATL": "Atlanta",
    "EGLC": "Londres",
}

# ============================================================================
# KELLY RISK MANAGEMENT
# ============================================================================

BANKROLL_TOTAL = 1000.0
FRACTIONAL_KELLY = 0.25
MIN_BET = 5.0
MAX_BET_PCT = 0.15

# ============================================================================
# SYNOPTIC DATA API (PWS)
# ============================================================================

import os as _os
SYNOPTIC_API_TOKEN = _os.environ.get("SYNOPTIC_API_TOKEN", "")
SYNOPTIC_BASE_URL = "https://api.synopticdata.com/v2"

# PWS search configuration per station
PWS_SEARCH_CONFIG = {
    "KLGA": {"radius_km": 25, "max_stations": 20},
    "KATL": {"radius_km": 30, "max_stations": 20},
    "EGLC": {"radius_km": 20, "max_stations": 20},
}

# Maximum age of PWS observations (minutes)
PWS_MAX_AGE_MINUTES = 90

# Minimum stations for valid consensus
PWS_MIN_SUPPORT = 3