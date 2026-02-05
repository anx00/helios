"""
CAMS Aerosol Fetcher - Copernicus Atmosphere Monitoring Service

Fetches Aerosol Optical Depth (AOD) data to detect smoke/haze
that standard weather models miss.

Impact: -2°F to -4°F cooling when smoke present
"""
import httpx
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional
import asyncio


# CAMS API Configuration (Atmosphere Data Store)
CAMS_API_URL = "https://ads.atmosphere.copernicus.eu/api"
CAMS_API_KEY = "540d6eb0-bb86-4094-8a6b-ff9896cc5534"


@dataclass
class AerosolData:
    """Aerosol observation data."""
    station_id: str
    latitude: float
    longitude: float
    aod_550nm: float  # Total AOD at 550nm wavelength
    dust_aod: Optional[float]
    smoke_aod: Optional[float]
    observation_time: datetime
    quality: str  # "good", "moderate", "poor"
    
    @property
    def has_significant_aerosols(self) -> bool:
        """Check if aerosols are significant enough to affect temperature."""
        return self.aod_550nm > 0.3
    
    @property
    def physics_adjustment_f(self) -> float:
        """Calculate temperature adjustment based on AOD."""
        if self.aod_550nm > 0.5:
            return -2.0  # Heavy smoke/haze
        elif self.aod_550nm > 0.3:
            return -1.0  # Moderate haze
        return 0.0
    
    @property
    def physics_reason(self) -> str:
        """Get reason string for physics adjustment."""
        if self.aod_550nm > 0.5:
            return f"Smoke-2F (AOD={self.aod_550nm:.2f})"
        elif self.aod_550nm > 0.3:
            return f"Haze-1F (AOD={self.aod_550nm:.2f})"
        return ""


async def fetch_aod_simple(
    latitude: float,
    longitude: float,
    station_id: str = "UNKNOWN"
) -> Optional[AerosolData]:
    """
    Fetch PM2.5 from OpenAQ as AOD proxy.
    
    PM2.5 > 35 μg/m³ correlates with AOD > 0.3 (haze)
    PM2.5 > 55 μg/m³ correlates with AOD > 0.5 (smoke)
    """
    try:
        # OpenAQ v2 API - no key required
        openaq_url = (
            f"https://api.openaq.org/v2/latest"
            f"?coordinates={latitude},{longitude}"
            f"&radius=50000"  # 50km radius
            f"&parameter=pm25"
            f"&limit=1"
        )
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(openaq_url)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    measurements = results[0].get("measurements", [])
                    for m in measurements:
                        if m.get("parameter") == "pm25":
                            pm25 = m.get("value", 0)
                            
                            # Convert PM2.5 to approximate AOD
                            # Based on empirical relationship
                            if pm25 > 55:
                                aod = 0.6  # Heavy smoke/haze
                            elif pm25 > 35:
                                aod = 0.4  # Moderate haze
                            elif pm25 > 20:
                                aod = 0.2  # Light haze
                            else:
                                aod = 0.1  # Clear
                            
                            return AerosolData(
                                station_id=station_id,
                                latitude=latitude,
                                longitude=longitude,
                                aod_550nm=aod,
                                dust_aod=0.0,
                                smoke_aod=aod if pm25 > 35 else 0.0,
                                observation_time=datetime.now(timezone.utc),
                                quality=f"pm25={pm25:.0f}"
                            )
        
        # Fallback: baseline clear sky
        return AerosolData(
            station_id=station_id,
            latitude=latitude,
            longitude=longitude,
            aod_550nm=0.1,
            dust_aod=0.0,
            smoke_aod=0.0,
            observation_time=datetime.now(timezone.utc),
            quality="baseline"
        )
        
    except Exception as e:
        print(f"[AOD] OpenAQ error: {e}")
        return AerosolData(
            station_id=station_id,
            latitude=latitude,
            longitude=longitude,
            aod_550nm=0.1,
            dust_aod=0.0,
            smoke_aod=0.0,
            observation_time=datetime.now(timezone.utc),
            quality="fallback"
        )


async def fetch_aod_cdsapi(
    latitude: float,
    longitude: float,
    station_id: str = "UNKNOWN"
) -> Optional[AerosolData]:
    """
    Fetch AOD using official CAMS cdsapi library.
    
    Requires: pip install cdsapi xarray netcdf4
    
    The cdsapi configuration file (~/.cdsapirc) should contain:
    url: https://cds.climate.copernicus.eu/api
    key: YOUR_API_KEY
    """
    try:
        import cdsapi
        import xarray as xr
        import tempfile
        import os
        
        # Create client with credentials
        # v3.0: Add retry limits and short sleep to avoid hanging for hours
        client = cdsapi.Client(
            url=CAMS_API_URL,
            key=CAMS_API_KEY,
            retry_max=2,
            sleep_max=5
        )
        
        # Calculate date range (Use yesterday's run + 24h to ensure availability)
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")
        
        # Create temp file for download
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # v3.0: Run in thread to avoid blocking event loop and add a 60s timeout
            # Request CAMS global atmospheric composition forecast
            await asyncio.wait_for(
                asyncio.to_thread(
                    client.retrieve,
                    "cams-global-atmospheric-composition-forecasts",
                    {
                        "variable": "total_aerosol_optical_depth_550nm",
                        "date": date_str,
                        "time": "00:00",  # Use 00:00 run
                        "leadtime_hour": "24", # Forecast for +24h (Today)
                        "type": "forecast",
                        "format": "netcdf",
                        "area": [
                            latitude + 0.5, longitude - 0.5,
                            latitude - 0.5, longitude + 0.5
                        ],
                    },
                    tmp_path
                ),
                timeout=60.0
            )
            
            # Parse NetCDF (use context manager to ensure file is closed)
            with xr.open_dataset(tmp_path) as ds:
                # Extract AOD value (nearest point)
                var_name = list(ds.data_vars)[0]
                aod_value = float(ds[var_name].values.flatten()[0])
            
            return AerosolData(
                station_id=station_id,
                latitude=latitude,
                longitude=longitude,
                aod_550nm=aod_value,
                dust_aod=None,
                smoke_aod=None,
                observation_time=now,
                quality="cams"
            )
            
        except asyncio.TimeoutError:
            print(f"[CAMS] Timeout fetching AOD for {station_id}, falling back...")
            return await fetch_aod_simple(latitude, longitude, station_id)
        finally:
            # Cleanup temp file
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass
                
    except ImportError:
        print("[CAMS] cdsapi not installed. Run: pip install cdsapi xarray netcdf4")
        return await fetch_aod_simple(latitude, longitude, station_id)
    except Exception as e:
        print(f"[CAMS] API error: {e}")
        return await fetch_aod_simple(latitude, longitude, station_id)


# Main fetch function
async def fetch_aerosol_data(
    latitude: float,
    longitude: float,
    station_id: str = "UNKNOWN"
) -> Optional[AerosolData]:
    """
    Main entry point for aerosol data.
    Tries cdsapi first, falls back to simple method.
    """
    return await fetch_aod_cdsapi(latitude, longitude, station_id)


if __name__ == "__main__":
    # Test
    async def test():
        data = await fetch_aerosol_data(40.7769, -73.8740, "KLGA")
        if data:
            print(f"AOD: {data.aod_550nm}")
            print(f"Adjustment: {data.physics_adjustment_f}°F")
            print(f"Reason: {data.physics_reason or 'Clear'}")
    
    asyncio.run(test())
