"""
Helios Weather Lab - Wunderground Fetcher
Scrapes current day's maximum temperature from Weather Underground.
"""

import httpx
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from config import STATIONS
from synthesizer.reality_validator import validate_reality_floor


@dataclass
class WundergroundHourlyEntry:
    time: str  # Local time string (e.g. "12:51 PM")
    temp_f: float
    conditions: str = ""

@dataclass
class WundergroundObservation:
    """Observed maximum temperature data with cross-verification status."""
    station_id: str
    date: date
    high_temp_f: float              # Maximum observed temperature (verified)
    high_temp_time: Optional[str]   # When the max occurred (local time string)
    current_temp_f: Optional[float] # Current temperature
    forecast_high_f: Optional[float] = None # v5.1: WU Forecasted peak for the day
    source: str = "wunderground"
    
    # Hourly data
    hourly_history: list[WundergroundHourlyEntry] = None
    
    # Cross-verification fields
    metar_max_f: Optional[float] = None     # Max from official METAR
    wunderground_max_f: Optional[float] = None  # Max from Wunderground history
    wunderground_current_f: Optional[float] = None  # Real-time from WU
    verification_status: str = "unverified"  # "verified", "unverified", "discrepancy"
    confidence: str = "low"  # "high", "medium", "low"
    notes: str = ""  # Explanation of verification decision
    metar_racing: Optional[dict] = None # v8.0: Racing results from multi-source fetch


# Wunderground URL patterns
WUNDERGROUND_HISTORY_URL = "https://www.wunderground.com/history/daily/us/{region}/{city}/{station_id}/date/{year}-{month}-{day}"

# Station to Wunderground location mapping
# v7.0: Added 'units' field - 'e' for imperial (Fahrenheit), 'm' for metric (Celsius)
STATION_WU_MAP = {
    "KLGA": {"region": "ny", "city": "new-york-city", "country": "us", "units": "e"},
    "KATL": {"region": "ga", "city": "atlanta", "country": "us", "units": "e"},
    "EGLC": {"region": "gb/london", "city": "city-of-london", "country": "gb", "units": "m"},  # UK uses Celsius
}


async def fetch_wunderground_max(
    station_id: str,
    target_date: Optional[date] = None
) -> Optional[WundergroundObservation]:
    """
    Fetch the maximum observed temperature from Wunderground for a given date.
    
    Args:
        station_id: ICAO station identifier
        target_date: Date to fetch (defaults to today in station's timezone)
        
    Returns:
        WundergroundObservation with observed max, or None on error
    """
    if station_id not in STATION_WU_MAP:
        return None
    
    # Get station timezone
    station_config = STATIONS.get(station_id)
    if not station_config:
        return None
    
    local_tz = ZoneInfo(station_config.timezone)
    
    # Default to today in station's local time
    if target_date is None:
        target_date = datetime.now(local_tz).date()
    
    # Build URL - handle different country formats
    wu_config = STATION_WU_MAP[station_id]
    country = wu_config.get("country", "us")  # Default to US
    
    # UK uses different URL path format
    if country == "gb":
        url = f"https://www.wunderground.com/history/daily/gb/{wu_config['city']}/{station_id}/date/{target_date.year}-{target_date.month}-{target_date.day}"
    else:
        url = WUNDERGROUND_HISTORY_URL.format(
            region=wu_config["region"],
            city=wu_config["city"],
            station_id=station_id,
            year=target_date.year,
            month=target_date.month,
            day=target_date.day
        )
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            
            # Initialize
            high_temp_f = None
            high_temp_time = None
            current_temp_f = None
            forecast_high_f = None # v5.8
            hourly_history = []

            # Method 1: Use the Weather.com API directly (Most robust)
            # URL format: api.weather.com/v1/location/{station}:9:{country}/observations/historical.json
            api_key = "e1f10a1e78da46f5b10a1e78da96f525" # Public WU key
            date_str = target_date.strftime("%Y%m%d")
            country_code = "GB" if country == "gb" else "US"
            
            # v7.0: Use correct units for each station
            api_units = wu_config.get("units", "e")  # 'e' = imperial (F), 'm' = metric (C)
            use_metric = (api_units == "m")
            
            api_url = f"https://api.weather.com/v1/location/{station_id}:9:{country_code}/observations/historical.json?apiKey={api_key}&units={api_units}&startDate={date_str}&endDate={date_str}"
            
            try:
                api_response = await client.get(api_url, headers=headers, timeout=10.0)
                if api_response.status_code == 200:
                    api_data = api_response.json()
                    obs_list = api_data.get("observations", [])
                    for obs in obs_list:
                        temp = obs.get("temp")
                        if temp is not None:
                            # Time processing (v1 API uses valid_time_gmt)
                            obs_time = "Unknown"
                            valid_time = obs.get("valid_time_gmt")
                            if valid_time:
                                try:
                                    dt_obs = datetime.fromtimestamp(valid_time, tz=local_tz)
                                    obs_time = dt_obs.strftime("%I:%M %p")
                                except: pass
                            
                            cond = obs.get("wx_phrase", "")
                            
                            # v7.0: Convert Celsius to Fahrenheit if using metric units
                            temp_f = float(temp)
                            if use_metric:
                                temp_f = (float(temp) * 9/5) + 32
                            
                            hourly_history.append(WundergroundHourlyEntry(
                                time=obs_time,
                                temp_f=temp_f,
                                conditions=cond
                            ))
                            
                            if high_temp_f is None or temp_f > high_temp_f:
                                high_temp_f = temp_f
                                high_temp_time = obs_time

                else:
                    pass  # API error - will fallback to scraping
            except Exception as e:
                pass  # API failed - will fallback to scraping

            # Method 2: FALLBACK - Scraping & Script Parsing (If API failed or returned nothing)
            if not hourly_history:
                url = f"https://www.wunderground.com/history/daily/us/{wu_config['region']}/{wu_config['city']}/{station_id}/date/{target_date.strftime('%Y-%m-%d')}"
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Logic for Method 3 & 4 (search scripts for embedded JSON)
                scripts = soup.find_all("script")
                best_observations = []
                for script in scripts:
                    if script.string and '"observations":' in script.string:
                        try:
                            import re
                            import json
                            pattern = r'"observations":\s*(\[.*?\])(?:,"|\})'
                            matches = re.findall(pattern, script.string, re.DOTALL)
                            for m in matches:
                                try:
                                    data = json.loads(m)
                                    if isinstance(data, list) and len(data) > len(best_observations):
                                        best_observations = data
                                except: continue
                        except: continue
                
                if best_observations:
                    for obs in best_observations:
                        try:
                            temp = None
                            if "imperial" in obs and "temp" in obs["imperial"]:
                                temp = obs["imperial"]["temp"]
                            elif "temp" in obs:
                                temp = obs["temp"]
                            if temp is not None:
                                obs_time = "Unknown"
                                for time_key in ["validTimeLocal", "obsTimeLocal"]:
                                    if time_key in obs:
                                        try:
                                            dt_obs = datetime.fromisoformat(obs[time_key].replace("Z", "+00:00"))
                                            obs_time = dt_obs.strftime("%I:%M %p")
                                            break
                                        except: continue
                                hourly_history.append(WundergroundHourlyEntry(
                                    time=obs_time, temp_f=float(temp), conditions=obs.get("wxPhraseLong", "")
                                ))
                                if high_temp_f is None or float(temp) > high_temp_f:
                                    high_temp_f = float(temp)
                                    high_temp_time = obs_time
                        except: continue

            # Method 4: Optimized Dynamic Forecast Logic (v5.9)
            try:
                # 4.1: Fetch Daily Forecast for the baseline
                # v7.0: Use correct units for station
                forecast_api = f"https://api.weather.com/v3/wx/forecast/daily/5day?icaoCode={station_id}&units={api_units}&language=en-US&format=json&apiKey={api_key}"
                f_response = await client.get(forecast_api, headers=headers, timeout=10.0)
                
                daily_max = None
                if f_response.status_code == 200:
                    f_data = f_response.json()
                    # Find the index for target_date
                    target_iso = target_date.isoformat()
                    for i, dt_str in enumerate(f_data.get("validTimeLocal", [])):
                        if dt_str.startswith(target_iso):
                            max_list = f_data.get("calendarDayTemperatureMax", [])
                            if len(max_list) > i:
                                daily_max = float(max_list[i])
                                # v7.0: Convert C to F if using metric
                                if use_metric:
                                    daily_max = (daily_max * 9/5) + 32
                                break
                
                # 4.2: Fetch Hourly Forecast for "Live Adjustment"
                # Logic: Today's Dynamic Max = max(Observed so far, Predicted for rest of day)
                hourly_api = f"https://api.weather.com/v3/wx/forecast/hourly/2day?icaoCode={station_id}&units={api_units}&language=en-US&format=json&apiKey={api_key}"
                h_response = await client.get(hourly_api, headers=headers, timeout=10.0)
                
                upcoming_max = -999.0
                if h_response.status_code == 200:
                    h_data = h_response.json()
                    target_iso = target_date.isoformat()
                    upcoming_temps = [
                        t for j, t in enumerate(h_data.get("temperature", []))
                        if h_data.get("validTimeLocal", [])[j].startswith(target_iso)
                    ]
                    if upcoming_temps:
                        upcoming_max = max(upcoming_temps)
                        # v7.0: Convert C to F if using metric
                        if use_metric:
                            upcoming_max = (upcoming_max * 9/5) + 32
                
                # Final Forecast Calculation:
                # If it's today, we take the max of what happened vs what is predicted to happen
                # This makes the line "descend" or "adjust" as peak hours pass.
                today_local = datetime.now(local_tz).date()
                if target_date == today_local:
                    # Current observed max (from Method 1/2 or current)
                    current_highest = high_temp_f if high_temp_f is not None else -999.0
                    forecast_high_f = max(current_highest, upcoming_max)
                    if forecast_high_f < -900: # Fallback to daily
                        forecast_high_f = daily_max
                else:
                    # For future dates, use the higher of upcoming(hourly) or daily
                    forecast_high_f = max(daily_max if daily_max else -999.0, upcoming_max)
                
                if forecast_high_f is not None and forecast_high_f < -900:
                    forecast_high_f = daily_max if daily_max else None
                
                # Still get current real-time temp from the weather page
                # v7.0: Fix URL for non-US stations
                if country == "gb":
                    current_url = f"https://www.wunderground.com/weather/gb/{wu_config['city']}/{station_id}"
                else:
                    current_url = f"https://www.wunderground.com/weather/us/{wu_config['region']}/{wu_config['city']}/{station_id}"
                current_response = await client.get(current_url, headers=headers)
                if current_response.status_code == 200:
                    current_soup = BeautifulSoup(current_response.text, "html.parser")
                    current_temp_elem = current_soup.find("span", class_="wu-value-to")
                    if current_temp_elem:
                        try:
                            scraped_temp = float(current_temp_elem.text.strip())
                            # v7.0: For UK pages, the displayed temp is in Celsius - convert to F
                            if use_metric:
                                # Validate: UK temps should be reasonable (-10 to 35 C)
                                if -10 <= scraped_temp <= 35:
                                    current_temp_f = (scraped_temp * 9/5) + 32
                            else:
                                current_temp_f = scraped_temp
                        except: pass
            except Exception as e:
                pass  # Dynamic forecast error - will continue with available data
            
            # v5.7 FIX: Do NOT overwrite historical max with current temp
            # They are separate measurements:
            # - high_temp_f = max from historical hourly data (what has been recorded)
            # - current_temp_f = real-time reading (may not be in history yet)
            # The UI should show both separately, not combine them.
            
            if current_temp_f is not None:
                # Just add current to hourly history for reference, but DON'T change high_temp_f
                current_time_str = datetime.now(local_tz).strftime("%I:%M %p")
                already_exists = any(h.time == "Ahora" or h.time == current_time_str for h in hourly_history)
                if not already_exists:
                    hourly_history.append(WundergroundHourlyEntry(
                        time="Ahora",
                        temp_f=current_temp_f,
                        conditions="Current"
                    ))
            
            if high_temp_f is None and not hourly_history:
                return None
            
            return WundergroundObservation(
                station_id=station_id,
                date=target_date,
                high_temp_f=high_temp_f,
                high_temp_time=high_temp_time,
                current_temp_f=current_temp_f,
                forecast_high_f=forecast_high_f,
                source="wunderground",
                hourly_history=hourly_history
            )
            
    except httpx.HTTPStatusError as e:
        return None
    except httpx.RequestError as e:
        return None
    except Exception as e:
        return None


async def get_observed_max(
    station_id: str,
    target_date: Optional[date] = None
) -> Optional[WundergroundObservation]:
    """
    Get observed maximum temperature with cross-verification between sources.
    
    Implements strict verification logic:
    1. Get METAR official max (ground truth)
    2. Get Wunderground history max
    3. Get Wunderground current temp
    4. Cross-verify: Only trust values confirmed by METAR
    5. Apply confidence scoring based on agreement
    
    Args:
        station_id: ICAO station identifier
        target_date: Date to check (defaults to today)
        
    Returns:
        WundergroundObservation with verified max and confidence level
    """
    from collector.metar_fetcher import fetch_metar_history, fetch_metar
    from config import STATIONS
    
    # Get station timezone
    station_config = STATIONS.get(station_id)
    if not station_config:
        return None
    
    local_tz = ZoneInfo(station_config.timezone)
    now_local = datetime.now(local_tz)
    
    if target_date is None:
        target_date = now_local.date()
    
    # =========================================================================
    # STEP 1: Get METAR official max (GROUND TRUTH)
    # =========================================================================
    metar_max_f = None
    metar_max_time_str = None
    metar_lag_mins = 999
    metar_racing = None
    
    try:
        # v8.0: Fetch current METAR racing results in parallel with history
        current_metar_task = asyncio.create_task(fetch_metar(station_id))
        history_task = asyncio.create_task(fetch_metar_history(station_id, hours=24))
        
        current_metar, history = await asyncio.gather(current_metar_task, history_task)
        
        if current_metar:
            metar_racing = current_metar.racing_results

        if history:
            # Filter to target date only
            todays_obs = []
            for obs in history:
                if obs.temp_c is None:
                    continue
                obs_local = obs.observation_time.astimezone(local_tz)
                if obs_local.date() == target_date:
                    todays_obs.append(obs)
            
            if todays_obs:
                max_obs = max(todays_obs, key=lambda x: x.temp_c)
                metar_max_f = max_obs.temp_f
                metar_max_time_str = max_obs.observation_time.astimezone(local_tz).strftime("%H:%M")
                
                # Check lag
                latest_obs = max(todays_obs, key=lambda x: x.observation_time)
                metar_lag_mins = (datetime.now(timezone.utc) - latest_obs.observation_time).total_seconds() / 60
    except Exception as e:
        pass  # METAR error - will continue with other sources
    
    # =========================================================================
    # STEP 2: Get Wunderground HISTORY max (not current)
    # =========================================================================
    wu_history_max = None
    wu_current = None
    wu_hourly = []
    
    try:
        wu_data = await fetch_wunderground_max(station_id, target_date)
        if wu_data:
            wu_history_max = wu_data.high_temp_f  # This is already the MAX from history loop
            wu_current = wu_data.current_temp_f
            wu_hourly = wu_data.hourly_history or []
    except Exception as e:
        pass  # Wunderground error - will continue with METAR

    
    # =========================================================================
    # STEP 3: Cross-verification logic
    # =========================================================================
    TOLERANCE = 1.0
    final_max_f = None
    verification_status = "unverified"
    confidence = "low"
    notes = []
    source = "unknown"
    
    # Strict Verification Filter
    if metar_max_f is not None:
        final_max_f = metar_max_f # METAR is the anchor
        source = "metar"
        
        wu_potential_max = max([v for v in [wu_history_max, wu_current] if v is not None] + [metar_max_f])
        
        if wu_potential_max > metar_max_f + 1.5:
            # DISCREPANCY DETECTED
            if metar_lag_mins < 35:
                # METAR is fresh and DOES NOT see the 49F that WU claims. 
                # This is almost certainly PWS noise/error.
                verification_status = "suspicious"
                confidence = "low"
                notes.append(f"⚠️ SUSPICIOUS: WU says {wu_potential_max}°F but METAR ({metar_lag_mins:.0f}m ago) only shows {metar_max_f}°F")
                notes.append("Status: Ignorado por falta de confirmación NOAA (posible sensor PWS erróneo)")
            else:
                # METAR is stale. WU might be right, but we wait.
                verification_status = "pending"
                confidence = "medium"
                notes.append(f"⏳ PENDING: WU says {wu_potential_max}°F. METAR lag is {metar_lag_mins:.0f}m")
                notes.append("Status: Esperando confirmación METAR oficial antes de subir el suelo")
        
        elif wu_history_max is not None:
            diff = abs(wu_history_max - metar_max_f)
            if diff <= TOLERANCE:
                verification_status = "verified"
                confidence = "high"
                notes.append(f"METAR and WU agree (diff={diff:.1f}°F)")
            else:
                verification_status = "verified"
                confidence = "medium"
                notes.append(f"METAR={metar_max_f}°F, WU={wu_history_max}°F (diff={diff:.1f}°F). Using METAR ground truth.")
        else:
            verification_status = "verified"
            confidence = "medium"
            notes.append("Only METAR available, used as ground truth.")
    
    # Case 2: No METAR (Problematic), fall back to Wunderground with LOW confidence
    elif wu_history_max is not None:
        final_max_f = wu_history_max
        source = "wunderground"
        verification_status = "unverified"
        confidence = "low"
        notes.append("❌ NO METAR AVAILABLE. Using WU as fallback (UNVERIFIED)")
    
    # Case 3: Only WU current available (least trustworthy)
    elif wu_current is not None:
        final_max_f = wu_current
        source = "wunderground_current"
        verification_status = "unverified"
        confidence = "low"
        notes.append("⚠️ Only WU current available - extremely low confidence")
    
    else:
        return None

    # =========================================================================
    # v3.0: KLGA Anti-Ghost Peak (Triple Peak Validation)
    # =========================================================================
    if station_id == "KLGA" and final_max_f is not None:
        target_date_str = target_date.strftime("%Y-%m-%d")
        validated_f, is_new_peak = validate_reality_floor(station_id, target_date_str, final_max_f)
        
        if validated_f < final_max_f:
            notes.append(f"🛑 ANTI-GHOST PK: Suelo bloqueado en {validated_f:.1f}F (Esperando segunda lectura para {final_max_f:.1f}F)")
            final_max_f = validated_f
        elif is_new_peak:
            notes.append(f"✅ PEAK VERIFIED: Nuevo suelo confirmado a {final_max_f:.1f}F")
    
    return WundergroundObservation(
        station_id=station_id,
        date=target_date,
        high_temp_f=final_max_f,
        high_temp_time=metar_max_time_str,
        current_temp_f=wu_current,
        source=source,
        hourly_history=wu_hourly,
        metar_max_f=metar_max_f,
        wunderground_max_f=wu_history_max,
        wunderground_current_f=wu_current,
        verification_status=verification_status,
        confidence=confidence,
        notes="; ".join(notes),
        metar_racing=metar_racing
    )
