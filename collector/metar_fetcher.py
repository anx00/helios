"""
Helios Weather Lab - METAR Fetcher
Fetches and parses METAR data from NOAA Aviation Weather Center.
"""

import httpx
import math
import asyncio
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timezone, timedelta
import logging

from config import NOAA_METAR_URL
from collector.metar.tds_fetcher import fetch_metar_tds
from collector.metar.tgftp_fetcher import fetch_metar_tgftp

import time

logger = logging.getLogger("metar_fetcher")

# --- Phase 2: METAR Race Dedupe + Latency Registry ---
_last_published_obs: dict = {}  # station_id -> obs_time_utc (dedupe)
_race_latency_log: list = []    # List of {station, route, latency_ms, won}


@dataclass
class MetarData:
    """Parsed METAR observation data."""
    station_id: str
    observation_time: datetime
    temp_c: float
    dewpoint_c: float
    humidity_pct: float
    wind_dir_degrees: Optional[int]
    wind_speed_kt: Optional[int]
    sky_condition: str
    sky_conditions_list: List[str]
    visibility_miles: Optional[float]
    altimeter_hg: Optional[float]
    wind_trend: str                  # NEW: "Estable" or "Girando al [Dir]"
    raw_metar: str
    racing_results: Optional[dict] = None  # NEW: Store all sources {source: temp_f}
    
    @property
    def temp_f(self) -> float:
        """Temperature in Fahrenheit."""
        if self.temp_c is None:
            return 0.0
        return round((self.temp_c * 9/5) + 32, 1)


def calculate_humidity(temp_c: float, dewpoint_c: float) -> float:
    """
    Calculate relative humidity from temperature and dewpoint.
    Uses the Magnus-Tetens approximation.
    
    Args:
        temp_c: Temperature in Celsius
        dewpoint_c: Dewpoint in Celsius
        
    Returns:
        Relative humidity as percentage (0-100)
    """
    if temp_c is None or dewpoint_c is None:
        return 0.0
    
    # Magnus-Tetens coefficients
    a = 17.27
    b = 237.7
    
    # Calculate saturation vapor pressure
    gamma_t = (a * temp_c) / (b + temp_c)
    gamma_td = (a * dewpoint_c) / (b + dewpoint_c)
    
    # Relative humidity
    rh = 100 * math.exp(gamma_td - gamma_t)
    
    return round(min(100.0, max(0.0, rh)), 1)


def parse_sky_condition(sky_data: List[dict]) -> tuple[str, List[str]]:
    """
    Parse sky condition from METAR JSON response.
    
    Args:
        sky_data: List of sky condition dictionaries from NOAA API
        
    Returns:
        Tuple of (primary condition, list of all conditions)
    """
    if not sky_data:
        return "CLR", ["CLR"]
    
    conditions = []
    for layer in sky_data:
        cover = layer.get("cover", "CLR")
        conditions.append(cover)
    
    # Priority order for primary condition (most significant)
    priority = {"OVC": 4, "BKN": 3, "SCT": 2, "FEW": 1, "CLR": 0, "SKC": 0}
    
    primary = max(conditions, key=lambda x: priority.get(x, 0))
    
    return primary, conditions


async def fetch_metar_json(station_id: str) -> Optional[dict]:
    """Helper for the JSON API source."""
    url = f"{NOAA_METAR_URL}?ids={station_id}&format=json"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            if not data: return None
            obs = data[0]
            
            # Time parsing
            obs_time_raw = obs.get("obsTime")
            try:
                if isinstance(obs_time_raw, int):
                    obs_time = datetime.fromtimestamp(obs_time_raw, tz=timezone.utc)
                elif isinstance(obs_time_raw, str):
                    obs_time = datetime.fromisoformat(obs_time_raw.replace("Z", "+00:00"))
                else:
                    obs_time = datetime.now(timezone.utc)
            except:
                obs_time = datetime.now(timezone.utc)

            return {
                "station_id": station_id,
                "obs_time": obs_time,
                "temp": obs.get("temp"),
                "dewp": obs.get("dewp"),
                "wdir": obs.get("wdir"),
                "wspd": obs.get("wspd"),
                "visib": obs.get("visib"),
                "altim": obs.get("altim"),
                "clouds": obs.get("clouds", []),
                "raw_ob": obs.get("rawOb", ""),
                "source": "NOAA_JSON_API"
            }
    except Exception:
        return None

async def fetch_metar(station_id: str) -> Optional[MetarData]:
    """
    Fetch and parse METAR data for a station using Protocol RACING.
    Concurrently polls three NOAA sources and takes the winner.
    """
    logger.info(f"Starting METAR Race for {station_id}...")

    # Phase 2: Per-route latency measurement
    race_start = time.monotonic()

    async def timed_fetch(coro, source_name):
        """Wrap fetch in timing envelope."""
        t0 = time.monotonic()
        try:
            result = await coro
            elapsed_ms = (time.monotonic() - t0) * 1000
            if result:
                result["_route_latency_ms"] = round(elapsed_ms, 1)
            return result
        except Exception:
            return None

    tasks = [
        asyncio.create_task(timed_fetch(fetch_metar_json(station_id), "NOAA_JSON_API")),
        asyncio.create_task(timed_fetch(fetch_metar_tds(station_id), "AWC_TDS_XML")),
        asyncio.create_task(timed_fetch(fetch_metar_tgftp(station_id), "TGFTP_TXT"))
    ]

    # Aggressive 3s timeout
    done, pending = await asyncio.wait(tasks, timeout=3.0, return_when=asyncio.ALL_COMPLETED)

    for p in pending:
        if p.done():
            done.add(p)
        else:
            p.cancel()

    valid_results = []
    racing_log = {}

    for task in done:
        try:
            res = task.result()
            if res and res.get("temp") is not None:
                res["temp_f"] = round((res["temp"] * 9/5) + 32, 1)

                # Normalize time for comparison
                obs_time = res.get("obs_time")
                if isinstance(obs_time, str):
                    try:
                        if "T" in obs_time:
                            res["dt"] = datetime.fromisoformat(obs_time.replace("Z", "+00:00"))
                        else:
                            res["dt"] = datetime.strptime(obs_time, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
                    except:
                        res["dt"] = datetime.now(timezone.utc)
                else:
                    res["dt"] = obs_time or datetime.now(timezone.utc)

                valid_results.append(res)
                racing_log[res["source"]] = res["temp_f"]
        except asyncio.CancelledError:
            continue
        except Exception:
            continue

    if not valid_results:
        logger.warning(f"METAR Race failed for {station_id}: No valid sources returned in time")
        return None

    # Pick the winner (most recent obs_time, then highest temp)
    valid_results.sort(key=lambda x: (x["dt"], x["temp_f"]), reverse=True)
    winner = valid_results[0]

    # Phase 2: Log per-route latencies
    for r in valid_results:
        route_ms = r.get("_route_latency_ms", 0)
        _race_latency_log.append({
            "station": station_id,
            "route": r["source"],
            "latency_ms": route_ms,
            "won": r is winner,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    # Keep log bounded
    while len(_race_latency_log) > 500:
        _race_latency_log.pop(0)

    sources_str = ", ".join([f"{r['source']}={r['temp_f']}({r.get('_route_latency_ms',0):.0f}ms)" for r in valid_results])
    logger.info(f"METAR Race {station_id}: {sources_str} | Winner: {winner['source']}")

    # Phase 2: Dedupe — skip if same obs_time already published
    winner_obs_time = winner.get("dt")
    last_pub = _last_published_obs.get(station_id)
    is_duplicate = (last_pub is not None and winner_obs_time is not None
                    and abs((winner_obs_time - last_pub).total_seconds()) < 5)
    if is_duplicate:
        logger.debug(f"METAR dedupe: {station_id} obs_time={winner_obs_time} already published, skipping WorldState publish")

    
    # Map back to MetarData (Legacy backward compatibility)
    temp_c = winner.get("temp")
    dewpoint_c = winner.get("dewp")
    humidity = calculate_humidity(temp_c, dewpoint_c) if temp_c and dewpoint_c else None
    
    # Sky
    sky_data = winner.get("clouds", [])
    if isinstance(sky_data, list):
        primary_sky, all_sky = parse_sky_condition(sky_data)
    else:
        primary_sky, all_sky = "CLR", ["CLR"]
        
    # MODULE A: Wind Trend (from history - usually JSON API)
    wind_trend = "Estable"
    try:
        history = await asyncio.wait_for(fetch_metar_history(station_id, hours=1), timeout=5.0)
        if len(history) >= 2:
            w1 = history[0].wind_dir_degrees
            w2 = history[1].wind_dir_degrees
            if w1 is not None and w2 is not None:
                diff = (w1 - w2 + 180) % 360 - 180
                if abs(diff) > 30:
                    dirs = ["Norte", "Noreste", "Este", "Sureste", "Sur", "Suroeste", "Oeste", "Noroeste"]
                    dir_idx = int(((w1 + 22.5) % 360) / 45)
                    wind_trend = f"Girando al {dirs[dir_idx]}"
    except: pass

    # =================================================================
    # Phase 2 Event-Driven Publish (with dedupe + QC)
    # =================================================================
    if not is_duplicate:
        try:
            from core.models import OfficialObs
            from core.judge import JudgeAlignment
            from core.qc import QualityControl
            from core.world import get_world

            # 1. Judge Alignment
            temp_f_raw = JudgeAlignment.celsius_to_fahrenheit(temp_c)  # With decimals
            temp_f_aligned = JudgeAlignment.round_to_settlement(temp_f_raw)  # Rounded for settlement

            sky_aligned = JudgeAlignment.map_sky_condition(primary_sky)

            # 2. QC Checks
            qc_result = QualityControl.check_hard_rules(
                temp_c if temp_c is not None else 0.0,
                dewpoint_c
            )

            # 3. Create Official Event with QC results
            event = OfficialObs(
                obs_time_utc=winner["dt"],
                ingest_time_utc=datetime.now(timezone.utc),
                source=winner["source"],
                station_id=station_id,
                temp_c=temp_c if temp_c is not None else 0.0,
                temp_f=temp_f_aligned,  # Rounded for settlement
                temp_f_raw=temp_f_raw,  # With decimals for display
                dewpoint_c=dewpoint_c,
                wind_dir=winner.get("wdir"),
                wind_speed=winner.get("wspd"),
                sky_condition=sky_aligned,
                qc_passed=qc_result.is_valid,
                qc_flags=qc_result.flags,
            )

            # 4. Publish to World
            world = get_world()
            world.publish(event)

            # 5. Update QC state for this station
            world.set_qc_state(station_id, qc_result.status, qc_result.flags)

            # 6. Mark obs_time as published (dedupe)
            _last_published_obs[station_id] = winner_obs_time

        except Exception as e:
            logger.error(f"Failed to publish OfficialObs event: {e}")
            try:
                from core.world import get_world
                get_world().record_error(f"METAR_{station_id}", str(e))
            except:
                pass

    # Return Legacy format for current consumers (until full migration)
    return MetarData(
        station_id=station_id,
        observation_time=winner["dt"],
        temp_c=temp_c,
        dewpoint_c=dewpoint_c,
        humidity_pct=humidity or 0.0,
        wind_dir_degrees=winner.get("wdir"),
        wind_speed_kt=winner.get("wspd"),
        sky_condition=primary_sky,
        sky_conditions_list=all_sky,
        visibility_miles=winner.get("visib"),
        altimeter_hg=winner.get("altim"),
        wind_trend=wind_trend,
        raw_metar=winner.get("raw_ob", ""),
        racing_results=racing_log
    )


# Backward compatibility alias (used by web_server)
async def fetch_metar_race(station_id: str) -> Optional[MetarData]:
    return await fetch_metar(station_id)



async def fetch_metar_history(station_id: str, hours: int = 24) -> List[MetarData]:
    """
    Fetch historical METAR observations for a station.
    
    Args:
        station_id: ICAO station identifier
        hours: Number of hours of history to fetch
        
    Returns:
        List of MetarData objects
    """
    url = f"{NOAA_METAR_URL}?ids={station_id}&format=json&hours={hours}"
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return []
            
            results = []
            for obs in data:
                try:
                    obs_time_raw = obs.get("obsTime")
                    try:
                        if isinstance(obs_time_raw, int):
                            obs_time = datetime.fromtimestamp(obs_time_raw, tz=timezone.utc)
                        elif isinstance(obs_time_raw, str):
                            obs_time = datetime.fromisoformat(obs_time_raw.replace("Z", "+00:00"))
                        else:
                            obs_time = datetime.now(timezone.utc)
                    except (ValueError, TypeError, OSError):
                        obs_time = datetime.now(timezone.utc)
                    
                    temp_c = obs.get("temp")
                    dewpoint_c = obs.get("dewp")
                    humidity = calculate_humidity(temp_c, dewpoint_c) if temp_c and dewpoint_c else None
                    
                    sky_data = obs.get("clouds", [])
                    primary_sky, all_sky = parse_sky_condition(sky_data)
                    
                    results.append(MetarData(
                        station_id=station_id,
                        observation_time=obs_time,
                        temp_c=temp_c,
                        dewpoint_c=dewpoint_c,
                        humidity_pct=humidity,
                        wind_dir_degrees=obs.get("wdir"),
                        wind_speed_kt=obs.get("wspd"),
                        sky_condition=primary_sky,
                        sky_conditions_list=all_sky,
                        visibility_miles=obs.get("visib"),
                        altimeter_hg=obs.get("altim"),
                        wind_trend="Capturando...",  # Placeholder for history loop
                        raw_metar=obs.get("rawOb", ""),
                    ))
                except Exception as e:
                    print(f"⚠ Error parsing historical METAR: {e}")
                    continue
            
            return results
            
    except Exception as e:
        print(f"✗ METAR history error for {station_id}: {e}")
        return []

# Alias for compatibility with other modules expecting 'fetch_metar_data'
fetch_metar_data = fetch_metar
