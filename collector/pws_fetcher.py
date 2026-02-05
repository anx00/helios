"""
Helios Weather Lab - PWS (Personal Weather Station) Fetcher
Phase 2, Section 2.4: Cluster + Consensus + QC

Fetches nearby weather station observations and computes a robust
consensus using Median + MAD (Median Absolute Deviation).

Sources (priority order):
  1. Synoptic Data / MesoWest API (real PWS, requires SYNOPTIC_API_TOKEN)
  2. Open-Meteo Weather API (nearby grid points as pseudo-PWS fallback)

The output is an AuxObs(is_aggregate=True) with:
  - median temperature (the "consensus")
  - MAD (spread)
  - support (number of valid stations)
  - drift (consensus minus official METAR)
"""

import httpx
import asyncio
import statistics
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List

logger = logging.getLogger("pws_fetcher")

# ---------------------------------------------------------------------------
# Open-Meteo grid offsets (fallback pseudo-PWS)
# ---------------------------------------------------------------------------
_GRID_OFFSETS = {
    "KLGA": {
        "lat": 40.7769, "lon": -73.8740,
        "offsets": [
            (0.05, 0.0), (-0.05, 0.0), (0.0, 0.06), (0.0, -0.06),
            (0.10, 0.0), (-0.10, 0.0), (0.05, 0.06), (-0.05, -0.06),
            (0.15, 0.0), (0.0, -0.12),
        ],
    },
    "KATL": {
        "lat": 33.6407, "lon": -84.4277,
        "offsets": [
            (0.05, 0.0), (-0.05, 0.0), (0.0, 0.06), (0.0, -0.06),
            (0.10, 0.0), (-0.10, 0.0), (0.05, 0.06), (-0.05, -0.06),
            (0.15, 0.0), (0.0, -0.12),
        ],
    },
    "EGLC": {
        "lat": 51.5048, "lon": 0.0495,
        "offsets": [
            (0.04, 0.0), (-0.04, 0.0), (0.0, 0.06), (0.0, -0.06),
            (0.08, 0.0), (-0.08, 0.0), (0.04, 0.06), (-0.04, -0.06),
            (0.12, 0.0), (0.0, -0.10),
        ],
    },
}

# Station coordinates for Synoptic radius search
_STATION_COORDS = {
    "KLGA": (40.7769, -73.8740),
    "KATL": (33.6407, -84.4277),
    "EGLC": (51.5048, 0.0495),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PWSReading:
    """Single PWS observation."""
    lat: float
    lon: float
    temp_c: float
    label: str               # Station ID or grid label
    source: str = "UNKNOWN"  # "SYNOPTIC" or "OPEN_METEO"
    station_name: str = ""   # Human name (Synoptic only)
    distance_km: float = 0.0
    obs_time_utc: Optional[datetime] = None
    age_minutes: float = 0.0
    valid: bool = True
    qc_flag: Optional[str] = None


@dataclass
class PWSConsensus:
    """Cluster consensus result."""
    station_id: str
    median_temp_c: float
    median_temp_f: float
    mad: float                # Median Absolute Deviation
    support: int              # Number of valid stations
    total_queried: int
    readings: List[PWSReading]
    outliers: List[PWSReading]
    obs_time_utc: datetime
    drift_c: float = 0.0     # consensus - official METAR
    data_source: str = "UNKNOWN"  # "SYNOPTIC" or "OPEN_METEO"
    station_details: List[str] = field(default_factory=list)  # Detailed per-station log lines


# ---------------------------------------------------------------------------
# Synoptic Data API (primary source - real PWS)
# ---------------------------------------------------------------------------
async def _fetch_synoptic_cluster(
    station_id: str,
    radius_km: float = 40.0,  # Increased from 25km to get more stations
    max_stations: int = 50,   # Increased from 20 to get more stations
) -> Optional[List[PWSReading]]:
    """
    Fetch real PWS observations from Synoptic Data API.
    Returns list of PWSReading or None if unavailable.
    """
    token = os.environ.get("SYNOPTIC_API_TOKEN", "")
    if not token:
        logger.debug("No SYNOPTIC_API_TOKEN configured, skipping Synoptic source")
        return None

    coords = _STATION_COORDS.get(station_id)
    if not coords:
        logger.debug(f"No coordinates for {station_id}")
        return None

    lat, lon = coords
    now = datetime.now(timezone.utc)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.synopticdata.com/v2/stations/latest",
                params={
                    "token": token,
                    "radius": f"{lat},{lon},{radius_km}",
                    "vars": "air_temp",
                    "units": "temp|C",
                    "limit": max_stations,
                    "obtimezone": "UTC",
                    "status": "active",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Check API response code
        summary = data.get("SUMMARY", {})
        resp_code = summary.get("RESPONSE_CODE")
        if resp_code != 1:
            resp_msg = summary.get("RESPONSE_MESSAGE", "unknown")
            logger.warning(f"Synoptic API error for {station_id}: code={resp_code} msg={resp_msg}")
            return None

        stations_data = data.get("STATION", [])
        if not stations_data:
            logger.info(f"Synoptic returned 0 stations for {station_id}")
            return None

        readings = []
        skipped_age = 0
        skipped_parse = 0

        for stn in stations_data:
            try:
                stn_id = stn.get("STID", "?")
                stn_name = stn.get("NAME", "")
                stn_lat = float(stn.get("LATITUDE", 0))
                stn_lon = float(stn.get("LONGITUDE", 0))
                distance = float(stn.get("DISTANCE", 0))  # miles from Synoptic
                distance_km_val = distance * 1.60934  # convert miles to km

                # Get air_temp observation
                obs_data = stn.get("OBSERVATIONS", {})
                air_temp_entry = obs_data.get("air_temp_value_1")
                if not air_temp_entry:
                    skipped_parse += 1
                    continue

                temp_c = float(air_temp_entry.get("value", 0))
                obs_time_str = air_temp_entry.get("date_time", "")

                if obs_time_str:
                    obs_time = datetime.fromisoformat(
                        obs_time_str.replace("Z", "+00:00")
                    )
                else:
                    obs_time = now

                # Filter by age (max 90 minutes)
                age_min = (now - obs_time).total_seconds() / 60.0
                if age_min > 90:
                    skipped_age += 1
                    continue

                readings.append(PWSReading(
                    lat=stn_lat,
                    lon=stn_lon,
                    temp_c=temp_c,
                    label=stn_id,
                    source="SYNOPTIC",
                    station_name=stn_name,
                    distance_km=round(distance_km_val, 1),
                    obs_time_utc=obs_time,
                    age_minutes=round(age_min, 1),
                ))

            except Exception as e:
                skipped_parse += 1
                logger.debug(f"Synoptic station parse error: {e}")
                continue

        logger.info(
            f"Synoptic {station_id}: {len(readings)} valid stations "
            f"(skipped: {skipped_age} stale, {skipped_parse} parse errors) "
            f"from {len(stations_data)} total"
        )
        return readings if readings else None

    except httpx.HTTPStatusError as e:
        logger.warning(f"Synoptic HTTP error for {station_id}: {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Synoptic fetch failed for {station_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Open-Meteo grid fallback (pseudo-PWS)
# ---------------------------------------------------------------------------
async def _fetch_open_meteo_point(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    label: str,
) -> Optional[PWSReading]:
    """Fetch current temperature from Open-Meteo for a single grid point."""
    try:
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "current": "temperature_2m",
                "temperature_unit": "celsius",
            },
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        temp = data.get("current", {}).get("temperature_2m")
        if temp is not None:
            return PWSReading(
                lat=lat, lon=lon, temp_c=temp, label=label,
                source="OPEN_METEO", station_name=f"Grid {label}",
                obs_time_utc=datetime.now(timezone.utc),
            )
    except Exception as e:
        logger.debug(f"Open-Meteo point {label} failed: {e}")
    return None


async def _fetch_open_meteo_cluster(station_id: str) -> Optional[List[PWSReading]]:
    """Fetch Open-Meteo grid points as pseudo-PWS fallback."""
    cfg = _GRID_OFFSETS.get(station_id)
    if not cfg:
        return None

    base_lat = cfg["lat"]
    base_lon = cfg["lon"]
    offsets = cfg["offsets"]

    async with httpx.AsyncClient() as client:
        tasks = []
        for dlat, dlon in offsets:
            lat = base_lat + dlat
            lon = base_lon + dlon
            dist_km = ((dlat * 111) ** 2 + (dlon * 85) ** 2) ** 0.5
            direction = _bearing_label(dlat, dlon)
            label = f"{direction}_{dist_km:.0f}km"
            tasks.append(_fetch_open_meteo_point(client, lat, lon, label))

        results = await asyncio.gather(*tasks)

    readings = [r for r in results if r is not None]
    for r in readings:
        r.distance_km = round(((r.lat - base_lat) ** 2 * 111 ** 2 + (r.lon - base_lon) ** 2 * 85 ** 2) ** 0.5, 1)

    logger.info(f"Open-Meteo fallback {station_id}: {len(readings)}/{len(offsets)} grid points")
    return readings if readings else None


def _bearing_label(dlat: float, dlon: float) -> str:
    """Human label from lat/lon offset."""
    if abs(dlat) < 0.01 and abs(dlon) < 0.01:
        return "C"
    ns = "N" if dlat > 0.01 else ("S" if dlat < -0.01 else "")
    ew = "E" if dlon > 0.01 else ("W" if dlon < -0.01 else "")
    return ns + ew or "C"


# ---------------------------------------------------------------------------
# Main consensus pipeline
# ---------------------------------------------------------------------------
def _build_detail_lines(readings: List[PWSReading]) -> List[str]:
    """Build detailed per-station log lines for visibility."""
    lines = []
    for r in readings:
        temp_f = round((r.temp_c * 9 / 5) + 32, 1)
        age_str = f"age={r.age_minutes:.0f}min" if r.age_minutes > 0 else ""
        source_tag = "SYN" if r.source == "SYNOPTIC" else "OM"
        if r.source == "SYNOPTIC":
            lines.append(
                f"  [{source_tag}] [{r.label}] '{r.station_name}' "
                f"({r.lat:.4f}, {r.lon:.4f}) "
                f"dist={r.distance_km:.1f}km "
                f"temp={r.temp_c:.1f}C ({temp_f:.1f}F) "
                f"{age_str} "
                f"{'VALID' if r.valid else f'EXCLUDED: {r.qc_flag}'}"
            )
        else:
            lines.append(
                f"  [{source_tag}] [GRID {r.label}] "
                f"({r.lat:.4f}, {r.lon:.4f}) "
                f"dist={r.distance_km:.1f}km "
                f"temp={r.temp_c:.1f}C ({temp_f:.1f}F) "
                f"{'VALID' if r.valid else f'EXCLUDED: {r.qc_flag}'}"
            )
    return lines


async def fetch_pws_cluster(station_id: str, min_support: int = 3) -> Optional[PWSConsensus]:
    """
    Fetch PWS observations and compute robust consensus.

    Strategy:
      1. Fetch from BOTH Synoptic Data API (real PWS) AND Open-Meteo (grid points)
      2. Combine all readings from both sources
      3. Apply two-layer QC: hard rules + spatial MAD outlier detection
    """
    all_readings = []
    synoptic_count = 0
    openmeteo_count = 0

    # --- Fetch from Synoptic (real PWS stations) ---
    synoptic_readings = await _fetch_synoptic_cluster(station_id)
    if synoptic_readings:
        synoptic_count = len(synoptic_readings)
        all_readings.extend(synoptic_readings)
        logger.info(f"PWS {station_id}: Synoptic returned {synoptic_count} real stations")
    else:
        logger.info(f"PWS {station_id}: Synoptic unavailable or returned 0 stations")

    # --- Also fetch from Open-Meteo (grid points) ---
    openmeteo_readings = await _fetch_open_meteo_cluster(station_id)
    if openmeteo_readings:
        openmeteo_count = len(openmeteo_readings)
        all_readings.extend(openmeteo_readings)
        logger.info(f"PWS {station_id}: Open-Meteo returned {openmeteo_count} grid points")
    else:
        logger.info(f"PWS {station_id}: Open-Meteo unavailable")

    # Determine data source label
    if synoptic_count > 0 and openmeteo_count > 0:
        data_source = "SYNOPTIC+OPENMETEO"
    elif synoptic_count > 0:
        data_source = "SYNOPTIC"
    elif openmeteo_count > 0:
        data_source = "OPEN_METEO"
    else:
        logger.warning(f"PWS {station_id}: Both sources failed, no cluster data")
        return None

    readings = all_readings
    total_queried = len(readings)
    logger.info(f"PWS {station_id}: Combined {synoptic_count} Synoptic + {openmeteo_count} Open-Meteo = {total_queried} total")

    if len(readings) < min_support:
        logger.warning(f"PWS {station_id}: only {len(readings)}/{total_queried} readings, below min_support={min_support}")
        return None

    # --- QC Layer 1: Hard Rules ---
    from core.qc import QualityControl
    valid_readings = []
    for r in readings:
        qc = QualityControl.check_hard_rules(r.temp_c)
        if qc.is_valid:
            valid_readings.append(r)
        else:
            r.valid = False
            r.qc_flag = "HARD_RULE_FAIL"

    if len(valid_readings) < min_support:
        logger.warning(f"PWS {station_id}: only {len(valid_readings)} valid after hard rules QC")
        return None

    # --- QC Layer 2: Spatial MAD (applied on ALL combined readings) ---
    temps = [r.temp_c for r in valid_readings]
    median_val = statistics.median(temps)
    deviations = [abs(t - median_val) for t in temps]
    mad = statistics.median(deviations)

    # Mark outliers (> 3*MAD from median)
    outliers = []
    final_valid = []
    for r in valid_readings:
        if mad > 0 and abs(r.temp_c - median_val) > 3.0 * mad:
            r.valid = False
            r.qc_flag = "SPATIAL_OUTLIER_MAD"
            outliers.append(r)
        else:
            final_valid.append(r)

    # Recompute after removing outliers
    if len(final_valid) >= min_support:
        final_temps = [r.temp_c for r in final_valid]
        median_val = statistics.median(final_temps)
        mad = statistics.median([abs(t - median_val) for t in final_temps])
    elif len(final_valid) > 0:
        final_temps = [r.temp_c for r in final_valid]
        median_val = statistics.median(final_temps)
    else:
        return None

    median_f = round((median_val * 9 / 5) + 32, 1)

    # Build detailed station log
    detail_lines = _build_detail_lines(readings)

    # Log every station in detail
    logger.info(f"PWS {station_id} [{data_source}] — {len(readings)} stations queried:")
    for line in detail_lines:
        logger.info(line)
    if outliers:
        logger.info(f"  >> {len(outliers)} outliers removed by MAD filter")

    return PWSConsensus(
        station_id=station_id,
        median_temp_c=round(median_val, 2),
        median_temp_f=median_f,
        mad=round(mad, 3),
        support=len(final_valid),
        total_queried=total_queried,
        readings=final_valid,
        outliers=outliers,
        obs_time_utc=datetime.now(timezone.utc),
        data_source=data_source,
        station_details=detail_lines,
    )


async def fetch_and_publish_pws(station_id: str, official_temp_c: Optional[float] = None):
    """
    Full pipeline: fetch PWS cluster, compute consensus, QC, and publish to WorldState.

    Args:
        station_id: Station ICAO code
        official_temp_c: Current METAR temperature for drift calculation
    """
    consensus = await fetch_pws_cluster(station_id)
    if not consensus:
        return None

    # Compute drift vs official
    if official_temp_c is not None:
        consensus.drift_c = round(consensus.median_temp_c - official_temp_c, 2)

    # Publish to WorldState as AuxObs(is_aggregate=True)
    try:
        from core.models import AuxObs
        from core.world import get_world

        event = AuxObs(
            obs_time_utc=consensus.obs_time_utc,
            ingest_time_utc=datetime.now(timezone.utc),
            source=f"PWS_{consensus.data_source}",
            station_id=f"PWS_{station_id}",
            temp_c=consensus.median_temp_c,
            temp_f=consensus.median_temp_f,
            humidity=None,
            is_aggregate=True,
            support=consensus.support,
            drift=consensus.drift_c,
        )
        world = get_world()
        world.publish(event)

        # Store individual station details for UI breakdown
        from zoneinfo import ZoneInfo
        NYC = ZoneInfo("America/New_York")
        MAD = ZoneInfo("Europe/Madrid")

        all_readings = consensus.readings + consensus.outliers
        details = []
        for r in all_readings:
            temp_f = round((r.temp_c * 9 / 5) + 32, 1)
            # Convert obs_time to NYC and Madrid
            obs_nyc = r.obs_time_utc.astimezone(NYC).strftime("%Y-%m-%d %H:%M:%S") if r.obs_time_utc else None
            obs_mad = r.obs_time_utc.astimezone(MAD).strftime("%Y-%m-%d %H:%M:%S") if r.obs_time_utc else None
            obs_utc = r.obs_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ") if r.obs_time_utc else None

            details.append({
                "station_id": r.label,
                "station_name": r.station_name,
                "lat": round(r.lat, 4),
                "lon": round(r.lon, 4),
                "temp_c": round(r.temp_c, 1),
                "temp_f": temp_f,
                "distance_km": r.distance_km,
                "age_minutes": r.age_minutes,
                "obs_time_utc": obs_utc,
                "obs_time_nyc": obs_nyc,
                "obs_time_madrid": obs_mad,
                "source": r.source,
                "valid": r.valid,
                "qc_flag": r.qc_flag,
            })
        # Sort by distance
        details.sort(key=lambda x: x["distance_km"])
        world.set_pws_details(station_id, details)

        # Update QC with spatial info
        flags = []
        if consensus.outliers:
            flags.append(f"PWS_OUTLIERS_{len(consensus.outliers)}")
        if consensus.support < 5:
            flags.append("PWS_LOW_SUPPORT")
        if consensus.mad > 2.0:
            flags.append(f"PWS_HIGH_SPREAD_MAD_{consensus.mad:.1f}")
        if consensus.data_source == "OPEN_METEO":
            flags.append("PWS_FALLBACK_GRID")
        if flags:
            world.set_qc_state(f"PWS_{station_id}", "UNCERTAIN", flags)
        else:
            world.set_qc_state(f"PWS_{station_id}", "OK", [])

        logger.info(
            f"PWS {station_id} PUBLISHED: source={consensus.data_source} "
            f"median={consensus.median_temp_c}C ({consensus.median_temp_f}F) "
            f"MAD={consensus.mad:.2f} support={consensus.support}/{consensus.total_queried} "
            f"drift={consensus.drift_c:+.2f}C outliers={len(consensus.outliers)}"
        )

    except Exception as e:
        logger.error(f"PWS publish failed for {station_id}: {e}")
        try:
            from core.world import get_world
            get_world().record_error(f"PWS_{station_id}", str(e))
        except:
            pass

    return consensus


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    async def test():
        for station in ["KLGA", "KATL", "EGLC"]:
            print(f"\n{'='*60}")
            print(f"  PWS Cluster: {station}")
            print(f"{'='*60}")
            c = await fetch_pws_cluster(station)
            if c:
                print(f"\n  Source:    {c.data_source}")
                print(f"  Median:   {c.median_temp_c}C ({c.median_temp_f}F)")
                print(f"  MAD:      {c.mad}")
                print(f"  Support:  {c.support}/{c.total_queried}")
                print(f"  Outliers: {len(c.outliers)}")
                print(f"\n  Stations:")
                for line in c.station_details:
                    print(f"  {line}")
            else:
                print("  No consensus available")

    asyncio.run(test())
