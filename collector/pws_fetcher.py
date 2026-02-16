"""
Helios Weather Lab - PWS (Personal Weather Station) Fetcher
Phase 2, Section 2.4: Cluster + Consensus + QC

This implementation combines multiple public sources for dense nearby observations:
  1. Synoptic Data API (real station observations)
  2. NOAA MADIS Public surface service (CWOP/APRSWXNET + optional providers)
  3. Open-Meteo pseudo-station grid
  4. Wunderground PWS stations (weather.com)

Data flow:
  1. Query all enabled sources around target ICAO.
  2. Parse and normalize temperatures (Synoptic C, MADIS K -> C).
  3. Apply hard QC + spatial MAD outlier filtering.
  4. Publish robust consensus as AuxObs aggregate.
"""

import logging
import asyncio
import json
import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

logger = logging.getLogger("pws_fetcher")

# ---------------------------------------------------------------------------
# Config (with safe defaults)
# ---------------------------------------------------------------------------
try:
    from config import (
        SYNOPTIC_API_TOKEN,
        SYNOPTIC_BASE_URL,
        MADIS_CGI_URL,
        MADIS_TIMEOUT_SECONDS,
        MADIS_QC_LEVEL,
        MADIS_QC_TYPE,
        MADIS_ACCEPT_QCD,
        MADIS_PROVIDER_CONFIG,
        MADIS_RECWIN,
        PWS_SEARCH_CONFIG,
        PWS_MAX_AGE_MINUTES,
        PWS_MIN_SUPPORT,
        WUNDERGROUND_PWS_ENABLED,
        WUNDERGROUND_API_KEY,
        WUNDERGROUND_PWS_REGISTRY_PATH,
    )
except Exception:
    SYNOPTIC_API_TOKEN = ""
    SYNOPTIC_BASE_URL = "https://api.synopticdata.com/v2"
    MADIS_CGI_URL = "https://madis-data.ncep.noaa.gov/madisPublic/cgi-bin/madisXmlPublicDir"
    MADIS_TIMEOUT_SECONDS = 25.0
    MADIS_QC_LEVEL = 2
    MADIS_QC_TYPE = 0
    MADIS_ACCEPT_QCD = {"C", "S", "V"}
    MADIS_PROVIDER_CONFIG = {
        "KLGA": ["APRSWXNET", "MesoWest"],
        "KATL": ["APRSWXNET", "MesoWest"],
        "EGLC": ["APRSWXNET", "MesoWest"],
        "LTAC": ["APRSWXNET", "MesoWest"],
    }
    MADIS_RECWIN = 3
    PWS_SEARCH_CONFIG = {
        "KLGA": {"radius_km": 25, "max_stations": 20},
        "KATL": {"radius_km": 30, "max_stations": 20},
        "EGLC": {"radius_km": 20, "max_stations": 20},
        "LTAC": {"radius_km": 25, "max_stations": 20},
    }
    PWS_MAX_AGE_MINUTES = 90
    PWS_MIN_SUPPORT = 3
    WUNDERGROUND_PWS_ENABLED = True
    WUNDERGROUND_API_KEY = ""
    WUNDERGROUND_PWS_REGISTRY_PATH = "data/wu_pws_station_registry.json"

SYNOPTIC_TIMEOUT_SECONDS = 15.0
WUNDERGROUND_TIMEOUT_SECONDS = 8.0
_WUNDERGROUND_CURRENT_URL = "https://api.weather.com/v2/pws/observations/current"
_DEFAULT_WUNDERGROUND_STATION_IDS: Dict[str, List[str]] = {
    "KLGA": [
        "KNYNEWYO1591",
        "KNYNEWYO1552",
        "KNYNEWYO1974",
        "KNYNEWYO1771",
        "KNYTHEBR7",
        "KNYNEWYO1958",
        "KNYNEWYO1620",
        "KNYNEWYO1906",
        "KNYNEWYO1800",
        "KNYNEWYO1824",
    ],
    "KATL": [
        "KGAATLAN557",
        "KGAHAPEV1",
        "KGAEASTP2",
        "KGAFORES13",
        "KGAATLAN378",
        "KGARIVER19",
        "KGAATLAN1046",
        "KGAFORES15",
        "KGAATLAN972",
        "KGACONLE4",
    ],
}


# ---------------------------------------------------------------------------
# Station metadata
# ---------------------------------------------------------------------------
_STATION_COORDS: Dict[str, Tuple[float, float]] = {
    "KLGA": (40.7769, -73.8740),
    "KATL": (33.6407, -84.4277),
    "EGLC": (51.5048, 0.0495),
    "LTAC": (40.1281, 32.9951),
}

# Open-Meteo pseudo-station grid (for source diversity and diagnostics).
_GRID_OFFSETS: Dict[str, Dict[str, Any]] = {
    "KLGA": {
        "lat": 40.7769,
        "lon": -73.8740,
        "offsets": [
            (0.05, 0.0), (-0.05, 0.0), (0.0, 0.06), (0.0, -0.06),
            (0.10, 0.0), (-0.10, 0.0), (0.05, 0.06), (-0.05, -0.06),
            (0.15, 0.0), (0.0, -0.12),
        ],
    },
    "KATL": {
        "lat": 33.6407,
        "lon": -84.4277,
        "offsets": [
            (0.05, 0.0), (-0.05, 0.0), (0.0, 0.06), (0.0, -0.06),
            (0.10, 0.0), (-0.10, 0.0), (0.05, 0.06), (-0.05, -0.06),
            (0.15, 0.0), (0.0, -0.12),
        ],
    },
    "EGLC": {
        "lat": 51.5048,
        "lon": 0.0495,
        "offsets": [
            (0.04, 0.0), (-0.04, 0.0), (0.0, 0.06), (0.0, -0.06),
            (0.08, 0.0), (-0.08, 0.0), (0.04, 0.06), (-0.04, -0.06),
            (0.12, 0.0), (0.0, -0.10),
        ],
    },
    "LTAC": {
        "lat": 40.1281,
        "lon": 32.9951,
        "offsets": [
            (0.05, 0.0), (-0.05, 0.0), (0.0, 0.06), (0.0, -0.06),
            (0.10, 0.0), (-0.10, 0.0), (0.05, 0.06), (-0.05, -0.06),
            (0.15, 0.0), (0.0, -0.12),
        ],
    },
}

_STATION_STATE: Dict[str, str] = {
    "KLGA": "NY",
    "KATL": "GA",
    "EGLC": "",  # non-US
    "LTAC": "",  # non-US
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
    label: str
    source: str = "UNKNOWN"  # e.g. MADIS_APRSWXNET
    station_name: str = ""
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
    mad: float
    support: int
    total_queried: int
    readings: List[PWSReading]
    outliers: List[PWSReading]
    obs_time_utc: datetime
    drift_c: float = 0.0
    data_source: str = "UNKNOWN"
    station_details: List[str] = field(default_factory=list)
    # Precomputed QC metrics to feed HELIOS/UX comparisons.
    source_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km."""
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_obs_time_utc(value: str) -> Optional[datetime]:
    """
    Parse MADIS ObTime (e.g., 2026-02-08T01:19) as UTC.
    MADIS surface viewer timestamps are in GMT/UTC.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


def _kelvin_to_celsius(temp_k: float) -> float:
    """Convert Kelvin to Celsius."""
    return temp_k - 273.15


def _build_madis_query_params(
    station_id: str,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    providers: Sequence[str],
    max_age_minutes: int,
) -> List[Tuple[str, str]]:
    """
    Build MADIS CGI query parameters using a lat/lon bounding box.
    The endpoint accepts repeated keys (e.g., pvd, nvars).
    """
    lat_delta = radius_km / 111.0
    lon_scale = max(0.2, math.cos(math.radians(center_lat)))
    lon_delta = radius_km / (111.0 * lon_scale)

    latll = center_lat - lat_delta
    latur = center_lat + lat_delta
    lonll = center_lon - lon_delta
    lonur = center_lon + lon_delta

    params: List[Tuple[str, str]] = [
        ("time", "0"),
        ("minbck", f"-{int(max_age_minutes)}"),
        ("minfwd", "0"),
        ("recwin", str(MADIS_RECWIN)),
        ("timefilter", "0"),
        ("dfltrsel", "1"),  # lat/lon corners
        ("state", _STATION_STATE.get(station_id, "")),
        ("latll", f"{latll:.4f}"),
        ("lonll", f"{lonll:.4f}"),
        ("latur", f"{latur:.4f}"),
        ("lonur", f"{lonur:.4f}"),
        ("stanam", ""),
        ("stasel", "0"),
        ("pvdrsel", "1" if providers else "0"),
        ("varsel", "1"),  # selected variables
        ("nvars", "T"),  # temperature only
        ("qctype", str(MADIS_QC_TYPE)),  # MADIS QC
        ("qcsel", str(MADIS_QC_LEVEL)),
        ("xml", "1"),  # XML output
        ("csvmiss", "0"),
        ("rdr", ""),
    ]

    for provider in providers:
        params.append(("pvd", provider))

    return params


def _parse_madis_xml_records(
    xml_text: str,
    center_lat: float,
    center_lon: float,
    max_age_minutes: int,
    accepted_qcd: Iterable[str],
) -> List[PWSReading]:
    """
    Parse MADIS XML response into PWSReading entries.
    Keeps only valid temperature records with accepted QC descriptor.
    """
    if not xml_text:
        return []

    # MADIS returns an HTML error page for malformed requests.
    low = xml_text.lower()
    if "<html" in low and "misconfigured" in low:
        logger.warning("MADIS returned HTML error payload (misconfigured request)")
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"MADIS XML parse error: {e}")
        return []

    now = datetime.now(timezone.utc)
    allowed = {str(x).upper().strip() for x in accepted_qcd}
    dedupe: Dict[Tuple[str, str], PWSReading] = {}

    for rec in root.findall(".//record"):
        try:
            var_name = (rec.attrib.get("var") or "").upper().strip()
            if var_name not in {"V-T", "T"}:
                continue

            qcd = (rec.attrib.get("QCD") or "").upper().strip()
            if allowed and qcd and qcd not in allowed:
                continue

            provider = (rec.attrib.get("provider") or "MADIS").strip()
            shef_id = (rec.attrib.get("shef_id") or "?").strip()
            lat = float(rec.attrib.get("lat", "nan"))
            lon = float(rec.attrib.get("lon", "nan"))

            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue

            obs_time = _parse_obs_time_utc(rec.attrib.get("ObTime", ""))
            if not obs_time:
                continue

            age_min = (now - obs_time).total_seconds() / 60.0
            if age_min < 0 or age_min > max_age_minutes:
                continue

            temp_k = float(rec.attrib.get("data_value", "nan"))
            if not math.isfinite(temp_k):
                continue
            # Sanity bounds for near-surface air temperature in Kelvin.
            if temp_k < 180.0 or temp_k > 340.0:
                continue

            temp_c = _kelvin_to_celsius(temp_k)
            dist_km = _haversine_km(center_lat, center_lon, lat, lon)

            reading = PWSReading(
                lat=lat,
                lon=lon,
                temp_c=temp_c,
                label=shef_id,
                source=f"MADIS_{provider}",
                station_name=f"{provider} {shef_id}",
                distance_km=round(dist_km, 2),
                obs_time_utc=obs_time,
                age_minutes=round(age_min, 1),
            )

            key = (reading.source, reading.label)
            prev = dedupe.get(key)
            if prev is None:
                dedupe[key] = reading
            elif prev.obs_time_utc and reading.obs_time_utc and reading.obs_time_utc > prev.obs_time_utc:
                dedupe[key] = reading

        except Exception as e:
            logger.debug(f"MADIS record parse error: {e}")
            continue

    readings = list(dedupe.values())
    readings.sort(key=lambda r: (r.distance_km, r.age_minutes))
    return readings


# ---------------------------------------------------------------------------
# Synoptic source fetcher
# ---------------------------------------------------------------------------
async def _fetch_synoptic_cluster(
    station_id: str,
    radius_km: float,
    max_stations: int,
) -> Optional[List[PWSReading]]:
    """
    Fetch nearby observations from Synoptic Data API.
    """
    token = (SYNOPTIC_API_TOKEN or "").strip()
    if not token:
        logger.debug("No SYNOPTIC_API_TOKEN configured, skipping Synoptic source")
        return None

    coords = _STATION_COORDS.get(station_id)
    if not coords:
        logger.debug(f"No station coordinates for {station_id}")
        return None

    lat, lon = coords
    endpoint = f"{SYNOPTIC_BASE_URL}/stations/latest"

    try:
        async with httpx.AsyncClient(timeout=SYNOPTIC_TIMEOUT_SECONDS) as client:
            resp = await client.get(
                endpoint,
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
    except httpx.HTTPStatusError as e:
        logger.warning(f"Synoptic HTTP error for {station_id}: {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Synoptic request failed for {station_id}: {e}")
        return None

    summary = data.get("SUMMARY", {})
    response_code = summary.get("RESPONSE_CODE")
    try:
        response_code_val = int(response_code)
    except Exception:
        response_code_val = -1
    if response_code_val != 1:
        msg = summary.get("RESPONSE_MESSAGE", "unknown")
        logger.warning(f"Synoptic API error for {station_id}: code={response_code} msg={msg}")
        return None

    stations = data.get("STATION", [])
    if not stations:
        logger.info(f"Synoptic {station_id}: 0 valid readings")
        return None

    now = datetime.now(timezone.utc)
    dedupe: Dict[str, PWSReading] = {}
    skipped_no_temp = 0
    skipped_stale = 0
    skipped_parse = 0

    for stn in stations:
        try:
            stn_id = str(stn.get("STID", "?")).strip()
            stn_name = str(stn.get("NAME", "")).strip()
            stn_lat = float(stn.get("LATITUDE", "nan"))
            stn_lon = float(stn.get("LONGITUDE", "nan"))

            if not (math.isfinite(stn_lat) and math.isfinite(stn_lon)):
                skipped_parse += 1
                continue

            obs_data = stn.get("OBSERVATIONS", {}) or {}
            temp_entry = obs_data.get("air_temp_value_1")
            if not temp_entry:
                skipped_no_temp += 1
                continue

            if isinstance(temp_entry, dict):
                temp_raw = temp_entry.get("value")
                obs_time_raw = str(temp_entry.get("date_time", ""))
            else:
                temp_raw = temp_entry
                obs_time_raw = str(stn.get("DATE_TIME", ""))

            temp_c = float(temp_raw)
            if not math.isfinite(temp_c):
                skipped_parse += 1
                continue

            obs_time = _parse_obs_time_utc(obs_time_raw) or now
            age_min = (now - obs_time).total_seconds() / 60.0
            if age_min < -5 or age_min > PWS_MAX_AGE_MINUTES:
                skipped_stale += 1
                continue

            distance_km = _haversine_km(lat, lon, stn_lat, stn_lon)
            try:
                # Synoptic distance field is miles from query center.
                distance_km = float(stn.get("DISTANCE", 0.0)) * 1.60934
            except Exception:
                pass

            reading = PWSReading(
                lat=stn_lat,
                lon=stn_lon,
                temp_c=temp_c,
                label=stn_id,
                source="SYNOPTIC",
                station_name=stn_name,
                distance_km=round(distance_km, 2),
                obs_time_utc=obs_time,
                age_minutes=round(age_min, 1),
            )

            prev = dedupe.get(reading.label)
            if prev is None:
                dedupe[reading.label] = reading
            elif prev.obs_time_utc and reading.obs_time_utc and reading.obs_time_utc > prev.obs_time_utc:
                dedupe[reading.label] = reading

        except Exception as e:
            skipped_parse += 1
            logger.debug(f"Synoptic station parse error: {e}")
            continue

    readings = list(dedupe.values())
    readings.sort(key=lambda r: (r.distance_km, r.age_minutes))
    logger.info(
        f"Synoptic {station_id}: {len(readings)} readings "
        f"(skipped no_temp={skipped_no_temp}, stale={skipped_stale}, parse={skipped_parse}; "
        f"radius={radius_km}km)"
    )
    return readings or None


# ---------------------------------------------------------------------------
# Open-Meteo source fetcher (pseudo-stations)
# ---------------------------------------------------------------------------
def _bearing_label(dlat: float, dlon: float) -> str:
    if abs(dlat) < 0.01 and abs(dlon) < 0.01:
        return "C"
    ns = "N" if dlat > 0.01 else ("S" if dlat < -0.01 else "")
    ew = "E" if dlon > 0.01 else ("W" if dlon < -0.01 else "")
    return ns + ew or "C"


def _parse_open_meteo_obs_time(value: Any) -> Optional[datetime]:
    """Parse Open-Meteo current.time into UTC-aware datetime."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _normalize_open_meteo_payload(payload: Any) -> List[Dict[str, Any]]:
    """Normalize Open-Meteo response to a list of point payload dictionaries."""
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _build_open_meteo_grid_points(station_id: str) -> Optional[Tuple[float, float, List[Tuple[float, float, str]]]]:
    """Build absolute grid points for Open-Meteo pseudo-stations."""
    cfg = _GRID_OFFSETS.get(station_id)
    if not cfg:
        return None

    base_lat = float(cfg["lat"])
    base_lon = float(cfg["lon"])
    offsets: List[Tuple[float, float]] = list(cfg["offsets"])

    points: List[Tuple[float, float, str]] = []
    for dlat, dlon in offsets:
        lat = base_lat + dlat
        lon = base_lon + dlon
        dist_km = ((dlat * 111) ** 2 + (dlon * 85) ** 2) ** 0.5
        direction = _bearing_label(dlat, dlon)
        label = f"{direction}_{dist_km:.0f}km"
        points.append((lat, lon, label))

    return base_lat, base_lon, points


def _build_open_meteo_reading(
    *,
    base_lat: float,
    base_lon: float,
    lat: float,
    lon: float,
    label: str,
    payload: Dict[str, Any],
) -> Optional[PWSReading]:
    """Build PWSReading from one Open-Meteo point payload."""
    current = payload.get("current") or {}
    temp = current.get("temperature_2m")
    if temp is None:
        return None

    now_utc = datetime.now(timezone.utc)
    obs_time_utc = _parse_open_meteo_obs_time(current.get("time")) or now_utc
    age_minutes = max(0.0, (now_utc - obs_time_utc).total_seconds() / 60.0)

    return PWSReading(
        lat=lat,
        lon=lon,
        temp_c=float(temp),
        label=label,
        source="OPEN_METEO",
        station_name=f"Grid {label}",
        distance_km=round(_haversine_km(base_lat, base_lon, lat, lon), 1),
        obs_time_utc=obs_time_utc,
        age_minutes=round(age_minutes, 1),
    )


async def _fetch_open_meteo_point(
    client: httpx.AsyncClient,
    base_lat: float,
    base_lon: float,
    lat: float,
    lon: float,
    label: str,
) -> Optional[PWSReading]:
    try:
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "current": "temperature_2m",
                "temperature_unit": "celsius",
                "timezone": "UTC",
            },
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return _build_open_meteo_reading(
            base_lat=base_lat,
            base_lon=base_lon,
            lat=lat,
            lon=lon,
            label=label,
            payload=data if isinstance(data, dict) else {},
        )
    except Exception as e:
        logger.debug(f"Open-Meteo point {label} failed: {e}")
        return None


async def _fetch_open_meteo_cluster(station_id: str) -> Optional[List[PWSReading]]:
    grid = _build_open_meteo_grid_points(station_id)
    if not grid:
        return None

    base_lat, base_lon, points = grid
    if not points:
        return None

    readings: List[PWSReading] = []

    # Prefer single multi-point call (more robust than N concurrent point calls).
    try:
        latitudes = ",".join(f"{lat:.4f}" for lat, _, _ in points)
        longitudes = ",".join(f"{lon:.4f}" for _, lon, _ in points)
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitudes,
                    "longitude": longitudes,
                    "current": "temperature_2m",
                    "temperature_unit": "celsius",
                    "timezone": "UTC",
                },
            )
            resp.raise_for_status()
            payload = resp.json()

        entries = _normalize_open_meteo_payload(payload)
        if len(entries) != len(points):
            logger.warning(
                "Open-Meteo %s: batch returned %s points, expected %s",
                station_id,
                len(entries),
                len(points),
            )

        for idx, (lat, lon, label) in enumerate(points):
            if idx >= len(entries):
                break
            reading = _build_open_meteo_reading(
                base_lat=base_lat,
                base_lon=base_lon,
                lat=lat,
                lon=lon,
                label=label,
                payload=entries[idx],
            )
            if reading:
                readings.append(reading)

    except Exception as e:
        logger.warning(f"Open-Meteo batch request failed for {station_id}: {e}")

    # Fallback to point-by-point fetch if batch returns no usable readings.
    if not readings:
        async with httpx.AsyncClient() as client:
            tasks = [
                _fetch_open_meteo_point(client, base_lat, base_lon, lat, lon, label)
                for lat, lon, label in points
            ]
            results = await asyncio.gather(*tasks)
        readings = [r for r in results if r is not None]

    logger.info(f"Open-Meteo {station_id}: {len(readings)}/{len(points)} grid readings")
    return readings if readings else None


# ---------------------------------------------------------------------------
# Wunderground source fetcher (real PWS stations)
# ---------------------------------------------------------------------------
def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _station_distance_km(
    center_lat: float,
    center_lon: float,
    lat: Optional[float],
    lon: Optional[float],
    fallback: Optional[float] = None,
) -> float:
    if fallback is not None:
        return round(float(fallback), 1)
    if lat is None or lon is None:
        return 0.0
    return round(_haversine_km(center_lat, center_lon, lat, lon), 1)


def _dedupe_wunderground_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedupe: Dict[str, Dict[str, Any]] = {}
    for row in candidates:
        sid = str(row.get("station_id") or "").strip().upper()
        if not sid:
            continue
        prev = dedupe.get(sid)
        if prev is None:
            dedupe[sid] = row
            continue
        prev_dist = _float_or_none(prev.get("distance_km"))
        curr_dist = _float_or_none(row.get("distance_km"))
        if prev_dist is None and curr_dist is not None:
            dedupe[sid] = row
        elif prev_dist is not None and curr_dist is not None and curr_dist < prev_dist:
            dedupe[sid] = row
    return list(dedupe.values())


def _load_wunderground_candidates(station_id: str, max_stations: int) -> List[Dict[str, Any]]:
    station_id = station_id.upper()
    candidates: List[Dict[str, Any]] = []
    registry_path = Path(str(WUNDERGROUND_PWS_REGISTRY_PATH or "").strip() or "data/wu_pws_station_registry.json")

    if registry_path.exists():
        try:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            station_payload = payload.get(station_id, {}) if isinstance(payload, dict) else {}
            stations = station_payload.get("stations") if isinstance(station_payload, dict) else None
            if isinstance(stations, list):
                for item in stations:
                    if not isinstance(item, dict):
                        continue
                    sid = str(item.get("station_id") or "").strip().upper()
                    if not sid:
                        continue
                    candidates.append(
                        {
                            "station_id": sid,
                            "station_name": str(item.get("station_name") or ""),
                            "neighborhood": str(item.get("neighborhood") or ""),
                            "lat": _float_or_none(item.get("latitude")),
                            "lon": _float_or_none(item.get("longitude")),
                            "distance_km": _float_or_none(item.get("distance_km")),
                        }
                    )
            station_ids = station_payload.get("station_ids") if isinstance(station_payload, dict) else None
            if isinstance(station_ids, list):
                for sid in station_ids:
                    sid_str = str(sid or "").strip().upper()
                    if not sid_str:
                        continue
                    candidates.append({"station_id": sid_str})
        except Exception as e:
            logger.warning(f"Wunderground registry read failed ({registry_path}): {e}")

    if not candidates:
        for sid in _DEFAULT_WUNDERGROUND_STATION_IDS.get(station_id, []):
            candidates.append({"station_id": sid})

    deduped = _dedupe_wunderground_candidates(candidates)
    if max_stations > 0:
        deduped = deduped[:max_stations]
    return deduped


def _get_wunderground_api_key() -> str:
    return str(WUNDERGROUND_API_KEY or "").strip()


async def _fetch_wunderground_point(
    client: httpx.AsyncClient,
    *,
    center_lat: float,
    center_lon: float,
    api_key: str,
    candidate: Dict[str, Any],
) -> Optional[PWSReading]:
    station_id = str(candidate.get("station_id") or "").strip().upper()
    if not station_id:
        return None

    try:
        resp = await client.get(
            _WUNDERGROUND_CURRENT_URL,
            params={
                "stationId": station_id,
                "format": "json",
                "units": "e",
                "apiKey": api_key,
            },
            timeout=WUNDERGROUND_TIMEOUT_SECONDS,
        )
        if resp.status_code == 204:
            return None
        if resp.status_code != 200:
            logger.debug(f"Wunderground station {station_id} returned {resp.status_code}")
            return None

        payload = resp.json()
        observations = payload.get("observations") or []
        if not observations:
            return None
        obs = observations[0]
        imperial = obs.get("imperial") or {}

        temp_f = _float_or_none(imperial.get("temp"))
        if temp_f is None:
            temp_f = _float_or_none(obs.get("temp"))
        if temp_f is None:
            return None
        temp_c = (temp_f - 32.0) * 5.0 / 9.0

        obs_time = _parse_obs_time_utc(str(obs.get("obsTimeUtc") or ""))
        now_utc = datetime.now(timezone.utc)
        age_min = 0.0
        if obs_time:
            age_min = (now_utc - obs_time).total_seconds() / 60.0
            if age_min < -5 or age_min > PWS_MAX_AGE_MINUTES:
                return None
        else:
            obs_time = now_utc

        lat = _float_or_none(obs.get("lat"))
        lon = _float_or_none(obs.get("lon"))
        if lat is None:
            lat = _float_or_none(candidate.get("lat"))
        if lon is None:
            lon = _float_or_none(candidate.get("lon"))
        if lat is None:
            lat = center_lat
        if lon is None:
            lon = center_lon

        distance_km = _station_distance_km(
            center_lat,
            center_lon,
            lat,
            lon,
            fallback=_float_or_none(candidate.get("distance_km")),
        )
        neighborhood = str(obs.get("neighborhood") or "").strip()
        station_name = neighborhood or str(candidate.get("station_name") or "") or f"WU {station_id}"

        return PWSReading(
            lat=lat,
            lon=lon,
            temp_c=temp_c,
            label=station_id,
            source="WUNDERGROUND",
            station_name=station_name,
            distance_km=distance_km,
            obs_time_utc=obs_time,
            age_minutes=round(max(0.0, age_min), 1),
        )
    except Exception as e:
        logger.debug(f"Wunderground station {station_id} failed: {e}")
        return None


async def _fetch_wunderground_cluster(station_id: str, max_stations: int) -> Optional[List[PWSReading]]:
    if not WUNDERGROUND_PWS_ENABLED:
        return None

    coords = _STATION_COORDS.get(station_id)
    if not coords:
        return None
    center_lat, center_lon = coords

    candidates = _load_wunderground_candidates(station_id, max_stations=max_stations)
    if not candidates:
        logger.info(f"Wunderground {station_id}: no station candidates configured")
        return None

    api_key = _get_wunderground_api_key()
    if not api_key:
        logger.info(f"Wunderground {station_id}: API key not configured")
        return None

    limits = httpx.Limits(max_connections=max(16, len(candidates) * 2), max_keepalive_connections=8)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            _fetch_wunderground_point(
                client,
                center_lat=center_lat,
                center_lon=center_lon,
                api_key=api_key,
                candidate=candidate,
            )
            for candidate in candidates
        ]
        results = await asyncio.gather(*tasks)

    readings = [r for r in results if r is not None]
    readings.sort(key=lambda r: (r.distance_km, r.age_minutes))
    logger.info(f"Wunderground {station_id}: {len(readings)}/{len(candidates)} readings")
    return readings if readings else None


# ---------------------------------------------------------------------------
# MADIS source fetcher
# ---------------------------------------------------------------------------
async def _fetch_madis_cluster(
    station_id: str,
    radius_km: float,
    max_stations: int,
    providers: Sequence[str],
) -> Optional[List[PWSReading]]:
    """
    Fetch nearby PWS-like observations from MADIS public endpoint.
    """
    coords = _STATION_COORDS.get(station_id)
    if not coords:
        logger.debug(f"No station coordinates for {station_id}")
        return None

    lat, lon = coords
    params = _build_madis_query_params(
        station_id=station_id,
        center_lat=lat,
        center_lon=lon,
        radius_km=radius_km,
        providers=providers,
        max_age_minutes=PWS_MAX_AGE_MINUTES,
    )

    try:
        async with httpx.AsyncClient(timeout=MADIS_TIMEOUT_SECONDS) as client:
            resp = await client.get(MADIS_CGI_URL, params=params)
            resp.raise_for_status()
            raw = resp.text
    except httpx.HTTPStatusError as e:
        logger.warning(f"MADIS HTTP error for {station_id}: {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"MADIS request failed for {station_id}: {e}")
        return None

    readings = _parse_madis_xml_records(
        xml_text=raw,
        center_lat=lat,
        center_lon=lon,
        max_age_minutes=PWS_MAX_AGE_MINUTES,
        accepted_qcd=MADIS_ACCEPT_QCD,
    )

    if not readings:
        logger.info(f"MADIS {station_id}: 0 valid readings")
        return None

    if max_stations > 0 and len(readings) > max_stations:
        readings = readings[:max_stations]

    provider_counts: Dict[str, int] = {}
    for r in readings:
        provider = r.source.replace("MADIS_", "", 1)
        provider_counts[provider] = provider_counts.get(provider, 0) + 1

    provider_log = ", ".join(f"{k}:{v}" for k, v in sorted(provider_counts.items()))
    logger.info(
        f"MADIS {station_id}: {len(readings)} readings "
        f"(providers: {provider_log}; radius={radius_km}km)"
    )

    return readings


# ---------------------------------------------------------------------------
# Consensus pipeline
# ---------------------------------------------------------------------------
def _build_detail_lines(readings: List[PWSReading]) -> List[str]:
    """Build detailed per-station lines for logs and diagnostics."""
    lines: List[str] = []
    for r in readings:
        temp_f = round((r.temp_c * 9 / 5) + 32, 1)
        source_label = r.source
        if r.source.startswith("MADIS_"):
            provider = r.source.replace("MADIS_", "", 1)
            source_label = f"MADIS/{provider}"
        lines.append(
            f"  [{source_label}] [{r.label}] "
            f"({r.lat:.4f}, {r.lon:.4f}) "
            f"dist={r.distance_km:.1f}km "
            f"temp={r.temp_c:.1f}C ({temp_f:.1f}F) "
            f"age={r.age_minutes:.0f}min "
            f"{'VALID' if r.valid else f'EXCLUDED: {r.qc_flag}'}"
        )
    return lines


def _compute_qc_metrics(readings: List[PWSReading], min_support: int = 1) -> Optional[Dict[str, float]]:
    """
    Compute consensus metrics for a subset using the same QC policy:
    hard-rule filter + MAD outlier removal.
    """
    if not readings:
        return None

    from core.qc import QualityControl

    valid = [r for r in readings if QualityControl.check_hard_rules(r.temp_c).is_valid]
    if len(valid) < min_support:
        return None

    temps = [r.temp_c for r in valid]
    median_val = statistics.median(temps)
    deviations = [abs(t - median_val) for t in temps]
    mad = statistics.median(deviations)

    final = [r for r in valid if mad == 0 or abs(r.temp_c - median_val) <= 3.0 * mad]
    if len(final) < min_support:
        return None

    final_temps = [r.temp_c for r in final]
    final_median_c = statistics.median(final_temps)
    final_mad = 0.0
    if len(final_temps) > 1:
        final_mad = statistics.median([abs(t - final_median_c) for t in final_temps])

    final_median_f = round((final_median_c * 9 / 5) + 32, 1)
    return {
        "median_temp_c": round(final_median_c, 2),
        "median_temp_f": final_median_f,
        "mad": round(final_mad, 3),
        "support": len(final),
        "total_queried": len(readings),
    }


def _build_source_metrics(readings: List[PWSReading], min_support: int) -> Dict[str, Dict[str, float]]:
    """
    Build canonical HELIOS-facing metrics for source subsets.
    Keys:
      - all_sources_qc
      - synoptic_only_qc
      - madis_only_qc
      - open_meteo_only_qc
      - wunderground_only_qc
    """
    metrics: Dict[str, Dict[str, float]] = {}

    def _put(key: str, subset: List[PWSReading], subset_min_support: int) -> None:
        data = _compute_qc_metrics(subset, min_support=subset_min_support)
        if data:
            metrics[key] = data

    _put("all_sources_qc", readings, min_support)
    _put("synoptic_only_qc", [r for r in readings if r.source == "SYNOPTIC"], 1)
    _put("madis_only_qc", [r for r in readings if r.source.startswith("MADIS_")], 1)
    _put("open_meteo_only_qc", [r for r in readings if r.source == "OPEN_METEO"], 1)
    _put("wunderground_only_qc", [r for r in readings if r.source == "WUNDERGROUND"], 1)
    return metrics


def _resolve_station_source(readings: List[PWSReading]) -> str:
    has_synoptic = any(r.source == "SYNOPTIC" for r in readings)
    has_open_meteo = any(r.source == "OPEN_METEO" for r in readings)
    has_wunderground = any(r.source == "WUNDERGROUND" for r in readings)
    providers = sorted({r.source.replace("MADIS_", "", 1) for r in readings if r.source.startswith("MADIS_")})
    labels: List[str] = []
    if has_synoptic:
        labels.append("SYNOPTIC")
    if providers:
        labels.append(f"MADIS_{providers[0]}" if len(providers) == 1 else "MADIS_MULTI")
    if has_open_meteo:
        labels.append("OPEN_METEO")
    if has_wunderground:
        labels.append("WUNDERGROUND")
    if not labels:
        return "UNKNOWN"
    return "+".join(labels)


async def fetch_pws_cluster(station_id: str, min_support: int = PWS_MIN_SUPPORT) -> Optional[PWSConsensus]:
    """
    Fetch nearby observations from Synoptic + MADIS + Open-Meteo + Wunderground and compute robust consensus.

    Strategy:
      1. Pull readings from Synoptic (real stations), MADIS (CWOP/APRSWXNET + optional providers),
         Open-Meteo pseudo-stations, and Wunderground PWS stations
      2. Merge source readings
      3. Hard-rule QC filter
      4. Spatial MAD outlier filter
      5. Median consensus
    """
    station_id = station_id.upper()
    cfg = PWS_SEARCH_CONFIG.get(station_id, {"radius_km": 25, "max_stations": 20})
    radius_km = float(cfg.get("radius_km", 25))
    max_stations = int(cfg.get("max_stations", 20))
    providers = MADIS_PROVIDER_CONFIG.get(station_id, ["APRSWXNET"])

    synoptic_readings = await _fetch_synoptic_cluster(
        station_id=station_id,
        radius_km=radius_km,
        max_stations=max_stations,
    )
    madis_readings = await _fetch_madis_cluster(
        station_id=station_id,
        radius_km=radius_km,
        max_stations=max_stations,
        providers=providers,
    )
    open_meteo_readings = await _fetch_open_meteo_cluster(station_id)
    wunderground_readings = await _fetch_wunderground_cluster(station_id, max_stations=max_stations)

    readings: List[PWSReading] = []
    if synoptic_readings:
        readings.extend(synoptic_readings)
    if madis_readings:
        readings.extend(madis_readings)
    if open_meteo_readings:
        readings.extend(open_meteo_readings)
    if wunderground_readings:
        readings.extend(wunderground_readings)

    if not readings:
        logger.warning(f"PWS {station_id}: Synoptic, MADIS, Open-Meteo and Wunderground returned no data")
        return None

    synoptic_count = len(synoptic_readings) if synoptic_readings else 0
    madis_count = len(madis_readings) if madis_readings else 0
    open_meteo_count = len(open_meteo_readings) if open_meteo_readings else 0
    wunderground_count = len(wunderground_readings) if wunderground_readings else 0
    logger.info(
        "PWS %s: source mix Synoptic=%s, MADIS=%s, Open-Meteo=%s, Wunderground=%s, total=%s",
        station_id,
        synoptic_count,
        madis_count,
        open_meteo_count,
        wunderground_count,
        len(readings),
    )

    total_queried = len(readings)
    if total_queried < min_support:
        logger.warning(f"PWS {station_id}: {total_queried} readings, below min_support={min_support}")
        return None

    # QC Layer 1: Hard rules
    from core.qc import QualityControl

    valid_readings: List[PWSReading] = []
    for r in readings:
        qc = QualityControl.check_hard_rules(r.temp_c)
        if qc.is_valid:
            valid_readings.append(r)
        else:
            r.valid = False
            r.qc_flag = "HARD_RULE_FAIL"

    if len(valid_readings) < min_support:
        logger.warning(f"PWS {station_id}: only {len(valid_readings)} valid after hard-rule QC")
        return None

    # QC Layer 2: Spatial MAD outlier filtering
    temps = [r.temp_c for r in valid_readings]
    median_val = statistics.median(temps)
    deviations = [abs(t - median_val) for t in temps]
    mad = statistics.median(deviations)

    outliers: List[PWSReading] = []
    final_valid: List[PWSReading] = []
    for r in valid_readings:
        if mad > 0 and abs(r.temp_c - median_val) > 3.0 * mad:
            r.valid = False
            r.qc_flag = "SPATIAL_OUTLIER_MAD"
            outliers.append(r)
        else:
            final_valid.append(r)

    if not final_valid:
        logger.warning(f"PWS {station_id}: no readings left after MAD filtering")
        return None

    final_temps = [r.temp_c for r in final_valid]
    median_val = statistics.median(final_temps)
    if len(final_temps) > 1:
        mad = statistics.median([abs(t - median_val) for t in final_temps])

    if len(final_valid) < min_support:
        logger.warning(
            f"PWS {station_id}: only {len(final_valid)} valid after MAD, below min_support={min_support}"
        )
        return None

    median_f = round((median_val * 9 / 5) + 32, 1)
    data_source = _resolve_station_source(final_valid + outliers)
    detail_lines = _build_detail_lines(readings)

    logger.info(f"PWS {station_id} [{data_source}] - {len(readings)} stations queried")
    for line in detail_lines:
        logger.info(line)
    if outliers:
        logger.info(f"  >> {len(outliers)} outliers removed by MAD filter")

    source_metrics = _build_source_metrics(readings, min_support=min_support)

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
        source_metrics=source_metrics,
    )


async def fetch_and_publish_pws(station_id: str, official_temp_c: Optional[float] = None):
    """
    Full pipeline: fetch PWS cluster, compute consensus, QC, and publish to WorldState.
    """
    consensus = await fetch_pws_cluster(station_id)
    if not consensus:
        return None

    # Compute drift vs official
    if official_temp_c is not None:
        consensus.drift_c = round(consensus.median_temp_c - official_temp_c, 2)

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

        nyc_tz = ZoneInfo("America/New_York")
        mad_tz = ZoneInfo("Europe/Madrid")

        all_readings = consensus.readings + consensus.outliers
        details = []
        for r in all_readings:
            temp_f = round((r.temp_c * 9 / 5) + 32, 1)
            obs_nyc = r.obs_time_utc.astimezone(nyc_tz).strftime("%Y-%m-%d %H:%M:%S") if r.obs_time_utc else None
            obs_mad = r.obs_time_utc.astimezone(mad_tz).strftime("%Y-%m-%d %H:%M:%S") if r.obs_time_utc else None
            obs_utc = r.obs_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ") if r.obs_time_utc else None

            details.append(
                {
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
                }
            )

        details.sort(key=lambda x: x["distance_km"])
        world.set_pws_details(station_id, details)

        # Store HELIOS-facing PWS metrics (dual view: Synoptic-only vs all sources QC).
        metrics_payload: Dict[str, Dict[str, float]] = {}
        for key, value in (consensus.source_metrics or {}).items():
            metric = dict(value)
            if official_temp_c is not None:
                metric["drift_c"] = round(metric["median_temp_c"] - official_temp_c, 2)
            metrics_payload[key] = metric
        world.set_pws_metrics(station_id, metrics_payload)

        # Update QC state
        flags = []
        if consensus.outliers:
            flags.append(f"PWS_OUTLIERS_{len(consensus.outliers)}")
        if consensus.support < 5:
            flags.append("PWS_LOW_SUPPORT")
        if consensus.mad > 2.0:
            flags.append(f"PWS_HIGH_SPREAD_MAD_{consensus.mad:.1f}")
        has_madis = any(r.source.startswith("MADIS_") for r in consensus.readings)
        if has_madis and not any(r.source == "MADIS_APRSWXNET" for r in consensus.readings):
            flags.append("PWS_NO_CWOP")
        if consensus.readings and all(r.source == "OPEN_METEO" for r in consensus.readings):
            flags.append("PWS_FALLBACK_GRID")
        if consensus.readings and all(r.source == "WUNDERGROUND" for r in consensus.readings):
            flags.append("PWS_WUNDERGROUND_ONLY")

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
        except Exception:
            pass

    return consensus


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    import asyncio

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    async def test():
        for station in ["KLGA", "KATL", "EGLC", "LTAC"]:
            print(f"\n{'=' * 60}")
            print(f"  PWS Cluster (Synoptic + MADIS + Open-Meteo + Wunderground): {station}")
            print(f"{'=' * 60}")
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
