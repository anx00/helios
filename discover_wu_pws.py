"""
Discover and validate Weather Underground PWS stations near a target station.

Flow:
  1) Discover nearby PWS IDs using weather.com location service.
  2) Validate each ID by requesting current PWS observation.
  3) Persist only valid stations to a reusable JSON registry.

Example:
  python discover_wu_pws.py --station KLGA
  python discover_wu_pws.py --stations KLGA,KATL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from config import STATIONS


DISCOVERY_URL = "https://api.weather.com/v3/location/near"
PWS_CURRENT_URL = "https://api.weather.com/v2/pws/observations/current"
DEFAULT_REGISTRY_PATH = Path("data/wu_pws_station_registry.json")


@dataclass
class CandidateStation:
    station_id: str
    station_name: str = ""
    qc_status: Optional[int] = None
    update_time_utc: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    distance_km: Optional[float] = None


@dataclass
class ValidStation:
    station_id: str
    station_name: str
    qc_status: Optional[int]
    latitude: Optional[float]
    longitude: Optional[float]
    distance_km: Optional[float]
    obs_time_utc: Optional[str]
    temp_f: Optional[float]
    humidity: Optional[float]
    neighborhood: str


def _safe_index(values: Sequence[Any], idx: int, default: Any = None) -> Any:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return default
    if idx < 0 or idx >= len(values):
        return default
    return values[idx]


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _load_env_value_from_dotenv(env_name: str, env_path: Path = Path(".env")) -> Optional[str]:
    if not env_path.exists():
        return None
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == env_name:
                return v.strip().strip("\"").strip("'")
    except Exception:
        return None
    return None


def _resolve_api_key(cli_key: Optional[str]) -> str:
    if cli_key:
        return cli_key
    for name in ("WUNDERGROUND_API_KEY", "WU_API_KEY"):
        val = os.environ.get(name)
        if val:
            return val
        val = _load_env_value_from_dotenv(name)
        if val:
            return val
    raise ValueError(
        "Missing WU API key. Set WUNDERGROUND_API_KEY (or WU_API_KEY) in environment or .env, "
        "or pass --api-key."
    )


def _resolve_target(station_id: str, lat: Optional[float], lon: Optional[float]) -> Tuple[float, float]:
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    station = STATIONS.get(station_id.upper())
    if not station:
        raise ValueError(f"Unknown station {station_id}. Pass --lat and --lon.")
    return float(station.latitude), float(station.longitude)


async def discover_candidates(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    lat: float,
    lon: float,
    limit: int,
) -> List[CandidateStation]:
    params = {
        "geocode": f"{lat},{lon}",
        "product": "pws",
        "format": "json",
        "apiKey": api_key,
    }
    resp = await client.get(DISCOVERY_URL, params=params)
    resp.raise_for_status()
    payload = resp.json()
    loc = payload.get("location") or {}

    ids = loc.get("stationId") or []
    names = loc.get("stationName") or []
    qcs = loc.get("qcStatus") or []
    updates = loc.get("updateTimeUtc") or []
    lats = loc.get("latitude") or []
    lons = loc.get("longitude") or []

    candidates: List[CandidateStation] = []
    for idx, station_id in enumerate(ids):
        station_id = str(station_id).strip()
        if not station_id:
            continue
        cand_lat = _to_float(_safe_index(lats, idx))
        cand_lon = _to_float(_safe_index(lons, idx))
        dist = None
        if cand_lat is not None and cand_lon is not None:
            dist = round(_haversine_km(lat, lon, cand_lat, cand_lon), 3)
        candidates.append(
            CandidateStation(
                station_id=station_id,
                station_name=str(_safe_index(names, idx, "")),
                qc_status=_to_int(_safe_index(qcs, idx)),
                update_time_utc=_to_int(_safe_index(updates, idx)),
                latitude=cand_lat,
                longitude=cand_lon,
                distance_km=dist,
            )
        )

    if limit > 0:
        candidates = candidates[:limit]
    return candidates


async def fetch_current_observation(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    candidate: CandidateStation,
) -> Optional[ValidStation]:
    params = {
        "stationId": candidate.station_id,
        "format": "json",
        "units": "e",
        "apiKey": api_key,
    }
    resp = await client.get(PWS_CURRENT_URL, params=params)
    if resp.status_code == 204:
        return None
    if resp.status_code != 200:
        return None

    payload = resp.json()
    observations = payload.get("observations") or []
    if not observations:
        return None

    obs = observations[0]
    imperial = obs.get("imperial") or {}
    temp_f = _to_float(imperial.get("temp"))
    if temp_f is None:
        temp_f = _to_float(obs.get("temp"))

    humidity = _to_float(obs.get("humidity"))
    neighborhood = str(obs.get("neighborhood") or "")
    obs_time_utc = obs.get("obsTimeUtc")
    if obs_time_utc is not None:
        obs_time_utc = str(obs_time_utc)

    return ValidStation(
        station_id=candidate.station_id,
        station_name=candidate.station_name,
        qc_status=candidate.qc_status,
        latitude=candidate.latitude,
        longitude=candidate.longitude,
        distance_km=candidate.distance_km,
        obs_time_utc=obs_time_utc,
        temp_f=temp_f,
        humidity=humidity,
        neighborhood=neighborhood,
    )


async def validate_candidates(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    candidates: List[CandidateStation],
    concurrency: int,
) -> List[ValidStation]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(candidate: CandidateStation) -> Optional[ValidStation]:
        async with semaphore:
            return await fetch_current_observation(client, api_key=api_key, candidate=candidate)

    tasks = [_one(candidate) for candidate in candidates]
    results = await asyncio.gather(*tasks)
    valid = [r for r in results if r is not None]
    valid.sort(key=lambda v: (v.distance_km if v.distance_km is not None else float("inf"), v.station_id))
    return valid


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_registry(path: Path, registry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def update_registry(
    registry: Dict[str, Any],
    *,
    station_id: str,
    lat: float,
    lon: float,
    valid_stations: List[ValidStation],
) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    station_ids = [st.station_id for st in valid_stations]

    registry[station_id] = {
        "updated_at_utc": now_utc,
        "geocode": {"lat": lat, "lon": lon},
        "station_ids": station_ids,
        "stations": [asdict(st) for st in valid_stations],
        "source": "weather.com v3/location/near + v2/pws/observations/current",
    }
    return registry


async def run(args: argparse.Namespace) -> int:
    if args.stations:
        station_ids = [s.strip().upper() for s in str(args.stations).split(",") if s.strip()]
    else:
        station_ids = [args.station.upper().strip()]

    if not station_ids:
        raise ValueError("No stations provided. Use --station or --stations.")

    api_key = _resolve_api_key(args.api_key)
    out_path = Path(args.out)

    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=max(16, args.concurrency * 2), max_keepalive_connections=8)

    registry = load_registry(out_path)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for station_id in station_ids:
            if args.stations:
                lat, lon = _resolve_target(station_id, None, None)
            else:
                lat, lon = _resolve_target(station_id, args.lat, args.lon)

            candidates = await discover_candidates(
                client,
                api_key=api_key,
                lat=lat,
                lon=lon,
                limit=args.limit,
            )
            valid = await validate_candidates(
                client,
                api_key=api_key,
                candidates=candidates,
                concurrency=args.concurrency,
            )

            registry = update_registry(registry, station_id=station_id, lat=lat, lon=lon, valid_stations=valid)

            print(f"Station: {station_id} ({lat:.4f}, {lon:.4f})")
            print(f"Candidates discovered: {len(candidates)}")
            print(f"Valid current PWS: {len(valid)}")
            if valid:
                print("Top valid stations:")
                for st in valid[: min(10, len(valid))]:
                    dist = f"{st.distance_km:.1f}km" if st.distance_km is not None else "--"
                    temp = f"{st.temp_f:.1f}F" if st.temp_f is not None else "--"
                    print(f"  - {st.station_id:14s} {dist:>8s} {temp:>8s} {st.neighborhood}")
            print("-" * 60)

    save_registry(out_path, registry)
    print(f"Registry updated: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover and validate WU PWS stations near target station.")
    parser.add_argument("--station", default="KLGA", help="Target station (default: KLGA)")
    parser.add_argument(
        "--stations",
        default=None,
        help="Comma-separated station list (e.g. KLGA,KATL). Overrides --station.",
    )
    parser.add_argument("--lat", type=float, default=None, help="Override latitude")
    parser.add_argument("--lon", type=float, default=None, help="Override longitude")
    parser.add_argument("--limit", type=int, default=50, help="Max discovered stations to validate")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent validation requests")
    parser.add_argument("--timeout", type=float, default=12.0, help="HTTP timeout in seconds")
    parser.add_argument("--api-key", default=None, help="Weather.com API key (overrides env)")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_REGISTRY_PATH),
        help=f"Registry JSON output path (default: {DEFAULT_REGISTRY_PATH})",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
