from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx

from config import ACCUWEATHER_API_KEY, ACCUWEATHER_ENABLED, OPEN_METEO_URL, STATIONS, get_polymarket_temp_unit
from collector.metar_fetcher import fetch_metar, fetch_metar_history
from collector.wunderground_fetcher import (
    STATION_WU_MAP,
    _wu_metric_value_to_f,
    _wu_scrape_values_use_metric,
    fetch_wunderground_max,
)


_WU_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_NOAA_HEADERS = {
    "User-Agent": "helios/1.0 (weather endpoint; local development)",
    "Accept": "application/geo+json,application/json;q=0.9,*/*;q=0.8",
}

_OPEN_METEO_CACHE_TTL_SECONDS = 300
_OPEN_METEO_SOURCE_CACHE: Dict[str, Dict[str, Any]] = {}


def _celsius_to_fahrenheit(celsius: float) -> float:
    return round((float(celsius) * 9.0 / 5.0) + 32.0, 1)


def _fahrenheit_to_celsius(fahrenheit: float) -> float:
    return round((float(fahrenheit) - 32.0) * 5.0 / 9.0, 1)


def _round_optional(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return round(float(value), 1)
    except Exception:
        return None


def _serialize_temp_point(
    dt_local: datetime,
    *,
    temp_c: Optional[float] = None,
    temp_f: Optional[float] = None,
    conditions: Optional[str] = None,
) -> Dict[str, Any]:
    if temp_f is None and temp_c is not None:
        temp_f = _celsius_to_fahrenheit(temp_c)
    if temp_c is None and temp_f is not None:
        temp_c = _fahrenheit_to_celsius(temp_f)

    return {
        "timestamp_local": dt_local.isoformat(),
        "time_local": dt_local.strftime("%Y-%m-%d %H:%M"),
        "temp_c": _round_optional(temp_c),
        "temp_f": _round_optional(temp_f),
        "conditions": conditions or None,
    }


def _status_from_sections(
    *,
    historical_ok: bool,
    realtime_ok: bool,
    forecast_ok: bool,
) -> str:
    successes = sum(1 for flag in (historical_ok, realtime_ok, forecast_ok) if flag)
    if successes == 3:
        return "ok"
    if successes == 0:
        return "error"
    return "partial"


def _build_disabled_source(
    source: str,
    display_name: str,
    reason: str,
    *,
    notes: Optional[List[str]] = None,
    setup: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "source": source,
        "display_name": display_name,
        "status": "disabled",
        "historical_day": {
            "kind": None,
            "confirmed": False,
            "published_through_local": None,
            "max_temp_c": None,
            "max_temp_f": None,
            "max_temp_time_local": None,
            "points": [],
        },
        "realtime": {
            "kind": None,
            "as_of_local": None,
            "temp_c": None,
            "temp_f": None,
        },
        "forecast_hourly": {
            "kind": None,
            "horizon_days": 3,
            "points": [],
        },
        "notes": list(notes or []),
        "errors": [reason] if reason else [],
        "setup": dict(setup or {}),
    }


def _clone_source_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(payload)


def _get_cached_open_meteo_source(
    station_id: str,
    *,
    max_age_seconds: Optional[float] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[float]]:
    entry = _OPEN_METEO_SOURCE_CACHE.get(station_id)
    if not entry:
        return None, None

    age_seconds = max(0.0, time.monotonic() - float(entry.get("stored_at") or 0.0))
    if max_age_seconds is not None and age_seconds > max_age_seconds:
        return None, age_seconds

    return _clone_source_payload(dict(entry.get("source") or {})), age_seconds


def _store_cached_open_meteo_source(station_id: str, source: Dict[str, Any]) -> None:
    _OPEN_METEO_SOURCE_CACHE[station_id] = {
        "stored_at": time.monotonic(),
        "source": _clone_source_payload(source),
    }


def _station_country(station_id: str) -> str:
    return str((STATION_WU_MAP.get(station_id) or {}).get("country") or "").lower()


def _station_supports_noaa(station_id: str) -> bool:
    return _station_country(station_id) == "us"


def _preferred_unit(station_id: str) -> str:
    return str(get_polymarket_temp_unit(station_id) or "F").upper()


def _build_wu_hourly_url(station_id: str) -> str:
    wu_config = STATION_WU_MAP[station_id]
    country = str(wu_config.get("country", "us")).lower()
    city = str(wu_config.get("city", "")).strip("/")
    if country == "us":
        region = str(wu_config.get("region", "")).strip("/")
        return f"https://www.wunderground.com/hourly/us/{region}/{city}/{station_id}"
    return f"https://www.wunderground.com/hourly/{country}/{city}/{station_id}"


def _extract_longest_json_array(html: str, field_name: str) -> List[Any]:
    if not html:
        return []

    pattern = re.compile(rf'"{re.escape(field_name)}"\s*:\s*(\[[^\]]*\])', re.IGNORECASE)
    best: List[Any] = []
    for raw in pattern.findall(html):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, list) and len(parsed) > len(best):
            best = parsed
    return best


def _parse_local_iso(value: Any, station_tz: ZoneInfo) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=station_tz)
    return dt.astimezone(station_tz)


def _parse_wu_clock_time(
    value: Optional[str],
    *,
    target_date: date,
    station_tz: ZoneInfo,
) -> Optional[datetime]:
    if not value or value in {"Ahora", "Unknown"}:
        return None
    try:
        parsed = datetime.strptime(value.strip(), "%I:%M %p")
    except Exception:
        return None
    return datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        parsed.hour,
        parsed.minute,
        tzinfo=station_tz,
    )


def _build_wunderground_forecast_points(
    *,
    html: str,
    station_tz: ZoneInfo,
    now_local: datetime,
    default_use_metric: bool,
) -> List[Dict[str, Any]]:
    times = _extract_longest_json_array(html, "validTimeLocal")
    temps = _extract_longest_json_array(html, "temperature")
    phrases = _extract_longest_json_array(html, "wxPhraseLong")

    inferred_metric = _wu_scrape_values_use_metric(html)
    use_metric = default_use_metric if inferred_metric is None else bool(inferred_metric)

    last_date = now_local.date() + timedelta(days=2)
    rows: List[Dict[str, Any]] = []

    for idx, raw_time in enumerate(times):
        if idx >= len(temps):
            break
        raw_temp = temps[idx]
        if raw_temp is None:
            continue
        dt_local = _parse_local_iso(raw_time, station_tz)
        if not dt_local:
            continue
        if dt_local <= now_local or dt_local.date() > last_date:
            continue
        try:
            temp_f = _wu_metric_value_to_f(float(raw_temp), use_metric)
        except Exception:
            continue
        condition = None
        if idx < len(phrases) and phrases[idx]:
            condition = str(phrases[idx])
        rows.append(_serialize_temp_point(dt_local, temp_f=temp_f, conditions=condition))

    return rows


def _build_noaa_forecast_points(
    periods: List[Dict[str, Any]],
    *,
    station_tz: ZoneInfo,
    now_local: datetime,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    last_date = now_local.date() + timedelta(days=2)

    for period in periods or []:
        dt_local = _parse_local_iso(period.get("startTime"), station_tz)
        if not dt_local:
            continue
        if dt_local <= now_local or dt_local.date() > last_date:
            continue

        raw_temp = period.get("temperature")
        unit = str(period.get("temperatureUnit") or "").upper()
        if raw_temp is None:
            continue
        try:
            value = float(raw_temp)
        except Exception:
            continue

        temp_f = value if unit == "F" else _celsius_to_fahrenheit(value)
        rows.append(
            _serialize_temp_point(
                dt_local,
                temp_f=temp_f,
                conditions=period.get("shortForecast") or None,
            )
        )

    return rows


def _build_open_meteo_source_from_payload(
    station_id: str,
    payload: Dict[str, Any],
    *,
    now_local: Optional[datetime] = None,
) -> Dict[str, Any]:
    station = STATIONS[station_id]
    station_tz = ZoneInfo(station.timezone)

    timezone_name = str(payload.get("timezone") or station.timezone)
    try:
        station_tz = ZoneInfo(timezone_name)
    except Exception:
        station_tz = ZoneInfo(station.timezone)

    current_block = payload.get("current") or {}
    current_dt = _parse_local_iso(current_block.get("time"), station_tz)
    if current_dt is None:
        current_dt = now_local.astimezone(station_tz) if now_local else datetime.now(station_tz)

    hourly = payload.get("hourly") or {}
    hourly_times = hourly.get("time") or []
    hourly_temps = hourly.get("temperature_2m") or []

    all_points: List[tuple[datetime, float]] = []
    for raw_time, raw_temp in zip(hourly_times, hourly_temps):
        if raw_temp is None:
            continue
        dt_local = _parse_local_iso(raw_time, station_tz)
        if not dt_local:
            continue
        try:
            temp_c = float(raw_temp)
        except Exception:
            continue
        all_points.append((dt_local, temp_c))

    today_local = current_dt.date()
    last_date = today_local + timedelta(days=2)

    historical_points = [
        _serialize_temp_point(dt_local, temp_c=temp_c)
        for dt_local, temp_c in all_points
        if dt_local.date() == today_local and dt_local <= current_dt
    ]
    forecast_points = [
        _serialize_temp_point(dt_local, temp_c=temp_c)
        for dt_local, temp_c in all_points
        if dt_local > current_dt and dt_local.date() <= last_date
    ]

    historical_temps_f = [row["temp_f"] for row in historical_points if row.get("temp_f") is not None]
    historical_max_f = max(historical_temps_f) if historical_temps_f else None
    historical_max_time_local = None
    for row in historical_points:
        if historical_max_f is not None and row.get("temp_f") == historical_max_f:
            historical_max_time_local = row.get("timestamp_local")
            break

    realtime_temp_c = _round_optional(current_block.get("temperature_2m"))
    realtime_temp_f = _celsius_to_fahrenheit(realtime_temp_c) if realtime_temp_c is not None else None

    notes = [
        "Open-Meteo historical values are modeled data, not station-confirmed observations.",
    ]

    return {
        "source": "open_meteo",
        "display_name": "Open-Meteo",
        "status": _status_from_sections(
            historical_ok=bool(historical_points or historical_max_f is not None),
            realtime_ok=realtime_temp_c is not None,
            forecast_ok=bool(forecast_points),
        ),
        "historical_day": {
            "kind": "modeled",
            "confirmed": False,
            "published_through_local": historical_points[-1]["timestamp_local"] if historical_points else None,
            "max_temp_c": _fahrenheit_to_celsius(historical_max_f) if historical_max_f is not None else None,
            "max_temp_f": _round_optional(historical_max_f),
            "max_temp_time_local": historical_max_time_local,
            "points": historical_points,
        },
        "realtime": {
            "kind": "modeled_current",
            "as_of_local": current_dt.isoformat() if current_dt else None,
            "temp_c": _round_optional(realtime_temp_c),
            "temp_f": _round_optional(realtime_temp_f),
        },
        "forecast_hourly": {
            "kind": "forecast",
            "horizon_days": 3,
            "points": forecast_points,
        },
        "notes": notes,
        "errors": [],
    }


async def fetch_open_meteo_source(station_id: str) -> Dict[str, Any]:
    station = STATIONS[station_id]
    station_tz = ZoneInfo(station.timezone)
    now_local = datetime.now(station_tz)

    fresh_cached, fresh_age = _get_cached_open_meteo_source(
        station_id,
        max_age_seconds=_OPEN_METEO_CACHE_TTL_SECONDS,
    )
    if fresh_cached is not None:
        fresh_cached.setdefault("notes", [])
        fresh_cached["notes"] = [
            f"Open-Meteo snapshot served from cache ({int(fresh_age or 0)}s old) to reduce rate-limit pressure.",
            *list(fresh_cached.get("notes") or []),
        ]
        return fresh_cached

    params = {
        "latitude": station.latitude,
        "longitude": station.longitude,
        "timezone": station.timezone,
        "current": "temperature_2m",
        "hourly": "temperature_2m",
        "past_days": 1,
        "forecast_days": 3,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            last_status_code: Optional[int] = None
            for attempt in range(3):
                response = await client.get(OPEN_METEO_URL, params=params)
                last_status_code = int(response.status_code)

                if response.status_code == 429:
                    retry_after_raw = response.headers.get("Retry-After")
                    if attempt < 2:
                        try:
                            retry_after = float(retry_after_raw) if retry_after_raw else float(attempt + 1)
                        except Exception:
                            retry_after = float(attempt + 1)
                        await asyncio.sleep(max(0.5, min(retry_after, 3.0)))
                        continue

                    stale_cached, stale_age = _get_cached_open_meteo_source(station_id)
                    if stale_cached is not None:
                        stale_cached.setdefault("notes", [])
                        stale_cached["notes"] = [
                            f"Open-Meteo returned HTTP 429; serving cached snapshot ({int(stale_age or 0)}s old).",
                            *list(stale_cached.get("notes") or []),
                        ]
                        return stale_cached
                    raise RuntimeError("Open-Meteo rate limited the request (HTTP 429).")

                response.raise_for_status()
                payload = response.json()
                source = _build_open_meteo_source_from_payload(
                    station_id,
                    payload,
                    now_local=now_local,
                )
                _store_cached_open_meteo_source(station_id, source)
                return source

            raise RuntimeError(f"Open-Meteo unavailable (HTTP {last_status_code or 'unknown'}).")
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Open-Meteo unavailable (HTTP {exc.response.status_code}).") from exc
    except httpx.RequestError as exc:
        stale_cached, stale_age = _get_cached_open_meteo_source(station_id)
        if stale_cached is not None:
            stale_cached.setdefault("notes", [])
            stale_cached["notes"] = [
                f"Open-Meteo request failed; serving cached snapshot ({int(stale_age or 0)}s old).",
                *list(stale_cached.get("notes") or []),
            ]
            return stale_cached
        raise RuntimeError("Open-Meteo request failed.") from exc


async def _fetch_noaa_hourly_forecast_periods(station_id: str) -> List[Dict[str, Any]]:
    station = STATIONS[station_id]
    points_url = f"https://api.weather.gov/points/{station.latitude},{station.longitude}"

    async with httpx.AsyncClient(timeout=20.0, headers=_NOAA_HEADERS, follow_redirects=True) as client:
        points_response = await client.get(points_url)
        points_response.raise_for_status()
        points_data = points_response.json()
        hourly_url = ((points_data.get("properties") or {}).get("forecastHourly") or "").strip()
        if not hourly_url:
            raise ValueError("weather.gov forecastHourly URL missing")

        hourly_response = await client.get(hourly_url)
        hourly_response.raise_for_status()
        hourly_data = hourly_response.json()
        return list((((hourly_data.get("properties") or {}).get("periods")) or []))


async def _fetch_wunderground_hourly_html(station_id: str) -> str:
    url = _build_wu_hourly_url(station_id)
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(url, headers=_WU_HEADERS)
        response.raise_for_status()
        return response.text


async def fetch_wunderground_source(station_id: str) -> Dict[str, Any]:
    station = STATIONS[station_id]
    station_tz = ZoneInfo(station.timezone)
    now_local = datetime.now(station_tz)
    today_local = now_local.date()

    obs_result, hourly_result = await asyncio.gather(
        fetch_wunderground_max(station_id, today_local),
        _fetch_wunderground_hourly_html(station_id),
        return_exceptions=True,
    )

    errors: List[str] = []
    notes: List[str] = []

    historical_points: List[Dict[str, Any]] = []
    published_through_local: Optional[str] = None
    historical_max_f: Optional[float] = None
    historical_max_time_local: Optional[str] = None
    realtime_temp_f: Optional[float] = None

    if isinstance(obs_result, Exception) or not obs_result:
        errors.append("Wunderground observed history unavailable.")
    else:
        realtime_temp_f = _round_optional(obs_result.current_temp_f)
        historical_max_f = _round_optional(obs_result.high_temp_f)
        max_dt_local = _parse_wu_clock_time(
            obs_result.high_temp_time,
            target_date=today_local,
            station_tz=station_tz,
        )
        historical_max_time_local = max_dt_local.isoformat() if max_dt_local else None

        for entry in obs_result.hourly_history or []:
            if entry.time == "Ahora":
                continue
            dt_local = _parse_wu_clock_time(entry.time, target_date=today_local, station_tz=station_tz)
            if not dt_local:
                continue
            historical_points.append(
                _serialize_temp_point(
                    dt_local,
                    temp_f=entry.temp_f,
                    conditions=entry.conditions or None,
                )
            )

        historical_points.sort(key=lambda row: row["timestamp_local"])
        if historical_points:
            published_through_local = historical_points[-1]["timestamp_local"]
            notes.append(
                "Wunderground history is lagged versus realtime; the current reading can be newer than confirmed hourly history."
            )
        if len(historical_points) <= 1:
            notes.append(
                "Wunderground public history is currently exposing limited hourly rows for this station; treat the history section as partial."
            )

    forecast_points: List[Dict[str, Any]] = []
    if isinstance(hourly_result, Exception):
        errors.append("Wunderground hourly forecast unavailable.")
    else:
        default_use_metric = bool(STATION_WU_MAP.get(station_id, {}).get("units") == "m")
        forecast_points = _build_wunderground_forecast_points(
            html=hourly_result,
            station_tz=station_tz,
            now_local=now_local,
            default_use_metric=default_use_metric,
        )

    realtime_temp_c = _fahrenheit_to_celsius(realtime_temp_f) if realtime_temp_f is not None else None

    return {
        "source": "wunderground",
        "display_name": "Wunderground",
        "status": _status_from_sections(
            historical_ok=bool(historical_points or historical_max_f is not None),
            realtime_ok=realtime_temp_f is not None,
            forecast_ok=bool(forecast_points),
        ),
        "historical_day": {
            "kind": "observed",
            "confirmed": True,
            "published_through_local": published_through_local,
            "max_temp_c": _fahrenheit_to_celsius(historical_max_f) if historical_max_f is not None else None,
            "max_temp_f": _round_optional(historical_max_f),
            "max_temp_time_local": historical_max_time_local,
            "points": historical_points,
        },
        "realtime": {
            "kind": "observed_current",
            "as_of_local": now_local.isoformat(),
            "temp_c": _round_optional(realtime_temp_c),
            "temp_f": _round_optional(realtime_temp_f),
        },
        "forecast_hourly": {
            "kind": "forecast",
            "horizon_days": 3,
            "points": forecast_points,
        },
        "notes": notes,
        "errors": errors,
    }


async def fetch_noaa_source(station_id: str) -> Dict[str, Any]:
    if not _station_supports_noaa(station_id):
        raise ValueError("NOAA/NWS is only available for US stations")

    station = STATIONS[station_id]
    station_tz = ZoneInfo(station.timezone)

    current_task = fetch_metar(station_id)
    history_task = fetch_metar_history(station_id, hours=36)
    forecast_task = _fetch_noaa_hourly_forecast_periods(station_id)

    current_result, history_result, forecast_result = await asyncio.gather(
        current_task,
        history_task,
        forecast_task,
        return_exceptions=True,
    )

    errors: List[str] = []
    notes: List[str] = [
        "NOAA/NWS uses official airport METAR observations plus weather.gov hourly forecast grids.",
    ]

    current_obs = None if isinstance(current_result, Exception) else current_result
    history_rows = [] if isinstance(history_result, Exception) else list(history_result or [])
    forecast_periods = [] if isinstance(forecast_result, Exception) else list(forecast_result or [])

    if isinstance(current_result, Exception):
        errors.append(f"NOAA current observation unavailable: {current_result}")
    if isinstance(history_result, Exception):
        errors.append(f"NOAA historical observations unavailable: {history_result}")
    if isinstance(forecast_result, Exception):
        errors.append(f"NWS hourly forecast unavailable: {forecast_result}")

    if current_obs is not None:
        now_local = current_obs.observation_time.astimezone(station_tz)
    else:
        now_local = datetime.now(station_tz)

    today_local = now_local.date()

    historical_points: List[Dict[str, Any]] = []
    historical_max_f: Optional[float] = None
    historical_max_time_local: Optional[str] = None

    filtered_history = []
    for obs in history_rows:
        try:
            obs_local = obs.observation_time.astimezone(station_tz)
        except Exception:
            continue
        if obs_local.date() != today_local:
            continue
        filtered_history.append((obs_local, obs))

    filtered_history.sort(key=lambda item: item[0])

    for obs_local, obs in filtered_history:
        historical_points.append(
            _serialize_temp_point(
                obs_local,
                temp_f=getattr(obs, "temp_f", None),
                conditions=getattr(obs, "report_type", None),
            )
        )
        temp_f = _round_optional(getattr(obs, "temp_f", None))
        if temp_f is None:
            continue
        if historical_max_f is None or temp_f > historical_max_f:
            historical_max_f = temp_f
            historical_max_time_local = obs_local.isoformat()

    realtime_temp_f = _round_optional(getattr(current_obs, "temp_f", None)) if current_obs else None
    realtime_temp_c = _round_optional(getattr(current_obs, "temp_c", None)) if current_obs else None
    realtime_as_of = current_obs.observation_time.astimezone(station_tz).isoformat() if current_obs else None

    forecast_points = _build_noaa_forecast_points(
        forecast_periods,
        station_tz=station_tz,
        now_local=now_local,
    )

    return {
        "source": "noaa_nws",
        "display_name": "NOAA/NWS",
        "status": _status_from_sections(
            historical_ok=bool(historical_points or historical_max_f is not None),
            realtime_ok=realtime_temp_f is not None,
            forecast_ok=bool(forecast_points),
        ),
        "historical_day": {
            "kind": "observed",
            "confirmed": True,
            "published_through_local": historical_points[-1]["timestamp_local"] if historical_points else None,
            "max_temp_c": _fahrenheit_to_celsius(historical_max_f) if historical_max_f is not None else None,
            "max_temp_f": _round_optional(historical_max_f),
            "max_temp_time_local": historical_max_time_local,
            "points": historical_points,
        },
        "realtime": {
            "kind": "observed_current",
            "as_of_local": realtime_as_of,
            "temp_c": realtime_temp_c,
            "temp_f": realtime_temp_f,
        },
        "forecast_hourly": {
            "kind": "forecast",
            "horizon_days": 3,
            "points": forecast_points,
        },
        "notes": notes,
        "errors": errors,
    }


async def fetch_accuweather_source(station_id: str) -> Dict[str, Any]:
    notes = [
        "Scaffold only for future AccuWeather integration.",
        "Intended target contract: observed day history, realtime temperature, and hourly forecast through the next 2 days.",
    ]

    if not ACCUWEATHER_ENABLED:
        return _build_disabled_source(
            "accuweather",
            "AccuWeather",
            "AccuWeather scaffold disabled by configuration.",
            notes=notes,
            setup={"enabled_env": "ACCUWEATHER_ENABLED", "requires_env": "ACCUWEATHER_API_KEY"},
        )

    if not str(ACCUWEATHER_API_KEY or "").strip():
        notes.append("Set ACCUWEATHER_API_KEY in the environment to continue the implementation.")
        notes.append(
            f"Station {station_id} can later resolve location metadata from coordinates, so no fixed manual location map is required."
        )
        return _build_disabled_source(
            "accuweather",
            "AccuWeather",
            "Missing ACCUWEATHER_API_KEY.",
            notes=notes,
            setup={"enabled_env": "ACCUWEATHER_ENABLED", "requires_env": "ACCUWEATHER_API_KEY"},
        )

    notes.append("API key detected, but the provider-specific fetch flow is not implemented yet.")
    return _build_disabled_source(
        "accuweather",
        "AccuWeather",
        "AccuWeather integration scaffold exists but fetch logic is pending.",
        notes=notes,
        setup={"enabled_env": "ACCUWEATHER_ENABLED", "requires_env": "ACCUWEATHER_API_KEY"},
    )


def _build_error_source(source: str, display_name: str, error: Exception) -> Dict[str, Any]:
    return {
        "source": source,
        "display_name": display_name,
        "status": "error",
        "historical_day": {
            "kind": None,
            "confirmed": False,
            "published_through_local": None,
            "max_temp_c": None,
            "max_temp_f": None,
            "max_temp_time_local": None,
            "points": [],
        },
        "realtime": {
            "kind": None,
            "as_of_local": None,
            "temp_c": None,
            "temp_f": None,
        },
        "forecast_hourly": {
            "kind": None,
            "horizon_days": 3,
            "points": [],
        },
        "notes": [],
        "errors": [str(error)],
    }


async def fetch_external_sources_snapshot(station_id: str) -> Dict[str, Any]:
    station_id = str(station_id or "").strip().upper()
    station = STATIONS.get(station_id)
    if not station:
        raise ValueError(f"Invalid station: {station_id}")

    station_tz = ZoneInfo(station.timezone)
    generated_at_utc = datetime.now(timezone.utc)

    wu_task = fetch_wunderground_source(station_id) if station_id in STATION_WU_MAP else None
    om_task = fetch_open_meteo_source(station_id)
    noaa_task = fetch_noaa_source(station_id) if _station_supports_noaa(station_id) else None
    accuweather_task = fetch_accuweather_source(station_id)

    task_specs: List[tuple[str, str, Any]] = []
    if wu_task is not None:
        task_specs.append(("wunderground", "Wunderground", wu_task))
    task_specs.append(("open_meteo", "Open-Meteo", om_task))
    if noaa_task is not None:
        task_specs.append(("noaa_nws", "NOAA/NWS", noaa_task))
    task_specs.append(("accuweather", "AccuWeather", accuweather_task))

    results = await asyncio.gather(*(spec[2] for spec in task_specs), return_exceptions=True)

    sources: List[Dict[str, Any]] = []
    for (source_name, display_name, _), result in zip(task_specs, results):
        if isinstance(result, Exception):
            sources.append(_build_error_source(source_name, display_name, result))
            continue
        sources.append(result)

    return {
        "station_id": station_id,
        "station_name": station.name,
        "timezone": station.timezone,
        "preferred_unit": _preferred_unit(station_id),
        "generated_at_utc": generated_at_utc.isoformat(),
        "generated_at_local": generated_at_utc.astimezone(station_tz).isoformat(),
        "sources": sources,
    }
