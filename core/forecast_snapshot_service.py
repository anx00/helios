from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

from collector.external_sources_fetcher import fetch_open_meteo_source
from collector.lamp_fetcher import fetch_lamp
from collector.nbm_fetcher import fetch_nbm
from collector.wunderground_fetcher import fetch_wunderground_max
from config import STATIONS, get_active_stations, get_polymarket_temp_unit
from database import insert_forecast_source_snapshot


logger = logging.getLogger("forecast_snapshot_service")

SUPPORTED_FUTURE_SOURCES = ("WUNDERGROUND", "OPEN_METEO", "NBM", "LAMP")
DEFAULT_INTERVAL_SECONDS = max(60, int(os.getenv("HELIOS_FUTURE_SNAPSHOT_INTERVAL_SECONDS", "900")))
DEFAULT_MAX_CONCURRENCY = max(1, int(os.getenv("HELIOS_FUTURE_SNAPSHOT_MAX_CONCURRENCY", "3")))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _round_optional(value: Any, digits: int = 1) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _f_to_c(value_f: Any) -> Optional[float]:
    number = _safe_float(value_f)
    if number is None:
        return None
    return (number - 32.0) * 5.0 / 9.0


def _c_to_f(value_c: Any) -> Optional[float]:
    number = _safe_float(value_c)
    if number is None:
        return None
    return (number * 9.0 / 5.0) + 32.0


def _market_unit(station_id: str) -> str:
    return str(get_polymarket_temp_unit(station_id) or "F").upper()


def _convert_f_to_market_unit(value_f: Any, market_unit: str) -> Optional[float]:
    if str(market_unit or "F").upper() == "C":
        return _f_to_c(value_f)
    return _safe_float(value_f)


def _station_timezone(station_id: str) -> ZoneInfo:
    station = STATIONS[str(station_id).upper()]
    return ZoneInfo(station.timezone)


def _ensure_target_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _normalize_note_list(notes: Any) -> List[str]:
    if isinstance(notes, list):
        return [str(note) for note in notes if str(note or "").strip()]
    if isinstance(notes, str) and notes.strip():
        return [notes.strip()]
    return []


def _iso_local(value: Any, station_id: str) -> Optional[str]:
    if not value:
        return None
    station_tz = _station_timezone(station_id)
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(station_tz).isoformat()


def _build_snapshot_payload(
    *,
    station_id: str,
    source: str,
    target_date: date,
    target_day: int,
    status: str,
    forecast_high_f: Any = None,
    forecast_high_c: Any = None,
    peak_hour_local: Any = None,
    provider_updated_local: Any = None,
    notes: Any = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    market_unit = _market_unit(station_id)
    high_f = _round_optional(forecast_high_f, 1)
    high_c = _round_optional(forecast_high_c, 1)
    if high_f is None and high_c is not None:
        high_f = _round_optional(_c_to_f(high_c), 1)
    if high_c is None and high_f is not None:
        high_c = _round_optional(_f_to_c(high_f), 1)
    return {
        "station_id": str(station_id).upper(),
        "target_date": target_date.isoformat(),
        "target_day": int(target_day),
        "source": str(source or "").upper(),
        "status": str(status or "error"),
        "market_unit": market_unit,
        "forecast_high_market": _round_optional(_convert_f_to_market_unit(high_f, market_unit), 3),
        "forecast_high_f": high_f,
        "forecast_high_c": high_c,
        "peak_hour_local": int(peak_hour_local) if peak_hour_local is not None else None,
        "provider_updated_local": _iso_local(provider_updated_local, station_id) if provider_updated_local else None,
        "notes": _normalize_note_list(notes),
        "meta": dict(meta or {}),
    }


def normalize_source_daily_forecast(
    *,
    station_id: str,
    source: str,
    target_date: date | str,
    target_day: int,
    payload: Any,
) -> Dict[str, Any]:
    """Normalize heterogeneous external-source payloads into a single day-ahead daily-high snapshot."""
    station_id = str(station_id or "").upper()
    source_name = str(source or "").upper()
    resolved_target_date = _ensure_target_date(target_date)

    if source_name == "WUNDERGROUND":
        notes = _normalize_note_list(getattr(payload, "notes", None))
        high_f = _safe_float(getattr(payload, "forecast_high_f", None))
        return _build_snapshot_payload(
            station_id=station_id,
            source=source_name,
            target_date=resolved_target_date,
            target_day=target_day,
            status="ok" if high_f is not None else "partial",
            forecast_high_f=high_f,
            peak_hour_local=getattr(payload, "forecast_peak_hour_local", None),
            notes=notes,
            meta={
                "provider_source": getattr(payload, "source", "wunderground"),
                "forecast_peak_temp_f": _round_optional(getattr(payload, "forecast_peak_temp_f", None), 1),
                "observed_high_f": _round_optional(getattr(payload, "high_temp_f", None), 1),
            },
        )

    if source_name == "OPEN_METEO":
        payload_dict = dict(payload or {})
        forecast = payload_dict.get("forecast_hourly") or {}
        points = forecast.get("points") or []
        day_points: List[Dict[str, Any]] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            raw_ts = point.get("timestamp_local")
            if not raw_ts:
                continue
            try:
                dt_local = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            except Exception:
                continue
            if dt_local.date() != resolved_target_date:
                continue
            day_points.append(point)

        max_point = None
        if day_points:
            max_point = max(
                day_points,
                key=lambda item: _safe_float(item.get("temp_f"))
                if _safe_float(item.get("temp_f")) is not None
                else (_safe_float(item.get("temp_c")) or float("-inf")),
            )
        notes = _normalize_note_list(payload_dict.get("notes"))
        notes.extend(str(err) for err in (payload_dict.get("errors") or []) if str(err or "").strip())
        peak_hour = None
        if isinstance(max_point, dict) and max_point.get("timestamp_local"):
            try:
                peak_hour = datetime.fromisoformat(str(max_point["timestamp_local"]).replace("Z", "+00:00")).hour
            except Exception:
                peak_hour = None
        status = str(payload_dict.get("status") or ("ok" if max_point else "partial"))
        return _build_snapshot_payload(
            station_id=station_id,
            source=source_name,
            target_date=resolved_target_date,
            target_day=target_day,
            status=status,
            forecast_high_f=(max_point or {}).get("temp_f"),
            forecast_high_c=(max_point or {}).get("temp_c"),
            peak_hour_local=peak_hour,
            provider_updated_local=((payload_dict.get("realtime") or {}).get("as_of_local")),
            notes=notes,
            meta={
                "point_count": len(day_points),
                "display_name": payload_dict.get("display_name"),
            },
        )

    if source_name == "NBM":
        high_f = _safe_float(getattr(payload, "max_temp_f", None))
        min_f = _safe_float(getattr(payload, "min_temp_f", None))
        return _build_snapshot_payload(
            station_id=station_id,
            source=source_name,
            target_date=resolved_target_date,
            target_day=target_day,
            status="ok" if high_f is not None else "error",
            forecast_high_f=high_f,
            provider_updated_local=getattr(payload, "fetch_time", None),
            notes=[],
            meta={
                "provider_source": getattr(payload, "source", None),
                "min_temp_f": _round_optional(min_f, 1),
            },
        )

    if source_name == "LAMP":
        if int(target_day) != 1:
            return _build_snapshot_payload(
                station_id=station_id,
                source=source_name,
                target_date=resolved_target_date,
                target_day=target_day,
                status="omitted",
                notes=["LAMP only participates in the tomorrow horizon."],
                meta={"reason": "target_day_not_supported"},
            )
        high_f = _safe_float(getattr(payload, "max_temp_f", None))
        notes: List[str] = []
        if high_f is not None:
            notes.append("LAMP is used here as a near-term 25h proxy; daily slicing is approximate.")
        return _build_snapshot_payload(
            station_id=station_id,
            source=source_name,
            target_date=resolved_target_date,
            target_day=target_day,
            status="ok" if high_f is not None else "error",
            forecast_high_f=high_f,
            provider_updated_local=getattr(payload, "fetch_time", None),
            notes=notes,
            meta={
                "provider_source": getattr(payload, "source", None),
                "valid_hours": getattr(payload, "valid_hours", None),
                "confidence": _round_optional(getattr(payload, "confidence", None), 3),
            },
        )

    raise ValueError(f"Unsupported source for normalization: {source_name}")


async def _capture_source_snapshot(
    *,
    station_id: str,
    target_date: date,
    target_day: int,
    source: str,
    persist: bool,
) -> Dict[str, Any]:
    station_id = str(station_id or "").upper()
    source_name = str(source or "").upper()

    try:
        if source_name == "WUNDERGROUND":
            raw = await fetch_wunderground_max(station_id, target_date=target_date)
        elif source_name == "OPEN_METEO":
            raw = await fetch_open_meteo_source(station_id)
        elif source_name == "NBM":
            raw = await fetch_nbm(station_id, target_date=target_date)
        elif source_name == "LAMP":
            raw = await fetch_lamp(station_id)
        else:
            raise ValueError(f"Unsupported source: {source_name}")

        snapshot = normalize_source_daily_forecast(
            station_id=station_id,
            source=source_name,
            target_date=target_date,
            target_day=target_day,
            payload=raw,
        )
    except Exception as exc:
        snapshot = _build_snapshot_payload(
            station_id=station_id,
            source=source_name,
            target_date=target_date,
            target_day=target_day,
            status="error",
            notes=[str(exc)],
            meta={"error": str(exc)},
        )

    if persist:
        insert_forecast_source_snapshot(
            station_id=station_id,
            target_date=snapshot["target_date"],
            target_day=snapshot["target_day"],
            source=snapshot["source"],
            status=snapshot["status"],
            market_unit=snapshot["market_unit"],
            forecast_high_market=snapshot["forecast_high_market"],
            forecast_high_f=snapshot["forecast_high_f"],
            forecast_high_c=snapshot["forecast_high_c"],
            peak_hour_local=snapshot["peak_hour_local"],
            provider_updated_local=snapshot["provider_updated_local"],
            notes=snapshot["notes"],
            meta=snapshot["meta"],
        )
    return snapshot


async def capture_station_future_snapshots(
    station_id: str,
    *,
    target_days: Sequence[int] = (1, 2),
    persist: bool = True,
) -> Dict[str, Any]:
    station_id = str(station_id or "").upper()
    if station_id not in STATIONS:
        raise ValueError(f"Invalid station: {station_id}")

    station_tz = _station_timezone(station_id)
    local_today = datetime.now(station_tz).date()
    snapshots: List[Dict[str, Any]] = []

    for target_day in [int(day) for day in target_days]:
        if target_day < 1:
            continue
        target_date = local_today + timedelta(days=target_day)
        results = await asyncio.gather(
            *[
                _capture_source_snapshot(
                    station_id=station_id,
                    target_date=target_date,
                    target_day=target_day,
                    source=source,
                    persist=persist,
                )
                for source in SUPPORTED_FUTURE_SOURCES
            ],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Future snapshot capture failed for %s day=%s: %s", station_id, target_day, result)
                continue
            snapshots.append(result)

    return {
        "station_id": station_id,
        "snapshots": snapshots,
    }


async def capture_all_future_snapshots(
    *,
    station_ids: Optional[Iterable[str]] = None,
    target_days: Sequence[int] = (1, 2),
    max_concurrency: Optional[int] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    resolved_station_ids = [
        str(station_id).upper()
        for station_id in (station_ids or get_active_stations().keys())
        if str(station_id).upper() in STATIONS
    ]
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency or DEFAULT_MAX_CONCURRENCY)))
    results: List[Dict[str, Any]] = []

    async def _run_station(station_id: str) -> None:
        async with semaphore:
            try:
                results.append(
                    await capture_station_future_snapshots(
                        station_id,
                        target_days=target_days,
                        persist=persist,
                    )
                )
            except Exception as exc:
                logger.warning("Future snapshot station failure for %s: %s", station_id, exc)
                results.append({"station_id": station_id, "snapshots": [], "error": str(exc)})

    await asyncio.gather(*[_run_station(station_id) for station_id in resolved_station_ids])
    return {
        "station_count": len(resolved_station_ids),
        "target_days": [int(day) for day in target_days],
        "results": results,
    }


async def future_forecast_snapshot_loop(
    *,
    interval_seconds: Optional[int] = None,
    max_concurrency: Optional[int] = None,
) -> None:
    interval = max(60, int(interval_seconds or DEFAULT_INTERVAL_SECONDS))
    while True:
        try:
            await capture_all_future_snapshots(
                target_days=(1, 2),
                max_concurrency=max_concurrency or DEFAULT_MAX_CONCURRENCY,
                persist=True,
            )
        except Exception as exc:
            logger.warning("Future forecast snapshot loop failed: %s", exc)
        await asyncio.sleep(interval)
