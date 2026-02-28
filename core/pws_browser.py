from __future__ import annotations

from bisect import bisect_right
from datetime import date as date_cls
from datetime import datetime
from statistics import median
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from config import STATIONS, get_polymarket_temp_unit
from core.compactor import get_hybrid_reader
from core.pws_learning import get_pws_learning_store
from core.replay_engine import (
    _local_day_window_utc,
    _source_partition_dates_for_local_day,
    get_replay_engine,
)

UTC = ZoneInfo("UTC")
ES_TZ = ZoneInfo("Europe/Madrid")
MIN_TABLE_INTERVAL_MINUTES = 10

_CITY_NAME_OVERRIDES = {
    "KLGA": "New York",
    "KATL": "Atlanta",
    "KORD": "Chicago",
    "KMIA": "Miami",
    "KDAL": "Dallas",
    "LFPG": "Paris",
    "EGLC": "London",
    "LTAC": "Ankara",
}


def _market_city_name(station_id: str) -> str:
    override = _CITY_NAME_OVERRIDES.get(station_id)
    if override:
        return override
    station = STATIONS.get(station_id)
    if not station or not station.name:
        return station_id
    return str(station.name).split()[0]


def _resolve_station_timezone(station_id: str) -> ZoneInfo:
    station = STATIONS.get(station_id)
    tz_name = str(getattr(station, "timezone", "") or "")
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    return ZoneInfo("America/New_York")


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):
        try:
            py_dt = to_py()
            if isinstance(py_dt, datetime):
                return py_dt if py_dt.tzinfo else py_dt.replace(tzinfo=UTC)
        except Exception:
            return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except Exception:
            return None

    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        try:
            return datetime.fromisoformat(txt.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def _parse_event_ts(event: Dict[str, Any]) -> Optional[datetime]:
    return _parse_dt(event.get("obs_time_utc")) or _parse_dt(event.get("ts_ingest_utc"))


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _normalize_station_row_time(ts_utc: datetime, station_tz: ZoneInfo) -> Dict[str, str]:
    station_local = ts_utc.astimezone(station_tz)
    es_local = ts_utc.astimezone(ES_TZ)
    return {
        "ts_utc": ts_utc.isoformat(),
        "station_time": station_local.strftime("%Y-%m-%d %H:%M"),
        "station_time_short": station_local.strftime("%H:%M"),
        "es_time": es_local.strftime("%Y-%m-%d %H:%M"),
    }


def _temperature_pair(temp_c: Optional[float], temp_f: Optional[float]) -> Dict[str, Optional[float]]:
    tc = _float_or_none(temp_c)
    tf = _float_or_none(temp_f)
    if tc is None and tf is not None:
        tc = (tf - 32.0) * (5.0 / 9.0)
    if tf is None and tc is not None:
        tf = (tc * 9.0 / 5.0) + 32.0
    return {
        "temp_c": round(float(tc), 3) if tc is not None else None,
        "temp_f": round(float(tf), 3) if tf is not None else None,
    }


def _build_market_station_meta(station_id: str) -> Dict[str, Any]:
    station = STATIONS[station_id]
    return {
        "station_id": station_id,
        "name": station.name,
        "city": _market_city_name(station_id),
        "timezone": station.timezone,
        "market_unit": get_polymarket_temp_unit(station_id),
    }


def _lead_bucket_sort_key(bucket_key: str) -> tuple:
    raw = str(bucket_key or "")
    try:
        lo = int(raw.split("-", 1)[0])
    except Exception:
        lo = 999
    return (lo, raw)


def _learning_profile_subset(profile: Dict[str, Any]) -> Dict[str, Any]:
    bucket_summaries = profile.get("lead_bucket_summaries")
    if not isinstance(bucket_summaries, dict):
        bucket_summaries = {}

    ordered_buckets = []
    for key in sorted(bucket_summaries.keys(), key=_lead_bucket_sort_key):
        row = bucket_summaries.get(key)
        if not isinstance(row, dict):
            continue
        ordered_buckets.append({
            "bucket": str(row.get("bucket") or key),
            "count": int(row.get("count") or 0),
            "avg_lead_minutes": _float_or_none(row.get("avg_lead_minutes")),
            "bias_c": _float_or_none(row.get("bias_c")),
            "mae_c": _float_or_none(row.get("mae_c")),
            "rmse_c": _float_or_none(row.get("rmse_c")),
            "sigma_c": _float_or_none(row.get("sigma_c")),
            "score": _float_or_none(row.get("score")),
            "prob_within_0_5_c": _float_or_none(row.get("prob_within_0_5_c")),
            "prob_within_1_0_c": _float_or_none(row.get("prob_within_1_0_c")),
        })

    return {
        "weight": _float_or_none(profile.get("weight")),
        "weight_predictive": _float_or_none(profile.get("weight_predictive")),
        "now_score": _float_or_none(profile.get("now_score")),
        "lead_score": _float_or_none(profile.get("lead_score")),
        "lead_skill_score": _float_or_none(profile.get("lead_skill_score")),
        "predictive_score": _float_or_none(profile.get("predictive_score")),
        "next_metar_score": _float_or_none(profile.get("next_metar_score")),
        "now_samples": int(profile.get("now_samples") or 0),
        "lead_samples": int(profile.get("lead_samples") or 0),
        "lead_avg_minutes": _float_or_none(profile.get("lead_avg_minutes")),
        "now_mae_c": _float_or_none(profile.get("now_mae_c")),
        "lead_mae_c": _float_or_none(profile.get("lead_mae_c")),
        "lead_bias_c": _float_or_none(profile.get("lead_bias_c")),
        "lead_rmse_c": _float_or_none(profile.get("lead_rmse_c")),
        "next_metar_bias_c": _float_or_none(profile.get("next_metar_bias_c")),
        "next_metar_mae_c": _float_or_none(profile.get("next_metar_mae_c")),
        "next_metar_rmse_c": _float_or_none(profile.get("next_metar_rmse_c")),
        "next_metar_sigma_c": _float_or_none(profile.get("next_metar_sigma_c")),
        "next_metar_prob_within_0_5_c": _float_or_none(profile.get("next_metar_prob_within_0_5_c")),
        "next_metar_prob_within_1_0_c": _float_or_none(profile.get("next_metar_prob_within_1_0_c")),
        "quality_band": profile.get("quality_band"),
        "rank_eligible": bool(profile.get("rank_eligible")),
        "learning_phase": profile.get("learning_phase"),
        "lead_bucket_summaries": ordered_buckets,
    }


def _predictive_ranking_rows(station_id: str, limit: int = 24) -> List[Dict[str, Any]]:
    store = get_pws_learning_store()
    summary = store.get_predictive_summary(station_id, limit=limit, eligible_only=False)
    rows: List[Dict[str, Any]] = []
    for raw in summary.get("rows") or []:
        if not isinstance(raw, dict):
            continue
        learning_profile = _learning_profile_subset(raw)
        rows.append({
            "station_id": str(raw.get("station_id") or "").upper(),
            "station_name": raw.get("station_name") or str(raw.get("station_id") or "").upper(),
            "city": _market_city_name(station_id),
            "source": raw.get("source"),
            "weight": _float_or_none(raw.get("weight")),
            "weight_predictive": _float_or_none(raw.get("weight_predictive")),
            "next_metar_score": _float_or_none(raw.get("next_metar_score")),
            "predictive_score": _float_or_none(raw.get("predictive_score")),
            "lead_avg_minutes": _float_or_none(raw.get("lead_avg_minutes")),
            "lead_samples": int(raw.get("lead_samples") or 0),
            "next_metar_mae_c": _float_or_none(raw.get("next_metar_mae_c")),
            "next_metar_bias_c": _float_or_none(raw.get("next_metar_bias_c")),
            "next_metar_prob_within_0_5_c": _float_or_none(raw.get("next_metar_prob_within_0_5_c")),
            "next_metar_prob_within_1_0_c": _float_or_none(raw.get("next_metar_prob_within_1_0_c")),
            "rank_eligible": bool(raw.get("rank_eligible")),
            "learning_phase": raw.get("learning_phase"),
            "quality_band": raw.get("quality_band"),
            "learning_profile": learning_profile,
            "lead_bucket_summaries": learning_profile.get("lead_bucket_summaries", []),
            "chart_rows": [],
            "table_rows": [],
            "summary": {
                "current_metar_mae_c": None,
                "next_metar_mae_c": _float_or_none(raw.get("next_metar_mae_c")),
                "current_metar_matches": 0,
                "next_metar_matches": int(raw.get("lead_samples") or 0),
            },
        })
    return rows


def list_pws_browser_dates(station_id: str) -> List[str]:
    engine = get_replay_engine()
    dates = engine.list_available_dates(station_id)
    out: set[str] = set()
    for date_str in dates:
        try:
            channels = engine.list_channels_for_date(date_str, station_id)
        except Exception:
            channels = []
        if "pws" in channels:
            out.add(date_str)
    try:
        out.update(get_pws_learning_store().list_audit_dates(station_id))
    except Exception:
        pass
    return sorted(out, reverse=True)


def _load_station_day_events(station_id: str, date_str: str) -> List[Dict[str, Any]]:
    station_tz = _resolve_station_timezone(station_id)
    window = _local_day_window_utc(date_str, station_tz)
    if window is None:
        raise ValueError(f"Invalid date: {date_str}")

    start_utc, end_utc = window
    reader = get_hybrid_reader()
    events: List[Dict[str, Any]] = []

    for source_date in _source_partition_dates_for_local_day(date_str, station_tz):
        events.extend(reader.get_events_sorted(source_date, station_id, ["pws", "world"]))

    filtered: List[Dict[str, Any]] = []
    for event in events:
        ts = _parse_event_ts(event)
        if ts is None:
            continue
        if start_utc <= ts < end_utc:
            filtered.append(event)

    filtered.sort(key=lambda event: _parse_event_ts(event) or datetime.max.replace(tzinfo=UTC))
    return filtered


def _extract_metar_series(events: List[Dict[str, Any]], station_tz: ZoneInfo) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, Optional[float]]] = set()

    for event in events:
        if str(event.get("ch") or "") != "world":
            continue
        data = event.get("data") or {}
        if str(data.get("src") or "").upper() != "METAR":
            continue

        obs_dt = _parse_dt(event.get("obs_time_utc")) or _parse_event_ts(event)
        if obs_dt is None:
            continue

        temps = _temperature_pair(data.get("temp_c"), data.get("temp_f"))
        key = (obs_dt.isoformat(), temps["temp_c"])
        if key in seen:
            continue
        seen.add(key)

        row = {
            **_normalize_station_row_time(obs_dt, station_tz),
            **temps,
            "report_type": str(data.get("report_type") or "METAR").upper(),
            "is_speci": bool(data.get("is_speci")),
        }
        row["_dt"] = obs_dt
        rows.append(row)

    rows.sort(key=lambda row: row["_dt"])
    return rows


def _extract_legacy_station_ids(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for event in events:
        if str(event.get("ch") or "") != "pws":
            continue
        data = event.get("data") or {}
        for raw_id in data.get("pws_ids") or []:
            sid = str(raw_id or "").strip().upper()
            if not sid:
                continue
            counts[sid] = int(counts.get(sid, 0) or 0) + 1
    return [
        {"station_id": station_id, "occurrences": counts[station_id]}
        for station_id in sorted(counts.keys(), key=lambda sid: (-counts[sid], sid))
    ]


def _append_station_row(
    rows_by_station: Dict[str, List[Dict[str, Any]]],
    seen: set[tuple[str, str, Optional[float], Optional[str], bool]],
    raw: Dict[str, Any],
    station_tz: ZoneInfo,
) -> None:
    station_id = str(raw.get("station_id") or raw.get("label") or "").strip().upper()
    if not station_id:
        return

    obs_dt = _parse_dt(raw.get("obs_time_utc"))
    if obs_dt is None:
        return

    temps = _temperature_pair(raw.get("temp_c"), raw.get("temp_f"))
    if temps["temp_c"] is None and temps["temp_f"] is None:
        return

    source = str(raw.get("source") or "UNKNOWN").upper()
    valid = bool(raw.get("valid", True))
    key = (station_id, obs_dt.isoformat(), temps["temp_c"], source, valid)
    if key in seen:
        return
    seen.add(key)

    row = {
        "station_id": station_id,
        "station_name": str(raw.get("station_name") or station_id),
        "source": source,
        "distance_km": _float_or_none(raw.get("distance_km")),
        "age_minutes": _float_or_none(raw.get("age_minutes")),
        "valid": valid,
        "qc_flag": raw.get("qc_flag"),
        "weight": _float_or_none(raw.get("weight")),
        **_normalize_station_row_time(obs_dt, station_tz),
        **temps,
    }
    row["_dt"] = obs_dt
    rows_by_station.setdefault(station_id, []).append(row)


def _extract_station_histories(
    station_id: str,
    date_str: str,
    events: List[Dict[str, Any]],
    station_tz: ZoneInfo,
) -> tuple[Dict[str, List[Dict[str, Any]]], str]:
    rows_by_station: Dict[str, List[Dict[str, Any]]] = {}
    seen: set[tuple[str, str, Optional[float], Optional[str], bool]] = set()

    for event in events:
        if str(event.get("ch") or "") != "pws":
            continue
        data = event.get("data") or {}
        for key in ("pws_readings", "pws_outliers"):
            for raw in data.get(key) or []:
                if isinstance(raw, dict):
                    _append_station_row(rows_by_station, seen, raw, station_tz)

    if rows_by_station:
        for rows in rows_by_station.values():
            rows.sort(key=lambda row: row["_dt"])
        return rows_by_station, "recorded_pws_readings"

    requested_date = date_cls.fromisoformat(date_str)
    today_local = datetime.now(station_tz).date()
    if requested_date != today_local:
        return {}, "none"

    try:
        from core.world import get_world

        snapshot = get_world().get_snapshot()
        for raw in ((snapshot.get("pws_details") or {}).get(station_id)) or []:
            if isinstance(raw, dict):
                _append_station_row(rows_by_station, seen, raw, station_tz)
    except Exception:
        return {}, "none"

    if rows_by_station:
        for rows in rows_by_station.values():
            rows.sort(key=lambda row: row["_dt"])
        return rows_by_station, "live_snapshot"

    return {}, "none"


def _downsample_table_rows(rows: List[Dict[str, Any]], min_interval_minutes: int) -> List[Dict[str, Any]]:
    if not rows:
        return []

    deduped: List[Dict[str, Any]] = []
    for row in rows:
        if not deduped:
            deduped.append(row)
            continue
        prev = deduped[-1]
        if row["ts_utc"] == prev["ts_utc"] and row.get("temp_c") == prev.get("temp_c"):
            continue
        deduped.append(row)

    if len(deduped) <= 2:
        return deduped

    intervals: List[float] = []
    for idx in range(1, len(deduped)):
        delta_minutes = (deduped[idx]["_dt"] - deduped[idx - 1]["_dt"]).total_seconds() / 60.0
        if delta_minutes > 0:
            intervals.append(delta_minutes)

    if not intervals or median(intervals) >= float(min_interval_minutes):
        return deduped

    downsampled = [deduped[0]]
    for row in deduped[1:]:
        delta_minutes = (row["_dt"] - downsampled[-1]["_dt"]).total_seconds() / 60.0
        if delta_minutes >= float(min_interval_minutes):
            downsampled.append(row)

    if downsampled[-1]["ts_utc"] != deduped[-1]["ts_utc"]:
        downsampled.append(deduped[-1])
    return downsampled


def _match_metar_rows(
    rows: List[Dict[str, Any]],
    metar_series: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    if not metar_series:
        out = []
        for row in rows:
            enriched = dict(row)
            enriched["current_metar"] = None
            enriched["next_metar"] = None
            out.append(enriched)
        return out

    metar_times = [row["_dt"] for row in metar_series]
    out: List[Dict[str, Any]] = []

    for row in rows:
        idx = bisect_right(metar_times, row["_dt"]) - 1
        current_metar = metar_series[idx] if idx >= 0 else None
        next_metar = metar_series[idx + 1] if (idx + 1) < len(metar_series) else None

        enriched = dict(row)
        enriched["current_metar"] = (
            {
                "ts_utc": current_metar["ts_utc"],
                "station_time": current_metar["station_time"],
                "es_time": current_metar["es_time"],
                "temp_c": current_metar["temp_c"],
                "temp_f": current_metar["temp_f"],
                "report_type": current_metar["report_type"],
            }
            if current_metar
            else None
        )
        enriched["next_metar"] = (
            {
                "ts_utc": next_metar["ts_utc"],
                "station_time": next_metar["station_time"],
                "es_time": next_metar["es_time"],
                "temp_c": next_metar["temp_c"],
                "temp_f": next_metar["temp_f"],
                "report_type": next_metar["report_type"],
            }
            if next_metar
            else None
        )
        out.append(enriched)

    return out


def _metric_mae(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    errors: List[float] = []
    for row in rows:
        temp_c = _float_or_none(row.get("temp_c"))
        metar = row.get(key) or {}
        metar_c = _float_or_none(metar.get("temp_c"))
        if temp_c is None or metar_c is None:
            continue
        errors.append(abs(temp_c - metar_c))
    if not errors:
        return None
    return round(sum(errors) / len(errors), 3)


def _build_station_payload(
    market_station_id: str,
    station_rows: List[Dict[str, Any]],
    metar_series: List[Dict[str, Any]],
    learning_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    latest = station_rows[-1]
    chart_rows = [
        {
            "ts_utc": row["ts_utc"],
            "station_time": row["station_time"],
            "station_time_short": row["station_time_short"],
            "es_time": row["es_time"],
            "temp_c": row["temp_c"],
            "temp_f": row["temp_f"],
            "valid": row["valid"],
            "qc_flag": row["qc_flag"],
        }
        for row in station_rows
    ]

    table_rows = _match_metar_rows(
        _downsample_table_rows(station_rows, MIN_TABLE_INTERVAL_MINUTES),
        metar_series,
    )
    latest_temp = _temperature_pair(latest.get("temp_c"), latest.get("temp_f"))
    current_mae_c = _metric_mae(table_rows, "current_metar")
    next_mae_c = _metric_mae(table_rows, "next_metar")

    current_match_count = sum(1 for row in table_rows if (row.get("current_metar") or {}).get("temp_c") is not None)
    next_match_count = sum(1 for row in table_rows if (row.get("next_metar") or {}).get("temp_c") is not None)

    return {
        "station_id": latest["station_id"],
        "station_name": latest["station_name"],
        "city": _market_city_name(market_station_id),
        "source": latest["source"],
        "distance_km": latest["distance_km"],
        "valid": latest["valid"],
        "latest_obs_time_utc": latest["ts_utc"],
        "latest_station_time": latest["station_time"],
        "latest_es_time": latest["es_time"],
        "latest_temp_c": latest_temp["temp_c"],
        "latest_temp_f": latest_temp["temp_f"],
        "raw_records_count": len(chart_rows),
        "table_records_count": len(table_rows),
        "summary": {
            "current_metar_mae_c": current_mae_c,
            "next_metar_mae_c": next_mae_c,
            "current_metar_matches": current_match_count,
            "next_metar_matches": next_match_count,
        },
        "learning_profile": _learning_profile_subset(learning_profile or {}),
        "chart_rows": chart_rows,
        "table_rows": [
            {
                "ts_utc": row["ts_utc"],
                "station_time": row["station_time"],
                "es_time": row["es_time"],
                "temp_c": row["temp_c"],
                "temp_f": row["temp_f"],
                "valid": row["valid"],
                "qc_flag": row["qc_flag"],
                "current_metar": row["current_metar"],
                "next_metar": row["next_metar"],
            }
            for row in table_rows
        ],
    }


def _station_sort_key(payload: Dict[str, Any]) -> tuple:
    summary = payload.get("summary") or {}
    learning = payload.get("learning_profile") or {}
    next_matches = int(summary.get("next_metar_matches") or 0)
    current_matches = int(summary.get("current_metar_matches") or 0)
    next_mae_c = _float_or_none(summary.get("next_metar_mae_c"))
    current_mae_c = _float_or_none(summary.get("current_metar_mae_c"))
    next_score = _float_or_none(learning.get("next_metar_score"))
    distance_km = _float_or_none(payload.get("distance_km"))
    return (
        0 if next_matches > 0 else 1,
        next_mae_c if next_mae_c is not None else 9999.0,
        -(next_score if next_score is not None else -1.0),
        0 if current_matches > 0 else 1,
        current_mae_c if current_mae_c is not None else 9999.0,
        distance_km if distance_km is not None else 9999.0,
        str(payload.get("station_id") or ""),
    )


def build_pws_browser_payload(station_id: str, date_str: str) -> Dict[str, Any]:
    if station_id not in STATIONS:
        raise ValueError(f"Unknown station_id: {station_id}")

    station_tz = _resolve_station_timezone(station_id)
    events = _load_station_day_events(station_id, date_str)
    metar_series = _extract_metar_series(events, station_tz)
    rows_by_station, detail_source = _extract_station_histories(station_id, date_str, events, station_tz)
    legacy_station_ids = _extract_legacy_station_ids(events)
    learning_store = get_pws_learning_store()
    predictive_ranking = _predictive_ranking_rows(station_id)

    pws_stations = [
        _build_station_payload(
            station_id,
            rows,
            metar_series,
            learning_profile=learning_store.get_station_profile(
                station_id,
                rows[-1]["station_id"],
                rows[-1]["source"],
            ),
        )
        for rows in rows_by_station.values()
        if rows
    ]
    pws_stations.sort(key=_station_sort_key)

    detail_available = bool(pws_stations)
    message = None
    if not detail_available:
        if legacy_station_ids:
            message = (
                "Este dia fue grabado con el formato legacy de PWS: hay consenso e IDs, "
                "pero no lecturas individuales por estacion."
            )
            if predictive_ranking:
                message += " Aun asi puedes usar el ranking predictivo acumulado para ver qué PWS suele anticipar mejor el siguiente METAR."
        else:
            message = (
                "No hay lecturas PWS detalladas para este dia."
                if not predictive_ranking
                else "No hay detalle intradia para este dia, pero si ranking predictivo acumulado por estacion."
            )
    elif detail_source == "live_snapshot":
        message = (
            "Mostrando fallback live para hoy. Las lecturas historicas detalladas "
            "solo existen si el recorder guardo pws_readings."
        )

    return {
        "station_id": station_id,
        "date": date_str,
        "market_station": _build_market_station_meta(station_id),
        "detail_available": detail_available,
        "detail_source": detail_source,
        "message": message,
        "table_interval_minutes": MIN_TABLE_INTERVAL_MINUTES,
        "lead_window_minutes": [int(learning_store.lead_min_minutes), int(learning_store.lead_max_minutes)],
        "metar_series": [
            {
                "ts_utc": row["ts_utc"],
                "station_time": row["station_time"],
                "station_time_short": row["station_time_short"],
                "es_time": row["es_time"],
                "temp_c": row["temp_c"],
                "temp_f": row["temp_f"],
                "report_type": row["report_type"],
                "is_speci": row["is_speci"],
            }
            for row in metar_series
        ],
        "pws_stations": pws_stations,
        "predictive_ranking": predictive_ranking,
        "selected_pws_station_id": pws_stations[0]["station_id"] if pws_stations else None,
        "legacy_station_ids": legacy_station_ids,
        "stats": {
            "metar_count": len(metar_series),
            "pws_station_count": len(pws_stations),
            "legacy_station_count": len(legacy_station_ids),
            "detail_station_count": len(rows_by_station),
            "predictive_station_count": len(predictive_ranking),
        },
    }


def build_pws_audit_day_payload(station_id: str, date_str: str) -> Dict[str, Any]:
    if station_id not in STATIONS:
        raise ValueError(f"Unknown station_id: {station_id}")

    store = get_pws_learning_store()
    station_tz = _resolve_station_timezone(station_id)
    rows = store.load_audit_rows(station_id, date_str)

    grouped: Dict[str, Dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        pws_station_id = str(raw.get("station_id") or "").upper()
        if not pws_station_id:
            continue

        official_dt = _parse_dt(raw.get("official_obs_time_utc"))
        sample_dt = _parse_dt(raw.get("sample_obs_time_utc"))
        profile = grouped.get(pws_station_id)
        if profile is None:
            learning_profile = store.get_station_profile(
                station_id,
                pws_station_id,
                str(raw.get("source") or "UNKNOWN"),
            )
            profile = {
                "station_id": pws_station_id,
                "station_name": raw.get("station_name") or pws_station_id,
                "source": raw.get("source"),
                "distance_km": _float_or_none(raw.get("distance_km")),
                "learning_profile": _learning_profile_subset(learning_profile),
                "audit_rows": [],
                "now_count": 0,
                "lead_count": 0,
            }
            grouped[pws_station_id] = profile

        if str(raw.get("kind") or "").upper() == "NOW":
            profile["now_count"] = int(profile.get("now_count", 0) or 0) + 1
        elif str(raw.get("kind") or "").upper() == "LEAD":
            profile["lead_count"] = int(profile.get("lead_count", 0) or 0) + 1

        sample_time = _normalize_station_row_time(sample_dt, station_tz) if sample_dt else {
            "ts_utc": raw.get("sample_obs_time_utc"),
            "station_time": None,
            "station_time_short": None,
            "es_time": None,
        }
        official_time = _normalize_station_row_time(official_dt, station_tz) if official_dt else {
            "ts_utc": raw.get("official_obs_time_utc"),
            "station_time": None,
            "station_time_short": None,
            "es_time": None,
        }
        profile["audit_rows"].append({
            "kind": str(raw.get("kind") or "").upper(),
            "lead_bucket": raw.get("lead_bucket"),
            "lead_minutes": _float_or_none(raw.get("lead_minutes")),
            "sample_time": sample_time,
            "official_time": official_time,
            "sample_temp_c": _float_or_none(raw.get("sample_temp_c")),
            "sample_temp_f": _float_or_none(raw.get("sample_temp_f")),
            "official_temp_c": _float_or_none(raw.get("official_temp_c")),
            "official_temp_f": _float_or_none(raw.get("official_temp_f")),
            "signed_error_c": _float_or_none(raw.get("signed_error_c")),
            "abs_error_c": _float_or_none(raw.get("abs_error_c")),
            "hit_within_0_5_c": bool(raw.get("hit_within_0_5_c")),
            "hit_within_1_0_c": bool(raw.get("hit_within_1_0_c")),
        })

    stations = sorted(
        grouped.values(),
        key=lambda row: (
            -(_float_or_none(((row.get("learning_profile") or {}).get("next_metar_score"))) or -1.0),
            -int(row.get("lead_count") or 0),
            str(row.get("station_id") or ""),
        ),
    )

    return {
        "station_id": station_id,
        "date": date_str,
        "market_station": _build_market_station_meta(station_id),
        "row_count": len(rows),
        "stations": stations,
    }
