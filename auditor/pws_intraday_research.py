from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import shutil
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config import STATIONS
from core.autotrader import AutoTraderConfig
from core.pws_learning import PWSLearningStore
from core.trading_signal import (
    _market_unit,
    _repricing_influence,
    estimate_next_metar_projection,
)


UTC = timezone.utc
DEFAULT_REMOTE_ROOT = "/opt/helios"
DEFAULT_REMOTE_DB = "/opt/helios/helios_weather.db"
DEFAULT_REMOTE_AUDIT_DIR = "/opt/helios/data/pws_learning/audit"
DEFAULT_REMOTE_WEIGHTS = "/opt/helios/data/pws_learning/weights.json"
DEFAULT_STATIONS = tuple(sorted(STATIONS.keys()))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n", ""}:
        return False
    return None


def _parse_dt(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _fmt_num(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.{digits}f}%"


def _csv_write(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _quantile(values: Sequence[float], pct: float) -> Optional[float]:
    nums = sorted(float(v) for v in values if v is not None)
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    position = max(0.0, min(1.0, float(pct))) * (len(nums) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return nums[lo]
    frac = position - lo
    return nums[lo] + (nums[hi] - nums[lo]) * frac


def _corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return None
    return num / (den_x * den_y)


def _median(values: Iterable[float]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return float(statistics.median(nums))


def _sign_with_threshold(value: Optional[float], threshold: float) -> str:
    number = _safe_float(value)
    if number is None:
        return "NONE"
    if number >= threshold:
        return "UP"
    if number <= -threshold:
        return "DOWN"
    return "FLAT"


def _score_populated(row: Dict[str, Any]) -> int:
    return sum(1 for value in row.values() if value not in (None, ""))


def dedupe_predictions(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    duplicate_meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = (
            str(row.get("station_id") or "").upper(),
            str(row.get("target_date") or ""),
            str(row.get("timestamp") or ""),
        )
        if key not in grouped:
            grouped[key] = row
            duplicate_meta[key] = {"rows": 1}
            continue
        duplicate_meta[key]["rows"] += 1
        incumbent = grouped[key]
        if _score_populated(row) > _score_populated(incumbent):
            grouped[key] = row
    deduped = sorted(
        grouped.values(),
        key=lambda item: (
            str(item.get("station_id") or ""),
            str(item.get("target_date") or ""),
            str(item.get("timestamp") or ""),
        ),
    )
    duplicates = [
        {
            "station_id": station_id,
            "target_date": target_date,
            "timestamp": timestamp,
            "rows": meta["rows"],
        }
        for (station_id, target_date, timestamp), meta in sorted(duplicate_meta.items())
        if meta["rows"] > 1
    ]
    return deduped, duplicates


def _build_performance_index(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        row["_dt"] = _parse_dt(row.get("timestamp"))
        if row["_dt"] is None:
            continue
        key = (str(row.get("station_id") or "").upper(), str(row.get("target_date") or ""))
        grouped[key].append(row)
    for payload in grouped.values():
        payload.sort(key=lambda item: item["_dt"])
    return grouped


def _build_market_velocity_index(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        row["_dt"] = _parse_dt(row.get("timestamp"))
        if row["_dt"] is None:
            continue
        key = (
            str(row.get("station_id") or "").upper(),
            str(row.get("target_date") or ""),
            str(row.get("bracket_name") or ""),
        )
        grouped[key].append(row)
    for payload in grouped.values():
        payload.sort(key=lambda item: item["_dt"])
    return grouped


def _latest_snapshot_at_or_before(rows: Sequence[Dict[str, Any]], dt: Optional[datetime]) -> Optional[Dict[str, Any]]:
    if dt is None:
        return None
    candidate = None
    for row in rows:
        row_dt = row.get("_dt")
        if row_dt is None or row_dt > dt:
            break
        candidate = row
    return candidate


def _first_snapshot_after(rows: Sequence[Dict[str, Any]], dt: Optional[datetime]) -> Optional[Dict[str, Any]]:
    if dt is None:
        return None
    for row in rows:
        row_dt = row.get("_dt")
        if row_dt is not None and row_dt >= dt:
            return row
    return None


def _latest_probability_at_or_before(rows: Sequence[Dict[str, Any]], dt: Optional[datetime]) -> Optional[float]:
    if dt is None:
        return None
    candidate = None
    for row in rows:
        row_dt = row.get("_dt")
        if row_dt is None or row_dt > dt:
            break
        candidate = row
    return _safe_float((candidate or {}).get("probability"))


def _final_real_by_station_date(perf_index: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> Dict[Tuple[str, str], Optional[float]]:
    finals: Dict[Tuple[str, str], Optional[float]] = {}
    for key, rows in perf_index.items():
        values = [_safe_float(row.get("wu_final")) for row in rows]
        actuals = [value for value in values if value is not None]
        if actuals:
            finals[key] = max(actuals)
            continue
        cumulative = [_safe_float(row.get("cumulative_max_f")) for row in rows]
        cumulative = [value for value in cumulative if value is not None]
        finals[key] = max(cumulative) if cumulative else None
    return finals


def _build_profile_maps(weights_payload: Dict[str, Any], station_ids: Sequence[str]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    profile_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    profile_rows: List[Dict[str, Any]] = []
    market_rows: List[Dict[str, Any]] = []
    markets = weights_payload.get("markets") if isinstance(weights_payload, dict) else {}
    for station_id in station_ids:
        market = (markets or {}).get(station_id, {}) if isinstance(markets, dict) else {}
        stations = market.get("stations") if isinstance(market, dict) else {}
        pending = market.get("pending") if isinstance(market, dict) else []
        market_rows.append(
            {
                "market_station_id": station_id,
                "tracked_station_count": len(stations) if isinstance(stations, dict) else 0,
                "pending_count": len(pending) if isinstance(pending, list) else 0,
                "updated_at": market.get("updated_at") if isinstance(market, dict) else None,
                "last_official_obs_time_utc": market.get("last_official_obs_time_utc") if isinstance(market, dict) else None,
            }
        )
        if not isinstance(stations, dict):
            continue
        for pws_station_id, raw in stations.items():
            if not isinstance(raw, dict):
                continue
            row = {
                "market_station_id": station_id,
                "station_id": str(pws_station_id or "").upper(),
                "source": raw.get("source"),
                "weight": _safe_float(raw.get("weight")),
                "weight_predictive": _safe_float(raw.get("weight_predictive")),
                "now_score": _safe_float(raw.get("now_score")),
                "lead_score": _safe_float(raw.get("lead_score")),
                "predictive_score": _safe_float(raw.get("predictive_score")),
                "next_metar_score": _safe_float(raw.get("next_metar_score")),
                "next_metar_mae_c": _safe_float(raw.get("next_metar_mae_c")),
                "next_metar_bias_c": _safe_float(raw.get("next_metar_bias_c")),
                "next_metar_sigma_c": _safe_float(raw.get("next_metar_sigma_c")),
                "now_samples": _safe_int(raw.get("now_samples")) or 0,
                "lead_samples": _safe_int(raw.get("lead_samples")) or 0,
                "rank_eligible": bool(raw.get("rank_eligible")),
                "learning_phase": raw.get("learning_phase"),
                "rank_reason": raw.get("rank_reason"),
                "source_prior": _safe_float(raw.get("source_prior")),
                "lead_avg_minutes": _safe_float(raw.get("lead_avg_minutes")),
            }
            profile_rows.append(row)
            profile_map[(station_id, row["station_id"])] = row
    return profile_map, profile_rows, market_rows


def _group_audit_rows(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        market_station_id = str(raw.get("market_station_id") or "").upper()
        market_date = str(raw.get("market_date") or "")
        official_obs_time_utc = str(raw.get("official_obs_time_utc") or "")
        if not market_station_id or not market_date or not official_obs_time_utc:
            continue
        row = dict(raw)
        row["_official_dt"] = _parse_dt(row.get("official_obs_time_utc"))
        row["_sample_dt"] = _parse_dt(row.get("sample_obs_time_utc"))
        grouped[(market_station_id, market_date, official_obs_time_utc)].append(row)
    for payload in grouped.values():
        payload.sort(key=lambda item: item.get("_sample_dt") or datetime.min.replace(tzinfo=UTC))
    return grouped


def _build_learning_metric(station_id: str, lead_rows: Sequence[Dict[str, Any]], profile_map: Dict[Tuple[str, str], Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    top_profiles: List[Dict[str, Any]] = []
    for row in lead_rows:
        station_key = str(row.get("station_id") or "").upper()
        profile = profile_map.get((station_id, station_key))
        if not profile:
            continue
        top_profiles.append(
            {
                "station_id": station_key,
                "next_metar_score": profile.get("next_metar_score"),
                "next_metar_bias_c": profile.get("next_metar_bias_c"),
                "next_metar_sigma_c": profile.get("next_metar_sigma_c"),
                "lead_score": profile.get("lead_score"),
                "now_score": profile.get("now_score"),
                "weight_predictive": profile.get("weight_predictive"),
                "weight": profile.get("weight"),
                "learning_phase": profile.get("learning_phase"),
                "rank_eligible": profile.get("rank_eligible"),
                "now_samples": profile.get("now_samples"),
                "lead_samples": profile.get("lead_samples"),
            }
        )
    return {"policy": {"lead_window_minutes": [5, 45]}, "top_profiles": top_profiles}, top_profiles


def _build_projection_rows(
    station_id: str,
    lead_rows: Sequence[Dict[str, Any]],
    profile_map: Dict[Tuple[str, str], Dict[str, Any]],
    official_obs_time_utc: Optional[datetime],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in lead_rows:
        station_key = str(raw.get("station_id") or "").upper()
        profile = profile_map.get((station_id, station_key), {})
        sample_dt = raw.get("_sample_dt")
        age_minutes = None
        if sample_dt is not None and official_obs_time_utc is not None:
            age_minutes = max(0.0, (official_obs_time_utc - sample_dt).total_seconds() / 60.0)
        out.append(
            {
                "valid": bool(_safe_bool(raw.get("valid"))),
                "station_id": station_key,
                "temp_c": _safe_float(raw.get("sample_temp_c")),
                "source": raw.get("source"),
                "distance_km": _safe_float(raw.get("distance_km")),
                "age_minutes": round(age_minutes, 3) if age_minutes is not None else _safe_float(raw.get("lead_minutes")),
                "obs_time_utc": raw.get("sample_obs_time_utc"),
                "learning_weight": profile.get("weight"),
                "learning_weight_predictive": profile.get("weight_predictive"),
                "learning_now_score": profile.get("now_score"),
                "learning_lead_score": profile.get("lead_score"),
                "learning_predictive_score": profile.get("predictive_score"),
                "learning_now_samples": profile.get("now_samples"),
                "learning_lead_samples": profile.get("lead_samples"),
            }
        )
    return out


def _station_actual_delta(current_temp_c: Optional[float], prev_temp_c: Optional[float], market_unit: str) -> Optional[float]:
    current = _safe_float(current_temp_c)
    previous = _safe_float(prev_temp_c)
    if current is None or previous is None:
        return None
    if market_unit == "F":
        return ((current * 9.0 / 5.0) + 32.0) - ((previous * 9.0 / 5.0) + 32.0)
    return current - previous


def build_cycle_rows(
    audit_rows: Sequence[Dict[str, Any]],
    perf_index: Dict[Tuple[str, str], List[Dict[str, Any]]],
    profile_map: Dict[Tuple[str, str], Dict[str, Any]],
    velocity_index: Optional[Dict[Tuple[str, str, str], List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    grouped = _group_audit_rows(audit_rows)
    station_officials: Dict[Tuple[str, str], List[Tuple[datetime, Dict[str, Any]]]] = defaultdict(list)
    for (station_id, market_date, _official_iso), rows in grouped.items():
        official_dt = rows[0].get("_official_dt")
        if official_dt is None:
            continue
        station_officials[(station_id, market_date)].append((official_dt, rows[0]))
    for payload in station_officials.values():
        payload.sort(key=lambda item: item[0])

    previous_map: Dict[Tuple[str, str, str], Tuple[datetime, Dict[str, Any]]] = {}
    for key, official_rows in station_officials.items():
        previous: Optional[Tuple[datetime, Dict[str, Any]]] = None
        for official_dt, sample_row in official_rows:
            official_iso = str(sample_row.get("official_obs_time_utc") or official_dt.isoformat())
            if previous is not None:
                previous_map[(key[0], key[1], official_iso)] = previous
            previous = (official_dt, sample_row)

    out: List[Dict[str, Any]] = []
    for (station_id, market_date, official_iso), rows in sorted(grouped.items()):
        if not rows:
            continue
        prev = previous_map.get((station_id, market_date, official_iso))
        if prev is None:
            continue
        current_official = rows[0]
        current_official_dt = current_official.get("_official_dt")
        prev_official_dt, prev_official_row = prev
        if current_official_dt is None:
            continue

        lead_rows = [
            row
            for row in rows
            if str(row.get("kind") or "").upper() == "LEAD" and _safe_bool(row.get("valid")) is True
        ]
        if not lead_rows:
            continue

        learning_metric, top_profiles = _build_learning_metric(station_id, lead_rows, profile_map)
        projection_rows = _build_projection_rows(station_id, lead_rows, profile_map, prev_official_dt)
        reference_dt = max((row.get("_sample_dt") for row in lead_rows if row.get("_sample_dt") is not None), default=None)
        consensus_c = _median(_safe_float(row.get("sample_temp_c")) for row in lead_rows)
        projection = estimate_next_metar_projection(
            station_id=station_id,
            rows=projection_rows,
            official_temp_c=_safe_float(prev_official_row.get("official_temp_c")),
            official_obs_utc=prev_official_dt,
            learning_metric=learning_metric,
            consensus_c=consensus_c,
            reference_utc=reference_dt,
        )

        market_unit = _market_unit(station_id)
        perf_rows = perf_index.get((station_id, market_date), [])
        ref_snapshot = _latest_snapshot_at_or_before(perf_rows, reference_dt)
        official_snapshot = _latest_snapshot_at_or_before(perf_rows, current_official_dt)
        official_after_snapshot = _first_snapshot_after(perf_rows, current_official_dt)
        prev_snapshot = _latest_snapshot_at_or_before(perf_rows, prev_official_dt)
        sample_market = _safe_float((ref_snapshot or {}).get("market_weighted_avg"))
        official_market = _safe_float((official_snapshot or official_after_snapshot or {}).get("market_weighted_avg"))
        prev_market = _safe_float((prev_snapshot or {}).get("market_weighted_avg"))
        actual_market_move = None
        if sample_market is not None and official_market is not None:
            actual_market_move = official_market - sample_market
        actual_market_from_prev = None
        if prev_market is not None and official_market is not None:
            actual_market_from_prev = official_market - prev_market

        actual_next_delta_market = _station_actual_delta(
            _safe_float(current_official.get("official_temp_c")),
            _safe_float(prev_official_row.get("official_temp_c")),
            market_unit,
        )
        market_threshold = 0.35 if market_unit == "C" else 0.6
        projection_direction = str((projection or {}).get("direction") or "FLAT").upper()
        actual_temp_direction = _sign_with_threshold(actual_next_delta_market, market_threshold)
        actual_market_direction = _sign_with_threshold(actual_market_from_prev, market_threshold)
        intraday_market_direction = _sign_with_threshold(actual_market_move, market_threshold)
        terminal_confidence = _safe_float((ref_snapshot or {}).get("confidence_score"))
        repricing_influence = (
            _repricing_influence(
                station_id=station_id,
                next_projection=projection,
                terminal_model={"confidence": terminal_confidence, "peak_hour": None},
                reference_utc=reference_dt or current_official_dt,
            )
            if projection.get("available")
            else 0.0
        )
        config = AutoTraderConfig()
        minimum_delta = 0.2 if market_unit == "C" else 0.35
        tactical_allowed = bool(
            projection.get("available")
            and _safe_float(projection.get("minutes_to_next")) is not None
            and float(_safe_float(projection.get("minutes_to_next")) or 0.0) >= float(config.tactical_min_minutes_to_next)
            and float(_safe_float(projection.get("minutes_to_next")) or 0.0) <= float(config.tactical_max_minutes_to_next)
            and projection_direction != "FLAT"
            and abs(float(_safe_float(projection.get("delta_market")) or 0.0)) >= minimum_delta
            and float(repricing_influence) >= 0.18
        )

        top_source = None
        top_weight = None
        if projection.get("contributors"):
            contributor = projection["contributors"][0]
            top_source = contributor.get("source")
            top_weight = _safe_float(contributor.get("weight"))
        top_bracket = (ref_snapshot or {}).get("market_top1_bracket")
        top_bracket_ref_prob = None
        top_bracket_official_prob = None
        top_bracket_prob_move = None
        if velocity_index and top_bracket:
            bracket_rows = velocity_index.get((station_id, market_date, str(top_bracket)), [])
            top_bracket_ref_prob = _latest_probability_at_or_before(bracket_rows, reference_dt)
            top_bracket_official_prob = _latest_probability_at_or_before(bracket_rows, current_official_dt)
            if top_bracket_ref_prob is not None and top_bracket_official_prob is not None:
                top_bracket_prob_move = top_bracket_official_prob - top_bracket_ref_prob

        out.append(
            {
                "station_id": station_id,
                "target_date": market_date,
                "previous_official_utc": prev_official_dt.isoformat(),
                "official_utc": current_official_dt.isoformat(),
                "reference_utc": reference_dt.isoformat() if reference_dt is not None else None,
                "lead_row_count": len(lead_rows),
                "lead_station_count": len({str(row.get("station_id") or "").upper() for row in lead_rows}),
                "consensus_c": consensus_c,
                "prev_official_temp_c": _safe_float(prev_official_row.get("official_temp_c")),
                "official_temp_c": _safe_float(current_official.get("official_temp_c")),
                "projection_available": bool(projection.get("available")),
                "expected_c": _safe_float(projection.get("expected_c")),
                "sigma_c": _safe_float(projection.get("sigma_c")),
                "expected_market": _safe_float(projection.get("expected_market")),
                "current_market": _safe_float(projection.get("current_market")),
                "delta_market": _safe_float(projection.get("delta_market")),
                "projection_direction": projection_direction,
                "projection_confidence": _safe_float(projection.get("confidence")),
                "projection_minutes_to_next": _safe_float(projection.get("minutes_to_next")),
                "actual_next_delta_market": actual_next_delta_market,
                "actual_temp_direction": actual_temp_direction,
                "actual_market_direction": actual_market_direction,
                "actual_intraday_market_direction": intraday_market_direction,
                "market_weighted_avg_reference": sample_market,
                "market_weighted_avg_prev_official": prev_market,
                "market_weighted_avg_official": official_market,
                "actual_market_move_from_reference": actual_market_move,
                "actual_market_move_from_prev": actual_market_from_prev,
                "market_move_matches_projection": int(projection_direction == intraday_market_direction) if intraday_market_direction != "NONE" else None,
                "temp_move_matches_projection": int(projection_direction == actual_temp_direction) if actual_temp_direction != "NONE" else None,
                "repricing_influence": round(float(repricing_influence), 6),
                "tactical_allowed": int(tactical_allowed),
                "reference_helios_pred": _safe_float((ref_snapshot or {}).get("helios_pred")),
                "reference_confidence_score": terminal_confidence,
                "reference_cumulative_max_f": _safe_float((ref_snapshot or {}).get("cumulative_max_f")),
                "reference_wu_final": _safe_float((ref_snapshot or {}).get("wu_final")),
                "reference_market_top1_bracket": (ref_snapshot or {}).get("market_top1_bracket"),
                "reference_market_top1_prob": _safe_float((ref_snapshot or {}).get("market_top1_prob")),
                "reference_market_top2_bracket": (ref_snapshot or {}).get("market_top2_bracket"),
                "reference_market_top2_prob": _safe_float((ref_snapshot or {}).get("market_top2_prob")),
                "reference_market_top3_bracket": (ref_snapshot or {}).get("market_top3_bracket"),
                "reference_market_top3_prob": _safe_float((ref_snapshot or {}).get("market_top3_prob")),
                "top_bracket_velocity_reference_prob": top_bracket_ref_prob,
                "top_bracket_velocity_official_prob": top_bracket_official_prob,
                "top_bracket_velocity_move": top_bracket_prob_move,
                "top_projection_source": top_source,
                "top_projection_weight": top_weight,
                "rank_ready_profile_count": sum(1 for profile in top_profiles if profile.get("rank_eligible")),
            }
        )
    return out


def _aggregate_stats(rows: Sequence[Dict[str, Any]], *, group_key: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_key) or "UNKNOWN")
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    for key, payload in sorted(grouped.items()):
        abs_errors = [_safe_float(row.get("abs_error_c")) for row in payload]
        abs_errors = [value for value in abs_errors if value is not None]
        signed = [_safe_float(row.get("signed_error_c")) for row in payload]
        signed = [value for value in signed if value is not None]
        hits_05 = [
            1.0 if _safe_bool(row.get("hit_within_0_5_c")) else 0.0
            for row in payload
            if _safe_bool(row.get("hit_within_0_5_c")) is not None
        ]
        hits_10 = [
            1.0 if _safe_bool(row.get("hit_within_1_0_c")) else 0.0
            for row in payload
            if _safe_bool(row.get("hit_within_1_0_c")) is not None
        ]
        out.append(
            {
                group_key: key,
                "row_count": len(payload),
                "station_count": len({str(row.get("station_id") or "").upper() for row in payload}),
                "mae_c": statistics.fmean(abs_errors) if abs_errors else None,
                "bias_c": statistics.fmean(signed) if signed else None,
                "hit_rate_0_5_c": statistics.fmean(hits_05) if hits_05 else None,
                "hit_rate_1_0_c": statistics.fmean(hits_10) if hits_10 else None,
                "p50_abs_error_c": _quantile(abs_errors, 0.5),
                "p90_abs_error_c": _quantile(abs_errors, 0.9),
            }
        )
    return out


def summarize_cycle_metrics(cycle_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    available = [row for row in cycle_rows if row.get("projection_available")]
    delta_vs_temp_pairs = [
        (
            float(_safe_float(row.get("delta_market")) or 0.0),
            float(_safe_float(row.get("actual_next_delta_market")) or 0.0),
        )
        for row in available
        if _safe_float(row.get("delta_market")) is not None and _safe_float(row.get("actual_next_delta_market")) is not None
    ]
    delta_vs_market_pairs = [
        (
            float(_safe_float(row.get("delta_market")) or 0.0),
            float(_safe_float(row.get("actual_market_move_from_prev")) or 0.0),
        )
        for row in available
        if _safe_float(row.get("delta_market")) is not None and _safe_float(row.get("actual_market_move_from_prev")) is not None
    ]
    temp_match = [float(row.get("temp_move_matches_projection")) for row in available if row.get("temp_move_matches_projection") is not None]
    market_match = [float(row.get("market_move_matches_projection")) for row in available if row.get("market_move_matches_projection") is not None]
    tactical_allowed = [row for row in available if int(row.get("tactical_allowed") or 0) == 1]
    tactical_blocked = [row for row in available if int(row.get("tactical_allowed") or 0) == 0]

    def _precision(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
        values = [float(row.get(key)) for row in rows if row.get(key) is not None]
        return statistics.fmean(values) if values else None

    return {
        "cycle_count": len(cycle_rows),
        "projection_count": len(available),
        "directional_hit_temp": statistics.fmean(temp_match) if temp_match else None,
        "directional_hit_market": statistics.fmean(market_match) if market_match else None,
        "corr_projection_vs_temp_move": _corr([x for x, _ in delta_vs_temp_pairs], [y for _, y in delta_vs_temp_pairs]) if delta_vs_temp_pairs else None,
        "corr_projection_vs_market_move": _corr([x for x, _ in delta_vs_market_pairs], [y for _, y in delta_vs_market_pairs]) if delta_vs_market_pairs else None,
        "avg_repricing_influence": statistics.fmean([float(_safe_float(row.get("repricing_influence")) or 0.0) for row in available]) if available else None,
        "tactical_allowed_rate": (len(tactical_allowed) / len(available)) if available else None,
        "tactical_allowed_temp_precision": _precision(tactical_allowed, "temp_move_matches_projection"),
        "tactical_allowed_market_precision": _precision(tactical_allowed, "market_move_matches_projection"),
        "tactical_blocked_temp_precision": _precision(tactical_blocked, "temp_move_matches_projection"),
        "tactical_blocked_market_precision": _precision(tactical_blocked, "market_move_matches_projection"),
    }


def summarize_station_metrics(cycle_rows: Sequence[Dict[str, Any]], finals: Dict[Tuple[str, str], Optional[float]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in cycle_rows:
        grouped[str(row.get("station_id") or "").upper()].append(row)

    out: List[Dict[str, Any]] = []
    for station_id, payload in sorted(grouped.items()):
        available = [row for row in payload if row.get("projection_available")]
        temp_hits = [float(row.get("temp_move_matches_projection")) for row in available if row.get("temp_move_matches_projection") is not None]
        market_hits = [float(row.get("market_move_matches_projection")) for row in available if row.get("market_move_matches_projection") is not None]
        ref_error = []
        market_error = []
        for row in payload:
            final_real = finals.get((station_id, str(row.get("target_date") or "")))
            helios_pred = _safe_float(row.get("reference_helios_pred"))
            market_avg = _safe_float(row.get("market_weighted_avg_reference"))
            if final_real is not None and helios_pred is not None:
                ref_error.append(abs(helios_pred - final_real))
            if final_real is not None and market_avg is not None:
                market_error.append(abs(market_avg - final_real))
        out.append(
            {
                "station_id": station_id,
                "cycle_count": len(payload),
                "projection_count": len(available),
                "directional_hit_temp": statistics.fmean(temp_hits) if temp_hits else None,
                "directional_hit_market": statistics.fmean(market_hits) if market_hits else None,
                "helios_vs_final_mae_f": statistics.fmean(ref_error) if ref_error else None,
                "market_vs_final_mae_f": statistics.fmean(market_error) if market_error else None,
                "tactical_allowed_rate": (
                    sum(int(row.get("tactical_allowed") or 0) for row in available) / len(available)
                ) if available else None,
            }
        )
    return out


def summarize_individual_pws_metrics(
    audit_rows: Sequence[Dict[str, Any]],
    profile_map: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for raw in audit_rows:
        market_station_id = str(raw.get("market_station_id") or "").upper()
        station_id = str(raw.get("station_id") or "").upper()
        if not market_station_id or not station_id:
            continue
        grouped[(market_station_id, station_id)].append(raw)

    out: List[Dict[str, Any]] = []
    for (market_station_id, pws_station_id), payload in sorted(grouped.items()):
        lead_rows = [row for row in payload if str(row.get("kind") or "").upper() == "LEAD"]
        now_rows = [row for row in payload if str(row.get("kind") or "").upper() == "NOW"]

        def _metric(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
            return [float(val) for val in (_safe_float(row.get(key)) for row in rows) if val is not None]

        def _bool_metric(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
            values = []
            for row in rows:
                parsed = _safe_bool(row.get(key))
                if parsed is not None:
                    values.append(1.0 if parsed else 0.0)
            return values

        profile = profile_map.get((market_station_id, pws_station_id), {})
        lead_abs = _metric(lead_rows, "abs_error_c")
        lead_signed = _metric(lead_rows, "signed_error_c")
        now_abs = _metric(now_rows, "abs_error_c")
        now_signed = _metric(now_rows, "signed_error_c")
        lead_hits_05 = _bool_metric(lead_rows, "hit_within_0_5_c")
        lead_hits_10 = _bool_metric(lead_rows, "hit_within_1_0_c")
        now_hits_05 = _bool_metric(now_rows, "hit_within_0_5_c")
        now_hits_10 = _bool_metric(now_rows, "hit_within_1_0_c")
        lead_minutes = _metric(lead_rows, "lead_minutes")
        distance_values = _metric(payload, "distance_km")
        observed_rank = (
            (statistics.fmean(lead_hits_10) if lead_hits_10 else 0.0) * 100.0
            - (statistics.fmean(lead_abs) if lead_abs else 9.0) * 12.0
            + min(len(lead_rows), 20) * 0.2
        )
        out.append(
            {
                "market_station_id": market_station_id,
                "pws_station_id": pws_station_id,
                "source": payload[0].get("source"),
                "station_name": payload[0].get("station_name"),
                "distance_km": statistics.fmean(distance_values) if distance_values else None,
                "row_count": len(payload),
                "lead_row_count": len(lead_rows),
                "now_row_count": len(now_rows),
                "lead_mae_c": statistics.fmean(lead_abs) if lead_abs else None,
                "lead_bias_c": statistics.fmean(lead_signed) if lead_signed else None,
                "lead_hit_rate_0_5_c": statistics.fmean(lead_hits_05) if lead_hits_05 else None,
                "lead_hit_rate_1_0_c": statistics.fmean(lead_hits_10) if lead_hits_10 else None,
                "lead_avg_minutes": statistics.fmean(lead_minutes) if lead_minutes else None,
                "now_mae_c": statistics.fmean(now_abs) if now_abs else None,
                "now_bias_c": statistics.fmean(now_signed) if now_signed else None,
                "now_hit_rate_0_5_c": statistics.fmean(now_hits_05) if now_hits_05 else None,
                "now_hit_rate_1_0_c": statistics.fmean(now_hits_10) if now_hits_10 else None,
                "current_weight_predictive": _safe_float(profile.get("weight_predictive")),
                "current_weight": _safe_float(profile.get("weight")),
                "current_next_metar_score": _safe_float(profile.get("next_metar_score")),
                "current_predictive_score": _safe_float(profile.get("predictive_score")),
                "current_rank_eligible": bool(profile.get("rank_eligible")),
                "current_learning_phase": profile.get("learning_phase"),
                "current_now_samples": _safe_int(profile.get("now_samples")),
                "current_lead_samples": _safe_int(profile.get("lead_samples")),
                "observed_rank_score": round(observed_rank, 4),
            }
        )
    return out


def build_top_pws_by_market(
    pws_metrics: Sequence[Dict[str, Any]],
    *,
    min_lead_rows: int = 8,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pws_metrics:
        grouped[str(row.get("market_station_id") or "").upper()].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for market_station_id, payload in sorted(grouped.items()):
        eligible = [row for row in payload if int(_safe_int(row.get("lead_row_count")) or 0) >= int(min_lead_rows)]
        if not eligible:
            eligible = list(payload)
        eligible.sort(
            key=lambda row: (
                float(_safe_float(row.get("lead_mae_c")) or 999.0),
                -float(_safe_float(row.get("lead_hit_rate_1_0_c")) or -1.0),
                -int(_safe_int(row.get("lead_row_count")) or 0),
                -float(_safe_float(row.get("observed_rank_score")) or -999.0),
                0 if _safe_bool(row.get("current_rank_eligible")) else 1,
                -float(_safe_float(row.get("current_weight_predictive")) or -1.0),
            ),
        )
        for rank, row in enumerate(eligible[:top_n], start=1):
            summary_rows.append(
                {
                    "market_station_id": market_station_id,
                    "rank": rank,
                    **row,
                }
            )
    return summary_rows


def summarize_profile_source_metrics(profile_rows: Sequence[Dict[str, Any]], observed_source_metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    observed_map = {str(row.get("source") or ""): row for row in observed_source_metrics}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        grouped[str(row.get("source") or "UNKNOWN")].append(row)
    out: List[Dict[str, Any]] = []
    for source, payload in sorted(grouped.items()):
        weights = [_safe_float(row.get("weight_predictive")) for row in payload]
        weights = [value for value in weights if value is not None]
        scores = [_safe_float(row.get("next_metar_score")) for row in payload]
        scores = [value for value in scores if value is not None]
        rank_ready = [1.0 if row.get("rank_eligible") else 0.0 for row in payload]
        observed = observed_map.get(source, {})
        out.append(
            {
                "source": source,
                "profile_count": len(payload),
                "avg_weight_predictive": statistics.fmean(weights) if weights else None,
                "avg_next_metar_score": statistics.fmean(scores) if scores else None,
                "rank_ready_rate": statistics.fmean(rank_ready) if rank_ready else None,
                "observed_mae_c": observed.get("mae_c"),
                "observed_hit_rate_1_0_c": observed.get("hit_rate_1_0_c"),
                "source_prior_learning": PWSLearningStore.SOURCE_PRIOR.get(source, PWSLearningStore.SOURCE_PRIOR.get("UNKNOWN")),
            }
        )
    return out


def summarize_prediction_duplicates(duplicates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not duplicates:
        return {"duplicate_key_count": 0, "duplicate_row_excess": 0, "max_rows_per_key": 1}
    excess = sum(max(0, int(item.get("rows") or 0) - 1) for item in duplicates)
    max_rows = max(int(item.get("rows") or 0) for item in duplicates)
    return {
        "duplicate_key_count": len(duplicates),
        "duplicate_row_excess": excess,
        "max_rows_per_key": max_rows,
    }


def build_recommendations(
    *,
    cycle_summary: Dict[str, Any],
    source_metrics: Sequence[Dict[str, Any]],
    lead_bucket_metrics: Sequence[Dict[str, Any]],
    profile_source_metrics: Sequence[Dict[str, Any]],
    duplicate_summary: Dict[str, Any],
    market_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []

    observed_by_source = {str(row.get("source") or ""): row for row in source_metrics}
    profile_by_source = {str(row.get("source") or ""): row for row in profile_source_metrics}
    if observed_by_source:
        sorted_by_mae = sorted(
            observed_by_source.values(),
            key=lambda row: float(_safe_float(row.get("mae_c")) or 999.0),
        )
        best_source = str(sorted_by_mae[0].get("source") or "")
        worst_source = str(sorted_by_mae[-1].get("source") or "")
        best_weight = _safe_float((profile_by_source.get(best_source) or {}).get("avg_weight_predictive"))
        worst_weight = _safe_float((profile_by_source.get(worst_source) or {}).get("avg_weight_predictive"))
        if best_weight is not None and worst_weight is not None and best_weight <= worst_weight:
            recommendations.append(
                {
                    "priority": "high",
                    "area": "source_priors",
                    "title": f"Retune source priors: {best_source} is outperforming but not receiving a stronger predictive weight than {worst_source}",
                    "evidence": f"Observed MAE {best_source}={_fmt_num(_safe_float(sorted_by_mae[0].get('mae_c')), 3)}C vs {worst_source}={_fmt_num(_safe_float(sorted_by_mae[-1].get('mae_c')), 3)}C; avg predictive weight {best_source}={_fmt_num(best_weight, 3)} vs {worst_source}={_fmt_num(worst_weight, 3)}.",
                }
            )

    bucket_map = {str(row.get("lead_bucket") or ""): row for row in lead_bucket_metrics}
    bucket_a = bucket_map.get("05-15")
    bucket_b = bucket_map.get("15-30")
    if bucket_a and bucket_b:
        mae_a = _safe_float(bucket_a.get("mae_c"))
        mae_b = _safe_float(bucket_b.get("mae_c"))
        hit_a = _safe_float(bucket_a.get("hit_rate_1_0_c"))
        hit_b = _safe_float(bucket_b.get("hit_rate_1_0_c"))
        if mae_a is not None and mae_b is not None and hit_a is not None and hit_b is not None and mae_b + 0.08 < mae_a and hit_b > hit_a + 0.03:
            recommendations.append(
                {
                    "priority": "high",
                    "area": "lead_windows",
                    "title": "Promote the 15-30 minute lead window in tactical weighting",
                    "evidence": f"Observed 15-30 bucket MAE={_fmt_num(mae_b, 3)}C and hit@1.0C={_fmt_pct(hit_b, 1)} beat 05-15 MAE={_fmt_num(mae_a, 3)}C and hit@1.0C={_fmt_pct(hit_a, 1)}.",
                }
            )

    allowed_market_precision = _safe_float(cycle_summary.get("tactical_allowed_market_precision"))
    blocked_market_precision = _safe_float(cycle_summary.get("tactical_blocked_market_precision"))
    allowed_rate = _safe_float(cycle_summary.get("tactical_allowed_rate"))
    if allowed_market_precision is not None and blocked_market_precision is not None and allowed_rate is not None:
        if allowed_market_precision < 0.55 and allowed_rate > 0.35:
            recommendations.append(
                {
                    "priority": "high",
                    "area": "repricing_gates",
                    "title": "Tighten tactical repricing gates",
                    "evidence": f"Allowed tactical cycles are only converting at {_fmt_pct(allowed_market_precision, 1)} on market direction while allowed rate is {_fmt_pct(allowed_rate, 1)}.",
                }
            )
        elif allowed_market_precision > blocked_market_precision + 0.10 and allowed_rate < 0.20:
            recommendations.append(
                {
                    "priority": "medium",
                    "area": "repricing_gates",
                    "title": "Relax tactical gating slightly to admit more good repricing setups",
                    "evidence": f"Allowed tactical precision {_fmt_pct(allowed_market_precision, 1)} is materially above blocked precision {_fmt_pct(blocked_market_precision, 1)} while only {_fmt_pct(allowed_rate, 1)} of projections are admitted.",
                }
            )

    if duplicate_summary.get("duplicate_key_count", 0) > 0:
        recommendations.append(
            {
                "priority": "medium",
                "area": "analytics_guardrail",
                "title": "Do not use predictions without explicit dedupe",
                "evidence": f"Found {duplicate_summary.get('duplicate_key_count')} duplicated prediction keys and {duplicate_summary.get('duplicate_row_excess')} excess rows; max rows per key {duplicate_summary.get('max_rows_per_key')}.",
            }
        )

    pending_counts = [_safe_int(row.get("pending_count")) or 0 for row in market_rows]
    if pending_counts and statistics.fmean(pending_counts) > 150:
        recommendations.append(
            {
                "priority": "medium",
                "area": "warmup_pending",
                "title": "Inspect pending queue churn and warmup progression in PWS learning",
                "evidence": f"Average pending queue size across markets is {statistics.fmean(pending_counts):.1f}, which is high for a 10-day tactical window.",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "priority": "low",
                "area": "status",
                "title": "No strong retuning signal detected in the current window",
                "evidence": "Observed source ranking, lead buckets and tactical gating stay within expected ranges for this sample.",
            }
        )

    order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda item: (order.get(str(item.get("priority") or "low"), 99), str(item.get("area") or "")))
    return recommendations


def render_markdown_report(
    *,
    config: "ResearchConfig",
    cycle_summary: Dict[str, Any],
    source_metrics: Sequence[Dict[str, Any]],
    lead_bucket_metrics: Sequence[Dict[str, Any]],
    station_metrics: Sequence[Dict[str, Any]],
    top_pws_by_market: Sequence[Dict[str, Any]],
    profile_source_metrics: Sequence[Dict[str, Any]],
    duplicate_summary: Dict[str, Any],
    recommendations: Sequence[Dict[str, Any]],
    market_rows: Sequence[Dict[str, Any]],
    api_validation: Optional[Dict[str, Any]] = None,
) -> str:
    lines: List[str] = []
    lines.append(f"# PWS Intraday Research Report ({config.date_from} to {config.date_to})")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Stations: {', '.join(config.station_ids)}")
    lines.append(f"- Remote host: `{config.host}`")
    lines.append(f"- Window: `{config.date_from}` to `{config.date_to}`")
    lines.append("- Focus: intraday PWS repricing, next-official accuracy, market reaction, and tactical gating")
    lines.append("")
    lines.append("## Headline Metrics")
    lines.append("")
    lines.append(f"- Cycles analyzed: {cycle_summary.get('cycle_count', 0)}")
    lines.append(f"- Cycles with reconstructable projection: {cycle_summary.get('projection_count', 0)}")
    lines.append(f"- Directional hit vs next official: {_fmt_pct(_safe_float(cycle_summary.get('directional_hit_temp')), 1)}")
    lines.append(f"- Directional hit vs market move before official: {_fmt_pct(_safe_float(cycle_summary.get('directional_hit_market')), 1)}")
    lines.append(f"- Correlation projected delta vs actual next-official move: {_fmt_num(_safe_float(cycle_summary.get('corr_projection_vs_temp_move')), 3)}")
    lines.append(f"- Correlation projected delta vs market move: {_fmt_num(_safe_float(cycle_summary.get('corr_projection_vs_market_move')), 3)}")
    lines.append(f"- Tactical allowed rate: {_fmt_pct(_safe_float(cycle_summary.get('tactical_allowed_rate')), 1)}")
    lines.append(f"- Tactical allowed market precision: {_fmt_pct(_safe_float(cycle_summary.get('tactical_allowed_market_precision')), 1)}")
    lines.append(f"- Tactical blocked market precision: {_fmt_pct(_safe_float(cycle_summary.get('tactical_blocked_market_precision')), 1)}")
    lines.append(f"- Average repricing influence: {_fmt_num(_safe_float(cycle_summary.get('avg_repricing_influence')), 3)}")
    lines.append("")
    lines.append("## Findings")
    lines.append("")

    if source_metrics:
        best_source = min(source_metrics, key=lambda row: float(_safe_float(row.get("mae_c")) or 999.0))
        worst_source = max(source_metrics, key=lambda row: float(_safe_float(row.get("mae_c")) or -999.0))
        lines.append(f"- Best observed source by MAE is `{best_source.get('source')}` at {_fmt_num(_safe_float(best_source.get('mae_c')), 3)}C; worst is `{worst_source.get('source')}` at {_fmt_num(_safe_float(worst_source.get('mae_c')), 3)}C.")
    if lead_bucket_metrics:
        ordered = sorted(lead_bucket_metrics, key=lambda row: float(_safe_float(row.get("mae_c")) or 999.0))
        winner = ordered[0]
        lines.append(f"- Best lead bucket in this window is `{winner.get('lead_bucket')}` with MAE {_fmt_num(_safe_float(winner.get('mae_c')), 3)}C and hit@1.0C {_fmt_pct(_safe_float(winner.get('hit_rate_1_0_c')), 1)}.")
    if station_metrics:
        strongest = max(station_metrics, key=lambda row: float(_safe_float(row.get("directional_hit_market")) or -999.0))
        weakest = min(station_metrics, key=lambda row: float(_safe_float(row.get("directional_hit_market")) or 999.0))
        lines.append(f"- Strongest market alignment is `{strongest.get('station_id')}` at {_fmt_pct(_safe_float(strongest.get('directional_hit_market')), 1)}; weakest is `{weakest.get('station_id')}` at {_fmt_pct(_safe_float(weakest.get('directional_hit_market')), 1)}.")
    if duplicate_summary.get("duplicate_key_count", 0) > 0:
        lines.append(f"- `predictions` is not safe as an analysis truth table without dedupe: {duplicate_summary.get('duplicate_key_count')} duplicate keys and {duplicate_summary.get('duplicate_row_excess')} excess rows were found in-window.")
    if market_rows:
        avg_pending = statistics.fmean([float(_safe_int(row.get("pending_count")) or 0) for row in market_rows])
        lines.append(f"- Current PWS learning state still carries an average pending queue of {avg_pending:.1f} observations per market.")
    lines.append("")
    lines.append("## Adjustments Prioritized")
    lines.append("")
    for item in recommendations:
        lines.append(f"1. [{str(item.get('priority') or '').upper()}] {item.get('title')}")
        lines.append(f"   Evidence: {item.get('evidence')}")
    lines.append("")
    lines.append("## Source Metrics")
    lines.append("")
    lines.append("| Source | Rows | MAE C | Bias C | Hit @0.5C | Hit @1.0C | Avg Predictive Weight | Avg Next-METAR Score | Prior |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    profile_map = {str(row.get("source") or ""): row for row in profile_source_metrics}
    for row in source_metrics:
        source = str(row.get("source") or "")
        profile = profile_map.get(source, {})
        lines.append(f"| {source} | {int(row.get('row_count') or 0)} | {_fmt_num(_safe_float(row.get('mae_c')), 3)} | {_fmt_num(_safe_float(row.get('bias_c')), 3)} | {_fmt_pct(_safe_float(row.get('hit_rate_0_5_c')), 1)} | {_fmt_pct(_safe_float(row.get('hit_rate_1_0_c')), 1)} | {_fmt_num(_safe_float(profile.get('avg_weight_predictive')), 3)} | {_fmt_num(_safe_float(profile.get('avg_next_metar_score')), 2)} | {_fmt_num(_safe_float(profile.get('source_prior_learning')), 2)} |")
    lines.append("")
    lines.append("## Lead Buckets")
    lines.append("")
    lines.append("| Lead Bucket | Rows | MAE C | Bias C | Hit @0.5C | Hit @1.0C |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in lead_bucket_metrics:
        lines.append(f"| {row.get('lead_bucket')} | {int(row.get('row_count') or 0)} | {_fmt_num(_safe_float(row.get('mae_c')), 3)} | {_fmt_num(_safe_float(row.get('bias_c')), 3)} | {_fmt_pct(_safe_float(row.get('hit_rate_0_5_c')), 1)} | {_fmt_pct(_safe_float(row.get('hit_rate_1_0_c')), 1)} |")
    lines.append("")
    lines.append("## Station Scorecard")
    lines.append("")
    lines.append("| Station | Cycles | Projections | Temp Dir Hit | Market Dir Hit | Helios vs Final MAE F | Market vs Final MAE F | Tactical Allowed Rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in station_metrics:
        lines.append(f"| {row.get('station_id')} | {int(row.get('cycle_count') or 0)} | {int(row.get('projection_count') or 0)} | {_fmt_pct(_safe_float(row.get('directional_hit_temp')), 1)} | {_fmt_pct(_safe_float(row.get('directional_hit_market')), 1)} | {_fmt_num(_safe_float(row.get('helios_vs_final_mae_f')), 2)} | {_fmt_num(_safe_float(row.get('market_vs_final_mae_f')), 2)} | {_fmt_pct(_safe_float(row.get('tactical_allowed_rate')), 1)} |")
    lines.append("")
    lines.append("## Best Individual PWS Stations By Market")
    lines.append("")
    lines.append("| Market | Rank | PWS Station | Source | Lead Rows | Lead MAE C | Hit @1.0C | Distance km | Current Predictive Weight | Current Next-METAR Score | Phase |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in top_pws_by_market:
        lines.append(
            f"| {row.get('market_station_id')} | {int(row.get('rank') or 0)} | {row.get('pws_station_id')} | {row.get('source')} | {int(_safe_int(row.get('lead_row_count')) or 0)} | {_fmt_num(_safe_float(row.get('lead_mae_c')), 3)} | {_fmt_pct(_safe_float(row.get('lead_hit_rate_1_0_c')), 1)} | {_fmt_num(_safe_float(row.get('distance_km')), 1)} | {_fmt_num(_safe_float(row.get('current_weight_predictive')), 3)} | {_fmt_num(_safe_float(row.get('current_next_metar_score')), 2)} | {row.get('current_learning_phase') or 'n/a'} |"
        )
    if api_validation:
        lines.append("")
        lines.append("## API Validation")
        lines.append("")
        lines.append(f"- API base URL: `{api_validation.get('api_base_url')}`")
        lines.append(f"- Stations validated: {', '.join(api_validation.get('validated_stations') or [])}")
        lines.append(f"- Aggregate API day count: {api_validation.get('day_count')}")
        lines.append(f"- Aggregate API audit rows: {api_validation.get('audit_row_total')}")
    lines.append("")
    lines.append("## Code Paths Reviewed")
    lines.append("")
    lines.append("- `core/pws_learning.py`: source priors, lead buckets, warmup, rank eligibility, pending queue behavior.")
    lines.append("- `core/trading_signal.py`: `estimate_next_metar_projection`, source/distance/stability factors, repricing influence.")
    lines.append("- `core/trading_bracket_market.py`: tactical market repricing application.")
    lines.append("- `core/autotrader.py` and `core/papertrader.py`: tactical filters using `minutes_to_next`, `delta_market`, and repricing influence.")
    lines.append("- `core/nowcast_engine.py`: checked only as a secondary control to keep PWS soft-anchor separate from tactical trading logic.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Historical audit rows preserve scored PWS observations, but not the historical online-learning state at each moment; current `weights.json` is used as the best available proxy for ranking and predictive weights.")
    lines.append("- Market movement is measured using `performance_logs.market_weighted_avg` and nearest snapshots around sample/reference and official times; this captures tradable drift in temperature space, not exact order-book execution.")
    lines.append("- `predictions` is treated as secondary because duplicate snapshots exist for the same `station_id + target_date + timestamp` key.")
    lines.append("")
    return "\n".join(lines)


@dataclass
class ResearchConfig:
    host: str
    identity_file: str
    date_from: str
    date_to: str
    station_ids: List[str]
    output_dir: Path
    remote_root: str = DEFAULT_REMOTE_ROOT
    remote_db_path: str = DEFAULT_REMOTE_DB
    remote_audit_dir: str = DEFAULT_REMOTE_AUDIT_DIR
    remote_weights_path: str = DEFAULT_REMOTE_WEIGHTS
    api_base_url: Optional[str] = None


class RemoteExtractor:
    def __init__(self, config: ResearchConfig):
        self.config = config
        self._ssh_binary = shutil.which("ssh")
        if not self._ssh_binary:
            raise RuntimeError("ssh executable not found in PATH")

    def _run_remote_python(self, code: str, payload: Dict[str, Any], *, stdout_path: Optional[Path] = None) -> str:
        payload_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        remote_cmd = (
            f"cd {self.config.remote_root} && "
            f"PAYLOAD_B64='{payload_b64}' /opt/helios/venv/bin/python -"
        )
        cmd = [
            self._ssh_binary,
            "-i",
            self.config.identity_file,
            self.config.host,
            remote_cmd,
        ]
        if stdout_path is not None:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            with stdout_path.open("w", encoding="utf-8", newline="") as handle:
                completed = subprocess.run(
                    cmd,
                    input=code,
                    text=True,
                    stdout=handle,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or f"Remote command failed with code {completed.returncode}")
            return ""
        completed = subprocess.run(
            cmd,
            input=code,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"Remote command failed with code {completed.returncode}")
        return completed.stdout

    def export_sqlite_csv(self, table_name: str, columns: Sequence[str], dest_path: Path) -> Path:
        code = f"""
import base64
import csv
import json
import os
import sqlite3
import sys

payload = json.loads(base64.b64decode(os.environ['PAYLOAD_B64']).decode('utf-8'))
conn = sqlite3.connect(payload['db_path'])
conn.row_factory = sqlite3.Row
station_ids = [str(item).upper() for item in payload['station_ids']]
date_from = payload['date_from']
date_to = payload['date_to']
table_name = payload['table_name']
columns = payload['columns']
placeholders = ','.join('?' for _ in station_ids)
query = f"SELECT {{', '.join(columns)}} FROM {{table_name}} WHERE station_id IN ({{placeholders}}) AND target_date >= ? AND target_date <= ? ORDER BY station_id, target_date, timestamp"
writer = csv.writer(sys.stdout, lineterminator='\\n')
writer.writerow(columns)
for row in conn.execute(query, station_ids + [date_from, date_to]):
    writer.writerow([row[col] for col in columns])
"""
        self._run_remote_python(
            code,
            {
                "db_path": self.config.remote_db_path,
                "station_ids": self.config.station_ids,
                "date_from": self.config.date_from,
                "date_to": self.config.date_to,
                "table_name": table_name,
                "columns": list(columns),
            },
            stdout_path=dest_path,
        )
        return dest_path

    def export_audit_csv(self, dest_path: Path) -> Path:
        code = """
import base64
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

payload = json.loads(base64.b64decode(os.environ['PAYLOAD_B64']).decode('utf-8'))
audit_dir = Path(payload['audit_dir'])
stations = {str(item).upper() for item in payload['station_ids']}
date_from = date.fromisoformat(payload['date_from'])
date_to = date.fromisoformat(payload['date_to'])
fieldnames = [
    'market_station_id', 'market_date', 'official_obs_time_utc', 'official_temp_c', 'official_temp_f',
    'station_id', 'station_name', 'source', 'sample_obs_time_utc', 'sample_temp_c', 'sample_temp_f',
    'distance_km', 'kind', 'lead_minutes', 'lead_bucket', 'signed_error_c', 'abs_error_c',
    'hit_within_0_5_c', 'hit_within_1_0_c', 'valid'
]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator='\\n')
writer.writeheader()
for date_dir in sorted(audit_dir.glob('date=*')):
    if not date_dir.is_dir():
        continue
    date_str = date_dir.name.replace('date=', '', 1)
    try:
        current = date.fromisoformat(date_str)
    except Exception:
        continue
    if current < date_from or current > date_to:
        continue
    for station_dir in sorted(date_dir.glob('station=*')):
        station_id = station_dir.name.replace('station=', '', 1).upper()
        if station_id not in stations:
            continue
        events_path = station_dir / 'events.ndjson'
        if not events_path.exists():
            continue
        for raw_line in events_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload_row = json.loads(line)
            except Exception:
                continue
            writer.writerow({key: payload_row.get(key) for key in fieldnames})
"""
        self._run_remote_python(
            code,
            {
                "audit_dir": self.config.remote_audit_dir,
                "station_ids": self.config.station_ids,
                "date_from": self.config.date_from,
                "date_to": self.config.date_to,
            },
            stdout_path=dest_path,
        )
        return dest_path

    def fetch_weights(self, dest_path: Path) -> Path:
        code = """
import base64
import json
import os
from pathlib import Path

payload = json.loads(base64.b64decode(os.environ['PAYLOAD_B64']).decode('utf-8'))
path = Path(payload['weights_path'])
print(path.read_text(encoding='utf-8'), end='')
"""
        text = self._run_remote_python(code, {"weights_path": self.config.remote_weights_path})
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(text, encoding="utf-8")
        return dest_path


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _fetch_api_validation(config: ResearchConfig) -> Optional[Dict[str, Any]]:
    if not config.api_base_url:
        return None
    from urllib.parse import urlencode
    from urllib.request import urlopen

    aggregate = {
        "api_base_url": config.api_base_url,
        "validated_stations": [],
        "day_count": 0,
        "audit_row_total": 0,
    }
    for station_id in config.station_ids:
        query = urlencode(
            {
                "station_id": station_id,
                "date_from": config.date_from,
                "date_to": config.date_to,
                "hydrate_metar": "false",
            }
        )
        url = f"{config.api_base_url.rstrip('/')}/api/pws/export/city?{query}"
        try:
            with urlopen(url, timeout=30) as response:
                payload = json.load(response)
        except Exception as exc:
            aggregate["error"] = str(exc)
            return aggregate
        aggregate["validated_stations"].append(station_id)
        summary = payload.get("summary") if isinstance(payload, dict) else {}
        aggregate["day_count"] += int((summary or {}).get("day_count") or 0)
        aggregate["audit_row_total"] += int((summary or {}).get("audit_row_total") or 0)
    return aggregate


def run_research(config: ResearchConfig) -> Dict[str, Path]:
    extractor = RemoteExtractor(config)
    output_dir = config.output_dir
    raw_dir = output_dir / "raw"
    derived_dir = output_dir / "derived"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    performance_path = extractor.export_sqlite_csv(
        "performance_logs",
        [
            "timestamp", "station_id", "target_date", "helios_pred", "metar_actual", "wu_preliminary",
            "market_top1_bracket", "market_top1_prob", "market_top2_bracket", "market_top2_prob",
            "market_top3_bracket", "market_top3_prob", "market_weighted_avg", "cumulative_max_f",
            "wu_final", "confidence_score", "hrrr_max_raw_f", "current_deviation_f",
            "physics_adjustment_f", "verified_floor_f", "physics_reason", "market_all_brackets_json",
            "nbm_max_f", "lamp_max_f", "ensemble_base_f", "ensemble_floor_f", "ensemble_ceiling_f",
            "ensemble_spread_f", "ensemble_confidence",
        ],
        raw_dir / "performance_logs.csv",
    )
    predictions_path = extractor.export_sqlite_csv(
        "predictions",
        [
            "timestamp", "station_id", "target_date", "final_prediction_f", "hrrr_max_raw_f",
            "current_deviation_f", "physics_adjustment_f", "physics_reason", "verified_floor_f",
            "real_max_verified_f",
        ],
        raw_dir / "predictions.csv",
    )
    market_velocity_path = extractor.export_sqlite_csv(
        "market_velocity",
        ["timestamp", "station_id", "target_date", "bracket_name", "probability"],
        raw_dir / "market_velocity.csv",
    )
    audit_path = extractor.export_audit_csv(raw_dir / "pws_audit.csv")
    weights_path = extractor.fetch_weights(raw_dir / "weights.json")

    performance_rows = _load_csv(performance_path)
    prediction_rows = _load_csv(predictions_path)
    market_velocity_rows = _load_csv(market_velocity_path)
    audit_rows = _load_csv(audit_path)
    weights_payload = json.loads(weights_path.read_text(encoding="utf-8"))

    deduped_predictions, duplicate_rows = dedupe_predictions(prediction_rows)
    duplicate_summary = summarize_prediction_duplicates(duplicate_rows)
    perf_index = _build_performance_index(performance_rows)
    velocity_index = _build_market_velocity_index(market_velocity_rows)
    finals = _final_real_by_station_date(perf_index)
    profile_map, profile_rows, market_rows = _build_profile_maps(weights_payload, config.station_ids)
    cycle_rows = build_cycle_rows(audit_rows, perf_index, profile_map, velocity_index)
    cycle_summary = summarize_cycle_metrics(cycle_rows)
    source_metrics = _aggregate_stats(audit_rows, group_key="source")
    lead_bucket_metrics = _aggregate_stats([row for row in audit_rows if str(row.get("kind") or "").upper() == "LEAD"], group_key="lead_bucket")
    station_metrics = summarize_station_metrics(cycle_rows, finals)
    individual_pws_metrics = summarize_individual_pws_metrics(audit_rows, profile_map)
    top_pws_by_market = build_top_pws_by_market(individual_pws_metrics)
    profile_source_metrics = summarize_profile_source_metrics(profile_rows, source_metrics)
    recommendations = build_recommendations(
        cycle_summary=cycle_summary,
        source_metrics=source_metrics,
        lead_bucket_metrics=lead_bucket_metrics,
        profile_source_metrics=profile_source_metrics,
        duplicate_summary=duplicate_summary,
        market_rows=market_rows,
    )
    api_validation = _fetch_api_validation(config)

    deduped_prediction_path = derived_dir / "predictions_dedup.csv"
    _csv_write(deduped_prediction_path, deduped_predictions, [
        "timestamp", "station_id", "target_date", "final_prediction_f", "hrrr_max_raw_f",
        "current_deviation_f", "physics_adjustment_f", "physics_reason", "verified_floor_f",
        "real_max_verified_f",
    ])
    duplicate_path = derived_dir / "prediction_duplicates.csv"
    _csv_write(duplicate_path, duplicate_rows, ["station_id", "target_date", "timestamp", "rows"])
    cycle_path = derived_dir / "cycle_metrics.csv"
    _csv_write(cycle_path, cycle_rows, sorted({key for row in cycle_rows for key in row.keys()}))
    source_path = derived_dir / "source_metrics.csv"
    _csv_write(source_path, source_metrics, ["source", "row_count", "station_count", "mae_c", "bias_c", "hit_rate_0_5_c", "hit_rate_1_0_c", "p50_abs_error_c", "p90_abs_error_c"])
    lead_path = derived_dir / "lead_bucket_metrics.csv"
    _csv_write(lead_path, lead_bucket_metrics, ["lead_bucket", "row_count", "station_count", "mae_c", "bias_c", "hit_rate_0_5_c", "hit_rate_1_0_c", "p50_abs_error_c", "p90_abs_error_c"])
    station_path = derived_dir / "station_metrics.csv"
    _csv_write(station_path, station_metrics, ["station_id", "cycle_count", "projection_count", "directional_hit_temp", "directional_hit_market", "helios_vs_final_mae_f", "market_vs_final_mae_f", "tactical_allowed_rate"])
    pws_station_path = derived_dir / "pws_station_metrics.csv"
    _csv_write(pws_station_path, individual_pws_metrics, [
        "market_station_id", "pws_station_id", "source", "station_name", "distance_km", "row_count",
        "lead_row_count", "now_row_count", "lead_mae_c", "lead_bias_c", "lead_hit_rate_0_5_c",
        "lead_hit_rate_1_0_c", "lead_avg_minutes", "now_mae_c", "now_bias_c", "now_hit_rate_0_5_c",
        "now_hit_rate_1_0_c", "current_weight_predictive", "current_weight", "current_next_metar_score",
        "current_predictive_score", "current_rank_eligible", "current_learning_phase",
        "current_now_samples", "current_lead_samples", "observed_rank_score",
    ])
    top_pws_path = derived_dir / "top_pws_by_market.csv"
    _csv_write(top_pws_path, top_pws_by_market, [
        "market_station_id", "rank", "pws_station_id", "source", "station_name", "distance_km",
        "lead_row_count", "lead_mae_c", "lead_hit_rate_1_0_c", "lead_avg_minutes",
        "current_weight_predictive", "current_next_metar_score", "current_rank_eligible",
        "current_learning_phase", "observed_rank_score",
    ])
    profile_path = derived_dir / "current_profile_metrics.csv"
    _csv_write(profile_path, profile_source_metrics, ["source", "profile_count", "avg_weight_predictive", "avg_next_metar_score", "rank_ready_rate", "observed_mae_c", "observed_hit_rate_1_0_c", "source_prior_learning"])
    market_state_path = derived_dir / "market_learning_state.csv"
    _csv_write(market_state_path, market_rows, ["market_station_id", "tracked_station_count", "pending_count", "updated_at", "last_official_obs_time_utc"])

    report_text = render_markdown_report(
        config=config,
        cycle_summary=cycle_summary,
        source_metrics=source_metrics,
        lead_bucket_metrics=lead_bucket_metrics,
        station_metrics=station_metrics,
        top_pws_by_market=top_pws_by_market,
        profile_source_metrics=profile_source_metrics,
        duplicate_summary=duplicate_summary,
        recommendations=recommendations,
        market_rows=market_rows,
        api_validation=api_validation,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "cycle_summary": cycle_summary,
                "duplicate_summary": duplicate_summary,
                "recommendations": recommendations,
                "api_validation": api_validation,
                "market_velocity_row_count": len(market_velocity_rows),
                "output_files": {
                    "report": str(report_path),
                    "cycle_metrics": str(cycle_path),
                    "source_metrics": str(source_path),
                    "lead_bucket_metrics": str(lead_path),
                    "station_metrics": str(station_path),
                    "pws_station_metrics": str(pws_station_path),
                    "top_pws_by_market": str(top_pws_path),
                },
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "report": report_path,
        "summary": summary_path,
        "performance_logs": performance_path,
        "predictions_raw": predictions_path,
        "predictions_dedup": deduped_prediction_path,
        "prediction_duplicates": duplicate_path,
        "market_velocity": market_velocity_path,
        "audit_rows": audit_path,
        "weights": weights_path,
        "cycle_metrics": cycle_path,
        "source_metrics": source_path,
        "lead_bucket_metrics": lead_path,
        "station_metrics": station_path,
        "pws_station_metrics": pws_station_path,
        "top_pws_by_market": top_pws_path,
        "current_profile_metrics": profile_path,
        "market_learning_state": market_state_path,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> ResearchConfig:
    parser = argparse.ArgumentParser(description="Run read-only PWS intraday research against a remote Helios instance.")
    parser.add_argument("--host", required=True, help="SSH host, e.g. ubuntu@3.253.86.168")
    parser.add_argument("--identity-file", required=True, help="SSH identity file path")
    parser.add_argument("--date-from", default="2026-02-27")
    parser.add_argument("--date-to", default="2026-03-08")
    parser.add_argument("--stations", default=",".join(DEFAULT_STATIONS), help="Comma-separated station IDs")
    parser.add_argument("--output-dir", default=None, help="Output directory under forense/")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--remote-db-path", default=DEFAULT_REMOTE_DB)
    parser.add_argument("--remote-audit-dir", default=DEFAULT_REMOTE_AUDIT_DIR)
    parser.add_argument("--remote-weights-path", default=DEFAULT_REMOTE_WEIGHTS)
    parser.add_argument("--api-base-url", default=None, help="Optional base URL for validating /api/pws/export/city")
    args = parser.parse_args(argv)

    station_ids = [item.strip().upper() for item in str(args.stations or "").split(",") if item.strip()]
    if not station_ids:
        raise SystemExit("No stations provided")
    for station_id in station_ids:
        if station_id not in STATIONS:
            raise SystemExit(f"Unknown station_id: {station_id}")
    output_dir = Path(args.output_dir) if args.output_dir else Path("forense") / f"pws_intraday_{args.date_from}_{args.date_to}"
    return ResearchConfig(
        host=args.host,
        identity_file=args.identity_file,
        date_from=args.date_from,
        date_to=args.date_to,
        station_ids=station_ids,
        output_dir=output_dir,
        remote_root=args.remote_root,
        remote_db_path=args.remote_db_path,
        remote_audit_dir=args.remote_audit_dir,
        remote_weights_path=args.remote_weights_path,
        api_base_url=args.api_base_url,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = _parse_args(argv)
    outputs = run_research(config)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
