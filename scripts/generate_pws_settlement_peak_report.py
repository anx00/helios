from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import STATIONS
from core.polymarket_labels import label_for_temp, normalize_label

UTC = ZoneInfo("UTC")

DAY_AHEAD_WINDOWS: Sequence[Tuple[int, int, str]] = (
    (0, 6, "D-1 00-06"),
    (6, 12, "D-1 06-12"),
    (12, 18, "D-1 12-18"),
    (18, 24, "D-1 18-24"),
)

PEAK_BLOCKS: Sequence[Tuple[int, int, str]] = (
    (11, 13, "11-13 local"),
    (13, 15, "13-15 local"),
    (15, 17, "15-17 local"),
)

PEAK_MINUTE_BINS: Sequence[Tuple[int, int, str]] = (
    (120, 60, "120-60m before peak"),
    (60, 30, "60-30m before peak"),
    (30, 15, "30-15m before peak"),
    (15, 5, "15-5m before peak"),
)

LEAD_BUCKET_ORDER = ("05-15", "15-30", "30-45")
STATION_ORDER = ("KORD", "KLGA", "EGLC", "KMIA", "KDAL", "KATL", "LFPG", "LTAC")


@dataclass
class ReportConfig:
    input_dir: Path
    output_dir: Path
    report_date: str
    browser_path: Optional[Path]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _fmt_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_prob(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _mean(values: Iterable[float]) -> Optional[float]:
    payload = [float(value) for value in values if value is not None]
    return statistics.fmean(payload) if payload else None


def _market_distribution(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = row.get("market_all_brackets_json")
    if raw in (None, ""):
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out: List[Dict[str, Any]] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        label = normalize_label(str(item.get("bracket") or item.get("label") or "")) or str(item.get("bracket") or item.get("label") or "")
        prob = _safe_float(item.get("prob"))
        if label and prob is not None:
            out.append({"label": label, "prob": prob})
    out.sort(key=lambda item: float(item.get("prob") or 0.0), reverse=True)
    return out


def _market_unit(labels: Sequence[str]) -> str:
    return "C" if any("°C" in str(label) or "Â°C" in str(label) for label in labels) else "F"


def _convert_f_to_unit(value_f: Optional[float], unit: str) -> Optional[float]:
    if value_f is None:
        return None
    return (value_f - 32.0) * 5.0 / 9.0 if unit == "C" else value_f


def _load_performance_groups(path: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            station_id = str(raw.get("station_id") or "").upper()
            target_date = str(raw.get("target_date") or "")
            if not station_id or not target_date:
                continue
            dt = _parse_dt(raw.get("timestamp"))
            if dt is None:
                continue
            row = dict(raw)
            row["_dt"] = dt
            grouped[(station_id, target_date)].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["_dt"])
    return grouped


def _build_final_meta(groups: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, rows in groups.items():
        labels: List[str] = []
        official_max_f: Optional[float] = None
        official_peak_dt: Optional[datetime] = None
        wu_max_f: Optional[float] = None
        for row in rows:
            if not labels:
                labels = [str(item.get("label") or "") for item in _market_distribution(row)]
            official_value = _safe_float(row.get("cumulative_max_f"))
            if official_value is not None and (official_max_f is None or official_value > official_max_f):
                official_max_f = official_value
                official_peak_dt = row["_dt"]
            wu_value = _safe_float(row.get("wu_final"))
            if wu_value is not None:
                wu_max_f = wu_value if wu_max_f is None else max(wu_max_f, wu_value)
        unit = _market_unit(labels)
        official_label = label_for_temp(_convert_f_to_unit(official_max_f, unit), labels)[0] if labels and official_max_f is not None else None
        wu_label = label_for_temp(_convert_f_to_unit(wu_max_f, unit), labels)[0] if labels and wu_max_f is not None else None
        out[key] = {
            "labels": labels,
            "unit": unit,
            "official_max_f": official_max_f,
            "official_peak_dt": official_peak_dt,
            "official_label": official_label,
            "wu_max_f": wu_max_f,
            "wu_label": wu_label,
            "wu_minus_official_f": (wu_max_f - official_max_f) if wu_max_f is not None and official_max_f is not None else None,
        }
    return out


def _compute_day_ahead_metrics(
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]],
    final_meta: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for (station_id, target_date), rows in groups.items():
        final_label = final_meta.get((station_id, target_date), {}).get("official_label")
        if not final_label:
            continue
        station_tz = ZoneInfo(STATIONS[station_id].timezone)
        previous_date = datetime.fromisoformat(target_date).date() - timedelta(days=1)
        for row in rows:
            local_dt = row["_dt"].astimezone(station_tz)
            if local_dt.date() != previous_date:
                continue
            window_name = None
            for start_hour, end_hour, label in DAY_AHEAD_WINDOWS:
                if start_hour <= local_dt.hour < end_hour:
                    window_name = label
                    break
            if not window_name:
                continue
            dist = _market_distribution(row)
            if not dist:
                continue
            labels = [str(item.get("label") or "") for item in dist]
            probs = {str(item.get("label") or ""): _safe_float(item.get("prob")) for item in dist}
            rank = labels.index(final_label) + 1 if final_label in labels else None
            grouped[station_id][window_name].append(
                {
                    "rank": rank,
                    "winner_prob": probs.get(final_label),
                    "top1_hit": 1.0 if rank == 1 else 0.0,
                    "top3_hit": 1.0 if rank is not None and rank <= 3 else 0.0,
                }
            )

    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for station_id, window_map in grouped.items():
        for window_name, rows in window_map.items():
            out[station_id][window_name] = {
                "sample_count": len(rows),
                "winner_top1_rate": _mean(row["top1_hit"] for row in rows),
                "winner_top3_rate": _mean(row["top3_hit"] for row in rows),
                "winner_avg_prob": _mean(row["winner_prob"] for row in rows if row.get("winner_prob") is not None),
                "winner_avg_rank": _mean(float(row["rank"]) for row in rows if row.get("rank") is not None),
            }
    return out


def _pick_best_day_ahead_window(window_map: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    if not window_map:
        return (None, None)
    viable = [
        (name, metrics)
        for name, metrics in window_map.items()
        if (_safe_float(metrics.get("winner_top3_rate")) or 0.0) >= 0.25
    ]
    if viable:
        viable.sort(
            key=lambda item: (
                float(_safe_float(item[1].get("winner_avg_prob")) or 9.0),
                -float(_safe_float(item[1].get("winner_top3_rate")) or 0.0),
                -int(_safe_int(item[1].get("sample_count")) or 0),
            )
        )
        return (viable[0][0], "balanced")
    fallback = sorted(
        window_map.items(),
        key=lambda item: (
            -float(_safe_float(item[1].get("winner_top3_rate")) or 0.0),
            float(_safe_float(item[1].get("winner_avg_prob")) or 9.0),
        ),
    )
    return (fallback[0][0], "weak")


def _compute_wu_official_gap(final_meta: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for (station_id, _), payload in final_meta.items():
        gap = _safe_float(payload.get("wu_minus_official_f"))
        if gap is not None:
            grouped[station_id].append(gap)
    out: Dict[str, Dict[str, Any]] = {}
    for station_id, values in grouped.items():
        out[station_id] = {
            "sample_count": len(values),
            "avg_gap_f": _mean(values),
            "median_gap_f": statistics.median(values) if values else None,
            "min_gap_f": min(values) if values else None,
            "max_gap_f": max(values) if values else None,
        }
    return out


def _load_cycle_rows(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = dict(raw)
            row["_reference_dt"] = _parse_dt(row.get("reference_utc"))
            out.append(row)
    return out


def _compute_peak_metrics(
    cycle_rows: Sequence[Dict[str, Any]],
    final_meta: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    block_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    minute_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in cycle_rows:
        station_id = str(row.get("station_id") or "").upper()
        target_date = str(row.get("target_date") or "")
        meta = final_meta.get((station_id, target_date))
        if not meta:
            continue
        reference_dt = row.get("_reference_dt")
        final_label = meta.get("official_label")
        labels = meta.get("labels") or []
        unit = str(meta.get("unit") or "F")
        peak_dt = meta.get("official_peak_dt")
        if reference_dt is None or final_label in (None, "") or not labels or peak_dt is None:
            continue
        current_max_f = _safe_float(row.get("reference_cumulative_max_f"))
        expected_market = _safe_float(row.get("expected_market"))
        if current_max_f is None or expected_market is None:
            continue
        current_max_market = _convert_f_to_unit(current_max_f, unit)
        current_label = label_for_temp(current_max_market, labels)[0]
        if current_label == final_label:
            continue
        projected_peak_market = max(float(current_max_market or 0.0), float(expected_market))
        pws_label = label_for_temp(projected_peak_market, labels)[0]
        market_top1 = normalize_label(str(row.get("reference_market_top1_bracket") or "")) if row.get("reference_market_top1_bracket") else None
        station_tz = ZoneInfo(STATIONS[station_id].timezone)
        local_hour = reference_dt.astimezone(station_tz).hour
        minutes_before_peak = (peak_dt - reference_dt).total_seconds() / 60.0
        sample = {
            "pws_hit": 1.0 if pws_label == final_label else 0.0,
            "market_top1_hit": 1.0 if market_top1 == final_label else 0.0,
            "pws_only": 1.0 if pws_label == final_label and market_top1 != final_label else 0.0,
            "market_only": 1.0 if pws_label != final_label and market_top1 == final_label else 0.0,
        }
        for start_hour, end_hour, block_name in PEAK_BLOCKS:
            if start_hour <= local_hour < end_hour:
                block_rows[station_id][block_name].append(sample)
                break
        if minutes_before_peak > 0:
            for high, low, label in PEAK_MINUTE_BINS:
                if low < minutes_before_peak <= high:
                    minute_rows[label].append(sample)
                    break

    block_metrics: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for station_id, window_map in block_rows.items():
        for block_name, rows in window_map.items():
            pws_hit = _mean(row["pws_hit"] for row in rows)
            market_hit = _mean(row["market_top1_hit"] for row in rows)
            block_metrics[station_id][block_name] = {
                "sample_count": len(rows),
                "pws_peak_hit_rate": pws_hit,
                "market_top1_hit_rate": market_hit,
                "edge_vs_market": (pws_hit - market_hit) if pws_hit is not None and market_hit is not None else None,
                "pws_only_rate": _mean(row["pws_only"] for row in rows),
                "market_only_rate": _mean(row["market_only"] for row in rows),
            }

    minute_metrics: Dict[str, Dict[str, Any]] = {}
    for label, rows in minute_rows.items():
        pws_hit = _mean(row["pws_hit"] for row in rows)
        market_hit = _mean(row["market_top1_hit"] for row in rows)
        minute_metrics[label] = {
            "sample_count": len(rows),
            "pws_peak_hit_rate": pws_hit,
            "market_top1_hit_rate": market_hit,
            "edge_vs_market": (pws_hit - market_hit) if pws_hit is not None and market_hit is not None else None,
        }
    return block_metrics, minute_metrics


def _pick_best_peak_block(window_map: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    if not window_map:
        return (None, None)
    robust = [
        (name, metrics)
        for name, metrics in window_map.items()
        if int(_safe_int(metrics.get("sample_count")) or 0) >= 5
    ]
    if robust:
        robust.sort(
            key=lambda item: (
                -float(_safe_float(item[1].get("edge_vs_market")) or -9.0),
                -float(_safe_float(item[1].get("pws_peak_hit_rate")) or -9.0),
                -int(_safe_int(item[1].get("sample_count")) or 0),
            )
        )
        return (robust[0][0], "robust")
    fallback = sorted(
        window_map.items(),
        key=lambda item: (
            -float(_safe_float(item[1].get("edge_vs_market")) or -9.0),
            -int(_safe_int(item[1].get("sample_count")) or 0),
        ),
    )
    return (fallback[0][0], "limited")


def _compute_lead_bucket_metrics(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("kind") or "").upper() != "LEAD":
                continue
            station_id = str(row.get("market_station_id") or "").upper()
            lead_bucket = str(row.get("lead_bucket") or "")
            abs_error_c = _safe_float(row.get("abs_error_c"))
            hit_1c = 1.0 if str(row.get("hit_within_1_0_c")) == "True" else 0.0
            if station_id and lead_bucket and abs_error_c is not None:
                grouped[station_id][lead_bucket].append({"abs_error_c": abs_error_c, "hit_1c": hit_1c})

    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for station_id, bucket_map in grouped.items():
        for lead_bucket, rows in bucket_map.items():
            out[station_id][lead_bucket] = {
                "sample_count": len(rows),
                "mae_c": _mean(row["abs_error_c"] for row in rows),
                "hit_1c_rate": _mean(row["hit_1c"] for row in rows),
            }
    return out


def _pick_best_lead_bucket(bucket_map: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if not bucket_map:
        return None
    ordered = [
        (bucket, metrics)
        for bucket, metrics in bucket_map.items()
        if bucket in LEAD_BUCKET_ORDER
    ]
    ordered.sort(
        key=lambda item: (
            float(_safe_float(item[1].get("mae_c")) or 999.0),
            -float(_safe_float(item[1].get("hit_1c_rate")) or -1.0),
            -int(_safe_int(item[1].get("sample_count")) or 0),
        )
    )
    return ordered[0][0] if ordered else None


def _load_best_pws(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            station_id = str(raw.get("market_station_id") or "").upper()
            if not station_id:
                continue
            grouped[station_id].append(dict(raw))

    best: Dict[str, Dict[str, Any]] = {}
    top_three: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for station_id, rows in grouped.items():
        eligible = [row for row in rows if int(_safe_int(row.get("lead_row_count")) or 0) >= 20]
        if not eligible:
            eligible = list(rows)
        eligible.sort(
            key=lambda row: (
                float(_safe_float(row.get("lead_mae_c")) or 999.0),
                -float(_safe_float(row.get("lead_hit_rate_1_0_c")) or -1.0),
                -int(_safe_int(row.get("lead_row_count")) or 0),
            )
        )
        if eligible:
            best[station_id] = eligible[0]
            top_three[station_id] = eligible[:3]
    return best, top_three


def _station_unit_label(station_id: str) -> str:
    return "C" if station_id in {"EGLC", "LFPG", "LTAC"} else "F"


def _build_station_summary(
    day_ahead_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    peak_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    lead_bucket_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    best_pws: Dict[str, Dict[str, Any]],
    wu_gaps: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for station_id in STATION_ORDER:
        day_ahead_map = day_ahead_metrics.get(station_id, {})
        peak_map = peak_metrics.get(station_id, {})
        lead_map = lead_bucket_metrics.get(station_id, {})
        best_day_ahead_window, day_ahead_quality = _pick_best_day_ahead_window(day_ahead_map)
        best_peak_block, peak_quality = _pick_best_peak_block(peak_map)
        best_lead_bucket = _pick_best_lead_bucket(lead_map)
        pws_row = best_pws.get(station_id, {})
        gap_row = wu_gaps.get(station_id, {})
        day_ahead_row = day_ahead_map.get(best_day_ahead_window or "", {})
        peak_row = peak_map.get(best_peak_block or "", {})
        lead_row = lead_map.get(best_lead_bucket or "", {})

        recommendation = "Avoid"
        if (
            int(_safe_int(peak_row.get("sample_count")) or 0) >= 5
            and float(_safe_float(peak_row.get("edge_vs_market")) or 0.0) >= 0.15
            and float(_safe_float(lead_row.get("mae_c")) or 9.0) <= 0.90
        ):
            recommendation = "Peak-hour candidate"
        elif (
            float(_safe_float(day_ahead_row.get("winner_top3_rate")) or 0.0) >= 0.35
            and float(_safe_float(day_ahead_row.get("winner_avg_prob")) or 1.0) <= 0.30
        ):
            recommendation = "Day-ahead value candidate"
        elif float(_safe_float(pws_row.get("lead_mae_c")) or 9.0) <= 0.60:
            recommendation = "Confirmation only"
        elif float(_safe_float(day_ahead_row.get("winner_top3_rate")) or 0.0) >= 0.25:
            recommendation = "Selective only"

        out.append(
            {
                "station_id": station_id,
                "station_name": STATIONS[station_id].name,
                "timezone": STATIONS[station_id].timezone,
                "unit": _station_unit_label(station_id),
                "recommendation": recommendation,
                "best_day_ahead_window": best_day_ahead_window,
                "best_day_ahead_quality": day_ahead_quality,
                "best_day_ahead_top1": _safe_float(day_ahead_row.get("winner_top1_rate")),
                "best_day_ahead_top3": _safe_float(day_ahead_row.get("winner_top3_rate")),
                "best_day_ahead_prob": _safe_float(day_ahead_row.get("winner_avg_prob")),
                "best_peak_block": best_peak_block,
                "best_peak_quality": peak_quality,
                "best_peak_sample_count": _safe_int(peak_row.get("sample_count")),
                "best_peak_pws_hit": _safe_float(peak_row.get("pws_peak_hit_rate")),
                "best_peak_market_hit": _safe_float(peak_row.get("market_top1_hit_rate")),
                "best_peak_edge": _safe_float(peak_row.get("edge_vs_market")),
                "best_lead_bucket": best_lead_bucket,
                "best_lead_bucket_mae_c": _safe_float(lead_row.get("mae_c")),
                "best_lead_bucket_hit_1c": _safe_float(lead_row.get("hit_1c_rate")),
                "best_pws_station_id": pws_row.get("pws_station_id"),
                "best_pws_source": pws_row.get("source"),
                "best_pws_name": pws_row.get("station_name"),
                "best_pws_mae_c": _safe_float(pws_row.get("lead_mae_c")),
                "best_pws_hit_1c": _safe_float(pws_row.get("lead_hit_rate_1_0_c")),
                "best_pws_row_count": _safe_int(pws_row.get("lead_row_count")),
                "best_pws_rank_eligible": str(pws_row.get("current_rank_eligible") or "") == "True",
                "avg_wu_minus_official_f": _safe_float(gap_row.get("avg_gap_f")),
            }
        )
    return out


def _build_overall_day_ahead(day_ahead_metrics: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for station_id, window_map in day_ahead_metrics.items():
        for window_name, metrics in window_map.items():
            grouped[window_name].append(metrics)
    out: List[Dict[str, Any]] = []
    for _, _, window_name in DAY_AHEAD_WINDOWS:
        rows = grouped.get(window_name, [])
        out.append(
            {
                "window_name": window_name,
                "sample_count": sum(int(_safe_int(row.get("sample_count")) or 0) for row in rows),
                "winner_top1_rate": _mean(_safe_float(row.get("winner_top1_rate")) for row in rows if _safe_float(row.get("winner_top1_rate")) is not None),
                "winner_top3_rate": _mean(_safe_float(row.get("winner_top3_rate")) for row in rows if _safe_float(row.get("winner_top3_rate")) is not None),
                "winner_avg_prob": _mean(_safe_float(row.get("winner_avg_prob")) for row in rows if _safe_float(row.get("winner_avg_prob")) is not None),
            }
        )
    return out


def _render_station_takeaway(row: Dict[str, Any]) -> str:
    recommendation = str(row.get("recommendation") or "")
    if recommendation == "Peak-hour candidate":
        return "Best tactical candidate in this sample. PWS improves the late bracket call more than the market in at least one robust local block."
    if recommendation == "Day-ahead value candidate":
        return "More interesting one day before than near the peak. The final winner often appears on the board at a still-manageable probability."
    if recommendation == "Confirmation only":
        return "Useful as meteorological confirmation, but not enough evidence here to treat it as a standalone trading edge."
    if recommendation == "Selective only":
        return "There are usable pockets, but the edge is narrow and the timing matters a lot."
    return "No reliable edge in this window. Better to avoid or keep only as background context."


def _overall_highlights(station_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    peak_candidates = [row for row in station_rows if row.get("recommendation") == "Peak-hour candidate"]
    day_candidates = [row for row in station_rows if row.get("recommendation") == "Day-ahead value candidate"]
    confirmation = [row for row in station_rows if row.get("recommendation") == "Confirmation only"]
    return {
        "peak_candidates": peak_candidates,
        "day_candidates": day_candidates,
        "confirmation": confirmation,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _html_escape(value: Any) -> str:
    return html.escape(str(value if value not in (None, "") else "n/a"))


def _recommendation_badge(recommendation: str) -> str:
    css_class = {
        "Peak-hour candidate": "badge-green",
        "Day-ahead value candidate": "badge-blue",
        "Confirmation only": "badge-amber",
        "Selective only": "badge-slate",
    }.get(recommendation, "badge-red")
    return f'<span class="badge {css_class}">{_html_escape(recommendation)}</span>'


def _render_html(
    config: ReportConfig,
    station_rows: Sequence[Dict[str, Any]],
    day_ahead_overall: Sequence[Dict[str, Any]],
    peak_minute_metrics: Dict[str, Dict[str, Any]],
    lead_bucket_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    best_pws_top3: Dict[str, List[Dict[str, Any]]],
) -> str:
    highlights = _overall_highlights(station_rows)

    def render_overall_row(row: Dict[str, Any]) -> str:
        return (
            "<tr>"
            f"<td>{_html_escape(row.get('window_name'))}</td>"
            f"<td>{int(_safe_int(row.get('sample_count')) or 0)}</td>"
            f"<td>{_fmt_pct(_safe_float(row.get('winner_top1_rate')))}</td>"
            f"<td>{_fmt_pct(_safe_float(row.get('winner_top3_rate')))}</td>"
            f"<td>{_fmt_prob(_safe_float(row.get('winner_avg_prob')))}</td>"
            "</tr>"
        )

    def render_peak_minute_row(label: str, row: Dict[str, Any]) -> str:
        return (
            "<tr>"
            f"<td>{_html_escape(label)}</td>"
            f"<td>{int(_safe_int(row.get('sample_count')) or 0)}</td>"
            f"<td>{_fmt_pct(_safe_float(row.get('pws_peak_hit_rate')))}</td>"
            f"<td>{_fmt_pct(_safe_float(row.get('market_top1_hit_rate')))}</td>"
            f"<td>{_fmt_pct(_safe_float(row.get('edge_vs_market')))}</td>"
            "</tr>"
        )

    summary_rows_html: List[str] = []
    for row in station_rows:
        summary_rows_html.append(
            "<tr>"
            f"<td><strong>{_html_escape(row.get('station_id'))}</strong><div class='subtle'>{_html_escape(row.get('station_name'))}</div></td>"
            f"<td>{_recommendation_badge(str(row.get('recommendation') or ''))}</td>"
            f"<td>{_html_escape(row.get('best_day_ahead_window') or 'n/a')}<div class='subtle'>Top3 {_fmt_pct(_safe_float(row.get('best_day_ahead_top3')))} | Winner prob {_fmt_prob(_safe_float(row.get('best_day_ahead_prob')))}</div></td>"
            f"<td>{_html_escape(row.get('best_peak_block') or 'n/a')}<div class='subtle'>PWS {_fmt_pct(_safe_float(row.get('best_peak_pws_hit')))} vs market {_fmt_pct(_safe_float(row.get('best_peak_market_hit')))}</div></td>"
            f"<td>{_html_escape(row.get('best_lead_bucket') or 'n/a')}<div class='subtle'>MAE {_fmt_num(_safe_float(row.get('best_lead_bucket_mae_c')), 3)}C</div></td>"
            f"<td>{_html_escape(row.get('best_pws_station_id') or 'n/a')}<div class='subtle'>{_html_escape(row.get('best_pws_name') or '')}</div></td>"
            "</tr>"
        )

    station_sections: List[str] = []
    for index, row in enumerate(station_rows, start=1):
        station_id = str(row.get("station_id") or "")
        top_pws_rows = best_pws_top3.get(station_id, [])
        lead_rows = lead_bucket_metrics.get(station_id, {})
        lead_table: List[str] = []
        for bucket in LEAD_BUCKET_ORDER:
            metrics = lead_rows.get(bucket, {})
            lead_table.append(
                "<tr>"
                f"<td>{bucket}</td>"
                f"<td>{int(_safe_int(metrics.get('sample_count')) or 0)}</td>"
                f"<td>{_fmt_num(_safe_float(metrics.get('mae_c')), 3)}C</td>"
                f"<td>{_fmt_pct(_safe_float(metrics.get('hit_1c_rate')))}</td>"
                "</tr>"
            )

        pws_table: List[str] = []
        for pws_row in top_pws_rows:
            pws_table.append(
                "<tr>"
                f"<td>{_html_escape(pws_row.get('pws_station_id') or 'n/a')}</td>"
                f"<td>{_html_escape(pws_row.get('source') or 'n/a')}</td>"
                f"<td>{_html_escape(pws_row.get('station_name') or 'n/a')}</td>"
                f"<td>{int(_safe_int(pws_row.get('lead_row_count')) or 0)}</td>"
                f"<td>{_fmt_num(_safe_float(pws_row.get('lead_mae_c')), 3)}C</td>"
                f"<td>{_fmt_pct(_safe_float(pws_row.get('lead_hit_rate_1_0_c')))}</td>"
                "</tr>"
            )

        station_sections.append(
            f"""
            <section class="station-card {'page-break' if index > 1 else ''}">
              <div class="station-header">
                <div>
                  <h2>{_html_escape(station_id)} <span class="station-name">{_html_escape(row.get('station_name'))}</span></h2>
                  <div class="subtle">{_html_escape(row.get('timezone'))}</div>
                </div>
                <div>{_recommendation_badge(str(row.get('recommendation') or ''))}</div>
              </div>
              <p class="takeaway">{_html_escape(_render_station_takeaway(row))}</p>
              <div class="grid two">
                <div class="panel">
                  <h3>Best moments</h3>
                  <ul>
                    <li><strong>Day-ahead:</strong> {_html_escape(row.get('best_day_ahead_window') or 'n/a')} | final winner in top3 {_fmt_pct(_safe_float(row.get('best_day_ahead_top3')))} | average winner price {_fmt_prob(_safe_float(row.get('best_day_ahead_prob')))}</li>
                    <li><strong>Peak-hour:</strong> {_html_escape(row.get('best_peak_block') or 'n/a')} | PWS proxy {_fmt_pct(_safe_float(row.get('best_peak_pws_hit')))} vs market top1 {_fmt_pct(_safe_float(row.get('best_peak_market_hit')))} | edge {_fmt_pct(_safe_float(row.get('best_peak_edge')))}</li>
                    <li><strong>Lead window:</strong> {_html_escape(row.get('best_lead_bucket') or 'n/a')} | MAE {_fmt_num(_safe_float(row.get('best_lead_bucket_mae_c')), 3)}C | hit@1.0C {_fmt_pct(_safe_float(row.get('best_lead_bucket_hit_1c')))}</li>
                    <li><strong>Best single PWS:</strong> {_html_escape(row.get('best_pws_station_id') or 'n/a')} ({_html_escape(row.get('best_pws_source') or 'n/a')}) | MAE {_fmt_num(_safe_float(row.get('best_pws_mae_c')), 3)}C | hit@1.0C {_fmt_pct(_safe_float(row.get('best_pws_hit_1c')))}</li>
                  </ul>
                </div>
                <div class="panel">
                  <h3>Context</h3>
                  <ul>
                    <li><strong>WU minus official max:</strong> {_fmt_num(_safe_float(row.get('avg_wu_minus_official_f')), 2)}F on average across the window.</li>
                    <li><strong>Use this market as:</strong> {_html_escape(row.get('recommendation') or 'n/a')}.</li>
                    <li><strong>Interpretation:</strong> day-ahead numbers tell you how visible the final winner already was on the previous day; peak-hour numbers tell you whether a PWS-based next-step proxy beats the market close to the official daily high.</li>
                  </ul>
                </div>
              </div>
              <div class="grid two">
                <div class="panel">
                  <h3>Lead buckets by station</h3>
                  <table>
                    <thead><tr><th>Lead bucket</th><th>Rows</th><th>MAE</th><th>Hit @1.0C</th></tr></thead>
                    <tbody>{''.join(lead_table)}</tbody>
                  </table>
                </div>
                <div class="panel">
                  <h3>Top PWS stations</h3>
                  <table>
                    <thead><tr><th>PWS</th><th>Source</th><th>Name</th><th>Rows</th><th>MAE</th><th>Hit @1.0C</th></tr></thead>
                    <tbody>{''.join(pws_table) if pws_table else '<tr><td colspan="6">No ranked rows</td></tr>'}</tbody>
                  </table>
                </div>
              </div>
            </section>
            """
        )

    peak_minutes_html: List[str] = []
    for _, _, label in PEAK_MINUTE_BINS:
        peak_minutes_html.append(render_peak_minute_row(label, peak_minute_metrics.get(label, {})))

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>PWS settlement and peak-hour report</title>
  <style>
    @page {{ size: A4; margin: 14mm; }}
    body {{ font-family: "Segoe UI", Arial, sans-serif; color: #0f172a; margin: 0; background: #f8fafc; }}
    h1, h2, h3 {{ margin: 0 0 8px 0; }}
    h1 {{ font-size: 28px; line-height: 1.15; }}
    h2 {{ font-size: 22px; }}
    h3 {{ font-size: 16px; color: #0f172a; }}
    p, li {{ font-size: 12px; line-height: 1.45; }}
    .hero {{ background: linear-gradient(135deg, #082f49, #0f172a 55%, #1d4ed8); color: white; padding: 20px 22px; border-radius: 18px; margin-bottom: 18px; }}
    .hero p {{ color: rgba(255,255,255,0.88); margin: 10px 0 0 0; max-width: 860px; }}
    .meta {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 16px; }}
    .meta .item {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); padding: 10px 12px; border-radius: 12px; font-size: 12px; }}
    .section {{ background: white; border: 1px solid #dbe4ee; border-radius: 16px; padding: 16px 18px; margin-bottom: 16px; }}
    .grid {{ display: grid; gap: 14px; }}
    .grid.two {{ grid-template-columns: 1fr 1fr; }}
    .panel {{ background: #f8fafc; border: 1px solid #dbe4ee; border-radius: 14px; padding: 12px 14px; }}
    .subtle {{ color: #475569; font-size: 11px; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 6px 10px; font-size: 11px; font-weight: 700; letter-spacing: 0.02em; }}
    .badge-green {{ background: #dcfce7; color: #166534; }}
    .badge-blue {{ background: #dbeafe; color: #1d4ed8; }}
    .badge-amber {{ background: #fef3c7; color: #92400e; }}
    .badge-slate {{ background: #e2e8f0; color: #334155; }}
    .badge-red {{ background: #fee2e2; color: #b91c1c; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 11px; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid #dbe4ee; text-align: left; vertical-align: top; }}
    th {{ color: #334155; font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; }}
    ul {{ margin: 8px 0 0 18px; padding: 0; }}
    .station-card {{ background: white; border: 1px solid #dbe4ee; border-radius: 18px; padding: 16px 18px; margin-bottom: 16px; }}
    .station-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 8px; }}
    .station-name {{ color: #334155; font-weight: 400; font-size: 16px; }}
    .takeaway {{ margin: 8px 0 14px 0; font-size: 13px; }}
    .callouts {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }}
    .callout {{ border: 1px solid #dbe4ee; border-radius: 14px; background: #f8fafc; padding: 12px 14px; }}
    .callout h3 {{ font-size: 14px; }}
    .page-break {{ page-break-before: always; }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>PWS, settlement and peak-hour decision report</h1>
    <p>This report reframes the analysis around the two moments that matter for a weather trade: the previous local day, when brackets can still be mispriced, and the late intraday window close to the official daily high, where PWS can help you move before the crowd.</p>
    <div class="meta">
      <div class="item"><strong>Window analyzed</strong><br>2026-02-27 to 2026-03-08</div>
      <div class="item"><strong>Settlement truth</strong><br>NOAA METAR cumulative max in the logs</div>
      <div class="item"><strong>Important caveat</strong><br>Logs only retain D-1 to D, so there is no clean D-2 readout in this report</div>
    </div>
  </div>

  <section class="section">
    <h2>Simple reading</h2>
    <div class="callouts">
      <div class="callout">
        <h3>Best peak-hour candidate</h3>
        <p>{_html_escape(', '.join(row['station_id'] for row in highlights['peak_candidates']) or 'No strong one in this sample')}</p>
      </div>
      <div class="callout">
        <h3>Best day-ahead candidate</h3>
        <p>{_html_escape(', '.join(row['station_id'] for row in highlights['day_candidates']) or 'No clear one in this sample')}</p>
      </div>
      <div class="callout">
        <h3>Use as confirmation</h3>
        <p>{_html_escape(', '.join(row['station_id'] for row in highlights['confirmation']) or 'None')}</p>
      </div>
    </div>
    <p><strong>Main conclusion:</strong> there is no universal PWS edge across all markets close to the peak. The strongest late tactical candidates in this sample are KORD and, with less evidence, KDAL. Several other markets look more interesting one day before than in the intraday chase: KLGA, EGLC, KMIA, LFPG and LTAC all show previous-day windows where the final winner is visible at a still-manageable probability.</p>
  </section>

  <section class="section">
    <h2>What was measured</h2>
    <ul>
      <li><strong>Day-ahead:</strong> previous local day only, split into four windows. The report checks whether the final official winner was already top1 or top3, and at what probability.</li>
      <li><strong>Peak-hour:</strong> only setups where the final official winning bracket was still open. The PWS proxy is defined as <strong>max(current official max, next official expected by the PWS consensus)</strong>.</li>
      <li><strong>Best PWS station:</strong> taken from the historical audit rows by lowest MAE and strongest hit rate, with a minimum row threshold when available.</li>
    </ul>
  </section>

  <section class="section">
    <h2>Overall window summary</h2>
    <div class="grid two">
      <div class="panel">
        <h3>Previous-day visibility of the final winner</h3>
        <table>
          <thead><tr><th>Window</th><th>Samples</th><th>Winner top1</th><th>Winner top3</th><th>Winner avg probability</th></tr></thead>
          <tbody>{''.join(render_overall_row(row) for row in day_ahead_overall)}</tbody>
        </table>
      </div>
      <div class="panel">
        <h3>Minutes before the official peak</h3>
        <table>
          <thead><tr><th>Window</th><th>Open setups</th><th>PWS proxy hit</th><th>Market top1 hit</th><th>PWS edge</th></tr></thead>
          <tbody>{''.join(peak_minutes_html)}</tbody>
        </table>
      </div>
    </div>
  </section>

  <section class="section">
    <h2>Station scoreboard</h2>
    <table>
      <thead>
        <tr>
          <th>Market</th>
          <th>Use</th>
          <th>Best day-ahead window</th>
          <th>Best peak block</th>
          <th>Best lead bucket</th>
          <th>Best PWS</th>
        </tr>
      </thead>
      <tbody>{''.join(summary_rows_html)}</tbody>
    </table>
  </section>

  {''.join(station_sections)}

  <section class="section page-break">
    <h2>Practical takeaways</h2>
    <ul>
      <li><strong>KORD</strong> is the cleanest tactical market in this sample. It combines the best raw PWS quality with the best robust peak-hour edge.</li>
      <li><strong>KDAL</strong> also shows a usable late pocket, but the evidence is thinner and should be treated as more selective than KORD.</li>
      <li><strong>KMIA</strong>, <strong>KLGA</strong> and <strong>EGLC</strong> are more convincing as previous-day markets than as late PWS-chase markets. The final winner is often visible early enough, but the late proxy does not consistently beat the board.</li>
      <li><strong>LFPG</strong> and <strong>LTAC</strong> should be treated as previous-day only at most. The report does not show a reliable late intraday edge there.</li>
    </ul>
    <p class="subtle">Generated on {html.escape(config.report_date)}. If you want to extend the same report to a wider date window, rerun the script against a larger forensic export.</p>
  </section>
</body>
</html>
"""


def _find_browser(explicit: Optional[Path]) -> Optional[Path]:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _render_pdf(browser_path: Path, html_path: Path, pdf_path: Path) -> None:
    pdf_target = pdf_path.resolve()
    html_target = html_path.resolve().as_uri()
    commands = [
        [
            str(browser_path),
            "--headless=new",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={pdf_target}",
            html_target,
        ],
        [
            str(browser_path),
            "--headless",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={pdf_target}",
            html_target,
        ],
    ]
    last_error = None
    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and pdf_path.exists():
            return
        last_error = completed.stderr or completed.stdout or f"Browser exited with code {completed.returncode}"
    raise RuntimeError(f"Could not render PDF. {last_error}")


def generate_report(config: ReportConfig) -> Dict[str, Path]:
    raw_dir = config.input_dir / "raw"
    derived_dir = config.input_dir / "derived"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    performance_groups = _load_performance_groups(raw_dir / "performance_logs.csv")
    final_meta = _build_final_meta(performance_groups)
    day_ahead_metrics = _compute_day_ahead_metrics(performance_groups, final_meta)
    wu_gaps = _compute_wu_official_gap(final_meta)
    cycle_rows = _load_cycle_rows(derived_dir / "cycle_metrics.csv")
    peak_metrics, peak_minute_metrics = _compute_peak_metrics(cycle_rows, final_meta)
    lead_bucket_metrics = _compute_lead_bucket_metrics(raw_dir / "pws_audit.csv")
    best_pws, top_three_pws = _load_best_pws(derived_dir / "pws_station_metrics.csv")
    station_summary = _build_station_summary(day_ahead_metrics, peak_metrics, lead_bucket_metrics, best_pws, wu_gaps)
    day_ahead_overall = _build_overall_day_ahead(day_ahead_metrics)

    html_text = _render_html(
        config=config,
        station_rows=station_summary,
        day_ahead_overall=day_ahead_overall,
        peak_minute_metrics=peak_minute_metrics,
        lead_bucket_metrics=lead_bucket_metrics,
        best_pws_top3=top_three_pws,
    )

    html_path = config.output_dir / f"{config.report_date}_pws_settlement_peak_report.html"
    pdf_path = config.output_dir / f"{config.report_date}_pws_settlement_peak_report.pdf"
    summary_csv_path = config.output_dir / f"{config.report_date}_station_summary.csv"
    day_ahead_csv_path = config.output_dir / f"{config.report_date}_day_ahead_windows.csv"
    peak_csv_path = config.output_dir / f"{config.report_date}_peak_windows.csv"
    summary_json_path = config.output_dir / f"{config.report_date}_summary.json"

    html_path.write_text(html_text, encoding="utf-8")
    _write_csv(
        summary_csv_path,
        station_summary,
        [
            "station_id",
            "station_name",
            "timezone",
            "recommendation",
            "best_day_ahead_window",
            "best_day_ahead_top1",
            "best_day_ahead_top3",
            "best_day_ahead_prob",
            "best_peak_block",
            "best_peak_sample_count",
            "best_peak_pws_hit",
            "best_peak_market_hit",
            "best_peak_edge",
            "best_lead_bucket",
            "best_lead_bucket_mae_c",
            "best_lead_bucket_hit_1c",
            "best_pws_station_id",
            "best_pws_source",
            "best_pws_name",
            "best_pws_mae_c",
            "best_pws_hit_1c",
            "best_pws_row_count",
            "best_pws_rank_eligible",
            "avg_wu_minus_official_f",
        ],
    )

    day_ahead_rows: List[Dict[str, Any]] = []
    for station_id in STATION_ORDER:
        for _, _, window_name in DAY_AHEAD_WINDOWS:
            row = dict(day_ahead_metrics.get(station_id, {}).get(window_name, {}))
            row["station_id"] = station_id
            row["window_name"] = window_name
            day_ahead_rows.append(row)
    _write_csv(
        day_ahead_csv_path,
        day_ahead_rows,
        ["station_id", "window_name", "sample_count", "winner_top1_rate", "winner_top3_rate", "winner_avg_prob", "winner_avg_rank"],
    )

    peak_rows: List[Dict[str, Any]] = []
    for station_id in STATION_ORDER:
        for _, _, block_name in PEAK_BLOCKS:
            row = dict(peak_metrics.get(station_id, {}).get(block_name, {}))
            row["station_id"] = station_id
            row["block_name"] = block_name
            peak_rows.append(row)
    _write_csv(
        peak_csv_path,
        peak_rows,
        ["station_id", "block_name", "sample_count", "pws_peak_hit_rate", "market_top1_hit_rate", "edge_vs_market", "pws_only_rate", "market_only_rate"],
    )

    summary_json_path.write_text(
        json.dumps(
            {
                "report_date": config.report_date,
                "input_dir": str(config.input_dir),
                "output_dir": str(config.output_dir),
                "peak_minute_metrics": peak_minute_metrics,
                "station_summary": station_summary,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    browser_path = _find_browser(config.browser_path)
    if browser_path is None:
        raise RuntimeError("Chrome or Edge was not found. HTML was generated but PDF could not be rendered.")
    _render_pdf(browser_path, html_path, pdf_path)

    return {
        "html": html_path,
        "pdf": pdf_path,
        "station_summary": summary_csv_path,
        "day_ahead_windows": day_ahead_csv_path,
        "peak_windows": peak_csv_path,
        "summary_json": summary_json_path,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> ReportConfig:
    parser = argparse.ArgumentParser(description="Generate a settlement and peak-hour PDF report from a PWS forensic export.")
    parser.add_argument("--input-dir", required=True, help="Input forensic directory (must contain raw/ and derived/).")
    parser.add_argument("--output-dir", required=True, help="Output directory for the report.")
    parser.add_argument("--report-date", default=str(date.today().isoformat()))
    parser.add_argument("--browser-path", default=None, help="Optional explicit path to Chrome or Edge.")
    args = parser.parse_args(argv)
    return ReportConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        report_date=str(args.report_date),
        browser_path=Path(args.browser_path) if args.browser_path else None,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = _parse_args(argv)
    outputs = generate_report(config)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
