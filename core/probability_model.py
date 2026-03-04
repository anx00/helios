from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from config import STATIONS, get_polymarket_temp_unit
from core.polymarket_labels import label_for_temp, sort_labels


DAYAHEAD_SOURCE_WEIGHTS = {
    1: {"WUNDERGROUND": 0.55, "NBM": 0.20, "OPEN_METEO": 0.15, "LAMP": 0.10},
    2: {"WUNDERGROUND": 0.60, "NBM": 0.25, "OPEN_METEO": 0.15, "LAMP": 0.00},
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _round_optional(value: Any, digits: int = 4) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _market_unit(station_id: str) -> str:
    return str(get_polymarket_temp_unit(station_id) or "F").upper()


def _sigma_floor(station_id: str, target_day: int) -> float:
    market_unit = _market_unit(station_id)
    if market_unit == "C":
        return 0.8 if int(target_day) == 1 else 1.0
    return 1.4 if int(target_day) == 1 else 1.8


def _spread_saturation(station_id: str) -> float:
    return 4.0 if _market_unit(station_id) == "F" else (4.0 * 5.0 / 9.0)


def _source_calibration_multiplier(
    source: str,
    calibration: Optional[Dict[str, Any]],
) -> float:
    if not isinstance(calibration, dict):
        return 1.0
    source_row = ((calibration.get("sources") or {}) if isinstance(calibration.get("sources"), dict) else {}).get(str(source or "").upper())
    if not isinstance(source_row, dict):
        return 1.0
    samples = int(_safe_float(source_row.get("samples")) or 0)
    if samples < 2:
        return 1.0
    mae = _safe_float(source_row.get("mae_market")) or 0.0
    bias = abs(_safe_float(source_row.get("bias_market")) or 0.0)
    bonus = 0.05 if samples >= 4 else 0.0
    score = 1.20 - (0.18 * mae) - (0.10 * bias) + bonus
    return _clamp(score, 0.65, 1.35)


def _calibration_confidence_multiplier(
    station_id: str,
    calibration: Optional[Dict[str, Any]],
) -> float:
    if not isinstance(calibration, dict) or not calibration.get("available"):
        return 1.0
    samples = int(_safe_float(calibration.get("samples")) or 0)
    if samples < 2:
        return 1.0
    saturation = max(0.5, _spread_saturation(station_id))
    mae = _safe_float(calibration.get("mae_market")) or 0.0
    bias = abs(_safe_float(calibration.get("bias_market")) or 0.0)
    normalized_mae = min(1.0, mae / saturation)
    normalized_bias = min(1.0, bias / max(0.5, saturation * 0.75))
    sample_credit = (min(samples, 6) / 6.0) * 0.08
    score = 1.0 + sample_credit - (0.30 * normalized_mae) - (0.18 * normalized_bias)
    return _clamp(score, 0.78, 1.08)


def _build_calibration_payload(
    calibration: Optional[Dict[str, Any]],
    terminal_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw = calibration if isinstance(calibration, dict) else {}
    source_rows = raw.get("sources") if isinstance(raw.get("sources"), dict) else {}
    ranking: List[Dict[str, Any]] = []
    for source, row in source_rows.items():
        if not isinstance(row, dict):
            continue
        ranking.append({
            "source": str(source or "").upper(),
            "samples": int(_safe_float(row.get("samples")) or 0),
            "mae_market": _round_optional(row.get("mae_market"), 3),
            "bias_market": _round_optional(row.get("bias_market"), 3),
            "calibration_multiplier": _round_optional(_source_calibration_multiplier(str(source or "").upper(), raw), 4),
        })
    ranking.sort(
        key=lambda row: (
            float(row["mae_market"]) if row.get("mae_market") is not None else 999.0,
            abs(float(row["bias_market"])) if row.get("bias_market") is not None else 999.0,
            -(int(row.get("samples") or 0)),
            str(row.get("source") or ""),
        )
    )
    contributors = list((terminal_model or {}).get("contributors") or []) if isinstance(terminal_model, dict) else []
    applied_sources = [
        str(row.get("source") or "").upper()
        for row in contributors
        if abs(float(_safe_float(row.get("calibration_multiplier")) or 1.0) - 1.0) >= 0.02
    ]
    if not applied_sources:
        applied_sources = [
            str(row.get("source") or "").upper()
            for row in ranking
            if abs(float(_safe_float(row.get("calibration_multiplier")) or 1.0) - 1.0) >= 0.02
        ]
    return {
        "available": bool(raw.get("available")),
        "warming_up": bool(raw.get("warming_up")),
        "samples": int(_safe_float(raw.get("samples")) or 0),
        "mae_market": _round_optional(raw.get("mae_market"), 3),
        "bias_market": _round_optional(raw.get("bias_market"), 3),
        "source_ranking": ranking,
        "applied_sources": applied_sources,
        "active": bool(applied_sources),
        "confidence_multiplier": _round_optional((terminal_model or {}).get("confidence_multiplier"), 4),
        "confidence_base": _round_optional((terminal_model or {}).get("confidence_base"), 4),
        "confidence_final": _round_optional((terminal_model or {}).get("confidence"), 4),
    }


def _recommendation_rank(value: Optional[str]) -> int:
    rec = str(value or "").upper()
    if rec.startswith("BUY"):
        return 3
    if rec.startswith("LEAN"):
        return 2
    if rec.startswith("WATCH"):
        return 1
    return 0


def _current_trade_row(signal_payload: Optional[Dict[str, Any]], target_day: int) -> Optional[Dict[str, Any]]:
    if not isinstance(signal_payload, dict):
        return None
    tactical = signal_payload.get("best_tactical_trade")
    tactical_context = signal_payload.get("tactical_context") if isinstance(signal_payload.get("tactical_context"), dict) else {}
    if int(target_day) == 0 and tactical_context.get("enabled") and isinstance(tactical, dict):
        return tactical
    trade = signal_payload.get("best_terminal_trade")
    return trade if isinstance(trade, dict) else None


def _actionable_summary(signal_payload: Optional[Dict[str, Any]], target_day: int) -> Dict[str, Any]:
    trade = _current_trade_row(signal_payload, target_day)
    if not isinstance(trade, dict):
        return {
            "available": False,
            "label": None,
            "side": None,
            "recommendation": "BLOCK",
            "edge_points": None,
            "entry_price": None,
            "fair_prob": None,
            "reason": "No trade candidate",
        }

    use_tactical = bool(
        int(target_day) == 0
        and isinstance(signal_payload, dict)
        and isinstance(signal_payload.get("tactical_context"), dict)
        and signal_payload["tactical_context"].get("enabled")
        and trade is signal_payload.get("best_tactical_trade")
    )
    policy_key = "tactical_policy" if use_tactical else "terminal_policy"
    side_key = "tactical_best_side" if use_tactical else "best_side"
    edge_key = "tactical_best_edge" if use_tactical else "best_edge"
    fair_key = "selected_tactical_fair" if use_tactical else "selected_fair"
    entry_key = "selected_tactical_entry" if use_tactical else "selected_entry"
    recommendation_key = "tactical_recommendation" if use_tactical else "recommendation"

    policy = trade.get(policy_key) if isinstance(trade.get(policy_key), dict) else {}
    selected_fair = _safe_float(trade.get(fair_key))
    best_edge = _safe_float(trade.get(edge_key))
    entry_price = _safe_float(trade.get(entry_key))
    recommendation = str(trade.get(recommendation_key) or trade.get("recommendation") or "BLOCK").upper()
    reasons = {str(reason or "") for reason in (policy.get("reasons") or [])}
    blocked_reasons = {
        "tail_too_small",
        "top_bucket_better_priced",
        "missing_market",
        "too_expensive",
        "too_far_from_top_bucket",
    }
    highlight = bool(
        policy.get("allowed")
        and best_edge is not None
        and best_edge >= 0.03
        and selected_fair is not None
        and selected_fair >= 0.08
        and entry_price is not None
        and not (reasons & blocked_reasons)
        and recommendation.startswith(("BUY_", "LEAN_"))
    )
    reason = policy.get("summary") or trade.get("policy_reason") or "No actionable edge"
    return {
        "available": highlight,
        "label": trade.get("label"),
        "side": trade.get(side_key) or trade.get("best_side"),
        "recommendation": recommendation if highlight else ("WATCH" if policy.get("allowed") else "BLOCK"),
        "edge_points": _round_optional((best_edge or 0.0) * 100.0, 4) if best_edge is not None else None,
        "entry_price": _round_optional(entry_price, 4),
        "fair_prob": _round_optional(selected_fair, 6),
        "reason": str(reason),
    }


def build_dayahead_terminal_model(
    *,
    station_id: str,
    target_day: int,
    source_snapshots: Sequence[Dict[str, Any]],
    calibration: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    weights = dict(DAYAHEAD_SOURCE_WEIGHTS.get(int(target_day), {}))
    if not weights:
        return None

    usable: List[Dict[str, Any]] = []
    for snapshot in source_snapshots:
        if not isinstance(snapshot, dict):
            continue
        source = str(snapshot.get("source") or "").upper()
        if source not in weights or weights[source] <= 0.0:
            continue
        if str(snapshot.get("status") or "").lower() != "ok":
            continue
        forecast_high_market = _safe_float(snapshot.get("forecast_high_market"))
        if forecast_high_market is None:
            continue
        calibration_multiplier = _source_calibration_multiplier(source, calibration)
        usable.append({
            "source": source,
            "weight": float(weights[source]),
            "base_weight": float(weights[source]),
            "calibration_multiplier": calibration_multiplier,
            "adjusted_weight": float(weights[source]) * calibration_multiplier,
            "forecast_high_market": forecast_high_market,
            "peak_hour_local": snapshot.get("peak_hour_local"),
            "status": snapshot.get("status"),
        })

    if not usable:
        return None

    total_weight = sum(row["adjusted_weight"] for row in usable)
    if total_weight <= 0:
        return None
    for row in usable:
        row["weight"] = row["adjusted_weight"] / total_weight

    mean_market = sum(row["weight"] * row["forecast_high_market"] for row in usable)
    variance = sum(row["weight"] * ((row["forecast_high_market"] - mean_market) ** 2) for row in usable)
    source_spread_market = math.sqrt(max(0.0, variance))
    sigma_market = max(source_spread_market * 1.35, _sigma_floor(station_id, int(target_day)))

    normalized_spread = min(1.0, source_spread_market / _spread_saturation(station_id))
    source_count = len(usable)
    wu_present = 1.0 if any(row["source"] == "WUNDERGROUND" for row in usable) else 0.0
    confidence_base = _clamp(
        0.25 + (0.20 * source_count) + (0.35 * (1.0 - normalized_spread)) + (0.20 * wu_present),
        0.25,
        0.95,
    )

    notes: List[str] = []
    missing_sources = [source for source in weights if weights[source] > 0.0 and source not in {row["source"] for row in usable}]
    if missing_sources:
        notes.append(f"missing_sources:{','.join(missing_sources)}")
    if source_count == 1:
        notes.append("low_source_coverage")
    if isinstance(calibration, dict):
        calibration_sources = [
            row["source"]
            for row in usable
            if abs(float(row.get("calibration_multiplier") or 1.0) - 1.0) >= 0.02
        ]
        if calibration_sources:
            notes.append(f"calibration_weighting:{','.join(calibration_sources)}")
        if calibration.get("warming_up"):
            notes.append("calibration_warming_up")
    confidence_multiplier = _calibration_confidence_multiplier(station_id, calibration)
    confidence = _clamp(confidence_base * confidence_multiplier, 0.25, 0.95)
    if source_count == 1:
        confidence = min(confidence, 0.55)
    if abs(confidence_multiplier - 1.0) >= 0.02:
        notes.append(f"calibration_confidence:{confidence_multiplier:.3f}")

    weighted_peak_hours = [
        (row["weight"], _safe_float(row.get("peak_hour_local")))
        for row in usable
        if _safe_float(row.get("peak_hour_local")) is not None
    ]
    peak_hour = None
    if weighted_peak_hours:
        peak_hour = sum(weight * float(hour) for weight, hour in weighted_peak_hours)

    return {
        "source": "DAYAHEAD_FUSION",
        "mean_market": round(mean_market, 4),
        "sigma_market": round(sigma_market, 4),
        "confidence": round(confidence, 4),
        "market_floor": None,
        "market_ceiling": None,
        "market_ceiling_exact": None,
        "peak_hour": round(peak_hour, 3) if peak_hour is not None else None,
        "current_hour": None,
        "hours_to_peak": None,
        "confidence_base": round(confidence_base, 4),
        "confidence_multiplier": round(confidence_multiplier, 4),
        "source_count": source_count,
        "source_spread_market": round(source_spread_market, 4),
        "contributors": [
            {
                "source": row["source"],
                "weight": round(row["weight"], 4),
                "base_weight": round(row["base_weight"], 4),
                "calibration_multiplier": round(row["calibration_multiplier"], 4),
                "adjusted_weight": round(row["adjusted_weight"], 4),
                "forecast_high_market": round(row["forecast_high_market"], 4),
                "peak_hour_local": row.get("peak_hour_local"),
            }
            for row in usable
        ],
        "notes": notes,
    }


def build_source_strip(source_snapshots: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snapshot in source_snapshots or []:
        if not isinstance(snapshot, dict):
            continue
        rows.append({
            "source": snapshot.get("source"),
            "status": snapshot.get("status"),
            "forecast_high_market": _round_optional(snapshot.get("forecast_high_market"), 3),
            "forecast_high_f": _round_optional(snapshot.get("forecast_high_f"), 1),
            "forecast_high_c": _round_optional(snapshot.get("forecast_high_c"), 1),
            "peak_hour_local": snapshot.get("peak_hour_local"),
            "provider_updated_local": snapshot.get("provider_updated_local"),
            "captured_at_utc": snapshot.get("captured_at_utc"),
            "notes": list(snapshot.get("notes") or []),
        })
    return rows


def build_source_history_series(source_history: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for snapshot in source_history or []:
        if not isinstance(snapshot, dict):
            continue
        source = str(snapshot.get("source") or "").upper()
        if not source:
            continue
        grouped[source].append({
            "captured_at_utc": snapshot.get("captured_at_utc"),
            "forecast_high_market": _round_optional(snapshot.get("forecast_high_market"), 3),
            "forecast_high_f": _round_optional(snapshot.get("forecast_high_f"), 1),
            "forecast_high_c": _round_optional(snapshot.get("forecast_high_c"), 1),
            "status": snapshot.get("status"),
        })
    return [
        {"source": source, "points": sorted(points, key=lambda row: str(row.get("captured_at_utc") or ""))}
        for source, points in sorted(grouped.items())
    ]


def build_probability_lab_card(
    *,
    station_id: str,
    station_name: str,
    target_day: int,
    target_date: str,
    market_payload: Optional[Dict[str, Any]],
    signal_payload: Optional[Dict[str, Any]],
    terminal_model: Optional[Dict[str, Any]] = None,
    source_snapshots: Optional[Sequence[Dict[str, Any]]] = None,
    reality: Optional[Dict[str, Any]] = None,
    pws_profiles: Optional[Sequence[Dict[str, Any]]] = None,
    calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    brackets = list((market_payload or {}).get("brackets") or [])
    ordered_market = sorted(brackets, key=lambda row: float(_safe_float((row or {}).get("yes_price")) or 0.0), reverse=True)
    top_market = ordered_market[0] if ordered_market else {}
    model_info = ((signal_payload or {}).get("model") if isinstance(signal_payload, dict) else {}) or {}
    actionable = _actionable_summary(signal_payload, target_day)
    calibration_payload = _build_calibration_payload(calibration, terminal_model)

    return {
        "station_id": station_id,
        "station_name": station_name,
        "target_date": target_date,
        "horizon": "INTRADAY" if int(target_day) == 0 else "DAY_AHEAD",
        "market": {
            "event_title": (market_payload or {}).get("event_title"),
            "top_label": top_market.get("name") or top_market.get("bracket"),
            "top_price": _round_optional(top_market.get("yes_price"), 4),
            "volume": _round_optional((market_payload or {}).get("total_volume"), 2),
            "status": "CLOSED" if ((market_payload or {}).get("market_status") or {}).get("event_closed") else "OPEN",
        },
        "model": {
            "source": model_info.get("source"),
            "mean": _round_optional(model_info.get("mean"), 4),
            "sigma": _round_optional(model_info.get("sigma"), 4),
            "confidence": _round_optional(model_info.get("confidence"), 4),
            "top_label": model_info.get("top_label"),
            "top_probability": _round_optional(model_info.get("top_label_probability"), 6),
            "source_count": terminal_model.get("source_count") if isinstance(terminal_model, dict) else None,
            "source_spread": _round_optional((terminal_model or {}).get("source_spread_market"), 4),
            "notes": list((terminal_model or {}).get("notes") or []),
            "calibration_active": calibration_payload.get("active"),
            "calibration_warming_up": calibration_payload.get("warming_up"),
            "calibration_samples": calibration_payload.get("samples"),
            "calibration_mae_market": calibration_payload.get("mae_market"),
            "calibration_bias_market": calibration_payload.get("bias_market"),
        },
        "actionable": actionable,
        "source_strip": build_source_strip(source_snapshots),
        "reality": dict(reality or {}),
        "pws": {
            "top_profiles": list(pws_profiles or [])[:5],
            "count": len(list(pws_profiles or [])),
            "strong": bool(pws_profiles),
        },
        "_sort": {
            "actionable_rank": _recommendation_rank(actionable.get("recommendation")),
            "actionable_edge": _safe_float(actionable.get("edge_points")) or -999.0,
            "confidence": _safe_float(model_info.get("confidence")) or -999.0,
            "volume": _safe_float((market_payload or {}).get("total_volume")) or -999.0,
        },
    }


def build_probability_lab_station_detail(
    *,
    station_id: str,
    target_day: int,
    target_date: str,
    market_payload: Optional[Dict[str, Any]],
    signal_payload: Optional[Dict[str, Any]],
    terminal_model: Optional[Dict[str, Any]] = None,
    source_snapshots: Optional[Sequence[Dict[str, Any]]] = None,
    source_history: Optional[Iterable[Dict[str, Any]]] = None,
    lookback_hours: int = 36,
    reality: Optional[Dict[str, Any]] = None,
    pws_profiles: Optional[Sequence[Dict[str, Any]]] = None,
    history_warming_up: bool = False,
    calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ladder: List[Dict[str, Any]] = []
    for row in list((signal_payload or {}).get("all_terminal_opportunities") or []):
        if not isinstance(row, dict):
            continue
        ladder.append({
            "label": row.get("label"),
            "fair_yes": _round_optional(row.get("fair_yes"), 6),
            "fair_no": _round_optional(row.get("fair_no"), 6),
            "market_yes": _round_optional(row.get("market_yes"), 6),
            "market_no": _round_optional(row.get("market_no"), 6),
            "selected_side": row.get("best_side"),
            "selected_fair": _round_optional(row.get("selected_fair"), 6),
            "selected_entry": _round_optional(row.get("selected_entry"), 6),
            "best_edge": _round_optional(row.get("best_edge"), 6),
            "edge_points": _round_optional(row.get("edge_points"), 4),
            "recommendation": row.get("recommendation"),
            "policy_reason": row.get("policy_reason"),
            "tactical_recommendation": row.get("tactical_recommendation"),
            "tactical_policy_reason": row.get("tactical_policy_reason"),
        })

    card = build_probability_lab_card(
        station_id=station_id,
        station_name=getattr(STATIONS.get(station_id), "name", station_id),
        target_day=target_day,
        target_date=target_date,
        market_payload=market_payload,
        signal_payload=signal_payload,
        terminal_model=terminal_model,
        source_snapshots=source_snapshots,
        reality=reality,
        pws_profiles=pws_profiles,
        calibration=calibration,
    )
    card.pop("_sort", None)

    return {
        "station_id": station_id,
        "target_day": int(target_day),
        "target_date": target_date,
        "summary": card,
        "source_detail": build_source_strip(source_snapshots),
        "source_history": {
            "lookback_hours": int(lookback_hours),
            "history_warming_up": bool(history_warming_up),
            "series": build_source_history_series(source_history),
        },
        "bracket_ladder": ladder,
        "pws_profiles": list(pws_profiles or []),
        "tactical": (signal_payload or {}).get("tactical_context") if isinstance(signal_payload, dict) else None,
        "reality": dict(reality or {}),
        "calibration": _build_calibration_payload(calibration, terminal_model),
    }


def sort_probability_lab_cards(cards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        list(cards),
        key=lambda row: (
            int((row.get("_sort") or {}).get("actionable_rank") or 0),
            float((row.get("_sort") or {}).get("actionable_edge") or -999.0),
            float((row.get("_sort") or {}).get("confidence") or -999.0),
            float((row.get("_sort") or {}).get("volume") or -999.0),
        ),
        reverse=True,
    )


def current_reality_bracket(value_market: Optional[float], labels: Sequence[str]) -> Optional[str]:
    if value_market is None or not labels:
        return None
    label, _ = label_for_temp(float(value_market), sort_labels(list(labels)))
    return label
