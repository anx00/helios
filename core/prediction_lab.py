from __future__ import annotations

import math
from datetime import date as date_cls
from datetime import datetime
from statistics import median
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from config import STATIONS, get_polymarket_temp_unit


UTC = ZoneInfo("UTC")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _station_tz(station_id: str) -> ZoneInfo:
    station = STATIONS.get(str(station_id or "").upper())
    tz_name = str(getattr(station, "timezone", "") or "")
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    return UTC


def _market_unit(station_id: str) -> str:
    return str(get_polymarket_temp_unit(station_id) or "F").upper()


def _parse_dt(value: Any, *, default_tz: Optional[ZoneInfo] = None) -> Optional[datetime]:
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
        return dt.replace(tzinfo=default_tz or UTC)
    return dt


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


def _convert_f_to_market_unit(value_f: Any, market_unit: str) -> Optional[float]:
    if str(market_unit or "F").upper() == "C":
        return _f_to_c(value_f)
    return _safe_float(value_f)


def _convert_c_to_market_unit(value_c: Any, market_unit: str) -> Optional[float]:
    if str(market_unit or "F").upper() == "C":
        return _safe_float(value_c)
    return _c_to_f(value_c)


def _round_or_none(value: Any, digits: int = 3) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(float(number), digits)


def _mean(values: Sequence[float]) -> Optional[float]:
    numbers = [float(value) for value in values if _safe_float(value) is not None]
    if not numbers:
        return None
    return sum(numbers) / float(len(numbers))


def _weighted_mean(pairs: Sequence[tuple[float, float]]) -> Optional[float]:
    total_weight = 0.0
    total_value = 0.0
    for value, weight in pairs:
        value_num = _safe_float(value)
        weight_num = _safe_float(weight)
        if value_num is None or weight_num is None or weight_num <= 0:
            continue
        total_weight += weight_num
        total_value += value_num * weight_num
    if total_weight <= 0:
        return None
    return total_value / total_weight


def _isoformat_or_none(value: Optional[datetime], *, tz: Optional[ZoneInfo] = None) -> Optional[str]:
    if value is None:
        return None
    if tz is not None:
        return value.astimezone(tz).isoformat()
    return value.isoformat()


def _preferred_point_value(point: Dict[str, Any], market_unit: str) -> Optional[float]:
    if str(market_unit).upper() == "C":
        return _safe_float(point.get("temp_c"))
    return _safe_float(point.get("temp_f"))


def _source_target_points(
    station_id: str,
    source_payload: Dict[str, Any],
    target_date: date_cls,
    market_unit: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    station_tz = _station_tz(station_id)
    for point in (source_payload.get("forecast_hourly") or {}).get("points") or []:
        if not isinstance(point, dict):
            continue
        dt_local = _parse_dt(point.get("timestamp_local"), default_tz=station_tz)
        if dt_local is None or dt_local.date() != target_date:
            continue
        value = _preferred_point_value(point, market_unit)
        if value is None:
            continue
        rows.append(
            {
                "timestamp_local": dt_local.isoformat(),
                "value": round(float(value), 3),
                "conditions": point.get("conditions"),
            }
        )
    rows.sort(key=lambda row: row.get("timestamp_local") or "")
    return rows


def _source_summary_row(
    station_id: str,
    source_payload: Dict[str, Any],
    target_date: date_cls,
    market_unit: str,
) -> Dict[str, Any]:
    source_name = str(source_payload.get("source") or "").strip()
    target_points = _source_target_points(station_id, source_payload, target_date, market_unit)
    target_max = None
    peak_time_local = None
    if target_points:
        target_max = max(float(row["value"]) for row in target_points)
        peak_row = next((row for row in target_points if float(row["value"]) == float(target_max)), None)
        peak_time_local = peak_row.get("timestamp_local") if peak_row else None

    historical = source_payload.get("historical_day") or {}
    realtime = source_payload.get("realtime") or {}
    return {
        "source": source_name,
        "display_name": source_payload.get("display_name") or source_name,
        "status": source_payload.get("status") or "unknown",
        "target_day_max": _round_or_none(target_max),
        "target_day_peak_time_local": peak_time_local,
        "target_hour_count": len(target_points),
        "realtime": _round_or_none(_preferred_point_value(realtime, market_unit)),
        "realtime_as_of_local": realtime.get("as_of_local"),
        "history_today_max": _round_or_none(
            historical.get("max_temp_c") if market_unit == "C" else historical.get("max_temp_f")
        ),
        "history_confirmed": bool(historical.get("confirmed")),
        "history_kind": historical.get("kind"),
        "notes": list(source_payload.get("notes") or []),
        "errors": list(source_payload.get("errors") or []),
        "target_curve": target_points,
    }


def build_prior_model(
    *,
    station_id: str,
    target_date: date_cls,
    external_sources: Optional[Dict[str, Any]],
    legacy_prediction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    market_unit = _market_unit(station_id)
    sources = list((external_sources or {}).get("sources") or [])
    source_rows = [
        _source_summary_row(station_id, source_payload, target_date, market_unit)
        for source_payload in sources
        if isinstance(source_payload, dict)
    ]

    maxima = [float(row["target_day_max"]) for row in source_rows if _safe_float(row.get("target_day_max")) is not None]
    source_count = len(maxima)
    consensus_max = _mean(maxima)
    consensus_spread = (max(maxima) - min(maxima)) if len(maxima) >= 2 else 0.0 if maxima else None
    confirmed_count = sum(1 for row in source_rows if row.get("history_confirmed"))
    confidence = None
    if source_count > 0:
        spread_scale = 3.0 if market_unit == "C" else 5.5
        spread_term = 1.0
        if consensus_spread is not None:
            spread_term = max(0.0, 1.0 - (float(consensus_spread) / spread_scale))
        confidence = max(
            0.2,
            min(
                0.92,
                0.25
                + min(0.3, source_count * 0.12)
                + (0.15 * (confirmed_count / max(1, len(source_rows))))
                + (0.22 * spread_term),
            ),
        )

    legacy_market = None
    if legacy_prediction:
        legacy_market = _convert_f_to_market_unit(legacy_prediction.get("final_prediction_f"), market_unit)

    headline = "No hourly forward curve available for the selected day."
    if consensus_max is not None:
        headline = (
            f"{source_count} sources point to a {round(float(consensus_max), 2)} {market_unit} peak "
            f"for {target_date.isoformat()}."
        )
        if consensus_spread is not None and source_count > 1:
            headline += f" Current spread is {round(float(consensus_spread), 2)} {market_unit}."

    notes = []
    if legacy_market is not None:
        notes.append(
            f"Legacy HELIOS physics reference is {round(float(legacy_market), 2)} {market_unit}; "
            "it is shown as context, not as the primary probabilistic driver."
        )
    notes.append(
        "Run-to-run forecast snapshots are not persisted yet, so this prior only sees the current forward curves."
    )

    return {
        "available": bool(source_rows),
        "market_unit": market_unit,
        "target_date": target_date.isoformat(),
        "source_count": source_count,
        "consensus_max": _round_or_none(consensus_max),
        "consensus_spread": _round_or_none(consensus_spread),
        "confidence": _round_or_none(confidence, 4),
        "headline": headline,
        "source_rows": source_rows,
        "notes": notes,
    }


def _source_rank(source_name: Any) -> int:
    normalized = str(source_name or "").strip().upper()
    if normalized == "SYNOPTIC":
        return 0
    if normalized.startswith("MADIS_MESOWEST"):
        return 1
    if normalized.startswith("MADIS_APRSWXNET"):
        return 2
    if normalized.startswith("MADIS_"):
        return 3
    if normalized == "WUNDERGROUND":
        return 4
    return 5


def _dedupe_current_pws_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_station: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict) or not bool(row.get("valid")):
            continue
        station_key = str(row.get("station_id") or "").upper()
        temp_c = _safe_float(row.get("temp_c"))
        if not station_key or temp_c is None:
            continue
        score = (
            _safe_float(row.get("learning_weight_predictive"))
            or _safe_float(row.get("weight_predictive"))
            or _safe_float(row.get("learning_weight"))
            or _safe_float(row.get("weight"))
            or 0.0
        )
        age_minutes = _safe_float(row.get("age_minutes"))
        freshness = 9999.0 if age_minutes is None else float(age_minutes)
        rank = _source_rank(row.get("source"))
        current = best_by_station.get(station_key)
        if current is None:
            best_by_station[station_key] = dict(row)
            continue
        current_score = (
            _safe_float(current.get("learning_weight_predictive"))
            or _safe_float(current.get("weight_predictive"))
            or _safe_float(current.get("learning_weight"))
            or _safe_float(current.get("weight"))
            or 0.0
        )
        current_age = _safe_float(current.get("age_minutes"))
        current_freshness = 9999.0 if current_age is None else float(current_age)
        current_rank = _source_rank(current.get("source"))
        if (score, -freshness, -rank) > (current_score, -current_freshness, -current_rank):
            best_by_station[station_key] = dict(row)
    return list(best_by_station.values())


def classify_pws_station(row: Dict[str, Any]) -> Dict[str, str]:
    next_total = int(row.get("next_metar_total") or 0)
    raw_updates = int(row.get("raw_updates_total") or 0)
    hit_rate = _safe_float(row.get("next_metar_hit_rate"))
    next_mae = _safe_float(row.get("next_metar_mae_c"))
    current_mae = _safe_float(row.get("current_metar_mae_c"))
    audit_hit_1 = _safe_float(row.get("audit_hit_rate_within_1_0_c"))
    delay_gain = None
    if current_mae is not None and next_mae is not None:
        delay_gain = current_mae - next_mae

    if raw_updates < 80 or next_total < 4:
        return {"code": "sparse", "label": "Sparse"}
    if hit_rate is not None and next_mae is not None and hit_rate >= 0.7 and next_mae <= 0.6:
        return {"code": "stable", "label": "Stable"}
    if delay_gain is not None and delay_gain >= 0.25 and hit_rate is not None and hit_rate >= 0.55:
        return {"code": "delayed_but_useful", "label": "Delayed but useful"}
    if hit_rate is not None and next_mae is not None and (hit_rate <= 0.4 or next_mae >= 1.2):
        return {"code": "noisy", "label": "Noisy"}
    if audit_hit_1 is not None and hit_rate is not None and abs(audit_hit_1 - hit_rate) >= 0.2:
        return {"code": "regime_dependent", "label": "Regime-dependent"}
    return {"code": "mixed", "label": "Mixed"}


def _pws_historical_focus(pws_history_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    summary = pws_history_summary or {}
    stations = [row for row in summary.get("stations") or [] if isinstance(row, dict)]
    top_rows = []
    for row in stations[:8]:
        current_mae = _safe_float(row.get("current_metar_mae_c"))
        next_mae = _safe_float(row.get("next_metar_mae_c"))
        delay_gain = None
        if current_mae is not None and next_mae is not None:
            delay_gain = current_mae - next_mae
        top_rows.append(
            {
                "station_id": row.get("station_id"),
                "station_name": row.get("station_name") or row.get("station_id"),
                "source": row.get("source"),
                "distance_km": _round_or_none(row.get("distance_km")),
                "classification": classify_pws_station(row),
                "next_metar_hit_rate": _round_or_none(row.get("next_metar_hit_rate"), 4),
                "next_metar_mae_c": _round_or_none(next_mae),
                "current_metar_mae_c": _round_or_none(current_mae),
                "audit_hit_rate_within_1_0_c": _round_or_none(row.get("audit_hit_rate_within_1_0_c"), 4),
                "raw_updates_total": int(row.get("raw_updates_total") or 0),
                "browser_day_count": int(row.get("browser_day_count") or 0),
                "audit_day_count": int(row.get("audit_day_count") or 0),
                "next_metar_total": int(row.get("next_metar_total") or 0),
                "delay_gain_c": _round_or_none(delay_gain),
            }
        )

    return {
        "available": bool(summary),
        "window_mode": summary.get("window_mode"),
        "window_day_count": int(summary.get("window_day_count") or 0),
        "available_day_count": len(summary.get("available_dates") or []),
        "summary": dict(summary.get("summary") or {}),
        "top_stations": top_rows,
        "notes": list(summary.get("notes") or []),
    }


def build_intraday_model(
    *,
    station_id: str,
    target_day: int,
    official: Optional[Dict[str, Any]],
    pws_details: Optional[Sequence[Dict[str, Any]]],
    pws_metrics: Optional[Dict[str, Any]],
    pws_history_summary: Optional[Dict[str, Any]],
    observed_max_f: Optional[float],
    trading_signal: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    market_unit = _market_unit(station_id)
    station_tz = _station_tz(station_id)
    history_focus = _pws_historical_focus(pws_history_summary)

    if int(target_day) != 0:
        return {
            "available": False,
            "market_unit": market_unit,
            "status": "inactive",
            "reason": "The intraday layer only activates for today.",
            "historical_ranking": history_focus,
        }

    official_obs = official or {}
    official_temp_c = _safe_float(official_obs.get("temp_c"))
    official_time = _parse_dt(official_obs.get("obs_time_utc"))
    official_market = _convert_c_to_market_unit(official_temp_c, market_unit)

    deduped_rows = _dedupe_current_pws_rows(pws_details or [])
    current_pairs: List[tuple[float, float]] = []
    current_values_market: List[float] = []
    top_current_rows: List[Dict[str, Any]] = []
    history_by_station = {
        str(row.get("station_id") or "").upper(): row
        for row in (history_focus.get("top_stations") or [])
        if isinstance(row, dict)
    }

    for row in deduped_rows:
        temp_c = _safe_float(row.get("temp_c"))
        temp_market = _convert_c_to_market_unit(temp_c, market_unit)
        if temp_market is None:
            continue
        current_values_market.append(float(temp_market))
        weight = (
            _safe_float(row.get("learning_weight_predictive"))
            or _safe_float(row.get("weight_predictive"))
            or _safe_float(row.get("learning_weight"))
            or _safe_float(row.get("weight"))
            or 1.0
        )
        current_pairs.append((float(temp_market), float(weight)))

        station_key = str(row.get("station_id") or "").upper()
        history_row = history_by_station.get(station_key, {})
        top_current_rows.append(
            {
                "station_id": station_key,
                "station_name": row.get("station_name") or station_key,
                "source": row.get("source"),
                "temp_market": _round_or_none(temp_market),
                "temp_c": _round_or_none(temp_c),
                "age_minutes": _round_or_none(row.get("age_minutes"), 2),
                "distance_km": _round_or_none(row.get("distance_km")),
                "weight_predictive": _round_or_none(weight, 4),
                "next_metar_score": _round_or_none(
                    row.get("learning_predictive_score")
                    or row.get("predictive_score")
                    or row.get("learning_lead_score")
                    or row.get("lead_score")
                ),
                "classification": classify_pws_station(history_row) if history_row else None,
            }
        )

    top_current_rows.sort(
        key=lambda row: (
            -(float(row.get("weight_predictive") or 0.0)),
            float(row.get("age_minutes") or 9999.0),
            float(row.get("distance_km") or 9999.0),
        )
    )

    median_market = _round_or_none(median(current_values_market) if current_values_market else None)
    weighted_mean_market = _round_or_none(_weighted_mean(current_pairs))
    dispersion_market = None
    if current_values_market:
        dispersion_market = max(current_values_market) - min(current_values_market)

    next_official = None
    if isinstance(trading_signal, dict):
        next_official = (
            ((trading_signal.get("modules") or {}).get("next_official"))
            or ((trading_signal.get("tactical_context") or {}).get("next_metar"))
        )
    observed_max_market = _convert_f_to_market_unit(observed_max_f, market_unit)

    consensus_gap = None
    if weighted_mean_market is not None and official_market is not None:
        consensus_gap = float(weighted_mean_market) - float(official_market)

    headline = "No live official or PWS data is available right now."
    if weighted_mean_market is not None:
        headline = (
            f"PWS consensus sits near {weighted_mean_market} {market_unit}"
            + (
                f", versus official {round(float(official_market), 3)} {market_unit}."
                if official_market is not None
                else "."
            )
        )
        if isinstance(next_official, dict) and next_official.get("available"):
            delta = _safe_float(next_official.get("delta_market"))
            direction = str(next_official.get("direction") or "FLAT")
            headline += (
                f" Next official projection is {direction}"
                + (f" ({round(float(delta), 3)} {market_unit})" if delta is not None else "")
                + "."
            )

    return {
        "available": bool(official_temp_c is not None or current_values_market),
        "market_unit": market_unit,
        "status": "live",
        "reason": None,
        "headline": headline,
        "official": {
            "temp_market": _round_or_none(official_market),
            "temp_c": _round_or_none(official_temp_c),
            "obs_time_local": _isoformat_or_none(official_time, tz=station_tz),
            "report_type": official_obs.get("report_type"),
            "is_speci": bool(official_obs.get("is_speci")),
        },
        "observed_max_market": _round_or_none(observed_max_market),
        "pws_now": {
            "station_count": len(current_values_market),
            "fresh_station_count": sum(
                1
                for row in top_current_rows
                if _safe_float(row.get("age_minutes")) is not None and float(row.get("age_minutes")) <= 15.0
            ),
            "median_market": median_market,
            "weighted_mean_market": weighted_mean_market,
            "dispersion_market": _round_or_none(dispersion_market),
            "consensus_gap_vs_official": _round_or_none(consensus_gap),
            "weighted_support": _round_or_none((pws_metrics or {}).get("weighted_support")),
            "top_current_stations": top_current_rows[:6],
        },
        "next_official": next_official if isinstance(next_official, dict) else {"available": False},
        "historical_ranking": history_focus,
    }


def _trade_snapshot(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    return {
        "label": row.get("label"),
        "best_side": row.get("best_side"),
        "tactical_best_side": row.get("tactical_best_side"),
        "selected_entry": _round_or_none(row.get("selected_entry"), 6),
        "selected_fair": _round_or_none(row.get("selected_fair"), 6),
        "best_edge": _round_or_none(row.get("best_edge"), 6),
        "edge_points": _round_or_none(row.get("edge_points"), 4),
        "recommendation": row.get("recommendation"),
        "policy_reason": row.get("policy_reason"),
        "tactical_recommendation": row.get("tactical_recommendation"),
        "tactical_edge_points": _round_or_none(row.get("tactical_edge_points"), 4),
    }


def build_market_model(
    *,
    station_id: str,
    market_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(market_payload, dict):
        return {
            "available": False,
            "reason": "Market payload unavailable.",
        }

    error = market_payload.get("error")
    if error:
        return {
            "available": False,
            "reason": str(error),
            "event_title": None,
            "event_slug": market_payload.get("event_slug"),
        }

    trading = market_payload.get("trading") or {}
    model = trading.get("model") or {}
    all_rows = [
        row for row in (trading.get("all_terminal_opportunities") or [])
        if isinstance(row, dict)
    ]
    all_rows_by_label = {
        str(row.get("label") or ""): row
        for row in all_rows
        if str(row.get("label") or "")
    }
    comparison = []
    for top_label in model.get("top_labels") or []:
        if not isinstance(top_label, dict):
            continue
        label = str(top_label.get("label") or "")
        row = all_rows_by_label.get(label)
        if not row:
            continue
        market_yes = _safe_float(row.get("yes_entry"))
        model_yes = _safe_float(row.get("fair_yes"))
        comparison.append(
            {
                "label": label,
                "model_yes": _round_or_none(model_yes, 6),
                "market_yes": _round_or_none(market_yes, 6),
                "edge_yes_points": _round_or_none(
                    ((float(model_yes) - float(market_yes)) * 100.0)
                    if model_yes is not None and market_yes is not None
                    else None,
                    4,
                ),
                "best_side": row.get("best_side"),
                "recommendation": row.get("recommendation"),
            }
        )

    top_market_buckets = []
    for bracket in (market_payload.get("brackets") or [])[:6]:
        if not isinstance(bracket, dict):
            continue
        top_market_buckets.append(
            {
                "label": bracket.get("name"),
                "yes_price": _round_or_none(bracket.get("yes_price"), 6),
                "volume": _round_or_none(bracket.get("volume"), 2),
            }
        )

    best_terminal = _trade_snapshot(trading.get("best_terminal_trade"))
    best_tactical = _trade_snapshot(trading.get("best_tactical_trade"))
    headline = (trading.get("policy") or {}).get("headline") or "Market loaded."
    if best_terminal and best_terminal.get("edge_points") is not None:
        headline = (
            f"Best live terminal idea: {best_terminal.get('label')} "
            f"{best_terminal.get('best_side')} for {best_terminal.get('edge_points')} pts."
        )

    return {
        "available": True,
        "reason": None,
        "event_title": market_payload.get("event_title"),
        "event_slug": market_payload.get("event_slug"),
        "target_date": market_payload.get("target_date"),
        "total_volume": _round_or_none(market_payload.get("total_volume"), 2),
        "ws_connected": bool(market_payload.get("ws_connected")),
        "headline": headline,
        "market_status": dict(market_payload.get("market_status") or {}),
        "top_market_buckets": top_market_buckets,
        "comparison": comparison,
        "model": {
            "source": model.get("source"),
            "mean": _round_or_none(model.get("mean")),
            "sigma": _round_or_none(model.get("sigma")),
            "confidence": _round_or_none(model.get("confidence"), 4),
            "top_label": model.get("top_label"),
            "top_label_probability": _round_or_none(model.get("top_label_probability"), 6),
            "top_labels": list(model.get("top_labels") or []),
        },
        "forecast_winner": trading.get("forecast_winner"),
        "best_terminal_trade": best_terminal,
        "best_tactical_trade": best_tactical,
        "policy": dict(trading.get("policy") or {}),
    }


def build_legacy_reference(
    *,
    station_id: str,
    legacy_prediction: Optional[Dict[str, Any]],
    component_breakdown: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not isinstance(legacy_prediction, dict):
        return {"available": False}
    market_unit = _market_unit(station_id)
    prediction_f = _safe_float(legacy_prediction.get("final_prediction_f"))
    return {
        "available": True,
        "timestamp": legacy_prediction.get("timestamp"),
        "prediction_f": _round_or_none(prediction_f),
        "prediction_market": _round_or_none(_convert_f_to_market_unit(prediction_f, market_unit)),
        "hrrr_f": _round_or_none(legacy_prediction.get("hrrr_max_raw_f")),
        "physics_adjustment_f": _round_or_none(legacy_prediction.get("physics_adjustment_f")),
        "delta_weight": _round_or_none(legacy_prediction.get("delta_weight"), 4),
        "component_breakdown": list(component_breakdown or []),
    }


def build_data_inventory(
    *,
    target_day: int,
    prior_model: Dict[str, Any],
    intraday_model: Dict[str, Any],
    market_model: Dict[str, Any],
    legacy_reference: Dict[str, Any],
    external_sources: Optional[Dict[str, Any]],
    feature_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    available: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    needs_study: List[Dict[str, Any]] = []

    for row in prior_model.get("source_rows") or []:
        if not isinstance(row, dict):
            continue
        if int(row.get("target_hour_count") or 0) > 0:
            available.append(
                {
                    "title": f"{row.get('display_name')} target curve",
                    "detail": f"{int(row.get('target_hour_count') or 0)} hourly rows for the selected day.",
                }
            )

    if legacy_reference.get("available"):
        available.append(
            {
                "title": "Legacy HELIOS reference",
                "detail": "Physics prediction kept as a comparison layer, not as the main model.",
            }
        )

    if market_model.get("available"):
        comparison_count = len(market_model.get("comparison") or [])
        available.append(
            {
                "title": "Live market pricing",
                "detail": f"{comparison_count} model-vs-market bucket comparisons available.",
            }
        )
    else:
        missing.append(
            {
                "title": "Polymarket market context",
                "detail": market_model.get("reason") or "No market event was available for the selected horizon.",
            }
        )

    if int(target_day) == 0:
        if intraday_model.get("available"):
            pws_count = int(((intraday_model.get("pws_now") or {}).get("station_count")) or 0)
            available.append(
                {
                    "title": "Official + PWS intraday layer",
                    "detail": f"{pws_count} current PWS stations plus official observation for same-day repricing.",
                }
            )
        else:
            missing.append(
                {
                    "title": "Same-day intraday layer",
                    "detail": intraday_model.get("reason") or "No official or PWS live context is available.",
                }
            )

    historical_ranking = intraday_model.get("historical_ranking") or {}
    window_day_count = int(historical_ranking.get("window_day_count") or 0)
    if window_day_count > 0:
        available.append(
            {
                "title": "Historical PWS ranking",
                "detail": f"Aggregated over {window_day_count} recorded market days.",
            }
        )

    feature = feature_summary or {}
    if feature.get("available") and not bool(feature.get("empty_env")):
        available.append(
            {
                "title": "Feature channel",
                "detail": f"{int(feature.get('env_key_count') or 0)} environment keys available in the latest feature snapshot.",
            }
        )
    else:
        missing.append(
            {
                "title": "Feature store for wind/cloud/humidity regimes",
                "detail": (
                    "The recorded `features.env` payload is empty right now, so regime variables are not feeding the model yet."
                    if feature.get("available")
                    else "No recent feature payload was found for this station."
                ),
            }
        )

    missing.append(
        {
            "title": "Historical forecast snapshots",
            "detail": "Current provider curves are available, but run-to-run snapshots are not persisted yet.",
        }
    )

    accuweather = next(
        (
            row
            for row in (external_sources or {}).get("sources") or []
            if isinstance(row, dict) and str(row.get("source") or "") == "accuweather"
        ),
        None,
    )
    if isinstance(accuweather, dict) and accuweather.get("status") != "ok":
        missing.append(
            {
                "title": "AccuWeather provider",
                "detail": "; ".join(accuweather.get("errors") or accuweather.get("notes") or []) or "Scaffold exists but the provider is not active.",
            }
        )

    if window_day_count > 0:
        audit_rows = int(((historical_ranking.get("summary") or {}).get("audit_row_total")) or 0)
        needs_study.append(
            {
                "title": "PWS delay vs next METAR",
                "detail": f"{audit_rows} audit rows are available to study time-to-convergence and delayed hits.",
            }
        )

    if int(prior_model.get("source_count") or 0) >= 2:
        needs_study.append(
            {
                "title": "Prior drift between sources",
                "detail": "The page can compare current source curves, but not yet how each source drifted during the prior day.",
            }
        )

    if feature.get("available") and bool(feature.get("empty_env")):
        needs_study.append(
            {
                "title": "Regime features",
                "detail": "Wind, cloud, humidity and precipitation variables need to be recorded before they can be studied as conditional effects.",
            }
        )

    return {
        "available": available,
        "missing": missing,
        "needs_study": needs_study,
    }


def build_research_backlog(
    *,
    target_day: int,
    prior_model: Dict[str, Any],
    intraday_model: Dict[str, Any],
    feature_summary: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    historical_ranking = intraday_model.get("historical_ranking") or {}
    backlog = [
        {
            "priority": "high",
            "title": "Persist hourly forecast snapshots",
            "detail": "Store Wunderground/Open-Meteo/NOAA hourly curves during the prior day so source drift can be measured against settlement.",
        },
        {
            "priority": "high",
            "title": "Promote PWS path metrics beyond hit/miss",
            "detail": "Add first-hit, last-hit, time-to-convergence and trimmed-mean features per station before ranking them for trading.",
        },
    ]

    feature = feature_summary or {}
    if not feature.get("available") or bool(feature.get("empty_env")):
        backlog.append(
            {
                "priority": "high",
                "title": "Populate the feature channel",
                "detail": "Wind, cloud cover, humidity, precipitation and solar variables are still missing from the recorded feature payload.",
            }
        )

    if int(target_day) == 0 and int(historical_ranking.get("window_day_count") or 0) > 0:
        backlog.append(
            {
                "priority": "medium",
                "title": "Segment PWS behaviour by regime",
                "detail": "Split station behaviour by hour, wind direction, cloud cover and rain so delayed-but-useful stations are not mixed with noisy ones.",
            }
        )

    if int(prior_model.get("source_count") or 0) < 3:
        backlog.append(
            {
                "priority": "medium",
                "title": "Expand provider breadth",
                "detail": "AccuWeather is still scaffold-only, so the prior layer lacks an additional independent forecast source.",
            }
        )

    return backlog


def build_prediction_payload(
    *,
    station_id: str,
    target_day: int,
    target_date: date_cls,
    generated_at: datetime,
    external_sources: Optional[Dict[str, Any]],
    market_payload: Optional[Dict[str, Any]],
    legacy_prediction: Optional[Dict[str, Any]],
    legacy_component_breakdown: Optional[List[Dict[str, Any]]],
    official: Optional[Dict[str, Any]],
    pws_details: Optional[Sequence[Dict[str, Any]]],
    pws_metrics: Optional[Dict[str, Any]],
    pws_history_summary: Optional[Dict[str, Any]],
    observed_max_f: Optional[float],
    feature_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    station = STATIONS[station_id]
    station_tz = _station_tz(station_id)
    market_unit = _market_unit(station_id)

    market_model = build_market_model(station_id=station_id, market_payload=market_payload)
    trading_signal = (market_payload or {}).get("trading") if isinstance(market_payload, dict) else None
    legacy_reference = build_legacy_reference(
        station_id=station_id,
        legacy_prediction=legacy_prediction,
        component_breakdown=legacy_component_breakdown,
    )
    prior_model = build_prior_model(
        station_id=station_id,
        target_date=target_date,
        external_sources=external_sources,
        legacy_prediction=legacy_prediction,
    )
    intraday_model = build_intraday_model(
        station_id=station_id,
        target_day=target_day,
        official=official,
        pws_details=pws_details,
        pws_metrics=pws_metrics,
        pws_history_summary=pws_history_summary,
        observed_max_f=observed_max_f,
        trading_signal=trading_signal if isinstance(trading_signal, dict) else None,
    )
    inventory = build_data_inventory(
        target_day=target_day,
        prior_model=prior_model,
        intraday_model=intraday_model,
        market_model=market_model,
        legacy_reference=legacy_reference,
        external_sources=external_sources,
        feature_summary=feature_summary,
    )
    backlog = build_research_backlog(
        target_day=target_day,
        prior_model=prior_model,
        intraday_model=intraday_model,
        feature_summary=feature_summary,
    )

    horizon_label = {
        0: "Same day",
        1: "1 day prior",
        2: "2 days prior",
    }.get(int(target_day), f"T+{int(target_day)}")

    headline_summary = prior_model.get("headline") or "No prior model available."
    if int(target_day) == 0 and intraday_model.get("available"):
        headline_summary = intraday_model.get("headline") or headline_summary
    if market_model.get("available") and market_model.get("headline"):
        headline_summary = f"{headline_summary} {market_model.get('headline')}"

    return {
        "station_id": station_id,
        "station_name": station.name,
        "timezone": station.timezone,
        "market_unit": market_unit,
        "generated_at_utc": generated_at.astimezone(UTC).isoformat(),
        "generated_at_local": generated_at.astimezone(station_tz).isoformat(),
        "target_day": int(target_day),
        "target_date": target_date.isoformat(),
        "horizon": {
            "code": "INTRADAY" if int(target_day) == 0 else "PRIOR",
            "label": horizon_label,
        },
        "headline": {
            "title": f"{station.name} probability lab",
            "summary": headline_summary,
            "prior_consensus": prior_model.get("consensus_max"),
            "market_best_edge_points": (market_model.get("best_terminal_trade") or {}).get("edge_points"),
            "intraday_direction": ((intraday_model.get("next_official") or {}).get("direction")),
        },
        "data_inventory": inventory,
        "prior_model": prior_model,
        "intraday_model": intraday_model,
        "market_model": market_model,
        "legacy_reference": legacy_reference,
        "feature_summary": feature_summary or {"available": False},
        "research_backlog": backlog,
        "timestamp": legacy_reference.get("timestamp"),
        "prediction_f": legacy_reference.get("prediction_f"),
        "component_breakdown": legacy_reference.get("component_breakdown") or [],
    }
