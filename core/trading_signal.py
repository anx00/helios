from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from config import STATIONS, get_polymarket_temp_unit
from core.market_pricing import normalize_probability_01, select_probability_01_from_quotes
from core.polymarket_labels import label_for_temp, normalize_label, parse_label, sort_labels


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


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, float(value)))


def _round_half_up(value: Any) -> Optional[int]:
    number = _safe_float(value)
    if number is None:
        return None
    return int(Decimal(str(number)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _parse_iso_utc(value: Any) -> Optional[datetime]:
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


def _convert_sigma_f_to_market_unit(sigma_f: Any, market_unit: str) -> Optional[float]:
    sigma = _safe_float(sigma_f)
    if sigma is None:
        return None
    if str(market_unit or "F").upper() == "C":
        return sigma * 5.0 / 9.0
    return sigma


def _convert_sigma_c_to_market_unit(sigma_c: Any, market_unit: str) -> Optional[float]:
    sigma = _safe_float(sigma_c)
    if sigma is None:
        return None
    if str(market_unit or "F").upper() == "C":
        return sigma
    return sigma * 9.0 / 5.0


def _market_timezone(station_id: str) -> ZoneInfo:
    station = STATIONS.get(str(station_id or "").upper())
    tz_name = str(getattr(station, "timezone", "") or "")
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    return UTC


def _market_unit(station_id: str) -> str:
    return str(get_polymarket_temp_unit(station_id)).upper()


def _normal_cdf(value: float, mean: float, sigma: float) -> float:
    sigma = max(0.05, float(sigma))
    z = (float(value) - float(mean)) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _discrete_prob(value: int, mean: float, sigma: float) -> float:
    hi = _normal_cdf(float(value) + 0.5, mean, sigma)
    lo = _normal_cdf(float(value) - 0.5, mean, sigma)
    return max(0.0, hi - lo)


def _gaussian_mass_for_label(
    label: str,
    mean: float,
    sigma: float,
    *,
    market_floor: Optional[int] = None,
    market_ceiling: Optional[int] = None,
) -> float:
    kind, low, high = parse_label(label)
    if kind == "range" and low is not None and high is not None:
        lo = int(low)
        hi = int(high)
        if market_floor is not None:
            lo = max(lo, int(market_floor))
        if market_ceiling is not None:
            hi = min(hi, int(market_ceiling))
        if lo > hi:
            return 0.0
        return sum(_discrete_prob(temp, mean, sigma) for temp in range(lo, hi + 1))

    if kind == "single" and low is not None:
        value = int(low)
        if market_floor is not None and value < int(market_floor):
            return 0.0
        if market_ceiling is not None and value > int(market_ceiling):
            return 0.0
        return _discrete_prob(value, mean, sigma)

    if kind == "below" and high is not None:
        lower_bound = float("-inf") if market_floor is None else (float(int(market_floor)) - 0.5)
        upper_value = int(high)
        if market_ceiling is not None:
            upper_value = min(upper_value, int(market_ceiling))
        upper_bound = float(upper_value) + 0.5
        if lower_bound >= upper_bound:
            return 0.0
        return max(0.0, _normal_cdf(upper_bound, mean, sigma) - _normal_cdf(lower_bound, mean, sigma))

    if kind == "above" and low is not None:
        lower_value = int(low)
        if market_floor is not None:
            lower_value = max(lower_value, int(market_floor))
        if market_ceiling is not None and lower_value > int(market_ceiling):
            return 0.0
        lower_bound = float(lower_value) - 0.5
        if market_ceiling is not None:
            upper_bound = float(int(market_ceiling)) + 0.5
            if lower_bound >= upper_bound:
                return 0.0
            return max(0.0, _normal_cdf(upper_bound, mean, sigma) - _normal_cdf(lower_bound, mean, sigma))
        return max(0.0, 1.0 - _normal_cdf(lower_bound, mean, sigma))

    return 0.0


def _normalize_partition(entries: Sequence[Dict[str, Any]], key: str = "fair_prob") -> List[Dict[str, Any]]:
    total = sum(max(0.0, float(entry.get(key) or 0.0)) for entry in entries)
    if total <= 0:
        return [dict(entry) for entry in entries]

    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        row = dict(entry)
        row[key] = round(max(0.0, float(row.get(key) or 0.0)) / total, 6)
        normalized.append(row)
    return normalized


def _quote_prob(value: Any) -> Optional[float]:
    return normalize_probability_01(value)


def compute_nominal_next_metar(
    official_obs_utc: Optional[datetime],
    reference_utc: Optional[datetime] = None,
) -> Dict[str, Optional[Any]]:
    if official_obs_utc is None:
        return {
            "current_obs": None,
            "next_obs": None,
            "minutes_to_next": None,
            "schedule_minute": None,
        }

    current_obs = official_obs_utc.astimezone(UTC)
    reference = (reference_utc or datetime.now(UTC)).astimezone(UTC)
    next_obs = current_obs.replace(second=0, microsecond=0)
    while next_obs <= reference:
        next_obs = next_obs + timedelta(minutes=60)

    return {
        "current_obs": current_obs,
        "next_obs": next_obs,
        "minutes_to_next": max(0.0, (next_obs - reference).total_seconds() / 60.0),
        "schedule_minute": current_obs.minute,
    }


def _source_reliability_factor(source: Any) -> float:
    normalized = str(source or "").strip().upper()
    if normalized == "SYNOPTIC":
        return 1.0
    if normalized.startswith("MADIS_APRSWXNET"):
        return 0.95
    if normalized.startswith("MADIS_"):
        return 0.9
    if normalized == "WUNDERGROUND":
        return 0.86
    if normalized == "OPEN_METEO":
        return 0.72
    return 0.82


def _lead_window_factor(lead_minutes: Optional[float], min_lead_minutes: float, max_lead_minutes: float) -> float:
    lead = _safe_float(lead_minutes)
    if lead is None:
        return 0.75
    if lead < 0:
        return 0.05
    if lead < min_lead_minutes:
        return _clamp(0.35 + (lead / max(1.0, min_lead_minutes)) * 0.35, 0.15, 0.7)
    if lead <= max_lead_minutes:
        target = min_lead_minutes + ((max_lead_minutes - min_lead_minutes) * 0.38)
        spread = max(8.0, (max_lead_minutes - min_lead_minutes) / 2.0)
        distance = abs(lead - target)
        return _clamp(1.0 - (distance / spread) * 0.22, 0.78, 1.0)
    if lead <= max_lead_minutes + 20.0:
        return 0.45
    return 0.2


def _freshness_factor(age_minutes: Optional[float]) -> float:
    age = _safe_float(age_minutes)
    if age is None:
        return 0.55
    if age <= 10.0:
        return 1.0
    if age <= 20.0:
        return 0.94
    if age <= 35.0:
        return 0.82
    if age <= 60.0:
        return 0.62
    if age <= 90.0:
        return 0.32
    return 0.14


def _spike_penalty(consensus_gap_c: float, official_gap_c: float) -> float:
    consensus_gap = abs(float(consensus_gap_c))
    official_gap = abs(float(official_gap_c))
    if consensus_gap >= 1.8 and official_gap >= 1.4:
        return 0.32
    if consensus_gap >= 1.2 and official_gap >= 1.0:
        return 0.52
    if consensus_gap >= 0.8:
        return 0.72
    if consensus_gap >= 0.45:
        return 0.88
    return 1.0


def _build_learning_profile_map(learning_metric: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    metric = learning_metric if isinstance(learning_metric, dict) else {}
    rows: List[Dict[str, Any]] = []
    for key in ("top_profiles", "rank_top_profiles"):
        payload = metric.get(key)
        if isinstance(payload, list):
            rows.extend(item for item in payload if isinstance(item, dict))

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        station_id = str(row.get("station_id") or "").upper()
        if station_id:
            out[station_id] = row
    return out


def estimate_next_metar_projection(
    *,
    station_id: str,
    rows: Sequence[Dict[str, Any]],
    official_temp_c: Optional[float],
    official_obs_utc: Optional[datetime],
    learning_metric: Optional[Dict[str, Any]],
    consensus_c: Optional[float],
    reference_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    market_unit = _market_unit(station_id)
    valid_rows = [
        row for row in rows
        if isinstance(row, dict)
        and bool(row.get("valid"))
        and _safe_float(row.get("temp_c")) is not None
    ]
    if not valid_rows or official_obs_utc is None:
        return {
            "available": False,
            "market_unit": market_unit,
        }

    schedule = compute_nominal_next_metar(official_obs_utc, reference_utc=reference_utc)
    next_obs = schedule.get("next_obs")
    minutes_to_next = _safe_float(schedule.get("minutes_to_next"))
    if next_obs is None:
        return {
            "available": False,
            "market_unit": market_unit,
        }

    learning = learning_metric if isinstance(learning_metric, dict) else {}
    policy = learning.get("policy") if isinstance(learning.get("policy"), dict) else {}
    lead_window = policy.get("lead_window_minutes") if isinstance(policy.get("lead_window_minutes"), list) else [5, 45]
    min_lead_minutes = _safe_float(lead_window[0]) if len(lead_window) >= 1 else 5.0
    max_lead_minutes = _safe_float(lead_window[1]) if len(lead_window) >= 2 else 45.0
    min_lead_minutes = 5.0 if min_lead_minutes is None else min_lead_minutes
    max_lead_minutes = 45.0 if max_lead_minutes is None else max_lead_minutes
    profile_map = _build_learning_profile_map(learning)

    projected_rows: List[Dict[str, Any]] = []
    weighted_sum = 0.0
    total_weight = 0.0

    for row in valid_rows:
        temp_c = _safe_float(row.get("temp_c"))
        if temp_c is None:
            continue

        station_key = str(row.get("station_id") or "").upper()
        profile = profile_map.get(station_key, {})

        predictive_weight = _safe_float(row.get("learning_weight_predictive"))
        if predictive_weight is None:
            predictive_weight = _safe_float(row.get("learning_weight"))
        if predictive_weight is None:
            predictive_weight = 1.0

        next_score = _safe_float(profile.get("next_metar_score"))
        if next_score is None:
            next_score = _safe_float(row.get("learning_predictive_score"))
        lead_score = _safe_float(profile.get("lead_score"))
        if lead_score is None:
            lead_score = _safe_float(row.get("learning_lead_score"))
        now_score = _safe_float(profile.get("now_score"))
        if now_score is None:
            now_score = _safe_float(row.get("learning_now_score"))

        score_terms: List[tuple[float, float]] = []
        if next_score is not None:
            score_terms.append((next_score / 100.0, 0.5))
        if lead_score is not None:
            score_terms.append((lead_score / 100.0, 0.32))
        if now_score is not None:
            score_terms.append((now_score / 100.0, 0.18))
        score_weight_sum = sum(weight for _, weight in score_terms)
        if score_weight_sum > 0:
            skill_raw = sum(value * weight for value, weight in score_terms) / score_weight_sum
            skill_factor = _clamp(skill_raw, 0.18, 1.0)
        else:
            skill_factor = 0.45

        now_samples = _safe_float(row.get("learning_now_samples")) or 0.0
        lead_samples = _safe_float(row.get("learning_lead_samples")) or 0.0
        sample_factor = _clamp(
            0.45 + 0.28 * min(1.0, now_samples / 8.0) + 0.27 * min(1.0, lead_samples / 6.0),
            0.4,
            1.0,
        )

        age_minutes = _safe_float(row.get("age_minutes"))
        row_obs = _parse_iso_utc(row.get("obs_time_utc"))
        if row_obs is not None:
            lead_minutes = max(0.0, (next_obs - row_obs).total_seconds() / 60.0)
        elif age_minutes is not None and minutes_to_next is not None:
            lead_minutes = age_minutes + minutes_to_next
        else:
            lead_minutes = None

        lead_factor = _lead_window_factor(lead_minutes, min_lead_minutes, max_lead_minutes)
        freshness_factor = _freshness_factor(age_minutes)
        source_factor = _source_reliability_factor(row.get("source"))
        distance_km = _safe_float(row.get("distance_km")) or 9999.0
        distance_factor = _clamp(1.0 - (distance_km / 35.0), 0.35, 1.0)

        bias_c = _safe_float(profile.get("next_metar_bias_c")) or 0.0
        base_projected_c = temp_c - bias_c
        consensus_gap_c = (base_projected_c - consensus_c) if consensus_c is not None else 0.0
        official_gap_c = (base_projected_c - official_temp_c) if official_temp_c is not None else 0.0
        stability_factor = _spike_penalty(consensus_gap_c, official_gap_c)

        reversion_blend = _clamp(
            max(0.0, abs(consensus_gap_c) - 0.35) * 0.22
            + (1.0 - skill_factor) * 0.18
            + (1.0 - freshness_factor) * 0.14,
            0.0,
            0.55,
        )
        projected_c = (
            ((base_projected_c * (1.0 - reversion_blend)) + (consensus_c * reversion_blend))
            if consensus_c is not None
            else base_projected_c
        )

        final_weight = (
            predictive_weight
            * skill_factor
            * sample_factor
            * freshness_factor
            * source_factor
            * distance_factor
            * lead_factor
            * stability_factor
        )
        if final_weight <= 0.025:
            continue

        sigma_c = _safe_float(profile.get("next_metar_sigma_c"))
        if sigma_c is None:
            sigma_c = max(0.25, 1.15 - (0.6 * skill_factor))

        contributor = {
            "station_id": station_key,
            "source": row.get("source"),
            "temp_c": round(temp_c, 3),
            "projected_c": round(projected_c, 3),
            "weight": round(final_weight, 4),
            "skill_factor": round(skill_factor, 4),
            "lead_minutes": round(float(lead_minutes), 2) if lead_minutes is not None else None,
            "age_minutes": round(float(age_minutes), 2) if age_minutes is not None else None,
            "next_metar_sigma_c": round(float(sigma_c), 4),
        }
        projected_rows.append(contributor)
        total_weight += final_weight
        weighted_sum += projected_c * final_weight

    if total_weight <= 0 or not projected_rows:
        return {
            "available": False,
            "market_unit": market_unit,
        }

    expected_c = weighted_sum / total_weight
    weighted_var = 0.0
    weighted_skill = 0.0
    weighted_freshness = 0.0
    for row in projected_rows:
        row_weight = float(row["weight"])
        row_mean = float(row["projected_c"])
        row_sigma = float(row["next_metar_sigma_c"])
        weighted_var += row_weight * ((row_sigma ** 2) + ((row_mean - expected_c) ** 2))
        weighted_skill += row_weight * float(row["skill_factor"])
        age_minutes = _safe_float(row.get("age_minutes"))
        weighted_freshness += row_weight * _freshness_factor(age_minutes)

    sigma_c = math.sqrt(max(0.04, weighted_var / total_weight))
    sigma_c = _clamp(sigma_c, 0.2, 1.8)

    weighted_skill /= total_weight
    weighted_freshness /= total_weight

    expected_market = _convert_c_to_market_unit(expected_c, market_unit)
    sigma_market = _convert_sigma_c_to_market_unit(sigma_c, market_unit)
    current_market = _convert_c_to_market_unit(official_temp_c, market_unit)

    aligned_market = _round_half_up(expected_market)
    current_aligned_market = _round_half_up(current_market)
    delta_market = None
    if expected_market is not None and current_market is not None:
        delta_market = float(expected_market) - float(current_market)

    direction = "FLAT"
    if delta_market is not None:
        threshold = 0.35 if market_unit == "C" else 0.6
        if delta_market >= threshold:
            direction = "UP"
        elif delta_market <= -threshold:
            direction = "DOWN"

    agreement_term = 1.0 - min(1.0, sigma_c / 1.8)
    confidence = _clamp(
        0.25
        + 0.35 * weighted_skill
        + 0.20 * agreement_term
        + 0.20 * weighted_freshness,
        0.2,
        0.96,
    )

    if expected_market is not None and sigma_market is not None:
        center = _round_half_up(expected_market) or 0
        values = range(center - 2, center + 3)
        probs = []
        for value in values:
            prob = _discrete_prob(value, float(expected_market), max(0.15, float(sigma_market)))
            probs.append({"value": int(value), "probability": round(prob, 6)})
        total_prob = sum(item["probability"] for item in probs)
        if total_prob > 0:
            probabilities = [
                {
                    "value": item["value"],
                    "probability": round(item["probability"] / total_prob, 6),
                }
                for item in probs
            ]
        else:
            probabilities = []
    else:
        probabilities = []

    projected_rows.sort(key=lambda row: float(row.get("weight") or 0.0), reverse=True)

    return {
        "available": True,
        "market_unit": market_unit,
        "expected_c": round(float(expected_c), 4),
        "sigma_c": round(float(sigma_c), 4),
        "expected_market": round(float(expected_market), 4) if expected_market is not None else None,
        "sigma_market": round(float(sigma_market), 4) if sigma_market is not None else None,
        "aligned_market": aligned_market,
        "current_market": round(float(current_market), 4) if current_market is not None else None,
        "current_aligned_market": current_aligned_market,
        "delta_market": round(float(delta_market), 4) if delta_market is not None else None,
        "direction": direction,
        "minutes_to_next": round(float(minutes_to_next), 2) if minutes_to_next is not None else None,
        "next_obs_utc": next_obs.isoformat(),
        "quality": round(float(confidence), 4),
        "confidence": round(float(confidence), 4),
        "contributors": projected_rows[:10],
        "probabilities": probabilities,
    }


def estimate_next_metar_c(
    *,
    station_id: str,
    rows: Sequence[Dict[str, Any]],
    official_temp_c: Optional[float],
    official_obs_utc: Optional[datetime],
    learning_metric: Optional[Dict[str, Any]],
    consensus_c: Optional[float],
    reference_utc: Optional[datetime] = None,
) -> Optional[float]:
    projection = estimate_next_metar_projection(
        station_id=station_id,
        rows=rows,
        official_temp_c=official_temp_c,
        official_obs_utc=official_obs_utc,
        learning_metric=learning_metric,
        consensus_c=consensus_c,
        reference_utc=reference_utc,
    )
    return _safe_float(projection.get("expected_c"))


def _coerce_nowcast_distribution(nowcast_distribution: Any) -> Optional[Dict[str, Any]]:
    if nowcast_distribution is None:
        return None
    if isinstance(nowcast_distribution, dict):
        return dict(nowcast_distribution)
    if hasattr(nowcast_distribution, "to_dict"):
        try:
            payload = nowcast_distribution.to_dict()
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
    result: Dict[str, Any] = {}
    for field_name in ("tmax_mean_f", "tmax_sigma_f", "confidence", "t_peak_expected_hour", "target_date"):
        value = getattr(nowcast_distribution, field_name, None)
        if value is not None:
            result[field_name] = value
    return result or None


def _extract_nowcast_model(
    *,
    station_id: str,
    target_day: int,
    nowcast_distribution: Optional[Dict[str, Any]],
    nowcast_state: Optional[Dict[str, Any]],
    prediction: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    market_unit = _market_unit(station_id)
    if target_day == 0 and nowcast_distribution:
        mean_f = _safe_float(nowcast_distribution.get("tmax_mean_f"))
        sigma_f = _safe_float(nowcast_distribution.get("tmax_sigma_f"))
        if mean_f is not None and sigma_f is not None:
            confidence = _clamp(_safe_float(nowcast_distribution.get("confidence")) or 0.55, 0.2, 0.98)
            raw_floor_f = None
            if isinstance(nowcast_state, dict):
                raw_floor_f = _safe_float(nowcast_state.get("max_so_far_raw_f"))
                if raw_floor_f is None:
                    raw_floor_f = _safe_float(nowcast_state.get("max_so_far_f"))
            market_floor = None
            if raw_floor_f is not None and raw_floor_f > -900:
                converted_floor = _convert_f_to_market_unit(raw_floor_f, market_unit)
                market_floor = _round_half_up(converted_floor)

            peak_hour = int(nowcast_distribution.get("t_peak_expected_hour") or 14)
            market_ceiling = None
            exact_market_ceiling = None
            current_hour = None
            hours_to_peak = None
            if isinstance(nowcast_state, dict):
                breakdown = nowcast_state.get("tmax_breakdown") if isinstance(nowcast_state.get("tmax_breakdown"), dict) else {}
                peak_hour = int(
                    _safe_float(breakdown.get("peak_hour"))
                    or _safe_float(((nowcast_state.get("base_forecast") or {}).get("t_peak_hour") if isinstance(nowcast_state.get("base_forecast"), dict) else None))
                    or peak_hour
                )
                current_hour = _safe_float(breakdown.get("current_hour"))
                hours_to_peak = _safe_float(breakdown.get("hours_to_peak"))

                post_peak_cap_f = _safe_float(breakdown.get("post_peak_cap_f"))
                if post_peak_cap_f is None:
                    remaining_max_f = _safe_float(breakdown.get("remaining_max_f"))
                    if remaining_max_f is not None and current_hour is not None and current_hour >= peak_hour + 1:
                        post_peak_cap_f = remaining_max_f
                if post_peak_cap_f is not None:
                    exact_market_ceiling = _convert_f_to_market_unit(post_peak_cap_f, market_unit)
                    market_ceiling = _round_half_up(exact_market_ceiling)

            mean_market = _convert_f_to_market_unit(mean_f, market_unit)
            sigma_market = _convert_sigma_f_to_market_unit(sigma_f, market_unit)
            return {
                "source": "NOWCAST",
                "mean_market": round(float(mean_market), 4) if mean_market is not None else None,
                "sigma_market": round(float(max(0.15, sigma_market or 0.0)), 4) if sigma_market is not None else None,
                "confidence": round(float(confidence), 4),
                "market_floor": market_floor,
                "market_ceiling": market_ceiling,
                "market_ceiling_exact": round(float(exact_market_ceiling), 4) if exact_market_ceiling is not None else None,
                "peak_hour": peak_hour,
                "current_hour": round(float(current_hour), 4) if current_hour is not None else None,
                "hours_to_peak": round(float(hours_to_peak), 4) if hours_to_peak is not None else None,
            }

    if not isinstance(prediction, dict):
        return None

    mean_f = _safe_float(prediction.get("final_prediction_f"))
    if mean_f is None:
        return None

    delta_weight = _clamp(_safe_float(prediction.get("delta_weight")) or 0.45, 0.0, 1.0)
    current_deviation = abs(_safe_float(prediction.get("current_deviation_f")) or 0.0)
    physics_adjustment = abs(_safe_float(prediction.get("physics_adjustment_f")) or 0.0)

    base_sigma_f = 2.15 if target_day == 0 else 2.65
    sigma_f = (
        base_sigma_f
        - (0.65 * delta_weight)
        + max(0.0, current_deviation - 1.5) * 0.09
        + max(0.0, physics_adjustment - 2.0) * 0.04
    )
    sigma_f = _clamp(sigma_f, 1.05 if target_day == 0 else 1.45, 4.25)

    confidence = 0.48 + (0.26 * delta_weight)
    if target_day > 0:
        confidence -= 0.08
    confidence -= min(0.08, max(0.0, current_deviation - 2.0) * 0.02)
    confidence = _clamp(confidence, 0.25, 0.82)

    mean_market = _convert_f_to_market_unit(mean_f, market_unit)
    sigma_market = _convert_sigma_f_to_market_unit(sigma_f, market_unit)
    return {
        "source": "PHYSICS",
        "mean_market": round(float(mean_market), 4) if mean_market is not None else None,
        "sigma_market": round(float(max(0.15, sigma_market or 0.0)), 4) if sigma_market is not None else None,
        "confidence": round(float(confidence), 4),
        "market_floor": None,
        "market_ceiling": None,
        "market_ceiling_exact": None,
        "peak_hour": None,
        "current_hour": None,
        "hours_to_peak": None,
    }


def _coerce_target_date(value: Any, reference_utc: datetime, station_id: str, target_day: int) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    tz = _market_timezone(station_id)
    return reference_utc.astimezone(tz).date() + timedelta(days=int(target_day))


def _build_fair_probabilities(
    *,
    labels: Sequence[str],
    mean_market: float,
    sigma_market: float,
    market_floor: Optional[int],
    market_ceiling: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label in sort_labels(labels):
        fair_prob = _gaussian_mass_for_label(
            label,
            mean_market,
            sigma_market,
            market_floor=market_floor,
            market_ceiling=market_ceiling,
        )
        rows.append({"label": label, "fair_prob": fair_prob})
    return _normalize_partition(rows, key="fair_prob")


def _repricing_influence(
    *,
    station_id: str,
    next_projection: Optional[Dict[str, Any]],
    terminal_model: Dict[str, Any],
    reference_utc: datetime,
) -> float:
    if not next_projection or not bool(next_projection.get("available")):
        return 0.0
    next_conf = _safe_float(next_projection.get("confidence")) or 0.0
    terminal_conf = _safe_float(terminal_model.get("confidence")) or 0.0
    minutes_to_next = _safe_float(next_projection.get("minutes_to_next")) or 60.0
    next_delta = abs(_safe_float(next_projection.get("delta_market")) or 0.0)
    if next_delta <= (0.2 if _market_unit(station_id) == "C" else 0.35):
        return 0.0

    peak_hour = terminal_model.get("peak_hour")
    if peak_hour is None:
        peak_term = 0.55
    else:
        station_now = reference_utc.astimezone(_market_timezone(station_id))
        hours_from_peak = max(0.0, float(station_now.hour + station_now.minute / 60.0) - float(peak_hour))
        peak_term = _clamp(1.0 - (hours_from_peak / 6.0), 0.35, 1.0)

    minutes_term = _clamp(1.0 - (minutes_to_next / 90.0), 0.35, 1.0)
    return _clamp(
        0.18 + 0.40 * next_conf + 0.18 * terminal_conf + 0.16 * peak_term + 0.08 * minutes_term,
        0.18,
        0.85,
    )


def build_trading_signal(
    *,
    station_id: str,
    target_day: int,
    target_date: Any,
    brackets: Sequence[Dict[str, Any]],
    prediction: Optional[Dict[str, Any]] = None,
    nowcast_distribution: Any = None,
    nowcast_state: Optional[Dict[str, Any]] = None,
    official: Optional[Dict[str, Any]] = None,
    pws_details: Optional[Sequence[Dict[str, Any]]] = None,
    pws_metrics: Optional[Dict[str, Any]] = None,
    reference_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    ref_utc = (reference_utc or datetime.now(UTC)).astimezone(UTC)
    market_unit = _market_unit(station_id)
    target_date_obj = _coerce_target_date(target_date, ref_utc, station_id, target_day)
    nowcast_payload = _coerce_nowcast_distribution(nowcast_distribution)
    terminal_model = _extract_nowcast_model(
        station_id=station_id,
        target_day=target_day,
        nowcast_distribution=nowcast_payload,
        nowcast_state=nowcast_state,
        prediction=prediction,
    )

    if not terminal_model:
        return {
            "available": False,
            "station_id": station_id,
            "target_day": int(target_day),
            "target_date": target_date_obj.isoformat(),
            "market_unit": market_unit,
        }

    labels = [
        normalize_label(str(row.get("name") or row.get("bracket") or ""))
        for row in brackets
        if isinstance(row, dict) and str(row.get("name") or row.get("bracket") or "").strip()
    ]
    ordered_labels = sort_labels(labels)
    fair_rows = _build_fair_probabilities(
        labels=ordered_labels,
        mean_market=float(terminal_model["mean_market"]),
        sigma_market=max(0.15, float(terminal_model["sigma_market"])),
        market_floor=terminal_model.get("market_floor"),
        market_ceiling=terminal_model.get("market_ceiling"),
    )
    fair_map = {row["label"]: row["fair_prob"] for row in fair_rows}

    next_projection = None
    if target_day == 0:
        official_temp_c = None
        official_obs_utc = None
        if isinstance(official, dict):
            official_temp_c = _safe_float(official.get("temp_c"))
            official_obs_utc = _parse_iso_utc(official.get("obs_time_utc"))

        learning_metric = None
        if isinstance(pws_metrics, dict):
            learning_metric = (pws_metrics.get("learning") if isinstance(pws_metrics.get("learning"), dict) else {})

        consensus_c = None
        if pws_details:
            valid_temps = [
                _safe_float(row.get("temp_c"))
                for row in pws_details
                if isinstance(row, dict) and bool(row.get("valid"))
            ]
            nums = [temp for temp in valid_temps if temp is not None]
            if nums:
                nums.sort()
                mid = len(nums) // 2
                if len(nums) % 2 == 0:
                    consensus_c = (nums[mid - 1] + nums[mid]) / 2.0
                else:
                    consensus_c = nums[mid]

        next_projection = estimate_next_metar_projection(
            station_id=station_id,
            rows=list(pws_details or []),
            official_temp_c=official_temp_c,
            official_obs_utc=official_obs_utc,
            learning_metric=learning_metric,
            consensus_c=consensus_c,
            reference_utc=ref_utc,
        )

    tactical_map: Dict[str, float] = {}
    tactical_mean = None
    tactical_influence = 0.0
    if target_day == 0 and next_projection and bool(next_projection.get("available")):
        next_delta = _safe_float(next_projection.get("delta_market")) or 0.0
        tactical_influence = _repricing_influence(
            station_id=station_id,
            next_projection=next_projection,
            terminal_model=terminal_model,
            reference_utc=ref_utc,
        )
        if tactical_influence > 0:
            tactical_mean = float(terminal_model["mean_market"]) + (next_delta * tactical_influence)
            tactical_rows = _build_fair_probabilities(
                labels=ordered_labels,
                mean_market=tactical_mean,
                sigma_market=max(0.15, float(terminal_model["sigma_market"])),
                market_floor=terminal_model.get("market_floor"),
                market_ceiling=terminal_model.get("market_ceiling"),
            )
            tactical_map = {row["label"]: row["fair_prob"] for row in tactical_rows}

    opportunities: List[Dict[str, Any]] = []
    for row in brackets:
        if not isinstance(row, dict):
            continue
        label = normalize_label(str(row.get("name") or row.get("bracket") or ""))
        if not label:
            continue

        fair_yes = float(fair_map.get(label, 0.0))
        tactical_yes = float(tactical_map.get(label, fair_yes))

        yes_bid = _quote_prob(row.get("ws_yes_best_bid", row.get("ws_best_bid")))
        yes_ask = _quote_prob(row.get("ws_yes_best_ask", row.get("ws_best_ask")))
        yes_mid = _quote_prob(row.get("ws_yes_mid", row.get("ws_mid")))
        yes_ref = _quote_prob(row.get("yes_price"))
        market_yes = select_probability_01_from_quotes(
            best_bid=yes_bid,
            best_ask=yes_ask,
            mid=yes_mid,
            reference=yes_ref,
            wide_spread_threshold=0.04,
            reference_tolerance=0.005,
        )

        no_bid = _quote_prob(row.get("ws_no_best_bid"))
        no_ask = _quote_prob(row.get("ws_no_best_ask"))
        no_mid = _quote_prob(row.get("ws_no_mid"))
        no_ref = _quote_prob(row.get("no_price"))
        market_no = select_probability_01_from_quotes(
            best_bid=no_bid,
            best_ask=no_ask,
            mid=no_mid,
            reference=no_ref,
            wide_spread_threshold=0.04,
            reference_tolerance=0.005,
        )

        yes_entry = yes_ask if yes_ask is not None else yes_ref
        no_entry = no_ask if no_ask is not None else no_ref
        edge_yes = (fair_yes - yes_entry) if yes_entry is not None else None
        edge_no = ((1.0 - fair_yes) - no_entry) if no_entry is not None else None
        tactical_edge_yes = (tactical_yes - yes_entry) if yes_entry is not None else None
        tactical_edge_no = ((1.0 - tactical_yes) - no_entry) if no_entry is not None else None

        best_side = "YES"
        best_edge = edge_yes if edge_yes is not None else -999.0
        tactical_best_edge = tactical_edge_yes if tactical_edge_yes is not None else -999.0
        if edge_no is not None and edge_no > best_edge:
            best_side = "NO"
            best_edge = edge_no
        if tactical_edge_no is not None and tactical_edge_no > tactical_best_edge:
            tactical_best_edge = tactical_edge_no

        selected_entry = yes_entry if best_side == "YES" else no_entry
        selected_market = market_yes if best_side == "YES" else market_no
        selected_fair = fair_yes if best_side == "YES" else (1.0 - fair_yes)
        selected_tactical_fair = tactical_yes if best_side == "YES" else (1.0 - tactical_yes)

        side_spread = None
        side_depth = None
        side_staleness = None
        if best_side == "YES":
            side_spread = _safe_float(row.get("ws_yes_spread", row.get("ws_spread")))
            side_depth = _safe_float(row.get("ws_yes_ask_depth", row.get("ws_ask_depth")))
            side_staleness = _safe_float(row.get("ws_yes_staleness_ms", row.get("ws_staleness_ms")))
        else:
            side_spread = _safe_float(row.get("ws_no_spread"))
            side_depth = _safe_float(row.get("ws_no_ask_depth"))
            side_staleness = _safe_float(row.get("ws_no_staleness_ms"))

        spread_factor = 0.74 if side_spread is None else _clamp(1.0 - (side_spread / 0.08), 0.35, 1.0)
        depth_factor = 0.62 if side_depth is None else _clamp(0.55 + min(0.45, side_depth / 300.0), 0.55, 1.0)
        stale_factor = 0.72 if side_staleness is None else _clamp(1.0 - (side_staleness / 90000.0), 0.4, 1.0)
        quality = spread_factor * depth_factor * stale_factor * float(terminal_model["confidence"])

        recommendation = "HOLD"
        if best_edge >= 0.07:
            recommendation = f"BUY_{best_side}"
        elif best_edge >= 0.04:
            recommendation = f"LEAN_{best_side}"
        elif best_edge >= 0.02:
            recommendation = f"WATCH_{best_side}"

        tactical_recommendation = "HOLD"
        if tactical_best_edge >= 0.06:
            tactical_recommendation = f"BUY_{best_side}"
        elif tactical_best_edge >= 0.03:
            tactical_recommendation = f"LEAN_{best_side}"

        opportunities.append({
            "label": label,
            "volume": round(float(_safe_float(row.get("volume")) or 0.0), 2),
            "bucket_yes_probability": round(fair_yes, 6),
            "bucket_no_probability": round(1.0 - fair_yes, 6),
            "fair_yes": round(fair_yes, 6),
            "fair_no": round(1.0 - fair_yes, 6),
            "tactical_yes": round(tactical_yes, 6),
            "tactical_no": round(1.0 - tactical_yes, 6),
            "market_yes": round(float(market_yes), 6) if market_yes is not None else None,
            "market_no": round(float(market_no), 6) if market_no is not None else None,
            "yes_entry": round(float(yes_entry), 6) if yes_entry is not None else None,
            "no_entry": round(float(no_entry), 6) if no_entry is not None else None,
            "selected_market": round(float(selected_market), 6) if selected_market is not None else None,
            "selected_entry": round(float(selected_entry), 6) if selected_entry is not None else None,
            "selected_fair": round(float(selected_fair), 6),
            "selected_tactical_fair": round(float(selected_tactical_fair), 6),
            "edge_yes": round(float(edge_yes), 6) if edge_yes is not None else None,
            "edge_no": round(float(edge_no), 6) if edge_no is not None else None,
            "tactical_edge_yes": round(float(tactical_edge_yes), 6) if tactical_edge_yes is not None else None,
            "tactical_edge_no": round(float(tactical_edge_no), 6) if tactical_edge_no is not None else None,
            "best_side": best_side,
            "best_edge": round(float(best_edge), 6),
            "tactical_best_edge": round(float(tactical_best_edge), 6),
            "edge_points": round(float(best_edge) * 100.0, 4),
            "tactical_edge_points": round(float(tactical_best_edge) * 100.0, 4),
            "score": round(float(best_edge) * 100.0 * quality, 4),
            "tactical_score": round(float(tactical_best_edge) * 100.0 * quality, 4),
            "recommendation": recommendation,
            "tactical_recommendation": tactical_recommendation,
        })

    opportunities.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    tactical_ranked = sorted(opportunities, key=lambda row: float(row.get("tactical_score") or 0.0), reverse=True)

    top_labels = sorted(fair_rows, key=lambda row: float(row.get("fair_prob") or 0.0), reverse=True)
    top_label = top_labels[0]["label"] if top_labels else None
    top_label_prob = top_labels[0]["fair_prob"] if top_labels else None
    top_label_complement_prob = (1.0 - float(top_label_prob)) if top_label_prob is not None else None

    expected_label, _ = label_for_temp(float(terminal_model["mean_market"]), ordered_labels)

    best_terminal_trade = next(
        (row for row in opportunities if float(row.get("best_edge") or 0.0) > 0.0),
        None,
    )
    best_tactical_trade = next(
        (row for row in tactical_ranked if float(row.get("tactical_best_edge") or 0.0) > 0.0),
        None,
    )

    final_market_module = {
        "type": "terminal_distribution",
        "source": terminal_model["source"],
        "mean": round(float(terminal_model["mean_market"]), 4),
        "sigma": round(float(terminal_model["sigma_market"]), 4),
        "quality": round(float(terminal_model["confidence"]), 4),
        "top_bucket": top_label,
        "top_bucket_probability": round(float(top_label_prob), 6) if top_label_prob is not None else None,
        "top_bucket_complement_probability": round(float(top_label_complement_prob), 6) if top_label_complement_prob is not None else None,
        "expected_bucket": expected_label,
        "ceiling": terminal_model.get("market_ceiling"),
    }

    return {
        "available": True,
        "station_id": station_id,
        "target_day": int(target_day),
        "target_date": target_date_obj.isoformat(),
        "market_unit": market_unit,
        "horizon": "INTRADAY" if int(target_day) == 0 else "DAY_AHEAD",
        "model": {
            "source": terminal_model["source"],
            "mean": round(float(terminal_model["mean_market"]), 4),
            "sigma": round(float(terminal_model["sigma_market"]), 4),
            "quality": round(float(terminal_model["confidence"]), 4),
            "confidence": round(float(terminal_model["confidence"]), 4),
            "observed_floor": terminal_model.get("market_floor"),
            "ceiling": terminal_model.get("market_ceiling"),
            "expected_label": expected_label,
            "top_label": top_label,
            "top_label_probability": round(float(top_label_prob), 6) if top_label_prob is not None else None,
            "top_label_complement_probability": round(float(top_label_complement_prob), 6) if top_label_complement_prob is not None else None,
            "top_labels": [
                {
                    "label": row["label"],
                    "probability": round(float(row["fair_prob"]), 6),
                }
                for row in top_labels[:5]
            ],
        },
        "best_terminal_trade": best_terminal_trade,
        "best_tactical_trade": best_tactical_trade,
        "terminal_opportunities": opportunities[:8],
        "tactical_context": {
            "enabled": bool(tactical_map),
            "repricing_influence": round(float(tactical_influence), 4),
            "tactical_mean": round(float(tactical_mean), 4) if tactical_mean is not None else None,
            "next_metar": next_projection,
        },
        "modules": {
            "final_market": final_market_module,
            "next_official": next_projection,
            "market_pricing": {
                "type": "market_pricing",
                "repricing_influence": round(float(tactical_influence), 4),
                "best_terminal_trade": best_terminal_trade,
                "best_tactical_trade": best_tactical_trade,
            },
        },
    }
