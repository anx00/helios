from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
from zoneinfo import ZoneInfo

import aiohttp

from collector.metar_fetcher import fetch_metar_history, fetch_metar_race
from collector.pws_fetcher import fetch_and_publish_pws
from config import STATIONS, get_polymarket_temp_unit
from core.world import get_world
from database import (
    get_latest_prediction,
    get_latest_prediction_for_date,
    is_telegram_station_subscription_enabled,
)

logger = logging.getLogger("telegram_bot")
_LIVE_TELEGRAM_BOT: Optional["HeliosTelegramBot"] = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_allowed_chat_ids(raw: str) -> Set[str]:
    values: Set[str] = set()
    for part in (raw or "").split(","):
        item = part.strip()
        if item:
            values.add(item)
    return values


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, decimals: int = 2, suffix: str = "") -> str:
    num = _safe_float(value)
    if num is None:
        return "n/a"
    return f"{num:.{decimals}f}{suffix}"


def _f_to_c(value_f: Any) -> Optional[float]:
    num = _safe_float(value_f)
    if num is None:
        return None
    return (num - 32.0) * 5.0 / 9.0


def _c_to_f(value_c: Any) -> Optional[float]:
    num = _safe_float(value_c)
    if num is None:
        return None
    return (num * 9.0 / 5.0) + 32.0


def _station_temp_unit(station_id: str) -> str:
    return "C" if get_polymarket_temp_unit(station_id).upper() == "C" else "F"


def _fmt_station_temp_from_f(value_f: Any, station_id: str, decimals: int = 1) -> str:
    unit = _station_temp_unit(station_id)
    if unit == "C":
        return _fmt(_f_to_c(value_f), decimals, "C")
    return _fmt(value_f, decimals, "F")


def _fmt_station_temp_from_c(value_c: Any, station_id: str, decimals: int = 1) -> str:
    unit = _station_temp_unit(station_id)
    if unit == "C":
        return _fmt(value_c, decimals, "C")
    return _fmt(_c_to_f(value_c), decimals, "F")


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if _safe_float(v) is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, float(value)))


def _weighted_median(entries: Iterable[tuple[float, float]]) -> Optional[float]:
    valid = sorted((float(value), float(weight)) for value, weight in entries if float(weight) > 0)
    if not valid:
        return None
    total_weight = sum(weight for _, weight in valid)
    if total_weight <= 0:
        return None
    midpoint = total_weight / 2.0
    running = 0.0
    for value, weight in valid:
        running += weight
        if running >= midpoint:
            return value
    return valid[-1][0]


def _parse_iso_utc(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(ZoneInfo("UTC"))
    except Exception:
        return None


def _pws_source_bucket(source: str) -> str:
    src = str(source or "").upper()
    if src == "SYNOPTIC":
        return "Synoptic"
    if src.startswith("MADIS_"):
        return "MADIS"
    if src == "OPEN_METEO":
        return "Open-Meteo"
    if src == "WUNDERGROUND":
        return "Wunderground"
    return "Other"


def _pws_source_tag(source: str) -> str:
    src = str(source or "").upper()
    if src == "SYNOPTIC":
        return "SYN"
    if src.startswith("MADIS_"):
        return "MAD"
    if src == "OPEN_METEO":
        return "OM"
    if src == "WUNDERGROUND":
        return "WU"
    return "OTH"


def _is_valid_pws_row(row: Dict[str, Any]) -> bool:
    return bool(row.get("valid")) and _safe_float(row.get("temp_c")) is not None


def _pws_weight(row: Dict[str, Any]) -> float:
    weight = _safe_float(row.get("learning_weight"))
    if weight is not None and weight > 0:
        return max(0.05, weight)
    return 1.0


def _normalize_pws_source(source: Any) -> str:
    return str(source or "").strip().upper()


def _rank_pws_rows(rows: Sequence[Dict[str, Any]], metric_key: str) -> List[Dict[str, Any]]:
    def _metric(row: Dict[str, Any]) -> Optional[float]:
        return _safe_float(row.get(metric_key))

    def _sort_key(row: Dict[str, Any]) -> tuple:
        metric = _metric(row)
        weight = _safe_float(row.get("learning_weight"))
        dist = _safe_float(row.get("distance_km"))
        return (
            0 if metric is not None else 1,
            -(metric if metric is not None else 0.0),
            0 if weight is not None else 1,
            -(weight if weight is not None else 0.0),
            dist if dist is not None else 9999.0,
            str(row.get("station_id") or ""),
        )

    return sorted(list(rows), key=_sort_key)


def _compute_pws_weighted_consensus_c(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    entries: List[tuple[float, float]] = []
    for row in rows:
        temp_c = _safe_float(row.get("temp_c"))
        if temp_c is None:
            continue
        entries.append((temp_c, _pws_weight(row)))
    return _weighted_median(entries)


def _compute_nominal_next_metar(
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

    current_obs = official_obs_utc.astimezone(ZoneInfo("UTC"))
    reference = (reference_utc or datetime.now(ZoneInfo("UTC"))).astimezone(ZoneInfo("UTC"))
    next_obs = current_obs.replace(second=0, microsecond=0)
    while next_obs <= reference:
        next_obs = next_obs.replace(second=0, microsecond=0) + timedelta(minutes=60)

    return {
        "current_obs": current_obs,
        "next_obs": next_obs,
        "minutes_to_next": max(0.0, (next_obs - reference).total_seconds() / 60.0),
        "schedule_minute": current_obs.minute,
    }


def _source_reliability_factor(source: Any) -> float:
    normalized = _normalize_pws_source(source)
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
    if lead <= max_lead_minutes + 20:
        return 0.45
    return 0.2


def _freshness_factor(age_minutes: Optional[float]) -> float:
    age = _safe_float(age_minutes)
    if age is None:
        return 0.55
    if age <= 10:
        return 1.0
    if age <= 20:
        return 0.94
    if age <= 35:
        return 0.82
    if age <= 60:
        return 0.62
    if age <= 90:
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


def _build_learning_profile_map(learning_metric: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    top_profiles = learning_metric.get("top_profiles")
    rank_top_profiles = learning_metric.get("rank_top_profiles")
    if isinstance(top_profiles, list):
        rows.extend(p for p in top_profiles if isinstance(p, dict))
    if isinstance(rank_top_profiles, list):
        rows.extend(p for p in rank_top_profiles if isinstance(p, dict))

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        station_id = str(row.get("station_id") or "").upper()
        if station_id:
            out[station_id] = row
    return out


def _estimate_next_metar_c(
    rows: Sequence[Dict[str, Any]],
    official_temp_c: Optional[float],
    official_obs_utc: Optional[datetime],
    learning_metric: Optional[Dict[str, Any]],
    consensus_c: Optional[float],
    reference_utc: Optional[datetime] = None,
) -> Optional[float]:
    valid_rows = [row for row in rows if _is_valid_pws_row(row)]
    if not valid_rows or official_obs_utc is None:
        return None

    schedule = _compute_nominal_next_metar(official_obs_utc, reference_utc=reference_utc)
    next_obs = schedule.get("next_obs")
    minutes_to_next = _safe_float(schedule.get("minutes_to_next"))
    if next_obs is None:
        return None

    learning = learning_metric if isinstance(learning_metric, dict) else {}
    policy = learning.get("policy") if isinstance(learning.get("policy"), dict) else {}
    lead_window = policy.get("lead_window_minutes") if isinstance(policy.get("lead_window_minutes"), list) else [5, 45]
    min_lead_minutes = _safe_float(lead_window[0]) if len(lead_window) >= 1 else None
    max_lead_minutes = _safe_float(lead_window[1]) if len(lead_window) >= 2 else None
    min_lead_minutes = 5.0 if min_lead_minutes is None else min_lead_minutes
    max_lead_minutes = 45.0 if max_lead_minutes is None else max_lead_minutes
    profile_map = _build_learning_profile_map(learning)

    total_weight = 0.0
    projected_sum = 0.0

    for row in valid_rows:
        temp_c = _safe_float(row.get("temp_c"))
        if temp_c is None:
            continue

        station_id = str(row.get("station_id") or "").upper()
        profile = profile_map.get(station_id, {})
        predictive_weight = _safe_float(row.get("learning_weight_predictive"))
        if predictive_weight is None:
            predictive_weight = _safe_float(row.get("learning_weight"))
        if predictive_weight is None:
            predictive_weight = _pws_weight(row)

        next_score = _safe_float(profile.get("next_metar_score"))
        if next_score is None:
            next_score = _safe_float(row.get("learning_predictive_score"))
        lead_score = _safe_float(row.get("learning_lead_score"))
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
        fresh_factor = _freshness_factor(age_minutes)
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
            + (1.0 - fresh_factor) * 0.14,
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
            * fresh_factor
            * source_factor
            * distance_factor
            * lead_factor
            * stability_factor
        )
        if final_weight <= 0.025:
            continue

        total_weight += final_weight
        projected_sum += projected_c * final_weight

    if total_weight <= 0:
        return None
    return projected_sum / total_weight


def _format_station_clock(value: Any, station_id: str) -> str:
    dt = _parse_iso_utc(value)
    if dt is None:
        return "n/a"
    station = STATIONS.get(station_id)
    tz_name = getattr(station, "timezone", None) if station else None
    try:
        tz = ZoneInfo(str(tz_name)) if tz_name else ZoneInfo("UTC")
    except Exception:
        tz = ZoneInfo("UTC")
    return dt.astimezone(tz).strftime("%H:%M")


def _build_pws_mobile_snapshot(
    station_id: str,
    details: Sequence[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]],
    official_temp_c: Optional[float],
    official_obs_utc: Optional[datetime],
) -> Dict[str, Any]:
    valid_rows = [row for row in details if isinstance(row, dict) and _is_valid_pws_row(row)]
    learning_metric = (metrics or {}).get("learning") if isinstance(metrics, dict) else {}
    top_weighted = _rank_pws_rows(valid_rows, "learning_weight")[:10]
    top_now = _rank_pws_rows(valid_rows, "learning_now_score")[:10]
    top_lead = _rank_pws_rows(valid_rows, "learning_lead_score")[:10]

    total_consensus_c = _compute_pws_weighted_consensus_c(valid_rows)
    weighted_consensus_c = _compute_pws_weighted_consensus_c(top_weighted)
    now_consensus_c = _compute_pws_weighted_consensus_c(top_now)
    lead_consensus_c = _compute_pws_weighted_consensus_c(top_lead)
    estimated_next_metar_c = _estimate_next_metar_c(
        rows=valid_rows,
        official_temp_c=official_temp_c,
        official_obs_utc=official_obs_utc,
        learning_metric=learning_metric if isinstance(learning_metric, dict) else {},
        consensus_c=total_consensus_c,
    )

    return {
        "total_consensus_c": total_consensus_c,
        "top10_weighted_consensus_c": weighted_consensus_c,
        "top10_now_consensus_c": now_consensus_c,
        "top10_lead_consensus_c": lead_consensus_c,
        "estimated_next_metar_c": estimated_next_metar_c,
        "top10_weighted_rows": top_weighted,
    }


@dataclass(frozen=True)
class TelegramBotSettings:
    token: str
    allowed_chat_ids: Optional[Set[str]]
    poll_timeout_seconds: int = 25
    max_message_chars: int = 3900

    @classmethod
    def from_env(cls) -> Optional["TelegramBotSettings"]:
        # Backward compatible alias: some deployments still use TELEGRAM_BOT_API_KEY.
        token = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_API_KEY") or "").strip()
        if not token:
            return None

        if not _env_bool("TELEGRAM_BOT_ENABLED", True):
            return None

        allowed_raw = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
        allowed = _parse_allowed_chat_ids(allowed_raw)
        return cls(
            token=token,
            allowed_chat_ids=allowed if allowed else None,
            poll_timeout_seconds=max(10, int(os.getenv("TELEGRAM_POLL_TIMEOUT_SECONDS", "25"))),
            max_message_chars=max(1000, int(os.getenv("TELEGRAM_MAX_MESSAGE_CHARS", "3900"))),
        )


class HeliosTelegramBot:
    """Telegram bot with station selection and HELIOS mobile summaries."""

    def __init__(self, settings: TelegramBotSettings):
        self.settings = settings
        self._api_base_url = f"https://api.telegram.org/bot{settings.token}"
        self._offset: Optional[int] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._chat_station: Dict[str, str] = {}
        self._known_chat_ids: Set[str] = set()
        self._unauthorized_warned: Set[str] = set()
        self._last_subscription_push_key_by_station: Dict[str, str] = {}

    def _bot_station_ids(self) -> List[str]:
        """Stations exposed in Telegram bot (all configured stations, not only market-active)."""
        return list(STATIONS.keys())

    def _subscription_recipients(self) -> List[str]:
        if self.settings.allowed_chat_ids:
            return sorted(str(chat_id) for chat_id in self.settings.allowed_chat_ids)
        return sorted(self._known_chat_ids)

    async def broadcast_system_message(self, text: str) -> int:
        """
        Broadcast a control/system message to subscription recipients.
        Returns count of successful chats.
        """
        if not text:
            return 0
        recipients = self._subscription_recipients()
        if not recipients or self._session is None:
            return 0
        results = await asyncio.gather(
            *(self._send_text(chat_id, text) for chat_id in recipients),
            return_exceptions=True,
        )
        sent_ok = 0
        for res in results:
            if isinstance(res, Exception):
                logger.warning("Telegram system broadcast failed: %s", res)
            else:
                sent_ok += 1
        return sent_ok

    def _format_subscription_metar_message(self, data: Dict[str, Any]) -> Optional[str]:
        station_id = str(data.get("station_id") or "").upper().strip()
        if not station_id:
            return None
        station = STATIONS.get(station_id)
        station_name = station.name if station else station_id
        station_tz = ZoneInfo(station.timezone) if station else ZoneInfo("UTC")

        report_type = str(data.get("report_type") or ("SPECI" if data.get("is_speci") else "METAR")).upper()
        source = str(data.get("source") or "UNKNOWN")
        obs_utc = _parse_iso_utc(data.get("obs_time_utc"))
        obs_local = obs_utc.astimezone(station_tz) if obs_utc else None

        temp_f_display = data.get("temp_f_raw", data.get("temp_f"))
        settlement_f = data.get("temp_f")
        wind_dir = data.get("wind_dir")
        wind_speed = data.get("wind_speed")
        if wind_dir is not None and wind_speed is not None:
            wind_text = f"{wind_dir}@{wind_speed}kt"
        else:
            wind_text = "n/a"

        qc_passed = data.get("qc_passed")
        if qc_passed is True:
            qc_text = "PASS"
        elif qc_passed is False:
            qc_text = "FAIL"
        else:
            qc_text = "n/a"
        qc_flags = data.get("qc_flags") or []
        if isinstance(qc_flags, list) and qc_flags:
            qc_text = f"{qc_text} ({', '.join(str(x) for x in qc_flags[:4])})"

        source_age = data.get("source_age_s")
        source_age_text = f"{int(float(source_age))}s" if _safe_float(source_age) is not None else "n/a"
        source_prefix = "NOAA" if source.upper().startswith("NOAA") else "Official"
        label = f"{source_prefix} {report_type}"
        settlement_text = _fmt_station_temp_from_f(settlement_f, station_id, 0)
        raw_metar = str(data.get("raw_metar") or "").strip()

        lines = [
            f"{label} [Subscription] - {station_id} ({station_name})",
            f"Obs UTC:   {obs_utc.isoformat() if obs_utc else (data.get('obs_time_utc') or 'n/a')}",
            f"Obs local: {obs_local.isoformat() if obs_local else 'n/a'}",
            f"Source:    {source} (age={source_age_text})",
            "",
            f"Temp:      {_fmt_station_temp_from_f(temp_f_display, station_id, 1)}",
            f"Settlement:{settlement_text}",
            f"Dewpoint:  {_fmt(data.get('dewpoint_c'), 1, 'C')}",
            f"Wind:      {wind_text}",
            f"Sky:       {data.get('sky_condition') or 'n/a'}",
            f"QC:        {qc_text}",
        ]
        if raw_metar:
            lines.extend(["", f"RAW: {raw_metar}"])
        return "\n".join(lines)

    async def _run_world_subscription_publisher(self) -> None:
        """Listen to WorldState updates and push subscribed official METAR/SPECI to Telegram."""
        world = get_world()
        queue = world.subscribe_sse()
        logger.info("Telegram METAR subscription listener started")
        try:
            while True:
                payload = await queue.get()
                try:
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("type") != "official":
                        continue
                    data = payload.get("data")
                    if not isinstance(data, dict):
                        continue

                    station_id = str(data.get("station_id") or "").upper().strip()
                    if not station_id:
                        continue
                    if not is_telegram_station_subscription_enabled(station_id):
                        continue

                    report_type = str(data.get("report_type") or ("SPECI" if data.get("is_speci") else "METAR")).upper()
                    dedupe_key = "|".join(
                        [
                            station_id,
                            str(data.get("obs_time_utc") or ""),
                            report_type,
                            str(data.get("temp_f_raw", data.get("temp_f")) or ""),
                            str(data.get("source") or ""),
                        ]
                    )
                    if self._last_subscription_push_key_by_station.get(station_id) == dedupe_key:
                        continue

                    recipients = self._subscription_recipients()
                    if not recipients:
                        continue

                    message = self._format_subscription_metar_message(data)
                    if not message:
                        continue

                    results = await asyncio.gather(
                        *(self._send_text(chat_id, message) for chat_id in recipients),
                        return_exceptions=True,
                    )
                    failures = [r for r in results if isinstance(r, Exception)]
                    if failures:
                        for err in failures[:3]:
                            logger.warning("Telegram subscription push failed: %s", err)
                    self._last_subscription_push_key_by_station[station_id] = dedupe_key
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Telegram subscription listener event error: %s", exc)
        except asyncio.CancelledError:
            logger.info("Telegram METAR subscription listener cancelled")
            raise
        finally:
            world.unsubscribe_sse(queue)

    async def run_forever(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.settings.poll_timeout_seconds + 15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self._session = session
            world_push_task = asyncio.create_task(self._run_world_subscription_publisher())
            try:
                me = await self._safe_get_me()
                if me:
                    logger.info("Telegram bot online as @%s", me.get("username", "unknown"))
                else:
                    logger.info("Telegram bot online")

                while True:
                    try:
                        updates = await self._fetch_updates()
                        for update in updates:
                            update_id = update.get("update_id")
                            if isinstance(update_id, int):
                                self._offset = update_id + 1
                            await self._handle_update(update)
                    except asyncio.CancelledError:
                        logger.info("Telegram bot loop cancelled")
                        raise
                    except Exception as exc:
                        logger.warning("Telegram polling error: %s", exc)
                        await asyncio.sleep(3)
            finally:
                world_push_task.cancel()
                try:
                    await world_push_task
                except asyncio.CancelledError:
                    pass
                self._session = None

    async def _safe_get_me(self) -> Optional[Dict[str, Any]]:
        try:
            result = await self._api_post("getMe", {})
            return result if isinstance(result, dict) else None
        except Exception as exc:
            logger.warning("Telegram getMe failed: %s", exc)
            return None

    async def _fetch_updates(self) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "timeout": self.settings.poll_timeout_seconds,
            "allowed_updates": ["message", "edited_message"],
        }
        if self._offset is not None:
            payload["offset"] = self._offset
        result = await self._api_post(
            "getUpdates",
            payload,
            timeout_seconds=self.settings.poll_timeout_seconds + 10,
        )
        if not isinstance(result, list):
            return []
        return [item for item in result if isinstance(item, dict)]

    async def _api_post(
        self,
        method: str,
        payload: Dict[str, Any],
        timeout_seconds: Optional[int] = None,
    ) -> Any:
        if self._session is None:
            raise RuntimeError("Telegram session not initialized")

        url = f"{self._api_base_url}/{method}"
        timeout = aiohttp.ClientTimeout(total=timeout_seconds) if timeout_seconds else None
        async with self._session.post(url, json=payload, timeout=timeout) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"Telegram HTTP {response.status}: {body[:240]}")
        data = json.loads(body)
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error in {method}: {data}")
        return data.get("result")

    async def _handle_update(self, update: Dict[str, Any]) -> None:
        message = update.get("message") or update.get("edited_message")
        if not isinstance(message, dict):
            return

        chat = message.get("chat") or {}
        chat_id = str(chat.get("id", "")).strip()
        if not chat_id:
            return
        if not self._is_chat_allowed(chat_id):
            if chat_id not in self._unauthorized_warned:
                self._unauthorized_warned.add(chat_id)
                await self._send_text(chat_id, "Chat no autorizado para este bot.")
            return
        self._known_chat_ids.add(chat_id)

        text = str(message.get("text") or "").strip()
        if not text:
            return
        await self._dispatch_text(chat_id, text)

    def _is_chat_allowed(self, chat_id: str) -> bool:
        allowed = self.settings.allowed_chat_ids
        return not allowed or chat_id in allowed

    async def _dispatch_text(self, chat_id: str, text: str) -> None:
        active_ids = self._bot_station_ids()
        upper_text = text.upper()
        if upper_text in active_ids:
            self._chat_station[chat_id] = upper_text
            await self._send_text(
                chat_id,
                f"Estacion activa: {upper_text} ({STATIONS[upper_text].name})",
                reply_markup=self._station_keyboard(active_ids),
            )
            return

        parts = text.split()
        command = parts[0].split("@", 1)[0].lower()
        args = parts[1:]

        if command in {"/start", "/help", "/ayuda"}:
            await self._cmd_start(chat_id, active_ids)
            return
        if command in {"/stations", "/estaciones"}:
            await self._cmd_stations(chat_id, active_ids)
            return
        if command in {"/station", "/estacion"}:
            await self._cmd_station(chat_id, args, active_ids)
            return
        if command == "/helios":
            await self._cmd_helios(chat_id, args)
            return
        if command == "/metar":
            await self._cmd_metar(chat_id, args)
            return
        if command == "/pws":
            await self._cmd_pws(chat_id, args)
            return
        if command in {"/pwsraw", "/pws_full", "/pwsfull"}:
            await self._cmd_pwsraw(chat_id, args)
            return
        if command in {"/resumen", "/status"}:
            await self._cmd_resumen(chat_id, args)
            return

        await self._send_text(
            chat_id,
            "Comando no reconocido. Usa /help para ver comandos.",
            reply_markup=self._station_keyboard(active_ids),
        )

    async def _cmd_start(self, chat_id: str, active_ids: Sequence[str]) -> None:
        station_id = self._current_station_for_chat(chat_id, active_ids)
        msg = (
            "HELIOS Telegram Bot\n\n"
            "Comandos:\n"
            "/helios [ICAO] - Prediccion HELIOS\n"
            "/pws [ICAO] - Consenso PWS simple\n"
            "/metar [ICAO] - Temperatura METAR actual\n"
            "/resumen [ICAO] - Resumen corto\n"
            "/stations - Estaciones\n"
            "/station ICAO - Cambiar estacion\n\n"
            "/pwsraw [ICAO] - Debug JSON\n\n"
            f"Estacion actual: {station_id}\n"
            "Tambien puedes tocar un boton de estacion."
        )
        await self._send_text(chat_id, msg, reply_markup=self._station_keyboard(active_ids))

    async def _cmd_stations(self, chat_id: str, active_ids: Sequence[str]) -> None:
        selected = self._current_station_for_chat(chat_id, active_ids)
        lines = [f"Estacion actual: {selected}", ""]
        for sid in active_ids:
            station = STATIONS[sid]
            marker = "-> " if sid == selected else ""
            lines.append(f"{marker}{sid} - {station.name}")
        await self._send_text(chat_id, "\n".join(lines), reply_markup=self._station_keyboard(active_ids))

    async def _cmd_station(self, chat_id: str, args: Sequence[str], active_ids: Sequence[str]) -> None:
        if not args:
            current = self._current_station_for_chat(chat_id, active_ids)
            await self._send_text(
                chat_id,
                f"Estacion actual: {current}\nUsa /station ICAO para cambiar.",
                reply_markup=self._station_keyboard(active_ids),
            )
            return

        wanted = str(args[0]).upper()
        if wanted not in active_ids:
            await self._send_text(
                chat_id,
                f"Estacion invalida: {wanted}. Usa /stations para ver opciones.",
                reply_markup=self._station_keyboard(active_ids),
            )
            return

        self._chat_station[chat_id] = wanted
        await self._send_text(
            chat_id,
            f"Estacion: {wanted} ({STATIONS[wanted].name})",
            reply_markup=self._station_keyboard(active_ids),
        )

    async def _cmd_helios(self, chat_id: str, args: Sequence[str]) -> None:
        station_id = self._resolve_station_from_args(chat_id, args)
        if not station_id:
            await self._send_text(chat_id, "No hay estaciones disponibles en el bot.")
            return

        station = STATIONS[station_id]
        now_local = datetime.now(ZoneInfo(station.timezone))
        requested_target_date = now_local.date().isoformat()

        pred = (
            get_latest_prediction_for_date(station_id, requested_target_date)
            or get_latest_prediction(station_id)
        )
        if not pred:
            await self._send_text(chat_id, f"No hay predicciones HELIOS para {station_id}.")
            return

        prediction_text = _fmt_station_temp_from_f(pred.get("final_prediction_f"), station_id, 1)
        await self._send_text(chat_id, f"HELIOS {station_id}: {prediction_text}")

    async def _cmd_metar(self, chat_id: str, args: Sequence[str]) -> None:
        station_id = self._resolve_station_from_args(chat_id, args)
        if not station_id:
            await self._send_text(chat_id, "No hay estaciones disponibles en el bot.")
            return

        station = STATIONS[station_id]
        metar, history = await asyncio.gather(
            fetch_metar_race(station_id),
            fetch_metar_history(station_id, hours=12),
        )
        if not metar:
            await self._send_text(chat_id, f"No se pudo obtener METAR NOAA para {station_id}.")
            return

        temp_text = _fmt_station_temp_from_c(metar.temp_c, station_id, 1)
        history_rows = sorted(
            [row for row in (history or []) if getattr(row, "observation_time", None) is not None],
            key=lambda row: row.observation_time,
            reverse=True,
        )

        lines = [f"METAR {station_id}: {temp_text}", "", "Ultimos 5:"]
        seen: Set[tuple[str, Optional[float]]] = set()
        for row in history_rows:
            obs_time = row.observation_time.astimezone(ZoneInfo(station.timezone))
            row_temp_c = _safe_float(getattr(row, "temp_c", None))
            dedupe_key = (row.observation_time.isoformat(), row_temp_c)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            row_temp_text = _fmt_station_temp_from_c(row_temp_c, station_id, 1)
            lines.append(f"{obs_time.strftime('%Y-%m-%d %H:%M')} | {row_temp_text}")
            if len(seen) >= 5:
                break

        await self._send_text(chat_id, "\n".join(lines))

    async def _cmd_pws(self, chat_id: str, args: Sequence[str]) -> None:
        station_id = self._resolve_station_from_args(chat_id, args)
        if not station_id:
            await self._send_text(chat_id, "No hay estaciones disponibles en el bot.")
            return

        metar, consensus, details, metrics = await self._refresh_pws_state(station_id)
        if not details and not consensus:
            await self._send_text(chat_id, f"No hay datos PWS disponibles para {station_id}.")
            return

        snapshot = _build_pws_mobile_snapshot(
            station_id=station_id,
            details=[row for row in details if isinstance(row, dict)],
            metrics=metrics if isinstance(metrics, dict) else {},
            official_temp_c=metar.temp_c if metar else None,
            official_obs_utc=metar.observation_time if metar else None,
        )

        lines = [
            f"Total Consensus: {_fmt_station_temp_from_c(snapshot.get('total_consensus_c'), station_id, 1)}",
            f"Consensus top 10 weighted: {_fmt_station_temp_from_c(snapshot.get('top10_weighted_consensus_c'), station_id, 1)}",
            f"Consensus top 10 now: {_fmt_station_temp_from_c(snapshot.get('top10_now_consensus_c'), station_id, 1)}",
            f"Consensus top 10 lead: {_fmt_station_temp_from_c(snapshot.get('top10_lead_consensus_c'), station_id, 1)}",
            f"Estimated next metar: {_fmt_station_temp_from_c(snapshot.get('estimated_next_metar_c'), station_id, 1)}",
            "",
            "Top 10 PWS weighted:",
        ]

        for idx, row in enumerate(snapshot.get("top10_weighted_rows") or [], start=1):
            station_code = str(row.get("station_id") or "n/a")
            last_update = _format_station_clock(row.get("obs_time_utc"), station_id)
            temp_text = _fmt_station_temp_from_c(row.get("temp_c"), station_id, 1)
            lines.append(f"{idx}. {station_code} | {last_update} | {temp_text}")

        await self._send_text(chat_id, "\n".join(lines))

    async def _cmd_pwsraw(self, chat_id: str, args: Sequence[str]) -> None:
        station_id = self._resolve_station_from_args(chat_id, args)
        if not station_id:
            await self._send_text(chat_id, "No hay estaciones disponibles en el bot.")
            return

        await self._send_text(chat_id, f"Actualizando PWS RAW para {station_id}...")
        _metar, consensus, details, metrics = await self._refresh_pws_state(station_id)
        if not details and not consensus:
            await self._send_text(chat_id, f"No hay datos PWS disponibles para {station_id}.")
            return

        consensus_payload = None
        if consensus:
            consensus_payload = {
                "station_id": consensus.station_id,
                "median_temp_c": consensus.median_temp_c,
                "median_temp_f": consensus.median_temp_f,
                "mad": consensus.mad,
                "support": consensus.support,
                "total_queried": consensus.total_queried,
                "obs_time_utc": consensus.obs_time_utc.isoformat() if consensus.obs_time_utc else None,
                "drift_c": consensus.drift_c,
                "data_source": consensus.data_source,
                "weighted_support": consensus.weighted_support,
                "station_details": consensus.station_details,
                "source_metrics": consensus.source_metrics,
                "learning_weights": consensus.learning_weights,
                "learning_profiles": consensus.learning_profiles,
            }

        full_payload = {
            "station_id": station_id,
            "timestamp_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
            "consensus": consensus_payload,
            "pws_metrics": metrics,
            "pws_details": details,
        }
        payload_text = json.dumps(full_payload, ensure_ascii=True, indent=2)
        await self._send_text(chat_id, f"PWS payload completo:\n{payload_text}")

    async def _refresh_pws_state(
        self,
        station_id: str,
    ):
        metar = await fetch_metar_race(station_id)
        official_temp_c = metar.temp_c if metar else None
        official_obs_time_utc = metar.observation_time if metar else None
        consensus = await fetch_and_publish_pws(
            station_id,
            official_temp_c=official_temp_c,
            official_obs_time_utc=official_obs_time_utc,
        )
        snapshot = get_world().get_snapshot()
        details = ((snapshot.get("pws_details") or {}).get(station_id)) or []
        metrics = ((snapshot.get("pws_metrics") or {}).get(station_id)) or {}
        return metar, consensus, details, metrics

    async def _cmd_resumen(self, chat_id: str, args: Sequence[str]) -> None:
        station_id = self._resolve_station_from_args(chat_id, args)
        if not station_id:
            await self._send_text(chat_id, "No hay estaciones disponibles en el bot.")
            return

        station = STATIONS[station_id]
        now_local = datetime.now(ZoneInfo(station.timezone))
        target_date = now_local.date().isoformat()
        pred = (
            get_latest_prediction_for_date(station_id, target_date)
            or get_latest_prediction(station_id)
        )
        metar = await fetch_metar_race(station_id)
        pws_snapshot = get_world().get_snapshot()
        pws_details = ((pws_snapshot.get("pws_details") or {}).get(station_id)) or []
        pws_metrics = ((pws_snapshot.get("pws_metrics") or {}).get(station_id)) or {}
        pws_mobile = _build_pws_mobile_snapshot(
            station_id=station_id,
            details=[row for row in pws_details if isinstance(row, dict)],
            metrics=pws_metrics if isinstance(pws_metrics, dict) else {},
            official_temp_c=metar.temp_c if metar else None,
            official_obs_utc=metar.observation_time if metar else None,
        )

        lines = [f"Resumen {station_id}"]
        lines.append(
            f"HELIOS: {_fmt_station_temp_from_f(pred.get('final_prediction_f'), station_id, 1) if pred else 'n/a'}"
        )
        lines.append(
            f"METAR: {_fmt_station_temp_from_c(metar.temp_c, station_id, 1) if metar else 'n/a'}"
        )
        lines.append(
            f"PWS total consensus: {_fmt_station_temp_from_c(pws_mobile.get('total_consensus_c'), station_id, 1)}"
        )
        lines.append(
            f"Estimated next metar: {_fmt_station_temp_from_c(pws_mobile.get('estimated_next_metar_c'), station_id, 1)}"
        )

        await self._send_text(chat_id, "\n".join(lines))

    def _resolve_station_from_args(self, chat_id: str, args: Sequence[str]) -> Optional[str]:
        active = self._bot_station_ids()
        if not active:
            return None

        if args:
            wanted = str(args[0]).upper()
            if wanted in active:
                self._chat_station[chat_id] = wanted
                return wanted
            return self._current_station_for_chat(chat_id, active)
        return self._current_station_for_chat(chat_id, active)

    def _current_station_for_chat(self, chat_id: str, active_ids: Sequence[str]) -> str:
        saved = self._chat_station.get(chat_id)
        if saved in active_ids:
            return saved
        default_station = active_ids[0]
        self._chat_station[chat_id] = default_station
        return default_station

    def _station_keyboard(self, active_ids: Sequence[str]) -> Dict[str, Any]:
        ids = list(active_ids)
        rows = [ids[i : i + 2] for i in range(0, len(ids), 2)]
        return {
            "keyboard": rows,
            "resize_keyboard": True,
            "one_time_keyboard": False,
        }

    async def _send_text(
        self,
        chat_id: str,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> None:
        chunks = self._chunk_text(text or "")
        for idx, chunk in enumerate(chunks):
            payload: Dict[str, Any] = {"chat_id": chat_id, "text": chunk}
            if idx == 0 and reply_markup:
                payload["reply_markup"] = reply_markup
            await self._api_post("sendMessage", payload)

    def _chunk_text(self, text: str) -> List[str]:
        limit = self.settings.max_message_chars
        if len(text) <= limit:
            return [text]

        chunks: List[str] = []
        current = ""
        for raw_line in text.splitlines(keepends=True):
            line = raw_line
            while len(line) > limit:
                piece = line[:limit]
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(piece)
                line = line[limit:]

            if len(current) + len(line) > limit:
                if current:
                    chunks.append(current)
                current = line
            else:
                current += line

        if current:
            chunks.append(current)
        return chunks


def build_telegram_bot_from_env() -> Optional[HeliosTelegramBot]:
    global _LIVE_TELEGRAM_BOT
    settings = TelegramBotSettings.from_env()
    if not settings:
        _LIVE_TELEGRAM_BOT = None
        return None
    _LIVE_TELEGRAM_BOT = HeliosTelegramBot(settings)
    return _LIVE_TELEGRAM_BOT


def get_live_telegram_bot() -> Optional[HeliosTelegramBot]:
    """Return the bot instance created at startup, if enabled."""
    return _LIVE_TELEGRAM_BOT
