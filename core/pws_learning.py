"""
Online learning for PWS station reliability by market.

Dual-score model:
- now_score: accuracy against METAR at aligned timestamps.
- lead_score: cumulative lead hits when PWS leads METAR (short horizon, 5-45m by default).

The consensus weight uses now_score as primary signal and lead reliability as a
bounded modifier, so a station that is "early" is not treated as accurate
"right now".
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from config import STATIONS
logger = logging.getLogger("pws_learning")

DEFAULT_PATH = Path("data/pws_learning/weights.json")
DEFAULT_AUDIT_DIR = Path("data/pws_learning/audit")
LEAD_BUCKETS = (
    ("05-15", 5.0, 15.0),
    ("15-30", 15.0, 30.0),
    ("30-45", 30.0, 45.0),
)
PROBABILITY_TOLERANCES_C = (0.5, 1.0)
MIN_SIGMA_C = 0.15


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _as_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        try:
            return datetime.fromisoformat(txt.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _normalize_source(source: str) -> str:
    src = str(source or "UNKNOWN").upper()
    if src.startswith("MADIS_"):
        if src == "MADIS_APRSWXNET":
            return "MADIS_APRSWXNET"
        return "MADIS_OTHER"
    return src


def _quality_band(score: float) -> str:
    s = float(score)
    if s >= 88.0:
        return "EXCELLENT"
    if s >= 76.0:
        return "GOOD"
    if s >= 62.0:
        return "FAIR"
    if s >= 48.0:
        return "WEAK"
    return "POOR"


def _round_float(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _lead_bucket_key(lead_minutes: Optional[float]) -> Optional[str]:
    try:
        lead_val = float(lead_minutes)
    except Exception:
        return None
    if lead_val < 0:
        return None
    for key, lo, hi in LEAD_BUCKETS:
        if lead_val < lo:
            continue
        if key == LEAD_BUCKETS[-1][0]:
            if lead_val <= hi:
                return key
            continue
        if lead_val < hi:
            return key
    return None


def _market_timezone(market_station_id: str) -> ZoneInfo:
    station = STATIONS.get(str(market_station_id).upper())
    tz_name = str(getattr(station, "timezone", "") or "")
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    return ZoneInfo("America/New_York")


class PWSLearningStore:
    """
    Persists station reliability and exposes dynamic station weights.

    Keys:
    - market_station_id: e.g. KLGA
    - pws_station_id: station from Synoptic/MADIS/WU/Open-Meteo
    """

    SOURCE_PRIOR: Dict[str, float] = {
        "SYNOPTIC": 1.15,
        "MADIS_APRSWXNET": 1.08,
        "MADIS_OTHER": 1.02,
        "WUNDERGROUND": 0.95,
        "OPEN_METEO": 0.75,
        "UNKNOWN": 1.00,
    }

    def __init__(
        self,
        path: Path = DEFAULT_PATH,
        audit_dir: Optional[Path] = None,
        alpha_now: float = 0.18,
        alpha_lead: float = 0.12,
        now_error_scale_c: float = 1.5,
        lead_error_scale_c: float = 2.0,
        lead_hit_tolerance_c: Optional[float] = None,
        min_weight: float = 0.25,
        max_weight: float = 3.5,
        confidence_samples_now: int = 24,
        confidence_samples_lead: int = 12,
        now_alignment_minutes: int = 22,
        lead_min_minutes: int = 5,
        lead_max_minutes: int = 45,
        lead_target_minutes: Optional[int] = None,
        ranking_min_now_samples: int = 2,
        ranking_min_lead_samples: int = 1,
        pending_max_age_hours: int = 8,
        max_pending_per_market: int = 6000,
    ):
        self.path = Path(path)
        self.audit_dir = Path(audit_dir) if audit_dir is not None else (self.path.parent / DEFAULT_AUDIT_DIR.name)

        self.alpha_now = float(alpha_now)
        self.alpha_lead = float(alpha_lead)

        self.now_error_scale_c = float(now_error_scale_c)
        self.lead_error_scale_c = float(lead_error_scale_c)
        if lead_hit_tolerance_c is None:
            lead_hit_tolerance_c = self.lead_error_scale_c
        self.lead_hit_tolerance_c = max(0.1, float(lead_hit_tolerance_c))

        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

        self.confidence_samples_now = max(1, int(confidence_samples_now))
        self.confidence_samples_lead = max(1, int(confidence_samples_lead))

        self.now_alignment_minutes = max(1, int(now_alignment_minutes))
        self.lead_min_minutes = max(1, int(lead_min_minutes))
        self.lead_max_minutes = max(self.lead_min_minutes + 1, int(lead_max_minutes))
        if lead_target_minutes is None:
            lead_target_minutes = int(round((self.lead_min_minutes + self.lead_max_minutes) / 2.0))
        self.lead_target_minutes = max(self.lead_min_minutes, min(int(lead_target_minutes), self.lead_max_minutes))
        self.ranking_min_now_samples = max(0, int(ranking_min_now_samples))
        self.ranking_min_lead_samples = max(0, int(ranking_min_lead_samples))
        self.pending_max_age_hours = max(1, int(pending_max_age_hours))
        self.max_pending_per_market = max(100, int(max_pending_per_market))

        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "schema_version": 2,
            "updated_at": None,
            "markets": {},
        }
        self._load()

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._state.update(payload)
        except Exception as e:
            logger.warning("Failed to load PWS learning file %s: %s", self.path, e)

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._state, ensure_ascii=True, indent=2)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")

        for attempt in range(5):
            try:
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(self.path)
                return
            except PermissionError:
                # Windows can transiently lock files during rapid replace.
                if attempt < 4:
                    time.sleep(0.01 * (attempt + 1))
                    continue
            except Exception:
                raise

        # Final fallback: direct overwrite.
        self.path.write_text(payload, encoding="utf-8")

    def _market_locked(self, market_station_id: str) -> Dict[str, Any]:
        markets = self._state.setdefault("markets", {})
        key = str(market_station_id).upper()
        bucket = markets.get(key)
        if not isinstance(bucket, dict):
            bucket = {}
            markets[key] = bucket
        if not isinstance(bucket.get("stations"), dict):
            # Backward compatibility: schema v1 stored station records directly in market bucket.
            legacy: Dict[str, Any] = {}
            reserved = {"updated_at", "pending", "stations", "last_official_obs_time_utc"}
            for k, v in list(bucket.items()):
                if k in reserved:
                    continue
                if isinstance(v, dict) and (
                    ("station_id" in v)
                    or ("weight" in v)
                    or ("samples" in v)
                    or ("ema_abs_error_c" in v)
                ):
                    legacy[str(k).upper()] = v
                    bucket.pop(k, None)
            bucket["stations"] = legacy
        if not isinstance(bucket.get("pending"), list):
            bucket["pending"] = []
        return bucket

    # ---------------------------------------------------------------------
    # Core metrics
    # ---------------------------------------------------------------------
    def _source_prior(self, source: str) -> float:
        norm = _normalize_source(source)
        return float(self.SOURCE_PRIOR.get(norm, self.SOURCE_PRIOR["UNKNOWN"]))

    def _calc_rel(self, ema_err_c: Optional[float], scale_c: float) -> float:
        if ema_err_c is None:
            return 0.55
        err = max(0.0, float(ema_err_c))
        return math.exp(-err / max(0.1, float(scale_c)))

    def _rank_policy(self) -> Dict[str, int]:
        return {
            "rank_min_now_samples": int(self.ranking_min_now_samples),
            "rank_min_lead_samples": int(self.ranking_min_lead_samples),
        }

    def _rank_eligibility(self, now_samples: int, lead_samples: int) -> Dict[str, Any]:
        now_n = max(0, int(now_samples))
        lead_n = max(0, int(lead_samples))
        need_now = max(0, int(self.ranking_min_now_samples) - now_n)
        need_lead = max(0, int(self.ranking_min_lead_samples) - lead_n)
        eligible = (need_now == 0) and (need_lead == 0)

        if eligible:
            reason = "READY"
            phase = "READY"
        else:
            reason_parts: List[str] = []
            if need_now > 0:
                reason_parts.append(f"need_now={need_now}")
            if need_lead > 0:
                reason_parts.append(f"need_lead={need_lead}")
            reason = " ".join(reason_parts) if reason_parts else "WARMUP"
            phase = "WARMUP"

        return {
            "rank_eligible": bool(eligible),
            "learning_phase": phase,
            "samples_total": int(now_n + lead_n),
            "rank_warmup_remaining_now": int(need_now),
            "rank_warmup_remaining_lead": int(need_lead),
            "rank_reason": reason,
            **self._rank_policy(),
        }

    def _recompute_station_locked(self, rec: Dict[str, Any], source_fallback: str = "UNKNOWN") -> None:
        source = str(rec.get("source") or source_fallback or "UNKNOWN")
        prior = self._source_prior(source)

        now_samples = int(rec.get("now_samples", 0) or 0)
        lead_samples = int(rec.get("lead_samples", 0) or 0)

        now_rel = self._calc_rel(rec.get("now_ema_abs_error_c"), self.now_error_scale_c)
        lead_ema_rel = self._calc_rel(rec.get("lead_ema_abs_error_c"), self.lead_error_scale_c)
        lead_hits_raw = rec.get("lead_hits")
        try:
            lead_hits = int(lead_hits_raw)
        except Exception:
            lead_hits = None
        if lead_samples <= 0:
            lead_hits = 0
        elif lead_hits is None:
            # Backfill legacy records approximately from historical EMA score.
            lead_hits = int(round(float(lead_ema_rel) * float(lead_samples)))
        lead_hits = max(0, min(int(lead_hits), int(lead_samples))) if lead_samples > 0 else 0
        lead_rel = (float(lead_hits) / float(lead_samples)) if lead_samples > 0 else float(lead_ema_rel)

        now_conf = min(1.0, now_samples / float(self.confidence_samples_now))
        lead_conf = min(1.0, lead_samples / float(self.confidence_samples_lead))

        # Weight for "current temperature consensus" (primary: now_rel).
        if now_samples == 0:
            rel_now = (0.72 * now_rel) + (0.28 * lead_rel * max(0.25, lead_conf))
        elif now_conf < 0.5:
            rel_now = (0.82 * now_rel) + (0.18 * lead_rel * lead_conf)
        else:
            rel_now = (0.90 * now_rel) + (0.10 * lead_rel * lead_conf)

        dynamic_now = 0.65 + (1.30 * rel_now)
        base_now = ((1.0 - now_conf) * 1.0) + (now_conf * dynamic_now)
        lead_modifier = 1.0 + (0.10 * (lead_rel - 0.5) * lead_conf)  # +/-5% max
        # Source prior is useful at cold start, but should fade as evidence accrues.
        evidence_now = min(1.0, max(0.0, max(now_conf, 0.35 * lead_conf)))
        prior_now = 1.0 + ((prior - 1.0) * (1.0 - evidence_now))
        weight_now = _clamp(prior_now * base_now * lead_modifier, self.min_weight, self.max_weight)

        # Predictive weight (for user-facing diagnostics).
        rel_pred = (0.55 * now_rel) + (0.45 * lead_rel)
        conf_pred = max(0.35 * now_conf, lead_conf)
        dynamic_pred = 0.70 + (1.30 * rel_pred)
        base_pred = ((1.0 - conf_pred) * 1.0) + (conf_pred * dynamic_pred)
        evidence_pred = min(1.0, max(0.0, max(now_conf, lead_conf)))
        prior_pred = 1.0 + ((prior - 1.0) * (1.0 - evidence_pred))
        weight_pred = _clamp(prior_pred * base_pred, self.min_weight, self.max_weight)

        rec["source"] = source
        rec["source_prior"] = round(float(prior), 4)
        rec["source_prior_now_effective"] = round(float(prior_now), 4)
        rec["source_prior_predictive_effective"] = round(float(prior_pred), 4)
        rec["weight"] = round(float(weight_now), 4)  # active consensus weight
        rec["weight_now"] = round(float(weight_now), 4)
        rec["weight_predictive"] = round(float(weight_pred), 4)

        rec["now_score"] = round(float(100.0 * now_rel), 2)
        rec["lead_hits"] = int(lead_hits)
        rec["lead_score"] = int(lead_hits)
        rec["lead_hit_rate"] = round(float(100.0 * lead_rel), 2) if lead_samples > 0 else None
        rec["predictive_score"] = round(float(100.0 * rel_pred), 2)
        rec["quality_band"] = _quality_band(rec["predictive_score"])

        now_stats = rec.get("now_stats") if isinstance(rec.get("now_stats"), dict) else {}
        lead_stats = rec.get("lead_stats") if isinstance(rec.get("lead_stats"), dict) else {}
        now_summary = self._error_stats_summary(
            count=int(now_stats.get("count", 0) or 0),
            err_sum_c=now_stats.get("err_sum_c"),
            abs_err_sum_c=now_stats.get("abs_err_sum_c"),
            err_sq_sum_c=now_stats.get("err_sq_sum_c"),
            hit_counts={
                "0.5": int(now_stats.get("hits_within_0.5_c", 0) or 0),
                "1.0": int(now_stats.get("hits_within_1.0_c", 0) or 0),
            },
            score_scale_c=self.now_error_scale_c,
        )
        lead_summary = self._error_stats_summary(
            count=int(lead_stats.get("count", 0) or 0),
            err_sum_c=lead_stats.get("err_sum_c"),
            abs_err_sum_c=lead_stats.get("abs_err_sum_c"),
            err_sq_sum_c=lead_stats.get("err_sq_sum_c"),
            lead_minutes_sum=lead_stats.get("lead_minutes_sum"),
            hit_counts={
                "0.5": int(lead_stats.get("hits_within_0.5_c", 0) or 0),
                "1.0": int(lead_stats.get("hits_within_1.0_c", 0) or 0),
            },
            score_scale_c=self.lead_error_scale_c,
        )
        rec["now_mae_c"] = now_summary.get("mae_c")
        rec["now_bias_c"] = now_summary.get("bias_c")
        rec["now_rmse_c"] = now_summary.get("rmse_c")
        rec["now_hit_rate_0_5_c"] = (
            _round_float(float(now_summary["prob_within_0_5_c"]) * 100.0, 2)
            if now_summary.get("prob_within_0_5_c") is not None
            else None
        )
        rec["now_hit_rate_1_0_c"] = (
            _round_float(float(now_summary["prob_within_1_0_c"]) * 100.0, 2)
            if now_summary.get("prob_within_1_0_c") is not None
            else None
        )

        rec["lead_mae_c"] = lead_summary.get("mae_c")
        rec["lead_bias_c"] = lead_summary.get("bias_c")
        rec["lead_rmse_c"] = lead_summary.get("rmse_c")
        rec["lead_sigma_c"] = lead_summary.get("sigma_c")
        rec["lead_avg_minutes"] = lead_summary.get("avg_lead_minutes")
        rec["lead_hit_rate_0_5_c"] = (
            _round_float(float(lead_summary["prob_within_0_5_c"]) * 100.0, 2)
            if lead_summary.get("prob_within_0_5_c") is not None
            else None
        )
        rec["lead_hit_rate_1_0_c"] = (
            _round_float(float(lead_summary["prob_within_1_0_c"]) * 100.0, 2)
            if lead_summary.get("prob_within_1_0_c") is not None
            else None
        )
        rec["lead_skill_score"] = lead_summary.get("score")
        rec["next_metar_score"] = lead_summary.get("score")
        rec["next_metar_bias_c"] = lead_summary.get("bias_c")
        rec["next_metar_mae_c"] = lead_summary.get("mae_c")
        rec["next_metar_rmse_c"] = lead_summary.get("rmse_c")
        rec["next_metar_sigma_c"] = lead_summary.get("sigma_c")
        rec["next_metar_prob_within_0_5_c"] = lead_summary.get("prob_within_0_5_c")
        rec["next_metar_prob_within_1_0_c"] = lead_summary.get("prob_within_1_0_c")

        bucket_map = rec.get("lead_bucket_stats") if isinstance(rec.get("lead_bucket_stats"), dict) else {}
        bucket_summaries: Dict[str, Dict[str, Any]] = {}
        for key, bucket in bucket_map.items():
            if not isinstance(bucket, dict):
                continue
            bucket_copy = dict(bucket)
            bucket_copy.setdefault("bucket", key)
            bucket_summaries[str(key)] = self._summarize_bucket_locked(bucket_copy)
        rec["lead_bucket_summaries"] = bucket_summaries
        rec.update(self._rank_eligibility(now_samples=now_samples, lead_samples=lead_samples))

    def _ensure_station_locked(self, stations: Dict[str, Any], sid: str, source: str) -> Dict[str, Any]:
        rec = stations.get(sid)
        if not isinstance(rec, dict):
            rec = {
                "station_id": sid,
                "source": source,
                "now_samples": 0,
                "lead_samples": 0,
                "lead_hits": 0,
            }
            stations[sid] = rec
        elif not rec.get("source"):
            rec["source"] = source
        return rec

    def _update_ema(self, prev: Optional[float], sample: float, alpha: float) -> float:
        if prev is None:
            return float(sample)
        return ((1.0 - alpha) * float(prev)) + (alpha * float(sample))

    def _error_stats_summary(
        self,
        *,
        count: int,
        err_sum_c: Optional[float],
        abs_err_sum_c: Optional[float],
        err_sq_sum_c: Optional[float],
        lead_minutes_sum: Optional[float] = None,
        hit_counts: Optional[Dict[str, int]] = None,
        score_scale_c: Optional[float] = None,
    ) -> Dict[str, Any]:
        n = max(0, int(count or 0))
        if n <= 0:
            return {
                "count": 0,
                "bias_c": None,
                "mae_c": None,
                "rmse_c": None,
                "sigma_c": None,
                "avg_lead_minutes": None,
                "prob_within_0_5_c": None,
                "prob_within_1_0_c": None,
                "score": None,
            }

        err_sum = float(err_sum_c or 0.0)
        abs_sum = float(abs_err_sum_c or 0.0)
        sq_sum = float(err_sq_sum_c or 0.0)

        bias_c = err_sum / float(n)
        mae_c = abs_sum / float(n)
        rmse_c = math.sqrt(max(0.0, sq_sum / float(n)))
        sigma_sq = max(0.0, (sq_sum / float(n)) - (bias_c ** 2))
        sigma_c = max(MIN_SIGMA_C, math.sqrt(sigma_sq))

        out: Dict[str, Any] = {
            "count": n,
            "bias_c": _round_float(bias_c, 4),
            "mae_c": _round_float(mae_c, 4),
            "rmse_c": _round_float(rmse_c, 4),
            "sigma_c": _round_float(sigma_c, 4),
            "avg_lead_minutes": (
                _round_float(float(lead_minutes_sum or 0.0) / float(n), 3)
                if lead_minutes_sum is not None
                else None
            ),
            "prob_within_0_5_c": None,
            "prob_within_1_0_c": None,
            "score": None,
        }

        hit_map = hit_counts or {}
        out["prob_within_0_5_c"] = (
            _round_float(float(int(hit_map.get("0.5", 0) or 0)) / float(n), 4)
            if "0.5" in hit_map or hit_map
            else None
        )
        out["prob_within_1_0_c"] = (
            _round_float(float(int(hit_map.get("1.0", 0) or 0)) / float(n), 4)
            if "1.0" in hit_map or hit_map
            else None
        )

        scale = float(score_scale_c or self.lead_error_scale_c or 2.0)
        out["score"] = _round_float(100.0 * math.exp(-mae_c / max(0.1, scale)), 2)
        return out

    def _ensure_counter_locked(self, rec: Dict[str, Any], key: str) -> Dict[str, Any]:
        bucket = rec.get(key)
        if not isinstance(bucket, dict):
            bucket = {}
            rec[key] = bucket
        return bucket

    def _record_hit_counter_locked(self, bucket: Dict[str, Any], abs_err_c: float) -> None:
        for tol in PROBABILITY_TOLERANCES_C:
            tol_key = f"hits_within_{tol:.1f}_c"
            bucket[tol_key] = int(bucket.get(tol_key, 0) or 0) + (1 if abs_err_c <= float(tol) else 0)

    def _update_now_stats_locked(self, rec: Dict[str, Any], signed_err_c: float) -> None:
        abs_err_c = abs(float(signed_err_c))
        bucket = self._ensure_counter_locked(rec, "now_stats")
        bucket["count"] = int(bucket.get("count", 0) or 0) + 1
        bucket["err_sum_c"] = float(bucket.get("err_sum_c", 0.0) or 0.0) + float(signed_err_c)
        bucket["abs_err_sum_c"] = float(bucket.get("abs_err_sum_c", 0.0) or 0.0) + abs_err_c
        bucket["err_sq_sum_c"] = float(bucket.get("err_sq_sum_c", 0.0) or 0.0) + (float(signed_err_c) ** 2)
        self._record_hit_counter_locked(bucket, abs_err_c)

    def _update_lead_stats_locked(self, rec: Dict[str, Any], signed_err_c: float, lead_minutes: float) -> None:
        abs_err_c = abs(float(signed_err_c))
        overall = self._ensure_counter_locked(rec, "lead_stats")
        overall["count"] = int(overall.get("count", 0) or 0) + 1
        overall["err_sum_c"] = float(overall.get("err_sum_c", 0.0) or 0.0) + float(signed_err_c)
        overall["abs_err_sum_c"] = float(overall.get("abs_err_sum_c", 0.0) or 0.0) + abs_err_c
        overall["err_sq_sum_c"] = float(overall.get("err_sq_sum_c", 0.0) or 0.0) + (float(signed_err_c) ** 2)
        overall["lead_minutes_sum"] = float(overall.get("lead_minutes_sum", 0.0) or 0.0) + float(lead_minutes)
        self._record_hit_counter_locked(overall, abs_err_c)

        bucket_key = _lead_bucket_key(lead_minutes)
        if not bucket_key:
            return

        bucket_map = self._ensure_counter_locked(rec, "lead_bucket_stats")
        bucket = bucket_map.get(bucket_key)
        if not isinstance(bucket, dict):
            bucket = {"bucket": bucket_key}
            bucket_map[bucket_key] = bucket
        bucket["count"] = int(bucket.get("count", 0) or 0) + 1
        bucket["err_sum_c"] = float(bucket.get("err_sum_c", 0.0) or 0.0) + float(signed_err_c)
        bucket["abs_err_sum_c"] = float(bucket.get("abs_err_sum_c", 0.0) or 0.0) + abs_err_c
        bucket["err_sq_sum_c"] = float(bucket.get("err_sq_sum_c", 0.0) or 0.0) + (float(signed_err_c) ** 2)
        bucket["lead_minutes_sum"] = float(bucket.get("lead_minutes_sum", 0.0) or 0.0) + float(lead_minutes)
        self._record_hit_counter_locked(bucket, abs_err_c)

    def _summarize_bucket_locked(self, bucket: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "bucket": str(bucket.get("bucket") or ""),
            **self._error_stats_summary(
                count=int(bucket.get("count", 0) or 0),
                err_sum_c=_round_float(bucket.get("err_sum_c"), 8),
                abs_err_sum_c=_round_float(bucket.get("abs_err_sum_c"), 8),
                err_sq_sum_c=_round_float(bucket.get("err_sq_sum_c"), 8),
                lead_minutes_sum=_round_float(bucket.get("lead_minutes_sum"), 8),
                hit_counts={
                    "0.5": int(bucket.get("hits_within_0.5_c", 0) or 0),
                    "1.0": int(bucket.get("hits_within_1.0_c", 0) or 0),
                },
                score_scale_c=self.lead_error_scale_c,
            ),
        }

    def _audit_file_path(self, market_station_id: str, official_ts: datetime) -> Path:
        station_tz = _market_timezone(market_station_id)
        market_date = official_ts.astimezone(station_tz).date().isoformat()
        market_key = str(market_station_id).upper()
        return self.audit_dir / f"date={market_date}" / f"station={market_key}" / "events.ndjson"

    def _append_audit_rows_locked(
        self,
        market_station_id: str,
        official_ts: datetime,
        rows: List[Dict[str, Any]],
    ) -> None:
        if not rows:
            return
        path = self._audit_file_path(market_station_id, official_ts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    # ---------------------------------------------------------------------
    # Pending queue for lead labels
    # ---------------------------------------------------------------------
    def _reading_to_sample(self, reading: Any, fallback_ts: datetime) -> Optional[Dict[str, Any]]:
        sid = str(getattr(reading, "label", "") or "").upper()
        if not sid:
            return None

        temp_c = getattr(reading, "temp_c", None)
        if temp_c is None:
            return None
        try:
            temp_val = float(temp_c)
        except Exception:
            return None

        source = str(getattr(reading, "source", "UNKNOWN"))
        obs = _as_utc(getattr(reading, "obs_time_utc", None)) or fallback_ts
        valid = bool(getattr(reading, "valid", True))
        obs_iso = obs.isoformat()

        return {
            "station_id": sid,
            "source": source,
            "station_name": str(getattr(reading, "station_name", "") or sid),
            "distance_km": _round_float(getattr(reading, "distance_km", None), 4),
            "temp_c": round(temp_val, 4),
            "valid": valid,
            "obs_time_utc": obs_iso,
            "key": f"{sid}|{obs_iso}",
        }

    def _cleanup_pending_locked(self, market: Dict[str, Any], ref_time: datetime) -> None:
        pending = market.get("pending", [])
        if not isinstance(pending, list):
            market["pending"] = []
            return

        cutoff = ref_time - timedelta(hours=self.pending_max_age_hours)
        kept: List[Dict[str, Any]] = []
        for row in pending:
            if not isinstance(row, dict):
                continue
            ts = _as_utc(row.get("obs_time_utc"))
            if ts is None:
                continue
            if ts < cutoff:
                continue
            kept.append(row)

        # Keep most recent if still too large.
        if len(kept) > self.max_pending_per_market:
            kept.sort(key=lambda r: str(r.get("obs_time_utc") or ""))
            kept = kept[-self.max_pending_per_market :]

        market["pending"] = kept

    def ingest_pending(
        self,
        market_station_id: str,
        readings: Iterable[Any],
        obs_time_utc: Optional[datetime] = None,
    ) -> int:
        fallback_ts = _as_utc(obs_time_utc) or _utc_now()
        market_key = str(market_station_id).upper()
        added = 0

        with self._lock:
            market = self._market_locked(market_key)
            pending = market["pending"]
            keys = {
                str(row.get("key"))
                for row in pending
                if isinstance(row, dict) and row.get("key")
            }

            for r in readings:
                sample = self._reading_to_sample(r, fallback_ts)
                if not sample:
                    continue
                if sample["key"] in keys:
                    continue
                pending.append(sample)
                keys.add(sample["key"])
                added += 1

            self._cleanup_pending_locked(market, fallback_ts)
            if added > 0:
                market["updated_at"] = _utc_now_iso()
                self._state["updated_at"] = _utc_now_iso()
                self._save_locked()

        return added

    # ---------------------------------------------------------------------
    # Public read APIs
    # ---------------------------------------------------------------------
    def get_weight(
        self,
        market_station_id: str,
        pws_station_id: str,
        source: str = "UNKNOWN",
    ) -> float:
        profile = self.get_station_profile(market_station_id, pws_station_id, source)
        return float(profile.get("weight", self._source_prior(source)))

    def get_weights_for_readings(
        self,
        market_station_id: str,
        readings: Iterable[Any],
    ) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for r in readings:
            sid = str(getattr(r, "label", "") or "").upper()
            if not sid:
                continue
            src = str(getattr(r, "source", "UNKNOWN"))
            weights[sid] = self.get_weight(market_station_id, sid, src)
        return weights

    def get_station_profile(
        self,
        market_station_id: str,
        pws_station_id: str,
        source: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        market_key = str(market_station_id).upper()
        station_key = str(pws_station_id).upper()
        src = str(source or "UNKNOWN")
        prior = self._source_prior(src)

        with self._lock:
            market = self._state.get("markets", {}).get(market_key, {})
            stations = market.get("stations", {}) if isinstance(market, dict) else {}
            rec = stations.get(station_key) if isinstance(stations, dict) else None
            if not isinstance(rec, dict):
                rank_meta = self._rank_eligibility(now_samples=0, lead_samples=0)
                return {
                    "station_id": station_key,
                    "source": src,
                    "weight": round(float(prior), 4),
                    "weight_now": round(float(prior), 4),
                    "weight_predictive": round(float(prior), 4),
                    "now_score": None,
                    "lead_score": None,
                    "predictive_score": None,
                    "now_samples": 0,
                    "lead_samples": 0,
                    "lead_hits": 0,
                    "quality_band": "INIT",
                    "lead_skill_score": None,
                    "next_metar_score": None,
                    "next_metar_bias_c": None,
                    "next_metar_mae_c": None,
                    "next_metar_rmse_c": None,
                    "next_metar_sigma_c": None,
                    "next_metar_prob_within_0_5_c": None,
                    "next_metar_prob_within_1_0_c": None,
                    "lead_bucket_summaries": {},
                    **rank_meta,
                }
            if (
                rec.get("weight") is None
                or rec.get("now_score") is None
                or rec.get("lead_score") is None
                or rec.get("rank_eligible") is None
                or rec.get("next_metar_score") is None
            ):
                self._recompute_station_locked(rec, rec.get("source") or src)
            return dict(rec)

    def get_profiles_for_readings(
        self,
        market_station_id: str,
        readings: Iterable[Any],
    ) -> Dict[str, Dict[str, Any]]:
        profiles: Dict[str, Dict[str, Any]] = {}
        for r in readings:
            sid = str(getattr(r, "label", "") or "").upper()
            if not sid:
                continue
            src = str(getattr(r, "source", "UNKNOWN"))
            profiles[sid] = self.get_station_profile(market_station_id, sid, src)
        return profiles

    def get_top_profiles(
        self,
        market_station_id: str,
        limit: int = 10,
        sort_by: str = "weight",
        eligible_only: bool = False,
    ) -> List[Dict[str, Any]]:
        market_key = str(market_station_id).upper()
        key = str(sort_by or "weight")
        with self._lock:
            market = self._state.get("markets", {}).get(market_key, {})
            stations = market.get("stations", {}) if isinstance(market, dict) else {}
            rows: List[Dict[str, Any]] = []
            for rec in stations.values():
                if not isinstance(rec, dict):
                    continue
                if (
                    rec.get("weight") is None
                    or rec.get("now_score") is None
                    or rec.get("lead_score") is None
                    or rec.get("rank_eligible") is None
                    or rec.get("next_metar_score") is None
                ):
                    self._recompute_station_locked(rec, rec.get("source") or "UNKNOWN")
                rows.append(dict(rec))

        if eligible_only:
            rows = [r for r in rows if bool(r.get("rank_eligible"))]

        def _sort_value(row: Dict[str, Any]) -> float:
            try:
                return float(row.get(key) or 0.0)
            except Exception:
                return 0.0

        rows.sort(key=_sort_value, reverse=True)
        return rows[: max(1, int(limit))]

    def get_top_weights(
        self,
        market_station_id: str,
        limit: int = 10,
        eligible_only: bool = False,
    ) -> Dict[str, float]:
        top = self.get_top_profiles(
            market_station_id,
            limit=limit,
            sort_by="weight",
            eligible_only=eligible_only,
        )
        return {
            str(p.get("station_id")): round(float(p.get("weight", 0.0)), 4)
            for p in top
            if p.get("station_id")
        }

    def get_market_summary(self, market_station_id: str) -> Dict[str, Any]:
        market_key = str(market_station_id).upper()
        with self._lock:
            market = self._state.get("markets", {}).get(market_key, {})
            stations = market.get("stations", {}) if isinstance(market, dict) else {}
            rows: List[Dict[str, Any]] = []
            for rec in stations.values():
                if not isinstance(rec, dict):
                    continue
                if (
                    rec.get("weight") is None
                    or rec.get("now_score") is None
                    or rec.get("lead_score") is None
                    or rec.get("rank_eligible") is None
                    or rec.get("next_metar_score") is None
                ):
                    self._recompute_station_locked(rec, rec.get("source") or "UNKNOWN")
                rows.append(dict(rec))

        rows.sort(key=lambda r: float(r.get("weight") or 0.0), reverse=True)
        ready_rows = [r for r in rows if bool(r.get("rank_eligible"))]
        top_any = rows[0] if rows else None
        top_ready = ready_rows[0] if ready_rows else None

        return {
            "tracked_station_count": len(rows),
            "rank_ready_station_count": len(ready_rows),
            "warmup_station_count": max(0, len(rows) - len(ready_rows)),
            "rank_ready": bool(ready_rows),
            "top_any_station_id": top_any.get("station_id") if isinstance(top_any, dict) else None,
            "top_ready_station_id": top_ready.get("station_id") if isinstance(top_ready, dict) else None,
            **self._rank_policy(),
        }

    def get_predictive_summary(
        self,
        market_station_id: str,
        limit: int = 20,
        eligible_only: bool = False,
    ) -> Dict[str, Any]:
        rows = self.get_top_profiles(
            market_station_id=market_station_id,
            limit=max(1, int(limit)),
            sort_by="next_metar_score",
            eligible_only=eligible_only,
        )
        return {
            "market_station_id": str(market_station_id).upper(),
            "lead_window_minutes": [int(self.lead_min_minutes), int(self.lead_max_minutes)],
            "lead_buckets": [bucket[0] for bucket in LEAD_BUCKETS],
            "rows": rows,
        }

    def list_audit_dates(self, market_station_id: str) -> List[str]:
        market_key = str(market_station_id).upper()
        if not self.audit_dir.exists():
            return []
        dates: List[str] = []
        for path in self.audit_dir.glob("date=*/station=*/events.ndjson"):
            if path.parent.name != f"station={market_key}":
                continue
            if not path.is_file():
                continue
            date_name = path.parent.parent.name
            if not date_name.startswith("date="):
                continue
            dates.append(date_name.replace("date=", "", 1))
        return sorted(set(dates), reverse=True)

    def load_audit_rows(self, market_station_id: str, date_str: str) -> List[Dict[str, Any]]:
        market_key = str(market_station_id).upper()
        path = self.audit_dir / f"date={date_str}" / f"station={market_key}" / "events.ndjson"
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
        except Exception as e:
            logger.warning("Failed to load PWS audit rows from %s: %s", path, e)
        rows.sort(key=lambda row: str(row.get("official_obs_time_utc") or row.get("sample_obs_time_utc") or ""))
        return rows

    # ---------------------------------------------------------------------
    # Update logic with official labels
    # ---------------------------------------------------------------------
    def update_with_official(
        self,
        market_station_id: str,
        official_temp_c: Optional[float],
        readings: Iterable[Any],
        obs_time_utc: Optional[datetime] = None,
        official_obs_time_utc: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Update station scores with official label.

        now_score update:
        - compares only readings close to official obs timestamp.

        lead_score update:
        - counts cumulative lead hits from pending readings whose timestamp is
          5-45 minutes before official obs timestamp.
        """
        readings_list = list(readings)
        fallback_ts = _as_utc(obs_time_utc) or _utc_now()
        official_ts = _as_utc(official_obs_time_utc) or fallback_ts

        # Always ingest current snapshot into pending queue first.
        # This allows lead updates on future METAR ticks.
        self.ingest_pending(market_station_id, readings_list, obs_time_utc=fallback_ts)

        if official_temp_c is None:
            return {}
        try:
            official = float(official_temp_c)
        except Exception:
            return {}

        market_key = str(market_station_id).upper()
        updated: Dict[str, float] = {}
        touched_ids: set[str] = set()
        audit_rows: List[Dict[str, Any]] = []

        with self._lock:
            market = self._market_locked(market_key)
            stations = market["stations"]

            # NOW labels: aligned only.
            official_iso = official_ts.isoformat()
            for r in readings_list:
                sample = self._reading_to_sample(r, fallback_ts)
                if not sample:
                    continue
                if not bool(sample.get("valid", True)):
                    # Invalid/outlier samples should not train reliability scores.
                    continue

                sample_ts = _as_utc(sample["obs_time_utc"])
                if sample_ts is None:
                    continue
                delta_min = abs((sample_ts - official_ts).total_seconds()) / 60.0
                if delta_min > float(self.now_alignment_minutes):
                    continue

                sid = sample["station_id"]
                rec = self._ensure_station_locked(stations, sid, sample["source"])

                # One NOW label per station per official timestamp.
                if rec.get("last_now_official_obs_time_utc") == official_iso:
                    continue

                signed_err_c = float(sample["temp_c"]) - official
                err = abs(signed_err_c)
                if not bool(sample.get("valid", True)):
                    err += 0.15

                rec["now_samples"] = int(rec.get("now_samples", 0) or 0) + 1
                rec["now_ema_abs_error_c"] = round(
                    self._update_ema(rec.get("now_ema_abs_error_c"), err, self.alpha_now), 4
                )
                self._update_now_stats_locked(rec, signed_err_c)
                rec["last_now_abs_error_c"] = round(float(err), 4)
                rec["last_now_alignment_min"] = round(float(delta_min), 2)
                rec["last_now_official_obs_time_utc"] = official_iso
                rec["last_now_sample_obs_time_utc"] = sample["obs_time_utc"]
                rec["updated_at"] = _utc_now_iso()
                touched_ids.add(sid)
                audit_rows.append({
                    "market_station_id": market_key,
                    "market_date": official_ts.astimezone(_market_timezone(market_key)).date().isoformat(),
                    "official_obs_time_utc": official_iso,
                    "official_temp_c": round(float(official), 4),
                    "official_temp_f": round((float(official) * 9.0 / 5.0) + 32.0, 2),
                    "station_id": sid,
                    "station_name": sample.get("station_name"),
                    "source": sample.get("source"),
                    "sample_obs_time_utc": sample.get("obs_time_utc"),
                    "sample_temp_c": round(float(sample["temp_c"]), 4),
                    "sample_temp_f": round((float(sample["temp_c"]) * 9.0 / 5.0) + 32.0, 2),
                    "distance_km": sample.get("distance_km"),
                    "kind": "NOW",
                    "lead_minutes": round(float(delta_min), 3),
                    "lead_bucket": "NOW",
                    "signed_error_c": round(float(signed_err_c), 4),
                    "abs_error_c": round(float(abs(float(sample["temp_c"]) - official)), 4),
                    "hit_within_0_5_c": bool(abs(float(sample["temp_c"]) - official) <= 0.5),
                    "hit_within_1_0_c": bool(abs(float(sample["temp_c"]) - official) <= 1.0),
                    "valid": True,
                })

            # LEAD labels: pending queue resolution (5-30 min before official).
            # Use at most ONE lead sample per station per official METAR timestamp:
            # choose the candidate closest to lead_target_minutes.
            pending = market.get("pending", [])
            keep_pending: List[Dict[str, Any]] = []
            lead_candidates_by_station: Dict[str, Dict[str, Any]] = {}
            lead_candidate_count_by_station: Dict[str, int] = {}
            for row in pending:
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("station_id") or "").upper()
                if not sid:
                    continue

                row_ts = _as_utc(row.get("obs_time_utc"))
                if row_ts is None:
                    continue

                lead_min = (official_ts - row_ts).total_seconds() / 60.0
                if lead_min < float(self.lead_min_minutes):
                    keep_pending.append(row)
                    continue
                if lead_min > float(self.lead_max_minutes):
                    # Too old to score as lead and no longer useful.
                    continue

                lead_candidate_count_by_station[sid] = int(lead_candidate_count_by_station.get(sid, 0) or 0) + 1
                candidate = dict(row)
                candidate["_lead_min"] = float(lead_min)
                prev = lead_candidates_by_station.get(sid)
                if not isinstance(prev, dict):
                    lead_candidates_by_station[sid] = candidate
                    continue

                prev_lead = float(prev.get("_lead_min", 0.0) or 0.0)
                prev_gap = abs(prev_lead - float(self.lead_target_minutes))
                cur_gap = abs(float(lead_min) - float(self.lead_target_minutes))
                if cur_gap < prev_gap:
                    lead_candidates_by_station[sid] = candidate
                    continue
                if cur_gap > prev_gap:
                    continue

                prev_ts = _as_utc(prev.get("obs_time_utc"))
                if prev_ts is None or row_ts > prev_ts:
                    lead_candidates_by_station[sid] = candidate

            for sid, row in lead_candidates_by_station.items():
                temp_val = row.get("temp_c")
                try:
                    temp_c = float(temp_val)
                except Exception:
                    continue
                if row.get("valid") is False:
                    continue

                rec = self._ensure_station_locked(stations, sid, str(row.get("source") or "UNKNOWN"))
                # One LEAD label per station per official timestamp.
                if rec.get("last_lead_official_obs_time_utc") == official_iso:
                    continue

                signed_err_c = temp_c - official
                err = abs(signed_err_c)

                lead_min = float(row.get("_lead_min", 0.0) or 0.0)
                rec["lead_samples"] = int(rec.get("lead_samples", 0) or 0) + 1
                rec["lead_ema_abs_error_c"] = round(
                    self._update_ema(rec.get("lead_ema_abs_error_c"), err, self.alpha_lead), 4
                )
                self._update_lead_stats_locked(rec, signed_err_c, lead_min)
                lead_hit = bool(err <= float(self.lead_hit_tolerance_c))
                rec["lead_hits"] = int(rec.get("lead_hits", 0) or 0) + (1 if lead_hit else 0)
                rec["last_lead_abs_error_c"] = round(float(err), 4)
                rec["last_lead_hit"] = lead_hit
                rec["last_lead_minutes"] = round(float(lead_min), 2)
                rec["last_lead_official_obs_time_utc"] = official_iso
                rec["last_lead_sample_obs_time_utc"] = row.get("obs_time_utc")
                rec["last_lead_candidates_in_window"] = int(lead_candidate_count_by_station.get(sid, 1) or 1)
                rec["updated_at"] = _utc_now_iso()
                touched_ids.add(sid)
                lead_bucket = _lead_bucket_key(lead_min)
                audit_rows.append({
                    "market_station_id": market_key,
                    "market_date": official_ts.astimezone(_market_timezone(market_key)).date().isoformat(),
                    "official_obs_time_utc": official_iso,
                    "official_temp_c": round(float(official), 4),
                    "official_temp_f": round((float(official) * 9.0 / 5.0) + 32.0, 2),
                    "station_id": sid,
                    "station_name": row.get("station_name"),
                    "source": row.get("source"),
                    "sample_obs_time_utc": row.get("obs_time_utc"),
                    "sample_temp_c": round(float(temp_c), 4),
                    "sample_temp_f": round((float(temp_c) * 9.0 / 5.0) + 32.0, 2),
                    "distance_km": row.get("distance_km"),
                    "kind": "LEAD",
                    "lead_minutes": round(float(lead_min), 3),
                    "lead_bucket": lead_bucket,
                    "signed_error_c": round(float(signed_err_c), 4),
                    "abs_error_c": round(float(err), 4),
                    "hit_within_0_5_c": bool(err <= 0.5),
                    "hit_within_1_0_c": bool(err <= 1.0),
                    "valid": True,
                })

            market["pending"] = keep_pending
            self._cleanup_pending_locked(market, official_ts)

            for sid, rec in stations.items():
                if not isinstance(rec, dict):
                    continue
                self._recompute_station_locked(rec, rec.get("source", "UNKNOWN"))
                if sid in touched_ids:
                    updated[sid] = float(rec.get("weight", self._source_prior(rec.get("source", "UNKNOWN"))))

            market["last_official_obs_time_utc"] = official_iso
            market["updated_at"] = _utc_now_iso()
            self._state["updated_at"] = _utc_now_iso()
            self._append_audit_rows_locked(market_key, official_ts, audit_rows)
            self._save_locked()

        return updated


_store: Optional[PWSLearningStore] = None
_store_lock = threading.Lock()


def get_pws_learning_store(path: Path = DEFAULT_PATH) -> PWSLearningStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = PWSLearningStore(path=path)
        return _store
