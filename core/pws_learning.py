"""
Online learning for PWS station reliability by market.

Dual-score model:
- now_score: accuracy against METAR at aligned timestamps.
- lead_score: predictive signal quality when PWS leads METAR by ~1 hour.

The consensus weight uses now_score as primary signal and lead_score as a
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

logger = logging.getLogger("pws_learning")

DEFAULT_PATH = Path("data/pws_learning/weights.json")


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
        alpha_now: float = 0.18,
        alpha_lead: float = 0.12,
        now_error_scale_c: float = 1.5,
        lead_error_scale_c: float = 2.0,
        min_weight: float = 0.25,
        max_weight: float = 3.5,
        confidence_samples_now: int = 24,
        confidence_samples_lead: int = 12,
        now_alignment_minutes: int = 22,
        lead_min_minutes: int = 35,
        lead_max_minutes: int = 85,
        pending_max_age_hours: int = 8,
        max_pending_per_market: int = 6000,
    ):
        self.path = Path(path)

        self.alpha_now = float(alpha_now)
        self.alpha_lead = float(alpha_lead)

        self.now_error_scale_c = float(now_error_scale_c)
        self.lead_error_scale_c = float(lead_error_scale_c)

        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

        self.confidence_samples_now = max(1, int(confidence_samples_now))
        self.confidence_samples_lead = max(1, int(confidence_samples_lead))

        self.now_alignment_minutes = max(1, int(now_alignment_minutes))
        self.lead_min_minutes = max(1, int(lead_min_minutes))
        self.lead_max_minutes = max(self.lead_min_minutes + 1, int(lead_max_minutes))
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

    def _recompute_station_locked(self, rec: Dict[str, Any], source_fallback: str = "UNKNOWN") -> None:
        source = str(rec.get("source") or source_fallback or "UNKNOWN")
        prior = self._source_prior(source)

        now_samples = int(rec.get("now_samples", 0) or 0)
        lead_samples = int(rec.get("lead_samples", 0) or 0)

        now_rel = self._calc_rel(rec.get("now_ema_abs_error_c"), self.now_error_scale_c)
        lead_rel = self._calc_rel(rec.get("lead_ema_abs_error_c"), self.lead_error_scale_c)

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
        weight_now = _clamp(prior * base_now * lead_modifier, self.min_weight, self.max_weight)

        # Predictive weight (for user-facing diagnostics).
        rel_pred = (0.55 * now_rel) + (0.45 * lead_rel)
        conf_pred = max(0.35 * now_conf, lead_conf)
        dynamic_pred = 0.70 + (1.30 * rel_pred)
        base_pred = ((1.0 - conf_pred) * 1.0) + (conf_pred * dynamic_pred)
        weight_pred = _clamp(prior * base_pred, self.min_weight, self.max_weight)

        rec["source"] = source
        rec["weight"] = round(float(weight_now), 4)  # active consensus weight
        rec["weight_now"] = round(float(weight_now), 4)
        rec["weight_predictive"] = round(float(weight_pred), 4)

        rec["now_score"] = round(float(100.0 * now_rel), 2)
        rec["lead_score"] = round(float(100.0 * lead_rel), 2)
        rec["predictive_score"] = round(float(100.0 * rel_pred), 2)
        rec["quality_band"] = _quality_band(rec["predictive_score"])

    def _ensure_station_locked(self, stations: Dict[str, Any], sid: str, source: str) -> Dict[str, Any]:
        rec = stations.get(sid)
        if not isinstance(rec, dict):
            rec = {
                "station_id": sid,
                "source": source,
                "now_samples": 0,
                "lead_samples": 0,
            }
            stations[sid] = rec
        elif not rec.get("source"):
            rec["source"] = source
        return rec

    def _update_ema(self, prev: Optional[float], sample: float, alpha: float) -> float:
        if prev is None:
            return float(sample)
        return ((1.0 - alpha) * float(prev)) + (alpha * float(sample))

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
                return {
                    "station_id": station_key,
                    "source": src,
                    "weight": round(float(prior), 4),
                    "weight_now": round(float(prior), 4),
                    "weight_predictive": round(float(prior), 4),
                    "now_score": 55.0,
                    "lead_score": 55.0,
                    "predictive_score": 55.0,
                    "now_samples": 0,
                    "lead_samples": 0,
                    "quality_band": "INIT",
                }
            if (
                rec.get("weight") is None
                or rec.get("now_score") is None
                or rec.get("lead_score") is None
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
    ) -> List[Dict[str, Any]]:
        market_key = str(market_station_id).upper()
        key = str(sort_by or "weight")
        with self._lock:
            market = self._state.get("markets", {}).get(market_key, {})
            stations = market.get("stations", {}) if isinstance(market, dict) else {}
            rows = [dict(v) for v in stations.values() if isinstance(v, dict)]
        rows.sort(key=lambda r: float(r.get(key) or 0.0), reverse=True)
        return rows[: max(1, int(limit))]

    def get_top_weights(self, market_station_id: str, limit: int = 10) -> Dict[str, float]:
        top = self.get_top_profiles(market_station_id, limit=limit, sort_by="weight")
        return {
            str(p.get("station_id")): round(float(p.get("weight", 0.0)), 4)
            for p in top
            if p.get("station_id")
        }

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
        - compares pending readings whose timestamp is 35-85 minutes before
          official obs timestamp.
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

        with self._lock:
            market = self._market_locked(market_key)
            stations = market["stations"]

            # NOW labels: aligned only.
            official_iso = official_ts.isoformat()
            for r in readings_list:
                sample = self._reading_to_sample(r, fallback_ts)
                if not sample:
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

                err = abs(float(sample["temp_c"]) - official)
                if not bool(sample.get("valid", True)):
                    err += 0.15

                rec["now_samples"] = int(rec.get("now_samples", 0) or 0) + 1
                rec["now_ema_abs_error_c"] = round(
                    self._update_ema(rec.get("now_ema_abs_error_c"), err, self.alpha_now), 4
                )
                rec["last_now_abs_error_c"] = round(float(err), 4)
                rec["last_now_alignment_min"] = round(float(delta_min), 2)
                rec["last_now_official_obs_time_utc"] = official_iso
                rec["last_now_sample_obs_time_utc"] = sample["obs_time_utc"]
                rec["updated_at"] = _utc_now_iso()
                touched_ids.add(sid)

            # LEAD labels: pending queue resolution (35-85 min before official).
            pending = market.get("pending", [])
            keep_pending: List[Dict[str, Any]] = []
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

                temp_val = row.get("temp_c")
                try:
                    temp_c = float(temp_val)
                except Exception:
                    continue

                rec = self._ensure_station_locked(stations, sid, str(row.get("source") or "UNKNOWN"))
                err = abs(temp_c - official)
                if row.get("valid") is False:
                    err += 0.15

                rec["lead_samples"] = int(rec.get("lead_samples", 0) or 0) + 1
                rec["lead_ema_abs_error_c"] = round(
                    self._update_ema(rec.get("lead_ema_abs_error_c"), err, self.alpha_lead), 4
                )
                rec["last_lead_abs_error_c"] = round(float(err), 4)
                rec["last_lead_minutes"] = round(float(lead_min), 2)
                rec["last_lead_official_obs_time_utc"] = official_iso
                rec["last_lead_sample_obs_time_utc"] = row.get("obs_time_utc")
                rec["updated_at"] = _utc_now_iso()
                touched_ids.add(sid)

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
