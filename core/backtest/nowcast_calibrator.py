"""
Nowcast Calibrator - Auto-calibration of NowcastEngine parameters.

Re-runs the NowcastEngine on recorded world observations (METAR data)
with candidate NowcastConfig parameter sets, then evaluates predictions
against ground truth (DayLabel) using Brier/MAE metrics.

This runs independently from the trading policy calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import product
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from core.nowcast_engine import NowcastConfig, NowcastEngine
from core.models import OfficialObs

from .dataset import DatasetBuilder
from .labels import DayLabel
from .metrics import brier_score_multi, mae

logger = logging.getLogger("backtest.nowcast_calibrator")

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")

# Nowcast params eligible for calibration, with default grid values
DEFAULT_NOWCAST_GRID: Dict[str, List[Any]] = {
    "bias_decay_hours": [4.0, 6.0, 8.0],
    "bias_alpha_normal": [0.15, 0.2, 0.3, 0.4],
    "base_sigma_f": [1.5, 2.0, 2.5],
    "sigma_qc_penalty_uncertain": [0.3, 0.5, 0.7],
    "sigma_stale_source_penalty": [0.2, 0.3, 0.5],
}


@dataclass
class NowcastCalibrationResult:
    """Result of a nowcast calibration run."""
    station_id: str
    train_dates: Tuple[date, date]
    val_dates: Tuple[date, date]

    best_params: Dict[str, Any] = field(default_factory=dict)
    best_train_score: float = float("inf")
    best_val_score: float = float("inf")

    all_results: List[Dict[str, Any]] = field(default_factory=list)
    total_combinations: int = 0
    combinations_tested: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "train_dates": [d.isoformat() for d in self.train_dates],
            "val_dates": [d.isoformat() for d in self.val_dates],
            "best_params": self.best_params,
            "best_train_score": self.best_train_score,
            "best_val_score": self.best_val_score,
            "total_combinations": self.total_combinations,
            "combinations_tested": self.combinations_tested,
            "all_results": self.all_results,
        }


class NowcastCalibrator:
    """
    Calibrates NowcastEngine parameters by replaying recorded observations.

    For each candidate NowcastConfig:
      - For each day in the date range:
        1. Load world events from tape (via DatasetBuilder)
        2. Create a fresh NowcastEngine with candidate params
        3. Feed METAR observations chronologically
        4. Compare final tmax_mean prediction to the DayLabel ground truth
      - Compute aggregate Brier score + MAE across all days
    Return the NowcastConfig with the lowest composite loss.
    """

    def __init__(
        self,
        recordings_dir: str = "data/recordings",
        parquet_dir: str = "data/parquet",
    ):
        self.builder = DatasetBuilder(recordings_dir, parquet_dir)

    def calibrate(
        self,
        station_id: str,
        train_start: date,
        train_end: date,
        val_start: date,
        val_end: date,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        max_combinations: int = 500,
    ) -> NowcastCalibrationResult:
        """
        Run grid search over nowcast parameters.

        Args:
            station_id: ICAO station code
            train_start/train_end: Training period
            val_start/val_end: Validation period
            param_grid: Dict mapping param names to candidate values.
                        Defaults to DEFAULT_NOWCAST_GRID.
            max_combinations: Cap on combos to test

        Returns:
            NowcastCalibrationResult with best params
        """
        grid = param_grid or DEFAULT_NOWCAST_GRID

        result = NowcastCalibrationResult(
            station_id=station_id,
            train_dates=(train_start, train_end),
            val_dates=(val_start, val_end),
        )

        # Pre-load datasets for train and val periods
        train_datasets = self.builder.build_datasets_for_range(
            station_id, train_start, train_end
        )
        val_datasets = self.builder.build_datasets_for_range(
            station_id, val_start, val_end
        )

        # Filter to days with labels and world observations
        train_days = [
            ds for ds in train_datasets
            if ds.label and ds.label.y_tmax_aligned is not None and ds.world_events
        ]
        val_days = [
            ds for ds in val_datasets
            if ds.label and ds.label.y_tmax_aligned is not None and ds.world_events
        ]

        if not train_days:
            logger.warning("No usable training days for nowcast calibration")
            return result
        if not val_days:
            logger.warning("No usable validation days for nowcast calibration")
            return result

        logger.info(
            "Nowcast calibration: %d train days, %d val days",
            len(train_days), len(val_days),
        )

        # Generate parameter combinations
        param_names = sorted(grid.keys())
        param_values = [grid[k] for k in param_names]
        all_combos = list(product(*param_values))
        result.total_combinations = len(all_combos)

        if len(all_combos) > max_combinations:
            import random
            all_combos = random.sample(all_combos, max_combinations)
            logger.info(
                "Sampling %d of %d combinations",
                max_combinations, result.total_combinations,
            )

        best_val_loss = float("inf")

        for i, combo in enumerate(all_combos):
            params = dict(zip(param_names, combo))
            result.combinations_tested += 1

            config = self._make_config(params)

            train_loss = self._evaluate(station_id, config, train_days)
            val_loss = self._evaluate(station_id, config, val_days)

            entry = {
                "params": params,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            result.all_results.append(entry)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                result.best_params = params
                result.best_train_score = train_loss
                result.best_val_score = val_loss
                logger.info(
                    "New best nowcast params (combo %d): val_loss=%.4f params=%s",
                    i + 1, val_loss, params,
                )

        logger.info(
            "Nowcast calibration complete: best_val_loss=%.4f, tested %d combos",
            result.best_val_score, result.combinations_tested,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_config(params: Dict[str, Any]) -> NowcastConfig:
        """Build a NowcastConfig from a param dict, using defaults for missing keys."""
        return NowcastConfig(
            bias_decay_hours=params.get("bias_decay_hours", 6.0),
            bias_alpha_normal=params.get("bias_alpha_normal", 0.3),
            base_sigma_f=params.get("base_sigma_f", 2.0),
            sigma_qc_penalty_uncertain=params.get("sigma_qc_penalty_uncertain", 0.5),
            sigma_stale_source_penalty=params.get("sigma_stale_source_penalty", 0.3),
        )

    def _evaluate(
        self,
        station_id: str,
        config: NowcastConfig,
        day_datasets: List,
    ) -> float:
        """
        Evaluate a NowcastConfig across multiple days.

        Returns composite loss = 0.5 * mean_brier + 0.5 * normalized_mae.
        Lower is better.
        """
        brier_scores: List[float] = []
        mae_pairs: List[Tuple[float, float]] = []  # (predicted, actual)

        for ds in day_datasets:
            predicted_tmax, predicted_probs = self._replay_day(
                station_id, config, ds
            )
            label: DayLabel = ds.label

            # MAE contribution (tmax point prediction)
            if predicted_tmax is not None and label.y_tmax_aligned is not None:
                mae_pairs.append((predicted_tmax, label.y_tmax_aligned))

            # Brier contribution (bucket probabilities)
            if (
                predicted_probs
                and label.y_bucket_index is not None
                and 0 <= label.y_bucket_index < len(predicted_probs)
            ):
                brier = brier_score_multi(predicted_probs, label.y_bucket_index)
                brier_scores.append(brier)

        mean_brier = (
            sum(brier_scores) / len(brier_scores) if brier_scores else 1.0
        )
        mean_mae = mae(mae_pairs) if mae_pairs else 10.0
        # Normalize MAE to roughly [0,1] range (5F error -> 1.0)
        normalized_mae = min(mean_mae / 5.0, 2.0)

        return 0.5 * mean_brier + 0.5 * normalized_mae

    def _replay_day(
        self,
        station_id: str,
        config: NowcastConfig,
        dataset,
    ) -> Tuple[Optional[float], List[float]]:
        """
        Replay one day's METAR observations through a fresh NowcastEngine.

        Returns:
            (predicted_tmax, predicted_bucket_probs)
        """
        engine = NowcastEngine(station_id, config)

        # If there are features events with HRRR data, set up base forecast
        for event in dataset.features_events:
            data = event.get("data", event)
            hourly = data.get("t_hourly_f") or data.get("hourly_temps_f")
            if hourly and isinstance(hourly, list) and len(hourly) >= 12:
                ts_str = event.get("ts_ingest_utc") or event.get("ts_utc")
                model_time = self._parse_ts(ts_str)
                if model_time:
                    engine.update_base_forecast(
                        hourly_temps_f=hourly,
                        model_run_time=model_time,
                        model_source=data.get("model_source", "HRRR"),
                        target_date=dataset.market_date,
                    )
                break

        last_dist = None
        for event in dataset.world_events:
            data = event.get("data", event)
            temp_f = data.get("temp_f")
            if temp_f is None:
                continue

            ts_str = event.get("ts_ingest_utc") or event.get("ts_utc")
            obs_time = self._parse_ts(ts_str)
            if obs_time is None:
                continue

            obs = OfficialObs(
                obs_time_utc=obs_time,
                station_id=station_id,
                temp_f=temp_f,
                temp_f_raw=data.get("temp_f_raw", temp_f),
                source=data.get("source", "METAR"),
            )
            qc = data.get("qc_state", "OK")
            last_dist = engine.update_observation(obs, qc_state=qc)

        if last_dist is None:
            return None, []

        predicted_tmax = last_dist.tmax_mean_f
        predicted_probs = [b.probability for b in last_dist.p_bucket]

        return predicted_tmax, predicted_probs

    @staticmethod
    def _parse_ts(ts_val) -> Optional[datetime]:
        """Parse a timestamp value to datetime."""
        if ts_val is None:
            return None
        if isinstance(ts_val, datetime):
            if ts_val.tzinfo is None:
                return ts_val.replace(tzinfo=UTC)
            return ts_val
        if isinstance(ts_val, str):
            try:
                dt = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except ValueError:
                return None
        return None
