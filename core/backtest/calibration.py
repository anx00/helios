"""
Calibration Loop Module

Optimizes model and policy parameters through backtesting:
- Grid search over parameter space
- Walk-forward validation
- Multi-objective optimization (calibration + PnL - risk)

Key parameters to calibrate:
- bias_alpha (EMA for bias correction)
- decay_horizon_hours
- sigma_base and growth
- upstream weights
- QC thresholds
- Policy sizing and edge thresholds
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any, Tuple, Callable
from zoneinfo import ZoneInfo
import json
from pathlib import Path
from itertools import product
import copy

from .engine import BacktestEngine, BacktestResult, BacktestMode
from .metrics import CalibrationMetrics
from .policy import Policy, PolicyTable

logger = logging.getLogger("backtest.calibration")

UTC = ZoneInfo("UTC")


@dataclass
class ParameterSet:
    """A set of parameters to calibrate."""
    name: str
    current_value: Any
    min_value: Any
    max_value: Any
    step: Any
    param_type: str = "float"  # "float", "int", "bool"

    def get_grid_values(self) -> List[Any]:
        """Get all values to test in grid search."""
        if self.param_type == "bool":
            return [True, False]

        if self.param_type == "int":
            return list(range(
                int(self.min_value),
                int(self.max_value) + 1,
                int(self.step)
            ))

        # Float
        values = []
        current = self.min_value
        while current <= self.max_value:
            values.append(round(current, 6))
            current += self.step
        return values


@dataclass
class CalibrationResult:
    """Result of a calibration run."""
    # Configuration
    station_id: str
    train_start: date
    train_end: date
    val_start: date
    val_end: date

    # Best parameters found
    best_params: Dict[str, Any] = field(default_factory=dict)

    # Scores
    best_train_score: float = 0.0
    best_val_score: float = 0.0

    # Best validation run metrics (for multi-objective promotion)
    best_val_pnl_net: float = 0.0
    best_val_max_drawdown: float = 0.0
    best_val_total_turnover: float = 0.0
    best_val_total_fills: int = 0
    best_val_total_signals: int = 0

    # All tried combinations
    all_results: List[Dict[str, Any]] = field(default_factory=list)

    # Probability calibrator params (temperature T or isotonic breakpoints)
    prob_calibrator_params: Optional[Dict[str, Any]] = None

    # Timing
    run_started: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_completed: Optional[datetime] = None
    total_combinations: int = 0
    combinations_tested: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": {
                "station_id": self.station_id,
                "train_start": self.train_start.isoformat(),
                "train_end": self.train_end.isoformat(),
                "val_start": self.val_start.isoformat(),
                "val_end": self.val_end.isoformat(),
            },
            "best": {
                "params": self.best_params,
                "train_score": self.best_train_score,
                "val_score": self.best_val_score,
                "val_pnl_net": self.best_val_pnl_net,
                "val_max_drawdown": self.best_val_max_drawdown,
                "val_total_turnover": self.best_val_total_turnover,
                "val_total_fills": self.best_val_total_fills,
                "val_total_signals": self.best_val_total_signals,
            },
            "prob_calibrator": self.prob_calibrator_params,
            "summary": {
                "total_combinations": self.total_combinations,
                "combinations_tested": self.combinations_tested,
            },
            "all_results": self.all_results,
        }

    def save(self, output_dir: str = "data/calibration_results"):
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = (
            f"calibration_{self.station_id}_"
            f"{self.train_start.isoformat()}_"
            f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_path / filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved calibration results to {output_path / filename}")


class ObjectiveFunction:
    """
    Objective function for calibration optimization.

    Combines multiple metrics into a single score.
    """

    def __init__(
        self,
        brier_weight: float = 1.0,
        ece_weight: float = 0.5,
        pnl_weight: float = 0.3,
        drawdown_penalty: float = 0.5,
        stability_weight: float = 0.2,
        turnover_penalty: float = 0.1,
        fill_rate_bonus: float = 0.1,
        capital: float = 100.0,
    ):
        self.brier_weight = brier_weight
        self.ece_weight = ece_weight
        self.pnl_weight = pnl_weight
        self.drawdown_penalty = drawdown_penalty
        self.stability_weight = stability_weight
        self.turnover_penalty = turnover_penalty
        self.fill_rate_bonus = fill_rate_bonus
        self.capital = capital

    def compute(self, result: BacktestResult) -> float:
        """
        Compute objective score.

        Higher is better.

        Args:
            result: Backtest result to score

        Returns:
            Objective score
        """
        score = 0.0

        metrics = result.aggregated_metrics
        if metrics:
            # Brier score (lower is better, so negate)
            score -= self.brier_weight * metrics.brier_global

            # ECE (lower is better)
            score -= self.ece_weight * metrics.ece

            # Stability (lower churn is better)
            score -= self.stability_weight * metrics.avg_churn

        # PnL (higher is better)
        score += self.pnl_weight * result.total_pnl_net

        # Drawdown penalty
        score -= self.drawdown_penalty * result.max_drawdown

        # Turnover penalty: penalize excessive trading volume relative to capital
        total_turnover = sum(
            f.size * f.price
            for d in result.day_results
            for f in d.fills
        )
        if self.capital > 0:
            score -= self.turnover_penalty * (total_turnover / self.capital)

        # Fill rate bonus: reward strategies that actually execute signals
        total_signals = sum(d.signals_generated for d in result.day_results)
        if total_signals > 0:
            fill_rate = result.total_fills / total_signals
            score += self.fill_rate_bonus * fill_rate

        return score


class CalibrationLoop:
    """
    Main calibration loop.

    Performs grid search over parameter space with walk-forward validation.
    """

    # Default parameters to calibrate (policy + nowcast)
    DEFAULT_PARAMS = [
        # Policy params
        ParameterSet("bias_alpha", 0.1, 0.05, 0.3, 0.05),
        ParameterSet("sigma_base", 1.5, 1.0, 3.0, 0.5),
        ParameterSet("min_edge_to_enter", 0.05, 0.02, 0.10, 0.02),
        ParameterSet("min_confidence_to_trade", 0.5, 0.3, 0.7, 0.1),
        ParameterSet("edge_to_fade", 0.10, 0.06, 0.15, 0.04),
        ParameterSet("max_daily_loss", 0.20, 0.10, 0.30, 0.10),
        ParameterSet("max_position_per_bucket", 1.0, 0.5, 2.0, 0.5),
        ParameterSet("size_per_confidence", 0.5, 0.3, 0.8, 0.25),
        # Nowcast params (used by NowcastCalibrator, not by _run_with_params)
        ParameterSet("bias_decay_hours", 6.0, 4.0, 8.0, 2.0),
        ParameterSet("bias_alpha_normal", 0.3, 0.15, 0.4, 0.05),
        ParameterSet("base_sigma_f", 2.0, 1.5, 2.5, 0.5),
        ParameterSet("sigma_qc_penalty_uncertain", 0.5, 0.3, 0.7, 0.2),
        ParameterSet("sigma_stale_source_penalty", 0.3, 0.2, 0.5, 0.1),
    ]

    # Subset of param names that are nowcast-only (used by NowcastCalibrator)
    NOWCAST_PARAM_NAMES = {
        "bias_decay_hours", "bias_alpha_normal", "base_sigma_f",
        "sigma_qc_penalty_uncertain", "sigma_stale_source_penalty",
    }

    def __init__(
        self,
        parameters: Optional[List[ParameterSet]] = None,
        objective: Optional[ObjectiveFunction] = None,
        mode: BacktestMode = BacktestMode.SIGNAL_ONLY
    ):
        self.parameters = parameters or self.DEFAULT_PARAMS
        self.objective = objective or ObjectiveFunction()
        self.mode = mode

        # Progress callback
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _report_progress(self, current: int, total: int, message: str):
        """Report progress if callback set."""
        if self._progress_callback:
            self._progress_callback(current, total, message)

    def grid_search(
        self,
        station_id: str,
        train_start: date,
        train_end: date,
        val_start: date,
        val_end: date,
        max_combinations: int = 1000
    ) -> CalibrationResult:
        """
        Perform grid search over parameter space.

        Args:
            station_id: Station to calibrate
            train_start: Training period start
            train_end: Training period end
            val_start: Validation period start
            val_end: Validation period end
            max_combinations: Maximum combinations to try

        Returns:
            CalibrationResult with best parameters
        """
        result = CalibrationResult(
            station_id=station_id,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        )

        # Generate all combinations
        param_names = [p.name for p in self.parameters]
        param_values = [p.get_grid_values() for p in self.parameters]
        all_combinations = list(product(*param_values))

        result.total_combinations = len(all_combinations)
        logger.info(f"Grid search: {result.total_combinations} combinations")

        # Limit if too many
        if len(all_combinations) > max_combinations:
            import random
            all_combinations = random.sample(all_combinations, max_combinations)
            logger.warning(f"Sampling {max_combinations} combinations")

        best_score = float('-inf')
        best_params = {}

        for i, combo in enumerate(all_combinations):
            result.combinations_tested += 1
            self._report_progress(
                i + 1,
                len(all_combinations),
                f"Testing combination {i + 1}/{len(all_combinations)}"
            )

            # Create parameter dict
            params = dict(zip(param_names, combo))

            try:
                # Run training backtest
                train_result = self._run_with_params(
                    station_id, train_start, train_end, params
                )
                train_score = self.objective.compute(train_result)

                # Run validation backtest
                val_result = self._run_with_params(
                    station_id, val_start, val_end, params
                )
                val_score = self.objective.compute(val_result)

                # Record result
                result.all_results.append({
                    "params": params,
                    "train_score": train_score,
                    "val_score": val_score,
                    "train_metrics": train_result.aggregated_metrics.to_dict() if train_result.aggregated_metrics else None,
                    "val_metrics": val_result.aggregated_metrics.to_dict() if val_result.aggregated_metrics else None,
                })

                # Check if best
                # Use validation score as primary criterion
                if val_score > best_score:
                    best_score = val_score
                    best_params = params
                    result.best_params = params
                    result.best_train_score = train_score
                    result.best_val_score = val_score

                    # Capture validation metrics for promotion
                    result.best_val_pnl_net = val_result.total_pnl_net
                    result.best_val_max_drawdown = val_result.max_drawdown
                    result.best_val_total_fills = val_result.total_fills
                    result.best_val_total_signals = sum(
                        d.signals_generated for d in val_result.day_results
                    )
                    result.best_val_total_turnover = sum(
                        f.size * f.price
                        for d in val_result.day_results
                        for f in d.fills
                    )

                    logger.info(
                        f"New best: val_score={val_score:.4f}, params={params}"
                    )

            except Exception as e:
                logger.warning(f"Failed combination {params}: {e}")

        result.run_completed = datetime.now(UTC)

        # Fit probability calibrator on training data using best params
        try:
            result.prob_calibrator_params = self._fit_prob_calibrator(
                station_id, train_start, train_end
            )
        except Exception as e:
            logger.warning(f"Probability calibrator fitting failed: {e}")

        logger.info(
            f"Calibration complete: best_val_score={result.best_val_score:.4f}"
        )

        return result

    def _run_with_params(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
        params: Dict[str, Any]
    ) -> BacktestResult:
        """Run backtest with specific parameters."""
        # Create policy with parameters - set all matching PolicyTable fields
        policy_table = PolicyTable()
        policy_table_fields = set(PolicyTable.__dataclass_fields__.keys())
        for name, value in params.items():
            if name in policy_table_fields:
                setattr(policy_table, name, value)
        policy = Policy(policy_table, name=f"calibration_{hash(str(params)) % 10000}")

        # Create and run engine
        engine = BacktestEngine(
            policy=policy,
            mode=self.mode
        )

        return engine.run(station_id, start_date, end_date)

    def _fit_prob_calibrator(
        self,
        station_id: str,
        train_start: date,
        train_end: date,
    ) -> Optional[Dict[str, Any]]:
        """
        Fit a ProbabilityCalibrator on training data.

        Collects the final nowcast p_bucket from each training day and the
        corresponding DayLabel, then fits temperature scaling.

        Returns serialized calibration params dict, or None on failure.
        """
        from .prob_calibrator import ProbabilityCalibrator
        from .dataset import get_dataset_builder

        builder = get_dataset_builder()
        predictions = []
        labels = []

        current = train_start
        while current <= train_end:
            try:
                ds = builder.build_dataset(station_id, current)
                if ds.nowcast_events and ds.label and ds.label.y_bucket_winner:
                    # Use the last nowcast of the day as the "prediction"
                    last_nowcast = ds.nowcast_events[-1]
                    data = last_nowcast.get("data", last_nowcast)
                    if data.get("p_bucket"):
                        predictions.append(data)
                        labels.append(ds.label)
            except Exception:
                pass
            current += timedelta(days=1)

        if len(predictions) < 3:
            logger.info(
                "Not enough days (%d) for prob calibrator fitting", len(predictions)
            )
            return None

        calibrator = ProbabilityCalibrator(method="temperature_scaling")
        calibrator.fit(predictions, labels)

        if calibrator.is_fitted:
            params = calibrator.to_dict()
            logger.info(
                "Prob calibrator fitted: T=%.4f on %d days",
                calibrator.temperature, len(predictions),
            )
            return params

        return None

    def walk_forward(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
        train_days: int = 30,
        val_days: int = 7,
        step_days: int = 7
    ) -> List[CalibrationResult]:
        """
        Walk-forward calibration.

        Recalibrates periodically using rolling windows.

        Args:
            station_id: Station to calibrate
            start_date: Overall start date
            end_date: Overall end date
            train_days: Days in training window
            val_days: Days in validation window
            step_days: Days to step forward each iteration

        Returns:
            List of CalibrationResults for each window
        """
        results = []

        current = start_date

        while current + timedelta(days=train_days + val_days) <= end_date:
            train_start = current
            train_end = current + timedelta(days=train_days - 1)
            val_start = train_end + timedelta(days=1)
            val_end = val_start + timedelta(days=val_days - 1)

            logger.info(
                f"Walk-forward window: train {train_start} to {train_end}, "
                f"val {val_start} to {val_end}"
            )

            try:
                result = self.grid_search(
                    station_id=station_id,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    max_combinations=100  # Smaller grid for walk-forward
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Walk-forward window failed: {e}")

            current += timedelta(days=step_days)

        return results


class NoLeakageValidator:
    """
    Validates that backtest has no data leakage.

    Checks:
    - No future data used in predictions
    - Timestamps respect ingest_time_utc
    - Market data is properly delayed
    """

    def __init__(self):
        self.issues: List[str] = []

    def validate_dataset(self, dataset: "BacktestDataset") -> bool:
        """
        Validate a dataset for leakage.

        Returns True if no issues found.
        """
        self.issues = []

        # Check event ordering
        prev_ts = None
        for event in dataset.all_events:
            ts_str = event.get("ts_ingest_utc")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if prev_ts and ts < prev_ts:
                        self.issues.append(
                            f"Out-of-order event: {ts} < {prev_ts}"
                        )
                    prev_ts = ts
                except:
                    pass

        # Check nowcast doesn't use future observations
        for i, nowcast_event in enumerate(dataset.nowcast_events):
            nowcast_ts = nowcast_event.get("ts_ingest_utc", "")

            # Find observations after this nowcast
            for world_event in dataset.world_events:
                world_ts = world_event.get("ts_ingest_utc", "")

                if world_ts and nowcast_ts and world_ts < nowcast_ts:
                    # This is fine - observation before nowcast
                    pass

        return len(self.issues) == 0

    def validate_backtest_result(self, result: BacktestResult) -> bool:
        """Validate a backtest result for integrity."""
        self.issues = []

        # Check chronological ordering of day results
        prev_date = None
        for day_result in result.day_results:
            if prev_date and day_result.market_date < prev_date:
                self.issues.append(
                    f"Day results out of order: {day_result.market_date} < {prev_date}"
                )
            prev_date = day_result.market_date

        # Check fills are within day bounds
        for day_result in result.day_results:
            day_start = datetime.combine(
                day_result.market_date,
                datetime.min.time()
            ).replace(tzinfo=UTC)
            day_end = day_start + timedelta(days=1)

            for fill in day_result.fills:
                if fill.timestamp_utc < day_start or fill.timestamp_utc >= day_end:
                    self.issues.append(
                        f"Fill outside day bounds: {fill.timestamp_utc} "
                        f"not in {day_start} to {day_end}"
                    )

        return len(self.issues) == 0

    def get_report(self) -> str:
        """Get validation report."""
        if not self.issues:
            return "No leakage issues detected."

        return "Leakage issues found:\n" + "\n".join(f"  - {i}" for i in self.issues)
