"""
Backtest Engine - Main Orchestration

Runs backtests by:
1. Loading datasets
2. Iterating through timeline
3. Applying policy to generate signals
4. Simulating execution
5. Calculating metrics
6. Generating results

Supports two modes:
- Signal-only: Just evaluate prediction quality
- Execution-aware: Full simulation with trading
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple, Callable
from zoneinfo import ZoneInfo
import math
import json
from pathlib import Path

from .dataset import BacktestDataset, DatasetBuilder, TimelineState
from .labels import DayLabel, LabelManager
from .metrics import MetricsCalculator, CalibrationMetrics
from .policy import (
    Policy, PolicySignal, PolicyEvaluationResult, NoTradeReason,
    create_conservative_policy
)
from .simulator import ExecutionSimulator, Fill
from core.polymarket_labels import normalize_label, normalize_p_bucket, format_range_label

logger = logging.getLogger("backtest.engine")

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


class BacktestMode(Enum):
    """Backtest mode."""
    SIGNAL_ONLY = "signal_only"          # Just evaluate predictions
    EXECUTION_AWARE = "execution_aware"  # Full trading simulation


@dataclass
class DayResult:
    """Results for a single day."""
    station_id: str
    market_date: date

    # Metrics
    metrics: Optional[CalibrationMetrics] = None

    # Trading results (execution-aware only)
    signals_generated: int = 0
    fills: List[Fill] = field(default_factory=list)
    pnl_gross: float = 0.0
    pnl_net: float = 0.0  # After fees/slippage
    max_drawdown: float = 0.0
    turnover_usd: float = 0.0

    # Label info
    actual_winner: Optional[str] = None
    actual_tmax: Optional[float] = None
    predicted_winner: Optional[str] = None
    predicted_tmax: Optional[float] = None

    # Data quality
    total_events: int = 0
    nowcast_count: int = 0
    metar_count: int = 0
    market_count: int = 0
    staleness_avg: float = 0.0

    # === NEW FIELDS for FIX_BACKTEST ===

    # Timeline coverage
    timeline_steps: int = 0
    timesteps_with_nowcast: int = 0
    timesteps_with_market: int = 0

    # Three-layer outputs (forward references, populated after dataclass defined)
    prediction_points: List[Any] = field(default_factory=list)
    decision_points: List[Any] = field(default_factory=list)
    orders: List[Any] = field(default_factory=list)  # Maker orders lifecycle

    # Reason counters
    reason_counters: Dict[str, int] = field(default_factory=dict)

    # Timeline summary
    first_ts_utc: Optional[datetime] = None
    last_ts_utc: Optional[datetime] = None
    interval_seconds: float = 60.0

    # Data status
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        import math
        # Helper to convert inf/nan to None for JSON serialization
        def safe_float(v):
            if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
                return None
            return v

        return {
            "station_id": self.station_id,
            "market_date": self.market_date.isoformat(),
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "trading": {
                "signals_generated": self.signals_generated,
                "fills_count": len(self.fills),
                "pnl_gross": safe_float(self.pnl_gross),
                "pnl_net": safe_float(self.pnl_net),
                "max_drawdown": safe_float(self.max_drawdown),
                "turnover_usd": safe_float(self.turnover_usd),
            },
            "predictions": {
                "actual_winner": self.actual_winner,
                "actual_tmax": safe_float(self.actual_tmax),
                "predicted_winner": self.predicted_winner,
                "predicted_tmax": safe_float(self.predicted_tmax),
            },
            "data_quality": {
                "total_events": self.total_events,
                "nowcast_count": self.nowcast_count,
                "metar_count": self.metar_count,
                "market_count": self.market_count,
                "staleness_avg": safe_float(self.staleness_avg),
            },
            # === NEW ===
            "coverage": {
                "timeline_steps": self.timeline_steps,
                "timesteps_with_nowcast": self.timesteps_with_nowcast,
                "timesteps_with_market": self.timesteps_with_market,
            },
            "timeline_summary": {
                "first_ts_utc": self.first_ts_utc.isoformat() if self.first_ts_utc else None,
                "last_ts_utc": self.last_ts_utc.isoformat() if self.last_ts_utc else None,
                "interval_seconds": self.interval_seconds,
            },
            "status": self.status,
            "reason_counters": self.reason_counters,
            "prediction_points": [p.to_dict() for p in self.prediction_points],
            "decision_points": [d.to_dict() for d in self.decision_points],
            "orders": [o.to_dict() if hasattr(o, 'to_dict') else o for o in self.orders],
            "fills": [f.to_dict() for f in self.fills],
        }


@dataclass
class PredictionPoint:
    """A single prediction at a specific timestep."""
    ts_utc: datetime
    ts_nyc: datetime

    # Prediction data
    top_bucket: str                    # argmax bucket label
    top_bucket_prob: float             # probability of top bucket
    p_bucket: List[Dict[str, Any]]     # full distribution (compacted)
    tmax_mean: Optional[float] = None  # point prediction
    tmax_sigma: Optional[float] = None # uncertainty

    # Context
    confidence: float = 0.0
    qc_state: str = "UNKNOWN"

    # Market snapshot at this point
    market_snapshot: Optional[Dict[str, Any]] = None  # mid/spread/depth

    # Staleness
    nowcast_age_seconds: float = 0.0
    market_age_seconds: float = float('inf')

    def to_dict(self) -> Dict[str, Any]:
        import math
        def safe_float(v):
            if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
                return None
            return v

        return {
            "ts_utc": self.ts_utc.isoformat(),
            "ts_nyc": self.ts_nyc.isoformat(),
            "top_bucket": self.top_bucket,
            "top_bucket_prob": self.top_bucket_prob,
            "p_bucket": self.p_bucket,
            "tmax_mean": safe_float(self.tmax_mean),
            "tmax_sigma": safe_float(self.tmax_sigma),
            "confidence": self.confidence,
            "qc_state": self.qc_state,
            "market_snapshot": self.market_snapshot,
            "nowcast_age_seconds": safe_float(self.nowcast_age_seconds),
            "market_age_seconds": safe_float(self.market_age_seconds),
        }


@dataclass
class DecisionPoint:
    """A trading decision point at a specific timestep."""
    ts_utc: datetime
    ts_nyc: datetime

    # Decision outcome
    signals_count: int = 0
    signals: List[PolicySignal] = field(default_factory=list)
    no_trade_reasons: List[str] = field(default_factory=list)  # NoTradeReason values

    # Edge information (for top N buckets)
    edge_info: List[Dict[str, Any]] = field(default_factory=list)

    # Context
    policy_state: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.ts_utc.isoformat(),
            "ts_nyc": self.ts_nyc.isoformat(),
            "signals_count": self.signals_count,
            "signals": [s.to_dict() for s in self.signals],
            "no_trade_reasons": self.no_trade_reasons,
            "edge_info": self.edge_info,
            "policy_state": self.policy_state,
        }


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    # Configuration
    station_id: str
    start_date: date
    end_date: date
    mode: BacktestMode
    policy_name: str

    # Timing
    run_started: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_completed: Optional[datetime] = None
    run_duration_seconds: float = 0.0

    # Per-day results
    day_results: List[DayResult] = field(default_factory=list)

    # Aggregated metrics
    aggregated_metrics: Optional[CalibrationMetrics] = None

    # Aggregated trading stats
    total_pnl_gross: float = 0.0
    total_pnl_net: float = 0.0
    total_fills: int = 0
    total_turnover: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Data coverage
    days_with_data: int = 0
    days_with_labels: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        import math
        # Helper to convert inf/nan to None for JSON serialization
        def safe_float(v):
            if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
                return None
            return v

        return {
            "config": {
                "station_id": self.station_id,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "mode": self.mode.value,
                "policy_name": self.policy_name,
            },
            "timing": {
                "run_started": self.run_started.isoformat(),
                "run_completed": self.run_completed.isoformat() if self.run_completed else None,
                "run_duration_seconds": safe_float(self.run_duration_seconds),
            },
            "aggregated_metrics": self.aggregated_metrics.to_dict() if self.aggregated_metrics else None,
            "trading_summary": {
                "total_pnl_gross": safe_float(self.total_pnl_gross),
                "total_pnl_net": safe_float(self.total_pnl_net),
                "total_fills": self.total_fills,
                "total_turnover": safe_float(self.total_turnover),
                "win_rate": safe_float(self.win_rate),
                "sharpe_ratio": safe_float(self.sharpe_ratio),
                "max_drawdown": safe_float(self.max_drawdown),
            },
            "coverage": {
                "days_with_data": self.days_with_data,
                "days_with_labels": self.days_with_labels,
                "total_days": len(self.day_results),
            },
            "day_results": [d.to_dict() for d in self.day_results],
        }

    def save(self, output_dir: str = "data/backtest_results"):
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = (
            f"backtest_{self.station_id}_"
            f"{self.start_date.isoformat()}_"
            f"{self.end_date.isoformat()}_"
            f"{self.mode.value}.json"
        )

        with open(output_path / filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved backtest results to {output_path / filename}")

    def summary(self) -> str:
        """Generate text summary."""
        return f"""
Backtest Results: {self.station_id}
{'=' * 50}
Period: {self.start_date} to {self.end_date}
Mode: {self.mode.value}
Policy: {self.policy_name}
Duration: {self.run_duration_seconds:.1f}s

Coverage:
  Days with data: {self.days_with_data}
  Days with labels: {self.days_with_labels}

{self.aggregated_metrics.summary() if self.aggregated_metrics else "No metrics available"}

Trading Summary:
  Total PnL (gross): ${self.total_pnl_gross:.2f}
  Total PnL (net): ${self.total_pnl_net:.2f}
  Total Fills: {self.total_fills}
  Win Rate: {self.win_rate:.1%}
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Max Drawdown: {self.max_drawdown:.1%}
"""


class BacktestEngine:
    """
    Main backtest engine.

    Orchestrates the entire backtesting process.
    """

    def __init__(
        self,
        dataset_builder: Optional[DatasetBuilder] = None,
        policy: Optional[Policy] = None,
        simulator: Optional[ExecutionSimulator] = None,
        mode: BacktestMode = BacktestMode.SIGNAL_ONLY
    ):
        from .dataset import get_dataset_builder

        self.dataset_builder = dataset_builder or get_dataset_builder()
        self.policy = policy or create_conservative_policy()
        self.simulator = simulator or ExecutionSimulator()
        self.mode = mode

        # Metrics calculator
        self.metrics_calc = MetricsCalculator()

        # Progress callback
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress updates: (current, total, message)"""
        self._progress_callback = callback

    def _report_progress(self, current: int, total: int, message: str):
        """Report progress if callback set."""
        if self._progress_callback:
            self._progress_callback(current, total, message)

    def run(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
        interval_seconds: float = 60,
    ) -> BacktestResult:
        """
        Run backtest for a date range.

        Args:
            station_id: Station to backtest
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            interval_seconds: Timeline iteration interval

        Returns:
            BacktestResult with all results
        """
        result = BacktestResult(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            mode=self.mode,
            policy_name=self.policy.name,
        )

        # Reset metrics calculator
        self.metrics_calc.reset()

        # Iterate through dates
        current_date = start_date
        total_days = (end_date - start_date).days + 1
        day_idx = 0

        while current_date <= end_date:
            day_idx += 1
            self._report_progress(day_idx, total_days, f"Processing {current_date}")

            try:
                day_result = self._run_day(
                    station_id=station_id,
                    market_date=current_date,
                    interval_seconds=interval_seconds
                )
                # Skip empty dates without events.
                if (day_result.total_events or 0) <= 0:
                    current_date += timedelta(days=1)
                    continue

                result.day_results.append(day_result)
                result.days_with_data += 1

                if day_result.actual_winner:
                    result.days_with_labels += 1

            except Exception as e:
                logger.error(f"Error processing {current_date}: {e}")

            current_date += timedelta(days=1)

        # Compute aggregated metrics
        result.aggregated_metrics = self.metrics_calc.compute()

        # Aggregate trading stats
        result.total_fills = sum(len(d.fills) for d in result.day_results)
        result.total_pnl_gross = sum(d.pnl_gross for d in result.day_results)
        result.total_pnl_net = sum(d.pnl_net for d in result.day_results)
        result.total_turnover = sum(d.turnover_usd for d in result.day_results)

        # Calculate win rate
        profitable_days = sum(1 for d in result.day_results if d.pnl_net > 0)
        if result.day_results:
            result.win_rate = profitable_days / len(result.day_results)

        # Calculate Sharpe (simplified)
        daily_returns = [d.pnl_net for d in result.day_results if d.pnl_net != 0]
        if len(daily_returns) > 1:
            import statistics
            mean_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns)
            if std_return > 0:
                result.sharpe_ratio = mean_return / std_return * (252 ** 0.5)  # Annualized

        # Calculate max drawdown
        cumulative = 0.0
        peak = 0.0
        for d in result.day_results:
            cumulative += d.pnl_net
            peak = max(peak, cumulative)
            drawdown = (peak - cumulative) / peak if peak > 0 else 0
            result.max_drawdown = max(result.max_drawdown, drawdown)

        # Mark completion
        result.run_completed = datetime.now(UTC)
        result.run_duration_seconds = (
            result.run_completed - result.run_started
        ).total_seconds()

        logger.info(
            f"Backtest complete: {result.days_with_data} days, "
            f"PnL: ${result.total_pnl_net:.2f}"
        )

        return result

    def _run_day(
        self,
        station_id: str,
        market_date: date,
        interval_seconds: float
    ) -> DayResult:
        """Run backtest for a single day."""
        logger.debug(f"Running day: {market_date}")

        # Build dataset
        dataset = self.dataset_builder.build_dataset(station_id, market_date)
        integrity_issues = self._check_dataset_integrity(dataset)

        day_result = DayResult(
            station_id=station_id,
            market_date=market_date,
            total_events=dataset.total_events,
            nowcast_count=len(dataset.nowcast_events),
            metar_count=len(dataset.world_events),
            market_count=len(dataset.market_events),
            interval_seconds=interval_seconds,
        )
        if integrity_issues:
            day_result.status = "integrity_warning"
            day_result.reason_counters["integrity_issues"] = len(integrity_issues)
            logger.warning(
                "Dataset integrity issues for %s/%s: %s",
                station_id,
                market_date,
                "; ".join(integrity_issues[:5]),
            )

        # Get label
        if dataset.label:
            day_result.actual_winner = dataset.label.y_bucket_winner
            day_result.actual_tmax = dataset.label.y_tmax_aligned

        # Reset policy and simulator
        self.policy.reset()
        self.simulator.reset()

        # Collect nowcast sequence for metrics
        nowcast_sequence = []

        # Track staleness
        staleness_samples = []

        # === NEW: Initialize counters for FIX_BACKTEST ===
        from collections import defaultdict
        total_signals = 0  # FIX: accumulator for signals
        reason_counters = defaultdict(int)
        prediction_points = []
        decision_points = []

        timeline_steps = 0
        timesteps_with_nowcast = 0
        timesteps_with_market = 0

        # Iterate through timeline
        prev_state = None
        first_ts = None
        last_ts = None

        max_staleness_seconds = self.policy.table.max_staleness_seconds

        for state in dataset.iterate_timeline(interval_seconds):
            timeline_steps += 1

            if first_ts is None:
                first_ts = state.timestamp_utc
            last_ts = state.timestamp_utc

            # Track staleness
            staleness_samples.append(state.metar_age_seconds)

            # Track coverage
            if state.nowcast:
                timesteps_with_nowcast += 1

            has_market = (
                state.market is not None
                and not math.isinf(state.market_age_seconds)
                and state.market_age_seconds <= max_staleness_seconds
            )
            if has_market:
                timesteps_with_market += 1

            # Collect nowcasts for metrics (unique only)
            if state.nowcast and state.nowcast != (prev_state.nowcast if prev_state else None):
                nowcast_sequence.append(state.nowcast)

            # === NEW: Collect PredictionPoint at EVERY timestep with nowcast ===
            if state.nowcast:
                pred_point = self._build_prediction_point(state)
                prediction_points.append(pred_point)

            # In execution-aware mode, generate and execute signals
            if self.mode == BacktestMode.EXECUTION_AWARE:
                # Use real market data only (no synthetic fallback)
                market_state = state.market if has_market else None

                if state.nowcast:
                    # === NEW: Use evaluate_with_reasons ===
                    eval_result = self.policy.evaluate_with_reasons(
                        timestamp_utc=state.timestamp_utc,
                        nowcast=state.nowcast,
                        market_state=market_state,
                        qc_flags=state.qc_flags,
                        nowcast_age_seconds=state.nowcast_age_seconds,
                        market_age_seconds=state.market_age_seconds
                    )

                    # Hard gate: no trading without real market data
                    if market_state is None and eval_result.signals:
                        reason_counters["hard_gate_no_market"] += len(eval_result.signals)
                        eval_result.signals = []

                    signals = eval_result.signals

                    # === FIX: Accumulate signals properly ===
                    total_signals += len(signals)

                    # === NEW: Build DecisionPoint ===
                    decision_point = DecisionPoint(
                        ts_utc=state.timestamp_utc,
                        ts_nyc=state.timestamp_nyc,
                        signals_count=len(signals),
                        signals=signals,
                        no_trade_reasons=[r.value for r in eval_result.no_trade_reasons],
                        edge_info=[{"bucket": k, "edge": v} for k, v in eval_result.edge_by_bucket.items()],
                        policy_state=self.policy.get_state().value,
                    )
                    decision_points.append(decision_point)

                    # === NEW: Update reason counters ===
                    for reason in eval_result.no_trade_reasons:
                        reason_counters[reason.value] += 1

                    # Execute signals only if real market data available
                    if market_state is not None:
                        for signal in signals:
                            fill = self.simulator.execute_signal(signal, market_state)
                            if fill:
                                day_result.fills.append(fill)
                                self.policy.record_fill(
                                    fill.timestamp_utc,
                                    fill.bucket,
                                    fill.size,
                                    fill.price,
                                    signal.side
                                )

                        # Update maker orders
                        maker_fills = self.simulator.update(
                            state.timestamp_utc,
                            market_state,
                            interval_seconds
                        )
                        day_result.fills.extend(maker_fills)
                else:
                    # No nowcast at this timestep
                    reason_counters["no_nowcast"] += 1

            prev_state = state

        # === NEW: Store accumulated data ===
        day_result.signals_generated = total_signals  # FIX: use accumulated total
        day_result.timeline_steps = timeline_steps
        day_result.timesteps_with_nowcast = timesteps_with_nowcast
        day_result.timesteps_with_market = timesteps_with_market
        day_result.prediction_points = prediction_points
        day_result.decision_points = decision_points
        day_result.orders = self.simulator.get_orders()
        day_result.reason_counters = dict(reason_counters)
        day_result.first_ts_utc = first_ts
        day_result.last_ts_utc = last_ts

        # Compute turnover
        day_result.turnover_usd = sum(f.size * f.price for f in day_result.fills)

        if self.mode == BacktestMode.EXECUTION_AWARE and timesteps_with_market == 0:
            day_result.status = "no_market_data"
            day_result.signals_generated = 0
            day_result.fills = []
            day_result.pnl_gross = 0.0
            day_result.pnl_net = 0.0
            day_result.turnover_usd = 0.0

        # Calculate PnL at settlement
        if dataset.label and self.mode == BacktestMode.EXECUTION_AWARE:
            day_result.pnl_gross, day_result.pnl_net = self._calculate_day_pnl(
                day_result.fills,
                dataset.label
            )

        # Calculate average staleness (filter out infinity values)
        valid_staleness = [s for s in staleness_samples if not math.isinf(s)]
        if valid_staleness:
            day_result.staleness_avg = sum(valid_staleness) / len(valid_staleness)

        # Get predicted values from last nowcast
        if nowcast_sequence:
            last_nowcast = nowcast_sequence[-1]
            day_result.predicted_tmax = last_nowcast.get("tmax_mean_f")

            # Get predicted winner (highest probability bucket)
            p_bucket = normalize_p_bucket(last_nowcast.get("p_bucket", []))
            if p_bucket:
                max_prob = 0
                for b in p_bucket:
                    if isinstance(b, dict):
                        # Support both "prob" and "probability" keys
                        prob = b.get("prob", b.get("probability", 0))
                        label = b.get("label", "")
                        if prob > max_prob:
                            max_prob = prob
                            day_result.predicted_winner = normalize_label(label)

            # Fallback: derive bucket from tmax_mean if no predicted_winner
            if not day_result.predicted_winner and day_result.predicted_tmax is not None:
                # Derive bucket from temperature (e.g., 17.5°F -> "17-18" or "<18")
                tmax = day_result.predicted_tmax
                low = int(tmax)
                # Standard 2-degree buckets aligned to even numbers
                if low % 2 == 1:
                    low -= 1
                day_result.predicted_winner = format_range_label(low, low + 1)

        # Add day to metrics calculator
        if dataset.label and nowcast_sequence:
            self.metrics_calc.add_day(nowcast_sequence, dataset.label)

        # Calculate day metrics
        if dataset.label and nowcast_sequence:
            day_result.metrics = self.metrics_calc.compute_for_day(
                nowcast_sequence, dataset.label
            )

        return day_result

    def _check_dataset_integrity(self, dataset: BacktestDataset) -> List[str]:
        """
        Lightweight anti-leakage/integrity checks based on ts_ingest_utc ordering.
        """
        issues: List[str] = []
        prev_ts = None
        for event in dataset.all_events:
            ts_str = event.get("ts_ingest_utc") or event.get("ts_utc")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            except Exception:
                continue
            if prev_ts is not None and ts < prev_ts:
                issues.append(f"out_of_order:{ts.isoformat()}<{prev_ts.isoformat()}")
            prev_ts = ts

        # Verify nowcast events are not ahead of ingest ordering.
        prev_nowcast = None
        for event in dataset.nowcast_events:
            ts_str = event.get("ts_ingest_utc") or event.get("ts_utc")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            except Exception:
                continue
            if prev_nowcast is not None and ts < prev_nowcast:
                issues.append(f"nowcast_out_of_order:{ts.isoformat()}<{prev_nowcast.isoformat()}")
            prev_nowcast = ts

        return issues

    def _build_prediction_point(self, state: TimelineState) -> PredictionPoint:
        """Build a PredictionPoint from a TimelineState."""
        nowcast = state.nowcast

        # Extract top bucket (normalized to Polymarket labels)
        p_bucket = normalize_p_bucket(nowcast.get("p_bucket", []))
        top_bucket = ""
        top_bucket_prob = 0.0

        if p_bucket:
            for b in p_bucket:
                if isinstance(b, dict):
                    prob = b.get("prob", b.get("probability", 0))
                    if prob > top_bucket_prob:
                        top_bucket_prob = prob
                        top_bucket = normalize_label(b.get("label", ""))

        # Build market snapshot
        market_snapshot = None
        if state.market:
            # Compact snapshot: top bucket's mid/spread/depth
            if top_bucket and top_bucket in state.market:
                bucket_data = state.market[top_bucket]
                bid = bucket_data.get("best_bid", 0)
                ask = bucket_data.get("best_ask", 0)
                market_snapshot = {
                    "top_bucket": top_bucket,
                    "mid": (bid + ask) / 2 if bid and ask else None,
                    "spread": ask - bid if bid and ask else None,
                    "bid_depth": bucket_data.get("bid_depth"),
                    "ask_depth": bucket_data.get("ask_depth"),
                }

        return PredictionPoint(
            ts_utc=state.timestamp_utc,
            ts_nyc=state.timestamp_nyc,
            top_bucket=top_bucket,
            top_bucket_prob=top_bucket_prob,
            p_bucket=p_bucket,  # Store full distribution
            tmax_mean=nowcast.get("tmax_mean_f") or nowcast.get("tmax_mean"),
            tmax_sigma=nowcast.get("tmax_sigma_f") or nowcast.get("tmax_sigma"),
            confidence=nowcast.get("confidence", 0),
            qc_state=nowcast.get("qc_state", "UNKNOWN"),
            market_snapshot=market_snapshot,
            nowcast_age_seconds=state.nowcast_age_seconds,
            market_age_seconds=state.market_age_seconds,
        )

    def _build_market_state(self, state: TimelineState) -> Dict[str, Any]:
        """Build market state from timeline state or simulate."""
        if state.market:
            return state.market

        # Simulate market based on nowcast
        market_state = {}

        if state.nowcast:
            p_bucket = normalize_p_bucket(state.nowcast.get("p_bucket", []))
            for b in p_bucket:
                if isinstance(b, dict):
                    label = normalize_label(b.get("label", ""))
                    prob = b.get("prob", b.get("probability", 0))

                    # Simulate bid/ask around model price
                    spread = 0.04  # 4% spread
                    market_state[label] = {
                        "best_bid": max(0.01, prob - spread / 2),
                        "best_ask": min(0.99, prob + spread / 2),
                        "bid_depth": 100,
                        "ask_depth": 100,
                    }

        return market_state

    def _calculate_day_pnl(
        self,
        fills: List[Fill],
        label: DayLabel
    ) -> Tuple[float, float]:
        """
        Calculate PnL at settlement.

        Returns:
            (gross_pnl, net_pnl)
        """
        if not fills or not label.y_bucket_winner:
            return 0.0, 0.0

        gross_pnl = 0.0
        total_fees = 0.0

        for fill in fills:
            # Determine settlement value based on token type
            outcome = getattr(fill, "outcome", "yes")
            if outcome == "no":
                # NO token: settles at $1 if bucket LOSES, $0 if it wins
                settlement = 0.0 if fill.bucket == label.y_bucket_winner else 1.0
            else:
                # YES token: settles at $1 if bucket WINS, $0 if it loses
                settlement = 1.0 if fill.bucket == label.y_bucket_winner else 0.0

            # Calculate PnL
            if fill.side == "buy":
                # Bought at fill.price, settles at settlement
                pnl = (settlement - fill.price) * fill.size
            else:
                # Sold at fill.price, settles at settlement (closing position)
                pnl = (fill.price - settlement) * fill.size

            gross_pnl += pnl
            total_fees += fill.fees + fill.slippage

        net_pnl = gross_pnl - total_fees

        return gross_pnl, net_pnl


# =============================================================================
# Convenience functions
# =============================================================================

def run_signal_backtest(
    station_id: str,
    start_date: date,
    end_date: date
) -> BacktestResult:
    """Run a signal-only backtest."""
    engine = BacktestEngine(mode=BacktestMode.SIGNAL_ONLY)
    return engine.run(station_id, start_date, end_date)


def run_execution_backtest(
    station_id: str,
    start_date: date,
    end_date: date,
    policy: Optional[Policy] = None
) -> BacktestResult:
    """Run an execution-aware backtest."""
    engine = BacktestEngine(
        mode=BacktestMode.EXECUTION_AWARE,
        policy=policy or create_conservative_policy()
    )
    return engine.run(station_id, start_date, end_date)
