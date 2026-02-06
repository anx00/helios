"""
Comprehensive tests for Phase 5 Backtesting Module.

Tests cover:
- Metrics calculations
- Label management
- Policy evaluation
- Execution simulation
- Dataset building
- Engine orchestration
- Calibration loop
- Report generation
"""

import pytest
import math
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetrics:
    """Test metric calculations."""

    def test_brier_score_perfect(self):
        """Brier score should be 0 for perfect prediction."""
        from core.backtest.metrics import brier_score

        # Predicted 100% and it happened
        assert brier_score(1.0, True) == 0.0
        # Predicted 0% and it didn't happen
        assert brier_score(0.0, False) == 0.0

    def test_brier_score_worst(self):
        """Brier score should be 1 for worst prediction."""
        from core.backtest.metrics import brier_score

        # Predicted 100% but didn't happen
        assert brier_score(1.0, False) == 1.0
        # Predicted 0% but it happened
        assert brier_score(0.0, True) == 1.0

    def test_brier_score_middle(self):
        """Brier score for 50% prediction."""
        from core.backtest.metrics import brier_score

        # 50% prediction
        assert brier_score(0.5, True) == 0.25
        assert brier_score(0.5, False) == 0.25

    def test_brier_score_multi(self):
        """Multi-class Brier score."""
        from core.backtest.metrics import brier_score_multi

        # Perfect prediction
        probs = [0.0, 1.0, 0.0]
        assert brier_score_multi(probs, 1) == 0.0

        # Uniform prediction
        probs = [1/3, 1/3, 1/3]
        # Expected: avg of (1/3-0)^2, (1/3-1)^2, (1/3-0)^2
        expected = ((1/3)**2 + (1/3-1)**2 + (1/3)**2) / 3
        assert abs(brier_score_multi(probs, 1) - expected) < 0.0001

    def test_log_loss_perfect(self):
        """Log loss for perfect prediction."""
        from core.backtest.metrics import log_loss

        # Near-perfect (can't use exactly 1.0 due to log)
        assert log_loss(0.999, True) < 0.01
        assert log_loss(0.001, False) < 0.01

    def test_log_loss_bad(self):
        """Log loss for bad prediction is high."""
        from core.backtest.metrics import log_loss

        # Bad prediction
        assert log_loss(0.1, True) > 2.0
        assert log_loss(0.9, False) > 2.0

    def test_expected_calibration_error(self):
        """ECE calculation."""
        from core.backtest.metrics import expected_calibration_error

        # Perfectly calibrated: predictions match outcomes
        predictions = [
            (0.1, False), (0.1, False), (0.1, False), (0.1, False),
            (0.1, False), (0.1, False), (0.1, False), (0.1, False),
            (0.1, True), (0.1, False),  # 10% = 1/10 true
        ]
        ece = expected_calibration_error(predictions, n_bins=5)
        # Should be close to 0 for calibrated predictions
        assert ece < 0.2

    def test_reliability_diagram_data(self):
        """Reliability diagram data generation."""
        from core.backtest.metrics import reliability_diagram_data

        predictions = [(0.1, False), (0.5, True), (0.9, True)]
        data = reliability_diagram_data(predictions, n_bins=5)

        assert "bin_centers" in data
        assert "actual_freqs" in data
        assert len(data["bin_centers"]) == 5

    def test_sharpness(self):
        """Sharpness calculation."""
        from core.backtest.metrics import sharpness

        # Sharp prediction (low entropy)
        sharp_probs = [[0.9, 0.1, 0.0]]
        sharp_val = sharpness(sharp_probs)

        # Uncertain prediction (high entropy)
        uncertain_probs = [[0.33, 0.34, 0.33]]
        uncertain_val = sharpness(uncertain_probs)

        # Sharp should be higher than uncertain
        assert sharp_val > uncertain_val

    def test_mae_rmse(self):
        """MAE and RMSE calculations."""
        from core.backtest.metrics import mae, rmse

        predictions = [(10, 12), (20, 18), (30, 30)]

        # MAE = (2 + 2 + 0) / 3 = 4/3
        assert abs(mae(predictions) - 4/3) < 0.001

        # RMSE = sqrt((4 + 4 + 0) / 3) = sqrt(8/3)
        assert abs(rmse(predictions) - math.sqrt(8/3)) < 0.001

    def test_bias(self):
        """Bias calculation."""
        from core.backtest.metrics import bias

        # Over-predicting
        predictions = [(12, 10), (22, 20), (32, 30)]
        assert bias(predictions) == 2.0

        # Under-predicting
        predictions = [(8, 10), (18, 20), (28, 30)]
        assert bias(predictions) == -2.0

    def test_prediction_churn(self):
        """Prediction churn calculation."""
        from core.backtest.metrics import prediction_churn

        # Sequence with changing probabilities
        seq = [
            {"p_bucket": [{"probability": 0.5}, {"probability": 0.5}]},
            {"p_bucket": [{"probability": 0.7}, {"probability": 0.3}]},
            {"p_bucket": [{"probability": 0.7}, {"probability": 0.3}]},
        ]

        churn = prediction_churn(seq)
        # First to second: |0.5-0.7| + |0.5-0.3| = 0.4
        # Second to third: 0
        assert abs(churn - 0.4) < 0.001

    def test_flip_count(self):
        """Flip count calculation."""
        from core.backtest.metrics import flip_count

        # Sequence with one flip
        seq = [
            {"p_bucket": [{"probability": 0.6}, {"probability": 0.4}]},
            {"p_bucket": [{"probability": 0.4}, {"probability": 0.6}]},
            {"p_bucket": [{"probability": 0.3}, {"probability": 0.7}]},
        ]

        flips = flip_count(seq)
        assert flips == 1  # Winner changed once


# =============================================================================
# Labels Tests
# =============================================================================

class TestLabels:
    """Test label management."""

    def test_label_creation(self):
        """Create a DayLabel."""
        from core.backtest.labels import DayLabel

        label = DayLabel(
            station_id="KLGA",
            market_date=date(2026, 1, 15),
            y_bucket_winner="21-22°F",
            y_tmax_aligned=21.0,
            y_bucket_index=3
        )

        assert label.station_id == "KLGA"
        assert label.y_bucket_winner == "21-22°F"

    def test_label_to_dict(self):
        """Label serialization."""
        from core.backtest.labels import DayLabel

        label = DayLabel(
            station_id="KLGA",
            market_date=date(2026, 1, 15),
            y_bucket_winner="21-22°F"
        )

        d = label.to_dict()
        assert d["station_id"] == "KLGA"
        assert d["market_date"] == "2026-01-15"

    def test_label_manager_bucket_for_temp(self):
        """Get bucket for temperature."""
        from core.backtest.labels import LabelManager

        mgr = LabelManager(labels_dir="data/test_labels")

        # Test various temperatures
        bucket, idx = mgr.get_bucket_for_temp(21.3)
        assert bucket == "21-22°F"

        bucket, idx = mgr.get_bucket_for_temp(21.7)
        assert bucket == "22-23°F"

        bucket, idx = mgr.get_bucket_for_temp(17.0)
        assert bucket == "18°F or below"

    def test_label_manager_tpeak_bin(self):
        """Get t-peak bin for hour."""
        from core.backtest.labels import LabelManager

        mgr = LabelManager(labels_dir="data/test_labels")

        assert mgr.get_tpeak_bin(14) == "14-16"
        assert mgr.get_tpeak_bin(0) == "00-02"
        assert mgr.get_tpeak_bin(23) == "22-24"


# =============================================================================
# Policy Tests
# =============================================================================

class TestPolicy:
    """Test policy evaluation."""

    def test_policy_creation(self):
        """Create policies."""
        from core.backtest.policy import (
            create_conservative_policy,
            create_aggressive_policy
        )

        conservative = create_conservative_policy()
        aggressive = create_aggressive_policy()

        assert conservative.name == "conservative"
        assert aggressive.name == "aggressive"

    def test_policy_edge_calculation(self):
        """Calculate edge."""
        from core.backtest.policy import PolicyTable, Side

        table = PolicyTable()

        # Model says 70%, market at 60% -> 10% edge for BUY
        edge = table.calculate_edge(0.7, 0.6, Side.BUY)
        assert abs(edge - 0.1) < 0.0001

        # Model says 30%, market at 40% -> 10% edge for SELL
        edge = table.calculate_edge(0.3, 0.4, Side.SELL)
        assert abs(edge - 0.1) < 0.0001

    def test_policy_state_machine(self):
        """Test policy state transitions."""
        from core.backtest.policy import Policy, PolicyState

        policy = Policy()
        assert policy.get_state() == PolicyState.NEUTRAL

        # Simulate state change
        policy.set_state(PolicyState.BUILD_POSITION, datetime.now(UTC))
        assert policy.get_state() == PolicyState.BUILD_POSITION

    def test_policy_evaluate(self):
        """Test policy evaluation."""
        from core.backtest.policy import Policy, PolicyTable

        table = PolicyTable(
            min_edge_to_enter=0.05,
            min_confidence_to_trade=0.5
        )
        policy = Policy(table)

        nowcast = {
            "p_bucket": [
                {"label": "20-21°F", "probability": 0.7},
                {"label": "21-22°F", "probability": 0.3}
            ],
            "confidence": 0.8
        }

        market_state = {
            "20-21°F": {"best_bid": 0.55, "best_ask": 0.58},
            "21-22°F": {"best_bid": 0.25, "best_ask": 0.28}
        }

        signals = policy.evaluate(
            timestamp_utc=datetime.now(UTC),
            nowcast=nowcast,
            market_state=market_state
        )

        # Should generate at least one signal (model has edge)
        assert len(signals) > 0

    def test_policy_handles_partial_orderbook_levels(self):
        """Policy should not crash when bid/ask levels are partially missing."""
        from core.backtest.policy import Policy, PolicyTable

        table = PolicyTable(
            min_edge_to_enter=0.01,
            min_confidence_to_trade=0.1,
        )
        policy = Policy(table)

        nowcast = {
            "p_bucket": [{"label": "20-21°F", "probability": 0.70}],
            "confidence": 0.9,
        }

        # Real recordings can have one-sided books temporarily (bid or ask missing).
        market_state = {
            "20-21°F": {
                "best_bid": None,
                "best_ask": 0.40,
                "bid_depth": 0,
                "ask_depth": 120,
            }
        }

        result = policy.evaluate_with_reasons(
            timestamp_utc=datetime.now(UTC),
            nowcast=nowcast,
            market_state=market_state,
            qc_flags=[],
            nowcast_age_seconds=0.0,
            market_age_seconds=0.0,
        )
        assert result is not None
        assert isinstance(result.signals, list)


# =============================================================================
# Simulator Tests
# =============================================================================

class TestSimulator:
    """Test execution simulation."""

    def test_taker_model(self):
        """Test taker execution."""
        from core.backtest.simulator import TakerModel

        taker = TakerModel(fee_rate=0.02, base_slippage=0.005)

        market_state = {
            "21-22°F": {
                "best_bid": 0.50,
                "best_ask": 0.52,
                "bid_depth": 100,
                "ask_depth": 100
            }
        }

        fill = taker.execute(
            timestamp_utc=datetime.now(UTC),
            bucket="21-22°F",
            side="buy",
            size=10,
            market_state=market_state
        )

        # Should fill at ask + slippage
        assert fill.price >= 0.52
        assert fill.fees > 0
        assert fill.side == "buy"

    def test_maker_model_order(self):
        """Test maker order placement."""
        from core.backtest.simulator import MakerModel

        maker = MakerModel()

        order = maker.place_order(
            timestamp_utc=datetime.now(UTC),
            bucket="21-22°F",
            side="buy",
            size=10,
            limit_price=0.50
        )

        assert order.bucket == "21-22°F"
        assert order.status == "pending"
        assert order.limit_price == 0.50

    def test_execution_simulator(self):
        """Test combined execution simulator."""
        from core.backtest.simulator import ExecutionSimulator
        from core.backtest.policy import PolicySignal, Side, OrderType

        sim = ExecutionSimulator()

        signal = PolicySignal(
            timestamp_utc=datetime.now(UTC),
            bucket="21-22°F",
            side=Side.BUY,
            order_type=OrderType.TAKER,
            target_size=10,
            max_price=0.55,
            min_price=0.45,
            confidence=0.8,
            urgency=0.9
        )

        market_state = {
            "21-22°F": {"best_bid": 0.50, "best_ask": 0.52}
        }

        fill = sim.execute_signal(signal, market_state)

        assert fill is not None
        assert fill.bucket == "21-22°F"


# =============================================================================
# Dataset Tests
# =============================================================================

class TestDataset:
    """Test dataset building."""

    def test_timeline_state(self):
        """Test TimelineState."""
        from core.backtest.dataset import TimelineState

        state = TimelineState(
            timestamp_utc=datetime.now(UTC),
            timestamp_nyc=datetime.now(NYC)
        )

        assert state.is_stale()  # No data, should be stale

        state.metar_age_seconds = 100
        state.nowcast_age_seconds = 100
        assert not state.is_stale(max_age_seconds=600)


# =============================================================================
# Calibration Tests
# =============================================================================

class TestCalibration:
    """Test calibration functions."""

    def test_parameter_set_grid(self):
        """Test parameter grid generation."""
        from core.backtest.calibration import ParameterSet

        param = ParameterSet(
            name="test",
            current_value=0.1,
            min_value=0.0,
            max_value=0.25,
            step=0.1
        )

        grid = param.get_grid_values()
        # Should include 0.0, 0.1, 0.2
        assert len(grid) >= 3
        assert 0.0 in grid
        assert 0.1 in grid
        assert 0.2 in grid

    def test_objective_function(self):
        """Test objective function."""
        from core.backtest.calibration import ObjectiveFunction
        from core.backtest.engine import BacktestResult, BacktestMode
        from core.backtest.metrics import CalibrationMetrics

        obj = ObjectiveFunction(
            brier_weight=1.0,
            pnl_weight=0.5
        )

        # Create mock result
        result = BacktestResult(
            station_id="KLGA",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 7),
            mode=BacktestMode.SIGNAL_ONLY,
            policy_name="test"
        )
        result.aggregated_metrics = CalibrationMetrics(
            brier_global=0.1,
            ece=0.05
        )
        result.total_pnl_net = 10.0

        score = obj.compute(result)
        # Score should be positive (good PnL, decent metrics)
        assert isinstance(score, float)

    def test_no_leakage_validator(self):
        """Test leakage validation."""
        from core.backtest.calibration import NoLeakageValidator

        validator = NoLeakageValidator()
        report = validator.get_report()

        # No issues initially
        assert "No leakage" in report


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full system."""

    def test_imports(self):
        """Test all imports work."""
        from core.backtest import (
            BacktestEngine,
            BacktestMode,
            BacktestResult,
            Policy,
            ExecutionSimulator,
            MetricsCalculator,
            CalibrationMetrics,
            LabelManager,
            DatasetBuilder,
            CalibrationLoop,
            ReportGenerator,
        )

        # All imports should work
        assert BacktestEngine is not None
        assert BacktestMode is not None

    def test_metrics_calculator_flow(self):
        """Test full metrics calculation flow."""
        from core.backtest.metrics import MetricsCalculator
        from core.backtest.labels import DayLabel

        calc = MetricsCalculator()

        # Add mock day
        nowcast_seq = [
            {
                "p_bucket": [
                    {"label": "20-21°F", "probability": 0.6},
                    {"label": "21-22°F", "probability": 0.4}
                ],
                "tmax_mean_f": 21.0,
                "t_peak_bins": [
                    {"label_short": "12-14", "probability": 0.6},
                    {"label_short": "14-16", "probability": 0.4}
                ]
            }
        ]

        label = DayLabel(
            station_id="KLGA",
            market_date=date(2026, 1, 15),
            y_bucket_winner="20-21°F",
            y_bucket_index=0,
            y_tmax_aligned=20.5,
            y_t_peak_bin="12-14"
        )

        calc.add_day(nowcast_seq, label)
        metrics = calc.compute()

        assert metrics.n_days == 1
        assert metrics.brier_global >= 0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
