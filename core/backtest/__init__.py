"""
HELIOS Phase 5: Backtesting + Calibration Loop

This module provides comprehensive backtesting capabilities for the HELIOS
weather prediction and trading system.

Components:
- labels: Ground truth and target management
- dataset: Backtest dataset construction
- metrics: Prediction metrics (Brier, ECE, calibration)
- policy: Trading policy engine with state machine
- simulator: Execution simulation (taker/maker models)
- engine: Main backtest orchestration
- calibration: Parameter calibration loop
- report: HTML/MD report generation

Usage:
    from core.backtest import BacktestEngine, BacktestMode

    engine = BacktestEngine(mode=BacktestMode.SIGNAL_ONLY)
    result = engine.run("KLGA", date(2026, 1, 1), date(2026, 1, 15))
    print(result.summary())
"""

# Labels
from .labels import (
    LabelManager,
    DayLabel,
    get_label_manager,
)

# Dataset
from .dataset import (
    BacktestDataset,
    DatasetBuilder,
    TimelineState,
    get_dataset_builder,
)

# Metrics
from .metrics import (
    MetricsCalculator,
    CalibrationMetrics,
    brier_score,
    brier_score_multi,
    log_loss,
    log_loss_multi,
    expected_calibration_error,
    reliability_diagram_data,
    sharpness,
    mae,
    rmse,
    bias,
    prediction_churn,
    flip_count,
    tpeak_accuracy,
    tpeak_mass_near_truth,
)

# Policy
from .policy import (
    Policy,
    PolicyState,
    PolicyTable,
    PolicySignal,
    PolicyEvaluationResult,
    Position,
    Side,
    OrderType,
    NoTradeReason,
    MakerPassivePolicy,
    ToyPolicyDebug,
    create_conservative_policy,
    create_aggressive_policy,
    create_fade_policy,
    create_maker_passive_policy,
    create_toy_debug_policy,
)

# Simulator
from .simulator import (
    ExecutionSimulator,
    TakerModel,
    MakerModel,
    Fill,
    FillType,
    Order,
)

# Engine
from .engine import (
    BacktestEngine,
    BacktestResult,
    BacktestMode,
    DayResult,
    run_signal_backtest,
    run_execution_backtest,
)

# Calibration
from .calibration import (
    CalibrationLoop,
    CalibrationResult,
    ParameterSet,
    ObjectiveFunction,
    NoLeakageValidator,
)

# Report
from .report import ReportGenerator

__all__ = [
    # Labels
    "LabelManager",
    "DayLabel",
    "get_label_manager",
    # Dataset
    "BacktestDataset",
    "DatasetBuilder",
    "TimelineState",
    "get_dataset_builder",
    # Metrics
    "MetricsCalculator",
    "CalibrationMetrics",
    "brier_score",
    "brier_score_multi",
    "log_loss",
    "log_loss_multi",
    "expected_calibration_error",
    "reliability_diagram_data",
    "sharpness",
    "mae",
    "rmse",
    "bias",
    "prediction_churn",
    "flip_count",
    "tpeak_accuracy",
    "tpeak_mass_near_truth",
    # Policy
    "Policy",
    "PolicyState",
    "PolicyTable",
    "PolicySignal",
    "PolicyEvaluationResult",
    "Position",
    "Side",
    "OrderType",
    "NoTradeReason",
    "MakerPassivePolicy",
    "ToyPolicyDebug",
    "create_conservative_policy",
    "create_aggressive_policy",
    "create_fade_policy",
    "create_maker_passive_policy",
    "create_toy_debug_policy",
    # Simulator
    "ExecutionSimulator",
    "TakerModel",
    "MakerModel",
    "Fill",
    "FillType",
    "Order",
    # Engine
    "BacktestEngine",
    "BacktestResult",
    "BacktestMode",
    "DayResult",
    "run_signal_backtest",
    "run_execution_backtest",
    # Calibration
    "CalibrationLoop",
    "CalibrationResult",
    "ParameterSet",
    "ObjectiveFunction",
    "NoLeakageValidator",
    # Report
    "ReportGenerator",
]
