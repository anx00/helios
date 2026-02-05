"""
HELIOS Autotrader package.
"""

from .models import (
    DecisionContext,
    StrategyDecision,
    BanditState,
    PaperOrder,
    PaperFill,
    RiskSnapshot,
    LearningRunResult,
    PromotionDecision,
    TradingMode,
)
from .bandit import LinUCBBandit
from .risk import RiskConfig, RiskGate
from .paper_broker import PaperBroker
from .strategy import Strategy, build_strategy_catalog
from .storage import AutoTraderStorage
from .learning import LearningConfig, LearningRunner
from .service import AutoTraderConfig, AutoTraderService, get_autotrader_service
from .backtest_adapter import create_multi_bandit_policy, MultiBanditBacktestPolicy
from .execution_adapter_live import LiveExecutionAdapter, LiveExecutionConfig

__all__ = [
    "DecisionContext",
    "StrategyDecision",
    "BanditState",
    "PaperOrder",
    "PaperFill",
    "RiskSnapshot",
    "LearningRunResult",
    "PromotionDecision",
    "TradingMode",
    "LinUCBBandit",
    "RiskConfig",
    "RiskGate",
    "PaperBroker",
    "Strategy",
    "build_strategy_catalog",
    "AutoTraderStorage",
    "LearningConfig",
    "LearningRunner",
    "AutoTraderConfig",
    "AutoTraderService",
    "get_autotrader_service",
    "create_multi_bandit_policy",
    "MultiBanditBacktestPolicy",
    "LiveExecutionAdapter",
    "LiveExecutionConfig",
]
