"""
Strategy catalog and strategy contract for autotrader.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List

from core.backtest.policy import (
    Policy,
    PolicySignal,
    create_aggressive_policy,
    create_conservative_policy,
    create_fade_policy,
)

from .models import DecisionContext, StrategyDecision


class Strategy(ABC):
    name: str

    @abstractmethod
    def evaluate(self, context: DecisionContext) -> StrategyDecision:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


def _signal_to_action(signal: PolicySignal, force_maker: bool = False) -> Dict:
    order_type = "maker" if force_maker else signal.order_type.value
    return {
        "bucket": signal.bucket,
        "side": signal.side.value,
        "outcome": getattr(signal, "outcome", "yes"),
        "order_type": order_type,
        "target_size": signal.target_size,
        "max_price": signal.max_price,
        "min_price": signal.min_price,
        "confidence": signal.confidence,
        "urgency": signal.urgency,
        "reason": signal.reason,
    }


class PolicyBackedStrategy(Strategy):
    def __init__(
        self,
        name: str,
        policy: Policy,
        require_event_window: bool = False,
        force_maker: bool = False,
    ):
        self.name = name
        self._policy = policy
        self._require_event_window = require_event_window
        self._force_maker = force_maker

    def evaluate(self, context: DecisionContext) -> StrategyDecision:
        if self._require_event_window and not context.event_window_active:
            return StrategyDecision(
                ts_utc=context.ts_utc,
                station_id=context.station_id,
                strategy_name=self.name,
                actions=[],
                no_trade_reasons=["event_window_inactive"],
            )

        result = self._policy.evaluate_with_reasons(
            timestamp_utc=context.ts_utc,
            nowcast=context.nowcast,
            market_state=context.market_state,
            qc_flags=[],
            nowcast_age_seconds=context.nowcast_age_seconds,
            market_age_seconds=context.market_age_seconds,
        )

        actions = [_signal_to_action(s, force_maker=self._force_maker) for s in result.signals]
        reasons = [r.value for r in result.no_trade_reasons]
        return StrategyDecision(
            ts_utc=context.ts_utc,
            station_id=context.station_id,
            strategy_name=self.name,
            actions=actions,
            no_trade_reasons=reasons,
        )

    def reset(self) -> None:
        self._policy.reset()


def build_strategy_catalog() -> Dict[str, Strategy]:
    return {
        "conservative_edge": PolicyBackedStrategy(
            name="conservative_edge",
            policy=create_conservative_policy(),
        ),
        "aggressive_edge": PolicyBackedStrategy(
            name="aggressive_edge",
            policy=create_aggressive_policy(),
        ),
        "fade_event_qc": PolicyBackedStrategy(
            name="fade_event_qc",
            policy=create_fade_policy(),
            require_event_window=True,
        ),
        "maker_passive": PolicyBackedStrategy(
            name="maker_passive",
            policy=create_aggressive_policy(),
            force_maker=True,
        ),
    }
