"""
Backtest adapters for autotrader-centric policy options.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from core.backtest.policy import (
    Policy,
    PolicyEvaluationResult,
    PolicyState,
    create_aggressive_policy,
    create_conservative_policy,
    create_fade_policy,
    create_maker_passive_policy,
)

from .bandit import LinUCBBandit


class MultiBanditBacktestPolicy:
    """
    Lightweight policy wrapper for backtest endpoint compatibility.
    Uses existing policies and optionally LinUCB selector.
    """

    def __init__(self, selection_mode: str = "static", risk_profile: str = "risk_first"):
        self.name = "multi_bandit"
        self.selection_mode = selection_mode
        self.risk_profile = risk_profile
        self._policies: Dict[str, Policy] = {
            "conservative_edge": create_conservative_policy(),
            "aggressive_edge": create_aggressive_policy(),
            "fade_event_qc": create_fade_policy(),
            "maker_passive": create_maker_passive_policy(),
        }
        self._selected = "conservative_edge"
        self.table = self._policies[self._selected].table
        self._bandit = LinUCBBandit(self._policies.keys(), feature_dim=6, alpha=0.8)

    def reset(self) -> None:
        for policy in self._policies.values():
            policy.reset()
        self._selected = "conservative_edge"
        self.table = self._policies[self._selected].table
        self._bandit.reset()

    def _features(
        self,
        nowcast: Optional[Dict],
        nowcast_age_seconds: Optional[float],
        market_age_seconds: Optional[float],
    ) -> List[float]:
        confidence = float((nowcast or {}).get("confidence", 0.0))
        sigma = float((nowcast or {}).get("tmax_sigma_f") or (nowcast or {}).get("tmax_sigma") or 0.0)
        bucket = (nowcast or {}).get("p_bucket", [])
        top_prob = 0.0
        if bucket and isinstance(bucket, list):
            for b in bucket:
                if isinstance(b, dict):
                    top_prob = max(top_prob, float(b.get("prob", b.get("probability", 0.0))))
        return [
            confidence,
            sigma,
            float(nowcast_age_seconds or 0.0),
            float(market_age_seconds or 0.0),
            top_prob,
            1.0 if confidence > 0.6 else 0.0,
        ]

    def _pick_policy(
        self,
        nowcast: Optional[Dict],
        nowcast_age_seconds: Optional[float],
        market_age_seconds: Optional[float],
    ) -> str:
        if self.selection_mode != "linucb":
            return "conservative_edge"
        x = self._features(nowcast, nowcast_age_seconds, market_age_seconds)
        return self._bandit.select(x).strategy_name

    def evaluate_with_reasons(
        self,
        timestamp_utc: datetime,
        nowcast: Optional[Dict],
        market_state: Optional[Dict] = None,
        qc_flags: Optional[List[str]] = None,
        nowcast_age_seconds: Optional[float] = None,
        market_age_seconds: Optional[float] = None
    ) -> PolicyEvaluationResult:
        self._selected = self._pick_policy(nowcast, nowcast_age_seconds, market_age_seconds)
        active = self._policies[self._selected]
        self.table = active.table

        result = active.evaluate_with_reasons(
            timestamp_utc=timestamp_utc,
            nowcast=nowcast,
            market_state=market_state,
            qc_flags=qc_flags,
            nowcast_age_seconds=nowcast_age_seconds,
            market_age_seconds=market_age_seconds,
        )

        # Keep signal trace readable in day_detail.
        for signal in result.signals:
            signal.reason = f"{signal.reason}; selected={self._selected}"

        # Proxy reward as summed edge for quick online adaptation in backtest.
        if self.selection_mode == "linucb":
            edge_proxy = sum(abs(v) for v in result.edge_by_bucket.values())
            self._bandit.update(self._selected, self._features(nowcast, nowcast_age_seconds, market_age_seconds), edge_proxy)
        return result

    def record_fill(
        self,
        timestamp_utc: datetime,
        bucket: str,
        size: float,
        price: float,
        side
    ) -> None:
        self._policies[self._selected].record_fill(timestamp_utc, bucket, size, price, side)

    def record_pnl(self, pnl: float) -> None:
        self._policies[self._selected].record_pnl(pnl)

    def get_state(self) -> PolicyState:
        return self._policies[self._selected].get_state()


def create_multi_bandit_policy(selection_mode: str = "static", risk_profile: str = "risk_first") -> MultiBanditBacktestPolicy:
    return MultiBanditBacktestPolicy(selection_mode=selection_mode, risk_profile=risk_profile)
