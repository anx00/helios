"""
Risk gate for autotrader decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

from .models import DecisionContext, RiskSnapshot


@dataclass
class RiskConfig:
    max_daily_loss: float = 0.10
    max_total_exposure: float = 1.5
    max_position_per_bucket: float = 0.5
    max_orders_per_hour: int = 120
    cooldown_seconds: int = 30
    max_nowcast_age_seconds: int = 600
    max_market_age_seconds: int = 600
    blocked_qc_states: tuple[str, ...] = ("STALE", "GAP", "ERROR", "OUTLIER", "SEVERE")


class RiskGate:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def evaluate(
        self,
        context: DecisionContext,
        positions: Dict[str, Dict[str, float]],
        daily_pnl: float,
        orders_last_hour: int,
        last_trade_ts: datetime | None,
    ) -> RiskSnapshot:
        reasons: List[str] = []
        total_exposure = sum(abs(float(v.get("size", 0.0))) for v in positions.values())

        if daily_pnl <= -abs(self.config.max_daily_loss):
            reasons.append("daily_loss_limit")

        if total_exposure > self.config.max_total_exposure:
            reasons.append("max_total_exposure")

        for bucket, info in positions.items():
            size = abs(float(info.get("size", 0.0)))
            if size > self.config.max_position_per_bucket:
                reasons.append(f"max_position_per_bucket:{bucket}")

        if orders_last_hour > self.config.max_orders_per_hour:
            reasons.append("max_orders_per_hour")

        if (
            last_trade_ts is not None
            and (context.ts_utc - last_trade_ts) < timedelta(seconds=self.config.cooldown_seconds)
        ):
            reasons.append("cooldown_blocked")

        if context.nowcast_age_seconds > self.config.max_nowcast_age_seconds:
            reasons.append("stale_nowcast")

        if context.market_age_seconds > self.config.max_market_age_seconds:
            reasons.append("stale_market")

        # If Polymarket is already targeting a different settlement day (e.g., today's market is mature/resolved),
        # block trading until we have a day+1 compatible model distribution.
        if getattr(context, "target_mismatch", False):
            reasons.append("target_mismatch")
        if getattr(context, "market_mature", False):
            reasons.append("market_mature")

        qc = (context.qc_state or "").upper()
        if qc and qc in self.config.blocked_qc_states:
            reasons.append("qc_blocked")

        return RiskSnapshot(
            ts_utc=context.ts_utc,
            station_id=context.station_id,
            blocked=bool(reasons),
            reasons=reasons,
            daily_pnl=float(daily_pnl),
            total_exposure=float(total_exposure),
            orders_last_hour=int(orders_last_hour),
        )
