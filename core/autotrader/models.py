"""
Autotrader data models.

These types are shared by strategy selection, execution, storage, and API.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TradingMode(str, Enum):
    PAPER = "paper"
    SEMI_AUTO = "semi_auto"
    LIVE_AUTO = "live_auto"


@dataclass
class DecisionContext:
    ts_utc: datetime
    ts_nyc: datetime
    station_id: str
    nowcast: Dict[str, Any]
    market_state: Dict[str, Any]
    nowcast_target_date: str = ""
    market_target_date: str = ""
    target_mismatch: bool = False
    market_mature: bool = False
    health_state: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    tmax_sigma_f: float = 0.0
    nowcast_age_seconds: float = float("inf")
    market_age_seconds: float = float("inf")
    spread_top: float = 0.0
    depth_imbalance: float = 0.0
    volatility_short: float = 0.0
    prediction_churn_short: float = 0.0
    event_window_active: bool = False
    qc_state: str = "UNKNOWN"
    hour_nyc: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ts_utc"] = self.ts_utc.isoformat()
        d["ts_nyc"] = self.ts_nyc.isoformat()
        return d


@dataclass
class StrategyDecision:
    ts_utc: datetime
    station_id: str
    strategy_name: str
    actions: List[Dict[str, Any]] = field(default_factory=list)
    no_trade_reasons: List[str] = field(default_factory=list)
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.ts_utc.isoformat(),
            "station_id": self.station_id,
            "strategy_name": self.strategy_name,
            "actions": self.actions,
            "no_trade_reasons": self.no_trade_reasons,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class BanditState:
    version: str
    alpha: float
    feature_dim: int
    strategies: List[str]
    state: Dict[str, Any]
    updated_at_utc: datetime

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["updated_at_utc"] = self.updated_at_utc.isoformat()
        return d


@dataclass
class PaperOrder:
    ts_utc: datetime
    station_id: str
    strategy_name: str
    order_id: str
    bucket: str
    side: str
    size: float
    limit_price: float
    order_type: str
    status: str
    outcome: str = "yes"
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.ts_utc.isoformat(),
            "station_id": self.station_id,
            "strategy_name": self.strategy_name,
            "order_id": self.order_id,
            "bucket": self.bucket,
            "side": self.side,
            "size": self.size,
            "limit_price": self.limit_price,
            "order_type": self.order_type,
            "status": self.status,
            "outcome": self.outcome,
            "raw": self.raw,
        }


@dataclass
class PaperFill:
    ts_utc: datetime
    station_id: str
    strategy_name: str
    bucket: str
    side: str
    size: float
    price: float
    fill_type: str
    fees: float = 0.0
    slippage: float = 0.0
    outcome: str = "yes"
    order_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.ts_utc.isoformat(),
            "station_id": self.station_id,
            "strategy_name": self.strategy_name,
            "bucket": self.bucket,
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "fill_type": self.fill_type,
            "fees": self.fees,
            "slippage": self.slippage,
            "outcome": self.outcome,
            "order_id": self.order_id,
            "raw": self.raw,
        }


@dataclass
class RiskSnapshot:
    ts_utc: datetime
    station_id: str
    blocked: bool
    reasons: List[str] = field(default_factory=list)
    daily_pnl: float = 0.0
    total_exposure: float = 0.0
    orders_last_hour: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.ts_utc.isoformat(),
            "station_id": self.station_id,
            "blocked": self.blocked,
            "reasons": self.reasons,
            "daily_pnl": self.daily_pnl,
            "total_exposure": self.total_exposure,
            "orders_last_hour": self.orders_last_hour,
        }


@dataclass
class LearningRunResult:
    run_id: str
    station_id: str
    ts_started_utc: datetime
    ts_completed_utc: Optional[datetime] = None
    status: str = "running"
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    val_start: Optional[str] = None
    val_end: Optional[str] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_train_score: float = 0.0
    best_val_score: float = 0.0
    notes: str = ""
    # Multi-objective metrics from validation run
    val_pnl_net: float = 0.0
    val_max_drawdown: float = 0.0
    val_turnover_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "station_id": self.station_id,
            "ts_started_utc": self.ts_started_utc.isoformat(),
            "ts_completed_utc": self.ts_completed_utc.isoformat() if self.ts_completed_utc else None,
            "status": self.status,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "best_params": self.best_params,
            "best_train_score": self.best_train_score,
            "best_val_score": self.best_val_score,
            "notes": self.notes,
            "val_pnl_net": self.val_pnl_net,
            "val_max_drawdown": self.val_max_drawdown,
            "val_turnover_ratio": self.val_turnover_ratio,
        }


@dataclass
class PromotionDecision:
    version_id: str
    station_id: str
    promoted: bool
    reason: str
    metrics: Dict[str, Any]
    ts_utc: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "station_id": self.station_id,
            "promoted": self.promoted,
            "reason": self.reason,
            "metrics": self.metrics,
            "ts_utc": self.ts_utc.isoformat(),
        }
