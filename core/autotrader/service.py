"""
Autotrader runtime service.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from core.polymarket_labels import normalize_label

from .bandit import LinUCBBandit
from .learning import LearningConfig, LearningRunner
from .models import DecisionContext, StrategyDecision, TradingMode
from .paper_broker import PaperBroker
from .risk import RiskConfig, RiskGate
from .storage import AutoTraderStorage
from .strategy import Strategy, build_strategy_catalog

logger = logging.getLogger("autotrader.service")

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


@dataclass
class AutoTraderConfig:
    station_id: str = "KLGA"
    mode: TradingMode = TradingMode.PAPER
    selection_mode: str = "linucb"  # static | linucb
    decision_interval_seconds: int = 60
    lambda_drawdown: float = 0.20
    lambda_turnover: float = 0.01
    default_strategy_static: str = "conservative_edge"


class AutoTraderService:
    _instance: Optional["AutoTraderService"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        config: Optional[AutoTraderConfig] = None,
        storage: Optional[AutoTraderStorage] = None,
    ):
        if self._initialized:
            return

        self.config = config or AutoTraderConfig()
        self.storage = storage or AutoTraderStorage()
        self.strategies: Dict[str, Strategy] = build_strategy_catalog()
        self.bandit = LinUCBBandit(self.strategies.keys(), feature_dim=10, alpha=0.8)
        self.risk_gate = RiskGate(RiskConfig())
        self.broker = PaperBroker()
        self.learning = LearningRunner(self.storage)

        self._running = False
        self._paused = False
        self._risk_off = False
        self._task: Optional[asyncio.Task] = None

        self._last_trade_ts: Optional[datetime] = None
        self._orders_recent: deque[datetime] = deque(maxlen=1000)
        self._mid_history: deque[float] = deque(maxlen=20)
        self._prob_history: deque[float] = deque(maxlen=20)
        self._recent_decisions: deque[Dict[str, Any]] = deque(maxlen=500)
        self._selected_strategy_counts: Dict[str, int] = {k: 0 for k in self.strategies}
        self._strategy_rewards: Dict[str, float] = {k: 0.0 for k in self.strategies}
        self._last_context_summary: Dict[str, Any] = {}

        self._last_total_pnl = 0.0
        self._equity_peak = 0.0
        self._max_drawdown = 0.0
        self._last_mark_to_market = 0.0

        self._load_bandit_state()
        self._initialized = True

    def _load_bandit_state(self) -> None:
        state = self.storage.load_latest_bandit_state(scope=self.config.station_id)
        if state is None:
            return
        try:
            self.bandit = LinUCBBandit.from_state(state)
        except Exception:
            logger.warning("Failed to restore bandit state; using fresh state")

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._paused = False
        self._task = asyncio.create_task(self._loop())
        logger.info("Autotrader started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.storage.save_bandit_state(self.config.station_id, self.bandit.to_state())
        logger.info("Autotrader stopped")

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def risk_off(self) -> None:
        self._risk_off = True

    def risk_on(self) -> None:
        self._risk_off = False

    async def _loop(self) -> None:
        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(1)
                    continue
                await self.step_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Autotrader step failed: %s", exc)
            await asyncio.sleep(self.config.decision_interval_seconds)

    async def _build_context(self) -> Optional[DecisionContext]:
        from core.event_window import get_event_window_manager
        from core.nowcast_integration import get_nowcast_integration
        from market.polymarket_ws import get_ws_client

        now_utc = datetime.now(UTC)
        now_nyc = now_utc.astimezone(NYC)
        station = self.config.station_id

        integration = get_nowcast_integration()
        distribution = integration.get_distribution(station)
        if not distribution:
            return None
        nowcast = distribution.to_dict()

        client = await get_ws_client()
        market_state: Dict[str, Any] = {}
        market_age_seconds = float("inf")
        if client and client.state and hasattr(client.state, "orderbooks"):
            for token_id, book in client.state.orderbooks.items():
                info = client.market_info.get(token_id, "")
                if "|" not in info:
                    continue
                parts = info.split("|")
                if len(parts) >= 3:
                    sid, bracket, outcome = parts[0], parts[1], (parts[2] or "unknown").lower()
                elif len(parts) == 2:
                    sid, bracket, outcome = parts[0], parts[1], "yes"
                else:
                    continue
                if sid != station:
                    continue
                # Strategy pricing is based on YES probability for each bracket.
                if outcome != "yes":
                    continue
                snap = book.get_l2_snapshot(top_n=10)
                if not snap:
                    continue
                label = normalize_label(bracket)
                market_state[label] = {
                    "best_bid": snap.get("best_bid"),
                    "best_ask": snap.get("best_ask"),
                    "bid_depth": snap.get("bid_depth"),
                    "ask_depth": snap.get("ask_depth"),
                    "spread": snap.get("spread"),
                    "mid": snap.get("mid"),
                }
                staleness_ms = snap.get("staleness_ms")
                if staleness_ms is not None:
                    age = float(staleness_ms) / 1000.0
                    market_age_seconds = min(market_age_seconds, age)

        p_bucket = nowcast.get("p_bucket", [])
        confidence = float(nowcast.get("confidence", 0.0))
        tmax_sigma_f = float(nowcast.get("tmax_sigma_f") or nowcast.get("tmax_sigma") or 0.0)
        nowcast_age_seconds = 0.0
        gen_ts = nowcast.get("ts_generated_utc")
        if isinstance(gen_ts, str):
            try:
                gen_dt = datetime.fromisoformat(gen_ts.replace("Z", "+00:00"))
                nowcast_age_seconds = max(0.0, (now_utc - gen_dt).total_seconds())
            except Exception:
                nowcast_age_seconds = 0.0

        top_label = ""
        top_prob = 0.0
        for b in p_bucket:
            prob = float(b.get("probability", b.get("prob", 0.0)))
            if prob > top_prob:
                top_prob = prob
                top_label = normalize_label(b.get("label", ""))

        spread_top = 0.0
        depth_imbalance = 0.0
        mid_top = 0.0
        if top_label and top_label in market_state:
            top_market = market_state[top_label]
            spread_top = float(top_market.get("spread") or 0.0)
            bid_depth = float(top_market.get("bid_depth") or 0.0)
            ask_depth = float(top_market.get("ask_depth") or 0.0)
            if bid_depth + ask_depth > 0:
                depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
            mid_top = float(top_market.get("mid") or 0.0)

        if mid_top > 0.0:
            self._mid_history.append(mid_top)
        if top_prob > 0.0:
            self._prob_history.append(top_prob)

        volatility_short = 0.0
        if len(self._mid_history) >= 3:
            mids = list(self._mid_history)[-10:]
            avg = sum(mids) / len(mids)
            var = sum((m - avg) ** 2 for m in mids) / len(mids)
            volatility_short = var ** 0.5

        prediction_churn_short = 0.0
        if len(self._prob_history) >= 2:
            probs = list(self._prob_history)[-10:]
            prediction_churn_short = sum(abs(probs[i] - probs[i - 1]) for i in range(1, len(probs)))

        win_mgr = get_event_window_manager()
        event_window_active = win_mgr.has_active_window(station)

        return DecisionContext(
            ts_utc=now_utc,
            ts_nyc=now_nyc,
            station_id=station,
            nowcast=nowcast,
            market_state=market_state,
            confidence=confidence,
            tmax_sigma_f=tmax_sigma_f,
            nowcast_age_seconds=nowcast_age_seconds,
            market_age_seconds=market_age_seconds,
            spread_top=spread_top,
            depth_imbalance=depth_imbalance,
            volatility_short=volatility_short,
            prediction_churn_short=prediction_churn_short,
            event_window_active=event_window_active,
            qc_state=str(nowcast.get("qc_state") or nowcast.get("qc") or "UNKNOWN"),
            hour_nyc=now_nyc.hour,
        )

    def _feature_vector(self, ctx: DecisionContext) -> List[float]:
        qc = 1.0 if str(ctx.qc_state).upper() in {"OK", "GOOD"} else 0.0
        return [
            float(ctx.confidence),
            float(ctx.tmax_sigma_f),
            float(min(ctx.nowcast_age_seconds, 3600.0) / 3600.0),
            float(min(ctx.market_age_seconds, 3600.0) / 3600.0),
            float(ctx.spread_top),
            float(ctx.depth_imbalance),
            float(ctx.volatility_short),
            float(ctx.prediction_churn_short),
            1.0 if ctx.event_window_active else 0.0,
            qc + (float(ctx.hour_nyc) / 24.0),
        ]

    def _orders_last_hour(self, now_utc: datetime) -> int:
        cutoff = now_utc - timedelta(hours=1)
        while self._orders_recent and self._orders_recent[0] < cutoff:
            self._orders_recent.popleft()
        return len(self._orders_recent)

    async def step_once(self) -> Dict[str, Any]:
        ctx = await self._build_context()
        if ctx is None:
            return {"status": "no_nowcast"}

        # Evaluate all strategies.
        strategy_decisions: Dict[str, StrategyDecision] = {}
        for name, strategy in self.strategies.items():
            strategy_decisions[name] = strategy.evaluate(ctx)

        feature_vector = self._feature_vector(ctx)
        if self.config.selection_mode == "linucb":
            selected = self.bandit.select(feature_vector, candidates=self.strategies.keys())
            selected_name = selected.strategy_name
            scores = selected.scores
        else:
            selected_name = self.config.default_strategy_static
            scores = {k: 0.0 for k in self.strategies}

        decision = strategy_decisions[selected_name]
        risk = self.risk_gate.evaluate(
            context=ctx,
            positions=self.broker.get_positions(),
            daily_pnl=self.broker.realized_pnl,
            orders_last_hour=self._orders_last_hour(ctx.ts_utc),
            last_trade_ts=self._last_trade_ts,
        )

        if self._risk_off:
            risk.blocked = True
            risk.reasons.append("manual_risk_off")

        orders: List[Dict[str, Any]] = []
        fills: List[Dict[str, Any]] = []
        if not risk.blocked and decision.actions:
            new_orders, new_fills = self.broker.execute_actions(
                timestamp_utc=ctx.ts_utc,
                station_id=ctx.station_id,
                strategy_name=selected_name,
                actions=decision.actions,
                market_state=ctx.market_state,
            )
            orders.extend(o.to_dict() for o in new_orders)
            fills.extend(f.to_dict() for f in new_fills)
            for _ in decision.actions:
                self._orders_recent.append(ctx.ts_utc)
            if decision.actions:
                self._last_trade_ts = ctx.ts_utc

        upd_orders, upd_fills = self.broker.update(
            timestamp_utc=ctx.ts_utc,
            station_id=ctx.station_id,
            market_state=ctx.market_state,
            elapsed_seconds=float(self.config.decision_interval_seconds),
        )
        orders.extend(o.to_dict() for o in upd_orders)
        fills.extend(f.to_dict() for f in upd_fills)

        mtm = self.broker.mark_to_market(ctx.market_state)
        self._last_mark_to_market = mtm
        perf = self.broker.get_performance(mark_to_market_pnl=mtm)
        total_pnl = float(perf["total_pnl"])
        pnl_delta = total_pnl - self._last_total_pnl
        self._last_total_pnl = total_pnl

        self._equity_peak = max(self._equity_peak, total_pnl)
        dd = max(0.0, self._equity_peak - total_pnl)
        prev_max_dd = self._max_drawdown
        self._max_drawdown = max(self._max_drawdown, dd)
        dd_increment = max(0.0, self._max_drawdown - prev_max_dd)

        turnover_penalty = len(decision.actions)
        reward = pnl_delta - (self.config.lambda_drawdown * dd_increment) - (
            self.config.lambda_turnover * turnover_penalty
        )

        if self.config.selection_mode == "linucb":
            self.bandit.update(selected_name, feature_vector, reward)

        self._selected_strategy_counts[selected_name] += 1
        self._strategy_rewards[selected_name] += reward

        # Persist all strategy decisions for auditability.
        for name, d in strategy_decisions.items():
            self.storage.record_decision(
                ts_utc=ctx.ts_utc.isoformat(),
                station_id=ctx.station_id,
                strategy_name=name,
                selected=name == selected_name,
                context=ctx.to_dict(),
                decision=d.to_dict(),
                reward=reward if name == selected_name else None,
            )
        for order in orders:
            self.storage.record_order(order)
        for fill in fills:
            self.storage.record_fill(fill)
        self.storage.record_positions(
            ts_utc=ctx.ts_utc.isoformat(),
            station_id=ctx.station_id,
            realized_pnl=self.broker.realized_pnl,
            positions=self.broker.get_positions(),
        )
        self.storage.record_reward(
            ts_utc=ctx.ts_utc.isoformat(),
            station_id=ctx.station_id,
            strategy_name=selected_name,
            reward=reward,
            pnl_component=pnl_delta,
            drawdown_component=dd_increment,
            turnover_component=float(turnover_penalty),
        )
        self.storage.save_bandit_state(self.config.station_id, self.bandit.to_state())

        event = {
            "ts_utc": ctx.ts_utc.isoformat(),
            "station_id": ctx.station_id,
            "selected_strategy": selected_name,
            "scores": scores,
            "risk_blocked": risk.blocked,
            "risk_reasons": risk.reasons,
            "actions": decision.actions,
            "no_trade_reasons": decision.no_trade_reasons,
            "fills_count": len(fills),
            "reward": reward,
            "performance": perf,
        }
        nowcast_labels = {
            normalize_label(b.get("label", ""))
            for b in (ctx.nowcast.get("p_bucket") or [])
            if isinstance(b, dict) and b.get("label")
        }
        market_labels = set(ctx.market_state.keys())
        overlap_count = len(nowcast_labels & market_labels)
        self._last_context_summary = {
            "ts_utc": ctx.ts_utc.isoformat(),
            "confidence": ctx.confidence,
            "tmax_sigma_f": ctx.tmax_sigma_f,
            "nowcast_age_seconds": ctx.nowcast_age_seconds,
            "market_age_seconds": ctx.market_age_seconds,
            "spread_top": ctx.spread_top,
            "depth_imbalance": ctx.depth_imbalance,
            "volatility_short": ctx.volatility_short,
            "prediction_churn_short": ctx.prediction_churn_short,
            "event_window_active": ctx.event_window_active,
            "qc_state": ctx.qc_state,
            "market_bucket_count": len(ctx.market_state),
            "nowcast_bucket_count": len(nowcast_labels),
            "market_overlap_count": overlap_count,
        }
        self._recent_decisions.append(event)
        return event

    def run_learning_once(self) -> Dict[str, Any]:
        cfg = LearningConfig(station_id=self.config.station_id)
        return self.learning.run_once(cfg)

    def get_status(self) -> Dict[str, Any]:
        perf = self.broker.get_performance(mark_to_market_pnl=self._last_mark_to_market)
        return {
            "running": self._running,
            "paused": self._paused,
            "risk_off": self._risk_off,
            "station_id": self.config.station_id,
            "mode": self.config.mode.value,
            "selection_mode": self.config.selection_mode,
            "decision_interval_seconds": self.config.decision_interval_seconds,
            "strategies": list(self.strategies.keys()),
            "performance": perf,
            "max_drawdown_abs": self._max_drawdown,
            "strategy_selection_counts": self._selected_strategy_counts,
            "strategy_rewards": self._strategy_rewards,
            "last_context": self._last_context_summary,
        }

    def get_strategies(self) -> Dict[str, Any]:
        total = sum(self._selected_strategy_counts.values()) or 1
        out = []
        for name in self.strategies:
            count = self._selected_strategy_counts.get(name, 0)
            out.append(
                {
                    "name": name,
                    "selected_count": count,
                    "selected_weight": count / total,
                    "reward_sum": self._strategy_rewards.get(name, 0.0),
                }
            )
        return {"strategies": out}

    def get_positions(self) -> Dict[str, Any]:
        return {"positions": self.broker.get_positions()}

    def get_performance(self) -> Dict[str, Any]:
        return self.broker.get_performance(mark_to_market_pnl=self._last_mark_to_market)

    def get_decisions(self, limit: int = 200) -> Dict[str, Any]:
        return {"decisions": list(self._recent_decisions)[-limit:]}

    def get_orders(self, limit: int = 200) -> Dict[str, Any]:
        return {"orders": self.broker.get_orders(limit)}

    def get_fills(self, limit: int = 200) -> Dict[str, Any]:
        return {"fills": self.broker.get_fills(limit)}

    def get_learning_runs(self, limit: int = 20) -> Dict[str, Any]:
        return {"runs": self.storage.list_learning_runs(limit)}


_autotrader_service: Optional[AutoTraderService] = None


def get_autotrader_service() -> AutoTraderService:
    global _autotrader_service
    if _autotrader_service is None:
        _autotrader_service = AutoTraderService()
    return _autotrader_service
