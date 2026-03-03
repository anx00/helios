"""Live papertrading shadow module for HELIOS.

A separate runtime that shadows the autotrader's decision logic but executes
paper orders against live L2 orderbook data via WebSocket. Never sends real
orders to the exchange.

Reuses shared decision functions from core/autotrader.py:
  - evaluate_trade_candidate()
  - compute_trade_budget_usd()
  - apply_orderbook_guardrails()
  - apply_exit_guardrails()

Uses market/paper_execution.py for realistic L2 book-walking fill simulation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx

from config import STATIONS
from core.autotrader import (
    AutoTradeCandidate,
    AutoTraderConfig,
    TopOfBook,
    _build_candidate_from_trade,
    _normalize_state,
    _position_key,
    _recalculate_open_risk,
    apply_exit_guardrails,
    apply_orderbook_guardrails,
    compute_trade_budget_usd,
    evaluate_trade_candidate,
    load_autotrader_config_from_env,
    resolve_station_market_target,
)
from core.polymarket_labels import normalize_label
from market.paper_execution import PaperExecutionEngine, compact_book_snapshot

logger = logging.getLogger("papertrader")

UTC = timezone.utc
_LOG_WRITE_LOCK = threading.Lock()
PAPER_STATE_SCHEMA_VERSION = 2

PAPER_STRATEGY_DEFS: Tuple[Dict[str, str], ...] = (
    {
        "id": "winner_value_core",
        "title": "Winner Value",
        "subtitle": "Top bucket YES only",
        "description": "Conservative lane that only backs the model winner when the live ask is still cheap enough.",
        "accent": "#35d4ff",
    },
    {
        "id": "terminal_value_legacy",
        "title": "Legacy Terminal",
        "subtitle": "Current terminal selector",
        "description": "Baseline lane that keeps the old terminal-value behavior for side-by-side comparison.",
        "accent": "#f59e0b",
    },
    {
        "id": "tactical_reprice",
        "title": "Tactical Reprice",
        "subtitle": "Next official catalyst",
        "description": "Short-horizon repricing trades around the next official observation when tactical pressure is strong.",
        "accent": "#22c55e",
    },
    {
        "id": "event_fade_qc",
        "title": "Event Fade QC",
        "subtitle": "Fade overheated moves",
        "description": "Contrarian NO lane that only fades stretched intraday buckets when the next official is supportive.",
        "accent": "#f472b6",
    },
    {
        "id": "micro_value_ensemble",
        "title": "Micro Value",
        "subtitle": "Ensemble counting $1 bets",
        "description": "gopfan2-style micro-bets: compares ensemble member probabilities vs market prices, takes $1 positions on 8%+ edge.",
        "accent": "#a855f7",
    },
)
PAPER_STRATEGY_ORDER: Tuple[str, ...] = tuple(spec["id"] for spec in PAPER_STRATEGY_DEFS)
PAPER_STRATEGY_BY_ID: Dict[str, Dict[str, str]] = {spec["id"]: spec for spec in PAPER_STRATEGY_DEFS}
PAPER_STRATEGY_ID_MAP: Dict[str, str] = {
    "terminal_value": "terminal_value_legacy",
    "terminal_value_legacy": "terminal_value_legacy",
    "winner_value_core": "winner_value_core",
    "tactical_reprice": "tactical_reprice",
    "event_fade_qc": "event_fade_qc",
    "micro_value_ensemble": "micro_value_ensemble",
}


# ---------------------------------------------------------------------------
# Helpers (mirrored from autotrader for independence)
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def _minutes_since(value: Any, reference_utc: Optional[datetime] = None) -> Optional[float]:
    dt = _parse_iso_datetime(value)
    if dt is None:
        return None
    return max(0.0, ((reference_utc or _utc_now()) - dt).total_seconds() / 60.0)


def _opportunity_row_for_label(payload: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
    normalized = normalize_label(label) or label
    for row in ((payload.get("trading") or {}).get("all_terminal_opportunities") or []):
        if normalize_label(str(row.get("label") or "")) == normalized:
            return row
    return None


def _held_side_fair(row: Optional[Dict[str, Any]], side: str, strategy: str) -> Optional[float]:
    if not isinstance(row, dict):
        return None
    side_key = str(side or "YES").upper()
    if strategy == "tactical_reprice":
        return _safe_float(row.get("tactical_yes" if side_key == "YES" else "tactical_no"))
    return _safe_float(row.get("fair_yes" if side_key == "YES" else "fair_no"))


def _held_side_policy(row: Optional[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    source = row.get("tactical_policy") if strategy == "tactical_reprice" else row.get("terminal_policy")
    return dict(source or {})


def _held_side_edge_points(row: Optional[Dict[str, Any]], side: str, strategy: str) -> Optional[float]:
    if not isinstance(row, dict):
        return None
    side_key = str(side or "YES").upper()
    if strategy == "tactical_reprice":
        key = "tactical_edge_yes" if side_key == "YES" else "tactical_edge_no"
    else:
        key = "edge_yes" if side_key == "YES" else "edge_no"
    edge = _safe_float(row.get(key))
    return None if edge is None else edge * 100.0


def _position_risk_now_usd(position: Dict[str, Any]) -> float:
    current_value = _safe_float(position.get("current_value_usd"))
    if current_value is not None and current_value >= 0.0:
        return float(current_value)
    shares_open = float(_safe_float(position.get("shares_open")) or _safe_float(position.get("shares")) or 0.0)
    current_price = _safe_float(position.get("current_price"))
    if current_price is not None and shares_open > 0.0:
        return float(current_price) * shares_open
    return float(_safe_float(position.get("cost_basis_open_usd")) or _safe_float(position.get("notional_usd")) or 0.0)


def _book_from_bracket(bracket: Dict[str, Any], side: str) -> Optional[TopOfBook]:
    if not isinstance(bracket, dict):
        return None
    key = "yes" if str(side or "YES").upper() == "YES" else "no"
    token_id = str((bracket.get("yes_token_id") if key == "yes" else bracket.get("no_token_id")) or "").strip()
    if not token_id:
        return None
    return TopOfBook(
        token_id=token_id,
        best_bid=_safe_float(bracket.get(f"ws_{key}_best_bid")),
        best_ask=_safe_float(bracket.get(f"ws_{key}_best_ask")),
        bid_size=float(_safe_float(bracket.get(f"ws_{key}_bid_depth")) or 0.0),
        ask_size=float(_safe_float(bracket.get(f"ws_{key}_ask_depth")) or 0.0),
        spread=_safe_float(bracket.get(f"ws_{key}_spread")),
        min_order_size=None,
        last_trade_price=None,
        raw=bracket,
    )


def _tail_jsonl(path: str, limit: int = 8) -> List[Dict[str, Any]]:
    try:
        p = Path(path)
        if not p.exists():
            return []
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        result = []
        for line in reversed(lines[-limit * 2:]):
            try:
                result.append(json.loads(line))
            except Exception:
                continue
            if len(result) >= limit:
                break
        return result
    except Exception:
        return []


def _nyc_trading_day() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _normalize_strategy_id(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return PAPER_STRATEGY_ID_MAP.get(raw, "terminal_value_legacy")


def _strategy_position_key(strategy_id: str, base_key: str) -> str:
    return f"{strategy_id}::{base_key}"


def _position_base_key(position: Dict[str, Any], fallback_key: Optional[str] = None) -> str:
    explicit = str(position.get("base_position_key") or "").strip()
    if explicit:
        return explicit
    station_id = str(position.get("station_id") or "").strip().upper()
    target_date = str(position.get("target_date") or "").strip()
    label = str(position.get("label") or "").strip()
    side = str(position.get("side") or "YES").strip().upper()
    if station_id and target_date and label and side:
        return _position_key(station_id, target_date, label, side)
    raw = str(fallback_key or "").strip()
    if "::" in raw:
        return raw.split("::", 1)[1]
    return raw


def _default_strategy_state(
    config: "PaperTraderConfig",
    strategy_id: str,
    *,
    now_iso: Optional[str] = None,
    trading_day: Optional[str] = None,
) -> Dict[str, Any]:
    now_iso = now_iso or _utc_now().isoformat()
    trading_day = trading_day or _nyc_trading_day()
    meta = PAPER_STRATEGY_BY_ID[strategy_id]
    return {
        "strategy_id": strategy_id,
        "title": meta["title"],
        "subtitle": meta["subtitle"],
        "description": meta["description"],
        "paper_equity_usd": config.initial_bankroll_usd,
        "initial_bankroll_usd": config.initial_bankroll_usd,
        "positions": {},
        "trades_today": 0,
        "spent_today_usd": 0.0,
        "gross_buys_today_usd": 0.0,
        "gross_sells_today_usd": 0.0,
        "realized_pnl_today_usd": 0.0,
        "closed_trades_today": 0,
        "open_risk_usd": 0.0,
        "trading_day": trading_day,
        "last_trade_utc": None,
        "total_fills": 0,
        "total_partial_fills": 0,
        "total_rejections": 0,
        "total_slippage_usd": 0.0,
        "total_fill_notional_usd": 0.0,
        "exits_by_reason": {},
        "total_realized_pnl_usd": 0.0,
        "total_entries": 0,
        "total_exits": 0,
        "session_started_utc": now_iso,
    }


def _ensure_strategy_state(
    raw_state: Optional[Dict[str, Any]],
    config: "PaperTraderConfig",
    strategy_id: str,
    *,
    now_iso: Optional[str] = None,
    trading_day: Optional[str] = None,
) -> Dict[str, Any]:
    state = _default_strategy_state(config, strategy_id, now_iso=now_iso, trading_day=trading_day)
    if isinstance(raw_state, dict):
        for key, value in raw_state.items():
            state[key] = value
    if not isinstance(state.get("positions"), dict):
        state["positions"] = {}
    if not isinstance(state.get("exits_by_reason"), dict):
        state["exits_by_reason"] = {}
    if _safe_float(state.get("initial_bankroll_usd")) is None or float(_safe_float(state.get("initial_bankroll_usd")) or 0.0) <= 0.0:
        state["initial_bankroll_usd"] = config.initial_bankroll_usd
    if _safe_float(state.get("paper_equity_usd")) is None:
        state["paper_equity_usd"] = float(state["initial_bankroll_usd"])
    state["strategy_id"] = strategy_id
    meta = PAPER_STRATEGY_BY_ID[strategy_id]
    state["title"] = meta["title"]
    state["subtitle"] = meta["subtitle"]
    state["description"] = meta["description"]
    return state


def _position_record_for_strategy(
    position: Dict[str, Any],
    *,
    strategy_id: Optional[str] = None,
    fallback_key: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    normalized_strategy = _normalize_strategy_id(strategy_id or position.get("strategy_id") or position.get("strategy"))
    base_key = _position_base_key(position, fallback_key=fallback_key)
    full_key = _strategy_position_key(normalized_strategy, base_key)
    normalized = dict(position)
    normalized["strategy_id"] = normalized_strategy
    normalized["strategy"] = normalized_strategy
    normalized["base_position_key"] = base_key
    normalized["position_key"] = full_key
    return normalized_strategy, base_key, normalized


def _repair_position_maps(state: Dict[str, Any], config: "PaperTraderConfig") -> None:
    now_iso = _utc_now().isoformat()
    trading_day = _nyc_trading_day()
    raw_strategies = state.get("strategies") if isinstance(state.get("strategies"), dict) else {}
    strategies = {
        strategy_id: _ensure_strategy_state(raw_strategies.get(strategy_id), config, strategy_id, now_iso=now_iso, trading_day=trading_day)
        for strategy_id in PAPER_STRATEGY_ORDER
    }
    global_positions = state.get("positions") if isinstance(state.get("positions"), dict) else {}
    normalized_global: Dict[str, Dict[str, Any]] = {}
    if global_positions:
        for raw_key, raw_position in global_positions.items():
            if not isinstance(raw_position, dict):
                continue
            strategy_id, base_key, normalized = _position_record_for_strategy(raw_position, fallback_key=str(raw_key))
            full_key = _strategy_position_key(strategy_id, base_key)
            normalized_global[full_key] = normalized
            strategies[strategy_id]["positions"][base_key] = dict(normalized)
    else:
        for strategy_id, strategy_state in strategies.items():
            normalized_positions: Dict[str, Dict[str, Any]] = {}
            for raw_key, raw_position in ((strategy_state.get("positions") or {}).items() if isinstance(strategy_state.get("positions"), dict) else []):
                if not isinstance(raw_position, dict):
                    continue
                _, base_key, normalized = _position_record_for_strategy(
                    raw_position,
                    strategy_id=strategy_id,
                    fallback_key=str(raw_key),
                )
                normalized_positions[base_key] = dict(normalized)
                normalized_global[normalized["position_key"]] = dict(normalized)
            strategy_state["positions"] = normalized_positions
    state["positions"] = normalized_global
    state["strategies"] = strategies


def _refresh_aggregate_state(state: Dict[str, Any], config: "PaperTraderConfig") -> None:
    strategies = state.get("strategies") or {}
    state["schema_version"] = PAPER_STATE_SCHEMA_VERSION
    state["strategy_count"] = len(PAPER_STRATEGY_ORDER)
    state["initial_bankroll_usd"] = config.initial_bankroll_usd
    state["aggregate_initial_bankroll_usd"] = 0.0
    state["paper_equity_usd"] = 0.0
    state["open_risk_usd"] = 0.0
    state["trades_today"] = 0
    state["spent_today_usd"] = 0.0
    state["gross_buys_today_usd"] = 0.0
    state["gross_sells_today_usd"] = 0.0
    state["realized_pnl_today_usd"] = 0.0
    state["closed_trades_today"] = 0
    state["total_fills"] = 0
    state["total_partial_fills"] = 0
    state["total_rejections"] = 0
    state["total_slippage_usd"] = 0.0
    state["total_fill_notional_usd"] = 0.0
    state["total_realized_pnl_usd"] = 0.0
    state["total_entries"] = 0
    state["total_exits"] = 0
    merged_exit_reasons: Dict[str, int] = {}
    last_trade_values: List[str] = []
    session_started_values: List[str] = []
    for strategy_id in PAPER_STRATEGY_ORDER:
        strategy_state = _ensure_strategy_state(strategies.get(strategy_id), config, strategy_id)
        strategy_state["open_risk_usd"] = _recalculate_open_risk(strategy_state.get("positions") or {})
        strategies[strategy_id] = strategy_state
        state["aggregate_initial_bankroll_usd"] += float(_safe_float(strategy_state.get("initial_bankroll_usd")) or 0.0)
        state["paper_equity_usd"] += float(_safe_float(strategy_state.get("paper_equity_usd")) or 0.0)
        state["open_risk_usd"] += float(_safe_float(strategy_state.get("open_risk_usd")) or 0.0)
        state["trades_today"] += int(_safe_float(strategy_state.get("trades_today")) or 0)
        state["spent_today_usd"] += float(_safe_float(strategy_state.get("spent_today_usd")) or 0.0)
        state["gross_buys_today_usd"] += float(_safe_float(strategy_state.get("gross_buys_today_usd")) or 0.0)
        state["gross_sells_today_usd"] += float(_safe_float(strategy_state.get("gross_sells_today_usd")) or 0.0)
        state["realized_pnl_today_usd"] += float(_safe_float(strategy_state.get("realized_pnl_today_usd")) or 0.0)
        state["closed_trades_today"] += int(_safe_float(strategy_state.get("closed_trades_today")) or 0)
        state["total_fills"] += int(_safe_float(strategy_state.get("total_fills")) or 0)
        state["total_partial_fills"] += int(_safe_float(strategy_state.get("total_partial_fills")) or 0)
        state["total_rejections"] += int(_safe_float(strategy_state.get("total_rejections")) or 0)
        state["total_slippage_usd"] += float(_safe_float(strategy_state.get("total_slippage_usd")) or 0.0)
        state["total_fill_notional_usd"] += float(_safe_float(strategy_state.get("total_fill_notional_usd")) or 0.0)
        state["total_realized_pnl_usd"] += float(_safe_float(strategy_state.get("total_realized_pnl_usd")) or 0.0)
        state["total_entries"] += int(_safe_float(strategy_state.get("total_entries")) or 0)
        state["total_exits"] += int(_safe_float(strategy_state.get("total_exits")) or 0)
        for reason, count in (strategy_state.get("exits_by_reason") or {}).items():
            merged_exit_reasons[str(reason)] = merged_exit_reasons.get(str(reason), 0) + int(_safe_float(count) or 0)
        last_trade = str(strategy_state.get("last_trade_utc") or "").strip()
        if last_trade:
            last_trade_values.append(last_trade)
        session_started = str(strategy_state.get("session_started_utc") or "").strip()
        if session_started:
            session_started_values.append(session_started)
    state["strategies"] = strategies
    state["paper_equity_usd"] = round(float(state["paper_equity_usd"]), 4)
    state["aggregate_initial_bankroll_usd"] = round(float(state["aggregate_initial_bankroll_usd"]), 4)
    state["open_risk_usd"] = round(float(state["open_risk_usd"]), 4)
    state["spent_today_usd"] = round(float(state["spent_today_usd"]), 6)
    state["gross_buys_today_usd"] = round(float(state["gross_buys_today_usd"]), 6)
    state["gross_sells_today_usd"] = round(float(state["gross_sells_today_usd"]), 6)
    state["realized_pnl_today_usd"] = round(float(state["realized_pnl_today_usd"]), 6)
    state["total_slippage_usd"] = round(float(state["total_slippage_usd"]), 6)
    state["total_fill_notional_usd"] = round(float(state["total_fill_notional_usd"]), 6)
    state["total_realized_pnl_usd"] = round(float(state["total_realized_pnl_usd"]), 6)
    state["exits_by_reason"] = merged_exit_reasons
    state["trading_day"] = _nyc_trading_day()
    if last_trade_values:
        state["last_trade_utc"] = max(last_trade_values)
    if session_started_values:
        state["session_started_utc"] = min(session_started_values)


def _migrate_legacy_top_level_state(state: Dict[str, Any], config: "PaperTraderConfig") -> None:
    raw_strategies = state.get("strategies")
    if isinstance(raw_strategies, dict) and raw_strategies:
        return
    legacy_state = _default_strategy_state(config, "terminal_value_legacy")
    for key in (
        "paper_equity_usd",
        "initial_bankroll_usd",
        "trades_today",
        "spent_today_usd",
        "gross_buys_today_usd",
        "gross_sells_today_usd",
        "realized_pnl_today_usd",
        "closed_trades_today",
        "open_risk_usd",
        "trading_day",
        "last_trade_utc",
        "total_fills",
        "total_partial_fills",
        "total_rejections",
        "total_slippage_usd",
        "total_fill_notional_usd",
        "exits_by_reason",
        "total_realized_pnl_usd",
        "total_entries",
        "total_exits",
        "session_started_utc",
    ):
        if key in state:
            legacy_state[key] = state.get(key)
    state["strategies"] = {"terminal_value_legacy": legacy_state}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PaperTraderConfig:
    """Configuration for the papertrading shadow runtime."""

    enabled: bool = False
    station_ids: List[str] = field(default_factory=list)
    target_day: int = 0
    target_date: str = ""
    signal_source: str = "internal"
    signal_base_url: str = "http://127.0.0.1:8000"
    loop_seconds: int = 60

    # Virtual capital
    initial_bankroll_usd: float = 100.0

    # Risk parameters (mirror autotrader defaults)
    bankroll_usd: float = 100.0
    fractional_kelly: float = 0.20
    min_edge_points: float = 6.0
    min_fair_probability: float = 0.18
    max_entry_price: float = 0.40
    min_trade_usd: float = 1.0
    max_trade_usd: float = 0.0
    max_total_exposure_usd: float = 25.0
    max_station_exposure_usd: float = 25.0
    daily_loss_limit_usd: float = 10.0
    max_trades_per_day: int = 12
    cooldown_seconds: int = 300
    max_spread_points: float = 8.0
    min_ask_depth_contracts: float = 15.0
    max_depth_participation: float = 0.25
    max_open_positions: int = 12
    max_station_positions: int = 0
    min_signal_confidence: float = 0.35
    min_confidence_budget_multiplier: float = 0.60
    signal_sigma_soft_cap_f: float = 2.0
    signal_sigma_hard_cap_f: float = 4.0
    signal_sigma_min_budget_multiplier: float = 0.50
    distribution_top_probability_soft_floor: float = 0.40
    distribution_top_probability_hard_floor: float = 0.22
    distribution_top_gap_soft_floor: float = 0.08
    distribution_top_gap_hard_floor: float = 0.015
    distribution_min_budget_multiplier: float = 0.60

    # Strategy toggles
    allow_terminal_value: bool = True
    allow_tactical_reprice: bool = True
    tactical_min_edge_points: float = 5.0
    tactical_min_minutes_to_next: float = 8.0
    tactical_max_minutes_to_next: float = 75.0
    secondary_yes_edge_bonus_pts: float = 8.0
    require_buy_recommendation: bool = True

    # Exit thresholds
    terminal_take_profit_points: float = 4.5
    terminal_stop_loss_points: float = 5.0
    tactical_take_profit_points: float = 3.5
    tactical_stop_loss_points: float = 3.0
    tactical_timeout_minutes: int = 75
    exit_edge_buffer_points: float = 1.5

    # Execution
    order_mode: str = "limit_fak"
    limit_slippage_points: float = 1.0
    prefer_ws_book: bool = True
    latency_ms: float = 50.0

    # Paths
    state_path: str = "data/papertrader_state.json"
    runtime_path: str = "data/papertrader_runtime.json"
    log_path: str = "logs/papertrader_trades.jsonl"

    def to_autotrader_config(self) -> AutoTraderConfig:
        """Adapter: convert to AutoTraderConfig for shared decision functions."""
        return AutoTraderConfig(
            enabled=self.enabled,
            mode="paper",
            station_ids=list(self.station_ids),
            target_day=self.target_day,
            target_date=self.target_date,
            signal_source=self.signal_source,
            signal_base_url=self.signal_base_url,
            loop_seconds=self.loop_seconds,
            bankroll_usd=self.bankroll_usd,
            fractional_kelly=self.fractional_kelly,
            min_edge_points=self.min_edge_points,
            min_fair_probability=self.min_fair_probability,
            max_entry_price=self.max_entry_price,
            min_trade_usd=self.min_trade_usd,
            max_trade_usd=self.max_trade_usd,
            max_total_exposure_usd=self.max_total_exposure_usd,
            max_station_exposure_usd=self.max_station_exposure_usd,
            daily_loss_limit_usd=self.daily_loss_limit_usd,
            max_trades_per_day=self.max_trades_per_day,
            cooldown_seconds=self.cooldown_seconds,
            max_spread_points=self.max_spread_points,
            min_ask_depth_contracts=self.min_ask_depth_contracts,
            max_depth_participation=self.max_depth_participation,
            max_open_positions=self.max_open_positions,
            max_station_positions=self.max_station_positions,
            min_signal_confidence=self.min_signal_confidence,
            min_confidence_budget_multiplier=self.min_confidence_budget_multiplier,
            signal_sigma_soft_cap_f=self.signal_sigma_soft_cap_f,
            signal_sigma_hard_cap_f=self.signal_sigma_hard_cap_f,
            signal_sigma_min_budget_multiplier=self.signal_sigma_min_budget_multiplier,
            distribution_top_probability_soft_floor=self.distribution_top_probability_soft_floor,
            distribution_top_probability_hard_floor=self.distribution_top_probability_hard_floor,
            distribution_top_gap_soft_floor=self.distribution_top_gap_soft_floor,
            distribution_top_gap_hard_floor=self.distribution_top_gap_hard_floor,
            distribution_min_budget_multiplier=self.distribution_min_budget_multiplier,
            allow_terminal_value=self.allow_terminal_value,
            allow_tactical_reprice=self.allow_tactical_reprice,
            tactical_min_edge_points=self.tactical_min_edge_points,
            tactical_min_minutes_to_next=self.tactical_min_minutes_to_next,
            tactical_max_minutes_to_next=self.tactical_max_minutes_to_next,
            secondary_yes_edge_bonus_pts=self.secondary_yes_edge_bonus_pts,
            require_buy_recommendation=self.require_buy_recommendation,
            terminal_take_profit_points=self.terminal_take_profit_points,
            terminal_stop_loss_points=self.terminal_stop_loss_points,
            tactical_take_profit_points=self.tactical_take_profit_points,
            tactical_stop_loss_points=self.tactical_stop_loss_points,
            tactical_timeout_minutes=self.tactical_timeout_minutes,
            exit_edge_buffer_points=self.exit_edge_buffer_points,
            order_mode=self.order_mode,
            limit_slippage_points=self.limit_slippage_points,
            prefer_ws_book=self.prefer_ws_book,
            state_path=self.state_path,
            log_path=self.log_path,
        )


def load_papertrader_config() -> PaperTraderConfig:
    """Load config from environment variables."""
    shared = load_autotrader_config_from_env()
    config = PaperTraderConfig(
        station_ids=list(shared.station_ids),
        target_day=shared.target_day,
        target_date=shared.target_date,
        signal_source=shared.signal_source,
        signal_base_url=shared.signal_base_url,
        loop_seconds=shared.loop_seconds,
        bankroll_usd=shared.bankroll_usd,
        fractional_kelly=shared.fractional_kelly,
        min_edge_points=shared.min_edge_points,
        min_fair_probability=shared.min_fair_probability,
        max_entry_price=shared.max_entry_price,
        min_trade_usd=shared.min_trade_usd,
        max_trade_usd=shared.max_trade_usd,
        max_total_exposure_usd=shared.max_total_exposure_usd,
        max_station_exposure_usd=shared.max_station_exposure_usd,
        daily_loss_limit_usd=shared.daily_loss_limit_usd,
        max_trades_per_day=shared.max_trades_per_day,
        cooldown_seconds=shared.cooldown_seconds,
        max_spread_points=shared.max_spread_points,
        min_ask_depth_contracts=shared.min_ask_depth_contracts,
        max_depth_participation=shared.max_depth_participation,
        max_open_positions=shared.max_open_positions,
        max_station_positions=shared.max_station_positions,
        min_signal_confidence=shared.min_signal_confidence,
        min_confidence_budget_multiplier=shared.min_confidence_budget_multiplier,
        signal_sigma_soft_cap_f=shared.signal_sigma_soft_cap_f,
        signal_sigma_hard_cap_f=shared.signal_sigma_hard_cap_f,
        signal_sigma_min_budget_multiplier=shared.signal_sigma_min_budget_multiplier,
        distribution_top_probability_soft_floor=shared.distribution_top_probability_soft_floor,
        distribution_top_probability_hard_floor=shared.distribution_top_probability_hard_floor,
        distribution_top_gap_soft_floor=shared.distribution_top_gap_soft_floor,
        distribution_top_gap_hard_floor=shared.distribution_top_gap_hard_floor,
        distribution_min_budget_multiplier=shared.distribution_min_budget_multiplier,
        allow_terminal_value=shared.allow_terminal_value,
        allow_tactical_reprice=shared.allow_tactical_reprice,
        tactical_min_edge_points=shared.tactical_min_edge_points,
        tactical_min_minutes_to_next=shared.tactical_min_minutes_to_next,
        tactical_max_minutes_to_next=shared.tactical_max_minutes_to_next,
        secondary_yes_edge_bonus_pts=shared.secondary_yes_edge_bonus_pts,
        require_buy_recommendation=shared.require_buy_recommendation,
        terminal_take_profit_points=shared.terminal_take_profit_points,
        terminal_stop_loss_points=shared.terminal_stop_loss_points,
        tactical_take_profit_points=shared.tactical_take_profit_points,
        tactical_stop_loss_points=shared.tactical_stop_loss_points,
        tactical_timeout_minutes=shared.tactical_timeout_minutes,
        exit_edge_buffer_points=shared.exit_edge_buffer_points,
        order_mode=shared.order_mode,
        limit_slippage_points=shared.limit_slippage_points,
        prefer_ws_book=shared.prefer_ws_book,
    )
    config.enabled = os.environ.get("HELIOS_PAPERTRADER_ENABLED", "").strip().lower() in ("1", "true", "yes")
    raw_stations = os.environ.get("HELIOS_PAPERTRADER_STATIONS", "")
    if raw_stations.strip():
        config.station_ids = [s.strip().upper() for s in raw_stations.split(",") if s.strip()]
    raw_bankroll = os.environ.get("HELIOS_PAPERTRADER_BANKROLL_USD", "")
    if raw_bankroll.strip():
        config.initial_bankroll_usd = float(raw_bankroll)
        config.bankroll_usd = config.initial_bankroll_usd
    return config


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _default_paper_state(config: PaperTraderConfig) -> Dict[str, Any]:
    now_iso = _utc_now().isoformat()
    nyc_today = _nyc_trading_day()
    state = {
        "schema_version": PAPER_STATE_SCHEMA_VERSION,
        "paper_equity_usd": config.initial_bankroll_usd * len(PAPER_STRATEGY_ORDER),
        "initial_bankroll_usd": config.initial_bankroll_usd,
        "aggregate_initial_bankroll_usd": config.initial_bankroll_usd * len(PAPER_STRATEGY_ORDER),
        "positions": {},
        "pending_orders": {},
        "strategies": {
            strategy_id: _default_strategy_state(config, strategy_id, now_iso=now_iso, trading_day=nyc_today)
            for strategy_id in PAPER_STRATEGY_ORDER
        },
        "trades_today": 0,
        "spent_today_usd": 0.0,
        "gross_buys_today_usd": 0.0,
        "gross_sells_today_usd": 0.0,
        "realized_pnl_today_usd": 0.0,
        "closed_trades_today": 0,
        "open_risk_usd": 0.0,
        "trading_day": nyc_today,
        "last_trade_utc": None,
        "total_fills": 0,
        "total_partial_fills": 0,
        "total_rejections": 0,
        "total_slippage_usd": 0.0,
        "total_fill_notional_usd": 0.0,
        "exits_by_reason": {},
        "total_realized_pnl_usd": 0.0,
        "total_entries": 0,
        "total_exits": 0,
        "session_started_utc": now_iso,
    }
    _refresh_aggregate_state(state, config)
    return state


def _paper_session_has_activity(state: Dict[str, Any]) -> bool:
    positions = state.get("positions") or {}
    if isinstance(positions, dict) and positions:
        return True

    strategies = state.get("strategies") or {}
    if isinstance(strategies, dict):
        for strategy_state in strategies.values():
            if not isinstance(strategy_state, dict):
                continue
            if strategy_state.get("positions"):
                return True
            strategy_equity = float(_safe_float(strategy_state.get("paper_equity_usd")) or 0.0)
            strategy_initial = float(_safe_float(strategy_state.get("initial_bankroll_usd")) or 0.0)
            if abs(strategy_equity - strategy_initial) > 1e-9:
                return True
            for key in (
                "trades_today",
                "gross_buys_today_usd",
                "gross_sells_today_usd",
                "realized_pnl_today_usd",
                "open_risk_usd",
                "total_fills",
                "total_partial_fills",
                "total_rejections",
                "total_slippage_usd",
                "total_fill_notional_usd",
                "total_realized_pnl_usd",
                "total_entries",
                "total_exits",
            ):
                if abs(float(_safe_float(strategy_state.get(key)) or 0.0)) > 1e-9:
                    return True

    counters = (
        "trades_today",
        "gross_buys_today_usd",
        "gross_sells_today_usd",
        "realized_pnl_today_usd",
        "open_risk_usd",
        "total_fills",
        "total_partial_fills",
        "total_rejections",
        "total_slippage_usd",
        "total_fill_notional_usd",
        "total_realized_pnl_usd",
        "total_entries",
        "total_exits",
    )
    for key in counters:
        if abs(float(_safe_float(state.get(key)) or 0.0)) > 1e-9:
            return True

    paper_equity = float(_safe_float(state.get("paper_equity_usd")) or 0.0)
    initial_bankroll = float(
        _safe_float(state.get("aggregate_initial_bankroll_usd"))
        or _safe_float(state.get("initial_bankroll_usd"))
        or 0.0
    )
    return abs(paper_equity - initial_bankroll) > 1e-9


def load_paper_state(path: str, config: PaperTraderConfig) -> Dict[str, Any]:
    try:
        p = Path(path)
        if p.exists():
            state = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(state, dict):
                defaults = _default_paper_state(config)
                for key, value in defaults.items():
                    state.setdefault(key, value)
                if not isinstance(state.get("positions"), dict):
                    state["positions"] = {}
                if not isinstance(state.get("pending_orders"), dict):
                    state["pending_orders"] = {}
                if not isinstance(state.get("exits_by_reason"), dict):
                    state["exits_by_reason"] = {}
                _migrate_legacy_top_level_state(state, config)
                _repair_position_maps(state, config)
                if not _paper_session_has_activity(state):
                    state["initial_bankroll_usd"] = config.initial_bankroll_usd
                    for strategy_id in PAPER_STRATEGY_ORDER:
                        state["strategies"][strategy_id] = _default_strategy_state(config, strategy_id)
                _refresh_aggregate_state(state, config)
                return state
    except Exception as exc:
        logger.warning("Failed to load paper state from %s: %s", path, exc)
    return _default_paper_state(config)


def save_paper_state(path: str, state: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------------------
# PaperTrader runtime
# ---------------------------------------------------------------------------


class PaperTrader:
    """Live papertrading shadow runtime.

    Runs the same decision logic as the autotrader but executes paper orders
    against live L2 WebSocket orderbook data. Never sends real orders.
    """

    def __init__(self, config: Optional[PaperTraderConfig] = None):
        self.config = config or PaperTraderConfig()
        self.execution = PaperExecutionEngine(latency_ms=self.config.latency_ms)
        self.state = load_paper_state(self.config.state_path, self.config)
        self._sync_config_bankroll_with_state()
        self._recent_events: List[Dict[str, Any]] = []

    def _atc(self) -> AutoTraderConfig:
        """Shortcut: get adapted AutoTraderConfig."""
        return self.config.to_autotrader_config()

    # -- State persistence --

    def _save_state(self) -> None:
        save_paper_state(self.config.state_path, self.state)

    def _sync_config_bankroll_with_state(self) -> None:
        first_strategy = self._strategy_state(PAPER_STRATEGY_ORDER[0])
        state_initial = float(
            _safe_float(first_strategy.get("initial_bankroll_usd"))
            or _safe_float(self.state.get("initial_bankroll_usd"))
            or self.config.initial_bankroll_usd
        )
        self.config.initial_bankroll_usd = state_initial
        self.config.bankroll_usd = state_initial

    def _strategy_state(self, strategy_id: str) -> Dict[str, Any]:
        normalized = _normalize_strategy_id(strategy_id)
        strategies = self.state.setdefault("strategies", {})
        if not isinstance(strategies.get(normalized), dict):
            strategies[normalized] = _default_strategy_state(self.config, normalized)
        strategy_state = strategies[normalized]
        if not isinstance(strategy_state.get("positions"), dict):
            strategy_state["positions"] = {}
        if not isinstance(strategy_state.get("exits_by_reason"), dict):
            strategy_state["exits_by_reason"] = {}
        return strategy_state

    def _refresh_state_totals(self) -> None:
        _refresh_aggregate_state(self.state, self.config)

    def _resolve_position_keys(self, position_key: str) -> Tuple[str, str, str]:
        raw = str(position_key or "").strip()
        positions = self.state.get("positions") or {}
        if raw in positions:
            position = dict(positions.get(raw) or {})
            strategy_id = _normalize_strategy_id(position.get("strategy_id") or position.get("strategy"))
            base_key = str(position.get("base_position_key") or _position_base_key(position, fallback_key=raw))
            return raw, strategy_id, base_key
        if "::" in raw:
            strategy_id, base_key = raw.split("::", 1)
            return raw, _normalize_strategy_id(strategy_id), base_key
        matches = []
        for full_key, position in positions.items():
            if not isinstance(position, dict):
                continue
            base_key = str(position.get("base_position_key") or _position_base_key(position, fallback_key=full_key))
            if base_key == raw or str(full_key).endswith(f"::{raw}"):
                matches.append((str(full_key), dict(position)))
        if len(matches) == 1:
            full_key, position = matches[0]
            strategy_id = _normalize_strategy_id(position.get("strategy_id") or position.get("strategy"))
            base_key = str(position.get("base_position_key") or _position_base_key(position, fallback_key=full_key))
            return full_key, strategy_id, base_key
        return raw, _normalize_strategy_id(None), raw

    def _store_position_record(self, position: Dict[str, Any], *, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        normalized_strategy, base_key, normalized = _position_record_for_strategy(position, strategy_id=strategy_id)
        full_key = normalized["position_key"]
        self.state.setdefault("positions", {})[full_key] = dict(normalized)
        strategy_state = self._strategy_state(normalized_strategy)
        strategy_state.setdefault("positions", {})[base_key] = dict(normalized)
        return normalized

    def _record_fill_metrics(self, fill: Any, strategy_id: Optional[str] = None) -> None:
        filled_size = float(_safe_float(getattr(fill, "filled_size", None)) or 0.0)
        if filled_size <= 0.0:
            return
        target = self._strategy_state(strategy_id) if strategy_id else self.state
        target["total_fills"] = int(_safe_float(target.get("total_fills")) or 0) + 1
        if str(getattr(fill, "status", "") or "").upper() == "PARTIAL":
            target["total_partial_fills"] = int(_safe_float(target.get("total_partial_fills")) or 0) + 1
        slippage_usd = float(_safe_float(getattr(fill, "slippage_vs_best", None)) or 0.0) * filled_size
        target["total_slippage_usd"] = round(
            float(_safe_float(target.get("total_slippage_usd")) or 0.0) + slippage_usd,
            6,
        )
        target["total_fill_notional_usd"] = round(
            float(_safe_float(target.get("total_fill_notional_usd")) or 0.0)
            + float(_safe_float(getattr(fill, "notional_usd", None)) or 0.0),
            6,
        )
        if strategy_id:
            self._refresh_state_totals()

    def _record_rejection(self, strategy_id: Optional[str] = None) -> None:
        target = self._strategy_state(strategy_id) if strategy_id else self.state
        target["total_rejections"] = int(_safe_float(target.get("total_rejections")) or 0) + 1
        if strategy_id:
            self._refresh_state_totals()

    def _append_journal(self, event: Dict[str, Any]) -> None:
        path = Path(self.config.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(event)
        record.setdefault("ts_utc", _utc_now().isoformat())
        line = (json.dumps(record, ensure_ascii=True, default=str) + "\n").encode("utf-8")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        with _LOG_WRITE_LOCK:
            fd = os.open(str(path), flags, 0o666)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        self._recent_events.insert(0, record)
        self._recent_events = self._recent_events[:20]

    def _clear_journal(self) -> None:
        path = Path(self.config.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_WRITE_LOCK:
            path.write_text("", encoding="utf-8")

    # -- Daily reset --

    def _reset_daily_counters_if_needed(self) -> None:
        nyc_today = _nyc_trading_day()
        needs_reset = self.state.get("trading_day") != nyc_today
        for strategy_id in PAPER_STRATEGY_ORDER:
            strategy_state = self._strategy_state(strategy_id)
            if strategy_state.get("trading_day") != nyc_today:
                needs_reset = True
                strategy_state["trading_day"] = nyc_today
                strategy_state["trades_today"] = 0
                strategy_state["spent_today_usd"] = 0.0
                strategy_state["gross_buys_today_usd"] = 0.0
                strategy_state["gross_sells_today_usd"] = 0.0
                strategy_state["realized_pnl_today_usd"] = 0.0
                strategy_state["closed_trades_today"] = 0
        if needs_reset:
            self.state["trading_day"] = nyc_today
            self._refresh_state_totals()

    def reset_session(self) -> Dict[str, Any]:
        """Reset every paper lane back to a clean session."""
        self.config.enabled = False
        self.state = _default_paper_state(self.config)
        self._sync_config_bankroll_with_state()
        self._recent_events = []
        self._save_state()
        self._clear_journal()
        return dict(self.state)

    # -- Signal fetching --

    async def _fetch_signal_payload(
        self,
        station_id: str,
        target_day: int,
        *,
        target_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.config.signal_source == "api":
            url = f"{self.config.signal_base_url.rstrip('/')}/api/trade/{station_id}"
            async with httpx.AsyncClient(timeout=20.0) as client:
                params: Dict[str, Any] = {"target_day": target_day, "depth": 5}
                if target_date:
                    params["target_date"] = target_date
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()

        from web_server import get_polymarket_dashboard_data

        return await get_polymarket_dashboard_data(
            station_id,
            target_day=target_day,
            target_date=target_date,
            depth=5,
        )

    # -- L2 book access --

    def _get_live_book_from_payload(self, payload: Dict[str, Any], label: str, side: str) -> Optional[TopOfBook]:
        """Extract TopOfBook from the signal payload brackets (WS enriched)."""
        for row in (payload.get("brackets") or []):
            row_label = normalize_label(str(row.get("name") or row.get("bracket") or ""))
            if row_label == normalize_label(label):
                return _book_from_bracket(row, side)
        return None

    def _get_l2_from_payload(self, payload: Dict[str, Any], label: str, side: str) -> Dict[str, Any]:
        """Extract L2 depth data from payload brackets for fill simulation."""
        key = "yes" if str(side or "YES").upper() == "YES" else "no"
        for row in (payload.get("brackets") or []):
            row_label = normalize_label(str(row.get("name") or row.get("bracket") or ""))
            if row_label == normalize_label(label):
                return {
                    "best_bid": _safe_float(row.get(f"ws_{key}_best_bid")),
                    "best_ask": _safe_float(row.get(f"ws_{key}_best_ask")),
                    "mid": _safe_float(row.get(f"ws_{key}_mid")),
                    "spread": _safe_float(row.get(f"ws_{key}_spread")),
                    "bid_depth": float(_safe_float(row.get(f"ws_{key}_bid_depth")) or 0.0),
                    "ask_depth": float(_safe_float(row.get(f"ws_{key}_ask_depth")) or 0.0),
                    "bids": list(row.get(f"ws_{key}_bids") or []),
                    "asks": list(row.get(f"ws_{key}_asks") or []),
                }
        return {"bids": [], "asks": [], "best_bid": None, "best_ask": None}

    def _terminal_opportunities(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = list(((payload.get("trading") or {}).get("all_terminal_opportunities") or []))
        rows.sort(
            key=lambda row: (
                float(_safe_float(row.get("policy_score")) or -1.0),
                float(_safe_float(row.get("score")) or -1.0),
            ),
            reverse=True,
        )
        return [dict(row) for row in rows if isinstance(row, dict)]

    def _build_strategy_candidate_from_row(
        self,
        payload: Dict[str, Any],
        strategy_state: Dict[str, Any],
        strategy_id: str,
        trade_row: Dict[str, Any],
    ) -> Tuple[Optional[AutoTradeCandidate], List[str]]:
        candidate, reasons = _build_candidate_from_trade(
            payload,
            self._atc(),
            _normalize_state(strategy_state),
            trade_row=dict(trade_row or {}),
            strategy=strategy_id,
        )
        if candidate is None:
            return None, reasons
        base_key = candidate.position_key
        full_key = _strategy_position_key(strategy_id, base_key)
        candidate.position_key = full_key
        candidate.strategy = strategy_id
        candidate.raw_row = dict(candidate.raw_row or {})
        candidate.raw_row["strategy_lane"] = strategy_id
        return candidate, reasons

    def _select_winner_value_candidate(
        self,
        payload: Dict[str, Any],
        strategy_state: Dict[str, Any],
    ) -> Tuple[Optional[AutoTradeCandidate], List[str]]:
        trading = payload.get("trading") or {}
        forecast = trading.get("forecast_winner") or {}
        forecast_label = normalize_label(str(forecast.get("label") or ""))
        if not forecast_label:
            return None, ["missing_forecast_winner"]
        opportunities = self._terminal_opportunities(payload)
        row = next(
            (
                dict(item)
                for item in opportunities
                if normalize_label(str(item.get("label") or "")) == forecast_label
                and str(item.get("best_side") or "").upper() == "YES"
                and bool((item.get("terminal_policy") or {}).get("allowed"))
            ),
            None,
        )
        if not row:
            return None, ["winner_lane_no_yes_candidate"]
        return self._build_strategy_candidate_from_row(payload, strategy_state, "winner_value_core", row)

    def _select_terminal_legacy_candidate(
        self,
        payload: Dict[str, Any],
        strategy_state: Dict[str, Any],
    ) -> Tuple[Optional[AutoTradeCandidate], List[str]]:
        best_terminal = dict(((payload.get("trading") or {}).get("best_terminal_trade") or {}))
        if not best_terminal:
            return None, ["missing_best_terminal_trade"]
        return self._build_strategy_candidate_from_row(payload, strategy_state, "terminal_value_legacy", best_terminal)

    def _select_tactical_candidate(
        self,
        payload: Dict[str, Any],
        strategy_state: Dict[str, Any],
    ) -> Tuple[Optional[AutoTradeCandidate], List[str]]:
        best_tactical = dict(((payload.get("trading") or {}).get("best_tactical_trade") or {}))
        if not best_tactical:
            return None, ["missing_best_tactical_trade"]
        return self._build_strategy_candidate_from_row(payload, strategy_state, "tactical_reprice", best_tactical)

    def _select_event_fade_candidate(
        self,
        payload: Dict[str, Any],
        strategy_state: Dict[str, Any],
    ) -> Tuple[Optional[AutoTradeCandidate], List[str]]:
        trading = payload.get("trading") or {}
        next_projection = ((trading.get("tactical_context") or {}).get("next_metar") or {})
        next_direction = str(next_projection.get("direction") or "FLAT").upper()
        repricing_influence = float(_safe_float((trading.get("tactical_context") or {}).get("repricing_influence")) or 0.0)
        if int(payload.get("target_day") or self.config.target_day) != 0:
            return None, ["event_fade_intraday_only"]
        if next_direction != "DOWN":
            return None, ["no_downside_event_pressure"]
        if repricing_influence < 0.18:
            return None, ["repricing_quality_too_low"]
        opportunities = self._terminal_opportunities(payload)
        row = next(
            (
                dict(item)
                for item in opportunities
                if str(item.get("best_side") or "").upper() == "NO"
                and bool((item.get("terminal_policy") or {}).get("allowed"))
                and (item.get("distance_from_top") is None or int(item.get("distance_from_top") or 0) <= 1)
                and float(_safe_float(item.get("best_edge")) or 0.0) >= 0.05
                and float(_safe_float(item.get("no_entry")) or 1.0) <= 0.55
            ),
            None,
        )
        if not row:
            return None, ["event_fade_no_candidate"]
        return self._build_strategy_candidate_from_row(payload, strategy_state, "event_fade_qc", row)

    def _select_micro_value_candidate(
        self,
        payload: Dict[str, Any],
        strategy_state: Dict[str, Any],
    ) -> Tuple[Optional[AutoTradeCandidate], List[str]]:
        """Ensemble counting micro-value strategy (gopfan2-style).

        Compares ensemble member probabilities against market prices.
        Takes the bucket with the highest edge above threshold.
        Fixed $1 positions, bypassing complex Kelly sizing.
        """
        ensemble_snap = getattr(self, "_current_ensemble", None)
        if not ensemble_snap or not ensemble_snap.bucket_probabilities:
            return None, ["no_ensemble_data"]

        station_id = str(payload.get("station_id") or "")
        target_date = str(payload.get("target_date") or "")
        opportunities = self._terminal_opportunities(payload)
        if not opportunities:
            return None, ["no_opportunities"]

        min_edge = 0.08  # 8% minimum edge

        best_candidate = None
        best_edge = 0.0

        for opp in opportunities:
            label = normalize_label(str(opp.get("label") or ""))
            if not label:
                continue

            # Get ensemble probability for this bucket
            ensemble_prob = ensemble_snap.get_bucket_prob(label)
            if ensemble_prob is None:
                continue

            # Get market price (best ask for YES side)
            yes_entry = _safe_float(opp.get("selected_entry") or opp.get("yes_entry"))
            no_entry = _safe_float(opp.get("no_entry"))

            # Determine best side: BUY YES if ensemble > market, or BUY NO if ensemble < market
            if yes_entry is not None and yes_entry > 0:
                yes_edge = ensemble_prob - yes_entry
                if yes_edge > min_edge and yes_edge > best_edge and yes_entry <= 0.40:
                    # Find bracket for token_id
                    bracket = None
                    for row in (payload.get("brackets") or []):
                        if normalize_label(str(row.get("name") or row.get("bracket") or "")) == label:
                            bracket = row
                            break

                    token_id = str((bracket or {}).get("yes_token_id") or "") if bracket else ""
                    position_key = _position_key(station_id, target_date, label, "YES")
                    full_key = _strategy_position_key("micro_value_ensemble", position_key)

                    # Check for duplicate position
                    positions = strategy_state.get("positions") or {}
                    if full_key in positions and str((positions[full_key] or {}).get("status") or "OPEN").upper() == "OPEN":
                        continue

                    best_edge = yes_edge
                    best_candidate = AutoTradeCandidate(
                        station_id=station_id,
                        target_date=target_date,
                        target_day=int(payload.get("target_day") or 0),
                        label=label,
                        side="YES",
                        token_id=token_id,
                        market_price=yes_entry,
                        fair_price=ensemble_prob,
                        edge_points=round(yes_edge * 100, 2),
                        model_probability=ensemble_prob,
                        recommendation=f"BUY_YES_ENSEMBLE({ensemble_prob:.0%}vs{yes_entry:.0%})",
                        policy_reason=f"ensemble_edge_{yes_edge:.0%}",
                        forecast_winner_label=None,
                        forecast_edge_points=None,
                        tactical_alignment="neutral",
                        why=f"Ensemble {ensemble_prob:.1%} vs market {yes_entry:.1%} = +{yes_edge:.1%} edge",
                        position_key=full_key,
                        strategy="micro_value_ensemble",
                        strategy_score=yes_edge * 100,
                        signal_confidence=min(1.0, ensemble_snap.total_members / 50.0),
                        signal_sigma_f=None,
                        distribution_top_probability=ensemble_prob,
                        distribution_top_gap=yes_edge,
                        bucket_rank=None,
                        opened_next_obs_utc=None,
                        raw_row=dict(opp),
                    )

            # Also check NO side: if ensemble says low prob but market is high
            if no_entry is not None and no_entry > 0:
                no_edge = (1.0 - ensemble_prob) - no_entry
                if no_edge > min_edge and no_edge > best_edge and no_entry <= 0.50:
                    bracket = None
                    for row in (payload.get("brackets") or []):
                        if normalize_label(str(row.get("name") or row.get("bracket") or "")) == label:
                            bracket = row
                            break

                    token_id = str((bracket or {}).get("no_token_id") or "") if bracket else ""
                    position_key = _position_key(station_id, target_date, label, "NO")
                    full_key = _strategy_position_key("micro_value_ensemble", position_key)

                    positions = strategy_state.get("positions") or {}
                    if full_key in positions and str((positions[full_key] or {}).get("status") or "OPEN").upper() == "OPEN":
                        continue

                    best_edge = no_edge
                    best_candidate = AutoTradeCandidate(
                        station_id=station_id,
                        target_date=target_date,
                        target_day=int(payload.get("target_day") or 0),
                        label=label,
                        side="NO",
                        token_id=token_id,
                        market_price=no_entry,
                        fair_price=1.0 - ensemble_prob,
                        edge_points=round(no_edge * 100, 2),
                        model_probability=1.0 - ensemble_prob,
                        recommendation=f"BUY_NO_ENSEMBLE({1.0-ensemble_prob:.0%}vs{no_entry:.0%})",
                        policy_reason=f"ensemble_edge_{no_edge:.0%}",
                        forecast_winner_label=None,
                        forecast_edge_points=None,
                        tactical_alignment="neutral",
                        why=f"Ensemble NO {1.0-ensemble_prob:.1%} vs market {no_entry:.1%} = +{no_edge:.1%} edge",
                        position_key=full_key,
                        strategy="micro_value_ensemble",
                        strategy_score=no_edge * 100,
                        signal_confidence=min(1.0, ensemble_snap.total_members / 50.0),
                        signal_sigma_f=None,
                        distribution_top_probability=1.0 - ensemble_prob,
                        distribution_top_gap=no_edge,
                        bucket_rank=None,
                        opened_next_obs_utc=None,
                        raw_row=dict(opp),
                    )

        if best_candidate is None:
            return None, ["no_ensemble_edge_above_threshold"]

        return best_candidate, []

    def _evaluate_strategy_candidate(
        self,
        strategy_id: str,
        payload: Dict[str, Any],
    ) -> Tuple[Optional[AutoTradeCandidate], List[str]]:
        strategy_state = self._strategy_state(strategy_id)
        if strategy_id == "winner_value_core":
            return self._select_winner_value_candidate(payload, strategy_state)
        if strategy_id == "terminal_value_legacy":
            return self._select_terminal_legacy_candidate(payload, strategy_state)
        if strategy_id == "tactical_reprice":
            return self._select_tactical_candidate(payload, strategy_state)
        if strategy_id == "event_fade_qc":
            return self._select_event_fade_candidate(payload, strategy_state)
        if strategy_id == "micro_value_ensemble":
            return self._select_micro_value_candidate(payload, strategy_state)
        return None, ["unknown_strategy"]

    # -- Exit decision (replicated from AutoTrader._evaluate_exit_decision) --

    def _evaluate_exit_decision(
        self,
        position: Dict[str, Any],
        payload: Dict[str, Any],
        book: Optional[TopOfBook],
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        if not isinstance(position, dict) or str(position.get("status") or "OPEN").upper() != "OPEN":
            return None, ["position_not_open"]
        if book is None or _safe_float(book.best_bid) is None:
            return None, ["no_live_bid"]

        strategy = str(position.get("strategy") or "terminal_value")
        side = str(position.get("side") or "YES").upper()
        entry_price = float(_safe_float(position.get("entry_price")) or 0.0)
        shares_open = float(_safe_float(position.get("shares_open")) or _safe_float(position.get("shares")) or 0.0)
        if shares_open <= 0:
            return None, ["no_open_shares"]

        held_minutes = _minutes_since(position.get("opened_at_utc"))
        opportunity_row = _opportunity_row_for_label(payload, str(position.get("label") or ""))
        fair_now = _held_side_fair(opportunity_row, side, strategy)
        policy_now = _held_side_policy(opportunity_row, strategy)
        edge_now_points = _held_side_edge_points(opportunity_row, side, strategy)
        trading = (payload.get("trading") or {})
        next_projection = ((trading.get("tactical_context") or {}).get("next_metar") or {})
        next_direction = str(next_projection.get("direction") or "FLAT").upper()
        next_obs_utc = _parse_iso_datetime(next_projection.get("next_obs_utc"))
        opened_next_obs_utc = _parse_iso_datetime(position.get("opened_next_obs_utc"))
        current_best_bid = float(_safe_float(book.best_bid) or 0.0)
        next_obs_slip_minutes = None
        if opened_next_obs_utc is not None and next_obs_utc is not None:
            next_obs_slip_minutes = (next_obs_utc - opened_next_obs_utc).total_seconds() / 60.0

        buffer_points = float(self.config.exit_edge_buffer_points) / 100.0
        take_profit_points = (
            float(self.config.tactical_take_profit_points) if strategy == "tactical_reprice"
            else float(self.config.terminal_take_profit_points)
        ) / 100.0
        stop_loss_points = (
            float(self.config.tactical_stop_loss_points) if strategy == "tactical_reprice"
            else float(self.config.terminal_stop_loss_points)
        ) / 100.0

        target_floor = entry_price + take_profit_points
        if fair_now is not None and fair_now > 0:
            target_floor = min(target_floor, max(0.01, fair_now - buffer_points))
        stop_floor = max(0.01, entry_price - stop_loss_points)

        force_exit = False
        reason = None
        reasons: List[str] = []
        fair_value_absorbed = bool(
            fair_now is not None
            and current_best_bid >= max(0.01, float(fair_now) - buffer_points)
        )

        if (
            fair_now is not None
            and fair_value_absorbed
            and current_best_bid >= target_floor
            and current_best_bid > entry_price + 0.0025
        ):
            reason = "take_profit"
        elif strategy == "tactical_reprice":
            if held_minutes is not None and held_minutes >= float(self.config.tactical_timeout_minutes):
                force_exit = True
                reason = "timeout_exit"
            elif opened_next_obs_utc is not None and next_obs_utc is not None and next_obs_utc > (opened_next_obs_utc + timedelta(minutes=5)):
                force_exit = True
                reason = "post_official_reprice"
            elif not policy_now.get("allowed", True):
                force_exit = True
                reason = "signal_flip"
            elif (side == "YES" and next_direction == "DOWN") or (side == "NO" and next_direction == "UP"):
                force_exit = True
                reason = "next_official_against_position"
            elif current_best_bid <= stop_floor and held_minutes is not None and held_minutes >= 2.0:
                force_exit = True
                reason = "stop_loss"
        else:
            if fair_now is not None and fair_now < entry_price - buffer_points:
                force_exit = True
                reason = "model_broke"
            elif current_best_bid <= stop_floor and edge_now_points is not None and edge_now_points <= 0.0:
                force_exit = True
                reason = "stop_loss"

        if not reason:
            return None, ["hold"]

        exit_plan, exit_reasons = apply_exit_guardrails(
            position,
            book,
            self._atc(),
            target_floor_price=target_floor,
            force_exit=force_exit,
        )
        if not exit_plan:
            return None, exit_reasons or [reason]

        exit_plan["reason"] = reason
        exit_plan["strategy"] = strategy
        exit_plan["target_floor"] = round(float(target_floor), 6)
        exit_plan["stop_floor"] = round(float(stop_floor), 6)
        exit_plan["fair_now"] = round(float(fair_now), 6) if fair_now is not None else None
        exit_plan["edge_now_points"] = round(float(edge_now_points), 4) if edge_now_points is not None else None
        exit_plan["decision_context"] = {
            "held_minutes": round(float(held_minutes), 3) if held_minutes is not None else None,
            "entry_price": round(float(entry_price), 6),
            "current_best_bid": round(float(current_best_bid), 6),
            "force_exit": force_exit,
        }
        return exit_plan, reasons

    # -- Position management --

    async def _manage_open_positions(self, station_id: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        exit_events = []
        positions = dict(self.state.get("positions") or {})
        for pkey, position in list(positions.items()):
            if not isinstance(position, dict):
                continue
            if str(position.get("status") or "").upper() != "OPEN":
                continue
            if str(position.get("station_id") or "").upper() != station_id.upper():
                continue

            label = str(position.get("label") or "")
            side = str(position.get("side") or "YES")
            book = self._get_live_book_from_payload(payload, label, side)

            exit_plan, exit_reasons = self._evaluate_exit_decision(position, payload, book)
            if not exit_plan:
                continue

            # Execute paper sell
            l2 = self._get_l2_from_payload(payload, label, side)
            min_price = float(_safe_float(exit_plan.get("min_price")) or 0.0)
            size = float(_safe_float(exit_plan.get("size")) or 0.0)

            fill = self.execution.execute_limit_fak_sell(
                token_id=str(position.get("token_id") or ""),
                min_price=min_price,
                size=size,
                book_snapshot=l2,
            )

            strategy_id = _normalize_strategy_id(position.get("strategy_id") or position.get("strategy"))
            if fill.filled_size > 0:
                persisted = self._persist_exit(pkey, exit_plan, fill)
                self._record_fill_metrics(fill, strategy_id=strategy_id)
                event = {
                    "event": "exit_fill",
                    "station_id": station_id,
                    "strategy_id": strategy_id,
                    "label": label,
                    "side": side,
                    "entry_price": float(_safe_float(position.get("entry_price")) or 0.0),
                    "position_key": pkey,
                    "reason": exit_plan.get("reason"),
                    "fill": fill.to_dict(),
                    "persisted": persisted,
                }
                self._append_journal(event)
                exit_events.append(event)
            else:
                self._record_rejection(strategy_id=strategy_id)
                event = {
                    "event": "exit_rejected",
                    "station_id": station_id,
                    "strategy_id": strategy_id,
                    "label": label,
                    "side": side,
                    "entry_price": float(_safe_float(position.get("entry_price")) or 0.0),
                    "position_key": pkey,
                    "reason": exit_plan.get("reason"),
                    "fill_status": fill.status,
                }
                self._append_journal(event)

        return exit_events

    def _persist_trade(self, candidate: AutoTradeCandidate, plan: Dict[str, Any], fill) -> None:
        """Persist a new paper position from a filled entry order."""
        now_iso = _utc_now().isoformat()
        notional = fill.notional_usd if fill.notional_usd > 0 else float(plan.get("notional_usd") or 0.0)
        entry_price = fill.avg_price if fill.avg_price > 0 else float(plan.get("limit_price") or 0.0)
        shares = fill.filled_size if fill.filled_size > 0 else float(plan.get("size") or 0.0)
        strategy_id = _normalize_strategy_id(candidate.strategy)
        strategy_state = self._strategy_state(strategy_id)
        raw_position_key = str(candidate.position_key or "").strip()
        base_key = raw_position_key.split("::", 1)[1] if "::" in raw_position_key else raw_position_key
        full_key = _strategy_position_key(strategy_id, base_key)

        position = {
            "station_id": candidate.station_id,
            "target_date": candidate.target_date,
            "label": candidate.label,
            "side": candidate.side,
            "token_id": candidate.token_id,
            "status": "OPEN",
            "mode": "paper_shadow",
            "strategy": strategy_id,
            "strategy_id": strategy_id,
            "managed_by_bot": True,
            "source": "papertrader",
            "strategy_score": round(float(candidate.strategy_score), 6),
            "entry_price": round(entry_price, 6),
            "avg_price": round(entry_price, 6),
            "shares": round(shares, 6),
            "shares_open": round(shares, 6),
            "notional_usd": round(notional, 4),
            "cost_basis_open_usd": round(notional, 4),
            "current_value_usd": round(notional, 4),
            "opened_at_utc": now_iso,
            "opened_next_obs_utc": candidate.opened_next_obs_utc,
            "thesis": candidate.why,
            "realized_pnl_usd": 0.0,
            "unrealized_pnl_usd": 0.0,
            "fills": [],
            "base_position_key": base_key,
            "position_key": full_key,
        }
        self._store_position_record(position, strategy_id=strategy_id)

        gross = float(_safe_float(strategy_state.get("gross_buys_today_usd")) or 0.0)
        strategy_state["gross_buys_today_usd"] = round(gross + notional, 6)
        strategy_state["spent_today_usd"] = strategy_state["gross_buys_today_usd"]
        strategy_state["trades_today"] = int(_safe_float(strategy_state.get("trades_today")) or 0) + 1
        strategy_state["total_entries"] = int(_safe_float(strategy_state.get("total_entries")) or 0) + 1
        strategy_state["last_trade_utc"] = now_iso
        strategy_state["paper_equity_usd"] = round(
            float(_safe_float(strategy_state.get("paper_equity_usd")) or strategy_state.get("initial_bankroll_usd") or self.config.initial_bankroll_usd)
            - notional,
            4,
        )
        self._record_fill_metrics(fill, strategy_id=strategy_id)
        strategy_state["open_risk_usd"] = _recalculate_open_risk(strategy_state.get("positions") or {})
        self._refresh_state_totals()

    def _persist_exit(self, position_key: str, exit_plan: Dict[str, Any], fill) -> Dict[str, Any]:
        """Persist a paper exit (partial or full close)."""
        full_key, strategy_id, base_key = self._resolve_position_keys(position_key)
        position = dict((self.state.get("positions") or {}).get(full_key) or {})
        if not position:
            return {}
        strategy_state = self._strategy_state(strategy_id)

        shares_open = float(_safe_float(position.get("shares_open")) or _safe_float(position.get("shares")) or 0.0)
        size = min(shares_open, fill.filled_size)
        if size <= 0:
            return {}

        entry_price = float(_safe_float(position.get("entry_price")) or 0.0)
        cost_basis_open = float(_safe_float(position.get("cost_basis_open_usd")) or (shares_open * entry_price))
        average_cost = cost_basis_open / shares_open if shares_open > 0 else entry_price
        cost_basis_closed = average_cost * size
        proceeds_usd = fill.notional_usd
        realized_pnl = proceeds_usd - cost_basis_closed
        remaining_shares = max(0.0, shares_open - size)
        remaining_cost_basis = max(0.0, cost_basis_open - cost_basis_closed)
        now_iso = _utc_now().isoformat()

        fill_record = {
            "closed_at_utc": now_iso,
            "mode": "paper_shadow",
            "reason": exit_plan.get("reason"),
            "size": round(size, 6),
            "price": round(fill.avg_price, 6),
            "proceeds_usd": round(proceeds_usd, 6),
            "realized_pnl_usd": round(realized_pnl, 6),
            "slippage_vs_best": round(fill.slippage_vs_best, 6),
            "levels_consumed": fill.levels_consumed,
        }

        position["fills"] = list(position.get("fills") or []) + [fill_record]
        position["shares_open"] = round(remaining_shares, 6)
        position["cost_basis_open_usd"] = round(remaining_cost_basis, 6)
        position["realized_pnl_usd"] = round(float(_safe_float(position.get("realized_pnl_usd")) or 0.0) + realized_pnl, 6)
        position["last_exit_reason"] = exit_plan.get("reason")
        position["last_exit_utc"] = now_iso
        if remaining_shares <= 1e-9:
            position["status"] = "CLOSED"
            position["closed_at_utc"] = now_iso

        position["strategy"] = strategy_id
        position["strategy_id"] = strategy_id
        position["base_position_key"] = base_key
        position["position_key"] = full_key
        self._store_position_record(position, strategy_id=strategy_id)

        strategy_state["gross_sells_today_usd"] = round(
            float(_safe_float(strategy_state.get("gross_sells_today_usd")) or 0.0) + proceeds_usd, 6
        )
        strategy_state["realized_pnl_today_usd"] = round(
            float(_safe_float(strategy_state.get("realized_pnl_today_usd")) or 0.0) + realized_pnl, 6
        )
        strategy_state["total_realized_pnl_usd"] = round(
            float(_safe_float(strategy_state.get("total_realized_pnl_usd")) or 0.0) + realized_pnl, 6
        )
        strategy_state["total_exits"] = int(_safe_float(strategy_state.get("total_exits")) or 0) + 1
        strategy_state["closed_trades_today"] = int(_safe_float(strategy_state.get("closed_trades_today")) or 0) + 1
        strategy_state["paper_equity_usd"] = round(
            float(_safe_float(strategy_state.get("paper_equity_usd")) or strategy_state.get("initial_bankroll_usd") or self.config.initial_bankroll_usd)
            + proceeds_usd,
            4,
        )
        # Track exit reasons
        reason = str(exit_plan.get("reason") or "unknown")
        exits_by_reason = strategy_state.get("exits_by_reason") or {}
        exits_by_reason[reason] = int(_safe_float(exits_by_reason.get(reason)) or 0) + 1
        strategy_state["exits_by_reason"] = exits_by_reason
        strategy_state["open_risk_usd"] = _recalculate_open_risk(strategy_state.get("positions") or {})
        self._refresh_state_totals()

        return {
            "position_key": full_key,
            "strategy_id": strategy_id,
            "remaining_shares": round(remaining_shares, 6),
            "realized_pnl_usd": round(realized_pnl, 6),
            "reason": exit_plan.get("reason"),
        }

    # -- Mark to market --

    def _mark_to_market_all(self, payload_cache: Optional[Dict[str, Dict]] = None) -> None:
        """Update current_value and unrealized PnL for all open positions."""
        positions = self.state.get("positions") or {}
        for pkey, position in list(positions.items()):
            if not isinstance(position, dict):
                continue
            if str(position.get("status") or "").upper() != "OPEN":
                continue
            shares_open = float(_safe_float(position.get("shares_open")) or 0.0)
            if shares_open <= 0:
                continue

            entry_price = float(_safe_float(position.get("entry_price")) or 0.0)
            station_id = str(position.get("station_id") or "")
            label = str(position.get("label") or "")
            side = str(position.get("side") or "YES")

            # Try to get live book from cached payloads
            current_price = None
            if payload_cache and station_id in payload_cache:
                pl = payload_cache[station_id]
                book = self._get_live_book_from_payload(pl, label, side)
                if book and _safe_float(book.best_bid) is not None:
                    current_price = float(book.best_bid)

            if current_price is not None:
                position["current_price"] = round(current_price, 6)
                position["current_value_usd"] = round(shares_open * current_price, 4)
                cost_basis = float(_safe_float(position.get("cost_basis_open_usd")) or (shares_open * entry_price))
                position["unrealized_pnl_usd"] = round(position["current_value_usd"] - cost_basis, 4)
                self._store_position_record(position, strategy_id=_normalize_strategy_id(position.get("strategy_id") or position.get("strategy")))

        self._refresh_state_totals()

    # -- Main execution loop --

    async def run_station_once(self, station_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run one tick for a single station.

        If ``payload`` is provided it will be used directly, skipping the
        network fetch.  ``run_once`` pre-fetches payloads so they can also be
        passed to ``_mark_to_market_all``.
        """
        atc = self._atc()
        resolved_target_day, resolved_target_date = resolve_station_market_target(
            station_id, atc,
            override_target_date=self.config.target_date or None,
        )

        if payload is None:
            try:
                payload = await self._fetch_signal_payload(
                    station_id, resolved_target_day, target_date=resolved_target_date,
                )
            except Exception as exc:
                result = {"station_id": station_id, "status": "error", "error": str(exc)}
                self._append_journal({"event": "fetch_error", **result})
                return result

        # Pre-fetch ensemble data for micro_value_ensemble strategy
        self._current_ensemble = None
        if "micro_value_ensemble" in PAPER_STRATEGY_ORDER:
            try:
                from collector.ensemble_fetcher import fetch_ensemble_snapshot
                from datetime import date as _date

                td = _date.fromisoformat(resolved_target_date) if resolved_target_date else None
                self._current_ensemble = await fetch_ensemble_snapshot(station_id, td)
            except Exception as exc:
                logger.debug("Ensemble fetch for %s: %s", station_id, exc)

        # Manage exits first
        exit_events = await self._manage_open_positions(station_id, payload)
        strategy_results: List[Dict[str, Any]] = []
        any_fill = False
        any_hold = bool(exit_events)

        for strategy_id in PAPER_STRATEGY_ORDER:
            strategy_state = self._strategy_state(strategy_id)
            candidate, reasons = self._evaluate_strategy_candidate(strategy_id, payload)
            if not candidate:
                strategy_results.append({"strategy_id": strategy_id, "status": "skip", "reasons": reasons})
                if "no_price_anomaly" in reasons:
                    self._append_journal({
                        "event": "price_anomaly",
                        "station_id": station_id,
                        "strategy_id": strategy_id,
                        "reasons": reasons,
                        "source": "candidate_build",
                    })
                continue

            live_book = self._get_live_book_from_payload(payload, candidate.label, candidate.side)
            if live_book is None or live_book.best_ask is None:
                strategy_results.append(
                    {
                        "strategy_id": strategy_id,
                        "status": "skip",
                        "reasons": ["no_live_book"],
                        "candidate": asdict(candidate),
                    }
                )
                continue

            # Micro value uses fixed $1 budget (gopfan2 style)
            if strategy_id == "micro_value_ensemble":
                equity = float(_safe_float(strategy_state.get("paper_equity_usd")) or 0.0)
                budget_usd = min(1.0, equity * 0.10) if equity > 0 else 1.0
            else:
                budget_usd = compute_trade_budget_usd(
                    candidate,
                    atc,
                    _normalize_state(strategy_state),
                    available_balance_usd=float(_safe_float(strategy_state.get("paper_equity_usd")) or 0.0),
                )
            if budget_usd < float(self.config.min_trade_usd):
                strategy_results.append(
                    {
                        "strategy_id": strategy_id,
                        "status": "skip",
                        "reasons": ["budget_too_small"],
                        "candidate": asdict(candidate),
                    }
                )
                continue

            plan, plan_reasons = apply_orderbook_guardrails(candidate, live_book, atc, budget_usd)
            if not plan:
                strategy_results.append(
                    {
                        "strategy_id": strategy_id,
                        "status": "skip",
                        "reasons": plan_reasons,
                        "candidate": asdict(candidate),
                    }
                )
                if "no_ask_price_anomaly" in plan_reasons:
                    self._append_journal({
                        "event": "price_anomaly",
                        "station_id": station_id,
                        "strategy_id": strategy_id,
                        "label": candidate.label,
                        "side": candidate.side,
                        "market_price": candidate.market_price,
                        "fair_price": candidate.fair_price,
                        "live_ask": float(_safe_float(live_book.best_ask) or 0),
                        "reasons": plan_reasons,
                        "source": "orderbook_guardrails",
                    })
                continue

            l2 = self._get_l2_from_payload(payload, candidate.label, candidate.side)
            limit_price = float(plan["limit_price"])
            size = float(plan["size"])
            if self.config.order_mode == "market_fok":
                fill = self.execution.execute_market_fok_buy(candidate.token_id, limit_price, size, l2)
            else:
                fill = self.execution.execute_limit_fak_buy(candidate.token_id, limit_price, size, l2)

            if fill.filled_size <= 0:
                self._record_rejection(strategy_id=strategy_id)
                rejected = {
                    "strategy_id": strategy_id,
                    "station_id": station_id,
                    "status": "paper_rejected",
                    "reasons": ["no_fill"],
                    "label": candidate.label,
                    "side": candidate.side,
                    "candidate": asdict(candidate),
                    "plan": plan,
                    "fill_status": fill.status,
                }
                strategy_results.append(rejected)
                self._append_journal({"event": "entry_rejected", **rejected})
                continue

            self._persist_trade(candidate, plan, fill)
            any_fill = True
            signal_snap = {
                "fair_price": round(candidate.fair_price, 6),
                "market_price": round(candidate.market_price, 6),
                "edge_points": round(candidate.edge_points, 4),
                "strategy": candidate.strategy,
                "recommendation": candidate.recommendation,
            }
            book_snap = compact_book_snapshot(l2)
            filled = {
                "strategy_id": strategy_id,
                "station_id": station_id,
                "status": "paper_fill",
                "candidate": asdict(candidate),
                "plan": plan,
                "fill": fill.to_dict(),
            }
            strategy_results.append(filled)
            self._append_journal(
                {
                    "event": "entry_fill",
                    "strategy_id": strategy_id,
                    "station_id": station_id,
                    "label": candidate.label,
                    "side": candidate.side,
                    "position_key": candidate.position_key,
                    "signal_snapshot": signal_snap,
                    "book_snapshot": book_snap,
                    "order": {"order_type": self.config.order_mode, "limit_price": limit_price, "size": size},
                    "fill": fill.to_dict(),
                }
            )

        status = "paper_fill" if any_fill else "hold" if any_hold else "skip"
        result = {
            "station_id": station_id,
            "status": status,
            "exits": exit_events,
            "strategy_results": strategy_results,
        }
        if any_fill or exit_events:
            self._append_journal(
                {
                    "event": "station_tick",
                    "station_id": station_id,
                    "status": status,
                    "fills": sum(1 for item in strategy_results if item.get("status") == "paper_fill"),
                    "exits": len(exit_events),
                }
            )
        self._save_state()
        return result

    async def run_once(self) -> List[Dict[str, Any]]:
        """Run one tick across all configured stations."""
        self.state = load_paper_state(self.config.state_path, self.config)
        self._reset_daily_counters_if_needed()

        results = []
        payload_cache: Dict[str, Dict] = {}
        atc = self._atc()

        # Pre-fetch payloads for every station so we can pass them to both
        # run_station_once and _mark_to_market_all (MTM requires live bid
        # prices from the current tick, not from a stale cache).
        for station_id in self.config.station_ids:
            try:
                resolved_target_day, resolved_target_date = resolve_station_market_target(
                    station_id, atc,
                    override_target_date=self.config.target_date or None,
                )
                payload = await self._fetch_signal_payload(
                    station_id, resolved_target_day, target_date=resolved_target_date,
                )
                payload_cache[station_id] = payload
            except Exception as exc:
                logger.warning("Failed to pre-fetch payload for %s: %s", station_id, exc)

        for station_id in self.config.station_ids:
            try:
                result = await self.run_station_once(station_id, payload=payload_cache.get(station_id))
                results.append(result)
            except Exception as exc:
                logger.exception("Error in papertrader tick for %s", station_id)
                results.append({"station_id": station_id, "status": "error", "error": str(exc)})

        # Mark to market using the payloads fetched in this tick
        self._mark_to_market_all(payload_cache=payload_cache)
        self._save_state()
        return results

    async def run_loop(self) -> None:
        """Continuous loop running run_once() every loop_seconds."""
        logger.info("Papertrader loop starting: stations=%s, interval=%ds", self.config.station_ids, self.config.loop_seconds)
        while self.config.enabled:
            try:
                results = await self.run_once()
                fills = sum(
                    1
                    for station_result in results
                    for strategy_result in (station_result.get("strategy_results") or [])
                    if strategy_result.get("status") == "paper_fill"
                )
                exits = sum(len(r.get("exits") or []) for r in results)
                if fills or exits:
                    logger.info("Papertrader tick: %d fills, %d exits across %d stations", fills, exits, len(results))
            except Exception:
                logger.exception("Papertrader loop error")
            await asyncio.sleep(self.config.loop_seconds)
        logger.info("Papertrader loop stopped")

    # -- Status snapshot --

    def _get_status_snapshot_legacy(self) -> Dict[str, Any]:
        """Build a comprehensive status snapshot for the API/UI."""
        self.state = load_paper_state(self.config.state_path, self.config)
        positions = self.state.get("positions") or {}
        open_positions = []
        closed_positions = []
        for pkey, pos in positions.items():
            if not isinstance(pos, dict):
                continue
            pos_copy = dict(pos)
            pos_copy["position_key"] = pkey
            if str(pos.get("status") or "").upper() == "OPEN":
                open_positions.append(pos_copy)
            else:
                closed_positions.append(pos_copy)

        open_risk = float(_safe_float(self.state.get("open_risk_usd")) or 0.0)
        total_realized = float(_safe_float(self.state.get("total_realized_pnl_usd")) or 0.0)
        total_unrealized = sum(float(_safe_float(p.get("unrealized_pnl_usd")) or 0.0) for p in open_positions)
        total_entries = int(_safe_float(self.state.get("total_entries")) or 0)
        total_exits = int(_safe_float(self.state.get("total_exits")) or 0)
        total_fills = int(_safe_float(self.state.get("total_fills")) or 0)
        total_partial = int(_safe_float(self.state.get("total_partial_fills")) or 0)
        total_rejections = int(_safe_float(self.state.get("total_rejections")) or 0)
        total_attempts = total_fills + total_rejections
        fill_ratio = round(total_fills / total_attempts, 4) if total_attempts > 0 else 0.0
        total_slippage = float(_safe_float(self.state.get("total_slippage_usd")) or 0.0)
        total_fill_notional = float(_safe_float(self.state.get("total_fill_notional_usd")) or 0.0)
        # Slippage in bps = (total slippage $ / total fill notional $) × 10000
        # Using notional instead of fill count gives a true basis-point figure.
        avg_slippage_bps = round((total_slippage / total_fill_notional) * 10000, 2) if total_fill_notional > 0 else 0.0

        # Hit rate: fraction of closed positions with positive realized PnL
        winners = sum(1 for p in closed_positions if float(_safe_float(p.get("realized_pnl_usd")) or 0.0) > 0)
        hit_rate = round(winners / len(closed_positions), 4) if closed_positions else 0.0

        recent = self._recent_events[:8] if self._recent_events else _tail_jsonl(self.config.log_path, limit=8)

        return {
            "mode": "paper_shadow",
            "station_ids": list(self.config.station_ids),
            "target_date": self.config.target_date,
            "loop_seconds": self.config.loop_seconds,
            "paper_equity_usd": round(float(_safe_float(self.state.get("paper_equity_usd")) or self.config.initial_bankroll_usd), 4),
            "initial_bankroll_usd": self.config.initial_bankroll_usd,
            "open_positions": open_positions,
            "open_positions_count": len(open_positions),
            "closed_positions_count": len(closed_positions),
            "open_risk_usd": round(open_risk, 4),
            "realized_pnl_usd": round(total_realized, 4),
            "realized_pnl_today_usd": round(float(_safe_float(self.state.get("realized_pnl_today_usd")) or 0.0), 4),
            "unrealized_pnl_usd": round(total_unrealized, 4),
            "trades_today": int(_safe_float(self.state.get("trades_today")) or 0),
            "total_entries": total_entries,
            "total_exits": total_exits,
            "total_fills": total_fills,
            "total_partial_fills": total_partial,
            "total_rejections": total_rejections,
            "fill_ratio": fill_ratio,
            "hit_rate": hit_rate,
            "avg_slippage_bps": avg_slippage_bps,
            "total_slippage_usd": round(total_slippage, 6),
            "total_fill_notional_usd": round(total_fill_notional, 4),
            "exits_by_reason": dict(self.state.get("exits_by_reason") or {}),
            "max_total_exposure_usd": self.config.max_total_exposure_usd,
            "recent_events": recent,
            "closed_positions": closed_positions,
        }

    def get_status_snapshot(self) -> Dict[str, Any]:
        """Build a multi-strategy status snapshot for the API/UI."""
        self.state = load_paper_state(self.config.state_path, self.config)
        positions = self.state.get("positions") or {}
        all_positions: List[Dict[str, Any]] = []
        for pkey, pos in positions.items():
            if not isinstance(pos, dict):
                continue
            pos_copy = dict(pos)
            pos_copy["position_key"] = pkey
            pos_copy["strategy_id"] = _normalize_strategy_id(pos_copy.get("strategy_id") or pos_copy.get("strategy"))
            all_positions.append(pos_copy)

        def _sort_key(position: Dict[str, Any]) -> str:
            return str(position.get("closed_at_utc") or position.get("last_exit_utc") or position.get("opened_at_utc") or "")

        open_positions = [position for position in all_positions if str(position.get("status") or "").upper() == "OPEN"]
        closed_positions = [position for position in all_positions if str(position.get("status") or "").upper() != "OPEN"]
        open_positions.sort(key=_sort_key, reverse=True)
        closed_positions.sort(key=_sort_key, reverse=True)

        open_risk = float(_safe_float(self.state.get("open_risk_usd")) or 0.0)
        total_realized = float(_safe_float(self.state.get("total_realized_pnl_usd")) or 0.0)
        total_unrealized = sum(float(_safe_float(position.get("unrealized_pnl_usd")) or 0.0) for position in open_positions)
        total_entries = int(_safe_float(self.state.get("total_entries")) or 0)
        total_exits = int(_safe_float(self.state.get("total_exits")) or 0)
        total_fills = int(_safe_float(self.state.get("total_fills")) or 0)
        total_partial = int(_safe_float(self.state.get("total_partial_fills")) or 0)
        total_rejections = int(_safe_float(self.state.get("total_rejections")) or 0)
        total_attempts = total_fills + total_rejections
        fill_ratio = round(total_fills / total_attempts, 4) if total_attempts > 0 else 0.0
        total_slippage = float(_safe_float(self.state.get("total_slippage_usd")) or 0.0)
        total_fill_notional = float(_safe_float(self.state.get("total_fill_notional_usd")) or 0.0)
        avg_slippage_bps = round((total_slippage / total_fill_notional) * 10000, 2) if total_fill_notional > 0 else 0.0
        winners = sum(1 for position in closed_positions if float(_safe_float(position.get("realized_pnl_usd")) or 0.0) > 0)
        hit_rate = round(winners / len(closed_positions), 4) if closed_positions else 0.0
        recent = self._recent_events[:24] if self._recent_events else _tail_jsonl(self.config.log_path, limit=24)

        strategy_snapshots: List[Dict[str, Any]] = []
        for strategy_id in PAPER_STRATEGY_ORDER:
            strategy_state = self._strategy_state(strategy_id)
            strategy_positions = [position for position in all_positions if position.get("strategy_id") == strategy_id]
            strategy_open = [position for position in strategy_positions if str(position.get("status") or "").upper() == "OPEN"]
            strategy_closed = [position for position in strategy_positions if str(position.get("status") or "").upper() != "OPEN"]
            strategy_unrealized = sum(float(_safe_float(position.get("unrealized_pnl_usd")) or 0.0) for position in strategy_open)
            strategy_fills = int(_safe_float(strategy_state.get("total_fills")) or 0)
            strategy_rejections = int(_safe_float(strategy_state.get("total_rejections")) or 0)
            strategy_attempts = strategy_fills + strategy_rejections
            strategy_fill_ratio = round(strategy_fills / strategy_attempts, 4) if strategy_attempts > 0 else 0.0
            strategy_slippage = float(_safe_float(strategy_state.get("total_slippage_usd")) or 0.0)
            strategy_notional = float(_safe_float(strategy_state.get("total_fill_notional_usd")) or 0.0)
            strategy_hit_rate = (
                round(
                    sum(1 for position in strategy_closed if float(_safe_float(position.get("realized_pnl_usd")) or 0.0) > 0) / len(strategy_closed),
                    4,
                )
                if strategy_closed
                else 0.0
            )
            strategy_recent = [
                dict(event)
                for event in recent
                if _normalize_strategy_id((event or {}).get("strategy_id")) == strategy_id
            ][:8]
            meta = PAPER_STRATEGY_BY_ID[strategy_id]
            strategy_snapshots.append(
                {
                    "strategy_id": strategy_id,
                    "title": meta["title"],
                    "subtitle": meta["subtitle"],
                    "description": meta["description"],
                    "accent": meta["accent"],
                    "initial_bankroll_usd": round(float(_safe_float(strategy_state.get("initial_bankroll_usd")) or self.config.initial_bankroll_usd), 4),
                    "paper_equity_usd": round(float(_safe_float(strategy_state.get("paper_equity_usd")) or self.config.initial_bankroll_usd), 4),
                    "free_equity_usd": round(
                        float(_safe_float(strategy_state.get("paper_equity_usd")) or self.config.initial_bankroll_usd)
                        - float(_safe_float(strategy_state.get("open_risk_usd")) or 0.0),
                        4,
                    ),
                    "open_risk_usd": round(float(_safe_float(strategy_state.get("open_risk_usd")) or 0.0), 4),
                    "realized_pnl_usd": round(float(_safe_float(strategy_state.get("total_realized_pnl_usd")) or 0.0), 4),
                    "realized_pnl_today_usd": round(float(_safe_float(strategy_state.get("realized_pnl_today_usd")) or 0.0), 4),
                    "unrealized_pnl_usd": round(strategy_unrealized, 4),
                    "trades_today": int(_safe_float(strategy_state.get("trades_today")) or 0),
                    "total_entries": int(_safe_float(strategy_state.get("total_entries")) or 0),
                    "total_exits": int(_safe_float(strategy_state.get("total_exits")) or 0),
                    "total_fills": strategy_fills,
                    "total_partial_fills": int(_safe_float(strategy_state.get("total_partial_fills")) or 0),
                    "total_rejections": strategy_rejections,
                    "fill_ratio": strategy_fill_ratio,
                    "hit_rate": strategy_hit_rate,
                    "avg_slippage_bps": round((strategy_slippage / strategy_notional) * 10000, 2) if strategy_notional > 0 else 0.0,
                    "last_trade_utc": strategy_state.get("last_trade_utc"),
                    "open_positions_count": len(strategy_open),
                    "closed_positions_count": len(strategy_closed),
                    "open_positions": strategy_open[:6],
                    "closed_positions_recent": strategy_closed[:12],
                    "recent_events": strategy_recent,
                    "exits_by_reason": dict(strategy_state.get("exits_by_reason") or {}),
                }
            )

        return {
            "mode": "paper_multi_strategy",
            "station_ids": list(self.config.station_ids),
            "target_date": self.config.target_date,
            "loop_seconds": self.config.loop_seconds,
            "strategy_count": len(PAPER_STRATEGY_ORDER),
            "strategy_order": list(PAPER_STRATEGY_ORDER),
            "aggregate_initial_bankroll_usd": round(float(_safe_float(self.state.get("aggregate_initial_bankroll_usd")) or 0.0), 4),
            "paper_equity_usd": round(float(_safe_float(self.state.get("paper_equity_usd")) or 0.0), 4),
            "initial_bankroll_usd": self.config.initial_bankroll_usd,
            "free_equity_usd": round(float(_safe_float(self.state.get("paper_equity_usd")) or 0.0) - open_risk, 4),
            "open_positions": open_positions,
            "open_positions_count": len(open_positions),
            "closed_positions_count": len(closed_positions),
            "open_risk_usd": round(open_risk, 4),
            "realized_pnl_usd": round(total_realized, 4),
            "realized_pnl_today_usd": round(float(_safe_float(self.state.get("realized_pnl_today_usd")) or 0.0), 4),
            "unrealized_pnl_usd": round(total_unrealized, 4),
            "trades_today": int(_safe_float(self.state.get("trades_today")) or 0),
            "total_entries": total_entries,
            "total_exits": total_exits,
            "total_fills": total_fills,
            "total_partial_fills": total_partial,
            "total_rejections": total_rejections,
            "fill_ratio": fill_ratio,
            "hit_rate": hit_rate,
            "avg_slippage_bps": avg_slippage_bps,
            "total_slippage_usd": round(total_slippage, 6),
            "total_fill_notional_usd": round(total_fill_notional, 4),
            "exits_by_reason": dict(self.state.get("exits_by_reason") or {}),
            "max_total_exposure_usd": self.config.max_total_exposure_usd,
            "max_open_positions": self.config.max_open_positions,
            "recent_events": recent[:16],
            "closed_positions": closed_positions[:50],
            "strategies": strategy_snapshots,
        }
