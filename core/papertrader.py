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
    nyc_today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    return {
        "paper_equity_usd": config.initial_bankroll_usd,
        "initial_bankroll_usd": config.initial_bankroll_usd,
        "positions": {},
        "pending_orders": {},
        "trades_today": 0,
        "spent_today_usd": 0.0,
        "gross_buys_today_usd": 0.0,
        "gross_sells_today_usd": 0.0,
        "realized_pnl_today_usd": 0.0,
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


def _paper_session_has_activity(state: Dict[str, Any]) -> bool:
    positions = state.get("positions") or {}
    if isinstance(positions, dict) and positions:
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
    initial_bankroll = float(_safe_float(state.get("initial_bankroll_usd")) or 0.0)
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
                initial_bankroll = _safe_float(state.get("initial_bankroll_usd"))
                if initial_bankroll is None or initial_bankroll <= 0:
                    state["initial_bankroll_usd"] = config.initial_bankroll_usd
                    initial_bankroll = config.initial_bankroll_usd
                if not _paper_session_has_activity(state):
                    state["initial_bankroll_usd"] = config.initial_bankroll_usd
                    state["paper_equity_usd"] = config.initial_bankroll_usd
                elif _safe_float(state.get("paper_equity_usd")) is None:
                    state["paper_equity_usd"] = float(initial_bankroll)
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
        state_initial = float(_safe_float(self.state.get("initial_bankroll_usd")) or self.config.initial_bankroll_usd)
        self.config.initial_bankroll_usd = state_initial
        self.config.bankroll_usd = state_initial

    def _record_fill_metrics(self, fill: Any) -> None:
        filled_size = float(_safe_float(getattr(fill, "filled_size", None)) or 0.0)
        if filled_size <= 0.0:
            return
        self.state["total_fills"] = int(_safe_float(self.state.get("total_fills")) or 0) + 1
        if str(getattr(fill, "status", "") or "").upper() == "PARTIAL":
            self.state["total_partial_fills"] = int(_safe_float(self.state.get("total_partial_fills")) or 0) + 1
        slippage_usd = float(_safe_float(getattr(fill, "slippage_vs_best", None)) or 0.0) * filled_size
        self.state["total_slippage_usd"] = round(
            float(_safe_float(self.state.get("total_slippage_usd")) or 0.0) + slippage_usd,
            6,
        )
        self.state["total_fill_notional_usd"] = round(
            float(_safe_float(self.state.get("total_fill_notional_usd")) or 0.0)
            + float(_safe_float(getattr(fill, "notional_usd", None)) or 0.0),
            6,
        )

    def _record_rejection(self) -> None:
        self.state["total_rejections"] = int(_safe_float(self.state.get("total_rejections")) or 0) + 1

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

    # -- Daily reset --

    def _reset_daily_counters_if_needed(self) -> None:
        nyc_today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        if self.state.get("trading_day") != nyc_today:
            self.state["trading_day"] = nyc_today
            self.state["trades_today"] = 0
            self.state["spent_today_usd"] = 0.0
            self.state["gross_buys_today_usd"] = 0.0
            self.state["gross_sells_today_usd"] = 0.0
            self.state["realized_pnl_today_usd"] = 0.0
            self.state["closed_trades_today"] = 0

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

            if fill.filled_size > 0:
                persisted = self._persist_exit(pkey, exit_plan, fill)
                self._record_fill_metrics(fill)
                event = {
                    "event": "exit_fill",
                    "station_id": station_id,
                    "position_key": pkey,
                    "reason": exit_plan.get("reason"),
                    "fill": fill.to_dict(),
                    "persisted": persisted,
                }
                self._append_journal(event)
                exit_events.append(event)
            else:
                self._record_rejection()
                event = {
                    "event": "exit_rejected",
                    "station_id": station_id,
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

        self.state["positions"][candidate.position_key] = {
            "station_id": candidate.station_id,
            "target_date": candidate.target_date,
            "label": candidate.label,
            "side": candidate.side,
            "token_id": candidate.token_id,
            "status": "OPEN",
            "mode": "paper_shadow",
            "strategy": candidate.strategy,
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
            "position_key": candidate.position_key,
        }

        gross = float(_safe_float(self.state.get("gross_buys_today_usd")) or 0.0)
        self.state["gross_buys_today_usd"] = round(gross + notional, 6)
        self.state["spent_today_usd"] = self.state["gross_buys_today_usd"]
        self.state["trades_today"] = int(_safe_float(self.state.get("trades_today")) or 0) + 1
        self.state["total_entries"] = int(_safe_float(self.state.get("total_entries")) or 0) + 1
        # _record_fill_metrics handles total_fills, total_partial_fills,
        # total_slippage_usd AND total_fill_notional_usd in one place.
        self._record_fill_metrics(fill)
        self.state["last_trade_utc"] = now_iso
        self.state["paper_equity_usd"] = round(
            float(_safe_float(self.state.get("paper_equity_usd")) or self.config.initial_bankroll_usd) - notional, 4
        )
        self.state["open_risk_usd"] = _recalculate_open_risk(self.state.get("positions") or {})

    def _persist_exit(self, position_key: str, exit_plan: Dict[str, Any], fill) -> Dict[str, Any]:
        """Persist a paper exit (partial or full close)."""
        position = dict((self.state.get("positions") or {}).get(position_key) or {})
        if not position:
            return {}

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

        self.state["positions"][position_key] = position
        self.state["gross_sells_today_usd"] = round(
            float(_safe_float(self.state.get("gross_sells_today_usd")) or 0.0) + proceeds_usd, 6
        )
        self.state["realized_pnl_today_usd"] = round(
            float(_safe_float(self.state.get("realized_pnl_today_usd")) or 0.0) + realized_pnl, 6
        )
        self.state["total_realized_pnl_usd"] = round(
            float(_safe_float(self.state.get("total_realized_pnl_usd")) or 0.0) + realized_pnl, 6
        )
        self.state["total_exits"] = int(_safe_float(self.state.get("total_exits")) or 0) + 1
        self.state["closed_trades_today"] = int(_safe_float(self.state.get("closed_trades_today")) or 0) + 1
        self.state["paper_equity_usd"] = round(
            float(_safe_float(self.state.get("paper_equity_usd")) or self.config.initial_bankroll_usd) + proceeds_usd, 4
        )
        # Track exit reasons
        reason = str(exit_plan.get("reason") or "unknown")
        exits_by_reason = self.state.get("exits_by_reason") or {}
        exits_by_reason[reason] = int(_safe_float(exits_by_reason.get(reason)) or 0) + 1
        self.state["exits_by_reason"] = exits_by_reason
        self.state["open_risk_usd"] = _recalculate_open_risk(self.state.get("positions") or {})

        return {
            "position_key": position_key,
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

        self.state["open_risk_usd"] = _recalculate_open_risk(positions)

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

        # Manage exits first
        exit_events = await self._manage_open_positions(station_id, payload)

        # Evaluate trade candidate
        candidate, reasons = evaluate_trade_candidate(payload, atc, self.state)
        if not candidate:
            status = "hold" if exit_events else "skip"
            result = {"station_id": station_id, "status": status, "reasons": reasons, "exits": exit_events}
            if status != "skip" or exit_events:
                self._append_journal({"event": "tick", **result})
            return result

        # Check duplicate position
        existing = self.state.get("positions", {}).get(candidate.position_key)
        if existing and str(existing.get("status") or "").upper() == "OPEN":
            result = {"station_id": station_id, "status": "skip", "reasons": ["duplicate_position"], "exits": exit_events}
            return result

        # Get live book for guardrails
        live_book = self._get_live_book_from_payload(payload, candidate.label, candidate.side)
        if live_book is None or live_book.best_ask is None:
            result = {"station_id": station_id, "status": "skip", "reasons": ["no_live_book"], "exits": exit_events}
            self._append_journal({"event": "no_book", **result})
            return result

        # Compute budget
        budget_usd = compute_trade_budget_usd(candidate, atc, self.state)
        if budget_usd < float(self.config.min_trade_usd):
            result = {"station_id": station_id, "status": "skip", "reasons": ["budget_too_small"], "candidate": asdict(candidate), "exits": exit_events}
            self._append_journal({"event": "budget_skip", **result})
            return result

        # Apply guardrails
        plan, plan_reasons = apply_orderbook_guardrails(candidate, live_book, atc, budget_usd)
        if not plan:
            result = {"station_id": station_id, "status": "skip", "reasons": plan_reasons, "candidate": asdict(candidate), "exits": exit_events}
            self._append_journal({"event": "guardrail_skip", **result})
            return result

        # Execute paper order against L2 book
        l2 = self._get_l2_from_payload(payload, candidate.label, candidate.side)
        limit_price = float(plan["limit_price"])
        size = float(plan["size"])

        if self.config.order_mode == "market_fok":
            fill = self.execution.execute_market_fok_buy(candidate.token_id, limit_price, size, l2)
        else:
            fill = self.execution.execute_limit_fak_buy(candidate.token_id, limit_price, size, l2)

        if fill.filled_size <= 0:
            self._record_rejection()
            result = {
                "station_id": station_id,
                "status": "paper_rejected",
                "reasons": ["no_fill"],
                "candidate": asdict(candidate),
                "plan": plan,
                "fill_status": fill.status,
                "exits": exit_events,
            }
            self._append_journal({"event": "entry_rejected", **result})
            return result

        # Persist paper position (total_partial_fills counted inside _record_fill_metrics)
        self._persist_trade(candidate, plan, fill)

        signal_snap = {
            "fair_price": round(candidate.fair_price, 6),
            "market_price": round(candidate.market_price, 6),
            "edge_points": round(candidate.edge_points, 4),
            "strategy": candidate.strategy,
            "recommendation": candidate.recommendation,
        }
        book_snap = compact_book_snapshot(l2)

        result = {
            "station_id": station_id,
            "status": "paper_fill",
            "candidate": asdict(candidate),
            "plan": plan,
            "fill": fill.to_dict(),
            "exits": exit_events,
        }
        self._append_journal({
            "event": "entry_fill",
            "station_id": station_id,
            "position_key": candidate.position_key,
            "signal_snapshot": signal_snap,
            "book_snapshot": book_snap,
            "order": {"order_type": self.config.order_mode, "limit_price": limit_price, "size": size},
            "fill": fill.to_dict(),
        })

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
                fills = sum(1 for r in results if r.get("status") == "paper_fill")
                exits = sum(len(r.get("exits") or []) for r in results)
                if fills or exits:
                    logger.info("Papertrader tick: %d fills, %d exits across %d stations", fills, exits, len(results))
            except Exception:
                logger.exception("Papertrader loop error")
            await asyncio.sleep(self.config.loop_seconds)
        logger.info("Papertrader loop stopped")

    # -- Status snapshot --

    def get_status_snapshot(self) -> Dict[str, Any]:
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
