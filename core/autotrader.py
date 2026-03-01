from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx

from config import STATIONS
from core.polymarket_labels import normalize_label
from market.polymarket_execution import (
    PolymarketExecutionClient,
    TopOfBook,
    load_polymarket_execution_config_from_env,
    quantize_buy_order_size,
    quantize_market_buy_amount,
    quantize_share_size,
)


logger = logging.getLogger("autotrader")
UTC = ZoneInfo("UTC")
_PORTFOLIO_LOCK: Optional[asyncio.Lock] = None


def _get_portfolio_lock() -> asyncio.Lock:
    global _PORTFOLIO_LOCK
    if _PORTFOLIO_LOCK is None:
        _PORTFOLIO_LOCK = asyncio.Lock()
    return _PORTFOLIO_LOCK


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, float(value)))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_list(name: str, default: Optional[List[str]] = None) -> List[str]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return list(default or [])
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _position_key(station_id: str, target_date: str, label: str, side: str) -> str:
    return f"{station_id}|{target_date}|{label}|{side}".upper()


def _normalize_iso_date(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError:
        return ""


def resolve_station_market_target(
    station_id: str,
    config: "AutoTraderConfig",
    *,
    override_target_day: Optional[int] = None,
    override_target_date: Optional[str] = None,
) -> Tuple[int, str]:
    station = STATIONS.get(str(station_id or "").upper())
    tz = ZoneInfo(station.timezone) if station else UTC
    local_today = datetime.now(tz).date()

    normalized_target_date = _normalize_iso_date(
        override_target_date if override_target_date is not None else config.target_date
    )
    if normalized_target_date:
        selected_date = date.fromisoformat(normalized_target_date)
        return int((selected_date - local_today).days), normalized_target_date

    target_day = int(config.target_day if override_target_day is None else override_target_day)
    selected_date = local_today + timedelta(days=target_day)
    return target_day, selected_date.isoformat()


@dataclass
class AutoTraderConfig:
    enabled: bool = False
    mode: str = "paper"
    station_ids: List[str] = field(default_factory=list)
    target_day: int = 0
    target_date: str = ""
    signal_source: str = "internal"
    signal_base_url: str = "http://127.0.0.1:8000"
    loop_seconds: int = 60
    bankroll_usd: float = 20.0
    fractional_kelly: float = 0.20
    min_edge_points: float = 6.0
    min_fair_probability: float = 0.18
    max_entry_price: float = 0.40
    min_trade_usd: float = 1.0
    max_trade_usd: float = 2.5
    max_total_exposure_usd: float = 6.0
    max_station_exposure_usd: float = 6.0
    daily_loss_limit_usd: float = 4.0
    max_trades_per_day: int = 4
    cooldown_seconds: int = 900
    max_spread_points: float = 8.0
    min_ask_depth_contracts: float = 15.0
    max_depth_participation: float = 0.25
    max_open_positions: int = 4
    max_station_positions: int = 0
    secondary_yes_edge_bonus_pts: float = 8.0
    require_buy_recommendation: bool = True
    allow_terminal_value: bool = True
    allow_tactical_reprice: bool = True
    tactical_min_edge_points: float = 5.0
    tactical_min_minutes_to_next: float = 8.0
    tactical_max_minutes_to_next: float = 75.0
    tactical_take_profit_points: float = 3.5
    tactical_stop_loss_points: float = 3.0
    tactical_timeout_minutes: int = 75
    terminal_take_profit_points: float = 4.5
    terminal_stop_loss_points: float = 5.0
    exit_edge_buffer_points: float = 1.5
    order_mode: str = "limit_fak"
    limit_slippage_points: float = 1.0
    prefer_ws_book: bool = True
    state_path: str = "data/autotrader_state.json"
    log_path: str = "logs/autotrader_trades.jsonl"
    kill_switch_path: str = "data/AUTOTRADE_STOP"

    @property
    def is_live(self) -> bool:
        return str(self.mode or "paper").strip().lower() == "live"


def load_autotrader_config_from_env() -> AutoTraderConfig:
    config = AutoTraderConfig(
        enabled=_env_bool("HELIOS_AUTOTRADE_ENABLED", False),
        mode=str(os.environ.get("HELIOS_AUTOTRADE_MODE", "paper")).strip().lower() or "paper",
        station_ids=_env_list("HELIOS_AUTOTRADE_STATIONS"),
        target_day=_env_int("HELIOS_AUTOTRADE_TARGET_DAY", 0),
        target_date=str(os.environ.get("HELIOS_AUTOTRADE_TARGET_DATE", "")).strip(),
        signal_source=str(os.environ.get("HELIOS_AUTOTRADE_SIGNAL_SOURCE", "internal")).strip().lower() or "internal",
        signal_base_url=str(os.environ.get("HELIOS_AUTOTRADE_SIGNAL_BASE_URL", "http://127.0.0.1:8000")).strip() or "http://127.0.0.1:8000",
        loop_seconds=_env_int("HELIOS_AUTOTRADE_LOOP_SECONDS", 60),
        bankroll_usd=_env_float("HELIOS_AUTOTRADE_BANKROLL_USD", 20.0),
        fractional_kelly=_env_float("HELIOS_AUTOTRADE_FRACTIONAL_KELLY", 0.20),
        min_edge_points=_env_float("HELIOS_AUTOTRADE_MIN_EDGE_PTS", 6.0),
        min_fair_probability=_env_float("HELIOS_AUTOTRADE_MIN_FAIR_PCT", 0.18),
        max_entry_price=_env_float("HELIOS_AUTOTRADE_MAX_ENTRY_PCT", 0.40),
        min_trade_usd=_env_float("HELIOS_AUTOTRADE_MIN_TRADE_USD", 1.0),
        max_trade_usd=_env_float("HELIOS_AUTOTRADE_MAX_TRADE_USD", 2.5),
        max_total_exposure_usd=_env_float("HELIOS_AUTOTRADE_MAX_TOTAL_EXPOSURE_USD", 6.0),
        max_station_exposure_usd=_env_float("HELIOS_AUTOTRADE_MAX_STATION_EXPOSURE_USD", 3.0),
        daily_loss_limit_usd=_env_float("HELIOS_AUTOTRADE_DAILY_LOSS_LIMIT_USD", 4.0),
        max_trades_per_day=_env_int("HELIOS_AUTOTRADE_MAX_TRADES_PER_DAY", 4),
        cooldown_seconds=_env_int("HELIOS_AUTOTRADE_COOLDOWN_SECONDS", 900),
        max_spread_points=_env_float("HELIOS_AUTOTRADE_MAX_SPREAD_PTS", 8.0),
        min_ask_depth_contracts=_env_float("HELIOS_AUTOTRADE_MIN_ASK_DEPTH_CONTRACTS", 15.0),
        max_depth_participation=_env_float("HELIOS_AUTOTRADE_MAX_DEPTH_PARTICIPATION", 0.25),
        max_open_positions=_env_int("HELIOS_AUTOTRADE_MAX_OPEN_POSITIONS", 4),
        max_station_positions=_env_int("HELIOS_AUTOTRADE_MAX_STATION_POSITIONS", 2),
        secondary_yes_edge_bonus_pts=_env_float("HELIOS_AUTOTRADE_SECONDARY_YES_EDGE_BONUS_PTS", 8.0),
        require_buy_recommendation=_env_bool("HELIOS_AUTOTRADE_REQUIRE_BUY_RECOMMENDATION", True),
        allow_terminal_value=_env_bool("HELIOS_AUTOTRADE_ALLOW_TERMINAL_VALUE", True),
        allow_tactical_reprice=_env_bool("HELIOS_AUTOTRADE_ALLOW_TACTICAL_REPRICE", True),
        tactical_min_edge_points=_env_float("HELIOS_AUTOTRADE_TACTICAL_MIN_EDGE_PTS", 5.0),
        tactical_min_minutes_to_next=_env_float("HELIOS_AUTOTRADE_TACTICAL_MIN_MINUTES_TO_NEXT", 8.0),
        tactical_max_minutes_to_next=_env_float("HELIOS_AUTOTRADE_TACTICAL_MAX_MINUTES_TO_NEXT", 75.0),
        tactical_take_profit_points=_env_float("HELIOS_AUTOTRADE_TACTICAL_TAKE_PROFIT_PTS", 3.5),
        tactical_stop_loss_points=_env_float("HELIOS_AUTOTRADE_TACTICAL_STOP_LOSS_PTS", 3.0),
        tactical_timeout_minutes=_env_int("HELIOS_AUTOTRADE_TACTICAL_TIMEOUT_MINUTES", 75),
        terminal_take_profit_points=_env_float("HELIOS_AUTOTRADE_TERMINAL_TAKE_PROFIT_PTS", 4.5),
        terminal_stop_loss_points=_env_float("HELIOS_AUTOTRADE_TERMINAL_STOP_LOSS_PTS", 5.0),
        exit_edge_buffer_points=_env_float("HELIOS_AUTOTRADE_EXIT_EDGE_BUFFER_PTS", 1.5),
        order_mode=str(os.environ.get("HELIOS_AUTOTRADE_ORDER_MODE", "limit_fak")).strip().lower() or "limit_fak",
        limit_slippage_points=_env_float("HELIOS_AUTOTRADE_LIMIT_SLIPPAGE_PTS", 1.0),
        prefer_ws_book=_env_bool("HELIOS_AUTOTRADE_PREFER_WS_BOOK", True),
        state_path=str(os.environ.get("HELIOS_AUTOTRADE_STATE_PATH", "data/autotrader_state.json")).strip() or "data/autotrader_state.json",
        log_path=str(os.environ.get("HELIOS_AUTOTRADE_LOG_PATH", "logs/autotrader_trades.jsonl")).strip() or "logs/autotrader_trades.jsonl",
        kill_switch_path=str(os.environ.get("HELIOS_AUTOTRADE_KILL_SWITCH_PATH", "data/AUTOTRADE_STOP")).strip() or "data/AUTOTRADE_STOP",
    )
    config.max_station_exposure_usd = float(config.max_total_exposure_usd)
    config.max_station_positions = 0
    return config


@dataclass
class AutoTradeCandidate:
    station_id: str
    target_date: str
    target_day: int
    label: str
    side: str
    token_id: str
    market_price: float
    fair_price: float
    edge_points: float
    model_probability: float
    recommendation: str
    policy_reason: str
    forecast_winner_label: Optional[str]
    forecast_edge_points: Optional[float]
    tactical_alignment: str
    why: str
    position_key: str
    strategy: str = "terminal_value"
    strategy_score: float = 0.0
    opened_next_obs_utc: Optional[str] = None
    raw_row: Dict[str, Any] = field(default_factory=dict)


def _default_state(now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    day = (now_utc or _utc_now()).date().isoformat()
    return {
        "trading_day": day,
        "spent_today_usd": 0.0,
        "gross_buys_today_usd": 0.0,
        "gross_sells_today_usd": 0.0,
        "realized_pnl_today_usd": 0.0,
        "open_risk_usd": 0.0,
        "trades_today": 0,
        "closed_trades_today": 0,
        "last_trade_utc": None,
        "positions": {},
    }


def _recalculate_open_risk(positions: Dict[str, Any]) -> float:
    total = 0.0
    for position in (positions or {}).values():
        if not isinstance(position, dict):
            continue
        if str(position.get("status") or "OPEN").upper() != "OPEN":
            continue
        total += float(_safe_float(position.get("cost_basis_open_usd")) or _safe_float(position.get("notional_usd")) or 0.0)
    return round(total, 6)


def _normalize_state(state: Optional[Dict[str, Any]], now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    normalized = _default_state(now_utc=now_utc)
    if isinstance(state, dict):
        normalized.update(state)

    positions = normalized.get("positions")
    if not isinstance(positions, dict):
        positions = {}

    clean_positions: Dict[str, Any] = {}
    for key, raw_position in positions.items():
        if not isinstance(raw_position, dict):
            continue
        position = dict(raw_position)
        status = str(position.get("status") or "OPEN").upper()
        shares_total = float(_safe_float(position.get("shares")) or _safe_float(position.get("shares_open")) or 0.0)
        shares_open = float(_safe_float(position.get("shares_open")) or shares_total)
        entry_price = float(_safe_float(position.get("entry_price")) or 0.0)
        notional_usd = float(_safe_float(position.get("notional_usd")) or (shares_total * entry_price))
        cost_basis_open_usd = float(_safe_float(position.get("cost_basis_open_usd")) or notional_usd)

        position["status"] = status
        position["shares"] = round(shares_total, 6)
        position["shares_open"] = round(max(0.0, shares_open), 6)
        position["entry_price"] = round(entry_price, 6)
        position["notional_usd"] = round(notional_usd, 6)
        position["cost_basis_open_usd"] = round(max(0.0, cost_basis_open_usd), 6)
        position["realized_pnl_usd"] = round(float(_safe_float(position.get("realized_pnl_usd")) or 0.0), 6)
        position["strategy"] = str(position.get("strategy") or "terminal_value")
        position["fills"] = list(position.get("fills") or [])
        clean_positions[str(key)] = position

    normalized["positions"] = clean_positions
    normalized["spent_today_usd"] = round(float(_safe_float(normalized.get("spent_today_usd")) or 0.0), 6)
    normalized["gross_buys_today_usd"] = round(
        float(_safe_float(normalized.get("gross_buys_today_usd")) or _safe_float(normalized.get("spent_today_usd")) or 0.0),
        6,
    )
    normalized["gross_sells_today_usd"] = round(float(_safe_float(normalized.get("gross_sells_today_usd")) or 0.0), 6)
    normalized["realized_pnl_today_usd"] = round(float(_safe_float(normalized.get("realized_pnl_today_usd")) or 0.0), 6)
    normalized["trades_today"] = int(_safe_float(normalized.get("trades_today")) or 0)
    normalized["closed_trades_today"] = int(_safe_float(normalized.get("closed_trades_today")) or 0)
    normalized["open_risk_usd"] = _recalculate_open_risk(clean_positions)
    return normalized


def load_autotrader_state(path: str, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    state = _default_state(now_utc=now_utc)
    target = Path(path)
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                state.update(data)
        except Exception:
            logger.warning("Failed to load autotrader state from %s", target)
    current_day = (now_utc or _utc_now()).date().isoformat()
    if str(state.get("trading_day")) != current_day:
        state = _default_state(now_utc=now_utc)
    return _normalize_state(state, now_utc=now_utc)


def save_autotrader_state(path: str, state: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _station_exposure_usd(
    state: Dict[str, Any],
    station_id: str,
    target_date: str,
    live_positions: Optional[List[Dict[str, Any]]] = None,
) -> float:
    total = 0.0
    for position in _merged_open_positions(state, live_positions=live_positions):
        if str(position.get("station_id")) != station_id:
            continue
        if str(position.get("target_date")) != target_date:
            continue
        total += float(_safe_float(position.get("cost_basis_open_usd")) or _safe_float(position.get("notional_usd")) or 0.0)
    return total


def _is_station_position_limit_enabled(config: AutoTraderConfig) -> bool:
    return int(config.max_station_positions) > 0


def _open_positions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, position in (state.get("positions") or {}).items():
        if not isinstance(position, dict):
            continue
        if str(position.get("status") or "OPEN").upper() != "OPEN":
            continue
        row = dict(position)
        row.setdefault("position_key", key)
        row.setdefault("source", "local_state")
        row["notional_usd"] = float(_safe_float(row.get("notional_usd")) or 0.0)
        row["cost_basis_open_usd"] = float(_safe_float(row.get("cost_basis_open_usd")) or row["notional_usd"])
        row["shares_open"] = float(_safe_float(row.get("shares_open")) or _safe_float(row.get("shares")) or 0.0)
        rows.append(row)
    rows.sort(key=lambda row: str(row.get("opened_at_utc") or ""), reverse=True)
    return rows


def _position_identity(row: Dict[str, Any]) -> str:
    station_id = str(row.get("station_id") or "").upper()
    target_date = str(row.get("target_date") or "")
    label = normalize_label(str(row.get("label") or row.get("title") or "")) or str(row.get("label") or row.get("title") or "")
    side = str(row.get("side") or "").upper()
    return f"{station_id}|{target_date}|{label}|{side}".upper()


def _merged_open_positions(
    state: Dict[str, Any],
    live_positions: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in _open_positions(state):
        identity = _position_identity(row)
        if not identity:
            continue
        prepared = dict(row)
        prepared["position_identity"] = identity
        merged[identity] = prepared
    for row in list(live_positions or []):
        if not isinstance(row, dict):
            continue
        identity = _position_identity(row)
        if not identity:
            continue
        if identity in merged:
            merged[identity].setdefault("current_value_usd", float(_safe_float(row.get("current_value_usd")) or 0.0))
            merged[identity].setdefault("avg_price", float(_safe_float(row.get("avg_price")) or 0.0))
            merged[identity].setdefault("current_price", float(_safe_float(row.get("current_price")) or 0.0))
            merged[identity].setdefault("source", "local_state")
            continue
        prepared = dict(row)
        prepared.setdefault("source", "polymarket_live")
        prepared.setdefault("position_key", identity)
        prepared["position_identity"] = identity
        prepared["cost_basis_open_usd"] = float(_safe_float(prepared.get("cost_basis_open_usd")) or _safe_float(prepared.get("initialValue")) or 0.0)
        prepared["notional_usd"] = float(_safe_float(prepared.get("cost_basis_open_usd")) or 0.0)
        prepared["shares_open"] = float(_safe_float(prepared.get("shares_open")) or _safe_float(prepared.get("shares")) or 0.0)
        merged[identity] = prepared
    rows = list(merged.values())
    rows.sort(
        key=lambda row: (
            str(row.get("opened_at_utc") or ""),
            float(_safe_float(row.get("current_value_usd")) or 0.0),
            float(_safe_float(row.get("cost_basis_open_usd")) or 0.0),
        ),
        reverse=True,
    )
    return rows


def _station_state_summary(
    state: Dict[str, Any],
    station_id: str,
    live_positions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized_station = str(station_id or "").upper()
    positions = [
        row
        for row in _merged_open_positions(state, live_positions=live_positions)
        if str(row.get("station_id") or "").upper() == normalized_station
    ]
    open_risk = sum(float(row.get("cost_basis_open_usd") or 0.0) for row in positions)
    last_trade = next((row.get("opened_at_utc") for row in positions if row.get("opened_at_utc")), None)
    strategies: Dict[str, int] = {}
    for row in positions:
        strategy = str(row.get("strategy") or ("external_live" if row.get("source") == "polymarket_live" else "terminal_value"))
        strategies[strategy] = strategies.get(strategy, 0) + 1
    return {
        "open_risk_usd": round(open_risk, 6),
        "open_positions": len(positions),
        "positions": positions[:6],
        "last_trade_utc": last_trade,
        "strategies": strategies,
    }


def _global_state_summary(
    state: Dict[str, Any],
    live_positions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    positions = _merged_open_positions(state, live_positions=live_positions)
    open_risk = sum(float(row.get("cost_basis_open_usd") or 0.0) for row in positions)
    last_trade = next(
        (
            row.get("opened_at_utc") or row.get("closed_at_utc")
            for row in positions
            if row.get("opened_at_utc") or row.get("closed_at_utc")
        ),
        state.get("last_trade_utc"),
    )
    strategies: Dict[str, int] = {}
    for row in positions:
        strategy = str(row.get("strategy") or ("external_live" if row.get("source") == "polymarket_live" else "terminal_value"))
        strategies[strategy] = strategies.get(strategy, 0) + 1
    return {
        "open_risk_usd": round(open_risk, 6),
        "open_positions": len(positions),
        "positions": positions[:10],
        "last_trade_utc": last_trade,
        "strategies": strategies,
    }


def _portfolio_summary(
    state: Dict[str, Any],
    config: AutoTraderConfig,
    *,
    live_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    live_positions = list((live_snapshot or {}).get("open_positions") or [])
    positions = _merged_open_positions(state, live_positions=live_positions)
    open_risk = round(sum(float(_safe_float(row.get("cost_basis_open_usd")) or 0.0) for row in positions), 6)
    gross_buys = float(_safe_float(state.get("gross_buys_today_usd")) or _safe_float(state.get("spent_today_usd")) or 0.0)
    gross_sells = float(_safe_float(state.get("gross_sells_today_usd")) or 0.0)
    realized_pnl = float(_safe_float(state.get("realized_pnl_today_usd")) or 0.0)
    realized_loss = max(0.0, -realized_pnl)
    live_value = float(_safe_float((live_snapshot or {}).get("portfolio_value_usd")) or 0.0)
    free_collateral = float(_safe_float((live_snapshot or {}).get("free_collateral_usd")) or 0.0)
    local_positions = _open_positions(state)
    return {
        "bankroll_usd": float(config.bankroll_usd),
        "open_risk_usd": open_risk,
        "available_exposure_usd": round(max(0.0, float(config.max_total_exposure_usd) - open_risk), 6),
        "free_collateral_usd": free_collateral,
        "portfolio_value_usd": live_value,
        "positions_market_value_usd": float(_safe_float((live_snapshot or {}).get("positions_market_value_usd")) or 0.0),
        "positions_cost_basis_usd": float(_safe_float((live_snapshot or {}).get("positions_cost_basis_usd")) or open_risk),
        "spent_today_usd": gross_buys,
        "gross_buys_today_usd": gross_buys,
        "gross_sells_today_usd": gross_sells,
        "realized_pnl_today_usd": realized_pnl,
        "remaining_daily_loss_usd": round(max(0.0, float(config.daily_loss_limit_usd) - realized_loss), 6),
        "open_positions": positions[:10],
        "open_positions_count": len(positions),
        "tracked_open_positions_count": len(local_positions),
        "external_open_positions_count": max(0, len(positions) - len(local_positions)),
        "trades_today": int(_safe_float(state.get("trades_today")) or 0),
        "closed_trades_today": int(_safe_float(state.get("closed_trades_today")) or 0),
        "last_trade_utc": state.get("last_trade_utc"),
    }


def _find_bracket(payload: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
    normalized_label = normalize_label(label) or label
    for row in (payload.get("brackets") or []):
        row_label = normalize_label(str(row.get("name") or row.get("bracket") or ""))
        if row_label == normalized_label:
            return row
    return None


def _count_station_open_positions(
    state: Dict[str, Any],
    station_id: str,
    live_positions: Optional[List[Dict[str, Any]]] = None,
) -> int:
    normalized_station = str(station_id or "").upper()
    return sum(
        1
        for row in _merged_open_positions(state, live_positions=live_positions)
        if str(row.get("station_id") or "").upper() == normalized_station
    )


def _build_candidate_from_trade(
    payload: Dict[str, Any],
    config: AutoTraderConfig,
    state: Dict[str, Any],
    *,
    trade_row: Dict[str, Any],
    strategy: str,
    live_positions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[AutoTradeCandidate], List[str]]:
    trading = (payload or {}).get("trading") or {}
    forecast = trading.get("forecast_winner") or {}
    forecast_label = str(forecast.get("label") or "") or None
    forecast_edge = _safe_float(forecast.get("edge_points"))

    tactical = strategy == "tactical_reprice"
    policy_key = "tactical_policy" if tactical else "terminal_policy"
    recommendation_key = "tactical_recommendation" if tactical else "recommendation"
    side_key = "tactical_best_side" if tactical else "best_side"
    edge_key = "tactical_edge_points" if tactical else "edge_points"
    entry_key = "selected_tactical_entry" if tactical else "selected_entry"
    fair_key = "selected_tactical_fair" if tactical else "selected_fair"
    score_key = "tactical_policy_score" if tactical else "policy_score"

    policy = trade_row.get(policy_key) or {}
    if not policy.get("allowed"):
        return None, ["policy_blocked"]

    recommendation = str(trade_row.get(recommendation_key) or "")
    if config.require_buy_recommendation and not recommendation.startswith("BUY_"):
        return None, ["recommendation_not_buy"]

    label = str(trade_row.get("label") or "")
    side = str(trade_row.get(side_key) or trade_row.get("best_side") or "YES").upper()
    edge_points = float(_safe_float(trade_row.get(edge_key)) or 0.0)
    market_price = _safe_float(trade_row.get(entry_key))
    fair_price = _safe_float(trade_row.get(fair_key))
    if not label or market_price is None or fair_price is None:
        return None, ["missing_trade_fields"]

    reasons: List[str] = []
    min_edge_points = float(config.tactical_min_edge_points if tactical else config.min_edge_points)
    if edge_points < min_edge_points:
        reasons.append("edge_too_small")
    if fair_price < float(config.min_fair_probability):
        reasons.append("fair_too_small")
    if market_price > float(config.max_entry_price):
        reasons.append("entry_too_expensive")

    if side == "YES" and forecast_label and label != forecast_label:
        required_edge = float(forecast_edge or 0.0) + float(config.secondary_yes_edge_bonus_pts)
        if edge_points < required_edge:
            reasons.append("secondary_yes_not_good_enough")

    bracket = _find_bracket(payload, label)
    if not bracket:
        reasons.append("missing_bracket")

    token_id = ""
    if bracket:
        token_id = str((bracket.get("yes_token_id") if side == "YES" else bracket.get("no_token_id")) or "").strip()
        if not token_id:
            reasons.append("missing_token_id")

    target_date = str(payload.get("target_date") or "")
    position_key = _position_key(str(payload.get("station_id") or ""), target_date, label, side)
    positions = state.get("positions") or {}
    if position_key in positions and str((positions[position_key] or {}).get("status") or "OPEN").upper() == "OPEN":
        reasons.append("duplicate_position")
    position_identity = _position_identity(
        {
            "station_id": str(payload.get("station_id") or ""),
            "target_date": target_date,
            "label": label,
            "side": side,
        }
    )
    merged_identities = {
        str(row.get("position_identity") or _position_identity(row))
        for row in _merged_open_positions(state, live_positions=live_positions)
        if isinstance(row, dict)
    }
    if position_identity in merged_identities:
        reasons.append("duplicate_position")

    next_projection = ((trading.get("tactical_context") or {}).get("next_metar") or {}) if tactical else {}
    tactical_alignment = "neutral"
    counterpart = (trading.get("best_terminal_trade") or {}) if tactical else (trading.get("best_tactical_trade") or {})
    counterpart_label = str(counterpart.get("label") or "")
    counterpart_side = str(counterpart.get("tactical_best_side") or counterpart.get("best_side") or "").upper()
    if counterpart_label and counterpart_side:
        if counterpart_label == label and counterpart_side == side:
            tactical_alignment = "aligned"
        elif counterpart_label == label and counterpart_side != side:
            tactical_alignment = "conflict"

    if tactical:
        minutes_to_next = _safe_float(next_projection.get("minutes_to_next"))
        next_direction = str(next_projection.get("direction") or "FLAT").upper()
        delta_market = abs(float(_safe_float(next_projection.get("delta_market")) or 0.0))
        repricing_influence = float(_safe_float(((trading.get("tactical_context") or {}).get("repricing_influence"))) or 0.0)
        market_unit = str(trading.get("market_unit") or payload.get("market_unit") or "F").upper()
        minimum_delta = 0.2 if market_unit == "C" else 0.35

        if minutes_to_next is None:
            reasons.append("missing_next_official")
        else:
            if minutes_to_next < float(config.tactical_min_minutes_to_next):
                reasons.append("too_close_to_next_official")
            if minutes_to_next > float(config.tactical_max_minutes_to_next):
                reasons.append("too_early_for_tactical")
        if next_direction == "FLAT":
            reasons.append("no_tactical_pressure")
        if delta_market < minimum_delta:
            reasons.append("repricing_too_small")
        if repricing_influence < 0.18:
            reasons.append("repricing_quality_too_low")

    if reasons:
        return None, reasons

    return AutoTradeCandidate(
        station_id=str(payload.get("station_id") or ""),
        target_date=target_date,
        target_day=int(payload.get("target_day") or config.target_day),
        label=label,
        side=side,
        token_id=token_id,
        market_price=float(market_price),
        fair_price=float(fair_price),
        edge_points=edge_points,
        model_probability=float(fair_price),
        recommendation=recommendation,
        policy_reason=str(trade_row.get("policy_reason") or trade_row.get("tactical_policy_reason") or policy.get("summary") or ""),
        forecast_winner_label=forecast_label,
        forecast_edge_points=forecast_edge,
        tactical_alignment=tactical_alignment,
        why=str(policy.get("summary") or trade_row.get("policy_reason") or ""),
        position_key=position_key,
        strategy=strategy,
        strategy_score=float(_safe_float(trade_row.get(score_key)) or 0.0),
        opened_next_obs_utc=str(next_projection.get("next_obs_utc") or "") if tactical else None,
        raw_row=dict(trade_row),
    ), []


def evaluate_trade_candidate(
    payload: Dict[str, Any],
    config: AutoTraderConfig,
    state: Optional[Dict[str, Any]] = None,
    *,
    live_positions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[AutoTradeCandidate], List[str]]:
    trading = (payload or {}).get("trading") or {}
    market_status = (payload or {}).get("market_status") or {}
    if not trading or not trading.get("available"):
        return None, ["signal_unavailable"]
    if bool(market_status.get("event_closed")):
        return None, ["event_closed"]
    if bool(market_status.get("is_mature")):
        return None, ["event_mature"]

    normalized_state = _normalize_state(state or {})
    if len(_merged_open_positions(normalized_state, live_positions=live_positions)) >= int(config.max_open_positions):
        return None, ["portfolio_full"]
    if _is_station_position_limit_enabled(config) and _count_station_open_positions(
        normalized_state,
        str(payload.get("station_id") or ""),
        live_positions=live_positions,
    ) >= int(config.max_station_positions):
        return None, ["station_full"]

    terminal_candidate = None
    terminal_reasons: List[str] = []
    if config.allow_terminal_value:
        terminal_candidate, terminal_reasons = _build_candidate_from_trade(
            payload,
            config,
            normalized_state,
            trade_row=dict(trading.get("best_terminal_trade") or {}),
            strategy="terminal_value",
            live_positions=live_positions,
        )

    tactical_candidate = None
    tactical_reasons: List[str] = []
    if config.allow_tactical_reprice and int(payload.get("target_day") or config.target_day) == 0:
        tactical_candidate, tactical_reasons = _build_candidate_from_trade(
            payload,
            config,
            normalized_state,
            trade_row=dict(trading.get("best_tactical_trade") or {}),
            strategy="tactical_reprice",
            live_positions=live_positions,
        )

    if terminal_candidate and tactical_candidate:
        same_trade = (
            terminal_candidate.label == tactical_candidate.label
            and terminal_candidate.side == tactical_candidate.side
        )
        if same_trade:
            tactical_candidate = None
            tactical_reasons = ["tactical_duplicates_terminal"]
        else:
            tactical_needs_clear_win = (
                tactical_candidate.strategy_score < (terminal_candidate.strategy_score * 1.18)
                and tactical_candidate.edge_points < (terminal_candidate.edge_points + 4.0)
            )
            if tactical_needs_clear_win:
                tactical_candidate = None
                tactical_reasons = ["terminal_better_priced"]

    candidates = [candidate for candidate in (terminal_candidate, tactical_candidate) if candidate is not None]
    if not candidates:
        combined = terminal_reasons + tactical_reasons
        return None, combined or ["no_candidate"]

    candidates.sort(
        key=lambda candidate: (
            candidate.strategy == "terminal_value",
            float(candidate.strategy_score),
            float(candidate.edge_points),
            float(candidate.fair_price),
        ),
        reverse=True,
    )
    return candidates[0], []


def compute_trade_budget_usd(
    candidate: AutoTradeCandidate,
    config: AutoTraderConfig,
    state: Dict[str, Any],
    *,
    available_balance_usd: Optional[float] = None,
    live_positions: Optional[List[Dict[str, Any]]] = None,
) -> float:
    realized_pnl_today = float(_safe_float(state.get("realized_pnl_today_usd")) or 0.0)
    realized_loss_today = max(0.0, -realized_pnl_today)
    remaining_daily_loss = float(config.daily_loss_limit_usd) - realized_loss_today
    merged_positions = _merged_open_positions(state, live_positions=live_positions)
    merged_open_risk = sum(float(_safe_float(row.get("cost_basis_open_usd")) or 0.0) for row in merged_positions)
    remaining_total_exposure = float(config.max_total_exposure_usd) - float(merged_open_risk)
    open_positions_count = len(merged_positions)
    station_positions_count = _count_station_open_positions(state, candidate.station_id, live_positions=live_positions)

    if int(_safe_float(state.get("trades_today")) or 0) >= int(config.max_trades_per_day):
        return 0.0
    if open_positions_count >= int(config.max_open_positions):
        return 0.0
    if _is_station_position_limit_enabled(config) and station_positions_count >= int(config.max_station_positions):
        return 0.0

    last_trade_raw = str(state.get("last_trade_utc") or "").strip()
    if last_trade_raw:
        try:
            last_trade_dt = datetime.fromisoformat(last_trade_raw.replace("Z", "+00:00"))
            seconds_since = (_utc_now() - last_trade_dt.astimezone(UTC)).total_seconds()
            if seconds_since < float(config.cooldown_seconds):
                return 0.0
        except Exception:
            pass

    effective_cap = min(
        float(config.max_trade_usd),
        remaining_daily_loss,
        remaining_total_exposure,
    )
    if available_balance_usd is not None:
        effective_cap = min(effective_cap, float(available_balance_usd))
    if effective_cap <= 0:
        return 0.0

    fair = _clamp(float(candidate.fair_price), 0.001, 0.999)
    price = _clamp(float(candidate.market_price), 0.001, 0.999)
    kelly = max(0.0, (fair - price) / max(0.05, 1.0 - price))
    raw_size = float(config.bankroll_usd) * float(config.fractional_kelly) * kelly

    if candidate.strategy == "tactical_reprice":
        raw_size *= 0.85

    if candidate.tactical_alignment == "aligned":
        raw_size *= 1.15
    elif candidate.tactical_alignment == "conflict":
        raw_size *= 0.65

    if raw_size > 0.0:
        raw_size = max(raw_size, float(config.min_trade_usd))

    return round(max(0.0, min(effective_cap, raw_size)), 6)


def apply_orderbook_guardrails(
    candidate: AutoTradeCandidate,
    book: TopOfBook,
    config: AutoTraderConfig,
    budget_usd: float,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    ask = _safe_float(book.best_ask)
    if ask is None or ask <= 0:
        return None, ["no_live_ask"]

    reasons: List[str] = []
    if ask > float(config.max_entry_price):
        reasons.append("live_ask_too_expensive")
    if ask > float(candidate.fair_price) - (float(config.min_edge_points) / 100.0):
        reasons.append("live_edge_too_small")

    spread_points = (_safe_float(book.spread) or 0.0) * 100.0
    if spread_points > float(config.max_spread_points):
        reasons.append("spread_too_wide")

    ask_size = float(_safe_float(book.ask_size) or 0.0)
    if ask_size < float(config.min_ask_depth_contracts):
        reasons.append("ask_depth_too_small")

    max_payup = min(
        float(candidate.fair_price) - (float(config.min_edge_points) / 100.0),
        float(ask) + (float(config.limit_slippage_points) / 100.0),
    )
    if max_payup < float(ask):
        reasons.append("no_price_headroom")
        return None, reasons

    shares_from_budget = float(budget_usd) / max(0.0001, float(max_payup))
    shares_cap = ask_size * float(config.max_depth_participation)
    size = quantize_buy_order_size(min(shares_from_budget, shares_cap), max_payup)
    min_order_size = float(_safe_float(book.min_order_size) or 0.0)
    if min_order_size > 0 and size < min_order_size:
        reasons.append("below_min_order_size")

    notional_usd = quantize_market_buy_amount(size * float(max_payup))
    if notional_usd < float(config.min_trade_usd):
        reasons.append("trade_too_small")

    if reasons:
        return None, reasons

    return {
        "token_id": candidate.token_id,
        "limit_price": round(float(max_payup), 6),
        "size": round(float(size), 6),
        "notional_usd": round(float(notional_usd), 6),
        "spread_points": round(float(spread_points), 4),
        "ask_size": round(float(ask_size), 6),
        "best_ask": round(float(ask), 6),
        "max_payup_price": round(float(max_payup), 6),
    }, []


def apply_exit_guardrails(
    position: Dict[str, Any],
    book: TopOfBook,
    config: AutoTraderConfig,
    *,
    target_floor_price: Optional[float] = None,
    force_exit: bool = False,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    bid = _safe_float(book.best_bid)
    if bid is None or bid <= 0:
        return None, ["no_live_bid"]

    shares_open = float(_safe_float(position.get("shares_open")) or _safe_float(position.get("shares")) or 0.0)
    if shares_open <= 0:
        return None, ["no_open_shares"]

    reasons: List[str] = []
    bid_size = float(_safe_float(book.bid_size) or 0.0)
    size_cap = shares_open
    if bid_size > 0:
        size_cap = max(0.0, bid_size * float(config.max_depth_participation))
    size = quantize_share_size(min(shares_open, size_cap) if size_cap > 0 else shares_open, 2)

    min_order_size = float(_safe_float(book.min_order_size) or 0.0)
    if min_order_size > 0 and size < min_order_size:
        reasons.append("below_min_order_size")

    if size <= 0:
        reasons.append("no_bid_depth")

    slippage_floor = max(0.01, float(bid) - (float(config.limit_slippage_points) / 100.0))
    if force_exit:
        min_price = slippage_floor
    else:
        if target_floor_price is not None and float(bid) + 1e-9 < float(target_floor_price):
            reasons.append("bid_below_exit_floor")
        desired_floor = min(float(bid), float(target_floor_price)) if target_floor_price is not None else float(bid)
        min_price = max(slippage_floor, desired_floor)

    if reasons:
        return None, reasons

    proceeds_usd = float(size) * float(bid)
    return {
        "token_id": str(position.get("token_id") or ""),
        "size": round(float(size), 6),
        "best_bid": round(float(bid), 6),
        "min_price": round(float(min_price), 6),
        "proceeds_usd": round(float(proceeds_usd), 6),
        "bid_size": round(float(bid_size), 6),
    }, []


def _tail_jsonl(path: str, limit: int = 8) -> List[Dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()[-limit:]
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


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
        last_trade_price=_safe_float(bracket.get(f"ws_{key}_mid")),
        raw=bracket,
    )


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


class AutoTrader:
    def __init__(
        self,
        config: Optional[AutoTraderConfig] = None,
        execution: Optional[PolymarketExecutionClient] = None,
    ):
        self.config = config or load_autotrader_config_from_env()
        self.execution = execution or PolymarketExecutionClient(load_polymarket_execution_config_from_env())
        self.state = load_autotrader_state(self.config.state_path)

    def _append_log(self, event: Dict[str, Any]) -> None:
        path = Path(self.config.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(event)
        record.setdefault("ts_utc", _utc_now().isoformat())
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _persist_state(self) -> None:
        self.state = _normalize_state(self.state)
        save_autotrader_state(self.config.state_path, self.state)

    def _sync_live_positions_into_state(self, live_positions: Optional[List[Dict[str, Any]]] = None) -> None:
        positions = self.state.setdefault("positions", {})
        if not isinstance(positions, dict):
            self.state["positions"] = {}
            positions = self.state["positions"]

        seen_identities = set()
        now_iso = _utc_now().isoformat()
        for row in list(live_positions or []):
            if not isinstance(row, dict):
                continue
            identity = str(row.get("position_identity") or _position_identity(row))
            if not identity:
                continue
            seen_identities.add(identity)
            position_key = str(row.get("position_key") or identity)
            existing = dict(positions.get(position_key) or positions.get(identity) or {})
            strategy = str(existing.get("strategy") or "external_live")
            fills = list(existing.get("fills") or [])
            positions[position_key] = {
                "position_identity": identity,
                "station_id": str(row.get("station_id") or existing.get("station_id") or "").upper(),
                "target_date": str(row.get("target_date") or existing.get("target_date") or ""),
                "label": str(row.get("label") or existing.get("label") or row.get("title") or ""),
                "title": str(row.get("title") or existing.get("title") or ""),
                "side": str(row.get("side") or existing.get("side") or "").upper(),
                "token_id": str(row.get("token_id") or existing.get("token_id") or ""),
                "status": "OPEN",
                "mode": str(existing.get("mode") or "live"),
                "strategy": strategy,
                "source": "polymarket_live",
                "strategy_score": float(_safe_float(existing.get("strategy_score")) or 0.0),
                "entry_price": float(_safe_float(row.get("entry_price")) or _safe_float(row.get("avg_price")) or _safe_float(existing.get("entry_price")) or 0.0),
                "avg_price": float(_safe_float(row.get("avg_price")) or _safe_float(existing.get("avg_price")) or 0.0),
                "current_price": float(_safe_float(row.get("current_price")) or _safe_float(existing.get("current_price")) or 0.0),
                "shares": float(_safe_float(row.get("shares")) or _safe_float(row.get("shares_open")) or _safe_float(existing.get("shares")) or 0.0),
                "shares_open": float(_safe_float(row.get("shares_open")) or _safe_float(row.get("shares")) or _safe_float(existing.get("shares_open")) or 0.0),
                "notional_usd": float(_safe_float(row.get("cost_basis_open_usd")) or _safe_float(existing.get("notional_usd")) or 0.0),
                "cost_basis_open_usd": float(_safe_float(row.get("cost_basis_open_usd")) or _safe_float(existing.get("cost_basis_open_usd")) or 0.0),
                "current_value_usd": float(_safe_float(row.get("current_value_usd")) or _safe_float(existing.get("current_value_usd")) or 0.0),
                "cash_pnl_usd": float(_safe_float(row.get("cash_pnl_usd")) or _safe_float(existing.get("cash_pnl_usd")) or 0.0),
                "realized_pnl_usd": float(_safe_float(row.get("realized_pnl_usd")) or _safe_float(existing.get("realized_pnl_usd")) or 0.0),
                "opened_at_utc": str(existing.get("opened_at_utc") or row.get("opened_at_utc") or now_iso),
                "opened_next_obs_utc": str(existing.get("opened_next_obs_utc") or ""),
                "thesis": str(existing.get("thesis") or "Imported from live Polymarket portfolio."),
                "fills": fills,
                "result": existing.get("result") or {},
                "live_synced_at_utc": now_iso,
            }

        for key, position in list(positions.items()):
            if not isinstance(position, dict):
                continue
            if str(position.get("source") or "") != "polymarket_live":
                continue
            identity = str(position.get("position_identity") or _position_identity(position))
            if identity in seen_identities:
                positions[key]["position_identity"] = identity
                continue
            if str(position.get("status") or "OPEN").upper() != "OPEN":
                continue
            positions[key]["status"] = "CLOSED"
            positions[key]["shares_open"] = 0.0
            positions[key]["cost_basis_open_usd"] = 0.0
            positions[key]["closed_at_utc"] = now_iso

    def _position_live_book(self, position: Dict[str, Any], payload: Dict[str, Any]) -> Optional[TopOfBook]:
        bracket = _find_bracket(payload, str(position.get("label") or ""))
        if self.config.prefer_ws_book and bracket:
            book = _book_from_bracket(bracket, str(position.get("side") or "YES"))
            if book and book.best_bid is not None:
                return book
        token_id = str(position.get("token_id") or "").strip()
        if not token_id:
            return None
        return self.execution.get_top_of_book(token_id)

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

        if fair_now is not None and current_best_bid >= target_floor and current_best_bid > entry_price + 0.0025:
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
            elif not policy_now.get("allowed", True) and current_best_bid >= stop_floor:
                force_exit = True
                reason = "policy_flip"
            elif current_best_bid <= stop_floor and edge_now_points is not None and edge_now_points <= 0.0:
                force_exit = True
                reason = "stop_loss"

        if not reason:
            return None, ["hold"]

        exit_plan, exit_reasons = apply_exit_guardrails(
            position,
            book,
            self.config,
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
        return exit_plan, reasons

    def _persist_trade(self, candidate: AutoTradeCandidate, plan: Dict[str, Any], result: Dict[str, Any], *, mode: str) -> None:
        now_iso = _utc_now().isoformat()
        self.state["positions"][candidate.position_key] = {
            "station_id": candidate.station_id,
            "target_date": candidate.target_date,
            "label": candidate.label,
            "side": candidate.side,
            "token_id": candidate.token_id,
            "status": "OPEN",
            "mode": mode,
            "strategy": candidate.strategy,
            "strategy_score": round(float(candidate.strategy_score), 6),
            "entry_price": plan["limit_price"],
            "shares": plan["size"],
            "shares_open": plan["size"],
            "notional_usd": plan["notional_usd"],
            "cost_basis_open_usd": plan["notional_usd"],
            "opened_at_utc": now_iso,
            "opened_next_obs_utc": candidate.opened_next_obs_utc,
            "thesis": candidate.why,
            "realized_pnl_usd": 0.0,
            "fills": [],
            "result": result,
        }
        gross_buys = float(_safe_float(self.state.get("gross_buys_today_usd")) or _safe_float(self.state.get("spent_today_usd")) or 0.0)
        self.state["gross_buys_today_usd"] = round(gross_buys + float(plan["notional_usd"]), 6)
        self.state["spent_today_usd"] = self.state["gross_buys_today_usd"]
        self.state["trades_today"] = int(_safe_float(self.state.get("trades_today")) or 0) + 1
        self.state["last_trade_utc"] = now_iso
        self.state["open_risk_usd"] = _recalculate_open_risk(self.state.get("positions") or {})
        self._persist_state()

    def _persist_exit(
        self,
        position_key: str,
        exit_plan: Dict[str, Any],
        result: Dict[str, Any],
        *,
        mode: str,
    ) -> Dict[str, Any]:
        position = dict((self.state.get("positions") or {}).get(position_key) or {})
        if not position:
            return {}

        shares_open = float(_safe_float(position.get("shares_open")) or _safe_float(position.get("shares")) or 0.0)
        size = min(shares_open, float(_safe_float(exit_plan.get("size")) or 0.0))
        if size <= 0:
            return {}

        entry_price = float(_safe_float(position.get("entry_price")) or 0.0)
        cost_basis_open = float(_safe_float(position.get("cost_basis_open_usd")) or _safe_float(position.get("notional_usd")) or (shares_open * entry_price))
        average_cost = cost_basis_open / shares_open if shares_open > 0 else entry_price
        cost_basis_closed = average_cost * size
        proceeds_usd = float(_safe_float(exit_plan.get("proceeds_usd")) or 0.0)
        realized_pnl = proceeds_usd - cost_basis_closed
        remaining_shares = max(0.0, shares_open - size)
        remaining_cost_basis = max(0.0, cost_basis_open - cost_basis_closed)
        now_iso = _utc_now().isoformat()

        fill = {
            "closed_at_utc": now_iso,
            "mode": mode,
            "reason": exit_plan.get("reason"),
            "size": round(size, 6),
            "price": round(float(_safe_float(exit_plan.get("best_bid")) or 0.0), 6),
            "proceeds_usd": round(proceeds_usd, 6),
            "realized_pnl_usd": round(realized_pnl, 6),
            "result": result,
        }

        position["fills"] = list(position.get("fills") or []) + [fill]
        position["shares_open"] = round(remaining_shares, 6)
        position["cost_basis_open_usd"] = round(remaining_cost_basis, 6)
        position["realized_pnl_usd"] = round(float(_safe_float(position.get("realized_pnl_usd")) or 0.0) + realized_pnl, 6)
        position["last_exit_reason"] = exit_plan.get("reason")
        position["last_exit_utc"] = now_iso
        if remaining_shares <= 1e-9:
            position["status"] = "CLOSED"
            position["closed_at_utc"] = now_iso

        self.state["positions"][position_key] = position
        self.state["gross_sells_today_usd"] = round(float(_safe_float(self.state.get("gross_sells_today_usd")) or 0.0) + proceeds_usd, 6)
        self.state["realized_pnl_today_usd"] = round(float(_safe_float(self.state.get("realized_pnl_today_usd")) or 0.0) + realized_pnl, 6)
        self.state["closed_trades_today"] = int(_safe_float(self.state.get("closed_trades_today")) or 0) + 1
        self.state["last_trade_utc"] = now_iso
        self.state["open_risk_usd"] = _recalculate_open_risk(self.state.get("positions") or {})
        self._persist_state()
        return {
            "position_key": position_key,
            "remaining_shares": round(remaining_shares, 6),
            "realized_pnl_usd": round(realized_pnl, 6),
            "reason": exit_plan.get("reason"),
            "mode": mode,
        }

    def get_status_snapshot(self, station_id: Optional[str] = None) -> Dict[str, Any]:
        self.state = load_autotrader_state(self.config.state_path)
        execution_config = self.execution.config
        recent_events = _tail_jsonl(self.config.log_path, limit=8)
        live_snapshot: Dict[str, Any] = {}
        live_snapshot_error = ""
        if execution_config.is_live_ready:
            try:
                live_snapshot = self.execution.get_live_portfolio_snapshot()
            except Exception as exc:
                live_snapshot_error = str(exc)
        live_positions = list(live_snapshot.get("open_positions") or [])
        station_focus = str(station_id or "").upper()
        station_state = (
            _station_state_summary(self.state, station_focus, live_positions=live_positions)
            if station_focus else _global_state_summary(self.state, live_positions=live_positions)
        )
        return {
            "enabled": bool(self.config.enabled),
            "mode": self.config.mode,
            "order_mode": self.config.order_mode,
            "prefer_ws_book": bool(self.config.prefer_ws_book),
            "stations": list(self.config.station_ids or []),
            "target_day": int(self.config.target_day),
            "target_date": str(self.config.target_date or ""),
            "loop_seconds": int(self.config.loop_seconds),
            "risk": {
                "bankroll_usd": float(self.config.bankroll_usd),
                "max_trade_usd": float(self.config.max_trade_usd),
                "max_total_exposure_usd": float(self.config.max_total_exposure_usd),
                "max_station_exposure_usd": float(self.config.max_station_exposure_usd),
                "daily_loss_limit_usd": float(self.config.daily_loss_limit_usd),
                "max_open_positions": int(self.config.max_open_positions),
                "max_station_positions": int(self.config.max_station_positions),
                "station_limits_enabled": _is_station_position_limit_enabled(self.config),
            },
            "strategy": {
                "terminal_enabled": bool(self.config.allow_terminal_value),
                "tactical_enabled": bool(self.config.allow_tactical_reprice),
                "tactical_window_minutes": [
                    float(self.config.tactical_min_minutes_to_next),
                    float(self.config.tactical_max_minutes_to_next),
                ],
                "terminal_take_profit_pts": float(self.config.terminal_take_profit_points),
                "tactical_take_profit_pts": float(self.config.tactical_take_profit_points),
                "terminal_stop_loss_pts": float(self.config.terminal_stop_loss_points),
                "tactical_stop_loss_pts": float(self.config.tactical_stop_loss_points),
                "tactical_timeout_minutes": int(self.config.tactical_timeout_minutes),
            },
            "wallet": {
                "live_ready": bool(execution_config.is_live_ready),
                "has_private_key": bool(execution_config.has_private_key),
                "has_api_creds": bool(execution_config.has_api_creds),
                "signature_type": int(execution_config.signature_type),
                "has_funder": bool(str(execution_config.funder or "").strip()),
                "account_user": live_snapshot.get("user"),
                "free_collateral_usd": float(_safe_float(live_snapshot.get("free_collateral_usd")) or 0.0),
                "portfolio_value_usd": float(_safe_float(live_snapshot.get("portfolio_value_usd")) or 0.0),
                "collateral_allowance_ready": bool(live_snapshot.get("collateral_allowance_ready")),
                "error": live_snapshot_error or None,
            },
            "portfolio": _portfolio_summary(self.state, self.config, live_snapshot=live_snapshot),
            "state": {
                "trading_day": self.state.get("trading_day"),
                "spent_today_usd": float(_safe_float(self.state.get("spent_today_usd")) or 0.0),
                "realized_pnl_today_usd": float(_safe_float(self.state.get("realized_pnl_today_usd")) or 0.0),
                "open_risk_usd": float(station_state.get("open_risk_usd") or 0.0),
                "trades_today": int(_safe_float(self.state.get("trades_today")) or 0),
                "closed_trades_today": int(_safe_float(self.state.get("closed_trades_today")) or 0),
                "last_trade_utc": station_state.get("last_trade_utc") or self.state.get("last_trade_utc"),
                "open_positions": int(station_state.get("open_positions") or 0),
                "positions": station_state.get("positions") or [],
                "strategies": station_state.get("strategies") or {},
            },
            "kill_switch_active": Path(self.config.kill_switch_path).exists(),
            "recent_events": recent_events,
        }

    async def _manage_open_positions(self, station_id: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        target_date = str(payload.get("target_date") or "")
        for position_key, position in list((self.state.get("positions") or {}).items()):
            if not isinstance(position, dict):
                continue
            if str(position.get("station_id") or "").upper() != str(station_id or "").upper():
                continue
            if str(position.get("target_date") or "") != target_date:
                continue
            if str(position.get("status") or "OPEN").upper() != "OPEN":
                continue

            try:
                live_book = self._position_live_book(position, payload)
                exit_plan, exit_reasons = self._evaluate_exit_decision(position, payload, live_book)
                if not exit_plan:
                    continue

                if not self.config.is_live:
                    response = {"mode": "paper"}
                    persisted = self._persist_exit(position_key, exit_plan, response, mode="paper")
                    event = {
                        "station_id": station_id,
                        "status": "paper_exit",
                        "position_key": position_key,
                        "position": position,
                        "exit_plan": exit_plan,
                        "persisted": persisted,
                    }
                    self._append_log(event)
                    events.append(event)
                    continue

                self.execution.ensure_conditional_ready(str(position.get("token_id") or ""), float(exit_plan["size"]))
                if self.config.order_mode == "market_fok":
                    response = self.execution.place_market_sell(
                        token_id=str(position.get("token_id") or ""),
                        size=float(exit_plan["size"]),
                        min_price=float(exit_plan["min_price"]),
                    )
                else:
                    response = self.execution.place_limit_sell(
                        token_id=str(position.get("token_id") or ""),
                        price=float(exit_plan["min_price"]),
                        size=float(exit_plan["size"]),
                    )
                summary = self.execution.summarize_order_response(
                    response,
                    fallback_size=float(exit_plan["size"]),
                    fallback_price=float(exit_plan["best_bid"]),
                    fallback_notional=float(exit_plan["proceeds_usd"]),
                    assume_full_on_matched=True,
                )
                if float(summary.get("matched_size") or 0.0) <= 0.0:
                    event = {
                        "station_id": station_id,
                        "status": "live_exit_unfilled",
                        "position_key": position_key,
                        "position": position,
                        "exit_plan": exit_plan,
                        "summary": summary,
                        "response": response,
                    }
                    self._append_log(event)
                    events.append(event)
                    continue

                reconciled_plan = dict(exit_plan)
                reconciled_plan["size"] = float(summary["matched_size"])
                reconciled_plan["proceeds_usd"] = float(summary["matched_notional"])
                persisted = self._persist_exit(position_key, reconciled_plan, {"response": response, "summary": summary}, mode="live")
                event = {
                    "station_id": station_id,
                    "status": "live_exit_sent",
                    "position_key": position_key,
                    "position": position,
                    "exit_plan": reconciled_plan,
                    "persisted": persisted,
                    "summary": summary,
                    "response": response,
                }
                self._append_log(event)
                events.append(event)
            except Exception as exc:
                logger.exception("Autotrader exit failed for %s %s", station_id, position_key)
                event = {
                    "station_id": station_id,
                    "status": "exit_error",
                    "position_key": position_key,
                    "error": str(exc),
                }
                self._append_log(event)
                events.append(event)
        return events

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

    async def run_station_once(
        self,
        station_id: str,
        target_day: Optional[int] = None,
        *,
        target_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_target_day, resolved_target_date = resolve_station_market_target(
            station_id,
            self.config,
            override_target_day=target_day,
            override_target_date=target_date,
        )
        payload = await self._fetch_signal_payload(
            station_id,
            resolved_target_day,
            target_date=resolved_target_date,
        )
        live_snapshot: Dict[str, Any] = {}
        live_positions: List[Dict[str, Any]] = []
        available_balance_usd = None
        live_snapshot_ok = False
        if self.config.is_live:
            try:
                live_snapshot = self.execution.get_live_portfolio_snapshot(force_refresh=True)
                live_positions = list(live_snapshot.get("open_positions") or [])
                available_balance_usd = _safe_float(live_snapshot.get("free_collateral_usd"))
                live_snapshot_ok = True
            except Exception:
                live_snapshot = {}
                live_positions = []
                available_balance_usd = None
        async with _get_portfolio_lock():
            self.state = load_autotrader_state(self.config.state_path)
            if live_snapshot_ok:
                self._sync_live_positions_into_state(live_positions)
                self._persist_state()
            exit_events = await self._manage_open_positions(station_id, payload)
            candidate, reasons = evaluate_trade_candidate(
                payload,
                self.config,
                self.state,
                live_positions=live_positions,
            )
            if not candidate:
                status = "hold" if exit_events else "skip"
                result = {"station_id": station_id, "status": status, "reasons": reasons, "exits": exit_events}
                self._append_log(result)
                return result

        live_book = None
        if self.config.prefer_ws_book:
            for row in (payload.get("brackets") or []):
                row_label = normalize_label(str(row.get("name") or row.get("bracket") or ""))
                if row_label == normalize_label(candidate.label):
                    live_book = _book_from_bracket(row, candidate.side)
                    break
        if live_book is None or live_book.best_ask is None:
            live_book = self.execution.get_top_of_book(candidate.token_id)

        async with _get_portfolio_lock():
            self.state = load_autotrader_state(self.config.state_path)
            if live_snapshot_ok:
                self._sync_live_positions_into_state(live_positions)
                self._persist_state()
            merged_position_identities = {
                str(row.get("position_identity") or _position_identity(row))
                for row in _merged_open_positions(self.state, live_positions=live_positions)
                if isinstance(row, dict)
            }
            candidate_identity = _position_identity(
                {
                    "station_id": candidate.station_id,
                    "target_date": candidate.target_date,
                    "label": candidate.label,
                    "side": candidate.side,
                }
            )
            if candidate.position_key in (self.state.get("positions") or {}) or candidate_identity in merged_position_identities:
                result = {"station_id": station_id, "status": "skip", "reasons": ["duplicate_position"], "candidate": asdict(candidate), "exits": exit_events}
                self._append_log(result)
                return result

            budget_usd = compute_trade_budget_usd(
                candidate,
                self.config,
                self.state,
                available_balance_usd=available_balance_usd,
                live_positions=live_positions,
            )
            if budget_usd < float(self.config.min_trade_usd):
                result = {"station_id": station_id, "status": "skip", "reasons": ["budget_too_small"], "candidate": asdict(candidate), "exits": exit_events}
                self._append_log(result)
                return result

            plan, plan_reasons = apply_orderbook_guardrails(candidate, live_book, self.config, budget_usd)
            if not plan:
                result = {"station_id": station_id, "status": "skip", "reasons": plan_reasons, "candidate": asdict(candidate), "exits": exit_events}
                self._append_log(result)
                return result

            if not self.config.is_live:
                result = {"station_id": station_id, "status": "paper_fill", "candidate": asdict(candidate), "plan": plan, "exits": exit_events}
                self._persist_trade(candidate, plan, {"mode": "paper"}, mode="paper")
                self._append_log(result)
                return result

            try:
                self.execution.ensure_collateral_ready(float(plan["notional_usd"]))
                if self.config.order_mode == "market_fok":
                    response = self.execution.place_market_buy(
                        token_id=plan["token_id"],
                        amount_usd=float(plan["notional_usd"]),
                        max_price=float(plan["max_payup_price"]),
                    )
                else:
                    response = self.execution.place_limit_buy(
                        token_id=plan["token_id"],
                        price=float(plan["limit_price"]),
                        size=float(plan["size"]),
                    )
            except Exception as exc:
                message = str(exc)
                reasons = ["live_order_failed"]
                lowered = message.lower()
                if "not enough balance / allowance" in lowered:
                    reasons = ["insufficient_balance_or_allowance"]
                elif "allowance" in lowered:
                    reasons = ["insufficient_allowance"]
                elif "balance" in lowered or "collateral" in lowered:
                    reasons = ["insufficient_balance"]
                result = {
                    "station_id": station_id,
                    "status": "skip",
                    "reasons": reasons,
                    "candidate": asdict(candidate),
                    "plan": plan,
                    "exits": exit_events,
                    "error": message,
                }
                self._append_log(result)
                return result
            summary = self.execution.summarize_order_response(
                response,
                fallback_size=float(plan["size"]),
                fallback_price=float(plan["limit_price"]),
                fallback_notional=float(plan["notional_usd"]),
                assume_full_on_matched=True,
            )
            if float(summary.get("matched_size") or 0.0) <= 0.0:
                result = {
                    "station_id": station_id,
                    "status": "live_unfilled",
                    "candidate": asdict(candidate),
                    "plan": plan,
                    "exits": exit_events,
                    "summary": summary,
                    "response": response,
                }
                self._append_log(result)
                return result

            reconciled_plan = dict(plan)
            reconciled_plan["size"] = float(summary["matched_size"])
            reconciled_plan["notional_usd"] = float(summary["matched_notional"])
            result = {
                "station_id": station_id,
                "status": "live_sent",
                "candidate": asdict(candidate),
                "plan": reconciled_plan,
                "exits": exit_events,
                "summary": summary,
                "response": response,
            }
            self._persist_trade(candidate, reconciled_plan, {"response": response, "summary": summary}, mode="live")
            self._append_log(result)
            return result

    async def run_once(self) -> List[Dict[str, Any]]:
        self.state = load_autotrader_state(self.config.state_path)
        if not self.config.enabled:
            event = {"status": "disabled"}
            self._append_log(event)
            return [event]

        kill_switch = Path(self.config.kill_switch_path)
        if kill_switch.exists():
            event = {"status": "killed", "path": str(kill_switch)}
            self._append_log(event)
            return [event]

        station_ids = list(self.config.station_ids or [])
        if not station_ids:
            event = {"status": "no_stations_configured"}
            self._append_log(event)
            return [event]
        results: List[Dict[str, Any]] = []
        for station_id in station_ids:
            if station_id not in STATIONS:
                results.append({"station_id": station_id, "status": "skip", "reasons": ["invalid_station"]})
                continue
            try:
                results.append(
                    await self.run_station_once(
                        station_id,
                        target_day=self.config.target_day,
                        target_date=self.config.target_date or None,
                    )
                )
            except Exception as exc:
                logger.exception("Autotrader station run failed for %s", station_id)
                event = {"station_id": station_id, "status": "error", "error": str(exc)}
                self._append_log(event)
                results.append(event)
        return results

    async def run_loop(self) -> None:
        while True:
            await self.run_once()
            await asyncio.sleep(max(5, int(self.config.loop_seconds)))
