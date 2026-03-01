from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from math import floor, gcd
from typing import Any, Dict, List, Optional

import requests
from eth_account import Account

from config import POLYMARKET_CITY_SLUGS
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, AssetType, BalanceAllowanceParams, OrderArgs, OrderType


_DATA_API_BASE = "https://data-api.polymarket.com"
_LIVE_CACHE_TTL_SECONDS = 3.0
_CITY_SLUG_TO_STATION = {str(value).lower(): key for key, value in POLYMARKET_CITY_SLUGS.items()}
_STATION_TITLE_HINTS = {
    "KLGA": ("new york city", "new york", "nyc"),
    "KATL": ("atlanta",),
    "KORD": ("chicago",),
    "KMIA": ("miami",),
    "KDAL": ("dallas",),
    "LFPG": ("paris",),
    "EGLC": ("london",),
    "LTAC": ("ankara",),
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_first_nonblank(*names: str, default: str = "") -> str:
    for name in names:
        raw = str(os.environ.get(name, "")).strip()
        if raw:
            return raw
    return default


def _round_down(value: float, decimals: int) -> float:
    scale = 10 ** max(0, int(decimals))
    return floor(float(value) * scale) / scale


def _price_fraction(price: float) -> tuple[int, int]:
    raw = str(float(price)).rstrip("0").rstrip(".")
    decimal = Decimal(raw or "0")
    exponent = decimal.as_tuple().exponent
    if exponent >= 0:
        return int(decimal), 1
    scale = 10 ** (-exponent)
    numerator = int(decimal * scale)
    return numerator, scale


def quantize_share_size(size: float, decimals: int = 2) -> float:
    return max(0.0, _round_down(float(size), decimals))


def quantize_market_buy_amount(amount_usd: float) -> float:
    return max(0.0, _round_down(float(amount_usd), 2))


def quantize_buy_order_size(size: float, price: float) -> float:
    rounded_size = quantize_share_size(size, 2)
    if rounded_size <= 0 or price <= 0:
        return 0.0

    numerator, denominator = _price_fraction(price)
    if numerator <= 0 or denominator <= 0:
        return rounded_size

    # Polymarket buy orders require the collateral leg to resolve cleanly to cents.
    size_step_cents = max(1, denominator // gcd(abs(numerator), denominator))
    size_step = max(0.01, size_step_cents / 100.0)
    steps = floor(rounded_size / size_step)
    if steps <= 0:
        return 0.0
    return quantize_share_size(steps * size_step, 2)


def _extract_numeric(payload: Any, *keys: str) -> Optional[float]:
    if payload is None:
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, str):
        return _safe_float(payload)
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                parsed = _extract_numeric(payload.get(key), "value", "amount", "decimal", "raw")
                if parsed is not None:
                    return parsed
        for fallback_key in ("decimal", "balance", "allowance", "amount", "value", "raw"):
            if fallback_key in payload:
                parsed = _extract_numeric(payload.get(fallback_key), "value", "amount", "decimal", "raw")
                if parsed is not None:
                    return parsed
    return None


def _extract_text(payload: Any, *keys: str) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return str(value)
    return ""


def _normalize_usdc_amount(value: Any) -> Optional[float]:
    if value is None:
        return None
    raw_text = str(value).strip()
    if not raw_text:
        return None
    try:
        if "." in raw_text:
            return float(raw_text)
        raw_int = int(raw_text)
        return raw_int / 1_000_000.0
    except (TypeError, ValueError):
        parsed = _safe_float(value)
        if parsed is None:
            return None
        if abs(parsed) >= 1000 and float(parsed).is_integer():
            return parsed / 1_000_000.0
        return parsed


def _has_positive_allowance(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    allowances = raw.get("allowances")
    if isinstance(allowances, dict):
        for amount in allowances.values():
            text = str(amount or "").strip()
            if not text:
                continue
            try:
                if int(text) > 0:
                    return True
            except ValueError:
                parsed = _safe_float(text)
                if parsed is not None and parsed > 0:
                    return True
    direct = raw.get("allowance")
    if direct is not None:
        parsed = _normalize_usdc_amount(direct)
        return parsed is not None and parsed > 0
    return False


def _funded_user_address(private_key: str, funder: str) -> str:
    funder_raw = str(funder or "").strip()
    if funder_raw:
        return funder_raw
    private_key_raw = str(private_key or "").strip()
    if not private_key_raw:
        return ""
    return str(Account.from_key(private_key_raw).address)


def _infer_station_id(position: Dict[str, Any]) -> Optional[str]:
    for key in ("eventSlug", "slug"):
        slug = str(position.get(key) or "").strip().lower()
        if not slug:
            continue
        for city_slug, station_id in _CITY_SLUG_TO_STATION.items():
            if city_slug and city_slug in slug:
                return station_id
    title = str(position.get("title") or "").strip().lower()
    for station_id, hints in _STATION_TITLE_HINTS.items():
        if any(hint in title for hint in hints):
            return station_id
    return None


def _extract_market_label(title: str) -> str:
    raw = str(title or "").strip()
    if not raw:
        return ""
    match = re.search(r"\bbe\s+(.+?)\s+on\s+[a-z]+\s+\d{1,2}\??$", raw, flags=re.IGNORECASE)
    if not match:
        return raw
    label = match.group(1).strip()
    if label.lower().startswith("between "):
        label = label[8:].strip()
    return label


def _normalize_live_position(position: Dict[str, Any]) -> Dict[str, Any]:
    title = str(position.get("title") or "").strip()
    station_id = _infer_station_id(position)
    target_date = str(position.get("endDate") or "").strip()
    if target_date:
        try:
            target_date = datetime.fromisoformat(target_date.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            target_date = target_date.split("T", 1)[0]
    label = _extract_market_label(title)
    side = str(position.get("outcome") or "").strip().upper()
    return {
        "source": "polymarket_live",
        "station_id": station_id,
        "target_date": target_date,
        "label": label or title,
        "side": side,
        "token_id": str(position.get("asset") or "").strip(),
        "opposite_token_id": str(position.get("oppositeAsset") or "").strip(),
        "position_key": f"{station_id or 'UNKNOWN'}|{target_date}|{label or title}|{side}".upper(),
        "title": title,
        "event_slug": str(position.get("eventSlug") or "").strip(),
        "slug": str(position.get("slug") or "").strip(),
        "cost_basis_open_usd": float(_safe_float(position.get("initialValue")) or 0.0),
        "current_value_usd": float(_safe_float(position.get("currentValue")) or 0.0),
        "realized_pnl_usd": float(_safe_float(position.get("realizedPnl")) or 0.0),
        "cash_pnl_usd": float(_safe_float(position.get("cashPnl")) or 0.0),
        "shares_open": float(_safe_float(position.get("size")) or 0.0),
        "shares": float(_safe_float(position.get("size")) or 0.0),
        "avg_price": float(_safe_float(position.get("avgPrice")) or 0.0),
        "entry_price": float(_safe_float(position.get("avgPrice")) or 0.0),
        "current_price": float(_safe_float(position.get("curPrice")) or 0.0),
        "opened_at_utc": None,
        "status": "OPEN",
        "raw": position,
    }


def _extract_order_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    direct = _extract_text(payload, "orderID", "orderId", "id")
    if direct:
        return direct
    order = payload.get("order")
    if isinstance(order, dict):
        return _extract_text(order, "id", "orderID", "orderId")
    return ""


def _matched_size_from_amounts(payload: Any, fallback_size: float, fallback_notional: float) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    amounts = []
    for key in ("makingAmount", "making_amount", "takingAmount", "taking_amount"):
        value = _extract_numeric(payload, key)
        if value is not None:
            amounts.append(float(value))
    if not amounts:
        return None
    target_size = max(0.000001, float(fallback_size))
    target_notional = max(0.000001, float(fallback_notional))
    scored = []
    for amount in amounts:
        score_size = abs(amount - target_size)
        score_notional = abs(amount - target_notional)
        scored.append((min(score_size, score_notional), score_size <= score_notional, amount))
    scored.sort(key=lambda row: (row[0], not row[1]))
    best = scored[0]
    return float(best[2])


@dataclass
class PolymarketExecutionConfig:
    host: str = "https://clob.polymarket.com"
    chain_id: int = 137
    private_key: str = ""
    signature_type: int = 0
    funder: str = ""
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    derive_api_creds: bool = True
    auto_approve_allowances: bool = False

    @property
    def has_private_key(self) -> bool:
        return bool(str(self.private_key or "").strip())

    @property
    def has_api_creds(self) -> bool:
        return bool(self.api_key and self.api_secret and self.api_passphrase)

    @property
    def is_live_ready(self) -> bool:
        return self.has_private_key and (self.has_api_creds or self.derive_api_creds)


@dataclass
class TopOfBook:
    token_id: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    bid_size: float
    ask_size: float
    spread: Optional[float]
    min_order_size: Optional[float]
    last_trade_price: Optional[float]
    raw: Any


def load_polymarket_execution_config_from_env() -> PolymarketExecutionConfig:
    return PolymarketExecutionConfig(
        host=_env_first_nonblank("HELIOS_POLY_CLOB_HOST", default="https://clob.polymarket.com"),
        chain_id=int(_env_first_nonblank("HELIOS_POLY_CHAIN_ID", default="137") or "137"),
        private_key=_env_first_nonblank("HELIOS_POLY_PRIVATE_KEY", "POLYMARKET_PRIVATE_KEY"),
        signature_type=int(_env_first_nonblank("HELIOS_POLY_SIGNATURE_TYPE", default="0") or "0"),
        funder=_env_first_nonblank("HELIOS_POLY_FUNDER", "POLYMARKET_FUNDER"),
        api_key=_env_first_nonblank("HELIOS_POLY_API_KEY", "POLYMARKET_API_KEY"),
        api_secret=_env_first_nonblank("HELIOS_POLY_API_SECRET", "POLYMARKET_API_SECRET"),
        api_passphrase=_env_first_nonblank("HELIOS_POLY_API_PASSPHRASE", "POLYMARKET_API_PASSPHRASE"),
        derive_api_creds=_env_bool("HELIOS_POLY_DERIVE_API_CREDS", True),
        auto_approve_allowances=_env_bool("HELIOS_POLY_AUTO_APPROVE_ALLOWANCES", False),
    )


class PolymarketExecutionClient:
    def __init__(self, config: Optional[PolymarketExecutionConfig] = None):
        self.config = config or load_polymarket_execution_config_from_env()
        self._public_client: Optional[ClobClient] = None
        self._auth_client: Optional[ClobClient] = None
        self._live_snapshot_cache: Optional[Dict[str, Any]] = None
        self._live_snapshot_cache_ts: float = 0.0

    def _build_client(self, *, authenticated: bool) -> ClobClient:
        if authenticated and not self.config.has_private_key:
            raise RuntimeError("Missing HELIOS_POLY_PRIVATE_KEY for live execution")

        client = ClobClient(
            self.config.host,
            chain_id=self.config.chain_id,
            key=self.config.private_key or None,
            signature_type=self.config.signature_type,
            funder=self.config.funder or None,
        )
        if not authenticated:
            return client

        if self.config.has_api_creds:
            client.set_api_creds(
                ApiCreds(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    api_passphrase=self.config.api_passphrase,
                )
            )
            return client

        if not self.config.derive_api_creds:
            raise RuntimeError("Missing Polymarket API credentials and HELIOS_POLY_DERIVE_API_CREDS is disabled")

        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        return client

    def public_client(self) -> ClobClient:
        if self._public_client is None:
            self._public_client = self._build_client(authenticated=False)
        return self._public_client

    def auth_client(self) -> ClobClient:
        if self._auth_client is None:
            self._auth_client = self._build_client(authenticated=True)
        return self._auth_client

    def account_user_address(self) -> str:
        return _funded_user_address(self.config.private_key, self.config.funder)

    def _data_api_get(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        response = requests.get(
            f"{_DATA_API_BASE}/{path.lstrip('/')}",
            params=params or {},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def get_top_of_book(self, token_id: str) -> TopOfBook:
        summary = self.public_client().get_order_book(token_id)
        bids = list(getattr(summary, "bids", None) or [])
        asks = list(getattr(summary, "asks", None) or [])

        best_bid = _safe_float(getattr(bids[0], "price", None)) if bids else None
        best_ask = _safe_float(getattr(asks[0], "price", None)) if asks else None
        bid_size = _safe_float(getattr(bids[0], "size", None)) if bids else None
        ask_size = _safe_float(getattr(asks[0], "size", None)) if asks else None
        spread = None
        if best_bid is not None and best_ask is not None:
            spread = round(best_ask - best_bid, 6)

        return TopOfBook(
            token_id=token_id,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=float(bid_size or 0.0),
            ask_size=float(ask_size or 0.0),
            spread=spread,
            min_order_size=_safe_float(getattr(summary, "min_order_size", None)),
            last_trade_price=_safe_float(getattr(summary, "last_trade_price", None)),
            raw=summary,
        )

    def get_collateral_status(self) -> Dict[str, Any]:
        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            token_id="",
            signature_type=self.config.signature_type,
        )
        raw = self.auth_client().get_balance_allowance(params)
        return {
            "balance": _normalize_usdc_amount((raw or {}).get("balance") if isinstance(raw, dict) else None),
            "allowance": float("inf") if _has_positive_allowance(raw) else 0.0,
            "allowance_ready": _has_positive_allowance(raw),
            "raw": raw,
        }

    def get_conditional_status(self, token_id: str) -> Dict[str, Any]:
        params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL,
            token_id=token_id,
            signature_type=self.config.signature_type,
        )
        raw = self.auth_client().get_balance_allowance(params)
        return {
            "balance": _extract_numeric(raw, "balance"),
            "allowance": float("inf") if _has_positive_allowance(raw) else 0.0,
            "allowance_ready": _has_positive_allowance(raw),
            "raw": raw,
        }

    def ensure_collateral_ready(self, required_amount: float) -> Dict[str, Any]:
        status = self.get_collateral_status()
        balance = _safe_float(status.get("balance"))
        allowance_ready = bool(status.get("allowance_ready"))

        if balance is not None and balance + 1e-9 < float(required_amount):
            raise RuntimeError(f"Insufficient collateral balance: {balance:.2f} < {required_amount:.2f}")

        if allowance_ready:
            return status

        if not self.config.auto_approve_allowances:
            raise RuntimeError("Collateral allowance is too low and HELIOS_POLY_AUTO_APPROVE_ALLOWANCES is disabled")

        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            token_id="",
            signature_type=self.config.signature_type,
        )
        self.auth_client().update_balance_allowance(params)
        return self.get_collateral_status()

    def ensure_conditional_ready(self, token_id: str, required_size: float) -> Dict[str, Any]:
        status = self.get_conditional_status(token_id)
        balance = _safe_float(status.get("balance"))
        allowance_ready = bool(status.get("allowance_ready"))

        if balance is not None and balance + 1e-9 < float(required_size):
            raise RuntimeError(f"Insufficient token balance: {balance:.4f} < {required_size:.4f}")

        if allowance_ready:
            return status

        if not self.config.auto_approve_allowances:
            raise RuntimeError("Conditional token allowance is too low and HELIOS_POLY_AUTO_APPROVE_ALLOWANCES is disabled")

        params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL,
            token_id=token_id,
            signature_type=self.config.signature_type,
        )
        self.auth_client().update_balance_allowance(params)
        return self.get_conditional_status(token_id)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        raw = self.auth_client().get_order(order_id)
        return raw if isinstance(raw, dict) else {"raw": raw}

    def get_live_portfolio_snapshot(self, *, force_refresh: bool = False) -> Dict[str, Any]:
        now = time.monotonic()
        if (
            not force_refresh
            and self._live_snapshot_cache is not None
            and (now - self._live_snapshot_cache_ts) < _LIVE_CACHE_TTL_SECONDS
        ):
            return dict(self._live_snapshot_cache)

        user = self.account_user_address()
        collateral = self.get_collateral_status() if user else {"balance": None, "allowance": 0.0, "allowance_ready": False, "raw": None}
        value_payload = self._data_api_get("value", params={"user": user}) if user else []
        positions_payload = self._data_api_get("positions", params={"user": user}) if user else []

        portfolio_value = None
        if isinstance(value_payload, list) and value_payload:
            portfolio_value = _safe_float((value_payload[0] or {}).get("value"))

        normalized_positions: List[Dict[str, Any]] = []
        if isinstance(positions_payload, list):
            normalized_positions = [_normalize_live_position(item) for item in positions_payload if isinstance(item, dict)]
            normalized_positions.sort(
                key=lambda row: float(_safe_float(row.get("current_value_usd")) or 0.0),
                reverse=True,
            )

        cost_basis = round(sum(float(_safe_float(row.get("cost_basis_open_usd")) or 0.0) for row in normalized_positions), 6)
        market_value = round(sum(float(_safe_float(row.get("current_value_usd")) or 0.0) for row in normalized_positions), 6)
        snapshot = {
            "user": user,
            "free_collateral_usd": float(_safe_float(collateral.get("balance")) or 0.0),
            "collateral_allowance_ready": bool(collateral.get("allowance_ready")),
            "portfolio_value_usd": float(portfolio_value if portfolio_value is not None else market_value),
            "positions_market_value_usd": market_value,
            "positions_cost_basis_usd": cost_basis,
            "open_positions_count": len(normalized_positions),
            "open_positions": normalized_positions,
            "raw_positions": positions_payload if isinstance(positions_payload, list) else [],
            "raw_value": value_payload,
        }
        self._live_snapshot_cache = snapshot
        self._live_snapshot_cache_ts = now
        return dict(snapshot)

    def summarize_order_response(
        self,
        response: Dict[str, Any],
        *,
        fallback_size: float,
        fallback_price: float,
        fallback_notional: Optional[float] = None,
        assume_full_on_matched: bool = False,
    ) -> Dict[str, Any]:
        payload = response if isinstance(response, dict) else {"raw": response}
        fallback_notional = float(fallback_notional if fallback_notional is not None else (float(fallback_size) * float(fallback_price)))
        success_raw = payload.get("success")
        success = True if success_raw is None else bool(success_raw)
        status = str(payload.get("status") or payload.get("state") or "").strip().lower()
        error_msg = _extract_text(payload, "errorMsg", "error", "message")
        order_id = _extract_order_id(payload)

        matched_size = _extract_numeric(payload, "size_matched", "matched_size", "filledSize", "filled_size")
        if matched_size is None and isinstance(payload.get("order"), dict):
            matched_size = _extract_numeric(payload.get("order"), "size_matched", "matched_size", "filledSize", "filled_size")
        if matched_size is None:
            matched_size = _matched_size_from_amounts(payload, float(fallback_size), float(fallback_notional))

        if matched_size is None and order_id:
            try:
                order = self.get_order(order_id)
                matched_size = _extract_numeric(order, "size_matched", "matched_size", "filledSize", "filled_size")
                if matched_size is None and isinstance(order.get("order"), dict):
                    matched_size = _extract_numeric(order.get("order"), "size_matched", "matched_size", "filledSize", "filled_size")
            except Exception:
                matched_size = None

        if matched_size is None:
            if not success or error_msg:
                matched_size = 0.0
            elif status in {"live", "unmatched", "delayed"}:
                matched_size = 0.0
            elif assume_full_on_matched and status in {"matched", "filled"}:
                matched_size = float(fallback_size)
            elif assume_full_on_matched and not status:
                matched_size = float(fallback_size)
            else:
                matched_size = 0.0

        matched_size = max(0.0, min(float(fallback_size), float(matched_size)))
        matched_notional = round(float(matched_size) * float(fallback_price), 6)
        return {
            "success": bool(success and not error_msg),
            "status": status or ("matched" if matched_size > 0 else "unknown"),
            "order_id": order_id or None,
            "matched_size": round(float(matched_size), 6),
            "matched_notional": matched_notional,
            "error": error_msg or None,
            "raw": payload,
        }

    def place_limit_buy(
        self,
        *,
        token_id: str,
        price: float,
        size: float,
        order_type: str = OrderType.FAK,
    ) -> Dict[str, Any]:
        price = round(float(price), 6)
        size = quantize_buy_order_size(size, price)
        if size <= 0:
            raise RuntimeError("Buy order size quantized to zero")
        fee_rate_bps = int(self.auth_client().get_fee_rate_bps(token_id))
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side="BUY",
            fee_rate_bps=fee_rate_bps,
        )
        signed_order = self.auth_client().create_order(order_args)
        response = self.auth_client().post_order(signed_order, orderType=order_type, post_only=False)
        if isinstance(response, dict):
            return response
        return {"raw": response}

    def place_limit_sell(
        self,
        *,
        token_id: str,
        price: float,
        size: float,
        order_type: str = OrderType.FAK,
    ) -> Dict[str, Any]:
        price = round(float(price), 6)
        size = quantize_share_size(size, 2)
        if size <= 0:
            raise RuntimeError("Sell order size quantized to zero")
        fee_rate_bps = int(self.auth_client().get_fee_rate_bps(token_id))
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side="SELL",
            fee_rate_bps=fee_rate_bps,
        )
        signed_order = self.auth_client().create_order(order_args)
        response = self.auth_client().post_order(signed_order, orderType=order_type, post_only=False)
        if isinstance(response, dict):
            return response
        return {"raw": response}

    def place_market_buy(
        self,
        *,
        token_id: str,
        amount_usd: float,
        max_price: Optional[float] = None,
        order_type: str = OrderType.FOK,
    ) -> Dict[str, Any]:
        from py_clob_client.clob_types import MarketOrderArgs

        amount_usd = quantize_market_buy_amount(amount_usd)
        if amount_usd <= 0:
            raise RuntimeError("Buy amount quantized to zero")
        fee_rate_bps = int(self.auth_client().get_fee_rate_bps(token_id))
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=amount_usd,
            side="BUY",
            price=round(float(max_price), 6) if max_price is not None else 0,
            fee_rate_bps=fee_rate_bps,
            order_type=order_type,
        )
        signed_order = self.auth_client().create_market_order(order_args)
        response = self.auth_client().post_order(signed_order, orderType=order_type, post_only=False)
        if isinstance(response, dict):
            return response
        return {"raw": response}

    def place_market_sell(
        self,
        *,
        token_id: str,
        size: float,
        min_price: Optional[float] = None,
        order_type: str = OrderType.FOK,
    ) -> Dict[str, Any]:
        from py_clob_client.clob_types import MarketOrderArgs

        size = quantize_share_size(size, 2)
        if size <= 0:
            raise RuntimeError("Sell amount quantized to zero")
        fee_rate_bps = int(self.auth_client().get_fee_rate_bps(token_id))
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=size,
            side="SELL",
            price=round(float(min_price), 6) if min_price is not None else 0,
            fee_rate_bps=fee_rate_bps,
            order_type=order_type,
        )
        signed_order = self.auth_client().create_market_order(order_args)
        response = self.auth_client().post_order(signed_order, orderType=order_type, post_only=False)
        if isinstance(response, dict):
            return response
        return {"raw": response}
