"""Paper execution engine — simulates fills by walking live L2 orderbook data.

This module provides realistic fill simulation for the papertrader shadow
trading system. Instead of assuming instant fills at limit price, it walks
the orderbook level-by-level to compute actual fill prices, partial fills,
and slippage.

Supports:
  - limit_fak (Fill-and-Kill): consume levels up to limit price, partial OK
  - market_fok (Fill-or-Kill): full fill required or reject entirely
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PaperFill:
    """Result of attempting to fill an order against the orderbook."""

    filled_size: float = 0.0
    avg_price: float = 0.0
    notional_usd: float = 0.0
    slippage_vs_best: float = 0.0
    levels_consumed: int = 0
    per_level: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "REJECTED"  # FILLED / PARTIAL / REJECTED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperOrder:
    """A simulated order with full lifecycle tracking."""

    order_id: str
    kind: str  # "entry" / "exit"
    station_id: str
    target_date: str
    label: str
    side: str  # "YES" / "NO"
    token_id: str
    order_type: str  # "limit_fak" / "market_fok"
    direction: str  # "buy" / "sell"
    requested_size: float
    limit_price: float  # max price for buy, min price for sell
    status: str = "NEW"  # NEW/WORKING/PARTIAL/FILLED/CANCELED/EXPIRED/REJECTED
    filled_size: float = 0.0
    remaining_size: float = 0.0
    avg_fill_price: float = 0.0
    fill_notional_usd: float = 0.0
    slippage_vs_best: float = 0.0
    submitted_at_utc: str = ""
    filled_at_utc: str = ""
    book_snapshot_submit: Optional[Dict[str, Any]] = None
    book_snapshot_fill: Optional[Dict[str, Any]] = None
    fills: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        if self.remaining_size == 0.0 and self.requested_size > 0:
            self.remaining_size = self.requested_size
        if self.fills is None:
            self.fills = []

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Book snapshot helpers
# ---------------------------------------------------------------------------


def compact_book_snapshot(l2_snap: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact summary from a LocalOrderBook.get_l2_snapshot() result."""
    return {
        "best_bid": l2_snap.get("best_bid"),
        "best_ask": l2_snap.get("best_ask"),
        "mid": l2_snap.get("mid"),
        "spread": l2_snap.get("spread"),
        "bid_depth": l2_snap.get("bid_depth", 0.0),
        "ask_depth": l2_snap.get("ask_depth", 0.0),
        "bids": l2_snap.get("bids", [])[:10],
        "asks": l2_snap.get("asks", [])[:10],
    }


# ---------------------------------------------------------------------------
# Core fill algorithms
# ---------------------------------------------------------------------------


def _walk_book_buy(
    asks: List[Dict[str, Any]],
    limit_price: float,
    requested_size: float,
) -> PaperFill:
    """Walk ask levels ascending to simulate a buy order.

    For each ask level at or below limit_price, consume as much as possible.
    Returns a PaperFill with the aggregate result.
    """
    if not asks or requested_size <= 0 or limit_price <= 0:
        return PaperFill(status="REJECTED")

    # asks should already be sorted ascending by price from get_l2_snapshot
    sorted_asks = sorted(asks, key=lambda lvl: float(lvl["price"]))

    remaining = requested_size
    filled_size = 0.0
    fill_notional = 0.0
    per_level: List[Dict[str, Any]] = []
    best_ask = float(sorted_asks[0]["price"]) if sorted_asks else 0.0

    for level in sorted_asks:
        level_price = float(level["price"])
        level_size = float(level["size"])

        if level_price > limit_price:
            break
        if level_size <= 0:
            continue

        consumed = min(remaining, level_size)
        level_notional = consumed * level_price
        filled_size += consumed
        fill_notional += level_notional
        remaining -= consumed

        per_level.append({
            "price": round(level_price, 6),
            "size": round(consumed, 6),
            "notional": round(level_notional, 4),
        })

        if remaining <= 1e-9:
            break

    if filled_size <= 0:
        return PaperFill(status="REJECTED")

    avg_price = fill_notional / filled_size
    slippage = avg_price - best_ask if best_ask > 0 else 0.0

    status = "FILLED" if remaining <= 1e-9 else "PARTIAL"
    return PaperFill(
        filled_size=round(filled_size, 6),
        avg_price=round(avg_price, 6),
        notional_usd=round(fill_notional, 4),
        slippage_vs_best=round(slippage, 6),
        levels_consumed=len(per_level),
        per_level=per_level,
        status=status,
    )


def _walk_book_sell(
    bids: List[Dict[str, Any]],
    min_price: float,
    requested_size: float,
) -> PaperFill:
    """Walk bid levels descending to simulate a sell order.

    For each bid level at or above min_price, consume as much as possible.
    Returns a PaperFill with the aggregate result.
    """
    if not bids or requested_size <= 0:
        return PaperFill(status="REJECTED")

    # bids should already be sorted descending from get_l2_snapshot
    sorted_bids = sorted(bids, key=lambda lvl: float(lvl["price"]), reverse=True)

    remaining = requested_size
    filled_size = 0.0
    fill_notional = 0.0
    per_level: List[Dict[str, Any]] = []
    best_bid = float(sorted_bids[0]["price"]) if sorted_bids else 0.0

    for level in sorted_bids:
        level_price = float(level["price"])
        level_size = float(level["size"])

        if level_price < min_price:
            break
        if level_size <= 0:
            continue

        consumed = min(remaining, level_size)
        level_notional = consumed * level_price
        filled_size += consumed
        fill_notional += level_notional
        remaining -= consumed

        per_level.append({
            "price": round(level_price, 6),
            "size": round(consumed, 6),
            "notional": round(level_notional, 4),
        })

        if remaining <= 1e-9:
            break

    if filled_size <= 0:
        return PaperFill(status="REJECTED")

    avg_price = fill_notional / filled_size
    # For sells, slippage is negative when avg < best_bid (got worse price)
    slippage = best_bid - avg_price if best_bid > 0 else 0.0

    status = "FILLED" if remaining <= 1e-9 else "PARTIAL"
    return PaperFill(
        filled_size=round(filled_size, 6),
        avg_price=round(avg_price, 6),
        notional_usd=round(fill_notional, 4),
        slippage_vs_best=round(slippage, 6),
        levels_consumed=len(per_level),
        per_level=per_level,
        status=status,
    )


# ---------------------------------------------------------------------------
# Paper Execution Engine
# ---------------------------------------------------------------------------


class PaperExecutionEngine:
    """Simulates order execution against live L2 orderbook snapshots.

    Walks the book level-by-level for realistic fill simulation.
    Tracks all orders in memory for lifecycle management.
    """

    def __init__(self, latency_ms: float = 50.0):
        self.latency_ms = latency_ms
        self._orders: Dict[str, PaperOrder] = {}

    # -- Stateless fill execution (returns PaperFill) --

    def execute_limit_fak_buy(
        self,
        token_id: str,
        limit_price: float,
        size: float,
        book_snapshot: Dict[str, Any],
    ) -> PaperFill:
        """Limit FAK buy: walk asks up to limit_price, partial fill OK."""
        asks = book_snapshot.get("asks", [])
        return _walk_book_buy(asks, limit_price, size)

    def execute_limit_fak_sell(
        self,
        token_id: str,
        min_price: float,
        size: float,
        book_snapshot: Dict[str, Any],
    ) -> PaperFill:
        """Limit FAK sell: walk bids down to min_price, partial fill OK."""
        bids = book_snapshot.get("bids", [])
        return _walk_book_sell(bids, min_price, size)

    def execute_market_fok_buy(
        self,
        token_id: str,
        max_price: float,
        size: float,
        book_snapshot: Dict[str, Any],
    ) -> PaperFill:
        """Market FOK buy: full fill required or reject entirely."""
        asks = book_snapshot.get("asks", [])
        fill = _walk_book_buy(asks, max_price, size)
        # FOK semantics: reject if not fully filled
        if fill.status == "PARTIAL":
            return PaperFill(status="REJECTED")
        return fill

    def execute_market_fok_sell(
        self,
        token_id: str,
        min_price: float,
        size: float,
        book_snapshot: Dict[str, Any],
    ) -> PaperFill:
        """Market FOK sell: full fill required or reject entirely."""
        bids = book_snapshot.get("bids", [])
        fill = _walk_book_sell(bids, min_price, size)
        # FOK semantics: reject if not fully filled
        if fill.status == "PARTIAL":
            return PaperFill(status="REJECTED")
        return fill

    # -- Stateful order management --

    def place_order(
        self,
        *,
        kind: str,
        station_id: str,
        target_date: str,
        label: str,
        side: str,
        token_id: str,
        order_type: str,
        direction: str,
        size: float,
        limit_price: float,
        book_snapshot: Dict[str, Any],
    ) -> PaperOrder:
        """Place a paper order and attempt immediate execution.

        For FAK orders, attempts to fill immediately against the book.
        For FOK orders, fills completely or rejects.
        """
        now = datetime.now(timezone.utc).isoformat()
        order = PaperOrder(
            order_id=str(uuid.uuid4())[:12],
            kind=kind,
            station_id=station_id,
            target_date=target_date,
            label=label,
            side=side,
            token_id=token_id,
            order_type=order_type,
            direction=direction,
            requested_size=size,
            limit_price=limit_price,
            status="WORKING",
            submitted_at_utc=now,
            book_snapshot_submit=compact_book_snapshot(book_snapshot),
        )

        # Execute immediately (FAK/FOK are immediate-or-cancel by nature)
        if direction == "buy":
            if order_type == "limit_fak":
                fill = self.execute_limit_fak_buy(token_id, limit_price, size, book_snapshot)
            elif order_type == "market_fok":
                fill = self.execute_market_fok_buy(token_id, limit_price, size, book_snapshot)
            else:
                fill = PaperFill(status="REJECTED")
        else:  # sell
            if order_type == "limit_fak":
                fill = self.execute_limit_fak_sell(token_id, limit_price, size, book_snapshot)
            elif order_type == "market_fok":
                fill = self.execute_market_fok_sell(token_id, limit_price, size, book_snapshot)
            else:
                fill = PaperFill(status="REJECTED")

        # Apply fill result to order
        order.filled_size = fill.filled_size
        order.remaining_size = round(size - fill.filled_size, 6)
        order.avg_fill_price = fill.avg_price
        order.fill_notional_usd = fill.notional_usd
        order.slippage_vs_best = fill.slippage_vs_best
        order.fills = fill.per_level
        order.book_snapshot_fill = compact_book_snapshot(book_snapshot)

        if fill.status == "FILLED":
            order.status = "FILLED"
            order.filled_at_utc = now
        elif fill.status == "PARTIAL":
            # FAK: partial fill is terminal (remaining is killed)
            order.status = "PARTIAL"
            order.filled_at_utc = now
        else:
            order.status = "REJECTED"

        self._orders[order.order_id] = order
        return order

    def get_order(self, order_id: str) -> Optional[PaperOrder]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[PaperOrder]:
        return [o for o in self._orders.values() if o.status in ("NEW", "WORKING")]

    def get_all_orders(self) -> List[PaperOrder]:
        return list(self._orders.values())

    def cancel_order(self, order_id: str) -> Optional[PaperOrder]:
        order = self._orders.get(order_id)
        if order and order.status in ("NEW", "WORKING"):
            order.status = "CANCELED"
        return order

    def reset(self):
        """Clear all tracked orders."""
        self._orders.clear()
