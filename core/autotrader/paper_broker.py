"""
Paper broker backed by backtest execution simulator.

This provides:
- realistic taker/maker behavior via ExecutionSimulator
- position/inventory tracking
- realized PnL tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.backtest.policy import OrderType, PolicySignal, PolicyState, Side
from core.backtest.simulator import ExecutionSimulator, Fill, Order
from core.polymarket_labels import normalize_label

from .models import PaperFill, PaperOrder


@dataclass
class _Position:
    size: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0


class PaperBroker:
    def __init__(
        self,
        simulator: Optional[ExecutionSimulator] = None,
        *,
        allow_naked_short: bool = False,
        min_buy_notional_usd: float = 0.01,
    ):
        self.simulator = simulator or ExecutionSimulator()
        self.allow_naked_short = bool(allow_naked_short)
        self.min_buy_notional_usd = float(min_buy_notional_usd)
        self.positions: Dict[str, _Position] = {}
        self.realized_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.total_slippage: float = 0.0
        self.total_orders: int = 0
        self._known_order_ids: set[str] = set()
        self._order_strategy: Dict[str, str] = {}
        self._fills: List[PaperFill] = []
        self._orders: List[PaperOrder] = []

    def reset(self) -> None:
        self.simulator.reset()
        self.positions = {}
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_orders = 0
        self._known_order_ids = set()
        self._order_strategy = {}
        self._fills = []
        self._orders = []

    def execute_actions(
        self,
        timestamp_utc: datetime,
        station_id: str,
        strategy_name: str,
        actions: List[Dict[str, Any]],
        market_state: Optional[Dict[str, Any]],
    ) -> Tuple[List[PaperOrder], List[PaperFill]]:
        created_orders: List[PaperOrder] = []
        created_fills: List[PaperFill] = []

        for action in actions:
            side_val = (action.get("side") or "buy").lower()
            side = Side.BUY if side_val == "buy" else Side.SELL
            order_type_val = (action.get("order_type") or "taker").lower()
            order_type = OrderType.MAKER if order_type_val == "maker" else OrderType.TAKER
            bucket = normalize_label(str(action.get("bucket", "")))
            outcome = str(action.get("outcome", "yes")).lower()
            target_size = float(action.get("target_size", 0.0) or 0.0)
            if target_size <= 0.0:
                continue

            # Do not allow naked shorting in paper mode unless explicitly enabled.
            # SELL only valid for closing existing positions of the same outcome.
            if side == Side.SELL and not self.allow_naked_short:
                pos_key = f"{bucket}_{outcome}"
                current = self._get_position(pos_key).size
                if current <= 1e-9:
                    continue
                target_size = min(target_size, float(current))
                if target_size <= 0.0:
                    continue

            est_price = self._estimate_action_price(
                bucket=bucket,
                side=side,
                market_state=market_state,
                max_price=float(action.get("max_price", 0.99)),
                min_price=float(action.get("min_price", 0.01)),
            )
            if (
                side == Side.BUY
                and self.min_buy_notional_usd > 0.0
                and est_price > 0.0
                and (target_size * est_price) < self.min_buy_notional_usd
            ):
                target_size = self.min_buy_notional_usd / est_price

            signal = PolicySignal(
                timestamp_utc=timestamp_utc,
                bucket=bucket,
                side=side,
                order_type=order_type,
                target_size=target_size,
                max_price=float(action.get("max_price", 0.99)),
                min_price=float(action.get("min_price", 0.01)),
                confidence=float(action.get("confidence", 0.5)),
                urgency=float(action.get("urgency", 0.5)),
                ttl_seconds=float(action.get("ttl_seconds", 300)),
                reason=str(action.get("reason", "")),
                policy_state=PolicyState.BUILD_POSITION,
                outcome=outcome,
            )

            self.total_orders += 1
            fill = self.simulator.execute_signal(signal, market_state=market_state)
            if fill:
                paper_fill = self._convert_fill(fill, station_id, strategy_name)
                created_fills.append(paper_fill)
                self._fills.append(paper_fill)
                self._apply_fill(fill)

        new_orders = self._capture_new_orders(station_id, strategy_name)
        created_orders.extend(new_orders)
        self._orders.extend(new_orders)
        return created_orders, created_fills

    def _estimate_action_price(
        self,
        bucket: str,
        side: Side,
        market_state: Optional[Dict[str, Any]],
        max_price: float,
        min_price: float,
    ) -> float:
        """
        Estimate executable price for minimum notional checks.
        Uses orderbook first, then policy limits as fallback.
        """
        snap = (market_state or {}).get(bucket, {})
        bid = float(snap.get("best_bid", 0.0) or 0.0)
        ask = float(snap.get("best_ask", 0.0) or 0.0)
        mid = float(snap.get("mid", 0.0) or 0.0)

        if side == Side.BUY:
            if ask > 0.0:
                return ask
            if mid > 0.0:
                return mid
            return max(max_price, 0.0)
        if bid > 0.0:
            return bid
        if mid > 0.0:
            return mid
        return max(min_price, 0.0)

    def update(
        self,
        timestamp_utc: datetime,
        station_id: str,
        market_state: Optional[Dict[str, Any]],
        elapsed_seconds: float = 60.0,
    ) -> Tuple[List[PaperOrder], List[PaperFill]]:
        fills = self.simulator.update(timestamp_utc, market_state, elapsed_seconds)
        paper_fills: List[PaperFill] = []

        for fill in fills:
            strategy_name = self._order_strategy.get(fill.order_id or "", "maker_passive")
            paper_fill = self._convert_fill(fill, station_id, strategy_name)
            paper_fills.append(paper_fill)
            self._fills.append(paper_fill)
            self._apply_fill(fill)

        new_orders = self._capture_new_orders(station_id, strategy_name="maker_passive")
        self._orders.extend(new_orders)
        return new_orders, paper_fills

    def _capture_new_orders(self, station_id: str, strategy_name: str) -> List[PaperOrder]:
        out: List[PaperOrder] = []
        for order in self.simulator.get_orders():
            if not order.order_id or order.order_id in self._known_order_ids:
                continue
            self._known_order_ids.add(order.order_id)
            self._order_strategy[order.order_id] = strategy_name
            out.append(self._convert_order(order, station_id, strategy_name))
        return out

    def _convert_order(self, order: Order, station_id: str, strategy_name: str) -> PaperOrder:
        raw = order.to_dict() if hasattr(order, "to_dict") else {}
        return PaperOrder(
            ts_utc=order.timestamp_utc,
            station_id=station_id,
            strategy_name=strategy_name,
            order_id=order.order_id,
            bucket=normalize_label(order.bucket),
            side=order.side,
            size=float(order.size),
            limit_price=float(order.limit_price),
            order_type=order.order_type,
            status=order.status,
            outcome=getattr(order, "outcome", "yes"),
            raw=raw,
        )

    def _convert_fill(self, fill: Fill, station_id: str, strategy_name: str) -> PaperFill:
        raw = fill.to_dict() if hasattr(fill, "to_dict") else {}
        return PaperFill(
            ts_utc=fill.timestamp_utc,
            station_id=station_id,
            strategy_name=strategy_name,
            bucket=normalize_label(fill.bucket),
            side=fill.side,
            size=float(fill.size),
            price=float(fill.price),
            fill_type=fill.fill_type.value,
            fees=float(fill.fees),
            slippage=float(fill.slippage),
            outcome=getattr(fill, "outcome", "yes"),
            order_id=fill.order_id,
            raw=raw,
        )

    def _get_position(self, bucket: str) -> _Position:
        if bucket not in self.positions:
            self.positions[bucket] = _Position()
        return self.positions[bucket]

    def _apply_fill(self, fill: Fill) -> None:
        bucket = normalize_label(fill.bucket)
        outcome = getattr(fill, "outcome", "yes")
        pos_key = f"{bucket}_{outcome}"
        pos = self._get_position(pos_key)
        qty = float(fill.size if fill.side == "buy" else -fill.size)
        price = float(fill.price)
        fee_cost = float(fill.fees + fill.slippage)
        self.total_fees += float(fill.fees)
        self.total_slippage += float(fill.slippage)

        if abs(pos.size) < 1e-9 or pos.size * qty > 0:
            # Opening or adding in the same direction.
            new_size = pos.size + qty
            if abs(new_size) > 1e-9:
                if abs(pos.size) < 1e-9:
                    pos.avg_price = price
                else:
                    weighted = (abs(pos.size) * pos.avg_price) + (abs(qty) * price)
                    pos.avg_price = weighted / abs(new_size)
            else:
                pos.avg_price = 0.0
            pos.size = new_size
        else:
            # Closing or flipping.
            closing = min(abs(pos.size), abs(qty))
            if pos.size > 0 and qty < 0:
                self.realized_pnl += (price - pos.avg_price) * closing
            elif pos.size < 0 and qty > 0:
                self.realized_pnl += (pos.avg_price - price) * closing

            new_size = pos.size + qty
            if abs(new_size) < 1e-9:
                pos.size = 0.0
                pos.avg_price = 0.0
            elif pos.size * new_size < 0:
                # Flipped side: remaining inventory starts at current fill price.
                pos.size = new_size
                pos.avg_price = price
            else:
                pos.size = new_size

        self.realized_pnl -= fee_cost

    def mark_to_market(self, market_state: Optional[Dict[str, Any]]) -> float:
        if not market_state:
            for pos in self.positions.values():
                pos.unrealized_pnl = 0.0
            return 0.0

        total = 0.0
        for pos_key, pos in self.positions.items():
            if abs(pos.size) < 1e-9:
                pos.unrealized_pnl = 0.0
                continue
            # Position keys are "bucket_outcome" (e.g. "32-33°F_yes")
            if pos_key.endswith("_no"):
                bucket = pos_key[:-3]
                is_no = True
            elif pos_key.endswith("_yes"):
                bucket = pos_key[:-4]
                is_no = False
            else:
                bucket = pos_key
                is_no = False
            snap = market_state.get(bucket, {})
            yes_bid = float(snap.get("best_bid", 0.0) or 0.0)
            yes_ask = float(snap.get("best_ask", 0.0) or 0.0)
            if is_no:
                # NO mid ≈ 1 - YES mid
                yes_mid = (yes_bid + yes_ask) / 2.0 if yes_bid > 0.0 and yes_ask > 0.0 else 0.5
                mid = 1.0 - yes_mid
            else:
                mid = (yes_bid + yes_ask) / 2.0 if yes_bid > 0.0 and yes_ask > 0.0 else pos.avg_price
            if pos.size > 0:
                pos.unrealized_pnl = (mid - pos.avg_price) * abs(pos.size)
            else:
                pos.unrealized_pnl = (pos.avg_price - mid) * abs(pos.size)
            total += pos.unrealized_pnl
        return total

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for pos_key, pos in self.positions.items():
            if abs(pos.size) < 1e-9:
                continue
            # Parse "bucket_outcome" key
            if pos_key.endswith("_no"):
                bucket, outcome = pos_key[:-3], "no"
            elif pos_key.endswith("_yes"):
                bucket, outcome = pos_key[:-4], "yes"
            else:
                bucket, outcome = pos_key, "yes"
            result[pos_key] = {
                "bucket": bucket,
                "outcome": outcome,
                "size": pos.size,
                "avg_price": pos.avg_price,
                "unrealized_pnl": pos.unrealized_pnl,
            }
        return result

    def get_orders(self, limit: int = 200) -> List[Dict[str, Any]]:
        return [o.to_dict() for o in self._orders[-limit:]]

    def get_fills(self, limit: int = 200) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self._fills[-limit:]]

    def get_performance(self, mark_to_market_pnl: float = 0.0) -> Dict[str, float]:
        return {
            "realized_pnl": float(self.realized_pnl),
            "mark_to_market_pnl": float(mark_to_market_pnl),
            "total_pnl": float(self.realized_pnl + mark_to_market_pnl),
            "total_fees": float(self.total_fees),
            "total_slippage": float(self.total_slippage),
            "total_orders": int(self.total_orders),
            "total_fills": len(self._fills),
        }
