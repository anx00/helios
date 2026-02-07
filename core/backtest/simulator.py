"""
Execution Simulator Module

Simulates order execution for backtesting:
- TakerModel: Execute at market (best bid/ask + slippage)
- MakerModel: Queue-based fill simulation
- Adverse selection tracking

This is where most backtests get it wrong - be conservative here.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger("backtest.simulator")

UTC = ZoneInfo("UTC")


class FillType(Enum):
    """How the fill occurred."""
    TAKER_BUY = "taker_buy"      # Lifted the ask
    TAKER_SELL = "taker_sell"   # Hit the bid
    MAKER_BUY = "maker_buy"     # Bid was hit
    MAKER_SELL = "maker_sell"   # Ask was lifted
    SIMULATED = "simulated"     # No market data, simulated


@dataclass
class Fill:
    """
    A simulated fill.

    Represents execution of part or all of an order.
    """
    timestamp_utc: datetime
    bucket: str
    side: str  # "buy" or "sell"
    size: float
    price: float
    fill_type: FillType

    # Costs
    fees: float = 0.0
    slippage: float = 0.0

    # Adverse selection tracking
    price_1s_after: Optional[float] = None
    price_5s_after: Optional[float] = None
    price_30s_after: Optional[float] = None

    # Metadata
    order_id: Optional[str] = None
    signal_id: Optional[str] = None
    outcome: str = "yes"  # "yes" or "no" token (Polymarket binary market)

    @property
    def total_cost(self) -> float:
        """Total execution cost (fees + slippage)."""
        return self.fees + self.slippage

    @property
    def effective_price(self) -> float:
        """Price including costs."""
        if self.side == "buy":
            return self.price + self.total_cost
        else:
            return self.price - self.total_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "bucket": self.bucket,
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "fill_type": self.fill_type.value,
            "fees": self.fees,
            "slippage": self.slippage,
            "total_cost": self.total_cost,
            "effective_price": self.effective_price,
            "price_1s_after": self.price_1s_after,
            "price_5s_after": self.price_5s_after,
            "price_30s_after": self.price_30s_after,
            "outcome": self.outcome,
        }


@dataclass
class Order:
    """
    A pending order in the simulator.
    """
    order_id: str
    timestamp_utc: datetime
    bucket: str
    side: str  # "buy" or "sell"
    size: float
    limit_price: float
    order_type: str  # "taker" or "maker"

    # Execution state
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "pending"  # pending, partial, filled, cancelled, expired

    # Maker queue tracking
    queue_position: float = 0.0  # Estimated queue ahead
    time_at_price: float = 0.0   # Seconds at this price level

    # Metadata
    ttl_seconds: float = 300
    signal_id: Optional[str] = None
    outcome: str = "yes"  # "yes" or "no" token (Polymarket binary market)

    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size

    def is_active(self) -> bool:
        return self.status in ("pending", "partial")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "bucket": self.bucket,
            "side": self.side,
            "size": self.size,
            "limit_price": self.limit_price,
            "order_type": self.order_type,
            "filled_size": self.filled_size,
            "avg_fill_price": self.avg_fill_price,
            "status": self.status,
            "queue_position": self.queue_position,
            "time_at_price": self.time_at_price,
            "ttl_seconds": self.ttl_seconds,
            "signal_id": self.signal_id,
        }


class TakerModel:
    """
    Taker execution model.

    Executes immediately at best bid/ask with slippage.
    This is the "honest baseline" - if you can't win with taker, don't trade.
    """

    def __init__(
        self,
        fee_rate: float = 0.0,         # Polymarket: 0% fees on most markets
        base_slippage: float = 0.005,   # 0.5% base slippage
        size_impact: float = 0.01,      # Additional slippage per unit size
    ):
        self.fee_rate = fee_rate
        self.base_slippage = base_slippage
        self.size_impact = size_impact

    def execute(
        self,
        timestamp_utc: datetime,
        bucket: str,
        side: str,
        size: float,
        market_state: Optional[Dict] = None,
        signal_id: Optional[str] = None,
        min_order_usd: float = 0.0,
        outcome: str = "yes",
    ) -> Fill:
        """
        Execute a taker order.

        Args:
            timestamp_utc: Execution time
            bucket: Bucket to trade
            side: "buy" or "sell"
            size: Size to execute
            market_state: Market data with bid/ask (YES token book)
            signal_id: Optional signal reference
            min_order_usd: Minimum notional; size is bumped up if needed
            outcome: "yes" or "no" token to trade

        Returns:
            Fill object
        """
        # Get YES token prices from market state
        if market_state and bucket in market_state:
            bucket_data = market_state[bucket]
            yes_bid = bucket_data.get("best_bid", 0.5)
            yes_ask = bucket_data.get("best_ask", 0.5)
            bid_depth = bucket_data.get("bid_depth", 100)
            ask_depth = bucket_data.get("ask_depth", 100)
        else:
            # Simulated: assume 50/50 with 4% spread
            yes_bid = 0.48
            yes_ask = 0.52
            bid_depth = 100
            ask_depth = 100

        # Derive NO token prices from YES book (complementary)
        # NO ask ≈ 1 - YES bid (buying NO from someone selling NO)
        # NO bid ≈ 1 - YES ask (selling NO to someone buying NO)
        if outcome == "no":
            best_bid = max(0.01, 1.0 - yes_ask)
            best_ask = min(0.99, 1.0 - yes_bid)
        else:
            best_bid = yes_bid
            best_ask = yes_ask

        # Enforce minimum notional
        if min_order_usd > 0.0:
            ref_price = best_ask if side == "buy" else best_bid
            if ref_price and ref_price > 0 and size * ref_price < min_order_usd:
                size = min_order_usd / ref_price

        # Calculate execution price
        if side == "buy":
            # Buying: pay the ask + slippage
            base_price = best_ask
            depth_ratio = size / ask_depth if ask_depth > 0 else 1.0
            slippage = self.base_slippage + self.size_impact * depth_ratio
            price = base_price * (1 + slippage)
            fill_type = FillType.TAKER_BUY
        else:
            # Selling: receive the bid - slippage
            base_price = best_bid
            depth_ratio = size / bid_depth if bid_depth > 0 else 1.0
            slippage = self.base_slippage + self.size_impact * depth_ratio
            price = base_price * (1 - slippage)
            fill_type = FillType.TAKER_SELL

        # Calculate fees
        fees = price * size * self.fee_rate

        return Fill(
            timestamp_utc=timestamp_utc,
            bucket=bucket,
            side=side,
            size=size,
            price=price,
            fill_type=fill_type,
            fees=fees,
            slippage=abs(price - base_price),
            signal_id=signal_id,
            outcome=outcome,
        )


class MakerModel:
    """
    Maker execution model.

    Simulates queue-based fills for limit orders.
    More complex and more realistic than taker.

    WARNING: This is an approximation. Real maker fills are harder to predict.
    """

    def __init__(
        self,
        fee_rate: float = 0.0,          # Often 0 for makers (rebates)
        queue_consumption_rate: float = 0.1,  # Fraction of queue consumed per update
        min_time_at_price: float = 5.0,  # Min seconds before fill possible
        adverse_selection_prob: float = 0.3,  # Probability of being picked off
    ):
        self.fee_rate = fee_rate
        self.queue_consumption_rate = queue_consumption_rate
        self.min_time_at_price = min_time_at_price
        self.adverse_selection_prob = adverse_selection_prob

        # Pending orders
        self._orders: Dict[str, Order] = {}
        self._next_order_id = 1
        self._order_history: List[Order] = []

    def place_order(
        self,
        timestamp_utc: datetime,
        bucket: str,
        side: str,
        size: float,
        limit_price: float,
        market_state: Optional[Dict] = None,
        signal_id: Optional[str] = None,
        ttl_seconds: float = 300,
        outcome: str = "yes",
    ) -> Order:
        """
        Place a maker order.

        Returns:
            Order object (may not be filled immediately)
        """
        order_id = f"M{self._next_order_id:06d}"
        self._next_order_id += 1

        # Estimate queue position
        queue_position = self._estimate_queue(bucket, side, limit_price, market_state)

        order = Order(
            order_id=order_id,
            timestamp_utc=timestamp_utc,
            bucket=bucket,
            side=side,
            size=size,
            limit_price=limit_price,
            order_type="maker",
            outcome=outcome,
            queue_position=queue_position,
            ttl_seconds=ttl_seconds,
            signal_id=signal_id,
        )

        self._orders[order_id] = order
        self._order_history.append(order)
        return order

    def _estimate_queue(
        self,
        bucket: str,
        side: str,
        price: float,
        market_state: Optional[Dict]
    ) -> float:
        """Estimate queue ahead of us at this price."""
        if not market_state or bucket not in market_state:
            return 100.0  # Conservative default

        bucket_data = market_state[bucket]

        if side == "buy":
            # Bidding: how much is already at or above our price?
            best_bid = bucket_data.get("best_bid", 0.5)
            if price < best_bid:
                # We're behind the best bid
                return bucket_data.get("bid_depth", 100) * 2
            elif price == best_bid:
                # At best bid, behind existing queue
                return bucket_data.get("bid_depth", 100)
            else:
                # Improving the bid, at front
                return 0.0
        else:
            # Asking: how much is already at or below our price?
            best_ask = bucket_data.get("best_ask", 0.5)
            if price > best_ask:
                return bucket_data.get("ask_depth", 100) * 2
            elif price == best_ask:
                return bucket_data.get("ask_depth", 100)
            else:
                return 0.0

    def update(
        self,
        timestamp_utc: datetime,
        market_state: Optional[Dict],
        elapsed_seconds: float = 1.0
    ) -> List[Fill]:
        """
        Update all pending orders and check for fills.

        Call this at each time step.

        Returns:
            List of fills that occurred
        """
        fills = []
        orders_to_remove = []

        for order_id, order in self._orders.items():
            if not order.is_active():
                orders_to_remove.append(order_id)
                continue

            # Check expiration
            age = (timestamp_utc - order.timestamp_utc).total_seconds()
            if age > order.ttl_seconds:
                order.status = "expired"
                orders_to_remove.append(order_id)
                continue

            # Update time at price
            order.time_at_price += elapsed_seconds

            # Get current market
            if market_state and order.bucket in market_state:
                bucket_data = market_state[order.bucket]
                best_bid = bucket_data.get("best_bid", 0.5)
                best_ask = bucket_data.get("best_ask", 0.5)
            else:
                continue  # Can't simulate without market data

            # Check if price crossed (guaranteed fill)
            crossed = False
            if order.side == "buy" and best_ask <= order.limit_price:
                crossed = True
            elif order.side == "sell" and best_bid >= order.limit_price:
                crossed = True

            # Simulate queue consumption
            order.queue_position = max(
                0,
                order.queue_position - order.queue_position * self.queue_consumption_rate * elapsed_seconds
            )

            # Check fill conditions
            should_fill = False
            if crossed:
                should_fill = True
            elif order.queue_position <= 0 and order.time_at_price >= self.min_time_at_price:
                # Queue cleared and been waiting long enough
                should_fill = True

            if should_fill:
                fill = self._create_fill(order, timestamp_utc)
                fills.append(fill)

                order.filled_size = order.size
                order.avg_fill_price = fill.price
                order.status = "filled"
                orders_to_remove.append(order_id)

        # Clean up
        for order_id in orders_to_remove:
            del self._orders[order_id]

        return fills

    def _create_fill(self, order: Order, timestamp_utc: datetime) -> Fill:
        """Create a fill for an order."""
        if order.side == "buy":
            fill_type = FillType.MAKER_BUY
        else:
            fill_type = FillType.MAKER_SELL

        fees = order.limit_price * order.size * self.fee_rate

        return Fill(
            timestamp_utc=timestamp_utc,
            bucket=order.bucket,
            side=order.side,
            size=order.size,
            price=order.limit_price,
            fill_type=fill_type,
            fees=fees,
            slippage=0.0,  # Maker fills at limit
            order_id=order.order_id,
            signal_id=order.signal_id,
            outcome=getattr(order, "outcome", "yes"),
        )

    def cancel_all(self):
        """Cancel all pending orders."""
        for order in self._orders.values():
            order.status = "cancelled"
        self._orders.clear()

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return [o for o in self._orders.values() if o.is_active()]

    def get_order_history(self) -> List[Order]:
        """Get all orders (including filled/expired)."""
        return list(self._order_history)

    def clear_history(self):
        """Clear order history."""
        self._order_history = []

    def reset(self):
        """Reset maker model state."""
        self._orders.clear()
        self._order_history = []
        self._next_order_id = 1


class ExecutionSimulator:
    """
    Combined execution simulator.

    Handles both taker and maker orders.
    """

    def __init__(
        self,
        taker_model: Optional[TakerModel] = None,
        maker_model: Optional[MakerModel] = None,
        min_order_usd: float = 0.01,
    ):
        self.taker = taker_model or TakerModel()
        self.maker = maker_model or MakerModel()
        self.min_order_usd = min_order_usd

        # Fill history
        self._fills: List[Fill] = []

        # Statistics
        self._total_volume: float = 0.0
        self._total_fees: float = 0.0
        self._total_slippage: float = 0.0

    def reset(self):
        """Reset simulator state."""
        self._fills = []
        self._total_volume = 0.0
        self._total_fees = 0.0
        self._total_slippage = 0.0
        self.maker.reset()

    def execute_signal(
        self,
        signal: "PolicySignal",
        market_state: Optional[Dict] = None
    ) -> Optional[Fill]:
        """
        Execute a policy signal.

        Args:
            signal: PolicySignal to execute
            market_state: Current market data

        Returns:
            Fill if executed as taker, None if placed as maker
        """
        from .policy import OrderType

        outcome = getattr(signal, "outcome", "yes")

        if signal.order_type == OrderType.TAKER:
            fill = self.taker.execute(
                timestamp_utc=signal.timestamp_utc,
                bucket=signal.bucket,
                side=signal.side.value,
                size=signal.target_size,
                market_state=market_state,
                min_order_usd=self.min_order_usd,
                outcome=outcome,
            )
            self._record_fill(fill)
            return fill

        else:  # MAKER
            # Enforce minimum notional on maker orders
            target_size = signal.target_size
            limit_price = signal.max_price if signal.side.value == "buy" else signal.min_price
            if self.min_order_usd > 0.0 and limit_price > 0 and target_size * limit_price < self.min_order_usd:
                target_size = self.min_order_usd / limit_price

            self.maker.place_order(
                timestamp_utc=signal.timestamp_utc,
                bucket=signal.bucket,
                side=signal.side.value,
                size=target_size,
                limit_price=limit_price,
                market_state=market_state,
                ttl_seconds=signal.ttl_seconds,
                outcome=outcome,
            )
            return None

    def update(
        self,
        timestamp_utc: datetime,
        market_state: Optional[Dict],
        elapsed_seconds: float = 1.0
    ) -> List[Fill]:
        """
        Update simulator (process maker orders).

        Returns:
            List of maker fills
        """
        fills = self.maker.update(timestamp_utc, market_state, elapsed_seconds)
        for fill in fills:
            self._record_fill(fill)
        return fills

    def _record_fill(self, fill: Fill):
        """Record a fill and update statistics."""
        self._fills.append(fill)
        self._total_volume += fill.size
        self._total_fees += fill.fees
        self._total_slippage += fill.slippage

    def calculate_adverse_selection(
        self,
        market_history: List[Dict],
        intervals: List[int] = [1, 5, 30]
    ):
        """
        Calculate adverse selection metrics.

        Measures price movement after fills to detect "being picked off".

        Args:
            market_history: Time-ordered market snapshots
            intervals: Seconds after fill to measure
        """
        for fill in self._fills:
            fill_time = fill.timestamp_utc

            for seconds in intervals:
                target_time = fill_time + timedelta(seconds=seconds)

                # Find market state at target time
                price_after = None
                for snapshot in market_history:
                    snap_time = snapshot.get("timestamp_utc")
                    if isinstance(snap_time, str):
                        snap_time = datetime.fromisoformat(snap_time.replace("Z", "+00:00"))

                    if snap_time and snap_time >= target_time:
                        bucket_data = snapshot.get(fill.bucket, {})
                        price_after = (
                            bucket_data.get("best_bid", 0) +
                            bucket_data.get("best_ask", 0)
                        ) / 2
                        break

                # Store result
                if price_after is not None:
                    if seconds == 1:
                        fill.price_1s_after = price_after
                    elif seconds == 5:
                        fill.price_5s_after = price_after
                    elif seconds == 30:
                        fill.price_30s_after = price_after

    def get_fills(self) -> List[Fill]:
        """Get all fills."""
        return self._fills

    def get_orders(self) -> List[Order]:
        """Get all orders (including filled/expired)."""
        return self.maker.get_order_history()

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._fills:
            return {
                "total_fills": 0,
                "total_volume": 0.0,
                "total_fees": 0.0,
                "total_slippage": 0.0,
            }

        # Breakdown by type
        taker_fills = [f for f in self._fills if f.fill_type in (FillType.TAKER_BUY, FillType.TAKER_SELL)]
        maker_fills = [f for f in self._fills if f.fill_type in (FillType.MAKER_BUY, FillType.MAKER_SELL)]

        # Adverse selection analysis
        adverse_selection = []
        for fill in self._fills:
            if fill.price_5s_after is not None:
                # Positive = price moved against us after fill
                if fill.side == "buy":
                    adverse = fill.price_5s_after - fill.price
                else:
                    adverse = fill.price - fill.price_5s_after
                adverse_selection.append(adverse)

        avg_adverse = sum(adverse_selection) / len(adverse_selection) if adverse_selection else 0.0

        return {
            "total_fills": len(self._fills),
            "total_volume": self._total_volume,
            "total_fees": self._total_fees,
            "total_slippage": self._total_slippage,
            "avg_cost_per_fill": (self._total_fees + self._total_slippage) / len(self._fills),
            "by_type": {
                "taker_count": len(taker_fills),
                "taker_volume": sum(f.size for f in taker_fills),
                "maker_count": len(maker_fills),
                "maker_volume": sum(f.size for f in maker_fills),
            },
            "adverse_selection": {
                "avg_5s_move": avg_adverse,
                "samples": len(adverse_selection),
            },
        }
