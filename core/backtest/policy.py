"""
Policy Module - Trading Policy Engine

Defines trading policies that convert NowcastDistribution + market state
into actionable trading decisions.

Components:
- PolicyState: State machine for position management
- PolicyTable: Value levels and sizing rules
- Policy: Main policy engine combining state + rules
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
from zoneinfo import ZoneInfo

from core.polymarket_labels import normalize_label
logger = logging.getLogger("backtest.policy")

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


def _safe_float(value: Any) -> Optional[float]:
    """Return float value when valid, otherwise None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _resolve_mid_price(
    bucket_market: Optional[Dict[str, Any]],
    fallback_mid: float,
    default_spread: float = 0.04,
) -> float:
    """
    Build a robust mid-price from bucket market data.
    Falls back to model-prob-based synthetic spread if bid/ask is missing.
    """
    if not bucket_market:
        return fallback_mid

    bid = _safe_float(bucket_market.get("best_bid"))
    ask = _safe_float(bucket_market.get("best_ask"))

    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return (bid + (bid + default_spread)) / 2.0
    if ask is not None:
        return ((ask - default_spread) + ask) / 2.0
    return fallback_mid


class PolicyState(Enum):
    """
    Position management state machine.

    States prevent overtrading and manage risk through the day.
    """
    NEUTRAL = "neutral"           # No position, waiting for opportunity
    BUILD_POSITION = "build"      # Actively building position
    HOLD = "hold"                 # Position taken, holding
    REDUCE = "reduce"             # Actively reducing position
    FADE_EVENT = "fade"           # Short-term fade of suspected bad print
    RISK_OFF = "risk_off"         # Data quality issues, avoid trading


class Side(Enum):
    """Order side."""
    BUY = "buy"      # Buy tokens (YES or NO depending on outcome)
    SELL = "sell"    # Sell tokens (closing existing position only)


class OrderType(Enum):
    """Order execution type."""
    TAKER = "taker"  # Execute at market
    MAKER = "maker"  # Post limit order


class NoTradeReason(Enum):
    """Explicit reasons why no trade was generated."""
    NO_NOWCAST = "no_nowcast"             # No nowcast data available
    NO_MARKET_DATA = "no_market_data"     # No market state available
    STALE_NOWCAST = "stale_nowcast"       # Nowcast data too old
    STALE_MARKET = "stale_market"         # Market data too old
    LOW_CONFIDENCE = "low_confidence"      # Confidence below threshold
    QC_BLOCKED = "qc_blocked"             # QC flags prevent trading
    EDGE_TOO_SMALL = "edge_too_small"     # Edge below entry threshold
    DAILY_LOSS_LIMIT = "daily_loss_limit" # Hit daily loss limit
    COOLDOWN_BLOCKED = "cooldown_blocked" # In cooldown period
    INVENTORY_BLOCKED = "inventory_blocked"  # Position limits reached
    SIZE_TOO_SMALL = "size_too_small"     # Calculated size below minimum


@dataclass
class PolicySignal:
    """
    A trading signal from the policy.

    Represents what the policy "wants" to do, not necessarily
    what gets executed.
    """
    timestamp_utc: datetime
    bucket: str
    side: Side
    order_type: OrderType

    # Size and limits
    target_size: float      # Desired position size (0-1 normalized)
    max_price: float        # Max price willing to pay (for BUY)
    min_price: float        # Min price willing to accept (for SELL)

    # Confidence and urgency
    confidence: float       # Signal confidence (0-1)
    urgency: float          # How urgently we need to execute (0-1)

    # Risk parameters
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    ttl_seconds: float = 300  # Signal validity

    # Metadata
    reason: str = ""
    policy_state: PolicyState = PolicyState.NEUTRAL
    outcome: str = "yes"  # "yes" or "no" token (Polymarket binary market)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "bucket": self.bucket,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "target_size": self.target_size,
            "max_price": self.max_price,
            "min_price": self.min_price,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "ttl_seconds": self.ttl_seconds,
            "reason": self.reason,
            "policy_state": self.policy_state.value,
            "outcome": self.outcome,
        }


@dataclass
class PolicyEvaluationResult:
    """Result of a policy evaluation with explicit reasons."""
    signals: List[PolicySignal] = field(default_factory=list)
    no_trade_reasons: List[NoTradeReason] = field(default_factory=list)
    edge_by_bucket: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals_count": len(self.signals),
            "signals": [s.to_dict() for s in self.signals],
            "no_trade_reasons": [r.value for r in self.no_trade_reasons],
            "edge_by_bucket": self.edge_by_bucket,
        }


@dataclass
class Position:
    """Current position state."""
    bucket: str
    size: float = 0.0        # -1 to 1 (negative = short)
    avg_price: float = 0.0   # Average entry price
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Entry info
    entry_time: Optional[datetime] = None
    entry_count: int = 0

    def is_flat(self) -> bool:
        """Check if position is flat."""
        return abs(self.size) < 0.001


@dataclass
class PolicyTable:
    """
    Value levels and sizing rules by bucket.

    Determines:
    - Fair value for each bucket
    - Edge required to trade
    - Position sizing
    """
    # Edge thresholds
    min_edge_to_enter: float = 0.05      # 5% edge minimum
    min_edge_to_add: float = 0.08        # 8% edge to add to position
    edge_to_fade: float = 0.10           # 10% edge to fade events

    # Position sizing
    max_position_per_bucket: float = 1.0  # Max normalized position
    max_total_exposure: float = 3.0       # Total across all buckets
    size_per_confidence: float = 0.5      # Size = confidence * this

    # Risk limits
    max_daily_loss: float = 0.20          # 20% max daily loss
    position_decay_rate: float = 0.1      # Reduce position urgency over time

    # QC gating
    min_confidence_to_trade: float = 0.5
    max_staleness_seconds: float = 600

    def calculate_edge(
        self,
        model_prob: float,
        market_price: float,
        side: Side
    ) -> float:
        """
        Calculate edge for a potential trade.

        Args:
            model_prob: Model's probability for the bucket
            market_price: Current market price
            side: BUY or SELL

        Returns:
            Edge (positive = favorable)
        """
        if side == Side.BUY:
            # Buying YES: want model_prob > market_price
            return model_prob - market_price
        else:
            # Selling YES: want model_prob < market_price
            return market_price - model_prob

    def calculate_size(
        self,
        edge: float,
        confidence: float,
        current_position: float
    ) -> float:
        """
        Calculate position size for a trade.

        Uses a simple Kelly-like sizing: size proportional to edge * confidence.

        Args:
            edge: Expected edge
            confidence: Model confidence
            current_position: Current position in bucket

        Returns:
            Target position delta
        """
        # Base size from confidence
        base_size = confidence * self.size_per_confidence

        # Scale by edge (more edge = larger size)
        edge_multiplier = min(2.0, edge / self.min_edge_to_enter)
        target_size = base_size * edge_multiplier

        # Cap at max
        max_delta = self.max_position_per_bucket - abs(current_position)
        return min(target_size, max_delta)


class Policy:
    """
    Main policy engine.

    Converts nowcast + market state into trading signals.
    """

    def __init__(
        self,
        policy_table: Optional[PolicyTable] = None,
        name: str = "default"
    ):
        self.name = name
        self.table = policy_table or PolicyTable()

        # State machine
        self._state = PolicyState.NEUTRAL
        self._state_entered_at: Optional[datetime] = None

        # Positions by bucket
        self._positions: Dict[str, Position] = {}

        # Risk tracking
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0

        # Cooldowns
        self._last_trade_time: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None

    def reset(self):
        """Reset policy state for new day."""
        self._state = PolicyState.NEUTRAL
        self._state_entered_at = None
        self._positions = {}
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_trade_time = None
        self._cooldown_until = None

    def get_state(self) -> PolicyState:
        """Get current policy state."""
        return self._state

    def set_state(self, state: PolicyState, timestamp: datetime):
        """Set policy state."""
        if state != self._state:
            logger.debug(f"Policy state: {self._state.value} -> {state.value}")
            self._state = state
            self._state_entered_at = timestamp

    def get_position(self, bucket: str) -> Position:
        """Get position for a bucket."""
        if bucket not in self._positions:
            self._positions[bucket] = Position(bucket=bucket)
        return self._positions[bucket]

    def evaluate(
        self,
        timestamp_utc: datetime,
        nowcast: Dict,
        market_state: Optional[Dict] = None,
        qc_flags: Optional[List[str]] = None
    ) -> List[PolicySignal]:
        """
        Evaluate current state and generate signals.

        Args:
            timestamp_utc: Current timestamp
            nowcast: Nowcast output with p_bucket, tmax_mean, confidence
            market_state: Market data (bid/ask by bucket)
            qc_flags: Current QC flags

        Returns:
            List of PolicySignals (may be empty)
        """
        signals = []

        # Check QC gating
        confidence = nowcast.get("confidence", 0)
        if confidence < self.table.min_confidence_to_trade:
            self.set_state(PolicyState.RISK_OFF, timestamp_utc)
            return signals

        if qc_flags and any(f in qc_flags for f in ["STALE", "GAP", "ERROR"]):
            self.set_state(PolicyState.RISK_OFF, timestamp_utc)
            return signals

        # Check daily loss limit
        if self._daily_pnl < -self.table.max_daily_loss:
            self.set_state(PolicyState.RISK_OFF, timestamp_utc)
            return signals

        # Check cooldown
        if self._cooldown_until and timestamp_utc < self._cooldown_until:
            return signals

        # Get bucket probabilities
        p_bucket = nowcast.get("p_bucket", [])
        if not p_bucket:
            return signals

        # Convert to dict format
        bucket_probs = {}
        for b in p_bucket:
            if isinstance(b, dict):
                label = normalize_label(b.get("label", ""))
                prob = b.get("probability", b.get("prob", 0))
            else:
                continue
            bucket_probs[label] = prob

        # Evaluate each bucket
        for bucket, model_prob in bucket_probs.items():
            signal = self._evaluate_bucket(
                timestamp_utc=timestamp_utc,
                bucket=bucket,
                model_prob=model_prob,
                confidence=confidence,
                market_state=market_state
            )
            if signal:
                signals.append(signal)

        # Update state based on signals
        if signals:
            if self._state == PolicyState.NEUTRAL:
                self.set_state(PolicyState.BUILD_POSITION, timestamp_utc)
        elif self._state == PolicyState.BUILD_POSITION:
            # No signals, check if we should hold
            total_position = sum(abs(p.size) for p in self._positions.values())
            if total_position > 0:
                self.set_state(PolicyState.HOLD, timestamp_utc)
            else:
                self.set_state(PolicyState.NEUTRAL, timestamp_utc)

        return signals

    def _evaluate_bucket(
        self,
        timestamp_utc: datetime,
        bucket: str,
        model_prob: float,
        confidence: float,
        market_state: Optional[Dict]
    ) -> Optional[PolicySignal]:
        """Evaluate a single bucket for trading opportunity."""

        # Get market price (or use model prob as fallback)
        market_price = model_prob  # Default: no edge
        if market_state:
            bucket_market = market_state.get(bucket, {})
            market_price = _resolve_mid_price(bucket_market, fallback_mid=model_prob)

        # Get current position
        position = self.get_position(bucket)

        # Determine direction
        # BUY YES = bracket is underpriced (model > market)
        # BUY NO  = bracket is overpriced  (model < market)
        if model_prob > market_price:
            outcome = "yes"
            edge = model_prob - market_price
        else:
            outcome = "no"
            edge = market_price - model_prob

        # Check edge threshold
        min_edge = self.table.min_edge_to_enter
        if not position.is_flat():
            min_edge = self.table.min_edge_to_add

        if edge < min_edge:
            return None

        # Calculate size
        target_size = self.table.calculate_size(edge, confidence, position.size)
        if target_size < 0.01:
            return None

        # Determine order type based on urgency
        urgency = min(1.0, edge / 0.20)  # More edge = more urgent
        order_type = OrderType.TAKER if urgency > 0.7 else OrderType.MAKER

        # Calculate price limits for the token being traded
        if outcome == "yes":
            # Buying YES: pay up to model_prob minus safety margin
            max_price = model_prob - (self.table.min_edge_to_enter / 2)
            min_price = 0.0
        else:
            # Buying NO: NO price ≈ 1 - YES price
            no_model_price = 1.0 - model_prob
            max_price = no_model_price - (self.table.min_edge_to_enter / 2)
            min_price = 0.0

        return PolicySignal(
            timestamp_utc=timestamp_utc,
            bucket=bucket,
            side=Side.BUY,
            order_type=order_type,
            target_size=target_size,
            max_price=max_price,
            min_price=min_price,
            confidence=confidence,
            urgency=urgency,
            outcome=outcome,
            reason=f"Edge={edge:.3f}, Model={model_prob:.3f}, Market={market_price:.3f}",
            policy_state=self._state,
        )

    def evaluate_with_reasons(
        self,
        timestamp_utc: datetime,
        nowcast: Optional[Dict],
        market_state: Optional[Dict] = None,
        qc_flags: Optional[List[str]] = None,
        nowcast_age_seconds: Optional[float] = None,
        market_age_seconds: Optional[float] = None
    ) -> PolicyEvaluationResult:
        """
        Evaluate current state and generate signals with explicit reasons.

        Returns:
            PolicyEvaluationResult with signals and no-trade reasons
        """
        result = PolicyEvaluationResult()

        # Check for missing nowcast
        if not nowcast:
            result.no_trade_reasons.append(NoTradeReason.NO_NOWCAST)
            return result

        # Check staleness gating
        if (
            nowcast_age_seconds is not None
            and not math.isinf(nowcast_age_seconds)
            and nowcast_age_seconds > self.table.max_staleness_seconds
        ):
            self.set_state(PolicyState.RISK_OFF, timestamp_utc)
            result.no_trade_reasons.append(NoTradeReason.STALE_NOWCAST)
            return result

        if (
            market_age_seconds is not None
            and not math.isinf(market_age_seconds)
            and market_age_seconds > self.table.max_staleness_seconds
        ):
            self.set_state(PolicyState.RISK_OFF, timestamp_utc)
            result.no_trade_reasons.append(NoTradeReason.STALE_MARKET)
            return result

        # Check QC gating
        confidence = nowcast.get("confidence", 0)
        if confidence < self.table.min_confidence_to_trade:
            self.set_state(PolicyState.RISK_OFF, timestamp_utc)
            result.no_trade_reasons.append(NoTradeReason.LOW_CONFIDENCE)
            return result

        if qc_flags and any(f in qc_flags for f in ["STALE", "GAP", "ERROR"]):
            self.set_state(PolicyState.RISK_OFF, timestamp_utc)
            result.no_trade_reasons.append(NoTradeReason.QC_BLOCKED)
            return result

        # Check daily loss limit
        if self._daily_pnl < -self.table.max_daily_loss:
            self.set_state(PolicyState.RISK_OFF, timestamp_utc)
            result.no_trade_reasons.append(NoTradeReason.DAILY_LOSS_LIMIT)
            return result

        # Check cooldown
        if self._cooldown_until and timestamp_utc < self._cooldown_until:
            result.no_trade_reasons.append(NoTradeReason.COOLDOWN_BLOCKED)
            return result

        # Get bucket probabilities
        p_bucket = nowcast.get("p_bucket", [])
        if not p_bucket:
            result.no_trade_reasons.append(NoTradeReason.NO_NOWCAST)
            return result

        # Check for missing market data
        if not market_state:
            result.no_trade_reasons.append(NoTradeReason.NO_MARKET_DATA)

        # Convert to dict format
        bucket_probs = {}
        for b in p_bucket:
            if isinstance(b, dict):
                label = normalize_label(b.get("label", ""))
                prob = b.get("probability", b.get("prob", 0))
                bucket_probs[label] = prob

        # Track if any bucket had edge but was rejected
        had_edge_rejection = False
        had_size_rejection = False

        # Evaluate each bucket
        for bucket, model_prob in bucket_probs.items():
            signal, rejection_reason = self._evaluate_bucket_with_reason(
                timestamp_utc=timestamp_utc,
                bucket=bucket,
                model_prob=model_prob,
                confidence=confidence,
                market_state=market_state
            )

            # Track edge for diagnostics
            if market_state and bucket in market_state:
                bucket_market = market_state[bucket]
                mid = _resolve_mid_price(bucket_market, fallback_mid=model_prob)
                edge = model_prob - mid
                result.edge_by_bucket[bucket] = edge

            if signal:
                result.signals.append(signal)
            elif rejection_reason == NoTradeReason.EDGE_TOO_SMALL:
                had_edge_rejection = True
            elif rejection_reason == NoTradeReason.SIZE_TOO_SMALL:
                had_size_rejection = True

        # Add consolidated rejection reasons
        if not result.signals:
            if had_edge_rejection:
                result.no_trade_reasons.append(NoTradeReason.EDGE_TOO_SMALL)
            if had_size_rejection:
                result.no_trade_reasons.append(NoTradeReason.SIZE_TOO_SMALL)

        # Update state based on signals
        if result.signals:
            if self._state == PolicyState.NEUTRAL:
                self.set_state(PolicyState.BUILD_POSITION, timestamp_utc)
        elif self._state == PolicyState.BUILD_POSITION:
            total_position = sum(abs(p.size) for p in self._positions.values())
            if total_position > 0:
                self.set_state(PolicyState.HOLD, timestamp_utc)
            else:
                self.set_state(PolicyState.NEUTRAL, timestamp_utc)

        return result

    def _evaluate_bucket_with_reason(
        self,
        timestamp_utc: datetime,
        bucket: str,
        model_prob: float,
        confidence: float,
        market_state: Optional[Dict]
    ) -> Tuple[Optional[PolicySignal], Optional[NoTradeReason]]:
        """Evaluate a single bucket and return signal or rejection reason."""

        # Get market price (or use model prob as fallback)
        market_price = model_prob
        if market_state and bucket in market_state:
            bucket_market = market_state[bucket]
            market_price = _resolve_mid_price(bucket_market, fallback_mid=model_prob)

        # Get current position
        position = self.get_position(bucket)

        # Determine direction and edge
        # BUY YES = bracket is underpriced (model > market)
        # BUY NO  = bracket is overpriced  (model < market)
        if model_prob > market_price:
            outcome = "yes"
            edge = model_prob - market_price
        else:
            outcome = "no"
            edge = market_price - model_prob

        # Check edge threshold
        min_edge = self.table.min_edge_to_enter
        if not position.is_flat():
            min_edge = self.table.min_edge_to_add

        if edge < min_edge:
            return None, NoTradeReason.EDGE_TOO_SMALL

        # Calculate size
        target_size = self.table.calculate_size(edge, confidence, position.size)
        if target_size < 0.01:
            return None, NoTradeReason.SIZE_TOO_SMALL

        # Determine order type based on urgency
        urgency = min(1.0, edge / 0.20)
        order_type = OrderType.TAKER if urgency > 0.7 else OrderType.MAKER

        # Calculate price limits for the token being traded
        if outcome == "yes":
            max_price = model_prob - (self.table.min_edge_to_enter / 2)
            min_price = 0.0
        else:
            no_model_price = 1.0 - model_prob
            max_price = no_model_price - (self.table.min_edge_to_enter / 2)
            min_price = 0.0

        signal = PolicySignal(
            timestamp_utc=timestamp_utc,
            bucket=bucket,
            side=Side.BUY,
            order_type=order_type,
            target_size=target_size,
            max_price=max_price,
            min_price=min_price,
            confidence=confidence,
            urgency=urgency,
            outcome=outcome,
            reason=f"Edge={edge:.3f}, Model={model_prob:.3f}, Market={market_price:.3f}",
            policy_state=self._state,
        )

        return signal, None

    def record_fill(
        self,
        timestamp_utc: datetime,
        bucket: str,
        size: float,
        price: float,
        side: Side
    ):
        """Record a fill and update position."""
        position = self.get_position(bucket)

        # Update position
        if side == Side.BUY:
            delta = size
        else:
            delta = -size

        # Calculate new average price
        if position.is_flat():
            position.avg_price = price
            position.entry_time = timestamp_utc
        else:
            total_size = position.size + delta
            if abs(total_size) > 0.001:
                # Weighted average
                position.avg_price = (
                    position.avg_price * abs(position.size) +
                    price * abs(delta)
                ) / abs(total_size)

        position.size += delta
        position.entry_count += 1

        self._last_trade_time = timestamp_utc
        self._daily_trades += 1

    def record_pnl(self, pnl: float):
        """Record realized PnL."""
        self._daily_pnl += pnl

    def get_summary(self) -> Dict[str, Any]:
        """Get policy state summary."""
        return {
            "name": self.name,
            "state": self._state.value,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "positions": {
                b: {
                    "size": p.size,
                    "avg_price": p.avg_price,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for b, p in self._positions.items()
                if not p.is_flat()
            },
            "total_exposure": sum(abs(p.size) for p in self._positions.values()),
        }


# =============================================================================
# Pre-defined policies
# =============================================================================

def create_conservative_policy() -> Policy:
    """Create a conservative policy with strict risk limits."""
    table = PolicyTable(
        min_edge_to_enter=0.08,
        min_edge_to_add=0.12,
        max_position_per_bucket=0.5,
        max_total_exposure=1.5,
        min_confidence_to_trade=0.7,
        max_daily_loss=0.10,
    )
    return Policy(table, name="conservative")


def create_aggressive_policy() -> Policy:
    """Create an aggressive policy with looser risk limits."""
    table = PolicyTable(
        min_edge_to_enter=0.03,
        min_edge_to_add=0.05,
        max_position_per_bucket=1.0,
        max_total_exposure=5.0,
        min_confidence_to_trade=0.4,
        max_daily_loss=0.30,
    )
    return Policy(table, name="aggressive")


def create_fade_policy() -> Policy:
    """Create a policy specialized for fading events."""
    table = PolicyTable(
        min_edge_to_enter=0.10,
        min_edge_to_add=0.15,
        edge_to_fade=0.08,
        max_position_per_bucket=0.3,
        max_total_exposure=1.0,
        min_confidence_to_trade=0.6,
        max_daily_loss=0.15,
    )
    return Policy(table, name="fade")


class MakerPassivePolicy(Policy):
    """
    Maker-biased strategy:
    - uses normal edge logic
    - forces all generated signals to maker orders
    """

    def evaluate_with_reasons(
        self,
        timestamp_utc: datetime,
        nowcast: Optional[Dict],
        market_state: Optional[Dict] = None,
        qc_flags: Optional[List[str]] = None,
        nowcast_age_seconds: Optional[float] = None,
        market_age_seconds: Optional[float] = None
    ) -> PolicyEvaluationResult:
        result = super().evaluate_with_reasons(
            timestamp_utc=timestamp_utc,
            nowcast=nowcast,
            market_state=market_state,
            qc_flags=qc_flags,
            nowcast_age_seconds=nowcast_age_seconds,
            market_age_seconds=market_age_seconds,
        )
        for signal in result.signals:
            signal.order_type = OrderType.MAKER
            signal.reason = f"{signal.reason}; maker_passive"
        return result


def create_maker_passive_policy() -> MakerPassivePolicy:
    """Create a policy that prefers posting passive liquidity."""
    table = PolicyTable(
        min_edge_to_enter=0.04,
        min_edge_to_add=0.06,
        max_position_per_bucket=0.6,
        max_total_exposure=2.0,
        min_confidence_to_trade=0.5,
        max_daily_loss=0.12,
    )
    return MakerPassivePolicy(table, name="maker_passive")


# =============================================================================
# Debug/Test Policies
# =============================================================================

class ToyPolicyDebug(Policy):
    """
    Debug policy that generates 1 taker buy signal per hour.

    Used for testing that the backtest pipeline can produce fills.
    """

    def __init__(self):
        super().__init__(PolicyTable(), name="toy_debug")
        self._last_signal_hour: Optional[int] = None

    def reset(self):
        super().reset()
        self._last_signal_hour = None

    def evaluate_with_reasons(
        self,
        timestamp_utc: datetime,
        nowcast: Optional[Dict],
        market_state: Optional[Dict] = None,
        qc_flags: Optional[List[str]] = None,
        nowcast_age_seconds: Optional[float] = None,
        market_age_seconds: Optional[float] = None
    ) -> PolicyEvaluationResult:
        """Generate 1 taker buy per hour for testing."""
        result = PolicyEvaluationResult()

        if not nowcast:
            result.no_trade_reasons.append(NoTradeReason.NO_NOWCAST)
            return result

        # One signal per hour
        current_hour = timestamp_utc.hour
        if self._last_signal_hour == current_hour:
            result.no_trade_reasons.append(NoTradeReason.COOLDOWN_BLOCKED)
            return result

        # Get top predicted bucket
        p_bucket = nowcast.get("p_bucket", [])
        if not p_bucket:
            result.no_trade_reasons.append(NoTradeReason.NO_NOWCAST)
            return result

        # Find top bucket
        top_bucket = ""
        top_prob = 0
        for b in p_bucket:
            if isinstance(b, dict):
                prob = b.get("prob", b.get("probability", 0))
                if prob > top_prob:
                    top_prob = prob
                    top_bucket = normalize_label(b.get("label", ""))

        if not top_bucket:
            result.no_trade_reasons.append(NoTradeReason.NO_NOWCAST)
            return result

        # Create a simple taker buy signal
        signal = PolicySignal(
            timestamp_utc=timestamp_utc,
            bucket=top_bucket,
            side=Side.BUY,
            order_type=OrderType.TAKER,
            target_size=0.1,  # Small size
            max_price=0.99,
            min_price=0.01,
            confidence=1.0,
            urgency=1.0,
            reason="ToyDebug: 1 buy per hour",
            policy_state=PolicyState.BUILD_POSITION,
        )

        result.signals.append(signal)
        self._last_signal_hour = current_hour

        return result


def create_toy_debug_policy() -> ToyPolicyDebug:
    """Create a debug policy for pipeline testing."""
    return ToyPolicyDebug()
