"""Tests for market/paper_execution.py — L2 book-walking fill simulation."""

import pytest

from market.paper_execution import (
    PaperExecutionEngine,
    PaperFill,
    PaperOrder,
    _walk_book_buy,
    _walk_book_sell,
    compact_book_snapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_book(asks=None, bids=None):
    """Build a minimal L2 book snapshot."""
    return {
        "best_bid": float(bids[0]["price"]) if bids else None,
        "best_ask": float(asks[0]["price"]) if asks else None,
        "mid": None,
        "spread": None,
        "bid_depth": sum(float(b["size"]) for b in (bids or [])),
        "ask_depth": sum(float(a["size"]) for a in (asks or [])),
        "bids": bids or [],
        "asks": asks or [],
    }


THREE_LEVEL_ASK_BOOK = _make_book(
    asks=[
        {"price": 0.35, "size": 50.0},
        {"price": 0.36, "size": 80.0},
        {"price": 0.37, "size": 100.0},
    ],
    bids=[
        {"price": 0.33, "size": 60.0},
        {"price": 0.32, "size": 40.0},
        {"price": 0.31, "size": 30.0},
    ],
)


# ---------------------------------------------------------------------------
# limit_fak BUY tests
# ---------------------------------------------------------------------------

class TestLimitFakBuy:

    def test_walks_multiple_ask_levels(self):
        """Buy 100 shares at limit 0.37 — should walk 2 full levels + partial 3rd."""
        engine = PaperExecutionEngine()
        fill = engine.execute_limit_fak_buy(
            token_id="tok1",
            limit_price=0.37,
            size=100.0,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.status == "FILLED"
        assert fill.filled_size == 100.0
        # 50 @ 0.35 + 50 @ 0.36 = 17.5 + 18.0 = 35.5
        expected_notional = 50 * 0.35 + 50 * 0.36
        assert abs(fill.notional_usd - expected_notional) < 0.01
        assert fill.avg_price == pytest.approx(expected_notional / 100.0, abs=1e-4)
        assert fill.levels_consumed == 2
        assert fill.slippage_vs_best >= 0  # avg >= best_ask

    def test_partial_insufficient_depth(self):
        """Request more than the book has — partial fill."""
        engine = PaperExecutionEngine()
        fill = engine.execute_limit_fak_buy(
            token_id="tok1",
            limit_price=0.37,
            size=300.0,  # book only has 230 shares across 3 levels
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.status == "PARTIAL"
        assert fill.filled_size == 230.0  # 50 + 80 + 100
        assert fill.levels_consumed == 3

    def test_rejected_limit_below_best_ask(self):
        """Limit price below best ask — nothing fills."""
        engine = PaperExecutionEngine()
        fill = engine.execute_limit_fak_buy(
            token_id="tok1",
            limit_price=0.34,  # below best ask of 0.35
            size=50.0,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.status == "REJECTED"
        assert fill.filled_size == 0.0

    def test_exact_one_level(self):
        """Request exactly what the first level offers."""
        engine = PaperExecutionEngine()
        fill = engine.execute_limit_fak_buy(
            token_id="tok1",
            limit_price=0.35,
            size=50.0,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.status == "FILLED"
        assert fill.filled_size == 50.0
        assert fill.avg_price == pytest.approx(0.35, abs=1e-6)
        assert fill.levels_consumed == 1
        assert fill.slippage_vs_best == pytest.approx(0.0, abs=1e-6)

    def test_empty_book_rejected(self):
        """Empty ask book — rejected."""
        engine = PaperExecutionEngine()
        fill = engine.execute_limit_fak_buy(
            token_id="tok1",
            limit_price=0.50,
            size=10.0,
            book_snapshot=_make_book(asks=[], bids=[]),
        )
        assert fill.status == "REJECTED"


# ---------------------------------------------------------------------------
# limit_fak SELL tests
# ---------------------------------------------------------------------------

class TestLimitFakSell:

    def test_walks_multiple_bid_levels(self):
        """Sell 80 shares at min 0.31 — walks 2 levels."""
        engine = PaperExecutionEngine()
        fill = engine.execute_limit_fak_sell(
            token_id="tok1",
            min_price=0.31,
            size=80.0,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.status == "FILLED"
        assert fill.filled_size == 80.0
        # 60 @ 0.33 + 20 @ 0.32 = 19.8 + 6.4 = 26.2
        expected_notional = 60 * 0.33 + 20 * 0.32
        assert abs(fill.notional_usd - expected_notional) < 0.01
        assert fill.levels_consumed == 2

    def test_partial_sell(self):
        """Sell more than bid depth."""
        engine = PaperExecutionEngine()
        fill = engine.execute_limit_fak_sell(
            token_id="tok1",
            min_price=0.31,
            size=200.0,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.status == "PARTIAL"
        assert fill.filled_size == 130.0  # 60 + 40 + 30
        assert fill.levels_consumed == 3

    def test_rejected_min_above_best_bid(self):
        """Min price above best bid — nothing fills."""
        engine = PaperExecutionEngine()
        fill = engine.execute_limit_fak_sell(
            token_id="tok1",
            min_price=0.34,  # above best bid of 0.33
            size=50.0,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.status == "REJECTED"
        assert fill.filled_size == 0.0


# ---------------------------------------------------------------------------
# market_fok tests
# ---------------------------------------------------------------------------

class TestMarketFokBuy:

    def test_full_fill(self):
        """FOK buy with sufficient depth — fills."""
        engine = PaperExecutionEngine()
        fill = engine.execute_market_fok_buy(
            token_id="tok1",
            max_price=0.37,
            size=100.0,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.status == "FILLED"
        assert fill.filled_size == 100.0

    def test_reject_insufficient_depth(self):
        """FOK buy with insufficient depth — rejected (not partial)."""
        engine = PaperExecutionEngine()
        fill = engine.execute_market_fok_buy(
            token_id="tok1",
            max_price=0.37,
            size=300.0,  # more than 230 available
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.status == "REJECTED"
        assert fill.filled_size == 0.0


class TestMarketFokSell:

    def test_full_fill(self):
        """FOK sell with sufficient depth — fills."""
        engine = PaperExecutionEngine()
        fill = engine.execute_market_fok_sell(
            token_id="tok1",
            min_price=0.31,
            size=60.0,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.status == "FILLED"
        assert fill.filled_size == 60.0

    def test_reject_insufficient(self):
        """FOK sell with insufficient depth — rejected."""
        engine = PaperExecutionEngine()
        fill = engine.execute_market_fok_sell(
            token_id="tok1",
            min_price=0.31,
            size=200.0,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.status == "REJECTED"
        assert fill.filled_size == 0.0


# ---------------------------------------------------------------------------
# Order lifecycle tests
# ---------------------------------------------------------------------------

class TestOrderLifecycle:

    def test_place_order_filled(self):
        """place_order returns a FILLED order with snapshots."""
        engine = PaperExecutionEngine()
        order = engine.place_order(
            kind="entry",
            station_id="KLGA",
            target_date="2026-03-02",
            label="33-34°F",
            side="YES",
            token_id="tok1",
            order_type="limit_fak",
            direction="buy",
            size=50.0,
            limit_price=0.35,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert order.status == "FILLED"
        assert order.filled_size == 50.0
        assert order.remaining_size == pytest.approx(0.0, abs=1e-6)
        assert order.avg_fill_price == pytest.approx(0.35, abs=1e-6)
        assert order.book_snapshot_submit is not None
        assert order.book_snapshot_fill is not None
        assert order.submitted_at_utc != ""
        assert order.filled_at_utc != ""
        assert len(order.fills) == 1  # single level

    def test_place_order_rejected(self):
        """place_order with price below ask → REJECTED."""
        engine = PaperExecutionEngine()
        order = engine.place_order(
            kind="entry",
            station_id="KLGA",
            target_date="2026-03-02",
            label="33-34°F",
            side="YES",
            token_id="tok1",
            order_type="limit_fak",
            direction="buy",
            size=50.0,
            limit_price=0.30,  # below best ask
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert order.status == "REJECTED"
        assert order.filled_size == 0.0

    def test_place_order_partial(self):
        """place_order with insufficient depth → PARTIAL."""
        engine = PaperExecutionEngine()
        order = engine.place_order(
            kind="entry",
            station_id="KLGA",
            target_date="2026-03-02",
            label="33-34°F",
            side="YES",
            token_id="tok1",
            order_type="limit_fak",
            direction="buy",
            size=300.0,
            limit_price=0.37,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert order.status == "PARTIAL"
        assert order.filled_size == 230.0
        assert order.remaining_size == pytest.approx(70.0, abs=1e-4)

    def test_order_tracked_in_engine(self):
        """Orders are tracked and retrievable."""
        engine = PaperExecutionEngine()
        order = engine.place_order(
            kind="entry",
            station_id="KLGA",
            target_date="2026-03-02",
            label="33-34°F",
            side="YES",
            token_id="tok1",
            order_type="limit_fak",
            direction="buy",
            size=50.0,
            limit_price=0.35,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        retrieved = engine.get_order(order.order_id)
        assert retrieved is not None
        assert retrieved.order_id == order.order_id

    def test_sell_order(self):
        """place_order for sell direction."""
        engine = PaperExecutionEngine()
        order = engine.place_order(
            kind="exit",
            station_id="KLGA",
            target_date="2026-03-02",
            label="33-34°F",
            side="YES",
            token_id="tok1",
            order_type="limit_fak",
            direction="sell",
            size=60.0,
            limit_price=0.31,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert order.status == "FILLED"
        assert order.filled_size == 60.0
        assert order.direction == "sell"


# ---------------------------------------------------------------------------
# Slippage calculation tests
# ---------------------------------------------------------------------------

class TestSlippage:

    def test_zero_slippage_single_level(self):
        """Buy exactly at best ask — zero slippage."""
        engine = PaperExecutionEngine()
        fill = engine.execute_limit_fak_buy(
            token_id="tok1",
            limit_price=0.35,
            size=50.0,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        assert fill.slippage_vs_best == pytest.approx(0.0, abs=1e-6)

    def test_positive_slippage_multilevel(self):
        """Buy across levels — positive slippage vs best ask."""
        engine = PaperExecutionEngine()
        fill = engine.execute_limit_fak_buy(
            token_id="tok1",
            limit_price=0.37,
            size=130.0,  # 50 @ 0.35 + 80 @ 0.36
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        # avg = (50*0.35 + 80*0.36) / 130 = (17.5 + 28.8) / 130 = 0.35615...
        # slippage = 0.35615 - 0.35 = ~0.00615
        assert fill.slippage_vs_best > 0
        assert fill.slippage_vs_best == pytest.approx(
            fill.avg_price - 0.35, abs=1e-4
        )

    def test_sell_slippage(self):
        """Sell across levels — positive slippage (got worse than best bid)."""
        engine = PaperExecutionEngine()
        fill = engine.execute_limit_fak_sell(
            token_id="tok1",
            min_price=0.31,
            size=100.0,  # 60 @ 0.33 + 40 @ 0.32
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        # avg = (60*0.33 + 40*0.32) / 100 = (19.8 + 12.8) / 100 = 0.326
        # slippage = best_bid - avg = 0.33 - 0.326 = 0.004
        assert fill.slippage_vs_best > 0
        assert fill.slippage_vs_best == pytest.approx(0.33 - fill.avg_price, abs=1e-4)


# ---------------------------------------------------------------------------
# Book snapshot tests
# ---------------------------------------------------------------------------

class TestBookSnapshot:

    def test_compact_snapshot(self):
        snap = compact_book_snapshot(THREE_LEVEL_ASK_BOOK)
        assert snap["best_ask"] == 0.35
        assert snap["best_bid"] == 0.33
        assert len(snap["asks"]) == 3
        assert len(snap["bids"]) == 3
        assert "bid_depth" in snap
        assert "ask_depth" in snap

    def test_order_to_dict(self):
        """PaperOrder.to_dict() serializes cleanly."""
        engine = PaperExecutionEngine()
        order = engine.place_order(
            kind="entry",
            station_id="KLGA",
            target_date="2026-03-02",
            label="33-34°F",
            side="YES",
            token_id="tok1",
            order_type="limit_fak",
            direction="buy",
            size=50.0,
            limit_price=0.35,
            book_snapshot=THREE_LEVEL_ASK_BOOK,
        )
        d = order.to_dict()
        assert isinstance(d, dict)
        assert d["status"] == "FILLED"
        assert d["order_id"] == order.order_id

    def test_fill_to_dict(self):
        """PaperFill.to_dict() serializes cleanly."""
        fill = PaperFill(
            filled_size=50.0,
            avg_price=0.35,
            notional_usd=17.5,
            slippage_vs_best=0.0,
            levels_consumed=1,
            per_level=[{"price": 0.35, "size": 50.0, "notional": 17.5}],
            status="FILLED",
        )
        d = fill.to_dict()
        assert isinstance(d, dict)
        assert d["status"] == "FILLED"
