"""Integration tests for core/papertrader.py — shadow trading runtime."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.papertrader import (
    PaperTrader,
    PaperTraderConfig,
    _default_paper_state,
    load_paper_state,
    save_paper_state,
)
from market.paper_execution import PaperFill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> PaperTraderConfig:
    cfg = PaperTraderConfig(
        station_ids=["KLGA"],
        initial_bankroll_usd=100.0,
        bankroll_usd=100.0,
        target_date="2026-03-05",
        min_trade_usd=1.0,
        max_total_exposure_usd=50.0,
        daily_loss_limit_usd=20.0,
        max_open_positions=10,
        max_trades_per_day=10,
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_filled_fill(size=25.0, price=0.36, slippage=0.01) -> PaperFill:
    return PaperFill(
        filled_size=size,
        avg_price=price,
        notional_usd=round(size * price, 4),
        slippage_vs_best=slippage,
        levels_consumed=2,
        per_level=[{"price": price, "size": size, "notional": size * price}],
        status="FILLED",
    )


def _make_candidate(label="33-34F", side="YES"):
    from core.autotrader import AutoTradeCandidate
    pkey = f"KLGA|2026-03-05|{label}|{side}"
    return AutoTradeCandidate(
        station_id="KLGA",
        target_date="2026-03-05",
        target_day=0,
        label=label,
        side=side,
        token_id="tok_abc123",
        market_price=0.35,
        fair_price=0.42,
        edge_points=7.0,
        model_probability=0.42,
        recommendation="BUY_TERMINAL",
        policy_reason="edge_ok",
        forecast_winner_label=label,
        forecast_edge_points=7.0,
        tactical_alignment="neutral",
        why="terminal value test",
        position_key=pkey,
        strategy="terminal_value",
        strategy_score=0.88,
    )


def _make_trader(tmp_path, **cfg_kwargs) -> PaperTrader:
    state_path = str(tmp_path / "papertrader_state.json")
    log_path = str(tmp_path / "papertrader_trades.jsonl")
    cfg = _make_config(state_path=state_path, log_path=log_path, **cfg_kwargs)
    return PaperTrader(config=cfg)


def _legacy_state(subject):
    state = subject.state if hasattr(subject, "state") else subject
    return state["strategies"]["terminal_value_legacy"]


def _legacy_position_key(trader, base_key: str) -> str:
    return next(key for key in trader.state["positions"] if key.endswith(f"::{base_key}"))


def _legacy_position(trader, base_key: str):
    return trader.state["positions"][_legacy_position_key(trader, base_key)]


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:

    def test_default_state_has_correct_equity(self, tmp_path):
        cfg = _make_config()
        state = _default_paper_state(cfg)
        assert state["paper_equity_usd"] == 400.0
        assert state["aggregate_initial_bankroll_usd"] == 400.0
        assert _legacy_state(state)["paper_equity_usd"] == 100.0
        assert state["positions"] == {}
        assert state["trades_today"] == 0

    def test_save_and_reload_state(self, tmp_path):
        cfg = _make_config()
        path = str(tmp_path / "state.json")
        state = _default_paper_state(cfg)
        state["strategies"]["terminal_value_legacy"]["paper_equity_usd"] = 87.5
        state["strategies"]["terminal_value_legacy"]["trades_today"] = 3
        save_paper_state(path, state)
        loaded = load_paper_state(path, cfg)
        assert loaded["paper_equity_usd"] == 387.5
        assert _legacy_state(loaded)["paper_equity_usd"] == 87.5
        assert loaded["trades_today"] == 3

    def test_missing_file_returns_default(self, tmp_path):
        cfg = _make_config()
        path = str(tmp_path / "nonexistent.json")
        state = load_paper_state(path, cfg)
        assert state["paper_equity_usd"] == 400.0

    def test_corrupted_file_returns_default(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{{invalid json{{", encoding="utf-8")
        cfg = _make_config()
        state = load_paper_state(str(path), cfg)
        assert state["paper_equity_usd"] == 400.0


# ---------------------------------------------------------------------------
# _persist_trade (entry accounting)
# ---------------------------------------------------------------------------

class TestPersistTrade:

    def test_equity_decreases_on_entry(self, tmp_path):
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()
        plan = {"limit_price": 0.36, "size": 25.0, "notional_usd": 9.0}
        fill = _make_filled_fill(size=25.0, price=0.36)

        trader._persist_trade(candidate, plan, fill)

        assert trader.state["paper_equity_usd"] == pytest.approx(391.0, abs=0.01)
        assert _legacy_state(trader)["paper_equity_usd"] == pytest.approx(91.0, abs=0.01)

    def test_position_created_correctly(self, tmp_path):
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()
        plan = {"limit_price": 0.36, "size": 25.0, "notional_usd": 9.0}
        fill = _make_filled_fill(size=25.0, price=0.36)

        trader._persist_trade(candidate, plan, fill)

        pos = _legacy_position(trader, "KLGA|2026-03-05|33-34F|YES")
        assert pos["status"] == "OPEN"
        assert pos["mode"] == "paper_shadow"
        assert pos["shares"] == 25.0
        assert pos["shares_open"] == 25.0
        assert pos["entry_price"] == pytest.approx(0.36, abs=1e-6)
        assert pos["station_id"] == "KLGA"

    def test_trades_today_increments(self, tmp_path):
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()
        plan = {"limit_price": 0.36, "size": 25.0, "notional_usd": 9.0}
        fill = _make_filled_fill(size=25.0, price=0.36)

        trader._persist_trade(candidate, plan, fill)
        assert trader.state["trades_today"] == 1
        assert trader.state["total_entries"] == 1

    def test_open_risk_updated(self, tmp_path):
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()
        fill = _make_filled_fill(size=25.0, price=0.36)
        plan = {"limit_price": 0.36, "size": 25.0, "notional_usd": 9.0}

        trader._persist_trade(candidate, plan, fill)
        # open_risk should reflect the position's cost basis
        assert trader.state["open_risk_usd"] > 0

    def test_slippage_tracked(self, tmp_path):
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()
        fill = _make_filled_fill(size=25.0, price=0.36, slippage=0.01)
        plan = {"limit_price": 0.37, "size": 25.0, "notional_usd": 9.0}

        trader._persist_trade(candidate, plan, fill)
        # 0.01 slippage * 25 shares = 0.25 slippage usd
        assert trader.state["total_slippage_usd"] == pytest.approx(0.25, abs=1e-4)


# ---------------------------------------------------------------------------
# _persist_exit (exit accounting)
# ---------------------------------------------------------------------------

class TestPersistExit:

    def _setup_open_position(self, trader) -> str:
        candidate = _make_candidate()
        fill = _make_filled_fill(size=50.0, price=0.35)
        plan = {"limit_price": 0.35, "size": 50.0, "notional_usd": 17.5}
        trader._persist_trade(candidate, plan, fill)
        return candidate.position_key

    def test_realized_pnl_computed(self, tmp_path):
        trader = _make_trader(tmp_path)
        pkey = self._setup_open_position(trader)

        # Exit at 0.42 = profit of (0.42-0.35)*50 = $3.50
        exit_fill = _make_filled_fill(size=50.0, price=0.42)
        exit_plan = {"reason": "take_profit", "size": 50.0, "min_price": 0.40, "proceeds_usd": 21.0}
        persisted = trader._persist_exit(pkey, exit_plan, exit_fill)

        assert persisted["realized_pnl_usd"] == pytest.approx(3.5, abs=0.01)

    def test_position_closes_when_fully_exited(self, tmp_path):
        trader = _make_trader(tmp_path)
        pkey = self._setup_open_position(trader)

        exit_fill = _make_filled_fill(size=50.0, price=0.42)
        exit_plan = {"reason": "take_profit", "size": 50.0, "min_price": 0.40, "proceeds_usd": 21.0}
        trader._persist_exit(pkey, exit_plan, exit_fill)

        assert _legacy_position(trader, pkey)["status"] == "CLOSED"
        assert _legacy_position(trader, pkey)["shares_open"] == pytest.approx(0.0, abs=1e-6)

    def test_equity_increases_after_exit(self, tmp_path):
        trader = _make_trader(tmp_path)
        pkey = self._setup_open_position(trader)

        equity_after_entry = trader.state["paper_equity_usd"]
        exit_fill = _make_filled_fill(size=50.0, price=0.42)
        exit_plan = {"reason": "take_profit", "size": 50.0, "min_price": 0.40, "proceeds_usd": 21.0}
        trader._persist_exit(pkey, exit_plan, exit_fill)

        assert trader.state["paper_equity_usd"] > equity_after_entry

    def test_exit_reason_tracked(self, tmp_path):
        trader = _make_trader(tmp_path)
        pkey = self._setup_open_position(trader)

        exit_fill = _make_filled_fill(size=50.0, price=0.30)
        exit_plan = {"reason": "stop_loss", "size": 50.0, "min_price": 0.28, "proceeds_usd": 15.0}
        trader._persist_exit(pkey, exit_plan, exit_fill)

        assert trader.state["exits_by_reason"].get("stop_loss") == 1

    def test_partial_exit_leaves_shares_open(self, tmp_path):
        trader = _make_trader(tmp_path)
        pkey = self._setup_open_position(trader)

        # Exit only 25 of 50 shares
        exit_fill = _make_filled_fill(size=25.0, price=0.42)
        exit_plan = {"reason": "take_profit", "size": 25.0, "min_price": 0.40, "proceeds_usd": 10.5}
        trader._persist_exit(pkey, exit_plan, exit_fill)

        pos = _legacy_position(trader, pkey)
        assert pos["status"] == "OPEN"
        assert pos["shares_open"] == pytest.approx(25.0, abs=1e-4)

    def test_full_entry_exit_cycle_pnl(self, tmp_path):
        """End-to-end: entry at 0.35, exit at 0.42 → net PnL = $3.50."""
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()

        entry_fill = _make_filled_fill(size=50.0, price=0.35)
        plan = {"limit_price": 0.35, "size": 50.0, "notional_usd": 17.5}
        trader._persist_trade(candidate, plan, entry_fill)

        equity_after_entry = trader.state["paper_equity_usd"]

        exit_fill = _make_filled_fill(size=50.0, price=0.42)
        exit_plan = {"reason": "take_profit", "size": 50.0, "min_price": 0.40, "proceeds_usd": 21.0}
        trader._persist_exit(candidate.position_key, exit_plan, exit_fill)

        # equity should be initial - cost + proceeds = 100 - 17.5 + 21 = 103.5
        assert trader.state["paper_equity_usd"] == pytest.approx(403.5, abs=0.01)
        assert _legacy_state(trader)["paper_equity_usd"] == pytest.approx(103.5, abs=0.01)
        assert trader.state["total_realized_pnl_usd"] == pytest.approx(3.5, abs=0.01)
        assert _legacy_position(trader, candidate.position_key)["status"] == "CLOSED"


# ---------------------------------------------------------------------------
# Daily counter reset
# ---------------------------------------------------------------------------

class TestDailyReset:

    def test_counters_reset_on_new_day(self, tmp_path):
        trader = _make_trader(tmp_path)
        trader.state["trading_day"] = "2026-03-01"  # old day
        trader.state["trades_today"] = 5
        trader.state["realized_pnl_today_usd"] = -2.0
        trader.state["gross_buys_today_usd"] = 10.0

        trader._reset_daily_counters_if_needed()

        # Should have reset because today != 2026-03-01
        assert trader.state["trades_today"] == 0
        assert trader.state["realized_pnl_today_usd"] == 0.0
        assert trader.state["gross_buys_today_usd"] == 0.0

    def test_counters_not_reset_same_day(self, tmp_path):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        trader = _make_trader(tmp_path)
        today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        trader.state["trading_day"] = today
        trader.state["trades_today"] = 3

        trader._reset_daily_counters_if_needed()

        assert trader.state["trades_today"] == 3


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------

class TestRiskLimits:

    def test_exposure_cap_blocks_new_entries(self, tmp_path):
        """When open_risk >= max_total_exposure, budget should be 0."""
        trader = _make_trader(tmp_path, max_total_exposure_usd=10.0)
        # Add a real open position that exceeds the cap
        # compute_trade_budget_usd sums _position_risk_now_usd from actual positions
        trader.state["positions"]["KLGA|2026-03-05|existing|YES"] = {
            "status": "OPEN",
            "station_id": "KLGA",
            "target_date": "2026-03-05",
            "label": "existing",
            "side": "YES",
            "current_value_usd": 11.0,  # above 10.0 cap
            "cost_basis_open_usd": 10.0,
            "shares_open": 28.0,
            "managed_by_bot": True,
            "source": "papertrader",
        }

        from core.autotrader import compute_trade_budget_usd
        candidate = _make_candidate(label="other-label")
        atc = trader._atc()
        budget = compute_trade_budget_usd(candidate, atc, trader.state)
        assert budget == 0.0

    def test_daily_loss_limit_blocks_entries(self, tmp_path):
        """When realized losses exceed daily limit, budget = 0."""
        trader = _make_trader(tmp_path, daily_loss_limit_usd=5.0)
        trader.state["realized_pnl_today_usd"] = -5.5  # exceeded limit

        from core.autotrader import compute_trade_budget_usd
        candidate = _make_candidate()
        atc = trader._atc()
        budget = compute_trade_budget_usd(candidate, atc, trader.state)
        assert budget == 0.0

    def test_max_trades_per_day_blocks_entries(self, tmp_path):
        """When trades_today >= max_trades_per_day, budget = 0."""
        trader = _make_trader(tmp_path, max_trades_per_day=3)
        trader.state["trades_today"] = 3

        from core.autotrader import compute_trade_budget_usd
        candidate = _make_candidate()
        atc = trader._atc()
        budget = compute_trade_budget_usd(candidate, atc, trader.state)
        assert budget == 0.0


# ---------------------------------------------------------------------------
# Journal events
# ---------------------------------------------------------------------------

class TestJournalEvents:

    def test_journal_written_on_entry(self, tmp_path):
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()
        fill = _make_filled_fill(size=25.0, price=0.36)
        plan = {"limit_price": 0.36, "size": 25.0, "notional_usd": 9.0}

        trader._persist_trade(candidate, plan, fill)
        trader._append_journal({
            "event": "entry_fill",
            "station_id": "KLGA",
            "position_key": candidate.position_key,
            "fill": fill.to_dict(),
        })

        log_path = Path(trader.config.log_path)
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "entry_fill"
        assert record["station_id"] == "KLGA"
        assert "ts_utc" in record

    def test_multiple_journal_events_appended(self, tmp_path):
        trader = _make_trader(tmp_path)

        trader._append_journal({"event": "tick", "status": "skip"})
        trader._append_journal({"event": "tick", "status": "skip"})
        trader._append_journal({"event": "entry_fill", "status": "paper_fill"})

        log_path = Path(trader.config.log_path)
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        events = [json.loads(l)["event"] for l in lines]
        assert events == ["tick", "tick", "entry_fill"]


# ---------------------------------------------------------------------------
# Mark to market
# ---------------------------------------------------------------------------

class TestMarkToMarket:

    def test_mark_to_market_updates_value(self, tmp_path):
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()
        fill = _make_filled_fill(size=50.0, price=0.35)
        plan = {"limit_price": 0.35, "size": 50.0, "notional_usd": 17.5}
        trader._persist_trade(candidate, plan, fill)

        # Build a mock payload where the bracket shows best_bid=0.42
        mock_payload = {
            "brackets": [{
                "name": "33-34F",
                "bracket": "33-34F",
                "yes_token_id": "tok_abc123",   # required by _book_from_bracket
                "no_token_id": "tok_no123",
                "ws_yes_best_bid": 0.42,
                "ws_yes_best_ask": 0.43,
                "ws_yes_bid_depth": 100.0,
                "ws_yes_ask_depth": 80.0,
                "ws_yes_spread": 0.01,
                "ws_yes_bids": [{"price": 0.42, "size": 100}],
                "ws_yes_asks": [{"price": 0.43, "size": 80}],
            }]
        }
        payload_cache = {"KLGA": mock_payload}
        trader._mark_to_market_all(payload_cache=payload_cache)

        pos = _legacy_position(trader, candidate.position_key)
        assert pos.get("current_price") == pytest.approx(0.42, abs=1e-6)
        # current_value = 50 * 0.42 = 21.0
        assert pos.get("current_value_usd") == pytest.approx(21.0, abs=0.01)
        # unrealized_pnl = 21.0 - 17.5 = +3.5
        assert pos.get("unrealized_pnl_usd") == pytest.approx(3.5, abs=0.01)

    def test_mark_to_market_no_payload_leaves_value_unchanged(self, tmp_path):
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()
        fill = _make_filled_fill(size=50.0, price=0.35)
        plan = {"limit_price": 0.35, "size": 50.0, "notional_usd": 17.5}
        trader._persist_trade(candidate, plan, fill)

        trader._mark_to_market_all(payload_cache={})

        pos = _legacy_position(trader, candidate.position_key)
        # Without book data, current_value unchanged from cost basis
        assert pos.get("current_price") is None


# ---------------------------------------------------------------------------
# Status snapshot
# ---------------------------------------------------------------------------

class TestStatusSnapshot:

    def test_snapshot_initial(self, tmp_path):
        trader = _make_trader(tmp_path)
        status = trader.get_status_snapshot()
        assert status["mode"] == "paper_multi_strategy"
        assert status["paper_equity_usd"] == 400.0
        assert status["aggregate_initial_bankroll_usd"] == 400.0
        assert len(status["strategies"]) == 4
        assert status["open_positions_count"] == 0
        assert status["fill_ratio"] == 0.0
        assert status["hit_rate"] == 0.0

    def test_snapshot_with_open_position(self, tmp_path):
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()
        fill = _make_filled_fill(size=25.0, price=0.36)
        plan = {"limit_price": 0.36, "size": 25.0, "notional_usd": 9.0}
        trader._persist_trade(candidate, plan, fill)
        trader._save_state()

        status = trader.get_status_snapshot()
        assert status["open_positions_count"] == 1
        assert status["total_fills"] == 1
        assert status["total_entries"] == 1
        assert status["open_risk_usd"] > 0

    def test_snapshot_fill_ratio(self, tmp_path):
        trader = _make_trader(tmp_path)
        lane = _legacy_state(trader)
        lane["total_fills"] = 7
        lane["total_rejections"] = 3
        trader._refresh_state_totals()
        trader._save_state()

        status = trader.get_status_snapshot()
        assert status["fill_ratio"] == pytest.approx(0.7, abs=1e-4)

    def test_snapshot_hit_rate_with_winners(self, tmp_path):
        """Hit rate = winners / closed positions."""
        trader = _make_trader(tmp_path)
        # Add 3 closed positions: 2 winners, 1 loser
        for i, pnl in enumerate([2.0, -0.5, 1.0]):
            pkey = f"KLGA|2026-03-05|label{i}|YES"
            trader.state["positions"][pkey] = {
                "status": "CLOSED",
                "realized_pnl_usd": pnl,
            }
        trader._save_state()

        status = trader.get_status_snapshot()
        assert status["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)

    def test_config_to_autotrader_adapter(self, tmp_path):
        """PaperTraderConfig.to_autotrader_config() produces a valid AutoTraderConfig."""
        cfg = _make_config()
        atc = cfg.to_autotrader_config()
        assert atc.bankroll_usd == cfg.bankroll_usd
        assert atc.fractional_kelly == cfg.fractional_kelly
        assert atc.min_edge_points == cfg.min_edge_points
        assert atc.mode == "paper"

    # ── Bug-fix regression tests ────────────────────────────────────────────

    def test_avg_slippage_bps_uses_notional_not_fill_count(self, tmp_path):
        """[P2a] avg_slippage_bps should be (slippage_usd / notional_usd) × 10000,
        NOT (slippage_usd / fills) × 10000."""
        trader = _make_trader(tmp_path)
        # Slippage = $0.50, notional = $100, bps = 50
        lane = _legacy_state(trader)
        lane["total_slippage_usd"] = 0.50
        lane["total_fill_notional_usd"] = 100.0
        lane["total_fills"] = 2  # irrelevant to the bps formula
        trader._refresh_state_totals()
        trader._save_state()

        status = trader.get_status_snapshot()
        # 0.50 / 100.0 * 10000 = 50 bps  (not 0.50/2*10000 = 2500 bps)
        assert status["avg_slippage_bps"] == pytest.approx(50.0, abs=0.1)

    def test_entry_persist_trade_populates_total_fill_notional(self, tmp_path):
        """[P2c] _persist_trade must increment total_fill_notional_usd via
        _record_fill_metrics so avg_slippage_bps has a valid denominator."""
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()
        fill = _make_filled_fill(size=25.0, price=0.36)  # notional = 9.0
        plan = {"limit_price": 0.36, "size": 25.0, "notional_usd": 9.0}
        trader._persist_trade(candidate, plan, fill)

        assert trader.state["total_fill_notional_usd"] == pytest.approx(9.0, abs=0.01)
        assert trader.state["total_fills"] == 1

    def test_exit_fill_metrics_recorded(self, tmp_path):
        """[P2b] After _manage_open_positions processes an exit fill, total_fills
        and total_fill_notional_usd must be updated."""
        trader = _make_trader(tmp_path)
        candidate = _make_candidate()
        fill_in = _make_filled_fill(size=25.0, price=0.35)
        plan = {"limit_price": 0.35, "size": 25.0, "notional_usd": 8.75}
        trader._persist_trade(candidate, plan, fill_in)

        # Manually simulate what _manage_open_positions does on exit
        exit_fill = _make_filled_fill(size=25.0, price=0.42)  # notional = 10.5
        exit_plan = {"reason": "take_profit", "min_price": 0.40, "size": 25.0}
        trader._persist_exit(candidate.position_key, exit_plan, exit_fill)
        trader._record_fill_metrics(exit_fill, strategy_id="terminal_value_legacy")

        # total_fills: 1 entry + 1 exit = 2
        assert trader.state["total_fills"] == 2
        # total_fill_notional: 8.75 entry + 10.5 exit
        assert trader.state["total_fill_notional_usd"] == pytest.approx(8.75 + 10.5, abs=0.01)

    def test_exit_rejection_increments_rejections(self, tmp_path):
        """[P2b] When an exit fill is rejected (filled_size=0), total_rejections
        must increment via _record_rejection."""
        trader = _make_trader(tmp_path)
        assert trader.state.get("total_rejections", 0) == 0
        trader._record_rejection()
        assert trader.state["total_rejections"] == 1

    def test_snapshot_total_fill_notional_exposed(self, tmp_path):
        """[P2a] Status snapshot must expose total_fill_notional_usd for UI."""
        trader = _make_trader(tmp_path)
        _legacy_state(trader)["total_fill_notional_usd"] = 55.5
        trader._refresh_state_totals()
        trader._save_state()

        status = trader.get_status_snapshot()
        assert "total_fill_notional_usd" in status
        assert status["total_fill_notional_usd"] == pytest.approx(55.5, abs=0.01)
