from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.autotrader import (
    AutoTradeCandidate,
    AutoTraderConfig,
    AutoTrader,
    apply_orderbook_guardrails,
    compute_trade_budget_usd,
    evaluate_trade_candidate,
)
from market.polymarket_execution import TopOfBook
from market.polymarket_execution import (
    PolymarketExecutionClient,
    PolymarketExecutionConfig,
    quantize_buy_order_size,
    quantize_market_buy_amount,
)


def _sample_payload():
    return {
        "station_id": "KATL",
        "target_date": "2026-02-28",
        "target_day": 0,
        "market_status": {"event_closed": False, "is_mature": False},
        "brackets": [
            {"name": "68-69 F", "yes_token_id": "YES1", "no_token_id": "NO1"},
            {"name": "70-71 F", "yes_token_id": "YES2", "no_token_id": "NO2"},
        ],
        "trading": {
            "available": True,
            "market_unit": "F",
            "forecast_winner": {"label": "68-69 F", "edge_points": 3.0},
            "best_terminal_trade": {
                "label": "68-69 F",
                "best_side": "YES",
                "selected_entry": 0.24,
                "selected_fair": 0.32,
                "edge_points": 8.0,
                "recommendation": "BUY_YES",
                "policy_reason": "Main scenario and underpriced.",
                "terminal_policy": {"allowed": True},
            },
            "best_tactical_trade": {
                "label": "68-69 F",
                "tactical_best_side": "YES",
            },
            "tactical_context": {
                "repricing_influence": 0.35,
                "next_metar": {
                    "available": True,
                    "direction": "UP",
                    "delta_market": 0.6,
                    "minutes_to_next": 35,
                    "next_obs_utc": "2026-02-28T16:00:00+00:00",
                },
            },
        },
    }


def test_evaluate_trade_candidate_accepts_forecast_winner_yes():
    config = AutoTraderConfig(enabled=True, station_ids=["KATL"])
    candidate, reasons = evaluate_trade_candidate(_sample_payload(), config, state={"positions": {}})

    assert reasons == []
    assert candidate is not None
    assert candidate.label == "68-69 F"
    assert candidate.side == "YES"
    assert candidate.token_id == "YES1"
    assert candidate.tactical_alignment == "aligned"
    assert candidate.strategy == "terminal_value"


def test_evaluate_trade_candidate_blocks_secondary_yes_without_extra_edge():
    payload = _sample_payload()
    payload["trading"]["best_terminal_trade"] = {
        "label": "70-71 F",
        "best_side": "YES",
        "selected_entry": 0.19,
        "selected_fair": 0.26,
        "edge_points": 7.0,
        "recommendation": "BUY_YES",
        "policy_reason": "Cheap but secondary.",
        "terminal_policy": {"allowed": True},
    }
    payload["trading"]["forecast_winner"] = {"label": "68-69 F", "edge_points": 2.0}

    config = AutoTraderConfig(enabled=True, station_ids=["KATL"], secondary_yes_edge_bonus_pts=8.0)
    candidate, reasons = evaluate_trade_candidate(payload, config, state={"positions": {}})

    assert candidate is None
    assert "secondary_yes_not_good_enough" in reasons


def test_compute_trade_budget_respects_remaining_limits():
    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL"],
        bankroll_usd=20.0,
        fractional_kelly=0.2,
        min_trade_usd=1.0,
        max_trade_usd=2.5,
        max_total_exposure_usd=6.0,
        max_station_exposure_usd=3.0,
        daily_loss_limit_usd=4.0,
    )
    candidate = AutoTradeCandidate(
        station_id="KATL",
        target_date="2026-02-28",
        target_day=0,
        label="68-69 F",
        side="YES",
        token_id="YES1",
        market_price=0.24,
        fair_price=0.32,
        edge_points=8.0,
        model_probability=0.32,
        recommendation="BUY_YES",
        policy_reason="test",
        forecast_winner_label="68-69 F",
        forecast_edge_points=8.0,
        tactical_alignment="aligned",
        why="test",
        position_key="KATL|2026-02-28|68-69 F|YES",
    )
    state = {
        "spent_today_usd": 1.0,
        "open_risk_usd": 2.0,
        "trades_today": 1,
        "last_trade_utc": None,
        "positions": {},
    }

    budget = compute_trade_budget_usd(candidate, config, state, available_balance_usd=2.0)

    assert budget >= 1.0
    assert budget <= 2.0


def test_apply_orderbook_guardrails_uses_live_book_and_depth_caps():
    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL"],
        min_trade_usd=1.0,
        max_entry_price=0.40,
        min_edge_points=6.0,
        max_spread_points=8.0,
        min_ask_depth_contracts=10.0,
        max_depth_participation=0.25,
    )
    candidate = AutoTradeCandidate(
        station_id="KATL",
        target_date="2026-02-28",
        target_day=0,
        label="68-69 F",
        side="YES",
        token_id="YES1",
        market_price=0.24,
        fair_price=0.32,
        edge_points=8.0,
        model_probability=0.32,
        recommendation="BUY_YES",
        policy_reason="test",
        forecast_winner_label="68-69 F",
        forecast_edge_points=8.0,
        tactical_alignment="aligned",
        why="test",
        position_key="KATL|2026-02-28|68-69 F|YES",
    )
    book = TopOfBook(
        token_id="YES1",
        best_bid=0.23,
        best_ask=0.24,
        bid_size=40.0,
        ask_size=20.0,
        spread=0.01,
        min_order_size=1.0,
        last_trade_price=0.24,
        raw={},
    )

    plan, reasons = apply_orderbook_guardrails(candidate, book, config, budget_usd=2.0)

    assert reasons == []
    assert plan is not None
    assert plan["limit_price"] == 0.25
    assert plan["size"] <= 5.0


def test_quantize_buy_order_size_respects_cents_precision_for_buy_collateral():
    assert quantize_buy_order_size(8.33, 0.25) == 8.32
    assert quantize_buy_order_size(10.52, 0.19) == 10.0
    assert quantize_market_buy_amount(2.0899) == 2.08


def test_apply_orderbook_guardrails_caps_buy_budget_at_limit_price():
    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL"],
        min_trade_usd=1.0,
        max_entry_price=0.40,
        min_edge_points=6.0,
        max_spread_points=8.0,
        min_ask_depth_contracts=10.0,
        max_depth_participation=1.0,
        limit_slippage_points=1.0,
    )
    candidate = AutoTradeCandidate(
        station_id="KATL",
        target_date="2026-02-28",
        target_day=0,
        label="70-71 F",
        side="YES",
        token_id="YES2",
        market_price=0.19,
        fair_price=0.29,
        edge_points=10.0,
        model_probability=0.29,
        recommendation="BUY_YES",
        policy_reason="test",
        forecast_winner_label="68-69 F",
        forecast_edge_points=4.0,
        tactical_alignment="aligned",
        why="test",
        position_key="KATL|2026-02-28|70-71 F|YES",
    )
    book = TopOfBook(
        token_id="YES2",
        best_bid=0.18,
        best_ask=0.19,
        bid_size=50.0,
        ask_size=50.0,
        spread=0.01,
        min_order_size=1.0,
        last_trade_price=0.19,
        raw={},
    )

    plan, reasons = apply_orderbook_guardrails(candidate, book, config, budget_usd=2.0)

    assert reasons == []
    assert plan is not None
    assert plan["limit_price"] == 0.2
    assert plan["size"] == 10.0
    assert plan["notional_usd"] == 2.0


def test_status_snapshot_shows_global_portfolio_and_station_filtered_positions(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "katl.jsonl"
    state_path.write_text(
        """{
  "trading_day": "2026-02-28",
  "spent_today_usd": 3.5,
  "open_risk_usd": 3.5,
  "trades_today": 2,
  "positions": {
    "KATL|2026-02-28|68-69 F|YES": {
      "station_id": "KATL",
      "target_date": "2026-02-28",
      "label": "68-69 F",
      "side": "YES",
      "notional_usd": 1.5,
      "opened_at_utc": "2026-02-28T15:00:00+00:00",
      "status": "OPEN"
    },
    "KLGA|2026-02-28|48-49 F|NO": {
      "station_id": "KLGA",
      "target_date": "2026-02-28",
      "label": "48-49 F",
      "side": "NO",
      "notional_usd": 2.0,
      "opened_at_utc": "2026-02-28T15:05:00+00:00",
      "status": "OPEN"
    }
  }
}""",
        encoding="utf-8",
    )
    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL"],
        bankroll_usd=20.0,
        max_total_exposure_usd=4.0,
        daily_loss_limit_usd=4.0,
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config)

    snapshot = trader.get_status_snapshot("KATL")

    assert snapshot["portfolio"]["open_positions_count"] == 2
    assert snapshot["portfolio"]["open_risk_usd"] == 3.5
    assert snapshot["state"]["open_positions"] == 1
    assert snapshot["state"]["open_risk_usd"] == 1.5


def test_status_snapshot_without_station_returns_global_state_summary(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "katl.jsonl"
    state_path.write_text(
        """{
  "trading_day": "2026-02-28",
  "open_risk_usd": 2.75,
  "positions": {
    "KATL|2026-02-28|68-69 F|YES": {
      "station_id": "KATL",
      "label": "68-69 F",
      "side": "YES",
      "strategy": "terminal_value",
      "cost_basis_open_usd": 1.25,
      "opened_at_utc": "2026-02-28T15:00:00+00:00",
      "status": "OPEN"
    },
    "KLGA|2026-02-28|48-49 F|NO": {
      "station_id": "KLGA",
      "label": "48-49 F",
      "side": "NO",
      "strategy": "tactical_reprice",
      "cost_basis_open_usd": 1.50,
      "opened_at_utc": "2026-02-28T15:05:00+00:00",
      "status": "OPEN"
    }
  }
}""",
        encoding="utf-8",
    )
    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL", "KLGA"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config)

    snapshot = trader.get_status_snapshot()

    assert snapshot["state"]["open_positions"] == 2
    assert snapshot["state"]["open_risk_usd"] == 2.75
    assert snapshot["state"]["strategies"] == {"terminal_value": 1, "tactical_reprice": 1}


def test_evaluate_trade_candidate_can_pick_tactical_reprice():
    payload = _sample_payload()
    payload["trading"]["best_terminal_trade"] = {}
    payload["trading"]["best_tactical_trade"] = {
        "label": "70-71 F",
        "tactical_best_side": "YES",
        "selected_tactical_entry": 0.12,
        "selected_tactical_fair": 0.20,
        "tactical_edge_points": 8.0,
        "tactical_recommendation": "BUY_YES",
        "tactical_policy": {"allowed": True, "summary": "Next official can reprice this bucket."},
        "tactical_policy_score": 5.6,
    }

    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL"],
        allow_terminal_value=False,
        secondary_yes_edge_bonus_pts=0.0,
    )
    candidate, reasons = evaluate_trade_candidate(payload, config, state={"positions": {}})

    assert reasons == []
    assert candidate is not None
    assert candidate.strategy == "tactical_reprice"
    assert candidate.label == "70-71 F"
    assert candidate.side == "YES"
    assert candidate.token_id == "YES2"


def test_persist_exit_closes_position_and_updates_realized_pnl(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "katl.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config)
    trader.state["positions"] = {
        "KATL|2026-02-28|68-69 F|YES": {
            "station_id": "KATL",
            "target_date": "2026-02-28",
            "label": "68-69 F",
            "side": "YES",
            "token_id": "YES1",
            "status": "OPEN",
            "strategy": "tactical_reprice",
            "entry_price": 0.20,
            "shares": 5.0,
            "shares_open": 5.0,
            "notional_usd": 1.0,
            "cost_basis_open_usd": 1.0,
            "opened_at_utc": "2026-02-28T15:00:00+00:00",
            "realized_pnl_usd": 0.0,
            "fills": [],
        }
    }
    trader.state["open_risk_usd"] = 1.0
    trader._persist_state()

    persisted = trader._persist_exit(
        "KATL|2026-02-28|68-69 F|YES",
        {
            "size": 5.0,
            "best_bid": 0.28,
            "proceeds_usd": 1.4,
            "reason": "take_profit",
        },
        {"mode": "paper"},
        mode="paper",
    )

    assert persisted["reason"] == "take_profit"
    assert round(persisted["realized_pnl_usd"], 2) == 0.40
    snapshot = trader.get_status_snapshot("KATL")
    assert snapshot["portfolio"]["open_risk_usd"] == 0.0
    assert snapshot["portfolio"]["realized_pnl_today_usd"] == 0.4
    assert snapshot["state"]["open_positions"] == 0


def test_summarize_order_response_uses_matched_amounts():
    client = PolymarketExecutionClient(PolymarketExecutionConfig())

    summary = client.summarize_order_response(
        {
            "success": True,
            "status": "matched",
            "makingAmount": "5.0",
            "takingAmount": "1.20",
        },
        fallback_size=5.0,
        fallback_price=0.24,
    )

    assert summary["success"] is True
    assert summary["matched_size"] == 5.0
    assert summary["matched_notional"] == 1.2
