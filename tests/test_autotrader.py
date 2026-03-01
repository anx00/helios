import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime, timezone
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
    load_autotrader_config_from_env,
    resolve_station_market_target,
)
from market.polymarket_execution import TopOfBook
from market.polymarket_execution import (
    PolymarketExecutionClient,
    PolymarketExecutionConfig,
    quantize_buy_order_size,
    quantize_market_buy_amount,
)


def _today_iso():
    return datetime.now(timezone.utc).date().isoformat()


def _sample_payload():
    return {
        "station_id": "KATL",
        "target_date": "2026-02-28",
        "target_day": 0,
        "confidence": 0.95,
        "tmax_sigma_f": 1.8,
        "nowcast": {
            "confidence": 0.95,
            "tmax_sigma_f": 1.8,
            "p_bucket": [
                {"label": "68-69 F", "probability": 0.44},
                {"label": "70-71 F", "probability": 0.24},
                {"label": "66-67 F", "probability": 0.18},
            ],
        },
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
    assert candidate.signal_confidence == 0.95
    assert candidate.signal_sigma_f == 1.8
    assert candidate.distribution_top_probability == 0.44
    assert candidate.distribution_top_gap == 0.20
    assert candidate.bucket_rank == 1


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


def test_evaluate_trade_candidate_ignores_station_position_limit_when_disabled():
    payload = _sample_payload()
    config = AutoTraderConfig(enabled=True, station_ids=["KATL"], max_station_positions=0)
    candidate, reasons = evaluate_trade_candidate(
        payload,
        config,
        state={"positions": {}},
        live_positions=[
            {
                "source": "polymarket_live",
                "station_id": "KATL",
                "target_date": "2026-02-28",
                "label": "64-65 F",
                "side": "YES",
                "cost_basis_open_usd": 1.0,
            },
            {
                "source": "polymarket_live",
                "station_id": "KATL",
                "target_date": "2026-02-28",
                "label": "66-67 F",
                "side": "YES",
                "cost_basis_open_usd": 1.0,
            },
        ],
    )

    assert reasons == []
    assert candidate is not None


def test_evaluate_trade_candidate_ignores_tracking_only_open_position_limit():
    payload = _sample_payload()
    config = AutoTraderConfig(enabled=True, station_ids=["KATL"], max_open_positions=1)
    candidate, reasons = evaluate_trade_candidate(
        payload,
        config,
        state={"positions": {}},
        live_positions=[
            {
                "source": "polymarket_live",
                "station_id": "EGLC",
                "target_date": "2026-02-28",
                "label": "14C OR HIGHER",
                "side": "YES",
                "cost_basis_open_usd": 5.0,
                "managed_by_bot": False,
            }
        ],
    )

    assert reasons == []
    assert candidate is not None


def test_evaluate_trade_candidate_blocks_low_signal_confidence():
    payload = _sample_payload()
    payload["confidence"] = 0.2
    payload["nowcast"]["confidence"] = 0.2
    config = AutoTraderConfig(enabled=True, station_ids=["KATL"], min_signal_confidence=0.35)

    candidate, reasons = evaluate_trade_candidate(payload, config, state={"positions": {}})

    assert candidate is None
    assert "signal_confidence_too_low" in reasons


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


def test_compute_trade_budget_auto_mode_ignores_per_trade_cap_when_zero():
    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL"],
        bankroll_usd=20.0,
        fractional_kelly=0.2,
        min_trade_usd=1.0,
        max_trade_usd=0.0,
        max_total_exposure_usd=6.0,
        max_station_exposure_usd=6.0,
        daily_loss_limit_usd=6.0,
    )
    candidate = AutoTradeCandidate(
        station_id="KATL",
        target_date="2026-02-28",
        target_day=0,
        label="68-69 F",
        side="YES",
        token_id="YES1",
        market_price=0.10,
        fair_price=0.90,
        edge_points=80.0,
        model_probability=0.90,
        recommendation="BUY_YES",
        policy_reason="test",
        forecast_winner_label="68-69 F",
        forecast_edge_points=80.0,
        tactical_alignment="neutral",
        why="test",
        position_key="KATL|2026-02-28|68-69 F|YES",
    )

    budget = compute_trade_budget_usd(candidate, config, state={"positions": {}}, available_balance_usd=6.0)

    assert budget == 1.066667


def test_compute_trade_budget_scales_down_with_diffuse_distribution():
    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL"],
        bankroll_usd=20.0,
        fractional_kelly=0.2,
        min_trade_usd=0.5,
        max_trade_usd=0.0,
        max_total_exposure_usd=10.0,
        max_station_exposure_usd=10.0,
        daily_loss_limit_usd=10.0,
        distribution_top_probability_soft_floor=0.40,
        distribution_top_probability_hard_floor=0.22,
        distribution_top_gap_soft_floor=0.08,
        distribution_top_gap_hard_floor=0.015,
        distribution_min_budget_multiplier=0.60,
    )
    concentrated = AutoTradeCandidate(
        station_id="KATL",
        target_date="2026-02-28",
        target_day=0,
        label="68-69 F",
        side="YES",
        token_id="YES1",
        market_price=0.10,
        fair_price=0.90,
        edge_points=80.0,
        model_probability=0.90,
        recommendation="BUY_YES",
        policy_reason="test",
        forecast_winner_label="68-69 F",
        forecast_edge_points=80.0,
        tactical_alignment="neutral",
        why="test",
        position_key="KATL|2026-02-28|68-69 F|YES",
        signal_confidence=0.95,
        signal_sigma_f=1.5,
        distribution_top_probability=0.60,
        distribution_top_gap=0.20,
        bucket_rank=1,
    )
    diffuse = AutoTradeCandidate(
        station_id="KATL",
        target_date="2026-02-28",
        target_day=0,
        label="68-69 F",
        side="YES",
        token_id="YES1",
        market_price=0.10,
        fair_price=0.90,
        edge_points=80.0,
        model_probability=0.90,
        recommendation="BUY_YES",
        policy_reason="test",
        forecast_winner_label="68-69 F",
        forecast_edge_points=80.0,
        tactical_alignment="neutral",
        why="test",
        position_key="KATL|2026-02-28|68-69 F|YES",
        signal_confidence=0.95,
        signal_sigma_f=1.5,
        distribution_top_probability=0.24,
        distribution_top_gap=0.02,
        bucket_rank=1,
    )

    concentrated_budget = compute_trade_budget_usd(concentrated, config, state={"positions": {}}, available_balance_usd=10.0)
    diffuse_budget = compute_trade_budget_usd(diffuse, config, state={"positions": {}}, available_balance_usd=10.0)

    assert concentrated_budget == 1.723077
    assert diffuse_budget == 1.086864
    assert diffuse_budget < concentrated_budget


def test_compute_trade_budget_auto_mode_splits_remaining_exposure_across_open_slots():
    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL"],
        bankroll_usd=20.0,
        fractional_kelly=0.2,
        min_trade_usd=1.0,
        max_trade_usd=0.0,
        max_total_exposure_usd=6.0,
        max_station_exposure_usd=6.0,
        daily_loss_limit_usd=6.0,
        max_open_positions=4,
    )
    candidate = AutoTradeCandidate(
        station_id="KATL",
        target_date="2026-02-28",
        target_day=0,
        label="68-69 F",
        side="YES",
        token_id="YES1",
        market_price=0.10,
        fair_price=0.90,
        edge_points=80.0,
        model_probability=0.90,
        recommendation="BUY_YES",
        policy_reason="test",
        forecast_winner_label="68-69 F",
        forecast_edge_points=80.0,
        tactical_alignment="neutral",
        why="test",
        position_key="KATL|2026-02-28|68-69 F|YES",
    )
    state = {
        "positions": {
            "KLGA|2026-02-28|48-49 F|YES": {
                "station_id": "KLGA",
                "target_date": "2026-02-28",
                "label": "48-49 F",
                "side": "YES",
                "status": "OPEN",
                "managed_by_bot": True,
                "cost_basis_open_usd": 2.0,
                "shares_open": 10.0,
                "entry_price": 0.2,
            }
        }
    }

    budget = compute_trade_budget_usd(candidate, config, state=state, available_balance_usd=6.0)

    assert budget == 1.333333


def test_compute_trade_budget_blocks_when_live_portfolio_already_uses_cap():
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

    budget = compute_trade_budget_usd(
        candidate,
        config,
        state={"positions": {}},
        available_balance_usd=20.0,
        live_positions=[
            {
                "source": "polymarket_live",
                "station_id": "KATL",
                "target_date": "2026-02-28",
                "label": "70-71 F",
                "side": "YES",
                "cost_basis_open_usd": 6.5,
            }
        ],
    )

    assert budget == 0.0


def test_compute_trade_budget_respects_station_exposure_limit():
    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL"],
        bankroll_usd=20.0,
        fractional_kelly=0.2,
        min_trade_usd=1.0,
        max_trade_usd=2.5,
        max_total_exposure_usd=10.0,
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

    budget = compute_trade_budget_usd(
        candidate,
        config,
        state={"positions": {}},
        available_balance_usd=20.0,
        live_positions=[
            {
                "source": "polymarket_live",
                "station_id": "KATL",
                "target_date": "2026-02-28",
                "label": "70-71 F",
                "side": "YES",
                "cost_basis_open_usd": 3.5,
            }
        ],
    )

    assert budget == 0.0


def test_load_autotrader_config_from_env_keeps_station_limits(monkeypatch):
    monkeypatch.setenv("HELIOS_AUTOTRADE_MAX_TOTAL_EXPOSURE_USD", "9")
    monkeypatch.setenv("HELIOS_AUTOTRADE_MAX_STATION_EXPOSURE_USD", "2.5")
    monkeypatch.setenv("HELIOS_AUTOTRADE_MAX_STATION_POSITIONS", "3")
    monkeypatch.setenv("HELIOS_AUTOTRADE_MAX_TRADE_USD", "0")
    monkeypatch.setenv("HELIOS_AUTOTRADE_ASSUME_LIVE_POSITIONS_MANAGED", "false")

    config = load_autotrader_config_from_env()

    assert config.max_total_exposure_usd == 9.0
    assert config.max_station_exposure_usd == 2.5
    assert config.max_station_positions == 3
    assert config.max_trade_usd == 0.0
    assert config.assume_live_positions_managed is False


def test_compute_trade_budget_uses_mark_risk_not_historical_cost():
    config = AutoTraderConfig(
        enabled=True,
        station_ids=["KATL"],
        bankroll_usd=20.0,
        fractional_kelly=0.2,
        min_trade_usd=1.0,
        max_trade_usd=5.0,
        max_total_exposure_usd=20.0,
        daily_loss_limit_usd=10.0,
        max_open_positions=10,
    )
    candidate = AutoTradeCandidate(
        station_id="KATL",
        target_date="2026-02-28",
        target_day=0,
        label="68-69 F",
        side="YES",
        token_id="YES1",
        market_price=0.24,
        fair_price=0.40,
        edge_points=16.0,
        model_probability=0.40,
        recommendation="BUY_YES",
        policy_reason="test",
        forecast_winner_label="68-69 F",
        forecast_edge_points=16.0,
        tactical_alignment="aligned",
        why="test",
        position_key="KATL|2026-02-28|68-69 F|YES",
    )

    budget = compute_trade_budget_usd(
        candidate,
        config,
        state={"positions": {}},
        available_balance_usd=25.0,
        live_positions=[
            {
                "source": "polymarket_live",
                "station_id": "EGLC",
                "target_date": "2026-02-28",
                "label": "11C",
                "side": "YES",
                "cost_basis_open_usd": 16.0,
                "current_value_usd": 2.0,
                "managed_by_bot": False,
            }
        ],
    )

    assert budget == 1.0


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
    today = _today_iso()
    state_path.write_text(
        f"""{{
  "trading_day": "{today}",
  "spent_today_usd": 3.5,
  "open_risk_usd": 3.5,
  "trades_today": 2,
  "positions": {{
    "KATL|{today}|68-69 F|YES": {{
      "station_id": "KATL",
      "target_date": "{today}",
      "label": "68-69 F",
      "side": "YES",
      "notional_usd": 1.5,
      "opened_at_utc": "2026-02-28T15:00:00+00:00",
      "status": "OPEN"
    }},
    "KLGA|{today}|48-49 F|NO": {{
      "station_id": "KLGA",
      "target_date": "{today}",
      "label": "48-49 F",
      "side": "NO",
      "notional_usd": 2.0,
      "opened_at_utc": "2026-02-28T15:05:00+00:00",
      "status": "OPEN"
    }}
  }}
}}""",
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
    today = _today_iso()
    state_path.write_text(
        f"""{{
  "trading_day": "{today}",
  "open_risk_usd": 2.75,
  "positions": {{
    "KATL|{today}|68-69 F|YES": {{
      "station_id": "KATL",
      "label": "68-69 F",
      "side": "YES",
      "strategy": "terminal_value",
      "cost_basis_open_usd": 1.25,
      "opened_at_utc": "2026-02-28T15:00:00+00:00",
      "status": "OPEN"
    }},
    "KLGA|{today}|48-49 F|NO": {{
      "station_id": "KLGA",
      "label": "48-49 F",
      "side": "NO",
      "strategy": "tactical_reprice",
      "cost_basis_open_usd": 1.50,
      "opened_at_utc": "2026-02-28T15:05:00+00:00",
      "status": "OPEN"
    }}
  }}
}}""",
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


def test_resolve_station_market_target_prefers_explicit_date():
    config = AutoTraderConfig(enabled=True, station_ids=["KATL"], target_day=0, target_date="2026-03-05")

    resolved_day, resolved_date = resolve_station_market_target("KATL", config)

    assert resolved_date == "2026-03-05"
    assert isinstance(resolved_day, int)


def test_run_station_once_ignores_closed_state_duplicate(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        mode="paper",
        station_ids=["KATL"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config, execution=_ReduceOnlyExecutionClient())
    trader.state["positions"] = {
        "KATL|2026-02-28|68-69 F|YES": {
            "station_id": "KATL",
            "target_date": "2026-02-28",
            "label": "68-69 F",
            "side": "YES",
            "token_id": "YES1",
            "status": "CLOSED",
            "managed_by_bot": True,
            "shares": 10.0,
            "shares_open": 0.0,
            "entry_price": 0.24,
            "notional_usd": 2.4,
            "cost_basis_open_usd": 0.0,
        }
    }
    trader._persist_state()

    async def _fake_fetch(*args, **kwargs):
        return _sample_payload()

    trader._fetch_signal_payload = _fake_fetch

    result = asyncio.run(trader.run_station_once("KATL", target_date="2026-02-28"))

    assert "duplicate_position" not in result.get("reasons", [])


def test_run_station_once_records_pending_entry_and_blocks_duplicate_reentry(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    execution = _PendingExecutionClient()
    config = AutoTraderConfig(
        enabled=True,
        mode="live",
        station_ids=["KATL"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config, execution=execution)

    async def _fake_fetch(*args, **kwargs):
        return _sample_payload()

    trader._fetch_signal_payload = _fake_fetch

    first = asyncio.run(trader.run_station_once("KATL", target_date="2026-02-28"))

    assert first["status"] == "live_entry_pending"
    assert execution.buy_calls == 1
    assert trader.state["positions"] == {}
    pending_orders = trader.state.get("pending_orders") or {}
    assert len(pending_orders) == 1

    second = asyncio.run(trader.run_station_once("KATL", target_date="2026-02-28"))

    assert second["status"] == "skip"
    assert "pending_entry_in_flight" in second["reasons"]
    assert execution.buy_calls == 1


def test_reconcile_pending_entry_persists_fill_and_clears_pending(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    execution = _PendingExecutionClient()
    config = AutoTraderConfig(
        enabled=True,
        mode="live",
        station_ids=["KATL"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config, execution=execution)

    async def _fake_fetch(*args, **kwargs):
        return _sample_payload()

    trader._fetch_signal_payload = _fake_fetch
    asyncio.run(trader.run_station_once("KATL", target_date="2026-02-28"))

    pending = next(iter((trader.state.get("pending_orders") or {}).values()))
    order_id = pending["order_id"]
    execution.orders[order_id] = {
        "success": True,
        "status": "matched",
        "orderID": order_id,
        "size_matched": str(pending["size"]),
    }

    events = trader._reconcile_pending_orders()

    assert any(event["status"] == "pending_entry_filled" for event in events)
    assert trader.state["pending_orders"] == {}
    position = trader.state["positions"]["KATL|2026-02-28|68-69 F|YES"]
    assert position["status"] == "OPEN"
    assert position["shares_open"] == pending["size"]


def test_manage_open_positions_records_pending_exit_and_blocks_duplicate_retry(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    execution = _PendingExecutionClient()
    config = AutoTraderConfig(
        enabled=True,
        mode="live",
        station_ids=["EGLC"],
        tactical_timeout_minutes=1,
        max_depth_participation=1.0,
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config, execution=execution)
    trader.state["positions"] = {
        "EGLC|2026-03-01|14C OR HIGHER|YES": {
            "station_id": "EGLC",
            "target_date": "2026-03-01",
            "label": "14C OR HIGHER",
            "side": "YES",
            "token_id": "tok1",
            "status": "OPEN",
            "mode": "live",
            "strategy": "tactical_reprice",
            "managed_by_bot": True,
            "entry_price": 0.04,
            "shares": 200.0,
            "shares_open": 200.0,
            "notional_usd": 8.0,
            "cost_basis_open_usd": 8.0,
            "current_value_usd": 9.0,
            "cash_pnl_usd": 1.0,
            "opened_at_utc": "2026-03-01T00:00:00+00:00",
        }
    }
    trader._persist_state()

    first = asyncio.run(
        trader._manage_open_positions(
            "EGLC",
            {"station_id": "EGLC", "target_date": "2026-03-01", "trading": {}},
        )
    )

    assert first
    assert first[0]["status"] == "live_exit_pending"
    assert execution.sell_calls == 1
    assert len(trader.state.get("pending_orders") or {}) == 1

    second = asyncio.run(
        trader._manage_open_positions(
            "EGLC",
            {"station_id": "EGLC", "target_date": "2026-03-01", "trading": {}},
        )
    )

    assert second == []
    assert execution.sell_calls == 1


def test_reconcile_pending_exit_closes_position_and_clears_pending(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    execution = _PendingExecutionClient()
    config = AutoTraderConfig(
        enabled=True,
        mode="live",
        station_ids=["EGLC"],
        tactical_timeout_minutes=1,
        max_depth_participation=1.0,
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config, execution=execution)
    trader.state["positions"] = {
        "EGLC|2026-03-01|14C OR HIGHER|YES": {
            "station_id": "EGLC",
            "target_date": "2026-03-01",
            "label": "14C OR HIGHER",
            "side": "YES",
            "token_id": "tok1",
            "status": "OPEN",
            "mode": "live",
            "strategy": "tactical_reprice",
            "managed_by_bot": True,
            "entry_price": 0.04,
            "shares": 200.0,
            "shares_open": 200.0,
            "notional_usd": 8.0,
            "cost_basis_open_usd": 8.0,
            "current_value_usd": 9.0,
            "cash_pnl_usd": 1.0,
            "opened_at_utc": "2026-03-01T00:00:00+00:00",
        }
    }
    trader._persist_state()
    asyncio.run(
        trader._manage_open_positions(
            "EGLC",
            {"station_id": "EGLC", "target_date": "2026-03-01", "trading": {}},
        )
    )

    pending = next(iter((trader.state.get("pending_orders") or {}).values()))
    order_id = pending["order_id"]
    execution.orders[order_id] = {
        "success": True,
        "status": "matched",
        "orderID": order_id,
        "size_matched": str(pending["size"]),
    }

    events = trader._reconcile_pending_orders()

    assert any(event["status"] == "pending_exit_filled" for event in events)
    assert trader.state["pending_orders"] == {}
    position = trader.state["positions"]["EGLC|2026-03-01|14C OR HIGHER|YES"]
    assert position["status"] == "CLOSED"
    assert position["shares_open"] == 0.0


def test_terminal_policy_flip_alone_does_not_force_exit(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        mode="paper",
        station_ids=["KATL"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config)
    position = {
        "station_id": "KATL",
        "target_date": "2026-02-28",
        "label": "68-69 F",
        "side": "YES",
        "token_id": "YES1",
        "status": "OPEN",
        "strategy": "terminal_value",
        "managed_by_bot": True,
        "entry_price": 0.20,
        "shares": 5.0,
        "shares_open": 5.0,
        "notional_usd": 1.0,
        "cost_basis_open_usd": 1.0,
        "opened_at_utc": "2026-02-28T15:00:00+00:00",
    }
    payload = {
        "station_id": "KATL",
        "target_date": "2026-02-28",
        "trading": {
            "all_terminal_opportunities": [
                {
                    "label": "68-69 F",
                    "fair_yes": 0.23,
                    "edge_yes": 0.02,
                    "terminal_policy": {"allowed": False},
                }
            ]
        },
    }
    book = TopOfBook(
        token_id="YES1",
        best_bid=0.19,
        best_ask=0.20,
        bid_size=50.0,
        ask_size=50.0,
        spread=0.01,
        min_order_size=1.0,
        last_trade_price=0.195,
        raw={},
    )

    exit_plan, reasons = trader._evaluate_exit_decision(position, payload, book)

    assert exit_plan is None
    assert reasons == ["hold"]


def test_take_profit_waits_while_fair_still_has_room(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        mode="paper",
        station_ids=["KATL"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config)
    position = {
        "station_id": "KATL",
        "target_date": "2026-02-28",
        "label": "68-69 F",
        "side": "YES",
        "token_id": "YES1",
        "status": "OPEN",
        "strategy": "terminal_value",
        "managed_by_bot": True,
        "entry_price": 0.20,
        "shares": 5.0,
        "shares_open": 5.0,
        "notional_usd": 1.0,
        "cost_basis_open_usd": 1.0,
        "opened_at_utc": "2026-02-28T15:00:00+00:00",
    }
    payload = {
        "station_id": "KATL",
        "target_date": "2026-02-28",
        "trading": {
            "all_terminal_opportunities": [
                {
                    "label": "68-69 F",
                    "fair_yes": 0.35,
                    "edge_yes": 0.10,
                    "terminal_policy": {"allowed": True},
                }
            ]
        },
    }
    book = TopOfBook(
        token_id="YES1",
        best_bid=0.27,
        best_ask=0.28,
        bid_size=50.0,
        ask_size=50.0,
        spread=0.01,
        min_order_size=1.0,
        last_trade_price=0.275,
        raw={},
    )

    exit_plan, reasons = trader._evaluate_exit_decision(position, payload, book)

    assert exit_plan is None
    assert reasons == ["hold"]


def test_take_profit_exits_when_bid_has_caught_up_to_fair(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        mode="paper",
        station_ids=["KATL"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config)
    position = {
        "station_id": "KATL",
        "target_date": "2026-02-28",
        "label": "68-69 F",
        "side": "YES",
        "token_id": "YES1",
        "status": "OPEN",
        "strategy": "terminal_value",
        "managed_by_bot": True,
        "entry_price": 0.20,
        "shares": 5.0,
        "shares_open": 5.0,
        "notional_usd": 1.0,
        "cost_basis_open_usd": 1.0,
        "opened_at_utc": "2026-02-28T15:00:00+00:00",
    }
    payload = {
        "station_id": "KATL",
        "target_date": "2026-02-28",
        "trading": {
            "all_terminal_opportunities": [
                {
                    "label": "68-69 F",
                    "fair_yes": 0.26,
                    "edge_yes": 0.01,
                    "terminal_policy": {"allowed": True},
                }
            ]
        },
    }
    book = TopOfBook(
        token_id="YES1",
        best_bid=0.25,
        best_ask=0.26,
        bid_size=50.0,
        ask_size=50.0,
        spread=0.01,
        min_order_size=1.0,
        last_trade_price=0.255,
        raw={},
    )

    exit_plan, reasons = trader._evaluate_exit_decision(position, payload, book)

    assert reasons == []
    assert exit_plan is not None
    assert exit_plan["reason"] == "take_profit"
    assert exit_plan["decision_context"]["fair_value_absorbed"] is True
    assert exit_plan["decision_context"]["current_best_bid"] == 0.25
    assert exit_plan["decision_context"]["policy_allowed"] is True


def test_append_log_serializes_concurrent_writes_without_corrupting_jsonl(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        mode="paper",
        station_ids=["KATL"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config)

    def _write(seq: int) -> None:
        trader._append_log({"status": "threaded_test", "seq": seq})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_write, range(200)))

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200
    rows = [json.loads(line) for line in lines]
    assert sorted(int(row["seq"]) for row in rows) == list(range(200))


def test_live_entry_balance_failure_sets_station_cooldown_and_blocks_retries(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    execution = _InsufficientBalanceExecutionClient()
    config = AutoTraderConfig(
        enabled=True,
        mode="live",
        station_ids=["KATL"],
        order_failure_cooldown_seconds=1800,
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config, execution=execution)

    async def _fake_fetch(*args, **kwargs):
        return _sample_payload()

    trader._fetch_signal_payload = _fake_fetch

    first = asyncio.run(trader.run_station_once("KATL", target_date="2026-02-28"))

    assert first["status"] == "skip"
    assert first["reasons"] == ["insufficient_balance_or_allowance"]
    assert execution.buy_calls == 1
    cooldown = trader.state["station_cooldowns"]["KATL"]
    assert cooldown["reason"] == "insufficient_balance_or_allowance"

    second = asyncio.run(trader.run_station_once("KATL", target_date="2026-02-28"))

    assert second["status"] == "skip"
    assert second["reasons"] == ["station_order_failure_cooldown"]
    assert execution.buy_calls == 1


class _FakeExecutionClient(PolymarketExecutionClient):
    def __init__(self, snapshot):
        super().__init__(PolymarketExecutionConfig(private_key="0xabc", derive_api_creds=True))
        self._snapshot = snapshot

    def get_live_portfolio_snapshot(self, *, force_refresh: bool = False):
        return dict(self._snapshot)


class _ReduceOnlyExecutionClient(PolymarketExecutionClient):
    def __init__(self):
        super().__init__(PolymarketExecutionConfig())

    def get_top_of_book(self, token_id: str):
        return TopOfBook(
            token_id=token_id,
            best_bid=0.045,
            best_ask=0.046,
            bid_size=500.0,
            ask_size=500.0,
            spread=0.001,
            min_order_size=1.0,
            last_trade_price=0.045,
            raw={},
        )


class _PendingExecutionClient(PolymarketExecutionClient):
    def __init__(self):
        super().__init__(PolymarketExecutionConfig(private_key="0xabc", derive_api_creds=True))
        self.buy_calls = 0
        self.sell_calls = 0
        self.orders = {}
        self._snapshot = {
            "user": "0xuser",
            "free_collateral_usd": 25.0,
            "collateral_allowance_ready": True,
            "portfolio_value_usd": 25.0,
            "positions_market_value_usd": 0.0,
            "positions_cost_basis_usd": 0.0,
            "open_positions": [],
        }

    def get_live_portfolio_snapshot(self, *, force_refresh: bool = False):
        return dict(self._snapshot)

    def ensure_collateral_ready(self, required_amount: float):
        return {"balance": 25.0, "allowance_ready": True}

    def ensure_conditional_ready(self, token_id: str, required_size: float):
        return {"balance": required_size, "allowance_ready": True}

    def get_top_of_book(self, token_id: str):
        if token_id == "tok1":
            return TopOfBook(
                token_id=token_id,
                best_bid=0.045,
                best_ask=0.046,
                bid_size=500.0,
                ask_size=500.0,
                spread=0.001,
                min_order_size=1.0,
                last_trade_price=0.045,
                raw={},
            )
        return TopOfBook(
            token_id=token_id,
            best_bid=0.23,
            best_ask=0.24,
            bid_size=50.0,
            ask_size=50.0,
            spread=0.01,
            min_order_size=1.0,
            last_trade_price=0.24,
            raw={},
        )

    def place_limit_buy(self, *, token_id: str, price: float, size: float, order_type: str = "FAK"):
        self.buy_calls += 1
        order_id = f"buy-{self.buy_calls}"
        self.orders[order_id] = {"success": True, "status": "delayed", "orderID": order_id}
        return dict(self.orders[order_id])

    def place_limit_sell(self, *, token_id: str, price: float, size: float, order_type: str = "FAK"):
        self.sell_calls += 1
        order_id = f"sell-{self.sell_calls}"
        self.orders[order_id] = {"success": True, "status": "delayed", "orderID": order_id}
        return dict(self.orders[order_id])

    def get_order(self, order_id: str):
        return dict(self.orders.get(order_id) or {"success": True, "status": "delayed", "orderID": order_id})


class _InsufficientBalanceExecutionClient(PolymarketExecutionClient):
    def __init__(self):
        super().__init__(PolymarketExecutionConfig(private_key="0xabc", derive_api_creds=True))
        self.buy_calls = 0
        self._snapshot = {
            "user": "0xuser",
            "free_collateral_usd": 25.0,
            "collateral_allowance_ready": True,
            "portfolio_value_usd": 25.0,
            "positions_market_value_usd": 0.0,
            "positions_cost_basis_usd": 0.0,
            "open_positions": [],
        }

    def get_live_portfolio_snapshot(self, *, force_refresh: bool = False):
        return dict(self._snapshot)

    def ensure_collateral_ready(self, required_amount: float):
        return {"balance": 25.0, "allowance_ready": True}

    def get_top_of_book(self, token_id: str):
        return TopOfBook(
            token_id=token_id,
            best_bid=0.23,
            best_ask=0.24,
            bid_size=50.0,
            ask_size=50.0,
            spread=0.01,
            min_order_size=1.0,
            last_trade_price=0.24,
            raw={},
        )

    def place_limit_buy(self, *, token_id: str, price: float, size: float, order_type: str = "FAK"):
        self.buy_calls += 1
        raise RuntimeError("PolyApiException[status_code=400, error_message={'error': 'not enough balance / allowance'}]")


def test_status_snapshot_uses_live_polymarket_positions_for_exposure(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    today = _today_iso()
    state_path.write_text(
        f"""{{
  "trading_day": "{today}",
  "positions": {{}}
}}""",
        encoding="utf-8",
    )
    config = AutoTraderConfig(
        enabled=True,
        mode="live",
        station_ids=["KATL", "EGLC"],
        max_total_exposure_usd=20.0,
        max_station_exposure_usd=5.0,
        state_path=str(state_path),
        log_path=str(log_path),
    )
    live_snapshot = {
        "user": "0xuser",
        "free_collateral_usd": 0.840014,
        "collateral_allowance_ready": True,
        "portfolio_value_usd": 20.1959,
        "positions_market_value_usd": 20.1959,
        "positions_cost_basis_usd": 26.92,
        "open_positions": [
            {
                "source": "polymarket_live",
                "station_id": "KATL",
                "target_date": today,
                "label": "76F OR HIGHER",
                "side": "NO",
                "cost_basis_open_usd": 13.16,
                "current_value_usd": 13.90,
                "cash_pnl_usd": 0.74,
                "shares_open": 36.6,
            },
            {
                "source": "polymarket_live",
                "station_id": "EGLC",
                "target_date": today,
                "label": "14C OR HIGHER",
                "side": "YES",
                "cost_basis_open_usd": 5.73,
                "current_value_usd": 6.21,
                "cash_pnl_usd": 0.48,
                "shares_open": 133.6,
            },
            {
                "source": "polymarket_live",
                "station_id": "KDAL",
                "target_date": "2026-02-28",
                "label": "88F OR HIGHER",
                "side": "YES",
                "cost_basis_open_usd": 8.03,
                "current_value_usd": 0.08,
                "cash_pnl_usd": -7.95,
                "shares_open": 168.8,
            },
        ],
    }
    trader = AutoTrader(config=config, execution=_FakeExecutionClient(live_snapshot))

    snapshot = trader.get_status_snapshot()

    assert snapshot["portfolio"]["open_positions_count"] == 3
    assert snapshot["portfolio"]["open_risk_usd"] == 20.19
    assert snapshot["portfolio"]["free_collateral_usd"] == 0.840014
    assert snapshot["portfolio"]["portfolio_value_usd"] == 20.1959
    assert snapshot["portfolio"]["effective_bankroll_usd"] == 20.0
    assert snapshot["portfolio"]["live_only_open_positions_count"] == 3
    assert snapshot["portfolio"]["external_open_positions_count"] == 0
    assert snapshot["portfolio"]["managed_open_positions_count"] == 3
    assert snapshot["portfolio"]["tracking_only_positions_count"] == 0
    assert snapshot["wallet"]["free_collateral_usd"] == 0.840014
    assert snapshot["state"]["strategies"] == {"managed_live": 3}
    assert snapshot["risk"]["max_trade_cap_enabled"] is False


def test_sync_live_positions_into_state_imports_managed_positions_by_default(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        mode="live",
        station_ids=["KATL"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config)

    trader._sync_live_positions_into_state(
        [
            {
                "source": "polymarket_live",
                "station_id": "KATL",
                "target_date": "2026-03-01",
                "label": "76F OR HIGHER",
                "side": "NO",
                "token_id": "tok1",
                "cost_basis_open_usd": 13.16,
                "current_value_usd": 13.90,
                "cash_pnl_usd": 0.74,
                "shares_open": 36.57,
                "avg_price": 0.3599,
            }
        ]
    )

    positions = trader.state.get("positions") or {}
    assert len(positions) == 1
    row = next(iter(positions.values()))
    assert row["source"] == "autotrader_live"
    assert row["strategy"] == "managed_live"
    assert row["managed_by_bot"] is True
    assert row["cost_basis_open_usd"] == 13.16
    assert row["status"] == "OPEN"


def test_sync_live_positions_marks_missing_bot_live_position_as_closed_unreconciled(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        mode="live",
        station_ids=["KATL"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config)
    trader.state["positions"] = {
        "KATL|2026-03-01|76F OR HIGHER|NO": {
            "station_id": "KATL",
            "target_date": "2026-03-01",
            "label": "76F OR HIGHER",
            "side": "NO",
            "token_id": "tok1",
            "status": "OPEN",
            "source": "autotrader_live",
            "strategy": "managed_live",
            "managed_by_bot": True,
            "shares": 36.57,
            "shares_open": 36.57,
            "entry_price": 0.3599,
            "cost_basis_open_usd": 13.16,
            "current_value_usd": 13.9,
            "live_synced_at_utc": "2026-03-01T12:00:00+00:00",
        }
    }

    trader._sync_live_positions_into_state([])

    row = trader.state["positions"]["KATL|2026-03-01|76F OR HIGHER|NO"]
    assert row["status"] == "CLOSED_UNRECONCILED"
    assert row["shares_open"] == 36.57
    assert row["cost_basis_open_usd"] == 13.16
    assert row["close_reconciliation_status"] == "missing_from_live_snapshot"
    assert row["unreconciled_close_detected_at_utc"]
    trader._persist_state()
    snapshot = trader.get_status_snapshot()
    assert snapshot["portfolio"]["open_positions_count"] == 0
    assert snapshot["portfolio"]["unreconciled_closed_positions_count"] == 1
    assert snapshot["state"]["unreconciled_positions"] == 1
    log_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(row.get("status") == "live_position_missing_unreconciled" for row in log_rows)


def test_status_snapshot_repairs_legacy_live_positions_to_managed_when_account_is_bot_only(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    today = _today_iso()
    state_path.write_text(
        f"""{{
  "trading_day": "{today}",
  "positions": {{
    "KATL|{today}|76F OR HIGHER|NO": {{
      "station_id": "KATL",
      "target_date": "{today}",
      "label": "76F OR HIGHER",
      "side": "NO",
      "token_id": "tok1",
      "status": "OPEN",
      "source": "polymarket_live",
      "strategy": "external_live",
      "managed_by_bot": false,
      "shares": 10.0,
      "shares_open": 10.0,
      "entry_price": 0.2,
      "cost_basis_open_usd": 2.0,
      "current_value_usd": 1.5
    }}
  }}
}}""",
        encoding="utf-8",
    )
    config = AutoTraderConfig(
        enabled=True,
        mode="live",
        station_ids=["KATL"],
        assume_live_positions_managed=True,
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config)

    snapshot = trader.get_status_snapshot()

    row = snapshot["portfolio"]["open_positions"][0]
    assert row["managed_by_bot"] is True
    assert row["source"] == "autotrader_live"
    assert row["strategy"] == "managed_live"
    reloaded = json.loads(state_path.read_text(encoding="utf-8"))
    persisted = reloaded["positions"][f"KATL|{today}|76F OR HIGHER|NO"]
    assert persisted["managed_by_bot"] is True
    assert persisted["source"] == "autotrader_live"
    assert persisted["strategy"] == "managed_live"


def test_reduce_only_does_not_sell_explicit_tracking_only_live_position_in_paper_mode(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        mode="paper",
        station_ids=["EGLC"],
        max_total_exposure_usd=5.0,
        max_trade_usd=2.0,
        max_depth_participation=1.0,
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config, execution=_ReduceOnlyExecutionClient())
    trader._sync_live_positions_into_state(
        [
            {
                "source": "polymarket_live",
                "station_id": "EGLC",
                "target_date": "2026-03-01",
                "label": "14C OR HIGHER",
                "side": "YES",
                "token_id": "tok1",
                "cost_basis_open_usd": 8.0,
                "current_value_usd": 9.0,
                "cash_pnl_usd": 1.0,
                "shares_open": 200.0,
                "avg_price": 0.04,
                "current_price": 0.045,
                "managed_by_bot": False,
            }
        ]
    )
    trader._persist_state()

    events = asyncio.run(trader._manage_reduce_only_portfolio())

    assert events
    assert events[0]["status"] == "reduce_only_hold"
    assert events[0]["reasons"] == ["no_managed_positions_to_trim"]
    row = next(iter((trader.state.get("positions") or {}).values()))
    assert row["management_override"] == "tracking_only"
    snapshot = trader.get_status_snapshot()
    assert snapshot["portfolio"]["open_risk_usd"] == 9.0
    assert snapshot["portfolio"]["reduce_only_active"] is True


def test_reduce_only_trims_bot_managed_position_in_paper_mode(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        mode="paper",
        station_ids=["EGLC"],
        max_total_exposure_usd=5.0,
        max_trade_usd=2.0,
        max_depth_participation=1.0,
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config, execution=_ReduceOnlyExecutionClient())
    trader.state["positions"] = {
        "EGLC|2026-03-01|14C OR HIGHER|YES": {
            "station_id": "EGLC",
            "target_date": "2026-03-01",
            "label": "14C OR HIGHER",
            "side": "YES",
            "token_id": "tok1",
            "status": "OPEN",
            "mode": "live",
            "strategy": "terminal_value",
            "managed_by_bot": True,
            "entry_price": 0.04,
            "shares": 200.0,
            "shares_open": 200.0,
            "notional_usd": 8.0,
            "cost_basis_open_usd": 8.0,
            "current_value_usd": 9.0,
            "cash_pnl_usd": 1.0,
            "opened_at_utc": "2026-03-01T10:00:00+00:00",
        }
    }
    trader._persist_state()

    events = asyncio.run(trader._manage_reduce_only_portfolio())

    assert events
    assert events[0]["status"] == "paper_reduce_only_exit"
    snapshot = trader.get_status_snapshot()
    assert 5.0 < snapshot["portfolio"]["open_risk_usd"] < 9.0
    assert snapshot["portfolio"]["reduce_only_active"] is True


def test_manage_open_positions_skips_explicit_tracking_only_positions(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        mode="paper",
        station_ids=["EGLC"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config, execution=_ReduceOnlyExecutionClient())
    trader._sync_live_positions_into_state(
        [
            {
                "source": "polymarket_live",
                "station_id": "EGLC",
                "target_date": "2026-03-01",
                "label": "14C OR HIGHER",
                "side": "YES",
                "token_id": "tok1",
                "cost_basis_open_usd": 8.0,
                "current_value_usd": 9.0,
                "cash_pnl_usd": 1.0,
                "shares_open": 200.0,
                "avg_price": 0.04,
                "current_price": 0.045,
                "managed_by_bot": False,
            }
        ]
    )
    trader._persist_state()

    events = asyncio.run(
        trader._manage_open_positions(
            "EGLC",
            {"station_id": "EGLC", "target_date": "2026-03-01", "trading": {}},
        )
    )

    assert events == []
    snapshot = trader.get_status_snapshot("EGLC")
    assert snapshot["state"]["open_positions"] == 1


def test_run_station_once_skips_reentry_when_unreconciled_position_exists(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    log_path = tmp_path / "autotrader.jsonl"
    config = AutoTraderConfig(
        enabled=True,
        mode="paper",
        station_ids=["KATL"],
        state_path=str(state_path),
        log_path=str(log_path),
    )
    trader = AutoTrader(config=config, execution=_ReduceOnlyExecutionClient())
    trader.state["positions"] = {
        "KATL|2026-02-28|68-69 F|YES": {
            "station_id": "KATL",
            "target_date": "2026-02-28",
            "label": "68-69 F",
            "side": "YES",
            "token_id": "YES1",
            "status": "CLOSED_UNRECONCILED",
            "source": "autotrader_live",
            "strategy": "managed_live",
            "managed_by_bot": True,
            "shares": 10.0,
            "shares_open": 10.0,
            "entry_price": 0.24,
            "notional_usd": 2.4,
            "cost_basis_open_usd": 2.4,
            "current_value_usd": 2.0,
            "unreconciled_close_detected_at_utc": "2026-02-28T16:00:00+00:00",
        }
    }
    trader._persist_state()

    async def _fake_fetch(*args, **kwargs):
        return _sample_payload()

    trader._fetch_signal_payload = _fake_fetch

    result = asyncio.run(trader.run_station_once("KATL", target_date="2026-02-28"))

    assert result["status"] == "skip"
    assert "unreconciled_position_exists" in result["reasons"]


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


def test_persist_exit_reconciles_closed_unreconciled_position(tmp_path):
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
            "status": "CLOSED_UNRECONCILED",
            "source": "autotrader_live",
            "strategy": "managed_live",
            "managed_by_bot": True,
            "entry_price": 0.20,
            "shares": 5.0,
            "shares_open": 5.0,
            "notional_usd": 1.0,
            "cost_basis_open_usd": 1.0,
            "current_value_usd": 1.1,
            "unreconciled_close_detected_at_utc": "2026-02-28T15:10:00+00:00",
            "close_reconciliation_status": "pending_exit_missing_from_live_snapshot",
            "fills": [],
        }
    }
    trader.state["open_risk_usd"] = 0.0
    trader._persist_state()

    persisted = trader._persist_exit(
        "KATL|2026-02-28|68-69 F|YES",
        {
            "size": 5.0,
            "best_bid": 0.28,
            "proceeds_usd": 1.4,
            "reason": "take_profit",
        },
        {"mode": "live"},
        mode="live",
    )

    assert persisted["reason"] == "take_profit"
    reloaded = json.loads(state_path.read_text(encoding="utf-8"))
    row = reloaded["positions"]["KATL|2026-02-28|68-69 F|YES"]
    assert row["status"] == "CLOSED"
    assert row["close_reconciliation_status"] == "confirmed_by_order_fill"
    assert row["reconciled_close_at_utc"]
    assert round(row["realized_pnl_usd"], 2) == 0.40


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


def test_summarize_order_response_marks_delayed_as_pending():
    client = PolymarketExecutionClient(PolymarketExecutionConfig())

    summary = client.summarize_order_response(
        {
            "success": True,
            "status": "delayed",
            "orderID": "abc123",
        },
        fallback_size=5.0,
        fallback_price=0.24,
    )

    assert summary["success"] is True
    assert summary["pending"] is True
    assert summary["matched_size"] == 0.0
    assert summary["status"] == "delayed"
