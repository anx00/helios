from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from core.autotrader.bandit import LinUCBBandit
from core.autotrader.learning import LearningConfig, LearningRunner
from core.autotrader.models import DecisionContext
from core.autotrader.paper_broker import PaperBroker
from core.autotrader.risk import RiskConfig, RiskGate
from core.autotrader.storage import AutoTraderStorage
from core.autotrader.strategy import build_strategy_catalog

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


def _sample_context() -> DecisionContext:
    now = datetime.now(UTC)
    now_nyc = now.astimezone(NYC)
    return DecisionContext(
        ts_utc=now,
        ts_nyc=now_nyc,
        station_id="KLGA",
        nowcast={
            "confidence": 0.8,
            "tmax_sigma_f": 1.2,
            "p_bucket": [
                {"label": "30-31°F", "probability": 0.65},
                {"label": "32-33°F", "probability": 0.35},
            ],
            "qc_state": "OK",
        },
        market_state={
            "30-31°F": {"best_bid": 0.52, "best_ask": 0.56, "bid_depth": 80, "ask_depth": 90, "spread": 0.04, "mid": 0.54},
            "32-33°F": {"best_bid": 0.31, "best_ask": 0.35, "bid_depth": 50, "ask_depth": 60, "spread": 0.04, "mid": 0.33},
        },
        confidence=0.8,
        tmax_sigma_f=1.2,
        nowcast_age_seconds=30,
        market_age_seconds=10,
        spread_top=0.04,
        depth_imbalance=-0.05,
        volatility_short=0.01,
        prediction_churn_short=0.02,
        event_window_active=True,
        qc_state="OK",
        hour_nyc=now_nyc.hour,
    )


def test_strategy_contract_returns_actions_or_reasons():
    catalog = build_strategy_catalog()
    ctx = _sample_context()

    for name, strategy in catalog.items():
        decision = strategy.evaluate(ctx)
        assert decision.strategy_name == name
        assert decision.station_id == "KLGA"
        assert isinstance(decision.actions, list)
        assert isinstance(decision.no_trade_reasons, list)


def test_risk_gate_blocks_staleness_and_loss():
    ctx = _sample_context()
    cfg = RiskConfig(max_daily_loss=0.1, max_nowcast_age_seconds=60, max_market_age_seconds=60)
    gate = RiskGate(cfg)

    stale_ctx = _sample_context()
    stale_ctx.nowcast_age_seconds = 999
    stale_ctx.market_age_seconds = 999

    risk = gate.evaluate(
        context=stale_ctx,
        positions={"30-31°F": {"size": 0.1}},
        daily_pnl=-0.2,
        orders_last_hour=0,
        last_trade_ts=None,
    )
    assert risk.blocked
    assert "daily_loss_limit" in risk.reasons
    assert "stale_nowcast" in risk.reasons
    assert "stale_market" in risk.reasons


def test_paper_broker_executes_taker_and_updates_position():
    broker = PaperBroker()
    ctx = _sample_context()
    actions = [
        {
            "bucket": "30-31°F",
            "side": "buy",
            "order_type": "taker",
            "target_size": 0.2,
            "max_price": 0.8,
            "min_price": 0.0,
            "confidence": 0.8,
            "urgency": 0.9,
            "reason": "unit test",
        }
    ]

    orders, fills = broker.execute_actions(
        timestamp_utc=ctx.ts_utc,
        station_id=ctx.station_id,
        strategy_name="conservative_edge",
        actions=actions,
        market_state=ctx.market_state,
    )
    assert len(fills) >= 1
    pos = broker.get_positions()
    assert "30-31°F" in pos
    assert pos["30-31°F"]["size"] > 0

    # maker updates should be safe even without pending orders
    upd_orders, upd_fills = broker.update(
        timestamp_utc=ctx.ts_utc + timedelta(seconds=60),
        station_id=ctx.station_id,
        market_state=ctx.market_state,
        elapsed_seconds=60,
    )
    assert isinstance(upd_orders, list)
    assert isinstance(upd_fills, list)


def test_bandit_state_roundtrip_and_update():
    bandit = LinUCBBandit(["a", "b"], feature_dim=4, alpha=0.8)
    x = [0.5, 0.1, 0.2, 1.0]
    pick1 = bandit.select(x)
    bandit.update("a", x, reward=1.0)
    state = bandit.to_state()
    restored = LinUCBBandit.from_state(state)
    pick2 = restored.select(x)
    assert pick1.strategy_name in {"a", "b"}
    assert pick2.strategy_name in {"a", "b"}


def test_learning_window_split_shape():
    storage = AutoTraderStorage(db_path="data/test_autotrader.db")
    runner = LearningRunner(storage=storage, registry_dir="data/test_model_registry")
    cfg = LearningConfig(station_id="KLGA", train_days=53, val_days=7)
    train_start, train_end, val_start, val_end = runner._build_windows(cfg)
    assert (train_end - train_start).days + 1 == 53
    assert (val_end - val_start).days + 1 == 7
    assert (val_start - train_end).days == 1
