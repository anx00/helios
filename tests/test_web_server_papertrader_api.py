from types import SimpleNamespace

import pytest

import web_server


def _base_papertrader_config(initial_bankroll=50.0):
    return SimpleNamespace(
        initial_bankroll_usd=initial_bankroll,
        bankroll_usd=initial_bankroll,
        station_ids=["EGLC"],
        target_day=0,
        target_date="2026-03-03",
        enabled=False,
        max_total_exposure_usd=4.0,
    )


def test_build_papertrader_config_ignores_legacy_bankroll_override(monkeypatch):
    monkeypatch.setattr(web_server, "load_papertrader_config", lambda: _base_papertrader_config(50.0))
    monkeypatch.setattr(
        web_server,
        "_load_papertrader_runtime_overrides",
        lambda: {"station_ids": ["EGLC"], "initial_bankroll_usd": 100.0},
    )
    monkeypatch.setattr(web_server, "_load_autotrader_runtime_overrides", lambda: {})
    monkeypatch.setattr(web_server, "_default_autotrader_target_date", lambda station_ids, target_day: "2026-03-03")

    cfg = web_server._build_papertrader_config()

    assert cfg.initial_bankroll_usd == 50.0
    assert cfg.bankroll_usd == 50.0


def test_build_papertrader_config_respects_explicit_runtime_bankroll_override(monkeypatch):
    monkeypatch.setattr(web_server, "load_papertrader_config", lambda: _base_papertrader_config(50.0))
    monkeypatch.setattr(
        web_server,
        "_load_papertrader_runtime_overrides",
        lambda: {"station_ids": ["EGLC"], "initial_bankroll_usd": 75.0, "initial_bankroll_source": "runtime"},
    )
    monkeypatch.setattr(web_server, "_load_autotrader_runtime_overrides", lambda: {})
    monkeypatch.setattr(web_server, "_default_autotrader_target_date", lambda station_ids, target_day: "2026-03-03")

    cfg = web_server._build_papertrader_config()

    assert cfg.initial_bankroll_usd == 75.0
    assert cfg.bankroll_usd == 75.0


@pytest.mark.asyncio
async def test_reset_papertrader_session_uses_resolved_initial_bankroll(monkeypatch):
    saved = {}

    class FakeTrader:
        def __init__(self):
            self.config = SimpleNamespace(initial_bankroll_usd=100.0, bankroll_usd=100.0)
            self.reset_calls = 0

        def reset_session(self):
            self.reset_calls += 1

    trader = FakeTrader()

    monkeypatch.setattr(web_server, "_papertrader_running", lambda: False)
    monkeypatch.setattr(
        web_server,
        "_build_papertrader_config",
        lambda current_enabled=None: SimpleNamespace(initial_bankroll_usd=50.0),
    )
    monkeypatch.setattr(web_server, "_load_papertrader_runtime_overrides", lambda: {"station_ids": ["EGLC"]})
    monkeypatch.setattr(web_server, "_save_papertrader_runtime_overrides", lambda payload: saved.update(payload))
    monkeypatch.setattr(web_server, "get_papertrader_instance", lambda refresh=False: trader)
    monkeypatch.setattr(
        web_server,
        "get_papertrader_status_snapshot",
        lambda: {"initial_bankroll_usd": trader.config.initial_bankroll_usd},
    )

    status = await web_server.reset_papertrader_session()

    assert saved["initial_bankroll_usd"] == 50.0
    assert trader.config.initial_bankroll_usd == 50.0
    assert trader.config.bankroll_usd == 50.0
    assert trader.reset_calls == 1
    assert status == {"initial_bankroll_usd": 50.0}


@pytest.mark.asyncio
async def test_reset_papertrader_session_ignores_legacy_runtime_bankroll_override(monkeypatch):
    saved = {}

    class FakeTrader:
        def __init__(self):
            self.config = SimpleNamespace(initial_bankroll_usd=100.0, bankroll_usd=100.0)
            self.reset_calls = 0

        def reset_session(self):
            self.reset_calls += 1

    trader = FakeTrader()

    monkeypatch.setattr(web_server, "_papertrader_running", lambda: False)
    monkeypatch.setattr(web_server, "load_papertrader_config", lambda: _base_papertrader_config(50.0))
    monkeypatch.setattr(
        web_server,
        "_load_papertrader_runtime_overrides",
        lambda: {"station_ids": ["EGLC"], "initial_bankroll_usd": 100.0},
    )
    monkeypatch.setattr(web_server, "_load_autotrader_runtime_overrides", lambda: {})
    monkeypatch.setattr(web_server, "_default_autotrader_target_date", lambda station_ids, target_day: "2026-03-03")
    monkeypatch.setattr(web_server, "_save_papertrader_runtime_overrides", lambda payload: saved.update(payload))
    monkeypatch.setattr(web_server, "get_papertrader_instance", lambda refresh=False: trader)
    monkeypatch.setattr(
        web_server,
        "get_papertrader_status_snapshot",
        lambda: {"initial_bankroll_usd": trader.config.initial_bankroll_usd},
    )

    status = await web_server.reset_papertrader_session()

    assert saved["initial_bankroll_usd"] == 50.0
    assert "initial_bankroll_source" not in saved
    assert trader.config.initial_bankroll_usd == 50.0
    assert trader.config.bankroll_usd == 50.0
    assert trader.reset_calls == 1
    assert status == {"initial_bankroll_usd": 50.0}


@pytest.mark.asyncio
async def test_update_papertrader_config_marks_bankroll_as_runtime_override(monkeypatch):
    saved = {}

    monkeypatch.setattr(web_server, "_load_papertrader_runtime_overrides", lambda: {"station_ids": ["EGLC"]})
    monkeypatch.setattr(web_server, "_save_papertrader_runtime_overrides", lambda payload: saved.update(payload))
    monkeypatch.setattr(web_server, "get_papertrader_instance", lambda refresh=False: None)
    monkeypatch.setattr(web_server, "get_papertrader_status_snapshot", lambda: {"ok": True})

    response = await web_server.update_papertrader_config_api(
        web_server.PapertraderRuntimeConfigUpdate(initial_bankroll_usd=75.0)
    )

    assert saved["initial_bankroll_usd"] == 75.0
    assert saved["initial_bankroll_source"] == "runtime"
    assert response == {"ok": True, "status": {"ok": True}}
