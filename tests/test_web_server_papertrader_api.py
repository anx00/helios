from types import SimpleNamespace

import pytest

import web_server


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
