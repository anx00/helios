import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.recorder import Recorder


def _fresh_recorder(tmp_path, monkeypatch, *, persist_channels="world,pws,features,nowcast,market,health,event_window"):
    monkeypatch.setenv("HELIOS_RECORDER_PERSIST_CHANNELS", persist_channels)
    monkeypatch.setenv("HELIOS_RECORDER_RING_SIZE", "100")
    monkeypatch.setenv("HELIOS_RECORDER_L2_RING_SIZE", "5")
    Recorder._instance = None
    return Recorder(base_path=str(tmp_path / "recordings"))


def test_recorder_excludes_l2_snap_from_persistence_by_default(tmp_path, monkeypatch):
    recorder = _fresh_recorder(tmp_path, monkeypatch)

    assert "l2_snap" not in recorder._writers
    assert "market" in recorder._writers
    assert "l2_snap" in recorder._ring_buffers


def test_record_l2_snap_compacts_depth_arrays(tmp_path, monkeypatch):
    recorder = _fresh_recorder(tmp_path, monkeypatch)

    asyncio.run(
        recorder.record_l2_snap(
            station_id="KATL",
            market_state={
                "__meta__": {"book_count": 2},
                "68-69 F": {
                    "yes_token_id": "yes1",
                    "best_bid": 0.31,
                    "best_ask": 0.33,
                    "bids": [[0.31, 120], [0.30, 90]],
                    "asks": [[0.33, 110], [0.34, 80]],
                    "yes_bids": [[0.31, 120]],
                    "yes_asks": [[0.33, 110]],
                    "no_bids": [[0.67, 75]],
                    "no_asks": [[0.69, 60]],
                },
            },
        )
    )

    recent = recorder.get_recent("l2_snap", 1)
    assert len(recent) == 1
    payload = recent[0]["data"]["68-69 F"]
    assert payload["yes_token_id"] == "yes1"
    assert payload["best_bid"] == 0.31
    assert "bids" not in payload
    assert "asks" not in payload
    assert "yes_bids" not in payload
    assert "no_asks" not in payload


def test_record_market_keeps_only_compact_summary(tmp_path, monkeypatch):
    recorder = _fresh_recorder(tmp_path, monkeypatch)

    asyncio.run(
        recorder.record_market(
            station_id="KATL",
            market_state={
                "__meta__": {
                    "book_count": 2,
                    "staleness_ms_avg": 120.0,
                    "publisher_lag_ms": 35.0,
                },
                "68-69 F": {
                    "token_id": "yes1",
                    "yes_token_id": "yes1",
                    "no_token_id": "no1",
                    "best_bid": 0.31,
                    "best_ask": 0.33,
                    "spread": 0.02,
                    "mid": 0.32,
                    "bid_depth": 120.0,
                    "ask_depth": 110.0,
                    "yes_best_bid": 0.31,
                    "yes_best_ask": 0.33,
                    "no_best_bid": 0.67,
                    "no_best_ask": 0.69,
                    "staleness_ms": 87.0,
                    "ts_provider": 123456789,
                    "bids": [[0.31, 120], [0.30, 90]],
                    "asks": [[0.33, 110], [0.34, 80]],
                },
            },
        )
    )

    recent = recorder.get_recent("market", 1)
    assert len(recent) == 1
    payload = recent[0]["data"]["68-69 F"]
    assert payload["best_bid"] == 0.31
    assert payload["no_best_ask"] == 0.69
    assert "token_id" not in payload
    assert "staleness_ms" not in payload
    assert "bids" not in payload
    assert recent[0]["data"]["__meta__"]["book_count"] == 2
