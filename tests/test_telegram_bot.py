import asyncio
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collector.metar_fetcher import MetarData
import core.telegram_bot as telegram_bot
from core.telegram_bot import (
    HeliosTelegramBot,
    TelegramBotSettings,
    _compute_nominal_next_metar,
    _compute_pws_weighted_consensus_c,
    _estimate_next_metar_c,
    _fmt_station_temp_from_c,
)


UTC = ZoneInfo("UTC")


def test_weighted_consensus_prefers_heaviest_station():
    rows = [
        {"valid": True, "temp_c": 10.0, "learning_weight": 1.0},
        {"valid": True, "temp_c": 12.0, "learning_weight": 4.0},
        {"valid": True, "temp_c": 14.0, "learning_weight": 1.0},
    ]

    assert _compute_pws_weighted_consensus_c(rows) == 12.0


def test_nominal_next_metar_preserves_report_minute():
    official_obs = datetime(2026, 2, 28, 10, 51, tzinfo=UTC)
    reference = datetime(2026, 2, 28, 11, 5, tzinfo=UTC)

    result = _compute_nominal_next_metar(official_obs, reference_utc=reference)

    assert result["schedule_minute"] == 51
    assert result["next_obs"] == datetime(2026, 2, 28, 11, 51, tzinfo=UTC)
    assert round(result["minutes_to_next"], 1) == 46.0


def test_estimated_next_metar_uses_bias_corrected_projection():
    rows = [
        {
            "valid": True,
            "station_id": "PWS1",
            "temp_c": 12.0,
            "source": "SYNOPTIC",
            "distance_km": 5.0,
            "age_minutes": 5.0,
            "obs_time_utc": "2026-02-28T10:15:00Z",
            "learning_weight": 2.0,
            "learning_weight_predictive": 2.0,
            "learning_now_score": 90,
            "learning_lead_score": 70,
            "learning_predictive_score": 80,
            "learning_now_samples": 10,
            "learning_lead_samples": 8,
        }
    ]
    learning_metric = {
        "policy": {"lead_window_minutes": [5, 45]},
        "top_profiles": [
            {
                "station_id": "PWS1",
                "next_metar_score": 80,
                "next_metar_bias_c": 1.0,
            }
        ],
    }

    estimated_c = _estimate_next_metar_c(
        rows=rows,
        official_temp_c=11.0,
        official_obs_utc=datetime(2026, 2, 28, 10, 0, tzinfo=UTC),
        learning_metric=learning_metric,
        consensus_c=12.0,
        reference_utc=datetime(2026, 2, 28, 10, 20, tzinfo=UTC),
    )

    assert estimated_c == pytest.approx(11.18152, rel=1e-4)


def _metar_row(temp_c: float, obs_time: datetime) -> MetarData:
    return MetarData(
        station_id="KLGA",
        observation_time=obs_time,
        temp_c=temp_c,
        dewpoint_c=None,
        humidity_pct=0.0,
        wind_dir_degrees=None,
        wind_speed_kt=None,
        sky_condition="CLR",
        sky_conditions_list=["CLR"],
        visibility_miles=None,
        altimeter_hg=None,
        wind_trend="Stable",
        raw_metar="METAR",
        report_type="METAR",
    )


def test_cmd_metar_includes_last_five_history_rows(monkeypatch):
    current = _metar_row(10.0, datetime(2026, 2, 28, 10, 51, tzinfo=UTC))
    history = [
        _metar_row(10.0, datetime(2026, 2, 28, 10, 51, tzinfo=UTC)),
        _metar_row(9.0, datetime(2026, 2, 28, 9, 51, tzinfo=UTC)),
        _metar_row(8.0, datetime(2026, 2, 28, 8, 51, tzinfo=UTC)),
        _metar_row(7.0, datetime(2026, 2, 28, 7, 51, tzinfo=UTC)),
        _metar_row(6.0, datetime(2026, 2, 28, 6, 51, tzinfo=UTC)),
        _metar_row(5.0, datetime(2026, 2, 28, 5, 51, tzinfo=UTC)),
    ]

    async def fake_fetch_metar_race(_station_id: str):
        return current

    async def fake_fetch_metar_history(_station_id: str, hours: int = 12):
        assert hours == 12
        return history

    sent_messages = []

    async def fake_send_text(chat_id: str, text: str, reply_markup=None):
        sent_messages.append((chat_id, text, reply_markup))

    monkeypatch.setattr(telegram_bot, "fetch_metar_race", fake_fetch_metar_race)
    monkeypatch.setattr(telegram_bot, "fetch_metar_history", fake_fetch_metar_history)

    bot = HeliosTelegramBot(TelegramBotSettings(token="x", allowed_chat_ids=None))
    bot._chat_station["chat"] = "KLGA"
    monkeypatch.setattr(bot, "_send_text", fake_send_text)

    asyncio.run(bot._cmd_metar("chat", []))

    assert len(sent_messages) == 1
    text = sent_messages[0][1]
    expected_current = _fmt_station_temp_from_c(10.0, "KLGA", 1)
    expected_last = _fmt_station_temp_from_c(6.0, "KLGA", 1)

    assert f"METAR KLGA: {expected_current}" in text
    assert "Ultimos 5:" in text
    assert "2026-02-28 05:51" in text
    assert "2026-02-28 01:51" in text
    assert expected_last in text
    assert "2026-02-28 00:51" not in text


def test_cmd_trade_formats_compact_signal(monkeypatch):
    sent_messages = []

    async def fake_send_text(chat_id: str, text: str, reply_markup=None):
        sent_messages.append((chat_id, text, reply_markup))

    async def fake_load_trading_signal(self, station_id: str, target_day: int):
        assert station_id == "KLGA"
        assert target_day == 0
        return {
            "available": True,
            "model": {
                "mean": 49.2,
                "sigma": 1.1,
                "confidence": 0.73,
                "top_label": "48-49°F",
                "top_label_probability": 0.34,
            },
            "best_terminal_trade": {
                "label": "48-49°F",
                "best_side": "YES",
                "yes_entry": 0.27,
                "fair_yes": 0.34,
                "best_edge": 0.07,
            },
            "best_tactical_trade": {
                "label": "50-51°F",
                "best_side": "YES",
                "yes_entry": 0.18,
                "tactical_yes": 0.24,
                "tactical_best_edge": 0.06,
            },
            "tactical_context": {
                "enabled": True,
                "next_metar": {
                    "available": True,
                    "expected_market": 50.0,
                    "direction": "UP",
                    "minutes_to_next": 18.0,
                    "confidence": 0.68,
                }
            },
        }

    bot = HeliosTelegramBot(TelegramBotSettings(token="x", allowed_chat_ids=None))
    bot._chat_station["chat"] = "KLGA"
    monkeypatch.setattr(bot, "_send_text", fake_send_text)
    monkeypatch.setattr(HeliosTelegramBot, "_load_trading_signal", fake_load_trading_signal)

    asyncio.run(bot._cmd_trade("chat", []))

    assert len(sent_messages) == 1
    text = sent_messages[0][1]
    assert "Trade KLGA d0" in text
    assert "Model: 49.2F sigma 1.1 conf 73%" in text
    assert "Top bucket: 48-49°F 34%" in text
    assert "Final: YES 48-49°F @ 27% fair 34% edge 7%" in text
    assert "Next METAR: 50.0F up 18m conf 68%" in text
    assert "Tactical: YES 50-51°F @ 18% fair 24% edge 6%" in text
