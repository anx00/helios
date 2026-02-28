from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.trading_signal import build_trading_signal


UTC = ZoneInfo("UTC")


def _row_by_prefix(rows, prefix):
    for row in rows:
        if str(row.get("label") or "").startswith(prefix):
            return row
    raise KeyError(prefix)


def test_build_trading_signal_intraday_prefers_underpriced_terminal_bucket():
    signal = build_trading_signal(
        station_id="KLGA",
        target_day=0,
        target_date="2026-02-28",
        brackets=[
            {"name": "46-47 F", "yes_price": 0.08, "no_price": 0.92, "ws_yes_best_ask": 0.09, "ws_no_best_ask": 0.95},
            {"name": "48-49 F", "yes_price": 0.24, "no_price": 0.76, "ws_yes_best_ask": 0.25, "ws_no_best_ask": 0.80},
            {"name": "50-51 F", "yes_price": 0.52, "no_price": 0.48, "ws_yes_best_ask": 0.54, "ws_no_best_ask": 0.50},
            {"name": "52-53 F", "yes_price": 0.12, "no_price": 0.88, "ws_yes_best_ask": 0.13, "ws_no_best_ask": 0.91},
        ],
        nowcast_distribution={
            "tmax_mean_f": 49.0,
            "tmax_sigma_f": 1.0,
            "confidence": 0.82,
            "t_peak_expected_hour": 15,
        },
        nowcast_state={"max_so_far_raw_f": 47.2},
        official={"temp_c": 9.5, "obs_time_utc": "2026-02-28T15:00:00Z"},
        reference_utc=datetime(2026, 2, 28, 15, 20, tzinfo=UTC),
    )

    assert signal["available"] is True
    assert signal["model"]["source"] == "NOWCAST"
    assert signal["best_terminal_trade"]["label"].startswith("48-49")
    assert signal["best_terminal_trade"]["best_side"] == "YES"
    assert signal["best_terminal_trade"]["best_edge"] > 0.2
    assert signal["best_terminal_trade"]["selected_fair"] == signal["best_terminal_trade"]["fair_yes"]
    assert signal["best_terminal_trade"]["edge_points"] > 20.0
    assert signal["modules"]["final_market"]["top_bucket"] == signal["model"]["top_label"]


def test_build_trading_signal_day_ahead_falls_back_to_physics_prediction():
    signal = build_trading_signal(
        station_id="KLGA",
        target_day=1,
        target_date="2026-03-01",
        brackets=[
            {"name": "52-53 F", "yes_price": 0.19, "no_price": 0.81},
            {"name": "54-55 F", "yes_price": 0.26, "no_price": 0.74},
            {"name": "56-57 F", "yes_price": 0.31, "no_price": 0.69},
        ],
        prediction={
            "final_prediction_f": 55.0,
            "delta_weight": 0.6,
            "current_deviation_f": 0.8,
            "physics_adjustment_f": 0.5,
        },
        reference_utc=datetime(2026, 2, 28, 12, 0, tzinfo=UTC),
    )

    assert signal["available"] is True
    assert signal["horizon"] == "DAY_AHEAD"
    assert signal["model"]["source"] == "PHYSICS"
    assert signal["best_terminal_trade"]["label"].startswith("54-55")
    assert signal["best_terminal_trade"]["best_side"] == "YES"


def test_build_trading_signal_creates_tactical_pressure_from_predictive_pws():
    signal = build_trading_signal(
        station_id="EGLC",
        target_day=0,
        target_date="2026-02-28",
        brackets=[
            {"name": "9-10 C", "yes_price": 0.39, "no_price": 0.61, "ws_yes_best_ask": 0.40, "ws_no_best_ask": 0.66},
            {"name": "11-12 C", "yes_price": 0.17, "no_price": 0.83, "ws_yes_best_ask": 0.18, "ws_no_best_ask": 0.88},
            {"name": "13 C or higher", "yes_price": 0.03, "no_price": 0.97, "ws_yes_best_ask": 0.04, "ws_no_best_ask": 0.98},
        ],
        nowcast_distribution={
            "tmax_mean_f": 50.0,
            "tmax_sigma_f": 1.4,
            "confidence": 0.78,
            "t_peak_expected_hour": 14,
        },
        nowcast_state={"max_so_far_raw_f": 49.0},
        official={"temp_c": 10.0, "obs_time_utc": "2026-02-28T10:00:00Z"},
        pws_details=[
            {
                "valid": True,
                "station_id": "PWS1",
                "temp_c": 11.8,
                "source": "SYNOPTIC",
                "distance_km": 4.0,
                "age_minutes": 3.0,
                "obs_time_utc": "2026-02-28T10:22:00Z",
                "learning_weight": 2.0,
                "learning_weight_predictive": 2.0,
                "learning_now_score": 84,
                "learning_lead_score": 81,
                "learning_predictive_score": 83,
                "learning_now_samples": 12,
                "learning_lead_samples": 8,
            },
            {
                "valid": True,
                "station_id": "PWS2",
                "temp_c": 11.5,
                "source": "MADIS_APRSWXNET",
                "distance_km": 6.0,
                "age_minutes": 5.0,
                "obs_time_utc": "2026-02-28T10:20:00Z",
                "learning_weight": 1.7,
                "learning_weight_predictive": 1.7,
                "learning_now_score": 79,
                "learning_lead_score": 75,
                "learning_predictive_score": 77,
                "learning_now_samples": 10,
                "learning_lead_samples": 6,
            },
        ],
        pws_metrics={
            "learning": {
                "policy": {"lead_window_minutes": [5, 45]},
                "top_profiles": [
                    {"station_id": "PWS1", "next_metar_score": 83, "next_metar_bias_c": 0.1, "lead_score": 81, "now_score": 84},
                    {"station_id": "PWS2", "next_metar_score": 77, "next_metar_bias_c": 0.0, "lead_score": 75, "now_score": 79},
                ],
            }
        },
        reference_utc=datetime(2026, 2, 28, 10, 25, tzinfo=UTC),
    )

    assert signal["available"] is True
    assert signal["tactical_context"]["enabled"] is True
    assert signal["tactical_context"]["next_metar"]["direction"] == "UP"
    assert signal["tactical_context"]["next_metar"]["quality"] == signal["tactical_context"]["next_metar"]["confidence"]
    assert signal["best_tactical_trade"] is not None
    assert signal["best_tactical_trade"]["label"].startswith("11-12")
    assert signal["best_tactical_trade"]["tactical_best_side"] == "YES"


def test_build_trading_signal_respects_post_peak_ceiling_for_upside_buckets():
    signal = build_trading_signal(
        station_id="LFPG",
        target_day=0,
        target_date="2026-02-28",
        brackets=[
            {"name": "14 C", "yes_price": 0.50, "no_price": 0.50, "ws_yes_best_ask": 0.52, "ws_no_best_ask": 0.05},
            {"name": "15 C", "yes_price": 0.05, "no_price": 0.95, "ws_yes_best_ask": 0.05, "ws_no_best_ask": 0.97},
            {"name": "16 C or higher", "yes_price": 0.01, "no_price": 0.99, "ws_yes_best_ask": 0.01, "ws_no_best_ask": 0.995},
        ],
        nowcast_distribution={
            "tmax_mean_f": 57.0,
            "tmax_sigma_f": 2.2,
            "confidence": 0.8,
            "t_peak_expected_hour": 14,
        },
        nowcast_state={
            "max_so_far_raw_f": 57.2,
            "tmax_breakdown": {
                "peak_hour": 14,
                "current_hour": 16.0,
                "hours_to_peak": 0.0,
                "post_peak_cap_f": 59.0,
                "remaining_max_f": 59.0,
            },
        },
        official={"temp_c": 14.0, "obs_time_utc": "2026-02-28T15:00:00Z"},
        reference_utc=datetime(2026, 2, 28, 15, 20, tzinfo=UTC),
    )

    rows = signal["all_terminal_opportunities"]
    row_16 = _row_by_prefix(rows, "16")

    assert signal["model"]["ceiling"] == 15
    assert row_16["fair_yes"] == 0.0
    assert row_16["best_side"] == "NO"
    assert row_16["terminal_policy"]["allowed"] is False


def test_build_trading_signal_blocks_late_upside_chase_after_peak():
    signal = build_trading_signal(
        station_id="LFPG",
        target_day=0,
        target_date="2026-02-28",
        brackets=[
            {"name": "14 C", "yes_price": 0.52, "no_price": 0.48, "ws_yes_best_ask": 0.53, "ws_no_best_ask": 0.05},
            {"name": "15 C", "yes_price": 0.05, "no_price": 0.95, "ws_yes_best_ask": 0.05, "ws_no_best_ask": 0.98},
            {"name": "16 C or higher", "yes_price": 0.01, "no_price": 0.99, "ws_yes_best_ask": 0.01, "ws_no_best_ask": 0.995},
        ],
        nowcast_distribution={
            "tmax_mean_f": 57.0,
            "tmax_sigma_f": 1.8,
            "confidence": 0.84,
            "t_peak_expected_hour": 14,
        },
        nowcast_state={
            "max_so_far_raw_f": 57.2,
            "tmax_breakdown": {
                "peak_hour": 14,
                "current_hour": 16.2,
                "hours_to_peak": 0.0,
                "post_peak_cap_f": 59.0,
                "remaining_max_f": 59.0,
            },
        },
        official={"temp_c": 14.0, "obs_time_utc": "2026-02-28T15:00:00Z"},
        reference_utc=datetime(2026, 2, 28, 15, 20, tzinfo=UTC),
    )

    row_15 = _row_by_prefix(signal["all_terminal_opportunities"], "15")

    assert all(not str(row.get("label") or "").startswith("15") for row in signal["terminal_opportunities"])
    assert row_15["terminal_policy"]["allowed"] is False
    assert "late_upside_chase" in row_15["terminal_policy"]["reasons"]
    assert signal["best_terminal_trade"]["label"].startswith("14")
    assert signal["best_terminal_trade"]["best_side"] == "NO"
