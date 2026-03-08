import pytest

from core.probability_model import (
    build_dayahead_terminal_model,
    build_probability_lab_card,
    build_probability_lab_station_detail,
)


def test_build_dayahead_terminal_model_renormalizes_weights_when_source_missing():
    model = build_dayahead_terminal_model(
        station_id="KLGA",
        target_day=1,
        source_snapshots=[
            {"source": "WUNDERGROUND", "status": "ok", "forecast_high_market": 50.0, "peak_hour_local": 15},
            {"source": "NBM", "status": "ok", "forecast_high_market": 52.0, "peak_hour_local": 16},
            {"source": "OPEN_METEO", "status": "ok", "forecast_high_market": 49.0, "peak_hour_local": 14},
            {"source": "LAMP", "status": "error", "forecast_high_market": None, "peak_hour_local": None},
        ],
    )

    assert model is not None
    expected_mean = ((0.55 * 50.0) + (0.20 * 52.0) + (0.15 * 49.0)) / 0.90
    assert model["mean_market"] == pytest.approx(expected_mean, rel=1e-4)
    assert model["source_count"] == 3
    assert model["sigma_market"] >= 1.4
    assert "missing_sources:LAMP" in model["notes"]


def test_probability_lab_card_does_not_highlight_small_tail_edge():
    signal_payload = {
        "model": {
            "source": "DAYAHEAD_FUSION",
            "mean": 48.9,
            "sigma": 1.6,
            "confidence": 0.79,
            "top_label": "48-49 F",
            "top_label_probability": 0.34,
        },
        "best_terminal_trade": {
            "label": "53-54 F",
            "best_side": "YES",
            "best_edge": 0.01,
            "selected_fair": 0.05,
            "selected_entry": 0.04,
            "recommendation": "WATCH_YES",
            "terminal_policy": {"allowed": True, "summary": "Probability is too small for board action.", "reasons": []},
            "policy_reason": "Probability is too small for board action.",
        },
        "tactical_context": {"enabled": False},
    }

    card = build_probability_lab_card(
        station_id="KLGA",
        station_name="New York LaGuardia",
        target_day=1,
        target_date="2026-03-05",
        market_payload={"event_title": "Test", "total_volume": 1000.0, "market_status": {}, "brackets": []},
        signal_payload=signal_payload,
        terminal_model={"source_count": 4, "source_spread_market": 1.1, "notes": []},
        source_snapshots=[],
        reality={},
        pws_profiles=[],
    )

    assert card["actionable"]["available"] is False
    assert card["actionable"]["recommendation"] == "WATCH"


def test_build_dayahead_terminal_model_uses_calibration_to_shift_weights():
    snapshots = [
        {"source": "WUNDERGROUND", "status": "ok", "forecast_high_market": 50.0, "peak_hour_local": 15},
        {"source": "NBM", "status": "ok", "forecast_high_market": 56.0, "peak_hour_local": 16},
    ]
    calibration = {
        "available": True,
        "warming_up": False,
        "samples": 4,
        "mae_market": 1.1,
        "bias_market": 0.2,
        "sources": {
            "WUNDERGROUND": {"samples": 4, "mae_market": 0.5, "bias_market": 0.1},
            "NBM": {"samples": 4, "mae_market": 2.2, "bias_market": 1.0},
        },
    }

    plain_model = build_dayahead_terminal_model(
        station_id="KLGA",
        target_day=1,
        source_snapshots=snapshots,
    )
    calibrated_model = build_dayahead_terminal_model(
        station_id="KLGA",
        target_day=1,
        source_snapshots=snapshots,
        calibration=calibration,
    )

    assert plain_model is not None
    assert calibrated_model is not None
    assert calibrated_model["mean_market"] < plain_model["mean_market"]
    assert "calibration_weighting:WUNDERGROUND,NBM" in calibrated_model["notes"]
    assert calibrated_model["confidence"] < plain_model["confidence"]
    assert calibrated_model["confidence_multiplier"] == pytest.approx(0.9588, rel=1e-3)
    assert "calibration_confidence:0.959" in calibrated_model["notes"]
    contributors = {row["source"]: row for row in calibrated_model["contributors"]}
    assert contributors["WUNDERGROUND"]["calibration_multiplier"] == pytest.approx(1.15, rel=1e-3)
    assert contributors["NBM"]["calibration_multiplier"] == pytest.approx(0.754, rel=1e-3)
    assert contributors["WUNDERGROUND"]["weight"] > contributors["NBM"]["weight"]


def test_probability_lab_card_exposes_calibration_summary():
    signal_payload = {
        "model": {
            "source": "DAYAHEAD_FUSION",
            "mean": 48.9,
            "sigma": 1.6,
            "confidence": 0.79,
            "top_label": "48-49 F",
            "top_label_probability": 0.34,
        },
        "best_terminal_trade": {
            "label": "48-49 F",
            "best_side": "YES",
            "best_edge": 0.05,
            "selected_fair": 0.33,
            "selected_entry": 0.28,
            "recommendation": "LEAN_YES",
            "terminal_policy": {"allowed": True, "summary": "Actionable edge.", "reasons": []},
            "policy_reason": "Actionable edge.",
        },
        "tactical_context": {"enabled": False},
    }
    calibration = {
        "available": True,
        "warming_up": False,
        "samples": 5,
        "mae_market": 1.2,
        "bias_market": -0.3,
        "sources": {
            "WUNDERGROUND": {"samples": 5, "mae_market": 0.6, "bias_market": -0.1},
            "NBM": {"samples": 5, "mae_market": 1.8, "bias_market": 0.7},
        },
    }
    terminal_model = {
        "source_count": 2,
        "source_spread_market": 1.4,
        "notes": ["calibration_weighting:WUNDERGROUND,NBM"],
        "contributors": [
            {"source": "WUNDERGROUND", "calibration_multiplier": 1.14},
            {"source": "NBM", "calibration_multiplier": 0.86},
        ],
    }

    card = build_probability_lab_card(
        station_id="KLGA",
        station_name="New York LaGuardia",
        target_day=1,
        target_date="2026-03-05",
        market_payload={"event_title": "Test", "total_volume": 1000.0, "market_status": {}, "brackets": []},
        signal_payload=signal_payload,
        terminal_model=terminal_model,
        source_snapshots=[],
        reality={},
        pws_profiles=[],
        calibration=calibration,
    )

    assert card["model"]["calibration_active"] is True
    assert card["model"]["calibration_warming_up"] is False
    assert card["model"]["calibration_samples"] == 5
    assert card["model"]["calibration_mae_market"] == pytest.approx(1.2)
    assert card["model"]["calibration_bias_market"] == pytest.approx(-0.3)


def test_probability_lab_station_detail_prefers_tactical_fields_for_intraday_ladder():
    detail = build_probability_lab_station_detail(
        station_id="LFPG",
        target_day=0,
        target_date="2026-03-08",
        market_payload={"event_title": "Test", "total_volume": 1000.0, "market_status": {}, "brackets": []},
        signal_payload={
            "model": {
                "source": "NOWCAST",
                "mean": 14.2,
                "sigma": 0.9,
                "confidence": 0.88,
                "top_label": "15°C",
                "top_label_probability": 0.52,
            },
            "best_terminal_trade": {
                "label": "15°C",
                "best_side": "YES",
                "best_edge": 0.02,
                "selected_fair": 0.52,
                "selected_entry": 0.50,
                "recommendation": "WATCH_YES",
                "terminal_policy": {"allowed": True, "summary": "Terminal view only.", "reasons": []},
                "policy_reason": "Terminal view only.",
            },
            "best_tactical_trade": {
                "label": "15°C",
                "tactical_best_side": "NO",
                "tactical_best_edge": 0.08,
                "selected_tactical_fair": 0.66,
                "selected_tactical_entry": 0.58,
                "tactical_recommendation": "BUY_NO",
                "tactical_policy": {"allowed": True, "summary": "Tactical repricing says fade 15°C.", "reasons": []},
                "tactical_policy_reason": "Tactical repricing says fade 15°C.",
            },
            "all_terminal_opportunities": [{
                "label": "15°C",
                "fair_yes": 0.52,
                "fair_no": 0.48,
                "market_yes": 0.50,
                "market_no": 0.50,
                "best_side": "YES",
                "selected_fair": 0.52,
                "selected_entry": 0.50,
                "best_edge": 0.02,
                "edge_points": 2.0,
                "recommendation": "WATCH_YES",
                "policy_reason": "Terminal view only.",
                "tactical_best_side": "NO",
                "selected_tactical_fair": 0.66,
                "selected_tactical_entry": 0.58,
                "tactical_best_edge": 0.08,
                "tactical_edge_points": 8.0,
                "tactical_recommendation": "BUY_NO",
                "tactical_policy_reason": "Tactical repricing says fade 15°C.",
            }],
            "tactical_context": {"enabled": True},
        },
        terminal_model=None,
        source_snapshots=[],
        source_history=[],
        reality={},
        pws_profiles=[],
        calibration=None,
    )

    row = detail["bracket_ladder"][0]
    assert row["recommendation"] == "WATCH_YES"
    assert row["active_recommendation"] == "BUY_NO"
    assert row["active_selected_side"] == "NO"
    assert row["active_policy_reason"] == "Tactical repricing says fade 15°C."
