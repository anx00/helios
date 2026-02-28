from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.telegram_bot import (
    _compute_nominal_next_metar,
    _compute_pws_weighted_consensus_c,
    _estimate_next_metar_c,
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
