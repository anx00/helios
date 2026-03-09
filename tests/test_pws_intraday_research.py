from auditor.pws_intraday_research import (
    build_top_pws_by_market,
    build_recommendations,
    dedupe_predictions,
    summarize_prediction_duplicates,
)


def test_dedupe_predictions_keeps_most_complete_row():
    deduped, duplicates = dedupe_predictions(
        [
            {
                "station_id": "KLGA",
                "target_date": "2026-03-01",
                "timestamp": "2026-03-01T10:00:00+00:00",
                "final_prediction_f": "55.0",
                "physics_reason": "",
            },
            {
                "station_id": "KLGA",
                "target_date": "2026-03-01",
                "timestamp": "2026-03-01T10:00:00+00:00",
                "final_prediction_f": "55.0",
                "physics_reason": "more_complete",
                "verified_floor_f": "53.0",
            },
        ]
    )

    assert len(deduped) == 1
    assert deduped[0]["physics_reason"] == "more_complete"
    assert duplicates == [
        {
            "station_id": "KLGA",
            "target_date": "2026-03-01",
            "timestamp": "2026-03-01T10:00:00+00:00",
            "rows": 2,
        }
    ]


def test_build_recommendations_flags_predictions_dedupe_and_gate_tuning():
    recommendations = build_recommendations(
        cycle_summary={
            "tactical_allowed_market_precision": 0.48,
            "tactical_blocked_market_precision": 0.44,
            "tactical_allowed_rate": 0.41,
        },
        source_metrics=[
            {"source": "SYNOPTIC", "mae_c": 0.40, "hit_rate_1_0_c": 0.82},
            {"source": "OPEN_METEO", "mae_c": 0.75, "hit_rate_1_0_c": 0.61},
        ],
        lead_bucket_metrics=[
            {"lead_bucket": "05-15", "mae_c": 0.64, "hit_rate_1_0_c": 0.64},
            {"lead_bucket": "15-30", "mae_c": 0.49, "hit_rate_1_0_c": 0.72},
        ],
        profile_source_metrics=[
            {"source": "SYNOPTIC", "avg_weight_predictive": 0.9},
            {"source": "OPEN_METEO", "avg_weight_predictive": 1.0},
        ],
        duplicate_summary={"duplicate_key_count": 3, "duplicate_row_excess": 8, "max_rows_per_key": 4},
        market_rows=[{"pending_count": 10}],
    )

    titles = [row["title"] for row in recommendations]

    assert any("Retune source priors" in title for title in titles)
    assert any("Promote the 15-30 minute lead window" in title for title in titles)
    assert any("Tighten tactical repricing gates" in title for title in titles)
    assert any("Do not use predictions without explicit dedupe" in title for title in titles)


def test_summarize_prediction_duplicates_counts_excess_rows():
    summary = summarize_prediction_duplicates(
        [
            {"station_id": "KLGA", "target_date": "2026-03-01", "timestamp": "A", "rows": 2},
            {"station_id": "KATL", "target_date": "2026-03-01", "timestamp": "B", "rows": 5},
        ]
    )

    assert summary == {
        "duplicate_key_count": 2,
        "duplicate_row_excess": 5,
        "max_rows_per_key": 5,
    }


def test_build_top_pws_by_market_prioritizes_observed_performance():
    ranking = build_top_pws_by_market(
        [
            {
                "market_station_id": "KLGA",
                "pws_station_id": "PWS_BAD",
                "lead_row_count": 12,
                "lead_mae_c": 0.72,
                "lead_hit_rate_1_0_c": 0.70,
                "current_weight_predictive": 0.90,
                "current_rank_eligible": True,
                "observed_rank_score": 50.0,
            },
            {
                "market_station_id": "KLGA",
                "pws_station_id": "PWS_GOOD",
                "lead_row_count": 10,
                "lead_mae_c": 0.55,
                "lead_hit_rate_1_0_c": 0.82,
                "current_weight_predictive": 0.85,
                "current_rank_eligible": False,
                "observed_rank_score": 62.0,
            },
            {
                "market_station_id": "KLGA",
                "pws_station_id": "PWS_OK",
                "lead_row_count": 14,
                "lead_mae_c": 0.61,
                "lead_hit_rate_1_0_c": 0.78,
                "current_weight_predictive": 0.80,
                "current_rank_eligible": True,
                "observed_rank_score": 58.0,
            },
        ],
        top_n=3,
    )

    assert [row["pws_station_id"] for row in ranking] == ["PWS_GOOD", "PWS_OK", "PWS_BAD"]
