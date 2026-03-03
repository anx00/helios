import pytest

from collector.ensemble_fetcher import EnsembleSnapshot, _build_bucket_probabilities


def _make_snapshot(values, unit):
    return EnsembleSnapshot(
        station_id="TEST",
        target_date="2026-03-03",
        unit=unit,
        generated_utc="2026-03-03T00:00:00+00:00",
        total_members=len(values),
        member_tmax_values=list(values),
        ensemble_mean=0.0,
        ensemble_median=0.0,
        ensemble_std=0.0,
        ensemble_min=min(values),
        ensemble_max=max(values),
        bucket_probabilities=_build_bucket_probabilities(list(values), unit),
        models_used=["test"],
    )


def test_celsius_single_bucket_labels_and_lookup():
    snapshot = _make_snapshot([9.2, 9.6, 10.1, 10.7], "C")

    labels = [bucket.label for bucket in snapshot.bucket_probabilities]
    assert "9°C" in labels
    assert "10°C" in labels
    assert "11°C" in labels

    assert snapshot.get_bucket_prob("9 C") == pytest.approx(0.25, abs=1e-6)
    assert snapshot.get_bucket_prob("9-10 C") == pytest.approx(0.75, abs=1e-6)
    assert snapshot.get_bucket_prob("10 C or higher") == pytest.approx(0.75, abs=1e-6)


def test_fahrenheit_terminal_bucket_lookup():
    snapshot = _make_snapshot([-3.2, -2.6, -1.4, -0.6], "F")

    labels = [bucket.label for bucket in snapshot.bucket_probabilities]
    assert "26-27°F" in labels
    assert "28-29°F" in labels
    assert "30-31°F" in labels

    assert snapshot.get_bucket_prob("26 F or below") == pytest.approx(0.25, abs=1e-6)
    assert snapshot.get_bucket_prob("26-27 F") == pytest.approx(0.5, abs=1e-6)
    assert snapshot.get_bucket_prob("30 F or higher") == pytest.approx(0.25, abs=1e-6)
