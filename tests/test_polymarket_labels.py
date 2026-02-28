import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.polymarket_labels import (
    normalize_label,
    parse_label,
    label_for_temp,
    sort_labels,
    normalize_p_bucket,
    normalize_market_snapshot,
)


def test_normalize_label_range_and_edges():
    deg = "\u00b0"
    assert normalize_label("33-34") == f"33-34{deg}F"
    assert normalize_label(f"33-34{deg}F") == f"33-34{deg}F"
    assert normalize_label("28 F or below") == f"28{deg}F or below"
    assert normalize_label("39°F or higher") == f"39{deg}F or higher"
    assert normalize_label("<18") == f"18{deg}F or below"
    assert normalize_label(">=40") == f"40{deg}F or higher"


def test_parse_label():
    deg = "\u00b0"
    assert parse_label(f"33-34{deg}F") == ("range", 33, 34)
    assert parse_label(f"28{deg}F or below") == ("below", None, 28)
    assert parse_label(f"39{deg}F or higher") == ("above", 39, None)
    assert parse_label("40°F") == ("single", 40, 40)


def test_celsius_labels_are_supported():
    deg = "\u00b0"
    assert normalize_label(f"9-10{deg}C") == f"9-10{deg}C"
    assert parse_label(f"11{deg}C or higher") == ("above", 11, None)
    assert parse_label(f"8{deg}C or below") == ("below", None, 8)


def test_label_for_temp():
    deg = "\u00b0"
    labels = [
        f"28{deg}F or below",
        f"29-30{deg}F",
        f"31-32{deg}F",
        f"33{deg}F or higher",
    ]
    label, idx = label_for_temp(28.1, labels)
    assert label == f"28{deg}F or below"
    assert idx == 0

    label, idx = label_for_temp(31.2, labels)
    assert label == f"31-32{deg}F"
    assert idx == 2

    label, idx = label_for_temp(35.0, labels)
    assert label == f"33{deg}F or higher"
    assert idx == 3


def test_sort_labels():
    deg = "\u00b0"
    labels = [
        f"40{deg}F or higher",
        f"29-30{deg}F",
        f"28{deg}F or below",
        f"31-32{deg}F",
    ]
    sorted_labels = sort_labels(labels)
    assert sorted_labels == [
        f"28{deg}F or below",
        f"29-30{deg}F",
        f"31-32{deg}F",
        f"40{deg}F or higher",
    ]


def test_normalize_p_bucket_and_market_snapshot():
    deg = "\u00b0"
    p_bucket = [
        {"label": "20-21", "probability": 0.6},
        {"label": f"21-22{deg}F", "probability": 0.4},
    ]
    normalized = normalize_p_bucket(p_bucket)
    assert normalized[0]["label"] == f"20-21{deg}F"
    assert normalized[1]["label"] == f"21-22{deg}F"

    snapshot = {
        "20-21": {"best_bid": 0.4},
        f"21-22{deg}F": {"best_bid": 0.3},
        "__meta__": {"staleness_ms": 10},
    }
    normalized_snapshot = normalize_market_snapshot(snapshot)
    assert f"20-21{deg}F" in normalized_snapshot
    assert f"21-22{deg}F" in normalized_snapshot
    assert "__meta__" in normalized_snapshot
