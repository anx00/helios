import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.market_pricing import (
    normalize_probability_pct,
    normalize_probability_01,
    select_probability_01_from_quotes,
    select_probability_pct_from_quotes,
)


def test_normalize_probability_pct_from_fraction_and_percent():
    assert normalize_probability_pct(0.38) == 38.0
    assert normalize_probability_pct(38) == 38.0
    assert normalize_probability_pct("0.425") == 42.5
    assert normalize_probability_pct(-1) is None
    assert normalize_probability_pct(140) is None


def test_select_probability_pct_uses_mid_on_tight_spread():
    selected = select_probability_pct_from_quotes(
        best_bid_pct=38.0,
        best_ask_pct=41.0,
        mid_pct=39.5,
        reference_pct=None,
        wide_spread_threshold_pct=4.0,
    )
    assert selected == 39.5


def test_select_probability_pct_uses_bid_on_wide_spread():
    selected = select_probability_pct_from_quotes(
        best_bid_pct=38.0,
        best_ask_pct=49.0,
        mid_pct=43.5,
        reference_pct=None,
        wide_spread_threshold_pct=4.0,
    )
    assert selected == 38.0


def test_select_probability_pct_prefers_reference_inside_band():
    selected = select_probability_pct_from_quotes(
        best_bid_pct=38.0,
        best_ask_pct=49.0,
        mid_pct=43.5,
        reference_pct=39.0,
        wide_spread_threshold_pct=4.0,
    )
    assert selected == 39.0


def test_select_probability_01_from_quotes():
    selected = select_probability_01_from_quotes(
        best_bid=0.38,
        best_ask=0.49,
        mid=0.435,
        reference=None,
        wide_spread_threshold=0.04,
    )
    assert selected == 0.38
    assert normalize_probability_01(38) == 0.38
