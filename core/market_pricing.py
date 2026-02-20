"""
Shared market probability normalization and quote selection helpers.

Goal:
- Keep probability parsing consistent across replay/live paths.
- Avoid overstating probabilities in wide-spread books by defaulting to
  executable prices (best bid) when midpoint is unreliable.
"""

from __future__ import annotations

import math
from typing import Any, Optional


def normalize_probability_pct(value: Any) -> Optional[float]:
    """
    Normalize a probability-like value into [0, 100] percent.

    Accepted inputs:
    - 0..1 (fraction) -> scaled to 0..100
    - 0..100 (already percent)
    """
    try:
        n = float(value)
    except Exception:
        return None
    if not math.isfinite(n) or n < 0:
        return None
    if n <= 1.000001:
        return round(n * 100.0, 4)
    if n <= 100.000001:
        return round(n, 4)
    return None


def normalize_probability_01(value: Any) -> Optional[float]:
    """Normalize probability-like value into [0, 1]."""
    pct = normalize_probability_pct(value)
    if pct is None:
        return None
    return round(pct / 100.0, 6)


def select_probability_pct_from_quotes(
    best_bid_pct: Optional[float],
    best_ask_pct: Optional[float],
    mid_pct: Optional[float] = None,
    reference_pct: Optional[float] = None,
    wide_spread_threshold_pct: float = 4.0,
    reference_tolerance_pct: float = 0.5,
) -> Optional[float]:
    """
    Select a robust probability (percent) from quotes.

    Policy:
    - Tight spread: midpoint is acceptable.
    - Wide spread: default to executable best bid (conservative, tradable).
    - If a reference price exists and sits inside the quote band, prefer it.
    """
    bid = best_bid_pct if best_bid_pct is not None else None
    ask = best_ask_pct if best_ask_pct is not None else None
    mid = mid_pct if mid_pct is not None else None
    ref = reference_pct if reference_pct is not None else None

    if bid is not None and ask is not None:
        lo = min(bid, ask)
        hi = max(bid, ask)
        spread = hi - lo
        if ref is not None and (lo - reference_tolerance_pct) <= ref <= (hi + reference_tolerance_pct):
            return round(ref, 4)
        if spread <= max(0.1, float(wide_spread_threshold_pct)):
            if mid is not None:
                if (lo - reference_tolerance_pct) <= mid <= (hi + reference_tolerance_pct):
                    return round(mid, 4)
            return round((bid + ask) / 2.0, 4)
        return round(bid, 4)

    if bid is not None:
        return round(bid, 4)

    if ask is not None:
        if ref is not None and ref <= (ask + reference_tolerance_pct):
            return round(ref, 4)
        return round(ask, 4)

    if mid is not None:
        return round(mid, 4)

    if ref is not None:
        return round(ref, 4)

    return None


def select_probability_01_from_quotes(
    best_bid: Any,
    best_ask: Any,
    mid: Any = None,
    reference: Any = None,
    wide_spread_threshold: float = 0.04,
    reference_tolerance: float = 0.005,
) -> Optional[float]:
    """
    Select robust probability in [0, 1] directly from raw quote-like values.
    """
    bid_pct = normalize_probability_pct(best_bid)
    ask_pct = normalize_probability_pct(best_ask)
    mid_pct = normalize_probability_pct(mid)
    ref_pct = normalize_probability_pct(reference)

    selected_pct = select_probability_pct_from_quotes(
        best_bid_pct=bid_pct,
        best_ask_pct=ask_pct,
        mid_pct=mid_pct,
        reference_pct=ref_pct,
        wide_spread_threshold_pct=max(0.1, float(wide_spread_threshold) * 100.0),
        reference_tolerance_pct=max(0.05, float(reference_tolerance) * 100.0),
    )
    if selected_pct is None:
        return None
    return round(selected_pct / 100.0, 6)
