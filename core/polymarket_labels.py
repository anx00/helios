"""
Utilities for Polymarket temperature bracket labels.

Canonical Fahrenheit formats:
- "33-34°F"
- "28°F or below"
- "39°F or higher"
"""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, List, Optional, Tuple

_DEGREE_F = "°F"
_DEGREE_C = "°C"
_RE_NUMBER = re.compile(r"-?\d+")
_RE_DECIMAL = re.compile(r"-?\d+(?:\.\d+)?")


def _clean_label(label: str) -> str:
    if label is None:
        return ""
    cleaned = str(label).strip()
    cleaned = cleaned.replace("Ã‚Â°", "°")
    cleaned = cleaned.replace("Â°F", "°F")
    cleaned = cleaned.replace("Â°C", "°C")
    cleaned = cleaned.replace("Â°", "°")
    cleaned = cleaned.replace("º", "°")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _extract_number(label: str) -> Optional[int]:
    match = _RE_NUMBER.search(_clean_label(label))
    if not match:
        return None
    try:
        return int(round(float(match.group(0))))
    except ValueError:
        return None


def _contains_celsius(label: str) -> bool:
    lower = _clean_label(label).lower()
    return ("°c" in lower) or bool(re.search(r"\bc\b", lower))


def _format_suffix(unit: str) -> str:
    return _DEGREE_C if str(unit).upper() == "C" else _DEGREE_F


def _format_range_label_for_unit(low: int, high: int, unit: str) -> str:
    return f"{int(low)}-{int(high)}{_format_suffix(unit)}"


def _format_below_label_for_unit(threshold: int, unit: str) -> str:
    return f"{int(threshold)}{_format_suffix(unit)} or below"


def _format_above_label_for_unit(threshold: int, unit: str) -> str:
    return f"{int(threshold)}{_format_suffix(unit)} or higher"


def _extract_range_bounds(label: str) -> Optional[Tuple[int, int]]:
    cleaned = _clean_label(label)
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", cleaned)
    if not match:
        return None
    try:
        low = int(round(float(match.group(1))))
        high = int(round(float(match.group(2))))
    except ValueError:
        return None
    return (low, high)


def _round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def format_range_label(low: int, high: int) -> str:
    return _format_range_label_for_unit(low, high, "F")


def format_below_label(threshold: int) -> str:
    return _format_below_label_for_unit(threshold, "F")


def format_above_label(threshold: int) -> str:
    return _format_above_label_for_unit(threshold, "F")


def normalize_label(label: str) -> str:
    """
    Normalize a bucket/bracket label to canonical Polymarket format.
    """
    if not label:
        return label

    s = _clean_label(label)
    lower = s.lower()
    unit = "C" if _contains_celsius(s) else "F"

    if lower.startswith("<"):
        num = _extract_number(s)
        return _format_below_label_for_unit(num, unit) if num is not None else s

    if lower.startswith(">") or lower.startswith("≥") or lower.startswith("â‰¥"):
        num = _extract_number(s)
        return _format_above_label_for_unit(num, unit) if num is not None else s

    if "or below" in lower:
        num = _extract_number(s)
        return _format_below_label_for_unit(num, unit) if num is not None else s

    if "or higher" in lower or "or above" in lower:
        num = _extract_number(s)
        return _format_above_label_for_unit(num, unit) if num is not None else s

    bounds = _extract_range_bounds(s)
    if bounds is not None:
        low, high = bounds
        return _format_range_label_for_unit(low, high, unit)

    num = _extract_number(s)
    if num is not None:
        return f"{num}{_format_suffix(unit)}"

    return s


def parse_label(label: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse a Polymarket label into (kind, low, high).

    kind: "below" | "above" | "range" | "single" | "unknown"
    low/high are integers when available.
    """
    if not label:
        return ("unknown", None, None)

    s = normalize_label(label)
    lower = s.lower()

    if "or below" in lower:
        num = _extract_number(s)
        return ("below", None, num)

    if "or higher" in lower or "or above" in lower:
        num = _extract_number(s)
        return ("above", num, None)

    bounds = _extract_range_bounds(s)
    if bounds is not None:
        low, high = bounds
        return ("range", low, high)

    num = _extract_number(s)
    if num is not None:
        return ("single", num, num)

    return ("unknown", None, None)


def sort_labels(labels: Iterable[str]) -> List[str]:
    """
    Sort Polymarket labels in ascending temperature order.
    """

    def sort_key(label: str) -> Tuple[int, int]:
        kind, low, high = parse_label(label)
        if kind == "below":
            return (0, high if high is not None else -999)
        if kind in ("range", "single"):
            return (1, low if low is not None else 0)
        if kind == "above":
            return (2, low if low is not None else 999)
        return (3, 0)

    return sorted([normalize_label(l) for l in labels], key=sort_key)


def label_for_temp(temp_f: float, labels: Iterable[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Return (label, index) that matches temp_f based on Polymarket semantics.
    """
    if temp_f is None:
        return (None, None)

    temp_int = _round_half_up(float(temp_f))
    ordered = sort_labels(labels)
    for idx, label in enumerate(ordered):
        kind, low, high = parse_label(label)
        if kind == "below" and high is not None:
            if temp_int <= high:
                return (label, idx)
        elif kind == "above" and low is not None:
            if temp_int >= low:
                return (label, idx)
        elif kind == "range" and low is not None and high is not None:
            if low <= temp_int <= high:
                return (label, idx)
        elif kind == "single" and low is not None:
            if temp_int == low:
                return (label, idx)

    return (None, None)


def normalize_p_bucket(p_bucket: List[dict]) -> List[dict]:
    """
    Normalize labels inside a p_bucket list (dicts).
    """
    if not p_bucket:
        return p_bucket

    normalized = []
    for entry in p_bucket:
        if not isinstance(entry, dict):
            normalized.append(entry)
            continue
        label = entry.get("label") or entry.get("bucket")
        if label:
            entry = dict(entry)
            entry["label"] = normalize_label(label)
            entry.pop("bucket", None)
        normalized.append(entry)
    return normalized


def normalize_market_snapshot(snapshot: dict) -> dict:
    """
    Normalize keys of a market snapshot to canonical Polymarket labels.
    Keeps meta keys intact.
    """
    if not snapshot:
        return snapshot

    normalized = {}
    for key, value in snapshot.items():
        if isinstance(key, str) and key.startswith("__"):
            normalized[key] = value
            continue
        norm_key = normalize_label(key) if isinstance(key, str) else key
        if norm_key not in normalized:
            normalized[norm_key] = value
    return normalized
