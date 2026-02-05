"""
Utilities for Polymarket temperature bracket labels.

Canonical formats:
- "33-34°F"
- "28°F or below"
- "39°F or higher"
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

_DEGREE_F = "°F"
_RE_NUMBER = re.compile(r"-?\d+")


def _clean_label(label: str) -> str:
    if label is None:
        return ""
    # Normalize degree symbol artifacts and whitespace
    cleaned = str(label).strip()
    cleaned = cleaned.replace("Â°", "°")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _extract_number(label: str) -> Optional[int]:
    match = _RE_NUMBER.search(label)
    if not match:
        return None
    try:
        return int(round(float(match.group(0))))
    except ValueError:
        return None


def format_range_label(low: int, high: int) -> str:
    return f"{int(low)}-{int(high)}{_DEGREE_F}"


def format_below_label(threshold: int) -> str:
    return f"{int(threshold)}{_DEGREE_F} or below"


def format_above_label(threshold: int) -> str:
    return f"{int(threshold)}{_DEGREE_F} or higher"


def normalize_label(label: str) -> str:
    """
    Normalize a bucket/bracket label to canonical Polymarket format.
    """
    if not label:
        return label

    s = _clean_label(label)
    lower = s.lower()

    if lower.startswith("<"):
        num = _extract_number(s)
        return format_below_label(num) if num is not None else s

    if lower.startswith(">") or lower.startswith("≥"):
        num = _extract_number(s)
        return format_above_label(num) if num is not None else s

    if "or below" in lower:
        num = _extract_number(s)
        return format_below_label(num) if num is not None else s

    if "or higher" in lower or "or above" in lower:
        num = _extract_number(s)
        return format_above_label(num) if num is not None else s

    # Range format "XX-YY°F"
    if "-" in s:
        clean = s.replace("°F", "").replace("°", "")
        parts = clean.split("-")
        if len(parts) == 2:
            try:
                low = int(round(float(parts[0].strip())))
                high = int(round(float(parts[1].strip())))
                return format_range_label(low, high)
            except ValueError:
                return s

    # Single value "48°F" (keep as-is but normalized)
    num = _extract_number(s)
    if num is not None:
        if "°" in s or "f" in lower:
            return f"{num}{_DEGREE_F}"
        return f"{num}{_DEGREE_F}"

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

    if "-" in s:
        clean = s.replace("°F", "").replace("°", "")
        parts = clean.split("-")
        if len(parts) == 2:
            try:
                low = int(round(float(parts[0].strip())))
                high = int(round(float(parts[1].strip())))
                return ("range", low, high)
            except ValueError:
                return ("unknown", None, None)

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

    temp_int = int(round(temp_f))
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
        if isinstance(key, str):
            norm_key = normalize_label(key)
        else:
            norm_key = key
        # If collision occurs, prefer the first seen
        if norm_key not in normalized:
            normalized[norm_key] = value
    return normalized
