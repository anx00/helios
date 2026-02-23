from __future__ import annotations

from typing import Optional


def normalize_metar_report_type(raw_type: Optional[str], raw_metar: Optional[str] = None) -> str:
    """
    Normalize report type to one of:
    - "SPECI"
    - "METAR"

    NOAA JSON usually exposes `metarType`; XML/TGFTP can be inferred from raw text.
    """
    candidates = [str(raw_type or "").strip().upper()]
    raw = str(raw_metar or "").strip().upper()
    if raw:
        # Raw strings commonly start with "METAR ..." or "SPECI ..."
        if raw.startswith("SPECI "):
            return "SPECI"
        if raw.startswith("METAR "):
            return "METAR"
        candidates.append(raw.split(" ", 1)[0])

    for c in candidates:
        if c == "SPECI":
            return "SPECI"
        if c == "METAR":
            return "METAR"

    return "METAR"


def is_speci_report(raw_type: Optional[str], raw_metar: Optional[str] = None) -> bool:
    return normalize_metar_report_type(raw_type, raw_metar) == "SPECI"

