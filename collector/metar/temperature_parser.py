"""
Utilities to decode METAR temperature from raw text.

Key rule:
- Prefer T-group precision when present (e.g. T00220006 -> 2.2C / 0.6C).
- If T-group is absent, use the main temp/dewpoint group (e.g. 02/01) and
  expose an estimated range instead of fake exact precision.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import re
from typing import Optional, Dict, Any


_T_GROUP_RE = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
_MAIN_TEMP_RE = re.compile(r"\b(M?\d{2})/(M?\d{2}|//)\b")


def _c_to_f(temp_c: Optional[float]) -> Optional[float]:
    if temp_c is None:
        return None
    return round((temp_c * 9.0 / 5.0) + 32.0, 1)


def _round_half_up(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    d = Decimal(str(value))
    return int(d.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _decode_metar_signed(token: str) -> float:
    token = token.strip().upper()
    if token.startswith("M"):
        return -float(token[1:])
    return float(token)


def _decode_t_group(sign_bit: str, value_3d: str) -> float:
    sign = -1.0 if sign_bit == "1" else 1.0
    return sign * (int(value_3d) / 10.0)


def _without_remarks(raw_metar: str) -> str:
    # Main body before RMK to avoid parsing remark groups as temps.
    parts = raw_metar.split(" RMK ", 1)
    return parts[0]


def decode_temperature_from_raw(
    raw_metar: str,
    fallback_temp_c: Optional[float] = None,
    fallback_dewp_c: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Decode temperature from raw METAR.

    Returns dict with:
    - temp_c, dewp_c
    - temp_c_low/high, temp_f_low/high
    - settlement_f_low/high
    - has_t_group, used_main_group
    """
    raw = (raw_metar or "").strip()
    temp_c: Optional[float] = None
    dewp_c: Optional[float] = None
    has_t_group = False
    used_main_group = False
    used_fallback = False

    # 1) Exact precision from T-group, if present.
    if raw:
        t_match = _T_GROUP_RE.search(raw)
        if t_match:
            has_t_group = True
            temp_c = _decode_t_group(t_match.group(1), t_match.group(2))
            dewp_c = _decode_t_group(t_match.group(3), t_match.group(4))

    # 2) Fallback to main temp group (integer C; no extra precision).
    if temp_c is None and raw:
        main_body = _without_remarks(raw)
        mt = _MAIN_TEMP_RE.search(main_body)
        if mt:
            used_main_group = True
            temp_c = _decode_metar_signed(mt.group(1))
            dewp_token = mt.group(2)
            if dewp_token != "//":
                dewp_c = _decode_metar_signed(dewp_token)

    # 3) Last fallback: decoded endpoint fields.
    if temp_c is None:
        temp_c = fallback_temp_c
        if temp_c is not None:
            used_fallback = True
    if dewp_c is None:
        dewp_c = fallback_dewp_c

    # Build ranges.
    if temp_c is None:
        temp_c_low = None
        temp_c_high = None
    elif has_t_group or used_fallback:
        temp_c_low = temp_c
        temp_c_high = temp_c
    else:
        # No T-group: METAR main temp is rounded to integer C.
        # Use nearest-tenth plausible interval (n-0.5 to n+0.4).
        rounded_c = int(temp_c)
        temp_c_low = rounded_c - 0.5
        temp_c_high = rounded_c + 0.4

    temp_f = _c_to_f(temp_c)
    temp_f_low = _c_to_f(temp_c_low)
    temp_f_high = _c_to_f(temp_c_high)

    settlement_f_low = _round_half_up(temp_f_low)
    settlement_f_high = _round_half_up(temp_f_high)

    return {
        "temp_c": temp_c,
        "dewp_c": dewp_c,
        "temp_f": temp_f,
        "temp_c_low": temp_c_low,
        "temp_c_high": temp_c_high,
        "temp_f_low": temp_f_low,
        "temp_f_high": temp_f_high,
        "settlement_f_low": settlement_f_low,
        "settlement_f_high": settlement_f_high,
        "has_t_group": has_t_group,
        "used_main_group": used_main_group,
        "used_fallback": used_fallback,
    }
