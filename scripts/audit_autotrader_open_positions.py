from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import STATIONS
from core.polymarket_labels import normalize_label


UTC = ZoneInfo("UTC")


def _http_json(url: str) -> Dict[str, Any]:
    with urlopen(url, timeout=40) as response:
        payload = json.load(response)
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _station_today_iso(station_id: str) -> Optional[str]:
    station = STATIONS.get(str(station_id or "").upper())
    if not station:
        return None
    now_local = datetime.now(ZoneInfo(station.timezone))
    return now_local.date().isoformat()


def _target_day_phase(station_id: str, target_date: str) -> str:
    today_iso = _station_today_iso(station_id)
    if not today_iso or not target_date:
        return "unknown"
    if target_date < today_iso:
        return "past"
    if target_date > today_iso:
        return "future"
    return "today"


def _infer_settlement_value(label: str, side: str, winning_outcome: Optional[str]) -> Optional[float]:
    normalized_winner = normalize_label(str(winning_outcome or "")) or str(winning_outcome or "")
    normalized_label = normalize_label(str(label or "")) or str(label or "")
    if not normalized_winner:
        return None
    yes_settlement = 1.0 if normalized_label == normalized_winner else 0.0
    return yes_settlement if str(side or "YES").upper() == "YES" else 1.0 - yes_settlement


def _entry_price(row: Dict[str, Any]) -> float:
    avg_price = _safe_float(row.get("avg_price"))
    if avg_price is not None and avg_price > 0.0:
        return float(avg_price)
    entry_price = _safe_float(row.get("entry_price"))
    if entry_price is not None and entry_price > 0.0:
        return float(entry_price)
    shares_open = float(_safe_float(row.get("shares_open")) or _safe_float(row.get("shares")) or 0.0)
    cost_basis = float(_safe_float(row.get("cost_basis_open_usd")) or 0.0)
    return (cost_basis / shares_open) if shares_open > 0.0 else 0.0


def _classify_position(
    row: Dict[str, Any],
    market_status: Dict[str, Any],
    *,
    dust_value_threshold: float,
) -> Tuple[str, Optional[float], Optional[float]]:
    label = str(row.get("label") or "")
    side = str(row.get("side") or "YES").upper()
    winning_outcome = market_status.get("winning_outcome")
    settlement_value = _infer_settlement_value(label, side, winning_outcome)
    shares_open = float(_safe_float(row.get("shares_open")) or _safe_float(row.get("shares")) or 0.0)
    expected_pnl = None
    if settlement_value is not None and shares_open > 0.0:
        expected_pnl = round((settlement_value - _entry_price(row)) * shares_open, 6)

    resolved = bool(market_status.get("event_closed")) or bool(market_status.get("is_mature"))
    target_phase = _target_day_phase(str(row.get("station_id") or ""), str(row.get("target_date") or ""))
    current_value = float(_safe_float(row.get("current_value_usd")) or 0.0)

    if resolved and settlement_value is not None:
        return (
            "resolved_open_winner" if settlement_value >= 0.999 else "resolved_open_loser",
            settlement_value,
            expected_pnl,
        )
    if resolved:
        return "resolved_open_unknown", settlement_value, expected_pnl
    if target_phase == "past":
        return "past_target_open_unresolved", settlement_value, expected_pnl
    if current_value <= dust_value_threshold:
        return "dust_open_unresolved", settlement_value, expected_pnl
    return "live_open", settlement_value, expected_pnl


def _fmt_money(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def build_report(
    base_url: str,
    *,
    dust_value_threshold: float = 0.05,
) -> str:
    status_url = f"{base_url.rstrip('/')}/api/autotrader/status"
    status_payload = _http_json(status_url)
    open_positions = list((status_payload.get("portfolio") or {}).get("open_positions") or [])

    lines: List[str] = [
        "# Autotrader Open Position Audit",
        "",
        f"- Generated at: `{datetime.now(UTC).isoformat()}`",
        f"- Base URL: `{base_url}`",
        f"- Open positions audited: `{len(open_positions)}`",
        "",
    ]

    if not open_positions:
        lines.append("No open positions found.")
        return "\n".join(lines) + "\n"

    market_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    classification_counts: Counter[str] = Counter()
    total_cost_basis = 0.0
    total_current_value = 0.0
    total_expected_settlement_pnl = 0.0
    expected_settlement_pnl_known = 0
    audited_rows: List[Dict[str, Any]] = []

    for row in open_positions:
        station_id = str(row.get("station_id") or "").upper()
        target_date = str(row.get("target_date") or "")
        cache_key = (station_id, target_date)
        if cache_key not in market_cache:
            params = urlencode({"target_date": target_date})
            market_url = f"{base_url.rstrip('/')}/api/polymarket/{station_id}?{params}"
            try:
                market_cache[cache_key] = _http_json(market_url)
            except (HTTPError, URLError, TimeoutError):
                market_cache[cache_key] = {}
        market_payload = market_cache[cache_key]
        market_status = dict(market_payload.get("market_status") or {})
        classification, settlement_value, expected_settlement_pnl = _classify_position(
            row,
            market_status,
            dust_value_threshold=dust_value_threshold,
        )
        classification_counts[classification] += 1
        cost_basis = float(_safe_float(row.get("cost_basis_open_usd")) or 0.0)
        current_value = float(_safe_float(row.get("current_value_usd")) or 0.0)
        total_cost_basis += cost_basis
        total_current_value += current_value
        if expected_settlement_pnl is not None:
            total_expected_settlement_pnl += expected_settlement_pnl
            expected_settlement_pnl_known += 1

        audited_rows.append(
            {
                "station_id": station_id,
                "target_date": target_date,
                "label": str(row.get("label") or ""),
                "side": str(row.get("side") or "YES").upper(),
                "classification": classification,
                "target_phase": _target_day_phase(station_id, target_date),
                "winning_outcome": market_status.get("winning_outcome"),
                "is_mature": bool(market_status.get("is_mature")),
                "event_closed": bool(market_status.get("event_closed")),
                "max_probability": _safe_float(market_status.get("max_probability")),
                "entry_price": _entry_price(row),
                "current_price": _safe_float(row.get("current_price")),
                "shares_open": float(_safe_float(row.get("shares_open")) or _safe_float(row.get("shares")) or 0.0),
                "cost_basis_open_usd": cost_basis,
                "current_value_usd": current_value,
                "cash_pnl_usd": float(_safe_float(row.get("cash_pnl_usd")) or 0.0),
                "settlement_value": settlement_value,
                "expected_settlement_pnl_usd": expected_settlement_pnl,
            }
        )

    lines.extend(
        [
            "## Summary",
            "",
            f"- Total cost basis open: `{total_cost_basis:.4f} USD`",
            f"- Total current value open: `{total_current_value:.4f} USD`",
            f"- Mark-to-market open pnl: `{(total_current_value - total_cost_basis):+.4f} USD`",
            f"- Expected settlement pnl on positions with inferred winner: `{total_expected_settlement_pnl:+.4f} USD` across `{expected_settlement_pnl_known}` positions",
            "",
            "## Classification Counts",
            "",
        ]
    )
    for classification, count in classification_counts.most_common():
        lines.append(f"- `{classification}`: `{count}`")

    lines.extend(
        [
            "",
            "## Position Table",
            "",
            "| station | target_date | target_phase | label | side | class | mature | closed | winner | max_prob | entry | current | shares | cost_basis | current_value | cash_pnl | settle | expected_settle_pnl |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    audited_rows.sort(
        key=lambda row: (
            row["classification"],
            -float(row["cost_basis_open_usd"]),
            row["station_id"],
            row["label"],
        )
    )
    for row in audited_rows:
        lines.append(
            "| {station_id} | {target_date} | {target_phase} | {label} | {side} | {classification} | {is_mature} | {event_closed} | {winning_outcome} | {max_probability} | {entry_price} | {current_price} | {shares_open} | {cost_basis_open_usd} | {current_value_usd} | {cash_pnl_usd} | {settlement_value} | {expected_settlement_pnl_usd} |".format(
                station_id=row["station_id"],
                target_date=row["target_date"],
                target_phase=row["target_phase"],
                label=row["label"].replace("|", "/"),
                side=row["side"],
                classification=row["classification"],
                is_mature="1" if row["is_mature"] else "0",
                event_closed="1" if row["event_closed"] else "0",
                winning_outcome=(str(row["winning_outcome"] or "").replace("|", "/")),
                max_probability=_fmt_ratio(row["max_probability"]),
                entry_price=_fmt_ratio(row["entry_price"]),
                current_price=_fmt_ratio(row["current_price"]),
                shares_open=_fmt_ratio(row["shares_open"]),
                cost_basis_open_usd=_fmt_money(row["cost_basis_open_usd"]),
                current_value_usd=_fmt_money(row["current_value_usd"]),
                cash_pnl_usd=_fmt_money(row["cash_pnl_usd"]),
                settlement_value=_fmt_ratio(row["settlement_value"]),
                expected_settlement_pnl_usd=_fmt_money(row["expected_settlement_pnl_usd"]),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `resolved_open_loser` means the market already looks settled against the held side, but the position still appears open in the live portfolio snapshot.",
            "- `resolved_open_winner` means the market looks settled in favor of the held side, but the position still appears open in the live portfolio snapshot.",
            "- `past_target_open_unresolved` means the market day is already past in the station timezone, but HELIOS still does not see the position as closed or resolved.",
            "- This audit is diagnostic. It relies on HELIOS `/api/polymarket/{station}` market status and the current live portfolio snapshot.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit current autotrader open positions against live market status.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="")
    parser.add_argument("--dust-value-threshold", type=float, default=0.05)
    args = parser.parse_args()

    report = build_report(
        args.base_url,
        dust_value_threshold=max(0.0, float(args.dust_value_threshold)),
    )
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
