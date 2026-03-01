from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.polymarket_labels import normalize_label


UTC = timezone.utc


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_channel_events(root: Path, date_str: str, channel: str) -> List[Dict[str, Any]]:
    path = root / f"date={date_str}" / f"ch={channel}" / "events.ndjson"
    rows = _load_jsonl(path)
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        ts = _parse_iso_datetime(row.get("ts_ingest_utc"))
        if ts is None:
            continue
        enriched_row = dict(row)
        enriched_row["_ts"] = ts
        enriched.append(enriched_row)
    enriched.sort(key=lambda row: row["_ts"])
    return enriched


def _group_events_by_station(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        station_id = str(row.get("station_id") or "").upper()
        if not station_id:
            continue
        grouped[station_id].append(row)
    return grouped


def _last_event_at_or_before(rows: List[Dict[str, Any]], ts: datetime) -> Optional[Dict[str, Any]]:
    candidate = None
    for row in rows:
        row_ts = row.get("_ts")
        if not isinstance(row_ts, datetime):
            continue
        if row_ts <= ts:
            candidate = row
        else:
            break
    return candidate


def _first_event_at_or_after(rows: List[Dict[str, Any]], ts: datetime) -> Optional[Dict[str, Any]]:
    for row in rows:
        row_ts = row.get("_ts")
        if isinstance(row_ts, datetime) and row_ts >= ts:
            return row
    return None


def _last_event(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return rows[-1] if rows else None


def _find_bucket_payload(market_event: Optional[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
    if not isinstance(market_event, dict):
        return None
    data = market_event.get("data")
    if not isinstance(data, dict):
        return None
    target = normalize_label(label) or label
    for raw_label, payload in data.items():
        if raw_label == "__meta__":
            continue
        if normalize_label(str(raw_label)) == target and isinstance(payload, dict):
            return payload
    return None


def _extract_market_side_snapshot(market_event: Optional[Dict[str, Any]], label: str, side: str) -> Dict[str, Any]:
    payload = _find_bucket_payload(market_event, label)
    side_key = str(side or "YES").upper()
    if not isinstance(payload, dict):
        return {"best_bid": None, "best_ask": None, "mid": None, "spread": None}
    prefix = "yes" if side_key == "YES" else "no"
    best_bid = _safe_float(payload.get(f"{prefix}_best_bid"))
    best_ask = _safe_float(payload.get(f"{prefix}_best_ask"))
    mid = _safe_float(payload.get(f"{prefix}_mid"))
    spread = _safe_float(payload.get(f"{prefix}_spread"))
    if side_key == "YES":
        best_bid = _safe_float(payload.get("best_bid")) if best_bid is None else best_bid
        best_ask = _safe_float(payload.get("best_ask")) if best_ask is None else best_ask
        mid = _safe_float(payload.get("mid")) if mid is None else mid
        spread = _safe_float(payload.get("spread")) if spread is None else spread
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
    }


def _extract_nowcast_snapshot(nowcast_event: Optional[Dict[str, Any]], label: str, side: str) -> Dict[str, Any]:
    if not isinstance(nowcast_event, dict):
        return {"fair_prob": None, "confidence": None, "sigma_f": None}
    data = nowcast_event.get("data")
    if not isinstance(data, dict):
        return {"fair_prob": None, "confidence": None, "sigma_f": None}
    target = normalize_label(label) or label
    yes_prob = None
    for row in list(data.get("p_bucket") or []):
        if not isinstance(row, dict):
            continue
        if normalize_label(str(row.get("label") or "")) == target:
            yes_prob = _safe_float(row.get("probability"))
            if yes_prob is None:
                yes_prob = _safe_float(row.get("prob"))
            break
    if yes_prob is None:
        fair_prob = None
    elif str(side or "YES").upper() == "YES":
        fair_prob = yes_prob
    else:
        fair_prob = 1.0 - yes_prob
    return {
        "fair_prob": fair_prob,
        "confidence": _safe_float(data.get("confidence")),
        "sigma_f": _safe_float(data.get("tmax_sigma_f")),
    }


def _extract_pws_snapshot(pws_event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(pws_event, dict):
        return {
            "median_f": None,
            "support": None,
            "qc": None,
            "leader_station_id": None,
            "leader_temp_f": None,
            "leader_weight": None,
            "leader_quality_band": None,
        }
    data = pws_event.get("data")
    if not isinstance(data, dict):
        return {
            "median_f": None,
            "support": None,
            "qc": None,
            "leader_station_id": None,
            "leader_temp_f": None,
            "leader_weight": None,
            "leader_quality_band": None,
        }
    readings = [row for row in list(data.get("pws_readings") or []) if isinstance(row, dict)]
    readings.sort(key=lambda row: float(_safe_float(row.get("weight")) or 0.0), reverse=True)
    leader = readings[0] if readings else {}
    return {
        "median_f": _safe_float(data.get("median_f")),
        "support": int(_safe_float(data.get("support")) or 0) if data.get("support") is not None else None,
        "qc": str(data.get("qc") or "") or None,
        "leader_station_id": str(leader.get("station_id") or "") or None,
        "leader_temp_f": _safe_float(leader.get("temp_f")),
        "leader_weight": _safe_float(leader.get("weight")),
        "leader_quality_band": str(leader.get("quality_band") or "") or None,
    }


def _fmt_number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_signed_diff(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):+.{digits}f}"


@dataclass
class ExitAuditRow:
    ts_utc: datetime
    station_id: str
    position_key: str
    reason: str
    status: str
    label: str
    side: str
    exit_bid: Optional[float]
    exit_mid: Optional[float]
    fair_prob: Optional[float]
    confidence: Optional[float]
    sigma_f: Optional[float]
    pws_median_f: Optional[float]
    pws_qc: Optional[str]
    mid_plus_15m: Optional[float]
    bid_plus_15m: Optional[float]
    mid_plus_60m: Optional[float]
    bid_plus_60m: Optional[float]
    final_mid: Optional[float]
    final_bid: Optional[float]


def _collect_exit_rows(log_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    exits: List[Dict[str, Any]] = []
    for row in log_rows:
        exit_plan = row.get("exit_plan") or {}
        reason = str(exit_plan.get("reason") or "").strip()
        ts = _parse_iso_datetime(row.get("ts_utc"))
        if not reason or ts is None:
            continue
        position_key = str(row.get("position_key") or "").strip()
        if not position_key:
            continue
        exits.append(dict(row))
    exits.sort(key=lambda row: str(row.get("ts_utc") or ""))
    return exits


def _build_audit_rows(
    exit_rows: Iterable[Dict[str, Any]],
    market_by_station: Dict[str, List[Dict[str, Any]]],
    nowcast_by_station: Dict[str, List[Dict[str, Any]]],
    pws_by_station: Dict[str, List[Dict[str, Any]]],
) -> List[ExitAuditRow]:
    audit_rows: List[ExitAuditRow] = []
    for row in exit_rows:
        ts = _parse_iso_datetime(row.get("ts_utc"))
        if ts is None:
            continue
        station_id = str(row.get("station_id") or "").upper()
        position_key = str(row.get("position_key") or "")
        exit_plan = dict(row.get("exit_plan") or {})
        position = dict(row.get("position") or {})
        label = str(position.get("label") or "").strip()
        side = str(position.get("side") or "").strip().upper()
        market_rows = market_by_station.get(station_id, [])
        nowcast_rows = nowcast_by_station.get(station_id, [])
        pws_rows = pws_by_station.get(station_id, [])

        market_t0 = _last_event_at_or_before(market_rows, ts)
        market_15 = _first_event_at_or_after(market_rows, ts + timedelta(minutes=15))
        market_60 = _first_event_at_or_after(market_rows, ts + timedelta(minutes=60))
        market_final = _last_event(market_rows)
        nowcast_t0 = _last_event_at_or_before(nowcast_rows, ts)
        pws_t0 = _last_event_at_or_before(pws_rows, ts)

        market_t0_side = _extract_market_side_snapshot(market_t0, label, side)
        market_15_side = _extract_market_side_snapshot(market_15, label, side)
        market_60_side = _extract_market_side_snapshot(market_60, label, side)
        market_final_side = _extract_market_side_snapshot(market_final, label, side)
        nowcast_t0_side = _extract_nowcast_snapshot(nowcast_t0, label, side)
        pws_t0_summary = _extract_pws_snapshot(pws_t0)

        audit_rows.append(
            ExitAuditRow(
                ts_utc=ts,
                station_id=station_id,
                position_key=position_key,
                reason=str(exit_plan.get("reason") or ""),
                status=str(row.get("status") or ""),
                label=label,
                side=side,
                exit_bid=_safe_float(exit_plan.get("best_bid")),
                exit_mid=market_t0_side["mid"],
                fair_prob=nowcast_t0_side["fair_prob"],
                confidence=nowcast_t0_side["confidence"],
                sigma_f=nowcast_t0_side["sigma_f"],
                pws_median_f=pws_t0_summary["median_f"],
                pws_qc=pws_t0_summary["qc"],
                mid_plus_15m=market_15_side["mid"],
                bid_plus_15m=market_15_side["best_bid"],
                mid_plus_60m=market_60_side["mid"],
                bid_plus_60m=market_60_side["best_bid"],
                final_mid=market_final_side["mid"],
                final_bid=market_final_side["best_bid"],
            )
        )
    return audit_rows


def _render_markdown(
    date_str: str,
    audit_rows: List[ExitAuditRow],
    *,
    log_path: Path,
    recordings_root: Path,
) -> str:
    reason_counts = Counter(row.reason for row in audit_rows)
    status_counts = Counter(row.status for row in audit_rows)
    lines: List[str] = []
    lines.append(f"# Autotrader Exit Replay Audit - {date_str}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Log: `{log_path}`")
    lines.append(f"- Recordings root: `{recordings_root}`")
    lines.append(f"- Exit rows audited: `{len(audit_rows)}`")
    lines.append("")
    lines.append("## Exit Reasons")
    lines.append("")
    for reason, count in reason_counts.most_common():
        lines.append(f"- `{reason}`: `{count}`")
    lines.append("")
    lines.append("## Exit Statuses")
    lines.append("")
    for status, count in status_counts.most_common():
        lines.append(f"- `{status}`: `{count}`")
    lines.append("")
    lines.append("## Assessment Rules")
    lines.append("")
    lines.append("- `early_exit_15m`: the side mid at +15m is above the exit best bid.")
    lines.append("- `early_exit_60m`: the side mid at +60m is above the exit best bid.")
    lines.append("- `early_exit_final`: the final side mid is above the exit best bid.")
    lines.append("- These are diagnostics, not pnl-final truth; they measure whether the bot sold before the market improved for the same side.")
    lines.append("")
    lines.append("## Exit Table")
    lines.append("")
    lines.append("| ts_utc | station | reason | status | label | side | exit_bid | exit_mid | fair_prob | conf | sigma_f | pws_median_f | pws_qc | d15_mid-exit | d60_mid-exit | final_mid-exit |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for row in audit_rows:
        d15 = None if row.mid_plus_15m is None or row.exit_bid is None else (row.mid_plus_15m - row.exit_bid)
        d60 = None if row.mid_plus_60m is None or row.exit_bid is None else (row.mid_plus_60m - row.exit_bid)
        d_final = None if row.final_mid is None or row.exit_bid is None else (row.final_mid - row.exit_bid)
        lines.append(
            "| "
            + " | ".join(
                [
                    row.ts_utc.isoformat(),
                    row.station_id,
                    row.reason,
                    row.status,
                    row.label.replace("|", "/"),
                    row.side,
                    _fmt_number(row.exit_bid, 4),
                    _fmt_number(row.exit_mid, 4),
                    _fmt_number(row.fair_prob, 4),
                    _fmt_number(row.confidence, 2),
                    _fmt_number(row.sigma_f, 2),
                    _fmt_number(row.pws_median_f, 2),
                    str(row.pws_qc or ""),
                    _fmt_signed_diff(d15, 4),
                    _fmt_signed_diff(d60, 4),
                    _fmt_signed_diff(d_final, 4),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- If `take_profit` exits show positive `d15/d60/final` repeatedly, the current take-profit is too early or too price-insensitive.")
    lines.append("- If `policy_flip` exits show positive `d15/d60/final`, that reinforces removing `policy_flip` as a hard exit trigger.")
    lines.append("- If `model_broke` exits still look bad, the model-broke condition needs tighter definition than `fair_now < entry_price - buffer`.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit autotrader exits against replay data.")
    parser.add_argument("--date", required=True, help="Partition date in YYYY-MM-DD format.")
    parser.add_argument(
        "--logs",
        default="forense/remote/opt/helios/logs/autotrader_trades.jsonl",
        help="Path to autotrader JSONL log file.",
    )
    parser.add_argument(
        "--recordings-root",
        default="forense/remote/opt/helios/data/recordings",
        help="Root folder containing date=YYYY-MM-DD replay channels.",
    )
    parser.add_argument(
        "--output",
        default="forense/AUTOTRADER_EXIT_REPLAY_AUDIT.md",
        help="Output Markdown report path.",
    )
    args = parser.parse_args()

    log_path = Path(args.logs)
    recordings_root = Path(args.recordings_root)
    output_path = Path(args.output)

    log_rows = _load_jsonl(log_path)
    exit_rows = [
        row for row in _collect_exit_rows(log_rows)
        if str(row.get("ts_utc") or "").startswith(args.date)
    ]

    market_rows = _load_channel_events(recordings_root, args.date, "market")
    nowcast_rows = _load_channel_events(recordings_root, args.date, "nowcast")
    pws_rows = _load_channel_events(recordings_root, args.date, "pws")

    audit_rows = _build_audit_rows(
        exit_rows,
        _group_events_by_station(market_rows),
        _group_events_by_station(nowcast_rows),
        _group_events_by_station(pws_rows),
    )
    markdown = _render_markdown(
        args.date,
        audit_rows,
        log_path=log_path,
        recordings_root=recordings_root,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {len(audit_rows)} exit rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
