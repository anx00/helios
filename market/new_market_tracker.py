from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from config import STATIONS, get_active_stations
from market.discovery import fetch_event_for_station_date

logger = logging.getLogger("market_birth_tracker")

MADRID_TZ = ZoneInfo("Europe/Madrid")
STATE_FILENAME = "state.json"
RE_NUM = re.compile(r"-?\d+")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _iso_madrid(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(MADRID_TZ).isoformat()


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()) or "unknown"


def _parse_outcome_prices(raw: Any) -> Tuple[Optional[float], Optional[float]]:
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, list):
            return (None, None)
        yes = float(raw[0]) if len(raw) > 0 and raw[0] is not None else None
        no = float(raw[1]) if len(raw) > 1 and raw[1] is not None else None
        return (yes, no)
    except Exception:
        return (None, None)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _label_sort_key(label: str) -> Tuple[int, int, int, str]:
    s = str(label or "").strip().lower()
    nums = []
    for tok in RE_NUM.findall(s):
        try:
            nums.append(int(round(float(tok))))
        except Exception:
            continue

    if "or below" in s or s.startswith("<"):
        anchor = nums[0] if nums else -999
        return (0, anchor, anchor, s)
    if "or higher" in s or "or above" in s or s.startswith(">") or "≥" in s:
        anchor = nums[0] if nums else 999
        return (2, anchor, anchor, s)

    if len(nums) >= 2:
        return (1, nums[0], nums[1], s)
    if len(nums) == 1:
        return (1, nums[0], nums[0], s)
    return (3, 0, 0, s)


def _extract_brackets(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    markets = event.get("markets") or []
    if not isinstance(markets, list):
        return rows

    for market in markets:
        if not isinstance(market, dict):
            continue
        label = str(market.get("groupItemTitle") or market.get("question") or "").strip()
        yes_price, no_price = _parse_outcome_prices(market.get("outcomePrices"))

        rows.append(
            {
                "label": label,
                "yes_price": yes_price,
                "no_price": no_price,
                "best_bid": _float_or_none(market.get("bestBid")),
                "best_ask": _float_or_none(market.get("bestAsk")),
                "last_trade_price": _float_or_none(market.get("lastTradePrice")),
                "spread": _float_or_none(market.get("spread")),
                "volume": _float_or_none(market.get("volume")),
                "liquidity": _float_or_none(market.get("liquidity")),
                "market_id": str(market.get("id") or ""),
                "market_slug": str(market.get("slug") or ""),
                "market_created_at_utc": str(market.get("createdAt") or "") or None,
                "market_updated_at_utc": str(market.get("updatedAt") or "") or None,
            }
        )

    rows.sort(key=lambda r: _label_sort_key(str(r.get("label") or "")))
    return rows


def _snapshot_fingerprint(snap: Dict[str, Any]) -> str:
    payload = []
    for row in snap.get("brackets") or []:
        payload.append(
            (
                row.get("label"),
                row.get("yes_price"),
                row.get("no_price"),
                row.get("best_bid"),
                row.get("best_ask"),
                row.get("last_trade_price"),
            )
        )
    marker = (
        snap.get("event_updated_at_utc"),
        tuple(payload),
    )
    return json.dumps(marker, ensure_ascii=True, separators=(",", ":"))


def _build_snapshot(
    *,
    event: Dict[str, Any],
    captured_at_utc: datetime,
    event_created_at_utc: Optional[datetime],
    detected_at_utc: Optional[datetime],
) -> Dict[str, Any]:
    event_updated_at = _parse_iso(str(event.get("updatedAt") or ""))
    brackets = _extract_brackets(event)
    out: Dict[str, Any] = {
        "captured_at_utc": _iso_utc(captured_at_utc),
        "captured_at_madrid": _iso_madrid(captured_at_utc),
        "event_updated_at_utc": _iso_utc(event_updated_at),
        "markets_count": len(brackets),
        "brackets": brackets,
    }
    if event_created_at_utc is not None:
        out["seconds_since_event_created"] = round(
            (captured_at_utc - event_created_at_utc).total_seconds(), 3
        )
    if detected_at_utc is not None:
        out["seconds_since_detection"] = round((captured_at_utc - detected_at_utc).total_seconds(), 3)
    return out


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _default_state() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "initialized": False,
        "created_at_utc": _iso_utc(_now_utc()),
        "updated_at_utc": _iso_utc(_now_utc()),
        "known_events": {},  # key -> metadata
    }


def _record_key(station_id: str, target_date: date) -> str:
    return f"{station_id.upper()}|{target_date.isoformat()}"


def _resolve_station_ids(args: argparse.Namespace) -> List[str]:
    if args.stations:
        ids = [s.strip().upper() for s in str(args.stations).split(",") if s.strip()]
    elif args.station:
        ids = [str(args.station).strip().upper()]
    else:
        ids = sorted(get_active_stations().keys())

    valid: List[str] = []
    for sid in ids:
        if sid in STATIONS:
            valid.append(sid)
        else:
            logger.warning("Skipping unknown station: %s", sid)
    return valid


def _iter_target_dates(today: date, min_days_ahead: int, max_days_ahead: int) -> Iterable[date]:
    start = max(0, int(min_days_ahead))
    end = max(start, int(max_days_ahead))
    for offset in range(start, end + 1):
        yield today + timedelta(days=offset)


async def _scan_station_dates(
    station_ids: List[str],
    *,
    base_date: date,
    min_days_ahead: int,
    max_days_ahead: int,
) -> Dict[str, Dict[str, Any]]:
    targets: List[Tuple[str, date]] = []
    for sid in station_ids:
        for d in _iter_target_dates(base_date, min_days_ahead, max_days_ahead):
            targets.append((sid, d))

    async def _one(sid: str, target: date) -> Tuple[str, date, Optional[Dict[str, Any]]]:
        ev = await fetch_event_for_station_date(sid, target)
        return (sid, target, ev)

    results = await asyncio.gather(*[_one(sid, d) for sid, d in targets], return_exceptions=True)
    found: Dict[str, Dict[str, Any]] = {}
    for item in results:
        if isinstance(item, Exception):
            logger.warning("Scan request failed: %s", item)
            continue
        sid, d, ev = item
        if ev:
            found[_record_key(sid, d)] = ev
    return found


def _new_record_relpath(station_id: str, target_date: str, event_uid: str) -> str:
    return str(Path(station_id.upper()) / f"{target_date}__{_safe_filename(event_uid)}.json")


def _register_known_event(
    state: Dict[str, Any],
    *,
    out_dir: Path,
    station_id: str,
    target_date: date,
    event: Dict[str, Any],
    poll_seconds: int,
    capture_minutes: int,
    create_record: bool,
    detected_at_utc: datetime,
) -> Optional[Path]:
    key = _record_key(station_id, target_date)
    event_uid = str(event.get("id") or event.get("slug") or "")
    event_slug = str(event.get("slug") or "")
    event_created_dt = _parse_iso(str(event.get("createdAt") or ""))
    event_updated_dt = _parse_iso(str(event.get("updatedAt") or ""))
    record_relpath = None
    record_path: Optional[Path] = None

    if create_record:
        record_relpath = _new_record_relpath(station_id, target_date.isoformat(), event_uid or event_slug or "event")
        record_path = out_dir / record_relpath

        reference_dt = event_created_dt or detected_at_utc
        window_end_dt = reference_dt + timedelta(minutes=max(1, int(capture_minutes)))
        detection_latency = None
        if event_created_dt is not None:
            detection_latency = round((detected_at_utc - event_created_dt).total_seconds(), 3)

        record = {
            "schema_version": 1,
            "station_id": station_id.upper(),
            "station_name": getattr(STATIONS.get(station_id.upper()), "name", station_id.upper()),
            "target_date": target_date.isoformat(),
            "event_id": str(event.get("id") or ""),
            "event_slug": event_slug,
            "event_title": str(event.get("title") or ""),
            "event_created_at_utc": _iso_utc(event_created_dt),
            "event_created_at_madrid": _iso_madrid(event_created_dt),
            "event_updated_at_utc": _iso_utc(event_updated_dt),
            "detected_first_seen_at_utc": _iso_utc(detected_at_utc),
            "detected_first_seen_at_madrid": _iso_madrid(detected_at_utc),
            "detection_latency_seconds_vs_event_created": detection_latency,
            "capture_reference": "event_created_at_utc" if event_created_dt is not None else "first_seen_at_utc",
            "capture_window_minutes": int(capture_minutes),
            "capture_window_end_utc": _iso_utc(window_end_dt),
            "capture_window_end_madrid": _iso_madrid(window_end_dt),
            "capture_completed": False,
            "poll_interval_seconds": int(poll_seconds),
            "notes": [],
            "snapshots": [],
            "_last_snapshot_fingerprint": None,
        }
        if detection_latency is not None and detection_latency > capture_minutes * 60:
            record["notes"].append(
                "detected_after_capture_window_closed"
            )
        _save_json(record_path, record)

    state["known_events"][key] = {
        "event_uid": event_uid,
        "event_slug": event_slug,
        "event_created_at_utc": _iso_utc(event_created_dt),
        "event_updated_at_utc": _iso_utc(event_updated_dt),
        "first_seen_at_utc": _iso_utc(detected_at_utc),
        "record_relpath": record_relpath,
        "capture_completed": False if create_record else None,
    }
    return record_path


def _append_snapshot_if_needed(
    record_path: Path,
    *,
    event: Dict[str, Any],
    now_utc: datetime,
) -> bool:
    record = _load_json(record_path, {})
    if not isinstance(record, dict) or not record:
        return False

    if bool(record.get("capture_completed")):
        return False

    window_end_dt = _parse_iso(str(record.get("capture_window_end_utc") or ""))
    detected_dt = _parse_iso(str(record.get("detected_first_seen_at_utc") or ""))
    event_created_dt = _parse_iso(str(record.get("event_created_at_utc") or ""))
    if window_end_dt is not None and now_utc > window_end_dt:
        record["capture_completed"] = True
        record["completed_at_utc"] = _iso_utc(now_utc)
        record["completed_at_madrid"] = _iso_madrid(now_utc)
        _save_json(record_path, record)
        return False

    snapshot = _build_snapshot(
        event=event,
        captured_at_utc=now_utc,
        event_created_at_utc=event_created_dt,
        detected_at_utc=detected_dt,
    )
    fp = _snapshot_fingerprint(snapshot)
    if fp == record.get("_last_snapshot_fingerprint"):
        return False

    snaps = record.get("snapshots")
    if not isinstance(snaps, list):
        snaps = []
        record["snapshots"] = snaps
    snaps.append(snapshot)
    record["event_updated_at_utc"] = _iso_utc(_parse_iso(str(event.get("updatedAt") or "")))
    record["_last_snapshot_fingerprint"] = fp
    _save_json(record_path, record)
    return True


def _finalize_expired_records(state: Dict[str, Any], out_dir: Path, now_utc: datetime) -> int:
    closed = 0
    for meta in list((state.get("known_events") or {}).values()):
        if not isinstance(meta, dict):
            continue
        relpath = meta.get("record_relpath")
        if not relpath:
            continue
        record_path = out_dir / str(relpath)
        if not record_path.exists():
            continue
        record = _load_json(record_path, {})
        if not isinstance(record, dict):
            continue
        if bool(record.get("capture_completed")):
            meta["capture_completed"] = True
            continue
        window_end_dt = _parse_iso(str(record.get("capture_window_end_utc") or ""))
        if window_end_dt is not None and now_utc > window_end_dt:
            record["capture_completed"] = True
            record["completed_at_utc"] = _iso_utc(now_utc)
            record["completed_at_madrid"] = _iso_madrid(now_utc)
            _save_json(record_path, record)
            meta["capture_completed"] = True
            closed += 1
    return closed


def _print_scan_summary(
    *,
    found_events: Dict[str, Dict[str, Any]],
    created_records: List[Path],
    captured_snapshots: int,
    finalized_records: int,
    bootstrapped_known: int,
    backfill_existing: bool,
) -> None:
    if bootstrapped_known:
        mode = "backfill+track" if backfill_existing else "mark-known-only"
        logger.info("Bootstrap inicial: %s eventos existentes (%s)", bootstrapped_known, mode)
    if created_records:
        for path in created_records:
            logger.info("Nuevo mercado detectado -> %s", path)
    if captured_snapshots:
        logger.info("Snapshots guardados en este ciclo: %s", captured_snapshots)
    if finalized_records:
        logger.info("Capturas cerradas por ventana expirada: %s", finalized_records)
    if found_events and not created_records and captured_snapshots == 0 and finalized_records == 0:
        logger.debug("Scan sin novedades (%s eventos visibles)", len(found_events))


async def run_tracker(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    state_path = out_dir / STATE_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)

    state = _load_json(state_path, _default_state())
    if not isinstance(state, dict):
        state = _default_state()
    state.setdefault("schema_version", 1)
    state.setdefault("initialized", False)
    state.setdefault("known_events", {})

    station_ids = _resolve_station_ids(args)
    if not station_ids:
        raise ValueError("No valid stations to monitor.")

    logger.info(
        "Monitoring Polymarket temperature market creation for %s stations (%s..%s days ahead, base_tz=%s), poll=%ss, capture=%sm",
        len(station_ids),
        args.min_days_ahead,
        args.max_days_ahead,
        args.calendar_tz,
        args.poll_seconds,
        args.capture_minutes,
    )

    while True:
        cycle_start = _now_utc()
        calendar_tz = ZoneInfo(str(args.calendar_tz))
        base_date = datetime.now(calendar_tz).date()
        found = await _scan_station_dates(
            station_ids,
            base_date=base_date,
            min_days_ahead=args.min_days_ahead,
            max_days_ahead=args.max_days_ahead,
        )

        created_records: List[Path] = []
        captured_snapshots = 0
        finalized_records = 0
        bootstrapped_known = 0

        if not bool(state.get("initialized")):
            for key, event in found.items():
                try:
                    station_id, target_date_str = key.split("|", 1)
                    target_dt = date.fromisoformat(target_date_str)
                except Exception:
                    continue
                rec = _register_known_event(
                    state,
                    out_dir=out_dir,
                    station_id=station_id,
                    target_date=target_dt,
                    event=event,
                    poll_seconds=args.poll_seconds,
                    capture_minutes=args.capture_minutes,
                    create_record=bool(args.backfill_existing),
                    detected_at_utc=cycle_start,
                )
                if rec is not None:
                    created_records.append(rec)
                bootstrapped_known += 1
            state["initialized"] = True
            state["bootstrapped_at_utc"] = _iso_utc(cycle_start)
            state["bootstrapped_at_madrid"] = _iso_madrid(cycle_start)

            # If the user requested backfill on first run, capture an initial snapshot
            # in the same cycle for all bootstrapped records still within the window.
            if args.backfill_existing:
                for key, event in found.items():
                    meta = (state.get("known_events") or {}).get(key)
                    if not isinstance(meta, dict):
                        continue
                    relpath = meta.get("record_relpath")
                    if not relpath:
                        continue
                    record_path = out_dir / str(relpath)
                    if _append_snapshot_if_needed(record_path, event=event, now_utc=cycle_start):
                        captured_snapshots += 1
                finalized_records = _finalize_expired_records(state, out_dir, cycle_start)

        else:
            known_events = state.get("known_events") or {}
            for key, event in found.items():
                try:
                    station_id, target_date_str = key.split("|", 1)
                    target_dt = date.fromisoformat(target_date_str)
                except Exception:
                    continue

                event_uid = str(event.get("id") or event.get("slug") or "")
                meta = known_events.get(key)
                prev_uid = str(meta.get("event_uid") or "") if isinstance(meta, dict) else ""
                if not meta or prev_uid != event_uid:
                    rec = _register_known_event(
                        state,
                        out_dir=out_dir,
                        station_id=station_id,
                        target_date=target_dt,
                        event=event,
                        poll_seconds=args.poll_seconds,
                        capture_minutes=args.capture_minutes,
                        create_record=True,
                        detected_at_utc=cycle_start,
                    )
                    if rec is not None:
                        created_records.append(rec)

            # Append snapshots for any still-open captures present in this scan.
            for key, event in found.items():
                meta = (state.get("known_events") or {}).get(key)
                if not isinstance(meta, dict):
                    continue
                relpath = meta.get("record_relpath")
                if not relpath:
                    continue
                record_path = out_dir / str(relpath)
                if _append_snapshot_if_needed(record_path, event=event, now_utc=cycle_start):
                    captured_snapshots += 1

            finalized_records = _finalize_expired_records(state, out_dir, cycle_start)

        state["updated_at_utc"] = _iso_utc(cycle_start)
        state["updated_at_madrid"] = _iso_madrid(cycle_start)
        _save_json(state_path, state)

        _print_scan_summary(
            found_events=found,
            created_records=created_records,
            captured_snapshots=captured_snapshots,
            finalized_records=finalized_records,
            bootstrapped_known=bootstrapped_known,
            backfill_existing=bool(args.backfill_existing),
        )

        if args.run_once:
            return 0

        elapsed = (_now_utc() - cycle_start).total_seconds()
        sleep_s = max(1.0, float(args.poll_seconds) - elapsed)
        await asyncio.sleep(sleep_s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Detect new Polymarket temperature markets and capture bracket prices "
            "during the first N minutes after creation."
        )
    )
    p.add_argument("--station", default=None, help="Single station ICAO (e.g. EGLC).")
    p.add_argument(
        "--stations",
        default=None,
        help="Comma-separated station ICAOs. Overrides --station. Default: all active stations.",
    )
    p.add_argument("--min-days-ahead", type=int, default=0, help="Min target date offset to monitor (default: 0=today).")
    p.add_argument("--max-days-ahead", type=int, default=4, help="Max target date offset to monitor (default: 4).")
    p.add_argument(
        "--calendar-tz",
        default="Europe/Madrid",
        help="Timezone used to interpret day offsets (hoy/manana/pasado-manana). Default: Europe/Madrid.",
    )
    p.add_argument("--poll-seconds", type=int, default=30, help="Polling interval in seconds (default: 30).")
    p.add_argument(
        "--capture-minutes",
        type=int,
        default=10,
        help="Capture window length from event creation in minutes (default: 10).",
    )
    p.add_argument(
        "--out-dir",
        default="data/polymarket_market_births",
        help="Output directory for state + per-event JSON records.",
    )
    p.add_argument(
        "--backfill-existing",
        action="store_true",
        help="On first run, create records for already visible events instead of only marking them as known.",
    )
    p.add_argument("--run-once", action="store_true", help="Run a single scan cycle and exit.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s | %(message)s",
    )
    try:
        return asyncio.run(run_tracker(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.error("Tracker failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
