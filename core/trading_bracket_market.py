from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


def evaluate_bracket_market_impl(
    *,
    station_id: str,
    target_day: int,
    target_date: Any,
    brackets: Sequence[Dict[str, Any]],
    terminal_model: Optional[Dict[str, Any]],
    official: Optional[Dict[str, Any]] = None,
    pws_details: Optional[Sequence[Dict[str, Any]]] = None,
    pws_metrics: Optional[Dict[str, Any]] = None,
    reference_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    from core import trading_signal as ts

    ref_utc = (reference_utc or datetime.now(ts.UTC)).astimezone(ts.UTC)
    market_unit = ts._market_unit(station_id)
    target_date_obj = ts._coerce_target_date(target_date, ref_utc, station_id, target_day)
    if not terminal_model:
        return {
            "available": False,
            "station_id": station_id,
            "target_day": int(target_day),
            "target_date": target_date_obj.isoformat(),
            "market_unit": market_unit,
        }

    labels = [
        ts.normalize_label(str(row.get("name") or row.get("bracket") or ""))
        for row in brackets
        if isinstance(row, dict) and str(row.get("name") or row.get("bracket") or "").strip()
    ]
    ordered_labels = ts.sort_labels(labels)

    official_temp_c = None
    official_obs_utc = None
    if isinstance(official, dict):
        official_temp_c = ts._safe_float(official.get("temp_c"))
        official_obs_utc = ts._parse_iso_utc(official.get("obs_time_utc"))
    official_market = ts._convert_c_to_market_unit(official_temp_c, market_unit)

    next_projection = None
    if target_day == 0:
        learning_metric = None
        if isinstance(pws_metrics, dict):
            learning_metric = (pws_metrics.get("learning") if isinstance(pws_metrics.get("learning"), dict) else {})

        consensus_c = None
        if pws_details:
            valid_temps = [
                ts._safe_float(row.get("temp_c"))
                for row in pws_details
                if isinstance(row, dict) and bool(row.get("valid"))
            ]
            nums = [temp for temp in valid_temps if temp is not None]
            if nums:
                nums.sort()
                mid = len(nums) // 2
                if len(nums) % 2 == 0:
                    consensus_c = (nums[mid - 1] + nums[mid]) / 2.0
                else:
                    consensus_c = nums[mid]

        next_projection = ts.estimate_next_metar_projection(
            station_id=station_id,
            rows=list(pws_details or []),
            official_temp_c=official_temp_c,
            official_obs_utc=official_obs_utc,
            learning_metric=learning_metric,
            consensus_c=consensus_c,
            reference_utc=ref_utc,
        )

    terminal_model = ts._reconcile_intraday_terminal_model(
        station_id=station_id,
        target_day=target_day,
        terminal_model=terminal_model,
        official=official,
        next_projection=next_projection,
        reference_utc=ref_utc,
    )

    fair_rows = ts._build_fair_probabilities(
        labels=ordered_labels,
        mean_market=float(terminal_model["mean_market"]),
        sigma_market=max(0.15, float(terminal_model["sigma_market"])),
        market_floor=terminal_model.get("market_floor"),
        market_ceiling=terminal_model.get("market_ceiling"),
    )
    fair_map = {row["label"]: row["fair_prob"] for row in fair_rows}

    tactical_map: Dict[str, float] = {}
    tactical_mean = None
    tactical_influence = 0.0
    if target_day == 0 and next_projection and bool(next_projection.get("available")):
        next_delta = ts._safe_float(next_projection.get("delta_market")) or 0.0
        tactical_influence = ts._repricing_influence(
            station_id=station_id,
            next_projection=next_projection,
            terminal_model=terminal_model,
            reference_utc=ref_utc,
        )
        if tactical_influence > 0:
            tactical_mean = float(terminal_model["mean_market"]) + (next_delta * tactical_influence)
            tactical_rows = ts._build_fair_probabilities(
                labels=ordered_labels,
                mean_market=tactical_mean,
                sigma_market=max(0.15, float(terminal_model["sigma_market"])),
                market_floor=terminal_model.get("market_floor"),
                market_ceiling=terminal_model.get("market_ceiling"),
            )
            tactical_map = {row["label"]: row["fair_prob"] for row in tactical_rows}

    opportunities: List[Dict[str, Any]] = []
    for row in brackets:
        if not isinstance(row, dict):
            continue
        label = ts.normalize_label(str(row.get("name") or row.get("bracket") or ""))
        if not label:
            continue

        fair_yes = float(fair_map.get(label, 0.0))
        tactical_yes = float(tactical_map.get(label, fair_yes))

        yes_bid = ts._quote_prob(row.get("ws_yes_best_bid", row.get("ws_best_bid")))
        yes_ask = ts._quote_prob(row.get("ws_yes_best_ask", row.get("ws_best_ask")))
        yes_mid = ts._quote_prob(row.get("ws_yes_mid", row.get("ws_mid")))
        yes_ref = ts._quote_prob(row.get("yes_price"))
        market_yes = ts.select_probability_01_from_quotes(
            best_bid=yes_bid,
            best_ask=yes_ask,
            mid=yes_mid,
            reference=yes_ref,
            wide_spread_threshold=0.04,
            reference_tolerance=0.005,
        )

        no_bid = ts._quote_prob(row.get("ws_no_best_bid"))
        no_ask = ts._quote_prob(row.get("ws_no_best_ask"))
        no_mid = ts._quote_prob(row.get("ws_no_mid"))
        no_ref = ts._quote_prob(row.get("no_price"))
        market_no = ts.select_probability_01_from_quotes(
            best_bid=no_bid,
            best_ask=no_ask,
            mid=no_mid,
            reference=no_ref,
            wide_spread_threshold=0.04,
            reference_tolerance=0.005,
        )

        yes_entry = yes_ask if yes_ask is not None else yes_ref
        no_entry = no_ask if no_ask is not None else no_ref
        edge_yes = (fair_yes - yes_entry) if yes_entry is not None else None
        edge_no = ((1.0 - fair_yes) - no_entry) if no_entry is not None else None
        tactical_edge_yes = (tactical_yes - yes_entry) if yes_entry is not None else None
        tactical_edge_no = ((1.0 - tactical_yes) - no_entry) if no_entry is not None else None

        best_side = "YES"
        best_edge = edge_yes if edge_yes is not None else -999.0
        tactical_best_side = "YES"
        tactical_best_edge = tactical_edge_yes if tactical_edge_yes is not None else -999.0
        if edge_no is not None and edge_no > best_edge:
            best_side = "NO"
            best_edge = edge_no
        if tactical_edge_no is not None and tactical_edge_no > tactical_best_edge:
            tactical_best_side = "NO"
            tactical_best_edge = tactical_edge_no

        selected_entry = yes_entry if best_side == "YES" else no_entry
        selected_market = market_yes if best_side == "YES" else market_no
        selected_fair = fair_yes if best_side == "YES" else (1.0 - fair_yes)
        selected_tactical_entry = yes_entry if tactical_best_side == "YES" else no_entry
        selected_tactical_market = market_yes if tactical_best_side == "YES" else market_no
        selected_tactical_fair = tactical_yes if tactical_best_side == "YES" else (1.0 - tactical_yes)

        side_spread = None
        side_depth = None
        side_staleness = None
        if best_side == "YES":
            side_spread = ts._safe_float(row.get("ws_yes_spread", row.get("ws_spread")))
            side_depth = ts._safe_float(row.get("ws_yes_ask_depth", row.get("ws_ask_depth")))
            side_staleness = ts._safe_float(row.get("ws_yes_staleness_ms", row.get("ws_staleness_ms")))
        else:
            side_spread = ts._safe_float(row.get("ws_no_spread"))
            side_depth = ts._safe_float(row.get("ws_no_ask_depth"))
            side_staleness = ts._safe_float(row.get("ws_no_staleness_ms"))

        spread_factor = 0.74 if side_spread is None else ts._clamp(1.0 - (side_spread / 0.08), 0.35, 1.0)
        depth_factor = 0.62 if side_depth is None else ts._clamp(0.55 + min(0.45, side_depth / 300.0), 0.55, 1.0)
        stale_factor = 0.72 if side_staleness is None else ts._clamp(1.0 - (side_staleness / 90000.0), 0.4, 1.0)
        quality = spread_factor * depth_factor * stale_factor * float(terminal_model["confidence"])

        opportunities.append({
            "label": label,
            "volume": round(float(ts._safe_float(row.get("volume")) or 0.0), 2),
            "bucket_yes_probability": round(fair_yes, 6),
            "bucket_no_probability": round(1.0 - fair_yes, 6),
            "fair_yes": round(fair_yes, 6),
            "fair_no": round(1.0 - fair_yes, 6),
            "tactical_yes": round(tactical_yes, 6),
            "tactical_no": round(1.0 - tactical_yes, 6),
            "market_yes": round(float(market_yes), 6) if market_yes is not None else None,
            "market_no": round(float(market_no), 6) if market_no is not None else None,
            "yes_entry": round(float(yes_entry), 6) if yes_entry is not None else None,
            "no_entry": round(float(no_entry), 6) if no_entry is not None else None,
            "selected_market": round(float(selected_market), 6) if selected_market is not None else None,
            "selected_entry": round(float(selected_entry), 6) if selected_entry is not None else None,
            "selected_fair": round(float(selected_fair), 6),
            "tactical_best_side": tactical_best_side,
            "selected_tactical_market": round(float(selected_tactical_market), 6) if selected_tactical_market is not None else None,
            "selected_tactical_entry": round(float(selected_tactical_entry), 6) if selected_tactical_entry is not None else None,
            "selected_tactical_fair": round(float(selected_tactical_fair), 6),
            "edge_yes": round(float(edge_yes), 6) if edge_yes is not None else None,
            "edge_no": round(float(edge_no), 6) if edge_no is not None else None,
            "tactical_edge_yes": round(float(tactical_edge_yes), 6) if tactical_edge_yes is not None else None,
            "tactical_edge_no": round(float(tactical_edge_no), 6) if tactical_edge_no is not None else None,
            "best_side": best_side,
            "best_edge": round(float(best_edge), 6),
            "tactical_best_edge": round(float(tactical_best_edge), 6),
            "edge_points": round(float(best_edge) * 100.0, 4),
            "tactical_edge_points": round(float(tactical_best_edge) * 100.0, 4),
            "score": round(float(best_edge) * 100.0 * quality, 4),
            "tactical_score": round(float(tactical_best_edge) * 100.0 * quality, 4),
        })

    top_labels = sorted(fair_rows, key=lambda row: float(row.get("fair_prob") or 0.0), reverse=True)
    top_label = top_labels[0]["label"] if top_labels else None
    top_label_prob = top_labels[0]["fair_prob"] if top_labels else None
    top_label_complement_prob = (1.0 - float(top_label_prob)) if top_label_prob is not None else None

    expected_label, _ = ts.label_for_temp(float(terminal_model["mean_market"]), ordered_labels)
    label_positions = {label: idx for idx, label in enumerate(ordered_labels)}
    top_label_position = label_positions.get(top_label) if top_label else None
    top_bucket_row = next((row for row in opportunities if row.get("label") == top_label), None)
    top_bucket_yes_edge = ts._safe_float(top_bucket_row.get("edge_yes")) if isinstance(top_bucket_row, dict) else None

    terminal_ranked: List[Dict[str, Any]] = []
    tactical_ranked: List[Dict[str, Any]] = []
    blocked_terminal_count = 0
    blocked_tactical_count = 0
    for row in opportunities:
        row_label = str(row.get("label") or "")
        row_position = label_positions.get(row_label)
        row["distance_from_top"] = (
            abs(int(row_position) - int(top_label_position))
            if row_position is not None and top_label_position is not None
            else None
        )
        row["top_bucket_yes_edge"] = round(float(top_bucket_yes_edge), 6) if top_bucket_yes_edge is not None else None
        terminal_policy = ts._evaluate_trade_policy(
            row=row,
            target_day=target_day,
            market_unit=market_unit,
            terminal_model=terminal_model,
            top_label=top_label,
            official_market=official_market,
            next_projection=next_projection,
            tactical=False,
        )
        tactical_policy = ts._evaluate_trade_policy(
            row=row,
            target_day=target_day,
            market_unit=market_unit,
            terminal_model=terminal_model,
            top_label=top_label,
            official_market=official_market,
            next_projection=next_projection,
            tactical=True,
        )

        row["terminal_policy"] = terminal_policy
        row["tactical_policy"] = tactical_policy
        row["policy_reason"] = terminal_policy["summary"]
        row["tactical_policy_reason"] = tactical_policy["summary"]
        row["policy_score"] = terminal_policy["policy_score"]
        row["tactical_policy_score"] = tactical_policy["policy_score"]

        terminal_edge = float(row.get("best_edge") or 0.0)
        tactical_edge = float(row.get("tactical_best_edge") or 0.0)
        if terminal_policy["allowed"]:
            if terminal_edge >= 0.06:
                row["recommendation"] = f"BUY_{row['best_side']}"
            elif terminal_edge >= 0.04:
                row["recommendation"] = f"LEAN_{row['best_side']}"
            else:
                row["recommendation"] = f"WATCH_{row['best_side']}"
            terminal_ranked.append(row)
        else:
            row["recommendation"] = "BLOCK"
            blocked_terminal_count += 1

        tactical_side = row.get("tactical_best_side") or row.get("best_side") or "YES"
        if tactical_policy["allowed"]:
            if tactical_edge >= 0.05:
                row["tactical_recommendation"] = f"BUY_{tactical_side}"
            elif tactical_edge >= 0.03:
                row["tactical_recommendation"] = f"LEAN_{tactical_side}"
            else:
                row["tactical_recommendation"] = f"WATCH_{tactical_side}"
            tactical_ranked.append(row)
        else:
            row["tactical_recommendation"] = "BLOCK"
            blocked_tactical_count += 1

    terminal_ranked.sort(
        key=lambda row: (
            float(row.get("policy_score") or -1.0),
            float(row.get("score") or -1.0),
        ),
        reverse=True,
    )
    tactical_ranked.sort(
        key=lambda row: (
            float(row.get("tactical_policy_score") or -1.0),
            float(row.get("tactical_score") or -1.0),
        ),
        reverse=True,
    )

    best_terminal_trade = next(
        (row for row in terminal_ranked if float(row.get("best_edge") or 0.0) > 0.0),
        None,
    )
    best_tactical_trade = next(
        (row for row in tactical_ranked if float(row.get("tactical_best_edge") or 0.0) > 0.0),
        None,
    )

    final_market_module = {
        "type": "terminal_distribution",
        "source": terminal_model["source"],
        "mean": round(float(terminal_model["mean_market"]), 4),
        "sigma": round(float(terminal_model["sigma_market"]), 4),
        "quality": round(float(terminal_model["confidence"]), 4),
        "top_bucket": top_label,
        "top_bucket_probability": round(float(top_label_prob), 6) if top_label_prob is not None else None,
        "top_bucket_complement_probability": round(float(top_label_complement_prob), 6) if top_label_complement_prob is not None else None,
        "expected_bucket": expected_label,
        "ceiling": terminal_model.get("market_ceiling"),
    }

    policy_summary = {
        "mode": "AUTO_GUARDRAILS",
        "terminal_allowed_count": len(terminal_ranked),
        "tactical_allowed_count": len(tactical_ranked),
        "blocked_terminal_count": blocked_terminal_count,
        "blocked_tactical_count": blocked_tactical_count,
        "headline": (
            f"{len(terminal_ranked)} ideas pass live policy"
            if terminal_ranked
            else "No trade passes the live policy right now"
        ),
    }
    forecast_winner = None
    if isinstance(top_bucket_row, dict):
        forecast_winner = {
            "label": top_label,
            "model_probability": round(float(top_bucket_row.get("fair_yes") or 0.0), 6),
            "market_probability": round(float(top_bucket_row.get("yes_entry")), 6) if top_bucket_row.get("yes_entry") is not None else None,
            "edge_points": round(float((top_bucket_row.get("edge_yes") or 0.0) * 100.0), 4),
        }

    return {
        "available": True,
        "station_id": station_id,
        "target_day": int(target_day),
        "target_date": target_date_obj.isoformat(),
        "market_unit": market_unit,
        "horizon": "INTRADAY" if int(target_day) == 0 else "DAY_AHEAD",
        "model": {
            "source": terminal_model["source"],
            "mean": round(float(terminal_model["mean_market"]), 4),
            "sigma": round(float(terminal_model["sigma_market"]), 4),
            "quality": round(float(terminal_model["confidence"]), 4),
            "confidence": round(float(terminal_model["confidence"]), 4),
            "observed_floor": terminal_model.get("market_floor"),
            "ceiling": terminal_model.get("market_ceiling"),
            "expected_label": expected_label,
            "top_label": top_label,
            "top_label_probability": round(float(top_label_prob), 6) if top_label_prob is not None else None,
            "top_label_complement_probability": round(float(top_label_complement_prob), 6) if top_label_complement_prob is not None else None,
            "top_labels": [
                {
                    "label": row["label"],
                    "probability": round(float(row["fair_prob"]), 6),
                }
                for row in top_labels[:5]
            ],
        },
        "forecast_winner": forecast_winner,
        "best_terminal_trade": best_terminal_trade,
        "best_tactical_trade": best_tactical_trade,
        "all_terminal_opportunities": opportunities,
        "terminal_opportunities": terminal_ranked[:8],
        "tactical_context": {
            "enabled": bool(tactical_map),
            "repricing_influence": round(float(tactical_influence), 4),
            "tactical_mean": round(float(tactical_mean), 4) if tactical_mean is not None else None,
            "next_metar": next_projection,
        },
        "policy": policy_summary,
        "modules": {
            "final_market": final_market_module,
            "next_official": next_projection,
            "market_pricing": {
                "type": "market_pricing",
                "policy": policy_summary,
                "repricing_influence": round(float(tactical_influence), 4),
                "best_terminal_trade": best_terminal_trade,
                "best_tactical_trade": best_tactical_trade,
            },
        },
    }
