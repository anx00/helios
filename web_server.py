# Load environment variables FIRST (before any other imports)
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import re
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

# Silence verbose loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger("web_server")

from database import (
    get_accuracy_report,
    get_latest_prediction,
    get_observed_max_for_target_date,
    iter_performance_history_by_target_date,
    get_telegram_station_subscription,
    set_telegram_station_subscription,
)
from config import STATIONS, get_active_stations, get_polymarket_temp_unit
import sys
try:
    # Prevent UnicodeEncodeError on Windows when parts of the stack use emoji/box-drawing
    # in debug prints (e.g., main.collect_and_predict).
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.append(".")
from main import collect_and_predict
from market.polymarket_checker import build_event_slug, fetch_event_data, check_market_resolution
from market.discovery import fetch_market_token_ids, fetch_market_token_ids_for_station_date, fetch_event_for_station_date
from market.polymarket_ws import get_ws_client
from opportunity import check_bet_opportunity
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from core.polymarket_labels import normalize_label, parse_label
from core.hourly_curve import (
    infer_synthetic_peak_hour,
    nudge_curve_to_target_max,
    shift_curve_toward_peak,
)
from core.market_pricing import select_probability_01_from_quotes

def build_synthetic_hourly_curve_f(
    now_utc: datetime,
    station_tz: ZoneInfo,
    current_temp_f: Optional[float],
    tmax_f: float,
    peak_hour: int = 14,
) -> List[float]:
    """
    Build a simple 24h hourly curve in °F that:
    - follows a gradual diurnal shape and passes near the current observed temperature
    - rises to `tmax_f` around `peak_hour`
    - cools modestly after peak

    This is a fallback used when we cannot fetch an hourly model curve (e.g. Open-Meteo 429s),
    but we do have a reliable daily forecast high from Wunderground (settlement source).
    """
    try:
        tmax = float(tmax_f)
    except Exception:
        return []

    anchor = None
    if current_temp_f is not None:
        try:
            anchor = float(current_temp_f)
        except Exception:
            anchor = None

    if anchor is None or anchor <= -900:
        anchor = tmax - 8.0

    # Keep anchor <= tmax; if we're already above forecasted max, clip.
    anchor = min(anchor, tmax)

    now_local = now_utc.astimezone(station_tz)
    anchor_hour = int(max(0, min(23, now_local.hour)))
    peak_hour = int(max(0, min(23, peak_hour)))

    # If we're already at/after the peak hour, force a near-peak anchor for shape stability.
    if anchor_hour >= peak_hour:
        anchor_hour = max(0, peak_hour - 1)

    delta = max(0.0, tmax - anchor)
    end_temp = anchor + (0.25 * delta)  # modest cooling by late evening

    # Avoid flattening all hours before the current anchor hour.
    diurnal_range = max(8.0, min(20.0, (tmax - anchor) + 10.0))
    overnight_low = min(anchor - 3.0, tmax - diurnal_range)
    midnight_temp = min(anchor - 1.0, overnight_low + 1.5)
    dawn_hour = 6

    def _interp(h: int, h0: int, t0: float, h1: int, t1: float) -> float:
        if h1 <= h0:
            return float(t1)
        x = (h - h0) / float(h1 - h0)
        return float(t0) + (float(t1) - float(t0)) * x

    curve: List[float] = []
    for h in range(24):
        if h <= anchor_hour:
            if anchor_hour == 0:
                t = anchor
            elif anchor_hour <= dawn_hour:
                t = _interp(h, 0, midnight_temp, max(1, anchor_hour), anchor)
            elif h <= dawn_hour:
                t = _interp(h, 0, midnight_temp, dawn_hour, overnight_low)
            else:
                t = _interp(h, dawn_hour, overnight_low, anchor_hour, anchor)
        elif h <= peak_hour:
            t = _interp(h, anchor_hour, anchor, peak_hour, tmax)
        else:
            t = _interp(h, peak_hour, tmax, 23, end_temp)
        curve.append(round(float(t), 1))

    return curve


def align_hourly_curve_with_wu(
    hourly_temps_f: List[float],
    wu_high_f: Optional[float],
    wu_peak_hour_local: Optional[int],
) -> tuple[List[float], str]:
    """
    Align hourly curve with WU settlement hints (timing + max), shape-preserving.
    """
    if not hourly_temps_f:
        return [], "HRRR"

    curve = [round(float(t), 1) for t in hourly_temps_f if t is not None]
    if not curve:
        return [], "HRRR"

    source_tags = ["HRRR"]

    if wu_peak_hour_local is not None:
        try:
            current_peak = int(max(range(len(curve)), key=lambda i: curve[i]))
            target_peak = int(wu_peak_hour_local)
            if abs(target_peak - current_peak) >= 2:
                curve = shift_curve_toward_peak(curve, target_peak, strength=0.75)
                source_tags.append("WU_TIME")
        except Exception:
            pass

    if wu_high_f is not None:
        try:
            current_max = max(curve)
            if abs(float(wu_high_f) - float(current_max)) >= 1.5:
                anchor = int(wu_peak_hour_local) if wu_peak_hour_local is not None else int(
                    max(range(len(curve)), key=lambda i: curve[i])
                )
                curve = nudge_curve_to_target_max(
                    curve,
                    float(wu_high_f),
                    anchor_peak_hour=anchor,
                    spread_hours=10.0,
                    max_delta_f=12.0,
                )
                source_tags.append("WU_MAX")
        except Exception:
            pass

    return curve, "+".join(source_tags)

def _default_station() -> str:
    """Return the first active station ID for use as default."""
    active = get_active_stations()
    return next(iter(active)) if active else "KLGA"

DEFAULT_STATION = _default_station()
SELECTED_STATION_COOKIE = "helios_selected_station"

_STATION_CITY_NAME_OVERRIDES = {
    "KLGA": "New York",
    "KATL": "Atlanta",
    "KORD": "Chicago",
    "KMIA": "Miami",
    "KDAL": "Dallas",
    "LFPG": "Paris",
    "EGLC": "London",
    "LTAC": "Ankara",
}


def _normalize_station_id(
    station_id: Optional[str],
    active: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not station_id:
        return None
    sid = str(station_id).strip().upper()
    if not sid:
        return None
    station_pool = active if active is not None else get_active_stations()
    return sid if sid in station_pool else None


def _station_city_name(station_id: str, station: Any) -> str:
    override = _STATION_CITY_NAME_OVERRIDES.get(station_id)
    if override:
        return override
    raw = str(getattr(station, "name", "") or "").strip()
    if not raw:
        return station_id
    return raw.split()[0]


def _build_station_selector_options(active: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    active_map = active if active is not None else get_active_stations()
    options: List[Dict[str, str]] = []
    for sid, stn in active_map.items():
        city_name = _station_city_name(sid, stn)
        options.append({
            "id": sid,
            "city_name": city_name,
            "label": f"{city_name} ({sid})",
            "name": stn.name,
            "timezone": stn.timezone,
            "market_unit": get_polymarket_temp_unit(sid),
        })
    return options


def _resolve_selected_station(request: Request, fallback: Optional[str] = None) -> str:
    active = get_active_stations()
    candidates = [
        request.query_params.get("station_id"),
        request.cookies.get(SELECTED_STATION_COOKIE),
        fallback,
        DEFAULT_STATION,
    ]
    for candidate in candidates:
        sid = _normalize_station_id(candidate, active)
        if sid:
            return sid
    return DEFAULT_STATION


def _set_selected_station_cookie(response: Response, station_id: str) -> None:
    response.set_cookie(
        key=SELECTED_STATION_COOKIE,
        value=station_id,
        max_age=60 * 60 * 24 * 365,
        samesite="lax",
        httponly=False,
        path="/",
    )


def _render_template(
    request: Request,
    template_name: str,
    context: Optional[Dict[str, Any]] = None,
) -> HTMLResponse:
    payload: Dict[str, Any] = dict(context or {})
    selected_station = _normalize_station_id(
        payload.get("selected_station_id") or payload.get("station_id")
    ) or _resolve_selected_station(request)
    payload["request"] = request
    payload.setdefault("selected_station_id", selected_station)
    response = templates.TemplateResponse(template_name, payload)
    if selected_station:
        _set_selected_station_cookie(response, selected_station)
    return response

app = FastAPI(title="Helios Weather Interface")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Add global context for all templates (active stations)
def get_template_context():
    """Global context available in all templates."""
    active = get_active_stations()
    return {
        "active_stations": list(active.keys()),
        "all_stations": STATIONS,
        "station_selector_options": _build_station_selector_options(active),
        "default_station_id": DEFAULT_STATION,
    }

# Inject into Jinja2 environment
templates.env.globals.update(get_template_context())

# Cache last-known token lists to survive intermittent Gamma API hiccups
_WS_TOKEN_CACHE: Dict[str, List[Tuple[str, str, str]]] = {}

# Models source of truth
class StationConfig(BaseModel):
    id: str
    name: str
    timezone: str


class TelegramMetarSubscriptionToggle(BaseModel):
    enabled: bool

@app.get("/api/stations")
def get_stations():
    """Return list of active stations."""
    active = get_active_stations()
    by_id = {opt["id"]: opt for opt in _build_station_selector_options(active)}
    return [
        {
            "id": sid,
            "name": stn.name,
            "city_name": by_id.get(sid, {}).get("city_name", _station_city_name(sid, stn)),
            "display_label": by_id.get(sid, {}).get("label", f"{sid}"),
            "timezone": stn.timezone,
            "market_unit": get_polymarket_temp_unit(sid),
        }
        for sid, stn in active.items()
    ]

def parse_physics_breakdown(physics_reason: str) -> list:
    """Parse the physics reason string into a structured list with values."""
    breakdown = []
    import re
    
    icon_map = {
        "Advection": "🌊 Advection",
        "Heating": "🔥 Heating Velocity",
        "Cooling": "❄️ Cooling Velocity",
        "Soil": "🌱 Soil Moisture",
        "Cloud": "☁️ Cloud Cover",
        "Breeze": "💨 Sea Breeze",
        "Smoke": "🌫️ Smoke/Haze",
        "Haze": "🌫️ Smoke/Haze",
        "SST": "🌡️ SST Anomaly",
        "Decay": "🌙 Sunset Decay",
        "Floor": "⚠️ Reality Check"
    }

    if physics_reason:
        parts = physics_reason.split(", ")
        for p in parts:
            p = p.strip()
            if not p: continue
            value = None
            match = re.search(r'([+-]?\d+\.?\d*)', p)
            if match:
                try:
                    value = float(match.group(1))
                except:
                    pass
            label = p.split("-")[0].split("+")[0]
            if "(" in label: label = label.split("(")[0]
            pretty_label = label
            for key, val in icon_map.items():
                if key in label:
                    pretty_label = val
                    break
            breakdown.append({"raw": p, "label": pretty_label, "original_label": label, "value": value})
    return breakdown

@app.get("/api/prediction/{station_id}")
async def get_prediction_api(station_id: str, target_day: int = 0):
    """Get latest prediction + live market data for a station.
    
    Args:
        station_id: Station identifier (e.g., KLGA)
        target_day: 0 for today (default), 1 for tomorrow
    """
    if station_id not in STATIONS:
        return {"error": "Invalid station"}
    
    # Get the proper target date based on station timezone
    station = STATIONS[station_id]
    local_now = datetime.now(ZoneInfo(station.timezone))
    target_date = (local_now + timedelta(days=target_day)).date()
    target_date_str = target_date.isoformat()
    
    # Get prediction for the specific target date
    from database import get_latest_prediction_for_date
    pred = get_latest_prediction_for_date(station_id, target_date_str)
    
    # Fallback to most recent if no prediction for target date
    if not pred:
        pred = get_latest_prediction(station_id)
        if not pred:
            return {"error": "No prediction found", "target_date": target_date_str}
    
    breakdown = parse_physics_breakdown(pred.get("physics_reason", ""))
    opp = None
    try:
        event_data = await fetch_event_for_station_date(station_id, target_date)
        if not event_data:
            event_slug = build_event_slug(station_id, target_date)
            event_data = await fetch_event_data(event_slug)
        polymarket_options = []
        if event_data:
            markets = event_data.get("markets", [])
            for market in markets:
                outcome_prices_raw = market.get("outcomePrices", [])
                try:
                    import json
                    if isinstance(outcome_prices_raw, str):
                        outcome_prices = json.loads(outcome_prices_raw)
                    else:
                        outcome_prices = outcome_prices_raw
                    if outcome_prices and len(outcome_prices) > 0:
                        yes_price = float(outcome_prices[0])
                        outcome_title = market.get("groupItemTitle", "")
                        polymarket_options.append((outcome_title, yes_price))
                except:
                    continue
            polymarket_options.sort(key=lambda x: x[1], reverse=True)
            if polymarket_options:
                opp_data = check_bet_opportunity(pred["final_prediction_f"], polymarket_options, station_id, target_date)
                if opp_data:
                    opp = {"type": opp_data.confidence, "diff": opp_data.difference, "market_fav": opp_data.market_favorite, "market_prob": opp_data.market_favorite_prob, "recommendation": opp_data.recommendation, "roi": opp_data.estimated_roi}
    except Exception as e:
        print(f"Market fetch error: {e}")
    
    return {
        "station_id": station_id, 
        "target_date": target_date_str,
        "target_day": target_day,
        "timestamp": pred["timestamp"], 
        "prediction_f": pred["final_prediction_f"], 
        "hrrr_f": pred["hrrr_max_raw_f"], 
        "raw_deviation_f": pred["current_deviation_f"], 
        "delta_weight": pred.get("delta_weight", 1.0), 
        "physics_delta": pred["physics_adjustment_f"], 
        "component_breakdown": breakdown, 
        "opportunity": opp
    }

@app.get("/api/market/{station_id}")
async def get_market_data(station_id: str, target_day: int = 0):
    """Get live Polymarket data for a station.
    
    Args:
        station_id: Station identifier
        target_day: 0 for today (default), 1 for tomorrow
    """
    if station_id not in STATIONS: return {"error": "Invalid station"}
    station = STATIONS[station_id]
    local_now = datetime.now(ZoneInfo(station.timezone))
    target_date = (local_now + timedelta(days=target_day)).date()
    event_data = await fetch_event_for_station_date(station_id, target_date)
    if not event_data:
        event_slug = build_event_slug(station_id, target_date)
        event_data = await fetch_event_data(event_slug)
        if not event_data:
            return {"error": "Market not found", "slug": event_slug, "target_date": target_date.isoformat()}
    else:
        event_slug = build_event_slug(station_id, target_date)
    event_title = event_data.get("title", "Unknown Event")
    markets = event_data.get("markets", [])
    total_volume = 0.0
    outcomes = []
    import json
    for market in markets:
        outcome_title = market.get("groupItemTitle", "Unknown")
        try: volume = float(market.get("volume", "0"))
        except: volume = 0.0
        total_volume += volume
        outcome_prices_raw = market.get("outcomePrices", [])
        try:
            if isinstance(outcome_prices_raw, str): outcome_prices = json.loads(outcome_prices_raw)
            else: outcome_prices = outcome_prices_raw
            yes_price = float(outcome_prices[0]) if outcome_prices else 0.0
        except: yes_price = 0.0
        outcomes.append({"title": outcome_title, "probability": yes_price, "volume": volume})
    outcomes.sort(key=lambda x: x["probability"], reverse=True)
    return {"station_id": station_id, "target_date": target_date.isoformat(), "target_day": target_day, "event_title": event_title, "event_slug": event_slug, "total_volume": total_volume, "outcomes": outcomes}

@app.on_event("startup")
async def startup_event():
    """Start background monitoring tasks."""
    from database import init_database
    from core.telegram_bot import build_telegram_bot_from_env
    init_database()
    # Task 0: Immediate PWS fetch (no delay)
    asyncio.create_task(pws_loop())
    # Task 1: Heavy predictions (every 5 mins)
    asyncio.create_task(prediction_loop())
    # Task 2: Fast market/state snapshots (every 1 min for 'continuous' look)
    asyncio.create_task(snapshot_loop())
    # Task 3: Real-time WebSocket Mirror
    asyncio.create_task(websocket_loop())
    # Task 4: Phase 3 Nowcast Integration
    asyncio.create_task(nowcast_loop())
    # Task 5: Phase 4 Recorder
    asyncio.create_task(recorder_loop())
    # Task 6: Phase 6 Autotrader (paper-first)
    asyncio.create_task(autotrader_loop())
    # Task 7: Nightly offline learning cycle
    asyncio.create_task(learning_nightly_loop())
    # Task 8: Telegram bot (optional, enabled with TELEGRAM_BOT_TOKEN)
    telegram_bot = build_telegram_bot_from_env()
    if telegram_bot:
        asyncio.create_task(telegram_bot.run_forever())
        logger.info("Telegram bot enabled")

async def _resolve_station_tokens(
    station_id: str,
    target_date,
    now_local: datetime,
    logger: logging.Logger,
) -> List[Tuple[str, str, str]]:
    """Resolve token IDs for a station/date with robust fallbacks and caching."""
    def _parse_tokens_from_event_payload(event_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        rows: List[Tuple[str, str, str]] = []
        if not isinstance(event_data, dict):
            return rows
        markets = event_data.get("markets", [])
        if not isinstance(markets, list):
            return rows
        for market in markets:
            if not isinstance(market, dict):
                continue
            bracket = str(market.get("groupItemTitle") or market.get("title") or "").strip() or "Unknown"
            raw = market.get("clobTokenIds", [])
            ids: List[str] = []
            if isinstance(raw, str):
                try:
                    import json
                    parsed = json.loads(raw)
                except Exception:
                    parsed = []
                if isinstance(parsed, list):
                    ids = [str(x) for x in parsed if x]
            elif isinstance(raw, list):
                ids = [str(x) for x in raw if x]

            yes_id = ids[0] if len(ids) > 0 else None
            no_id = ids[1] if len(ids) > 1 else None
            if yes_id:
                rows.append((yes_id, bracket, "yes"))
            if no_id:
                rows.append((no_id, bracket, "no"))
        return rows

    tokens: List[Tuple[str, str, str]] = []

    # Candidate dates (ordered, deduped): target -> today -> tomorrow
    today = now_local.date()
    tomorrow = today + timedelta(days=1)
    candidate_dates: List[Any] = []
    for cand in [target_date, today, tomorrow]:
        if cand not in candidate_dates:
            candidate_dates.append(cand)

    for cand_date in candidate_dates:
        # Primary: station/date resolver
        tokens = await fetch_market_token_ids_for_station_date(station_id, cand_date)
        if tokens:
            break

        # Secondary: slug resolver
        slug = build_event_slug(station_id, cand_date)
        tokens = await fetch_market_token_ids(slug)
        if tokens:
            break

        # Tertiary: raw event payload parser (resilient to discovery regressions)
        event_data = await fetch_event_for_station_date(station_id, cand_date)
        if event_data:
            tokens = _parse_tokens_from_event_payload(event_data)
            if tokens:
                break

    if not tokens and station_id in _WS_TOKEN_CACHE:
        logger.warning(f"WS token fetch failed; using cached tokens for {station_id}")
        tokens = _WS_TOKEN_CACHE[station_id]

    if tokens:
        _WS_TOKEN_CACHE[station_id] = tokens

    return tokens


def _parse_market_info(raw_info: str) -> Tuple[str, str, str]:
    """
    Parse market metadata string.
    Format:
    - new: "STATION|BRACKET|OUTCOME"
    - legacy: "STATION|BRACKET" (assume YES)
    """
    if not raw_info:
        return "Unknown", "Unknown", "unknown"

    parts = raw_info.split("|")
    if len(parts) >= 3:
        return parts[0], parts[1], (parts[2] or "unknown").lower()
    if len(parts) == 2:
        return parts[0], parts[1], "yes"
    return "Unknown", raw_info, "unknown"


async def pws_loop():
    """Fetch PWS cluster data immediately and every 2 minutes."""
    from config import get_active_stations
    from collector.pws_fetcher import fetch_and_publish_pws
    from collector.metar_fetcher import fetch_metar_race
    from core.recorder import get_recorder
    logger = logging.getLogger("pws_loop")
    logger.info("Starting PWS loop (immediate + 2-min intervals)...")
    recorder = get_recorder()

    while True:
        try:
            active = get_active_stations()
            for station_id in active.keys():
                # Get current METAR for drift calculation
                official_temp_c = None
                official_obs_time_utc = None
                try:
                    metar = await fetch_metar_race(station_id)
                    if metar:
                        official_temp_c = metar.temp_c
                        official_obs_time_utc = metar.observation_time
                except Exception as e:
                    logger.debug(f"METAR fetch for PWS drift failed: {e}")

                logger.info(f"PWS fetch for {station_id}...")
                consensus = await fetch_and_publish_pws(
                    station_id,
                    official_temp_c=official_temp_c,
                    official_obs_time_utc=official_obs_time_utc,
                )
                if consensus:
                    qc_state = "OK"
                    if consensus.outliers or consensus.support < 5 or consensus.mad > 2.0:
                        qc_state = "UNCERTAIN"
                    pws_ids = [r.label for r in consensus.readings]
                    all_readings = consensus.readings + consensus.outliers
                    pws_all_ids = [r.label for r in all_readings]

                    def _serialize_reading(r):
                        sid = str(r.label).upper()
                        profile = consensus.learning_profiles.get(sid, {})
                        return {
                            "station_id": r.label,
                            "station_name": r.station_name,
                            "source": r.source,
                            "obs_time_utc": r.obs_time_utc.isoformat() if r.obs_time_utc else None,
                            "temp_c": round(float(r.temp_c), 3),
                            "temp_f": round((float(r.temp_c) * 9.0 / 5.0) + 32.0, 1),
                            "distance_km": round(float(r.distance_km), 3),
                            "age_minutes": round(float(r.age_minutes), 3),
                            "valid": bool(r.valid),
                            "qc_flag": r.qc_flag,
                            "weight": profile.get("weight", consensus.learning_weights.get(sid)),
                            "weight_now": profile.get("weight_now"),
                            "weight_predictive": profile.get("weight_predictive"),
                            "now_score": profile.get("now_score"),
                            "lead_score": profile.get("lead_score"),
                            "predictive_score": profile.get("predictive_score"),
                            "now_samples": profile.get("now_samples"),
                            "lead_samples": profile.get("lead_samples"),
                            "quality_band": profile.get("quality_band"),
                            "rank_eligible": profile.get("rank_eligible"),
                            "learning_phase": profile.get("learning_phase"),
                            "samples_total": profile.get("samples_total"),
                            "rank_warmup_remaining_now": profile.get("rank_warmup_remaining_now"),
                            "rank_warmup_remaining_lead": profile.get("rank_warmup_remaining_lead"),
                            "rank_min_now_samples": profile.get("rank_min_now_samples"),
                            "rank_min_lead_samples": profile.get("rank_min_lead_samples"),
                        }

                    pws_readings = [_serialize_reading(r) for r in consensus.readings]
                    pws_outliers = [_serialize_reading(r) for r in consensus.outliers]
                    mad_f = round(consensus.mad * 9 / 5, 2)
                    asyncio.create_task(recorder.record_pws(
                        station_id=station_id,
                        median_f=consensus.median_temp_f,
                        mad_f=mad_f,
                        support=consensus.support,
                        pws_ids=pws_ids,
                        pws_all_ids=pws_all_ids,
                        pws_readings=pws_readings,
                        pws_outliers=pws_outliers,
                        station_weights=consensus.learning_weights,
                        station_learning=consensus.learning_profiles,
                        weighted_support=consensus.weighted_support,
                        qc_state=qc_state,
                        obs_time_utc=consensus.obs_time_utc
                    ))

            await asyncio.sleep(120)  # Every 2 minutes
        except Exception as e:
            logger.error(f"PWS loop error: {e}")
            await asyncio.sleep(30)

async def websocket_loop():
    """Maintain WebSocket connection and subscriptions."""
    from config import get_active_stations
    from market.polymarket_checker import get_target_date
    logger = logging.getLogger("ws_loop")
    client = await get_ws_client()

    # 1. Start listening (blocking)
    asyncio.create_task(client.listen())

    # 2. Periodically refresh subscriptions (every 5 mins)
    while True:
        try:
            # Gather all tokens we need to track
            # (For now, just today's markets for active stations)
            all_token_ids = []
            active = get_active_stations()

            for station_id in active:
                now = datetime.now(ZoneInfo(active[station_id].timezone))
                target_date = await get_target_date(station_id, now)
                tokens = await _resolve_station_tokens(station_id, target_date, now, logger)
                if not tokens:
                    logger.warning(f"WS token resolution failed for {station_id} ({target_date}); skipping")
                    continue

                for tid, bracket, outcome in tokens:
                    all_token_ids.append(tid)
                    # Register metadata
                    client.set_market_info(tid, f"{station_id}|{bracket}|{outcome}")
            
            if all_token_ids:
                await client.subscribe_to_markets(all_token_ids)
                
            await asyncio.sleep(300) # Refresh every 5 mins
        except Exception as e:
            logger.error(f"WS Loop error: {e}")
            await asyncio.sleep(60)
async def nowcast_loop():
    """Phase 3: Nowcast Engine integration loop."""
    import asyncio
    from collector.hrrr_fetcher import fetch_hrrr
    from collector.metar_fetcher import fetch_metar_history
    from collector.wunderground_fetcher import fetch_wunderground_max
    from core.nowcast_integration import (
        extract_hourly_temps_f_for_date,
        get_nowcast_integration,
        initialize_nowcast_integration,
    )
    from core.judge import JudgeAlignment
    from config import get_active_stations
    logger = logging.getLogger("nowcast_loop")
    logger.info("Starting Phase 3 Nowcast Engine...")

    try:
        # Initialize with active stations
        active = get_active_stations()
        await initialize_nowcast_integration(
            stations=list(active.keys()),
            periodic_interval=60.0
        )
        logger.info(f"Nowcast Integration initialized for stations: {list(active.keys())}")

        # ------------------------------------------------------------------
        # BaseForecast bootstrap/refresh
        # ------------------------------------------------------------------
        integration = get_nowcast_integration()

        # ------------------------------------------------------------------
        # METAR Max Backfill (NOAA history)
        # ------------------------------------------------------------------
        # Rationale: our event-driven METAR ingest can miss an observation due to transient
        # network/race failures. That breaks the hard constraint: Tmax >= max_observed.
        #
        # This loop periodically reconciles "max so far" using NOAA's historical feed,
        # and updates the nowcast DailyState if we detect a higher observed max (or a newer
        # official observation).
        async def _backfill_noaa_metar_state_for_station(station_id: str) -> bool:
            station = STATIONS.get(station_id)
            if not station:
                return False

            engine = integration.get_engine(station_id)
            now_utc = datetime.now(ZoneInfo("UTC"))

            try:
                target_date = engine._get_target_date(now_utc)
            except Exception:
                station_tz = ZoneInfo(station.timezone)
                target_date = datetime.now(station_tz).date()

            station_tz = ZoneInfo(station.timezone)
            now_local = now_utc.astimezone(station_tz)

            # Fetch enough hours to cover from midnight local for the *target_date*.
            # If the target_date differs (rare), just grab a full day.
            if now_local.date() == target_date:
                hours_since_midnight = now_local.hour + 1
                hours = max(hours_since_midnight + 2, 6)
            else:
                hours = 26

            try:
                history = await fetch_metar_history(station_id, hours=hours)
            except Exception:
                return False

            obs: List[tuple[datetime, float, float]] = []
            for h in history or []:
                try:
                    h_local = h.observation_time.astimezone(station_tz)
                    if h_local.date() != target_date:
                        continue
                    if h.temp_c is None:
                        continue
                    temp_f_raw = float(JudgeAlignment.celsius_to_fahrenheit(h.temp_c))
                    temp_f_aligned = float(JudgeAlignment.round_to_settlement(temp_f_raw))
                    obs.append((h.observation_time, temp_f_raw, temp_f_aligned))
                except Exception:
                    continue

            if not obs:
                return False

            obs.sort(key=lambda x: x[0])
            latest_time, latest_raw_f, latest_aligned_f = obs[-1]
            max_time, max_raw_f, max_aligned_f = max(obs, key=lambda x: x[1])

            state = engine.get_or_create_daily_state(target_date=target_date)

            updated = False
            if max_aligned_f > state.max_so_far_aligned_f or max_raw_f > state.max_so_far_raw_f:
                state.update_max(max_aligned_f, max_raw_f, max_time)
                updated = True

            if state.last_obs_time_utc is None or latest_time > state.last_obs_time_utc:
                state.last_obs_time_utc = latest_time
                state.last_obs_temp_f = latest_raw_f
                state.last_obs_aligned_f = latest_aligned_f
                state.last_obs_source = "NOAA_HISTORY_BACKFILL"
                state.last_update_utc = datetime.now(ZoneInfo("UTC"))
                updated = True

            if updated:
                try:
                    engine.generate_distribution()
                except Exception:
                    pass

            return updated

        async def _noaa_metar_backfill_loop() -> None:
            while True:
                try:
                    active_now = get_active_stations()
                    tasks = []
                    for sid in list(active_now.keys()):
                        tasks.append(_backfill_noaa_metar_state_for_station(sid))
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    logger.warning("Nowcast METAR backfill loop error: %s", e)
                await asyncio.sleep(120)

        # Run once right away, then keep reconciling in the background.
        try:
            active_now = get_active_stations()
            await asyncio.gather(
                *[_backfill_noaa_metar_state_for_station(sid) for sid in list(active_now.keys())],
                return_exceptions=True,
            )
        except Exception:
            pass
        asyncio.create_task(_noaa_metar_backfill_loop())

        async def _refresh_base_forecast_for_station(station_id: str) -> bool:
            station = active.get(station_id)
            if not station:
                return False

            engine = integration.get_engine(station_id)
            now_utc = datetime.now(ZoneInfo("UTC"))
            try:
                target_date = engine._get_target_date(now_utc)  # Use settlement alignment logic.
            except Exception:
                station_tz = ZoneInfo(station.timezone)
                target_date = datetime.now(station_tz).date()

            # Skip refresh if we already have a valid base forecast for today.
            base = engine.get_base_forecast()
            if base and base.is_valid() and base.target_date == target_date:
                return False

            station_tz = ZoneInfo(station.timezone)

            # Fetch WU high first (settlement-aligned). We'll use it for both:
            # - HRRR curve peak adjustment when available
            # - Fallback base curve if Open-Meteo is rate-limited/unavailable
            wu_high: Optional[float] = None
            wu_current: Optional[float] = None
            wu_peak_hour: Optional[int] = None
            try:
                wu = await fetch_wunderground_max(station_id, target_date=target_date)
                wu_high = float(wu.forecast_high_f) if wu and wu.forecast_high_f is not None else None
                wu_current = float(wu.current_temp_f) if wu and wu.current_temp_f is not None else None
                wu_peak_hour = int(wu.forecast_peak_hour_local) if wu and wu.forecast_peak_hour_local is not None else None
            except Exception as e:
                logger.warning("Nowcast BaseForecast WU fetch failed (%s): %s", station_id, e)

            forecast = await fetch_hrrr(station.latitude, station.longitude, target_days_ahead=0)
            if not forecast or not getattr(forecast, "hourly_temps_c", None):
                if wu_high is None:
                    logger.warning("Nowcast BaseForecast refresh failed (%s): missing hourly HRRR data", station_id)
                    return False

                # Fallback: build a synthetic hourly curve anchored to current obs and WU forecast high.
                anchor_temp_f: Optional[float] = None
                try:
                    state = engine.get_or_create_daily_state(target_date=target_date)
                    if state.last_obs_time_utc:
                        anchor_temp_f = float(state.last_obs_temp_f or state.last_obs_aligned_f)
                except Exception:
                    pass
                if anchor_temp_f is None:
                    anchor_temp_f = wu_current

                fallback_peak_hour = infer_synthetic_peak_hour(
                    now_utc=now_utc,
                    station_tz=station_tz,
                    current_temp_f=anchor_temp_f,
                    tmax_f=float(wu_high),
                    hinted_peak_hour=wu_peak_hour,
                )
                hourly_temps_f = build_synthetic_hourly_curve_f(
                    now_utc=now_utc,
                    station_tz=station_tz,
                    current_temp_f=anchor_temp_f,
                    tmax_f=float(wu_high),
                    peak_hour=fallback_peak_hour,
                )
                if not hourly_temps_f:
                    logger.warning("Nowcast BaseForecast WU fallback failed (%s): could not build synthetic curve", station_id)
                    return False

                integration.on_hrrr_update(
                    station_id=station_id,
                    hourly_temps_f=hourly_temps_f,
                    model_run_time=now_utc,
                    model_source="WU_FORECAST",
                    target_date=target_date,
                )
                logger.info(
                    "Nowcast BaseForecast updated (%s): tmax_base=%.1fF (WU fallback)",
                    station_id,
                    max(hourly_temps_f),
                )
                return True

            hourly_temps_f = extract_hourly_temps_f_for_date(
                forecast.hourly_temps_c,
                getattr(forecast, "hourly_times", None),
                target_date,
                station_tz,
            )
            if not hourly_temps_f:
                if wu_high is None:
                    logger.warning("Nowcast BaseForecast refresh failed (%s): could not extract 24h curve", station_id)
                    return False

                anchor_temp_f: Optional[float] = None
                try:
                    state = engine.get_or_create_daily_state(target_date=target_date)
                    if state.last_obs_time_utc:
                        anchor_temp_f = float(state.last_obs_temp_f or state.last_obs_aligned_f)
                except Exception:
                    pass
                if anchor_temp_f is None:
                    anchor_temp_f = wu_current

                fallback_peak_hour = infer_synthetic_peak_hour(
                    now_utc=now_utc,
                    station_tz=station_tz,
                    current_temp_f=anchor_temp_f,
                    tmax_f=float(wu_high),
                    hinted_peak_hour=wu_peak_hour,
                )
                hourly_temps_f = build_synthetic_hourly_curve_f(
                    now_utc=now_utc,
                    station_tz=station_tz,
                    current_temp_f=anchor_temp_f,
                    tmax_f=float(wu_high),
                    peak_hour=fallback_peak_hour,
                )
                if not hourly_temps_f:
                    logger.warning("Nowcast BaseForecast WU fallback failed (%s): could not build synthetic curve", station_id)
                    return False

                integration.on_hrrr_update(
                    station_id=station_id,
                    hourly_temps_f=hourly_temps_f,
                    model_run_time=now_utc,
                    model_source="WU_FORECAST",
                    target_date=target_date,
                )
                logger.info(
                    "Nowcast BaseForecast updated (%s): tmax_base=%.1fF (WU fallback)",
                    station_id,
                    max(hourly_temps_f),
                )
                return True

            # WU is the settlement source for these markets. Align both timing and daily max
            # while preserving the HRRR/Open-Meteo shape.
            model_source = "HRRR"
            try:
                aligned_curve, aligned_source = align_hourly_curve_with_wu(
                    hourly_temps_f=hourly_temps_f,
                    wu_high_f=wu_high,
                    wu_peak_hour_local=wu_peak_hour,
                )
                if aligned_curve:
                    hourly_temps_f = aligned_curve
                    model_source = aligned_source
            except Exception as e:
                logger.warning("Nowcast BaseForecast WU align failed (%s): %s", station_id, e)

            integration.on_hrrr_update(
                station_id=station_id,
                hourly_temps_f=hourly_temps_f,
                model_run_time=forecast.fetch_time,
                model_source=model_source,
                target_date=target_date,
            )
            try:
                peak_hour_now = int(max(range(len(hourly_temps_f)), key=lambda i: hourly_temps_f[i]))
            except Exception:
                peak_hour_now = -1
            logger.info(
                "Nowcast BaseForecast updated (%s): tmax_base=%.1fF peak=%02d:00 source=%s",
                station_id,
                max(hourly_temps_f),
                peak_hour_now,
                model_source,
            )
            return True

        # Keep BaseForecast fresh.
        # If we're missing/stale (common right after boot or after transient API errors),
        # retry quickly to avoid early-day nowcasts using the naive fallback.
        sleep_s: int = 0
        while True:
            if sleep_s:
                await asyncio.sleep(sleep_s)

            try:
                active = get_active_stations()
                missing: List[str] = []

                for sid in list(active.keys()):
                    try:
                        await _refresh_base_forecast_for_station(sid)
                    except Exception as e:
                        logger.warning("Nowcast BaseForecast refresh failed (%s): %s", sid, e)

                    try:
                        station = active.get(sid)
                        if not station:
                            missing.append(sid)
                            continue

                        engine = integration.get_engine(sid)
                        now_utc = datetime.now(ZoneInfo("UTC"))
                        try:
                            target_date = engine._get_target_date(now_utc)
                        except Exception:
                            station_tz = ZoneInfo(station.timezone)
                            target_date = datetime.now(station_tz).date()

                        base = engine.get_base_forecast()
                        ok = bool(
                            base
                            and base.is_valid()
                            and base.target_date == target_date
                            and getattr(base, "t_hourly_f", None)
                        )
                        if not ok:
                            missing.append(sid)
                    except Exception:
                        missing.append(sid)

                sleep_s = 60 if missing else (20 * 60)
            except Exception as e:
                logger.warning("Nowcast BaseForecast refresh loop error: %s", e)
                sleep_s = 60
    except Exception as e:
        logger.error(f"Nowcast integration failed: {e}")


async def recorder_loop():
    """Phase 4: Initialize and run the event recorder."""
    from core.recorder import initialize_recorder, get_recorder
    from core.event_window import get_event_window_manager
    from core.nowcast_integration import get_nowcast_integration
    logger = logging.getLogger("recorder_loop")
    logger.info("Starting Phase 4 Recorder...")

    try:
        # Initialize recorder
        recorder = await initialize_recorder("data/recordings")

        # Get integration to register for events
        integration = get_nowcast_integration()
        window_manager = get_event_window_manager()

        # Register callback to record nowcast outputs
        def on_nowcast_update(station_id: str, distribution):
            if distribution:
                import asyncio
                asyncio.create_task(recorder.record_nowcast(
                    station_id=station_id,
                    tmax_mean_f=distribution.tmax_mean_f,
                    tmax_sigma_f=distribution.tmax_sigma_f,
                    p_bucket=[{"label": b.label, "prob": b.probability} for b in distribution.p_bucket],
                    t_peak_bins=[{"label": b.label, "prob": b.probability} for b in distribution.t_peak_bins],
                    confidence=distribution.confidence,
                    qc_state="OK",
                    bias_f=distribution.bias_applied_f
                ))

        integration.register_callback(on_nowcast_update)

        # Register callback to record METAR observations (for backtest ground truth)
        def on_metar_observation(obs):
            if obs:
                import asyncio
                asyncio.create_task(recorder.record_metar(
                    station_id=obs.station_id,
                    raw=getattr(obs, 'raw', ''),
                    temp_c=obs.temp_c,
                    temp_f=obs.temp_f,
                    temp_aligned=getattr(obs, 'temp_f_raw', obs.temp_f),
                    obs_time_utc=obs.obs_time_utc,
                    dewpoint_c=obs.dewpoint_c,
                    wind_dir=obs.wind_dir,
                    wind_speed=obs.wind_speed,
                    source=getattr(obs, 'source', 'METAR'),
                    report_type=getattr(obs, 'report_type', 'METAR'),
                    is_speci=bool(getattr(obs, 'is_speci', False)),
                ))
                try:
                    obs_source = getattr(obs, 'source', 'METAR')
                    obs_report_type = str(getattr(obs, 'report_type', 'METAR') or 'METAR').upper()
                    source_label = f"{obs_source}[SPECI]" if obs_report_type == "SPECI" else obs_source
                    asyncio.create_task(window_manager.on_metar(
                        station_id=obs.station_id,
                        temp_f=obs.temp_f,
                        source=source_label
                    ))
                    qc_flags = getattr(obs, 'qc_flags', []) or []
                    if qc_flags or not getattr(obs, 'qc_passed', True):
                        asyncio.create_task(window_manager.on_qc_outlier(
                            station_id=obs.station_id,
                            flags=qc_flags
                        ))
                except Exception:
                    pass

        integration.register_metar_callback(on_metar_observation)

        # Register window open callback to record
        def on_window_open(window):
            import asyncio
            asyncio.create_task(recorder.record_event_window(
                window_id=window.window_id,
                action="START",
                reason=window.reason,
                station_id=window.station_id
            ))

        def on_window_close(window):
            import asyncio
            asyncio.create_task(recorder.record_event_window(
                window_id=window.window_id,
                action="END",
                reason=f"Duration: {window.duration_seconds:.1f}s",
                station_id=window.station_id
            ))

        window_manager.on_window_open(on_window_open)
        window_manager.on_window_close(on_window_close)

        async def l2_snapshot_loop():
            from config import get_active_stations
            from market.polymarket_ws import get_ws_client
            logger = logging.getLogger("l2_snapshot_loop")

            while True:
                try:
                    active = get_active_stations()
                    if not active:
                        await asyncio.sleep(5)
                        continue

                    client = await get_ws_client()
                    if not client.state or not hasattr(client.state, "orderbooks"):
                        await asyncio.sleep(1)
                        continue

                    station_books = {sid: {} for sid in active.keys()}
                    station_staleness = {sid: [] for sid in active.keys()}

                    for token_id, book in client.state.orderbooks.items():
                        raw_info = client.market_info.get(token_id, "")
                        m_station, bracket, outcome = _parse_market_info(raw_info)
                        if m_station not in active:
                            continue

                        snap = book.get_l2_snapshot(top_n=10)
                        if not snap:
                            continue

                        bucket = station_books[m_station].setdefault(bracket, {})
                        side = str(outcome or "").lower()
                        if side == "yes":
                            bucket.update({
                                "token_id": token_id,  # Backward-compatible YES token
                                "yes_token_id": token_id,
                                "best_bid": snap.get("best_bid"),
                                "best_ask": snap.get("best_ask"),
                                "spread": snap.get("spread"),
                                "mid": snap.get("mid"),
                                "bid_depth": snap.get("bid_depth"),
                                "ask_depth": snap.get("ask_depth"),
                                "bids": snap.get("bids"),
                                "asks": snap.get("asks"),
                                "staleness_ms": snap.get("staleness_ms"),
                                "ts_provider": snap.get("ts_provider"),
                                "yes_best_bid": snap.get("best_bid"),
                                "yes_best_ask": snap.get("best_ask"),
                                "yes_spread": snap.get("spread"),
                                "yes_mid": snap.get("mid"),
                                "yes_bid_depth": snap.get("bid_depth"),
                                "yes_ask_depth": snap.get("ask_depth"),
                                "yes_staleness_ms": snap.get("staleness_ms"),
                                "yes_ts_provider": snap.get("ts_provider"),
                            })
                        elif side == "no":
                            bucket.update({
                                "no_token_id": token_id,
                                "no_best_bid": snap.get("best_bid"),
                                "no_best_ask": snap.get("best_ask"),
                                "no_spread": snap.get("spread"),
                                "no_mid": snap.get("mid"),
                                "no_bid_depth": snap.get("bid_depth"),
                                "no_ask_depth": snap.get("ask_depth"),
                                "no_bids": snap.get("bids"),
                                "no_asks": snap.get("asks"),
                                "no_staleness_ms": snap.get("staleness_ms"),
                                "no_ts_provider": snap.get("ts_provider"),
                            })
                        else:
                            continue

                        staleness_val = snap.get("staleness_ms")
                        if staleness_val is not None:
                            station_staleness[m_station].append(staleness_val)

                    metrics = client.state.metrics if client and client.state else {}

                    for station_id, buckets in station_books.items():
                        if not buckets:
                            continue

                        staleness_vals = station_staleness.get(station_id, [])
                        meta = {
                            "book_count": len(buckets),
                            "staleness_ms_avg": round(sum(staleness_vals) / len(staleness_vals), 1) if staleness_vals else None,
                            "staleness_ms_max": max(staleness_vals) if staleness_vals else None,
                            "publisher_lag_ms": metrics.get("publisher_lag_ms") if metrics else None,
                            "resync_count": metrics.get("resync_count") if metrics else None,
                            "reconnect_count": metrics.get("reconnect_count") if metrics else None
                        }

                        buckets["__meta__"] = meta
                        asyncio.create_task(recorder.record_l2_snap(
                            station_id=station_id,
                            market_state=buckets
                        ))

                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"L2 snapshot loop error: {e}")
                    await asyncio.sleep(2)

        async def features_health_loop():
            from config import get_active_stations
            from core.world import get_world
            from market.polymarket_ws import get_ws_client
            logger = logging.getLogger("features_health_loop")
            utc = ZoneInfo("UTC")

            while True:
                try:
                    active = get_active_stations()
                    if not active:
                        await asyncio.sleep(30)
                        continue

                    world = get_world()
                    snapshot = world.get_snapshot()
                    now_utc = datetime.now(utc)

                    env = snapshot.get("env", {})
                    env_features = {}
                    staleness = {}

                    for key, data in env.items():
                        env_features[key] = data
                        ts_str = data.get("obs_time_utc") or data.get("ingest_time_utc")
                        if ts_str:
                            try:
                                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                staleness[key] = round((now_utc - ts).total_seconds(), 1)
                            except Exception:
                                staleness[key] = None
                        else:
                            staleness[key] = None

                    client = await get_ws_client()
                    metrics = client.state.metrics if client and client.state else {}
                    ws_connected = client.is_connected if client else False

                    for station_id in active.keys():
                        latencies = {}
                        sources_status = {}

                        for source_name, health in snapshot.get("health", {}).items():
                            if source_name.startswith("ENV_") or station_id in source_name:
                                latencies[source_name] = health.get("source_age_s", health.get("source_age", 0))
                                sources_status[source_name] = health.get("status", "UNKNOWN")

                        latencies["ws_publisher_lag_ms"] = metrics.get("publisher_lag_ms") if metrics else None
                        sources_status["ws_connected"] = "OK" if ws_connected else "DISCONNECTED"

                        asyncio.create_task(recorder.record_features(
                            station_id=station_id,
                            features={
                                "env": env_features,
                                "qc": snapshot.get("qc", {}).get(station_id)
                            },
                            staleness=staleness
                        ))

                        asyncio.create_task(recorder.record_health(
                            station_id=station_id,
                            latencies=latencies,
                            sources_status=sources_status,
                            reconnects=metrics.get("reconnect_count", 0),
                            gaps=metrics.get("resync_count", 0)
                        ))

                    await asyncio.sleep(60)

                except Exception as e:
                    logger.error(f"Features/health loop error: {e}")
                    await asyncio.sleep(30)

        asyncio.create_task(l2_snapshot_loop())
        asyncio.create_task(features_health_loop())

        logger.info("Phase 4 Recorder initialized and listening for events")

    except Exception as e:
        logger.error(f"Recorder initialization failed: {e}")


async def autotrader_loop():
    """Phase 6: Autotrader runtime (paper-first, KLGA)."""
    logger = logging.getLogger("autotrader_loop")
    try:
        from core.autotrader import get_autotrader_service
        service = get_autotrader_service()
        await service.start()
        logger.info("Autotrader service started")
        while True:
            await asyncio.sleep(3600)
    except Exception as e:
        logger.error(f"Autotrader loop failed: {e}")


async def learning_nightly_loop():
    """Nightly offline learning cycle (NYC schedule)."""
    logger = logging.getLogger("learning_nightly_loop")
    last_run_date = None
    while True:
        try:
            now_nyc = datetime.now(ZoneInfo("America/New_York"))
            should_run = (
                now_nyc.hour == 3
                and now_nyc.minute >= 15
                and last_run_date != now_nyc.date()
            )
            if should_run:
                from core.autotrader import get_autotrader_service
                service = get_autotrader_service()
                result = await asyncio.to_thread(service.run_learning_once)
                last_run_date = now_nyc.date()
                logger.info("Nightly learning completed: %s", result.get("learning_run", {}).get("status"))
        except Exception as e:
            logger.error(f"Nightly learning failed: {e}")
        await asyncio.sleep(60)


# =============================================================================
# Phase 3: Nowcast API Endpoints
# =============================================================================

@app.get("/api/v3/nowcast/snapshot")
async def get_nowcast_snapshot():
    """
    Get the full Phase 3 Nowcast Engine snapshot.
    Includes distributions for all stations.
    """
    from core.nowcast_integration import get_nowcast_integration
    integration = get_nowcast_integration()
    return integration.get_snapshot()

@app.get("/api/v3/nowcast/status")
async def get_nowcast_status():
    """
    Get Nowcast Integration status for monitoring.
    Shows trigger counts, recent updates, and engine health.
    """
    from core.nowcast_integration import get_nowcast_integration
    integration = get_nowcast_integration()
    return integration.get_status()


@app.get("/api/v3/nowcast/{station_id}")
async def get_nowcast_distribution(station_id: str):
    """
    Get nowcast distribution for a specific station.
    Returns probability distribution, t_peak, confidence, and explanations.
    """
    if station_id not in STATIONS:
        return {"error": "Invalid station"}

    from core.nowcast_integration import get_nowcast_integration
    integration = get_nowcast_integration()
    engine = integration.get_engine(station_id)

    distribution = integration.get_distribution(station_id)
    if not distribution:
        # Generate one on demand
        distribution = engine.generate_distribution()

    if not distribution:
        return {"error": "No distribution available"}

    payload = distribution.to_dict()

    # Optional: expose hourly temperature curve for UI debugging/inspection.
    try:
        base = engine.get_base_forecast()
        if (
            base
            and base.is_valid()
            and base.target_date == distribution.target_date
            and getattr(base, "t_hourly_f", None)
        ):
            base_hourly = [round(float(t), 1) for t in list(base.t_hourly_f)]
            payload["base_hourly_f"] = base_hourly
            payload["base_peak_hour"] = int(getattr(base, "t_peak_hour", 14))
            payload["base_tmax_f"] = round(float(getattr(base, "t_max_base_f", max(base_hourly) if base_hourly else 0.0)), 1)

            # Build a shape-preserving "nowcast adjusted" hourly curve:
            # 1) move peak timing toward the expected nowcast peak hour,
            # 2) nudge max toward nowcast Tmax mean.
            tmax = float(distribution.tmax_mean_f)
            base_peak = int(payload.get("base_peak_hour") or 14)
            target_peak = int(distribution.t_peak_expected_hour) if distribution.t_peak_expected_hour is not None else base_peak
            shaped = list(base_hourly)
            if abs(target_peak - base_peak) >= 1:
                shaped = shift_curve_toward_peak(shaped, target_peak, strength=0.45)
            adjusted = nudge_curve_to_target_max(
                shaped,
                tmax,
                anchor_peak_hour=target_peak,
                spread_hours=10.0,
                max_delta_f=15.0,
            )
            payload["nowcast_hourly_f"] = adjusted
            payload["nowcast_curve_target_peak_hour"] = int(target_peak)
            if adjusted:
                payload["nowcast_hourly_peak_hour"] = int(max(range(len(adjusted)), key=lambda i: adjusted[i]))
    except Exception:
        pass

    return payload


@app.get("/api/v3/nowcast/{station_id}/state")
async def get_nowcast_state(station_id: str):
    """
    Get the internal DailyState for debugging.
    Shows bias, trend, uncertainty, and health tracking.
    """
    if station_id not in STATIONS:
        return {"error": "Invalid station"}

    from core.nowcast_integration import get_nowcast_integration
    integration = get_nowcast_integration()
    engine = integration.get_engine(station_id)

    return engine.get_state_snapshot()


@app.get("/api/v3/nowcast/{station_id}/update")
async def trigger_nowcast_update(station_id: str):
    """
    Manually trigger a nowcast update for a station.
    Useful for testing and debugging.
    """
    if station_id not in STATIONS:
        return {"error": "Invalid station"}

    from core.nowcast_integration import get_nowcast_integration, extract_hourly_temps_f_for_date
    from collector.metar_fetcher import fetch_metar_race
    from collector.hrrr_fetcher import fetch_hrrr
    from collector.wunderground_fetcher import fetch_wunderground_max
    from datetime import timezone

    integration = get_nowcast_integration()
    engine = integration.get_engine(station_id)

    # Fetch latest HRRR for base forecast (best effort)
    try:
        station = STATIONS[station_id]
        station_tz = ZoneInfo(station.timezone)
        target_date = datetime.now(station_tz).date()
        now_utc = datetime.now(ZoneInfo("UTC"))

        wu_high: Optional[float] = None
        wu_current: Optional[float] = None
        wu_peak_hour: Optional[int] = None
        try:
            wu = await fetch_wunderground_max(station_id, target_date=target_date)
            wu_high = float(wu.forecast_high_f) if wu and wu.forecast_high_f is not None else None
            wu_current = float(wu.current_temp_f) if wu and wu.current_temp_f is not None else None
            wu_peak_hour = int(wu.forecast_peak_hour_local) if wu and wu.forecast_peak_hour_local is not None else None
        except Exception:
            wu_high = None
            wu_current = None
            wu_peak_hour = None

        forecast = await fetch_hrrr(station.latitude, station.longitude, target_days_ahead=0)
        hourly_temps_f = None

        if forecast and forecast.hourly_temps_c:
            hourly_temps_f = extract_hourly_temps_f_for_date(
                forecast.hourly_temps_c,
                forecast.hourly_times,
                target_date,
                station_tz
            )

        if hourly_temps_f:
            # Align both peak timing and max with WU hints.
            model_source = "HRRR"
            try:
                aligned_curve, aligned_source = align_hourly_curve_with_wu(
                    hourly_temps_f=hourly_temps_f,
                    wu_high_f=wu_high,
                    wu_peak_hour_local=wu_peak_hour,
                )
                if aligned_curve:
                    hourly_temps_f = aligned_curve
                    model_source = aligned_source
            except Exception:
                pass

            integration.on_hrrr_update(
                station_id=station_id,
                hourly_temps_f=hourly_temps_f,
                model_run_time=forecast.fetch_time if forecast else now_utc,
                model_source=model_source,
                target_date=target_date
            )
        elif wu_high is not None:
            # Fallback: Open-Meteo is rate-limited/unavailable; use WU high to avoid nonsense early-day nowcasts.
            anchor_temp_f: Optional[float] = None
            try:
                state = engine.get_or_create_daily_state(target_date=target_date)
                if state.last_obs_time_utc:
                    anchor_temp_f = float(state.last_obs_temp_f or state.last_obs_aligned_f)
            except Exception:
                pass
            if anchor_temp_f is None:
                anchor_temp_f = wu_current

            fallback_peak_hour = infer_synthetic_peak_hour(
                now_utc=now_utc,
                station_tz=station_tz,
                current_temp_f=anchor_temp_f,
                tmax_f=float(wu_high),
                hinted_peak_hour=wu_peak_hour,
            )
            synth = build_synthetic_hourly_curve_f(
                now_utc=now_utc,
                station_tz=station_tz,
                current_temp_f=anchor_temp_f,
                tmax_f=float(wu_high),
                peak_hour=fallback_peak_hour,
            )
            if synth:
                integration.on_hrrr_update(
                    station_id=station_id,
                    hourly_temps_f=synth,
                    model_run_time=now_utc,
                    model_source="WU_FORECAST",
                    target_date=target_date
                )
    except Exception:
        pass

    # Fetch latest METAR
    try:
        metar = await fetch_metar_race(station_id)
        if metar:
            # Update engine with METAR
            from core.models import OfficialObs
            obs = OfficialObs(
                obs_time_utc=metar.observation_time,
                source=getattr(metar, "source", "METAR_RACE"),
                station_id=station_id,
                temp_c=metar.temp_c,
                temp_f=metar.temp_f,
                temp_f_raw=metar.temp_f,
                temp_f_low=metar.temp_f_low,
                temp_f_high=metar.temp_f_high,
                settlement_f_low=metar.settlement_f_low,
                settlement_f_high=metar.settlement_f_high,
                has_t_group=metar.has_t_group,
                dewpoint_c=metar.dewpoint_c,
                wind_dir=metar.wind_dir_degrees,
                wind_speed=metar.wind_speed_kt,
                sky_condition=metar.sky_condition,
                report_type=getattr(metar, "report_type", "METAR"),
                is_speci=bool(getattr(metar, "is_speci", False)),
            )
            distribution = integration.on_metar(obs)
            return {
                "success": True,
                "metar_temp_f": metar.temp_f,
                "metar_report_type": getattr(metar, "report_type", "METAR"),
                "metar_is_speci": bool(getattr(metar, "is_speci", False)),
                "distribution": distribution.to_dict() if distribution else None
            }
    except Exception as e:
        return {"error": str(e)}

    return {"error": "Failed to fetch METAR"}


# =============================================================================
# Phase 4: Replay & Storage Endpoints
# =============================================================================

@app.get("/api/v4/replay/dates")
async def get_replay_dates(station_id: str = None):
    """Get list of dates with recorded data."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()
    dates = await asyncio.to_thread(engine.list_available_dates, station_id)
    return {"dates": dates}


@app.get("/api/v4/replay/dates/{date_str}/channels")
async def get_replay_channels(date_str: str, station_id: str = None):
    """Get channels available for a date."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()
    channels = await asyncio.to_thread(engine.list_channels_for_date, date_str, station_id)
    return {"channels": channels}


@app.post("/api/v4/replay/session")
async def create_replay_session(date: str, station_id: str = None):
    """Create a new replay session."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()

    session = await engine.create_session(date, station_id)
    if not session:
        channels = await asyncio.to_thread(engine.list_channels_for_date, date, station_id)
        all_channels = (
            await asyncio.to_thread(engine.list_channels_for_date, date, None)
            if station_id else channels
        )
        available_dates = (await asyncio.to_thread(engine.list_available_dates, station_id))[:10]

        # Build descriptive error
        if not all_channels:
            reason = f"No recorded data exists for {date}. The recorder may not have been running."
        elif station_id and not channels:
            reason = f"Data exists for {date} but not for station {station_id}. Try 'All stations'."
        else:
            reason = f"No events found for {date}" + (f" / {station_id}" if station_id else "")

        return {
            "error": reason,
            "date": date,
            "station_id": station_id,
            "channels_for_date": all_channels,
            "channels_for_station": channels,
            "available_dates": available_dates,
            "date_reference": (
                f"{STATIONS[station_id].timezone} market date"
                if station_id and station_id in STATIONS
                else "NYC market date"
            ),
        }

    return session.get_state()


@app.get("/api/v4/replay/session/{session_id}/state")
async def get_replay_state(session_id: str):
    """Get current state of a replay session."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()

    session = engine.get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    return session.get_state()


@app.get("/api/v4/replay/session/{session_id}/events")
async def get_replay_events(session_id: str, n: int = 5):
    """Get events around current position."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()

    session = engine.get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    return session.get_events_around(n)


@app.get("/api/v4/replay/session/{session_id}/categories")
async def get_replay_categories(session_id: str):
    """Get category summary for replay cards."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()

    session = engine.get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    return session.get_category_summary()


@app.get("/api/v4/replay/session/{session_id}/pws_learning")
async def get_replay_pws_learning(
    session_id: str,
    max_points: int = 80,
    top_n: int = 6,
    trend_points: int = 240,
    trend_mode: str = "hourly",
):
    """Get PWS learning ranking + leader evolution for replay."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()

    session = engine.get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    return session.get_pws_learning_summary(
        max_points=max_points,
        top_n=top_n,
        trend_points=trend_points,
        trend_mode=trend_mode,
    )


@app.post("/api/v4/replay/session/{session_id}/play")
async def replay_play(session_id: str, speed: float = 1.0):
    """Start/resume playback."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()

    session = engine.get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    await session.play(speed)
    return {"success": True, "state": session.state.value}


@app.post("/api/v4/replay/session/{session_id}/pause")
async def replay_pause(session_id: str):
    """Pause playback."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()

    session = engine.get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    await session.pause()
    return {"success": True, "state": session.state.value}


@app.post("/api/v4/replay/session/{session_id}/stop")
async def replay_stop(session_id: str):
    """Stop playback and reset."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()

    session = engine.get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    await session.stop()
    return {"success": True, "state": session.state.value}


@app.post("/api/v4/replay/session/{session_id}/seek")
async def replay_seek(session_id: str, percent: float):
    """Seek to percentage position."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()

    session = engine.get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    session.seek_percent(percent)
    return {"success": True, "progress": session.clock.get_progress_percent()}


@app.post("/api/v4/replay/session/{session_id}/speed")
async def replay_speed(session_id: str, speed: float):
    """Change playback speed."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()

    session = engine.get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    # Change speed (works during playback without restart)
    session.set_speed(speed)
    return {"success": True, "speed": speed}


@app.post("/api/v4/replay/session/{session_id}/jump/{jump_type}")
async def replay_jump(session_id: str, jump_type: str):
    """Jump to next/prev event."""
    from core.replay_engine import get_replay_engine
    engine = get_replay_engine()

    session = engine.get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    success = False
    if jump_type == "next_metar":
        success = session.jump_to_next_metar()
    elif jump_type == "prev_metar":
        success = session.jump_to_prev_metar()
    elif jump_type == "next_window":
        success = session.jump_to_next_window()

    return {"success": success, "index": session._event_index}


@app.get("/api/v4/recorder/stats")
async def get_recorder_stats():
    """Get recorder statistics."""
    from core.recorder import get_recorder
    recorder = get_recorder()
    return recorder.get_stats()


@app.get("/api/v4/recorder/recent/{channel}")
async def get_recorder_recent(channel: str, n: int = 50):
    """Get recent events from a channel."""
    from core.recorder import get_recorder
    recorder = get_recorder()
    return {"events": recorder.get_recent(channel, n)}


@app.post("/api/v4/compactor/run")
async def run_compactor():
    """Manually trigger compaction of all pending dates."""
    from core.compactor import get_compactor
    compactor = get_compactor()
    result = compactor.compact_all_pending()
    return result


@app.get("/api/v4/compactor/stats")
async def get_compactor_stats():
    """Get compactor statistics."""
    from core.compactor import get_compactor
    compactor = get_compactor()
    return compactor.get_stats()


@app.get("/api/v4/windows/stats")
async def get_window_stats():
    """Get event window statistics."""
    from core.event_window import get_event_window_manager
    manager = get_event_window_manager()
    return manager.get_stats()


@app.get("/api/v4/windows/active")
async def get_active_windows():
    """Get currently active event windows."""
    from core.event_window import get_event_window_manager
    manager = get_event_window_manager()
    return {"windows": [w.to_dict() for w in manager.get_active_windows()]}


@app.get("/api/v2/world/snapshot")
async def get_world_snapshot():
    """
    Get the full Phase 2 Event-Driven World State.
    Source of Truth for the new UI.
    """
    from core.world import get_world
    return get_world().get_snapshot()


@app.get("/api/v2/world/metar_subscription/{station_id}")
async def get_world_metar_subscription(station_id: str):
    """Get Telegram METAR/SPECI push subscription state for a station."""
    station_id = str(station_id or "").upper().strip()
    if station_id not in STATIONS:
        return {"error": "Invalid station"}
    state = get_telegram_station_subscription(station_id)
    state["station_id"] = station_id
    return state


@app.post("/api/v2/world/metar_subscription/{station_id}")
async def set_world_metar_subscription(station_id: str, payload: TelegramMetarSubscriptionToggle):
    """Enable/disable Telegram METAR/SPECI push subscription for a station."""
    station_id = str(station_id or "").upper().strip()
    if station_id not in STATIONS:
        return {"error": "Invalid station"}
    state = set_telegram_station_subscription(
        station_id,
        bool(payload.enabled),
        updated_by="world_ui",
    )
    state["station_id"] = station_id
    return state


@app.get("/api/v2/world/metar_history/{station_id}")
async def get_world_metar_history(station_id: str):
    """Return today's METAR observations (NYC settlement day) from NOAA."""
    if station_id not in STATIONS:
        return {"error": "Invalid station"}
    from collector.metar_fetcher import fetch_metar_history
    from core.judge import JudgeAlignment
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(STATIONS[station_id].timezone)
    now_local = datetime.now(tz)
    today = now_local.date()

    # Fetch enough hours to cover from midnight local
    hours_since_midnight = now_local.hour + 1
    history = await fetch_metar_history(station_id, hours=max(hours_since_midnight + 2, 6))

    rows = []
    for h in history:
        h_local = h.observation_time.astimezone(tz)
        if h_local.date() != today:
            continue
        temp_f_raw = JudgeAlignment.celsius_to_fahrenheit(h.temp_c)
        temp_f_settlement = JudgeAlignment.round_to_settlement(temp_f_raw)
        rows.append({
            "obs_time_utc": h.observation_time.isoformat(),
            "obs_time_local": h_local.strftime("%H:%M"),
            "temp_c": h.temp_c,
            "temp_f_raw": temp_f_raw,
            "temp_f": temp_f_settlement,
            "dewpoint_c": h.dewpoint_c,
            "wind_dir": h.wind_dir_degrees,
            "wind_speed": h.wind_speed_kt,
            "sky": h.sky_condition,
            "report_type": getattr(h, "report_type", "METAR"),
            "is_speci": bool(getattr(h, "is_speci", False)),
            "raw_metar": h.raw_metar,
        })

    # Sort by obs time ascending (oldest first)
    rows.sort(key=lambda r: r["obs_time_utc"])

    # Compute running max
    running_max = None
    for r in rows:
        if running_max is None or r["temp_f_raw"] > running_max:
            running_max = r["temp_f_raw"]
        r["running_max_f"] = running_max

    return {"station_id": station_id, "date": today.isoformat(), "observations": rows}


@app.get("/api/v2/world/stream")
async def world_stream():
    """
    Phase 2 SSE endpoint: real-time push from WorldState.
    Clients receive events as they happen, no polling needed.
    Usage: const es = new EventSource('/api/v2/world/stream');
    """
    from core.world import get_world
    import json as _json

    world = get_world()
    queue = world.subscribe_sse()

    async def event_generator():
        try:
            # Send initial snapshot
            snapshot = world.get_snapshot()
            yield f"event: snapshot\ndata: {_json.dumps(snapshot, default=str)}\n\n"

            while True:
                try:
                    event_data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"event: update\ndata: {_json.dumps(event_data, default=str)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            world.unsubscribe_sse(queue)

    from starlette.responses import StreamingResponse as SSEResponse
    return SSEResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/realtime/{station_id}")
async def get_realtime_data(station_id: str):
    """
    Ultra-fast endpoint for alert checking.
    Returns ONLY the latest NOAA racing values without touching the database.
    Designed for 2-second polling from frontend alerts.
    """
    if station_id not in STATIONS:
        return {"error": "Invalid station"}
    
    from collector.metar_fetcher import fetch_metar
    
    try:
        current = await fetch_metar(station_id)
        
        if not current:
            return {"error": "No METAR data available", "station_id": station_id}
        
        racing = current.racing_results or {}
        
        return {
            "station_id": station_id,
            "noaa": current.temp_f,
            "noaa_low_f": current.temp_f_low,
            "noaa_high_f": current.temp_f_high,
            "settlement_low_f": current.settlement_f_low,
            "settlement_high_f": current.settlement_f_high,
            "has_t_group": current.has_t_group,
            "json_api": racing.get("NOAA_JSON_API"),
            "tds_xml": racing.get("AWC_TDS_XML"),
            "tgftp_txt": racing.get("NOAA_TG_FTP_TXT"),
            "report_type": getattr(current, "report_type", "METAR"),
            "is_speci": bool(getattr(current, "is_speci", False)),
            "racing_report_types": getattr(current, "racing_report_types", None),
            "observation_time": current.observation_time.isoformat() if current.observation_time else None,
            "timestamp": datetime.now(ZoneInfo("UTC")).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "station_id": station_id}

@app.get("/api/realtime/market/{station_id}")
async def get_realtime_market(station_id: str, depth: int = 10):
    """
    Get ultra-low latency market data from local memory mirror.
    Includes SEARA-grade observability metrics.
    """
    try:
        if station_id not in STATIONS:
            return {"error": "Invalid station"}
            
        client = await get_ws_client()
        # Allow reading even if disconnected, to see stale state (debug)
        # But alert if not connected
             
        data = []

        if not client.state or not hasattr(client.state, 'orderbooks'):
             return {"error": "Client state not initialized"}

        current_ts = datetime.now().timestamp()

        for token_id, book in client.state.orderbooks.items():
            raw_info = client.market_info.get(token_id, "Unknown|Unknown")
            m_station, m_bracket, m_outcome = _parse_market_info(raw_info)

            # Filter by station
            if m_station != station_id:
                continue

            if depth and depth > 0:
                snap = book.get_l2_snapshot(top_n=depth)
            else:
                snap = book.get_snapshot()  # O(1) cached snapshot
            snap["bracket"] = m_bracket
            snap["outcome"] = m_outcome
            data.append(snap)

        # If empty, try a quick on-demand subscription refresh for this station
        if not data:
            try:
                from market.polymarket_checker import get_target_date
                station = STATIONS[station_id]
                now = datetime.now(ZoneInfo(station.timezone))
                target_date = await get_target_date(station_id, now)
                logger = logging.getLogger("ws_loop")
                tokens = await _resolve_station_tokens(station_id, target_date, now, logger)
                if tokens:
                    await client.subscribe_to_markets([t[0] for t in tokens])
                    for tid, bracket, outcome in tokens:
                        client.set_market_info(tid, f"{station_id}|{bracket}|{outcome}")
                    # Rebuild view after warm-up subscription
                    for token_id, book in client.state.orderbooks.items():
                        raw_info = client.market_info.get(token_id, "Unknown|Unknown")
                        m_station, m_bracket, m_outcome = _parse_market_info(raw_info)
                        if m_station != station_id:
                            continue
                        if depth and depth > 0:
                            snap = book.get_l2_snapshot(top_n=depth)
                        else:
                            snap = book.get_snapshot()
                        snap["bracket"] = m_bracket
                        snap["outcome"] = m_outcome
                        data.append(snap)
            except Exception:
                pass

        return {
            "station_id": station_id,
            "status": "connected" if client.is_connected else "disconnected",
            "server_ts": current_ts,
            "metrics": client.state.metrics,
            "count": len(data),
            "markets": data
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/polymarket/{station_id}")
async def get_polymarket_dashboard_data(station_id: str, target_day: int = 0, depth: int = 5):
    """Unified Polymarket endpoint: Gamma API prices + WS orderbook + crowd wisdom + maturity."""
    import json as _json
    if station_id not in STATIONS:
        return {"error": "Invalid station"}

    station = STATIONS[station_id]
    local_now = datetime.now(ZoneInfo(station.timezone))
    target_date = (local_now + timedelta(days=target_day)).date()

    # 1. Fetch event data (same pattern as /api/market/)
    event_data = await fetch_event_for_station_date(station_id, target_date)
    if not event_data:
        event_slug = build_event_slug(station_id, target_date)
        event_data = await fetch_event_data(event_slug)
        if not event_data:
            return {"error": "Market not found", "target_date": target_date.isoformat()}
    event_slug = build_event_slug(station_id, target_date)
    event_title = event_data.get("title", "Unknown Event")
    markets = event_data.get("markets", [])

    # 2. Parse brackets from Gamma API
    total_volume = 0.0
    brackets = []
    for market in markets:
        bracket_name = market.get("groupItemTitle", "Unknown")
        try:
            volume = float(market.get("volume", "0"))
        except (ValueError, TypeError):
            volume = 0.0
        total_volume += volume

        # Parse outcomePrices (JSON string)
        outcome_prices_raw = market.get("outcomePrices", [])
        try:
            if isinstance(outcome_prices_raw, str):
                outcome_prices = _json.loads(outcome_prices_raw)
            else:
                outcome_prices = outcome_prices_raw
            yes_price = float(outcome_prices[0]) if outcome_prices else 0.0
        except (ValueError, TypeError, IndexError):
            yes_price = 0.0

        # Parse clobTokenIds (JSON string): [YES token, NO token]
        clob_raw = market.get("clobTokenIds", "[]")
        try:
            if isinstance(clob_raw, str):
                clob_ids = _json.loads(clob_raw)
            else:
                clob_ids = clob_raw
            yes_token = clob_ids[0] if len(clob_ids) > 0 else None
            no_token = clob_ids[1] if len(clob_ids) > 1 else None
        except (ValueError, TypeError, IndexError):
            yes_token = None
            no_token = None

        brackets.append({
            "name": bracket_name,
            "yes_price": yes_price,
            "no_price": round(1.0 - yes_price, 4),
            "volume": volume,
            "token_id": yes_token,  # backwards-compatible alias for YES token
            "yes_token_id": yes_token,
            "no_token_id": no_token,
            "ws_best_bid": None, "ws_best_ask": None, "ws_spread": None,
            "ws_mid": None, "ws_bid_depth": None, "ws_ask_depth": None,
            "ws_bids": [], "ws_asks": [], "ws_staleness_ms": None,
            "ws_yes_best_bid": None, "ws_yes_best_ask": None, "ws_yes_spread": None,
            "ws_yes_mid": None, "ws_yes_bid_depth": None, "ws_yes_ask_depth": None,
            "ws_yes_bids": [], "ws_yes_asks": [], "ws_yes_staleness_ms": None,
            "ws_no_best_bid": None, "ws_no_best_ask": None, "ws_no_spread": None,
            "ws_no_mid": None, "ws_no_bid_depth": None, "ws_no_ask_depth": None,
            "ws_no_bids": [], "ws_no_asks": [], "ws_no_staleness_ms": None,
        })

    # 3. Enrich with WebSocket orderbook data (join by token_id)
    ws_connected = False
    ws_metrics = {}
    try:
        client = await get_ws_client()
        ws_connected = client.is_connected
        ws_metrics = client.state.metrics if client.state else {}

        # Ensure both YES and NO token books are subscribed for the requested market.
        missing_tokens: list[str] = []
        if client.state and hasattr(client.state, "orderbooks"):
            for b in brackets:
                for token_key in ("yes_token_id", "no_token_id"):
                    token_id = b.get(token_key)
                    if token_id and token_id not in client.state.orderbooks:
                        missing_tokens.append(token_id)
                        outcome = "yes" if token_key == "yes_token_id" else "no"
                        client.set_market_info(token_id, f"{station_id}|{b.get('name', 'Unknown')}|{outcome}")
            if missing_tokens:
                unique_missing = list(dict.fromkeys(missing_tokens))
                await client.subscribe_to_markets(unique_missing)

        if client.state and hasattr(client.state, 'orderbooks'):
            for b in brackets:
                yes_token_id = b.get("yes_token_id")
                no_token_id = b.get("no_token_id")

                if yes_token_id and yes_token_id in client.state.orderbooks:
                    snap_yes = client.state.orderbooks[yes_token_id].get_l2_snapshot(top_n=depth)
                    b["ws_yes_best_bid"] = snap_yes.get("best_bid")
                    b["ws_yes_best_ask"] = snap_yes.get("best_ask")
                    b["ws_yes_spread"] = snap_yes.get("spread")
                    b["ws_yes_mid"] = snap_yes.get("mid")
                    b["ws_yes_bid_depth"] = snap_yes.get("bid_depth")
                    b["ws_yes_ask_depth"] = snap_yes.get("ask_depth")
                    b["ws_yes_bids"] = snap_yes.get("bids", [])
                    b["ws_yes_asks"] = snap_yes.get("asks", [])
                    b["ws_yes_staleness_ms"] = snap_yes.get("staleness_ms")

                    # Backwards-compatible fields continue representing YES orderbook.
                    b["ws_best_bid"] = b["ws_yes_best_bid"]
                    b["ws_best_ask"] = b["ws_yes_best_ask"]
                    b["ws_spread"] = b["ws_yes_spread"]
                    b["ws_mid"] = b["ws_yes_mid"]
                    b["ws_bid_depth"] = b["ws_yes_bid_depth"]
                    b["ws_ask_depth"] = b["ws_yes_ask_depth"]
                    b["ws_bids"] = b["ws_yes_bids"]
                    b["ws_asks"] = b["ws_yes_asks"]
                    b["ws_staleness_ms"] = b["ws_yes_staleness_ms"]

                if no_token_id and no_token_id in client.state.orderbooks:
                    snap_no = client.state.orderbooks[no_token_id].get_l2_snapshot(top_n=depth)
                    b["ws_no_best_bid"] = snap_no.get("best_bid")
                    b["ws_no_best_ask"] = snap_no.get("best_ask")
                    b["ws_no_spread"] = snap_no.get("spread")
                    b["ws_no_mid"] = snap_no.get("mid")
                    b["ws_no_bid_depth"] = snap_no.get("bid_depth")
                    b["ws_no_ask_depth"] = snap_no.get("ask_depth")
                    b["ws_no_bids"] = snap_no.get("bids", [])
                    b["ws_no_asks"] = snap_no.get("asks", [])
                    b["ws_no_staleness_ms"] = snap_no.get("staleness_ms")
    except Exception:
        pass

    # 4. Market maturity
    from market.polymarket_checker import is_event_mature
    is_mature, winning_outcome, max_prob = is_event_mature(event_data)

    # 5. Crowd wisdom (sentiment + shifts)
    from market.crowd_wisdom import detect_market_shift, get_crowd_sentiment
    target_date_str = target_date.isoformat()
    sentiment = get_crowd_sentiment(station_id, target_date_str)
    shifts = detect_market_shift(station_id, target_date_str, threshold=0.03)

    # Sort brackets by yes_price descending
    brackets.sort(key=lambda x: x["yes_price"], reverse=True)

    return {
        "station_id": station_id,
        "target_date": target_date.isoformat(),
        "target_day": target_day,
        "event_title": event_title,
        "event_slug": event_slug,
        "total_volume": total_volume,
        "market_status": {
            "is_mature": is_mature,
            "winning_outcome": winning_outcome,
            "max_probability": max_prob,
            "event_closed": event_data.get("closed", False),
            "event_active": event_data.get("active", True),
        },
        "ws_connected": ws_connected,
        "ws_metrics": ws_metrics,
        "brackets": brackets,
        "sentiment": sentiment,
        "shifts": shifts,
        "server_ts": datetime.now().timestamp(),
    }


async def prediction_loop():
    """Run HELIOS physics predictions every 5 minutes."""
    from main import collect_and_predict
    from config import get_active_stations
    logger = logging.getLogger("prediction_loop")
    logger.info("Starting background prediction loop (5-min intervals)...")

    # Run immediately on startup for active stations
    active = get_active_stations()
    logger.info(f"Active stations: {list(active.keys())}")
    for station_id in active.keys():
        try:
            logger.info(f"Initial HELIOS prediction ({station_id})...")
            await collect_and_predict(station_id, target_days=[0, 1])
        except Exception as e:
            logger.error(f"Initial prediction failed for {station_id}: {e}")

    while True:
        try:
            now = datetime.now()
            # Aligned to 5 minute marks
            wait_seconds = (5 - (now.minute % 5)) * 60 - now.second
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

            # Run predictions for active stations only
            active = get_active_stations()
            for station_id in active.keys():
                logger.info(f"Executing periodic HELIOS prediction ({station_id} Today/Tomorrow)...")
                await collect_and_predict(station_id, target_days=[0, 1])

            await asyncio.sleep(10) # Avoid double-running
        except Exception as e:
            logger.error(f"Prediction loop error: {e}")
            await asyncio.sleep(60)

async def snapshot_loop():
    """Capture market and reality snapshots every 1 minute."""
    from monitor import snapshot_current_state, reconcile_history
    logger = logging.getLogger("snapshot_loop")
    logger.info("Starting background snapshot loop (3-sec intervals for near-realtime)...")
    while True:
        try:
            await snapshot_current_state()
            
            # Reconcile history once an hour at :05
            if datetime.now().minute == 5 and datetime.now().second < 10:
                await reconcile_history()
                
            await asyncio.sleep(3)  # Ultra-fast polling
        except Exception as e:
            logger.error(f"Snapshot loop error: {e}")
            await asyncio.sleep(5)

@app.post("/api/refresh/{station_id}")
async def refresh_station(station_id: str):
    """Force a prediction refresh and save snapshot to analytics."""
    from main import collect_and_predict
    from monitor import snapshot_current_state
    logger = logging.getLogger("refresh")
    try:
        logger.info(f"Manual refresh triggered for {station_id}")
        await collect_and_predict(station_id, target_days=[0, 1])
        await snapshot_current_state()
        return {"success": True, "message": "Data refreshed and synced to analytics"}
    except Exception as e:
        logger.error(f"Refresh error for {station_id}: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/v12_debug/{station_id}")
async def get_v12_debug(station_id: str):
    """
    v12.0 Debug Endpoint - Shows real-time status of all v12.0 protections.
    Access: /api/v12_debug/KLGA
    """
    if station_id not in STATIONS:
        return {"error": "Invalid station"}
    
    from collector.metar_fetcher import fetch_metar
    from collector.nbm_fetcher import fetch_nbm
    from collector.lamp_fetcher import fetch_lamp
    from synthesizer.ensemble import calculate_ensemble_v11
    from database import get_latest_prediction_for_date, get_performance_history_by_target_date
    from synthesizer.advection import calculate_advection
    from collector.advection_fetcher import fetch_upstream_weather
    
    station = STATIONS[station_id]
    tz = ZoneInfo(station.timezone)
    now_local = datetime.now(tz)
    today_str = now_local.date().isoformat()
    
    # Get current METAR
    current = await fetch_metar(station_id)
    metar_current = current.temp_f if current else None
    
    # Advection Calculation
    advection_report = None
    wind_speed_kmh = 0.0
    wind_dir = 0
    upstream_temp_c = None
    
    if current and current.temp_c is not None:
        # Get wind data (METAR uses knots, advection uses km/h)
        wspd_kt = current.wind_speed_kt or 0
        wind_speed_kmh = wspd_kt * 1.852
        wind_dir = current.wind_dir_degrees or 0
        
        # Fetch upstream
        upstream_data = await fetch_upstream_weather(
            base_lat=station.latitude,
            base_lon=station.longitude,
            wind_dir_degrees=wind_dir
        )
        
        if upstream_data:
            advection_report = calculate_advection(
                current_temp_c=current.temp_c,
                upstream_data=upstream_data,
                local_wind_kmh=wind_speed_kmh
            )
            upstream_temp_c = upstream_data.temperature_c
    
    # Get observed max from today's logs
    logs = get_performance_history_by_target_date(station_id, today_str)
    observed_max = None
    if logs:
        temps = [l.get("cumulative_max_f") or l.get("metar_actual") for l in logs if l.get("cumulative_max_f") or l.get("metar_actual")]
        if temps:
            observed_max = max([t for t in temps if t])
    
    # Get prediction
    pred = get_latest_prediction_for_date(station_id, today_str)
    hrrr_val = pred.get("hrrr_max_raw_f") if pred else 55.0
    
    # Fetch NBM/LAMP
    try:
        nbm_data = await fetch_nbm(station_id, now_local.date())
        lamp_data = await fetch_lamp(station_id)
    except:
        nbm_data = None
        lamp_data = None
    
    # Run v12.0 ensemble calculation
    result = calculate_ensemble_v11(
        hrrr_max_f=hrrr_val,
        nbm_max_f=nbm_data.max_temp_f if nbm_data else None,
        lamp_max_f=lamp_data.max_temp_f if lamp_data else None,
        lamp_confidence=lamp_data.confidence if lamp_data else 0.5,
        metar_actual_f=metar_current,
        local_hour=now_local.hour,
        local_minute=now_local.minute,
        observed_max_f=observed_max,
        wu_forecast_high=pred.get("wu_forecast_high") if pred else None
    )
    
    return {
        "station_id": station_id,
        "local_time": now_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "local_hour": now_local.hour,
        "local_minute": now_local.minute,
        
        # Physics Context (Advection & Wind)
        "physics_context": {
            "wind_speed_kt": current.wind_speed_kt if current else 0,
            "wind_dir_degrees": current.wind_dir_degrees if current else 0,
            "wind_trend": current.wind_trend if current else "N/A",
            "metar_report_type": getattr(current, "report_type", "METAR") if current else None,
            "metar_is_speci": bool(getattr(current, "is_speci", False)) if current else False,
            "advection_rate_c_h": advection_report.rate_c_per_hour if advection_report else 0.0,
            "upstream_temp_c": upstream_temp_c,
            "description": advection_report.description if advection_report else "No data",
            "is_heating": (advection_report.rate_c_per_hour > 0.1) if advection_report else False,
            "is_cooling": (advection_report.rate_c_per_hour < -0.1) if advection_report else False
        },
        
        # Inputs
        "inputs": {
            "metar_current_f": metar_current,
            "metar_report_type": getattr(current, "report_type", "METAR") if current else None,
            "metar_is_speci": bool(getattr(current, "is_speci", False)) if current else False,
            "observed_max_f": observed_max,
            "hrrr_max_f": hrrr_val,
            "nbm_max_f": nbm_data.max_temp_f if nbm_data else None,
            "lamp_max_f": lamp_data.max_temp_f if lamp_data else None,
        },
        
        # v12.0 Protection Status
        "v12_protections": {
            "sunset_lock_active": result.sunset_lock_active,
            "sunset_lock_threshold": "METAR < observed_max - 0.5°F",
            "sunset_lock_condition_met": (observed_max - metar_current) > 0.5 if observed_max and metar_current else False,
            
            "ultimate_squeeze_active": result.ultimate_squeeze_active,
            "ultimate_squeeze_threshold": "After 14:30 NYC, max spread = 1.2°F",
            
            "ghost_prediction_warning": result.ghost_prediction_warning,
            "ghost_reason": result.ghost_reason,
            "ghost_threshold": "HELIOS > METAR + 3.0°F after 14:00",
        },
        
        # Ensemble Results
        "ensemble_result": {
            "base_weighted_f": result.base_weighted_f,
            "floor_f": result.helios_floor_f,
            "ceiling_f": result.helios_ceiling_f,
            "spread_f": result.ensemble_spread_f,
            "allowed_spread_f": result.allowed_spread_f,
            "confidence": result.confidence_level,
            "strategy_note": result.strategy_note,
            "observed_max_used_f": result.observed_max_used_f,
        }
    }


@app.get("/api/analytics")
def get_analytics_api(station_id: str = DEFAULT_STATION, date: str = None):
    """Get performance data for the dashboard, filtered by TARGET DATE."""
    from database import get_performance_history_by_target_date
    from zoneinfo import ZoneInfo
    if date: target_date = date
    else:
        nyc_tz = ZoneInfo("America/New_York")
        target_date = datetime.now(nyc_tz).date().isoformat()
    logs = get_performance_history_by_target_date(station_id, target_date)
    
    # Filter logs to only include data starting from 00:00 of the target_date
    # This prevents the chart from showing data from the previous day
    start_of_day = datetime.fromisoformat(target_date)
    filtered_logs = []
    for log in logs:
        ts = datetime.fromisoformat(log["timestamp"])
        # Handle timezone if necessary, but assuming logs and target_date align in date
        # Check if log timestamp date matches target_date
        # Note: logs are in UTC usually, but we want to filter by the station's local day
        # For simplicity, if target_date string is in log timestamp, keep it.
        # Better: Convert log timestamp to target timezone (NYC) and check date.
        ts_nyc = ts.astimezone(ZoneInfo("America/New_York"))
        if ts_nyc.date().isoformat() == target_date:
            filtered_logs.append(log)
            
    logs = filtered_logs
    chronological_logs = logs[::-1]
    
    labels = []
    helios_data = []
    cumulative_max_data = [] # v6.0: NOAA METAR (single source)
    wu_forecast_data = []
    wu_live_data = [] # v6.0: WU real-time temp
    hrrr_raw_data = [] # v6.2: HRRR raw max
    # v8.0: Racing METAR Sources
    metar_json_data = []
    metar_xml_data = []
    metar_txt_data = []
    
    # We want to show the 3 sensations (brackets) that are CURRENTLY top 3
    latest_market_log = None
    for log in logs: # logs is DESC
        if log.get("market_top1_bracket"):
            latest_market_log = log
            break
            
    # v6.2: Optimized bracket tracking - only track top 5 most active brackets
    # Pre-parse JSON data once to avoid repeated parsing in loops
    import json
    parsed_json_cache = {}  # log index -> parsed brackets dict
    
    for i, log in enumerate(chronological_logs):
        json_data = log.get("market_all_brackets_json")
        if json_data:
            try:
                brackets = json.loads(json_data)
                parsed_json_cache[i] = {item["bracket"]: item.get("prob", 0.0) for item in brackets if item.get("bracket")}
            except:
                parsed_json_cache[i] = {}
        else:
            # Fallback to top3 columns
            parsed_json_cache[i] = {}
            if log.get("market_top1_bracket"):
                parsed_json_cache[i][log["market_top1_bracket"]] = log.get("market_top1_prob", 0.0)
            if log.get("market_top2_bracket"):
                parsed_json_cache[i][log["market_top2_bracket"]] = log.get("market_top2_prob", 0.0)
            if log.get("market_top3_bracket"):
                parsed_json_cache[i][log["market_top3_bracket"]] = log.get("market_top3_prob", 0.0)
    
    # Get top 5 brackets from the latest data point with valid market data
    top_brackets = []
    for i in range(len(chronological_logs) - 1, -1, -1):
        if parsed_json_cache.get(i):
            sorted_items = sorted(parsed_json_cache[i].items(), key=lambda x: x[1], reverse=True)
            top_brackets = [b for b, p in sorted_items[:5]]  # Top 5 only for performance
            break
    
    # Fallback
    if not top_brackets and latest_market_log:
        top_brackets = [
            latest_market_log.get("market_top1_bracket"),
            latest_market_log.get("market_top2_bracket"),
            latest_market_log.get("market_top3_bracket")
        ]
        top_brackets = [b for b in top_brackets if b]
    
    # Initialize probability tracking
    market_probs = {b: [] for b in top_brackets}
    debug_data = []
    raw_timestamps = []
    
    # Pre-compute timezone objects once
    nyc_tz = ZoneInfo("America/New_York")
    es_tz = ZoneInfo("Europe/Madrid")
    
    for i, log in enumerate(chronological_logs):
        # Timestamps
        ts = datetime.fromisoformat(log["timestamp"]).replace(tzinfo=ZoneInfo("UTC"))
        raw_timestamps.append(log["timestamp"] + 'Z')
        
        ts_nyc = ts.astimezone(nyc_tz)
        ts_es = ts.astimezone(es_tz)
        labels.append(f"{ts_nyc.strftime('%H:%M')} ({ts_es.strftime('%H:%M')})")
        
        # Temperature data
        helios_data.append(log["helios_pred"])
        cumulative_max_data.append(log.get("metar_actual"))
        wu_forecast_data.append(log.get("wu_forecast_high"))
        wu_live_data.append(log.get("wu_preliminary"))
        hrrr_raw_data.append(log.get("hrrr_max_raw_f")) # v6.2: HRRR raw
        
        # v8.0: Racing METAR Data
        metar_json_data.append(log.get("metar_json_api_f"))
        metar_xml_data.append(log.get("metar_tds_xml_f"))
        metar_txt_data.append(log.get("metar_tgftp_txt_f"))

        # Debug components
        debug_data.append({
            "hrrr": log.get("hrrr_max_raw_f"),
            "delta": log.get("current_deviation_f"),
            "physics": log.get("physics_adjustment_f"),
            "soil": log.get("soil_moisture"),
            "radiation": log.get("radiation"),
            "sky": log.get("sky_condition"),
            "wind": log.get("wind_dir"),
            # v10.0: Ensemble data
            "nbm": log.get("nbm_max_f"),
            "lamp": log.get("lamp_max_f"),
            "lamp_conf": log.get("lamp_confidence"),
            "ensemble_base": log.get("ensemble_base_f"),
            "ensemble_floor": log.get("ensemble_floor_f"),
            "ensemble_ceiling": log.get("ensemble_ceiling_f"),
            "ensemble_spread": log.get("ensemble_spread_f"),
            "ensemble_confidence": log.get("ensemble_confidence"),
            "hrrr_outlier": log.get("hrrr_outlier_detected")
        })
        
        # Market probabilities (using pre-parsed cache)
        log_bracket_probs = parsed_json_cache.get(i, {})
        total_prob = sum(log_bracket_probs.values()) if log_bracket_probs else 0
        is_market_valid = total_prob > 0
        
        for b in top_brackets:
            if not is_market_valid:
                market_probs[b].append(None)
            else:
                prob = log_bracket_probs.get(b, 0.0)
                market_probs[b].append(prob * 100 if prob is not None else None)

    # Alpha calculation based on the latest data
    # Alpha = HELIOS prediction - Midpoint of TOP 1 Polymarket bracket
    alpha_val = 0.0
    alpha_direction = "neutral"
    if chronological_logs:
        last = chronological_logs[-1]
        h_last = last.get("helios_pred")
        top1_bracket = last.get("market_top1_bracket")  # e.g., "48-49°F"
        
        if h_last is not None and top1_bracket:
            # Parse bracket to get midpoint (e.g., "48-49°F" -> 48.5, or "48°F or higher" -> 48)
            try:
                cleaned = top1_bracket.replace("°F", "").replace(" or higher", "").replace(" or lower", "")
                bracket_nums = cleaned.split("-")
                if len(bracket_nums) == 2:
                    low = float(bracket_nums[0])
                    high = float(bracket_nums[1])
                    top1_midpoint = (low + high) / 2
                else:
                    # Single value like "48" from "48°F or higher"
                    top1_midpoint = float(bracket_nums[0])
                alpha_val = round(abs(h_last - top1_midpoint), 1)
                alpha_direction = "long" if h_last > top1_midpoint else "short"
            except:
                pass  # Fall back to 0 alpha 

    # Helper function to get bracket from temperature
    def get_bracket_for_temp(temp_f):
        if temp_f is None:
            return None
        low = int(temp_f)
        # Brackets are typically "48-49°F", "46-47°F", etc.
        if low % 2 == 1:  # Odd number, round down to even
            low -= 1
        high = low + 1
        return f"{low}-{high}°F"

    # Calculate HELIOS bracket range for shading (use latest prediction)
    helios_bracket = None
    helios_bracket_min = None
    helios_bracket_max = None
    if chronological_logs:
        latest_helios = chronological_logs[-1].get("helios_pred")
        if latest_helios:
            helios_bracket = get_bracket_for_temp(latest_helios)
            low = int(latest_helios)
            if low % 2 == 1:
                low -= 1
            helios_bracket_min = low
            helios_bracket_max = low + 2  # Range is 2°F

    # Prepare TEMPERATURE chart datasets (Panel Superior)
    temp_datasets = [
        {
            "label": "HELIOS Prediction", 
            "data": helios_data, 
            "borderColor": "#0078FF", 
            "backgroundColor": "#0078FF22",
            "tension": 0.5,
            "pointRadius": 0,
            "borderWidth": 2.5,
            "order": 1
        },
        {
            "label": "NOAA METAR", # v6.0: Renamed from Cumulative Max
            "data": cumulative_max_data, 
            "borderColor": "#00FF7F", 
            "backgroundColor": "#00FF7F22",
            "tension": 0.5,
            "pointRadius": 0,
            "borderWidth": 2.5,
            "order": 2
        },
        {
            "label": "WU Forecast High", 
            "data": wu_forecast_data, 
            "borderColor": "#F4D03F", 
            "tension": 0.5,
            "pointRadius": 0,
            "borderWidth": 2,
            "fill": False,
            "order": 3
        },
        {
            "label": "WU Live Temp", # v6.0: Real-time WU reading
            "data": wu_live_data, 
            "borderColor": "#FF6B35", # Orange
            "tension": 0.5,
            "pointRadius": 0,
            "borderWidth": 1.5,
            "borderDash": [3, 3],
            "fill": False,
            "order": 4
        },
        {
            "label": "HRRR Raw Max", # v6.2: Raw model output
            "data": hrrr_raw_data, 
            "borderColor": "#87CEEB", # Light blue
            "tension": 0.5,
            "pointRadius": 0,
            "borderWidth": 1.5,
            "borderDash": [5, 5],
            "fill": False,
            "order": 5
        },
        # v8.0: Racing METAR Extra Datasets (Distinct Colors)
        {
            "label": "NOAA JSON API",
            "data": metar_json_data,
            "borderColor": "#10B981", # Emerald
            "tension": 0.5,
            "pointRadius": 0,
            "borderWidth": 1.2,
            "fill": False,
            "hidden": True, # Hidden by default to avoid clutter
            "order": 6
        },
        {
            "label": "AWC TDS (XML)",
            "data": metar_xml_data,
            "borderColor": "#8B5CF6", # Violet
            "tension": 0.5,
            "pointRadius": 0,
            "borderWidth": 1.2,
            "fill": False,
            "hidden": True,
            "order": 7
        },
        {
            "label": "NOAA FTP (TXT)",
            "data": metar_txt_data,
            "borderColor": "#EC4899", # Pink
            "tension": 0.5,
            "pointRadius": 0,
            "borderWidth": 1.2,
            "fill": False,
            "hidden": True,
            "order": 8
        }
    ]

    # Prepare PROBABILITY chart datasets (Panel Inferior)
    # v6.0: Extended color palette for all brackets
    market_colors_extended = [
        "#00d4ff",  # Cyan
        "#0078FF",  # Blue
        "#F4C724",  # Yellow
        "#FF6B35",  # Orange
        "#00FF7F",  # Green
        "#AF7AC5",  # Purple
        "#FF69B4",  # Pink
        "#20B2AA",  # Teal
        "#FFD700",  # Gold
        "#DC143C",  # Crimson
        "#9370DB",  # Medium Purple
        "#32CD32",  # Lime Green
    ]
    
    # Sort brackets by their current probability (most likely first)
    sorted_brackets = sorted(
        top_brackets,
        key=lambda b: market_probs[b][-1] if market_probs[b] and market_probs[b][-1] is not None else 0,
        reverse=True
    )
    
    prob_datasets = []
    for i, b in enumerate(sorted_brackets):
        if not b: continue
        prob_datasets.append({
            "label": b,
            "data": market_probs[b],
            "borderColor": market_colors_extended[i % len(market_colors_extended)],
            "backgroundColor": market_colors_extended[i % len(market_colors_extended)] + "22",
            "tension": 0.5, # v6.6: Super smooth lines
            "pointRadius": 0,
            "borderWidth": 2.5 if i < 3 else 1.5,  # Thicker lines for top 3
            "fill": False
        })

    # Sidebar data
    sidebar_data = {}
    if latest_market_log:
        sidebar_data = {
            "top1": {"bracket": latest_market_log.get("market_top1_bracket"), "prob": latest_market_log.get("market_top1_prob")},
            "top2": {"bracket": latest_market_log.get("market_top2_bracket"), "prob": latest_market_log.get("market_top2_prob")},
            "top3": {"bracket": latest_market_log.get("market_top3_bracket"), "prob": latest_market_log.get("market_top3_prob")},
        }

    # Process table data with divergence detection
    # v6.1: Sample logs to show only one entry every 15 minutes in the table
    # This keeps the table clean while charts keep full resolution
    table_with_divergence = []
    last_block = None
    
    for i, log in enumerate(logs):
        ts = datetime.fromisoformat(log["timestamp"]).replace(tzinfo=ZoneInfo("UTC"))
        ts_nyc = ts.astimezone(ZoneInfo("America/New_York"))
        
        # Group by 15-minute blocks (e.g., 14:00, 14:15, 14:30)
        block = f"{ts_nyc.hour}:{ (ts_nyc.minute // 15) * 15 }"
        
        # Keep the very first log (absolute latest) OR the first log we hit for each 15-min block
        # Since logs are DESC, the first one we hit for a block is the latest for that block.
        if i == 0 or block != last_block:
            last_block = block
            
            helios_pred = log.get("helios_pred")
            market_top1 = log.get("market_top1_bracket")
            market_top1_prob = log.get("market_top1_prob")
            
            # Calculate divergence
            pred_bracket = get_bracket_for_temp(helios_pred) if helios_pred else None
            divergence_state = "UNKNOWN"
            potential_roi = None
            
            if pred_bracket and market_top1:
                if pred_bracket == market_top1:
                    divergence_state = "ALIGNED"
                else:
                    divergence_state = "DIVERGENCE"
                    # Find probability of HELIOS bracket in this log
                    helios_prob = None
                    if pred_bracket == log.get("market_top1_bracket"):
                        helios_prob = log.get("market_top1_prob")
                    elif pred_bracket == log.get("market_top2_bracket"):
                        helios_prob = log.get("market_top2_prob")
                    elif pred_bracket == log.get("market_top3_bracket"):
                        helios_prob = log.get("market_top3_prob")
                    
                    if helios_prob and helios_prob > 0:
                        potential_roi = round((1 / helios_prob - 1) * 100, 1)
            
            # Add new fields to log
            log_enhanced = dict(log)
            log_enhanced["helios_bracket"] = pred_bracket
            log_enhanced["divergence_state"] = divergence_state
            log_enhanced["potential_roi"] = potential_roi
            table_with_divergence.append(log_enhanced)
            
            # Limit table to 50 sampled entries
            if len(table_with_divergence) >= 50:
                break

    # v6.3: Backend Performance Optimization via Downsampling
    # If we have too many points (e.g. > 800), downsample arrays to prevent frontend lag
    target_points = 800
    if len(raw_timestamps) > target_points:
        step = len(raw_timestamps) // target_points
        if step < 1: step = 1
        
        # Apply downsampling to all parallel arrays
        indices = [i for i in range(0, len(raw_timestamps), step)]
        # Always include the last point
        if indices[-1] != len(raw_timestamps) - 1:
            indices.append(len(raw_timestamps) - 1)
            
        # Helper to slice list by indices
        def downsample_list(lst):
            return [lst[i] for i in indices]
            
        raw_timestamps = downsample_list(raw_timestamps)
        labels = downsample_list(labels)
        debug_data = downsample_list(debug_data)
        
        # Downsample datasets
        for ds in temp_datasets:
            ds["data"] = downsample_list(ds["data"])
        for ds in prob_datasets:
            ds["data"] = downsample_list(ds["data"])

    # v6.8: Persistent Alert Registry - Fetch from database
    from database import get_alerts_for_date
    db_alerts = get_alerts_for_date(station_id, target_date)
    
    alerts = []
    for alert in db_alerts:
        ts = datetime.fromisoformat(alert["timestamp"]).replace(tzinfo=ZoneInfo("UTC"))
        ts_cet = ts.astimezone(ZoneInfo("Europe/Madrid")).strftime("%H:%M")
        alerts.append({
            "id": f"{alert['alert_type']}_{alert['id']}",
            "title": alert["title"],
            "desc": alert["description"],
            "level": alert["level"],
            "diff": alert["diff_value"],
            "time": ts_cet
        })

    # v13.0: Market Resolution Detection
    # Check if the observed max has definitively resolved the market
    market_resolution_data = None
    if chronological_logs:
        # Get latest observed max (cumulative_max_f or metar_actual)
        observed_max_f = None
        observed_max_time = None
        for log in reversed(chronological_logs):
            cum_max = log.get("cumulative_max_f") or log.get("metar_actual")
            if cum_max is not None:
                if observed_max_f is None or cum_max > observed_max_f:
                    observed_max_f = cum_max
                    # Get time from timestamp
                    try:
                        ts = datetime.fromisoformat(log["timestamp"]).replace(tzinfo=ZoneInfo("UTC"))
                        ts_local = ts.astimezone(ZoneInfo(station.timezone))
                        observed_max_time = ts_local.strftime("%H:%M")
                    except:
                        pass
        
        # Get all brackets from latest log with valid market data
        all_brackets = []
        for i in range(len(chronological_logs) - 1, -1, -1):
            if parsed_json_cache.get(i):
                all_brackets = [(b, p) for b, p in parsed_json_cache[i].items()]
                break
        
        # Check resolution
        if all_brackets and observed_max_f:
            resolution = check_market_resolution(all_brackets, observed_max_f, observed_max_time)
            market_resolution_data = {
                "is_resolved": resolution.is_resolved,
                "winning_bracket": resolution.winning_bracket,
                "observed_max_f": resolution.observed_max_f,
                "resolution_threshold_f": resolution.resolution_threshold_f,
                "resolution_time": resolution.resolution_time,
                "remaining_upside": resolution.remaining_upside,
                "all_resolved_brackets": resolution.all_resolved_brackets
            }


    return {
        # Dual panel chart structure
        "chart_temperature": {
            "timestamps": raw_timestamps,
            "labels": labels,
            "datasets": temp_datasets
        },
        "chart_probability": {
            "timestamps": raw_timestamps,
            "labels": labels,
            "datasets": prob_datasets
        },
        "debug": debug_data,
        "alerts": alerts, # v6.4: Live alerts
        # HELIOS bracket range for shading
        "helios_bracket": helios_bracket,
        "helios_bracket_range": {
            "min": helios_bracket_min,
            "max": helios_bracket_max
        },
        # Legacy support - combined chart
        "chart": {
            "labels": labels,
            "datasets": temp_datasets + prob_datasets
        },
        # Table with divergence data
        "table": table_with_divergence,
        "sidebar": sidebar_data,
        "alpha": {"value": alpha_val, "direction": alpha_direction},
        # Stats (for KPI cards)
        "stats": {
            "total_logs": len(logs),
            "win_rate": 50.0,  # Placeholder
            "mae": 1.5  # Placeholder
        },
        # v13.0: Market Resolution Detection
        "market_resolution": market_resolution_data
    }

# v6.1: CSV Export API for full data analysis
from fastapi.responses import StreamingResponse
import io, csv

@app.get("/api/export_csv")
def export_csv_api(station_id: str = DEFAULT_STATION, window: str = "1h", date: str = None):
    """
    Export raw performance data as CSV for analysis.
    
    Window options:
      - 15m: Last 15 minutes
      - 30m: Last 30 minutes  
      - 1h: Last 1 hour
      - 6h: Last 6 hours
      - day: Full day (uses date param)
    """
    from database import get_performance_history_by_target_date
    
    # Determine time window
    nyc_tz = ZoneInfo("America/New_York")
    now_nyc = datetime.now(nyc_tz)
    
    window_minutes = {
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "6h": 360,
        "12h": 720,
        "day": 1440  # Full day
    }.get(window, 60)
    
    # Get data for target date
    if date:
        target_date = date
    else:
        target_date = now_nyc.date().isoformat()
    
    logs = get_performance_history_by_target_date(station_id, target_date)
    
    # Filter by time window (except for 'day' which gets all)
    if window != "day":
        cutoff = datetime.now(ZoneInfo("UTC")) - timedelta(minutes=window_minutes)
        filtered_logs = []
        for log in logs:
            ts = datetime.fromisoformat(log["timestamp"]).replace(tzinfo=ZoneInfo("UTC"))
            if ts >= cutoff:
                filtered_logs.append(log)
        logs = filtered_logs
    
    # Chronological order for CSV
    logs = logs[::-1]
    
    # Define CSV columns with clear labels
    columns = [
        # Timestamps
        ("timestamp_utc", "Timestamp (UTC)"),
        ("timestamp_nyc", "Timestamp (NYC)"),
        ("target_date", "Target Date"),
        
        # === HELIOS PREDICTION ===
        ("helios_pred", "HELIOS Prediction (°F)"),
        ("confidence_score", "HELIOS Confidence"),
        
        # === HELIOS CALCULATION COMPONENTS ===
        ("hrrr_max_raw_f", "HRRR Raw Max (°F)"),
        ("current_deviation_f", "Current Deviation (°F)"),
        ("physics_adjustment_f", "Physics Adjustment (°F)"),
        ("verified_floor_f", "Reality Floor (°F)"),
        ("physics_reason", "Physics Reason"),
        
        # === PHYSICS VARIABLES ===
        ("soil_moisture", "Soil Moisture (0-1)"),
        ("radiation", "Radiation (W/m²)"),
        ("sky_condition", "Sky Condition"),
        ("wind_dir", "Wind Direction (°)"),
        ("velocity_ratio", "Heating Velocity Ratio"),
        ("aod_550nm", "Aerosol Optical Depth (550nm)"),
        ("sst_delta_c", "SST Delta (°C)"),
        ("advection_adj_c", "Advection Adjustment (°C)"),
        
        # === OBSERVED TEMPERATURES ===
        ("metar_actual", "METAR Current Temp (°F)"),
        ("cumulative_max_f", "Cumulative Max Today (°F)"),
        ("wu_preliminary", "WU Current Temp (°F)"),
        ("wu_forecast_high", "WU Forecast High (°F)"),
        ("wu_final", "WU Verified Max (°F)"),
        
        # === POLYMARKET TOP 3 ===
        ("market_top1_bracket", "Polymarket #1 Bracket"),
        ("market_top1_prob", "Polymarket #1 Probability"),
        ("market_top2_bracket", "Polymarket #2 Bracket"),
        ("market_top2_prob", "Polymarket #2 Probability"),
        ("market_top3_bracket", "Polymarket #3 Bracket"),
        ("market_top3_prob", "Polymarket #3 Probability"),
        ("market_weighted_avg", "Polymarket Weighted Avg (°F)"),
        
        # === ALL POLYMARKET OPTIONS (JSON) ===
        ("market_all_brackets_json", "All Polymarket Options (JSON)"),
    ]
    
    # Build CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([col[1] for col in columns])
    
    # Write data rows
    for log in logs:
        ts_utc = datetime.fromisoformat(log["timestamp"]).replace(tzinfo=ZoneInfo("UTC"))
        ts_nyc = ts_utc.astimezone(nyc_tz)
        
        row = []
        for col_key, col_label in columns:
            if col_key == "timestamp_utc":
                row.append(ts_utc.isoformat())
            elif col_key == "timestamp_nyc":
                row.append(ts_nyc.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                value = log.get(col_key, "")
                # Format floats to 2 decimal places
                if isinstance(value, float):
                    row.append(f"{value:.4f}" if "prob" in col_key else f"{value:.2f}")
                else:
                    row.append(value if value is not None else "")
        writer.writerow(row)
    
    # Prepare response
    output.seek(0)
    filename = f"helios_{station_id}_{target_date}_{window}.csv"
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/history")
def get_history_api():
    """Get 7-day accuracy report."""
    return get_accuracy_report(7)

@app.get("/api/observed/{station_id}")
async def get_observed_max_api(station_id: str):
    """Get observed maximum temperature for today with cross-verification."""
    if station_id not in STATIONS: return {"error": "Invalid station"}
    try:
        from collector.wunderground_fetcher import get_observed_max
        obs = await get_observed_max(station_id)
        if not obs: return {"error": "No observation data available"}
        pred = get_latest_prediction(station_id)
        prediction_f = pred["final_prediction_f"] if pred else None
        diff_f = round(prediction_f - obs.high_temp_f, 1) if prediction_f and obs.high_temp_f else None
        
        # Serialize hourly history
        hourly_history = []
        if obs.hourly_history:
            for entry in obs.hourly_history:
                hourly_history.append({
                    "time": entry.time,
                    "temp_f": entry.temp_f,
                    "conditions": entry.conditions
                })
        
        # Determine status based on difference
        status = None
        if diff_f is not None:
            if abs(diff_f) <= 1.0:
                status = "accurate"
            elif diff_f > 0:
                status = "above_observed"
            else:
                status = "below_observed"
        
        # Check if peak has likely passed (after 4 PM local)
        station = STATIONS[station_id]
        local_now = datetime.now(ZoneInfo(station.timezone))
        peak_passed = local_now.hour >= 16
        
        return {
            "station_id": station_id, 
            "date": obs.date.isoformat(), 
            "observed_max_f": obs.high_temp_f, 
            "observed_max_time": obs.high_temp_time,
            "source": obs.source,
            "prediction_f": prediction_f, 
            "difference_f": diff_f,
            "status": status,
            "peak_passed": peak_passed,
            "verification": {
                "metar_max_f": obs.metar_max_f,
                "wunderground_max_f": obs.wunderground_max_f,
                "wunderground_current_f": obs.wunderground_current_f,
                "verification_status": obs.verification_status,
                "confidence": obs.confidence,
                "notes": obs.notes,
                # v8.0: Racing Results
                "metar_racing": getattr(obs, 'metar_racing', None)
            },
            "hourly_history": hourly_history
        }
    except Exception as e: 
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/station/{station_id}", response_class=HTMLResponse)
async def station_detail(request: Request, station_id: str):
    """Serve the station detail page."""
    station_id = (station_id or "").upper()
    if station_id not in STATIONS:
        return _render_template(request, "index.html", {"active_page": "home"})
    station = STATIONS[station_id]
    return _render_template(request, "station.html", {
        "station_id": station_id,
        "station_name": station.name,
        "timezone": station.timezone,
    })

@app.get("/station/{station_id}/analytics", response_class=HTMLResponse)
async def read_analytics(request: Request, station_id: str):
    station_id = (station_id or "").upper()
    station = STATIONS.get(station_id)
    if not station:
        return _render_template(request, "home.html", {"active_page": "home"})
    return _render_template(request, "analytics.html", {
        "station_id": station_id,
        "station_name": station.name,
        "active_page": "analytics"
    })

# v12.0: Prediction page - real-time calculation display
@app.get("/station/{station_id}/prediction", response_class=HTMLResponse)
async def read_prediction(request: Request, station_id: str):
    station_id = (station_id or "").upper()
    station = STATIONS.get(station_id)
    if not station:
        return _render_template(request, "home.html", {"active_page": "home"})
    return _render_template(request, "prediction.html", {
        "station_id": station_id,
        "station_name": station.name,
        "active_page": "prediction"
    })

# v12.0: Data page - historical data display
@app.get("/station/{station_id}/data", response_class=HTMLResponse)
async def read_data(request: Request, station_id: str):
    station_id = (station_id or "").upper()
    station = STATIONS.get(station_id)
    if not station:
        return _render_template(request, "home.html", {"active_page": "home"})
    
    # Get today's date for default
    from datetime import date
    today = date.today().isoformat()
    
    return _render_template(request, "data.html", {
        "station_id": station_id,
        "station_name": station.name,
        "active_page": "data",
        "today": today
    })

# v7.0: Fullscreen chart view
@app.get("/station/{station_id}/chart/{chart_type}", response_class=HTMLResponse)
async def fullscreen_chart(request: Request, station_id: str, chart_type: str):
    """Serve the fullscreen chart page."""
    station_id = (station_id or "").upper()
    if station_id not in STATIONS:
        return _render_template(request, "home.html", {"active_page": "home"})
    if chart_type not in ["temperature", "probability"]:
        return RedirectResponse(url=f"/station/{station_id}/analytics", status_code=302)
    
    station = STATIONS[station_id]
    chart_titles = {
        "temperature": "TARGET vs REALITY",
        "probability": "MARKET BATTLE"
    }
    
    return _render_template(request, "fullscreen_chart.html", {
        "station_id": station_id,
        "station_name": station.name,
        "chart_type": chart_type,
        "chart_title": chart_titles.get(chart_type, "Chart"),
        "active_page": "analytics"
    })

# v6.0+: Generic station-scoped pages resolved from persisted selection (cookie/query)
@app.get("/analytics", response_class=HTMLResponse)
async def read_analytics_redirect(request: Request, station_id: str | None = None):
    resolved_station = _resolve_selected_station(request, fallback=station_id)
    return await read_analytics(request, resolved_station)


@app.get("/prediction", response_class=HTMLResponse)
async def read_prediction_generic(request: Request, station_id: str | None = None):
    resolved_station = _resolve_selected_station(request, fallback=station_id)
    return await read_prediction(request, resolved_station)


@app.get("/data", response_class=HTMLResponse)
async def read_data_generic(request: Request, station_id: str | None = None):
    resolved_station = _resolve_selected_station(request, fallback=station_id)
    return await read_data(request, resolved_station)

# Phase 2: World Dashboard - Real-time Event-Driven UI
@app.get("/world", response_class=HTMLResponse)
async def world_dashboard(request: Request):
    """Phase 2 World State Dashboard - all sources in real-time."""
    return _render_template(request, "world.html", {
        "active_page": "world"
    })


# Phase 3: Nowcast Dashboard - Distribution-based predictions
# =============================================================================
# Phase 5: Backtest Endpoints
# =============================================================================

@app.get("/api/v5/backtest/dates")
async def get_backtest_dates(station_id: str = None):
    """List dates available for backtesting."""
    from fastapi.responses import JSONResponse
    try:
        from core.backtest import get_dataset_builder
        builder = get_dataset_builder()
        dates = builder.list_available_dates(station_id)
        return {"dates": [d.isoformat() for d in dates]}
    except Exception as e:
        logger.error(f"Failed to get backtest dates: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "dates": []}
        )


def _parse_backtest_mode(mode: str):
    """Parse API mode aliases into BacktestMode."""
    from core.backtest import BacktestMode

    mode_norm = (mode or "").strip().lower()
    if mode_norm in ("signal_only", "signal"):
        return BacktestMode.SIGNAL_ONLY
    if mode_norm in ("execution_aware", "execution"):
        return BacktestMode.EXECUTION_AWARE
    # Fallback to execution for unknown values
    return BacktestMode.EXECUTION_AWARE


def _build_backtest_policy(policy: str, selection_mode: str, risk_profile: str):
    """Instantiate policy object from request params."""
    from core.backtest import (
        create_conservative_policy,
        create_aggressive_policy,
        create_fade_policy,
        create_maker_passive_policy,
    )
    from core.autotrader import create_multi_bandit_policy

    policy_norm = (policy or "conservative").strip().lower()

    if policy_norm == "aggressive":
        return create_aggressive_policy()
    if policy_norm == "fade":
        return create_fade_policy()
    if policy_norm == "maker_passive":
        return create_maker_passive_policy()
    if policy_norm == "multi_bandit":
        return create_multi_bandit_policy(
            selection_mode=(selection_mode or "static"),
            risk_profile=(risk_profile or "risk_first"),
        )
    if policy_norm == "toy_debug":
        from core.backtest import create_toy_debug_policy
        return create_toy_debug_policy()
    return create_conservative_policy()


def _trim_backtest_payload(payload: dict, report_level: str):
    """Trim heavy timeline payload based on report level."""
    report_norm = (report_level or "summary").strip().lower()
    if report_norm == "summary":
        for day in payload.get("day_results", []):
            day.pop("prediction_points", None)
            day.pop("decision_points", None)
            day.pop("orders", None)
            day.pop("fills", None)
    elif report_norm == "day_detail":
        for day in payload.get("day_results", []):
            day["prediction_points"] = day.get("prediction_points", [])[:200]
            day["decision_points"] = day.get("decision_points", [])[:200]
            day["orders"] = day.get("orders", [])[:200]
            day["fills"] = day.get("fills", [])[:200]
    # full_timeline -> unchanged


def _run_backtest_core(
    station_id: str,
    start_date: str,
    end_date: str,
    mode: str,
    policy: str,
    selection_mode: str,
    risk_profile: str,
    report_level: str,
):
    """
    Run backtest and return normalized tuple:
      (payload, error_payload, status_code)
    """
    from datetime import date
    from core.backtest import BacktestEngine, get_dataset_builder

    bt_mode = _parse_backtest_mode(mode)
    bt_policy = _build_backtest_policy(policy, selection_mode, risk_profile)
    engine = BacktestEngine(policy=bt_policy, mode=bt_mode)

    result = engine.run(
        station_id=station_id,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
    )

    payload = result.to_dict()
    coverage = payload.get("coverage", {})
    if (coverage.get("days_with_data") or 0) == 0:
        builder = get_dataset_builder()
        available_dates = [d.isoformat() for d in builder.list_available_dates(station_id)[:10]]
        return (
            None,
            {
                "error": "No recorded data found for requested range/station",
                "station_id": station_id,
                "start_date": start_date,
                "end_date": end_date,
                "available_dates": available_dates,
                "date_reference": "NYC market date",
            },
            422,
        )

    _trim_backtest_payload(payload, report_level)
    payload["request"] = {
        "mode": mode,
        "policy": policy,
        "selection_mode": selection_mode,
        "risk_profile": risk_profile,
        "report_level": report_level,
    }
    return payload, None, 200


def _safe_float(value: Optional[float], default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _score_suite_run(payload: dict) -> float:
    """
    Composite score for auto-compare ranking.
    Higher is better.
    """
    trading = payload.get("trading_summary", {}) or {}
    metrics = payload.get("aggregated_metrics", {}) or {}
    calibration = metrics.get("calibration", {}) or {}
    coverage = payload.get("coverage", {}) or {}

    pnl = _safe_float(trading.get("total_pnl_net"), 0.0)
    sharpe = _safe_float(trading.get("sharpe_ratio"), 0.0)
    win_rate = _safe_float(trading.get("win_rate"), 0.0)
    max_dd = _safe_float(trading.get("max_drawdown"), 0.0)
    fills = _safe_float(trading.get("total_fills"), 0.0)
    brier = _safe_float(calibration.get("brier_global"), 0.35)

    days_with_data = int(coverage.get("days_with_data") or 0)
    days_with_labels = int(coverage.get("days_with_labels") or 0)
    coverage_ratio = (days_with_labels / days_with_data) if days_with_data > 0 else 0.0

    # Pragmatic weighted score: reward PnL/risk-adjusted returns and penalize drawdown/calibration error.
    score = 0.0
    score += pnl * 0.40
    score += sharpe * 8.0
    score += win_rate * 20.0
    score -= max_dd * 90.0
    score -= brier * 18.0
    score += coverage_ratio * 12.0
    score += min(fills, 300.0) * 0.02
    return round(score, 4)


@app.post("/api/v5/backtest/run")
async def run_backtest(
    station_id: str,
    start_date: str,
    end_date: str,
    mode: str = "signal",
    policy: str = "conservative",
    selection_mode: str = "static",
    risk_profile: str = "risk_first",
    report_level: str = "summary",
):
    """Run a backtest."""
    from fastapi.responses import JSONResponse

    try:
        payload, err, status = _run_backtest_core(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            policy=policy,
            selection_mode=selection_mode,
            risk_profile=risk_profile,
            report_level=report_level,
        )
        if status != 200:
            return JSONResponse(
                status_code=status,
                content=err or {"error": "Unknown backtest error"},
            )
        return payload

    except Exception as e:
        import traceback
        logger.error(f"Backtest failed: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )


@app.post("/api/v5/backtest/run_suite")
async def run_backtest_suite(
    station_id: str,
    start_date: str,
    end_date: str,
    suite_name: str = "default_auto_compare_v1",
    report_level: str = "summary",
):
    """
    Run curated strategy suite with streaming progress.

    Streams NDJSON: progress events then final result.
    This endpoint powers Simple mode in /backtest.
    """
    import json as _json

    suite_runs = [
        {
            "run_id": "conservative_exec",
            "label": "Conservative (Execution)",
            "mode": "execution",
            "policy": "conservative",
            "selection_mode": "static",
            "risk_profile": "risk_first",
        },
        {
            "run_id": "aggressive_exec",
            "label": "Aggressive (Execution)",
            "mode": "execution",
            "policy": "aggressive",
            "selection_mode": "static",
            "risk_profile": "pnl_first",
        },
        {
            "run_id": "fade_exec",
            "label": "Fade Event QC (Execution)",
            "mode": "execution",
            "policy": "fade",
            "selection_mode": "static",
            "risk_profile": "risk_first",
        },
        {
            "run_id": "maker_passive_exec",
            "label": "Maker Passive (Execution)",
            "mode": "execution",
            "policy": "maker_passive",
            "selection_mode": "static",
            "risk_profile": "risk_first",
        },
        {
            "run_id": "multi_bandit_risk",
            "label": "Multi-Bandit LinUCB (Risk First)",
            "mode": "execution",
            "policy": "multi_bandit",
            "selection_mode": "linucb",
            "risk_profile": "risk_first",
        },
        {
            "run_id": "multi_bandit_pnl",
            "label": "Multi-Bandit LinUCB (PnL First)",
            "mode": "execution",
            "policy": "multi_bandit",
            "selection_mode": "linucb",
            "risk_profile": "pnl_first",
        },
    ]

    total = len(suite_runs)

    async def generate():
        run_summaries = []
        successful_runs = []
        first_error = None

        for i, run_cfg in enumerate(suite_runs):
            # Stream progress: which strategy is running now
            yield _json.dumps({
                "type": "progress",
                "index": i,
                "total": total,
                "label": run_cfg["label"],
            }) + "\n"

            payload, err, status = await asyncio.to_thread(
                _run_backtest_core,
                station_id=station_id,
                start_date=start_date,
                end_date=end_date,
                mode=run_cfg["mode"],
                policy=run_cfg["policy"],
                selection_mode=run_cfg["selection_mode"],
                risk_profile=run_cfg["risk_profile"],
                report_level=report_level,
            )

            if status != 200:
                if first_error is None:
                    first_error = err or {"error": "Unknown suite run error"}
                run_summaries.append(
                    {
                        "run_id": run_cfg["run_id"],
                        "label": run_cfg["label"],
                        "request": {
                            "mode": run_cfg["mode"],
                            "policy": run_cfg["policy"],
                            "selection_mode": run_cfg["selection_mode"],
                            "risk_profile": run_cfg["risk_profile"],
                        },
                        "status": "error",
                        "error": (err or {}).get("error", "Unknown error"),
                    }
                )
                continue

            score = _score_suite_run(payload)
            trading = payload.get("trading_summary", {}) or {}
            calibration = (payload.get("aggregated_metrics", {}) or {}).get("calibration", {}) or {}
            coverage = payload.get("coverage", {}) or {}

            summary = {
                "run_id": run_cfg["run_id"],
                "label": run_cfg["label"],
                "request": {
                    "mode": run_cfg["mode"],
                    "policy": run_cfg["policy"],
                    "selection_mode": run_cfg["selection_mode"],
                    "risk_profile": run_cfg["risk_profile"],
                },
                "status": "ok",
                "score": score,
                "trading": {
                    "total_pnl_net": _safe_float(trading.get("total_pnl_net")),
                    "win_rate": _safe_float(trading.get("win_rate")),
                    "sharpe_ratio": _safe_float(trading.get("sharpe_ratio")),
                    "max_drawdown": _safe_float(trading.get("max_drawdown")),
                    "total_fills": int(trading.get("total_fills") or 0),
                },
                "calibration": {
                    "brier_global": _safe_float(calibration.get("brier_global"), 0.0),
                    "ece": _safe_float(calibration.get("ece"), 0.0),
                },
                "coverage": coverage,
            }
            run_summaries.append(summary)
            successful_runs.append((summary, payload))

        # Build final result
        if not successful_runs:
            yield _json.dumps({
                "type": "result",
                "data": {
                    "error": "No successful runs in strategy suite",
                    "suite_name": suite_name,
                    "first_error": first_error,
                    "runs": run_summaries,
                },
            }) + "\n"
            return

        successful_runs.sort(key=lambda x: x[0]["score"], reverse=True)
        recommended_summary, recommended_payload = successful_runs[0]

        recommendation_reason = (
            f"Best composite score {recommended_summary['score']:.2f}, "
            f"PnL ${recommended_summary['trading']['total_pnl_net']:.2f}, "
            f"Sharpe {recommended_summary['trading']['sharpe_ratio']:.2f}, "
            f"Max DD {(recommended_summary['trading']['max_drawdown'] * 100):.1f}%."
        )

        ok_runs = [r for r in run_summaries if r.get("status") == "ok"]
        err_runs = [r for r in run_summaries if r.get("status") != "ok"]
        ok_runs.sort(key=lambda x: x.get("score", -999999), reverse=True)

        yield _json.dumps({
            "type": "result",
            "data": {
                "kind": "suite",
                "suite_name": suite_name,
                "station_id": station_id,
                "start_date": start_date,
                "end_date": end_date,
                "runs": ok_runs + err_runs,
                "recommendation": {
                    "run_id": recommended_summary["run_id"],
                    "label": recommended_summary["label"],
                    "reason": recommendation_reason,
                },
                "recommended_result": recommended_payload,
                "guide": {
                    "mode": "simple_auto_compare",
                    "note": "Use Advanced mode to tune one strategy manually after selecting a baseline from this ranking.",
                },
            },
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/api/v5/backtest/labels/{station_id}")
async def get_labels(station_id: str, start_date: str = None, end_date: str = None):
    """Get labels for a station."""
    from core.backtest import get_label_manager
    mgr = get_label_manager()

    if start_date and end_date:
        from datetime import date
        labels = mgr.get_labels_for_range(
            station_id,
            date.fromisoformat(start_date),
            date.fromisoformat(end_date)
        )
        return {"labels": [l.to_dict() for l in labels]}

    return {"stats": mgr.get_stats()}


@app.get("/api/v5/backtest/metrics")
async def get_metric_functions():
    """List available metric functions."""
    return {
        "calibration": ["brier_score", "log_loss", "ece", "sharpness"],
        "point": ["mae", "rmse", "bias"],
        "stability": ["prediction_churn", "flip_count"],
        "tpeak": ["accuracy", "mass_near_truth"]
    }


@app.get("/api/v5/backtest/policies")
async def get_policies():
    """List available trading policies."""
    return {
        "policies": [
            {"name": "conservative", "description": "Strict risk limits, high confidence required"},
            {"name": "aggressive", "description": "Looser limits, more trading"},
            {"name": "fade", "description": "Specialized for fading events"},
            {"name": "maker_passive", "description": "Passive maker-style policy with queue-based fills"},
            {"name": "multi_bandit", "description": "Strategy selector wrapper (static or LinUCB)"},
            {"name": "toy_debug", "description": "Debug policy: 1 taker buy/hour for pipeline testing"}
        ]
    }


@app.get("/api/v5/backtest/day_detail/{station_id}")
async def get_day_detail(
    station_id: str,
    date: str,
    mode: str = "execution_aware",
    limit: int | None = None,
    downsample: int = 1
):
    """
    Get detailed day-level backtest results for debugging.

    Returns:
        coverage: Data availability stats
        timeline_summary: First/last timestamp, interval
        reason_counters: Why trades did/didn't happen
        outputs: predictions, signals, orders, fills
    """
    from datetime import date as date_type
    from fastapi.responses import JSONResponse

    try:
        from core.backtest import (
            BacktestEngine, BacktestMode,
            create_conservative_policy
        )

        bt_mode = BacktestMode.EXECUTION_AWARE if mode == "execution_aware" else BacktestMode.SIGNAL_ONLY
        engine = BacktestEngine(
            policy=create_conservative_policy(),
            mode=bt_mode
        )

        market_date = date_type.fromisoformat(date)
        day_result = engine._run_day(
            station_id=station_id,
            market_date=market_date,
            interval_seconds=60
        )

        # Downsample/limit for UI performance
        predictions = day_result.prediction_points
        decisions = day_result.decision_points
        if downsample and downsample > 1:
            predictions = predictions[::downsample]
            decisions = decisions[::downsample]
        if limit and limit > 0:
            predictions = predictions[:limit]
            decisions = decisions[:limit]

        # Build detailed response
        response = {
            "station_id": station_id,
            "market_date": date,
            "mode": mode,

            "coverage": {
                "has_label": day_result.actual_winner is not None,
                "nowcast_events_count": day_result.nowcast_count,
                "market_events_count": len([p for p in day_result.prediction_points if p.market_snapshot]),
                "timeline_steps": day_result.timeline_steps,
                "timesteps_with_nowcast": day_result.timesteps_with_nowcast,
                "timesteps_with_market": day_result.timesteps_with_market,
                "raw_market_events_in_store": day_result.market_count,
                "raw_nowcast_events_in_store": day_result.nowcast_count,
                "raw_world_events_in_store": day_result.metar_count,
            },

            "timeline_summary": {
                "first_ts_utc": day_result.first_ts_utc.isoformat() if day_result.first_ts_utc else None,
                "last_ts_utc": day_result.last_ts_utc.isoformat() if day_result.last_ts_utc else None,
                "interval_seconds": day_result.interval_seconds,
            },

            "reason_counters": day_result.reason_counters,

            "outputs": {
                "predictions": [p.to_dict() for p in predictions],
                "decisions": [d.to_dict() for d in decisions],
                "signals_total": day_result.signals_generated,
                "orders": [o.to_dict() if hasattr(o, "to_dict") else o for o in day_result.orders],
                "fills": [f.to_dict() for f in day_result.fills],
            },

            "summary": {
                "predicted_winner": day_result.predicted_winner,
                "actual_winner": day_result.actual_winner,
                "predicted_tmax": day_result.predicted_tmax,
                "actual_tmax": day_result.actual_tmax,
                "pnl_net": day_result.pnl_net,
                "status": day_result.status,
            }
        }

        return response

    except Exception as e:
        import traceback
        logger.error(f"Day detail failed: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )


# =============================================================================
# Phase 6: Autotrader + Learning API
# =============================================================================


class AutoTraderControlRequest(BaseModel):
    action: str
    station_id: Optional[str] = None


class LearningRunRequest(BaseModel):
    station_id: str = DEFAULT_STATION
    train_days: int = 53
    val_days: int = 7
    max_combinations: int = 50
    mode: str = "signal_only"


class BacktestReplayRequest(BaseModel):
    station_id: str
    start_date: str
    end_date: str
    mode: str = "execution"
    selection_mode: str = "linucb"
    risk_profile: str = "risk_first"


def _to_float_or_none(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values) -> Optional[float]:
    for v in values:
        out = _to_float_or_none(v)
        if out is not None:
            return out
    return None


def _resolve_yes_probability_from_bracket(bracket: Dict[str, Any]) -> Optional[float]:
    yes_bid_direct = _first_float(bracket.get("ws_yes_best_bid"), bracket.get("ws_best_bid"))
    yes_ask_direct = _first_float(bracket.get("ws_yes_best_ask"), bracket.get("ws_best_ask"))
    no_bid = _first_float(bracket.get("ws_no_best_bid"))
    no_ask = _first_float(bracket.get("ws_no_best_ask"))

    implied_yes_bid_from_no = (1.0 - no_ask) if no_ask is not None else None
    implied_yes_ask_from_no = (1.0 - no_bid) if no_bid is not None else None

    bid_candidates = [v for v in [yes_bid_direct, implied_yes_bid_from_no] if v is not None]
    ask_candidates = [v for v in [yes_ask_direct, implied_yes_ask_from_no] if v is not None]
    yes_bid = max(bid_candidates) if bid_candidates else None
    yes_ask = min(ask_candidates) if ask_candidates else None

    yes_mid = _first_float(bracket.get("ws_yes_mid"), bracket.get("ws_mid"))
    yes_price_ref = _first_float(bracket.get("yes_price"))
    selected = select_probability_01_from_quotes(
        best_bid=yes_bid,
        best_ask=yes_ask,
        mid=yes_mid,
        reference=yes_price_ref,
        wide_spread_threshold=0.04,
        reference_tolerance=0.005,
    )
    if selected is None:
        return None
    return max(0.0, min(1.0, float(selected)))


def _convert_market_label_to_f(label: str, market_unit: str) -> str:
    """
    Convert Celsius market labels to Fahrenheit labels so downstream logic
    (nowcast buckets, calibration, observed-max floor) stays unit-consistent.
    """
    clean = str(label or "")
    for _ in range(2):
        try:
            fixed = clean.encode("latin1").decode("utf-8")
            if fixed == clean:
                break
            clean = fixed
        except Exception:
            break
    clean = clean.replace("\u00C2", "")
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return ""
    if str(market_unit or "F").upper() != "C":
        return clean

    def c_to_f(v: float) -> float:
        return (v * 9.0 / 5.0) + 32.0

    lower = clean.lower()
    # If label already carries Fahrenheit, keep as-is.
    if re.search("\\d\\s*[\\u00B0\\u00BA]?\\s*f\\b", lower):
        return clean

    range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", clean)
    if range_match:
        low_f = int(round(c_to_f(float(range_match.group(1)))))
        high_f = int(round(c_to_f(float(range_match.group(2)))))
        if high_f < low_f:
            low_f, high_f = high_f, low_f
        return f"{low_f}-{high_f}°F"

    num_match = re.search(r"-?\d+(?:\.\d+)?", clean)
    if not num_match:
        return clean

    threshold_f = int(round(c_to_f(float(num_match.group(0)))))
    if "or below" in lower:
        return f"{threshold_f}°F or below"
    if "or above" in lower or "or higher" in lower:
        return f"{threshold_f}°F or higher"
    return f"{threshold_f}°F"


def _normalize_prob_dict(raw: Dict[str, float]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for label, prob in raw.items():
        if not label:
            continue
        p = max(0.0, float(prob))
        cleaned[normalize_label(label)] = p
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in cleaned.items()}


def _bucket_is_impossible_with_observed_max(label: str, observed_max_f: Optional[float]) -> bool:
    if observed_max_f is None:
        return False
    aligned_obs = int(round(float(observed_max_f)))
    kind, low, high = parse_label(label)
    if kind == "range" and high is not None:
        return high < aligned_obs
    if kind == "below" and high is not None:
        return high < aligned_obs
    if kind == "single" and low is not None:
        return low < aligned_obs
    return False


def _top_bucket_summary(p_bucket: List[Dict[str, Any]]) -> Dict[str, Any]:
    top_label = ""
    top_prob = 0.0
    for b in p_bucket:
        if not isinstance(b, dict):
            continue
        prob = float(b.get("probability", b.get("prob", 0.0)) or 0.0)
        if prob > top_prob:
            top_prob = prob
            top_label = normalize_label(str(b.get("label", b.get("bucket", ""))))
    return {"label": top_label, "probability": top_prob}


def _calibrate_nowcast_with_market(
    raw_p_bucket: List[Dict[str, Any]],
    market_probs: Dict[str, float],
    *,
    observed_max_f: Optional[float],
    nowcast_target_date: Optional[str],
    requested_target_date: str,
) -> Dict[str, Any]:
    raw_entries: List[Dict[str, Any]] = []
    raw_probs: Dict[str, float] = {}
    for b in raw_p_bucket or []:
        if not isinstance(b, dict):
            continue
        label = normalize_label(str(b.get("label", b.get("bucket", ""))))
        if not label:
            continue
        prob = float(b.get("probability", b.get("prob", 0.0)) or 0.0)
        raw_entries.append({"label": label, "probability": prob})
        raw_probs[label] = prob

    raw_probs = _normalize_prob_dict(raw_probs)
    market_probs = _normalize_prob_dict(market_probs)

    market_top_prob = max(market_probs.values()) if market_probs else 0.0
    blend_weight = 0.20
    if market_top_prob >= 0.90:
        blend_weight += 0.55
    elif market_top_prob >= 0.80:
        blend_weight += 0.40
    elif market_top_prob >= 0.65:
        blend_weight += 0.25
    elif market_top_prob >= 0.50:
        blend_weight += 0.15
    if observed_max_f is not None:
        blend_weight += 0.10
    if nowcast_target_date and nowcast_target_date != requested_target_date:
        blend_weight += 0.20
    blend_weight = max(0.0, min(0.90, blend_weight))

    ordered_labels = [e["label"] for e in raw_entries]
    for label, _ in sorted(market_probs.items(), key=lambda kv: kv[1], reverse=True):
        if label not in ordered_labels:
            ordered_labels.append(label)

    filtered_by_floor: List[str] = []
    blended: Dict[str, float] = {}
    for label in ordered_labels:
        p_now = raw_probs.get(label, 0.0)
        p_mkt = market_probs.get(label, 0.0)
        p = ((1.0 - blend_weight) * p_now) + (blend_weight * p_mkt)
        if _bucket_is_impossible_with_observed_max(label, observed_max_f):
            p = 0.0
            filtered_by_floor.append(label)
        blended[label] = p

    total = sum(blended.values())
    if total <= 0:
        # Fallback to raw distribution if all got filtered.
        blended = dict(raw_probs)
        total = sum(blended.values())

    if total > 0:
        for k in list(blended.keys()):
            blended[k] = blended[k] / total

    calibrated_bucket = [
        {"label": label, "probability": prob}
        for label, prob in sorted(blended.items(), key=lambda kv: kv[1], reverse=True)
    ]
    raw_bucket = [
        {"label": label, "probability": prob}
        for label, prob in sorted(raw_probs.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return {
        "raw_bucket": raw_bucket,
        "calibrated_bucket": calibrated_bucket,
        "blend_weight": blend_weight,
        "overlap_count": len(set(raw_probs.keys()) & set(market_probs.keys())),
        "market_count": len(market_probs),
        "raw_count": len(raw_probs),
        "filtered_by_observed_max": filtered_by_floor,
        "raw_top": _top_bucket_summary(raw_bucket),
        "calibrated_top": _top_bucket_summary(calibrated_bucket),
    }


@app.get("/api/v6/autotrader/nowcast")
async def autotrader_nowcast(target_day: int = 0):
    from core.autotrader import get_autotrader_service
    from core.nowcast_integration import get_nowcast_integration

    service = get_autotrader_service()
    station_id = service.config.station_id
    station = STATIONS.get(station_id)
    if not station:
        return {"error": f"Unknown station for autotrader: {station_id}"}

    station_tz = ZoneInfo(station.timezone)
    requested_target_date = (datetime.now(station_tz) + timedelta(days=int(target_day))).date().isoformat()

    integration = get_nowcast_integration()
    distribution = integration.get_distribution(station_id)
    if not distribution:
        # Best effort generation if the engine exists but has no cached distribution yet.
        try:
            engine = integration.get_engine(station_id)
            distribution = engine.generate_distribution()
        except Exception:
            distribution = None
    if not distribution:
        return {
            "error": "No nowcast distribution available",
            "station_id": station_id,
            "requested_target_date": requested_target_date,
        }

    raw = distribution.to_dict()
    market_payload = await get_polymarket_dashboard_data(station_id=station_id, target_day=target_day, depth=5)

    market_probs: Dict[str, float] = {}
    market_error = None
    station_market_unit = get_polymarket_temp_unit(station_id)
    if isinstance(market_payload, dict):
        market_error = market_payload.get("error")
        for b in market_payload.get("brackets", []) if isinstance(market_payload.get("brackets"), list) else []:
            raw_label = str(b.get("name", ""))
            label = normalize_label(_convert_market_label_to_f(raw_label, station_market_unit))
            if not label:
                continue
            yes_prob = _resolve_yes_probability_from_bracket(b)
            if yes_prob is None:
                continue
            market_probs[label] = yes_prob

    # Prefer nowcast engine state (backfilled from NOAA history) over DB-only observed max.
    observed_max_f_db = get_observed_max_for_target_date(station_id, requested_target_date)
    observed_max_f_state: Optional[float] = None
    try:
        if raw.get("target_date") == requested_target_date:
            state = integration.get_engine(station_id).get_or_create_daily_state(target_date=distribution.target_date)
            if state.max_so_far_aligned_f > -900:
                observed_max_f_state = float(state.max_so_far_aligned_f)
    except Exception:
        observed_max_f_state = None

    observed_max_f = observed_max_f_db
    if observed_max_f_state is not None:
        observed_max_f = observed_max_f_state if observed_max_f is None else max(float(observed_max_f), observed_max_f_state)
    calibrated = _calibrate_nowcast_with_market(
        raw_p_bucket=raw.get("p_bucket") or [],
        market_probs=market_probs,
        observed_max_f=observed_max_f,
        nowcast_target_date=raw.get("target_date"),
        requested_target_date=requested_target_date,
    )

    warnings: List[str] = []
    if market_error:
        warnings.append(f"market_error:{market_error}")
    if raw.get("target_date") and raw.get("target_date") != requested_target_date:
        warnings.append("nowcast_target_mismatch")

    return {
        **raw,
        "station_id": station_id,
        "target_day": int(target_day),
        "requested_target_date": requested_target_date,
        "observed_max_f": observed_max_f,
        "market_probs": market_probs,
        "raw_top": calibrated["raw_top"],
        "calibrated": {
            "p_bucket": calibrated["calibrated_bucket"],
            "top": calibrated["calibrated_top"],
            "blend_weight": calibrated["blend_weight"],
        },
        "diagnostics": {
            "raw_count": calibrated["raw_count"],
            "market_count": calibrated["market_count"],
            "overlap_count": calibrated["overlap_count"],
            "filtered_by_observed_max": calibrated["filtered_by_observed_max"],
            "warnings": warnings,
        },
    }


@app.get("/api/v6/autotrader/status")
async def autotrader_status():
    from core.autotrader import get_autotrader_service
    return get_autotrader_service().get_status()


@app.post("/api/v6/autotrader/control")
async def autotrader_control(req: AutoTraderControlRequest):
    from core.autotrader import get_autotrader_service
    service = get_autotrader_service()
    action = (req.action or "").strip().lower()

    if action == "start":
        await service.start()
    elif action == "pause":
        service.pause()
    elif action == "resume":
        service.resume()
    elif action == "risk_off":
        service.risk_off()
    elif action == "risk_on":
        service.risk_on()
    elif action == "change_station":
        if not req.station_id:
            return {"ok": False, "error": "station_id required for change_station"}
        try:
            service.change_station(req.station_id)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
    else:
        return {"ok": False, "error": f"unsupported action: {req.action}"}

    return {"ok": True, "action": action, "status": service.get_status()}


@app.get("/api/v6/autotrader/strategies")
async def autotrader_strategies():
    from core.autotrader import get_autotrader_service
    return get_autotrader_service().get_strategies()


@app.get("/api/v6/autotrader/positions")
async def autotrader_positions():
    from core.autotrader import get_autotrader_service
    return get_autotrader_service().get_positions()


@app.get("/api/v6/autotrader/performance")
async def autotrader_performance():
    from core.autotrader import get_autotrader_service
    return get_autotrader_service().get_performance()


@app.get("/api/v6/autotrader/decisions")
async def autotrader_decisions(limit: int = 200):
    from core.autotrader import get_autotrader_service
    return get_autotrader_service().get_decisions(limit=limit)


@app.get("/api/v6/autotrader/orders")
async def autotrader_orders(limit: int = 200):
    from core.autotrader import get_autotrader_service
    return get_autotrader_service().get_orders(limit=limit)


@app.get("/api/v6/autotrader/fills")
async def autotrader_fills(limit: int = 200):
    from core.autotrader import get_autotrader_service
    return get_autotrader_service().get_fills(limit=limit)


@app.get("/api/v6/autotrader/intraday")
async def autotrader_intraday(target_day: int = 0, limit: int = 2000):
    from core.autotrader import get_autotrader_service

    service = get_autotrader_service()
    payload = service.get_intraday_timeline(target_day=target_day, limit=limit)

    station_id = payload.get("station_id")
    target_date = ((payload.get("window") or {}).get("target_date")) or ""
    observed_max_f = None
    market_series: List[Dict[str, Any]] = []
    if station_id and target_date:
        observed_max_f_db = get_observed_max_for_target_date(station_id, target_date)
        observed_max_f = observed_max_f_db
        try:
            from core.nowcast_integration import get_nowcast_integration
            td = date.fromisoformat(str(target_date))
            state = get_nowcast_integration().get_engine(station_id).get_or_create_daily_state(target_date=td)
            if state.max_so_far_aligned_f > -900:
                state_max = float(state.max_so_far_aligned_f)
                observed_max_f = state_max if observed_max_f is None else max(float(observed_max_f), state_max)
        except Exception:
            pass
        try:
            for row in iter_performance_history_by_target_date(station_id, target_date):
                market_series.append(
                    {
                        "ts_utc": row.get("timestamp"),
                        "market_top1_bracket": row.get("market_top1_bracket"),
                        "market_top1_prob": row.get("market_top1_prob"),
                        "helios_pred": row.get("helios_pred"),
                        "cumulative_max_f": row.get("cumulative_max_f"),
                    }
                )
        except Exception:
            pass

    payload["observed_max_f"] = observed_max_f
    payload["market_series"] = market_series
    return payload


@app.post("/api/v6/learning/run")
async def learning_run(req: LearningRunRequest):
    from core.autotrader import LearningConfig, get_autotrader_service

    cfg = LearningConfig(
        station_id=req.station_id,
        train_days=req.train_days,
        val_days=req.val_days,
        max_combinations=req.max_combinations,
        mode=req.mode,
    )
    service = get_autotrader_service()
    return service.learning.run_once(cfg)


@app.get("/api/v6/learning/runs")
async def learning_runs(limit: int = 20):
    from core.autotrader import get_autotrader_service
    return get_autotrader_service().get_learning_runs(limit=limit)


@app.post("/api/v6/autotrader/backtest/replay")
async def autotrader_backtest_replay(req: BacktestReplayRequest):
    from datetime import date
    from core.backtest import BacktestEngine, BacktestMode, create_conservative_policy
    from core.autotrader import create_multi_bandit_policy

    bt_mode = BacktestMode.SIGNAL_ONLY if req.mode == "signal" else BacktestMode.EXECUTION_AWARE
    start_date = date.fromisoformat(req.start_date)
    end_date = date.fromisoformat(req.end_date)

    baseline_engine = BacktestEngine(
        policy=create_conservative_policy(),
        mode=bt_mode,
    )
    baseline = baseline_engine.run(req.station_id, start_date, end_date)

    bandit_engine = BacktestEngine(
        policy=create_multi_bandit_policy(
            selection_mode=req.selection_mode,
            risk_profile=req.risk_profile,
        ),
        mode=bt_mode,
    )
    bandit = bandit_engine.run(req.station_id, start_date, end_date)

    return {
        "config": {
            "station_id": req.station_id,
            "start_date": req.start_date,
            "end_date": req.end_date,
            "mode": req.mode,
            "selection_mode": req.selection_mode,
            "risk_profile": req.risk_profile,
        },
        "baseline": {
            "policy": "conservative",
            "trading_summary": baseline.to_dict().get("trading_summary"),
            "aggregated_metrics": baseline.to_dict().get("aggregated_metrics"),
            "coverage": baseline.to_dict().get("coverage"),
        },
        "candidate": {
            "policy": "multi_bandit",
            "trading_summary": bandit.to_dict().get("trading_summary"),
            "aggregated_metrics": bandit.to_dict().get("aggregated_metrics"),
            "coverage": bandit.to_dict().get("coverage"),
        },
    }


# =============================================================================
# ATENEA: AI Diagnostic Copilot (Phase Auxiliar)
# =============================================================================

from core.atenea import (
    get_atenea_chat,
    get_evidence_builder,
    get_intent_router,
    ScreenContext
)


class AteneaChatRequest(BaseModel):
    """Request body for Atenea chat."""
    query: str
    screen: str = "unknown"
    station_id: str | None = None
    mode: str = "LIVE"
    time_range: dict | None = None
    selected_token_ids: list | None = None


@app.post("/api/atenea/chat")
async def atenea_chat(request: AteneaChatRequest):
    """
    Main Atenea chat endpoint.

    Processes a user question with screen context and returns
    an evidence-backed response.
    """
    try:
        chat = get_atenea_chat()

        # Build screen context
        screen_context = {
            "screen": request.screen,
            "station_id": request.station_id,
            "mode": request.mode,
        }
        if request.time_range:
            screen_context["time_range"] = request.time_range
        if request.selected_token_ids:
            screen_context["selected_token_ids"] = request.selected_token_ids

        # Inject live state if available
        try:
            from core.nowcast_integration import get_nowcast_integration
            integration = get_nowcast_integration()
            if integration:
                snapshot = integration.get_snapshot()
                chat.set_live_state(
                    nowcast_state=snapshot.get("distributions", {}),
                    health_state=snapshot.get("states", {})
                )
        except Exception as e:
            logging.warning(f"Failed to inject nowcast state: {e}")

        # Also try to get world state
        try:
            from core.world import get_world
            world = get_world()
            world_snapshot = world.get_snapshot()
            chat.evidence_builder.set_live_state(
                world_state=world_snapshot
            )
        except Exception as e:
            logging.warning(f"Failed to inject world state: {e}")

        # Process chat
        response = await chat.chat(request.query, screen_context)

        return {
            "success": True,
            "response": response.to_dict()
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/api/atenea/context/live")
async def atenea_live_context(station_id: str | None = None):
    """
    Get current live context for Atenea.

    Returns the current state of market, world, nowcast, and health.
    """
    try:
        evidence_builder = get_evidence_builder()

        # Try to inject live state
        try:
            from core.nowcast_integration import get_nowcast_integration
            integration = get_nowcast_integration()
            if integration:
                snapshot = integration.get_snapshot()
                evidence_builder.set_live_state(
                    nowcast_state=snapshot.get("distributions", {})
                )
        except Exception as e:
            logging.warning(f"Failed to inject nowcast state: {e}")

        try:
            from core.world import get_world
            world = get_world()
            evidence_builder.set_live_state(
                world_state=world.get_snapshot()
            )
        except Exception as e:
            logging.warning(f"Failed to inject world state: {e}")

        evidences = evidence_builder.gather_live_context(station_id)

        return {
            "success": True,
            "evidences": [e.to_dict() for e in evidences],
            "station_id": station_id
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/atenea/intent")
async def atenea_classify_intent(query: str, screen: str = "unknown"):
    """
    Classify a query's intent without generating a response.

    Useful for debugging and understanding how Atenea routes questions.
    """
    try:
        router = get_intent_router()
        intent = router.classify(query, {"screen": screen})
        requirements = router.get_evidence_requirements(intent)

        return {
            "success": True,
            "intent": intent.to_dict(),
            "evidence_requirements": requirements
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.delete("/api/atenea/history")
async def atenea_clear_history():
    """Clear Atenea conversation history."""
    try:
        chat = get_atenea_chat()
        chat.clear_history()
        return {"success": True, "message": "History cleared"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/atenea/history")
async def atenea_get_history():
    """Get Atenea conversation history."""
    try:
        chat = get_atenea_chat()
        return {
            "success": True,
            "history": chat.get_history()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Phase 5: Backtest Dashboard
@app.get("/backtest", response_class=HTMLResponse)
async def backtest_dashboard(request: Request):
    """Phase 5 Backtest Dashboard."""
    return _render_template(request, "backtest.html", {
        "active_page": "backtest"
    })


@app.get("/autotrader", response_class=HTMLResponse)
async def autotrader_dashboard(request: Request):
    """Phase 6 Autotrader Dashboard - realtime signals, risk, and execution."""
    return _render_template(request, "autotrader.html", {
        "active_page": "autotrader"
    })


@app.get("/autotrader/tutorial", response_class=HTMLResponse)
async def autotrader_tutorial(request: Request):
    """Autotrader tutorial page with architecture, concepts, and operating guide."""
    return _render_template(request, "autotrader_tutorial.html", {
        "active_page": "autotrader"
    })


@app.get("/nowcast", response_class=HTMLResponse)
async def nowcast_dashboard(request: Request):
    """Phase 3 Nowcast Dashboard - P(bucket), t_peak, confidence."""
    return _render_template(request, "nowcast.html", {
        "active_page": "nowcast"
    })


# Phase 3: Model Debug Dashboard
@app.get("/nowcast/debug", response_class=HTMLResponse)
async def nowcast_debug(request: Request):
    """Phase 3 Model Debug Dashboard - internal state visualization."""
    return _render_template(request, "nowcast_debug.html", {
        "active_page": "nowcast"
    })


# Phase 4: Replay Dashboard
@app.get("/replay", response_class=HTMLResponse)
async def replay_dashboard(request: Request):
    """Phase 4 Replay Dashboard - session replay with timeline."""
    active = get_active_stations()
    station_market_units = {
        sid: get_polymarket_temp_unit(sid)
        for sid in active.keys()
    }
    return _render_template(request, "replay.html", {
        "active_page": "replay",
        "station_market_units": station_market_units,
    })


@app.get("/polymarket", response_class=HTMLResponse)
async def polymarket_dashboard(request: Request):
    """Polymarket Dashboard - live market data, orderbook, sentiment."""
    return _render_template(request, "polymarket.html", {
        "active_page": "polymarket"
    })


# ATENEA: AI Diagnostic Copilot
@app.get("/atenea", response_class=HTMLResponse)
async def atenea_dashboard(request: Request):
    """ATENEA - AI Diagnostic Copilot with evidence-based answers."""
    return _render_template(request, "atenea.html", {
        "active_page": "atenea"
    })


# v12.0: Home page with sidebar
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return _render_template(request, "home.html", {
        "active_page": "home"
    })


if __name__ == "__main__":
    import uvicorn
    from database import init_database
    init_database()
    uvicorn.run(app, host="0.0.0.0", port=8000)
