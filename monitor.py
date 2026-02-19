"""
Helios Weather Lab - Performance Monitor
Background worker for logging real-time performance and reconciling historical data.
"""

import asyncio
from datetime import datetime, timedelta
import logging
from typing import Optional

from config import STATIONS, get_active_stations
from database import insert_performance_log, get_pending_reconciliation_logs, update_performance_wu_final, get_latest_prediction, get_latest_prediction_for_date, get_latest_performance_log
from collector.wunderground_fetcher import fetch_wunderground_max
from collector.metar_fetcher import fetch_metar, fetch_metar_history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monitor")

# v8.2: Global state cache to allow independent fast/slow updates.
# Keep it dynamic so newly active stations (e.g., LTAC) are not silently skipped.
_state_cache = {
    sid: {"history": None, "wu_obs": None, "market_today": None, "market_tmr": None}
    for sid in get_active_stations().keys()
}
_weather_cache_time = {}  # station_id -> datetime (for TTL tracking)


def _ensure_station_cache(station_id: str) -> None:
    if station_id not in _state_cache:
        _state_cache[station_id] = {
            "history": None,
            "wu_obs": None,
            "market_today": None,
            "market_tmr": None,
        }

async def background_slow_refresh(station_id, today_local, tomorrow_local):
    """Refreshes slow data sources in the background without blocking the fast METAR line."""
    global _state_cache
    _ensure_station_cache(station_id)
    try:
        from market.polymarket_checker import get_market_all_options
        from collector.wunderground_fetcher import fetch_wunderground_max
        from collector.metar_fetcher import fetch_metar_history

        tasks = [
            fetch_metar_history(station_id, hours=20),
            fetch_wunderground_max(station_id),
            get_market_all_options(station_id, today_local),
            get_market_all_options(station_id, tomorrow_local)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        _state_cache[station_id]["history"] = results[0] if not isinstance(results[0], Exception) else _state_cache[station_id]["history"]
        _state_cache[station_id]["wu_obs"] = results[1] if not isinstance(results[1], Exception) else _state_cache[station_id]["wu_obs"]
        _state_cache[station_id]["market_today"] = results[2] if not isinstance(results[2], Exception) else _state_cache[station_id]["market_today"]
        _state_cache[station_id]["market_tmr"] = results[3] if not isinstance(results[3], Exception) else _state_cache[station_id]["market_tmr"]
        
        logger.debug(f"✅ Background refresh complete for {station_id}")
    except Exception as e:
        logger.error(f"❌ Background refresh error for {station_id}: {e}")

async def snapshot_current_state():
    """
    Capture current state for all stations.
    v8.2: Decoupled architecture. METAR is fetched every cycle (FAST), 
    while Market/History are fetched in the background (SLOW).
    """
    global _weather_cache_time, _state_cache
    
    active_ids = list(get_active_stations().keys())
    for station_id in active_ids:
        try:
            _ensure_station_cache(station_id)
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(STATIONS[station_id].timezone)
            now_local = datetime.now(tz)
            today_local = now_local.date()
            tomorrow_local = today_local + timedelta(days=1)
            
            # 1. Background Refresh Trigger (every 30s)
            now = datetime.now()
            station_cache_time = _weather_cache_time.get(station_id)
            if station_cache_time is None or (now - station_cache_time).total_seconds() > 30: # 30s TTL
                _weather_cache_time[station_id] = now
                # Fire and forget slow refresh
                asyncio.create_task(background_slow_refresh(station_id, today_local, tomorrow_local))
            
            # 2. FAST PATH: Fetch Live METAR Racing Data ONLY
            # This is what drives the chart line responsiveness
            current = await fetch_metar(station_id)
            if not current:
                continue

            # 3. Pull latest available data from state cache
            st_state = _state_cache[station_id]
            history = st_state["history"]
            wu_obs = st_state["wu_obs"]
            market_today = st_state["market_today"]
            market_tmr = st_state["market_tmr"]

            # Process METAR History
            todays_temps = []
            if history:
                for h in history:
                    h_local = h.observation_time.astimezone(tz)
                    if h_local.date() == today_local:
                         todays_temps.append(h.temp_f)
            
            # Live METAR
            metar_current = current.temp_f if current else None
            
            # Racing Extra Data (Today only)
            racing_results = current.racing_results if current else {}
            metar_json = racing_results.get("NOAA_JSON_API")
            metar_xml = racing_results.get("AWC_TDS_XML")
            metar_txt = racing_results.get("NOAA_TG_FTP_TXT")

            # Cumulative Max (Today)
            cumulative_max_today = 0.0
            if todays_temps:
                cumulative_max_today = max(todays_temps)
            
            # v8.1: Protocol RACING - Check all individual sources for a higher max
            all_current_temps = [metar_current] if metar_current else []
            if racing_results:
                all_current_temps.extend([v for v in racing_results.values() if v is not None])
            
            if all_current_temps:
                max_now = max(all_current_temps)
                if max_now > cumulative_max_today:
                    cumulative_max_today = max_now
                
            # WU Max (Today)
            wu_val_today = wu_obs.high_temp_f if wu_obs else None
            log_wu_forecast = wu_obs.forecast_high_f if wu_obs else None
            
            # 4. Iterate Target Days [0, 1]
            for offset in [0, 1]:

                target_date = today_local + timedelta(days=offset)
                target_date_str = target_date.isoformat()
                market_data = market_today if offset == 0 else market_tmr
                
                # A. Get HELIOS Prediction for this specific target
                pred = get_latest_prediction_for_date(station_id, target_date_str)
                helios_val = pred["final_prediction_f"] if pred else 0.0
                confidence = pred["delta_weight"] if pred else 0.0
                
                # C. Determine Actuals (Only for Today)
                if offset == 0:
                    log_metar_actual = metar_current
                    log_wu_preliminary = wu_obs.current_temp_f if wu_obs else None  # v6.0: Use current, not max
                    log_cumulative_max = cumulative_max_today
                    # log_wu_forecast already set above
                else:
                    # For future, actuals are None
                    log_metar_actual = None
                    log_wu_preliminary = None
                    log_cumulative_max = None
                    log_wu_forecast = None
                
                # D. Log to DB
                m_args = {}
                if market_data:
                    m_args = {
                        "market_top1_bracket": market_data.top1_bracket,
                        "market_top1_prob": market_data.top1_prob,
                        "market_top2_bracket": market_data.top2_bracket,
                        "market_top2_prob": market_data.top2_prob,
                        "market_top3_bracket": market_data.top3_bracket,
                        "market_top3_prob": market_data.top3_prob,
                        "market_weighted_avg": market_data.weighted_avg,
                        "market_all_brackets_json": market_data.all_brackets_json  # v6.0
                    }

                # v5.8 Safety Check: If temp drops > 10F suddenly, it's likely a sensor error
                if offset == 0:
                    last_log = get_latest_performance_log(station_id)
                    if last_log:
                        last_metar = last_log.get("metar_actual")
                        last_cum_max = last_log.get("cumulative_max_f")
                        
                        # Check METAR
                        if last_metar is not None and log_metar_actual is not None:
                            if log_metar_actual < (last_metar - 10.0):
                                logger.warning(f"Safety Triggered: METAR dropped from {last_metar} to {log_metar_actual}. Ignoring reading.")
                                log_metar_actual = last_metar # Carry forward last good value
                                
                        # Check Cumulative Max
                        if last_cum_max is not None and log_cumulative_max is not None:
                            if log_cumulative_max < (last_cum_max - 10.0):
                                logger.warning(f"Safety Triggered: Cumulative Max dropped from {last_cum_max} to {log_cumulative_max}. Ignoring reading.")
                                log_cumulative_max = last_cum_max # Carry forward last good value
                            
                        # v5.8 Market Safety: If all market probabilities are 0, it's a glitch/outage
                        current_m_total = (market_data.top1_prob + market_data.top2_prob + market_data.top3_prob) if market_data else 0
                        if market_data and current_m_total == 0:
                            logger.warning(f"Market Safety Triggered: Probabilities are 0. Carrying over last good market data.")
                            m_args = {
                                "market_top1_bracket": last_log.get("market_top1_bracket"),
                                "market_top1_prob": last_log.get("market_top1_prob"),
                                "market_top2_bracket": last_log.get("market_top2_bracket"),
                                "market_top2_prob": last_log.get("market_top2_prob"),
                                "market_top3_bracket": last_log.get("market_top3_bracket"),
                                "market_top3_prob": last_log.get("market_top3_prob"),
                                "market_weighted_avg": last_log.get("market_weighted_avg")
                            }
                        elif market_data and last_log.get("market_top1_prob"):
                            # If Top 1 drops by more than 40% absolute in 3-15 seconds, it's very suspicious
                            last_p1 = last_log.get("market_top1_prob")
                            if last_p1 > 0.4 and market_data.top1_prob < (last_p1 - 0.4):
                                logger.warning(f"Market Safety Triggered: Top 1 prob crashed from {last_p1} to {market_data.top1_prob}. Ignoring glitch.")
                                m_args["market_top1_prob"] = last_p1  # Fix the crash in the log reading
                
                # v10.0/v11.0: Fetch NBM/LAMP and calculate ensemble (Today only)
                ensemble_args = {}
                if offset == 0:
                    try:
                        from collector.nbm_fetcher import fetch_nbm
                        from collector.lamp_fetcher import fetch_lamp
                        from synthesizer.ensemble import calculate_ensemble_v11
                        from zoneinfo import ZoneInfo
                        
                        nbm_data = await fetch_nbm(station_id, target_date)
                        lamp_data = await fetch_lamp(station_id)
                        
                        hrrr_val = pred.get("hrrr_max_raw_f") if pred else None
                        sky_cond = pred.get("sky_condition") if pred else "SCT"
                        radiation = pred.get("radiation") if pred else None
                        
                        # v12.0: Get local hour AND minute for Ultimate Squeeze/Sunset Lock
                        station = STATIONS.get(station_id)
                        tz_name = station.timezone if station else "America/New_York"
                        now_tz = datetime.now(ZoneInfo(tz_name))
                        local_hour = now_tz.hour
                        local_minute = now_tz.minute
                        
                        if hrrr_val and (nbm_data or lamp_data):
                            ensemble = calculate_ensemble_v11(
                                hrrr_max_f=hrrr_val,
                                nbm_max_f=nbm_data.max_temp_f if nbm_data else None,
                                lamp_max_f=lamp_data.max_temp_f if lamp_data else None,
                                lamp_confidence=lamp_data.confidence if lamp_data else 0.5,
                                metar_actual_f=log_metar_actual,
                                local_hour=local_hour,
                                local_minute=local_minute,  # v12.0: For Ultimate Squeeze
                                radiation=radiation,
                                sky_condition=sky_cond,
                                wu_forecast_high=log_wu_forecast,
                                velocity_ratio=pred.get("velocity_ratio") if pred else None,
                                observed_max_f=log_cumulative_max  # v12.0: For Sunset Lock
                            )
                            ensemble_args = {
                                "nbm_max_f": nbm_data.max_temp_f if nbm_data else None,
                                "lamp_max_f": lamp_data.max_temp_f if lamp_data else None,
                                "lamp_confidence": lamp_data.confidence if lamp_data else None,
                                "ensemble_base_f": ensemble.base_weighted_f,
                                "ensemble_floor_f": ensemble.helios_floor_f,
                                "ensemble_ceiling_f": ensemble.helios_ceiling_f,
                                "ensemble_spread_f": ensemble.ensemble_spread_f,
                                "ensemble_confidence": ensemble.confidence_level,
                                "hrrr_outlier_detected": ensemble.hrrr_outlier_detected
                            }
                    except Exception as e:
                        logger.warning(f"Ensemble fetch error: {e}")

                insert_performance_log(
                    station_id=station_id,
                    helios_pred=helios_val,
                    metar_actual=log_metar_actual,
                    wu_preliminary=log_wu_preliminary,
                    confidence_score=confidence,
                    cumulative_max_f=log_cumulative_max,
                    target_date=target_date_str, # v4.1
                    wu_forecast_high=log_wu_forecast, # v5.1
                    # v5.2 & v5.3 Components
                    hrrr_max_raw_f=pred.get("hrrr_max_raw_f") if pred else None,
                    current_deviation_f=pred.get("current_deviation_f") if pred else None,
                    physics_adjustment_f=pred.get("physics_adjustment_f") if pred else None,
                    soil_moisture=pred.get("soil_moisture") if pred else None,
                    radiation=pred.get("radiation") if pred else None,
                    sky_condition=pred.get("sky_condition") if pred else None,
                    wind_dir=pred.get("wind_dir") if pred else None,
                    velocity_ratio=pred.get("velocity_ratio") if pred else None,
                    aod_550nm=pred.get("aod_550nm") if pred else None,
                    sst_delta_c=pred.get("sst_delta_c") if pred else None,
                    advection_adj_c=pred.get("advection_adj_c") if pred else None,
                    verified_floor_f=pred.get("verified_floor_f") if pred else None,
                    physics_reason=pred.get("physics_reason") if pred else None,
                    # v8.0 Racing Extras (Only for Today)
                    metar_json_api_f=metar_json if offset == 0 else None,
                    metar_tds_xml_f=metar_xml if offset == 0 else None,
                    metar_tgftp_txt_f=metar_txt if offset == 0 else None,
                    # v10.0 Ensemble
                    **ensemble_args,
                    **m_args
                )
                
                # v6.8: State-based Alert Detection (only for today)
                if offset == 0 and log_wu_preliminary is not None and log_metar_actual is not None:
                    from database import get_latest_alert_type, insert_market_alert
                    
                    diff = log_wu_preliminary - log_metar_actual
                    THRESHOLD = 0.5  # Difference threshold for divergence
                    
                    # Determine current state (3 states: NORMAL, BULL, BEAR)
                    if diff > THRESHOLD:
                        current_state = "BULL"  # WU inflated vs METAR
                    elif diff < -THRESHOLD:
                        current_state = "BEAR"  # WU lagging behind METAR
                    else:
                        current_state = "NORMAL"
                    
                    # Get previous state from last alert type
                    last_alert = get_latest_alert_type(station_id, target_date_str)
                    
                    # Infer previous state from last alert
                    # None or 'recovery' = was NORMAL
                    # 'bull_trap' = was BULL
                    # 'rezago' = was BEAR
                    previous_state = "NORMAL"
                    if last_alert == "bull_trap":
                        previous_state = "BULL"
                    elif last_alert == "rezago":
                        previous_state = "BEAR"
                    
                    # Detect transitions
                    if previous_state == "NORMAL" and current_state == "BULL":
                        # NORMAL -> BULL = BULL TRAP
                        insert_market_alert(
                            station_id=station_id,
                            target_date=target_date_str,
                            alert_type="bull_trap",
                            level="warning",
                            title="🐂 BULL TRAP",
                            description=f"WU ({log_wu_preliminary:.1f}°F) > METAR ({log_metar_actual:.1f}°F)",
                            diff_value=round(diff, 1),
                            wu_value=log_wu_preliminary,
                            metar_value=log_metar_actual
                        )
                        logger.warning(f"ALERT: BULL TRAP - WU={log_wu_preliminary:.1f}, METAR={log_metar_actual:.1f}, Diff={diff:.1f}")
                    
                    elif previous_state == "NORMAL" and current_state == "BEAR":
                        # NORMAL -> BEAR = REZAGO
                        insert_market_alert(
                            station_id=station_id,
                            target_date=target_date_str,
                            alert_type="rezago",
                            level="warning",
                            title="🐻 REZAGO WU",
                            description=f"WU ({log_wu_preliminary:.1f}°F) < METAR ({log_metar_actual:.1f}°F)",
                            diff_value=round(diff, 1),
                            wu_value=log_wu_preliminary,
                            metar_value=log_metar_actual
                        )
                        logger.warning(f"ALERT: REZAGO - WU={log_wu_preliminary:.1f}, METAR={log_metar_actual:.1f}, Diff={diff:.1f}")
                    
                    elif previous_state in ("BULL", "BEAR") and current_state == "NORMAL":
                        # BULL/BEAR -> NORMAL = RECOVERY
                        insert_market_alert(
                            station_id=station_id,
                            target_date=target_date_str,
                            alert_type="recovery",
                            level="info",
                            title="✅ RECOVERY",
                            description=f"WU ({log_wu_preliminary:.1f}°F) ≈ METAR ({log_metar_actual:.1f}°F)",
                            diff_value=round(diff, 1),
                            wu_value=log_wu_preliminary,
                            metar_value=log_metar_actual
                        )
                        logger.info(f"ALERT: RECOVERY - WU={log_wu_preliminary:.1f}, METAR={log_metar_actual:.1f}, Diff={diff:.1f}")
                
                mkt_str = f"MktAvg={market_data.weighted_avg:.1f}" if market_data else "NoMkt"

        except Exception as e:
            logger.error(f"Error snapshotting {station_id}: {e}")

async def reconcile_history():

    """
    Back-fill 'wu_final' for logs > 4 hours old.
    We look up the exact hourly history from WU for that timestamp.
    """
    logger.info("Starting history reconciliation...")
    
    # Get logs that need reconciliation (Look back 4 hours)
    pending_logs = get_pending_reconciliation_logs(lookback_hours=4)
    
    if not pending_logs:
        logger.info("No logs pending reconciliation.")
        return

    logger.info(f"Found {len(pending_logs)} logs to reconcile.")
    
    # Group by station/date to minimize API calls
    # Key: (station_id, date_str) -> Value: List of logs
    tasks_map = {}
    for log in pending_logs:
        ts = datetime.fromisoformat(log["timestamp"])
        key = (log["station_id"], ts.date())
        if key not in tasks_map:
            tasks_map[key] = []
        tasks_map[key].append(log)
    
    for (station_id, date_obj), logs in tasks_map.items():
        try:
            # Fetch full history for that day
            logger.info(f"Fetching full history for {station_id} on {date_obj}")
            wu_obs = await fetch_wunderground_max(station_id, target_date=date_obj)
            
            if not wu_obs or not wu_obs.hourly_history:
                logger.warning(f"No history found for {station_id} on {date_obj}")
                continue
                
            # History is a list of entries with 'time' (Str like "10:53 PM")
            # We need to match log timestamp (UTC) to station local time
            from zoneinfo import ZoneInfo
            station_tz = ZoneInfo(STATIONS[station_id].timezone)
            
            for log in logs:
                log_utc = datetime.fromisoformat(log["timestamp"])
                # Convert UTC log time to Station Local Time
                log_local = log_utc.astimezone(station_tz)
                
                # Find the closest history entry
                # WU history times are strings "%I:%M %p". We need to parse them roughly.
                # Simplified matching: match hour and nearest minute?
                # Actually, let's just parse all history entries into datetimes
                
                best_match = None
                min_diff = timedelta(minutes=60) # Tolerance
                
                for entry in wu_obs.hourly_history:
                    if entry.time == "Unknown" or entry.time == "Ahora":
                        continue
                        
                    try:
                        # Parse entry time (it's just time, assume same day as target_date)
                        # Be careful with day rollovers (late night)
                        t_parsed = datetime.strptime(entry.time, "%I:%M %p").time()
                        entry_dt = datetime.combine(date_obj, t_parsed).replace(tzinfo=station_tz)
                        
                        # Calculate diff
                        diff = abs(entry_dt - log_local)
                        if diff < min_diff:
                            min_diff = diff
                            best_match = entry
                    except:
                        continue
                
                if best_match and min_diff < timedelta(minutes=35):
                    # Found a matching historical record! Update DB.
                    update_performance_wu_final(log["id"], best_match.temp_f)
                    logger.info(f"Reconciled Log {log['id']}: WU Final = {best_match.temp_f} (Diff: {min_diff})")
                else:
                    logger.debug(f"No match for log {log['id']} at {log_local.strftime('%I:%M %p')}")

        except Exception as e:
            logger.error(f"Error reconciling {station_id} on {date_obj}: {e}")
