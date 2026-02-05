# Helios Weather Lab - Main Orchestrator (Physics Engine)
# Scheduler for data collection and deterministic physics predictions.

# Load environment variables FIRST (before any other imports)
from dotenv import load_dotenv
load_dotenv()

import asyncio
import schedule
import time
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import traceback
from typing import List, Optional

from collector.wunderground_fetcher import get_observed_max

from config import STATIONS, COLLECTION_INTERVAL_MINUTES, get_active_stations
from database import init_database, get_accuracy_report
from collector import collect_all_data
from synthesizer import calculate_physics_prediction
from synthesizer.bias_correction import get_7day_bias, apply_bias_correction
from registrar import log_prediction
from deviation import capture_morning_trajectory, calculate_deviation
from market import get_target_date
from market.crowd_wisdom import track_probability_velocity, detect_market_shift, get_crowd_sentiment

# v3.0: Persistent cache for volatility and confidence
STATION_CACHE = {}


async def capture_trajectories(target_stations: list[str] = None):
    """Capture the HRRR trajectory for all stations."""
    print(f"\n{'─'*70}")
    print(f"  Captura de Trayectoria Horaria")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*70}")
    
    stations_to_process = target_stations if target_stations else get_active_stations().keys()

    for station_id in stations_to_process:
        try:
            print(f"\n  📊 {station_id}...", end=" ")
            data = await collect_all_data(station_id)
            forecast = data.get("hrrr")
            
            if forecast:
                await capture_morning_trajectory(station_id, forecast)
            else:
                print(f"    ⚠ Sin datos de pronóstico")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print(f"\n{'─'*70}\n")


async def collect_and_predict(station_id: str, target_days: List[int] = None) -> bool:
    """
    Collect data and generate physics-based predictions for next 3 days.
    Uses deterministic rules (no AI/LLM).
    
    Args:
        station_id: Station ID
        target_days: Optional list of specific day offsets to run [0, 1]
    """
    try:
        # Get station config for timezone
        station = STATIONS.get(station_id)
        tz_name = station.timezone if station else "America/New_York"
        local_now = datetime.now(timezone.utc).astimezone(ZoneInfo(tz_name))
        local_hour = local_now.hour
        
        # Check Polymarket market status for today
        try:
            target_date = await get_target_date(station_id, local_now)
        except Exception as e:
            print(f"[{station_id}] [WARN] Market status check failed: {e}")
            target_date = local_now.date()
        
        # Calculate starting day (0 = today, 1 = tomorrow)
        starting_day = (target_date - local_now.date()).days
        
        # If starting tomorrow and it's late, skip
        # If starting tomorrow and it's late, skip (only if auto mode)
        if target_days is None and starting_day > 0 and local_hour >= 20:
            print(f"\n  ⏭️  {station_id}: Mercado resuelto, esperando a mañana")
            return True
        
        # Collect current METAR data (always for today)
        data_metar = await collect_all_data(station_id, 0)
        metar = data_metar.get("metar")
        
        if not metar:
            print(f"\n  ✗ {station_id}: Sin datos METAR")
            return False
        
        # Generate predictions
        days_label = str(target_days) if target_days else "Próximos 2 días"
        print(f"\n{'  '*0}╔{'═'*66}╗")
        print(f"{'  '*0}║  {station_id} - Predicciones ({days_label}){' '*20}║")
        print(f"{'  '*0}╚{'═'*66}╝")
        
        prediction_count = 0
        
        days_to_process = target_days if target_days is not None else range(starting_day, starting_day + 2)
        
        for day_offset in days_to_process:
            try:
                # Fetch forecast for this target day
                data = await collect_all_data(station_id, day_offset)
                forecast = data.get("hrrr")
                
                if not forecast or not forecast.hrrr:
                    print(f"\n  ⚠  Día +{day_offset}: Sin pronóstico disponible")
                    continue
                
                hrrr = forecast.hrrr
                prediction_date = local_now.date() + timedelta(days=day_offset)

                # ============================================================
                # v13.0: Extract peak hour from HRRR hourly data
                # ============================================================
                peak_hour = None
                if forecast.hourly_temps_c and day_offset == 0:
                    # For today, find the hour with max temp in first 24 entries
                    day_temps = forecast.hourly_temps_c[:24]
                    if day_temps:
                        max_temp = max(t for t in day_temps if t is not None)
                        for i, t in enumerate(day_temps):
                            if t == max_temp:
                                peak_hour = i
                                break

                    # ============================================================
                    # Phase 3: Update Nowcast BaseForecast (HRRR hourly curve)
                    # ============================================================
                    try:
                        from core.nowcast_integration import get_nowcast_integration, extract_hourly_temps_f_for_date
                        station_tz = ZoneInfo(STATIONS[station_id].timezone)
                        target_date = datetime.now(station_tz).date()
                        hourly_temps_f = extract_hourly_temps_f_for_date(
                            forecast.hourly_temps_c,
                            forecast.hourly_times,
                            target_date,
                            station_tz
                        )
                        if hourly_temps_f:
                            integration = get_nowcast_integration()
                            integration.on_hrrr_update(
                                station_id=station_id,
                                hourly_temps_f=hourly_temps_f,
                                model_run_time=forecast.fetch_time,
                                model_source="HRRR",
                                target_date=target_date
                            )
                    except Exception:
                        pass

                # ============================================================
                # v10.0 ENSEMBLE: Multi-Model Consensus (HRRR + NBM + LAMP)
                # ============================================================
                from synthesizer.physics import celsius_to_fahrenheit
                hrrr_max_raw_f = celsius_to_fahrenheit(hrrr.max_forecast_c)
                
                # Fetch GFS from other sources if available in data
                gfs_max_c = data.get("gfs_max_c")
                if gfs_max_c:
                    gfs_max_f = celsius_to_fahrenheit(gfs_max_c)
                    # Use average as requested by user
                    hrrr_base_f = (hrrr_max_raw_f + gfs_max_f) / 2
                else:
                    hrrr_base_f = hrrr_max_raw_f
                
                # Fetch NBM and LAMP for ensemble (today/tomorrow only)
                nbm_data = None
                lamp_data = None
                ensemble_result = None
                
                if day_offset <= 1:
                    try:
                        from collector.nbm_fetcher import fetch_nbm
                        from collector.lamp_fetcher import fetch_lamp
                        from synthesizer.ensemble import calculate_ensemble_v11
                        from database import get_performance_history_by_target_date
                        
                        nbm_data = await fetch_nbm(station_id, prediction_date)
                        lamp_data = await fetch_lamp(station_id)
                        
                        # v12.0: Get observed_max for Sunset Lock
                        observed_max_f = None
                        if day_offset == 0:
                            today_str = prediction_date.isoformat()
                            logs = get_performance_history_by_target_date(station_id, today_str)
                            if logs:
                                temps = [l.get("cumulative_max_f") or l.get("metar_actual") for l in logs 
                                        if l.get("cumulative_max_f") or l.get("metar_actual")]
                                if temps:
                                    observed_max_f = max([t for t in temps if t])
                        
                        # v12.0: Use full ensemble calculation with protections
                        ensemble_result = calculate_ensemble_v11(
                            hrrr_max_f=hrrr_base_f,
                            nbm_max_f=nbm_data.max_temp_f if nbm_data else None,
                            lamp_max_f=lamp_data.max_temp_f if lamp_data else None,
                            lamp_confidence=lamp_data.confidence if lamp_data else 0.5,
                            metar_actual_f=celsius_to_fahrenheit(metar.temp_c) if metar else None,
                            local_hour=local_hour,
                            local_minute=local_now.minute,  # v12.0
                            observed_max_f=observed_max_f   # v12.0: For Sunset Lock
                        )
                        
                        # v12.0: Use ensemble CEILING as the cap, not just base
                        # This ensures predictions respect v12.0 protections
                        if ensemble_result.sunset_lock_active or ensemble_result.ultimate_squeeze_active:
                            # When protections active, prediction cannot exceed ceiling
                            hrrr_max_f = min(ensemble_result.base_weighted_f, ensemble_result.helios_ceiling_f)
                            print(f"     ⚡ v12.0: {ensemble_result.strategy_note}")
                        elif ensemble_result.nbm_value_f or ensemble_result.lamp_value_f:
                            hrrr_max_f = ensemble_result.base_weighted_f
                        else:
                            hrrr_max_f = hrrr_base_f
                            
                    except Exception as e:
                        print(f"[WARN] Ensemble fetch failed: {e}")
                        hrrr_max_f = hrrr_base_f
                else:
                    hrrr_max_f = hrrr_base_f

                # ============================================================
                # BIAS CORRECTION: 7-day rolling bias for KLGA (v3.0)
                # ============================================================
                if station_id == "KLGA" and day_offset <= 1:
                    klga_bias = get_7day_bias(station_id)
                    if abs(klga_bias) > 0.1:
                        hrrr_max_f = apply_bias_correction(hrrr_max_f, klga_bias)

                # ============================================================
                # VOLATILITY FILTER: Ignore >2°F jumps between updates (v3.0)
                # ============================================================
                if station_id == "KLGA" and day_offset == 0:
                    cache_key = f"{station_id}_last_hrrr"
                    last_hrrr = STATION_CACHE.get(cache_key)
                    if last_hrrr:
                        jump = abs(hrrr_max_f - last_hrrr)
                        if jump > 2.0:
                            # Volatility detected - hold last value
                            print(f"     ⚠️  VOLATILITY: Salto de {jump:+.1f}F detectado. Filtrando...")
                            hrrr_max_f = last_hrrr
                    
                    # Update cache
                    STATION_CACHE[cache_key] = hrrr_max_f

                # Convert back to Celsius for the physics engine
                hrrr_max_c_corrected = (hrrr_max_f - 32) * 5/9

                # Calculate hours until target (for temporal decay)
                # Assume target is max temp time (typically 14:00-16:00 local)
                target_datetime = datetime.combine(prediction_date, datetime.min.time().replace(hour=15))
                target_datetime = target_datetime.replace(tzinfo=local_now.tzinfo)
                hours_until_target = int((target_datetime - local_now).total_seconds() / 3600)
                
                # Calculate deviation only for today and tomorrow (not meaningful for far future)
                deviation = None
                temp_hrrr_hourly_c = hrrr.current_temp_c or metar.temp_c
                current_delta_f = 0.0
                delta_explanation = ""
                
                if day_offset <= 1:
                    # Use deviation tracking for today/tomorrow
                    deviation = calculate_deviation(
                        station_id=station_id,
                        actual_temp_c=metar.temp_c,
                        actual_sky=metar.sky_condition,
                        observation_time=metar.observation_time,
                    )
                    
                    if deviation and deviation.has_trajectory:
                        temp_hrrr_hourly_c = deviation.predicted_temp_c
                    
                    # Apply temporal decay to delta
                    from deviation.temporal_decay import calculate_adjusted_delta
                    from synthesizer.physics import celsius_to_fahrenheit
                    
                    raw_delta_f = celsius_to_fahrenheit(metar.temp_c) - celsius_to_fahrenheit(temp_hrrr_hourly_c)
                    current_delta_f, delta_explanation = calculate_adjusted_delta(
                        current_delta=raw_delta_f,
                        hours_until_target=hours_until_target,
                        station_id=station_id,
                        local_hour=local_hour,  # Pass local hour for time-of-day weighting
                        station_timezone=STATIONS[station_id].timezone
                    )
                else:
                    # Far future: no delta
                    current_delta_f = 0.0
                    delta_explanation = "No delta (día futuro)"
                
                # Calculate heating velocity (Feature C)
                velocity_ratio = None
                velocity_explanation = ""
                
                if day_offset == 0 and deviation and deviation.has_trajectory:
                    # Only for today with trajectory data
                    from deviation.heating_velocity import calculate_heating_velocity
                    from database import get_trajectory_for_station
                    
                    # Get trajectory data
                    trajectory_data = get_trajectory_for_station(
                        station_id=station_id,
                        forecast_date=prediction_date
                    )
                    
                    if trajectory_data:
                        trajectory = [(row['hour_datetime'], row['temp_c']) for row in trajectory_data]
                        velocity_ratio, velocity_explanation = calculate_heating_velocity(
                            station_id=station_id,
                            current_temp_c=metar.temp_c,
                            time_now=local_now,
                            trajectory=trajectory
                        )
                
                # PHYSICS ENGINE: Calculate prediction  
                # For far future days (2+), use 0 deviation to avoid false precision
                if day_offset >= 2:
                    # Use HRRR directly with physics adjustments only
                    # To force delta=0, make temp_actual == temp_hrrr_hourly
                    prediction = calculate_physics_prediction(
                        station_id=station_id,
                        temp_actual_c=hrrr_max_c_corrected,  # Force delta = 0
                        temp_hrrr_hourly_c=hrrr_max_c_corrected,
                        hrrr_max_c=hrrr_max_c_corrected,
                        soil_moisture=hrrr.soil_moisture,
                        radiation=hrrr.shortwave_radiation,
                        sky_condition="CLR",  # Don't use current sky for future
                        wind_dir=None,  # Don't use current wind for future
                        velocity_ratio=None,  # No velocity for far future
                        local_hour=local_hour,
                        peak_hour=peak_hour,  # v13.0: Pass peak hour
                    )
                else:
                    # Normal prediction with deviation and velocity for today/tomorrow
                    # Override the delta in temp values to use adjusted delta
                    from synthesizer.physics import celsius_to_fahrenheit
                    
                    # Calculate what temp_actual should be to produce our adjusted delta
                    hrrr_hourly_f = celsius_to_fahrenheit(temp_hrrr_hourly_c)
                    adjusted_actual_f = hrrr_hourly_f + current_delta_f
                    adjusted_actual_c = (adjusted_actual_f - 32) * 5/9
                    
                    # ============================================================
                    # ALPHA VARIABLES: Fetch SST and AOD for enhanced prediction
                    # ============================================================
                    aod_value = None
                    sst_delta_c = None
                    current_sst_c_val = None # NEW: For thermodynamic check
                    sst_display = None  # For output display
                    aod_display = None  # For output display
                    
                    if day_offset == 0:  # Only for today
                        try:
                            # Fetch SST from NOAA buoy
                            from collector.ndbc_fetcher import fetch_buoy_sst, is_wind_onshore
                            
                            sst_data = await fetch_buoy_sst(station_id)
                            if sst_data:
                                current_sst_c_val = sst_data.water_temp_c
                                sst_display = f"{sst_data.water_temp_f:.1f}°F (Buoy {sst_data.buoy_id})"
                                if is_wind_onshore(station_id, metar.wind_dir_degrees):
                                    # Calculate SST delta (expected - actual)
                                    expected_sst_c = 8.0  # Baseline winter expectation
                                    sst_delta_c = expected_sst_c - sst_data.water_temp_c

                                # Phase 2: Publish SST to WorldState
                                try:
                                    from core.models import EnvFeature
                                    from core.world import get_world
                                    sst_event = EnvFeature(
                                        obs_time_utc=sst_data.observation_time,
                                        source="NDBC",
                                        feature_type="SST",
                                        location_ref=sst_data.buoy_id,
                                        value=sst_data.water_temp_c,
                                        unit="C",
                                    )
                                    get_world().publish(sst_event)
                                except Exception as e_w:
                                    pass
                                
                            # Fetch AOD from CAMS
                            from collector.cams_fetcher import fetch_aerosol_data
                            
                            station_config = STATIONS[station_id]
                            aod_data = await fetch_aerosol_data(
                                station_config.latitude,
                                station_config.longitude,
                                station_id
                            )
                            if aod_data:
                                aod_value = aod_data.aod_550nm
                                aod_display = f"{aod_data.aod_550nm:.2f} ({aod_data.quality})"

                                # Phase 2: Publish AOD to WorldState
                                try:
                                    from core.models import EnvFeature
                                    from core.world import get_world
                                    aod_event = EnvFeature(
                                        obs_time_utc=aod_data.observation_time,
                                        source="CAMS",
                                        feature_type="AOD",
                                        location_ref=f"{aod_data.latitude},{aod_data.longitude}",
                                        value=aod_data.aod_550nm,
                                        unit="dimensionless",
                                    )
                                    get_world().publish(aod_event)
                                except Exception as e_w:
                                    pass

                        except Exception as e:
                            # Silent fail - alpha variables are optional
                            pass

                    # ============================================================
                    # FLUID DYNAMICS: Thermal Advection (Phase 1)
                    # ============================================================
                    advection_adj_c = None
                    advection_display = None
                    
                    if day_offset == 0:
                        try:
                             from collector.advection_fetcher import fetch_upstream_weather
                             from synthesizer.advection import calculate_advection
                             
                             if metar and metar.wind_dir_degrees is not None:
                                  station_conf = STATIONS[station_id]
                                  # Fetch weather 50km upstream to detect incoming air mass
                                  up_data = await fetch_upstream_weather(
                                       station_conf.latitude, 
                                       station_conf.longitude,
                                       metar.wind_dir_degrees,
                                       distance_km=50.0
                                  )
                                  
                                  if up_data:
                                       # Convert knots to km/h
                                       local_wind_kmh = (metar.wind_speed_kt or 0) * 1.852
                                       adv_rep = calculate_advection(adjusted_actual_c, up_data, local_wind_kmh)
                                       # Save display string
                                       advection_display = adv_rep.description

                                       # Empirical: Apply 3-hour integration of current rate
                                       if abs(adv_rep.rate_c_per_hour) > 0.05:
                                            advection_adj_c = adv_rep.rate_c_per_hour * 3.0
                                            # Safety cap (+/- 3C = +/- 5.4F)
                                            advection_adj_c = max(-3.0, min(3.0, advection_adj_c))

                                       # Phase 2: Publish upstream as AuxObs
                                       try:
                                            from core.models import AuxObs
                                            from core.world import get_world
                                            from datetime import timezone as _tz
                                            aux_event = AuxObs(
                                                obs_time_utc=datetime.now(_tz.utc),
                                                source="UPSTREAM_OPEN_METEO",
                                                station_id=f"UPSTREAM_{station_id}",
                                                temp_c=up_data.temperature_c,
                                                temp_f=round((up_data.temperature_c * 9/5) + 32, 1),
                                                drift=round(up_data.temperature_c - adjusted_actual_c, 2) if adjusted_actual_c else 0.0,
                                            )
                                            get_world().publish(aux_event)
                                       except Exception as e_w:
                                            pass

                        except Exception as e:
                             print(f"[WARN] Advection failed: {e}")

                    # ============================================================
                    # Phase 2: PWS CLUSTER CONSENSUS
                    # ============================================================
                    if day_offset == 0:
                        try:
                            from collector.pws_fetcher import fetch_and_publish_pws
                            official_temp = metar.temp_c if metar else None
                            await fetch_and_publish_pws(station_id, official_temp_c=official_temp)
                        except Exception as e:
                            print(f"[WARN] PWS cluster failed: {e}")

                    # ============================================================
                    # REALITY CHECK: Get verified floor BEFORE prediction
                    # ============================================================
                    verified_floor_f = None
                    obs_verified = None
                    if day_offset == 0:
                        try:
                            # Use the function imported at the top from collector.wunderground_fetcher
                            obs_verified = await get_observed_max(station_id)
                            if obs_verified and obs_verified.high_temp_f is not None:
                                verified_floor_f = obs_verified.high_temp_f
                        except Exception as e:
                            print(f"[WARN] Failed to fetch verified floor: {e}")

                    prediction = calculate_physics_prediction(
                        station_id=station_id,
                        temp_actual_c=adjusted_actual_c,
                        temp_hrrr_hourly_c=temp_hrrr_hourly_c,
                        hrrr_max_c=hrrr_max_c_corrected,
                        soil_moisture=hrrr.soil_moisture,
                        radiation=hrrr.shortwave_radiation,
                        sky_condition=metar.sky_condition,
                        wind_dir=metar.wind_dir_degrees,
                        velocity_ratio=velocity_ratio,
                        aod_550nm=aod_value,
                        sst_delta_c=sst_delta_c,
                        current_sst_c=current_sst_c_val,
                        advection_adj_c=advection_adj_c,
                        local_hour=local_hour,
                        verified_floor_f=verified_floor_f,
                        peak_hour=peak_hour,  # v13.0: Pass peak hour for Post-Peak Cap
                    )
                
                # ================================================================
                # EXTRA TAGS: Add verification status to physics reason if needed
                # ================================================================
                floor_explanation = ""
                if day_offset == 0 and obs_verified:
                    if "Floor" in prediction.physics_reason:
                        if obs_verified.verification_status == "suspicious":
                            # If we applied a floor but it's suspicious, tag it
                            from synthesizer.physics import PhysicsPrediction
                            new_reason = f"{prediction.physics_reason} [SUSPICIOUS]"
                            prediction = PhysicsPrediction(
                                hrrr_max_raw_f=prediction.hrrr_max_raw_f,
                                current_deviation_f=prediction.current_deviation_f,
                                delta_weight=prediction.delta_weight,
                                physics_adjustment_f=prediction.physics_adjustment_f,
                                physics_reason=new_reason,
                                final_prediction_f=prediction.final_prediction_f,
                            )
                        floor_explanation = f"Floor: {prediction.final_prediction_f:.1f}F ({obs_verified.verification_status})"
                
                # Log all predictions to database for multi-day tracking (v4.1)
                log_prediction(
                    station_id=station_id,
                    timestamp=metar.observation_time,
                    temp_actual_c=metar.temp_c,
                    temp_hrrr_hourly_c=temp_hrrr_hourly_c,
                    soil_moisture=hrrr.soil_moisture,
                    radiation=hrrr.shortwave_radiation,
                    sky_condition=metar.sky_condition,
                    wind_dir=metar.wind_dir_degrees,
                    prediction=prediction,
                    target_date=prediction_date.isoformat()
                )
                
                # Display prediction with detailed breakdown
                day_label = ["HOY", "MAÑANA", "PASADO"][min(day_offset, 2)]
                print(f"\n  📅 {day_label} ({prediction_date.strftime('%Y-%m-%d')})")
                
                # Fetch Polymarket data for this specific day
                polymarket_options = []
                try:
                    from market.discovery import fetch_event_for_station_date
                    event_data = await fetch_event_for_station_date(station_id, prediction_date)
                    if not event_data:
                        from market.polymarket_checker import build_event_slug, fetch_event_data
                        event_slug = build_event_slug(station_id, prediction_date)
                        event_data = await fetch_event_data(event_slug)
                    
                    if event_data:
                        # Get all markets with probabilities
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
                        
                        # Sort by probability descending
                        polymarket_options.sort(key=lambda x: x[1], reverse=True)
                        
                        # ============================================================
                        # v3.0: Track Market Velocity (Crowd Wisdom)
                        # ============================================================
                        if day_offset == 0:
                            track_probability_velocity(station_id, prediction_date.isoformat(), polymarket_options)
                except Exception as e:
                    pass
                
                # Display Polymarket data
                if polymarket_options:
                    print(f"     💹 Polymarket:")
                    for outcome, prob in polymarket_options:
                        bar_length = int(prob * 20)  # Max 20 chars
                        bar = "█" * bar_length + "░" * (20 - bar_length)
                        print(f"        {outcome:15s} {bar} {prob*100:5.1f}%")
                    
                    # Show Sentiment (v3.0)
                    if day_offset == 0:
                        sentiment = get_crowd_sentiment(station_id, prediction_date.isoformat())
                        if "SHIFT" in sentiment or "MOVE" in sentiment:
                            print(f"     🔥 WISDOM: {sentiment}!")
                else:
                    print(f"     💹 Polymarket: Mercado aún no creado")
                
                print(f"     ┌─ Componentes del Cálculo")
                print(f"     │  Base HRRR:        {prediction.hrrr_max_raw_f}°F")
                
                # v10.0: Show Ensemble data if available
                if ensemble_result and (ensemble_result.nbm_value_f or ensemble_result.lamp_value_f):
                    if ensemble_result.nbm_value_f:
                        print(f"     │  🌐 NBM Value:      {ensemble_result.nbm_value_f:.1f}°F")
                    if ensemble_result.lamp_value_f:
                        conf_pct = int(ensemble_result.lamp_confidence * 100)
                        print(f"     │  ✈️  LAMP Value:     {ensemble_result.lamp_value_f:.1f}°F (conf: {conf_pct}%)")
                    print(f"     │  ⚖️  Ensemble Base:  {ensemble_result.base_weighted_f:.1f}°F")
                    print(f"     │  🛡️  Safety Range:   {ensemble_result.helios_floor_f:.1f} - {ensemble_result.helios_ceiling_f:.1f}°F")
                    if ensemble_result.hrrr_outlier_detected:
                        print(f"     │  ⚠️  HRRR OUTLIER:   Ignorando HRRR (desviación > 3°F del NBM)")
                    if ensemble_result.confidence_level == "LOW":
                        print(f"     │  🔴 CONFIANZA:      BAJA - Spread > 4°F - Solo Split Bets")
                
                print(f"     │  Desviación Delta: {prediction.current_deviation_f:+.1f}°F")
                
                # HELIOS v2.2 Weighting
                weight_pct = int(prediction.delta_weight * 100)
                weighted_impact = prediction.current_deviation_f * prediction.delta_weight
                time_str = local_now.strftime("%H:%M")
                print(f"     │  Confianza Delta:  {weight_pct}% (Hora: {time_str}) -> Impacto real: {weighted_impact:+.1f}°F")
                
                # Show delta explanation if available
                if delta_explanation and day_offset <= 1:
                    print(f"     │    └ {delta_explanation}")
                
                # Show velocity if calculated
                if velocity_ratio and velocity_explanation:
                    print(f"     │  Velocidad:        {velocity_ratio:.2f}x ({velocity_explanation})")
                
                # Parse physics adjustments
                # ============================================================
                # RISK ANALYSIS (HELIOS v2.1)
                # ============================================================
                from synthesizer.risk_analyzer import analyze_night_risk, analyze_rounding_risk
                
                # Rounding Risk
                r_risk, r_dist, r_msg = analyze_rounding_risk(prediction.final_prediction_f)
                
                # Night Mode Risk (Today only)
                n_active, n_risk, n_max = False, "N/A", 0.0
                if day_offset == 0:
                    n_active, n_risk, n_max = analyze_night_risk(
                        current_temp_f=adjusted_actual_f, # Use estimated current temp
                        hourly_temps_c=forecast.hourly_temps_c,
                        hourly_times=forecast.hourly_times,
                        current_local_hour=local_hour
                    )

                physics_breakdown = prediction.physics_reason.split(", ")
                soil_adj = 0.0
                cloud_adj = 0.0
                wind_adj = 0.0
                smoke_adj = 0.0
                sst_adj_val = 0.0
                advection_val = 0.0
                
                for item in physics_breakdown:
                    val = 0.0
                    if any(char.isdigit() for char in item):
                        # Extract number before 'F' or end
                        try:
                            # Parse "Smoke-2F" -> -2.0
                            import re
                            match = re.search(r'([+-]?\d?\.?\d+)F', item)
                            if match:
                                val = float(match.group(1))
                        except:
                            pass

                    if "Soil" in item:
                        soil_adj = val
                    elif "Cloud" in item:
                        cloud_adj = val
                    elif "Breeze" in item:
                        wind_adj = val
                    elif "Smoke" in item or "Haze" in item:
                        smoke_adj = val
                    elif "SST" in item:
                        sst_adj_val = val
                    elif "Advection" in item:
                        advection_val = val
                
                # Print Alpha Variables Data
                if day_offset == 0:
                    if advection_display:
                        print(f"     │  🌊 Advección:      {advection_display}")
                    if aod_display:
                        print(f"     │  CAMS AOD:         {aod_display}")
                    if sst_display:
                        print(f"     │  NDBC SST:         {sst_display}")

                print(f"     │  Humedad Suelo:    {soil_adj:+.1f}°F")
                print(f"     │  Desajuste Nubes:  {cloud_adj:+.1f}°F")
                print(f"     │  Brisa Marina:     {wind_adj:+.1f}°F")
                print(f"     │  Humo/Aerosoles:   {smoke_adj:+.1f}°F")
                print(f"     │  Anomalía SST:     {sst_adj_val:+.1f}°F")
                print(f"     │  Ajuste Advección: {advection_val:+.1f}°F")
                print(f"     └─ Total Física:    {prediction.physics_adjustment_f:+.1f}°F")
                print(f"     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"     PREDICCIÓN FINAL:   {prediction.final_prediction_f}°F")
                
                # Show floor explanation if Reality Check was applied
                if floor_explanation and "Floor applied" in floor_explanation:
                    print(f"     ⚠️  REALITY CHECK: {floor_explanation}")

                # Display Risk Analysis (New for v2.1)
                print(f"     ──────────────────────────────────")
                if n_active:
                    night_trend = "Ascendente" if n_risk == "HIGH" else "Descendente"
                    print(f"     🌙  MODO NOCTURNO:   ACTIVO")
                    print(f"         └─ Tendencia HRRR: {night_trend} (Max futuro: {n_max:.1f}°F)")
                    print(f"         └─ Riesgo subida:  {'CRÍTICO ⚠️' if n_risk == 'HIGH' else 'BAJO 0% (DEFINITIVO)'}")
                else:
                    print(f"     ☀️  MODO DIURNO:     ACTIVO (Sin restricción nocturna)")

                r_icon = "⚠️" if r_risk == "CRITICAL" else "✅"
                print(f"     🎯  RIESGO REDONDEO: {r_risk} {r_icon}")
                print(f"         └─ Analisis:       {r_msg}")
                
                # Show comparison if Polymarket data available
                
                # Show comparison if Polymarket data available
                if polymarket_options:
                    # Check for betting opportunity
                    from opportunity import check_bet_opportunity, format_opportunity_display
                    
                    opp = check_bet_opportunity(
                        prediction=prediction.final_prediction_f,
                        polymarket_options=polymarket_options,
                        station_id=station_id,
                        target_date=prediction_date
                    )
                    
                    # ============================================================
                    # ALPHA CONFIDENCE: Reduce confidence if divergence > 2.0°F (v3.0)
                    # ============================================================
                    if station_id == "KLGA" and day_offset == 0:
                        try:
                            mkt_avg = 0.0
                            if opp and hasattr(opp, 'market_avg') and opp.market_avg:
                                mkt_avg = opp.market_avg
                            
                            if mkt_avg > 0:
                                divergence = abs(prediction.final_prediction_f - mkt_avg)
                                if divergence > 2.0:
                                    print(f"     ⚠️  DIVERGENCIA ALTA: {divergence:.1f}F vs Mercado. Bajando confianza al 50%.")
                                    # Override prediction delta_weight for registrar/logs
                                    from synthesizer.physics import PhysicsPrediction
                                    prediction = PhysicsPrediction(
                                        hrrr_max_raw_f=prediction.hrrr_max_raw_f,
                                        current_deviation_f=prediction.current_deviation_f,
                                        delta_weight=prediction.delta_weight * 0.5,
                                        physics_adjustment_f=prediction.physics_adjustment_f,
                                        physics_reason=f"{prediction.physics_reason}, DivDiscount({divergence:.1f}F)",
                                        final_prediction_f=prediction.final_prediction_f,
                                        velocity_ratio=prediction.velocity_ratio,
                                        aod_550nm=prediction.aod_550nm,
                                        sst_delta_c=prediction.sst_delta_c,
                                        advection_adj_c=prediction.advection_adj_c,
                                        verified_floor_f=prediction.verified_floor_f
                                    )
                        except Exception as e:
                            print(f"     [WARN] Error aplicando descuento de confianza: {e}")
                    
                    # Display opportunity analysis
                    print(format_opportunity_display(opp))
                
                if day_offset >= 2:
                    print(f"     ℹ️  Día futuro: Solo HRRR + humedad suelo")
                
                prediction_count += 1
                
            except Exception as e:
                print(f"\n  ✗ Día +{day_offset}: Error - {e}")
                continue
        
        print(f"\n  ✓ {prediction_count} predicciones generadas\n")
        return True
        
    except Exception as e:
        print(f"\n  ✗ {station_id}: {e}")
        traceback.print_exc()
        return False
        
        return True
        
    except Exception as e:
        print(f"\n[{station_id}] [ERR] {e}")
        traceback.print_exc()
        return False


async def collection_cycle(target_stations: List[str] = None, target_days: List[int] = None):
    """Run data collection for all stations."""
    # Common timezone for current stations (NYC/Atlanta)
    us_tz = ZoneInfo("America/New_York")
    us_now = datetime.now(timezone.utc).astimezone(us_tz)
    
    print(f"\n{'─'*70}")
    print(f"  Ciclo de Recolección")
    print(f"  Hora del Sistema: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mercado US/Este:  {us_now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*70}\n")
    
    success_count = 0
    
    stations_to_process = target_stations if target_stations else get_active_stations().keys()

    for station_id in stations_to_process:
        try:
            # Validate station_id exists
            if station_id not in STATIONS:
                print(f"  ⚠ Station {station_id} not found in config.")
                continue
                
            if await collect_and_predict(station_id, target_days):
                success_count += 1
        except Exception as e:
            print(f"  ✗ {station_id}: {e}")
            continue
    
    print(f"\n{'─'*70}")
    print(f"  ✓ Completado: {success_count}/{len(stations_to_process)} estaciones")
    print(f"{'─'*70}\n")



# =============================================================================
# CLI & Scheduler
# =============================================================================

def run_collection(target_stations=None, target_days=None):
    """Wrapper to run async collection in sync context."""
    asyncio.run(collection_cycle(target_stations, target_days))


def run_trajectory_capture(target_stations=None):
    """Wrapper to run async trajectory capture in sync context."""
    asyncio.run(capture_trajectories(target_stations))


def generate_report() -> str:
    """Generate accuracy report."""
    report_data = get_accuracy_report(7)
    
    lines = [
        "",
        "=" * 65,
        "PHYSICS ENGINE ACCURACY REPORT (Last 7 days)",
        "=" * 65,
        "",
    ]
    
    if not report_data:
        lines.append("No verified predictions available yet.")
    else:
        for station_id, stats in report_data.items():
            station = STATIONS.get(station_id)
            station_name = station.name if station else station_id
            
            lines.append(f"[{station_id}] {station_name}")
            lines.append(f"   Predictions: {stats['total_predictions']}")
            
            if stats.get('avg_physics_error_f'):
                lines.append(f"   Physics Error: +/-{stats['avg_physics_error_f']:.1f}F")
            if stats.get('avg_hrrr_error_f'):
                lines.append(f"   Raw HRRR Error: +/-{stats['avg_hrrr_error_f']:.1f}F")
            
            # Calculate improvement
            if stats.get('avg_physics_error_f') and stats.get('avg_hrrr_error_f'):
                improvement = stats['avg_hrrr_error_f'] - stats['avg_physics_error_f']
                if improvement > 0:
                    lines.append(f"   Improvement: +{improvement:.1f}F [ADDING VALUE]")
                else:
                    lines.append(f"   Improvement: {improvement:.1f}F [NEEDS TUNING]")
            
            lines.append("")
    
    lines.append("=" * 65)
    lines.append("")
    
    return "\n".join(lines)


def main():
    """Main entry point with scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HELIOS Weather Prediction Engine")
    parser.add_argument("-s", "--station", type=str, help="Station ID to target (e.g. KLGA)")
    parser.add_argument("-d", "--day", type=str, choices=["today", "tomorrow", "both"], default="both", help="Target day(s)")
    args = parser.parse_args()
    
    target_stations = [args.station] if args.station else None
    
    target_days = [0, 1]
    if args.day == "today":
        target_days = [0]
    elif args.day == "tomorrow":
        target_days = [1]
        
    print(f"\n{'═'*70}")
    print(f"  HELIOS Weather Lab - Motor de Física")
    print(f"  Sistema de Predicción Determinística de Temperatura")
    print(f"  Modo: 100% Sin conexión (Sin IA/LLM)")
    if target_stations:
        print(f"  Estaciones: {', '.join(target_stations)} (Filtrado)")
    else:
        print(f"  Estaciones: KLGA (NYC), KATL (Atlanta)")
    print(f"  Días: {args.day}")
    print(f"  Modelo: HRRR + Reglas Físicas")
    print(f"{'═'*70}\n")
    
    # Initialize database
    init_database()
    print("✓ Base de datos inicializada: helios_weather.db\n")
    
    # Capture initial trajectory
    print("Capturando trayectoria inicial...")
    run_trajectory_capture(target_stations)
    
    # Run initial collection
    print("Ejecutando recolección inicial de datos...")
    run_collection(target_stations, target_days)
    
    # Schedule trajectory capture at 07:00 AM
    schedule.every().day.at("07:00").do(run_trajectory_capture, target_stations)
    
    # Schedule regular collection
    schedule.every(COLLECTION_INTERVAL_MINUTES).minutes.do(run_collection, target_stations, target_days)
    
    print(f"\n{'─'*70}")
    print(f"  ✓ Captura de trayectoria programada a las 07:00 AM")
    print(f"  ✓ Recolección programada cada {COLLECTION_INTERVAL_MINUTES} minutos")
    print(f"{'─'*70}")
    print(f"\n  Motor de física en ejecución. Presiona Ctrl+C para detener.\n")
    
    # Main loop
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print(f"\n\n{'─'*70}")
        print("  Deteniendo sistema...")
        print(f"{'─'*70}\n")
        try:
            print(generate_report())
        except Exception as e:
            print(f"  No se pudo generar reporte: {e}")
        print("  Hasta luego!\n")
    except Exception as e:
        print(f"\n[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
