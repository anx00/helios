"""
Helios Weather Lab - Physics Engine (Deterministic)
100% offline prediction using physics-based rules.
No AI/LLM dependencies.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from zoneinfo import ZoneInfo

from config import STATIONS
from synthesizer.bias_correction import get_7day_bias, apply_bias_correction
from synthesizer.cold_memory import apply_cold_memory


@dataclass
class PhysicsPrediction:
    """Result from the deterministic physics engine."""
    hrrr_max_raw_f: float          # Base HRRR forecast
    current_deviation_f: float      # Delta (clamped, raw)
    delta_weight: float             # Time-based confidence weight (0.0 to 1.0)
    physics_adjustment_f: float     # Sum of penalties
    physics_reason: str             # Breakdown of rules
    final_prediction_f: float       # Formula result
    # v5.3: Component traceability
    velocity_ratio: Optional[float] = None
    aod_550nm: Optional[float] = None
    sst_delta_c: Optional[float] = None
    advection_adj_c: Optional[float] = None
    verified_floor_f: Optional[float] = None


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return round((celsius * 9/5) + 32, 1)


def get_temporal_confidence_weight(hour: Optional[int]) -> float:
    """
    HELIOS v3.0 - Smooth Dynamic Delta Decay.
    Uses exponential decay as peak hour approaches to avoid sudden jumps.
    Only applies to KLGA; other stations use legacy linear weights.
    """
    if hour is None:
        return 0.5  # Default if unknown
    
    # Before 9 AM: minimal trust (morning noise)
    if hour < 9:
        return 0.15
    
    # After peak (15:00): no more delta impact
    if hour >= 15:
        return 0.0
    
    # Smooth decay from 9 AM to 3 PM (peak)
    # Formula: weight decays faster as we approach peak
    hours_to_peak = 15 - hour  # 6 at 9AM, 0 at 3PM
    decay_factor = (hours_to_peak / 6) ** 1.5  # Non-linear decay
    return max(0.0, min(1.0, 0.6 * decay_factor))


def calculate_physics_prediction(
    station_id: str,
    temp_actual_c: float,
    temp_hrrr_hourly_c: float,
    hrrr_max_c: float,
    soil_moisture: Optional[float],
    radiation: Optional[float],
    sky_condition: str,
    wind_dir: Optional[int],
    velocity_ratio: Optional[float] = None,
    aod_550nm: Optional[float] = None,      # NEW: Aerosol optical depth
    sst_delta_c: Optional[float] = None,    # NEW: SST anomaly
    local_hour: Optional[int] = None,       # NEW: Local hour for temporal decay
    current_sst_c: Optional[float] = None,  # NEW: Absolute SST for thermodynamic check
    advection_adj_c: Optional[float] = None, # NEW: Explicit Advection adjustment
    verified_floor_f: Optional[float] = None, # NEW: Verified observation floor
    target_date_str: Optional[str] = None,    # v12.0: For Cold Memory persistence
    peak_hour: Optional[int] = None,          # v13.0: Expected peak hour from HRRR
) -> PhysicsPrediction:
    """
    Calculate prediction using deterministic physics rules.
    """
    # Convert to Fahrenheit for calculations
    temp_actual_f = celsius_to_fahrenheit(temp_actual_c)
    temp_hrrr_hourly_f = celsius_to_fahrenheit(temp_hrrr_hourly_c)
    hrrr_max_f = celsius_to_fahrenheit(hrrr_max_c)
    
    # =========================================================================
    # STEP A: Calculate Deviation (Delta)
    # =========================================================================
    delta = temp_actual_f - temp_hrrr_hourly_f
    
    # Safety clamp: limit to +/- 5°F to avoid sensor errors
    delta = max(-5.0, min(5.0, delta))
    delta = round(delta, 1)
    
    # =========================================================================
    # v7.0 TASK 1: Morning Delta Dampening
    # If before noon AND delta is negative, apply only 25% of the impact.
    # Rationale: A cold morning doesn't mean a cold afternoon if there's solar radiation.
    # =========================================================================
    morning_dampen_applied = False
    if local_hour is not None and local_hour < 12 and delta < 0:
        delta = delta * 0.25
        morning_dampen_applied = True
    
    # =========================================================================
    # STEP B: Calculate Physics Adjustment
    # =========================================================================
    adjustment = 0.0
    reasons = []
    if morning_dampen_applied:
        reasons.append("MorningDamp(25%)")
    
    # Calculate Time Decay for Physics (Post-Peak)
    # Physics rules (Sea Breeze, Soil Moisture) affect how high the temp gets.
    # Once the peak time has passed (e.g. 5 PM), these factors shouldn't 
    # lower the max prediction further.
    physics_weight = 1.0
    if local_hour is not None:
        if local_hour >= 17:
            physics_weight = 0.0 # Physics off after 5 PM
        elif local_hour >= 15:
            # Decay from 1.0 at 15:00 to 0.0 at 17:00
            physics_weight = max(0.0, 1.0 - (local_hour - 14) * 0.5)
    
    # Rule 0: Thermal Advection (Fluid Dynamics)
    has_explicit_advection = False
    cold_memory_note = ""
    if advection_adj_c is not None:
        # v3.0 Asymmetric Advection for KLGA/KATL/EGLC
        # Cold advection: 100% weight (dense, reliable)
        # Warm advection: 50% base, 75% if high solar radiation helps mixing
        if station_id in ["KATL", "EGLC", "KLGA"] and advection_adj_c > 0:
            if radiation is not None and radiation > 500:
                advection_adj_c = advection_adj_c * 0.75  # Solar helps mixing
            else:
                advection_adj_c = advection_adj_c * 0.50  # Conservative
        
        if abs(advection_adj_c) > 0.2:
            has_explicit_advection = True
            adv_f = advection_adj_c * 1.8  # C to F delta
            
            # v12.0: Apply Cold Memory persistence
            if target_date_str is not None and local_hour is not None and station_id == "KLGA":
                adv_f, cold_memory_note = apply_cold_memory(
                    station_id=station_id,
                    target_date=target_date_str,
                    current_physics_adj_f=adv_f,
                    current_hour=local_hour
                )
            
            adjustment += adv_f
            if cold_memory_note:
                reasons.append(f"Advection{adv_f:+.1f}F|{cold_memory_note}")
            else:
                reasons.append(f"Advection{adv_f:+.1f}F")
        else:
             reasons.append("Advection+0.0F(Neutral)")
    else:
        reasons.append("Advection+0.0F(NoData)")

    # Rule 1: Soil Moisture Penalty
    # KATL: +20% weight for continental thermal inertia
    # EGLC: +10% for urban moisture retention
    if station_id == "KATL":
        soil_multiplier = 1.2
    elif station_id == "EGLC":
        soil_multiplier = 1.1
    else:
        soil_multiplier = 1.0
    if soil_moisture is not None:
        if soil_moisture > 0.35:
            soil_penalty = 2.0 * soil_multiplier
            adjustment -= soil_penalty
            reasons.append(f"Soil-{soil_penalty:.1f}F(Wet)")
        elif soil_moisture > 0.25:
            soil_penalty = 1.0 * soil_multiplier
            adjustment -= soil_penalty
            reasons.append(f"Soil-{soil_penalty:.1f}F(Moist)")
        else:
            reasons.append("Soil+0.0F(Dry)")
    else:
        reasons.append("Soil+0.0F(NoData)")
            
    # Rule 2: Cloud Mismatch Penalty
    if radiation is not None and radiation > 400:
        if sky_condition in ["BKN", "OVC", "VV"]:
            adjustment -= 3.0
            reasons.append("Cloud-3.0F(Mismatch)")
        else:
            reasons.append("Cloud+0.0F(Clear)")
    elif radiation is not None:
         reasons.append("Cloud+0.0F(LowRad)")
    else:
         reasons.append("Cloud+0.0F(NoData)")
            
    # Rule 3: Sea Breeze / Marine Influence
    if not has_explicit_advection and station_id == "KLGA" and wind_dir is not None:
        is_onshore = (wind_dir >= 340 or wind_dir <= 70)
        if is_onshore:
            sea_effect = -2.0
            if current_sst_c is not None:
                if current_sst_c >= hrrr_max_c:
                     sea_effect = 0.0
                elif (hrrr_max_c - current_sst_c) < 3.0:
                     sea_effect = -0.5
            
            if sea_effect < 0:
                adjustment += sea_effect
                reasons.append(f"Breeze{sea_effect:.1f}F")
            else:
                reasons.append("Breeze+0.0F(WarmSea)")
        else:
            reasons.append("Breeze+0.0F(Offshore)")
    else:
        # Not applicable or no wind data
        if station_id == "KLGA":
             reasons.append("Breeze+0.0F(NoWind)")
        else:
             reasons.append("Breeze+0.0F(N/A)")
    
    # Rule 3b: Thames River Effect (EGLC - Docklands riverside)
    # East wind (40-120°) brings cooler air from Thames estuary
    if not has_explicit_advection and station_id == "EGLC" and wind_dir is not None:
        is_estuary_wind = (40 <= wind_dir <= 120)  # East to Southeast
        if is_estuary_wind:
            thames_effect_c = -0.8  # Cooling in Celsius
            thames_effect_f = thames_effect_c * 1.8
            adjustment += thames_effect_f
            reasons.append(f"Thames{thames_effect_f:.1f}F")
        else:
            reasons.append("Thames+0.0F(NoEffect)")
    elif station_id == "EGLC":
        reasons.append("Thames+0.0F(NoWind)")
    
    # Rule 3c: Urban Heat Island (EGLC - Docklands)
    # Daytime urban heat factor +0.3°C due to concrete/buildings
    if station_id == "EGLC" and local_hour is not None:
        if 10 <= local_hour <= 17:  # Daytime hours
            urban_heat_c = 0.3
            urban_heat_f = urban_heat_c * 1.8  # Convert to F
            adjustment += urban_heat_f
            reasons.append(f"UrbanHeat+{urban_heat_f:.1f}F")
        else:
            reasons.append("UrbanHeat+0.0F(Night)")
            
    # Rule 4: Heating Velocity (Momentum)
    if velocity_ratio is not None:
        if velocity_ratio > 1.2:
            adjustment += 1.0
            reasons.append("Heating+1.0F(Fast)")
        elif velocity_ratio < 0.8:
            adjustment -= 0.5
            reasons.append("Heating-0.5F(Slow)")
        else:
            reasons.append("Heating+0.0F(Normal)")
    else:
        reasons.append("Heating+0.0F(NoData)")
            
    # Rule 5: Aerosol/Smoke Penalty
    if aod_550nm is not None:
        if aod_550nm > 0.5:
            adjustment -= 2.0
            reasons.append(f"Smoke-2.0F(AOD={aod_550nm:.2f})")
        elif aod_550nm > 0.3:
            adjustment -= 1.0
            reasons.append(f"Haze-1.0F(AOD={aod_550nm:.2f})")
        else:
            reasons.append(f"Haze+0.0F(AOD={aod_550nm:.2f})")
    else:
        reasons.append("Haze+0.0F(NoData)")
            
    # Rule 6: SST Anomaly
    # v7.0 TASK 4: Momentum Priority - If heating velocity is positive, ignore negative SST adjustments
    sst_momentum_override = (velocity_ratio is not None and velocity_ratio > 1.0)
    
    if sst_delta_c is not None and station_id == "KLGA":
        if sst_delta_c > 2.0:
            sst_adj = -sst_delta_c * 0.5 * 1.8
            sst_adj = max(-3.0, sst_adj)
            # v7.0: If momentum is positive, don't apply negative SST cooling
            if sst_momentum_override and sst_adj < 0:
                reasons.append(f"SST+0.0F(MomentumOverride)")
            else:
                adjustment += sst_adj
                reasons.append(f"SST{sst_adj:.1f}F(Cold)")
        elif sst_delta_c < -2.0:
            sst_adj = -sst_delta_c * 0.3 * 1.8
            sst_adj = min(2.0, sst_adj)
            adjustment += sst_adj
            reasons.append(f"SST+{sst_adj:.1f}F(Warm)")
        else:
            reasons.append("SST+0.0F(Neutral)")
    else:
        if station_id == "KLGA":
             reasons.append("SST+0.0F(NoData)")
        else:
             reasons.append("SST+0.0F(N/A)")

    # Apply Post-Peak Decay
    if physics_weight < 1.0:
        old_adj = adjustment
        adjustment *= physics_weight
        if abs(old_adj - adjustment) > 0.1:
            reasons.append(f"Decay({int(physics_weight*100)}%)")
            
    adjustment = round(adjustment, 1)
    
    # =========================================================================
    # v7.0 TASK 3: Physical Intersection Filter
    # If physics adjustment and delta have the same sign, use only the more aggressive one.
    # Rationale: Avoid double-penalizing for the same phenomenon (e.g., cold morning + cold wind).
    # =========================================================================
    delta_weight = get_temporal_confidence_weight(local_hour)
    weighted_delta = delta * delta_weight
    
    if (weighted_delta < 0 and adjustment < 0) or (weighted_delta > 0 and adjustment > 0):
        # Same sign: use the more aggressive (larger absolute value)
        if abs(weighted_delta) > abs(adjustment):
            final_adjustment = weighted_delta
            reasons.append(f"Intersect(DeltaWins:{weighted_delta:+.1f}F)")
        else:
            final_adjustment = adjustment
            reasons.append(f"Intersect(PhysicsWins:{adjustment:+.1f}F)")
        final_prediction_f = hrrr_max_f + final_adjustment
    else:
        # Different signs or zero: sum them normally (HELIOS v2.2 legacy)
        final_prediction_f = hrrr_max_f + weighted_delta + adjustment
    
    # Rule 7: Reality Check (Floor)
    # The predicted daily maximum cannot be lower than the temperature already reached.
    # ONLY apply if a verified floor (METAR/WU) is provided.
    if verified_floor_f is not None:
        if final_prediction_f < verified_floor_f:
            final_prediction_f = verified_floor_f
            reasons.append(f"Floor({verified_floor_f:.1f}F)")
    
    # =========================================================================
    # v7.0 TASK 2: Solar Minimum Rise (Suelo de Predicción Solar)
    # Calculation: Min_Prediction = METAR_Actual + (Radiation_W/m2 / 100)
    # Rationale: The sun will always raise the temperature. If there's radiation, enforce a minimum.
    # =========================================================================
    if radiation is not None and radiation > 100 and local_hour is not None and local_hour < 15:
        solar_floor_f = temp_actual_f + (radiation / 100.0)
        if final_prediction_f < solar_floor_f:
            reasons.append(f"SolarFloor({solar_floor_f:.1f}F)")
            final_prediction_f = solar_floor_f

    # =========================================================================
    # v13.2 Rule 7b: Post-Peak Cap (FIXED TRIGGER)
    # Only apply cap 1+ hour AFTER the expected peak hour, not at the peak itself.
    # Margin decreases from 2.0°F at peak+1 to 0.5°F by sunset (17:00).
    # Rationale: Need time for actual peak to be reached before capping upside.
    # =========================================================================
    effective_peak_hour = peak_hour if peak_hour is not None else 15  # Default 15:00 if unknown

    if local_hour is not None and verified_floor_f is not None:
        if local_hour >= effective_peak_hour + 1:
            # Calculate hours past peak (0 at peak, 2 at sunset for peak=15)
            hours_past_peak = local_hour - effective_peak_hour
            sunset_hour = 17  # Hours until hard limit
            hours_to_sunset = max(0, sunset_hour - local_hour)

            # Margin decreases linearly from 2.0°F at peak to 0.5°F at sunset
            # Formula: margin = 0.5 + 1.5 * (hours_to_sunset / (sunset_hour - peak_hour))
            peak_to_sunset_span = max(1, sunset_hour - effective_peak_hour)  # Avoid div by 0
            if hours_to_sunset > 0:
                margin_f = 0.5 + 1.5 * (hours_to_sunset / peak_to_sunset_span)
            else:
                margin_f = 0.5  # At or past sunset

            post_peak_cap = verified_floor_f + margin_f

            if final_prediction_f > post_peak_cap:
                reasons.append(f"PostPeakCap({post_peak_cap:.1f}F,margin={margin_f:.1f})")
                final_prediction_f = post_peak_cap

    # Rule 8: Sunset Hard Limit (KLGA/KATL continental safety)
    # After sunset (~17:00), prediction cannot exceed observed max by more than 0.5°F
    # Prevents "hallucinations" seen in NYC on Jan 8-9
    if station_id in ["KLGA", "KATL"] and local_hour is not None and local_hour >= 17:
        if verified_floor_f is not None:
            sunset_cap = verified_floor_f + 0.5
            if final_prediction_f > sunset_cap:
                final_prediction_f = sunset_cap
                reasons.append(f"SunsetCap({sunset_cap:.1f}F)")
    
    # Rule 8b: Sunset Hard Limit (EGLC - Celsius, stricter +0.3°C cap)
    # London sunsets can be early in winter; stricter cap for urban verification
    if station_id == "EGLC" and local_hour is not None and local_hour >= 16:
        if verified_floor_f is not None:
            # Convert to Celsius for UK cap, then back to F for comparison
            verified_floor_c = (verified_floor_f - 32) * 5 / 9
            sunset_cap_c = verified_floor_c + 0.3
            sunset_cap_f = (sunset_cap_c * 9 / 5) + 32
            if final_prediction_f > sunset_cap_f:
                final_prediction_f = sunset_cap_f
                reasons.append(f"SunsetCap({sunset_cap_c:.1f}C)")

    # FUTURE: 7-day rolling bias correction (omitted for v1 KATL deployment)

    final_prediction_f = round(final_prediction_f, 1)
    
    # Build reason string
    if reasons:
        reason_str = ", ".join(reasons)
    else:
        reason_str = "NoAdjustment"
    
    return PhysicsPrediction(
        hrrr_max_raw_f=hrrr_max_f,
        current_deviation_f=delta,
        delta_weight=delta_weight,
        physics_adjustment_f=adjustment,
        physics_reason=reason_str,
        final_prediction_f=final_prediction_f,
        # v5.3: Traceability
        velocity_ratio=velocity_ratio,
        aod_550nm=aod_550nm,
        sst_delta_c=sst_delta_c,
        advection_adj_c=advection_adj_c,
        verified_floor_f=verified_floor_f
    )


def format_physics_log(
    station_id: str,
    timestamp: datetime,
    prediction: PhysicsPrediction,
) -> str:
    """
    Format a one-line log entry for the physics prediction.
    
    Format: [KLGA] 14:30 | Base HRRR: 60F | Delta: +2F | Physics: -3F (Clouds) | FINAL: 59F
    """
    # Get local time
    station = STATIONS.get(station_id)
    tz_name = station.timezone if station else "UTC"
    local_time = timestamp.astimezone(ZoneInfo(tz_name))
    date_str = local_time.strftime("%Y-%m-%d")
    time_str = local_time.strftime("%H:%M")
    
    # Format delta with sign
    raw_delta = prediction.current_deviation_f
    weight = prediction.delta_weight
    weighted_impact = raw_delta * weight
    delta_str = f"{raw_delta:+.1f}F (w={weight:.2f} -> {weighted_impact:+.1f}F)"
    
    # Format physics adjustment
    if prediction.physics_adjustment_f != 0:
        physics_str = f"{prediction.physics_adjustment_f:+.1f}F ({prediction.physics_reason})"
    else:
        physics_str = "0F"
    
    return (
        f"[{station_id}] {date_str} {time_str} | "
        f"Base HRRR: {prediction.hrrr_max_raw_f:.0f}F | "
        f"Delta: {delta_str} | "
        f"Physics: {physics_str} | "
        f"FINAL: {prediction.final_prediction_f:.1f}F"
    )
