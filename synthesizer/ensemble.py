"""
Helios Weather Lab - Ensemble Engine
Multi-model consensus prediction using HRRR, NBM, and LAMP.
v10.0: Triple Validation with Safety Range.
v11.0: Physical Ceiling, Range Squeeze, WU Alignment, Bet Zones.
v11.1: Sniper Mode - Hard Floor, WU Anchor, 3-tier zone classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timezone


@dataclass
class EnsembleResult:
    """Result from the multi-model ensemble calculation."""
    # Core weighted prediction
    base_weighted_f: float          # Weighted average of all models
    
    # Safety range (99% confidence)
    helios_floor_f: float           # Conservative lower bound
    helios_ceiling_f: float         # Upper bound
    ensemble_spread_f: float        # Ceiling - Floor
    
    # Confidence assessment
    confidence_level: str           # "HIGH", "MEDIUM", "LOW"
    
    # Model diagnostics
    hrrr_outlier_detected: bool     # True if HRRR > 3°F from NBM
    weights_used: Dict[str, float] = field(default_factory=dict)
    
    # Component values (for logging)
    hrrr_value_f: Optional[float] = None
    nbm_value_f: Optional[float] = None
    lamp_value_f: Optional[float] = None
    lamp_confidence: float = 0.5
    
    # Strategy recommendation
    strategy_note: str = ""
    
    # v11.0: Physical Ceiling and Bet Zones
    physical_ceiling_f: Optional[float] = None  # Max physically possible temp
    squeeze_applied: bool = False               # True if range was squeezed
    allowed_spread_f: Optional[float] = None    # Time-based allowed spread
    safe_brackets: List[str] = field(default_factory=list)      # Legacy (deprecated)
    forbidden_brackets: List[str] = field(default_factory=list) # Legacy (deprecated)
    
    # v11.1: 3-tier zone classification (Sniper Mode)
    core_brackets: List[str] = field(default_factory=list)      # 90% prob - NBM/LAMP aligned
    coverage_brackets: List[str] = field(default_factory=list)  # Low prob - within range but far from center
    kill_brackets: List[str] = field(default_factory=list)      # 0% prob - outside physical limits
    
    # v12.0: Sunset Lock, Ultimate Squeeze, Ghost Detection
    sunset_lock_active: bool = False         # True if ceiling locked after 14:00 due to temp drop
    ultimate_squeeze_active: bool = False    # True if 14:30+ squeeze to 1.2°F applied
    ghost_prediction_warning: bool = False   # True if HELIOS > METAR + 3.0°F after 14:00
    ghost_reason: str = ""                   # Warning message for ghost prediction
    observed_max_used_f: Optional[float] = None  # Cumulative max used for Sunset Lock


def calculate_weighted_base(
    hrrr_max_f: float,
    nbm_max_f: Optional[float] = None,
    lamp_max_f: Optional[float] = None,
    lamp_confidence: float = 0.5,
    metar_actual_f: Optional[float] = None
) -> EnsembleResult:
    """
    Calculate weighted base prediction using multi-model consensus.
    
    Formula (standard):
        Base = (HRRR × 0.4) + (NBM × 0.4) + (LAMP × 0.2)
    
    Dynamic adjustments:
        - If lamp_confidence > 0.8: LAMP weight → 0.3, HRRR → 0.3
        - If HRRR deviates > 3°F from NBM: Ignore HRRR, use NBM/LAMP only
    
    Safety Range:
        - Floor = min(NBM, LAMP, HRRR - 1.5°F, METAR_Actual)
        - Ceiling = max(NBM, LAMP, HRRR + 1.0°F)
        - If spread > 4°F: confidence = "LOW"
    
    Args:
        hrrr_max_f: HRRR model maximum temperature (°F)
        nbm_max_f: NBM maximum temperature (°F) or None
        lamp_max_f: LAMP maximum temperature (°F) or None
        lamp_confidence: LAMP 7-day rolling confidence (0.0-1.0)
        metar_actual_f: Current METAR observation (°F) for floor calc
        
    Returns:
        EnsembleResult with weighted prediction and safety range
    """
    # Initialize weights
    weights = {"hrrr": 0.4, "nbm": 0.4, "lamp": 0.2}
    hrrr_outlier = False
    
    # =========================================================================
    # STEP 1: Check for HRRR outlier (> 3°F deviation from NBM)
    # =========================================================================
    if nbm_max_f is not None:
        hrrr_nbm_deviation = abs(hrrr_max_f - nbm_max_f)
        
        if hrrr_nbm_deviation > 3.0:
            # HRRR is "hallucinating" - ignore it
            hrrr_outlier = True
            weights["hrrr"] = 0.0
            weights["nbm"] = 0.7
            weights["lamp"] = 0.3 if lamp_max_f else 0.0
            # If no LAMP, all weight goes to NBM
            if lamp_max_f is None:
                weights["nbm"] = 1.0
    
    # =========================================================================
    # STEP 2: Adjust LAMP weight based on confidence (if not in outlier mode)
    # =========================================================================
    if not hrrr_outlier and lamp_confidence > 0.8 and lamp_max_f is not None:
        # High LAMP confidence: boost LAMP, reduce HRRR
        weights["lamp"] = 0.3
        weights["hrrr"] = 0.3
        weights["nbm"] = 0.4
    
    # =========================================================================
    # STEP 3: Handle missing data gracefully
    # =========================================================================
    available_models = []
    if weights["hrrr"] > 0:
        available_models.append(("hrrr", hrrr_max_f, weights["hrrr"]))
    if nbm_max_f is not None and weights["nbm"] > 0:
        available_models.append(("nbm", nbm_max_f, weights["nbm"]))
    if lamp_max_f is not None and weights["lamp"] > 0:
        available_models.append(("lamp", lamp_max_f, weights["lamp"]))
    
    # Normalize weights if some data is missing
    total_weight = sum(w for _, _, w in available_models)
    if total_weight > 0:
        normalized_models = [(n, v, w/total_weight) for n, v, w in available_models]
    else:
        # Fallback: only HRRR available
        normalized_models = [("hrrr", hrrr_max_f, 1.0)]
    
    # =========================================================================
    # STEP 4: Calculate weighted average
    # =========================================================================
    base_weighted_f = sum(v * w for _, v, w in normalized_models)
    base_weighted_f = round(base_weighted_f, 1)
    
    # Update weights dict with actual used values
    final_weights = {name: round(weight, 2) for name, _, weight in normalized_models}
    
    # =========================================================================
    # STEP 5: Calculate Safety Range
    # =========================================================================
    candidates_floor = [hrrr_max_f - 1.5]  # HRRR floor
    candidates_ceiling = [hrrr_max_f + 1.0]  # HRRR ceiling
    
    if nbm_max_f is not None:
        candidates_floor.append(nbm_max_f)
        candidates_ceiling.append(nbm_max_f)
    
    if lamp_max_f is not None:
        candidates_floor.append(lamp_max_f)
        candidates_ceiling.append(lamp_max_f)
    
    if metar_actual_f is not None:
        candidates_floor.append(metar_actual_f)  # Never predict below observed
    
    helios_floor_f = round(min(candidates_floor), 1)
    helios_ceiling_f = round(max(candidates_ceiling), 1)
    ensemble_spread_f = round(helios_ceiling_f - helios_floor_f, 1)
    
    # =========================================================================
    # STEP 6: Determine confidence level
    # =========================================================================
    if ensemble_spread_f > 4.0:
        confidence_level = "LOW"
        strategy_note = "🛡️ SPREAD > 4°F: Solo apuestas de cobertura (Split Bets)"
    elif ensemble_spread_f > 2.5:
        confidence_level = "MEDIUM"
        strategy_note = f"⚠️ Spread moderado. Rango seguro: {helios_floor_f:.0f}-{helios_ceiling_f:.0f}°F"
    else:
        confidence_level = "HIGH"
        strategy_note = f"✅ Consenso sólido. Rango seguro: {helios_floor_f:.0f}-{helios_ceiling_f:.0f}°F"
    
    if hrrr_outlier:
        strategy_note = f"⚠️ HRRR OUTLIER (+{hrrr_nbm_deviation:.1f}°F vs NBM). Priorizando NBM. " + strategy_note
    
    return EnsembleResult(
        base_weighted_f=base_weighted_f,
        helios_floor_f=helios_floor_f,
        helios_ceiling_f=helios_ceiling_f,
        ensemble_spread_f=ensemble_spread_f,
        confidence_level=confidence_level,
        hrrr_outlier_detected=hrrr_outlier,
        weights_used=final_weights,
        hrrr_value_f=hrrr_max_f,
        nbm_value_f=nbm_max_f,
        lamp_value_f=lamp_max_f,
        lamp_confidence=lamp_confidence,
        strategy_note=strategy_note
    )


def should_apply_ensemble_base(
    ensemble_result: EnsembleResult,
    station_id: str
) -> Tuple[bool, float]:
    """
    Determine if physics adjustments should use ensemble_base instead of HRRR.
    
    Returns tuple of (should_use_ensemble, base_to_use_f).
    
    If ensemble is available and HRRR is not an outlier, we still prefer
    the ensemble base for more stable predictions.
    """
    if ensemble_result is None:
        return (False, 0.0)
    
    # Always use ensemble if we have NBM or LAMP data
    if ensemble_result.nbm_value_f is not None or ensemble_result.lamp_value_f is not None:
        return (True, ensemble_result.base_weighted_f)
    
    # No ensemble data, use HRRR
    return (False, ensemble_result.hrrr_value_f or 0.0)


def get_ensemble_display_lines(ensemble: EnsembleResult) -> list[str]:
    """
    Generate display lines for console output.
    
    Returns list of formatted strings for the prediction display.
    """
    lines = []
    
    if ensemble.nbm_value_f is not None:
        lines.append(f"     │  🌐 NBM Value:      {ensemble.nbm_value_f:.1f}°F")
    
    if ensemble.lamp_value_f is not None:
        conf_pct = int(ensemble.lamp_confidence * 100)
        lines.append(f"     │  ✈️  LAMP Value:     {ensemble.lamp_value_f:.1f}°F (conf: {conf_pct}%)")
    
    if ensemble.nbm_value_f or ensemble.lamp_value_f:
        lines.append(f"     │  ⚖️  Ensemble Base:  {ensemble.base_weighted_f:.1f}°F")
        lines.append(f"     │  🛡️  Safety Range:   {ensemble.helios_floor_f:.1f} - {ensemble.helios_ceiling_f:.1f}°F (Spread: {ensemble.ensemble_spread_f:.1f}°F)")
        
        # Confidence indicator
        if ensemble.confidence_level == "HIGH":
            lines.append(f"     │  ✅ Confianza:      ALTA")
        elif ensemble.confidence_level == "MEDIUM":
            lines.append(f"     │  ⚠️  Confianza:      MEDIA")
        else:
            lines.append(f"     │  🔴 Confianza:      BAJA - Solo Split Bets")
        
        if ensemble.hrrr_outlier_detected:
            lines.append(f"     │  ⚠️  HRRR OUTLIER:   Ignorando HRRR (desviación > 3°F)")
    
    return lines


# =============================================================================
# v11.0: Physical Ceiling & Range Squeeze Functions
# =============================================================================

def calculate_physical_ceiling(
    metar_actual_f: float,
    local_hour: int,
    radiation: Optional[float] = None,
    sky_condition: str = "SCT"
) -> float:
    """
    Calculate maximum physically possible temperature.
    
    Formula: Techo = METAR + (Horas_Sol_Restantes * Ajuste_Radiación)
    
    Args:
        metar_actual_f: Current METAR temperature in °F
        local_hour: Current local hour (0-23)
        radiation: Solar radiation in W/m² (from HRRR)
        sky_condition: METAR sky condition (CLR, FEW, SCT, BKN, OVC)
        
    Returns:
        Physical ceiling temperature in °F
    """
    # Hours until sunset (assume ~17:00 for winter NYC)
    SUNSET_HOUR = 17
    hours_remaining = max(0, SUNSET_HOUR - local_hour)
    
    # Radiation adjustment (°F per hour based on conditions)
    sky = sky_condition.upper() if sky_condition else "SCT"
    if "CLR" in sky or "FEW" in sky:
        gain_per_hour = 1.5  # Clear sky
    elif "SCT" in sky:
        gain_per_hour = 1.0  # Scattered clouds
    elif "BKN" in sky:
        gain_per_hour = 0.5  # Broken clouds
    else:  # OVC, rain, etc.
        gain_per_hour = 0.2  # Overcast
    
    # Radiation factor (normalize 0-1 based on typical range)
    rad_factor = min(1.0, radiation / 600) if radiation else 0.5
    
    physical_ceiling = metar_actual_f + (hours_remaining * gain_per_hour * rad_factor)
    return round(physical_ceiling, 1)


def calculate_allowed_spread(local_hour: int, local_minute: int = 0) -> float:
    """
    Returns maximum allowed spread between Floor and Ceiling.
    Converges as day progresses toward max temp time.
    
    v12.0: Ultimate Squeeze after 14:30 - max 1.2°F spread.
    
    Args:
        local_hour: Current local hour (0-23)
        local_minute: Current minute (0-59)
        
    Returns:
        Maximum allowed spread in °F
    """
    # v12.0: Ultimate Squeeze - after 14:30, force max 1.2°F
    if local_hour >= 15 or (local_hour == 14 and local_minute >= 30):
        return 1.2
    
    SPREAD_TABLE = {
        8: 6.0,   # 8 AM - wide range
        10: 4.0,
        12: 3.0,  # Noon
        14: 1.6,  # 2 PM - tight
        16: 0.8,  # 4 PM - very tight (legacy, now overridden by 1.2 above)
    }
    
    # Find closest hour bracket
    for hour, spread in sorted(SPREAD_TABLE.items(), reverse=True):
        if local_hour >= hour:
            return spread
    return 6.0  # Default early morning


def calculate_bet_zones(
    floor_f: float,
    ceiling_f: float,
    physical_ceiling_f: Optional[float] = None
) -> Tuple[List[str], List[str]]:
    """
    Calculate safe and forbidden betting brackets (legacy v11.0).
    
    Args:
        floor_f: Safety range floor in °F
        ceiling_f: Safety range ceiling in °F
        physical_ceiling_f: Physical ceiling limit (if calculated)
        
    Returns:
        Tuple of (safe_brackets, forbidden_brackets)
    """
    safe_brackets = []
    forbidden_brackets = []
    
    # Generate all possible brackets from 30°F to 70°F
    for low in range(30, 70, 2):
        high = low + 1
        bracket = f"{low}-{high}"
        midpoint = low + 0.5
        
        # Check if within safe range
        if floor_f <= midpoint <= ceiling_f:
            safe_brackets.append(bracket)
        elif physical_ceiling_f and midpoint > physical_ceiling_f:
            forbidden_brackets.append(bracket)
    
    return safe_brackets, forbidden_brackets


def calculate_bet_zones_v12(
    floor_f: float,
    ceiling_f: float,
    base_weighted_f: float,
    metar_actual_f: Optional[float] = None,
    allowed_spread: float = 4.0
) -> Tuple[List[str], List[str], List[str]]:
    """
    v11.2: Sniper Mode Betting Zones (Aggressive).
    
    CORE ZONE (Yellow): Top 2 brackets closest to weighted average.
    COVERAGE ZONE (Blue): Inside valid range but not Core.
    KILL ZONE (Red): > 2.5°F deviation or below METAR/Floor.
    
    Args:
        floor_f: Safe floor
        ceiling_f: Safe ceiling
        base_weighted_f: Ensemble weighted average
        metar_actual_f: Current METAR
        allowed_spread: Current allowed spread limit
        
    Returns:
        (core, coverage, kill)
    """
    core_brackets = []
    coverage_brackets = []
    kill_brackets = []
    
    # Generate brackets 30-70
    brackets = []
    for low in range(30, 70, 2):
        high = low + 1
        mid = low + 0.5
        brackets.append({
            "mid": mid,
            "name": f"{low}-{high}",
            "dist": abs(mid - base_weighted_f)
        })
    
    # Identify top 2 closest for CORE
    brackets_sorted = sorted(brackets, key=lambda x: x["dist"])
    valid_core = []
    
    # Filter valid candidates for Core (must be within physical limits)
    for b in brackets_sorted:
        if floor_f <= b["mid"] <= ceiling_f:
            valid_core.append(b)
            
    # Select top 2 valid
    core_names = [b["name"] for b in valid_core[:2]]
    
    for b in brackets:
        name = b["name"]
        mid = b["mid"]
        
        # KILL ZONE Conditions:
        # 1. Below METAR
        # 2. Below Floor
        # 3. Above Ceiling
        # 4. > 2.5°F from center (unless it's one of the valid cores)
        is_kill = False
        
        if metar_actual_f and mid < metar_actual_f:
            is_kill = True
        elif mid < floor_f or mid > ceiling_f:
            is_kill = True
        elif b["dist"] > 2.5:
             is_kill = True
             
        if is_kill:
            kill_brackets.append(name)
        elif name in core_names:
            core_brackets.append(name)
        else:
            coverage_brackets.append(name)
            
    return core_brackets, coverage_brackets, kill_brackets



def apply_wu_alignment(
    base_weighted_f: float,
    wu_forecast_high: Optional[float],
    velocity_ratio: Optional[float] = None
) -> Tuple[float, str]:
    """
    Apply WU Market Reality check.
    
    If our prediction > WU + 2°F, apply 50% penalty.
    Exception: Skip if METAR is rising fast (velocity_ratio > 1.2)
    
    Args:
        base_weighted_f: Our weighted prediction
        wu_forecast_high: WU forecast high for the day
        velocity_ratio: Current heating velocity ratio
        
    Returns:
        Tuple of (adjusted_base, adjustment_note)
    """
    if wu_forecast_high is None:
        return base_weighted_f, ""
    
    # Skip if METAR is rising fast
    if velocity_ratio and velocity_ratio > 1.2:
        return base_weighted_f, ""
    
    divergence = base_weighted_f - wu_forecast_high
    
    if divergence > 2.0:
        penalty = divergence * 0.5  # 50% pullback
        adjusted = round(base_weighted_f - penalty, 1)
        note = f"WU Reality: -{penalty:.1f}°F"
        return adjusted, note
    
    return base_weighted_f, ""


def calculate_ensemble_v11(
    hrrr_max_f: float,
    nbm_max_f: Optional[float] = None,
    lamp_max_f: Optional[float] = None,
    lamp_confidence: float = 0.5,
    metar_actual_f: Optional[float] = None,
    local_hour: int = 12,
    local_minute: int = 0,  # v12.0: For Ultimate Squeeze timing
    radiation: Optional[float] = None,
    sky_condition: str = "SCT",
    wu_forecast_high: Optional[float] = None,
    velocity_ratio: Optional[float] = None,
    observed_max_f: Optional[float] = None  # v12.0: Cumulative max for Sunset Lock
) -> EnsembleResult:
    """
    v12.0: Sunset Lock, Cold Memory, Ultimate Squeeze, Ghost Detection.
    
    1. Hard Floor Lock: Floor = max(METAR, Floor)
    2. Forced Squeeze: Time-based spread limits
    3. WU Sync: Pull ceiling if divergent
    4. v12.0 Sunset Lock: After 14:00, if METAR < max - 0.5°F, lock ceiling
    5. v12.0 Ultimate Squeeze: After 14:30, max spread = 1.2°F
    6. v12.0 Ghost Detection: Warning if HELIOS > METAR + 3.0°F after 14:00
    """
    # Step 1: Base Calculation
    result = calculate_weighted_base(
        hrrr_max_f=hrrr_max_f,
        nbm_max_f=nbm_max_f,
        lamp_max_f=lamp_max_f,
        lamp_confidence=lamp_confidence,
        metar_actual_f=metar_actual_f
    )
    
    adjusted_ceiling = result.helios_ceiling_f
    adjusted_base = result.base_weighted_f
    notes = []
    
    # v12.0 tracking flags
    sunset_lock_active = False
    ultimate_squeeze_active = False
    ghost_prediction_warning = False
    ghost_reason = ""
    
    # =========================================================================
    # v11.2 Step 1: HARD FLOOR LOCK
    # =========================================================================
    if metar_actual_f is not None:
        if result.helios_floor_f < metar_actual_f:
            diff = metar_actual_f - result.helios_floor_f
            result.helios_floor_f = round(metar_actual_f, 1)
            notes.append(f"Hard Floor Lock (+{diff:.1f}°F)")

    # =========================================================================
    # Step 2: Physical Ceiling (v11.0)
    # =========================================================================
    physical_ceiling = None
    if metar_actual_f is not None:
        physical_ceiling = calculate_physical_ceiling(
            metar_actual_f=metar_actual_f,
            local_hour=local_hour,
            radiation=radiation,
            sky_condition=sky_condition
        )
        if adjusted_ceiling > physical_ceiling:
            adjusted_ceiling = physical_ceiling
            notes.append(f"Techo físico: {physical_ceiling:.1f}°F")

    # =========================================================================
    # v12.0 Step 3: SUNSET LOCK (Cierre de Máxima)
    # After 14:00 NYC, if METAR drops >0.5°F from observed max, lock ceiling
    # =========================================================================
    if local_hour >= 14 and observed_max_f is not None and metar_actual_f is not None:
        temp_drop = observed_max_f - metar_actual_f
        if temp_drop > 0.5:
            # Temperature is falling - Sunset Lock condition is MET
            sunset_lock_active = True
            sunset_ceiling = round(observed_max_f + 0.2, 1)
            if adjusted_ceiling > sunset_ceiling:
                adjusted_ceiling = sunset_ceiling
            notes.append(f"🔒 Sunset Lock ({observed_max_f:.1f}+0.2)")

    # =========================================================================
    # v12.0 Step 4: ULTIMATE SQUEEZE (after 14:30)
    # =========================================================================
    # Use the updated calculate_allowed_spread with minute support
    allowed_spread = calculate_allowed_spread(local_hour, local_minute)
    
    # Check if Ultimate Squeeze is active (14:30+)
    if local_hour >= 15 or (local_hour == 14 and local_minute >= 30):
        ultimate_squeeze_active = True
    
    actual_spread = adjusted_ceiling - result.helios_floor_f
    squeeze_applied = False
    
    if actual_spread > allowed_spread:
        new_ceiling = result.helios_floor_f + allowed_spread
        adjusted_ceiling = round(new_ceiling, 1)
        squeeze_applied = True
        if ultimate_squeeze_active:
            notes.append(f"⚡ Ultimate Squeeze 1.2°F")
        else:
            notes.append(f"Forced Squeeze {allowed_spread}°F")

    # =========================================================================
    # v11.2 Step 5: WU SYNCHRONIZATION
    # =========================================================================
    current_center = (result.helios_floor_f + adjusted_ceiling) / 2
    if wu_forecast_high is not None:
        if abs(current_center - wu_forecast_high) > 3.0:
            old_c = adjusted_ceiling
            adjusted_ceiling = round((adjusted_ceiling + wu_forecast_high) / 2, 1)
            notes.append(f"WU Divergence Sync (-{old_c - adjusted_ceiling:.1f}°F)")
            result.confidence_level = "DIVERGENT - USE CAUTION"
    
    # Step 6: WU Alignment for base
    adjusted_base, wu_note = apply_wu_alignment(
        base_weighted_f=adjusted_base,
        wu_forecast_high=wu_forecast_high,
        velocity_ratio=velocity_ratio
    )
    if wu_note and "WU Divergence" not in str(notes):
        notes.append(wu_note)

    # =========================================================================
    # v12.0 Step 7: GHOST PREDICTION DETECTION
    # If HELIOS > METAR + 3.0°F after 14:00, flag as potential hallucination
    # =========================================================================
    if local_hour >= 14 and metar_actual_f is not None:
        if adjusted_base > metar_actual_f + 3.0:
            ghost_prediction_warning = True
            ghost_reason = "⚠️ WARNING: POTENTIAL HRRR HALLUCINATION"
            notes.append(ghost_reason)
            
            # Priority override: use LAMP or WU if available
            if lamp_max_f is not None and lamp_max_f < adjusted_base:
                old_base = adjusted_base
                adjusted_base = lamp_max_f
                notes.append(f"Ghost Override -> LAMP ({lamp_max_f:.1f}°F)")
            elif wu_forecast_high is not None and wu_forecast_high < adjusted_base:
                old_base = adjusted_base
                adjusted_base = wu_forecast_high
                notes.append(f"Ghost Override -> WU ({wu_forecast_high:.1f}°F)")

    # =========================================================================
    # Step 8: 3-Tier Classification
    # =========================================================================
    safe_brackets, forbidden_brackets = calculate_bet_zones(
        floor_f=result.helios_floor_f,
        ceiling_f=adjusted_ceiling,
        physical_ceiling_f=physical_ceiling
    )
    
    core, coverage, kill = calculate_bet_zones_v12(
        floor_f=result.helios_floor_f,
        ceiling_f=adjusted_ceiling,
        base_weighted_f=adjusted_base,
        metar_actual_f=metar_actual_f,
        allowed_spread=allowed_spread
    )
    
    # Update Result with all values
    result.helios_ceiling_f = adjusted_ceiling
    result.base_weighted_f = adjusted_base
    result.ensemble_spread_f = round(adjusted_ceiling - result.helios_floor_f, 1)
    result.physical_ceiling_f = physical_ceiling
    result.squeeze_applied = squeeze_applied
    result.allowed_spread_f = allowed_spread
    
    result.core_brackets = core
    result.coverage_brackets = coverage
    result.kill_brackets = kill
    result.safe_brackets = safe_brackets
    result.forbidden_brackets = forbidden_brackets
    
    # v12.0: Set new tracking fields
    result.sunset_lock_active = sunset_lock_active
    result.ultimate_squeeze_active = ultimate_squeeze_active
    result.ghost_prediction_warning = ghost_prediction_warning
    result.ghost_reason = ghost_reason
    result.observed_max_used_f = observed_max_f
    
    # Confidence Logic Update
    if "DIVERGENT" not in result.confidence_level:
        if result.ensemble_spread_f > 4.0:
             result.confidence_level = "LOW"
        elif result.ensemble_spread_f <= 3.0:
             result.confidence_level = "HIGH (SNIPER)"
        else:
             result.confidence_level = "MEDIUM"

    if notes:
        result.strategy_note += " | " + " | ".join(notes)
        
    return result


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Ensemble Engine...")
    print("=" * 60)
    
    # Test case 1: Normal operation
    print("\n1. Normal operation (all models agree):")
    result = calculate_weighted_base(
        hrrr_max_f=52.0,
        nbm_max_f=51.5,
        lamp_max_f=51.0,
        lamp_confidence=0.6
    )
    print(f"   Base: {result.base_weighted_f}°F")
    print(f"   Range: {result.helios_floor_f} - {result.helios_ceiling_f}°F")
    print(f"   Confidence: {result.confidence_level}")
    print(f"   Weights: {result.weights_used}")
    
    # Test case 2: HRRR outlier (like Jan 13-14)
    print("\n2. HRRR outlier (>3°F deviation from NBM):")
    result = calculate_weighted_base(
        hrrr_max_f=54.7,
        nbm_max_f=51.0,
        lamp_max_f=50.4,
        lamp_confidence=0.7
    )
    print(f"   Base: {result.base_weighted_f}°F")
    print(f"   Range: {result.helios_floor_f} - {result.helios_ceiling_f}°F")
    print(f"   Outlier detected: {result.hrrr_outlier_detected}")
    print(f"   Weights: {result.weights_used}")
    print(f"   Strategy: {result.strategy_note}")
    
    # Test case 3: High LAMP confidence
    print("\n3. High LAMP confidence (>80%):")
    result = calculate_weighted_base(
        hrrr_max_f=52.0,
        nbm_max_f=51.5,
        lamp_max_f=50.0,
        lamp_confidence=0.85
    )
    print(f"   Base: {result.base_weighted_f}°F")
    print(f"   Weights: {result.weights_used}")
    print(f"   Note: LAMP boosted to 30%")
    
    # Test case 4: Missing NBM data
    print("\n4. Missing NBM data:")
    result = calculate_weighted_base(
        hrrr_max_f=52.0,
        nbm_max_f=None,
        lamp_max_f=50.0,
        lamp_confidence=0.6
    )
    print(f"   Base: {result.base_weighted_f}°F")
    print(f"   Weights: {result.weights_used}")
    
    # Test case 5: Wide spread (low confidence)
    print("\n5. Wide spread (>4°F):")
    result = calculate_weighted_base(
        hrrr_max_f=55.0,
        nbm_max_f=50.0,
        lamp_max_f=49.0,
        metar_actual_f=48.0
    )
    print(f"   Base: {result.base_weighted_f}°F")
    print(f"   Range: {result.helios_floor_f} - {result.helios_ceiling_f}°F")
    print(f"   Spread: {result.ensemble_spread_f}°F")
    print(f"   Confidence: {result.confidence_level}")
    
    print("\n" + "=" * 60)
    print("Ensemble Engine test complete.")
