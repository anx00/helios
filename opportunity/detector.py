"""
Opportunity Detector - Identify betting opportunities on Polymarket

Compares HELIOS predictions against market consensus to find
mispriced contracts with potential value.
"""
from dataclasses import dataclass
from datetime import date
from typing import List, Tuple, Optional
import re


@dataclass
class BettingOpportunity:
    """Represents a potential betting opportunity"""
    station_id: str
    target_date: date
    our_prediction: float
    market_favorite: str
    market_favorite_prob: float
    market_center: float
    difference: float  # our_pred - market_center
    confidence: str     # HIGH/MEDIUM/LOW
    recommendation: str # BET_LOWER/BET_HIGHER/SKIP
    estimated_roi: Optional[float] = None
    
    def __str__(self):
        if self.confidence == "LOW":
            return f"SKIP - Prediction aligns with market ({self.difference:+.1f}°F)"
        
        direction = "LOWER" if self.difference < 0 else "HIGHER"
        return (f"{self.confidence} confidence opportunity: "
                f"Bet {direction} ranges ({self.difference:+.1f}°F from market)")


def extract_temps(outcome_str: str) -> List[float]:
    """
    Extract temperature values from outcome string.
    
    Examples:
        "40-41°F" → [40.0, 41.0]
        "46°F or higher" → [46.0]
        "35°F or below" → [35.0]
    """
    temps = re.findall(r'\d+', outcome_str)
    return [float(t) for t in temps]


def get_market_center(outcome_str: str) -> Optional[float]:
    """
    Get center temperature of a market outcome.
    
    Examples:
        "40-41°F" → 40.5
        "46°F or higher" → 46.0
        "35°F or below" → 35.0
    """
    temps = extract_temps(outcome_str)
    
    if not temps:
        return None
    elif len(temps) >= 2:
        # Range like "40-41"
        return (temps[0] + temps[1]) / 2
    else:
        # Single value like "46 or higher"
        return temps[0]


def check_bet_opportunity(
    prediction: float,
    polymarket_options: List[Tuple[str, float]],
    station_id: str,
    target_date: date
) -> BettingOpportunity:
    """
    Analyze if there's a betting opportunity.
    
    UPDATED (v2.2): Now rounds the prediction to find which market range it falls into,
    then compares our probability against the market's.
    
    Args:
        prediction: Our temperature prediction
        polymarket_options: List of (outcome_title, probability) tuples
        station_id: Station identifier
        target_date: Date of prediction
        
    Returns:
        BettingOpportunity object with analysis
    """
    if not polymarket_options:
        return BettingOpportunity(
            station_id=station_id,
            target_date=target_date,
            our_prediction=prediction,
            market_favorite="N/A",
            market_favorite_prob=0.0,
            market_center=0.0,
            difference=0.0,
            confidence="LOW",
            recommendation="SKIP"
        )
    
    # Round prediction to nearest integer
    rounded_pred = round(prediction)
    
    # Find which market range contains our rounded prediction
    our_range = None
    our_range_prob = 0.0
    
    for outcome, prob in polymarket_options:
        temps = extract_temps(outcome)
        if len(temps) >= 2:
            # Range like "48-49°F"
            low, high = temps[0], temps[1]
            if low <= rounded_pred <= high:
                our_range = outcome
                our_range_prob = prob
                break
        elif "or higher" in outcome.lower() and temps:
            # "52°F or higher"
            if rounded_pred >= temps[0]:
                our_range = outcome
                our_range_prob = prob
                break
        elif "or below" in outcome.lower() and temps:
            # "41°F or below"
            if rounded_pred <= temps[0]:
                our_range = outcome
                our_range_prob = prob
                break
    
    # Find market favorite (highest probability)
    favorite_outcome, favorite_prob = max(polymarket_options, key=lambda x: x[1])
    market_center = get_market_center(favorite_outcome)
    
    if market_center is None:
        market_center = prediction
    
    # If we found our range, use that for display
    if our_range:
        display_outcome = our_range
        display_prob = our_range_prob
    else:
        display_outcome = favorite_outcome
        display_prob = favorite_prob
    
    # Calculate difference from market favorite center
    diff = prediction - market_center
    abs_diff = abs(diff)
    
    # Confidence based on difference
    if abs_diff > 3.5:
        confidence = "HIGH"
    elif abs_diff > 2.5:
        confidence = "HIGH"
    elif abs_diff > 1.5:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # Recommendation
    if confidence == "LOW":
        recommendation = "SKIP"
        roi_percent = 0.0
    elif diff < 0:
        recommendation = "BET_LOWER"
        roi_percent = ((1.0 / 0.15) * 0.85) - 1.0
        roi_percent *= 100
    else:
        recommendation = "BET_HIGHER"
        roi_percent = ((1.0 / 0.15) * 0.85) - 1.0
        roi_percent *= 100
    
    return BettingOpportunity(
        station_id=station_id,
        target_date=target_date,
        our_prediction=prediction,
        market_favorite=display_outcome,
        market_favorite_prob=display_prob,
        market_center=market_center,
        difference=diff,
        confidence=confidence,
        recommendation=recommendation,
        estimated_roi=roi_percent if confidence != "LOW" else None
    )


def format_opportunity_display(opp: BettingOpportunity) -> str:
    """
    Format opportunity for console display.
    
    Returns colored/formatted string for terminal output.
    """
    if opp.confidence == "LOW":
        return f"     ℹ️  Sin oportunidad clara (diferencia: {opp.difference:+.1f}°F)"
    
    # Color codes
    if opp.confidence == "HIGH":
        indicator = "🟢"
        conf_label = "ALTA"
    else:
        indicator = "🟡"
        conf_label = "MEDIA"
    
    direction = "rangos BAJOS" if opp.recommendation == "BET_LOWER" else "rangos ALTOS"
    
    lines = []
    lines.append(f"     {indicator} OPORTUNIDAD DETECTADA ({conf_label})")
    lines.append(f"        Nuestra predicción: {opp.our_prediction:.1f}°F")
    lines.append(f"        Mercado favorece:   {opp.market_favorite} ({opp.market_favorite_prob*100:.0f}%)")
    lines.append(f"        Diferencia:         {opp.difference:+.1f}°F")
    lines.append(f"        Recomendación:      Apostar a {direction}")
    
    if opp.estimated_roi:
        lines.append(f"        ROI estimado:       +{opp.estimated_roi:.0f}%")
    
    return "\n".join(lines)
