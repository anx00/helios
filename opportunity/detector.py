"""
Opportunity Detector - Identify betting opportunities on Polymarket.

Compares HELIOS predictions against market consensus to find
mispriced contracts with potential value.
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Tuple, Optional
import re


_DEG = "\u00B0"


@dataclass
class BettingOpportunity:
    """Represents a potential betting opportunity."""

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

    def __str__(self) -> str:
        if self.confidence == "LOW":
            return f"SKIP - Prediction aligns with market ({self.difference:+.1f}{_DEG}F)"

        direction = "LOWER" if self.difference < 0 else "HIGHER"
        return (
            f"{self.confidence} confidence opportunity: "
            f"Bet {direction} ranges ({self.difference:+.1f}{_DEG}F from market)"
        )


def _normalize_outcome_text(outcome_str: str) -> str:
    return str(outcome_str or "").replace("\u00C3\u0082\u00C2\u00B0", _DEG).replace("\u00C2\u00B0", _DEG)


def _outcome_is_fahrenheit(outcome_str: str) -> bool:
    s = _normalize_outcome_text(outcome_str).lower()
    return (f"{_DEG}f" in s) or bool(re.search("\\d\\s*[\\u00B0\\u00BA]?\\s*f\\b", s))


def _outcome_is_celsius(outcome_str: str) -> bool:
    s = _normalize_outcome_text(outcome_str).lower()
    has_c = (f"{_DEG}c" in s) or bool(re.search("\\d\\s*[\\u00B0\\u00BA]?\\s*c\\b", s))
    return has_c and not _outcome_is_fahrenheit(s)


def _to_fahrenheit(temp: float, *, source_is_celsius: bool) -> float:
    return (temp * 9.0 / 5.0 + 32.0) if source_is_celsius else temp


def extract_temps(outcome_str: str, *, to_fahrenheit: bool = False) -> List[float]:
    """
    Extract temperature values from outcome string.

    Examples:
        "40-41{deg}F" -> [40.0, 41.0]
        "46{deg}F or higher" -> [46.0]
        "35{deg}F or below" -> [35.0]
        "6-7{deg}C" -> [44.6, 46.4]  (when to_fahrenheit=True)
    """
    clean = _normalize_outcome_text(outcome_str)
    source_is_celsius = _outcome_is_celsius(clean)
    # Range hyphens (e.g. "6-7") should not be interpreted as negative signs.
    clean_for_parse = re.sub(r"(?<=\d)-(?=\d)", "|", clean)
    temps = [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", clean_for_parse)]
    if to_fahrenheit:
        temps = [_to_fahrenheit(t, source_is_celsius=source_is_celsius) for t in temps]
    return temps


def get_market_center(outcome_str: str) -> Optional[float]:
    """
    Get center temperature of a market outcome in Fahrenheit.

    Examples:
        "40-41{deg}F" -> 40.5
        "46{deg}F or higher" -> 46.0
        "35{deg}F or below" -> 35.0
        "6-7{deg}C" -> 45.5
    """
    temps = extract_temps(outcome_str, to_fahrenheit=True)

    if not temps:
        return None
    if len(temps) >= 2:
        return (temps[0] + temps[1]) / 2.0
    return temps[0]


def check_bet_opportunity(
    prediction: float,
    polymarket_options: List[Tuple[str, float]],
    station_id: str,
    target_date: date,
) -> BettingOpportunity:
    """
    Analyze if there is a betting opportunity.

    Updated (v2.2): rounds prediction, finds the matching market bracket,
    then compares our estimate versus market center.
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
            recommendation="SKIP",
        )

    rounded_pred = round(prediction)

    # Find which market range contains our rounded prediction (in Fahrenheit).
    our_range = None
    our_range_prob = 0.0

    for outcome, prob in polymarket_options:
        temps = extract_temps(outcome, to_fahrenheit=True)
        lower = outcome.lower()
        if len(temps) >= 2:
            low, high = temps[0], temps[1]
            if low <= rounded_pred <= high:
                our_range = outcome
                our_range_prob = prob
                break
        elif "or higher" in lower and temps:
            if rounded_pred >= temps[0]:
                our_range = outcome
                our_range_prob = prob
                break
        elif "or below" in lower and temps:
            if rounded_pred <= temps[0]:
                our_range = outcome
                our_range_prob = prob
                break

    favorite_outcome, favorite_prob = max(polymarket_options, key=lambda x: x[1])
    market_center = get_market_center(favorite_outcome)
    if market_center is None:
        market_center = prediction

    if our_range:
        display_outcome = our_range
        display_prob = our_range_prob
    else:
        display_outcome = favorite_outcome
        display_prob = favorite_prob

    diff = prediction - market_center
    abs_diff = abs(diff)

    if abs_diff > 3.5:
        confidence = "HIGH"
    elif abs_diff > 2.5:
        confidence = "HIGH"
    elif abs_diff > 1.5:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    if confidence == "LOW":
        recommendation = "SKIP"
        roi_percent = 0.0
    elif diff < 0:
        recommendation = "BET_LOWER"
        roi_percent = ((1.0 / 0.15) * 0.85 - 1.0) * 100
    else:
        recommendation = "BET_HIGHER"
        roi_percent = ((1.0 / 0.15) * 0.85 - 1.0) * 100

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
        estimated_roi=roi_percent if confidence != "LOW" else None,
    )


def format_opportunity_display(opp: BettingOpportunity) -> str:
    """Format opportunity for console display."""
    if opp.confidence == "LOW":
        return f"     INFO  Sin oportunidad clara (diferencia: {opp.difference:+.1f}{_DEG}F)"

    if opp.confidence == "HIGH":
        indicator = "[HIGH]"
        conf_label = "ALTA"
    else:
        indicator = "[MED]"
        conf_label = "MEDIA"

    direction = "rangos BAJOS" if opp.recommendation == "BET_LOWER" else "rangos ALTOS"

    lines = []
    lines.append(f"     {indicator} OPORTUNIDAD DETECTADA ({conf_label})")
    lines.append(f"        Nuestra prediccion: {opp.our_prediction:.1f}{_DEG}F")
    lines.append(f"        Mercado favorece:   {opp.market_favorite} ({opp.market_favorite_prob * 100:.0f}%)")
    lines.append(f"        Diferencia:         {opp.difference:+.1f}{_DEG}F")
    lines.append(f"        Recomendacion:      Apostar a {direction}")

    if opp.estimated_roi:
        lines.append(f"        ROI estimado:       +{opp.estimated_roi:.0f}%")

    return "\n".join(lines)
