"""
Helios Weather Lab - Polymarket Market Checker
Checks Polymarket Gamma API for temperature market status to determine
if today's market is resolved/mature and predictions should target tomorrow.
"""

import httpx
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

from market.discovery import fetch_event_for_station_date

# City slug mappings for Polymarket event URLs
CITY_SLUGS = {
    "KLGA": "nyc",
    "KATL": "atlanta",
    "EGLC": "london",
}

# Spanish city names for logging
CITY_NAMES_SPANISH = {
    "KLGA": "Nueva York",
    "KATL": "Atlanta",
    "EGLC": "Londres",
}

# Spanish month names (abbreviated)
MONTHS_SPANISH = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}

# English month names (lowercase for slug)
MONTHS_ENGLISH = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}

# Polymarket API endpoint
POLYMARKET_EVENTS_URL = "https://gamma-api.polymarket.com/events"

# Probability threshold (98% = 0.98)
MARKET_MATURE_PROBABILITY = 0.98


@dataclass
class MarketStatus:
    """Market status information from Polymarket."""
    is_mature: bool
    winning_outcome: Optional[str]
    max_probability: float
    event_closed: bool
    event_active: bool
    target_date: date


@dataclass
class MarketTop3:
    """Top 3 market brackets for arbitrage dashboard."""
    top1_bracket: Optional[str]
    top1_prob: float
    top2_bracket: Optional[str]
    top2_prob: float
    top3_bracket: Optional[str]
    top3_prob: float
    weighted_avg: float  # Weighted average of bracket midpoints


@dataclass
class MarketAllOptions:
    """All market options for comprehensive visualization."""
    all_brackets: list  # List of (bracket_name, probability) tuples, sorted by prob desc
    top1_bracket: Optional[str]
    top1_prob: float
    top2_bracket: Optional[str]
    top2_prob: float
    top3_bracket: Optional[str]
    top3_prob: float
    weighted_avg: float
    all_brackets_json: str  # JSON string for DB storage


@dataclass
class MarketResolution:
    """Market resolution status based on observed temperature."""
    is_resolved: bool
    winning_bracket: Optional[str]
    observed_max_f: Optional[float]
    resolution_threshold_f: Optional[float]  # The bracket threshold that was crossed
    resolution_time: Optional[str]
    remaining_upside: bool  # True if higher brackets still possible
    all_resolved_brackets: list  # List of brackets that have been definitively resolved



def parse_bracket_midpoint(bracket: str) -> Optional[float]:
    """
    Parse bracket like '48-49' and return midpoint (48.5).
    Handle edge cases like '41°F or below' -> 40.5
    
    CELSIUS SUPPORT: Also handles '3°C', '6°C or higher', etc.
    Returns the value in the native unit (Celsius for C, Fahrenheit for F)
    """
    if not bracket:
        return None
    
    # Handle "X°F or below" or "X°C or below" format
    if "or below" in bracket.lower():
        try:
            temp = float(bracket.split("°")[0].strip())
            return temp - 0.5
        except:
            return None
    
    # Handle "X°F or above" or "X°C or above" or "X°C or higher" format
    if "or above" in bracket.lower() or "or higher" in bracket.lower():
        try:
            temp = float(bracket.split("°")[0].strip())
            return temp + 0.5
        except:
            return None
    
    # Handle single value format like "3°C" or "48°F"
    if "°C" in bracket or "°F" in bracket:
        try:
            clean = bracket.replace("°C", "").replace("°F", "").strip()
            # If no range separator, it's a single value
            if "-" not in clean:
                return float(clean)
        except:
            pass
    
    # Handle "XX-YY" format (e.g., "48-49°F" or "3-4°C")
    try:
        # Remove °F or °C suffix if present
        clean = bracket.replace("°F", "").replace("°C", "").replace("°", "").strip()
        parts = clean.split("-")
        if len(parts) == 2:
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return (low + high) / 2
    except:
        pass
    
    return None


def parse_bracket_threshold(bracket: str) -> Optional[float]:
    """
    Parse bracket and return the LOW threshold that must be reached to WIN this bracket.
    
    Examples:
    - "46°F or higher" -> 46.0 (must reach 46 to win)
    - "44-45°F" -> 44.0 (must reach 44 to enter this range)
    - "41°F or below" -> None (this wins if temp stays below, no threshold to reach)
    """
    if not bracket:
        return None
    
    bracket_lower = bracket.lower()
    
    # "X or higher" / "X or above" - threshold is X
    if "or higher" in bracket_lower or "or above" in bracket_lower:
        try:
            temp = float(bracket.split("°")[0].strip())
            return temp
        except:
            return None
    
    # "X or below" - no upward threshold (wins by staying low)
    if "or below" in bracket_lower:
        return None
    
    # Range format "XX-YY°F" - threshold is the low value
    try:
        clean = bracket.replace("°F", "").replace("°C", "").replace("°", "").strip()
        parts = clean.split("-")
        if len(parts) == 2:
            low = float(parts[0].strip())
            return low
    except:
        pass
    
    # Single value "48°F" - threshold is that value
    try:
        clean = bracket.replace("°F", "").replace("°C", "").replace("°", "").strip()
        if "-" not in clean:
            return float(clean)
    except:
        pass
    
    return None


def check_market_resolution(
    all_brackets: list,
    observed_max_f: Optional[float],
    observed_max_time: Optional[str] = None
) -> MarketResolution:
    """
    Check if the market has been resolved based on Polymarket probability.
    
    A market is considered "resolved" when a bracket has >= 98% probability
    (virtual certainty). Simply crossing a temperature threshold is NOT enough
    because the temperature could still change.
    
    Args:
        all_brackets: List of (bracket_name, probability) tuples from Polymarket
        observed_max_f: The observed maximum temperature in Fahrenheit
        observed_max_time: Time when the max was observed (for display)
    
    Returns:
        MarketResolution with resolution status
    """
    RESOLUTION_THRESHOLD = 0.98  # 98% probability = virtual certainty
    
    if not all_brackets:
        return MarketResolution(
            is_resolved=False,
            winning_bracket=None,
            observed_max_f=observed_max_f,
            resolution_threshold_f=None,
            resolution_time=None,
            remaining_upside=True,
            all_resolved_brackets=[]
        )
    
    # Find bracket with highest probability
    winning_bracket = None
    winning_prob = 0.0
    winning_threshold = None
    
    for bracket_name, prob in all_brackets:
        if prob > winning_prob:
            winning_prob = prob
            winning_bracket = bracket_name
            winning_threshold = parse_bracket_threshold(bracket_name)
    
    # Market is ONLY resolved if probability >= 98%
    is_resolved = winning_prob >= RESOLUTION_THRESHOLD
    
    # Check remaining upside based on observed max vs highest threshold
    remaining_upside = True
    if is_resolved and observed_max_f is not None:
        # Get the highest threshold from all brackets
        highest_threshold = None
        for bracket_name, prob in all_brackets:
            threshold = parse_bracket_threshold(bracket_name)
            if threshold is not None:
                if highest_threshold is None or threshold > highest_threshold:
                    highest_threshold = threshold
        
        # No remaining upside if we've crossed the highest threshold
        if highest_threshold is not None and observed_max_f >= highest_threshold:
            remaining_upside = False
    
    return MarketResolution(
        is_resolved=is_resolved,
        winning_bracket=winning_bracket if is_resolved else None,
        observed_max_f=observed_max_f,
        resolution_threshold_f=winning_threshold if is_resolved else None,
        resolution_time=observed_max_time if is_resolved else None,
        remaining_upside=remaining_upside,
        all_resolved_brackets=[winning_bracket] if is_resolved else []
    )



async def get_market_top3(station_id: str, target_date: date) -> Optional[MarketTop3]:
    """
    Fetch top 3 brackets from Polymarket for arbitrage comparison.
    
    Returns MarketTop3 with brackets sorted by probability and weighted avg.
    """
    import json
    
    event_data = await fetch_event_for_station_date(station_id, target_date)
    if not event_data:
        slug = build_event_slug(station_id, target_date)
        event_data = await fetch_event_data(slug)
    
    if not event_data:
        return None
    
    markets = event_data.get("markets", [])
    parsed_markets = []
    
    for market in markets:
        outcome_prices_raw = market.get("outcomePrices", [])
        bracket = market.get("groupItemTitle", "")
        
        try:
            if isinstance(outcome_prices_raw, str):
                outcome_prices = json.loads(outcome_prices_raw)
            else:
                outcome_prices = outcome_prices_raw
            
            if outcome_prices and len(outcome_prices) > 0:
                yes_price = float(outcome_prices[0])
                parsed_markets.append((bracket, yes_price))
        except:
            continue
    
    # Sort by probability descending
    parsed_markets.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 3
    top1 = parsed_markets[0] if len(parsed_markets) > 0 else (None, 0.0)
    top2 = parsed_markets[1] if len(parsed_markets) > 1 else (None, 0.0)
    top3 = parsed_markets[2] if len(parsed_markets) > 2 else (None, 0.0)
    
    # Calculate weighted average of midpoints
    weighted_sum = 0.0
    prob_sum = 0.0
    
    for bracket, prob in [top1, top2, top3]:
        midpoint = parse_bracket_midpoint(bracket)
        if midpoint is not None and prob > 0:
            weighted_sum += midpoint * prob
            prob_sum += prob
    
    weighted_avg = weighted_sum / prob_sum if prob_sum > 0 else 0.0
    
    return MarketTop3(
        top1_bracket=top1[0],
        top1_prob=top1[1],
        top2_bracket=top2[0],
        top2_prob=top2[1],
        top3_bracket=top3[0],
        top3_prob=top3[1],
        weighted_avg=weighted_avg
    )


async def get_market_all_options(station_id: str, target_date: date) -> Optional[MarketAllOptions]:
    """
    Fetch ALL brackets from Polymarket for comprehensive chart visualization.
    
    Returns MarketAllOptions with all brackets and their probabilities.
    """
    import json
    
    event_data = await fetch_event_for_station_date(station_id, target_date)
    if not event_data:
        slug = build_event_slug(station_id, target_date)
        event_data = await fetch_event_data(slug)
    
    if not event_data:
        return None
    
    markets = event_data.get("markets", [])
    parsed_markets = []
    
    for market in markets:
        outcome_prices_raw = market.get("outcomePrices", [])
        bracket = market.get("groupItemTitle", "")
        
        try:
            if isinstance(outcome_prices_raw, str):
                outcome_prices = json.loads(outcome_prices_raw)
            else:
                outcome_prices = outcome_prices_raw
            
            if outcome_prices and len(outcome_prices) > 0:
                yes_price = float(outcome_prices[0])
                parsed_markets.append((bracket, yes_price))
        except:
            continue
    
    # Sort by probability descending
    parsed_markets.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 3
    top1 = parsed_markets[0] if len(parsed_markets) > 0 else (None, 0.0)
    top2 = parsed_markets[1] if len(parsed_markets) > 1 else (None, 0.0)
    top3 = parsed_markets[2] if len(parsed_markets) > 2 else (None, 0.0)
    
    # Calculate weighted average of midpoints (using all options)
    weighted_sum = 0.0
    prob_sum = 0.0
    
    for bracket, prob in parsed_markets:
        midpoint = parse_bracket_midpoint(bracket)
        if midpoint is not None and prob > 0:
            weighted_sum += midpoint * prob
            prob_sum += prob
    
    weighted_avg = weighted_sum / prob_sum if prob_sum > 0 else 0.0
    
    # Create JSON for DB storage
    all_brackets_json = json.dumps([{"bracket": b, "prob": p} for b, p in parsed_markets])
    
    return MarketAllOptions(
        all_brackets=parsed_markets,
        top1_bracket=top1[0],
        top1_prob=top1[1],
        top2_bracket=top2[0],
        top2_prob=top2[1],
        top3_bracket=top3[0],
        top3_prob=top3[1],
        weighted_avg=weighted_avg,
        all_brackets_json=all_brackets_json
    )



def build_event_slug(station_code: str, target_date: date) -> str:
    """
    Build Polymarket event slug from station and date.
    
    Example: "highest-temperature-in-nyc-on-january-6"
    
    Args:
        station_code: ICAO station code (e.g., "KLGA")
        target_date: Date to build slug for
        
    Returns:
        Event slug string for Polymarket API
    """
    city = CITY_SLUGS.get(station_code, "nyc")
    month = MONTHS_ENGLISH[target_date.month]
    day = target_date.day  # No leading zero
    
    return f"highest-temperature-in-{city}-on-{month}-{day}"


async def fetch_event_data(event_slug: str) -> Optional[dict]:
    """
    Fetch event data from Polymarket Gamma API.
    
    Args:
        event_slug: Event slug to fetch
        
    Returns:
        Event JSON data or None on error
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            # v5.5: Add cache-busting timestamp to ensure real-time data
            ts = int(datetime.now().timestamp() * 1000)
            response = await client.get(
                POLYMARKET_EVENTS_URL,
                params={"slug": event_slug, "_ts": ts}
            )
            response.raise_for_status()
            data = response.json()
            
            # API returns array with single event
            if data and len(data) > 0:
                return data[0]
            else:
                return None

            
    except httpx.HTTPError as e:
        return None
    except Exception as e:
        return None


def is_event_mature(event_data: dict) -> Tuple[bool, Optional[str], float]:
    """
    Check if event is mature/resolved.
    
    AGGRESSIVE LOGIC:
    1. First check all market probabilities for >98% (VIRTUAL CERTAINTY)
    2. Then check event-level flags (closed/inactive) as fallback
    
    An event is considered mature if:
    - ANY market within event has probability > 98% (PRIORITY)
    - Event is closed (closed=True) or inactive (active=False)
    
    IMPORTANT: In Polymarket, outcomePrices = [YesPrice, NoPrice]
    - We ONLY care about YesPrice (index 0)
    - NoPrice is just (1 - YesPrice)
    
    Args:
        event_data: Event JSON from Polymarket API
        
    Returns:
        Tuple of (is_mature, winning_outcome_text, max_probability)
    """
    # PRIORITY: Check all markets in event for high probability (VIRTUAL CERTAINTY)
    markets = event_data.get("markets", [])
    max_prob = 0.0
    winning_outcome = None
    
    for market in markets:
        # Get outcome prices - they come as JSON STRING, not array!
        outcome_prices_raw = market.get("outcomePrices", [])
        
        # Parse probabilities
        try:
            # outcomePrices comes as a JSON string like: "[\"0.9995\", \"0.0005\"]"
            # We need to parse it first
            import json
            if isinstance(outcome_prices_raw, str):
                outcome_prices = json.loads(outcome_prices_raw)
            else:
                outcome_prices = outcome_prices_raw
            
            # CRITICAL: Only check the YES price (index 0), not the NO price!
            # outcomePrices = [YesPrice, NoPrice]
            # Example: ["0.98", "0.02"] means 98% chance this outcome happens
            if outcome_prices and len(outcome_prices) > 0:
                yes_price = float(outcome_prices[0])
                
                if yes_price > max_prob:
                    max_prob = yes_price
                    # Extract temperature range from groupItemTitle
                    winning_outcome = market.get("groupItemTitle", "")
                    
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            # Skip markets with parsing errors
            continue
    
    # Event is mature if any outcome has >= 98% probability (VIRTUAL CERTAINTY)
    if max_prob >= MARKET_MATURE_PROBABILITY:
        return (True, winning_outcome, max_prob)
    
    # FALLBACK: Check event-level flags
    event_closed = event_data.get("closed", False)
    event_active = event_data.get("active", True)
    
    if event_closed or not event_active:
        return (True, None, 1.0)
    
    # Market still has uncertainty
    return (False, winning_outcome, max_prob)


def log_market_status(
    station_code: str,
    event_data: Optional[dict],
    is_mature: bool,
    outcome: Optional[str],
    probability: float,
    target_date: date,
    is_today: bool
) -> None:
    """
    Log market status - now silent to reduce console noise.
    """
    pass  # Logging disabled


def format_date_spanish(d: date) -> str:
    """Format date in Spanish: 06-Ene-2026"""
    return f"{d.day:02d}-{MONTHS_SPANISH[d.month]}-{d.year}"


async def get_target_date(station_code: str, local_now: datetime) -> date:
    """
    Determine target date for predictions based on market status.
    
    PRIORITY LOGIC:
    1. Check today's market for virtual certainty (>98%) or closed status
    2. If today is mature, check if tomorrow's market exists
    3. Target tomorrow if today is mature, today if still active
    
    Args:
        station_code: ICAO station code
        local_now: Current local time
        
    Returns:
        Target date for predictions
    """
    today = local_now.date()
    tomorrow = today + timedelta(days=1)
    
    # Step 1: Resolve today's event (robust date lookup)
    today_event = await fetch_event_for_station_date(station_code, today)
    if not today_event:
        today_slug = build_event_slug(station_code, today)
        today_event = await fetch_event_data(today_slug)
    
    if today_event is None:
        # Today's market not found - target tomorrow
        log_market_status(station_code, None, True, None, 0.0, tomorrow, is_today=True)
        return tomorrow
    
    # Step 2: Check if today's market is mature
    is_mature_today, outcome_today, prob_today = is_event_mature(today_event)
    
    if is_mature_today:
        # Today's market has virtual certainty or is closed
        # Check if tomorrow's market exists
        tomorrow_event = await fetch_event_for_station_date(station_code, tomorrow)
        if not tomorrow_event:
            tomorrow_slug = build_event_slug(station_code, tomorrow)
            tomorrow_event = await fetch_event_data(tomorrow_slug)
        
        # Log today's status
        log_market_status(station_code, today_event, True, outcome_today, prob_today, tomorrow, is_today=True)
        
        if tomorrow_event:
            # Tomorrow's market exists - log it too
            is_mature_tmr, outcome_tmr, prob_tmr = is_event_mature(tomorrow_event)
            log_market_status(station_code, tomorrow_event, is_mature_tmr, outcome_tmr, prob_tmr, tomorrow, is_today=False)
        
        return tomorrow
    else:
        # Today's market still active - target today
        log_market_status(station_code, today_event, False, outcome_today, prob_today, today, is_today=True)
        return today
