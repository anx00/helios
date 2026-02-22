"""
Helios Weather Lab - Polymarket Market Checker
Checks Polymarket Gamma API for temperature market status to determine
if today's market is resolved/mature and predictions should target tomorrow.
"""

import httpx
import re
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

from market.discovery import fetch_event_for_station_date

# City slug mappings for Polymarket event URLs
CITY_SLUGS = {
    "KLGA": "nyc",
    "KATL": "atlanta",
    "KORD": "chicago",
    "KMIA": "miami",
    "KDAL": "dallas",
    "LFPG": "paris",
    "EGLC": "london",
    "LTAC": "ankara",
}

# Spanish city names for logging
CITY_NAMES_SPANISH = {
    "KLGA": "Nueva York",
    "KATL": "Atlanta",
    "KORD": "Chicago",
    "KMIA": "Miami",
    "KDAL": "Dallas",
    "LFPG": "Paris",
    "EGLC": "Londres",
    "LTAC": "Ankara",
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
UTC = ZoneInfo("UTC")


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



def _clean_bracket_text(bracket: str) -> str:
    s = str(bracket or "")
    # Handle common mojibake cases like "Â°" / "Ã‚Â°".
    for _ in range(2):
        try:
            fixed = s.encode("latin1").decode("utf-8")
            if fixed == s:
                break
            s = fixed
        except Exception:
            break
    s = s.replace("\u00C2", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _bracket_unit(clean: str) -> str:
    lower = clean.lower()
    has_f = bool(re.search("\\d\\s*[\\u00B0\\u00BA]?\\s*f\\b", lower))
    has_c = bool(re.search("\\d\\s*[\\u00B0\\u00BA]?\\s*c\\b", lower))
    if has_c and not has_f:
        return "C"
    return "F"


def parse_bracket_midpoint(bracket: str) -> Optional[float]:
    """
    Parse a Polymarket bracket and return midpoint in Fahrenheit.

    Supports both Fahrenheit and Celsius labels:
    - "48-49°F" -> 48.5
    - "41°F or below" -> 40.5
    - "6°C or higher" -> 42.35
    """
    if not bracket:
        return None

    clean = _clean_bracket_text(bracket)
    lower = clean.lower()
    is_celsius = _bracket_unit(clean) == "C"

    def to_f(value: float) -> float:
        return (value * 9.0 / 5.0 + 32.0) if is_celsius else value

    try:
        range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", clean)
        if range_match:
            low = float(range_match.group(1))
            high = float(range_match.group(2))
            return to_f((low + high) / 2.0)

        num_match = re.search(r"-?\d+(?:\.\d+)?", clean)
        if not num_match:
            return None

        temp = float(num_match.group(0))
        if "or below" in lower:
            return to_f(temp - 0.5)
        if "or above" in lower or "or higher" in lower:
            return to_f(temp + 0.5)
        return to_f(temp)
    except Exception:
        return None


def parse_bracket_threshold(bracket: str) -> Optional[float]:
    """
    Parse bracket and return the LOW threshold in Fahrenheit that must be
    reached to win this bracket.

    Examples:
    - "46°F or higher" -> 46.0
    - "44-45°F" -> 44.0
    - "6°C or higher" -> 42.8
    - "41°F or below" -> None
    """
    if not bracket:
        return None

    clean = _clean_bracket_text(bracket)
    lower = clean.lower()
    is_celsius = _bracket_unit(clean) == "C"

    def to_f(value: float) -> float:
        return (value * 9.0 / 5.0 + 32.0) if is_celsius else value

    # "X or below" wins by staying low; no upward threshold exists.
    if "or below" in lower:
        return None

    try:
        range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", clean)
        if range_match:
            return to_f(float(range_match.group(1)))
    except Exception:
        pass

    try:
        num_match = re.search(r"-?\d+(?:\.\d+)?", clean)
        if num_match:
            return to_f(float(num_match.group(0)))
    except Exception:
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

    ROLLOVER POLICY:
    1. Keep today's market while it is still open/active in station timezone
    2. Move to tomorrow only when today's market is closed/inactive (or past endDate)
    3. Prefer switching only when tomorrow's market is discoverable
    
    Args:
        station_code: ICAO station code
        local_now: Current local time
        
    Returns:
        Target date for predictions
    """
    def _parse_event_dt(ts: Optional[str]) -> Optional[datetime]:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return None

    def _is_closed_for_rollover(event_data: Optional[dict], now_local: datetime) -> bool:
        if not event_data:
            return False

        if bool(event_data.get("closed")):
            return True
        if not bool(event_data.get("active", True)):
            return True

        end_dt = _parse_event_dt(event_data.get("endDate"))
        if end_dt is not None:
            now_utc = now_local.astimezone(UTC) if now_local.tzinfo else now_local.replace(tzinfo=UTC)
            if end_dt <= now_utc:
                return True

        return False

    today = local_now.date()
    tomorrow = today + timedelta(days=1)

    # Step 1: Resolve today's event (robust date lookup).
    today_event = await fetch_event_for_station_date(station_code, today)
    if not today_event:
        today_slug = build_event_slug(station_code, today)
        today_event = await fetch_event_data(today_slug)

    if today_event is None:
        # Conservative fallback: switch only if tomorrow market is already available.
        tomorrow_event = await fetch_event_for_station_date(station_code, tomorrow)
        if not tomorrow_event:
            tomorrow_slug = build_event_slug(station_code, tomorrow)
            tomorrow_event = await fetch_event_data(tomorrow_slug)

        if tomorrow_event:
            log_market_status(station_code, None, True, None, 0.0, tomorrow, is_today=True)
            return tomorrow

        # Keep today if discovery is flaky/missing.
        log_market_status(station_code, None, False, None, 0.0, today, is_today=True)
        return today

    # Step 2: Assess maturity for observability only (not rollover trigger).
    is_mature_today, outcome_today, prob_today = is_event_mature(today_event)
    today_closed = _is_closed_for_rollover(today_event, local_now)

    if today_closed:
        tomorrow_event = await fetch_event_for_station_date(station_code, tomorrow)
        if not tomorrow_event:
            tomorrow_slug = build_event_slug(station_code, tomorrow)
            tomorrow_event = await fetch_event_data(tomorrow_slug)

        log_market_status(station_code, today_event, True, outcome_today, prob_today, tomorrow, is_today=True)

        if tomorrow_event:
            is_mature_tmr, outcome_tmr, prob_tmr = is_event_mature(tomorrow_event)
            log_market_status(station_code, tomorrow_event, is_mature_tmr, outcome_tmr, prob_tmr, tomorrow, is_today=False)
            return tomorrow

        # If tomorrow is not yet listed, keep today to avoid empty subscriptions.
        return today

    # Today's market still active: never roll just because probabilities are high.
    log_market_status(station_code, today_event, is_mature_today, outcome_today, prob_today, today, is_today=True)
    return today

