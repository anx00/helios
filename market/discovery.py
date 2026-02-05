
import httpx
import logging
import json
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple

from config import POLYMARKET_EVENTS_URL, POLYMARKET_CITY_SLUGS

logger = logging.getLogger("market_discovery")

WEATHER_TAG_ID = 84

MONTHS_ENGLISH = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}

def _build_slug_prefix(station_id: str, target_date: date) -> str:
    city = POLYMARKET_CITY_SLUGS.get(station_id, "nyc")
    month = MONTHS_ENGLISH[target_date.month]
    day = target_date.day
    return f"highest-temperature-in-{city}-on-{month}-{day}"

def _parse_iso(ts: Optional[str]) -> datetime:
    if not ts:
        return datetime.min
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.min

async def fetch_event_for_station_date(station_id: str, target_date: date) -> Optional[dict]:
    """
    Resolve the Polymarket event for a station/date.
    Uses the weather tag + end_date window and matches slug prefix.
    """
    slug_prefix = _build_slug_prefix(station_id, target_date)
    end_min = f"{target_date.isoformat()}T00:00:00Z"
    end_max = f"{target_date.isoformat()}T23:59:59Z"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            ts = int(datetime.now().timestamp() * 1000)
            response = await client.get(
                POLYMARKET_EVENTS_URL,
                params={
                    "tag_id": WEATHER_TAG_ID,
                    "end_date_min": end_min,
                    "end_date_max": end_max,
                    "limit": 200,
                    "_ts": ts,
                },
            )
            response.raise_for_status()
            data = response.json() or []
    except Exception as e:
        logger.error(f"Error resolving event for {station_id} {target_date}: {e}")
        return None

    matches = [e for e in data if (e.get("slug") or "").startswith(slug_prefix)]
    if not matches:
        return None

    # Prefer most recently updated event
    matches.sort(
        key=lambda e: _parse_iso(e.get("updatedAt") or e.get("createdAt")),
        reverse=True
    )
    return matches[0]

async def fetch_market_token_ids_for_station_date(station_id: str, target_date: date) -> List[Tuple[str, str]]:
    """
    Fetch token IDs for the station/date event (robust slug resolution).
    Returns list of (token_id, bracket_name) tuples.
    """
    event = await fetch_event_for_station_date(station_id, target_date)
    if not event:
        return []

    markets = event.get("markets", [])
    result = []
    for market in markets:
        clob_tokens = market.get("clobTokenIds")
        bracket = market.get("groupItemTitle", "")

        if clob_tokens:
            if isinstance(clob_tokens, str):
                clob_tokens = json.loads(clob_tokens)
            if clob_tokens:
                yes_token = clob_tokens[0]
                result.append((yes_token, bracket))

    return result

async def fetch_market_token_ids(event_slug: str) -> List[Tuple[str, str]]:
    """
    Fetch token IDs for all markets in an event.
    
    Returns list of (token_id, bracket_name) tuples.
    """
    url = POLYMARKET_EVENTS_URL
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            ts = int(datetime.now().timestamp() * 1000)
            response = await client.get(url, params={"slug": event_slug, "_ts": ts})
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return []
                
            event = data[0]
            markets = event.get("markets", [])
            
            result = []
            for market in markets:
                # The clobTokenIds field contains the YES/NO token IDs
                clob_tokens = market.get("clobTokenIds")
                bracket = market.get("groupItemTitle", "")
                
                if clob_tokens:
                    if isinstance(clob_tokens, str):
                        clob_tokens = json.loads(clob_tokens)
                    # First token is YES, second is NO
                    if clob_tokens:
                        yes_token = clob_tokens[0]
                        result.append((yes_token, bracket))
                        
            return result
            
    except Exception as e:
        logger.error(f"Error fetching token IDs: {e}")
        return []
