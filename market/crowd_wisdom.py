"""
HELIOS Weather Lab - Crowd Wisdom Module
Analyzes the velocity of probability changes in Polymarket to detect
when the market is rapidly shifting or ignoring real-time data.
"""

import json
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
from database import get_connection

def track_probability_velocity(station_id: str, target_date: str, brackets: List[Tuple[str, float]]) -> None:
    """
    Store current probability snapshot for velocity analysis.
    
    Args:
        station_id: ICAO station code
        target_date: ISO date string
        brackets: List of (bracket_name, probability) tuples
    """
    now = datetime.utcnow()
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            for bracket, prob in brackets:
                cursor.execute("""
                    INSERT INTO market_velocity (timestamp, station_id, target_date, bracket_name, probability)
                    VALUES (?, ?, ?, ?, ?)
                """, (now, station_id, target_date, bracket, prob))
            conn.commit()
    except Exception as e:
        print(f"  [WISDOM] Error tracking velocity: {e}")

def detect_market_shift(station_id: str, target_date: str, threshold: float = 0.05) -> List[Dict[str, Any]]:
    """
    Detect if any bracket has shifted significantly in the last 60 minutes.
    
    Returns:
        List of shift alerts with bracket, change, and velocity.
    """
    shifts = []
    now = datetime.utcnow()
    one_hour_ago = now - timedelta(minutes=60)
    
    try:
        with get_connection() as conn:
            # For each bracket that exists today
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT bracket_name 
                FROM market_velocity 
                WHERE station_id = ? AND target_date = ?
                  AND timestamp >= ?
            """, (station_id, target_date, (now - timedelta(hours=24)).isoformat()))
            
            brackets = [r[0] for r in cursor.fetchall()]
            
            for bracket in brackets:
                # Get oldest reading in the window
                cursor.execute("""
                    SELECT probability, timestamp FROM market_velocity
                    WHERE station_id = ? AND target_date = ? AND bracket_name = ?
                      AND timestamp >= ?
                    ORDER BY timestamp ASC LIMIT 1
                """, (station_id, target_date, bracket, one_hour_ago.isoformat()))
                old_row = cursor.fetchone()
                
                # Get latest reading
                cursor.execute("""
                    SELECT probability, timestamp FROM market_velocity
                    WHERE station_id = ? AND target_date = ? AND bracket_name = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (station_id, target_date, bracket))
                new_row = cursor.fetchone()
                
                if old_row and new_row:
                    old_prob, old_time = old_row
                    new_prob, new_time = new_row
                    
                    change = new_prob - old_prob
                    if abs(change) >= threshold:
                        shifts.append({
                            "bracket": bracket,
                            "change": change,
                            "velocity": change / 1.0, # Change per hour
                            "old_prob": old_prob,
                            "new_prob": new_prob
                        })
    except Exception as e:
        print(f"  [WISDOM] Error detecting shifts: {e}")
        
    return shifts

def get_crowd_sentiment(station_id: str, target_date: str) -> str:
    """
    Analyze shifts to determine general market sentiment.
    """
    shifts = detect_market_shift(station_id, target_date)
    if not shifts:
        return "Stable"
    
    # Sort by magnitude of change
    shifts.sort(key=lambda x: abs(x['change']), reverse=True)
    top_shift = shifts[0]
    
    if top_shift['change'] > 0.10:
        return f"AGGRESSIVE MOVE into {top_shift['bracket']}"
    elif top_shift['change'] > 0.05:
        return f"SHIFTING to {top_shift['bracket']}"
    
    return "Neutral"
