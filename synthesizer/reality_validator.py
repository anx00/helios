"""
HELIOS Weather Lab - Reality Validator (Anti-Ghost Peak)
Ensures temperature peaks are verified by multiple consecutive readings 
to avoid locking false "reality floors" based on single outliers.
"""

from typing import Optional, Tuple
from database import get_connection

def validate_reality_floor(station_id: str, target_date_str: str, current_temp_f: float) -> Tuple[float, bool]:
    """
    Validate if a new maximum temperature should be accepted as the reality floor.
    Requires at least 2 consecutive readings at or above the value.
    
    Args:
        station_id: ICAO station code
        target_date_str: ISO date string (YYYY-MM-DD)
        current_temp_f: The latest observed temperature
        
    Returns:
        Tuple of (validated_floor_f, is_new_peak)
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Get current validated floor and last seen unvalidated value
            # We use a small table 'reality_check_state' for this persistence
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reality_check_state (
                    station_id TEXT,
                    target_date TEXT,
                    validated_floor REAL,
                    last_unvalidated_val REAL,
                    consecutive_count INTEGER,
                    PRIMARY KEY (station_id, target_date)
                )
            """)
            
            cursor.execute("""
                SELECT validated_floor, last_unvalidated_val, consecutive_count
                FROM reality_check_state
                WHERE station_id = ? AND target_date = ?
            """, (station_id, target_date_str))
            
            row = cursor.fetchone()
            
            if not row:
                # Initialize state for this day
                cursor.execute("""
                    INSERT INTO reality_check_state (station_id, target_date, validated_floor, last_unvalidated_val, consecutive_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (station_id, target_date_str, current_temp_f, current_temp_f, 1))
                conn.commit()
                return current_temp_f, False

            validated_floor, last_val, count = row
            
            # Logic: If current_temp > validated_floor, we need to verify it
            if current_temp_f > validated_floor:
                # Is it the same or higher than the last unvalidated peek?
                # We allow a small tolerance (0.2F) to account for slight dips
                if current_temp_f >= (last_val - 0.2):
                    new_count = count + 1
                    if new_count >= 2:
                        # VALIDATED! New peak confirmed
                        new_floor = max(current_temp_f, last_val)
                        cursor.execute("""
                            UPDATE reality_check_state
                            SET validated_floor = ?, last_unvalidated_val = ?, consecutive_count = ?
                            WHERE station_id = ? AND target_date = ?
                        """, (new_floor, current_temp_f, new_count, station_id, target_date_str))
                        conn.commit()
                        return new_floor, True
                    else:
                        # First repeat, still unvalidated
                        cursor.execute("""
                            UPDATE reality_check_state
                            SET last_unvalidated_val = ?, consecutive_count = ?
                            WHERE station_id = ? AND target_date = ?
                        """, (current_temp_f, new_count, station_id, target_date_str))
                        conn.commit()
                        return validated_floor, False
                else:
                    # Value dropped significantly - reset count but keep current_temp as last_val
                    cursor.execute("""
                        UPDATE reality_check_state
                        SET last_unvalidated_val = ?, consecutive_count = 1
                        WHERE station_id = ? AND target_date = ?
                    """, (current_temp_f, station_id, target_date_str))
                    conn.commit()
                    return validated_floor, False
            else:
                # Current temp is below floor, just update last seen (reset stay-at-top count)
                cursor.execute("""
                    UPDATE reality_check_state
                    SET last_unvalidated_val = ?, consecutive_count = 0
                    WHERE station_id = ? AND target_date = ?
                """, (current_temp_f, station_id, target_date_str))
                conn.commit()
                return validated_floor, False

    except Exception as e:
        print(f"[RE-VALIDATOR] Error: {e}")
        return current_temp_f, False
