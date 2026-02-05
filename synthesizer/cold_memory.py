"""
HELIOS v12.0 - Cold Memory Module
Persists negative physics adjustments across updates to maintain thermal inertia.

Key Rule: If negative advection/SST adjustment persists for >3 hours,
the cold memory "locks" and decays at max 20%/hour even if HRRR removes it.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
from database import get_connection


@dataclass
class ColdMemoryState:
    """State for cold memory persistence tracking."""
    station_id: str
    target_date: str
    accumulated_adj_f: float  # Accumulated negative adjustment in °F
    hours_active: int         # Hours this adjustment has been active
    last_update_hour: int     # Last hour when updated
    locked: bool              # True if persistence threshold (3h) reached


def get_cold_memory(station_id: str, target_date: str) -> Optional[ColdMemoryState]:
    """
    Get current cold memory state for station/date.
    
    Returns:
        ColdMemoryState if exists, None otherwise
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Ensure table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cold_memory_state (
                    station_id TEXT,
                    target_date TEXT,
                    accumulated_adj_f REAL,
                    hours_active INTEGER,
                    last_update_hour INTEGER,
                    locked INTEGER,
                    PRIMARY KEY (station_id, target_date)
                )
            """)
            
            cursor.execute("""
                SELECT accumulated_adj_f, hours_active, last_update_hour, locked
                FROM cold_memory_state
                WHERE station_id = ? AND target_date = ?
            """, (station_id, target_date))
            
            row = cursor.fetchone()
            if row:
                return ColdMemoryState(
                    station_id=station_id,
                    target_date=target_date,
                    accumulated_adj_f=row[0],
                    hours_active=row[1],
                    last_update_hour=row[2],
                    locked=bool(row[3])
                )
            return None
            
    except Exception as e:
        print(f"[COLD_MEMORY] Error reading state: {e}")
        return None


def save_cold_memory(
    station_id: str,
    target_date: str,
    accumulated_adj_f: float,
    hours_active: int,
    last_update_hour: int,
    locked: bool
) -> None:
    """Save cold memory state to database."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO cold_memory_state 
                (station_id, target_date, accumulated_adj_f, hours_active, last_update_hour, locked)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (station_id, target_date, accumulated_adj_f, hours_active, last_update_hour, int(locked)))
            conn.commit()
    except Exception as e:
        print(f"[COLD_MEMORY] Error saving state: {e}")


def apply_cold_memory(
    station_id: str,
    target_date: str,
    current_physics_adj_f: float,
    current_hour: int
) -> Tuple[float, str]:
    """
    Apply cold memory persistence logic.
    
    Rules:
    1. If physics_adj < 0 for >3 consecutive hours, LOCK the cold memory
    2. Once locked, if HRRR tries to eliminate negative, resist with 20%/hour decay
    3. Always track the most negative adjustment seen
    
    Args:
        station_id: ICAO station code
        target_date: ISO date string (YYYY-MM-DD)
        current_physics_adj_f: Current physics adjustment in °F
        current_hour: Current local hour (0-23)
        
    Returns:
        Tuple of (adjusted_physics_f, reason_note)
    """
    state = get_cold_memory(station_id, target_date)
    
    # =========================================================================
    # Case 1: No existing state - initialize
    # =========================================================================
    if state is None:
        if current_physics_adj_f < -0.5:  # Meaningful negative adjustment
            save_cold_memory(
                station_id=station_id,
                target_date=target_date,
                accumulated_adj_f=current_physics_adj_f,
                hours_active=1,
                last_update_hour=current_hour,
                locked=False
            )
            return current_physics_adj_f, ""
        return current_physics_adj_f, ""
    
    # =========================================================================
    # Case 2: State exists - apply persistence logic
    # =========================================================================
    hours_since_update = current_hour - state.last_update_hour
    if hours_since_update < 0:
        hours_since_update += 24  # Handle day rollover
    
    # Update hours active (only if still receiving negative adjustments)
    if current_physics_adj_f < -0.5:
        new_hours = min(state.hours_active + hours_since_update, 12)  # Cap at 12h
        new_accumulated = min(current_physics_adj_f, state.accumulated_adj_f)  # Track most negative
        new_locked = new_hours >= 3  # Lock after 3 hours
        
        save_cold_memory(
            station_id=station_id,
            target_date=target_date,
            accumulated_adj_f=new_accumulated,
            hours_active=new_hours,
            last_update_hour=current_hour,
            locked=new_locked
        )
        
        if new_locked:
            return new_accumulated, f"ColdMem({new_hours}h)"
        return current_physics_adj_f, ""
    
    # =========================================================================
    # Case 3: HRRR is trying to remove the cold (physics_adj >= -0.5)
    # =========================================================================
    if state.locked:
        # Apply 20%/hour decay from last known cold value
        decay_per_hour = 0.20
        hours_elapsed = max(1, hours_since_update)
        decay_factor = max(0.0, 1.0 - (decay_per_hour * hours_elapsed))
        
        # Calculate decayed cold adjustment
        decayed_adj = state.accumulated_adj_f * decay_factor
        
        # If decay would bring us above -0.3°F, release the lock
        if decayed_adj > -0.3:
            save_cold_memory(
                station_id=station_id,
                target_date=target_date,
                accumulated_adj_f=0.0,
                hours_active=0,
                last_update_hour=current_hour,
                locked=False
            )
            return current_physics_adj_f, "ColdMem(Released)"
        
        # Otherwise, maintain the decayed cold adjustment
        save_cold_memory(
            station_id=station_id,
            target_date=target_date,
            accumulated_adj_f=decayed_adj,
            hours_active=state.hours_active,
            last_update_hour=current_hour,
            locked=True
        )
        
        # Use the more negative of: current adjustment or decayed memory
        if decayed_adj < current_physics_adj_f:
            return round(decayed_adj, 1), f"ColdMem(Resist:{decayed_adj:.1f}F)"
        return current_physics_adj_f, ""
    
    # Not locked yet, just use current value
    return current_physics_adj_f, ""


# =============================================================================
# TESTING
# =============================================================================
if __name__ == "__main__":
    print("Testing Cold Memory Module...")
    print("=" * 60)
    
    # Simulate 5 hours of cold advection
    test_date = "2026-01-14"
    
    print("\nSimulating cold advection buildup:")
    for hour in range(10, 15):
        adj, note = apply_cold_memory("KLGA", test_date, -3.0, hour)
        print(f"  Hour {hour}: Physics=-3.0F -> Result={adj:.1f}F {note}")
    
    print("\nSimulating HRRR trying to remove cold:")
    for hour in range(15, 19):
        adj, note = apply_cold_memory("KLGA", test_date, 0.0, hour)
        print(f"  Hour {hour}: Physics=0.0F -> Result={adj:.1f}F {note}")
    
    print("\n" + "=" * 60)
    print("Cold Memory test complete.")
