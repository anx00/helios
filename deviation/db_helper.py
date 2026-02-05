"""
Get prediction errors from database for historical trend analysis
"""
import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).parent / "helios_weather.db"

def get_last_n_prediction_errors(station_id: str, n: int = 3) -> list[float]:
    """
    Get the prediction errors (delta) from the last N days.
    
    Used for calculating historical trend in temporal decay.
    
    Args:
        station_id: Station identifier
        n: Number of recent days to retrieve
        
    Returns:
        List of deltas (actual - predicted), empty if insufficient data
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check if predictions table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='predictions'
        """)
        
        if not cursor.fetchone():
            # Table doesn't exist yet, return empty
            cursor.close()
            conn.close()
            return []
        
        # Get verified predictions with actual temps
        cursor.execute("""
            SELECT (temp_actual_f - physics_pred_f) as delta
            FROM predictions
            WHERE station_id = ?
              AND verified = 1
              AND temp_actual_f IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        """, (station_id, n))
        
        rows = cursor.fetchall()
        errors = [row[0] for row in rows]
        
        cursor.close()
        conn.close()
        return errors
        
    except Exception as e:
        # Silently return empty on any error (no historical data yet)
        return []


if __name__ == "__main__":
    # Test the function
    errors = get_last_n_prediction_errors("KLGA", 3)
    print(f"Last 3 errors for KLGA: {errors}")
