
import sqlite3
from datetime import datetime
from config import DATABASE_PATH
from zoneinfo import ZoneInfo

def backfill():
    print(f"Backfilling target_date in {DATABASE_PATH}...")
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all logs without target_date
    cursor.execute("SELECT id, timestamp, station_id FROM performance_logs WHERE target_date IS NULL")
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} logs to backfill.")
    
    updates = []
    
    for row in rows:
        # Assume existing logs targeted "Today" relative to their creation time
        # Convert UTC timestamp to Station Local Time (NYC for KLGA)
        ts_utc = datetime.fromisoformat(row['timestamp'])
        
        # Hardcoded for KLGA (NYC) as simple backfill
        nyc_tz = ZoneInfo("America/New_York")
        ts_nyc = ts_utc.astimezone(nyc_tz)
        target_date = ts_nyc.date().isoformat()
        
        updates.append((target_date, row['id']))
        
    if updates:
        cursor.executemany("UPDATE performance_logs SET target_date = ? WHERE id = ?", updates)
        conn.commit()
        print(f"Updated {len(updates)} rows.")
    
    conn.close()

if __name__ == "__main__":
    backfill()
