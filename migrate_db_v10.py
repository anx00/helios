"""
HELIOS v10.0 Database Migration
Adds NBM/LAMP ensemble columns to performance_logs table.
"""
import sqlite3
from config import DATABASE_PATH

def migrate():
    print("=" * 60)
    print("HELIOS v10.0 - Database Migration")
    print("Adding ensemble columns to performance_logs...")
    print("=" * 60)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get existing columns
    cursor.execute("PRAGMA table_info(performance_logs)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    
    # New columns for v10.0 ensemble
    new_columns = [
        ("nbm_max_f", "REAL"),
        ("lamp_max_f", "REAL"),
        ("lamp_confidence", "REAL"),
        ("ensemble_base_f", "REAL"),
        ("ensemble_floor_f", "REAL"),
        ("ensemble_ceiling_f", "REAL"),
        ("ensemble_spread_f", "REAL"),
        ("ensemble_confidence", "TEXT"),
        ("hrrr_outlier_detected", "INTEGER"),
    ]
    
    added = 0
    for col_name, col_type in new_columns:
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE performance_logs ADD COLUMN {col_name} {col_type}")
                print(f"  + Added column: {col_name} ({col_type})")
                added += 1
            except sqlite3.OperationalError as e:
                print(f"  ! Error adding {col_name}: {e}")
        else:
            print(f"  - Column exists: {col_name}")
    
    conn.commit()
    conn.close()
    
    print("=" * 60)
    print(f"Migration complete. Added {added} new columns.")
    print("=" * 60)

if __name__ == "__main__":
    migrate()
