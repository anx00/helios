import sqlite3
import os
from config import DATABASE_PATH

def migrate():
    print(f"Migrating database: {DATABASE_PATH}")
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # 1. Add target_date to predictions table
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN target_date TEXT")
        print("✓ Column 'target_date' added to 'predictions' table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("! Column 'target_date' already exists in 'predictions' table.")
        else:
            print(f"✗ Error adding column to 'predictions': {e}")

    # 2. Add target_date to performance_logs table (double check)
    try:
        cursor.execute("ALTER TABLE performance_logs ADD COLUMN target_date TEXT")
        print("✓ Column 'target_date' added to 'performance_logs' table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("! Column 'target_date' already exists in 'performance_logs' table.")
        else:
            print(f"✗ Error adding column to 'performance_logs': {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
