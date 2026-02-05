
import sqlite3
from config import DATABASE_PATH as DB_PATH

def migrate():
    print(f"Migrating database at {DB_PATH}...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    columns_to_add = [
        ("market_top1_bracket", "TEXT"),
        ("market_top1_prob", "REAL"),
        ("market_top2_bracket", "TEXT"),
        ("market_top2_prob", "REAL"),
        ("market_top3_bracket", "TEXT"),
        ("market_top3_prob", "REAL"),
        ("market_weighted_avg", "REAL"),
        ("cumulative_max_f", "REAL"),
        ("target_date", "DATE")  # v4.1: Track which day the prediction is for
    ]
    
    for col_name, col_type in columns_to_add:
        try:
            print(f"Adding column {col_name}...")
            cursor.execute(f"ALTER TABLE performance_logs ADD COLUMN {col_name} {col_type}")
            print("  Done.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"  Column {col_name} already exists. Skipping.")
            else:
                print(f"  Error adding {col_name}: {e}")
                
    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
