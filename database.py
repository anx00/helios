"""
Helios Weather Lab - Database Module (Physics Engine)
SQLite database for physics-based weather predictions.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Iterator, List
from contextlib import contextmanager

from config import DATABASE_PATH


# ============================================================================
# DATABASE SCHEMA (Physics Engine - Snapshot Format)
# ============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    station_id TEXT NOT NULL,
    
    -- Raw inputs
    temp_actual_f REAL,
    temp_hrrr_hourly_f REAL,
    soil_moisture REAL,
    radiation REAL,
    sky_condition TEXT,
    wind_dir INTEGER,
    
    -- Physics calculation
    hrrr_max_raw_f REAL,
    current_deviation_f REAL,
    delta_weight REAL,
    physics_adjustment_f REAL,
    physics_reason TEXT,
    final_prediction_f REAL,
    
    -- Verification (filled next day)
    real_max_verified_f REAL,
    verified_at DATETIME,
    
    -- Metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    target_date TEXT, -- v4.1: ISO Format YYYY-MM-DD
    
    -- v5.3: Extended Physics Variables
    velocity_ratio REAL,
    aod_550nm REAL,
    sst_delta_c REAL,
    advection_adj_c REAL,
    verified_floor_f REAL
);

CREATE INDEX IF NOT EXISTS idx_predictions_station_timestamp 
ON predictions(station_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_predictions_verified 
ON predictions(verified_at);

-- Trajectory tracking for deviation engine
CREATE TABLE IF NOT EXISTS model_path (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    forecast_date DATE NOT NULL,
    hour INTEGER NOT NULL,
    predicted_temp_c REAL NOT NULL,
    predicted_temp_f REAL NOT NULL,
    captured_at DATETIME NOT NULL,
    
    UNIQUE(station_id, forecast_date, hour)
);

CREATE INDEX IF NOT EXISTS idx_model_path_lookup
ON model_path(station_id, forecast_date, hour);

-- Performance Monitoring & Validation (v4.0 - Arbitrage Dashboard)
CREATE TABLE IF NOT EXISTS performance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,      -- UTC time of snapshot
    station_id TEXT NOT NULL,
    
    -- Snapshot values
    helios_pred REAL,                 -- Our prediction (final_f)
    metar_actual REAL,                -- Official METAR temp
    wu_preliminary REAL,              -- WU Max from history
    
    -- v4.0: Market Data (Top 3 Polymarket brackets)
    market_top1_bracket TEXT,         -- e.g., "48-49"
    market_top1_prob REAL,            -- e.g., 0.45
    market_top2_bracket TEXT,
    market_top2_prob REAL,
    market_top3_bracket TEXT,
    market_top3_prob REAL,
    market_weighted_avg REAL,         -- Weighted avg of top 3 midpoints
    cumulative_max_f REAL,            -- Maximum temp observed so far today
    wu_forecast_high REAL,            -- Wunderground predicted peak for the day
    
    -- v5.2: Prediction Components (Debug)
    hrrr_max_raw_f REAL,
    current_deviation_f REAL,
    physics_adjustment_f REAL,
    soil_moisture REAL,
    radiation REAL,
    sky_condition TEXT,
    wind_dir INTEGER,
    
    -- Reconciliation (filled later)
    wu_final REAL,                    -- WU History verified (after 4h)
    
    -- Metadata
    confidence_score REAL,            -- Prediction delta_weight
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- v5.3: Extended Physics Variables
    velocity_ratio REAL,
    aod_550nm REAL,
    sst_delta_c REAL,
    advection_adj_c REAL,
    verified_floor_f REAL,
    physics_reason TEXT,
    
    market_all_brackets_json TEXT,  -- JSON array of {bracket, prob}
    target_date TEXT,  -- v4.1: ISO Format YYYY-MM-DD
    
    -- v8.0: Racing METAR Sources (Extra Data)
    metar_json_api_f REAL,
    metar_tds_xml_f REAL,
    metar_tgftp_txt_f REAL
);

CREATE INDEX IF NOT EXISTS idx_perf_station_time
ON performance_logs(station_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_perf_pending_reconciliation
ON performance_logs(timestamp) WHERE wu_final IS NULL;

-- v6.8: Market Alerts Registry (State-based persistent alerts)
CREATE TABLE IF NOT EXISTS market_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    station_id TEXT NOT NULL,
    target_date TEXT NOT NULL,
    alert_type TEXT NOT NULL,         -- 'bull_trap', 'recovery', 'divergence'
    level TEXT NOT NULL,              -- 'warning', 'critical', 'info'
    title TEXT NOT NULL,
    description TEXT,
    diff_value REAL,
    wu_value REAL,
    metar_value REAL
);

CREATE INDEX IF NOT EXISTS idx_alerts_station_date
ON market_alerts(station_id, target_date);

-- v7.5: Market Velocity Tracking (Crowd Wisdom)
CREATE TABLE IF NOT EXISTS market_velocity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    station_id TEXT NOT NULL,
    target_date TEXT NOT NULL,
    bracket_name TEXT NOT NULL,
    probability REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_market_velocity_lookup
ON market_velocity(station_id, target_date, bracket_name, timestamp);

-- v8.4: Telegram METAR/SPECI push subscriptions (per station, controlled from /world)
CREATE TABLE IF NOT EXISTS telegram_station_subscriptions (
    station_id TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 0,
    updated_at DATETIME NOT NULL,
    updated_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_telegram_station_subscriptions_enabled
ON telegram_station_subscriptions(enabled);

-- Probability Lab: day-ahead forecast source snapshots
CREATE TABLE IF NOT EXISTS forecast_source_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at_utc DATETIME NOT NULL,
    station_id TEXT NOT NULL,
    target_date TEXT NOT NULL,
    target_day INTEGER NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    market_unit TEXT NOT NULL,
    forecast_high_market REAL,
    forecast_high_f REAL,
    forecast_high_c REAL,
    peak_hour_local INTEGER,
    provider_updated_local TEXT,
    notes_json TEXT,
    meta_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_forecast_source_snapshots_station_target
ON forecast_source_snapshots(station_id, target_date, source, captured_at_utc);

CREATE INDEX IF NOT EXISTS idx_forecast_source_snapshots_target_date
ON forecast_source_snapshots(target_date, captured_at_utc);

-- Probability Lab: latest resolved calibration state per station
CREATE TABLE IF NOT EXISTS probability_lab_calibration_state (
    station_id TEXT PRIMARY KEY,
    computed_at_utc DATETIME NOT NULL,
    available INTEGER NOT NULL DEFAULT 0,
    warming_up INTEGER NOT NULL DEFAULT 1,
    samples INTEGER NOT NULL DEFAULT 0,
    mae_market REAL,
    bias_market REAL,
    rows_json TEXT,
    sources_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_probability_lab_calibration_state_computed
ON probability_lab_calibration_state(computed_at_utc);

"""


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _auto_migrate(conn):
    """Auto-migrate: add any columns defined in SCHEMA but missing from the real DB."""
    # Build expected schema in a temporary in-memory DB
    tmp = sqlite3.connect(":memory:")
    tmp.executescript(SCHEMA)

    # Get all tables from the expected schema
    tmp_tables = {row[0] for row in tmp.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    migrated = 0
    for table in tmp_tables:
        # Expected columns with types from temp DB
        expected = {row[1]: row[2] for row in tmp.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()}

        # Actual columns in real DB
        actual = {row[1] for row in conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()}

        # Add missing columns
        for col_name, col_type in expected.items():
            if col_name not in actual:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                print(f"  [MIGRATE] {table}.{col_name} ({col_type})")
                migrated += 1

    tmp.close()
    if migrated:
        print(f"  [MIGRATE] Added {migrated} missing column(s)")


def init_database():
    """Initialize the database schema and auto-migrate missing columns."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        _auto_migrate(conn)
    print(f"[OK] Database initialized: {DATABASE_PATH}")


# ============================================================================
# PREDICTION OPERATIONS
# ============================================================================

def insert_prediction(data: Dict[str, Any]) -> int:
    """
    Insert a physics prediction snapshot.
    
    Args:
        data: Dictionary with prediction data
        
    Returns:
        The ID of the inserted record
    """
    columns = [
        "timestamp", "station_id",
        "temp_actual_f", "temp_hrrr_hourly_f",
        "soil_moisture", "radiation", "sky_condition", "wind_dir",
        "hrrr_max_raw_f", "current_deviation_f", "delta_weight",
        "physics_adjustment_f", "physics_reason", "final_prediction_f",
        "target_date",
        "velocity_ratio", "aod_550nm", "sst_delta_c", "advection_adj_c", "verified_floor_f"
    ]
    
    placeholders = ", ".join(["?" for _ in columns])
    column_names = ", ".join(columns)
    
    values = [data.get(col) for col in columns]
    
    with get_connection() as conn:
        cursor = conn.execute(
            f"INSERT INTO predictions ({column_names}) VALUES ({placeholders})",
            values
        )
        return cursor.lastrowid


def get_unverified_predictions(station_id: str, date: datetime) -> List[Dict]:
    """Get all unverified predictions for a station on a specific date."""
    date_str = date.strftime("%Y-%m-%d")
    
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM predictions 
            WHERE station_id = ? 
            AND DATE(timestamp) = ?
            AND real_max_verified_f IS NULL
            ORDER BY timestamp DESC
            """,
            (station_id, date_str)
        )
        return [dict(row) for row in cursor.fetchall()]


def update_verification(
    prediction_id: int,
    real_max_f: float,
) -> None:
    """Update a prediction with the verified actual maximum."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE predictions 
            SET real_max_verified_f = ?,
                verified_at = ?
            WHERE id = ?
            """,
            (real_max_f, datetime.now().isoformat(), prediction_id)
        )


def get_accuracy_report(days: int = 7) -> Dict[str, Any]:
    """Generate accuracy report for verified predictions."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT 
                station_id,
                COUNT(*) as total_predictions,
                AVG(ABS(final_prediction_f - real_max_verified_f)) as avg_error,
                AVG(ABS(hrrr_max_raw_f - real_max_verified_f)) as avg_hrrr_error,
                MIN(ABS(final_prediction_f - real_max_verified_f)) as best_prediction,
                MAX(ABS(final_prediction_f - real_max_verified_f)) as worst_prediction
            FROM predictions
            WHERE real_max_verified_f IS NOT NULL
            AND verified_at >= datetime('now', ?)
            GROUP BY station_id
            """,
            (f'-{days} days',)
        )
        
        results = {}
        for row in cursor.fetchall():
            results[row['station_id']] = {
                'total_predictions': row['total_predictions'],
                'avg_physics_error_f': row['avg_error'],
                'avg_hrrr_error_f': row['avg_hrrr_error'],
                'best_prediction_f': row['best_prediction'],
                'worst_prediction_f': row['worst_prediction'],
            }
        
        return results


# ============================================================================
# TRAJECTORY OPERATIONS (for deviation engine)
# ============================================================================

def store_model_path(
    station_id: str,
    forecast_date: str,
    hourly_temps: List[float],
    hourly_times: List[str],
) -> int:
    """Store hourly temperature trajectory for a day."""
    inserted = 0
    
    with get_connection() as conn:
        for i, (temp_c, time_str) in enumerate(zip(hourly_temps, hourly_times)):
            if temp_c is None:
                continue
                
            try:
                hour = int(time_str.split("T")[1].split(":")[0])
                temp_f = round((temp_c * 9/5) + 32, 1)
                
                conn.execute(
                    """
                    INSERT OR REPLACE INTO model_path 
                    (station_id, forecast_date, hour, predicted_temp_c, predicted_temp_f, captured_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (station_id, forecast_date, hour, temp_c, temp_f, datetime.now().isoformat())
                )
                inserted += 1
            except (IndexError, ValueError):
                continue
    
    return inserted


def get_predicted_temp_for_hour(
    station_id: str,
    forecast_date: str,
    hour: int
) -> Optional[Dict]:
    """Get the predicted temperature for a specific hour."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM model_path 
            WHERE station_id = ? 
            AND forecast_date = ? 
            AND hour = ?
            """,
            (station_id, forecast_date, hour)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_trajectory_for_station(station_id: str, forecast_date: Any) -> Optional[List[Dict]]:
    """
    Get hourly trajectory for a station on a specific date.
    
    Returns list of dicts with hour_datetime and temp_c.
    """
    date_str = str(forecast_date)
    
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT hour, predicted_temp_c
            FROM model_path
            WHERE station_id = ?
              AND forecast_date = ?
            ORDER BY hour
            """,
            (station_id, date_str)
        )
        
        rows = cursor.fetchall()
        if not rows:
            return None
        
        trajectory = []
        for row in rows:
            # Reconstruct datetime
            from datetime import date as dt_date, time as dt_time
            if isinstance(forecast_date, str):
                y, m, d = map(int, forecast_date.split('-'))
                base_date = dt_date(y, m, d)
            else:
                base_date = forecast_date
                
            hour_dt = datetime.combine(base_date, dt_time(hour=row['hour']))
            trajectory.append({
                'hour_datetime': hour_dt,
                'temp_c': row['predicted_temp_c']
            })

def get_latest_prediction(station_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent prediction for a station.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM predictions 
            WHERE station_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
            """,
            (station_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_latest_prediction_for_date(station_id: str, target_date: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent prediction for a specific target date.
    target_date should be YYYY-MM-DD.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM predictions 
            WHERE station_id = ? 
            AND target_date = ?
            ORDER BY timestamp DESC 
            LIMIT 1
            """,
            (station_id, target_date)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def insert_forecast_source_snapshot(
    *,
    station_id: str,
    target_date: str,
    target_day: int,
    source: str,
    status: str,
    market_unit: str,
    forecast_high_market: Optional[float] = None,
    forecast_high_f: Optional[float] = None,
    forecast_high_c: Optional[float] = None,
    peak_hour_local: Optional[int] = None,
    provider_updated_local: Optional[str] = None,
    notes: Optional[List[Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    captured_at_utc: Optional[str] = None,
) -> Optional[int]:
    """Insert a day-ahead forecast source snapshot unless the latest row is effectively identical."""
    captured_at = str(captured_at_utc or datetime.now(timezone.utc).isoformat())
    latest = get_latest_forecast_source_snapshot(station_id, target_date, source)

    def _rounded_tenth(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return round(float(value), 1)
        except Exception:
            return None

    def _parse_iso(value: Any) -> Optional[datetime]:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None

    if latest:
        latest_captured = _parse_iso(latest.get("captured_at_utc"))
        this_captured = _parse_iso(captured_at)
        if latest_captured and this_captured:
            age = abs((this_captured - latest_captured).total_seconds())
            latest_status = str(latest.get("status") or "").lower()
            next_status = str(status or "").lower()
            if (
                age <= timedelta(hours=12).total_seconds()
                and latest_status in {"ok", "partial"}
                and next_status == "error"
                and _rounded_tenth(latest.get("forecast_high_market")) is not None
                and _rounded_tenth(forecast_high_market) is None
            ):
                return None
            if age <= timedelta(minutes=20).total_seconds():
                if (
                    str(latest.get("status") or "") == str(status or "")
                    and _rounded_tenth(latest.get("forecast_high_market")) == _rounded_tenth(forecast_high_market)
                    and latest.get("peak_hour_local") == peak_hour_local
                ):
                    return None

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO forecast_source_snapshots (
                captured_at_utc,
                station_id,
                target_date,
                target_day,
                source,
                status,
                market_unit,
                forecast_high_market,
                forecast_high_f,
                forecast_high_c,
                peak_hour_local,
                provider_updated_local,
                notes_json,
                meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                captured_at,
                station_id,
                target_date,
                int(target_day),
                str(source or "").upper(),
                str(status or ""),
                str(market_unit or "").upper(),
                forecast_high_market,
                forecast_high_f,
                forecast_high_c,
                peak_hour_local,
                provider_updated_local,
                json.dumps(list(notes or []), ensure_ascii=True),
                json.dumps(dict(meta or {}), ensure_ascii=True),
            ),
        )
        return cursor.lastrowid


def _decode_forecast_source_snapshot(row: sqlite3.Row) -> Dict[str, Any]:
    payload = dict(row)
    try:
        payload["notes"] = json.loads(payload.pop("notes_json") or "[]")
    except Exception:
        payload["notes"] = []
    try:
        payload["meta"] = json.loads(payload.pop("meta_json") or "{}")
    except Exception:
        payload["meta"] = {}
    return payload


def get_latest_forecast_source_snapshot(
    station_id: str,
    target_date: str,
    source: str,
) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT *
            FROM forecast_source_snapshots
            WHERE station_id = ?
              AND target_date = ?
              AND source = ?
            ORDER BY captured_at_utc DESC, id DESC
            LIMIT 1
            """,
            (station_id, target_date, str(source or "").upper()),
        )
        row = cursor.fetchone()
        return _decode_forecast_source_snapshot(row) if row else None


def get_latest_forecast_source_snapshots(
    station_id: str,
    target_date: str,
) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT fss.*
            FROM forecast_source_snapshots AS fss
            JOIN (
                SELECT source, MAX(captured_at_utc) AS max_captured_at_utc
                FROM forecast_source_snapshots
                WHERE station_id = ?
                  AND target_date = ?
                GROUP BY source
            ) AS latest
              ON latest.source = fss.source
             AND latest.max_captured_at_utc = fss.captured_at_utc
            WHERE fss.station_id = ?
              AND fss.target_date = ?
            ORDER BY fss.source ASC, fss.id DESC
            """,
            (station_id, target_date, station_id, target_date),
        )
        rows = cursor.fetchall()
        deduped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            payload = _decode_forecast_source_snapshot(row)
            source = str(payload.get("source") or "").upper()
            if source and source not in deduped:
                deduped[source] = payload
        return list(deduped.values())


def get_forecast_source_history(
    station_id: str,
    target_date: str,
    lookback_hours: int = 36,
) -> List[Dict[str, Any]]:
    hours = max(1, int(lookback_hours))
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT *
            FROM forecast_source_snapshots
            WHERE station_id = ?
              AND target_date = ?
              AND captured_at_utc >= datetime('now', ?)
            ORDER BY captured_at_utc ASC, source ASC, id ASC
            """,
            (station_id, target_date, f"-{hours} hours"),
        )
        return [_decode_forecast_source_snapshot(row) for row in cursor.fetchall()]


def get_recent_forecast_target_dates(
    station_id: str,
    *,
    before_date: Optional[str] = None,
    limit: int = 7,
) -> List[str]:
    resolved_limit = max(1, int(limit))
    params: List[Any] = [station_id]
    where = ["station_id = ?"]
    if before_date:
        where.append("target_date < ?")
        params.append(before_date)

    with get_connection() as conn:
        cursor = conn.execute(
            f"""
            SELECT target_date
            FROM forecast_source_snapshots
            WHERE {' AND '.join(where)}
            GROUP BY target_date
            ORDER BY target_date DESC
            LIMIT ?
            """,
            (*params, resolved_limit),
        )
        return [str(row["target_date"]) for row in cursor.fetchall()]


def _decode_probability_lab_calibration_state(row: sqlite3.Row) -> Dict[str, Any]:
    payload = dict(row)
    payload["available"] = bool(payload.get("available"))
    payload["warming_up"] = bool(payload.get("warming_up"))
    payload["samples"] = int(payload.get("samples") or 0)
    try:
        payload["rows"] = json.loads(payload.pop("rows_json") or "[]")
    except Exception:
        payload["rows"] = []
    try:
        payload["sources"] = json.loads(payload.pop("sources_json") or "{}")
    except Exception:
        payload["sources"] = {}
    return payload


def upsert_probability_lab_calibration_state(
    station_id: str,
    payload: Dict[str, Any],
    *,
    computed_at_utc: Optional[str] = None,
) -> None:
    resolved_payload = dict(payload or {})
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO probability_lab_calibration_state (
                station_id,
                computed_at_utc,
                available,
                warming_up,
                samples,
                mae_market,
                bias_market,
                rows_json,
                sources_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(station_id) DO UPDATE SET
                computed_at_utc = excluded.computed_at_utc,
                available = excluded.available,
                warming_up = excluded.warming_up,
                samples = excluded.samples,
                mae_market = excluded.mae_market,
                bias_market = excluded.bias_market,
                rows_json = excluded.rows_json,
                sources_json = excluded.sources_json
            """,
            (
                str(station_id or "").upper(),
                str(computed_at_utc or datetime.now(timezone.utc).isoformat()),
                1 if resolved_payload.get("available") else 0,
                1 if resolved_payload.get("warming_up") else 0,
                int(resolved_payload.get("samples") or 0),
                resolved_payload.get("mae_market"),
                resolved_payload.get("bias_market"),
                json.dumps(list(resolved_payload.get("rows") or []), ensure_ascii=True),
                json.dumps(dict(resolved_payload.get("sources") or {}), ensure_ascii=True),
            ),
        )


def get_probability_lab_calibration_state(station_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT *
            FROM probability_lab_calibration_state
            WHERE station_id = ?
            LIMIT 1
            """,
            (str(station_id or "").upper(),),
        )
        row = cursor.fetchone()
        return _decode_probability_lab_calibration_state(row) if row else None



# ============================================================================
# PERFORMANCE MONITORING OPERATIONS
# ============================================================================

def insert_performance_log(
    station_id: str,
    helios_pred: float,
    metar_actual: Optional[float],
    wu_preliminary: Optional[float],
    confidence_score: float,
    # v4.0: Market data
    market_top1_bracket: Optional[str] = None,
    market_top1_prob: Optional[float] = None,
    market_top2_bracket: Optional[str] = None,
    market_top2_prob: Optional[float] = None,
    market_top3_bracket: Optional[str] = None,
    market_top3_prob: Optional[float] = None,
    market_weighted_avg: Optional[float] = None,
    cumulative_max_f: Optional[float] = None,
    target_date: Optional[str] = None, # v4.1: ISO Format YYYY-MM-DD
    wu_forecast_high: Optional[float] = None, # v5.1
    # v5.2: Prediction Components
    hrrr_max_raw_f: Optional[float] = None,
    current_deviation_f: Optional[float] = None,
    physics_adjustment_f: Optional[float] = None,
    soil_moisture: Optional[float] = None,
    radiation: Optional[float] = None,
    sky_condition: Optional[str] = None,
    wind_dir: Optional[int] = None,
    # v5.3: Extended Physics
    velocity_ratio: Optional[float] = None,
    aod_550nm: Optional[float] = None,
    sst_delta_c: Optional[float] = None,
    advection_adj_c: Optional[float] = None,
    verified_floor_f: Optional[float] = None,
    physics_reason: Optional[str] = None,
    # v6.0: All market options
    market_all_brackets_json: Optional[str] = None,
    # v8.0: Racing METAR Sources
    metar_json_api_f: Optional[float] = None,
    metar_tds_xml_f: Optional[float] = None,
    metar_tgftp_txt_f: Optional[float] = None,
    # v10.0: Multi-Model Ensemble
    nbm_max_f: Optional[float] = None,
    lamp_max_f: Optional[float] = None,
    lamp_confidence: Optional[float] = None,
    ensemble_base_f: Optional[float] = None,
    ensemble_floor_f: Optional[float] = None,
    ensemble_ceiling_f: Optional[float] = None,
    ensemble_spread_f: Optional[float] = None,
    ensemble_confidence: Optional[str] = None,
    hrrr_outlier_detected: Optional[bool] = None,
) -> int:
    """Insert a new performance snapshot with market data."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO performance_logs 
            (timestamp, station_id, helios_pred, metar_actual, wu_preliminary, confidence_score,
             market_top1_bracket, market_top1_prob, market_top2_bracket, market_top2_prob,
             market_top3_bracket, market_top3_prob, market_weighted_avg, cumulative_max_f, target_date, wu_forecast_high,
             hrrr_max_raw_f, current_deviation_f, physics_adjustment_f, soil_moisture, radiation, sky_condition, wind_dir,
             velocity_ratio, aod_550nm, sst_delta_c, advection_adj_c, verified_floor_f, physics_reason, market_all_brackets_json,
             metar_json_api_f, metar_tds_xml_f, metar_tgftp_txt_f,
             nbm_max_f, lamp_max_f, lamp_confidence, ensemble_base_f, ensemble_floor_f, ensemble_ceiling_f, ensemble_spread_f, ensemble_confidence, hrrr_outlier_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (datetime.utcnow().isoformat(), station_id, helios_pred, metar_actual, wu_preliminary, confidence_score,
             market_top1_bracket, market_top1_prob, market_top2_bracket, market_top2_prob,
             market_top3_bracket, market_top3_prob, market_weighted_avg, cumulative_max_f, target_date, wu_forecast_high,
             hrrr_max_raw_f, current_deviation_f, physics_adjustment_f, soil_moisture, radiation, sky_condition, wind_dir,
             velocity_ratio, aod_550nm, sst_delta_c, advection_adj_c, verified_floor_f, physics_reason, market_all_brackets_json,
             metar_json_api_f, metar_tds_xml_f, metar_tgftp_txt_f,
             nbm_max_f, lamp_max_f, lamp_confidence, ensemble_base_f, ensemble_floor_f, ensemble_ceiling_f, ensemble_spread_f, ensemble_confidence, 1 if hrrr_outlier_detected else 0 if hrrr_outlier_detected is not None else None)
        )
        return cursor.lastrowid




def update_performance_wu_final(log_id: int, wu_final: float) -> None:
    """Update a log entry with the reconciled final WU history value."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE performance_logs SET wu_final = ? WHERE id = ?",
            (wu_final, log_id)
        )


def get_performance_history(station_id: str, limit_hours: int = 24) -> List[Dict]:
    """Get performance logs for the dashboard."""
    with get_connection() as conn:
        cursor = conn.execute(
            f"""
            SELECT * FROM performance_logs
            WHERE station_id = ?
            AND timestamp >= datetime('now', '-{limit_hours} hours')
            ORDER BY timestamp DESC
            """,
            (station_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_pending_reconciliation_logs(lookback_hours: int = 4) -> List[Dict]:
    """Get logs that need WU reconciliation (older than X hours but missing final)."""
    with get_connection() as conn:
        cursor = conn.execute(
            f"""
            SELECT * FROM performance_logs
            WHERE wu_final IS NULL
            AND timestamp <= datetime('now', '-{lookback_hours} hours')
            AND timestamp >= datetime('now', '-24 hours')
            ORDER BY timestamp ASC
            """,
        )
        return [dict(row) for row in cursor.fetchall()]


def get_performance_history_by_date(station_id: str, date_str: str) -> List[Dict]:
    """Get performance logs for a specific date (in NYC timezone)."""
    # date_str is in format YYYY-MM-DD
    # We need to convert NYC date to UTC range
    from zoneinfo import ZoneInfo
    
    nyc_tz = ZoneInfo("America/New_York")
    
    # Parse the date and create start/end in NYC time
    year, month, day = map(int, date_str.split('-'))
    start_nyc = datetime(year, month, day, 0, 0, 0, tzinfo=nyc_tz)
    end_nyc = datetime(year, month, day, 23, 59, 59, tzinfo=nyc_tz)
    
    # Convert to UTC for database query
    start_utc = start_nyc.astimezone(ZoneInfo('UTC')).isoformat()
    end_utc = end_nyc.astimezone(ZoneInfo('UTC')).isoformat()
    
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM performance_logs
            WHERE station_id = ?
            AND timestamp >= ?
            AND timestamp <= ?
            ORDER BY timestamp DESC
            """,
            (station_id, start_utc, end_utc)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_performance_history_by_target_date(station_id: str, target_date: str) -> List[Dict]:
    """
    Get performance logs filtered by the PREDICTION target date.
    Used for showing the evolution of a specific day's forecast.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM performance_logs
            WHERE station_id = ?
            AND target_date = ?
            ORDER BY timestamp DESC
            """,
            (station_id, target_date)
        )
        return [dict(row) for row in cursor.fetchall()]


def iter_performance_history_by_target_date(
    station_id: str,
    target_date: str,
    since_utc: Optional[str] = None,
    ascending: bool = False,
    batch_size: int = 1000,
) -> Iterator[Dict[str, Any]]:
    """
    Stream performance logs for a target date.
    Useful for large exports/series without building a full list in memory.
    """
    order = "ASC" if ascending else "DESC"
    safe_batch = max(1, int(batch_size))

    query = f"""
        SELECT * FROM performance_logs
        WHERE station_id = ?
        AND target_date = ?
        {"AND timestamp >= ?" if since_utc else ""}
        ORDER BY timestamp {order}, id {order}
    """
    params: List[Any] = [station_id, target_date]
    if since_utc:
        params.append(since_utc)

    with get_connection() as conn:
        cursor = conn.execute(query, tuple(params))
        while True:
            rows = cursor.fetchmany(safe_batch)
            if not rows:
                break
            for row in rows:
                yield dict(row)


def get_observed_max_for_target_date(station_id: str, target_date: str) -> Optional[float]:
    """
    Return best observed max for a target date, preferring cumulative max when available.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT MAX(
                CASE
                    WHEN cumulative_max_f IS NOT NULL THEN cumulative_max_f
                    ELSE metar_actual
                END
            ) AS observed_max_f
            FROM performance_logs
            WHERE station_id = ?
            AND target_date = ?
            """,
            (station_id, target_date),
        )
        row = cursor.fetchone()
        if not row:
            return None
        value = row["observed_max_f"]
        return float(value) if value is not None else None


def get_latest_performance_log(station_id: str) -> Optional[Dict]:
    """Get the very latest performance log for a station."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM performance_logs
            WHERE station_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (station_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


# ============================================================================
# MARKET ALERTS OPERATIONS (v6.8)
# ============================================================================

def insert_market_alert(
    station_id: str,
    target_date: str,
    alert_type: str,
    level: str,
    title: str,
    description: str,
    diff_value: float,
    wu_value: Optional[float] = None,
    metar_value: Optional[float] = None
) -> int:
    """Insert a new market alert into the registry."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO market_alerts 
            (timestamp, station_id, target_date, alert_type, level, title, description, diff_value, wu_value, metar_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (datetime.utcnow().isoformat(), station_id, target_date, alert_type, level, title, description, diff_value, wu_value, metar_value)
        )
        return cursor.lastrowid


def get_alerts_for_date(station_id: str, target_date: str) -> List[Dict]:
    """Get all alerts for a specific target date, ordered by timestamp DESC."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM market_alerts
            WHERE station_id = ?
            AND target_date = ?
            ORDER BY timestamp DESC
            """,
            (station_id, target_date)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_latest_alert_type(station_id: str, target_date: str) -> Optional[str]:
    """Get the most recent alert type for a station/date to track state."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT alert_type FROM market_alerts
            WHERE station_id = ?
            AND target_date = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (station_id, target_date)
        )
        row = cursor.fetchone()
        return row["alert_type"] if row else None


# ============================================================================
# TELEGRAM METAR/SPECI SUBSCRIPTIONS (v8.4)
# ============================================================================

def get_telegram_station_subscription(station_id: str) -> Dict[str, Any]:
    """Get subscription state for a station (default disabled if no row exists)."""
    sid = str(station_id or "").upper().strip()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT station_id, enabled, updated_at, updated_by
            FROM telegram_station_subscriptions
            WHERE station_id = ?
            """,
            (sid,),
        )
        row = cursor.fetchone()
        if not row:
            return {
                "station_id": sid,
                "enabled": False,
                "updated_at": None,
                "updated_by": None,
            }
        return {
            "station_id": row["station_id"],
            "enabled": bool(row["enabled"]),
            "updated_at": row["updated_at"],
            "updated_by": row["updated_by"],
        }


def set_telegram_station_subscription(
    station_id: str,
    enabled: bool,
    updated_by: Optional[str] = None,
) -> Dict[str, Any]:
    """Create/update subscription state for a station."""
    sid = str(station_id or "").upper().strip()
    ts = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO telegram_station_subscriptions (station_id, enabled, updated_at, updated_by)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(station_id) DO UPDATE SET
                enabled = excluded.enabled,
                updated_at = excluded.updated_at,
                updated_by = excluded.updated_by
            """,
            (sid, 1 if enabled else 0, ts, updated_by),
        )
    return get_telegram_station_subscription(sid)


def is_telegram_station_subscription_enabled(station_id: str) -> bool:
    """Fast helper used by the Telegram bot listener."""
    sid = str(station_id or "").upper().strip()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT enabled
            FROM telegram_station_subscriptions
            WHERE station_id = ?
            LIMIT 1
            """,
            (sid,),
        )
        row = cursor.fetchone()
        return bool(row["enabled"]) if row else False


def list_enabled_telegram_station_subscriptions() -> List[str]:
    """Return all station IDs with subscription enabled."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT station_id
            FROM telegram_station_subscriptions
            WHERE enabled = 1
            ORDER BY station_id ASC
            """
        )
        return [str(row["station_id"]).upper() for row in cursor.fetchall()]
