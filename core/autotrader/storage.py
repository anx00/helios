"""
SQLite persistence for autotrader runtime and learning artifacts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import BanditState, LearningRunResult, PromotionDecision


class AutoTraderStorage:
    def __init__(self, db_path: str = "data/autotrader.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS autotrader_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    selected INTEGER NOT NULL,
                    context_json TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    reward REAL
                );

                CREATE TABLE IF NOT EXISTS autotrader_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    order_id TEXT,
                    strategy_name TEXT NOT NULL,
                    bucket TEXT,
                    side TEXT,
                    size REAL,
                    limit_price REAL,
                    order_type TEXT,
                    status TEXT,
                    raw_json TEXT
                );

                CREATE TABLE IF NOT EXISTS autotrader_fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    order_id TEXT,
                    bucket TEXT,
                    side TEXT,
                    size REAL,
                    price REAL,
                    fill_type TEXT,
                    fees REAL,
                    slippage REAL,
                    raw_json TEXT
                );

                CREATE TABLE IF NOT EXISTS autotrader_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    size REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS autotrader_rewards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    reward REAL NOT NULL,
                    pnl_component REAL,
                    drawdown_component REAL,
                    turnover_component REAL
                );

                CREATE TABLE IF NOT EXISTS bandit_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    strategy_scope TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS learning_runs (
                    run_id TEXT PRIMARY KEY,
                    ts_started_utc TEXT NOT NULL,
                    ts_completed_utc TEXT,
                    station_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_registry (
                    version_id TEXT PRIMARY KEY,
                    ts_created_utc TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    promotion_status TEXT NOT NULL,
                    notes TEXT
                );
                """
            )
            conn.commit()

    def record_decision(
        self,
        ts_utc: str,
        station_id: str,
        strategy_name: str,
        selected: bool,
        context: Dict[str, Any],
        decision: Dict[str, Any],
        reward: Optional[float],
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO autotrader_decisions
                (ts_utc, station_id, strategy_name, selected, context_json, decision_json, reward)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_utc,
                    station_id,
                    strategy_name,
                    1 if selected else 0,
                    json.dumps(context, default=str),
                    json.dumps(decision, default=str),
                    reward,
                ),
            )
            conn.commit()

    def record_order(self, order: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO autotrader_orders
                (ts_utc, station_id, order_id, strategy_name, bucket, side, size, limit_price, order_type, status, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.get("ts_utc"),
                    order.get("station_id"),
                    order.get("order_id"),
                    order.get("strategy_name"),
                    order.get("bucket"),
                    order.get("side"),
                    order.get("size"),
                    order.get("limit_price"),
                    order.get("order_type"),
                    order.get("status"),
                    json.dumps(order.get("raw", {}), default=str),
                ),
            )
            conn.commit()

    def record_fill(self, fill: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO autotrader_fills
                (ts_utc, station_id, strategy_name, order_id, bucket, side, size, price, fill_type, fees, slippage, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.get("ts_utc"),
                    fill.get("station_id"),
                    fill.get("strategy_name"),
                    fill.get("order_id"),
                    fill.get("bucket"),
                    fill.get("side"),
                    fill.get("size"),
                    fill.get("price"),
                    fill.get("fill_type"),
                    fill.get("fees"),
                    fill.get("slippage"),
                    json.dumps(fill.get("raw", {}), default=str),
                ),
            )
            conn.commit()

    def record_positions(
        self,
        ts_utc: str,
        station_id: str,
        realized_pnl: float,
        positions: Dict[str, Dict[str, float]],
    ) -> None:
        if not positions:
            return
        with self._lock, self._connect() as conn:
            for bucket, info in positions.items():
                conn.execute(
                    """
                    INSERT INTO autotrader_positions
                    (ts_utc, station_id, bucket, size, avg_price, unrealized_pnl, realized_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts_utc,
                        station_id,
                        bucket,
                        info.get("size", 0.0),
                        info.get("avg_price", 0.0),
                        info.get("unrealized_pnl", 0.0),
                        realized_pnl,
                    ),
                )
            conn.commit()

    def record_reward(
        self,
        ts_utc: str,
        station_id: str,
        strategy_name: str,
        reward: float,
        pnl_component: float,
        drawdown_component: float,
        turnover_component: float,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO autotrader_rewards
                (ts_utc, station_id, strategy_name, reward, pnl_component, drawdown_component, turnover_component)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_utc,
                    station_id,
                    strategy_name,
                    reward,
                    pnl_component,
                    drawdown_component,
                    turnover_component,
                ),
            )
            conn.commit()

    def save_bandit_state(self, scope: str, state: BanditState) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bandit_state (ts_utc, strategy_scope, state_json)
                VALUES (?, ?, ?)
                """,
                (
                    state.updated_at_utc.isoformat(),
                    scope,
                    json.dumps(state.to_dict(), default=str),
                ),
            )
            conn.commit()

    def load_latest_bandit_state(self, scope: str) -> Optional[BanditState]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT state_json FROM bandit_state
                WHERE strategy_scope = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (scope,),
            ).fetchone()
        if not row:
            return None
        data = json.loads(row["state_json"])
        return BanditState(
            version=data["version"],
            alpha=data["alpha"],
            feature_dim=data["feature_dim"],
            strategies=data["strategies"],
            state=data["state"],
            updated_at_utc=datetime.fromisoformat(data["updated_at_utc"]),
        )

    def add_learning_run(self, run: LearningRunResult, config: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO learning_runs
                (run_id, ts_started_utc, ts_completed_utc, station_id, status, config_json, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.ts_started_utc.isoformat(),
                    run.ts_completed_utc.isoformat() if run.ts_completed_utc else None,
                    run.station_id,
                    run.status,
                    json.dumps(config, default=str),
                    json.dumps(run.to_dict(), default=str),
                ),
            )
            conn.commit()

    def list_learning_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, ts_started_utc, ts_completed_utc, station_id, status, result_json
                FROM learning_runs
                ORDER BY ts_started_utc DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for row in rows:
            payload = json.loads(row["result_json"])
            out.append(payload)
        return out

    def add_model_registry_entry(
        self,
        version_id: str,
        ts_created_utc: str,
        station_id: str,
        artifact_path: str,
        metrics: Dict[str, Any],
        promotion_status: str,
        notes: str = "",
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_registry
                (version_id, ts_created_utc, station_id, artifact_path, metrics_json, promotion_status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    ts_created_utc,
                    station_id,
                    artifact_path,
                    json.dumps(metrics, default=str),
                    promotion_status,
                    notes,
                ),
            )
            conn.commit()

    def get_recent_decisions(self, limit: int = 200, station_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if station_id:
                rows = conn.execute(
                    """
                    SELECT ts_utc, station_id, strategy_name, selected, decision_json, reward
                    FROM autotrader_decisions
                    WHERE station_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (station_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT ts_utc, station_id, strategy_name, selected, decision_json, reward
                    FROM autotrader_decisions
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        out = []
        for row in rows:
            payload = json.loads(row["decision_json"])
            payload["ts_utc"] = row["ts_utc"]
            payload["station_id"] = row["station_id"]
            payload["strategy_name"] = row["strategy_name"]
            payload["selected"] = bool(row["selected"])
            payload["reward"] = row["reward"]
            out.append(payload)
        return out

    def get_recent_orders(self, limit: int = 200, station_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if station_id:
                rows = conn.execute(
                    """
                    SELECT ts_utc, station_id, order_id, strategy_name, bucket, side, size, limit_price, order_type, status
                    FROM autotrader_orders
                    WHERE station_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (station_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT ts_utc, station_id, order_id, strategy_name, bucket, side, size, limit_price, order_type, status
                    FROM autotrader_orders
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_recent_fills(self, limit: int = 200, station_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if station_id:
                rows = conn.execute(
                    """
                    SELECT ts_utc, station_id, strategy_name, order_id, bucket, side, size, price, fill_type, fees, slippage
                    FROM autotrader_fills
                    WHERE station_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (station_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT ts_utc, station_id, strategy_name, order_id, bucket, side, size, price, fill_type, fees, slippage
                    FROM autotrader_fills
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_latest_positions(self, station_id: str) -> Dict[str, Dict[str, float]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.bucket, p.size, p.avg_price, p.unrealized_pnl, p.realized_pnl
                FROM autotrader_positions p
                JOIN (
                    SELECT bucket, MAX(id) as max_id
                    FROM autotrader_positions
                    WHERE station_id = ?
                    GROUP BY bucket
                ) mx ON p.id = mx.max_id
                WHERE p.station_id = ?
                """,
                (station_id, station_id),
            ).fetchall()
        out: Dict[str, Dict[str, float]] = {}
        for row in rows:
            out[row["bucket"]] = {
                "size": float(row["size"]),
                "avg_price": float(row["avg_price"]),
                "unrealized_pnl": float(row["unrealized_pnl"]),
                "realized_pnl": float(row["realized_pnl"]),
            }
        return out
