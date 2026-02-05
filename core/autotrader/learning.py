"""
Offline learning cycle:
- walk-forward-like train/val split
- calibration loop
- model registry artifact + promotion decision
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from core.backtest import BacktestMode, CalibrationLoop

from .models import LearningRunResult, PromotionDecision
from .storage import AutoTraderStorage

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


@dataclass
class LearningConfig:
    station_id: str = "KLGA"
    train_days: int = 53  # D-60..D-8
    val_days: int = 7     # D-7..D-1
    max_combinations: int = 50
    mode: str = "signal_only"
    min_val_score_for_promotion: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "train_days": self.train_days,
            "val_days": self.val_days,
            "max_combinations": self.max_combinations,
            "mode": self.mode,
            "min_val_score_for_promotion": self.min_val_score_for_promotion,
        }


class LearningRunner:
    def __init__(
        self,
        storage: Optional[AutoTraderStorage] = None,
        registry_dir: str = "data/model_registry",
    ):
        self.storage = storage or AutoTraderStorage()
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def _build_windows(self, cfg: LearningConfig) -> Tuple[datetime.date, datetime.date, datetime.date, datetime.date]:
        today_nyc = datetime.now(NYC).date()
        val_end = today_nyc - timedelta(days=1)
        val_start = val_end - timedelta(days=cfg.val_days - 1)
        train_end = val_start - timedelta(days=1)
        train_start = train_end - timedelta(days=cfg.train_days - 1)
        return train_start, train_end, val_start, val_end

    def _promotion_decision(
        self,
        version_id: str,
        station_id: str,
        run: LearningRunResult,
        cfg: LearningConfig,
    ) -> PromotionDecision:
        promoted = run.status == "completed" and run.best_val_score >= cfg.min_val_score_for_promotion
        if run.status != "completed":
            reason = "learning_failed"
        elif promoted:
            reason = "meets_thresholds"
        else:
            reason = "below_thresholds"
        return PromotionDecision(
            version_id=version_id,
            station_id=station_id,
            promoted=promoted,
            reason=reason,
            metrics={
                "best_train_score": run.best_train_score,
                "best_val_score": run.best_val_score,
            },
            ts_utc=datetime.now(UTC),
        )

    def run_once(self, config: Optional[LearningConfig] = None) -> Dict[str, Any]:
        cfg = config or LearningConfig()
        run_id = uuid.uuid4().hex[:12]
        run = LearningRunResult(
            run_id=run_id,
            station_id=cfg.station_id,
            ts_started_utc=datetime.now(UTC),
            status="running",
        )
        self.storage.add_learning_run(run, cfg.to_dict())

        train_start, train_end, val_start, val_end = self._build_windows(cfg)
        run.train_start = train_start.isoformat()
        run.train_end = train_end.isoformat()
        run.val_start = val_start.isoformat()
        run.val_end = val_end.isoformat()

        try:
            mode = BacktestMode.SIGNAL_ONLY if cfg.mode == "signal_only" else BacktestMode.EXECUTION_AWARE
            loop = CalibrationLoop(mode=mode)
            result = loop.grid_search(
                station_id=cfg.station_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                max_combinations=cfg.max_combinations,
            )
            run.best_params = result.best_params
            run.best_train_score = result.best_train_score
            run.best_val_score = result.best_val_score
            run.status = "completed"
            run.ts_completed_utc = datetime.now(UTC)
        except Exception as exc:
            run.status = "failed"
            run.notes = str(exc)
            run.ts_completed_utc = datetime.now(UTC)

        version_id = f"{cfg.station_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        artifact = {
            "version_id": version_id,
            "config": cfg.to_dict(),
            "result": run.to_dict(),
        }
        artifact_path = self.registry_dir / f"{version_id}.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)

        promo = self._promotion_decision(version_id, cfg.station_id, run, cfg)

        self.storage.add_learning_run(run, cfg.to_dict())
        self.storage.add_model_registry_entry(
            version_id=version_id,
            ts_created_utc=datetime.now(UTC).isoformat(),
            station_id=cfg.station_id,
            artifact_path=str(artifact_path),
            metrics=promo.metrics,
            promotion_status="promoted" if promo.promoted else "rejected",
            notes=promo.reason,
        )

        return {
            "learning_run": run.to_dict(),
            "promotion": promo.to_dict(),
            "artifact_path": str(artifact_path),
        }
