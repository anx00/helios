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

from core.backtest import BacktestMode, CalibrationLoop, NowcastCalibrator

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
    # Multi-objective promotion thresholds
    min_val_pnl_net: float = -5.0       # Allow small losses
    max_val_drawdown: float = 0.30      # Max 30% drawdown
    max_val_turnover_ratio: float = 10.0  # Reasonable turnover cap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "train_days": self.train_days,
            "val_days": self.val_days,
            "max_combinations": self.max_combinations,
            "mode": self.mode,
            "min_val_score_for_promotion": self.min_val_score_for_promotion,
            "min_val_pnl_net": self.min_val_pnl_net,
            "max_val_drawdown": self.max_val_drawdown,
            "max_val_turnover_ratio": self.max_val_turnover_ratio,
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
        if run.status != "completed":
            reason = "learning_failed"
            promoted = False
        else:
            # Multi-objective promotion: must meet ALL thresholds
            rejections = []
            if run.best_val_score < cfg.min_val_score_for_promotion:
                rejections.append(f"val_score={run.best_val_score:.4f}<{cfg.min_val_score_for_promotion}")
            if run.val_pnl_net < cfg.min_val_pnl_net:
                rejections.append(f"val_pnl_net={run.val_pnl_net:.2f}<{cfg.min_val_pnl_net}")
            if run.val_max_drawdown > cfg.max_val_drawdown:
                rejections.append(f"val_drawdown={run.val_max_drawdown:.2f}>{cfg.max_val_drawdown}")
            if run.val_turnover_ratio > cfg.max_val_turnover_ratio:
                rejections.append(f"val_turnover={run.val_turnover_ratio:.2f}>{cfg.max_val_turnover_ratio}")

            promoted = len(rejections) == 0
            reason = "meets_thresholds" if promoted else f"below_thresholds({'; '.join(rejections)})"

        return PromotionDecision(
            version_id=version_id,
            station_id=station_id,
            promoted=promoted,
            reason=reason,
            metrics={
                "best_train_score": run.best_train_score,
                "best_val_score": run.best_val_score,
                "val_pnl_net": run.val_pnl_net,
                "val_max_drawdown": run.val_max_drawdown,
                "val_turnover_ratio": run.val_turnover_ratio,
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
            # --- Phase A: Nowcast parameter calibration ---
            nowcast_result = None
            try:
                nc = NowcastCalibrator()
                nowcast_result = nc.calibrate(
                    station_id=cfg.station_id,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    max_combinations=min(cfg.max_combinations, 200),
                )
            except Exception as nc_exc:
                run.notes += f"nowcast_calibration_skipped: {nc_exc}; "

            # --- Phase B: Policy/strategy calibration (grid search) ---
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
            run.val_pnl_net = result.best_val_pnl_net
            run.val_max_drawdown = result.best_val_max_drawdown
            # Compute turnover ratio: total_turnover / capital (use $100 default)
            capital = 100.0
            run.val_turnover_ratio = (
                result.best_val_total_turnover / capital
                if capital > 0 else 0.0
            )
            # Merge nowcast calibration results into best_params
            if nowcast_result and nowcast_result.best_params:
                run.best_params["nowcast"] = nowcast_result.best_params
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
        # Include prob calibrator params if fitted during grid search
        if run.status == "completed" and result.prob_calibrator_params:
            artifact["prob_calibrator"] = result.prob_calibrator_params
        # Include nowcast calibration results
        if run.status == "completed" and nowcast_result and nowcast_result.best_params:
            artifact["nowcast_calibration"] = nowcast_result.to_dict()
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
