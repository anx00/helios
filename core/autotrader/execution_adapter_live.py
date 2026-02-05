"""
Live execution adapter behind feature flag.

Default behavior is disabled (paper-first).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LiveExecutionConfig:
    enabled: bool = False
    mode: str = "paper"  # paper | semi_auto | live_auto


class LiveExecutionAdapter:
    def __init__(self, config: LiveExecutionConfig | None = None):
        self.config = config or LiveExecutionConfig(
            enabled=os.environ.get("HELIOS_LIVE_EXEC_ENABLED", "").strip().lower() in {"1", "true", "yes"},
            mode=os.environ.get("HELIOS_EXEC_MODE", "paper").strip().lower() or "paper",
        )

    def can_execute_live(self) -> bool:
        if not self.config.enabled:
            return False
        return self.config.mode in {"semi_auto", "live_auto"}

    def submit_order(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder hook for py_clob_client integration.
        """
        if not self.can_execute_live():
            return {
                "ok": False,
                "status": "disabled",
                "reason": "live_execution_feature_flag_off",
            }
        # Integration point:
        # - build py_clob_client order args
        # - sign and submit
        # - return broker response
        return {
            "ok": False,
            "status": "not_implemented",
            "reason": "live_adapter_stub",
        }
