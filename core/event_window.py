"""
Event Window Manager - Phase 4 (Section 4.4)

Manages "event windows" - periods of higher granularity capture
triggered by informative events.

Triggers:
- World-driven: METAR arrival, QC outlier
- Market-driven: Mid shock, spread regime change, depth cliff

During windows, we capture more granular data (deltas, ticks).
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Callable, Any
from zoneinfo import ZoneInfo

logger = logging.getLogger("event_window")

UTC = ZoneInfo("UTC")


class WindowTrigger(Enum):
    """Types of event window triggers."""
    METAR = "METAR"                      # New METAR observation
    QC_OUTLIER = "QC_OUTLIER"            # QC flagged outlier
    QC_CHANGE = "QC_CHANGE"              # QC state changed
    MID_SHOCK = "MID_SHOCK"              # Mid price shock
    SPREAD_REGIME = "SPREAD_REGIME"      # Spread regime change
    DEPTH_CLIFF = "DEPTH_CLIFF"          # Order book depth drop
    HRRR_UPDATE = "HRRR_UPDATE"          # New HRRR model run
    MANUAL = "MANUAL"                    # Manual trigger


@dataclass
class EventWindow:
    """
    Represents an active or closed event window.

    Attributes:
        window_id: Unique identifier for correlation
        trigger: What triggered this window
        reason: Human-readable description
        start_ts: When window opened
        end_ts: When window closed (None if still open)
        station_id: Associated station (if applicable)
        metadata: Additional context
    """
    window_id: str
    trigger: WindowTrigger
    reason: str
    start_ts: datetime
    end_ts: Optional[datetime] = None
    station_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.end_ts is None

    @property
    def duration_seconds(self) -> float:
        end = self.end_ts or datetime.now(UTC)
        return (end - self.start_ts).total_seconds()

    def to_dict(self) -> Dict:
        return {
            "window_id": self.window_id,
            "trigger": self.trigger.value,
            "reason": self.reason,
            "start_ts": self.start_ts.isoformat(),
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "station_id": self.station_id,
            "duration_seconds": self.duration_seconds,
            "is_active": self.is_active,
            "metadata": self.metadata
        }


@dataclass
class WindowConfig:
    """Configuration for event window durations."""
    metar_duration_seconds: float = 120.0       # 2 minutes after METAR
    qc_outlier_duration_seconds: float = 180.0  # 3 minutes after QC outlier
    qc_change_duration_seconds: float = 60.0    # 1 minute after QC change
    mid_shock_duration_seconds: float = 60.0    # 1 minute after mid shock
    spread_regime_duration_seconds: float = 120.0
    depth_cliff_duration_seconds: float = 60.0
    hrrr_duration_seconds: float = 300.0        # 5 minutes after HRRR
    manual_duration_seconds: float = 60.0

    # Thresholds for market triggers
    mid_shock_threshold_ticks: int = 5          # Ticks to trigger shock
    spread_wide_threshold: float = 0.05         # 5% spread = wide
    depth_cliff_min_levels: int = 3             # Min levels before cliff

    def get_duration(self, trigger: WindowTrigger) -> float:
        mapping = {
            WindowTrigger.METAR: self.metar_duration_seconds,
            WindowTrigger.QC_OUTLIER: self.qc_outlier_duration_seconds,
            WindowTrigger.QC_CHANGE: self.qc_change_duration_seconds,
            WindowTrigger.MID_SHOCK: self.mid_shock_duration_seconds,
            WindowTrigger.SPREAD_REGIME: self.spread_regime_duration_seconds,
            WindowTrigger.DEPTH_CLIFF: self.depth_cliff_duration_seconds,
            WindowTrigger.HRRR_UPDATE: self.hrrr_duration_seconds,
            WindowTrigger.MANUAL: self.manual_duration_seconds
        }
        return mapping.get(trigger, 60.0)


class EventWindowManager:
    """
    Manages event windows lifecycle.

    Responsibilities:
    - Open windows on triggers
    - Track active windows
    - Auto-close after duration
    - Notify listeners on open/close
    - Provide current window for correlation
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EventWindowManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, config: Optional[WindowConfig] = None):
        if self.initialized:
            return

        self.config = config or WindowConfig()

        # Active windows
        self._active_windows: Dict[str, EventWindow] = {}

        # Window history (last 100)
        self._history: List[EventWindow] = []
        self._max_history = 100

        # Callbacks
        self._on_open_callbacks: List[Callable[[EventWindow], None]] = []
        self._on_close_callbacks: List[Callable[[EventWindow], None]] = []

        # Background tasks for auto-close
        self._close_tasks: Dict[str, asyncio.Task] = {}

        # Stats
        self._total_windows = 0
        self._windows_by_trigger: Dict[str, int] = {t.value: 0 for t in WindowTrigger}

        self.initialized = True
        logger.info("EventWindowManager initialized")

    # =========================================================================
    # Window Lifecycle
    # =========================================================================

    async def open_window(
        self,
        trigger: WindowTrigger,
        reason: str,
        station_id: Optional[str] = None,
        duration_override: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> EventWindow:
        """
        Open a new event window.

        Args:
            trigger: What triggered this window
            reason: Human-readable description
            station_id: Associated station
            duration_override: Override default duration
            metadata: Additional context

        Returns:
            The opened EventWindow
        """
        window_id = str(uuid.uuid4())[:8]  # Short ID for readability
        now = datetime.now(UTC)

        window = EventWindow(
            window_id=window_id,
            trigger=trigger,
            reason=reason,
            start_ts=now,
            station_id=station_id,
            metadata=metadata or {}
        )

        self._active_windows[window_id] = window
        self._total_windows += 1
        self._windows_by_trigger[trigger.value] += 1

        logger.info(f"Window opened: {window_id} ({trigger.value}) - {reason}")

        # Notify callbacks
        for callback in self._on_open_callbacks:
            try:
                callback(window)
            except Exception as e:
                logger.warning(f"Window open callback error: {e}")

        # Schedule auto-close
        duration = duration_override or self.config.get_duration(trigger)
        self._close_tasks[window_id] = asyncio.create_task(
            self._auto_close(window_id, duration)
        )

        return window

    async def _auto_close(self, window_id: str, duration: float):
        """Auto-close window after duration."""
        try:
            await asyncio.sleep(duration)
            await self.close_window(window_id)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Auto-close error for {window_id}: {e}")

    async def close_window(self, window_id: str) -> Optional[EventWindow]:
        """
        Close an active window.

        Returns:
            The closed window, or None if not found
        """
        if window_id not in self._active_windows:
            return None

        window = self._active_windows.pop(window_id)
        window.end_ts = datetime.now(UTC)

        # Cancel auto-close task if still pending
        if window_id in self._close_tasks:
            self._close_tasks[window_id].cancel()
            del self._close_tasks[window_id]

        # Add to history
        self._history.append(window)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.info(
            f"Window closed: {window_id} ({window.trigger.value}) "
            f"duration={window.duration_seconds:.1f}s"
        )

        # Notify callbacks
        for callback in self._on_close_callbacks:
            try:
                callback(window)
            except Exception as e:
                logger.warning(f"Window close callback error: {e}")

        return window

    async def close_all(self):
        """Close all active windows."""
        window_ids = list(self._active_windows.keys())
        for window_id in window_ids:
            await self.close_window(window_id)

    # =========================================================================
    # Trigger Methods (convenience)
    # =========================================================================

    async def on_metar(
        self,
        station_id: str,
        temp_f: float,
        source: str = "METAR"
    ) -> EventWindow:
        """Trigger window for new METAR."""
        return await self.open_window(
            trigger=WindowTrigger.METAR,
            reason=f"New {source} observation: {temp_f:.1f}°F",
            station_id=station_id,
            metadata={"temp_f": temp_f, "source": source}
        )

    async def on_qc_outlier(
        self,
        station_id: str,
        flags: List[str]
    ) -> EventWindow:
        """Trigger window for QC outlier."""
        return await self.open_window(
            trigger=WindowTrigger.QC_OUTLIER,
            reason=f"QC outlier detected: {', '.join(flags)}",
            station_id=station_id,
            metadata={"flags": flags}
        )

    async def on_qc_change(
        self,
        station_id: str,
        old_state: str,
        new_state: str
    ) -> EventWindow:
        """Trigger window for QC state change."""
        return await self.open_window(
            trigger=WindowTrigger.QC_CHANGE,
            reason=f"QC changed: {old_state} -> {new_state}",
            station_id=station_id,
            metadata={"old_state": old_state, "new_state": new_state}
        )

    async def on_mid_shock(
        self,
        market_id: str,
        old_mid: float,
        new_mid: float,
        delta_ticks: int,
        station_id: Optional[str] = None,
    ) -> EventWindow:
        """Trigger window for mid price shock."""
        return await self.open_window(
            trigger=WindowTrigger.MID_SHOCK,
            reason=f"Mid shock: {old_mid:.4f} -> {new_mid:.4f} ({delta_ticks} ticks)",
            station_id=station_id,
            metadata={
                "market_id": market_id,
                "old_mid": old_mid,
                "new_mid": new_mid,
                "delta_ticks": delta_ticks
            }
        )

    async def on_spread_regime_change(
        self,
        market_id: str,
        old_regime: str,
        new_regime: str,
        spread: float,
        station_id: Optional[str] = None,
    ) -> EventWindow:
        """Trigger window for spread regime change."""
        return await self.open_window(
            trigger=WindowTrigger.SPREAD_REGIME,
            reason=f"Spread regime: {old_regime} -> {new_regime} ({spread:.2%})",
            station_id=station_id,
            metadata={
                "market_id": market_id,
                "old_regime": old_regime,
                "new_regime": new_regime,
                "spread": spread
            }
        )

    async def on_depth_cliff(
        self,
        market_id: str,
        old_depth: float,
        new_depth: float,
        station_id: Optional[str] = None,
    ) -> EventWindow:
        """Trigger window for abrupt depth collapse."""
        return await self.open_window(
            trigger=WindowTrigger.DEPTH_CLIFF,
            reason=f"Depth cliff: {old_depth:.1f} -> {new_depth:.1f}",
            station_id=station_id,
            metadata={
                "market_id": market_id,
                "old_depth": old_depth,
                "new_depth": new_depth,
            },
        )

    async def on_hrrr_update(
        self,
        station_id: str,
        model_run: str,
        t_max_f: float
    ) -> EventWindow:
        """Trigger window for new HRRR run."""
        return await self.open_window(
            trigger=WindowTrigger.HRRR_UPDATE,
            reason=f"New HRRR run: {model_run}, Tmax={t_max_f:.1f}°F",
            station_id=station_id,
            metadata={"model_run": model_run, "t_max_f": t_max_f}
        )

    # =========================================================================
    # Query Methods
    # =========================================================================

    def has_active_window(self, station_id: Optional[str] = None) -> bool:
        """Check if there's an active window."""
        if station_id:
            return any(
                w.station_id == station_id
                for w in self._active_windows.values()
            )
        return len(self._active_windows) > 0

    def get_active_windows(
        self,
        station_id: Optional[str] = None
    ) -> List[EventWindow]:
        """Get active windows, optionally filtered by station."""
        windows = list(self._active_windows.values())
        if station_id:
            windows = [w for w in windows if w.station_id == station_id]
        return windows

    def get_current_window_id(
        self,
        station_id: Optional[str] = None
    ) -> Optional[str]:
        """Get current window ID for correlation."""
        windows = self.get_active_windows(station_id)
        if windows:
            # Return most recent
            return sorted(windows, key=lambda w: w.start_ts, reverse=True)[0].window_id
        return None

    def get_history(
        self,
        station_id: Optional[str] = None,
        limit: int = 50
    ) -> List[EventWindow]:
        """Get window history."""
        history = self._history
        if station_id:
            history = [w for w in history if w.station_id == station_id]
        return history[-limit:]

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_window_open(self, callback: Callable[[EventWindow], None]):
        """Register callback for window open."""
        self._on_open_callbacks.append(callback)

    def on_window_close(self, callback: Callable[[EventWindow], None]):
        """Register callback for window close."""
        self._on_close_callbacks.append(callback)

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get window manager statistics."""
        return {
            "active_windows": len(self._active_windows),
            "total_windows": self._total_windows,
            "by_trigger": self._windows_by_trigger.copy(),
            "history_size": len(self._history),
            "active": [w.to_dict() for w in self._active_windows.values()]
        }


# =============================================================================
# Global accessor
# =============================================================================

_manager: Optional[EventWindowManager] = None


def get_event_window_manager() -> EventWindowManager:
    """Get the global EventWindowManager instance."""
    global _manager
    if _manager is None:
        _manager = EventWindowManager()
    return _manager
