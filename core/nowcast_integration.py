"""
Nowcast Integration Layer - Phase 3 (Section 3.7)

Bridges WorldState events to NowcastEngine updates.
Implements two update triggers:
- Trigger A: Periodic (every 60s)
- Trigger B: Event-driven (METAR, QC, HRRR)
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Callable, List
from zoneinfo import ZoneInfo

from core.world import WorldState, get_world
from core.models import OfficialObs, AuxObs, EnvFeature
from core.nowcast_engine import NowcastEngine, get_nowcast_engine
from core.nowcast_models import NowcastDistribution
from core.judge import JudgeAlignment

logger = logging.getLogger("nowcast_integration")

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


class NowcastIntegration:
    """
    Integration layer between WorldState and NowcastEngine.

    Responsibilities:
    - Listen to WorldState events and update NowcastEngine
    - Manage periodic updates
    - Provide unified access to nowcast distributions
    - Track update history for debugging
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NowcastIntegration, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        self._engines: Dict[str, NowcastEngine] = {}
        self._world: WorldState = get_world()
        self._running: bool = False
        self._periodic_task: Optional[asyncio.Task] = None
        self._sse_task: Optional[asyncio.Task] = None
        self._update_callbacks: List[Callable] = []
        self._metar_callbacks: List[Callable] = []  # Callbacks for METAR observations

        # Update history for debugging
        self._update_history: List[Dict] = []
        self._max_history = 100

        # Trigger counts
        self._trigger_counts = {
            "periodic": 0,
            "metar": 0,
            "aux": 0,
            "env": 0,
            "qc": 0,
            "hrrr": 0
        }

        self.initialized = True
        logger.info("NowcastIntegration initialized")

    def get_engine(self, station_id: str) -> NowcastEngine:
        """Get or create engine for station."""
        station_id = station_id.upper()
        if station_id not in self._engines:
            self._engines[station_id] = get_nowcast_engine(station_id)
        return self._engines[station_id]

    def get_distribution(self, station_id: str) -> Optional[NowcastDistribution]:
        """Get latest nowcast distribution for station."""
        engine = self.get_engine(station_id)
        return engine._last_distribution

    def get_all_distributions(self) -> Dict[str, NowcastDistribution]:
        """Get all current distributions."""
        return {
            sid: engine._last_distribution
            for sid, engine in self._engines.items()
            if engine._last_distribution
        }

    # =========================================================================
    # Event-Driven Updates (Trigger B)
    # =========================================================================

    def on_metar(self, obs: OfficialObs) -> NowcastDistribution:
        """
        Handle new METAR observation.

        This is the primary trigger for nowcast updates.
        """
        engine = self.get_engine(obs.station_id)

        # Get QC state from world
        qc_state = self._world.qc_states.get(obs.station_id)
        qc_status = qc_state.status if qc_state else "OK"

        # Update engine
        distribution = engine.update_observation(obs, qc_status)

        # Record update
        self._record_update("metar", obs.station_id, {
            "temp_f": obs.temp_f,
            "source": obs.source,
            "qc_status": qc_status
        })

        self._trigger_counts["metar"] += 1

        # Notify METAR callbacks (for recording)
        self._notify_metar_callbacks(obs)

        # Notify nowcast callbacks
        self._notify_callbacks(obs.station_id, distribution)

        return distribution

    def on_aux(self, aux: AuxObs):
        """Handle auxiliary observation (PWS, upstream)."""
        # Extract base station from aux station_id
        if "UPSTREAM_" in aux.station_id:
            station_id = aux.station_id.replace("UPSTREAM_", "")
        elif "PWS_" in aux.station_id:
            station_id = aux.station_id.replace("PWS_", "")
        else:
            station_id = aux.station_id

        engine = self.get_engine(station_id)
        engine.update_aux_observation(aux)

        self._record_update("aux", station_id, {
            "aux_station": aux.station_id,
            "temp_f": aux.temp_f,
            "drift": aux.drift
        })

        self._trigger_counts["aux"] += 1

    def on_env_feature(self, feature: EnvFeature):
        """Handle environment feature update (SST, AOD)."""
        # Apply to all active engines
        for station_id, engine in self._engines.items():
            engine.update_env_feature(feature)

        self._record_update("env", "ALL", {
            "type": feature.feature_type,
            "value": feature.value,
            "status": feature.status
        })

        self._trigger_counts["env"] += 1

    def on_qc_change(self, station_id: str, status: str, flags: List[str]):
        """Handle QC state change."""
        engine = self.get_engine(station_id)
        state = engine.get_or_create_daily_state()
        state.last_qc_state = status

        # Regenerate distribution with new QC state
        distribution = engine.generate_distribution()

        self._record_update("qc", station_id, {
            "status": status,
            "flags": flags
        })

        self._trigger_counts["qc"] += 1

        self._notify_callbacks(station_id, distribution)

        return distribution

    def on_hrrr_update(
        self,
        station_id: str,
        hourly_temps_f: List[float],
        model_run_time: datetime,
        model_source: str = "HRRR",
        features: Optional[Dict] = None,
        target_date: Optional[date] = None
    ) -> NowcastDistribution:
        """
        Handle new HRRR/ensemble forecast.

        Updates the Base Forecast layer.
        """
        engine = self.get_engine(station_id)

        # Update base forecast
        engine.update_base_forecast(
            hourly_temps_f=hourly_temps_f,
            model_run_time=model_run_time,
            model_source=model_source,
            target_date=target_date,
            additional_features=features
        )

        # Regenerate distribution
        distribution = engine.generate_distribution()

        self._record_update("hrrr", station_id, {
            "source": model_source,
            "t_max_base_f": engine._base_forecast.t_max_base_f,
            "t_peak_hour": engine._base_forecast.t_peak_hour
        })

        self._trigger_counts["hrrr"] += 1

        self._notify_callbacks(station_id, distribution)

        return distribution

    # =========================================================================
    # Periodic Updates (Trigger A)
    # =========================================================================

    async def _periodic_update_loop(self, interval_seconds: float = 60.0):
        """
        Background task for periodic updates.

        Recalculates distributions every interval without fetching new data.
        """
        logger.info(f"Starting periodic update loop (interval: {interval_seconds}s)")

        while self._running:
            try:
                await asyncio.sleep(interval_seconds)

                if not self._running:
                    break

                # Update all engines
                for station_id, engine in self._engines.items():
                    try:
                        distribution = engine.periodic_update()
                        if distribution:
                            self._notify_callbacks(station_id, distribution)
                    except Exception as e:
                        logger.warning(f"Periodic update failed for {station_id}: {e}")

                self._trigger_counts["periodic"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic update loop error: {e}")

    async def _sse_listener_loop(self):
        """
        Listen to WorldState SSE queue for event-driven updates.
        """
        logger.info("Starting SSE listener loop")

        queue = self._world.subscribe_sse()

        try:
            while self._running:
                try:
                    # Wait for next event with timeout
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=5.0
                    )

                    # Process event based on type
                    event_type = event.get("type")
                    data = event.get("data", {})

                    if event_type == "official":
                        # Reconstruct OfficialObs from data
                        obs = self._reconstruct_official_obs(data)
                        if obs:
                            self.on_metar(obs)

                    elif event_type == "aux":
                        aux = self._reconstruct_aux_obs(data)
                        if aux:
                            self.on_aux(aux)

                    elif event_type == "env":
                        feature = self._reconstruct_env_feature(data)
                        if feature:
                            self.on_env_feature(feature)

                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

        finally:
            self._world.unsubscribe_sse(queue)

    def _reconstruct_official_obs(self, data: Dict) -> Optional[OfficialObs]:
        """Reconstruct OfficialObs from serialized data."""
        try:
            return OfficialObs(
                obs_time_utc=datetime.fromisoformat(data["obs_time_utc"]),
                ingest_time_utc=datetime.fromisoformat(data["ingest_time_utc"]),
                source=data.get("source", "UNKNOWN"),
                station_id=data.get("station_id", "UNKNOWN"),
                temp_c=data.get("temp_c", 0.0),
                temp_f=data.get("temp_f", 0.0),
                temp_f_raw=data.get("temp_f_raw", data.get("temp_f", 0.0)),
                dewpoint_c=data.get("dewpoint_c"),
                wind_dir=data.get("wind_dir"),
                wind_speed=data.get("wind_speed"),
                sky_condition=data.get("sky_condition", "CLR"),
                qc_passed=data.get("qc_passed", False),
                qc_flags=data.get("qc_flags", [])
            )
        except Exception as e:
            logger.warning(f"Failed to reconstruct OfficialObs: {e}")
            return None

    def _reconstruct_aux_obs(self, data: Dict) -> Optional[AuxObs]:
        """Reconstruct AuxObs from serialized data."""
        try:
            return AuxObs(
                obs_time_utc=datetime.fromisoformat(data["obs_time_utc"]),
                ingest_time_utc=datetime.fromisoformat(data["ingest_time_utc"]),
                source=data.get("source", "UNKNOWN"),
                station_id=data.get("station_id", "UNKNOWN"),
                temp_c=data.get("temp_c", 0.0),
                temp_f=data.get("temp_f", 0.0),
                drift=data.get("drift", 0.0),
                support=data.get("support", 1),
                is_aggregate=data.get("is_aggregate", False)
            )
        except Exception as e:
            logger.warning(f"Failed to reconstruct AuxObs: {e}")
            return None

    def _reconstruct_env_feature(self, data: Dict) -> Optional[EnvFeature]:
        """Reconstruct EnvFeature from serialized data."""
        try:
            return EnvFeature(
                obs_time_utc=datetime.fromisoformat(data["obs_time_utc"]),
                ingest_time_utc=datetime.fromisoformat(data["ingest_time_utc"]),
                source=data.get("source", "UNKNOWN"),
                feature_type=data.get("feature_type", "SST"),
                location_ref=data.get("location_ref", ""),
                value=data.get("value", 0.0),
                unit=data.get("unit", ""),
                status=data.get("status", "OK")
            )
        except Exception as e:
            logger.warning(f"Failed to reconstruct EnvFeature: {e}")
            return None

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self, periodic_interval: float = 60.0, listen_sse: bool = True):
        """Start the integration service."""
        if self._running:
            return

        self._running = True

        # Start periodic update task
        self._periodic_task = asyncio.create_task(
            self._periodic_update_loop(periodic_interval)
        )

        # Start SSE listener task
        if listen_sse:
            self._sse_task = asyncio.create_task(
                self._sse_listener_loop()
            )

        logger.info("NowcastIntegration started")

    async def stop(self):
        """Stop the integration service."""
        self._running = False

        if self._periodic_task:
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass

        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass

        logger.info("NowcastIntegration stopped")

    # =========================================================================
    # Callbacks & Notifications
    # =========================================================================

    def register_callback(self, callback: Callable[[str, NowcastDistribution], None]):
        """Register a callback for distribution updates."""
        self._update_callbacks.append(callback)

    def register_metar_callback(self, callback: Callable[[OfficialObs], None]):
        """Register a callback for METAR observations."""
        self._metar_callbacks.append(callback)

    def unregister_callback(self, callback: Callable):
        """Unregister a callback."""
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)

    def _notify_callbacks(self, station_id: str, distribution: NowcastDistribution):
        """Notify all registered callbacks."""
        for callback in self._update_callbacks:
            try:
                callback(station_id, distribution)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def _notify_metar_callbacks(self, obs: OfficialObs):
        """Notify all registered METAR callbacks."""
        for callback in self._metar_callbacks:
            try:
                callback(obs)
            except Exception as e:
                logger.warning(f"METAR callback error: {e}")

    # =========================================================================
    # Debugging & History
    # =========================================================================

    def _record_update(self, trigger: str, station_id: str, details: Dict):
        """Record update to history."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "trigger": trigger,
            "station_id": station_id,
            "details": details
        }

        self._update_history.append(entry)

        # Trim history
        if len(self._update_history) > self._max_history:
            self._update_history = self._update_history[-self._max_history:]

    def get_status(self) -> Dict:
        """Get integration status for debugging."""
        return {
            "running": self._running,
            "engines": list(self._engines.keys()),
            "trigger_counts": self._trigger_counts.copy(),
            "recent_updates": self._update_history[-10:],
            "callbacks_registered": len(self._update_callbacks)
        }

    def get_snapshot(self) -> Dict:
        """
        Get full nowcast snapshot for all stations.

        Suitable for API response.
        """
        distributions = {}
        states = {}

        for station_id, engine in self._engines.items():
            if engine._last_distribution:
                distributions[station_id] = engine._last_distribution.to_dict()

            state = engine.get_or_create_daily_state()
            states[station_id] = engine.get_state_snapshot()

        return {
            "ts_generated_utc": datetime.now(UTC).isoformat(),
            "distributions": distributions,
            "states": states,
            "trigger_counts": self._trigger_counts.copy()
        }


# =============================================================================
# Helper: Hourly Temps Extraction for BaseForecast
# =============================================================================

def extract_hourly_temps_f_for_date(
    hourly_temps_c: List[float],
    hourly_times: Optional[List[str]],
    target_date: date,
    station_tz: ZoneInfo
) -> List[float]:
    """
    Build a 24h temperature list in °F for a specific local date.
    Falls back to the first 24 values if timestamps are missing.
    """
    if not hourly_temps_c:
        return []

    if not hourly_times or len(hourly_times) != len(hourly_temps_c):
        return [round((t * 9/5) + 32, 1) for t in hourly_temps_c[:24] if t is not None]

    daily = []
    for temp_c, ts in zip(hourly_temps_c, hourly_times):
        if temp_c is None or not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=station_tz)
            else:
                dt = dt.astimezone(station_tz)
        except Exception:
            continue
        if dt.date() == target_date:
            daily.append(round((temp_c * 9/5) + 32, 1))

    if len(daily) >= 24:
        return daily[:24]

    return [round((t * 9/5) + 32, 1) for t in hourly_temps_c[:24] if t is not None]

# =============================================================================
# Global Instance Accessor
# =============================================================================

_integration: Optional[NowcastIntegration] = None


def get_nowcast_integration() -> NowcastIntegration:
    """Get the global NowcastIntegration instance."""
    global _integration
    if _integration is None:
        _integration = NowcastIntegration()
    return _integration


async def initialize_nowcast_integration(
    stations: List[str] = None,
    periodic_interval: float = 60.0
):
    """
    Initialize and start the nowcast integration.

    Should be called at application startup.
    """
    integration = get_nowcast_integration()

    # Pre-create engines for specified stations
    if stations:
        for station_id in stations:
            integration.get_engine(station_id)

    # Start background tasks
    await integration.start(periodic_interval)

    return integration
