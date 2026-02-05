"""
Evidence Builder - Deterministic Data Access Layer for Atenea

Fetches and formats evidence from HELIOS subsystems:
- Market: orderbook, spreads, depth
- World: METAR, PWS, QC, features
- Nowcast: predictions, drivers, distributions
- Health: staleness, latencies, gaps
- Backtest: runs, metrics, coverage

Evidence is always traceable with IDs and timestamps.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger("atenea.evidence")

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


class EvidenceType(Enum):
    """Types of evidence that can be collected."""
    MARKET = "market"           # L2 snapshots, spreads, mid
    WORLD = "world"             # METAR, observations
    PWS = "pws"                 # PWS cluster consensus
    QC = "qc"                   # Quality control state
    FEATURES = "features"       # Environmental features (SST, AOD)
    NOWCAST = "nowcast"         # Nowcast outputs
    HEALTH = "health"           # System health metrics
    BACKTEST = "backtest"       # Backtest results
    EVENT_WINDOW = "event"      # Event windows
    REPLAY = "replay"           # Replay state


@dataclass
class Evidence:
    """
    A single piece of traceable evidence.

    Every evidence has:
    - Unique ID (E1, E2, etc.)
    - Type (market, world, nowcast, etc.)
    - Timestamp (when the data was valid)
    - Source (where it came from)
    - Summary (human-readable summary)
    - Data (the actual payload, for detailed inspection)
    """
    id: str                     # E1, E2, etc.
    type: EvidenceType
    timestamp_utc: datetime
    source: str                 # e.g., "METAR/KLGA", "L2_SNAP", "NOWCAST_1M"
    summary: str                # Human-readable summary
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/LLM."""
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "timestamp_nyc": self.timestamp_utc.astimezone(NYC).strftime("%Y-%m-%d %H:%M:%S NYC"),
            "source": self.source,
            "summary": self.summary,
            "event_id": self.event_id,
            "data": self.data
        }

    def format_citation(self) -> str:
        """Format as citation for LLM response."""
        ts_nyc = self.timestamp_utc.astimezone(NYC).strftime("%H:%M:%S NYC")
        return f"`{self.id}` ({self.type.value}): {self.source} @ {ts_nyc} — {self.summary}"


class EvidenceBuilder:
    """
    Builds evidence from HELIOS data sources.

    This is the deterministic "arm" that Atenea uses to fetch data.
    The LLM never directly queries data - this layer does it first.
    """

    def __init__(self):
        self._evidence_counter = 0
        # Live state references (injected from web_server)
        self._market_state = None
        self._world_state = None
        self._nowcast_state = None
        self._health_state = None

    def set_live_state(
        self,
        market_state=None,
        world_state=None,
        nowcast_state=None,
        health_state=None
    ):
        """Inject live state references for evidence gathering."""
        if market_state is not None:
            self._market_state = market_state
        if world_state is not None:
            self._world_state = world_state
        if nowcast_state is not None:
            self._nowcast_state = nowcast_state
        if health_state is not None:
            self._health_state = health_state

    def reset_counter(self):
        """Reset evidence ID counter for new query."""
        self._evidence_counter = 0

    def _next_id(self) -> str:
        """Get next evidence ID."""
        self._evidence_counter += 1
        return f"E{self._evidence_counter}"

    # =========================================================================
    # Live State Access
    # =========================================================================

    def get_market_evidence(
        self,
        station_id: Optional[str] = None,
        timestamp_utc: Optional[datetime] = None
    ) -> List[Evidence]:
        """Get market state evidence (orderbook, spreads)."""
        evidences = []

        # Normalize station_id: if it's a list, use first element or None for "all"
        if isinstance(station_id, list):
            station_id = station_id[0] if len(station_id) == 1 else None

        try:
            from market.polymarket_ws import get_ws_client_sync
            client = get_ws_client_sync()

            if not client:
                return evidences

            # Get orderbook state
            books = client.get_all_books()

            for token_id, book in books.items():
                market_info = client.get_market_info(token_id)
                if not market_info:
                    continue

                # Filter by station if needed
                if station_id and station_id not in market_info:
                    continue

                best_bid = max(book.get("bids", {}).keys(), default=0)
                best_ask = min(book.get("asks", {}).keys(), default=1)
                spread = best_ask - best_bid if best_ask > best_bid else 0

                bid_depth = sum(book.get("bids", {}).values())
                ask_depth = sum(book.get("asks", {}).values())

                evidences.append(Evidence(
                    id=self._next_id(),
                    type=EvidenceType.MARKET,
                    timestamp_utc=timestamp_utc or datetime.now(UTC),
                    source=f"L2/{market_info}",
                    summary=f"bid={best_bid:.2f} ask={best_ask:.2f} spread={spread:.4f} depth={bid_depth:.0f}/{ask_depth:.0f}",
                    data={
                        "token_id": token_id,
                        "market_info": market_info,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread": spread,
                        "bid_depth": bid_depth,
                        "ask_depth": ask_depth,
                        "mid": (best_bid + best_ask) / 2 if spread > 0 else None
                    }
                ))

        except Exception as e:
            logger.warning(f"Failed to get market evidence: {e}")

        return evidences

    def get_world_evidence(
        self,
        station_id: Optional[str] = None,
        timestamp_utc: Optional[datetime] = None
    ) -> List[Evidence]:
        """Get world state evidence (METAR, QC)."""
        evidences = []

        # Normalize station_id: if it's a list, use first element or None for "all"
        if isinstance(station_id, list):
            # If list has one element, use it; otherwise get all stations
            station_id = station_id[0] if len(station_id) == 1 else None

        try:
            from core.world import get_world
            world = get_world()
            snapshot = world.get_snapshot()

            # METAR observations (official)
            official_data = snapshot.get("official", {})

            if isinstance(official_data, dict):
                for obs_station_id, obs in official_data.items():
                    # If station_id filter is set, skip non-matching stations
                    if station_id and obs.get("station_id") != station_id:
                        continue

                    # Parse timestamp - it's already ISO format from serialization
                    ts_str = obs.get("ingest_time_utc", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        except ValueError:
                            ts = datetime.now(UTC)
                    else:
                        ts = datetime.now(UTC)

                    # Build QC summary from qc_passed and qc_flags
                    qc_status = "OK" if obs.get("qc_passed", True) else "FLAGGED"
                    qc_flags = obs.get("qc_flags", [])
                    if qc_flags:
                        # Ensure all items in qc_flags are strings
                        qc_flags_str = [str(f) for f in qc_flags]
                        qc_status = f"FLAGGED({','.join(qc_flags_str)})"

                    evidences.append(Evidence(
                        id=self._next_id(),
                        type=EvidenceType.WORLD,
                        timestamp_utc=ts,
                        source=f"METAR/{obs.get('station_id')}",
                        summary=f"temp={obs.get('temp_f')}°F src={obs.get('source')} qc={qc_status}",
                        data=obs,
                        event_id=obs.get("event_id")
                    ))

            # QC states
            qc_states = snapshot.get("qc", {})

            if isinstance(qc_states, dict) and station_id and station_id in qc_states:
                qc = qc_states[station_id]
                ts_str = qc.get("last_check_utc", "") or qc.get("last_check", "") if isinstance(qc, dict) else ""
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except ValueError:
                        ts = datetime.now(UTC)
                else:
                    ts = datetime.now(UTC)

                if isinstance(qc, dict):
                    qc_flags = qc.get('flags', qc.get('active_flags', []))
                    evidences.append(Evidence(
                        id=self._next_id(),
                        type=EvidenceType.QC,
                        timestamp_utc=ts,
                        source=f"QC/{station_id}",
                        summary=f"status={qc.get('status', qc.get('state', 'unknown'))} flags={qc_flags}",
                        data=qc
                    ))

            # PWS/Aux observations
            aux_data = snapshot.get("aux", {})

            if isinstance(aux_data, dict):
                for obs_station_id, obs in aux_data.items():
                    # Filter by station if needed (aux stations might have different naming)
                    if station_id and station_id not in obs.get("station_id", ""):
                        continue

                    ts_str = obs.get("ingest_time_utc", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        except ValueError:
                            ts = datetime.now(UTC)
                    else:
                        ts = datetime.now(UTC)

                    evidences.append(Evidence(
                        id=self._next_id(),
                        type=EvidenceType.PWS,
                        timestamp_utc=ts,
                        source=f"PWS/{obs.get('station_id')}",
                        summary=f"temp={obs.get('temp_f')}°F support={obs.get('support')} drift={obs.get('drift')}°F",
                        data=obs,
                        event_id=obs.get("event_id")
                    ))

            logger.debug(f"World evidence collected: {len(evidences)} items")

        except Exception as e:
            logger.warning(f"Failed to get world evidence: {e}")

        return evidences

    def get_nowcast_evidence(
        self,
        station_id: Optional[str] = None,
        timestamp_utc: Optional[datetime] = None
    ) -> List[Evidence]:
        """Get nowcast evidence (predictions, distributions)."""
        evidences = []

        # Normalize station_id: if it's a list, use first element or None for "all"
        if isinstance(station_id, list):
            station_id = station_id[0] if len(station_id) == 1 else None

        try:
            from core.nowcast_integration import get_nowcast_integration
            integration = get_nowcast_integration()

            snapshot = integration.get_snapshot()

            # If no station filter, get all distributions
            distributions = snapshot.get("distributions", {})
            stations_to_check = [station_id] if station_id else list(distributions.keys())

            for sid in stations_to_check:
                if sid not in distributions:
                    continue
                dist = distributions[sid]

                # Find highest probability bucket
                p_bucket = dist.get("p_bucket", [])
                top_bucket = None
                top_prob = 0
                for b in p_bucket:
                    prob = b.get("probability", 0)
                    if prob > top_prob:
                        top_prob = prob
                        top_bucket = b.get("label")

                # Handle None values safely
                tmax_mean = dist.get('tmax_mean_f')
                tmax_sigma = dist.get('tmax_sigma_f')
                confidence = dist.get('confidence')

                # Build summary with safe formatting
                sigma_str = f"{tmax_sigma:.1f}" if tmax_sigma is not None else "N/A"
                conf_str = f"{confidence:.0%}" if confidence is not None else "N/A"

                evidences.append(Evidence(
                    id=self._next_id(),
                    type=EvidenceType.NOWCAST,
                    timestamp_utc=datetime.fromisoformat(dist.get("timestamp", "").replace("Z", "+00:00")) if dist.get("timestamp") else datetime.now(UTC),
                    source=f"NOWCAST/{sid}",
                    summary=f"tmax={tmax_mean}°F σ={sigma_str} top_bucket={top_bucket}({top_prob:.1%}) conf={conf_str}",
                    data=dist
                ))

                # Engine state for this station
                if sid in snapshot.get("states", {}):
                    state = snapshot["states"][sid]

                    # Handle None values safely
                    obs_count = state.get('obs_count')
                    obs_max = state.get('obs_max_f')
                    bias_ema = state.get('bias_ema_f')
                    bias_str = f"{bias_ema:.2f}" if bias_ema is not None else "N/A"

                    evidences.append(Evidence(
                        id=self._next_id(),
                        type=EvidenceType.NOWCAST,
                        timestamp_utc=datetime.now(UTC),
                        source=f"NOWCAST_STATE/{sid}",
                        summary=f"obs_count={obs_count} obs_max={obs_max}°F bias={bias_str}°F",
                        data=state
                    ))

        except Exception as e:
            logger.warning(f"Failed to get nowcast evidence: {e}")

        return evidences

    def get_health_evidence(
        self,
        station_id: Optional[str] = None
    ) -> List[Evidence]:
        """Get system health evidence (staleness, latencies)."""
        evidences = []

        # Normalize station_id: if it's a list, use first element or None for "all"
        if isinstance(station_id, list):
            station_id = station_id[0] if len(station_id) == 1 else None

        try:
            from core.world import get_world
            world = get_world()
            snapshot = world.get_snapshot()

            # Source health (key is "health" in World.get_snapshot())
            for source_name, health in snapshot.get("health", {}).items():
                ts_str = health.get("last_seen_utc", "") or health.get("last_seen", "")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except ValueError:
                        ts = datetime.now(UTC)
                else:
                    ts = datetime.now(UTC)

                # Handle different possible key names for staleness
                staleness = health.get("staleness_seconds", health.get("staleness_s", 0))
                if staleness is None:
                    staleness = 0

                evidences.append(Evidence(
                    id=self._next_id(),
                    type=EvidenceType.HEALTH,
                    timestamp_utc=ts,
                    source=f"HEALTH/{source_name}",
                    summary=f"status={health.get('status', 'unknown')} staleness={staleness:.1f}s updates={health.get('update_count', health.get('updates', 0))} errors={health.get('error_count', health.get('errors', 0))}",
                    data=health
                ))

            # WebSocket health
            try:
                from market.polymarket_ws import get_ws_client_sync
                client = get_ws_client_sync()
                if client:
                    ws_stats = {
                        "connected": client._connected if hasattr(client, '_connected') else None,
                        "reconnect_count": client._reconnect_count if hasattr(client, '_reconnect_count') else 0,
                        "subscribed_tokens": len(client._subscribed_tokens) if hasattr(client, '_subscribed_tokens') else 0
                    }
                    evidences.append(Evidence(
                        id=self._next_id(),
                        type=EvidenceType.HEALTH,
                        timestamp_utc=datetime.now(UTC),
                        source="WS_CLIENT",
                        summary=f"connected={ws_stats['connected']} reconnects={ws_stats['reconnect_count']} subscriptions={ws_stats['subscribed_tokens']}",
                        data=ws_stats
                    ))
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Failed to get health evidence: {e}")

        return evidences

    def get_backtest_evidence(
        self,
        station_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> List[Evidence]:
        """Get backtest evidence (runs, metrics, coverage)."""
        evidences = []

        # Normalize station_id: if it's a list, use first element or None
        if isinstance(station_id, list):
            station_id = station_id[0] if len(station_id) == 1 else None

        if not station_id:
            return evidences  # Need a station for backtest queries

        try:
            from core.backtest import get_dataset_builder, get_label_manager

            # Available dates
            builder = get_dataset_builder()
            dates = builder.list_available_dates(station_id)

            evidences.append(Evidence(
                id=self._next_id(),
                type=EvidenceType.BACKTEST,
                timestamp_utc=datetime.now(UTC),
                source=f"BACKTEST_DATA/{station_id}",
                summary=f"available_dates={len(dates)} range={dates[-1] if dates else 'none'} to {dates[0] if dates else 'none'}",
                data={"dates": [d.isoformat() for d in dates[:10]], "total_dates": len(dates)}
            ))

            # Labels
            label_mgr = get_label_manager()
            stats = label_mgr.get_stats()

            evidences.append(Evidence(
                id=self._next_id(),
                type=EvidenceType.BACKTEST,
                timestamp_utc=datetime.now(UTC),
                source=f"LABELS/{station_id}",
                summary=f"total_labels={stats.get('total_labels', 0)} stations={stats.get('stations', [])}",
                data=stats
            ))

        except Exception as e:
            logger.warning(f"Failed to get backtest evidence: {e}")

        return evidences

    # =========================================================================
    # Historical/Window Access
    # =========================================================================

    def get_window_evidence(
        self,
        station_id: Optional[str] = None,
        start_utc: Optional[datetime] = None,
        end_utc: Optional[datetime] = None,
        channels: Optional[List[str]] = None
    ) -> List[Evidence]:
        """Get evidence for a time window (from recordings)."""
        evidences = []

        # Normalize station_id: if it's a list, use first element or None
        if isinstance(station_id, list):
            station_id = station_id[0] if len(station_id) == 1 else None

        if not station_id or not start_utc or not end_utc:
            return evidences  # Need station and time range

        try:
            from core.compactor import get_hybrid_reader
            reader = get_hybrid_reader()

            # Convert to date string
            date_str = start_utc.astimezone(NYC).strftime("%Y-%m-%d")

            # Get all events
            events = reader.get_events_sorted(date_str, station_id, channels)

            # Filter by time window
            for event in events:
                ts_str = event.get("ts_ingest_utc", "")
                if not ts_str:
                    continue

                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if start_utc <= ts <= end_utc:
                    ch = event.get("ch", "unknown")
                    data = event.get("data", event)

                    # Create appropriate evidence type
                    if ch == "world":
                        etype = EvidenceType.WORLD
                        summary = f"temp={data.get('temp_f')}°F src={data.get('src')}"
                    elif ch == "nowcast":
                        etype = EvidenceType.NOWCAST
                        summary = f"tmax={data.get('tmax_mean_f')}°F"
                    elif ch == "pws":
                        etype = EvidenceType.PWS
                        summary = f"median={data.get('median_f')}°F"
                    elif ch in ("market", "l2_snap"):
                        etype = EvidenceType.MARKET
                        summary = "market snapshot"
                    else:
                        etype = EvidenceType.EVENT_WINDOW
                        summary = f"channel={ch}"

                    evidences.append(Evidence(
                        id=self._next_id(),
                        type=etype,
                        timestamp_utc=ts,
                        source=f"{ch.upper()}/{station_id}",
                        summary=summary,
                        data=data,
                        event_id=event.get("event_id")
                    ))

                    # Limit evidence count
                    if len(evidences) >= 50:
                        break

        except Exception as e:
            logger.warning(f"Failed to get window evidence: {e}")

        return evidences

    # =========================================================================
    # Composite Evidence Gathering
    # =========================================================================

    def gather_live_context(
        self,
        station_id: Optional[str] = None,
        include_market: bool = True,
        include_world: bool = True,
        include_nowcast: bool = True,
        include_health: bool = True
    ) -> List[Evidence]:
        """Gather all live evidence for a station (or all stations if None)."""
        self.reset_counter()
        evidences = []

        if include_world:
            evidences.extend(self.get_world_evidence(station_id))

        if include_nowcast:
            evidences.extend(self.get_nowcast_evidence(station_id))

        if include_market:
            evidences.extend(self.get_market_evidence(station_id))

        if include_health:
            evidences.extend(self.get_health_evidence(station_id))

        return evidences

    def gather_diagnostic_context(
        self,
        station_id: Optional[str] = None,
        question: str = "",
        window_minutes: int = 10
    ) -> List[Evidence]:
        """
        Gather evidence relevant to a diagnostic question.

        Analyzes the question to determine what evidence is needed.
        station_id can be a string or list; will be normalized internally.
        """
        self.reset_counter()
        evidences = []

        q_lower = question.lower()

        # Always get nowcast and world
        evidences.extend(self.get_world_evidence(station_id))
        evidences.extend(self.get_nowcast_evidence(station_id))

        # Market-related questions
        if any(kw in q_lower for kw in ["market", "spread", "bid", "ask", "orderbook", "price", "latency", "delay"]):
            evidences.extend(self.get_market_evidence(station_id))

        # Health-related questions
        if any(kw in q_lower for kw in ["health", "staleness", "reconnect", "gap", "error", "delay", "latency", "lag"]):
            evidences.extend(self.get_health_evidence(station_id))

        # Backtest-related questions
        if any(kw in q_lower for kw in ["backtest", "predicted", "actual", "brier", "logloss", "coverage", "label"]):
            evidences.extend(self.get_backtest_evidence(station_id))

        # If asking about changes/history, get window evidence
        if any(kw in q_lower for kw in ["changed", "cambió", "salto", "jump", "history", "antes", "before"]):
            end_utc = datetime.now(UTC)
            start_utc = end_utc - timedelta(minutes=window_minutes)
            evidences.extend(self.get_window_evidence(station_id, start_utc, end_utc))

        return evidences

    def get_technical_docs_evidence(self, question: str = "") -> List[Evidence]:
        """
        Load relevant technical documentation as evidence when user asks
        about how HELIOS works internally.

        Detects technical questions and loads the appropriate doc(s).
        """
        evidences = []
        q_lower = question.lower()

        # Keywords that indicate technical/architecture questions
        arch_keywords = ["arquitectura", "architecture", "cómo funciona", "how does it work",
                        "sistema", "system", "pipeline", "flujo", "flow", "componentes"]
        math_keywords = ["matemáticas", "math", "fórmula", "formula", "ecuación", "equation",
                        "cálculo", "calculation", "bias", "sigma", "cdf", "probabilidad",
                        "probability", "distribución", "distribution", "ema", "decay"]
        data_keywords = ["datos", "data", "fuentes", "sources", "metar", "pws", "sst", "aod",
                        "hrrr", "gfs", "lamp", "nbm", "buoy", "synoptic", "drift", "support",
                        "staleness", "health"]
        pred_keywords = ["predicción", "prediction", "tmax", "bucket", "bracket", "confidence",
                        "confianza", "incertidumbre", "uncertainty", "cap", "floor", "constraint"]

        import os
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs")

        # Map keywords to doc files
        docs_to_load = []

        if any(kw in q_lower for kw in arch_keywords):
            docs_to_load.append(("HELIOS_ARCHITECTURE.md", "Architecture & System Overview"))

        if any(kw in q_lower for kw in math_keywords):
            docs_to_load.append(("HELIOS_MATH.md", "Mathematics & Formulas"))

        if any(kw in q_lower for kw in data_keywords):
            docs_to_load.append(("HELIOS_DATA_SOURCES.md", "Data Sources & Inputs"))

        if any(kw in q_lower for kw in pred_keywords):
            docs_to_load.append(("HELIOS_PREDICTIONS.md", "Prediction System"))

        # If no specific match, load all docs for general "how does it work" questions
        general_keywords = ["cómo", "how", "explica", "explain", "funciona", "works",
                           "qué es", "what is", "por qué", "why"]
        if not docs_to_load and any(kw in q_lower for kw in general_keywords):
            docs_to_load = [
                ("HELIOS_ARCHITECTURE.md", "Architecture & System Overview"),
                ("HELIOS_MATH.md", "Mathematics & Formulas"),
                ("HELIOS_DATA_SOURCES.md", "Data Sources & Inputs"),
                ("HELIOS_PREDICTIONS.md", "Prediction System")
            ]

        # Load each relevant doc
        for doc_filename, doc_title in docs_to_load:
            doc_path = os.path.join(docs_dir, doc_filename)
            try:
                if os.path.exists(doc_path):
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Truncate if too long (keep first ~4000 chars for context limits)
                    if len(content) > 6000:
                        content = content[:6000] + "\n\n[... Document truncated for brevity ...]"

                    evidences.append(Evidence(
                        id=self._next_id(),
                        type=EvidenceType.EVENT_WINDOW,  # Using EVENT_WINDOW for docs
                        timestamp_utc=datetime.now(UTC),
                        source=f"DOCS/{doc_filename}",
                        summary=f"Technical documentation: {doc_title}",
                        data={
                            "doc_type": "technical_reference",
                            "filename": doc_filename,
                            "title": doc_title,
                            "content": content
                        }
                    ))
            except Exception as e:
                logger.warning(f"Failed to load technical doc {doc_filename}: {e}")

        return evidences


# =============================================================================
# Global Singleton
# =============================================================================

_evidence_builder: Optional[EvidenceBuilder] = None


def get_evidence_builder() -> EvidenceBuilder:
    """Get the global EvidenceBuilder instance."""
    global _evidence_builder
    if _evidence_builder is None:
        _evidence_builder = EvidenceBuilder()
    return _evidence_builder
