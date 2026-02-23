"""
Context Builder for ATENEA

Builds compact context packs from gathered evidence for LLM consumption.
Handles:
- Screen context (current view, selected entities)
- Before/After snapshots for change analysis
- Event timeline compression
- Evidence formatting with IDs
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from .evidence import Evidence, EvidenceType


class ScreenContext(Enum):
    """Current screen/view in HELIOS"""
    MARKET_LIVE = "market_live"
    WORLD_LIVE = "world_live"
    FEATURES = "features"
    NOWCAST = "nowcast"
    REPLAY = "replay"
    BACKTEST_LAB = "backtest_lab"
    HEALTH = "health"
    UNKNOWN = "unknown"


@dataclass
class ContextPack:
    """
    Compact context package for LLM consumption.

    Contains all necessary information for answering a question
    with proper evidence citations.
    """
    # User query
    query: str

    # Screen context
    screen: ScreenContext
    station_id: Optional[str] = None
    mode: str = "LIVE"  # LIVE or REPLAY

    # Time context
    query_time: datetime = field(default_factory=datetime.utcnow)
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None

    # Evidence collection
    evidences: List[Evidence] = field(default_factory=list)

    # Snapshots for comparison (before/after)
    snapshot_before: Optional[Dict[str, Any]] = None
    snapshot_after: Optional[Dict[str, Any]] = None

    # Key metrics summary
    metrics_summary: Dict[str, Any] = field(default_factory=dict)

    # Event timeline (compressed)
    event_timeline: List[Dict[str, Any]] = field(default_factory=list)

    # What's missing (for "insufficient evidence" responses)
    missing_data: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "screen": self.screen.value,
            "station_id": self.station_id,
            "mode": self.mode,
            "query_time": self.query_time.isoformat() if self.query_time else None,
            "time_range": {
                "start": self.time_range_start.isoformat() if self.time_range_start else None,
                "end": self.time_range_end.isoformat() if self.time_range_end else None
            },
            "evidences": [e.to_dict() for e in self.evidences],
            "snapshot_before": self.snapshot_before,
            "snapshot_after": self.snapshot_after,
            "metrics_summary": self.metrics_summary,
            "event_timeline": self.event_timeline,
            "missing_data": self.missing_data,
            "evidence_count": len(self.evidences)
        }

    def to_llm_prompt(self) -> str:
        """
        Format context pack for LLM consumption - expert-friendly format.

        Groups data by type and presents it in a clear, actionable way
        that helps Atenea reason about the current state.
        """
        from zoneinfo import ZoneInfo
        lines = []

        # Header with context
        lines.append("## Current Context")
        lines.append(f"- Screen: {self.screen.value}")
        lines.append(f"- Station: {self.station_id or 'All stations'}")
        lines.append(f"- Mode: {self.mode}")
        try:
            nyc_time = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- Current Time: {nyc_time} NYC")
        except Exception:
            if self.query_time:
                lines.append(f"- Query Time: {self.query_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")

        # Time range if applicable
        if self.time_range_start and self.time_range_end:
            lines.append(f"## Analysis Window")
            lines.append(f"- From: {self.time_range_start.strftime('%H:%M:%S')}")
            lines.append(f"- To: {self.time_range_end.strftime('%H:%M:%S')}")
            lines.append("")

        # Group evidences by type for clearer presentation
        if self.evidences:
            evidence_by_type = {}
            for ev in self.evidences:
                etype = ev.type.value.upper()
                if etype not in evidence_by_type:
                    evidence_by_type[etype] = []
                evidence_by_type[etype].append(ev)

            for etype, evs in evidence_by_type.items():
                lines.append(f"## {etype} Data")
                for ev in evs:
                    ts_str = ev.timestamp_utc.strftime("%H:%M:%S") if ev.timestamp_utc else "N/A"
                    lines.append(f"- [{ev.source}] @ {ts_str}: {ev.summary}")

                    # Include key data points for richer context
                    if ev.data:
                        important_keys = ['temp_f', 'tmax_mean_f', 'tmax_sigma_f', 'confidence',
                                         'best_bid', 'best_ask', 'spread', 'mid', 'yes_price',
                                         'sentiment', 'brackets', 'brackets_summary', 'total_volume',
                                         'status', 'staleness', 'qc_flags', 'support', 'drift']
                        for k in important_keys:
                            if k in ev.data and ev.data[k] is not None:
                                v = ev.data[k]
                                # Format lists nicely
                                if isinstance(v, list) and len(v) > 0:
                                    if len(v) <= 5:
                                        lines.append(f"  - {k}: {', '.join(str(x) for x in v)}")
                                    else:
                                        lines.append(f"  - {k}: {', '.join(str(x) for x in v[:5])}... ({len(v)} total)")
                                elif not isinstance(v, (dict, list)):
                                    lines.append(f"  - {k}: {v}")
                lines.append("")
        else:
            lines.append("## Data Status")
            lines.append("- No live data available at this moment")
            lines.append("")

        # Key metrics summary
        if self.metrics_summary:
            lines.append("## Key Metrics")
            for key, value in self.metrics_summary.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

        # Snapshots for comparison
        if self.snapshot_before or self.snapshot_after:
            lines.append("## State Comparison")
            if self.snapshot_before:
                lines.append("**Before:**")
                lines.append(self._format_snapshot(self.snapshot_before))
            if self.snapshot_after:
                lines.append("**After:**")
                lines.append(self._format_snapshot(self.snapshot_after))
            lines.append("")

        # Event timeline
        if self.event_timeline:
            lines.append("## Recent Events")
            for event in self.event_timeline[:10]:
                ts = event.get("timestamp", "?")
                etype = event.get("type", "event")
                summary = event.get("summary", "")
                lines.append(f"- [{ts}] {etype}: {summary}")
            if len(self.event_timeline) > 10:
                lines.append(f"- ... and {len(self.event_timeline) - 10} more events")
            lines.append("")

        # Missing data warnings
        if self.missing_data:
            lines.append("## Data Gaps")
            for item in self.missing_data:
                lines.append(f"- {item}")
            lines.append("")

        return "\n".join(lines)

    def _format_snapshot(self, snapshot: Dict[str, Any]) -> str:
        """Format a snapshot dict for display"""
        lines = []
        for key, value in snapshot.items():
            if isinstance(value, dict):
                lines.append(f"    {key}:")
                for k, v in value.items():
                    lines.append(f"      {k}: {v}")
            else:
                lines.append(f"    {key}: {value}")
        return "\n".join(lines)

    def has_sufficient_evidence(self) -> bool:
        """Check if we have enough evidence to answer the query"""
        return len(self.evidences) > 0 and len(self.missing_data) == 0


class ContextBuilder:
    """
    Builds ContextPack from gathered evidence and screen state.

    Responsible for:
    - Compressing large datasets into LLM-friendly summaries
    - Creating before/after snapshots
    - Building event timelines
    - Tracking missing data
    """

    MAX_EVIDENCE_ITEMS = 10
    MAX_TIMELINE_EVENTS = 50

    def __init__(self):
        self._current_pack: Optional[ContextPack] = None

    def start_pack(
        self,
        query: str,
        screen: ScreenContext,
        station_id: Optional[str] = None,
        mode: str = "LIVE"
    ) -> "ContextBuilder":
        """Start building a new context pack"""
        self._current_pack = ContextPack(
            query=query,
            screen=screen,
            station_id=station_id,
            mode=mode,
            query_time=datetime.utcnow()
        )
        return self

    def set_time_range(
        self,
        start: Optional[datetime],
        end: Optional[datetime]
    ) -> "ContextBuilder":
        """Set the analysis time range"""
        if self._current_pack:
            self._current_pack.time_range_start = start
            self._current_pack.time_range_end = end
        return self

    def add_evidence(self, evidence: Evidence) -> "ContextBuilder":
        """Add evidence to the pack"""
        if self._current_pack:
            if len(self._current_pack.evidences) < self.MAX_EVIDENCE_ITEMS:
                self._current_pack.evidences.append(evidence)
        return self

    def add_evidences(self, evidences: List[Evidence]) -> "ContextBuilder":
        """Add multiple evidences"""
        for e in evidences:
            self.add_evidence(e)
        return self

    def set_snapshot_before(self, snapshot: Dict[str, Any]) -> "ContextBuilder":
        """Set the 'before' snapshot for comparison"""
        if self._current_pack:
            self._current_pack.snapshot_before = self._compress_snapshot(snapshot)
        return self

    def set_snapshot_after(self, snapshot: Dict[str, Any]) -> "ContextBuilder":
        """Set the 'after' snapshot for comparison"""
        if self._current_pack:
            self._current_pack.snapshot_after = self._compress_snapshot(snapshot)
        return self

    def add_metric(self, name: str, value: Any) -> "ContextBuilder":
        """Add a key metric to the summary"""
        if self._current_pack:
            self._current_pack.metrics_summary[name] = value
        return self

    def add_metrics(self, metrics: Dict[str, Any]) -> "ContextBuilder":
        """Add multiple metrics"""
        if self._current_pack:
            self._current_pack.metrics_summary.update(metrics)
        return self

    def add_timeline_event(
        self,
        timestamp: datetime,
        event_type: str,
        summary: str,
        event_id: Optional[str] = None
    ) -> "ContextBuilder":
        """Add an event to the timeline"""
        if self._current_pack:
            if len(self._current_pack.event_timeline) < self.MAX_TIMELINE_EVENTS:
                self._current_pack.event_timeline.append({
                    "timestamp": timestamp.strftime("%H:%M:%S"),
                    "type": event_type,
                    "summary": summary,
                    "event_id": event_id
                })
        return self

    def add_timeline_events(self, events: List[Dict[str, Any]]) -> "ContextBuilder":
        """Add multiple timeline events from raw event dicts"""
        for event in events:
            ts = event.get("timestamp") or event.get("ts")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    continue
            if ts:
                self.add_timeline_event(
                    timestamp=ts,
                    event_type=event.get("channel", event.get("type", "event")),
                    summary=self._summarize_event(event),
                    event_id=event.get("event_id")
                )
        return self

    def mark_missing(self, what: str) -> "ContextBuilder":
        """Mark data as missing"""
        if self._current_pack:
            self._current_pack.missing_data.append(what)
        return self

    def build(self) -> ContextPack:
        """Build and return the context pack"""
        if not self._current_pack:
            raise ValueError("No context pack started. Call start_pack() first.")

        pack = self._current_pack
        self._current_pack = None

        # Sort timeline by timestamp
        pack.event_timeline.sort(key=lambda e: e.get("timestamp", ""))

        return pack

    def _compress_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Compress a snapshot to essential fields"""
        compressed = {}

        # Market state compression
        if "market" in snapshot:
            market = snapshot["market"]
            compressed["market"] = {
                "spread": market.get("spread"),
                "mid": market.get("mid"),
                "bid_depth": market.get("bid_depth"),
                "ask_depth": market.get("ask_depth")
            }

        # World state compression
        if "world" in snapshot:
            world = snapshot["world"]
            compressed["world"] = {
                "metar_temp": world.get("metar_temp_f"),
                "pws_median": world.get("pws_median"),
                "pws_count": world.get("pws_count"),
                "qc_flags": world.get("qc_flags")
            }

        # Nowcast compression
        if "nowcast" in snapshot:
            nc = snapshot["nowcast"]
            compressed["nowcast"] = {
                "predicted_tmax": nc.get("predicted_tmax"),
                "confidence": nc.get("confidence"),
                "top_bucket": nc.get("top_bucket"),
                "top_bucket_prob": nc.get("top_bucket_prob")
            }

        # Health compression
        if "health" in snapshot:
            health = snapshot["health"]
            compressed["health"] = {
                "market_staleness_ms": health.get("market_staleness_ms"),
                "world_staleness_ms": health.get("world_staleness_ms"),
                "ws_reconnects": health.get("ws_reconnect_count")
            }

        # If snapshot doesn't have these categories, keep top-level keys
        if not compressed:
            # Keep only scalar values and small lists
            for key, value in snapshot.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    compressed[key] = value
                elif isinstance(value, list) and len(value) <= 5:
                    compressed[key] = value

        return compressed

    def _summarize_event(self, event: Dict[str, Any]) -> str:
        """Create a brief summary of an event"""
        channel = event.get("channel", "")

        if channel == "metar":
            temp = event.get("temp_f", event.get("temp_c", "?"))
            raw = event.get("raw", "")[:50]
            report_type = str(event.get("report_type") or "METAR").upper()
            kind = "SPECI" if report_type == "SPECI" else "METAR"
            return f"{kind} temp={temp}?F raw='{raw}...'"

        elif channel == "pws_agg":
            median = event.get("median", "?")
            count = event.get("count", "?")
            return f"PWS cluster median={median} n={count}"

        elif channel == "nowcast_1m":
            tmax = event.get("predicted_tmax", "?")
            top = event.get("top_bucket", "?")
            return f"Nowcast Tmax={tmax} top_bucket={top}"

        elif channel == "l2_snap_1s":
            spread = event.get("spread", "?")
            mid = event.get("mid", "?")
            return f"L2 spread={spread} mid={mid}"

        elif channel == "market_shock":
            shock_type = event.get("shock_type", "?")
            magnitude = event.get("magnitude", "?")
            return f"Market shock type={shock_type} mag={magnitude}"

        elif channel == "health_tick":
            staleness = event.get("market_staleness_ms", "?")
            return f"Health staleness={staleness}ms"

        else:
            # Generic summary
            return str(event.get("summary", event.get("data", "")))[:100]


# Convenience function
def create_context_builder() -> ContextBuilder:
    """Create a new ContextBuilder instance"""
    return ContextBuilder()
