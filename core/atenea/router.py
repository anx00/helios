"""
Intent Router for ATENEA

Classifies user questions into categories to determine what evidence to gather:
- MARKET: orderbook, spreads, shocks, trading
- WORLD: METAR, PWS, QC, observations
- NOWCAST: predictions, drivers, changes, confidence
- HEALTH: staleness, reconnects, gaps, latency
- BACKTEST: runs, coverage, metrics, debugging
- REPLAY: event windows, timeline analysis
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import re


class IntentCategory(Enum):
    """High-level intent categories"""
    MARKET = "market"
    WORLD = "world"
    NOWCAST = "nowcast"
    HEALTH = "health"
    BACKTEST = "backtest"
    REPLAY = "replay"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Classified intent with confidence and details"""
    category: IntentCategory
    confidence: float  # 0.0 to 1.0
    sub_intent: Optional[str] = None  # More specific intent within category
    entities: Dict[str, Any] = field(default_factory=dict)  # Extracted entities
    keywords_matched: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "sub_intent": self.sub_intent,
            "entities": self.entities,
            "keywords_matched": self.keywords_matched
        }


class IntentRouter:
    """
    Routes user questions to appropriate evidence gathering strategies.

    Uses keyword matching and pattern recognition to classify intent.
    In production, could be enhanced with an LLM classifier.
    """

    # Keyword patterns for each category
    MARKET_KEYWORDS = [
        "orderbook", "order book", "spread", "bid", "ask", "price",
        "trading", "trade", "position", "pnl", "profit", "loss",
        "shock", "jump", "spike", "volume", "liquidity", "market",
        "poly", "polymarket", "clob", "token", "contract"
    ]

    WORLD_KEYWORDS = [
        "metar", "weather", "observation", "temperature", "temp",
        "pws", "sensor", "station", "reading", "qc", "quality",
        "outlier", "anomaly", "wind", "humidity", "pressure",
        "dew point", "visibility", "ceiling", "cloud", "rain",
        "snow", "precipitation", "actual", "observed", "real"
    ]

    NOWCAST_KEYWORDS = [
        "nowcast", "prediction", "forecast", "predict", "model",
        "driver", "feature", "variable", "coefficient", "weight",
        "confidence", "probability", "tmax", "tmin", "expected",
        "change", "cambio", "why", "por qué", "porque", "explain",
        "explica", "reason", "causa", "bucket", "distribution"
    ]

    HEALTH_KEYWORDS = [
        "health", "status", "latency", "delay", "lag", "slow",
        "staleness", "stale", "reconnect", "disconnect", "gap",
        "missing", "websocket", "ws", "connection", "error",
        "timeout", "heartbeat", "ping", "uptime", "downtime"
    ]

    BACKTEST_KEYWORDS = [
        "backtest", "backtesting", "historical", "simulation",
        "coverage", "metric", "brier", "logloss", "log loss",
        "ece", "sharpness", "mae", "rmse", "bias", "accuracy",
        "calibration", "run", "result", "empty", "vacío", "dash",
        "predicted", "label", "falta", "missing"
    ]

    REPLAY_KEYWORDS = [
        "replay", "playback", "timeline", "event", "window",
        "historia", "history", "pasado", "past", "resume",
        "resumen", "summary", "qué pasó", "what happened"
    ]

    # Sub-intent patterns
    SUB_INTENT_PATTERNS = {
        IntentCategory.MARKET: {
            "spread_analysis": ["spread", "bid-ask", "tight", "wide"],
            "shock_detection": ["shock", "jump", "spike", "sudden"],
            "position_status": ["position", "pnl", "profit", "loss"],
            "orderbook_state": ["orderbook", "depth", "liquidity"]
        },
        IntentCategory.WORLD: {
            "metar_analysis": ["metar", "official", "airport"],
            "pws_quality": ["pws", "sensor", "outlier", "qc", "quality"],
            "observation_compare": ["compare", "difference", "vs", "versus"],
            "current_conditions": ["current", "now", "actual", "latest"]
        },
        IntentCategory.NOWCAST: {
            "prediction_explain": ["why", "por qué", "explain", "reason", "driver"],
            "confidence_analysis": ["confidence", "uncertainty", "sure", "certain"],
            "change_analysis": ["change", "cambio", "different", "moved"],
            "distribution_view": ["distribution", "bucket", "probability"]
        },
        IntentCategory.HEALTH: {
            "latency_diagnosis": ["latency", "delay", "slow", "lag"],
            "connection_status": ["connection", "websocket", "reconnect"],
            "data_gaps": ["gap", "missing", "stale", "staleness"],
            "system_health": ["health", "status", "uptime"]
        },
        IntentCategory.BACKTEST: {
            "run_debug": ["empty", "vacío", "dash", "missing", "wrong"],
            "metrics_explain": ["logloss", "brier", "ece", "sharpness"],
            "coverage_analysis": ["coverage", "days", "labels", "data"],
            "results_summary": ["result", "summary", "how", "performance"]
        },
        IntentCategory.REPLAY: {
            "event_window": ["window", "range", "between", "from", "to"],
            "event_summary": ["summary", "resume", "what happened"],
            "trigger_analysis": ["trigger", "cause", "root cause"]
        }
    }

    # Entity extraction patterns
    ENTITY_PATTERNS = {
        "station_id": r'\b([A-Z]{4})\b',  # ICAO codes like KLGA
        "timestamp": r'(\d{2}:\d{2}(?::\d{2})?)',  # HH:MM or HH:MM:SS
        "date": r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        "temperature": r'(\d{1,3}(?:\.\d+)?)\s*[°]?[FCfc]?',  # Temperature values
        "bucket": r'(<\s*\d{1,2}|>=\s*\d{1,2}|≥\s*\d{1,2}|\d{1,2}\s*-\s*\d{1,2}\s*[Â°]?[Ff]?|\d{1,2}\s*[Â°]?[Ff]\s+or\s+(?:below|higher|above))',
        "run_id": r'run[_-]?(\d+)',  # Backtest run ID
    }

    def __init__(self):
        """Initialize the router with compiled patterns"""
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.ENTITY_PATTERNS.items()
        }

    def classify(self, query: str, context: Optional[Dict[str, Any]] = None) -> Intent:
        """
        Classify a user query into an intent.

        Args:
            query: The user's question
            context: Optional screen context (current_screen, station_id, etc.)

        Returns:
            Intent with category, confidence, and extracted entities
        """
        query_lower = query.lower()

        # Score each category
        scores: Dict[IntentCategory, Tuple[float, List[str]]] = {}

        for category, keywords in [
            (IntentCategory.MARKET, self.MARKET_KEYWORDS),
            (IntentCategory.WORLD, self.WORLD_KEYWORDS),
            (IntentCategory.NOWCAST, self.NOWCAST_KEYWORDS),
            (IntentCategory.HEALTH, self.HEALTH_KEYWORDS),
            (IntentCategory.BACKTEST, self.BACKTEST_KEYWORDS),
            (IntentCategory.REPLAY, self.REPLAY_KEYWORDS),
        ]:
            matched = [kw for kw in keywords if kw in query_lower]
            if matched:
                # Score based on number and specificity of matches
                score = len(matched) / len(keywords) + len(matched) * 0.1
                scores[category] = (min(score, 1.0), matched)

        # Determine winner
        if not scores:
            # Check context for hints
            if context and "screen" in context:
                screen = context["screen"].lower()
                if "market" in screen:
                    return Intent(IntentCategory.MARKET, 0.3, entities=self._extract_entities(query, context))
                elif "world" in screen:
                    return Intent(IntentCategory.WORLD, 0.3, entities=self._extract_entities(query, context))
                elif "nowcast" in screen:
                    return Intent(IntentCategory.NOWCAST, 0.3, entities=self._extract_entities(query, context))
                elif "backtest" in screen:
                    return Intent(IntentCategory.BACKTEST, 0.3, entities=self._extract_entities(query, context))
                elif "replay" in screen:
                    return Intent(IntentCategory.REPLAY, 0.3, entities=self._extract_entities(query, context))
                elif "health" in screen:
                    return Intent(IntentCategory.HEALTH, 0.3, entities=self._extract_entities(query, context))

            return Intent(IntentCategory.GENERAL, 0.5, entities=self._extract_entities(query, context))

        # Get best category
        best_category = max(scores.keys(), key=lambda c: scores[c][0])
        best_score, matched_keywords = scores[best_category]

        # Determine sub-intent
        sub_intent = self._classify_sub_intent(query_lower, best_category)

        # Extract entities
        entities = self._extract_entities(query, context)

        return Intent(
            category=best_category,
            confidence=best_score,
            sub_intent=sub_intent,
            entities=entities,
            keywords_matched=matched_keywords
        )

    def _classify_sub_intent(self, query_lower: str, category: IntentCategory) -> Optional[str]:
        """Classify sub-intent within a category"""
        if category not in self.SUB_INTENT_PATTERNS:
            return None

        sub_patterns = self.SUB_INTENT_PATTERNS[category]
        best_sub = None
        best_count = 0

        for sub_name, keywords in sub_patterns.items():
            count = sum(1 for kw in keywords if kw in query_lower)
            if count > best_count:
                best_count = count
                best_sub = sub_name

        return best_sub

    def _extract_entities(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract entities from query and context"""
        entities = {}

        # Extract from query using patterns
        for name, pattern in self._compiled_patterns.items():
            matches = pattern.findall(query)
            if matches:
                # Store first match or all if multiple
                entities[name] = matches[0] if len(matches) == 1 else matches

        # Merge with context if provided
        if context:
            if "station_id" in context and "station_id" not in entities:
                entities["station_id"] = context["station_id"]
            if "time_range" in context:
                entities["time_range"] = context["time_range"]
            if "selected_token_ids" in context:
                entities["token_ids"] = context["selected_token_ids"]
            if "mode" in context:
                entities["mode"] = context["mode"]

        return entities

    def get_evidence_requirements(self, intent: Intent) -> Dict[str, Any]:
        """
        Determine what evidence needs to be gathered for an intent.

        Returns a specification for the Evidence Builder.
        """
        requirements = {
            "live_state": False,
            "historical_window": False,
            "window_minutes": 0,
            "channels": [],
            "include_snapshots": False
        }

        category = intent.category
        sub_intent = intent.sub_intent

        if category == IntentCategory.MARKET:
            requirements["live_state"] = True
            requirements["channels"] = ["l2_snap_1s", "market_shock"]
            if sub_intent in ["shock_detection", "spread_analysis"]:
                requirements["historical_window"] = True
                requirements["window_minutes"] = 10
                requirements["include_snapshots"] = True

        elif category == IntentCategory.WORLD:
            requirements["live_state"] = True
            requirements["channels"] = ["metar", "pws_agg", "pws_qc", "world_features"]
            if sub_intent in ["observation_compare", "pws_quality"]:
                requirements["historical_window"] = True
                requirements["window_minutes"] = 30

        elif category == IntentCategory.NOWCAST:
            requirements["live_state"] = True
            requirements["channels"] = ["nowcast_1m", "metar", "world_features"]
            if sub_intent in ["change_analysis", "prediction_explain"]:
                requirements["historical_window"] = True
                requirements["window_minutes"] = 15
                requirements["include_snapshots"] = True

        elif category == IntentCategory.HEALTH:
            requirements["live_state"] = True
            requirements["channels"] = ["health_tick"]
            if sub_intent in ["latency_diagnosis", "data_gaps"]:
                requirements["historical_window"] = True
                requirements["window_minutes"] = 5

        elif category == IntentCategory.BACKTEST:
            requirements["live_state"] = False
            requirements["channels"] = ["backtest"]
            # Backtest requires run_id from entities
            if "run_id" in intent.entities:
                requirements["run_id"] = intent.entities["run_id"]

        elif category == IntentCategory.REPLAY:
            requirements["live_state"] = False
            requirements["historical_window"] = True
            requirements["window_minutes"] = 30
            requirements["channels"] = [
                "l2_snap_1s", "metar", "pws_agg", "nowcast_1m", "health_tick"
            ]
            requirements["include_snapshots"] = True

        else:  # GENERAL or UNKNOWN
            requirements["live_state"] = True
            requirements["channels"] = ["nowcast_1m", "health_tick"]

        return requirements


# Singleton instance
_router_instance: Optional[IntentRouter] = None


def get_intent_router() -> IntentRouter:
    """Get or create the singleton IntentRouter instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = IntentRouter()
    return _router_instance
