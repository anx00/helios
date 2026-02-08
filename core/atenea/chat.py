"""
ATENEA Chat Service

Handles the end-to-end chat flow:
1. Receive user query + screen context
2. Route intent
3. Gather evidence
4. Build context pack
5. Call Gemini Flash with strict instructions
6. Validate response has evidence citations
7. Return formatted response
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

from .router import IntentRouter, Intent, IntentCategory, get_intent_router
from .evidence import EvidenceBuilder, Evidence, EvidenceType, get_evidence_builder
from .context import ContextBuilder, ContextPack, ScreenContext, create_context_builder


logger = logging.getLogger(__name__)


# System prompt for ATENEA
ATENEA_SYSTEM_PROMPT = """You are ATENEA, the expert AI assistant for HELIOS — a weather-based temperature forecasting and Polymarket trading system.

## YOUR ROLE

You are a senior meteorologist, data scientist, and quantitative trader rolled into one. You deeply understand:
- Weather forecasting: METAR observations, PWS clusters, NWP models, temperature dynamics
- HELIOS internals: nowcast engine, bias correction, probability distributions, QC systems
- Polymarket trading: orderbooks, spreads, market sentiment, probability pricing, crowd wisdom
- System operations: data staleness, latency, reconnects, health metrics

## YOUR CAPABILITIES

1. **Analyze current conditions**: Examine METAR temps, PWS consensus, environmental features (SST, AOD), and explain what they indicate
2. **Evaluate predictions**: Compare HELIOS forecasts vs observations, assess confidence levels, identify potential issues
3. **Interpret market state**: Analyze Polymarket prices, orderbook depth, spreads, and crowd sentiment
4. **Correlate data**: Connect weather observations with market movements, identify discrepancies
5. **Provide expert opinions**: When asked "is the prediction correct?", give a reasoned assessment based on available data
6. **Diagnose issues**: Identify data quality problems, system health issues, or anomalies

## DATA CONTEXT

You receive real-time data from HELIOS including:
- METAR observations (official airport temps)
- PWS cluster consensus (personal weather stations)
- Nowcast predictions (tmax_mean, sigma, probability distributions)
- Environmental features (SST, AOD, cloud cover)
- Polymarket prices (YES/NO probabilities per temperature bracket)
- Orderbook state (bids, asks, spread, depth, mid prices)
- Market sentiment (crowd wisdom, probability velocity, shifts)
- System health (staleness, latencies, reconnects)

## HOW TO RESPOND

1. **Be helpful and direct**: Answer the question fully. Don't refuse to help.
2. **Use available data**: Reference specific values from the context (temps, prices, spreads)
3. **Reason transparently**: Explain your thinking when giving opinions
4. **Acknowledge uncertainty**: If data is incomplete, say so but still provide your best assessment
5. **Be actionable**: Suggest what to watch, potential risks, or recommended actions
6. **Match user's language**: Respond in Spanish if asked in Spanish, English if in English

## EXAMPLES

User: "¿Crees que la predicción de HELIOS es correcta?"
Good response: "Basándome en los datos actuales, la predicción de HELIOS de 34°F parece razonable pero conservadora. El METAR actual muestra 32°F a las 14:00 con tendencia ascendente, y el cluster PWS marca 33.5°F. Sin embargo, el mercado de Polymarket está pricing el bracket 36-37°F al 42% vs nuestro 28% para ese rango. Esta discrepancia sugiere que el mercado espera un calentamiento mayor. Recomendaría vigilar las próximas lecturas METAR antes de las 15:00."

User: "¿Por qué el spread está tan amplio?"
Good response: "El spread de 0.08 en el bracket 34-35°F es inusualmente amplio (normal: 0.02-0.04). Esto indica baja liquidez, posiblemente porque: (1) estamos cerca del settlement y los market makers reducen exposición, (2) hay incertidumbre por el salto de temperatura de +2°F en el último METAR, o (3) el websocket tuvo 3 reconnects en la última hora afectando la sincronización."

## WHAT TO AVOID

- Don't refuse to answer because "no evidence" — use whatever data is available
- Don't be overly cautious — give your expert assessment
- Don't just list data — interpret it and draw conclusions
- Don't ask for more information if you can provide a useful answer with current context
"""


@dataclass
class ChatMessage:
    """A single chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Optional[Dict[str, Any]] = None  # Screen context when message was sent
    evidences: List[str] = field(default_factory=list)  # Evidence IDs cited

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "evidences": self.evidences
        }


@dataclass
class ChatResponse:
    """Response from ATENEA"""
    message: ChatMessage
    intent: Intent
    context_pack: ContextPack
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    processing_time_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message.to_dict(),
            "intent": self.intent.to_dict(),
            "context_pack": self.context_pack.to_dict(),
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "processing_time_ms": self.processing_time_ms
        }


class AteneaChat:
    """
    Main chat service for ATENEA.

    Orchestrates the full pipeline from query to evidence-backed response.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash"
    ):
        """
        Initialize the chat service.

        Args:
            api_key: Google AI API key (or from GOOGLE_API_KEY env var)
            model_name: Gemini model to use
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name

        if self.api_key and HAS_GENAI:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=ATENEA_SYSTEM_PROMPT
            )
        else:
            if not HAS_GENAI:
                logger.warning("google-generativeai not installed. Chat will use mock responses.")
            else:
                logger.warning("No Google API key configured. Chat will use mock responses.")
            self.model = None

        self.router = get_intent_router()
        self.evidence_builder = get_evidence_builder()

        # Conversation history (per session)
        self.history: List[ChatMessage] = []
        try:
            self._max_history_messages = int(os.environ.get("ATENEA_MAX_HISTORY_MESSAGES", "100"))
        except (TypeError, ValueError):
            self._max_history_messages = 100

    def set_live_state(
        self,
        market_state: Any = None,
        world_state: Any = None,
        nowcast_state: Any = None,
        health_state: Any = None
    ):
        """Inject live state references for evidence gathering"""
        self.evidence_builder.set_live_state(
            market_state=market_state,
            world_state=world_state,
            nowcast_state=nowcast_state,
            health_state=health_state
        )

    async def chat(
        self,
        query: str,
        screen_context: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """
        Process a user query and return an evidence-backed response.

        Args:
            query: User's question
            screen_context: Current screen state (screen, station_id, mode, etc.)

        Returns:
            ChatResponse with message, intent, context pack, and validation status
        """
        start_time = datetime.utcnow()

        # Default context
        if screen_context is None:
            screen_context = {}

        # 1. Route intent
        intent = self.router.classify(query, screen_context)
        logger.info(f"Intent classified: {intent.category.value} ({intent.confidence:.2f})")

        # 2. Determine evidence requirements
        requirements = self.router.get_evidence_requirements(intent)

        # 3. Gather evidence (pass query for technical docs detection)
        evidences = await self._gather_evidence(intent, requirements, screen_context, query)

        # 4. Build context pack
        context_pack = self._build_context_pack(query, intent, evidences, screen_context)

        # 5. Generate response
        response_text = await self._generate_response(query, context_pack)

        # 6. Validate response
        validation_passed, validation_errors = self._validate_response(response_text, context_pack)

        # 7. Extract cited evidences
        cited_evidences = self._extract_evidence_citations(response_text)

        # Create message
        message = ChatMessage(
            role="assistant",
            content=response_text,
            context=screen_context,
            evidences=cited_evidences
        )

        # Add to history
        self.history.append(ChatMessage(
            role="user",
            content=query,
            context=screen_context
        ))
        self.history.append(message)
        if self._max_history_messages > 0 and len(self.history) > self._max_history_messages:
            self.history = self.history[-self._max_history_messages:]

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ChatResponse(
            message=message,
            intent=intent,
            context_pack=context_pack,
            validation_passed=validation_passed,
            validation_errors=validation_errors,
            processing_time_ms=processing_time
        )

    async def _gather_evidence(
        self,
        intent: Intent,
        requirements: Dict[str, Any],
        context: Dict[str, Any],
        query: str = ""
    ) -> List[Evidence]:
        """
        Gather ALL available evidence for comprehensive context.

        Atenea is an expert assistant that needs full context to provide
        useful analysis, so we always gather everything regardless of intent.
        Also loads technical documentation when user asks about how HELIOS works.
        """
        evidences = []
        station_id = context.get("station_id") or intent.entities.get("station_id")

        # ALWAYS gather all live state - Atenea needs full context to be useful
        self.evidence_builder.reset_counter()

        # Check for technical questions and load relevant documentation
        tech_docs_ev = self.evidence_builder.get_technical_docs_evidence(query)
        if tech_docs_ev:
            evidences.extend(tech_docs_ev)

        # World data (METAR, PWS, QC)
        world_ev = self.evidence_builder.get_world_evidence(station_id)
        if world_ev:
            evidences.extend(world_ev)

        # Nowcast predictions
        nowcast_ev = self.evidence_builder.get_nowcast_evidence(station_id)
        if nowcast_ev:
            evidences.extend(nowcast_ev)

        # Market orderbook (WS)
        market_ev = self.evidence_builder.get_market_evidence(station_id)
        if market_ev:
            evidences.extend(market_ev)

        # System health
        health_ev = self.evidence_builder.get_health_evidence(station_id)
        if health_ev:
            evidences.extend(health_ev)

        # Polymarket Gamma API data (prices, volumes, sentiment)
        polymarket_ev = await self._get_polymarket_api_evidence(station_id)
        if polymarket_ev:
            evidences.extend(polymarket_ev)

        # Gather historical window if specifically requested
        if requirements.get("historical_window"):
            window_minutes = requirements.get("window_minutes", 10)
            channels = requirements.get("channels", [])

            from datetime import timedelta
            end_utc = datetime.utcnow()
            start_utc = end_utc - timedelta(minutes=window_minutes)

            window_ev = self.evidence_builder.get_window_evidence(
                station_id=station_id,
                start_utc=start_utc,
                end_utc=end_utc,
                channels=channels if channels else None
            )
            if window_ev:
                evidences.extend(window_ev)

        # Backtest specific
        if intent.category == IntentCategory.BACKTEST:
            run_id = requirements.get("run_id") or intent.entities.get("run_id")
            if run_id:
                backtest_ev = self.evidence_builder.get_backtest_evidence(run_id)
                if backtest_ev:
                    evidences.extend(backtest_ev)

        return evidences

    async def _get_polymarket_api_evidence(self, station_id: Optional[str]) -> List[Evidence]:
        """
        Get Polymarket Gamma API evidence (prices, volumes, market status).

        This provides YES/NO prices and market sentiment that complement
        the WebSocket orderbook data from the regular market evidence.
        """
        evidences = []

        if not station_id:
            return evidences

        try:
            from market.polymarket_checker import fetch_event_for_station_date
            from market.crowd_wisdom import get_crowd_sentiment, detect_market_shift
            from zoneinfo import ZoneInfo
            from stations import STATIONS
            import json as _json

            if station_id not in STATIONS:
                return evidences

            station = STATIONS[station_id]
            local_now = datetime.now(ZoneInfo(station.timezone))
            target_date = local_now.date()

            event_data = fetch_event_for_station_date(station_id, target_date)
            if not event_data:
                return evidences

            # Parse brackets with YES/NO prices
            brackets_summary = []
            brackets_data = []
            for market in event_data.get("markets", []):
                name = market.get("groupItemTitle", "")
                prices = _json.loads(market.get("outcomePrices", "[]"))
                yes_price = float(prices[0]) if prices else 0
                volume = float(market.get("volume", 0))
                brackets_summary.append(f"{name}: {yes_price:.0%} (${volume:,.0f})")
                brackets_data.append({
                    "name": name,
                    "yes_price": yes_price,
                    "volume": volume
                })

            # Market sentiment
            sentiment = get_crowd_sentiment(station_id)
            shifts = detect_market_shift(station_id)

            # Format shifts for summary
            shift_summary = ""
            if shifts:
                top_shift = shifts[0] if isinstance(shifts, list) else shifts
                if isinstance(top_shift, dict):
                    shift_summary = f" | Top shift: {top_shift.get('bracket', 'N/A')} ({top_shift.get('change', 0):+.1%})"

            evidences.append(Evidence(
                id=self.evidence_builder._next_id(),
                type=EvidenceType.MARKET,
                timestamp_utc=datetime.now(ZoneInfo("UTC")),
                source=f"POLYMARKET_API/{station_id}",
                summary=f"Event: {event_data.get('title', 'N/A')} | Top brackets: {', '.join(brackets_summary[:3])} | Sentiment: {sentiment}{shift_summary}",
                data={
                    "event_title": event_data.get("title"),
                    "target_date": str(target_date),
                    "total_volume": sum(float(m.get("volume", 0)) for m in event_data.get("markets", [])),
                    "brackets": brackets_data,
                    "brackets_summary": brackets_summary,
                    "sentiment": sentiment,
                    "shifts": shifts,
                    "active": event_data.get("active", False),
                    "closed": event_data.get("closed", False)
                }
            ))

        except Exception as e:
            logger.warning(f"Failed to get Polymarket API evidence: {e}")

        return evidences

    def _build_context_pack(
        self,
        query: str,
        intent: Intent,
        evidences: List[Evidence],
        context: Dict[str, Any]
    ) -> ContextPack:
        """Build context pack from gathered evidence"""
        # Determine screen context
        screen_str = context.get("screen", "unknown").lower()
        screen = ScreenContext.UNKNOWN
        for sc in ScreenContext:
            if sc.value in screen_str or screen_str in sc.value:
                screen = sc
                break

        builder = create_context_builder()
        builder.start_pack(
            query=query,
            screen=screen,
            station_id=context.get("station_id"),
            mode=context.get("mode", "LIVE")
        )

        # Add evidences
        builder.add_evidences(evidences)

        # Add metrics from evidence data
        for ev in evidences:
            if ev.data:
                if "spread" in ev.data:
                    builder.add_metric("spread", ev.data["spread"])
                if "mid" in ev.data:
                    builder.add_metric("mid", ev.data["mid"])
                if "predicted_tmax" in ev.data:
                    builder.add_metric("predicted_tmax", ev.data["predicted_tmax"])
                if "staleness_ms" in ev.data:
                    builder.add_metric("staleness_ms", ev.data["staleness_ms"])

        # Check for missing data
        if not evidences:
            builder.mark_missing("No evidence available for this query")
        elif intent.category == IntentCategory.MARKET and not any(e.type.value == "market" for e in evidences):
            builder.mark_missing("Market data not available")
        elif intent.category == IntentCategory.WORLD and not any(e.type.value == "world" for e in evidences):
            builder.mark_missing("World/observation data not available")

        return builder.build()

    async def _generate_response(self, query: str, context_pack: ContextPack) -> str:
        """Generate response using Gemini"""
        if not self.model:
            return self._generate_mock_response(query, context_pack)

        # Build prompt with context - expert-friendly, no citation requirements
        prompt = f"""{context_pack.to_llm_prompt()}

User question: {query}

Provide a helpful, expert response based on the available data. Reason through the data and give your assessment. Match the user's language (Spanish/English)."""

        try:
            # Build chat history for context
            history = []
            for msg in self.history[-6:]:  # Last 3 exchanges
                history.append({
                    "role": "user" if msg.role == "user" else "model",
                    "parts": [msg.content]
                })

            chat = self.model.start_chat(history=history)
            response = chat.send_message(prompt)

            return response.text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error al consultar el modelo: {str(e)}\n\nPor favor, verifica la configuración de la API key de Google."

    def _generate_mock_response(self, query: str, context_pack: ContextPack) -> str:
        """
        Generate a helpful mock response when no API key is configured.

        This provides a useful summary of available data even without the LLM,
        helping users understand what Atenea can see.
        """
        lines = []

        if not context_pack.evidences:
            lines.append("**Estado del sistema:**")
            lines.append("No hay datos disponibles en este momento. Esto puede indicar:")
            lines.append("- HELIOS no está conectado o está iniciando")
            lines.append("- Los feeds de datos están desconectados")
            lines.append("- No hay mercado activo para la estación seleccionada")
            lines.append("")
            lines.append("*Nota: Atenea está en modo demo (sin API key de Google). Configura GOOGLE_API_KEY para análisis completos.*")
            return "\n".join(lines)

        # Group evidence by type for organized summary
        evidence_by_type = {}
        for ev in context_pack.evidences:
            etype = ev.type.value
            if etype not in evidence_by_type:
                evidence_by_type[etype] = []
            evidence_by_type[etype].append(ev)

        lines.append("**Resumen de datos disponibles:**\n")

        # World data (METAR, observations)
        if "world" in evidence_by_type:
            lines.append("📡 **Observaciones:**")
            for ev in evidence_by_type["world"]:
                lines.append(f"  - {ev.summary}")
            lines.append("")

        # PWS data
        if "pws" in evidence_by_type:
            lines.append("🌡️ **PWS Cluster:**")
            for ev in evidence_by_type["pws"]:
                lines.append(f"  - {ev.summary}")
            lines.append("")

        # Nowcast predictions
        if "nowcast" in evidence_by_type:
            lines.append("🎯 **Predicción HELIOS:**")
            for ev in evidence_by_type["nowcast"]:
                lines.append(f"  - {ev.summary}")
            lines.append("")

        # Market data
        if "market" in evidence_by_type:
            lines.append("📊 **Mercado Polymarket:**")
            for ev in evidence_by_type["market"]:
                lines.append(f"  - {ev.summary}")
            lines.append("")

        # Health data
        if "health" in evidence_by_type:
            lines.append("💚 **Salud del sistema:**")
            for ev in evidence_by_type["health"]:
                lines.append(f"  - {ev.summary}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Atenea está en modo demo. Para análisis inteligente, configura GOOGLE_API_KEY.*")

        return "\n".join(lines)

    def _validate_response(
        self,
        response: str,
        context_pack: ContextPack
    ) -> tuple[bool, List[str]]:
        """
        Basic validation - just check for generation errors.

        We no longer require strict evidence citations since Atenea
        is now an expert assistant that can reason and opine.
        """
        errors = []

        # Only fail if response is empty or indicates an error
        if not response or response.strip() == "":
            errors.append("Empty response generated")

        if "Error al consultar el modelo" in response:
            errors.append("Model API error occurred")

        return len(errors) == 0, errors

    def _extract_evidence_citations(self, response: str) -> List[str]:
        """Extract evidence citation IDs from response"""
        citations = re.findall(r'E(\d+)', response)
        return [f"E{c}" for c in sorted(set(citations), key=int)]

    def clear_history(self):
        """Clear conversation history"""
        self.history = []

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as dicts"""
        return [msg.to_dict() for msg in self.history]


# Singleton instance
_chat_instance: Optional[AteneaChat] = None


def get_atenea_chat() -> AteneaChat:
    """Get or create the singleton AteneaChat instance"""
    global _chat_instance
    if _chat_instance is None:
        _chat_instance = AteneaChat()
    return _chat_instance


def configure_atenea(api_key: str, model_name: str = "gemini-2.0-flash"):
    """Configure ATENEA with API key"""
    global _chat_instance
    _chat_instance = AteneaChat(api_key=api_key, model_name=model_name)
    return _chat_instance
