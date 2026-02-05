"""
ATENEA - AI Diagnostic Copilot for HELIOS

Atenea provides evidence-based answers to diagnostic questions about:
- Market state (orderbook, spreads, shocks)
- World state (METAR, PWS, QC, features)
- Nowcast/predictions (drivers, changes, confidence)
- Backtest results (coverage, metrics, issues)
- System health (staleness, reconnects, gaps)

Key principle: Every answer must include traceable evidence (E1, E2, etc.)
"""

from .evidence import (
    EvidenceBuilder,
    Evidence,
    EvidenceType,
    get_evidence_builder
)
from .router import (
    IntentRouter,
    Intent,
    IntentCategory,
    get_intent_router
)
from .context import (
    ContextBuilder,
    ContextPack,
    ScreenContext
)
from .chat import (
    AteneaChat,
    ChatMessage,
    ChatResponse,
    get_atenea_chat
)

__all__ = [
    # Evidence
    "EvidenceBuilder",
    "Evidence",
    "EvidenceType",
    "get_evidence_builder",
    # Router
    "IntentRouter",
    "Intent",
    "IntentCategory",
    "get_intent_router",
    # Context
    "ContextBuilder",
    "ContextPack",
    "ScreenContext",
    # Chat
    "AteneaChat",
    "ChatMessage",
    "ChatResponse",
    "get_atenea_chat",
]
