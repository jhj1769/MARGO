"""Message protocol primitives shared by every MARGO agent."""

from margo.protocol.messages import (
    Directive,
    Message,
    MessageType,
    RankedItem,
    Rationale,
    RecommendationResult,
    TrendInterpretation,
    ValidationReport,
)
from margo.protocol.router import MessageBus

__all__ = [
    "Directive",
    "Message",
    "MessageType",
    "RankedItem",
    "Rationale",
    "RecommendationResult",
    "TrendInterpretation",
    "ValidationReport",
    "MessageBus",
]
