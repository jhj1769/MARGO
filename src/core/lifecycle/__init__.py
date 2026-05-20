"""4-phase lifecycle orchestration.

We expose a thin ``MargoOrchestrator`` that runs Phase 2 → 3 → 4 with the
refine-loop wired in. The implementation does NOT hard-depend on LangGraph;
the linear control flow is small enough that adding a state-machine library
would only add ceremony. Switching to LangGraph later is a localised change
inside :class:`MargoOrchestrator.run`.
"""

from core.lifecycle.orchestrator import MargoOrchestrator, MargoRunConfig

__all__ = ["MargoOrchestrator", "MargoRunConfig"]
