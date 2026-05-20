"""Common retriever interface used by MARGO Phase 3 and by ablation studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class RetrievalHit:
    item_id: str
    score: float


class BaseRetriever(Protocol):
    """All retrievers expose these two methods (Step 2.3 of the Plan)."""

    def retrieve(self, query: str, k: int = 100) -> Sequence[RetrievalHit]: ...

    def retrieve_with_directive(
        self,
        user_profile: str,
        directive_nl: str,
        k: int = 100,
    ) -> Sequence[RetrievalHit]: ...
