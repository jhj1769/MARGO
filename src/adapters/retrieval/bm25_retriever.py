"""BM25 sparse retriever — baseline + sanity check.

Uses ``rank-bm25`` if installed, otherwise falls back to a tiny pure-python
implementation that is good enough for unit tests (a few hundred items).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional, Sequence

from adapters.retrieval.base import RetrievalHit


_TOK = re.compile(r"[a-z0-9][a-z0-9\-]*")


def _tokenise(text: str) -> list[str]:
    return _TOK.findall(text.lower())


class BM25Retriever:
    """Drop-in baseline mirroring the BGE-M3 interface."""

    def __init__(self, item_ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(item_ids) != len(texts):
            raise ValueError("item_ids and texts length mismatch")
        self.item_ids = list(item_ids)
        self.docs = [_tokenise(t) for t in texts]

        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._bm25 = BM25Okapi(self.docs)
            self._mode = "rank_bm25"
        except ImportError:
            self._bm25 = None
            self._mode = "fallback"
            self._build_fallback_index()

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalHit]:
        q = _tokenise(query)
        if self._mode == "rank_bm25":
            scores = self._bm25.get_scores(q)  # type: ignore[union-attr]
        else:
            scores = self._fallback_scores(q)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [RetrievalHit(item_id=self.item_ids[i], score=float(scores[i])) for i in top_idx]

    def retrieve_with_directive(
        self,
        user_profile: str,
        directive_nl: str,
        k: int = 100,
        trend_keywords: Optional[Sequence[str]] = None,
    ) -> list[RetrievalHit]:
        parts = [user_profile, directive_nl]
        if trend_keywords:
            parts.append("Trending: " + ", ".join(trend_keywords[:10]))
        return self.retrieve("\n".join(parts), k=k)

    # ------------------------------------------------------------------ #
    # Tiny BM25 fallback (avoids hard dep on rank_bm25 in unit tests)     #
    # ------------------------------------------------------------------ #

    def _build_fallback_index(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        self._dl = [len(d) for d in self.docs]
        self._avgdl = sum(self._dl) / max(1, len(self._dl))
        df: Counter[str] = Counter()
        for d in self.docs:
            for tok in set(d):
                df[tok] += 1
        N = len(self.docs)
        self._idf = {
            tok: math.log(1 + (N - n + 0.5) / (n + 0.5)) for tok, n in df.items()
        }
        self._tf = [Counter(d) for d in self.docs]

    def _fallback_scores(self, q_tokens: list[str]) -> list[float]:
        k1, b = self._k1, self._b
        scores = [0.0] * len(self.docs)
        for i, tf in enumerate(self._tf):
            dl = self._dl[i] or 1
            for tok in q_tokens:
                if tok not in tf:
                    continue
                idf = self._idf.get(tok, 0.0)
                f = tf[tok]
                num = f * (k1 + 1)
                den = f + k1 * (1 - b + b * dl / self._avgdl)
                scores[i] += idf * num / den
        return scores
