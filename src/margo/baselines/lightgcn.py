"""LightGCN baseline retriever.

This is a thin wrapper that delegates training to RecBole if available;
otherwise it raises a clear error. The class implements the same
``retrieve``/``retrieve_with_directive`` API as the BGE retriever so the
orchestrator can swap retrievers via configuration (Plan §10.4 ablation F).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from margo.retrieval.base import RetrievalHit

log = logging.getLogger(__name__)


class LightGCNRetriever:
    """LightGCN-based candidate retriever (collaborative-filtering only).

    Notes
    -----
    LightGCN is *user-id keyed*, so ``retrieve_with_directive`` simply
    ignores the NL part. The orchestrator should be configured with
    ``use_item_self_describe=False`` when comparing against this baseline,
    otherwise the User Agent's NL reasoning unfairly boosts it.
    """

    def __init__(self, user_embeddings: np.ndarray, item_embeddings: np.ndarray,
                 user_ids: Sequence[str], item_ids: Sequence[str]) -> None:
        self.user_emb = user_embeddings
        self.item_emb = item_embeddings
        self.user_index = {u: i for i, u in enumerate(user_ids)}
        self.item_ids = list(item_ids)

    @classmethod
    def from_npz(cls, path: str | Path) -> "LightGCNRetriever":
        data = np.load(path, allow_pickle=True)
        return cls(
            user_embeddings=data["user_emb"],
            item_embeddings=data["item_emb"],
            user_ids=list(data["user_ids"]),
            item_ids=list(data["item_ids"]),
        )

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalHit]:
        # `query` is interpreted as a user_id (LightGCN is CF-only).
        if query not in self.user_index:
            log.warning("LightGCNRetriever: unknown user %s", query)
            return []
        scores = self.item_emb @ self.user_emb[self.user_index[query]]
        top = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [RetrievalHit(item_id=self.item_ids[i], score=float(scores[i])) for i in top]

    def retrieve_with_directive(
        self,
        user_profile: str,
        directive_nl: str,
        k: int = 100,
        **_,
    ) -> list[RetrievalHit]:
        # `user_profile` here must be the user_id; CF baselines cannot use NL.
        return self.retrieve(user_profile, k=k)


def train_lightgcn(*_, **__):  # pragma: no cover — relies on RecBole
    raise NotImplementedError(
        "Use scripts/train_lightgcn.py to train a LightGCN baseline. "
        "This stub exists to pin the public API only."
    )
