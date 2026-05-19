"""BGE-M3 dense retriever (the MARGO primary).

Plug-and-play wrapper around the pretrained ``BAAI/bge-m3`` model + FAISS.
Index construction lives in :mod:`scripts.build_index`; this module is
strictly the *online* path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from margo.retrieval.base import RetrievalHit

log = logging.getLogger(__name__)


class BGERetriever:
    """Dense retriever with optional directive-aware querying.

    Parameters
    ----------
    index_path:
        Path to the FAISS index produced by ``scripts/build_index.py``.
    item_ids:
        ``item_id`` for each row in the FAISS index, in insertion order.
    model_name:
        Override for the underlying BGE checkpoint.
    """

    def __init__(
        self,
        index_path: str | Path,
        item_ids: Sequence[str],
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: str = "cuda:0",
    ) -> None:
        import faiss  # type: ignore

        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "BGE-M3 requires `pip install FlagEmbedding`."
            ) from e

        # Force single-device mode for online queries. Multi-GPU encode
        # crashes when the input batch (typically 1 sentence) is smaller
        # than the number of GPUs, causing empty sub-batches.
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=[device])
        self.index = faiss.read_index(str(index_path))
        self.item_ids = list(item_ids)
        self._faiss = faiss
        if self.index.ntotal != len(self.item_ids):
            raise ValueError(
                f"FAISS index ntotal={self.index.ntotal} does not match "
                f"len(item_ids)={len(self.item_ids)}"
            )

    # ------------------------------------------------------------------ #
    # Online API                                                           #
    # ------------------------------------------------------------------ #

    def _embed(self, queries: Sequence[str]) -> np.ndarray:
        out = self.model.encode(list(queries))["dense_vecs"]
        emb = np.asarray(out, dtype="float32")
        self._faiss.normalize_L2(emb)
        return emb

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalHit]:
        emb = self._embed([query])
        scores, indices = self.index.search(emb, k)
        return [
            RetrievalHit(item_id=self.item_ids[i], score=float(s))
            for i, s in zip(indices[0], scores[0])
            if i >= 0
        ]

    def retrieve_with_directive(
        self,
        user_profile: str,
        directive_nl: str,
        k: int = 100,
        trend_keywords: Optional[Sequence[str]] = None,
    ) -> list[RetrievalHit]:
        """Compose a single semantic query from (profile, directive, trend).

        This is the part that makes the demo's *candidate pool changes when
        the MD edits the directive* moment work — Step 2.3 of the Plan.
        Directive is placed first and repeated to give it higher semantic weight.
        """
        parts = []
        if directive_nl:
            parts.append(f"Shopping intent: {directive_nl.strip()}")
        # Truncate profile to avoid drowning the directive signal
        profile_short = user_profile.strip()[:300]
        parts.append(f"Shopper style: {profile_short}")
        if trend_keywords:
            parts.append("Trending: " + ", ".join(trend_keywords[:10]))
        if directive_nl:
            parts.append(f"Must include: {directive_nl.strip()}")
        return self.retrieve("\n".join(parts), k=k)
