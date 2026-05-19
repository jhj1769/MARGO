"""Public façade — ``api.MargoEngine`` for the web demo & evaluation.

This module ties together every other module so external callers (the
FastAPI backend, scripts, notebooks) only ever depend on a single
``MargoEngine`` object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from agents.expert_agent import ExpertAgent
from agents.item_agent import ItemFacts
from agents.trend_agent import TrendAgent
from agents.user_agent import UserAgent
from domains.fashion.loader import load_processed
from domains.fashion.personas import EXPERT_PERSONA
from domains.fashion.vocabulary import build_attribute_table, build_fashion_vocabulary
from grounding.schema_validator import SchemaValidator
from grounding.snapshot import TrendSnapshotStore
from grounding.vocabulary import Vocabulary
from lifecycle.orchestrator import MargoOrchestrator, MargoRunConfig
from llm import LLMClient, PromptRegistry, get_default_client
from protocol.messages import RecommendationResult
from protocol.router import MessageBus

log = logging.getLogger(__name__)


@dataclass
class MargoEngineConfig:
    processed_dir: Path
    domain: str = "fashion"
    time_window: str = "2026-Q2"
    expert_persona: str = EXPERT_PERSONA
    snapshot_dir: Optional[Path] = None
    bm25_only: bool = False  # set True when BGE-M3 is not yet built


class MargoEngine:
    """Stateful façade. Keep one instance alive per process."""

    def __init__(self, cfg: MargoEngineConfig, *, llm: Optional[LLMClient] = None) -> None:
        self.cfg = cfg
        self.llm = llm or get_default_client()
        self.prompts = PromptRegistry()
        self.validator = SchemaValidator()
        self.bus = MessageBus()

        tables = load_processed(cfg.processed_dir)
        log.info("Loaded %d items / %d interactions", len(tables.items), len(tables.train))
        self.catalog: dict[str, ItemFacts] = {}
        self.item_attrs: dict[str, dict[str, Any]] = build_attribute_table(tables.items)
        for _, row in tables.items.iterrows():
            iid = str(row.get("parent_asin") or row.get("item_id"))
            self.catalog[iid] = ItemFacts(
                item_id=iid,
                title=str(row.get("title", "")),
                attributes=self.item_attrs.get(iid, {}),
            )

        self.vocabulary: Vocabulary = build_fashion_vocabulary(tables.items)

        # Retriever — BGE by default, BM25 as cheap fallback.
        self.retriever = self._build_retriever(cfg, tables.items)

        snapshot_store = (
            TrendSnapshotStore(cfg.snapshot_dir) if cfg.snapshot_dir else None
        )

        self.expert = ExpertAgent(
            persona=cfg.expert_persona,
            vocabulary={k: sorted(v) for k, v in self.vocabulary.buckets.items()},
            llm=self.llm,
            prompts=self.prompts,
            bus=self.bus,
            schema_validator=self.validator,
        )
        self.trend = TrendAgent(
            domain=cfg.domain,
            time_window=cfg.time_window,
            vocabulary=self.vocabulary,
            snapshot_store=snapshot_store,
            gtrends_snapshot_dir=cfg.snapshot_dir,
            llm=self.llm,
            prompts=self.prompts,
            bus=self.bus,
            schema_validator=self.validator,
        )

        self.orchestrator = MargoOrchestrator(
            expert=self.expert,
            trend=self.trend,
            retriever=self.retriever,
            catalog=self.catalog,
            item_attrs=self.item_attrs,
            bus=self.bus,
        )

        # User histories are looked up lazily by user_id. The NL-string list
        # is what the User Agent reads; the parallel parent_asin list is what
        # the web demo uses to render item cards (image, category, price).
        self._user_histories, self._user_history_items = _build_user_history_table(
            tables.train, tables.items
        )

    # ------------------------------------------------------------------ #
    # Public                                                              #
    # ------------------------------------------------------------------ #

    def recommend(
        self,
        user_id: str,
        brief: str,
        *,
        config: Optional[MargoRunConfig] = None,
    ) -> RecommendationResult:
        history = self._user_histories.get(user_id, [])
        user = UserAgent(
            user_id=user_id,
            history=history,
            llm=self.llm,
            prompts=self.prompts,
            bus=self.bus,
            schema_validator=self.validator,
        )
        return self.orchestrator.recommend(user, brief=brief, config=config)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _build_retriever(self, cfg: MargoEngineConfig, items: pd.DataFrame):
        index_path = cfg.processed_dir / "faiss_index.bin"
        ids_path = cfg.processed_dir / "item_ids.txt"

        if not cfg.bm25_only and index_path.exists() and ids_path.exists():
            import os

            from retrieval.bge_retriever import BGERetriever

            item_ids = ids_path.read_text(encoding="utf-8").splitlines()
            log.info("Using BGE-M3 retriever (%d items)", len(item_ids))
            device = os.getenv("MARGO_BGE_DEVICE", "cuda:0")
            return BGERetriever(index_path=index_path, item_ids=item_ids, device=device)

        from domains.fashion.loader import build_item_text
        from retrieval.bm25_retriever import BM25Retriever

        log.warning("BGE-M3 index not found; falling back to BM25.")
        texts = [build_item_text(r) for _, r in items.iterrows()]
        ids = [str(r.get("parent_asin") or r.get("item_id")) for _, r in items.iterrows()]
        return BM25Retriever(item_ids=ids, texts=texts)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _build_user_history_table(
    train: pd.DataFrame, items: pd.DataFrame
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Render each user's history.

    Returns two parallel maps keyed by user_id:
      * ``texts``    — NL strings the User Agent reads ("Title | price=$X")
      * ``item_ids`` — the underlying ``parent_asin`` sequence (chronological)

    The lists are kept index-aligned so callers (e.g. the web demo) can
    safely zip them.
    """
    item_text_lookup = {
        str(r["parent_asin"]): (
            f"{r.get('title', '')} | price=${r.get('price', '?')}"
        )
        for _, r in items.iterrows()
    }
    texts: dict[str, list[str]] = {}
    ids: dict[str, list[str]] = {}
    for uid, group in train.sort_values("timestamp").groupby("user_id"):
        seq = [str(i) for i in group["item_id"]]
        ids[str(uid)] = seq
        texts[str(uid)] = [item_text_lookup.get(i, "[unknown item]") for i in seq]
    return texts, ids
