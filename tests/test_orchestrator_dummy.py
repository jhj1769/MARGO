"""End-to-end smoke test using the dummy LLM backend.

This exercises every glue point (protocol, agents, lifecycle, retrieval,
metrics) without requiring an OpenAI / vLLM endpoint.
"""

from __future__ import annotations

import os

os.environ.setdefault("MARGO_LLM_BACKEND", "dummy")

from margo.agents.expert_agent import ExpertAgent
from margo.agents.item_agent import ItemFacts
from margo.agents.trend_agent import TrendAgent
from margo.agents.user_agent import UserAgent
from margo.grounding.vocabulary import Vocabulary
from margo.lifecycle.orchestrator import MargoOrchestrator, MargoRunConfig
from margo.llm import LLMClient
from margo.protocol.router import MessageBus
from margo.retrieval.bm25_retriever import BM25Retriever


def _build_catalog():
    raw = {
        "A": ("Beige trench coat", {"category": ["outerwear"], "color": "beige", "price": 110.0}),
        "B": ("Navy chinos", {"category": ["pants"], "color": "navy", "price": 60.0}),
        "C": ("Wool overcoat", {"category": ["outerwear"], "color": "charcoal", "price": 180.0}),
        "D": ("White cotton tee", {"category": ["tops"], "color": "white", "price": 25.0}),
    }
    catalog = {iid: ItemFacts(item_id=iid, title=title, attributes=attrs) for iid, (title, attrs) in raw.items()}
    item_attrs = {iid: attrs for iid, (_, attrs) in raw.items()}
    return catalog, item_attrs


def test_orchestrator_dummy_endtoend():
    llm = LLMClient(backend="dummy")
    bus = MessageBus()
    catalog, item_attrs = _build_catalog()
    texts = [f"{f.title}. category={f.attributes.get('category')} color={f.attributes.get('color')}"
             for f in catalog.values()]
    retriever = BM25Retriever(item_ids=list(catalog.keys()), texts=texts)

    vocab = Vocabulary(
        {
            "category": {"outerwear", "pants", "tops"},
            "color": {"beige", "navy", "charcoal", "white"},
            "silhouette": {"trench-coat", "chinos"},
        }
    )
    expert = ExpertAgent(persona="MD", vocabulary={k: sorted(v) for k, v in vocab.buckets.items()}, llm=llm, bus=bus)
    trend = TrendAgent(domain="fashion", time_window="2026-Q2", vocabulary=vocab, llm=llm, bus=bus)
    orchestrator = MargoOrchestrator(
        expert=expert,
        trend=trend,
        retriever=retriever,
        catalog=catalog,
        item_attrs=item_attrs,
        bus=bus,
    )

    user = UserAgent(
        user_id="u1",
        history=["bought white cotton tee | $25", "bought navy chinos | $60"],
        llm=llm,
        bus=bus,
    )
    res = orchestrator.recommend(
        user,
        brief="Casual→formal upsell. Boost outerwear.",
        config=MargoRunConfig(top_k=3, candidate_size=4, rerank_window=4, max_iterations=1),
        user_attrs={"avg_price": 42.5},
    )
    assert res.user_id == "u1"
    assert res.candidate_pool_size > 0
    assert res.iterations >= 1
    assert len(res.top_k) > 0  # dummy LLM responds with at least one stub item
