"""Retriever smoke test — exercises the BM25 baseline end-to-end."""

from __future__ import annotations

from retrieval.bm25_retriever import BM25Retriever


def test_bm25_returns_correct_top_hit():
    items = {
        "A": "Beige trench coat. Category: outerwear. Color: beige. Material: cotton.",
        "B": "Navy chinos. Category: pants. Color: navy. Material: cotton.",
        "C": "Wool overcoat. Category: outerwear. Color: charcoal. Material: wool.",
    }
    r = BM25Retriever(item_ids=list(items.keys()), texts=list(items.values()))
    hits = r.retrieve("beige trench coat", k=2)
    assert hits[0].item_id == "A"
    assert len(hits) == 2


def test_bm25_directive_query_includes_intent():
    items = {
        "A": "Beige trench coat. Category: outerwear.",
        "B": "Black t-shirt. Category: tops.",
    }
    r = BM25Retriever(item_ids=list(items.keys()), texts=list(items.values()))
    hits = r.retrieve_with_directive(
        user_profile="prefers casual cotton wear",
        directive_nl="boost outerwear for formal upsell",
        k=2,
    )
    assert hits[0].item_id == "A"
