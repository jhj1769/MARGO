"""Standard + governance + grounding metric correctness."""

from __future__ import annotations

from margo.evaluation.governance import dcr, tas
from margo.evaluation.grounding import ihr, vdr
from margo.evaluation.standard import hit_rate, ndcg
from margo.grounding.vocabulary import Vocabulary
from margo.protocol.messages import (
    Directive,
    RankedItem,
    Rationale,
    TrendInterpretation,
)


def _r(id: str, s: float = 0.5) -> RankedItem:
    return RankedItem(item_id=id, score=s, rationale=Rationale(personal="p", directive="d", trend="t"))


def test_hr_and_ndcg_basic():
    ranked = ["A", "B", "C", "D", "E"]
    assert hit_rate(ranked, "C", k=5) == 1.0
    assert hit_rate(ranked, "Z", k=5) == 0.0
    # NDCG@k with relevance at rank 3 (idx=2) → 1/log2(4) = 0.5
    assert abs(ndcg(ranked, "C", k=5) - 0.5) < 1e-6


def test_dcr_flags_price_violation():
    d = Directive(
        goal="upsell",
        structured_constraints={"price_diff_pct_max": 30},
        natural_language="keep within 30 percent of user's typical price.",
    )
    ranked = [_r("A"), _r("B")]
    item_attrs = {"A": {"price": 100.0}, "B": {"price": 300.0}}
    user_attrs = {"avg_price": 100.0}
    score = dcr(ranked, directive=d, item_attrs=item_attrs, user_attrs=user_attrs)
    assert score == 0.5  # only "A" passes


def test_tas_keyword_overlap():
    t = TrendInterpretation(
        domain="fashion", time_window="2026-Q2", summary="x",
        keywords=["beige", "trench"],
    )
    ranked = [_r("A"), _r("B")]
    item_attrs = {
        "A": {"color": "beige", "category": ["outerwear"]},
        "B": {"color": "red", "category": ["pants"]},
    }
    assert tas(ranked, trend=t, item_attrs=item_attrs) == 0.5


def test_ihr_counts_unknown_items():
    ranked = [_r("A"), _r("Z")]
    assert ihr(ranked, catalog_ids={"A"}) == 0.5


def test_vdr_on_drifting_text():
    vocab = Vocabulary({"silhouette": {"trench-coat"}, "color": {"beige"}})
    drift_text = "neon micro-skirt with synthwave panel"
    clean_text = "beige trench-coat"
    high = vdr([drift_text], vocab)
    low = vdr([clean_text], vocab)
    assert high > low
