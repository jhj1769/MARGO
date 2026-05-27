"""Tests for Phase A — Cohort-Conditional Trend Application.

The function under test is module-level ``apply_cohort_conditioning``
in ``core.agents.trend_agent``. It re-weights a global
:class:`TrendInterpretation` per user cohort by dropping cohort-
conflicting attributes/keywords. No LLM, no learning — pure rule-based.

What we verify:
  * No-op behaviour (empty cohort, malformed signature) — backward compat
  * Conflict-driven drop (explicit pairs in the conflict table)
  * Non-conflicting attributes are preserved
  * Multi-axis cohort combines conflicts (intersection of drop rules)
  * ``summary`` and ``raw_sources`` are NEVER touched (provenance preserved)
  * Lowercase / case-insensitive matching
"""

from __future__ import annotations

from core.agents.trend_agent import (
    _COHORT_CONFLICT_TABLE,
    _conflicts_with_cohort,
    _parse_cohort_signature,
    apply_cohort_conditioning,
)
from core.protocol.messages import TrendInterpretation


def _interp(
    rising: dict[str, list[str]] | None = None,
    keywords: list[str] | None = None,
) -> TrendInterpretation:
    return TrendInterpretation(
        domain="fashion",
        time_window="2025-Q2",
        summary="luxury minimalism rising; loud maximalism stable",
        keywords=keywords or [],
        rising_attributes=rising or {},
        raw_sources=["multisource://test"],
    )


# --------------------------------------------------------------------------- #
# Cohort signature parsing                                                    #
# --------------------------------------------------------------------------- #


def test_parse_cohort_signature_basic():
    sig = "bra:brand-diverse|cat:balanced|pri:mid-tier|sty:minimal-casual"
    parsed = _parse_cohort_signature(sig)
    assert parsed == {
        "bra": "brand-diverse",
        "cat": "balanced",
        "pri": "mid-tier",
        "sty": "minimal-casual",
    }


def test_parse_cohort_signature_handles_malformed():
    # Empty parts, missing colons, trailing pipes — all ignored gracefully
    assert _parse_cohort_signature("") == {}
    assert _parse_cohort_signature("|||") == {}
    assert _parse_cohort_signature("foo") == {}  # no colon
    assert _parse_cohort_signature("pri:budget|broken") == {"pri": "budget"}


# --------------------------------------------------------------------------- #
# Conflict matcher                                                            #
# --------------------------------------------------------------------------- #


def test_conflicts_with_cohort_price_tier():
    cohort = {"pri": "budget"}
    assert _conflicts_with_cohort("luxury knitwear", cohort) is True
    assert _conflicts_with_cohort("premium leather", cohort) is True
    assert _conflicts_with_cohort("denim", cohort) is False


def test_conflicts_with_cohort_style_opposites():
    cohort = {"sty": "minimal-casual"}
    assert _conflicts_with_cohort("maximalist prints", cohort) is True
    assert _conflicts_with_cohort("flashy", cohort) is True
    assert _conflicts_with_cohort("clean lines", cohort) is False


def test_conflicts_with_cohort_case_insensitive():
    cohort = {"pri": "budget"}
    assert _conflicts_with_cohort("LUXURY", cohort) is True
    assert _conflicts_with_cohort("HiGh-EnD", cohort) is True


def test_conflicts_with_cohort_empty_attribute_is_safe():
    assert _conflicts_with_cohort("", {"pri": "budget"}) is False
    assert _conflicts_with_cohort(None, {"pri": "budget"}) is False  # type: ignore[arg-type]


def test_conflicts_with_cohort_unknown_axis_value_is_noop():
    # Axis values not in the conflict table mean *no opinion* — never drop
    cohort = {"sty": "nonexistent-style"}
    assert _conflicts_with_cohort("luxury", cohort) is False
    assert _conflicts_with_cohort("anything", cohort) is False


# --------------------------------------------------------------------------- #
# apply_cohort_conditioning                                                   #
# --------------------------------------------------------------------------- #


def test_apply_no_op_when_signature_empty():
    interp = _interp(rising={"style": ["luxury"]}, keywords=["premium"])
    out = apply_cohort_conditioning(interp, "")
    assert out.rising_attributes == {"style": ["luxury"]}
    assert out.keywords == ["premium"]


def test_apply_no_op_when_cohort_unknown():
    interp = _interp(rising={"style": ["luxury"]}, keywords=["premium"])
    # Cohort has values not in the conflict table → nothing to drop
    out = apply_cohort_conditioning(interp, "sty:totally-novel|pri:totally-novel")
    assert out.rising_attributes == {"style": ["luxury"]}
    assert out.keywords == ["premium"]


def test_apply_drops_conflicting_style_attributes():
    interp = _interp(rising={
        "style": ["minimal", "maximalist", "flashy"],
        "color": ["beige", "neon"],
    })
    out = apply_cohort_conditioning(interp, "sty:minimal-casual")
    # "maximalist" and "flashy" conflict with minimal-casual; "minimal" survives
    assert out.rising_attributes["style"] == ["minimal"]
    # "color" axis has no cohort conflict for these values → preserved
    assert out.rising_attributes["color"] == ["beige", "neon"]


def test_apply_drops_conflicting_keywords():
    interp = _interp(
        rising={"style": ["minimal"]},
        keywords=["minimal silhouette", "luxury knitwear", "budget tee"],
    )
    out = apply_cohort_conditioning(interp, "pri:budget")
    # "luxury knitwear" conflicts; the other two survive
    assert "luxury knitwear" not in out.keywords
    assert "minimal silhouette" in out.keywords
    assert "budget tee" in out.keywords


def test_apply_drops_empty_axis_entries():
    """If every attribute under an axis gets dropped, drop the axis entry too."""
    interp = _interp(rising={
        "style": ["maximalist", "flashy"],   # both conflict with minimal-casual
        "fit": ["relaxed"],                  # survives
    })
    out = apply_cohort_conditioning(interp, "sty:minimal-casual")
    assert "style" not in out.rising_attributes  # axis fully dropped
    assert out.rising_attributes["fit"] == ["relaxed"]


def test_apply_multi_axis_cohort_combines_conflicts():
    """A budget × streetwear cohort drops both luxury *and* preppy items."""
    interp = _interp(rising={
        "style": ["streetwear-edge", "preppy-cardigan", "luxury-baroque"],
    })
    out = apply_cohort_conditioning(interp, "pri:budget|sty:streetwear")
    # 'preppy-cardigan' dropped by sty:streetwear (preppy is its opposite)
    # 'luxury-baroque' dropped by pri:budget (luxury keyword)
    # 'streetwear-edge' survives — it contains 'streetwear' which conflicts
    #   with sty:preppy if that were the cohort, but here the cohort IS
    #   streetwear, so no conflict applies to it.
    assert out.rising_attributes["style"] == ["streetwear-edge"]


def test_apply_preserves_summary_and_raw_sources():
    """provenance fields must NEVER be modified — they describe evidence."""
    interp = _interp(rising={"style": ["luxury"]})
    out = apply_cohort_conditioning(interp, "pri:budget")
    assert out.summary == interp.summary
    assert out.raw_sources == interp.raw_sources
    assert out.domain == interp.domain
    assert out.time_window == interp.time_window


def test_apply_returns_new_object_does_not_mutate():
    """apply_cohort_conditioning must return a copy; original untouched."""
    interp = _interp(rising={"style": ["luxury", "minimal"]})
    original_rising = dict(interp.rising_attributes)
    _ = apply_cohort_conditioning(interp, "pri:budget")
    assert interp.rising_attributes == original_rising  # original unchanged


# --------------------------------------------------------------------------- #
# Sanity: conflict table itself                                                #
# --------------------------------------------------------------------------- #


def test_conflict_table_uses_lowercase_keys_and_values():
    """The conflict table must be all-lowercase so case-insensitive
    matching in ``_conflicts_with_cohort`` works."""
    for cohort_value, conflict_set in _COHORT_CONFLICT_TABLE.items():
        assert cohort_value == cohort_value.lower(), f"key not lower: {cohort_value}"
        for c in conflict_set:
            assert c == c.lower(), f"value not lower: {c}"
