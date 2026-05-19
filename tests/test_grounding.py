"""Grounding-layer tests: vocabulary drift + schema validator + snapshot cache."""

from __future__ import annotations

from margo.grounding.schema_validator import SchemaValidator
from margo.grounding.snapshot import TrendSnapshotStore
from margo.grounding.vocabulary import Vocabulary
from margo.protocol.messages import TrendInterpretation


def test_vocabulary_check_text():
    vocab = Vocabulary({"silhouette": {"trench-coat", "blazer"}, "color": {"beige", "navy"}})
    in_n, out_n, out_tokens = vocab.check_text(
        "A beige trench-coat with a sharp blazer cut and unobtrusive neon panel"
    )
    assert in_n >= 3
    assert "neon" in out_tokens
    assert "panel" in out_tokens


def test_schema_validator_counts_failures():
    from pydantic import BaseModel

    class S(BaseModel):
        x: int

    v = SchemaValidator()
    v.parse({"x": 1}, S, agent_id="a")
    try:
        v.parse({"x": "nope"}, S, agent_id="a")
    except Exception:
        pass
    assert v.counter.total == 2
    assert v.counter.violations == 1
    assert 0.49 < v.counter.rate() < 0.51


def test_snapshot_roundtrip(tmp_path):
    store = TrendSnapshotStore(tmp_path)
    interp = TrendInterpretation(
        domain="fashion",
        time_window="2026-Q2",
        summary="Earth-tones rising; tailored silhouettes dominate.",
        keywords=["earth-tone", "tailored"],
    )
    store.put(interp)
    again = store.get("fashion", "2026-Q2")
    assert again is not None
    assert again.summary == interp.summary
