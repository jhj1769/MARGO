"""L4 — Keyword → structured fashion attributes (Enhancement 4).

Why this layer exists: a raw trend keyword like ``"y2k revival"`` says nothing
about which items in OUR catalogue it touches. Without an explicit semantic
mapping, the downstream Item Agent has to guess — which is exactly the
hallucination we're trying to remove.

Mapping strategy (hybrid):

    1. LLM proposes structured attributes
       ``{style: [...], category: [...], era: "...", material_or_color: [...]}``.
    2. For each proposed attribute, retrieve the top-K nearest items from the
       catalogue using BGE-M3 (or any callable retriever).
    3. Drop attribute terms that find zero catalogue matches above a
       similarity floor — these are model hallucinations with no grounding.
    4. Persist the resulting :class:`TrendSemantic` to a cache JSON keyed by
       a hash of the keyword so re-runs are free.

The retriever is injected (duck-typed). For tests we pass a synthetic
matcher and never hit BGE-M3.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Optional, Protocol

from pydantic import BaseModel, Field

from adapters.trends.multisource_schema import TrendSemantic

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public types                                                                #
# --------------------------------------------------------------------------- #


class _LLMSemanticOut(BaseModel):
    """LLM-side schema for the proposal step."""

    style: list[str] = Field(default_factory=list)
    category: list[str] = Field(default_factory=list)
    era_reference: Optional[str] = None
    material_or_color: list[str] = Field(default_factory=list)


class CatalogMatcher(Protocol):
    """Callable interface used to validate proposed attributes.

    Should return ``[(item_id, similarity), ...]`` sorted descending by
    similarity. ``query`` is the proposed attribute term.
    """

    def __call__(self, query: str, k: int) -> list[tuple[str, float]]: ...


# --------------------------------------------------------------------------- #
# Cache                                                                       #
# --------------------------------------------------------------------------- #


_CACHE_DIR_NAME = "trend_semantics"


def _semantic_path(keyword: str, processed_dir: Path) -> Path:
    digest = hashlib.sha1(keyword.encode("utf-8")).hexdigest()[:16]
    return Path(processed_dir) / _CACHE_DIR_NAME / f"{digest}.json"


def load_semantic_from_cache(
    keyword: str, processed_dir: Path
) -> Optional[TrendSemantic]:
    path = _semantic_path(keyword, processed_dir)
    if not path.exists():
        return None
    try:
        return TrendSemantic.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("failed to read semantic cache %s: %s", path, e)
        return None


def save_semantic_to_cache(
    semantic: TrendSemantic, processed_dir: Path
) -> Path:
    path = _semantic_path(semantic.keyword, processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(semantic.model_dump_json(indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #


_LLM_PROMPT = """
Map this fashion trend keyword to structured catalog-grounded attributes.

Keyword: {keyword}
Domain : fashion

Produce JSON:
{{
  "style": [...],              // concrete style descriptors (e.g. "low-rise", "tailored")
  "category": [...],           // item categories likely affected (e.g. "jeans", "blazer")
  "era_reference": "...",      // optional decade/era anchor (e.g. "2000s")
  "material_or_color": [...]   // optional material/color (e.g. "metallics", "satin")
}}

Use only concrete, catalogue-likely terms. If you cannot ground a field,
return an empty list (or null for era_reference). Avoid abstract marketing
phrases like "fresh vibes" or "modern energy".
""".strip()


def build_semantic_mapping(
    keyword: str,
    *,
    matcher: CatalogMatcher,
    llm_propose: Callable[[str], _LLMSemanticOut],
    min_catalog_matches: int = 2,
    similarity_floor: float = 0.30,
    top_k_match: int = 10,
    processed_dir: Optional[Path] = None,
) -> TrendSemantic:
    """Build a single keyword's semantic mapping (with optional caching).

    Parameters
    ----------
    matcher
        Function from query → ``[(item_id, similarity)]`` (already sorted).
        In production this wraps a BGE-M3 retriever; in tests it's a
        deterministic stub.
    llm_propose
        Function that takes the rendered prompt and returns a parsed
        ``_LLMSemanticOut``. Injecting the LLM call this way keeps the
        module independent of which backend is used.
    min_catalog_matches
        Drop a proposed attribute term if it finds fewer than this many
        items above ``similarity_floor``.
    """
    # Cache shortcut.
    if processed_dir is not None:
        cached = load_semantic_from_cache(keyword, processed_dir)
        if cached is not None:
            return cached

    # Step 1 — LLM proposes structured attributes.
    try:
        proposal = llm_propose(_LLM_PROMPT.format(keyword=keyword))
    except Exception as e:  # noqa: BLE001
        log.warning("semantic LLM proposal failed for %r: %s", keyword, e)
        return TrendSemantic(
            keyword=keyword,
            fashion_attributes={},
            catalog_match_method="fallback",
            matched_item_ids=[],
            confidence=0.0,
        )

    # Step 2 — validate each proposed term against the catalogue.
    grounded: dict[str, list[str]] = {}
    matched_ids: set[str] = set()
    total_terms_proposed = 0
    total_terms_kept = 0
    for axis_name, terms in (
        ("style", proposal.style),
        ("category", proposal.category),
        ("material_or_color", proposal.material_or_color),
    ):
        kept: list[str] = []
        for term in terms:
            term = (term or "").strip().lower()
            if not term:
                continue
            total_terms_proposed += 1
            hits = matcher(term, top_k_match)
            strong_hits = [(iid, s) for iid, s in hits if s >= similarity_floor]
            if len(strong_hits) >= min_catalog_matches:
                kept.append(term)
                matched_ids.update(iid for iid, _ in strong_hits)
                total_terms_kept += 1
        if kept:
            grounded[axis_name] = kept

    if proposal.era_reference:
        # Era references aren't catalog terms — keep as a tag without validation.
        grounded["era_reference"] = [proposal.era_reference.strip().lower()]

    # Confidence: fraction of proposed terms that survived validation.
    confidence = (
        total_terms_kept / total_terms_proposed if total_terms_proposed else 0.0
    )

    out = TrendSemantic(
        keyword=keyword,
        fashion_attributes=grounded,
        catalog_match_method="hybrid",
        matched_item_ids=sorted(matched_ids),
        confidence=round(confidence, 4),
    )
    if processed_dir is not None:
        save_semantic_to_cache(out, processed_dir)
    return out


# --------------------------------------------------------------------------- #
# Convenience wrappers                                                        #
# --------------------------------------------------------------------------- #


def make_llm_proposer(llm_client) -> Callable[[str], _LLMSemanticOut]:
    """Wrap an ``LLMClient`` so it returns ``_LLMSemanticOut`` directly.

    Used by the pipeline so callers don't have to thread ``_LLMSemanticOut``
    through their own code. Kept here (not in the LLM module) because the
    schema is private to L4.
    """
    def _propose(prompt: str) -> _LLMSemanticOut:
        return llm_client.complete_structured(
            prompt,
            _LLMSemanticOut,
            system="You output strict JSON describing fashion catalog attributes.",
            agent_id="trend-semantic-mapper",
        )
    return _propose


def make_bge_matcher(retriever) -> CatalogMatcher:
    """Adapt a BGE-style retriever to the ``CatalogMatcher`` protocol."""
    def _match(query: str, k: int) -> list[tuple[str, float]]:
        hits = retriever.retrieve(query, k=k)
        return [(h.item_id, float(h.score)) for h in hits]
    return _match
