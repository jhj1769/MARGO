# MARGO Enhancement — Final Implementation Directive (v3)

> **Target**: Enhance MARGO (current state: 2026-05-20) with agent-level reasoning upgrades.
> **Scope**: Methodology enhancement only. Evaluation metrics (TAS/DCR/etc) will be revisited separately later.
> **Baseline**: All work builds on existing 4-agent / 4-phase architecture. No framework restructure.

---

## 0. Changes from v2

- **DROPPED**: User-conditional trend broadcast (former Enhancement 4). Trend application is uniform across users. Rationale: separates "trend analysis quality" from "trend personalization" as contributions, simplifies ablation, removes hardest-to-validate component.
- **STRENGTHENED**: Former Enhancement 5 (multi-source adapter) is reorganized into a **5-layer Trend Analysis Pipeline** with explicit semantic mapping and keyword pool layers.
- **RENUMBERED**: Enhancements now 1, 1.5, 2, 3, 4, 5 (down from 7 items).
- **Trend flow simplified**: Item Agent uses `global_interp` (single per directive); no user-conditional re-weighting.

---

## 1. Context for Claude Code

Current MARGO state (as of 2026-05-20):
- 4 agents (User, Item, Expert, Trend) inheriting `BaseAgent`
- 4-phase lifecycle (init → directive → reasoning → validation+refine)
- Pydantic message protocol, in-process MessageBus
- BGE-M3 + FAISS retrieval, Google Trends snapshot store
- Pytest 18 passed
- Web demo (FastAPI + Next.js)

**Critical constraints**:
- Do NOT modify evaluation modules (`src/eval/`) in this round
- All schema extensions backward-compatible (Optional fields)
- All existing tests must continue passing

---

## 2. Enhancement Overview

| # | Enhancement | Agent | Effort | Week |
|---|---|---|---|---|
| 1 | Multi-axis preference state | User | M | 1 |
| 1.5 | Axis-derived cohort + peer signal | User | M | 2 |
| 2 | Audience profile inference | Item | M | 3 |
| 3 | Trend-grounded self-positioning | Item | S | 3 |
| 4 | Trend Analysis Pipeline (5-layer) | Trend | L | 4-5 |
| 5 | Expert ↔ Trend negotiation | Expert + Trend | L | 6-7 |
| - | Integration + Web demo update | All | M | 8 |

Total: ~8 weeks. Implement strictly in order.

**Removed from v2**: Enhancement 4 (user-conditional broadcast). Week 4 slot absorbed into expanded Trend Pipeline (now Enhancement 4).

---

## 3. Enhancement 1: User Agent Multi-Axis Preference

### 3.1 Concept

User preference decomposed into **4 explicit axes**:

| Axis | Meaning | Inference Method |
|---|---|---|
| `style_preference` | Style direction | LLM inference from item descriptions |
| `price_preference` | Price tier preference | Deterministic statistics (median + variance) |
| `category_preference` | Category distribution | Deterministic statistics |
| `brand_preference` | Brand loyalty / diversity | Deterministic statistics |

**Key principle**: Price/Category/Brand computed deterministically. Style is LLM-inferred. Reduces hallucination surface.

**Diverse user handling**: Each axis allows valid "mixed" / "balanced" / "diverse" values with appropriate confidence. Single labeling is never forced.

### 3.2 Axis Value Specs

**style_preference**
- Examples: `"minimal-casual"`, `"streetwear"`, `"preppy"`, `"feminine-romantic"`, `"athleisure"`, `"mixed"`
- Inference: LLM reads item descriptions/material/color
- Diverse user: `value="mixed"`, lower confidence, populate `secondary_values`

**price_preference**
- Computed from purchase history median:
  - `< $30` → `"budget"`
  - `$30-100` → `"mid-tier"`
  - `$100-300` → `"premium-open"`
  - `$300+` → `"luxury-aware"`
- High variance → lower confidence
- Output: `value` (dominant tier), `confidence` (1.0 - normalized variance)

**category_preference**
- Computed from category distribution:
  - Single category > 50% → `"{category}-focused"`
  - 2-3 categories with 25-50% each → `"{cat1}-{cat2}-mix"`
  - All major categories < 30% → `"balanced"`
- High confidence even when "balanced"

**brand_preference**
- Computed from brand distribution:
  - Single brand > 40% → `"brand-loyal:{brand_name}"`
  - Top 3 brands each < 30% → `"brand-diverse"`
  - Premium brand share dominant → `"premium-brand-curious"`

### 3.3 Schema (NEW)

**File**: `src/core/protocol/messages.py` (extend)

```python
from typing import Literal
from pydantic import BaseModel, Field

AxisName = Literal[
    "style_preference",
    "price_preference",
    "category_preference",
    "brand_preference",
]

class PreferenceAxis(BaseModel):
    name: AxisName
    value: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    secondary_values: list[str] = Field(default_factory=list)
    derived_from: Literal["statistical", "llm_inferred"]
    stability: float = Field(default=1.0, ge=0.0, le=1.0)

class UserPreferenceState(BaseModel):
    user_id: str
    profile_nl: str
    axes: list[PreferenceAxis]
    cohort_signature: str  # populated in Enhancement 1.5
    last_updated_at: str

    def get_axis(self, name: AxisName) -> PreferenceAxis | None:
        return next((a for a in self.axes if a.name == name), None)
```

### 3.4 Statistical Pre-computation

**Create**: `src/data/fashion/preference_stats.py`

```python
def compute_deterministic_axes(user_id: str, history: list[Interaction], items: dict) -> dict:
    """Compute price, category, brand axes from statistics.
    Returns dict with 3 axes (style left for LLM).
    Also computes stability by comparing recent vs full history.
    """
```

### 3.5 Prompt Changes

**Modify**: `prompts/phase1_initialization/user.system.j2` — add 4-axis explanation + provenance.

**Modify**: `prompts/phase3_reasoning/user.profile.j2`

Required output:
```json
{
  "profile_nl": "...",
  "axes": [
    {"name": "style_preference", "value": "...", "confidence": 0.X, "evidence": [...], "secondary_values": [...], "derived_from": "llm_inferred", "stability": 0.X},
    {"name": "price_preference", "...", "derived_from": "statistical", ...},
    {"name": "category_preference", "...", "derived_from": "statistical", ...},
    {"name": "brand_preference", "...", "derived_from": "statistical", ...}
  ]
}
```

**Modify**: `prompts/phase3_reasoning/user.evaluate.j2`

Per-axis evaluation (KAR-style):
```
For each of the 4 axes, output:
- match_score (0.0-1.0)
- one-sentence explanation

Then aggregate into final_score and three_layer_rationale (personal/directive/trend).
Personal layer must reference specific axes.
```

### 3.6 Agent Changes

**Modify**: `src/core/agents/user_agent.py`

```python
def build_profile(self, user_id: str, history: list[Interaction]) -> UserProfile:
    deterministic = compute_deterministic_axes(user_id, history, self.item_catalog)
    response = self.llm_call(prompt="user.profile", context={...})
    return UserProfile(preference_state=parsed_state, ...)

def evaluate_candidate(self, ...):
    # Use preference_state for per-axis evaluation
    ...

def update_preference_state(self, current: UserPreferenceState, interaction: Interaction):
    # Re-run deterministic for price/category/brand
    # Ask LLM for style + confidence reconciliation
    # Update stability based on new interaction
    ...
```

### 3.7 Tests

**Create**: `tests/test_user_preference_axes.py`
- `test_deterministic_axes_computation`
- `test_diverse_user_lower_confidence`
- `test_axis_schema_validation` → SVR increment on malformed
- `test_update_preference_state`
- `test_stability_score`

---

## 4. Enhancement 1.5: Axis-Derived Cohort + Peer Signal

### 4.1 Concept

User cohort emerges naturally from axis combinations (not from clustering algorithm).

**Two functions**:
- **Cohort Signature**: deterministic concat of axis values
- **Peer Signal**: collaborative signal from users sharing cohort

### 4.2 Cohort Signature

```python
def compute_cohort_signature(state: UserPreferenceState) -> str:
    sorted_axes = sorted(state.axes, key=lambda a: a.name)
    return "|".join(f"{a.name[:3]}:{a.value}" for a in sorted_axes)
    # Example: "bra:brand-diverse|cat:balanced|pri:mid-tier|sty:minimal-casual"
```

Populated in `UserPreferenceState.cohort_signature` during `build_profile`.

### 4.3 Cohort Stats (Offline)

**Create**: `scripts/build_cohort_stats.py`

```python
MIN_COHORT_SIZE = 5

all_users_with_state = load_all_user_states()
cohorts = defaultdict(list)
for user_state in all_users_with_state:
    cohorts[user_state.cohort_signature].append(user_state.user_id)

for cohort_sig, user_ids in cohorts.items():
    if len(user_ids) < MIN_COHORT_SIZE:
        continue

    item_buy_counts = defaultdict(int)
    for uid in user_ids:
        for interaction in history(uid):
            if interaction.rating >= 4.0:
                item_buy_counts[interaction.item_id] += 1

    cohort_stats = {
        "signature": cohort_sig,
        "size": len(user_ids),
        "user_ids": user_ids,
        "item_buy_ratios": {
            item_id: count / len(user_ids)
            for item_id, count in item_buy_counts.items()
        },
        "top_categories": ...,
        "top_brands": ...,
    }
    save(f"{processed_dir}/cohort_stats/{hash(cohort_sig)}.json", cohort_stats)
```

### 4.4 Cohort Coverage Diagnostic (NEW)

**Create**: `scripts/diagnose_cohort_coverage.py`

Run after Enhancement 1 completes. Report:
- Distribution of cohort sizes
- % of users falling into cohorts < MIN_COHORT_SIZE (fallback rate)
- Top-10 most populous cohort signatures

If fallback rate > 50%, axis values must be coarsened before proceeding (e.g., price 4 tiers → 3 tiers).

### 4.5 Schema

```python
class CohortStats(BaseModel):
    signature: str
    size: int
    item_buy_ratios: dict[str, float]
    top_categories: list[tuple[str, float]]
    top_brands: list[tuple[str, float]]

    def peer_signal_for(self, item_id: str) -> float:
        return self.item_buy_ratios.get(item_id, 0.0)
```

### 4.6 Loader

**Create**: `src/data/fashion/cohort_loader.py`

```python
def load_cohort_stats(signature: str, processed_dir: Path) -> CohortStats | None:
    path = processed_dir / "cohort_stats" / f"{hash_signature(signature)}.json"
    if not path.exists():
        return None
    return CohortStats.parse_file(path)
```

### 4.7 Agent Integration

**Modify**: `src/core/agents/user_agent.py`

```python
def get_peer_signal(
    self,
    state: UserPreferenceState,
    candidate_item_id: str,
) -> tuple[float, str]:
    cohort = load_cohort_stats(state.cohort_signature, self.processed_dir)
    if cohort is None or cohort.size < MIN_COHORT_SIZE:
        return 0.0, "Cohort too small for reliable peer signal"

    ratio = cohort.peer_signal_for(candidate_item_id)
    return ratio, f"{ratio:.0%} of users in your cohort ({cohort.size} users) purchased this item"
```

### 4.8 Prompt Changes

**Modify**: `prompts/phase3_reasoning/user.evaluate.j2`

```jinja
{% if peer_signal %}
Cohort peer signal:
{{ peer_signal.explanation_nl }}
Use this as one of multiple signals — high peer ratio is a positive indicator
but does not override your axes-based judgment.
{% endif %}
```

### 4.9 Tests

**Create**: `tests/test_cohort_peer_signal.py`
- `test_cohort_signature_deterministic`
- `test_peer_signal_lookup`
- `test_small_cohort_fallback`
- `test_signature_consistency`

---

## 5. Enhancement 2: Item Agent Audience Profile Inference

### 5.1 Concept

Each item maintains a `buyer_aggregate` profile derived from past buyers. Item Agent uses this to claim "who I serve well" in self-description.

With Enhancement 1.5, buyer aggregate can leverage cohort distribution.

Inspired by RecExplainer (KDD'24).

### 5.2 Offline Preprocessing

**Create**: `scripts/build_buyer_aggregate.py`

```python
MIN_BUYERS = 3

for item_id in catalog:
    buyers = interactions.filter(item_id=item_id, rating>=4.0).user_ids
    if len(buyers) < MIN_BUYERS:
        continue

    buyer_cohorts = [load_user_state(b).cohort_signature for b in buyers]
    cohort_dist = Counter(buyer_cohorts)

    aggregate = {
        "item_id": item_id,
        "buyer_count": len(buyers),
        "avg_price_history": ...,
        "median_history_length": ...,
        "category_distribution": ...,
        "brand_distribution": ...,
        "buyer_cohort_distribution": dict(cohort_dist),
        "dominant_cohorts": top_3_cohorts(cohort_dist),
        "evidence_buyer_ids": random.sample(buyers, min(5, len(buyers))),
    }
    save(...)
```

### 5.3 Schema

```python
class ItemAudienceProfile(BaseModel):
    item_id: str
    buyer_count: int
    avg_price_history: float | None
    median_history_length: int | None
    category_distribution: dict[str, float]
    brand_distribution: dict[str, float]
    buyer_cohort_distribution: dict[str, int]
    dominant_cohorts: list[str]
    evidence_buyer_ids: list[str]
    dominant_pattern_nl: str | None = None
    outlier_pattern_nl: str | None = None
```

### 5.4 Loader

**Create**: `src/data/fashion/audience_loader.py`

```python
def load_audience_profile(item_id: str, processed_dir: Path) -> ItemAudienceProfile | None:
    path = processed_dir / "buyer_aggregate" / f"{item_id}.json"
    if not path.exists():
        return None
    return ItemAudienceProfile.parse_file(path)
```

### 5.5 Prompt Changes

**Modify**: `prompts/phase3_reasoning/item.describe.j2`

```jinja
You are item {{ item_id }} with attributes: {{ attributes }}.

{% if audience %}
Your buyer aggregate:
- {{ audience.buyer_count }} verified buyers
- Buyer cohorts (top): {{ audience.dominant_cohorts }}
- Category distribution: {{ audience.category_distribution }}
- Brand distribution: {{ audience.brand_distribution }}
{% endif %}

Describe yourself in three parts:
1. attribute_summary (directive-aware: {{ directive }})
2. audience_fit_claim: who you serve well — grounded in cohort patterns
3. outlier_note (optional): if minority buyer cohort exists
```

### 5.6 Agent Changes

**Modify**: `src/core/agents/item_agent.py`

```python
def self_describe(
    self,
    item: Item,
    directive: Directive,
    audience_profile: ItemAudienceProfile | None = None,
    trend_interpretation: TrendInterpretation | None = None,  # global, see Enhancement 3 + 4
) -> ItemDescription:
    ...
```

### 5.7 Lifecycle Integration

**Modify**: `src/core/lifecycle/phase3_reasoning.py`

```python
global_interp = trend_agent.interpret_trend(snapshot, directive)  # single per directive

for user in user_batch:
    candidates = retriever.retrieve_with_directive(...)
    for item in candidates:
        audience = load_audience_profile(item.item_id, processed_dir)
        item_desc = item_agent.self_describe(
            item=item,
            directive=directive,
            audience_profile=audience,
            trend_interpretation=global_interp,  # uniform
        )
        score = user_agent.evaluate_candidate(
            user=user,
            item_desc=item_desc,
            directive=directive,
            trend_interpretation=global_interp,  # uniform
            peer_signal=user_agent.get_peer_signal(user.preference_state, item.item_id),
        )
```

### 5.8 Tests

**Create**: `tests/test_item_audience.py`
- Items below MIN_BUYERS → None from loader
- self_describe handles None gracefully
- Audience-populated description references cohort patterns

---

## 6. Enhancement 3: Item Agent Trend Self-Positioning

### 6.1 Schema Extension

**Extend**: `src/core/protocol/messages.py`

```python
class TrendPosition(BaseModel):
    lifecycle: Literal["rising", "stable", "declining", "niche"]
    alignment: Literal["aligned", "orthogonal", "counter"]
    value_proposition_nl: str | None = None

class ItemDescription(BaseModel):
    # ... existing fields
    trend_position: TrendPosition | None = None
```

### 6.2 Prompt

**Append to**: `prompts/phase3_reasoning/item.describe.j2`

```jinja
{% if trend_interpretation %}
Current trend signal (global): {{ trend_interpretation }}

Position yourself relative to this trend:
- lifecycle: rising | stable | declining | niche
- alignment: aligned | orthogonal | counter
- If declining or counter: provide value_proposition_nl
{% endif %}
```

### 6.3 Tests

**Extend**: `tests/test_item_audience.py`
- `trend_interpretation` provided → `trend_position` populated
- `lifecycle="declining"` → `value_proposition_nl` is non-null

---

## 7. Enhancement 4: Trend Analysis Pipeline (5-Layer Adapter Architecture)

### 7.1 Concept

The core trend contribution. **Trend analysis quality is the originality**, applied uniformly to all users (no per-user re-weighting).

Five explicit layers, each replaceable:

| Layer | Responsibility | Output |
|---|---|---|
| L1 Source | Per-source data fetch + raw normalization | `RawSourceData` |
| L2 Signal | Time-series → lifecycle label per source | `SourceSignal` |
| L3 Consensus | Reliability-weighted aggregation, disagreement preserved | `TrendSignal` |
| L4 Semantic | Keyword → structured fashion attributes | `TrendSemantic` |
| L5 Pool | Keyword pool from catalog + operator brief | `KeywordPool` |

Pipeline flow (offline):
```
L5 (pool) → L1 (fetch per source per keyword) → L2 (normalize to lifecycle)
  → L3 (consensus across sources) → L4 (semantic mapping)
  → TrendSnapshot persisted to processed_dir
```

At runtime, Trend Agent reads snapshot, calls LLM once to produce `TrendInterpretation` conditioned on directive.

### 7.2 Layer 5 — Keyword Pool

**Create**: `src/adapters/trends/keyword_pool.py`

```python
class KeywordPool(BaseModel):
    catalog_derived: list[str]   # static, offline
    brief_derived: list[str]     # dynamic, per directive
    merged: list[str]            # deduplicated union

def build_catalog_keywords(catalog: dict, top_k: int = 200) -> list[str]:
    """Extract style/category/brand tokens from item attributes.
    Frequency-weighted; drop singletons.
    """
    ...

def extract_brief_keywords(brief: OperatorBrief) -> list[str]:
    """LLM-extracted domain keywords from operator brief.
    Structured output: list of keyword strings, max ~20.
    """
    ...

def build_keyword_pool(catalog: dict, brief: OperatorBrief | None) -> KeywordPool:
    catalog_kw = build_catalog_keywords(catalog)
    brief_kw = extract_brief_keywords(brief) if brief else []
    merged = list(dict.fromkeys(catalog_kw + brief_kw))  # preserve order, dedupe
    return KeywordPool(catalog_derived=catalog_kw, brief_derived=brief_kw, merged=merged)
```

**Prompt**: `prompts/phase2_directive/trend.extract_brief_keywords.j2`

```jinja
Extract fashion-relevant keywords from the operator brief.
Brief: {{ brief.summary }}
Domain hints: {{ brief.domain }}

Output JSON: {"keywords": [...]}, max 20 entries.
Prefer concrete fashion terms (silhouettes, materials, eras, styles, brands)
over abstract marketing language.
```

### 7.3 Layer 1 — Source Adapter Base

**Create**: `src/adapters/trends/base.py`

```python
from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel

class RawSourceData(BaseModel):
    keyword: str
    source_name: str
    time_series: list[tuple[str, float]]  # [(YYYY-MM-DD, value), ...]
    raw_payload: dict  # debugging

class TrendSourceAdapter(ABC):
    source_name: str
    source_type: Literal["search", "media", "community", "editorial"]
    reliability_prior: float  # in [0, 1]

    @abstractmethod
    def fetch(self, keyword: str, time_window: tuple[str, str]) -> RawSourceData:
        ...
```

### 7.4 Layer 1 — Google Trends + Wikipedia Implementations

**Modify**: `src/adapters/trends/google_trends.py`

```python
class GoogleTrendsAdapter(TrendSourceAdapter):
    source_name = "google_trends"
    source_type = "search"
    reliability_prior = 0.8
    # refactor existing pytrends logic into fetch() returning RawSourceData
```

**Create**: `src/adapters/trends/wikipedia.py`

```python
class WikipediaAdapter(TrendSourceAdapter):
    source_name = "wikipedia_pageview"
    source_type = "media"
    reliability_prior = 0.6

    def fetch(self, keyword: str, time_window: tuple[str, str]) -> RawSourceData:
        """Wikimedia REST API pageview stats.
        Resolve keyword → article title via search API, then pageviews."""
        ...
```

### 7.5 Layer 2 — Signal Normalization (Windowed Lifecycle Detection)

**Create**: `src/adapters/trends/signal_normalizer.py`

```python
class SourceSignal(BaseModel):
    keyword: str
    source_name: str
    lifecycle: Literal["rising", "stable", "declining", "niche"]
    growth_ratio: float  # short_ma / long_ma
    short_ma: float
    long_ma: float
    volume_percentile: float  # how loud is this keyword overall
    confidence: float

SHORT_WINDOW_WEEKS = 4
LONG_WINDOW_WEEKS = 12
RISING_THRESHOLD = 1.20      # short MA 20% above long MA
DECLINING_THRESHOLD = 0.85   # short MA 15% below long MA
NICHE_VOLUME_PCT = 0.20      # bottom 20% of volume → niche

def normalize_to_signal(raw: RawSourceData, all_volumes_in_corpus: list[float]) -> SourceSignal:
    """Convert time series to lifecycle label using windowed MA ratio.

    rising:    growth_ratio >= RISING_THRESHOLD
    declining: growth_ratio <= DECLINING_THRESHOLD
    stable:    otherwise, AND volume_percentile > NICHE_VOLUME_PCT
    niche:     otherwise (low volume, no strong direction)
    """
    ...
```

**Why windowed MA, not raw threshold**: Single-point growth is noisy (week-over-week spikes from events). 4w/12w ratio captures sustained direction.

### 7.6 Layer 3 — Consensus with Preserved Disagreement

**Create**: `src/adapters/trends/consensus.py`

```python
class TrendSignal(BaseModel):
    keyword: str
    sources: dict[str, SourceSignal]
    consensus_score: float = 0.0        # 0 (full disagreement) → 1 (full agreement)
    aggregated_lifecycle: Literal["rising", "stable", "declining", "niche"]
    disagreement_flag: bool = False     # True if sources disagree on direction
    disagreement_nl: str | None = None  # short NL note for LLM context
    matched_catalog_attributes: list[str] = Field(default_factory=list)  # populated by L4

def compute_consensus(sources: dict[str, SourceSignal]) -> tuple[float, bool, str | None]:
    """Reliability-weighted agreement on lifecycle.
    Returns (consensus_score, disagreement_flag, disagreement_nl).
    Disagreement is FLAGGED, not collapsed."""
    ...

def aggregate_lifecycle(sources: dict[str, SourceSignal]) -> str:
    """Reliability-weighted majority vote with tie-breaking by reliability_prior."""
    ...
```

**Disagreement policy**: If `disagreement_flag=True`, the LLM interpretation prompt receives the disagreement explicitly rather than a single collapsed label. This preserves epistemic honesty downstream.

### 7.7 Layer 4 — Semantic Mapping (Hybrid)

**Create**: `src/adapters/trends/semantic_mapper.py`

```python
class TrendSemantic(BaseModel):
    keyword: str
    fashion_attributes: dict[str, list[str]]  # e.g., {"style": [...], "category": [...], "era": [...]}
    catalog_match_method: Literal["llm_proposed", "bge_validated", "hybrid"]
    matched_item_ids: list[str]
    confidence: float

def build_semantic_mapping(
    keyword: str,
    catalog: dict,
    embedder: BGEM3,  # reuse existing retrieval embedder
    llm_client: LLMClient,
) -> TrendSemantic:
    """Hybrid: LLM proposes structured attributes, BGE-M3 validates against catalog.

    Step 1: LLM produces {style: [...], category: [...], era: [...]} for keyword
    Step 2: For each proposed attribute, embed and find top-K matching items
    Step 3: Keep attributes with at least N catalog matches (drops hallucinated terms)
    """
    ...
```

**Prompt**: `prompts/offline/trend.semantic_mapping.j2`

```jinja
Map this fashion trend keyword to structured catalog-grounded attributes.

Keyword: {{ keyword }}
Domain: fashion

Produce JSON:
{
  "style": [...],       // concrete style descriptors
  "category": [...],    // item categories likely affected
  "era_reference": "...", // optional decade/era anchor
  "material_or_color": [...]  // optional
}

Use only concrete, catalog-likely terms. If you cannot ground a field, leave it empty.
```

**Built offline once per snapshot**: cached at `{processed_dir}/trend_semantics/{keyword_hash}.json`. Runtime is pure lookup.

### 7.8 Snapshot Schema (Full)

**Modify**: `src/adapters/trends/snapshot.py`

```python
class TrendSnapshot(BaseModel):
    snapshot_id: str
    time_window: tuple[str, str]
    keyword_pool: KeywordPool
    signals: list[TrendSignal]   # one per keyword
    semantics: dict[str, TrendSemantic]  # keyword -> semantic mapping
    created_at: str
```

### 7.9 Build Script (Full Pipeline)

**Modify**: `scripts/build_trend_snapshot.py`

```python
def build_snapshot(time_window, catalog, brief=None):
    # L5: pool
    pool = build_keyword_pool(catalog, brief)

    # L1 + L2 per source per keyword
    adapters = [GoogleTrendsAdapter(), WikipediaAdapter()]
    all_volumes = collect_volumes_for_normalization(adapters, pool, time_window)

    signals = []
    for keyword in pool.merged:
        source_signals = {}
        for adapter in adapters:
            raw = adapter.fetch(keyword, time_window)
            source_signals[adapter.source_name] = normalize_to_signal(raw, all_volumes)

        # L3: consensus
        consensus, disagreement, disagreement_nl = compute_consensus(source_signals)
        lifecycle = aggregate_lifecycle(source_signals)

        # L4: semantic (built per keyword, cached)
        semantic = load_or_build_semantic(keyword, catalog, embedder, llm)

        signals.append(TrendSignal(
            keyword=keyword,
            sources=source_signals,
            consensus_score=consensus,
            aggregated_lifecycle=lifecycle,
            disagreement_flag=disagreement,
            disagreement_nl=disagreement_nl,
            matched_catalog_attributes=flatten(semantic.fashion_attributes),
        ))

    return TrendSnapshot(
        snapshot_id=...,
        time_window=time_window,
        keyword_pool=pool,
        signals=signals,
        semantics={s.keyword: load_semantic(s.keyword) for s in signals},
        created_at=now(),
    )
```

### 7.10 Trend Agent Runtime

**Modify**: `src/core/agents/trend_agent.py`

```python
def interpret_trend(
    self,
    snapshot: TrendSnapshot,
    directive: Directive,
) -> TrendInterpretation:
    """Single global interpretation. No user conditioning.

    Filters snapshot.signals by directive relevance, prioritizes:
    1. High consensus_score + aligned with directive
    2. Disagreement-flagged keywords (surfaced with epistemic hedge)
    3. Rising lifecycle in directive-relevant categories
    """
    relevant = filter_signals_by_directive(snapshot, directive)
    return self.llm_call(prompt="trend.interpret_multisource", context={
        "signals": relevant,
        "semantics": {s.keyword: snapshot.semantics[s.keyword] for s in relevant},
        "directive": directive,
    })
```

### 7.11 Prompt Update

**Rename**: `trend.interpret_gtrends.j2` → `prompts/phase2_directive/trend.interpret_multisource.j2`

```jinja
For each relevant keyword, signals from multiple sources:
{% for signal in signals %}
- {{ signal.keyword }}: lifecycle={{ signal.aggregated_lifecycle }},
  consensus={{ signal.consensus_score }}
  {% if signal.disagreement_flag %}
  ⚠ Source disagreement: {{ signal.disagreement_nl }}
  {% endif %}
  fashion attributes: {{ semantics[signal.keyword].fashion_attributes }}
{% endfor %}

Directive: {{ directive }}

Produce a TrendInterpretation that:
- Emphasizes high-consensus, directive-aligned rising signals
- Surfaces disagreement-flagged signals with explicit hedging
- Anchors claims to fashion_attributes (not raw keywords)
```

### 7.12 Tests

**Create**: `tests/test_trend_pipeline.py`
- `test_keyword_pool_dedup` — catalog + brief union
- `test_signal_normalization_windowed` — synthetic time series → expected lifecycle
- `test_consensus_full_agreement` — both rising → high consensus
- `test_consensus_disagreement_flag` — Google rising, Wiki declining → flag=True
- `test_semantic_mapping_grounded` — LLM proposes "low-rise jeans" → BGE-M3 finds catalog items
- `test_semantic_drops_ungrounded` — hallucinated term with no catalog match → dropped
- `test_snapshot_roundtrip_json` — full snapshot serializes/deserializes

---

## 8. Enhancement 5: Expert ↔ Trend Negotiation

### 8.1 Concept

Phase 2 currently: Expert directive → Trend broadcast → done.

New: Trend challenges directive when conflict detected. Expert responds (accept/reject/counter). Bounded loop.

Pattern from LLM Debate (Du et al., ICML'24).

### 8.2 Schema

```python
class TrendDirectiveTension(BaseModel):
    type: Literal["attribute_conflict", "price_conflict", "audience_conflict"]
    severity: float = Field(..., ge=0.0, le=1.0)
    description_nl: str
    directive_element: str
    trend_element: str

class NegotiationMessage(BaseModel):
    turn: int
    from_agent: Literal["expert", "trend"]
    to_agent: Literal["expert", "trend"]
    message_type: Literal["challenge", "accept", "reject", "counter"]
    content_nl: str
    tensions: list[TrendDirectiveTension] = Field(default_factory=list)
    proposed_directive_delta: dict | None = None

class NegotiationLog(BaseModel):
    messages: list[NegotiationMessage] = Field(default_factory=list)
    final_outcome: Literal["consensus", "expert_held", "max_turns_reached"]

class Directive(BaseModel):
    # ... existing
    negotiation_log: NegotiationLog | None = None
```

### 8.3 Config

```python
class MargoRunConfig(BaseModel):
    # ... existing
    max_negotiation_turns: int = 1   # start conservative; raise after calibration
    tension_threshold: float = 0.7   # start conservative; lower after calibration
```

### 8.4 Phase 2 Flow (REVISED)

**Modify**: `src/core/lifecycle/phase2_directive.py`

```python
def run_phase2(expert, trend, brief, snapshot, config):
    directive = expert.issue_directive(brief)
    log = NegotiationLog()

    for turn in range(config.max_negotiation_turns):
        global_interp = trend.interpret_trend(snapshot, directive)
        tensions = trend.detect_tensions(directive, global_interp)

        if not tensions or max(t.severity for t in tensions) < config.tension_threshold:
            log.final_outcome = "consensus"
            break

        challenge = trend.challenge_directive(directive, global_interp, tensions, turn)
        log.messages.append(challenge)

        response = expert.respond_to_challenge(directive, challenge, brief, turn)
        log.messages.append(response)

        if response.message_type == "accept":
            directive = expert.apply_directive_delta(directive, response.proposed_directive_delta)
            log.final_outcome = "consensus"
            break
        elif response.message_type == "reject":
            log.final_outcome = "expert_held"
            break
        elif response.message_type == "counter":
            directive = expert.apply_directive_delta(directive, response.proposed_directive_delta)
            # continue loop
    else:
        log.final_outcome = "max_turns_reached"

    directive.negotiation_log = log
    return directive, global_interp, log
```

### 8.5 Prompts (NEW)

**Create**: `prompts/phase2_directive/trend.detect_tension.j2`

```jinja
Compare directive with trend interpretation. Identify tensions.

Directive: {{ directive }}
Trend interpretation: {{ trend_interpretation }}

For each tension output:
- type, severity (0.0-1.0), description_nl, directive_element, trend_element

Output JSON list (empty if no tension).
```

**Create**: `prompts/phase2_directive/trend.challenge.j2`

```jinja
Tensions: {{ tensions }}

Generate challenge to Expert:
- content_nl
- proposed_directive_delta (JSON patch)

Be concise and constructive.
```

**Create**: `prompts/phase2_directive/expert.respond_negotiation.j2`

```jinja
Challenge from Trend Agent: {{ challenge.content_nl }}
Proposed delta: {{ challenge.proposed_directive_delta }}

Current directive: {{ directive }}
Original brief (source of truth): {{ brief }}

Decide:
- accept: apply delta (consistent with brief)
- reject: hold directive (delta violates brief)
- counter: propose own delta

Reasoning MUST trace to brief. Output JSON.
```

### 8.6 Agent Changes

**Modify**: `src/core/agents/trend_agent.py`

```python
def detect_tensions(self, directive, trend_interp) -> list[TrendDirectiveTension]:
    ...

def challenge_directive(self, directive, trend_interp, tensions, turn) -> NegotiationMessage:
    ...
```

**Modify**: `src/core/agents/expert_agent.py`

```python
def respond_to_challenge(self, directive, challenge, brief, turn) -> NegotiationMessage:
    ...

def apply_directive_delta(self, directive, delta: dict) -> Directive:
    """Apply JSON patch-like delta, return new Directive."""
    ...
```

### 8.7 Web Trace Integration

**Modify**: `web/backend/services/margo_runner.py` — stream negotiation messages via WebSocket.

### 8.8 Tests

**Create**: `tests/test_negotiation.py`
- Tension detection
- Full loop: challenge → accept → consensus
- Reject path → expert_held
- Max turns reached
- Counter path → directive updates, loop continues
- Delta application correctness

---

## 9. Cross-Cutting Concerns

### 9.1 Backward Compatibility

All schema extensions use Optional fields. Existing tests must continue passing:

```bash
source margo/bin/activate
pytest tests/ -v
```

### 9.2 Dummy LLM Backend

Each new prompt needs corresponding dummy responses in `src/adapters/llm/client.py`.

### 9.3 SVR Counting

All new LLM outputs route through `SchemaValidator`.

### 9.4 Ablation Hooks (NEW)

Even though `src/eval/*` is untouched in this round, **each enhancement must be toggleable** via `MargoRunConfig` flags:

```python
class MargoRunConfig(BaseModel):
    enable_multi_axis: bool = True
    enable_peer_signal: bool = True
    enable_audience_profile: bool = True
    enable_trend_position: bool = True
    enable_multisource_trend: bool = True
    enable_negotiation: bool = True
```

This lets later evaluation work run ablations without code changes.

### 9.5 Evaluation Untouched

Do NOT modify in this round:
- `src/eval/*`
- `scripts/evaluate.py`

---

## 10. Implementation Timeline

| Week | Enhancement | Key Deliverables |
|---|---|---|
| 1 | User multi-axis (#1) | Schemas, deterministic stats, prompts, tests |
| 2 | Cohort + peer signal (#1.5) | cohort_stats script, loader, coverage diagnostic, peer_signal method |
| 3 | Item audience (#2) + trend position (#3) | buyer aggregate (cohort-aware), audience loader, position |
| 4-5 | Trend Pipeline (#4) | 5-layer build: keyword pool, Wikipedia adapter, windowed signal, consensus with disagreement, semantic mapper |
| 6-7 | Negotiation (#5) | Tension detection, challenge/respond, Phase 2 loop |
| 8 | Integration + Web demo | Full E2E, web trace, docs update |

Each week ends with `pytest` passing + `scripts/sanity_one_user.py` working.

After Week 2, **run `scripts/diagnose_cohort_coverage.py`** before proceeding — if fallback rate > 50%, coarsen axis values.

---

## 11. File Manifest

### New files
```
scripts/build_buyer_aggregate.py
scripts/build_cohort_stats.py
scripts/diagnose_cohort_coverage.py
src/data/fashion/preference_stats.py
src/data/fashion/audience_loader.py
src/data/fashion/cohort_loader.py
src/adapters/trends/base.py
src/adapters/trends/keyword_pool.py
src/adapters/trends/wikipedia.py
src/adapters/trends/signal_normalizer.py
src/adapters/trends/consensus.py
src/adapters/trends/semantic_mapper.py
prompts/phase3_reasoning/user.update_state.j2
prompts/phase2_directive/trend.extract_brief_keywords.j2
prompts/phase2_directive/trend.interpret_multisource.j2
prompts/phase2_directive/trend.detect_tension.j2
prompts/phase2_directive/trend.challenge.j2
prompts/phase2_directive/expert.respond_negotiation.j2
prompts/offline/trend.semantic_mapping.j2
tests/test_user_preference_axes.py
tests/test_cohort_peer_signal.py
tests/test_item_audience.py
tests/test_trend_pipeline.py
tests/test_negotiation.py
```

### Modified files
```
src/core/protocol/messages.py
src/core/agents/user_agent.py
src/core/agents/item_agent.py
src/core/agents/trend_agent.py
src/core/agents/expert_agent.py
src/core/lifecycle/phase2_directive.py
src/core/lifecycle/phase3_reasoning.py
src/core/lifecycle/orchestrator.py
src/core/validation/...
src/core/config.py  # ablation flags
src/adapters/llm/client.py
src/adapters/trends/google_trends.py
src/adapters/trends/snapshot.py
scripts/build_trend_snapshot.py
prompts/phase1_initialization/user.system.j2
prompts/phase3_reasoning/user.profile.j2
prompts/phase3_reasoning/user.evaluate.j2
prompts/phase3_reasoning/item.describe.j2
```

### Untouched
```
src/eval/*
scripts/evaluate.py
src/baselines/*
```

---

## 12. Related Work (for paper)

- **RLMRec (WWW'24)**: profile decomposition → adapted as fixed 4-axis
- **KAR (RecSys'24)**: factor-level reasoning → adapted as per-axis evaluation
- **RecExplainer (KDD'24)**: buyer-pattern descriptions → adapted as audience inference with cohort
- **LLM Debate (Du et al., ICML'24)**: debate termination → adapted as negotiation max_turns
- **AgentCF (WWW'24)**: contrasted as thin 1:1 reflection vs MARGO's thick multi-axis + cohort
- **MACRec (SIGIR'24)**: contrasted as discussion-only vs MARGO's active tension resolution

---

## 13. Commit Convention

```
feat(user): add 4-axis preference state (style/price/category/brand)
feat(user): add axis-derived cohort signature and peer signal
feat(item): add audience profile inference with cohort distribution
feat(item): add trend-grounded self-positioning
feat(trend): add 5-layer trend pipeline (keyword pool, multi-source, windowed signal, consensus, semantic)
feat(negotiation): add expert-trend tension detection and negotiation loop
test(<area>): ...
docs(research-summary): update with methodology enhancements
```

---

## 14. Confirmed Settings

| Setting | Value |
|---|---|
| 4 axes | style_preference, price_preference, category_preference, brand_preference |
| Buyer aggregate MIN_BUYERS | 3 |
| Cohort MIN_COHORT_SIZE | 5 |
| Negotiation max_turns | 1 (start conservative) |
| Tension threshold | 0.7 (start conservative) |
| Trend signal short MA | 4 weeks |
| Trend signal long MA | 12 weeks |
| Rising threshold (short/long ratio) | 1.20 |
| Declining threshold (short/long ratio) | 0.85 |
| Niche volume percentile | 0.20 |
| Trend sources | Google Trends (existing) + Wikipedia Pageview (new) |
| Trend application | Uniform across users (no per-user broadcast) |
| Keyword pool | catalog-derived ∪ brief-derived |
| Semantic mapping | LLM-proposed + BGE-M3 validated (hybrid) |
| Web demo update | Week 8 (after methodology) |
| Evaluation modules | Untouched (this round) |

---

End of directive.
