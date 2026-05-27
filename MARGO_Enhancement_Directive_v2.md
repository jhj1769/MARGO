# MARGO Enhancement — Final Implementation Directive (v2)

> **Target**: Enhance MARGO (current state: 2026-05-20) with agent-level reasoning upgrades.
> **Scope**: Methodology enhancement only. Evaluation metrics (TAS/DCR/etc) will be revisited separately later.
> **Baseline**: All work builds on existing 4-agent / 4-phase architecture. No framework restructure.

---

## 0. Context for Claude Code

Current MARGO state (as of 2026-05-20):
- 4 agents (User, Item, Expert, Trend) inheriting `BaseAgent`
- 4-phase lifecycle (init → directive → reasoning → validation+refine)
- Pydantic message protocol, in-process MessageBus
- BGE-M3 + FAISS retrieval, Google Trends snapshot store
- Pytest 18 passed
- Web demo (FastAPI + Next.js)

This document specifies **7 enhancements** with file paths, schemas, prompts, and tests.

**Critical constraints**:
- Do NOT modify evaluation modules (`src/eval/`) in this round
- All schema extensions backward-compatible (Optional fields)
- All existing tests must continue passing

---

## 1. Enhancement Overview

| # | Enhancement | Agent | Effort | Week |
|---|---|---|---|---|
| 1 | Multi-axis preference state | User | M | 1 |
| 1.5 | Axis-derived cohort + peer signal | User | M | 2 |
| 2 | Audience profile inference | Item | M | 3 |
| 3 | Trend-grounded self-positioning | Item | S | 3 |
| 4 | Context-conditional broadcast | Trend | M | 4 |
| 5 | Multi-source trend adapter (Wikipedia) | Trend | M | 5 |
| 6 | Expert ↔ Trend negotiation | Expert + Trend | L | 6-7 |
| - | Integration + Web demo update | All | M | 8 |

Total: ~8 weeks. Implement strictly in order.

---

## 2. Enhancement 1: User Agent Multi-Axis Preference

### 2.1 Concept

User preference decomposed into **4 explicit axes**:

| Axis | Meaning | Inference Method |
|---|---|---|
| `style_preference` | Style direction | LLM inference from item descriptions |
| `price_preference` | Price tier preference | Deterministic statistics (median + variance) |
| `category_preference` | Category distribution | Deterministic statistics |
| `brand_preference` | Brand loyalty / diversity | Deterministic statistics |

**Key principle**: Price/Category/Brand computed deterministically from stats. Style is LLM-inferred. Reduces hallucination surface.

**Diverse user handling**: Each axis allows valid "mixed" / "balanced" / "diverse" values with appropriate confidence. Single labeling is never forced.

### 2.2 Axis Value Specs

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
  - Single category > 50% → `"{category}-focused"` (e.g., `"outerwear-focused"`)
  - 2-3 categories with 25-50% each → `"{cat1}-{cat2}-mix"`
  - All major categories < 30% → `"balanced"`
- High confidence even when "balanced"

**brand_preference**
- Computed from brand distribution:
  - Single brand > 40% → `"brand-loyal:{brand_name}"`
  - Top 3 brands each < 30% → `"brand-diverse"`
  - Premium brand share dominant → `"premium-brand-curious"`

### 2.3 Schema (NEW)

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
    derived_from: Literal["statistical", "llm_inferred"]  # provenance
    stability: float = Field(default=1.0, ge=0.0, le=1.0)  # temporal consistency

class UserPreferenceState(BaseModel):
    user_id: str
    profile_nl: str  # keep existing NL summary
    axes: list[PreferenceAxis]
    cohort_signature: str  # populated in Enhancement 1.5
    last_updated_at: str
    
    def get_axis(self, name: AxisName) -> PreferenceAxis | None:
        return next((a for a in self.axes if a.name == name), None)
```

### 2.4 Statistical Pre-computation

**Create**: `src/data/fashion/preference_stats.py`

```python
def compute_deterministic_axes(user_id: str, history: list[Interaction], items: dict) -> dict:
    """Compute price, category, brand axes from statistics.
    Returns dict with 3 axes (style left for LLM).
    Also computes stability by comparing recent vs full history.
    """
```

### 2.5 Prompt Changes

**Modify**: `prompts/phase1_initialization/user.system.j2`

Add system instruction explaining 4 axes + provenance.

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

Restructure for per-axis evaluation (KAR-style factor reasoning):
```
For each of the 4 axes, output:
- match_score (0.0-1.0)
- one-sentence explanation

Then aggregate into final_score and three_layer_rationale (personal/directive/trend).
Personal layer must reference specific axes.
```

### 2.6 Agent Changes

**Modify**: `src/core/agents/user_agent.py`

```python
def build_profile(self, user_id: str, history: list[Interaction]) -> UserProfile:
    # Step 1: deterministic computation
    deterministic = compute_deterministic_axes(user_id, history, self.item_catalog)
    # Step 2: LLM for style + NL summary
    response = self.llm_call(prompt="user.profile", context={...})
    # Step 3: parse, validate, merge
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

### 2.7 Tests

**Create**: `tests/test_user_preference_axes.py`
- `test_deterministic_axes_computation`
- `test_diverse_user_lower_confidence`
- `test_axis_schema_validation` → SVR increment on malformed
- `test_update_preference_state`
- `test_stability_score` — recent shift → lower stability

---

## 3. Enhancement 1.5: Axis-Derived Cohort + Peer Signal

### 3.1 Concept

User cohort emerges naturally from axis combinations (not from clustering algorithm).

**Two functions**:

**Function 1 — Cohort Signature**: deterministic concat of axis values
**Function 2 — Peer Signal**: collaborative signal from users sharing cohort

### 3.2 Cohort Signature

```python
def compute_cohort_signature(state: UserPreferenceState) -> str:
    """Deterministic signature from 4 axes."""
    sorted_axes = sorted(state.axes, key=lambda a: a.name)
    return "|".join(f"{a.name[:3]}:{a.value}" for a in sorted_axes)
    # Example: "bra:brand-diverse|cat:balanced|pri:mid-tier|sty:minimal-casual"
```

Populated in `UserPreferenceState.cohort_signature` during `build_profile`.

### 3.3 Cohort Stats (Offline)

**Create**: `scripts/build_cohort_stats.py`

```python
# Pseudo
all_users_with_state = load_all_user_states()
cohorts = defaultdict(list)
for user_state in all_users_with_state:
    cohorts[user_state.cohort_signature].append(user_state.user_id)

# For each cohort, compute item-level peer stats
for cohort_sig, user_ids in cohorts.items():
    if len(user_ids) < MIN_COHORT_SIZE:  # e.g., 5
        continue
    
    item_buy_counts = defaultdict(int)
    for uid in user_ids:
        for interaction in history(uid):
            if interaction.rating >= 4.0:
                item_buy_counts[interaction.item_id] += 1
    
    cohort_stats = {
        "signature": cohort_sig,
        "size": len(user_ids),
        "user_ids": user_ids,  # for traceability
        "item_buy_ratios": {
            item_id: count / len(user_ids)
            for item_id, count in item_buy_counts.items()
        },
        "top_categories": ...,  # most-purchased categories
        "top_brands": ...,
    }
    save(f"{processed_dir}/cohort_stats/{hash(cohort_sig)}.json", cohort_stats)
```

### 3.4 Schema

**Extend**: `src/core/protocol/messages.py`

```python
class CohortStats(BaseModel):
    signature: str
    size: int
    item_buy_ratios: dict[str, float]  # item_id -> ratio of cohort that bought
    top_categories: list[tuple[str, float]]
    top_brands: list[tuple[str, float]]
    
    def peer_signal_for(self, item_id: str) -> float:
        """Ratio of cohort that bought this item (0.0 if not in cohort)."""
        return self.item_buy_ratios.get(item_id, 0.0)
```

### 3.5 Loader

**Create**: `src/data/fashion/cohort_loader.py`

```python
def load_cohort_stats(signature: str, processed_dir: Path) -> CohortStats | None:
    path = processed_dir / "cohort_stats" / f"{hash_signature(signature)}.json"
    if not path.exists():
        return None
    return CohortStats.parse_file(path)
```

### 3.6 Agent Integration

**Modify**: `src/core/agents/user_agent.py`

```python
def get_peer_signal(
    self,
    state: UserPreferenceState,
    candidate_item_id: str,
) -> tuple[float, str]:
    """Returns (peer_buy_ratio, explanation_nl)."""
    cohort = load_cohort_stats(state.cohort_signature, self.processed_dir)
    if cohort is None or cohort.size < MIN_COHORT_SIZE:
        return 0.0, "Cohort too small for reliable peer signal"
    
    ratio = cohort.peer_signal_for(candidate_item_id)
    return ratio, f"{ratio:.0%} of users in your cohort ({cohort.size} users) purchased this item"
```

### 3.7 Prompt Changes

**Modify**: `prompts/phase3_reasoning/user.evaluate.j2`

Add peer signal section:
```jinja
{% if peer_signal %}
Cohort peer signal:
{{ peer_signal.explanation_nl }}
Use this as one of multiple signals — high peer ratio is a positive indicator
but does not override your axes-based judgment.
{% endif %}
```

### 3.8 Item Agent Integration

Audience inference (Enhancement 2) leverages cohort distribution of buyers — natural fit since cohorts are well-defined.

### 3.9 Tests

**Create**: `tests/test_cohort_peer_signal.py`
- `test_cohort_signature_deterministic` — same axes → same signature
- `test_peer_signal_lookup` — known item → expected ratio
- `test_small_cohort_fallback` — cohort < MIN returns 0.0
- `test_signature_consistency` — order-invariant

---

## 4. Enhancement 2: Item Agent Audience Profile Inference

### 4.1 Concept

Each item maintains a `buyer_aggregate` profile derived from past buyers. Item Agent uses this to claim "who I serve well" in self-description.

With Enhancement 1.5, buyer aggregate can leverage cohort distribution.

Inspired by RecExplainer (KDD'24).

### 4.2 Offline Preprocessing

**Create**: `scripts/build_buyer_aggregate.py`

```python
MIN_BUYERS = 3

for item_id in catalog:
    buyers = interactions.filter(item_id=item_id, rating>=4.0).user_ids
    if len(buyers) < MIN_BUYERS:
        continue
    
    # Cohort distribution of buyers (leverages Enhancement 1.5)
    buyer_cohorts = [load_user_state(b).cohort_signature for b in buyers]
    cohort_dist = Counter(buyer_cohorts)
    
    aggregate = {
        "item_id": item_id,
        "buyer_count": len(buyers),
        "avg_price_history": ...,
        "median_history_length": ...,
        "category_distribution": ...,
        "brand_distribution": ...,
        "buyer_cohort_distribution": dict(cohort_dist),  # NEW
        "dominant_cohorts": top_3_cohorts(cohort_dist),  # NEW
        "evidence_buyer_ids": random.sample(buyers, min(5, len(buyers))),
    }
    save(...)
```

### 4.3 Schema

```python
class ItemAudienceProfile(BaseModel):
    item_id: str
    buyer_count: int
    avg_price_history: float | None
    median_history_length: int | None
    category_distribution: dict[str, float]
    brand_distribution: dict[str, float]
    buyer_cohort_distribution: dict[str, int]  # signature -> count
    dominant_cohorts: list[str]  # top 3
    evidence_buyer_ids: list[str]
    dominant_pattern_nl: str | None = None
    outlier_pattern_nl: str | None = None
```

### 4.4 Loader

**Create**: `src/data/fashion/audience_loader.py`

```python
def load_audience_profile(item_id: str, processed_dir: Path) -> ItemAudienceProfile | None:
    path = processed_dir / "buyer_aggregate" / f"{item_id}.json"
    if not path.exists():
        return None
    return ItemAudienceProfile.parse_file(path)
```

### 4.5 Prompt Changes

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

### 4.6 Agent Changes

**Modify**: `src/core/agents/item_agent.py`

```python
def self_describe(
    self,
    item: Item,
    directive: Directive,
    audience_profile: ItemAudienceProfile | None = None,
    trend_interpretation: TrendBroadcast | None = None,  # Enhancement 3
) -> ItemDescription:
    ...
```

### 4.7 Lifecycle Integration

**Modify**: `src/core/lifecycle/phase3_reasoning.py`

```python
audience = load_audience_profile(item.item_id, processed_dir)
description = item_agent.self_describe(
    item=item,
    directive=directive,
    audience_profile=audience,
    trend_interpretation=trend_broadcast,
)
```

### 4.8 Tests

**Create**: `tests/test_item_audience.py`
- Items below MIN_BUYERS → None from loader
- self_describe handles None gracefully
- Audience-populated description references cohort patterns

---

## 5. Enhancement 3: Item Agent Trend Self-Positioning

### 5.1 Schema Extension

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

### 5.2 Prompt

**Append to**: `prompts/phase3_reasoning/item.describe.j2`

```jinja
{% if trend_interpretation %}
Current trend signal: {{ trend_interpretation }}

Position yourself relative to this trend:
- lifecycle: rising | stable | declining | niche
- alignment: aligned | orthogonal | counter
- If declining or counter: provide value_proposition_nl
{% endif %}
```

### 5.3 Tests

**Extend**: `tests/test_item_audience.py`
- `trend_interpretation` provided → `trend_position` populated
- `lifecycle="declining"` → `value_proposition_nl` is non-null

---

## 6. Enhancement 4: Trend Agent Context-Conditional Broadcast

### 6.1 Concept

Two-stage interpretation:
- **Stage A**: Global interpretation (existing, directive-conditioned)
- **Stage B**: User-conditional broadcast — re-weighted per user's preference axes

Pattern from InteRecAgent.

### 6.2 Schema

```python
class TrendBroadcast(BaseModel):
    user_id: str | None = None
    relevant_signals: list[str]
    emphasis_nl: str
    boost_direction: Literal["boost", "neutral", "suppress"]
    source_global_interp_id: str
```

### 6.3 Prompt

**Create**: `prompts/phase3_reasoning/trend.broadcast.j2`

```jinja
You are the Trend Agent generating user-conditional broadcast.

Global trend interpretation:
{{ global_interpretation }}

Target user's preference axes:
- style: {{ user_axes.style_preference.value }} (conf {{ user_axes.style_preference.confidence }})
- price: {{ user_axes.price_preference.value }}
- category: {{ user_axes.category_preference.value }}
- brand: {{ user_axes.brand_preference.value }}

Generate conditional broadcast:
1. relevant_signals: filter trend keywords relevant to this user's axes
2. emphasis_nl: 1-2 sentence emphasis
3. boost_direction:
   - "boost" if user's axes ALIGN with rising trend
   - "suppress" if rising trend MISALIGNS with user's stable preferences
   - "neutral" otherwise

Output JSON matching TrendBroadcast schema.
```

### 6.4 Agent Changes

**Modify**: `src/core/agents/trend_agent.py`

```python
def conditional_broadcast(
    self,
    global_interpretation: TrendInterpretation,
    user_state: UserPreferenceState,
) -> TrendBroadcast:
    sig = self._axes_signature(user_state)
    if sig in self._broadcast_cache:
        return self._broadcast_cache[sig]
    
    result = self.llm_call(prompt="trend.broadcast", context={...})
    self._broadcast_cache[sig] = result
    return result

def _axes_signature(self, user_state: UserPreferenceState) -> str:
    """Reuse cohort_signature from Enhancement 1.5."""
    return user_state.cohort_signature
```

**Note**: Cache key reuses `cohort_signature` from Enhancement 1.5 — same cohort → same broadcast.

### 6.5 Lifecycle Integration

**Modify**: `src/core/lifecycle/phase3_reasoning.py`

```python
global_interp = trend_agent.interpret_trend(snapshot, directive)

for user in user_batch:
    user_broadcast = trend_agent.conditional_broadcast(global_interp, user.preference_state)
    candidates = retriever.retrieve_with_directive(...)
    for item in candidates:
        item_desc = item_agent.self_describe(
            item=item,
            directive=directive,
            audience_profile=load_audience(item),
            trend_interpretation=user_broadcast,
        )
        score = user_agent.evaluate_candidate(
            user=user, 
            item_desc=item_desc,
            directive=directive,
            trend_broadcast=user_broadcast,
            peer_signal=user_agent.get_peer_signal(user.preference_state, item.item_id),
        )
```

### 6.6 Tests

**Create**: `tests/test_trend_conditional.py`
- Same cohort_signature → cache hit
- Different axes → different broadcasts
- Boost logic verified

---

## 7. Enhancement 5: Multi-Source Trend Adapter (Wikipedia)

### 7.1 Concept

Plug-in architecture for trend sources. Google Trends + Wikipedia Pageview demonstrate multi-source framework.

Other sources (GDELT, Reddit) → future work.

### 7.2 Adapter Interface

**Create**: `src/adapters/trends/base.py`

```python
from abc import ABC, abstractmethod
from typing import Literal

class TrendSourceAdapter(ABC):
    source_name: str
    source_type: Literal["search", "media", "community", "editorial"]
    reliability_prior: float
    
    @abstractmethod
    def fetch(self, domain: str, time_window: str, keywords: list[str]) -> dict:
        ...
    
    @abstractmethod
    def normalize(self, raw: dict) -> list["NormalizedSignal"]:
        ...
```

### 7.3 Refactor Google Trends

**Modify**: `src/adapters/trends/google_trends.py`

```python
class GoogleTrendsAdapter(TrendSourceAdapter):
    source_name = "google_trends"
    source_type = "search"
    reliability_prior = 0.8
    # existing pytrends logic refactored to match interface
```

### 7.4 Wikipedia Adapter

**Create**: `src/adapters/trends/wikipedia.py`

```python
class WikipediaAdapter(TrendSourceAdapter):
    source_name = "wikipedia_pageview"
    source_type = "media"
    reliability_prior = 0.6
    
    def fetch(self, domain, time_window, keywords):
        """Wikimedia REST API pageview stats.
        Fashion-relevant pages: fashion item pages + 2023 popular movies/shows.
        """
        ...
    
    def normalize(self, raw):
        """Pageview delta → lifecycle signal."""
        ...
```

### 7.5 Snapshot Schema Extension

**Modify**: `src/adapters/trends/snapshot.py`

```python
class SourceSignal(BaseModel):
    source: str
    score: float
    growth: str
    raw_evidence: dict

class TrendSignal(BaseModel):
    keyword: str
    sources: dict[str, SourceSignal]
    consensus_score: float = 0.0
    aggregated_lifecycle: Literal["rising", "stable", "declining"]
    matched_catalog_attributes: list[str]
```

### 7.6 Consensus Scoring

**Create**: `src/adapters/trends/consensus.py`

```python
def compute_consensus(sources: dict[str, SourceSignal]) -> float:
    """Reliability-weighted agreement on lifecycle direction."""
    ...

def aggregate_lifecycle(sources: dict[str, SourceSignal]) -> Literal["rising", "stable", "declining"]:
    """Reliability-weighted majority vote."""
    ...
```

### 7.7 Build Script

**Modify**: `scripts/build_trend_snapshot.py`

```python
adapters = [GoogleTrendsAdapter(), WikipediaAdapter()]

for time_window in weeks_2023:
    for keyword in keyword_pool:
        sources_data = {}
        for adapter in adapters:
            raw = adapter.fetch(domain="fashion", time_window=time_window, keywords=[keyword])
            sources_data[adapter.source_name] = adapter.normalize(raw)
        
        signal = TrendSignal(
            keyword=keyword,
            sources=sources_data,
            consensus_score=compute_consensus(sources_data),
            aggregated_lifecycle=aggregate_lifecycle(sources_data),
            ...
        )
```

### 7.8 Prompt Update

**Modify**: rename `trend.interpret_gtrends.j2` → `trend.interpret_multisource.j2`

```jinja
For each keyword, signals from multiple sources:
{{ signals }}

High-consensus keywords (high consensus_score): emphasize strongly.
Single-source keywords: mention with uncertainty.
```

### 7.9 Tests

**Create**: `tests/test_multisource_trend.py`
- Wikipedia adapter fetch (mock)
- Consensus computation (2 sources rising → high)
- Aggregated lifecycle voting
- Snapshot JSON contains multi-source structure

---

## 8. Enhancement 6: Expert ↔ Trend Negotiation

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
    max_negotiation_turns: int = 2
    tension_threshold: float = 0.5
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

**Modify**: `web/backend/services/margo_runner.py`

Stream negotiation messages via WebSocket.

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

### 9.4 Evaluation Untouched

Do NOT modify:
- `src/eval/*`
- `scripts/evaluate.py`

---

## 10. Implementation Timeline

| Week | Enhancement | Key Deliverables |
|---|---|---|
| 1 | User multi-axis (#1) | Schemas, deterministic stats, prompts, tests |
| 2 | Cohort + peer signal (#1.5) | cohort_stats script, loader, peer_signal method |
| 3 | Item audience (#2) + trend position (#3) | buyer aggregate (cohort-aware), audience loader, position |
| 4 | Trend conditional broadcast (#4) | conditional_broadcast + cache reusing cohort_signature |
| 5 | Multi-source adapter (#5) | Wikipedia adapter, consensus scoring |
| 6-7 | Negotiation (#6) | Tension detection, challenge/respond, Phase 2 loop |
| 8 | Integration + Web demo | Full E2E, web trace, docs update |

Each week ends with `pytest` passing + `scripts/sanity_one_user.py` working.

---

## 11. File Manifest

### New files
```
scripts/build_buyer_aggregate.py
scripts/build_cohort_stats.py
src/data/fashion/preference_stats.py
src/data/fashion/audience_loader.py
src/data/fashion/cohort_loader.py
src/adapters/trends/base.py
src/adapters/trends/wikipedia.py
src/adapters/trends/consensus.py
prompts/phase3_reasoning/user.update_state.j2
prompts/phase3_reasoning/trend.broadcast.j2
prompts/phase2_directive/trend.interpret_multisource.j2
prompts/phase2_directive/trend.detect_tension.j2
prompts/phase2_directive/trend.challenge.j2
prompts/phase2_directive/expert.respond_negotiation.j2
tests/test_user_preference_axes.py
tests/test_cohort_peer_signal.py
tests/test_item_audience.py
tests/test_trend_conditional.py
tests/test_multisource_trend.py
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
- **InteRecAgent**: context-aware retrieval → adapted as user-conditional broadcast
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
feat(trend): add user-conditional broadcast (cohort-signature cached)
feat(trend): add multi-source plug-in architecture with Wikipedia adapter
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
| Negotiation max_turns | 2 |
| Tension threshold | 0.5 |
| Trend sources | Google Trends (existing) + Wikipedia Pageview (new) |
| Web demo update | Week 8 (after methodology) |
| Evaluation modules | Untouched |

---

End of directive.
