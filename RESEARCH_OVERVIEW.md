# MARGO — 연구 개요 (Overview)

> **MARGO**: *A Multi-Agent Framework for Stakeholder-Aware Recommendation Governance*
> 본 문서는 연구의 **배경 → 문제정의 → 방법론 → 실행 plan**을 한 호흡으로 정리한 single source of truth.
> 구현 현황·디렉토리·테스트 현황 등 운영 정보는 [`RESEARCH_SUMMARY.md`](./RESEARCH_SUMMARY.md) 참조.
> 갱신: 2026-05-27 (Phase A·B·C·D 완료 + **Trend Agent v5 (Trade-press 4-axis pipeline)** 17 시즌 빌드 완료 + **v3/v4 fallback retired** → repo `previous/`로 격리)

---

## 1. 연구 배경

### 1.1 추천 시스템의 전제 — User–Item 2-stakeholder

전통적 추천 시스템(CF, NCF, LightGCN 등)은 **사용자(User)** 와 **아이템(Item)** 두 주체 간의 상호작용 행렬을 학습한다.
"누가 무엇을 좋아하는가"를 *과거 상호작용*으로 환원하고, 사용자가 다음에 좋아할 아이템을 예측한다.

실제 커머스(특히 패션) 운영 환경에서는 두 외생 주체가 추천에 *직접* 개입한다.

| 외생 주체 | 실제 영향 | 기존 시스템 처리 |
|---|---|---|
| **운영자 (MD / PM)** | 시즌 캠페인, 카테고리 부스팅, 가격대 정책, 브랜드 노출 조정 | 모델 *바깥*의 룰·필터로 후처리 |
| **외부 트렌드** | 시즌 컬러, 소재, 실루엣, 스타일 트렌드, 사회·문화 이벤트 | 별도 피처 엔지니어링 또는 무시 |

운영 의도와 트렌드가 파이프라인 *밖*에서 처리되면 세 가지 한계가 누적된다.

1. **거버넌스 단절** — 추천 결과가 운영 의도(directive)를 얼마나 준수했는지 모델 안에서 추론·검증 불가.
2. **설명 분절** — "왜 이 상품이 노출되었는가"의 답이 'CF 점수 + 사후 룰'로 쪼개진다.
3. **유연성 부족** — 자연어 운영 의도를 즉시 반영할 경로가 없다.

### 1.2 LLM·Multi-Agent의 등장 — 그리고 *얇은 agent* 문제

최근 LLM 멀티에이전트 추천 연구(AgentCF, MACRec, RecAgent, KAR 등)는 *agent reasoning* 자체를 추천 신호로 사용한다. 두 공통 한계:

1. **여전히 User–Item 2-stakeholder를 유지**하면서 "에이전트화"만 함 — 운영자·트렌드는 first-class가 아님.
2. **모든 agent에 *동일한 형태의 얇은 reflection*** 적용 — agent는 사실상 *호출당 stateless한 prompt template wrapper*.

---

## 2. 문제 정의

### 2.1 핵심 연구 질문

> **운영자 의도와 외부 트렌드를 일급 객체로 포함하는 4-stakeholder LLM 멀티에이전트 거버넌스에서, 각 stakeholder의 *본질적 차이*에 맞춰 *서로 다른 reasoning 메커니즘*을 어떻게 설계하면 추천 품질·거버넌스 정합·해석가능성이 동시에 향상되는가?**

### 2.2 핵심 thesis — **Heterogeneous Stakeholder Reasoning**

> *"4-stakeholder recommendation governance에서 각 stakeholder는 본질이 다르기 때문에 *서로 다른 종류의 reasoning*을 필요로 한다. MARGO는 stakeholder의 *역할 본질*에서 도출된 4가지 차별화된 reasoning 메커니즘을 제안한다."*

| Stakeholder | 본질 | 도출된 reasoning |
|---|---|---|
| **User** | 시간 따라 변하고, *좋아하는 것 ≠ 거부하는 것* | **Trajectory + Dual-Layer (Realistic / Rejected)** |
| **Item** | 정적이지만 *피드백 누적*으로 self-aware 가능 | **Autobiographical Memory + Light Self-Correction** |
| **Trend** | 외부 신호 — *코호트별 적용*을 통해 governance와 통합 | **Cohort-Conditional Multi-Source Pipeline** |
| **Expert** | 사람 운영자 — *경험에서 학습*하는 존재 | **Directive Outcome Pattern Learning** |

### 2.3 설계 원칙 — "Light but Real"

1. **모든 신규 메커니즘은 ablation toggle**로 끌 수 있음
2. **Memory는 *prompt evidence로 추가*만** — 기존 reasoning path 변경 X
3. **Retrieval은 *추가 정보*이지 *대체*가 아님**
4. **Off일 때 = v3 동작과 동일** (NDCG/HR baseline 보호)
5. **Over-engineering 금지** — counterfactual / adversarial 등 과한 메커니즘 제외

### 2.4 평가의 다층성

| 층 | 지표 | 측정 대상 |
|---|---|---|
| 추천 품질 | NDCG@K, HR@K | 사용자 선호 적합도 |
| 거버넌스 | DCR (Directive Compliance Rate), TAS (Trend Alignment Score) | 운영 의도·트렌드 정합 |
| Grounding | IHR, VDR, SVR | hallucination·드리프트·schema 위반 |
| **Heterogeneous Reasoning (Phase E)** | Memory Growth Curve, Self-Correction Precision, **Trend Predictive Validity (TPC)**, Trajectory Reasoning Accuracy | 각 agent reasoning 메커니즘 효과 |

---

## 3. 방법론

### 3.1 4-Stakeholder Multi-Agent

| Agent | 역할 |
|---|---|
| **User Agent** | trajectory + dual-layer로 선호 reasoning, 후보 평가 |
| **Item Agent** | 자기 평가 history를 기반으로 self-aware self-description |
| **Expert Agent** | 운영자 — directive 발행·검증·refine + outcome pattern 학습 |
| **Trend Agent** | 다중 소스 트렌드 해석 + cohort-conditional 적용 |

에이전트 간 통신은 Pydantic typed message protocol과 in-process MessageBus로 한정.
모든 에이전트는 `BaseAgent`를 상속, LLM 호출·프롬프트 렌더·SVR 검증 공통 처리.

### 3.2 4-Phase Lifecycle

```
Phase 1 (offline)  : Agent Initialization     — User profile (trajectory 포함), cohort/audience 사전 계산
Phase 2            : Directive Generation     — Expert directive (with past-brief memory) + Trend interpret
Phase 3            : Multi-Agent Reasoning    — Retrieval → Item self_describe (with reception memory)
                                                → Cohort-conditional trend applied
                                                → User Top-K rerank (with trajectory + rejected layer)
Phase 4            : Validation & Refinement  — Expert 검증, Fail 시 directive refine → Phase 2
                                                + outcome event appended to Expert memory
```

- **Validation loop**: `max_iterations=3`, convergence score ≥ 0.85 시 종료
- **Negotiation loop** (코드 보존, default OFF): `enable_negotiation=False`. MD 의도 vs 트렌드 충돌 의미론이 충분히 grounded되지 않음.
- **Directive-aware retrieval**: profile + directive NL + trend keywords 합성한 단일 semantic query.

### 3.3 3-Layer Rationale

모든 Top-K 추천 항목은 세 층의 근거를 동반.

1. **Personal** — User Agent의 *다축 선호 + trajectory + (적용 시) rejection pattern 회피*
2. **Directive** — Expert Agent의 운영 의도 (*유사 brief outcome 학습 기반*)
3. **Trend** — Trend Agent의 다중 소스 해석 (*user cohort에 맞춰 재가중됨*)

---

## 4. Heterogeneous Stakeholder Reasoning — 각 Agent 구현 spec

### 4.1 User Agent — *Preference Trajectory + Dual-Layer*

**본질**: 사용자는 *시간 따라 변하고*, *좋아하는 것 ≠ 거부하는 것*.

#### 기존 (v3 완료)
- Multi-Axis Preference (style + price/category/brand) — [`messages.py:PreferenceAxis`](MARGO/src/core/protocol/messages.py)
- Axis-Derived Cohort + Peer Signal — [`user_agent.get_peer_signal`](MARGO/src/core/agents/user_agent.py)
- NL 페르소나 + 4축 evaluate

#### 신규 (Phase C 완료)
| 기능 | 구현 위치 |
|---|---|
| **Preference Trajectory** (recent 4w vs full, `stability` 채워짐) | [`preference_stats.compute_stability`](MARGO/src/data/fashion/preference_stats.py#L237) |
| **Rejected Layer Loader** (rating 1-2 → user_id 매핑) | [`rejected_loader.load_rejected_history`](MARGO/src/data/fashion/rejected_loader.py) |
| **Rejection Pattern Summariser** (top cats/brands/style hints) | [`rejection_pattern.summarise_rejection_pattern`](MARGO/src/data/fashion/rejection_pattern.py) |
| **UserAgent.get_rejection_pattern** (caching) | [`user_agent.py:233`](MARGO/src/core/agents/user_agent.py) |
| **Directive.policy_hint** (daily / trend_push / cohort_expansion) | [`messages.py:Directive.policy_hint`](MARGO/src/core/protocol/messages.py) |
| **Prompt** `user.evaluate.j2`에 rejected_summary + policy 섹션 | [prompts/phase3_reasoning/user.evaluate.j2](MARGO/prompts/phase3_reasoning/user.evaluate.j2) |

#### Ablation
- `enable_user_trajectory: bool = True` (preference_stats 자동 호출)
- `enable_rejected_layer: bool = True` ([api.py:159](MARGO/src/api.py))

#### 추천 quality 영향
- **Rejected layer on: NDCG 양의 영향 기대** (단순 metadata 아닌 실질 정보)
- 사용자 37.2%가 rejected 신호 보유 (1.35M rows)

---

### 4.2 Item Agent — *Autobiographical Memory*

**본질**: 정적이지만 *피드백이 누적되는 위치*. 시간이 가면서 self-aware해질 수 있다.

#### 기존 (v3 완료)
- Audience Profile (buyer_aggregate) — `MIN_BUYERS=3`
- Context-Aware Self-Describe
- Trend Self-Positioning (`TrendPosition`)

#### 신규 (Phase B 완료)
| 기능 | 구현 위치 |
|---|---|
| **ItemMemory** (per-item JSONL store, cohort filter) | [`memory/schemas.py:ItemMemory`](MARGO/src/core/memory/schemas.py) |
| **Reception Logging** (cohort × month 집계 가능) | [`make_item_reception_event`](MARGO/src/core/memory/schemas.py) |
| **Audience Claim Logging** (unverified, Phase E에서 verify) | [`make_item_audience_claim_event`](MARGO/src/core/memory/schemas.py) |
| **Cohort-Conditioned Retrieval** | [`_item_cohort_filter`](MARGO/src/core/memory/schemas.py) |
| **ItemAgent.memory wire** | [`item_agent.py:99`](MARGO/src/core/agents/item_agent.py) |
| **Self-Correction Warning** (verified accuracy < threshold) | [`_self_correction_warning`](MARGO/src/core/agents/item_agent.py) |
| **Prompt** `item.describe.j2`에 past_reception + claim_warning 섹션 | [prompts/phase3_reasoning/item.describe.j2](MARGO/prompts/phase3_reasoning/item.describe.j2) |
| **Phase 3 통합** (reception_event auto-append) | [`phase3_reasoning.py:run_phase3`](MARGO/src/core/lifecycle/phase3_reasoning.py) |

#### Ablation
- `enable_item_memory: bool = True` (memory_root 설정 시에만 효력)

#### 메모리 폭발 방지
- Storage: ~10KB/item × 50K items ≈ **500MB**
- Context: cohort slice만 LLM에 ~500 tokens

---

### 4.3 Trend Agent — *Trade-press 4-Axis Pipeline (v5)*

**본질**: 외부 신호 — *후행 catalog 행동으로 검증 가능*. governance 통합의 핵심은 *cohort별 적용*과 *Expert 우위 (advisory only)*.

#### v5 Pipeline — **Collect → Synthesize (2-step)**

LLM 시대의 패션 트렌드 신호는 *trade-press 본문 자체*가 가장 직접적이라는 관찰에서 출발. 수치적 시계열(Wikipedia pageview / GDELT mention count)을 Tavily search로 trade-press 본문에 직접 접근하는 흐름으로 단순화.

| Step | 책임 | 산출 |
|---|---|---|
| **Step 1 — Collect** | Tavily 4-axis search (deterministic, no LLM). 42개 fashion 도메인 allowlist + 시즌 publish-date 필터. | `CollectResult` (~45 articles) |
| **Step 2 — Synthesize** | 단일 LLM call (temperature=0). 4축 cross-check로 trend 발견 + lifecycle 결정. | `SeasonTrendSnapshot` |

**4 Axes — trend signal이 *어디서 originate하는가*** (직교적 분류):

| Axis | 의미 | Query 예 (2023-SS) | 주된 도메인 |
|---|---|---|---|
| **RUNWAY** | designer origin (패션쇼) | "spring 2023 fashion week runway designer collections key trends" | vogue, voguebusiness, businessoffashion |
| **PRESS** | editorial consensus (권위 매체 정리) | "defining fashion trends and aesthetics of spring 2023 editorial report" | elle, wwd, harpersbazaar, fashionista |
| **STREET** | consumer / wearability | "spring 2023 fashion street style shopping wearable items what to wear" | refinery29, whowhatwear, popsugar, cosmopolitan |
| **POP** | celebrity / TV·movie·music / viral | "spring 2023 viral fashion celebrity outfits tv movie music influence" | people, glamour, allure, mashable, variety |

**Schema (v5 — 단순화)**:

```python
SeasonTrendSnapshot:
  snapshot_id, season, window_start_iso, window_end_iso, snapshot_date
  summary, headline_themes (3-5)
  trends: list[TrendItem]
  sources_used (실제 hit한 도메인), article_count, notes

TrendItem:
  keyword
  trend_stage: niche | rising | mainstream    # 4축 cross-check 결과
  confidence:  high | medium | low
  rationale                                   # Vogue 등 직접 cite한 자연어
  citations: list[Citation]                   # title, url, domain, publish_date, excerpt
```

**trend_stage 분류 기준** (LLM이 본문 *언어*로 판단):
- `mainstream` — 3+ 축에 prominent (designer + press + (street | pop))
- `rising` — 2 축에 momentum 언어 ("next", "to watch", "designers are showing")
- `niche` — 1 축만, 또는 momentum 부재

**파일 명명**: `<processed_dir>/trend_cache/fashion_trend_<year>_<SS|FW>.json`

#### v5 17-시즌 빌드 결과 (2015-SS ~ 2023-SS)

| 지표 | 값 |
|---|---|
| 빌드 성공 | 17/17 (failed=0), elapsed 33.8min |
| 총 trends | 208 (시즌당 평균 12.2) |
| trend_stage 분포 | mainstream=83, rising=79, niche=46 (균형 분포) |
| confidence | high=83, medium=91, low=34 |
| **Citations 채움률** | **100%** (모든 trend에 실제 trade-press URL + excerpt) |
| Citation 출처 다양성 | 18 unique domains 활용 |
| Top 5 cited | vogue.com (153), whowhatwear (29), wwd (24), harpersbazaar (22), fashionista (22) |

#### 비대칭 권한 (Asymmetric Authority)

> **Trend는 evidence(자문)이지 directive(결정)가 아님.** Expert가 결정권자.

| 단계 | Trend의 역할 | Expert의 권한 |
|---|---|---|
| Phase 2 협상 | tension을 *advisory로* 제시 (사실적 오류만) | accept / reject / counter 자유 |
| Phase 3 retrieval | mainstream+rising keywords로 *boost* | directive 정렬된 boost만 |
| Phase 3 evaluate | 점수 가중에서 *secondary* | personal > directive > trend 순서 |

#### Phase A — Cohort-Conditional Application
| 기능 | 구현 위치 |
|---|---|
| **Rule-based Conflict Table** (cohort axis 값 ↔ 충돌 keyword substring) | [`trend_agent._COHORT_CONFLICT_TABLE`](MARGO/src/core/agents/trend_agent.py) |
| **apply_cohort_conditioning** (drop-only, keyword 기반 fallback) | [`trend_agent.apply_cohort_conditioning`](MARGO/src/core/agents/trend_agent.py) |
| **Phase 3 통합** (user evaluate 직전 호출) | [`phase3_reasoning.py`](MARGO/src/core/lifecycle/phase3_reasoning.py) |

#### Phase E로 미룸
- **Predictive Validity 평가** (TPC, Lead-Lag Score)
- **Light Learned Reliability** (predictive validity → axis prior calibration)
- **Disagreement Typology** (4-axis 시각 충돌 분석)

#### Ablation
- `enable_cohort_conditional_trend: bool = True`
- `enable_trend_snapshot: bool = True` (False면 v5 season snapshot 경로 우회 → live web search로 fallback)
- v5 단일 경로. v3 google-trends·v4 multisource fallback은 retire되어 `src/previous/`로 격리 (코드 재현용 보존, runtime 의존 X)

---

### 4.4 Expert Agent — *Directive Outcome Pattern Learning*

**본질**: 사람 운영자 — *경험에서 학습*하는 존재.

#### 기존 (v3 완료)
- `issue_directive` (brief → Directive)
- Hard Constraint Validator (`_check_structured`) — mechanical 5종
- `refine_directive`

#### 신규 (Phase D 완료)
| 기능 | 구현 위치 |
|---|---|
| **ExpertMemory** (per-persona JSONL) | [`memory/schemas.py:ExpertMemory`](MARGO/src/core/memory/schemas.py) |
| **ExpertDirectiveOutcomeEvent** | [`make_expert_outcome_event`](MARGO/src/core/memory/schemas.py) |
| **record_outcome** (Phase 4 후 자동 호출) | [`expert_agent.record_outcome`](MARGO/src/core/agents/expert_agent.py) |
| **Cosine TF-lite similar-brief retrieval** | [`expert_agent.retrieve_similar_briefs`](MARGO/src/core/agents/expert_agent.py) |
| **Tokeniser + Cosine** (sklearn 의존성 X) | [`_tokenise_brief`, `_cosine`](MARGO/src/core/agents/expert_agent.py) |
| **Prompt** `expert.directive.j2`에 past_similar_briefs 섹션 + policy_hint 선택 안내 | [prompts/phase2_directive/expert.directive.j2](MARGO/prompts/phase2_directive/expert.directive.j2) |
| **Orchestrator hook** (validation 후 outcome append) | [`orchestrator.recommend`](MARGO/src/core/lifecycle/orchestrator.py) |

#### Ablation
- `enable_expert_memory: bool = True` (memory_root 설정 시 효력)

---

## 5. Cross-Agent Synergy — 4 agent가 *서로 강화*하는 방식

| Synergy | 메커니즘 | 측정 (Phase E) |
|---|---|---|
| **User trajectory × Trend cohort-cond** | User의 drift가 cohort 변화 → Trend가 그 cohort에 reweight | drift 큰 user에서 cohort-cond trend의 추가 lift |
| **Item autobiography × User Rejected** | User reject pattern이 Item의 false audience claim을 잡음 | rejected-aware items의 audience claim accuracy 향상 |
| **Item reception × Trend predictive validity** | Item 점수 history가 trend prediction의 *ground truth* | trend predictive validity score |
| **Expert pattern × Iteration loop** | 학습된 expert pattern으로 refine 수렴 횟수 감소 | average refinement_count over time |

---

## 6. 그라운딩 & Runtime Validation

| 메커니즘 | 메트릭 | 위치 |
|---|---|---|
| Catalog item ID 검증 | IHR | `core/validation/` |
| Vocabulary 어휘 매칭 | VDR | `core/validation/vocabulary.py` |
| Pydantic schema 위반 | SVR | `BaseAgent.llm_call` 공통 경로 |

---

## 7. 데이터 & 인프라

### 7.1 데이터셋
| 항목 | 선택 | 비고 |
|---|---|---|
| **도메인** | Amazon Reviews 2023 — Clothing, Shoes & Jewelry | 트렌드·가격·브랜드 다양성 |
| **Positive interactions** | rating ≥ 4.0, 5-core, leave-one-out | `train/valid/test.parquet` (12.3M rows) |
| **Rejected interactions** | rating 1-2, 5-core entity 제한 | `rejected.parquet` (1.35M rows) |
| **Review text** | text + helpful_vote + verified_purchase | `reviews_text.parquet` (18.45M rows) |

### 7.2 Retrieval / Baseline
| 항목 | 선택 | 위치 |
|---|---|---|
| Primary Retriever | BGE-M3 + FAISS, directive-aware query | `adapters/retrieval/bge_retriever.py` |
| Fallback Retriever | BM25 | `adapters/retrieval/bm25_retriever.py` |
| CF Baseline | LightGCN (scaffolding — Phase F에서 구현) | `src/previous/baselines/lightgcn.py` (격리, 인터페이스만 pinned) |
| **AgentCF Baseline (Phase F)** | 1-shot reflection 비교용 | `src/previous/baselines/agentcf.py` (TODO) |

### 7.3 Trend Sources — v5 Trade-press Pipeline

**Primary**: Tavily search API (web search + content extraction), 42개 fashion 도메인 allowlist, 4-axis search plan (RUNWAY / PRESS / STREET / POP). 시즌당 4 queries × max_results 12 = ~48 articles.

| 신호 source | 역할 | 상태 |
|---|---|---|
| **Tavily Search API** | Vogue/WWD/Elle/BoF 등 trade-press 본문 직접 접근 | ✅ 활성 (무료 1000 calls/월) |
| **42-domain allowlist** | 8 tiers (high-fashion / streetwear / women's lifestyle / men's / indie / industry / lifestyle / mass-pop-culture) | ✅ 활성 |
| **LLM (Synthesize)** | 4-axis cross-check → trend_stage + rationale + citations | gpt-4o-mini (temperature=0) |
| ~~Legacy v3/v4 (GDELT / Wikipedia / Pinterest / YouTube / Google Trends)~~ | ~~multi-source consensus~~ | **Retired** — code at `src/previous/adapters/trends/`, data at `data/previous/trend_cache_legacy/` & `data/previous/external_trends/gdelt/` |

**왜 v5는 quantitative source(Wikipedia/GDELT)를 제거했나**:
- 17 시즌 v3/v4 빌드 분석 결과 Wikipedia pageview는 “관심도” 신호일 뿐 *방향성* 약함, GDELT는 bigram 키워드 매칭 실패율 높음
- LLM이 trade-press 본문 *언어* (“to watch”, “designers are showing”, “last season”)를 보고 lifecycle을 판단하는 것이 *수치 시계열*보다 직접적
- 또한 schema 단순화: SourceEvidence dict / attributes dict / matched_items 제거 → variance 큰 필드 정리, citations(직접 URL 인용)로 대체

### 7.4 LLM / 프롬프트
| 항목 | 선택 |
|---|---|
| LLM Backend | Qwen2.5-7B-Instruct (vLLM 로컬, port 8000) + GPT-4o (보조) |
| 환경변수 | `MARGO_LLM_BACKEND`, `MARGO_VLLM_BASE_URL` |
| 프롬프트 | Jinja2, phase별 폴더 — `prompts/phase{1..4}_*/` |

### 7.5 Memory Persistence (신규)
- 위치: `<memory_root>/{user,item,trend,expert}/<entity_id>.jsonl`
- 형식: Append-only JSONL (event당 한 줄)
- git에 안 올림 (`.gitignore` 포함됨)
- 활성화: `MARGO_MEMORY_ROOT=<path>` 환경변수 또는 `MargoEngineConfig.memory_root=<path>`

### 7.6 Repository Layout — *v5 cleanup (2026-05-27)*

연구 focus가 v5 trade-press pipeline으로 단일화되면서, runtime에 활성인 모듈/데이터만 root에 두고 v3·v4 잔재는 모두 `previous/` 격리. **삭제하지 않은 이유**: 코드 재현성 + 추후 quantitative source 재도입 가능성.

| 영역 | Live (사용 중) | Retired → `previous/` |
|---|---|---|
| **데이터** | `data/Amazon Fashion/processed/trend_cache/fashion_trend_*.json` (17 v5 시즌) + `wikipedia/`, `youtube/` 캐시 | `data/previous/trend_cache_legacy/` (multisource_*, season_fashion_*, fashion_2022_*, google_trends_*, raw/ — 5.6MB) · `data/previous/external_trends/gdelt/` (508MB) |
| **Trend adapters** | `src/adapters/trends/{season_pipeline, season_collect, season_synthesize, season_schema, seasons, web_search, snapshot}.py` | `src/previous/adapters/trends/` — v4 multisource pipeline (`base.py`, `pipeline.py`, `consensus.py`, `multisource_schema.py`, `semantic_mapper.py`, `signal_normalizer.py`) + 5 source adapters (gdelt, wikipedia, pinterest, youtube, google_trends, google_trends_adapter) + v3 (`snapshot_schema.py`, `keyword_pool.py`) |
| **Lifecycle** | `phase{2,3,4}_*.py`, `orchestrator.py` | `src/previous/core/lifecycle/phase1_initialization.py` (offline preprocessing — 호출처 없음) |
| **Baselines** | (없음 — Phase F에서 다시 활성화) | `src/previous/baselines/{__init__.py, lightgcn.py}` (인터페이스 scaffolding) |
| **Build scripts** | `build_all_seasons.py` (v5), `build_season_snapshot.py`, `build_index.py`, `build_buyer_aggregate.py`, `build_cohort_stats.py`, `build_negative_signals.py`, `build_user_states.py`, `evaluate*.py`, `sanity_*.py` | `scripts/previous/` — `build_trend_snapshot.py` (v3), `build_monthly_snapshots.py`, `build_multisource_trend_snapshot.py`, `build_all_seasons_v4.py`, `download_gdelt_filtered.py`, `pinterest_oauth.py`, `preprocess.py`, `train_lightgcn.py`, `probe_pytrends.py`, `diagnose_cohort_coverage.py` |
| **Tests** | 14 live test files (`135 tests pass`) | `tests/previous/` — `test_trend_pipeline.py`, `test_pinterest_adapter.py`, `test_youtube_adapter.py` (v4 pipeline + source adapters) |

**pytest 설정** (`pyproject.toml`): `norecursedirs = ["tests/previous", "previous"]` + `--ignore=tests/previous` — `previous/`는 collection에서 제외.

**TrendAgent 진입점 (단순화)**:
```
MargoEngine.__init__ → TrendAgent(processed_dir=cfg.processed_dir, ...)
  ↓ recommend()
TrendAgent.interpret_trend()
  ↓ _load_season_snapshot()  ← 단일 primary path
SeasonTrendSnapshot.to_interpretation() → TrendInterpretation
  → (없으면) live web search via WebSearcher  ← 유일 fallback
```

---

## 8. Ablation Flag 전체 정리 (현재 코드)

`MargoRunConfig` 모든 플래그는 default **True**. *Off하면 v3 동작*.

| Flag | Phase | 효과 |
|---|---|---|
| `enable_multi_axis` | v3 | 4축 deterministic 계산 (이거 off면 모든 cohort 메커니즘 무력화) |
| `enable_peer_signal` | v3 | cohort_stats → peer ratio |
| `enable_audience_profile` | v3 | item의 buyer_aggregate 로딩 |
| `enable_trend_position` | v3 | Item이 trend 대비 자기 position 명시 |
| `enable_trend_snapshot` | v5 | v5 season snapshot 사용 (False면 live web search fallback). v3/v4 multisource는 retired |
| `enable_negotiation` | v3 (default **False**) | Expert↔Trend 도전 |
| **`enable_cohort_conditional_trend`** | **A** | Trend 결과를 user cohort에 reweight |
| **`enable_item_memory`** | **B** | Item autobiography (memory_root 필요) |
| **`enable_rejected_layer`** | **C** | rating 1-2 evidence를 evaluate prompt에 |
| **`enable_expert_memory`** | **D** | Directive outcome log + similar brief retrieve (memory_root 필요) |

---

## 9. 차별점 (vs 관련 연구)

| 비교 대상 | MARGO의 차별점 |
|---|---|
| **AgentCF** (WWW'24) | 균일 thin reflection → MARGO는 **stakeholder별 차별화 reasoning + typed cohort-aggregated memory + self-correction** |
| **MACRec** (SIGIR'24) | 균일 discussion → MARGO는 **role-derived heterogeneous reasoning** |
| **RLMRec** (WWW'24) | Static profile → MARGO는 **profile + trajectory + dual-layer (realistic/rejected)** |
| **KAR** (RecSys'24) | factor-level reasoning → MARGO는 **per-axis match + 코호트 peer signal + 거버넌스 결합** |
| **RecExplainer** (KDD'24) | buyer pattern 설명 → MARGO는 **autobiographical memory + self-correction loop** |
| **Generative Agents** / **MemoryBank** | 일반 LLM agent memory → MARGO는 **추천 도메인 4-stakeholder role-derived memory** |
| **LLM trend 분석** (다수) | plausibility만 평가 + 균일 적용 → MARGO는 **predictive validity (Phase E) + cohort-conditional 적용** |
| **LightGCN 등 CF** | 운영 의도·트렌드를 모델 바깥 처리 | MARGO는 **first-class agent 통합** |

---

## 10. 기여 요약 (Contributions)

### Tier 1 — Framework
**C1. Heterogeneous Stakeholder Reasoning** — 4-stakeholder governance에서 각 stakeholder role 본질에 맞춰 *서로 다른 reasoning 메커니즘*이 필요함을 주장·구현·증명. 균일 reflection의 기존 LLM rec agent와 정면 대조.

### Tier 2 — Per-Agent Methodological Contributions
**C2.User: Preference Trajectory + Dual-Layer (Realistic/Rejected)** — 시간 trajectory + 양방향 선호. Rejected layer는 **NDCG 양의 영향 기대**.

**C2.Item: Cohort-Aggregated Autobiographical Memory + Light Self-Correction** — stateless wrapper → first-class agent로 격상. **Self-Correction Precision Curve**가 paper의 강한 figure.

**C2.Trend: Cohort-Conditional Multi-Source Pipeline** — 5-layer pipeline 결과를 *user cohort에 맞춰 reweight*. Phase E에 *predictive validity + learned reliability + disagreement typology* 확장.

**C2.Expert: Directive Outcome Pattern Learning** — Experiential operator. Governance iteration이 학습 데이터.

### Tier 3 — Synergy
**C3. Cross-Agent Reasoning Synergy** — 4 메커니즘이 *상호 강화*하는 합성 효과를 식별·평가.

### Tier 4 — Evaluation (Phase E)
**C4. Heterogeneous Reasoning Evaluation Framework** — Memory Growth Curve, Self-Correction Precision, **Trend Predictive Validity (TPC)**, Trajectory Reasoning Accuracy. Temporal split + cold/warm + per-agent ablation.

### Tier 5 — System
**C5. End-to-end MARGO Codebase** — 데이터 전처리·검색·4-phase 오케스트레이터·FastAPI+Next.js 웹 데모. v3 + Phase A·B·C·D 통합 완료 + v5 trade-press pipeline 단일화 (2026-05-27 정리: v3/v4 fallback `previous/` 격리), **135 tests pass** (Phase 0–D live suite).

---

## 11. Roadmap — 현재 status

| Phase | 작업 | 비용 | 상태 |
|---|---|---|---|
| ✅ M1-M6 | 4-agent / 4-phase 골격, v3 #1-#4 통합, multi-source trend pipeline, 웹 데모 | — | **완료 (87 tests)** |
| ✅ M7 | Rejected layer + Review text 데이터 인프라 | — | **완료** (`rejected.parquet` 1.35M + `reviews_text.parquet` 18.45M) |
| ✅ **Phase 0** | Memory infrastructure (base + persistence + 4 agent schemas) | — | **완료 (+18 tests)** |
| ✅ **Phase A** | Trend Cohort-Conditional Application | — | **완료 (+16 tests)** |
| ✅ **Phase B** | Item autobiographical memory + light self-correction | — | **완료 (+15 tests)** |
| ✅ **Phase C** | User trajectory + Rejected layer + binary dual-layer policy | — | **완료 (+14 tests)** |
| ✅ **Phase D** | Expert outcome log + similar brief retrieval | — | **완료 (+16 tests)** |
| 🟡 Web 통합 | margo_runner에 `MARGO_MEMORY_ROOT` 환경변수 통합 | — | **완료** (UI 변경 없음 — Phase F에서 확장) |
| ⏳ **Phase E** | 평가 framework — temporal split + per-agent ablation + **Predictive Validity** + **Disagreement Typology** + **Learned Reliability** + AgentCF baseline | 3-4주 | 다음 |
| ✅ **v5 Trend pipeline + repo cleanup** | Trade-press 4-axis pipeline (17 시즌 빌드 100% citations) + v3/v4 fallback → `previous/` 격리 + TrendAgent v5-only로 단순화 | — | **완료 (2026-05-27)** |
| ⏳ Phase F | Paper draft + final experiments + Web UI 확장 | 2-3주 | |

**Live test suite**: `pytest tests/` → **135 tests pass** (v4 pipeline 테스트 3개는 `tests/previous/`로 격리, pytest collection 제외).

---

## 12. 보류 / Future Work

### Pinterest 통합 (Phase A에서 시도)
- 코드 완료, Sandbox token 받음, Sandbox는 Trends API 접근 불가 확인
- Trial access 승인 대기 (24-72시간)
- 승인 시 token만 production token으로 교체 → 자동 4-source

### Social platform 신호 (Reddit / Twitter / Instagram / TikTok)
- Reddit: 2024-2025 personal app 사실상 봉쇄
- 외 paid 또는 academic 승인 수개월 소요
- **Paper limitation 섹션에 명시** — academic LLM 연구 공통 제약

### Negotiation Loop (코드 보존, default OFF)
- `enable_negotiation=False` 기본 — MD 의도 vs 트렌드 충돌 의미론이 grounded되지 않음
- 코드 + 10 테스트 보존
- Phase E의 *Predictive Validity + Disagreement Typology* 완성 후 evidence 단단해지면 재고

### Web UI 새 evidence 표시 (Phase F)
- 현재 백엔드에서 cohort-cond / rejected layer / item memory 작동 (MARGO_MEMORY_ROOT 설정 시)
- 다만 *Demo 화면이 새 evidence를 시각화하지는 않음*
- past_reception, claim_warning, past_briefs, dual-layer policy 입력 등 UI 확장은 Phase F에서

---

## 13. 관련 문서

- [`README.md`](./README.md) — 설치·빠른 시작
- [`RESEARCH_SUMMARY.md`](./RESEARCH_SUMMARY.md) — 구현 현황·디렉토리·테스트 현황
- [`MARGO_Implementation_Plan.md`](./MARGO_Implementation_Plan.md) — 상세 로드맵·체크리스트
- [`MARGO_Enhancement_Directive_v3.md`](./MARGO_Enhancement_Directive_v3.md) — v3 enhancement 원본 directive
- [`PRIVACY.md`](./PRIVACY.md) — 외부 API 사용 정책
- [`web/README.md`](./web/README.md) — 웹 데모 실행·프록시 설정

---

*POSTECH AIM Lab · MARGO 연구 프로토타입 · 2026-05-25 갱신 (Phase A·B·C·D 완료, 166 tests pass)*
