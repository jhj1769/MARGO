# MARGO 연구 정리

> **MARGO** — *A Multi-Agent Framework for Stakeholder-Aware Recommendation Governance*  
> 기존 약칭 **SAGE**(*Stakeholder-Aware Recommendation Governance*)에서 프로젝트명을 변경하였다.  
> 본 문서는 2026년 5월 기준 구현·실험·웹 데모까지의 연구 내용을 한곳에 정리한다.

---

## 1. 연구 배경과 문제 정의

### 1.1 한계: User–Item 2-stakeholder 추천

전통적 추천 시스템은 **사용자(User)** 와 **아이템(Item)** 두 이해관계자만을 전제한다. 그러나 실제 서비스(특히 패션·커머스)에서는 다음 주체가 추천 결과에 직접 개입한다.

| 주체 | 역할 | 기존 시스템에서의 처리 |
|------|------|------------------------|
| **운영자 (MD/PM)** | 시즌 캠페인, 카테고리 부스팅, 가격대 정책 | 오프라인 룰·필터로 후처리 |
| **외부 트렌드** | 시즌 컬러, 소재, 스타일 트렌드 | 피처 엔지니어링 또는 무시 |

운영 의도와 트렌드 맥락이 **추천 파이프라인 밖**에서 처리되면, (1) directive 준수를 보장하기 어렵고, (2) “왜 이 상품이 노출되었는지” 설명이 분절된다.

### 1.2 연구 질문

> **자연어로 표현된 운영 의도(directive)와 외부 트렌드 맥락을, LLM 멀티에이전트 협업 구조 안에서 어떻게 추천 거버넌스(governance)의 일급 객체로 통합할 수 있는가?**

MARGO는 이 질문에 대해 **4-stakeholder 멀티에이전트**와 **4-phase lifecycle**, **grounding layer**로 답을 제시한다.

---

## 2. MARGO 프레임워크 개요

### 2.1 약어와 포지셔닝

| 항목 | 내용 |
|------|------|
| **약자** | **MARGO** |
| **풀네임** | A Multi-Agent Framework for Stakeholder-Aware Recommendation Governance |
| **핵심 키워드** | Multi-Agent, Recommendation, Governance, Orchestration |
| **도메인** | Amazon Reviews 2023 — Clothing, Shoes & Jewelry (Fashion) |
| **LLM** | Qwen2.5-7B-Instruct (vLLM 로컬) + GPT-4o (보조) |

### 2.2 4-Stakeholder Multi-Agent

| Agent | 역할 | 주요 Skill |
|-------|------|------------|
| **User Agent** | 사용자 선호를 자연어로 reasoning | `query_preference`, `evaluate_candidate`, `update_profile` |
| **Item Agent** | directive·trend를 반영한 context-aware 자기서술 | `self_describe`, `update_reflection` |
| **Expert Agent** | 운영자 persona — directive 발행·검증·refine | `issue_directive`, `validate_recommendation`, `refine_directive` |
| **Trend Agent** | 웹/트렌드 스냅샷 해석·broadcast | `query_trend`, `interpret_trend`, `broadcast` |

에이전트 간 통신은 **Pydantic 기반 typed message protocol**(`margo.protocol`)과 in-process **MessageBus**로 제한되어, 자연어 reasoning이 스키마 밖으로 새지 않도록 설계하였다.

### 2.3 4-Phase Lifecycle

```
Phase 1 (offline)  : Agent Initialization — User/Item 프로필·설명 생성
Phase 2            : Directive Generation — Expert directive + Trend broadcast
Phase 3            : Multi-Agent Reasoning — Retrieval → Item self_describe → User evaluate
Phase 4            : Validation & Refinement — Expert 검증, Fail 시 directive refine → Phase 2
```

- **Validation loop**: `max_iterations=3`, convergence score ≥ 0.85 시 종료  
- **Orchestrator**: `MargoOrchestrator` (`margo.lifecycle.orchestrator`) — Phase 2→3→4를 imperative하게 연결 (LangGraph와 동등한 상태 전이, 디버깅 용이성 우선)

### 2.4 3-Layer Rationale

각 Top-K 추천 항목은 다음 세 층의 근거를 동반한다.

1. **Personal** — User Agent가 history·profile 기반으로 생성  
2. **Directive** — Expert Agent의 운영 의도(structured + NL hybrid)  
3. **Trend** — Trend Agent의 시즌·키워드 해석  

웹 데모(`/demo`)에서 카드 hover·Insight Stream으로 시각화한다.

---

## 3. 기술 스택과 설계 결정 (Locked)

| 항목 | 결정 | 근거 |
|------|------|------|
| **Primary Retriever** | BGE-M3 (`BAAI/bge-m3`) + FAISS | directive-aware semantic query, Item Agent NL description과 호환 |
| **Baseline Retriever** | BM25, LightGCN | modular ablation |
| **Trend Agent** | LLM + web search (Tavily/SerpAPI/stub) + Google Trends 스냅샷 | 외부 맥락의 재현 가능한 캐싱 |
| **Grounding** | Domain vocabulary + SchemaValidator + TrendSnapshotStore | IHR/VDR/SVR/CADR 측정·hallucination 억제 |
| **LLM Client** | `MARGO_LLM_BACKEND` = openai \| vllm \| dummy | 통합 인터페이스, usage 로깅 |
| **웹** | FastAPI + Next.js 14, WebSocket trace | split-screen MD 콘솔 + Consumer View 데모 |

---

## 4. 구현 현황 (코드베이스)

프로젝트 루트: `/home/hjjung/research/MARGO`

### 4.1 디렉토리 구조

```
MARGO/
├── src/margo/           # 핵심 프레임워크 패키지
│   ├── agents/          # User, Item, Expert, Trend
│   ├── protocol/        # Message, MessageBus
│   ├── lifecycle/       # phase1–4, MargoOrchestrator
│   ├── retrieval/       # BGE-M3, BM25, LightGCN
│   ├── grounding/       # vocabulary, schema_validator, snapshot
│   ├── llm/             # LLMClient, Jinja2 prompts
│   ├── trend_sources/   # web_search, google_trends, keyword_pool
│   ├── evaluation/      # standard, governance, grounding metrics
│   ├── domains/fashion/ # Amazon Fashion loader, personas, vocab
│   └── api.py           # MargoEngine public façade
├── scripts/             # preprocess, build_index, evaluate, trend snapshot
├── tests/               # protocol, retrieval, orchestrator, evaluation
├── web/
│   ├── backend/         # FastAPI, MargoRunner, mock fallback
│   └── frontend/        # Landing, Architecture, Interactive Demo
├── data/Amazon Fashion/ # raw + processed (gitignore)
├── MARGO_Implementation_Plan.md
└── RESEARCH_SUMMARY.md  # 본 문서
```

### 4.2 주요 진입점

| 용도 | 명령 / 모듈 |
|------|-------------|
| 전처리 | `python -m scripts.preprocess` |
| BGE 인덱스 | `python -m scripts.build_index` |
| 단일 사용자 sanity | `python -m scripts.sanity_one_user` |
| 배치 평가 | `python -m scripts.evaluate` |
| 트렌드 스냅샷 | `python -m scripts.build_trend_snapshot` |
| 웹 데모 일괄 실행 | `./web/run.sh` |
| 엔진 API | `MargoEngine.recommend(user_id, directive, k)` |

### 4.3 환경 변수 (`.env`)

| 변수 | 설명 |
|------|------|
| `MARGO_LLM_BACKEND` | `openai` / `vllm` / `dummy` |
| `MARGO_LLM_MODEL` | 모델명 (예: Qwen2.5-7B-Instruct) |
| `MARGO_VLLM_BASE_URL` | vLLM OpenAI-compatible endpoint |
| `MARGO_PROCESSED_DIR` | 전처리 데이터 경로 (엔진 모드 필수) |
| `MARGO_BGE_DEVICE` | BGE GPU 디바이스 |
| `MARGO_DEMO_MODE` | `mock` 시 canned 데이터 |
| `MARGO_TREND_BACKEND` | `auto` / `tavily` / `serpapi` / `stub` |

---

## 5. 데이터 파이프라인

### 5.1 Amazon Fashion

- **소스**: Amazon Reviews 2023 — Clothing, Shoes & Jewelry  
- **전처리** (`scripts/preprocess.py`):
  - 5-core filtering (user·item 각 ≥5 interactions)
  - rating ≥ 4.0 → positive (implicit feedback)
  - Leave-one-out split (최신 → test, 직전 → valid)
  - `train/valid/test/items.parquet`, `vocab_fashion.json`

### 5.2 Retrieval 인덱스

- Item text: `title + description + category + color + material + price`
- BGE-M3 embedding → FAISS (`item_embeddings.npy`, `faiss_index.bin`)
- **Directive-aware query**: `User: {profile}\nIntent: {directive_nl}`

---

## 6. 평가 지표

### 6.1 표준 추천 품질 (`margo.evaluation.standard`)

- **NDCG@K**, **HR@K** (leave-one-out)

### 6.2 Governance (`margo.evaluation.governance`)

- **DCR** (Directive Compliance Rate): Top-K 중 structured constraint 위반 없는 비율  
- **TAS** (Trend Alignment Score): Trend signal과 item attribute 정합

### 6.3 Grounding (`margo.evaluation.grounding`)

- **IHR** (Item Hallucination Rate): catalog 외 item ID  
- **VDR** (Vocabulary Drift Rate): trend output의 vocab 미매칭  
- **CADR** (Cross-Agent Disagreement Rate): Expert–Trend challenge 빈도  
- **SVR** (Schema Violation Rate): Pydantic schema 위반

### 6.4 계획된 Ablation

| ID | 제거/변경 | 관찰 지표 |
|----|-----------|-----------|
| A | Expert Agent | DCR |
| B | Trend Agent | TAS |
| C | Expert + Trend | AgentCF 수준 회귀 |
| D | Grounding layer | IHR, VDR, CADR, SVR |
| E | Directive 형식 (structured / NL / hybrid) | DCR, 해석 가능성 |
| F | Retriever (BGE / BM25 / LightGCN) | NDCG, directive 민감도 |

---

## 7. 웹 데모

### 7.1 화면 구성

| 경로 | 설명 |
|------|------|
| `/` | Landing — MARGO 소개, 4-agent·4-phase 설명 |
| `/architecture` | 프레임워크 다이어그램 |
| `/demo` | Split-screen: Consumer View + Expert Console + Insight Stream |

### 7.2 백엔드 모드

- **engine**: `MargoEngine` + Amazon Fashion processed 데이터  
- **mock**: `web/backend/data/mock.json` — GPU/LLM 없이 UI 개발·시연  

API prefix: `/api/margo/*`, WebSocket: `/ws/margo/trace`

### 7.3 데모 시나리오 (의도)

1. MD가 Expert Console에서 자연어 directive 입력 (예: casual→formal upsell, 카테고리 부스팅)  
2. BGE retriever가 **후보 pool 자체**가 바뀌는 것을 보여줌  
3. User/Item Agent reasoning 후 Top-K + 3-layer rationale 갱신  
4. WebSocket으로 phase별 agent 메시지 스트리밍  

---

## 8. Baseline 및 관련 연구

| 방법 | 역할 |
|------|------|
| **BM25** | sparse retrieval baseline |
| **LightGCN** | collaborative filtering baseline |
| **AgentCF** | LLM user/item profile 참고 구현 |
| **MACF** | multi-agent CF 비교 (arXiv:2511.18413) |

MARGO의 차별점: **governance loop**(Expert validation + refine)와 **trend grounding**, **directive-aware dense retrieval**의 결합.

---

## 9. 마일스톤 (Implementation Plan 대비)

| 단계 | 내용 | 상태 |
|------|------|------|
| M1 | 데이터 전처리, BGE 인덱스, BM25/LightGCN baseline | ✅ 구현됨 |
| M2 | LLM client, protocol, User/Item Agent | ✅ 구현됨 |
| M3 | Expert Agent, Trend Agent, snapshot | ✅ 구현됨 |
| M4 | Full orchestrator, evaluation scripts | ✅ 구현됨 |
| M5 | 웹 데모 (landing, architecture, demo, WS trace) | ✅ 구현됨 |
| — | 대규모 ablation·논문 수치 | 🔄 진행 예정 |

상세 체크리스트는 [`MARGO_Implementation_Plan.md`](./MARGO_Implementation_Plan.md) 참고.

---

## 10. SAGE → MARGO 리네이밍 (2026-05)

다음 항목이 일괄 변경되었다.

| 구분 | 이전 | 이후 |
|------|------|------|
| 프로젝트 폴더 | `SAGE/` | `MARGO/` |
| Python 패키지 | `src/sage` | `src/margo` |
| PyPI 이름 | `sage` | `margo` |
| 엔진 클래스 | `SageEngine` | `MargoEngine` |
| 웹 runner | `sage_runner.py` | `margo_runner.py` |
| 환경 변수 prefix | `SAGE_*` | `MARGO_*` |
| API 경로 | `/api/sage`, `/ws/sage` | `/api/margo`, `/ws/margo` |
| 구현 계획서 | `SAGE_Implementation_Plan.md` | `MARGO_Implementation_Plan.md` |

**유지된 것**: Tailwind 색상 토큰 `sage` (#7F8B5A), mock 상품명의 “Sage” 컬러 표현 — 디자인·상품 속성용으로 프로젝트 약자와 무관.

**가상환경**: README 기준 `python3 -m venv margo`. 기존 `sage/` venv가 있다면 새로 생성하는 것을 권장 (`pip install -e .`).

---

## 11. 향후 작업

1. **정량 실험 완료**: NDCG/HR + DCR/TAS + grounding metrics 전체 test set  
2. **Ablation (A–F)** 표준화 및 논문 table  
3. **Comparison mode**: MARGO vs LightGCN vs AgentCF 나란히 데모  
4. **Trend snapshot**: Google Trends 기반 캐시 파이프라인 운영화  
5. **논문/프리프린트**: Landing `#paper` 섹션 연결  

---

## 12. 참고 문서

- [`README.md`](./README.md) — 설치·빠른 시작  
- [`MARGO_Implementation_Plan.md`](./MARGO_Implementation_Plan.md) — 상세 로드맵·체크리스트  
- [`web/README.md`](./web/README.md) — 웹 데모 실행·프록시 설정  

---

*작성: POSTECH AIM Lab · 연구 프로토타입 MARGO · 2026년 5월*
