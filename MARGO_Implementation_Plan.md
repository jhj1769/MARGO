# MARGO Implementation Plan (Final)
**A Multi-Agent Framework for Stakeholder-Aware Recommendation Governance**

> 본 문서는 MARGO 프레임워크의 전체 구현 로드맵이다. 알고리즘과 웹 데모를 병행 개발하며, 각 단계별 산출물·기술 스택·결정 사항을 명시한다.

---

## 0. 핵심 설계 결정 (Locked Decisions)

| 항목 | 결정 | 비고 |
|---|---|---|
| **Trend Agent** | LLM-only + web search 통합 | 도메인 뉴스/인터넷 콘텐츠를 LLM이 해석 |
| **User/Item Profile** | LLM 기반 새로 생성 | AgentCF 방식 참고, directive-aware 확장 |
| **Candidate Retriever** | **BGE-M3 (dense retriever)** | Pretrained, directive-aware query 가능 |
| **개발 방식** | 알고리즘 + 웹 병행 | UI 피드백이 알고리즘 설계에 반영 |
| **Primary Dataset** | Amazon Reviews 2023 - Clothing, Shoes & Jewelry | Fashion 도메인 |
| **LLM Backbone** | Qwen2.5-7B (default) + GPT-4o (보조) | vLLM 로컬 서빙 |
| **Orchestration** | LangGraph | State machine으로 validation loop 표현 |

### Retriever 선택 근거 (BGE-M3)

- **Pretrained → 학습 불필요**: `BAAI/bge-m3` HuggingFace 모델 plug-and-play
- **Directive-aware retrieval 자연스러움**: `query = user_profile + directive_NL`을 의미 검색으로 던질 수 있음 → 후보 pool 자체가 directive에 따라 변화 (데모 임팩트 ↑)
- **MARGO 정신과 일치**: Item Agent의 자연어 description과 호환되는 의미 기반 검색
- **Multilingual**: 한국어 directive + 영어 item 동시 처리
- **Baseline 차별화**: BM25/LightGCN을 별도 baseline으로 두면 "modern retriever + MARGO reranker" 조합의 우월성 검증 가능

---

## 1. 프로젝트 구조

```
MARGO/
├── data/
│   └── Amazon Fashion/
│       ├── Clothing_Shoes_and_Jewelry.jsonl
│       ├── meta_Clothing_Shoes_and_Jewelry.jsonl
│       └── processed/                    # 전처리 결과
│           ├── train.parquet
│           ├── valid.parquet
│           ├── test.parquet
│           ├── items.parquet
│           ├── vocab_fashion.json
│           ├── item_embeddings.npy       # BGE-M3 결과
│           └── faiss_index.bin
├── src/
│   └── sage/
│       ├── __init__.py
│       ├── agents/                       # 4종 stakeholder agent
│       │   ├── base.py
│       │   ├── user_agent.py
│       │   ├── item_agent.py
│       │   ├── expert_agent.py
│       │   └── trend_agent.py
│       ├── protocol/                     # Message schema, routing
│       │   ├── messages.py
│       │   └── router.py
│       ├── lifecycle/                    # 4-phase orchestration
│       │   ├── phase1_initialization.py
│       │   ├── phase2_directive.py
│       │   ├── phase3_reasoning.py
│       │   └── phase4_validation.py
│       ├── grounding/                    # Hallucination 통제
│       │   ├── schema_validator.py
│       │   ├── vocabulary.py
│       │   └── snapshot.py
│       ├── retrieval/                    # Candidate retriever
│       │   ├── bge_retriever.py          # ⭐ Primary
│       │   ├── bm25_retriever.py         # Baseline
│       │   └── lightgcn_retriever.py     # Baseline
│       ├── llm/                          # LLM wrapper
│       │   ├── client.py                 # Qwen / GPT-4o unified interface
│       │   └── prompts/                  # System prompts per agent
│       ├── trend_sources/                # Trend Agent의 web search
│       │   └── web_search.py
│       ├── evaluation/                   # Metrics
│       │   ├── standard.py               # NDCG, HR
│       │   ├── governance.py             # DCR, TAS
│       │   └── grounding.py              # IHR, VDR, CADR, SVR
│       ├── domains/
│       │   └── fashion/
│       │       ├── personas.py           # MD persona
│       │       ├── vocabulary.py         # 도메인 vocab 자동 추출
│       │       └── loader.py             # Amazon Fashion loader
│       └── baselines/
│           ├── lightgcn.py
│           ├── agentcf.py
│           └── macf.py
├── scripts/
│   ├── preprocess.py                     # 5-core, leave-one-out split
│   ├── build_index.py                    # BGE-M3 embedding + FAISS
│   ├── train_lightgcn.py                 # baseline용
│   └── evaluate.py
├── web/                                  # 웹 데모 (병행 개발)
│   ├── backend/                          # FastAPI
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── consumer.py               # 일반 사용자 API
│   │   │   ├── expert.py                 # MD 콘솔 API
│   │   │   └── trace.py                  # Agent trace WebSocket
│   │   └── services/
│   │       └── margo_runner.py            # MARGO 호출 wrapper
│   └── frontend/                         # Next.js
│       ├── app/
│       │   ├── (consumer)/               # Layer A: 사용자 화면
│       │   │   ├── page.tsx
│       │   │   └── product/[id]/page.tsx
│       │   └── (expert)/                 # Layer B: MD 콘솔
│       │       ├── console/page.tsx
│       │       └── trace/page.tsx
│       └── components/
│           ├── product-grid.tsx
│           ├── rationale-card.tsx        # 3-layer rationale
│           ├── directive-editor.tsx
│           └── agent-trace-viewer.tsx
├── notebooks/                            # EDA, prototype
├── tests/
├── requirements.txt
└── README.md
```

---

## 2. Step-by-Step 구현 가이드

### **Step 1. 데이터 전처리 (M1, Week 1-2)**

#### 1.1 Raw 데이터 inspection
```bash
cd data/Amazon\ Fashion/
head -n 1 Clothing_Shoes_and_Jewelry.jsonl | python3 -m json.tool
head -n 1 meta_Clothing_Shoes_and_Jewelry.jsonl | python3 -m json.tool
wc -l *.jsonl  # row count
```

#### 1.2 전처리 파이프라인 (`scripts/preprocess.py`)
- [ ] JSONL → Pandas DataFrame 로딩 (chunked, 메모리 관리)
- [ ] 5-core filtering: user·item 모두 ≥5 interactions
- [ ] rating ≥ 4.0을 positive로 변환 (implicit feedback)
- [ ] Leave-one-out split: user별 가장 최근 interaction을 test로, 그 직전을 valid로
- [ ] Metadata join: `parent_asin` 기준
- [ ] 처리 결과 저장: `data/Amazon Fashion/processed/{train,valid,test}.parquet`, `items.parquet`

#### 1.3 도메인 Vocabulary 추출 (`src/margo/domains/fashion/vocabulary.py`)
- [ ] `categories` 컬럼에서 계층 unique values 추출
- [ ] `details.Color`, `details.Material` 등 attribute별 vocab 구성
- [ ] Pantone 색상 표준 추가 (옵션)
- [ ] JSON으로 저장: `data/Amazon Fashion/processed/vocab_fashion.json`

**산출물**: 전처리된 train/valid/test 데이터셋, 도메인 vocab

---

### **Step 2. BGE-M3 Retriever 구축 (M1, Week 2-3)** ⭐

#### 2.1 Item Text Corpus 생성
각 item에 대해 BGE-M3 입력용 text를 구성:
```
item_text = f"{title}. {description}. Category: {categories}. Color: {color}. Material: {material}. Price: ${price}."
```

#### 2.2 Item Embedding 구축 (`scripts/build_index.py`)
- [ ] `BAAI/bge-m3` 모델 로딩 (sentence-transformers 또는 FlagEmbedding)
- [ ] 모든 item text → embedding (배치 처리, GPU)
- [ ] FAISS index 구축 (IndexFlatIP 또는 IndexHNSWFlat)
- [ ] 저장: `processed/item_embeddings.npy`, `processed/faiss_index.bin`

```python
# 핵심 코드 패턴
from FlagEmbedding import BGEM3FlagModel
import faiss
import numpy as np

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
embeddings = model.encode(item_texts, batch_size=64)['dense_vecs']

index = faiss.IndexFlatIP(embeddings.shape[1])
faiss.normalize_L2(embeddings)
index.add(embeddings)
faiss.write_index(index, 'faiss_index.bin')
```

#### 2.3 BGE Retriever Wrapper (`src/margo/retrieval/bge_retriever.py`)
```python
class BGERetriever:
    def __init__(self, index_path, item_ids):
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.index = faiss.read_index(index_path)
        self.item_ids = item_ids
    
    def retrieve(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Query → top-k item ID + score 반환"""
        query_emb = self.model.encode([query])['dense_vecs']
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, k)
        return [(self.item_ids[i], s) for i, s in zip(indices[0], scores[0])]
    
    def retrieve_with_directive(self, user_profile: str, directive_nl: str, k: int = 100):
        """Directive-aware retrieval: user profile + directive를 query로 합침"""
        query = f"User: {user_profile}\nIntent: {directive_nl}"
        return self.retrieve(query, k)
```

**산출물**: BGE-M3 인덱스, retriever wrapper. directive 입력 시 후보 pool 자체가 변화하는 동작 확인

---

### **Step 3. Baselines (병행, M1-M2)**

#### 3.1 BM25 Baseline (`src/margo/retrieval/bm25_retriever.py`)
- [ ] `rank-bm25` 라이브러리 사용
- [ ] item text corpus로 BM25 index 구축

#### 3.2 LightGCN Baseline (`scripts/train_lightgcn.py`)
- [ ] RecBole 또는 PyTorch 직접 구현
- [ ] BPR loss로 학습
- [ ] NDCG@10, HR@10 측정

#### 3.3 AgentCF, MACF 재현 (`src/margo/baselines/`)
- [ ] AgentCF: https://github.com/RUCAIBox/AgentCF 코드 참고
- [ ] MACF: arXiv:2511.18413 — 공개 코드 확인 후 재현

**산출물**: Baseline 수치 (NDCG/HR) 확보

---

### **Step 4. LLM Infrastructure (M2, Week 1)**

#### 4.1 LLM Client (`src/margo/llm/client.py`)
- [ ] vLLM으로 Qwen2.5-7B 로컬 서빙 (OpenAI-compatible API)
- [ ] OpenAI GPT-4o 클라이언트
- [ ] 통합 인터페이스: `LLMClient.complete(prompt, schema=PydanticModel)`
- [ ] Structured output 강제 (Pydantic JSON schema)
- [ ] Token cost logging (efficiency metric용)

#### 4.2 Prompt 관리 (`src/margo/llm/prompts/`)
- [ ] 각 agent별 system prompt를 별도 파일로 분리
- [ ] Jinja2 템플릿 사용 (변수 주입)
- [ ] Version control 가능하게 구성

**산출물**: 통합 LLM client, prompt 라이브러리

---

### **Step 5. Message Protocol & Schema (M2, Week 1)**

#### 5.1 Message 정의 (`src/margo/protocol/messages.py`)
```python
from pydantic import BaseModel
from enum import Enum

class MessageType(str, Enum):
    DIRECTIVE = "directive"
    BROADCAST = "broadcast"
    NEGOTIATION = "negotiation"
    CONSULTATION = "consultation"
    COORDINATION = "coordination"

class Message(BaseModel):
    type: MessageType
    sender: str
    receivers: list[str]
    payload: dict  # agent-specific schema
    timestamp: float
```

#### 5.2 Schema Validator (`src/margo/grounding/schema_validator.py`)
- [ ] 모든 agent output을 Pydantic으로 강제
- [ ] Schema violation 시 retry 또는 logging (SVR 측정)

**산출물**: 메시지 protocol, Schema validator

---

### **Step 6. User Agent & Item Agent (M2, Week 2-3)**

#### 6.1 Base Agent (`src/margo/agents/base.py`)
```python
class BaseAgent:
    def __init__(self, agent_id, llm_client, system_prompt):
        self.id = agent_id
        self.llm = llm_client
        self.prompt = system_prompt
        self.memory = []
    
    def receive(self, message: Message): ...
    def act(self) -> Message: ...
```

#### 6.2 User Agent (`src/margo/agents/user_agent.py`)
- [ ] State: id, history (item list), profile (NL), memory (reflection log)
- [ ] **Skills**:
  - `query_preference(context)` → 자연어 선호 표명
  - `evaluate_candidate(item_desc, directive, trend)` → 후보 점수 + 3-layer rationale
  - `update_profile(interaction)` → reflection 업데이트
- [ ] Profile 초기화: history를 LLM에 주고 자연어 profile 생성

#### 6.3 Item Agent (`src/margo/agents/item_agent.py`)
- [ ] State: id, attributes (structured), description (text), audience_log
- [ ] **Skills**:
  - `self_describe(user_context, directive)` → context-aware 자연어 자기소개
  - `claim_audience_fit(user_profile, directive)` → user fit 주장
  - `update_reflection(interaction)` → 패턴 학습
- [ ] 초기화: metadata로 description 생성

**검증**: BGE Retriever + User/Item Agent만으로 simple recommendation pipeline 동작 확인

---

### **Step 7. Expert Agent (M3, Week 1-2)**

#### 7.1 Expert Agent (`src/margo/agents/expert_agent.py`)
- [ ] State: role (persona), current_directive, policy_history
- [ ] **Skills**:
  - `issue_directive(goal, constraint, boost)` → structured + NL hybrid directive 발행
  - `validate_recommendation(rec_list)` → directive 만족 여부 검증 (Pass/Fail + 이유)
  - `refine_directive(outcome)` → fail 시 directive 수정
- [ ] Persona: 패션 MD ("10년차 머천다이저, 캐주얼-포멀 transition 전문")

#### 7.2 Directive Schema
```python
class Directive(BaseModel):
    goal: str                    # NL: "캐주얼→포멀 upsell"
    structured_constraints: dict # {"price_diff_pct": 30, "boost_category": "trench"}
    natural_language: str        # 자유 NL 표현
    issued_at: float
```

#### 7.3 Validation Loop (`src/margo/lifecycle/phase4_validation.py`)
- [ ] **무한 루프 방지**: max_iterations = 3
- [ ] **Convergence criterion**: directive 만족도 점수 ≥ 0.85 시 종료
- [ ] 각 iteration 로그 저장

**산출물**: Expert governance loop 동작

---

### **Step 8. Trend Agent (M3, Week 3-4)**

#### 8.1 Web Search Integration (`src/margo/trend_sources/web_search.py`)
- [ ] 검색 도구: Tavily API, SerpAPI, 또는 자체 Bing/Google 호출
- [ ] Query 생성: domain + time_window + persona → "2026 spring fashion trends casual"
- [ ] 결과 fetching: 상위 5-10개 페이지 텍스트 추출
- [ ] 도메인별 source whitelist (vogue, businessoffashion, wgsn 등)

#### 8.2 Trend Agent (`src/margo/agents/trend_agent.py`)
- [ ] State: role (trend analyst), domain_context, time_context, interpretation_log
- [ ] **Skills**:
  - `query_trend(context_query)` → web search 실행
  - `interpret_for_recommendation(raw_data, user, directive)` → 추천 맥락으로 해석
  - `broadcast(interpretation)` → 다른 모든 agent에 주입
- [ ] LLM-only 해석: web search 결과를 LLM이 요약·맥락화

#### 8.3 Output Snapshot (`src/margo/grounding/snapshot.py`)
- [ ] (domain, time_window) 단위 캐싱 → 재현성 보장
- [ ] 캐시 키: `f"{domain}_{YYYY-MM}.json"`
- [ ] 캐시 hit 시 web search 생략, miss 시에만 호출
- [ ] Hallucination 발견 시 cached version 수동 수정 가능

#### 8.4 Vocabulary Grounding for Trend Output
- [ ] Trend Agent output을 fashion vocab과 대조
- [ ] vocab 미매칭 키워드 logging (VDR 측정)

**산출물**: Full 4-agent MARGO

---

### **Step 9. Full Lifecycle Orchestration (M4, Week 1)**

#### 9.1 LangGraph로 Phase 연결 (`src/margo/lifecycle/`)
```
Phase 1 (Init) → Phase 2 (Directive + Trend) → Phase 3 (Reasoning) → Phase 4 (Validation)
                                                                      ↓ (Fail)
                                                              Refined Directive → Phase 2
```

- [ ] Phase 1 (`phase1_initialization.py`): User/Item Agent profile 생성 (offline batch)
- [ ] Phase 2 (`phase2_directive.py`): Expert directive + Trend broadcast
- [ ] Phase 3 (`phase3_reasoning.py`):
  - **BGE Retriever로 directive-aware top-100 추출**
  - 각 Item Agent: self_describe (directive + trend 인지)
  - User Agent: evaluate_candidate → ranked top-K + 3-layer rationale
- [ ] Phase 4 (`phase4_validation.py`): Expert validate, refine loop

#### 9.2 Recommend API (`src/margo/api.py`)
```python
def recommend(user_id, directive, k=10) -> RecommendationResult:
    """End-to-end recommendation"""
```

---

### **Step 10. Evaluation (M4, Week 2-3)**

#### 10.1 Standard Metrics (`src/margo/evaluation/standard.py`)
- [ ] NDCG@10, HR@10 (leave-one-out 기반)

#### 10.2 Governance Metrics (`src/margo/evaluation/governance.py`)
- [ ] **DCR (Directive Compliance Rate)**: Top-K 중 directive constraint 만족 비율
  - 정량 constraint (price_diff 등) → 자동 측정
  - 정성 constraint (style fit 등) → LLM judge
- [ ] **TAS (Trend Alignment Score)**: Top-K가 trend signal과 일치 정도
  - Trend keyword와 item attribute 매칭 비율
  - 또는 LLM-based semantic similarity

#### 10.3 Grounding Metrics (`src/margo/evaluation/grounding.py`)
- [ ] **IHR (Item Hallucination Rate)**: 추천에 catalog 없는 ID 등장 비율
- [ ] **VDR (Vocabulary Drift Rate)**: Trend output 중 vocab 미매칭 비율
- [ ] **CADR (Cross-Agent Disagreement Rate)**: Expert가 Trend agent challenge 빈도
- [ ] **SVR (Schema Violation Rate)**: Agent output schema 위반 비율

#### 10.4 Ablation 실험
- [ ] (A) Expert Agent 제거 → DCR 변화
- [ ] (B) Trend Agent 제거 → TAS 변화
- [ ] (C) Expert + Trend 제거 → AgentCF 회귀
- [ ] (D) Grounding Layer 제거 → IHR/VDR/CADR/SVR 변화
- [ ] (E) Directive 형식: structured / NL / hybrid 비교
- [ ] **(F) Retriever 교체: BGE-M3 / BM25 / LightGCN** → modular 검증

**산출물**: 정량 평가 결과, ablation table

---

## 3. 웹 데모 구현 가이드

### **설계 철학**

웹은 **2개의 별도 화면**으로 구성:
- **Layer A (Consumer View)**: 실제 패션 커머스처럼 보이는 일반 사용자 화면
- **Layer B (Expert Console)**: MD가 directive 입력하고 결과 변화를 실시간으로 보는 콘솔

학회 데모 시 **split-screen**으로 두 화면을 동시에 보여줘서, "MD가 자연어로 정책을 주입하니 사용자 추천이 어떻게 바뀌는가"를 시연한다.

**BGE-M3 덕분에 가능한 시연**: directive 변경 → **후보 pool 자체가 변하는 모습** + reranking 변화를 동시에 보여줄 수 있음.

### **기술 스택**

| 영역 | 기술 |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js 14 (App Router) + TypeScript |
| Styling | Tailwind CSS + shadcn/ui |
| 실시간 통신 | WebSocket (agent trace streaming) |
| 시각화 | React Flow (agent message graph), Framer Motion (ranking change animation) |
| 상태관리 | Zustand |
| Backend ↔ MARGO | Python module import (FastAPI에서 직접 호출) |

### **디자인 방향**

- **Aesthetic**: Editorial / Magazine 스타일 (패션 도메인에 맞게)
- **Typography**: Display는 serif (e.g., Fraunces), body는 modern sans (e.g., Geist)
- **Color**: Off-white background + deep navy + 어스 톤 accent (terracotta or sage)
- **절대 피할 것**: 흔한 SaaS 대시보드 느낌 (purple gradient, generic cards)

---

### **Step W1. Backend API 설계 (M2-M3 병행)**

#### W1.1 FastAPI 구조 (`web/backend/main.py`)
```python
# 주요 엔드포인트
POST /api/consumer/recommend
  - body: {user_id, k}
  - 현재 활성 directive로 추천 → ranked list + 3-layer rationale 반환

POST /api/expert/directive
  - body: {goal, constraints, natural_language}
  - directive 발행, 전역 상태 업데이트

GET /api/expert/users
  - 모니터링 가능한 sample user list (5-10명)

POST /api/expert/retrieve_preview
  - body: {user_id, directive}
  - BGE-M3로 후보 pool만 미리 보여주는 엔드포인트 (reranking 전)

WS /ws/trace
  - 추천 요청 시 agent 간 메시지를 실시간 stream
```

#### W1.2 MARGO Runner (`web/backend/services/margo_runner.py`)
- [ ] MARGO pipeline을 호출하면서 매 step의 메시지를 WebSocket으로 emit
- [ ] Async 처리 (LLM call이 길 수 있으니)
- [ ] 캐싱: 동일 (user, directive) → 결과 캐시

---

### **Step W2. Consumer View (M4-M5)**

#### W2.1 메인 페이지 (`web/frontend/app/(consumer)/page.tsx`)
- [ ] 상품 그리드 (Amazon Fashion 이미지 + 가격 + 카테고리)
- [ ] 상단에 "For You" 섹션 → MARGO 추천 Top-K
- [ ] **3-layer rationale UI**: 각 추천 카드 hover 시 펼쳐지는 패널
  ```
  [Item Card]
    ▾ Why this for you?
      🧑 Personal: 평소 가격대(₩50-100k)에 적합하며...
      📋 From operator: 시즌 캠페인 'casual→formal' 의도와 부합
      📈 Trend: 2026 SS 미니멀 트렌치 트렌드 강세
  ```

#### W2.2 사용자 프로필 선택
- [ ] 데모 환경에서는 sample user 5-10명 중 선택 가능
- [ ] 선택 시 history, 평균 가격대 등 표시

---

### **Step W3. Expert Console (M5)**

#### W3.1 Directive Editor (`web/frontend/app/(expert)/console/page.tsx`)
- [ ] **3-panel 레이아웃**:
  - 왼쪽: Directive 입력 (NL textarea + structured constraint 폼)
  - 가운데: Active directive 표시 + history
  - 오른쪽: 선택한 사용자의 추천 결과 (Consumer View와 동기화)

#### W3.2 Real-time Ranking Animation
- [ ] Directive 변경 시 추천 순위가 Framer Motion으로 부드럽게 재정렬
- [ ] 변동된 아이템에 highlight (어떤 게 올라오고 내려갔는지)
- [ ] **후보 pool 변화 시각화**: BGE retrieval 결과가 directive에 따라 어떻게 바뀌는지 별도 패널

#### W3.3 Constraint 시각화
- [ ] price_diff, boost category 등 constraint를 visual chip으로
- [ ] 위반 시 빨간 표시 (validation 결과 반영)

---

### **Step W4. Agent Trace Viewer (M5)**

가장 임팩트 있는 데모 요소. MARGO 내부 동작을 투명하게 시각화.

#### W4.1 Trace 패널 (`web/frontend/components/agent-trace-viewer.tsx`)
- [ ] 추천 요청 발생 시 React Flow로 4-agent 노드 + Retriever 노드 표시
- [ ] WebSocket으로 streaming되는 메시지를 노드 사이에 흐르는 형태로 시각화:
  ```
  [Expert] ── directive ──▶ [User] [Item] [Trend] [Retriever]
                          
  [Trend]  ── broadcast ──▶ [User] [Item]
  
  [Retriever] ── candidate pool ──▶ [Item Agents (100)]
  
  [Item]   ── self_describe ──▶ [User]
  
  [User]   ── ranked output ──▶ [Result]
  
  [Expert] ── validate ──▶ [Result]
  ```
- [ ] 각 메시지 클릭 시 실제 payload (NL 텍스트) 표시
- [ ] Phase별로 timeline 구분

#### W4.2 Validation Loop 시각화
- [ ] Refine iteration이 발생하면 loop가 시각적으로 표현
- [ ] 각 iteration의 directive 변화를 diff로 표시

---

### **Step W5. Polish & Demo Mode (M6)**

- [ ] Landing page: MARGO 소개 (논문 abstract 수준의 1-pager)
- [ ] **Demo scenarios**: 미리 준비된 3개 시나리오 (campaign, seasonal, clearance)
  - 클릭 한 번으로 directive + user + expected outcome 자동 세팅
- [ ] **Comparison mode**: MARGO vs LightGCN vs AgentCF의 추천 결과 나란히 표시
- [ ] 반응형 디자인 (학회 발표 시 다양한 화면 크기)
- [ ] Loading state (LLM call 중 agent thinking 애니메이션)

---

## 4. 6개월 타임라인

| Month | 알고리즘 | 웹 데모 | Deliverable |
|---|---|---|---|
| M1 | Step 1-3 (전처리, BGE-M3 인덱스, baselines) | 디자인 sketch, tech setup | Baseline 수치, BGE retriever 동작 |
| M2 | Step 4-6 (LLM, User/Item Agent) | Step W1 (Backend API) | 2-agent prototype |
| M3 | Step 7-8 (Expert, Trend Agent) | Step W1 완성 | Full 4-agent on Fashion |
| M4 | Step 9-10 (Lifecycle, Evaluation) | Step W2 (Consumer View) | 1차 실험 결과, 기본 데모 |
| M5 | Ablation, Multi-domain (선택) | Step W3-W4 (Expert + Trace) | 모든 ablation, full demo |
| M6 | 논문 draft | Step W5 (Polish) | Submission-ready paper + 데모 영상 |

---

## 5. 즉시 시작 체크리스트

### 환경 셋업
- [ ] `requirements.txt` 작성
  - 핵심: `torch`, `transformers`, `vllm`, `langgraph`, `pydantic`, `pandas`
  - Retriever: `FlagEmbedding` (BGE-M3), `faiss-cpu` 또는 `faiss-gpu`, `rank-bm25`
  - Web: `fastapi`, `uvicorn`, `websockets`
- [ ] Python 3.10+ venv 또는 conda env
- [ ] GPU 환경 확인 (Qwen2.5-7B 서빙용, 최소 24GB VRAM)
- [ ] OpenAI API key (보조 모델용)
- [ ] Web search API key (Tavily 추천)

### 첫 주에 끝낼 것
1. 데이터 압축 풀고 첫 row inspection (이미 진행 중)
2. `preprocess.py` 작성 + 5-core 필터링
3. `vocabulary.py`로 fashion vocab 추출

### 둘째 주
1. `build_index.py`: BGE-M3로 모든 item embedding + FAISS index
2. `BGERetriever` wrapper 구현
3. Sanity check: sample user의 history + dummy directive로 후보 100개 뽑히는지 확인
4. BM25 baseline retriever도 같이 구현 (비교용)

### 셋째 주
1. LightGCN baseline 학습 (RecBole로 빠르게)
2. NDCG/HR 측정 → MARGO 비교 기준선 확보

### 넷째 주
1. vLLM으로 Qwen2.5-7B 로컬 서빙
2. `LLMClient` 클래스
3. Pydantic message schema
4. User Agent profile generation 시범 구현 (10명 user로 sanity check)

---

## 6. 위험 요소 모니터링

| Risk | Mitigation | 체크 시점 |
|---|---|---|
| BGE-M3가 패션 도메인에 약할 가능성 | Fashion-CLIP 등 도메인 특화 모델로 ablation, item title+description 충분히 풍부하게 구성 | Step 2 후 |
| Trend Agent web search 결과가 noisy | 도메인별 source whitelist (vogue, businessoffashion 등) | Step 8 |
| LLM cost 폭증 | 모든 LLM call 로깅, batch 처리, snapshot 캐싱 적극 활용 | 상시 |
| User/Item Agent profile이 generic해서 차별점 약화 | Profile 생성 시 directive-aware 변형 추가 | Step 6 후 |
| Validation loop가 NDCG 떨어뜨림 | DCR vs NDCG trade-off curve를 명시적으로 보고 | Step 10 |
| 웹과 알고리즘 동기화 어긋남 | API contract를 OpenAPI spec으로 먼저 정의 | Step W1 시작 시 |
| 학회 deadline 임박 시 Music 도메인 미완성 | 제안서대로 fallback 처리 (Fashion + News만으로 generality 주장) | M5 |

---

## 7. 참고 자료

- **BGE-M3**: https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_M3 (논문: arXiv:2402.03216)
- **FAISS**: https://github.com/facebookresearch/faiss
- **AgentCF 코드**: https://github.com/RUCAIBox/AgentCF (User/Item Agent 구현 참고)
- **LangGraph 튜토리얼**: https://langchain-ai.github.io/langgraph/
- **vLLM 서빙**: https://docs.vllm.ai/
- **shadcn/ui**: https://ui.shadcn.com/ (Frontend 컴포넌트)
- **React Flow**: https://reactflow.dev/ (Agent trace 시각화)
- **AgenticShop 데모** (WWW 2026): https://github.com/happysnail06/AgenticShop — split-screen 데모 레퍼런스
- **RecBole**: https://recbole.io/ (LightGCN baseline 빠르게)
- **Amazon Reviews 2023**: https://amazon-reviews-2023.github.io/

---

*이 문서는 living document. 구현 진행하면서 결정 변경/추가 사항 발생 시 업데이트.*
