# MARGO

**MARGO: A Multi-Agent Framework for Stakeholder-Aware Recommendation Governance**

MARGO는 기존 추천 시스템의 *User–Item 2-stakeholder* 구조를
**User · Item · Expert · Trend** 4-stakeholder 구조로 확장하여,
운영자(MD/PM)의 자연어 운영 의도와 외부 트렌드 맥락을 추천 과정에
직접 통합하는 LLM Multi-Agent Framework입니다.

> 자세한 동기·구조·실험 설계는 [`MARGO_Implementation_Plan.md`](./MARGO_Implementation_Plan.md) 참고.  
> 지금까지의 연구 내용 요약은 [`RESEARCH_SUMMARY.md`](./RESEARCH_SUMMARY.md) 참고.

---

## 1. 4-Stakeholder Multi-Agent

| Agent | 역할 | Skill |
|---|---|---|
| **User Agent**   | 사용자 선호 자연어 reasoning             | `query_preference`, `evaluate_candidate`, `update_profile` |
| **Item Agent**   | context-aware 자기서술                    | `self_describe`, `update_reflection` |
| **Expert Agent** | directive 발행·검증·refine (governance)  | `issue_directive`, `validate_recommendation`, `refine_directive` |
| **Trend Agent**  | 외부 맥락 해석·broadcast (LLM + Web)     | `query_trend`, `interpret_trend`, `broadcast` |

## 2. 4-Phase Lifecycle

```
Phase 1 (offline)   : Agent Initialization
Phase 2             : Directive Generation  (Expert + Trend)
Phase 3             : Multi-Agent Reasoning (Retrieval + Item self_describe + User evaluate)
Phase 4             : Validation & Refinement
                       └─ Fail → refined directive → Phase 2
```

각 추천 결과는 **3-Layer Rationale** (Personal / Directive / Trend)을 동반합니다.

## 3. 빠른 시작

```bash
# 0) 가상환경 (프로젝트 루트의 `margo/` — `src/` 소스와 별개)
python3 -m venv margo
source margo/bin/activate
pip install -r requirements.txt
pip install -e .

# 2) 전처리 (Amazon Fashion)
python -m scripts.preprocess --data-dir data/Amazon\ Fashion --out-dir data/Amazon\ Fashion/processed

# 3) BGE-M3 인덱스 빌드
python -m scripts.build_index --processed-dir data/Amazon\ Fashion/processed

# 4) (옵션) vLLM으로 Qwen 서빙
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 5) End-to-end recommendation
python -m scripts.evaluate --processed-dir data/Amazon\ Fashion/processed --k 10
```

`MARGO_LLM_BACKEND` 환경변수로 LLM 백엔드를 전환합니다 (`openai` / `vllm` / `dummy`).

## 4. 디렉토리

```
src/
├── agents/         # base + user/item/expert/trend
├── protocol/       # Pydantic message schema + router
├── lifecycle/      # 4-phase orchestration
├── grounding/      # vocabulary / schema validator / snapshot cache
├── retrieval/      # BGE-M3, BM25, LightGCN
├── llm/            # unified LLM client + jinja prompts
├── trend_sources/  # web search (Tavily / stub)
├── evaluation/     # NDCG/HR + DCR/TAS + IHR/VDR/CADR/SVR
├── domains/        # fashion personas / vocabulary / loader
├── baselines/      # LightGCN, AgentCF, MACF
└── api.py          # public MargoEngine façade
web/
├── backend/        # FastAPI · /api + /ws/trace · mock fallback
└── frontend/       # Next.js 14 + Tailwind · landing / architecture / demo
```

## 5. 웹 데모 실행

```bash
# 한 번에 (mock backend + Next.js dev)
./web/run.sh
# → http://localhost:3000

# 또는 두 터미널로 따로
pip install -r web/backend/requirements.txt
uvicorn web.backend.main:app --port 8001          # 백엔드 (mock 모드)
cd web/frontend && npm install && npm run dev     # 프론트엔드
```

자세한 사용법은 [`web/README.md`](./web/README.md) 참조.

## 6. 라이선스

연구용 (POSTECH AIM Lab).
