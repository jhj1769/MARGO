# MARGO — Web Demo

Editorial-style split-screen demo for the MARGO framework.
Inspired by [DLI Lab demos](https://dli.yonsei.ac.kr/demo).

```
web/
├── backend/   FastAPI · serves /api + /ws/trace · mock fallback when LLM is offline
└── frontend/  Next.js 14 + Tailwind · 3 pages (landing / architecture / demo)
```

The frontend renders even without the backend (marketing pages); the live
demo (`/demo`) calls the FastAPI endpoints. The backend itself starts in
**mock mode** automatically when neither `MARGO_PROCESSED_DIR` nor an LLM
endpoint is reachable — so the demo can be presented from a laptop.

---

## 1. Quick start (one-shot, two terminals)

```bash
# Terminal A — backend (mock mode, no GPU needed)
cd web/backend
pip install -r requirements.txt
uvicorn web.backend.main:app --reload --port 8001 --app-dir ../..

# Terminal B — frontend
cd web/frontend
npm install
npm run dev
# → http://localhost:3000
```

The frontend's `next.config.mjs` proxies `/api/margo/*` → `http://localhost:8001/api/*`
and `/ws/margo/*` → `ws://localhost:8001/ws/*`. Override with `MARGO_API_URL`.

---

## 2. Wiring the real MARGO engine

Once you have run `scripts/preprocess.py` + `scripts/build_index.py` from
the repo root, point the backend at the processed directory:

```bash
export MARGO_PROCESSED_DIR="$(pwd)/../data/Amazon Fashion/processed"
export MARGO_LLM_BACKEND=vllm                # or `openai`
export MARGO_VLLM_BASE_URL=http://localhost:8000/v1
uvicorn web.backend.main:app --port 8001 --app-dir ../..
```

The backend automatically switches to **engine mode** — the `/api/health`
response will report `{"mode": "engine"}` and the demo's status bar lights
up green.

To explicitly stay in mock mode, set `MARGO_DEMO_MODE=mock`.

---

## 3. Pages

| Path             | What it does                                                                 |
|------------------|------------------------------------------------------------------------------|
| `/`              | Landing — hero, 4-agent ring, 3-layer rationale preview, demo cards          |
| `/architecture`  | Framework figure with 4 phases + grounding metric table                      |
| `/demo`          | **Three-column workbench** — Expert Console · Consumer View · Agent Trace    |

The `/demo` workbench streams agent messages over WebSocket (`/ws/margo/trace`)
and renders them as a typed timeline (phase color, sender → receivers,
NL summary). The consumer view re-animates rank changes with Framer Motion.

---

## 4. API surface

```
GET    /api/health
GET    /api/consumer/users
POST   /api/consumer/recommend            { user_id, directive?, k? }
GET    /api/expert/active-directive
POST   /api/expert/directive              Directive payload
GET    /api/scenarios/
WS     /ws/trace                          send {action:"recommend", user_id, directive, k}
```

All endpoints accept JSON. The mock backend ships with four sample users
(Min-ji, Ji-woo, Yuna, Tae-ho), sixteen catalogue items, three rising
trends, and three preset scenarios.

---

## 5. Design system

- Palette: `paper` (#FAF8F4) · `ink` (#0F1B2D) · `terracotta` (#C4664E) · `sage` · `gold`
- Display: Fraunces (variable serif) · Body: Inter · Mono: JetBrains Mono
- Motion: Framer Motion (Top-K re-rank, trace timeline, hero floating cards)
- No CSS framework beyond Tailwind v3 + a small set of component classes in
  `globals.css` (.card, .surface, .chip, .btn-*)

---

## 6. Customising the mock catalogue

Edit `web/backend/data/mock.json`. The file's contract is:

```jsonc
{
  "users":     [{ "user_id", "name", "tagline", "avg_price", "affinity_tokens", … }],
  "catalog":   [{ "item_id", "title", "price", "category", "image_url", … }],
  "trends":    [{ "summary", "keywords", "rising_attributes", "match_cues" }],
  "scenarios": [{ "id", "title", "blurb", "default_user", "directive" }]
}
```

The mock ranker honours every structured constraint the real MARGO Expert
Agent checks (price gap, boost/forbid category, price band), so you can
demonstrate directive-driven ranking changes without touching the LLM.
