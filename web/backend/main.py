"""MARGO Web Demo — FastAPI backend.

Two operating modes, selected at startup:

* **engine**  — Load the real :class:`MargoEngine` against a processed
                Amazon Fashion dataset. Requires ``MARGO_PROCESSED_DIR``.
* **mock**    — In-memory canned data, used for the demo when no GPU/LLM
                is available. Activated automatically when the engine
                fails to initialize OR when ``MARGO_DEMO_MODE=mock``.

Run::

    uvicorn web.backend.main:app --reload --port 8001
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.backend.routes import consumer, expert, scenarios, trace
from web.backend.services.margo_runner import MargoRunner

log = logging.getLogger("margo.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    runner = MargoRunner.from_env()
    app.state.runner = runner
    log.info("MARGO backend up — mode=%s", runner.mode)
    yield
    log.info("MARGO backend shutdown")


app = FastAPI(
    title="MARGO Demo API",
    description=(
        "Multi-agent Recommendation Governance — demo backend. "
        "Falls back to canned data when the live MARGO engine is unavailable."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — wide-open is fine for a demo, locked-down for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("MARGO_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": app.state.runner.mode}


app.include_router(consumer.router, prefix="/api/consumer", tags=["consumer"])
app.include_router(expert.router, prefix="/api/expert", tags=["expert"])
app.include_router(scenarios.router, prefix="/api/scenarios", tags=["scenarios"])
app.include_router(trace.router, prefix="/ws", tags=["trace"])
