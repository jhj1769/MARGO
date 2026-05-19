"""Expert-console endpoints (Layer B)."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


class DirectiveBody(BaseModel):
    goal: str
    natural_language: str
    structured_constraints: dict[str, Any] = Field(default_factory=dict)


@router.get("/active-directive")
async def get_active(request: Request):
    return request.app.state.runner.get_active_directive() or {}


@router.post("/directive")
async def set_directive(request: Request, body: DirectiveBody):
    saved = request.app.state.runner.set_active_directive(body.model_dump())
    return saved


@router.get("/history")
async def history(request: Request):
    return {"history": list(reversed(request.app.state.runner.directive_history))[:20]}
