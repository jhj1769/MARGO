"""Consumer-facing endpoints (Layer A in the demo)."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


class RecommendRequest(BaseModel):
    user_id: str
    k: int = Field(default=6, ge=1, le=20)
    directive: Optional[dict[str, Any]] = None


@router.get("/users")
async def list_users(request: Request):
    return {"users": request.app.state.runner.list_users()}


@router.get("/users/{user_id}/history")
async def user_history(user_id: str, request: Request):
    """Per-user interaction history rendered as item cards.

    Backs the Step 1 expandable panel — image / title / category / price for
    each recent purchase. Note: in mock mode the runner returns a summary-only
    payload (items list is empty).
    """
    return request.app.state.runner.get_user_history(user_id)


@router.post("/recommend")
async def recommend(request: Request, body: RecommendRequest):
    """Run MARGO for one (user, directive) pair and return the Top-K + trace."""
    runner = request.app.state.runner
    return await runner.recommend(user_id=body.user_id, directive=body.directive, k=body.k)
