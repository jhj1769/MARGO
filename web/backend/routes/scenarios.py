"""Preset scenarios that one-click populate the demo."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("", include_in_schema=True)
@router.get("/")
async def list_scenarios(request: Request):
    return {"scenarios": request.app.state.runner.list_scenarios()}
