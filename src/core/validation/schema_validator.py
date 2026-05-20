"""Pydantic-backed schema validator with SVR accounting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class SVRCounter:
    total: int = 0
    violations: int = 0
    by_agent: dict[str, int] = field(default_factory=dict)

    def rate(self) -> float:
        return self.violations / self.total if self.total else 0.0


class SchemaValidator:
    """Wraps every agent's "raw → typed" parse and tracks SVR."""

    def __init__(self) -> None:
        self.counter = SVRCounter()

    def parse(
        self,
        payload: Any,
        schema: Type[T],
        *,
        agent_id: str = "anonymous",
    ) -> T:
        """Validate ``payload`` against ``schema``.

        Failures bubble as :class:`pydantic.ValidationError` so the caller
        can decide whether to retry. SVR statistics are updated either way.
        """
        self.counter.total += 1
        try:
            return schema.model_validate(payload)
        except ValidationError:
            self.counter.violations += 1
            self.counter.by_agent[agent_id] = self.counter.by_agent.get(agent_id, 0) + 1
            log.warning("Schema violation: agent=%s schema=%s", agent_id, schema.__name__)
            raise

    def reset(self) -> None:
        self.counter = SVRCounter()
