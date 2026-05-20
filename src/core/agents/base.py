"""Shared base class for every MARGO agent.

Each concrete agent (User / Item / Expert / Trend) extends this class with
its own *Skills* (the methods listed in Section 2 of the paper). The base
class only handles:

* LLM client wiring,
* prompt rendering,
* message bus inbox handling,
* structured-output validation + SVR accounting.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from core.validation.schema_validator import SchemaValidator
from adapters.llm import LLMClient, PromptRegistry, get_default_client
from core.protocol.messages import Message
from core.protocol.router import MessageBus

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseAgent:
    """Common skeleton — Phase 1 of the lifecycle hangs profile init here."""

    def __init__(
        self,
        agent_id: str,
        *,
        prompt_namespace: str,
        llm: Optional[LLMClient] = None,
        prompts: Optional[PromptRegistry] = None,
        bus: Optional[MessageBus] = None,
        schema_validator: Optional[SchemaValidator] = None,
    ) -> None:
        self.id = agent_id
        self.namespace = prompt_namespace
        self.llm = llm or get_default_client()
        self.prompts = prompts or PromptRegistry()
        self.bus = bus
        self.schema = schema_validator or SchemaValidator()
        self.inbox: list[Message] = []
        if self.bus is not None:
            self.bus.subscribe(self.id, self.inbox.append)

    # ------------------------------------------------------------------ #
    # Prompt helpers                                                      #
    # ------------------------------------------------------------------ #

    def render(self, template: str, **vars: Any) -> str:
        return self.prompts.render(f"{self.namespace}.{template}", **vars)

    def _system(self, **vars: Any) -> str:
        return self.render("system", **vars)

    # ------------------------------------------------------------------ #
    # Structured LLM call                                                  #
    # ------------------------------------------------------------------ #

    def _ask_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system: str,
        max_repairs: int = 1,
        temperature: Optional[float] = None,
    ) -> T:
        """Wrap ``LLMClient.complete_structured`` with SVR accounting."""
        try:
            obj = self.llm.complete_structured(
                prompt,
                schema=schema,
                system=system,
                agent_id=self.id,
                max_repairs=max_repairs,
                temperature=temperature,
            )
            # Successful parse — count it for SVR denominator.
            self.schema.counter.total += 1
            return obj
        except ValidationError as e:
            self.schema.counter.total += 1
            self.schema.counter.violations += 1
            self.schema.counter.by_agent[self.id] = (
                self.schema.counter.by_agent.get(self.id, 0) + 1
            )
            log.error("schema violation by agent %s: %s", self.id, e)
            raise

    # ------------------------------------------------------------------ #
    # Bus helpers                                                          #
    # ------------------------------------------------------------------ #

    def emit(self, msg: Message) -> None:
        if self.bus is not None:
            self.bus.publish(msg)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id}>"
