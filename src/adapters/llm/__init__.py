"""Unified LLM client + Jinja-based prompt registry."""

from adapters.llm.client import LLMClient, LLMResponse, get_default_client
from adapters.llm.prompts import PromptRegistry

__all__ = ["LLMClient", "LLMResponse", "get_default_client", "PromptRegistry"]
