"""Backend-agnostic LLM client.

Three back-ends are supported, selected via the ``MARGO_LLM_BACKEND``
environment variable:

* ``openai``  — OpenAI public API.
* ``vllm``    — Any OpenAI-compatible server (vLLM, LM Studio, …).
                ``MARGO_VLLM_BASE_URL`` defaults to ``http://localhost:8000/v1``.
* ``dummy``   — Deterministic offline echo client; used in unit tests so the
                pipeline can be exercised without a GPU.

Every call records token usage to ``LLMClient.usage_log`` so we can
later report cost per RecommendationResult (efficiency metric).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Data containers                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    raw: Any = None


@dataclass
class UsageLog:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    calls_by_agent: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Client                                                                       #
# --------------------------------------------------------------------------- #


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    """Thin wrapper that always returns a string and tracks usage.

    Parameters
    ----------
    backend:
        ``"openai" | "vllm" | "dummy"``. Defaults to ``$MARGO_LLM_BACKEND``
        or ``"dummy"`` if the variable is unset.
    model:
        The model name. Defaults to ``$MARGO_LLM_MODEL`` if set, otherwise
        ``gpt-4o-mini`` for the OpenAI backend and
        ``Qwen/Qwen2.5-7B-Instruct`` for vLLM.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        request_timeout: float = 60.0,
    ) -> None:
        self.backend = (backend or os.getenv("MARGO_LLM_BACKEND") or "dummy").lower()
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.usage_log = UsageLog()
        self._lock = threading.Lock()

        if self.backend == "openai":
            self.model = model or os.getenv("MARGO_LLM_MODEL") or "gpt-4o-mini"
            self._client = self._build_openai(base_url=base_url, api_key=api_key)
        elif self.backend == "vllm":
            self.model = model or os.getenv("MARGO_LLM_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
            self._client = self._build_openai(
                base_url=base_url
                or os.getenv("MARGO_VLLM_BASE_URL")
                or "http://localhost:8000/v1",
                # vLLM does not check the key but openai>=1.0 demands a non-empty string.
                api_key=api_key or os.getenv("MARGO_VLLM_API_KEY") or "EMPTY",
            )
        elif self.backend == "dummy":
            self.model = model or "dummy"
            self._client = None
        else:
            raise LLMClientError(f"Unknown MARGO_LLM_BACKEND={self.backend!r}")

    # ------------------------------------------------------------------ #
    # Construction helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_openai(base_url: Optional[str], api_key: Optional[str]):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise LLMClientError(
                "Install the `openai` package to use openai/vllm back-ends."
            ) from e
        return OpenAI(base_url=base_url, api_key=api_key)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        agent_id: str = "anonymous",
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a plain-text completion."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        if self.backend == "dummy":
            text = self._dummy_completion(prompt, system=system)
            prompt_tokens, completion_tokens = len(prompt.split()), len(text.split())
            raw: Any = None
        else:
            resp = self._chat_with_retry(
                messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            raw = resp
        elapsed = (time.time() - t0) * 1000.0

        with self._lock:
            self.usage_log.calls += 1
            self.usage_log.prompt_tokens += prompt_tokens
            self.usage_log.completion_tokens += completion_tokens
            self.usage_log.calls_by_agent[agent_id] = (
                self.usage_log.calls_by_agent.get(agent_id, 0) + 1
            )

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed,
            model=self.model,
            raw=raw,
        )

    def complete_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system: Optional[str] = None,
        agent_id: str = "anonymous",
        max_repairs: int = 1,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> T:
        """Generate text and force-parse it into ``schema``.

        We first ask the LLM for strict JSON. If the response fails to
        validate, we make up to ``max_repairs`` follow-up calls that include
        the validation error so the model can self-correct. Schema
        violations are tracked through the ``SchemaValidator`` separately —
        this routine only handles the *recovery* path.
        """
        schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        instructions = (
            "Respond with ONE valid JSON object — no prose, no markdown fences.\n"
            f"It MUST validate against this JSON schema:\n{schema_hint}"
        )
        sys_prompt = f"{system}\n\n{instructions}" if system else instructions

        last_err: Optional[Exception] = None
        attempt_prompt = prompt
        for attempt in range(max_repairs + 1):
            response = self.complete(
                attempt_prompt,
                system=sys_prompt,
                agent_id=agent_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                payload = _extract_json_block(response.text)
            except (LLMClientError, json.JSONDecodeError) as e:
                last_err = e
                log.warning(
                    "JSON extraction failed on attempt %d (agent=%s): %s",
                    attempt,
                    agent_id,
                    str(e)[:200],
                )
                attempt_prompt = (
                    f"Your previous output contained invalid JSON: {e}\n"
                    "Return ONLY a valid JSON object that strictly matches the schema. "
                    "No trailing commas, no comments, no unescaped characters."
                )
                continue
            try:
                return schema.model_validate(payload)
            except ValidationError as e:
                last_err = e
                log.warning(
                    "Structured parse failed on attempt %d (agent=%s): %s",
                    attempt,
                    agent_id,
                    e.errors()[:3],
                )
                attempt_prompt = (
                    f"Your previous output was invalid: {e}\n"
                    "Return ONLY a JSON object that strictly matches the schema."
                )

        raise LLMClientError(
            f"complete_structured failed after {max_repairs + 1} attempts: {last_err}"
        )

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _chat_with_retry(self, **kwargs):
        return self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            timeout=self.request_timeout,
            **kwargs,
        )

    def _dummy_completion(self, prompt: str, system: Optional[str]) -> str:
        """Cheap deterministic responder used by unit tests.

        It looks for the JSON schema instructions injected by
        ``complete_structured`` and produces a minimally-valid object so
        downstream code paths are exercised end-to-end.
        """
        if system and "JSON schema" in system:
            return _dummy_structured_response(system)
        return f"[dummy:{self.model}] {prompt[:200]}"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _repair_json(text: str) -> str:
    """Attempt common fixes for malformed JSON from small LLMs."""
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _salvage_ranked_entries(raw: str) -> Any:
    """Extract complete ranked entries from a truncated JSON array.

    When the LLM produces a long list of objects but the output gets
    cut off mid-entry, we recover all fully-formed entries before the break.
    """
    pattern = re.compile(
        r'\{\s*"item_id"\s*:\s*"([^"]+)"\s*,\s*"score"\s*:\s*([\d.]+)\s*,'
        r'\s*"rationale"\s*:\s*\{[^}]*"personal"\s*:\s*"[^"]*"\s*,'
        r'\s*"directive"\s*:\s*"[^"]*"\s*,\s*"trend"\s*:\s*"[^"]*"\s*\}\s*\}',
        re.DOTALL,
    )
    entries = []
    for m in pattern.finditer(raw):
        try:
            obj = json.loads(m.group(0))
            entries.append(obj)
        except json.JSONDecodeError:
            continue
    if entries:
        return {"ranked": entries}
    return None


def _extract_json_block(text: str) -> Any:
    """Best-effort extraction of a JSON value from an LLM completion."""
    text = text.strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    # Fast-path: response is already a JSON document.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: pick out the largest balanced {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = text[start : end + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try removing trailing commas
        try:
            return json.loads(_repair_json(raw))
        except json.JSONDecodeError:
            pass
        # Try truncation to last complete entry and close brackets
        last_good = raw.rfind('"}')
        if last_good > 0:
            truncated = raw[: last_good + 2]
            open_braces = truncated.count("{") - truncated.count("}")
            open_brackets = truncated.count("[") - truncated.count("]")
            truncated += "}" * max(0, open_braces) + "]" * max(0, open_brackets)
            # Might still have trailing comma issues
            truncated = re.sub(r",\s*([}\]])", r"\1", truncated)
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass
        # Last resort: regex-extract individual complete ranked entries
        salvaged = _salvage_ranked_entries(text)
        if salvaged:
            return salvaged
        raise LLMClientError(
            f"Malformed JSON in completion (repair failed): {raw[:300]!r}"
        )
    raise LLMClientError(f"No JSON object found in completion: {text[:200]!r}")


def _dummy_structured_response(system_with_schema: str) -> str:
    """Build a JSON document that satisfies the schema embedded in `system`.

    The implementation only knows how to fill ``string``, ``number``,
    ``integer``, ``boolean``, ``array`` and ``object`` types — that covers
    every schema used by MARGO agents.
    """
    match = re.search(r"\{[\s\S]+\}", system_with_schema)
    if not match:
        return "{}"
    try:
        schema = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "{}"
    return json.dumps(_fill_schema(schema, schema), ensure_ascii=False)


def _fill_schema(schema: dict, root: dict) -> Any:
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return _fill_schema(root.get("$defs", {}).get(ref, {}), root)
    if "anyOf" in schema:
        return _fill_schema(schema["anyOf"][0], root)
    t = schema.get("type")
    if t == "object" or "properties" in schema:
        out = {}
        for k, v in (schema.get("properties") or {}).items():
            out[k] = _fill_schema(v, root)
        return out
    if t == "array":
        return [_fill_schema(schema.get("items", {}), root)]
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return True
    if t == "string":
        if schema.get("enum"):
            return schema["enum"][0]
        return "stub"
    return None


# --------------------------------------------------------------------------- #
# Module-level singleton (convenience)                                         #
# --------------------------------------------------------------------------- #


_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    """Return a module-wide ``LLMClient`` configured from the environment."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
