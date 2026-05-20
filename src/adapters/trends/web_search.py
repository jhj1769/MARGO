"""Pluggable web search.

Three back-ends:

* ``tavily``   — production back-end (preferred for the paper experiments).
* ``serpapi``  — fall-back, same shape.
* ``stub``     — deterministic offline data injected via constructor; used
                 by unit tests and demo dry-runs.

The default is selected from ``MARGO_TREND_BACKEND``; when unset and no
API key is available the searcher falls back to ``stub``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSnippet:
    url: str
    snippet: str
    title: str = ""


@dataclass
class WebSearcher:
    """Frontend that picks an actual backend at construction time."""

    backend: str = field(default_factory=lambda: os.getenv("MARGO_TREND_BACKEND", "auto"))
    api_key: Optional[str] = None
    top_k: int = 5
    stub_responses: Optional[dict[str, list[WebSnippet]]] = None

    def __post_init__(self) -> None:
        if self.backend == "auto":
            if os.getenv("TAVILY_API_KEY"):
                self.backend = "tavily"
            elif os.getenv("SERPAPI_API_KEY"):
                self.backend = "serpapi"
            else:
                self.backend = "stub"
                log.info("WebSearcher falling back to 'stub' (no API key found).")
        if self.backend == "tavily" and self.api_key is None:
            self.api_key = os.getenv("TAVILY_API_KEY")
        if self.backend == "serpapi" and self.api_key is None:
            self.api_key = os.getenv("SERPAPI_API_KEY")

    # ------------------------------------------------------------------ #
    # Public                                                              #
    # ------------------------------------------------------------------ #

    def search(self, query: str) -> list[WebSnippet]:
        if self.backend == "tavily":
            return self._tavily(query)
        if self.backend == "serpapi":
            return self._serpapi(query)
        return self._stub(query)

    # ------------------------------------------------------------------ #
    # Backends                                                            #
    # ------------------------------------------------------------------ #

    def _tavily(self, query: str) -> list[WebSnippet]:  # pragma: no cover - network
        try:
            import requests  # type: ignore
        except ImportError as e:
            raise RuntimeError("`requests` required for tavily backend") from e
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": self.api_key, "query": query, "max_results": self.top_k},
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return [
            WebSnippet(url=x.get("url", ""), snippet=x.get("content", ""), title=x.get("title", ""))
            for x in results
        ]

    def _serpapi(self, query: str) -> list[WebSnippet]:  # pragma: no cover - network
        try:
            import requests  # type: ignore
        except ImportError as e:
            raise RuntimeError("`requests` required for serpapi backend") from e
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"q": query, "api_key": self.api_key, "num": self.top_k},
            timeout=30,
        )
        r.raise_for_status()
        organic = r.json().get("organic_results", [])
        return [
            WebSnippet(url=x.get("link", ""), snippet=x.get("snippet", ""), title=x.get("title", ""))
            for x in organic
        ]

    def _stub(self, query: str) -> list[WebSnippet]:
        if self.stub_responses and query in self.stub_responses:
            return self.stub_responses[query]
        log.warning("WebSearcher stub returning empty for query=%r", query)
        return []


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def load_stub_responses(path: str) -> dict[str, list[WebSnippet]]:
    """Load a JSON file of canned snippets for offline demos."""
    raw = json.loads(open(path, encoding="utf-8").read())
    out: dict[str, list[WebSnippet]] = {}
    for query, snippets in raw.items():
        out[query] = [WebSnippet(**s) for s in snippets]
    return out
