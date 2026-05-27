"""Pluggable web search.

Three back-ends:

* ``tavily``   — production back-end (preferred for the paper experiments).
* ``serpapi``  — fall-back, same shape.
* ``stub``     — deterministic offline data injected via constructor; used
                 by unit tests and demo dry-runs.

The default is selected from ``MARGO_WEB_SEARCH_BACKEND`` (separate from
``MARGO_TREND_BACKEND`` which controls the trend pipeline mode). When the
env var is unset and no API key is available the searcher falls back to
``stub``.

The Tavily backend supports the optional kwargs needed by the Discover
agent: ``include_domains`` (trade-press allowlist), ``start_date`` /
``end_date`` (publish-date filter for historical seasons), ``search_depth``
(``"basic"`` cheap, ``"advanced"`` returns ``raw_content`` for LLM reading).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSnippet:
    url: str
    snippet: str
    title: str = ""
    raw_content: str = ""              # full article body (advanced depth)
    published_date: Optional[str] = None
    score: Optional[float] = None      # Tavily relevance


_VALID_BACKENDS = ("tavily", "serpapi", "stub", "auto")


@dataclass
class WebSearcher:
    """Frontend that picks an actual backend at construction time."""

    backend: str = field(
        default_factory=lambda: os.getenv("MARGO_WEB_SEARCH_BACKEND", "auto")
    )
    api_key: Optional[str] = None
    top_k: int = 5
    stub_responses: Optional[dict[str, list[WebSnippet]]] = None

    def __post_init__(self) -> None:
        # Guard against env-var name collisions / typos: any unknown backend
        # value drops through to auto-detection instead of silently failing
        # later inside search().
        if self.backend not in _VALID_BACKENDS:
            log.info(
                "WebSearcher: unknown backend %r — auto-detecting instead.",
                self.backend,
            )
            self.backend = "auto"
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

    def search(
        self,
        query: str,
        *,
        max_results: Optional[int] = None,
        include_domains: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        start_date: Optional[str] = None,   # "YYYY-MM-DD"
        end_date: Optional[str] = None,
        search_depth: str = "basic",        # "basic" | "advanced"
    ) -> list[WebSnippet]:
        if self.backend == "tavily":
            return self._tavily(
                query,
                max_results=max_results or self.top_k,
                include_domains=list(include_domains) if include_domains else None,
                exclude_domains=list(exclude_domains) if exclude_domains else None,
                start_date=start_date,
                end_date=end_date,
                search_depth=search_depth,
            )
        if self.backend == "serpapi":
            return self._serpapi(query, max_results=max_results or self.top_k)
        return self._stub(query)

    # ------------------------------------------------------------------ #
    # Backends                                                            #
    # ------------------------------------------------------------------ #

    def _tavily(
        self,
        query: str,
        *,
        max_results: int,
        include_domains: Optional[list[str]],
        exclude_domains: Optional[list[str]],
        start_date: Optional[str],
        end_date: Optional[str],
        search_depth: str,
    ) -> list[WebSnippet]:  # pragma: no cover - network
        try:
            import requests  # type: ignore
        except ImportError as e:
            raise RuntimeError("`requests` required for tavily backend") from e
        payload: dict = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            # Always ask for body — discover/synthesize benefit from it and
            # Tavily's API silently omits raw_content unless this is set.
            "include_raw_content": search_depth == "advanced",
        }
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        r = requests.post(
            "https://api.tavily.com/search", json=payload, timeout=45
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return [
            WebSnippet(
                url=x.get("url", ""),
                snippet=x.get("content", ""),
                title=x.get("title", ""),
                raw_content=x.get("raw_content", "") or "",
                published_date=x.get("published_date"),
                score=x.get("score"),
            )
            for x in results
        ]

    def _serpapi(self, query: str, *, max_results: int) -> list[WebSnippet]:  # pragma: no cover - network
        try:
            import requests  # type: ignore
        except ImportError as e:
            raise RuntimeError("`requests` required for serpapi backend") from e
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"q": query, "api_key": self.api_key, "num": max_results},
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
