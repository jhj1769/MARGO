"""Step 1 — Collect: deterministic 4-axis Tavily survey of a season.

v5 replaces the v4 ReAct + auxiliary-source-fetch flow with a flat,
deterministic structure:

    For each of 4 axes (RUNWAY, PRESS, STREET, POP):
        Tavily search (domain allowlist + season-window date filter)

That's it. No keyword extraction LLM call at this stage — Synthesize sees
the raw article bodies and makes all judgements in one structured pass.

The 4 axes correspond to where a fashion-trend signal *originates*:
    RUNWAY  — designer / fashion-week reports
    PRESS   — editorial consensus from authority titles
    STREET  — consumer / wearability / shopping coverage
    POP     — celebrity, TV/movie/music influence, viral moments

A trend that shows up across multiple axes is mainstream; one that shows
up only on POP (e.g. one celebrity wore it) is niche; etc. Lifecycle
classification happens in Synthesize using exactly this cross-axis check.

Design constraint — asymmetric authority (see ``season_schema.py``):
This step deliberately does not see the operator brief. Trend Agent's
role is objective season analysis; operator intent lives in the Expert
Directive flow.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from adapters.trends.seasons import Season
from adapters.trends.web_search import WebSearcher, WebSnippet

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Trade-press domain allowlist                                                #
# --------------------------------------------------------------------------- #
#
# 47 fashion-relevant domains across 8 tiers. Same list is passed to every
# Tavily search; Tavily decides which subset ranks best per query.

FASHION_MEDIA_DOMAINS: list[str] = [
    # ── Tier 1: high-fashion authority ──────────────────────────────────
    "vogue.com", "vogue.co.uk", "voguebusiness.com",
    "wwd.com", "elle.com", "harpersbazaar.com",
    "businessoffashion.com",

    # ── Tier 2: streetwear / culture ────────────────────────────────────
    "highsnobiety.com", "hypebeast.com",
    "complex.com",
    "ssense.com",
    "dazeddigital.com",

    # ── Tier 3: women's lifestyle ───────────────────────────────────────
    "whowhatwear.com", "refinery29.com", "popsugar.com",
    "instyle.com", "marieclaire.com", "cosmopolitan.com",
    "thecut.com",
    "fashionista.com",

    # ── Tier 4: men's ───────────────────────────────────────────────────
    "gq.com", "esquire.com",
    "mrporter.com",

    # ── Tier 5: indie / avant-garde (early micro-trend signal) ──────────
    "i-d.co",
    "anothermag.com",
    "032c.com",
    "showstudio.com",
    "vmagazine.com",

    # ── Tier 6: industry / trade ────────────────────────────────────────
    "fashionunited.com", "fashionweekdaily.com",
    "thefashionspot.com",
    "nytimes.com",

    # ── Tier 7: lifestyle / general fashion ─────────────────────────────
    "vanityfair.com", "papermag.com", "nylon.com",

    # ── Tier 8: mass / pop culture (celebrity / film / viral) ───────────
    "people.com", "usmagazine.com",
    "glamour.com", "allure.com",
    "bustle.com",
    "mashable.com",
    "variety.com",
]


# --------------------------------------------------------------------------- #
# 4 axes — the entire search plan                                             #
# --------------------------------------------------------------------------- #


SEARCH_PLAN: list[tuple[str, str]] = [
    # 1) RUNWAY — designer origin (fashion week recaps, designer-driven)
    ("runway",
     "{phrase} fashion week runway designer collections key trends"),

    # 2) PRESS — editorial consensus (Vogue/Elle/WWD/Harper's best-of)
    ("press",
     "defining fashion trends and aesthetics of {phrase} editorial report"),

    # 3) STREET — consumer / wearability (Refinery29, WhoWhatWear, shopping)
    ("street",
     "{phrase} fashion street style shopping wearable items what to wear"),

    # 4) POP — celebrity, TV/movie/music influence, viral moments
    ("pop",
     "{phrase} viral fashion celebrity outfits tv movie music influence"),
]


AXES: list[str] = [name for name, _ in SEARCH_PLAN]


def _season_phrase(season: Season) -> str:
    """Convert internal season label to natural search phrasing.

    Trade press writes 'spring 2023' / 'fall 2017', not '2023-SS' / '2017-AW'.
    Using the natural phrasing yields far better Tavily hits.
    """
    half = "spring" if season.half == "SS" else "fall"
    return f"{half} {season.year}"


# --------------------------------------------------------------------------- #
# Result                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class CollectResult:
    """Output of Step 1. Articles grouped by axis + flat unique list."""

    season_label: str
    season_phrase: str
    snippets_by_axis: dict[str, list[WebSnippet]] = field(default_factory=dict)
    all_snippets: list[WebSnippet] = field(default_factory=list)  # deduped
    sources_used: list[str] = field(default_factory=list)         # unique domains
    elapsed_s: float = 0.0

    @property
    def article_count(self) -> int:
        return len(self.all_snippets)


# --------------------------------------------------------------------------- #
# Collector                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class SeasonCollector:
    """Executes the 4-axis Tavily search plan for one season."""

    searcher: WebSearcher
    per_search_max_results: int = 12
    raw_content_truncate: int = 1500          # chars retained per article

    def collect(self, season: Season) -> CollectResult:
        t0 = time.time()
        start_iso, end_iso = season.date_range()
        phrase = _season_phrase(season)

        result = CollectResult(
            season_label=season.label,
            season_phrase=phrase,
        )

        seen_urls: set[str] = set()
        seen_domains: set[str] = set()

        for axis_name, query_tpl in SEARCH_PLAN:
            query = query_tpl.format(phrase=phrase)
            snippets = self._do_search(query, start_iso=start_iso, end_iso=end_iso)
            log.info(
                "Collect [%s] '%s' → %d results", axis_name, query, len(snippets),
            )

            kept: list[WebSnippet] = []
            for s in snippets:
                # Truncate raw_content to keep prompt size bounded; preserve
                # the rest of the WebSnippet unchanged.
                if s.raw_content and len(s.raw_content) > self.raw_content_truncate:
                    s = WebSnippet(
                        url=s.url,
                        snippet=s.snippet,
                        title=s.title,
                        raw_content=s.raw_content[: self.raw_content_truncate] + "...",
                        published_date=s.published_date,
                        score=s.score,
                    )
                kept.append(s)
                if s.url and s.url not in seen_urls:
                    seen_urls.add(s.url)
                    result.all_snippets.append(s)
                domain = _extract_domain(s.url)
                if domain and domain not in seen_domains:
                    seen_domains.add(domain)

            result.snippets_by_axis[axis_name] = kept

        result.sources_used = sorted(seen_domains)
        result.elapsed_s = time.time() - t0
        log.info(
            "Collect %s: %d unique articles from %d domains in %.1fs",
            season.label, len(result.all_snippets), len(result.sources_used),
            result.elapsed_s,
        )
        return result

    # --------------------------------------------------------------- #
    # Internals                                                       #
    # --------------------------------------------------------------- #

    def _do_search(
        self,
        query: str,
        *,
        start_iso: str,
        end_iso: str,
    ) -> list[WebSnippet]:
        try:
            return self.searcher.search(
                query,
                max_results=self.per_search_max_results,
                include_domains=FASHION_MEDIA_DOMAINS,
                start_date=start_iso,
                end_date=end_iso,
                search_depth="advanced",
            )
        except Exception as e:                        # noqa: BLE001
            log.warning("Collect search failed for %r: %s", query, e)
            return []


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _extract_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    # Cheap parse — Tavily URLs are HTTPS with standard host. Avoid importing
    # urllib for one call.
    rest = url.split("://", 1)[-1]
    host = rest.split("/", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host or None
