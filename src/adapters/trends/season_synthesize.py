"""Step 2 — Synthesize: one LLM call turns 4-axis articles into TrendItems.

Inputs (from Step 1 ``SeasonCollector.collect``):
* season label + window dates
* articles grouped by axis (RUNWAY / PRESS / STREET / POP), each with
  title, url, domain, publish_date, body excerpt

Outputs (one LLM call, structured):
* per-trend: keyword + trend_stage + confidence + rationale + citations
* season-level: summary + headline_themes

trend_stage classification — the four-axis cross-check:

    Across the axes the term appears in, count the *substantive* coverage.

    mainstream  =  appears prominently on ≥3 axes
                   (designer-driven AND press-validated AND publicly visible)
    rising      =  appears on 2 axes with momentum language
                   ('next', 'to watch', 'emerging', 'just dropped',
                    'designers are showing')
    niche       =  appears on 1 axis only, or scattered mentions without
                   momentum

This is the v5 design: the LLM looks at *who said what* (not page-view
curves) and gives a single verdict per trend.

Asymmetric authority preserved — Synthesize describes the season, it does
NOT recommend actions to the Expert. That is the Expert Agent's role.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from adapters.trends.season_collect import AXES, CollectResult
from adapters.trends.season_schema import (
    Citation,
    Confidence,
    SeasonTrendSnapshot,
    TrendItem,
    TrendStage,
)
from adapters.trends.seasons import Season
from adapters.trends.web_search import WebSnippet

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LLM I/O schema                                                              #
# --------------------------------------------------------------------------- #


class _RawCitation(BaseModel):
    title: str
    url: str
    domain: str = ""
    publish_date: Optional[str] = None
    excerpt: str = ""


class _RawTrendItem(BaseModel):
    keyword: str
    trend_stage: TrendStage
    confidence: Confidence
    rationale: str
    citations: list[_RawCitation] = Field(default_factory=list)


class _RawSynthesisOut(BaseModel):
    summary: str
    headline_themes: list[str] = Field(default_factory=list)
    trends: list[_RawTrendItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Public entry                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class SynthesisResult:
    trends: list[TrendItem]
    summary: str
    headline_themes: list[str]
    elapsed_s: float


def synthesize(
    *,
    llm,                                              # LLMClient
    season: Season,
    collect: CollectResult,
    domain: str = "fashion",
    region: str = "US",
    max_tokens: int = 8192,
    request_timeout_s: float = 240.0,
) -> SynthesisResult:
    """Run the single Step-2 LLM call. Always returns; on parse failure
    returns an empty SynthesisResult and logs the issue.
    """
    t0 = time.time()

    prompt = _build_prompt(
        season=season,
        collect=collect,
        domain=domain,
        region=region,
    )
    system = _system_prompt(season=season, domain=domain)

    # Temporarily raise request_timeout for this larger call.
    original_timeout = getattr(llm, "request_timeout", None)
    try:
        if original_timeout is not None:
            llm.request_timeout = max(original_timeout, request_timeout_s)
        raw = llm.complete_structured(
            prompt,
            _RawSynthesisOut,
            system=system,
            agent_id=f"trend-synthesize-{season.label}",
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as e:
        log.warning("Synthesize LLM call failed: %s", e)
        return SynthesisResult(
            trends=[], summary="", headline_themes=[], elapsed_s=time.time() - t0,
        )
    finally:
        if original_timeout is not None:
            llm.request_timeout = original_timeout

    trends = [_to_trend_item(r) for r in raw.trends]
    log.info(
        "Synthesize: %d trends produced (%.1fs)",
        len(trends), time.time() - t0,
    )
    return SynthesisResult(
        trends=trends,
        summary=raw.summary,
        headline_themes=raw.headline_themes,
        elapsed_s=time.time() - t0,
    )


# --------------------------------------------------------------------------- #
# Prompt construction                                                          #
# --------------------------------------------------------------------------- #


def _system_prompt(season: Season, domain: str) -> str:
    return f"""You are the Synthesize step of a fashion trend pipeline. Your job
is to read fashion trade-press coverage for season **{season.label}** and
produce a structured analysis. You are an objective season analyst — you
describe what the press said. You do NOT recommend actions.

INPUT STRUCTURE

You will receive trade-press articles grouped by FOUR AXES, where each axis
captures a different *origin* of the trend signal:

  1. RUNWAY  — designer / fashion-week origin
               (Vogue runway, BoF, voguebusiness reports)
  2. PRESS   — editorial consensus from authority titles
               (Vogue/Elle/WWD/Harper's "best of season" coverage)
  3. STREET  — consumer / wearability / shopping
               (Refinery29, WhoWhatWear, real-people coverage)
  4. POP     — celebrity, TV/movie/music influence, viral moments
               (people, variety, mashable, allure)

YOUR JOB PER TREND

For each distinct trend term that surfaces across the articles, emit ONE
TrendItem with:

  - keyword         : 2-4 word lowercase phrase, used verbatim by articles.
                      GOOD: "low-rise jeans", "quiet luxury", "platform shoes",
                      "tomato girl", "barbiecore".
                      BAD: "bold colors", "modern aesthetics", "sustainable
                      fashion" (too abstract — not a trend, a category).

  - trend_stage     : niche | rising | mainstream  (the SEASON-MOMENT stage)
      * mainstream — appears prominently on ≥3 axes
                     (e.g. designer ✓ + press ✓ + STREET or POP ✓);
                     the season's hero, broad cross-axis coverage.
      * rising     — appears on 2 axes with momentum language
                     ("next", "to watch", "emerging", "just dropped",
                      "designers are showing", "trend to wear");
                     building momentum, not yet hero.
      * niche      — appears on 1 axis only, or scattered mentions without
                     momentum language; specific subculture or designer-only.

  - confidence      : high | medium | low
      * high   — multiple articles with consistent language; named authority
                 publications (Vogue / WWD / Elle / BoF) lead.
      * medium — fewer articles or split tone; one strong source + one weak.
      * low    — single mention, ambiguous language, or only POP-axis viral
                 without industry validation.

  - rationale       : 1-2 sentences. CITE which publications said what
                      ("Vogue's Spring 2023 runway recap led with this;
                      Refinery29 ran a shopping guide"). Name the axes when
                      relevant ("appeared across RUNWAY and PRESS").

  - citations       : 1-4 articles you cited. Each has title, url, domain,
                      publish_date if known, and a 1-2 sentence excerpt.

SEASON-LEVEL OUTPUT

  - summary         : one paragraph capturing the season's mood — what the
                      press, runway, street, and pop culture together said
                      about this season.
  - headline_themes : 3-5 short phrases naming the dominant story arcs
                      (e.g. "quiet luxury consolidation", "Y2K cool-off",
                       "barbiecore moment").

QUALITY RULES

  * Surface ONE TrendItem per distinct trend term — do not merge trends
    into umbrella concepts.
  * Do NOT invent citations (URLs / titles) that weren't in the input.
  * Do NOT invent trend terms not present in any article.
  * It is OK to mark something as `niche` or `low` confidence — the
    season-truth matters more than producing few too-confident trends.
  * If pop culture (POP axis) is the *only* signal (e.g. only one celebrity
    wore it), the trend is `niche` even if the celebrity is famous —
    mainstream requires multi-axis adoption.

EXTRACT AT LEAST 15 DISTINCT TRENDS (ideally 20-30). Each of the following
deserves a separate TrendItem when articles mention it:
  - each distinct silhouette (low-rise jeans, wide-leg trousers, mini skirts...)
  - each specific material / fabric / print (satin, mesh, sheer, metallic, denim...)
  - each color trend (tangerine, lime, butter yellow, hot pink...)
  - each aesthetic / core (quiet luxury, y2k, balletcore, barbiecore...)
  - each specific footwear style (platform shoes, mary janes, kitten heels...)
  - each accessory category (micro bag, claw clip, statement necklace...)
  - each named designer movement if it crosses axes (Miu Miu's micro skirts...)

A single article like "Top 14 Trends from Spring 2023" should yield ~10
separate TrendItems — one per distinct trend the article calls out.
Don't collapse them into umbrellas like "vibrant colors" or "playful textures"
unless that IS the only level the articles use."""


def _build_prompt(
    *,
    season: Season,
    collect: CollectResult,
    domain: str,
    region: str,
) -> str:
    start_iso, end_iso = season.date_range()

    # Articles grouped by axis. Each block: [i] title (pub_date) / url / body.
    axis_blocks = []
    for axis in AXES:
        snips = collect.snippets_by_axis.get(axis, [])
        if not snips:
            axis_blocks.append(f"### {axis.upper()}\n  (no results)\n")
            continue
        lines = [f"### {axis.upper()}"]
        for i, s in enumerate(snips, 1):
            title = (s.title or "").strip()[:120]
            body = (s.raw_content or s.snippet or "").strip()
            pub = f" [pub: {s.published_date}]" if s.published_date else ""
            lines.append(
                f"  [{i}] {title}{pub}\n      url: {s.url}\n      excerpt: {body}"
            )
        axis_blocks.append("\n".join(lines))
    articles_block = "\n\n".join(axis_blocks)

    return f"""Season:     {season.label}  ({collect.season_phrase})
Window:     {start_iso} → {end_iso}
Region:     {region}
Domain:     {domain}

Articles surveyed: {collect.article_count} unique
Domains hit:       {", ".join(collect.sources_used)}

ARTICLES GROUPED BY AXIS:

{articles_block}

Now produce the structured synthesis."""


# --------------------------------------------------------------------------- #
# Schema bridging                                                              #
# --------------------------------------------------------------------------- #


def _to_trend_item(raw: _RawTrendItem) -> TrendItem:
    return TrendItem(
        keyword=raw.keyword.lower().strip(),
        trend_stage=raw.trend_stage,
        confidence=raw.confidence,
        rationale=raw.rationale,
        citations=[
            Citation(
                title=c.title,
                url=c.url,
                domain=c.domain or _extract_domain(c.url),
                publish_date=c.publish_date,
                excerpt=c.excerpt,
            )
            for c in (raw.citations or [])
        ],
    )


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    rest = url.split("://", 1)[-1]
    host = rest.split("/", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host
