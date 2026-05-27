"""End-to-end 5-layer trend pipeline orchestrator (Enhancement 4).

Chains L5 → L1 → L2 → L3 → L4 into a single :class:`MultiSourceTrendSnapshot`.
The script in ``scripts/build_multisource_trend_snapshot.py`` is a thin glue
around :func:`build_multisource_snapshot`.

Design notes:

* All adapter calls are wrapped in try/except — a single source failing must
  never crash the pipeline. The bad source is recorded in ``source_metadata``.
* Volume percentiles are ranked *per source* before L2 normalisation so a
  source's quiet keywords are never compared against a louder source's loud
  ones.
* L4 is computed lazily on a per-keyword basis with optional caching so
  re-runs across overlapping pools are cheap.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from adapters.trends.base import TrendSourceAdapter
from adapters.trends.consensus import aggregate_keyword
from adapters.trends.multisource_schema import (
    MultiSourceTrendSignal,
    MultiSourceTrendSnapshot,
    RawSourceData,
    SourceSignal,
    TrendSemantic,
)
from adapters.trends.semantic_mapper import (
    CatalogMatcher,
    build_semantic_mapping,
    make_llm_proposer,
)
from adapters.trends.signal_normalizer import (
    normalize_to_signal,
    rank_volume_percentiles,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public entrypoint                                                           #
# --------------------------------------------------------------------------- #


def build_multisource_snapshot(
    *,
    keyword_pool: Mapping[str, list[str]],
    adapters: list[TrendSourceAdapter],
    time_window: tuple[str, str],
    domain: str = "fashion",
    time_window_label: Optional[str] = None,
    region: str = "US",
    llm_client: Optional[object] = None,
    catalog_matcher: Optional[CatalogMatcher] = None,
    processed_dir: Optional[Path] = None,
    skip_semantic: bool = False,
) -> MultiSourceTrendSnapshot:
    """Run the full L5→L4 pipeline and return a persisted-ready snapshot.

    Parameters
    ----------
    keyword_pool
        ``{"catalog_derived": [...], "brief_derived": [...], "merged": [...]}``.
        Only ``merged`` is analysed; the other two are recorded for provenance.
    adapters
        Source adapters in order of *priority* (most reliable first). Order
        doesn't affect math — reliability_prior does — but the snapshot's
        ``source_metadata`` is written in the supplied order.
    catalog_matcher
        Optional: when present *and* ``llm_client`` is present, L4 semantic
        mapping is computed for every keyword. Skip when you only want
        trend signals (e.g. fast dev iteration).
    skip_semantic
        Hard kill switch for L4. Useful in tests where the semantic step
        adds zero value.
    """
    t0 = time.time()
    merged_keywords = keyword_pool.get("merged") or []
    if not merged_keywords:
        log.warning("Empty merged keyword pool — snapshot will have no signals")

    # ---------- L1: fetch raw time-series for every (keyword, adapter) -----
    raws: list[RawSourceData] = []
    fetch_errors: dict[str, list[str]] = defaultdict(list)
    for kw in merged_keywords:
        for adapter in adapters:
            try:
                raw = adapter.fetch(kw, time_window, region=region)
            except Exception as e:  # noqa: BLE001 — adapters shouldn't raise, but defend
                log.warning("adapter %s raised on %r: %s", adapter.source_name, kw, e)
                fetch_errors[adapter.source_name].append(kw)
                continue
            if raw is not None:
                raws.append(raw)
            else:
                fetch_errors[adapter.source_name].append(kw)
    log.info("L1 fetched %d raw series across %d adapters (%.1fs)",
             len(raws), len(adapters), time.time() - t0)

    # ---------- L2: per-source volume percentile, then normalise -----------
    volume_percentiles = rank_volume_percentiles(raws)
    signals: list[SourceSignal] = []
    for raw in raws:
        pct = volume_percentiles.get((raw.keyword, raw.source_name), 0.5)
        signals.append(normalize_to_signal(raw, volume_percentile=pct))
    log.info("L2 normalised %d source signals", len(signals))

    # Group signals by keyword for L3.
    by_keyword: dict[str, dict[str, SourceSignal]] = defaultdict(dict)
    for sig in signals:
        by_keyword[sig.keyword][sig.source_name] = sig

    # ---------- L3: reliability-weighted consensus per keyword -------------
    reliability_priors = {a.source_name: a.reliability_prior for a in adapters}
    multi_signals: list[MultiSourceTrendSignal] = []
    for kw in merged_keywords:
        per_source = by_keyword.get(kw, {})
        if not per_source:
            # No adapter returned data for this keyword — skip from L3+.
            continue
        multi_signals.append(
            aggregate_keyword(kw, per_source, reliability_priors=reliability_priors)
        )
    log.info("L3 aggregated to %d multi-source trend signals", len(multi_signals))

    # ---------- L4: semantic mapping (optional) ----------------------------
    semantics: dict[str, TrendSemantic] = {}
    if not skip_semantic and llm_client is not None and catalog_matcher is not None:
        llm_propose = make_llm_proposer(llm_client)
        for ms in multi_signals:
            try:
                sem = build_semantic_mapping(
                    ms.keyword,
                    matcher=catalog_matcher,
                    llm_propose=llm_propose,
                    processed_dir=processed_dir,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("L4 semantic build failed for %r: %s", ms.keyword, e)
                continue
            semantics[ms.keyword] = sem
        log.info("L4 produced %d semantic mappings", len(semantics))

    # ---------- Assemble snapshot ------------------------------------------
    snapshot_label = time_window_label or f"{time_window[0]}__{time_window[1]}"
    snapshot_id = f"{domain}_{snapshot_label}".replace(":", "").replace("/", "_")

    source_metadata: dict[str, dict] = {}
    for adapter in adapters:
        source_metadata[adapter.source_name] = {
            "source_type": adapter.source_type,
            "reliability_prior": adapter.reliability_prior,
            "fetch_errors": fetch_errors.get(adapter.source_name, []),
        }

    snapshot = MultiSourceTrendSnapshot(
        snapshot_id=snapshot_id,
        domain=domain,
        time_window=snapshot_label,
        window_start_iso=time_window[0],
        window_end_iso=time_window[1],
        region=region,
        snapshot_date=datetime.now(timezone.utc).date().isoformat(),
        keyword_pool=dict(keyword_pool),
        signals=multi_signals,
        semantics=semantics,
        source_metadata=source_metadata,
        notes=f"Built by trends.pipeline in {time.time() - t0:.1f}s",
    )
    return snapshot


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #


_SNAPSHOT_DIR_NAME = "trend_cache"
_SNAPSHOT_FILE_PREFIX = "multisource"


def snapshot_path(processed_dir: Path, snapshot_id: str) -> Path:
    return (
        Path(processed_dir) / _SNAPSHOT_DIR_NAME
        / f"{_SNAPSHOT_FILE_PREFIX}_{snapshot_id}.json"
    )


def save_snapshot(snapshot: MultiSourceTrendSnapshot, processed_dir: Path) -> Path:
    path = snapshot_path(processed_dir, snapshot.snapshot_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    log.info("multi-source snapshot saved → %s", path)
    return path


def load_snapshot(path: Path) -> Optional[MultiSourceTrendSnapshot]:
    if not path.exists():
        return None
    try:
        return MultiSourceTrendSnapshot.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as e:  # noqa: BLE001
        log.warning("failed to load multisource snapshot %s: %s", path, e)
        return None
