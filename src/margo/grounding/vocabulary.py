"""Domain vocabulary container + drift detector.

The vocabulary is a *flat* set of normalised tokens grouped by attribute
class (``color``, ``material``, ``silhouette``, …). Anything an agent
emits outside this set is recorded toward the Vocabulary Drift Rate (VDR).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']{1,}")


def _normalise(token: str) -> str:
    t = unicodedata.normalize("NFKC", token).lower().strip()
    return t


class Vocabulary:
    """Attribute-bucketed vocabulary of allowed tokens."""

    def __init__(self, buckets: dict[str, set[str]]) -> None:
        self.buckets: dict[str, set[str]] = {k: {_normalise(v) for v in vs} for k, vs in buckets.items()}
        # `all_tokens` is the union, used for fast drift scoring.
        self.all_tokens: set[str] = set().union(*self.buckets.values()) if self.buckets else set()
        self.drift_counter: Counter[str] = Counter()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({k: sorted(v) for k, v in self.buckets.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: set(v) for k, v in data.items()})

    @classmethod
    def from_attribute_columns(cls, rows: Iterable[dict], columns: dict[str, str]) -> "Vocabulary":
        """Build a vocabulary by harvesting unique values from item rows.

        Parameters
        ----------
        rows:
            Iterable of item dicts (e.g. ``items_df.to_dict(orient='records')``).
        columns:
            Map ``bucket_name -> column_name``. Multi-value cells (lists,
            ``", "`` separated strings) are split.
        """
        buckets: dict[str, set[str]] = {b: set() for b in columns}
        for row in rows:
            for bucket, col in columns.items():
                v = row.get(col)
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    for x in v:
                        if x:
                            buckets[bucket].add(_normalise(str(x)))
                else:
                    for part in re.split(r"[,/;|]", str(v)):
                        part = part.strip()
                        if part:
                            buckets[bucket].add(_normalise(part))
        return cls(buckets)

    # ------------------------------------------------------------------ #
    # Drift checking                                                       #
    # ------------------------------------------------------------------ #

    def in_vocabulary(self, token: str) -> bool:
        return _normalise(token) in self.all_tokens

    def check_text(self, text: str) -> tuple[int, int, list[str]]:
        """Return ``(in_count, out_count, out_tokens)`` for tokens in ``text``.

        Only alphabetic tokens of length ≥ 2 are considered, so prices and
        sentence punctuation don't pollute the rate.
        """
        in_count = out_count = 0
        out_tokens: list[str] = []
        for raw in _TOKEN_RE.findall(text):
            tok = _normalise(raw)
            if tok in self.all_tokens:
                in_count += 1
            else:
                # Skip extremely common stop-words to keep VDR meaningful.
                if tok in _STOPWORDS:
                    continue
                out_count += 1
                out_tokens.append(tok)
                self.drift_counter[tok] += 1
        return in_count, out_count, out_tokens


_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with",
    "is", "are", "be", "this", "that", "it", "its", "from", "by", "as",
    "you", "your", "we", "our", "their", "they", "but", "if", "so",
    "user", "item", "directive", "trend", "category",
}
