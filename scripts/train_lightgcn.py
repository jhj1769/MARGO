"""Train a LightGCN baseline via RecBole (Step 3.2 of the Plan).

This script is a thin wrapper that delegates to RecBole's quick-start
runner. The output is converted into the ``.npz`` layout expected by
:class:`sage.baselines.LightGCNRetriever`.

Usage::

    python -m scripts.train_lightgcn \
        --processed-dir "data/Amazon Fashion/processed" \
        --out "data/Amazon Fashion/processed/lightgcn.npz"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=50)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        from recbole.quick_start import run_recbole  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "RecBole is not installed. `pip install recbole` (or train LightGCN in your "
            "own pipeline and save embeddings to the .npz layout)."
        ) from e

    # The actual RecBole config requires a converted atomic-file dataset.
    # That conversion belongs in a separate prep script; we keep this stub
    # honest by surfacing a clear error here rather than silently failing.
    raise NotImplementedError(
        "scripts/train_lightgcn.py: implement RecBole atomic-file conversion and "
        "then call run_recbole(model='LightGCN', dataset=..., config_file_list=[...])."
    )


if __name__ == "__main__":
    main()
