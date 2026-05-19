"""Step 2 — Build the BGE-M3 dense index over the processed item table.

Supports multi-GPU: each GPU encodes a shard in a separate process,
then the parent merges embeddings and builds the FAISS index.

Usage::

    # Auto-detect all GPUs
    python -m scripts.build_index --processed-dir "data/Amazon Fashion/processed"

    # Explicitly pick GPUs
    python -m scripts.build_index --processed-dir "data/Amazon Fashion/processed" --gpus 0,1,2
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from margo.domains.fashion.loader import build_item_text

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BGE-M3 + FAISS index over items.parquet.")
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--model", default="BAAI/bge-m3")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated GPU ids, e.g. '0,1,2'. Default: all visible.")
    return p.parse_args()


def _get_gpu_ids(gpus_arg: str | None) -> list[int]:
    if gpus_arg:
        return [int(x.strip()) for x in gpus_arg.split(",")]
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        return []
    return list(range(n))


def _encode_shard(
    gpu_id: int,
    texts: list[str],
    model_name: str,
    batch_size: int,
    use_fp16: bool,
    out_path: str,
):
    """Worker: encode a shard of texts on a single GPU and save to disk."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [GPU{gpu_id}] %(message)s")
    wlog = logging.getLogger(f"worker-{gpu_id}")

    from FlagEmbedding import BGEM3FlagModel
    wlog.info("loading model on GPU %d (%d texts)", gpu_id, len(texts))
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device="cuda")
    wlog.info("encoding start")
    t0 = time.time()
    embs = model.encode(texts, batch_size=batch_size)["dense_vecs"]
    embs = np.asarray(embs, dtype="float32")
    elapsed = time.time() - t0
    wlog.info("encoding done: %d vectors in %.1fs (%.0f texts/s)",
              len(embs), elapsed, len(texts) / elapsed)
    np.save(out_path, embs)
    wlog.info("saved shard → %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    items_path = args.processed_dir / "items.parquet"
    items = pd.read_parquet(items_path)
    log.info("loaded %d items", len(items))

    log.info("building item texts …")
    texts = [build_item_text(r) for _, r in items.iterrows()]
    item_ids = [str(r.get("parent_asin") or r.get("item_id")) for _, r in items.iterrows()]
    log.info("item texts ready (%d)", len(texts))

    gpu_ids = _get_gpu_ids(args.gpus)

    if not gpu_ids:
        log.warning("No GPUs found — falling back to single-process CPU encoding")
        from FlagEmbedding import BGEM3FlagModel
        import faiss

        model = BGEM3FlagModel(args.model, use_fp16=args.fp16)
        log.info("encoding %d texts on CPU …", len(texts))
        embeddings = np.asarray(
            model.encode(texts, batch_size=args.batch_size)["dense_vecs"], dtype="float32"
        )
    else:
        log.info("using %d GPUs: %s", len(gpu_ids), gpu_ids)
        n = len(texts)
        shard_size = (n + len(gpu_ids) - 1) // len(gpu_ids)
        shards = [texts[i * shard_size : (i + 1) * shard_size] for i in range(len(gpu_ids))]

        tmpdir = tempfile.mkdtemp(prefix="margo_bge_")
        shard_paths = [os.path.join(tmpdir, f"shard_{i}.npy") for i in range(len(gpu_ids))]

        ctx = mp.get_context("spawn")
        procs = []
        for i, gid in enumerate(gpu_ids):
            p = ctx.Process(
                target=_encode_shard,
                args=(gid, shards[i], args.model, args.batch_size, args.fp16, shard_paths[i]),
            )
            p.start()
            procs.append(p)
            log.info("launched worker GPU %d (PID %d, %d texts)", gid, p.pid, len(shards[i]))

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker PID {p.pid} exited with code {p.exitcode}")

        log.info("all workers done — merging shards")
        parts = [np.load(sp) for sp in shard_paths]
        embeddings = np.concatenate(parts, axis=0)

        for sp in shard_paths:
            os.remove(sp)
        os.rmdir(tmpdir)

    import faiss
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    out_emb = args.processed_dir / "item_embeddings.npy"
    out_index = args.processed_dir / "faiss_index.bin"
    out_ids = args.processed_dir / "item_ids.txt"
    np.save(out_emb, embeddings)
    faiss.write_index(index, str(out_index))
    out_ids.write_text("\n".join(item_ids), encoding="utf-8")
    log.info("wrote %s (%d vectors, dim=%d)", out_index, len(embeddings), dim)
    log.info("wrote %s / %s", out_emb, out_ids)


if __name__ == "__main__":
    main()
