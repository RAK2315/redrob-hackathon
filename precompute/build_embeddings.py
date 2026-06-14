"""OFFLINE precompute: embed all 100k candidate profiles with a local
sentence-transformer and save the matrix as a shipped artifact.

This is the untimed precomputation step. Embedding 100k profiles on CPU takes
~30-60 min (measured ~14 docs/s at 256 tokens, ~2x faster at 128). The live
rank.py loads the saved matrix and never runs the model, keeping the ranking
step well under the 5-minute / no-network / no-GPU budget.

Determinism: fixed model revision (all-MiniLM-L6-v2), fixed max_seq_length,
L2-normalized output (so cosine == dot product), single thread-safe encode.
Run with HF_HUB_OFFLINE so it never reaches the network.

Usage:
    python precompute/build_embeddings.py [--candidates PATH] [--max-seq-length 128]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ranker.io_utils import load_candidates, resolve_candidates_path  # noqa: E402
from ranker import jd_spec  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--max-seq-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    path = resolve_candidates_path(args.candidates)
    print(f"[embed] loading {path.name} ...", flush=True)
    cands = load_candidates(path)
    ids = [c["candidate_id"] for c in cands]
    texts = [jd_spec.candidate_text(c) for c in cands]
    print(f"[embed] {len(texts)} profiles", flush=True)

    from sentence_transformers import SentenceTransformer
    import torch
    torch.manual_seed(0)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = args.max_seq_length

    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine == dot product downstream
        show_progress_bar=True,
    )
    dt = time.time() - t0
    print(f"[embed] encoded in {dt/60:.1f} min ({len(texts)/dt:.1f} docs/s)", flush=True)

    emb16 = emb.astype(np.float16)
    np.save(ARTIFACTS / "embeddings.f16.npy", emb16)
    np.save(ARTIFACTS / "cand_ids.npy", np.array(ids))
    meta = {
        "model": MODEL_NAME,
        "dim": int(emb.shape[1]),
        "max_seq_length": args.max_seq_length,
        "n": len(ids),
        "normalized": True,
    }
    (ARTIFACTS / "embeddings_meta.json").write_text(__import__("json").dumps(meta, indent=2))
    print(f"[embed] saved artifacts/embeddings.f16.npy "
          f"({emb16.nbytes/1e6:.0f} MB, dim={emb.shape[1]})", flush=True)


if __name__ == "__main__":
    main()
