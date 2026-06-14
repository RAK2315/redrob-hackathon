"""Load precomputed embeddings and compute cosine similarity to the JD.

This is the only place dense vectors enter the LIVE path. We never run the
embedding model at rank time (it would blow the 5-min/no-network budget); we
load the shipped artifacts produced offline by precompute/build_embeddings.py.

If artifacts are absent (e.g. someone runs rank.py before precompute), we return
None so the caller falls back to structured-only scoring (cosine := 0) rather
than crashing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def load_cosine_map(artifacts_dir: Path | None = None) -> dict[str, float] | None:
    """Return {candidate_id: cosine_to_JD in [0,1]} or None if unavailable."""
    d = Path(artifacts_dir) if artifacts_dir else ARTIFACTS
    emb_path = d / "embeddings.f16.npy"
    ids_path = d / "cand_ids.npy"
    jd_path = d / "jd_vecs.npy"
    if not (emb_path.exists() and ids_path.exists() and jd_path.exists()):
        return None

    emb = np.load(emb_path).astype(np.float32)        # (N, dim), L2-normalized
    ids = np.load(ids_path, allow_pickle=True)
    jd = np.load(jd_path).astype(np.float32)[0]        # row 0 = mean query vector
    jd = jd / (np.linalg.norm(jd) + 1e-12)

    cos = emb @ jd                                     # cosine (vectors normalized)
    cos = np.clip(cos, 0.0, 1.0)                       # negatives -> 0 (no signal)
    return {str(cid): float(c) for cid, c in zip(ids, cos)}
