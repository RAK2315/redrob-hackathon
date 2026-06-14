"""OFFLINE precompute: embed the JD into a query vector for dense matching.

We embed the JD's decisive blocks verbatim (the "things you absolutely need"
requirements and the "ideal candidate" archetype) rather than the whole
document, so the cosine similarity rewards candidates whose prose matches what
the role actually needs — production embeddings/retrieval/ranking systems, vector
search, ranking evaluation — which is the tier-5 differentiator.

Saved as artifacts/jd_vecs.npy: row 0 is the mean query vector used at rank
time; subsequent rows are the individual block vectors (kept for inspection).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# Verbatim / near-verbatim excerpts of the decisive JD requirements.
JD_BLOCKS = [
    # "Things you absolutely need"
    "Production experience with embeddings-based retrieval systems "
    "(sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real users; "
    "handled embedding drift, index refresh, retrieval-quality regression in production. "
    "Production experience with vector databases or hybrid search infrastructure: "
    "Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. "
    "Strong Python and code quality. "
    "Designing evaluation frameworks for ranking systems: NDCG, MRR, MAP, "
    "offline-to-online correlation, A/B test interpretation.",
    # "Ideal candidate" archetype
    "Senior AI engineer, 6-8 years total experience, 4-5 years in applied ML/AI "
    "roles at product companies, not pure services. Has shipped at least one "
    "end-to-end ranking, search, or recommendation system to real users at "
    "meaningful scale. Strong opinions about retrieval (hybrid vs dense), "
    "evaluation (offline vs online), and LLM integration. Hands-on production "
    "engineer who writes code, builds intelligence-layer ranking and matching "
    "systems for candidate-job search.",
    # Nice-to-haves
    "LLM fine-tuning (LoRA, QLoRA, PEFT). Learning-to-rank models (XGBoost or "
    "neural). HR-tech, recruiting tech, or marketplace products. Distributed "
    "systems and large-scale inference optimization. Open-source contributions "
    "in AI/ML.",
]


def main() -> None:
    from sentence_transformers import SentenceTransformer
    import torch
    torch.manual_seed(0)
    ARTIFACTS.mkdir(exist_ok=True)

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = 256
    blocks = model.encode(
        [b.lower() for b in JD_BLOCKS],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query = blocks.mean(axis=0)
    query = query / (np.linalg.norm(query) + 1e-12)
    out = np.vstack([query[None, :], blocks]).astype(np.float32)
    np.save(ARTIFACTS / "jd_vecs.npy", out)
    print(f"[jd] saved artifacts/jd_vecs.npy shape={out.shape} "
          f"(row0=mean query, rows1-{len(JD_BLOCKS)}=blocks)")


if __name__ == "__main__":
    main()
