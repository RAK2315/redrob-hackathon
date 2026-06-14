"""OFFLINE: build the <=100-candidate sample + embedding subset for the HF Space.

Curates a representative 100-candidate slice so the live demo visibly separates
good fits from traps: strong matches (should rise), keyword stuffers (should sink),
and honeypots (should be auto-rejected). Writes everything the Space needs into
sandbox/ so the Space is self-contained and lightweight (numpy-only, no model).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.io_utils import load_candidates, resolve_candidates_path  # noqa: E402
from ranker.fit_score import fit_components  # noqa: E402
from ranker.stuffer import _n_ai_skills  # noqa: E402
from ranker.honeypot import is_honeypot  # noqa: E402
from ranker.pipeline import final_score  # noqa: E402
from ranker.dense import load_cosine_map  # noqa: E402
from ranker.combine import order_candidates  # noqa: E402
from ranker import jd_spec  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
SANDBOX = ROOT / "sandbox"
SAND_ART = SANDBOX / "artifacts"


def main() -> None:
    SAND_ART.mkdir(parents=True, exist_ok=True)
    cands = load_candidates(resolve_candidates_path())
    by_id = {c["candidate_id"]: c for c in cands}
    cm = load_cosine_map(ARTIFACTS)

    honeypots = [c["candidate_id"] for c in cands if is_honeypot(c)]
    stuffers = [c["candidate_id"] for c in cands
                if jd_spec.title_class(c) == "other" and _n_ai_skills(c) >= 5]

    scored = [(c["candidate_id"], final_score(c, fit_components(c), cm.get(c["candidate_id"], 0.0)))
              for c in cands if not is_honeypot(c)]
    ranked = [cid for cid, _ in order_candidates(scored)]
    strong = ranked[:70]                       # clear top fits

    pick: list[str] = []
    seen: set[str] = set()
    def take(ids, n):
        for cid in ids:
            if cid not in seen:
                pick.append(cid); seen.add(cid)
                n -= 1
                if n == 0:
                    break
    take(strong, 70)
    take(stuffers, 22)
    take(honeypots, 5)
    take(ranked[5000:5100], 100 - len(pick))   # a few mid/low fits as filler
    pick = pick[:100]

    # Write sample jsonl
    with open(SANDBOX / "sample_100.jsonl", "w", encoding="utf-8") as f:
        for cid in pick:
            f.write(json.dumps(by_id[cid]) + "\n")

    # Embedding subset aligned to the sample
    full_ids = list(np.load(ARTIFACTS / "cand_ids.npy", allow_pickle=True))
    pos = {str(c): i for i, c in enumerate(full_ids)}
    full_emb = np.load(ARTIFACTS / "embeddings.f16.npy")
    rows = [pos[c] for c in pick]
    np.save(SAND_ART / "embeddings.f16.npy", full_emb[rows])
    np.save(SAND_ART / "cand_ids.npy", np.array(pick))
    # jd vector is tiny; copy as-is
    np.save(SAND_ART / "jd_vecs.npy", np.load(ARTIFACTS / "jd_vecs.npy"))

    n_hp = sum(1 for c in pick if is_honeypot(by_id[c]))
    n_st = sum(1 for c in pick if by_id[c]["candidate_id"] in set(stuffers))
    print(f"[sandbox] wrote {len(pick)} candidates "
          f"({n_hp} honeypots, {n_st} stuffers, rest fits) -> sandbox/sample_100.jsonl")
    print(f"[sandbox] embedding subset -> sandbox/artifacts/ ({full_emb[rows].nbytes/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
