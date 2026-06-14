#!/usr/bin/env python3
"""Redrob candidate ranker — LIVE entry point (the <=5-min reproduce command).

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pipeline (all CPU, no network, no GPU):
  1. load candidates (.jsonl or .jsonl.gz)
  2. honeypot hard-reject (impossible profiles removed before ranking)
  3. structured JD-fit + precomputed dense cosine -> blended fit
  4. keyword-stuffer penalty
  5. behavioral/availability modifier + location factor (bounded, sentinel-safe)
  6. combine -> enforce monotonic score + candidate_id tie-break -> top 100
  7. attach grounded reasoning (cached if present, else live template)
  8. final gate: assert zero known honeypots in the output

Embeddings are PRECOMPUTED offline (precompute/build_embeddings.py) and shipped
as artifacts/; this script only loads them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ranker.io_utils import load_candidates, resolve_candidates_path
from ranker.honeypot import find_honeypots
from ranker.fit_score import fit_components
from ranker.pipeline import final_score
from ranker.dense import load_cosine_map
from ranker.combine import build_submission, order_candidates
from ranker.reasoning import template_reasoning

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def score_candidates(candidates: list[dict], cosine_map: dict[str, float] | None):
    """Return (scored, components) where scored is [(id, final_score)] over
    non-honeypot candidates and components maps id -> fit_components dict."""
    honeypots = find_honeypots(candidates)
    scored: list[tuple[str, float]] = []
    components: dict[str, dict] = {}
    for c in candidates:
        cid = c["candidate_id"]
        if cid in honeypots:
            continue
        fc = fit_components(c)
        cosine = cosine_map.get(cid, 0.0) if cosine_map else 0.0
        final = final_score(c, fc, cosine)
        scored.append((cid, final))
        fc["cosine"] = cosine
        fc["final"] = final
        components[cid] = fc
    return scored, components, honeypots


def main() -> None:
    ap = argparse.ArgumentParser(description="Redrob top-100 candidate ranker")
    ap.add_argument("--candidates", default=None, help="path to candidates.jsonl(.gz)")
    ap.add_argument("--out", default="submission.csv", help="output CSV path")
    args = ap.parse_args()

    t0 = time.time()
    path = resolve_candidates_path(args.candidates)
    candidates = load_candidates(path)
    print(f"[rank] loaded {len(candidates)} candidates from {path.name} "
          f"({time.time()-t0:.1f}s)", flush=True)

    cosine_map = load_cosine_map(ARTIFACTS)
    print(f"[rank] dense cosine: {'loaded' if cosine_map else 'UNAVAILABLE -> structured-only'}",
          flush=True)

    by_id = {c["candidate_id"]: c for c in candidates}
    scored, components, honeypots = score_candidates(candidates, cosine_map)
    print(f"[rank] honeypots rejected: {len(honeypots)}; scorable: {len(scored)}", flush=True)

    # Determine the final top-100 ordering first, then build reasoning for them.
    ordered = order_candidates(scored)[:100]
    cache_path = ARTIFACTS / "reasoning_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    reasoning_map = {}
    for rank, (cid, _score) in enumerate(ordered, 1):
        if cid in cache and cache[cid].strip():
            reasoning_map[cid] = cache[cid]
        else:
            reasoning_map[cid] = template_reasoning(by_id[cid], components[cid], rank)

    rows = build_submission(scored, reasoning_map, args.out)

    # --- Final DQ gate: no known honeypot may appear in the output ---
    out_ids = {r["candidate_id"] for r in rows}
    leaked = out_ids & honeypots
    assert not leaked, f"HONEYPOT LEAK in top-100: {leaked}"

    print(f"[rank] wrote {len(rows)} rows -> {args.out}", flush=True)
    print(f"[rank] DONE in {time.time()-t0:.1f}s; 0 honeypots in top-100", flush=True)


if __name__ == "__main__":
    main()
