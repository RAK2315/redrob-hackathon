"""OFFLINE precompute: build the reasoning cache for the top shortlist.

Runs a dry-run ranking, takes the top ~150 (a buffer beyond the final 100 in
case later tweaks reorder the boundary), and writes grounded reasoning to
artifacts/reasoning_cache.json keyed by candidate_id. rank.py joins this cache
at run time; any miss falls back to the live template (so the cache is an
optional enhancement, never a hard dependency).

Default path is the deterministic fact-template (identical to rank.py's live
fallback). If ANTHROPIC_API_KEY is set, the top shortlist is upgraded to
LLM-written reasoning fed ONLY that candidate's own fields, and every line is
passed through reasoning.verify_grounded() before being accepted — any line
that fails verification falls back to the template. No candidate data leaves the
machine unless a key is explicitly present.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.io_utils import load_candidates, resolve_candidates_path  # noqa: E402
from ranker.fit_score import fit_components  # noqa: E402
from ranker.pipeline import final_score  # noqa: E402
from ranker.dense import load_cosine_map  # noqa: E402
from ranker.combine import order_candidates  # noqa: E402
from ranker.honeypot import find_honeypots  # noqa: E402
from ranker.reasoning import template_reasoning, gather_facts, verify_grounded  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
SHORTLIST = 150


def _llm_reasoning(candidate, fc, rank, client):
    """Optional LLM upgrade. Feeds ONLY the candidate's own facts; verified."""
    facts = gather_facts(candidate, fc)
    prompt = (
        "Write a 1-2 sentence recruiter justification for ranking this candidate "
        f"at position {rank} for a Senior AI Engineer role focused on production "
        "retrieval/ranking/search systems. Use ONLY these facts; do not invent "
        "skills or employers. Be specific and honest about concerns.\n"
        f"FACTS: {json.dumps(facts, default=str)}"
    )
    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=160,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    if text and verify_grounded(text, candidate):
        return text
    return template_reasoning(candidate, fc, rank)  # fallback on any doubt


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    candidates = load_candidates(resolve_candidates_path())
    by_id = {c["candidate_id"]: c for c in candidates}
    honeypots = find_honeypots(candidates)
    cosine_map = load_cosine_map(ARTIFACTS)

    scored, comps = [], {}
    for c in candidates:
        cid = c["candidate_id"]
        if cid in honeypots:
            continue
        fc = fit_components(c)
        cosine = cosine_map.get(cid, 0.0) if cosine_map else 0.0
        scored.append((cid, final_score(c, fc, cosine)))
        comps[cid] = fc

    ordered = order_candidates(scored)[:SHORTLIST]

    client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            print("[reason] ANTHROPIC_API_KEY found -> LLM-upgraded reasoning")
        except Exception as e:  # noqa: BLE001
            print(f"[reason] LLM unavailable ({e}); using template")
    else:
        print("[reason] no API key -> deterministic template reasoning")

    t0 = time.time()
    cache = {}
    for rank, (cid, _s) in enumerate(ordered, 1):
        c, fc = by_id[cid], comps[cid]
        if client is not None:
            cache[cid] = _llm_reasoning(c, fc, rank, client)
        else:
            cache[cid] = template_reasoning(c, fc, rank)

    (ARTIFACTS / "reasoning_cache.json").write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[reason] wrote {len(cache)} entries to artifacts/reasoning_cache.json "
          f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
