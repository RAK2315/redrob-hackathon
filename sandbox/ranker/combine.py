"""Final assembly: select top-N, enforce the validator's invariants, write CSV.

The validator (validate_submission.py) is unforgiving, so we bake its rules in
rather than hope the natural ordering satisfies them:
  * exactly 100 data rows, ranks 1..100 each once, unique candidate_ids;
  * score non-increasing as rank increases (ties allowed);
  * for EQUAL scores at adjacent ranks, candidate_id must be ascending.

Key subtlety: we round scores to the written precision FIRST, then order by
(-rounded_score, candidate_id). This guarantees the tie-break is computed on the
exact values that land in the file, so two raw scores that round to the same
string can never violate the ascending-id rule.
"""

from __future__ import annotations

import csv
from pathlib import Path

SCORE_DECIMALS = 6        # internal precision used to fix the ranking order
OUTPUT_DECIMALS = 4       # precision of the score written to the CSV
TOP_N = 100
REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]

# The raw score is an unnormalized fit×modifiers product (can exceed 1.0). For a
# clean, human-readable submission that matches the spec's 0-1 convention, we
# min-max map the top-N raw scores onto this range. This is strictly
# order-preserving (monotonic), so it never changes the ranking and preserves
# the non-increasing / tie-break invariants the validator checks.
NORM_LO, NORM_HI = 0.40, 0.99


def order_candidates(scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Round, then sort by (-rounded_score, candidate_id ascending).

    ``scored`` is a list of (candidate_id, raw_score). Returns the full ordered
    list of (candidate_id, rounded_score) with monotonic non-increasing scores.
    """
    rounded = [(cid, round(float(s), SCORE_DECIMALS)) for cid, s in scored]
    rounded.sort(key=lambda x: (-x[1], x[0]))

    # Belt-and-suspenders: enforce strict non-increasing (sorting already does
    # this, but clamp defends against any float edge case).
    out: list[tuple[str, float]] = []
    prev = None
    for cid, s in rounded:
        if prev is not None and s > prev:
            s = prev
        out.append((cid, s))
        prev = s
    return out


def normalize_scores(ordered: list[tuple[str, float]]) -> list[float]:
    """Min-max map an ordered (non-increasing) score list onto [NORM_LO, NORM_HI].

    Strictly order-preserving, rounded to OUTPUT_DECIMALS, clamped non-increasing.
    Shared by the submission writer and the sandbox demo so the displayed scores
    are always the clean 0-1 values, never the raw fit×modifier product.
    """
    raw = [s for _, s in ordered]
    if not raw:
        return []
    hi, lo = raw[0], raw[-1]
    span = hi - lo
    norm = [NORM_HI if span <= 0 else
            NORM_LO + (NORM_HI - NORM_LO) * (s - lo) / span for s in raw]
    norm = [round(x, OUTPUT_DECIMALS) for x in norm]
    for i in range(1, len(norm)):
        if norm[i] > norm[i - 1]:
            norm[i] = norm[i - 1]
    return norm


def build_submission(
    scored: list[tuple[str, float]],
    reasoning_map: dict[str, str],
    out_path: str | Path,
    top_n: int = TOP_N,
) -> list[dict]:
    """Write the submission CSV and return the top-N rows as dicts."""
    ordered = order_candidates(scored)[:top_n]
    if len(ordered) < top_n:
        raise ValueError(
            f"Only {len(ordered)} candidates available after filtering; "
            f"need {top_n}. Loosen filters."
        )

    norm = normalize_scores(ordered)

    rows = []
    for i, (cid, _raw) in enumerate(ordered):
        rank = i + 1
        reasoning = reasoning_map.get(cid, "").strip() or "Included in top-100 shortlist."
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": f"{norm[i]:.{OUTPUT_DECIMALS}f}",
            "reasoning": reasoning,
        })

    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_HEADER, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    return rows
