# Redrob Candidate Ranker

Ranks the top 100 of 100,000 candidates for the Senior AI Engineer JD and writes
`submission.csv` (`candidate_id,rank,score,reasoning`).

Interpretable multi-stage scoring engine (no trained model — there are no
relevance labels). Designed for **correctness and inspectability**: there is no
leaderboard, so every score is decomposable and the top-50 is meant to be read
by a human.

## Data

`candidates.jsonl` (100k profiles, ~465 MB) is **not** committed — download it from
the challenge bundle and place it at the repo root (or pass `--candidates PATH`).
The precomputed `artifacts/` (embeddings keyed by `candidate_id`) **are** committed,
so the ranking step runs without rebuilding them.

## Reproduce the submission (the ranking step)

```bash
# Needs only numpy + the precomputed artifacts/ (shipped in this repo).
pip install numpy
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv      # -> "Submission is valid."
```

**Time guard:** the ranking step runs on CPU, no GPU, **no network**, and
completes in **under a minute (~30–50 s measured)** for the full 100k pool — well
inside the 5-minute / 16 GB budget. It loads precomputed embeddings; it never runs
the embedding model.

## Rebuild the precomputed artifacts (offline, untimed)

```bash
pip install -r requirements.txt
# ~50 min on CPU; embeds all 100k profiles with all-MiniLM-L6-v2 (cached locally).
python precompute/build_embeddings.py --max-seq-length 128
python precompute/build_jd_vector.py
python precompute/build_reasoning.py     # template reasoning; LLM-upgraded if ANTHROPIC_API_KEY set
```

Artifacts written to `artifacts/`: `embeddings.f16.npy` (~77 MB), `cand_ids.npy`,
`jd_vecs.npy`, `reasoning_cache.json`.

## Architecture

```
rank.py                     # live entry point (<5 min)
ranker/
  io_utils.py   # load .jsonl / .jsonl.gz transparently
  jd_spec.py    # JD rubric as explicit config (title classes, retrieval terms, ...)
  honeypot.py   # impossible-profile hard-reject (DQ insurance)
  fit_score.py  # structured JD-fit sub-scores + dense-cosine blend
  stuffer.py    # keyword-stuffer penalty (AI skills under non-AI title)
  signals.py    # bounded behavioral/availability modifier (sentinel-safe)
  combine.py    # monotonic-score + candidate_id tie-break + CSV writer
  reasoning.py  # grounded reasoning + hallucination verifier
  dense.py      # load precomputed embeddings, cosine to JD
precompute/     # offline artifact builders (untimed)
artifacts/      # shipped precomputed state
```

Pipeline (single formula in `ranker/pipeline.py`): honeypot reject → structured
fit (title + career substance dominate) → 0.70·structured + 0.30·cosine → stuffer
penalty → bounded availability modifier × location factor × notice-period factor ×
GitHub/OSS external-validation bonus × job-hop stability factor × mild desirability
factor (recruiter-saves/views/interview-completion) → sort with validator invariants
→ top 100 (scores normalized to 0–1) → grounded reasoning → assert 0 honeypots in output.

Every JD signal is encoded: "absolutely need" (retrieval/vector/ranking-eval →
career substance + cosine), 5–9 yrs, product-not-consulting, NLP/IR-not-CV/speech,
notice period ("30+ raises the bar"), open-source/external validation, and the
"do NOT want" set (keyword-stuffers, title-chasers/job-hoppers, all-consulting,
CV/speech-only, honeypots). Deliberate limitations: "4–5 yrs *applied ML* depth"
is proxied by current title + career substance rather than per-role ML tenure;
"recent-LangChain-only" is not special-cased (a real career with LangChain as one
skill is not the trap).

### Why these choices
- **No trained ranker:** single JD, hidden tiered ground truth, zero labels — a
  learned model would only imitate our own heuristic with added variance and no
  way to validate (no feedback loop).
- **Title + career substance dominate the skills list:** the JD is explicit that
  "most AI keywords" is the wrong answer; a non-AI title with stuffed AI skills is
  not a fit. The structured score (weight 0.70) prevents cosine from promoting
  stuffers.
- **Honeypot hard-reject:** >10% honeypots in the top 100 is an instant DQ;
  ~44 impossible profiles are filtered before ranking and the output is asserted
  clean.
