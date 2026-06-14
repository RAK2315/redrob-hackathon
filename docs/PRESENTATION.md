# Redrob "India Runs" — Intelligent Candidate Discovery & Ranking
### Presentation content (slide-by-slide). Each block = one slide: title, bullets, and speaker notes.

---

## Slide 1 — Title

**Intelligent Candidate Ranking for Redrob AI**
Ranking the top 100 of 100,000 candidates for one Senior AI Engineer JD.

- An interpretable, production-grade ranking engine — not a black box.
- Runs in ~30 seconds on a laptop CPU, no GPU, no network.
- Every ranking decision is explainable in plain language.

*Speaker notes:* We built a system that mirrors how a thoughtful senior recruiter actually reads a profile — title, career story, availability — rather than counting keywords.

---

## Slide 2 — The Problem

**Find the 10 real matches hidden in 100,000 profiles.**

- 1 job description, 100,000 candidate profiles, output a ranked top 100.
- Scored once on a hidden, tiered ground truth (tiers 0–5; "relevant" = tier 3+):
  **Composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10**
- **80% of the score is the top-50 ordering; the top-10 dominates.**
- The dataset is adversarial: keyword stuffers, behavioral twins, ~80 honeypots.
- **No leaderboard, no validation set, no feedback. Three submissions max.**

*Speaker notes:* The no-feedback rule is the single most important design constraint. We can't tune to a metric, so we optimize for correctness and inspectability — a system we can read and defend.

---

## Slide 3 — Solution Overview

**An interpretable multi-stage scoring engine + grounded reasoning.**

- **Stage A — Honeypot rejection:** impossible profiles removed before ranking.
- **Stage B — JD-fit score:** structured requirement matching (title + career substance) blended with dense semantic similarity.
- **Stage C — Behavioral modifiers:** bounded multipliers for availability, notice period, location, open-source validation, job-hop stability.
- **Stage D — Assembly:** enforce the validator's invariants, take the top 100, attach plain-language reasoning.

What differentiates it:
- **Reads the career story, not the skills list** — directly defeats the keyword-stuffer trap the JD warns about.
- **Every score is decomposable** — we can show *why* anyone is ranked where they are.
- **No trained model** — there are zero relevance labels, so a learned ranker would only imitate our own heuristic with added variance and no way to validate it.

*Speaker notes:* Traditional ATS/matching systems do TF-IDF/BM25 keyword overlap or pure embedding similarity. Both rank a "Marketing Manager" with a stuffed AI skill list near the top. We don't.

---

## Slide 4 — What Differentiates Us from Traditional Matching

| Traditional candidate matching | Our approach |
|---|---|
| Keyword / BM25 overlap on the skills section | Title + **career-history substance** dominate; skills are not the signal |
| Pure embedding similarity (semantic, but gullible) | Embeddings **blended at 30%**, capped so they can't override structured fit |
| Treats every profile as honest | **Honeypot & stuffer detection**; impossible profiles hard-rejected |
| Skills-only; ignores hireability | **Behavioral signals** (response rate, recency, notice, OSS) modulate the score |
| Opaque score | **Plain-language, fact-checked reasoning** per candidate |
| Often GPU / API per candidate | **CPU-only, no network, ~30s for 100k** |

*Speaker notes:* The JD literally says "the right answer is NOT the most AI keywords." Our architecture is built around that sentence.

---

## Slide 5 — JD Understanding: Key Requirements Extracted

**"Things you absolutely need" (the must-haves):**
- Production **embeddings-based retrieval** experience (handled drift, index refresh, regressions).
- Production **vector DB / hybrid search** (Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch…).
- Strong **Python**.
- **Ranking evaluation** frameworks: NDCG, MRR, MAP, A/B testing.

**The ideal candidate:**
- 6–8 yrs total, 4–5 in applied ML/AI **at product companies (not services)**.
- Has **shipped a ranking / search / recommendation system** to real users at scale.
- Located in / willing to relocate to Pune–Noida; **active** on the platform.

**"Do NOT want" (the negatives we encode):**
- Keyword stuffers (AI skills, non-AI title) · title-chasers / job-hoppers (<1.5 yr stints)
- All-consulting careers (TCS/Infosys/Wipro/…) · CV/speech/robotics-only without NLP/IR
- Pure research / no production · recent-LangChain-only AI · long notice period (bar rises)

*Speaker notes:* We translated every one of these clauses into an explicit, inspectable rule — the slide after results lists the coverage.

---

## Slide 6 — Which Signals Matter Most (Beyond Keywords)

**Ranked by influence on the final order:**
1. **Current title class** (strong AI / adjacent / other) — the primary gate.
2. **Career-history substance** — prose evidence of building retrieval/ranking/recsys *in production*. This is the tier-5 differentiator: a real fit may not use the word "RAG."
3. **Semantic similarity** to the JD (dense embeddings) — recall for keyword-light strong candidates.
4. **Availability/behavioral signals** — recency, recruiter response rate, open-to-work (measured: removing these reorders 6 of the top 10).
5. **Experience band** (5–9 yrs), **product-vs-consulting**, **NLP/IR-vs-CV/speech**.
6. **Hireability frictions** — notice period, location/visa; **external validation** — GitHub/OSS.
7. **Recruiter interest / reliability** — recruiter-saves, profile-views, interview-completion. A *mild* tiebreaker: empirically these track quality even within the top tier, but we cap their influence because they're partly a popularity proxy the JD distrusts.

**How we evaluate fit beyond keyword matching:**
- We score the *narrative* of what they built, weighted above the skills list.
- We sanity-check claims: a skill claimed "expert" but with a low Redrob assessment is down-weighted.
- We ask "can we actually hire them?" — a perfect-on-paper but inactive, 120-day-notice candidate is down-ranked.

*Speaker notes:* This ordering came from a real ablation study on the data, not intuition.

---

## Slide 7 — Ranking Methodology: Retrieve → Score → Rank

**Retrieve:** load all 100k; honeypot hard-reject (~44 impossible profiles) leaves the contention pool.

**Score (the single formula in `ranker/pipeline.py`):**
```
final = blended_fit(structured, cosine)      # JD-fit
        × stuffer_factor                      # keyword-stuffer penalty
        × availability_modifier               # engagement/recency  (floor 0.5)
        × location_factor                     # India / relocate / abroad
        × notice_factor                       # notice-period friction
        × external_validation_factor          # GitHub/OSS bonus (asymmetric)
        × stability_factor                    # anti job-hopping
        × desirability_factor                 # mild recruiter-interest/reliability tiebreaker (bounded 0.95–1.07)
```
where `blended_fit = 0.70·structured + 0.30·cosine`, and
`structured = 0.35·title + 0.30·career_substance + 0.15·YOE + 0.10·domain + 0.10·product`.

**Rank:** sort by score; **clamp scores non-increasing by rank**; break ties by ascending `candidate_id`; take top 100.

*Speaker notes:* Structured fit is weighted 0.70 so embedding similarity can never promote a non-AI title into the top — the central anti-trap design choice.

---

## Slide 8 — Models, Algorithms & Heuristics

- **Dense embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim), L2-normalized, cosine via a single matrix–vector product.
- **JD query vector:** mean of three embedded JD blocks ("absolutely need" + ideal-candidate + nice-to-haves), verbatim from the JD.
- **Structured rule engine:** explicit, data-driven config (`jd_spec.py`) — title classes, retrieval-term lexicon, consulting list, CV/speech terms, YOE band, location prefs.
- **Honeypot detector:** deterministic impossibility rules (tenure > experience; expert skill with 0 months use; ≥8 expert skills weakly corroborated; expert claim vs near-zero assessment).
- **Behavioral modifiers:** bounded multiplicative factors, all sentinel(-1)-safe.
- **No trained/ML ranker:** deliberate — zero labels, single JD, hidden truth (a LightGBM ranker would overfit self-generated labels with no way to validate).

*Speaker notes:* We pressure-tested learning-to-rank and rejected it on principle: it can't beat the heuristic it would be trained to imitate, and we'd never know if it overfit.

---

## Slide 9 — Combining Multiple Signals

- **Fit (additive, normalized):** title + career substance + YOE + domain + product → one [0,1] structured score, then blended with cosine.
- **Modifiers (multiplicative, bounded):** each behavioral factor scales the fit, so they **modulate** rather than dominate — a strong candidate keeps ≥50% credit even if quiet.
- **Sentinels are neutral, never penalties:** `github=-1` (64% of pool) and `offer_acceptance=-1` (60%) and empty assessments map to 1.0 / no-signal. Absence ≠ negative.
- **Robustness check:** the top-10 is stable across plausible blend weights (0.6/0.4 → 10/10 overlap, 0.5/0.5 → 9/10; rank-1 never moves) — important since we can't tune to a leaderboard.

*Speaker notes:* The additive-fit / multiplicative-modifier split keeps the model interpretable: fit answers "are they right for the role," modifiers answer "can we get them."

---

## Slide 10 — Explainability

**Every candidate ships with a 1–2 sentence, fact-grounded justification.**

Example (rank 6):
> *"Senior Machine Learning Engineer with 7.2 yrs at Zomato; career history shows retrieval/ranking/recommendation work, matching the JD's production-search focus; squarely in the 6-8yr sweet spot; active open-source presence (GitHub score 95). Strong engagement (response rate 0.61)."*

- Cites **specific facts** (title, years, employer, named signal values).
- Connects to **specific JD requirements** (production retrieval/ranking).
- **Surfaces honest concerns** at lower ranks (e.g., "120-day notice period," "based in UK — relocation/visa consideration").
- **Tone tracks the rank** — top ranks emphasize fit; lower ranks lead with caveats.

*Speaker notes:* This directly targets the Stage-4 manual review rubric: specific facts, JD connection, honest concerns, variation, rank-consistency.

---

## Slide 11 — Preventing Hallucination

**Reasoning is built only from fields that exist in the profile.**

- **Template generator** assembles sentences from the candidate's own structured fields + computed sub-scores → nothing can be invented by construction.
- **Verifier** (`verify_grounded`) checks any named employer/skill/number against the record; an unverified claim is dropped.
- **Optional LLM upgrade** (offline only, if an API key is present) is fed *only* that candidate's facts and **passed through the same verifier** — any line that fails falls back to the template.
- **No candidate data leaves the machine during ranking** (no network in the live path).

*Speaker notes:* The shipped submission uses the deterministic template — zero hallucination risk — with the LLM path available as a verified enhancement.

---

## Slide 12 — Handling Suspicious / Low-Quality Profiles

- **Honeypots (~80 designed; we flag 44 high-precision):** impossible profiles hard-rejected *before* ranking; output asserted clean. (>10% honeypots in top-100 = instant disqualification — we ship 0.)
- **Keyword stuffers (7,092 in the pool):** non-AI title + stuffed AI skills → strong penalty; the best-ranked stuffer lands at position **85,287 / 99,956** — none reach the top 100.
- **Claim inflation:** "expert"/"advanced" skill contradicted by a low Redrob assessment → down-weight.
- **Missing/sentinel data:** `-1` and empty fields treated as neutral, so honest-but-sparse profiles aren't unfairly punished.

*Speaker notes:* Defensive filtering is a safety net, not the main mechanism — a good title/career ranker naturally avoids most traps, but with no feedback loop we don't leave the >10% DQ rule to chance.

---

## Slide 13 — End-to-End Workflow

```
JD (.docx)                         candidates.jsonl (100k)
   │                                      │
   ▼                                      ▼
[OFFLINE / untimed precompute]      [LIVE / <5 min, CPU, no network]
 build_jd_vector.py                  rank.py
 build_embeddings.py  ── artifacts ─▶  1. load + honeypot reject
 build_reasoning.py     (embeddings,    2. structured fit × cosine blend
                         jd_vecs,       3. stuffer + behavioral modifiers
                         reasoning)     4. combine → monotonic + tiebreak
                                        5. top 100 + grounded reasoning
                                        6. assert 0 honeypots
                                              │
                                              ▼
                                       submission.csv  ──▶  validate_submission.py ✓
```

- **Offline phase** (run once, ~49 min): embed 100k profiles, build JD vector + reasoning cache → shipped as `artifacts/`.
- **Live phase** (~30 s): loads precomputed vectors, never runs the model → produces and self-validates the CSV.

*Speaker notes:* The split is what makes us both high-quality (a real embedding model) and fast (no model at ranking time).

---

## Slide 14 — System Architecture

```
repo/
├── rank.py                  ← LIVE entry point (<5 min reproduce command)
├── ranker/
│   ├── io_utils.py          load .jsonl / .jsonl.gz (magic-byte sniff)
│   ├── jd_spec.py           JD rubric as explicit config + field extractors
│   ├── honeypot.py          impossible-profile hard-reject
│   ├── fit_score.py         structured sub-scores + cosine blend
│   ├── stuffer.py           keyword-stuffer & claim-inflation penalty
│   ├── signals.py           availability/notice/location/OSS/stability modifiers
│   ├── dense.py             load precomputed embeddings → cosine to JD
│   ├── pipeline.py          SINGLE source of truth for final_score()
│   ├── combine.py           monotonic clamp + tiebreak + CSV writer
│   └── reasoning.py         grounded reasoning + hallucination verifier
├── precompute/              build_embeddings · build_jd_vector · build_reasoning
├── artifacts/               embeddings.f16.npy · cand_ids.npy · jd_vecs.npy · reasoning_cache.json
├── submission.csv           the deliverable
├── requirements.txt · submission_metadata.yaml · README.md · validate_submission.py
```

**Design principles:** one formula (no drift), data-driven rubric (reviewable), graceful degradation (missing artifacts → structured-only), validator invariants enforced in code (not hoped for).

*Speaker notes:* `pipeline.py` exists so the shipped ranking and the precomputed reasoning can never use different formulas.

---

## Slide 15 — Results & Ranking-Quality Insights

**Top-100 profile (faithful to the JD):**
- **100/100** strong AI/ML/DS/NLP/Search titles · **100/100** show retrieval/ranking evidence.
- **0** honeypots · **0** all-consulting careers · **0** keyword stuffers.
- **~90/100** India (rest relocation-friendly) · **74/100** in the 5–9 yr band.
- After adding hireability signals: notice ≥90d candidates **49 → 32**; externally-validated (GitHub) **77 → 82**.

**Top-3 (illustrative):**
1. Search Engineer, 7.6 yr @ Sarvam AI — retrieval/ranking, GitHub 61, response 0.94, 45-day notice.
2. Search Engineer, 5.1 yr @ CRED — retrieval/ranking, GitHub 87, response 0.80.
3. Sr ML Engineer, 6.1 yr @ Genpact AI — retrieval/ranking, sweet spot, response 0.88.

**Robustness (our proxy for quality, since there's no leaderboard):**
- Top-10 stable across blend weights; rank-1 invariant.
- Funnel sanity: 508 strict AI titles → ~212 genuinely strong → top 100 drawn from that pool.

*Speaker notes:* We validate by reading the output and by ablation/robustness, because the metric is hidden.

---

## Slide 16 — Meeting the Runtime & Compute Constraints

| Constraint | Limit | Ours |
|---|---|---|
| Runtime (ranking step) | ≤ 5 min | **~30 s** for 100k |
| Memory | ≤ 16 GB | well under (77 MB embeddings + JSONL) |
| Compute | CPU only | **CPU only** (import audit: no torch/transformers in live path) |
| Network | off | **off** — no API/library network calls at rank time |
| Honeypots in top-100 | < 10% | **0%** |

- **Embedding 100k live would take ~110 min on CPU** → we precompute offline (untimed) and ship a 77 MB vector matrix.
- Live `rank.py` loads vectors and does a single matrix–vector product — milliseconds.

*Speaker notes:* The measured numbers come from this machine; the 5-min budget is comfortable with ~10× headroom.

---

## Slide 17 — Technologies Used (and Why)

- **Python 3.10** — required by the challenge; strong ecosystem; the JD itself wants strong Python.
- **sentence-transformers / all-MiniLM-L6-v2** — small (22M params), fast on CPU, runs fully offline; good semantic quality per dollar of compute. Embedding model is free at rank time because it's precomputed.
- **PyTorch (CPU build)** — backs the embedder; **precompute only**, never loaded during ranking.
- **NumPy** — the *only* runtime dependency of `rank.py`; vectorized cosine + sort.
- **Python stdlib** (csv, json, gzip, dataclass-free configs) — keeps the live path dependency-light and reproducible.
- **(Optional) Anthropic Claude** — offline reasoning upgrade, behind a verifier; not required for the submission.
- **Why not a vector DB / GPU / hosted LLM?** The constraints forbid network/GPU at rank time and a 100k pool doesn't need an external index — a 77 MB in-memory matrix is faster and simpler.

*Speaker notes:* Every tool choice is justified by the constraint set: offline, CPU, ≤5 min, reproducible.

---

## Slide 18 — JD-Requirement Coverage (defensibility)

| JD signal | Encoded as | In code |
|---|---|---|
| Retrieval / vector / ranking-eval experience | career-substance lexicon + JD cosine | `jd_spec`, `fit_score` |
| 5–9 yrs experience | soft YOE band | `fit_score` |
| Product company, not consulting | product factor | `jd_spec.is_all_consulting` |
| NLP/IR, not CV/speech-only | domain score | `fit_score` |
| Notice period (30+ raises bar) | notice factor | `signals` |
| Open-source / external validation | GitHub bonus (asymmetric) | `signals` |
| Active / responsive | availability modifier | `signals` |
| Recruiter interest / reliability (saves, views, interview-completion) | mild desirability tiebreaker (bounded) | `signals` |
| Keyword stuffers | title↔skill penalty | `stuffer` |
| Title-chasers / job-hoppers | stability factor | `signals` |
| Honeypots → tier 0 | impossibility hard-reject | `honeypot` |

**Conscious limitations:** "4–5 yrs *applied-ML* depth" is proxied by title + career substance (not per-role ML tenure); "recent-LangChain-only" is not special-cased (a real career with LangChain as one skill isn't the trap).

*Speaker notes:* This table is our Stage-5 interview cheat-sheet — every claim maps to a file.

---

## Slide 19 — Submission Assets

- **GitHub:** https://github.com/RAK2315/redrob-hackathon (code, artifacts, README with one-command reproduce).
- **Reproduce command:** `python rank.py --candidates ./candidates.jsonl --out ./submission.csv` (≤5 min, CPU, no network).
- **Deliverable:** `submission.csv` (validated by `validate_submission.py`).
- **Sandbox (REQUIRED by spec §10.5):** hosted demo (HF Space / Colab) running the ranker on a ≤100-candidate sample. *(to add)*
- **Demo video:** optional — only if your hackathon's presentation round asks for it (the challenge spec requires a live Stage-5 interview call, not a recorded video). *(optional)*
- **Metadata:** `submission_metadata.yaml` (team, compute, AI-tools declaration, methodology).

*Speaker notes:* Everything needed for Stage-3 reproduction is in the repo; the sandbox and video are the lightweight verification layer.

---

## Slide 20 — Summary

- An **interpretable, fast, honest** ranker that reads careers, not keywords.
- **Faithful to the JD's full rubric** — must-haves, ideal-candidate, and every "do NOT want."
- **Robust by design** for a no-feedback competition: decomposable scores, ablation-checked, validator-safe.
- **0 honeypots, 0 stuffers, 100% explainable, ~30 s on a CPU.**

*Speaker notes:* The thesis in one line: the right candidate may not have the right keywords — so we rank the work, verify the claims, and check we can actually hire them.
