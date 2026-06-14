"""Structured JD-fit scoring + optional dense-cosine blend.

Returns interpretable sub-scores (each in [0, 1]) so a reviewer can read WHY a
candidate scored as they did, and so the reasoning module can cite the same
facts. The structured score dominates the blend (jd_spec.STRUCTURED_WEIGHT) so a
non-AI title cannot be promoted into the top ranks by embedding similarity
alone — the central trap the JD warns about.
"""

from __future__ import annotations

from . import jd_spec


def _yoe_score(yoe: float) -> float:
    """1.0 inside the ideal 6-8 band; high across 5-9; decays outside."""
    if yoe <= 0:
        return 0.0
    lo, hi = jd_spec.YOE_IDEAL_LO, jd_spec.YOE_IDEAL_HI       # 6, 8
    blo, bhi = jd_spec.YOE_MIN, jd_spec.YOE_MAX               # 5, 9
    if lo <= yoe <= hi:
        return 1.0
    if blo <= yoe < lo:                # 5-6: ramp 0.85 -> 1.0
        return 0.85 + 0.15 * (yoe - blo) / (lo - blo)
    if hi < yoe <= bhi:                # 8-9: ramp 1.0 -> 0.85
        return 0.85 + 0.15 * (bhi - yoe) / (bhi - hi)
    if yoe < blo:                      # below band: decay to ~0.3 at 0
        return max(0.3, 0.85 * yoe / blo)
    # above band: gentle decay (JD: seniority judgment may still fit)
    return max(0.3, 0.85 - 0.05 * (yoe - bhi))


def _career_substance_score(candidate: dict) -> float:
    """Prose evidence of building retrieval/ranking/recsys systems IN
    PRODUCTION — the tier-5 differentiator, independent of the skills list."""
    text = jd_spec.career_text(candidate)
    retr = jd_spec.count_terms(text, jd_spec.RETRIEVAL_TERMS)
    prod = jd_spec.count_terms(text, jd_spec.PRODUCTION_TERMS)
    retr_s = min(retr / 3.0, 1.0)      # 3+ distinct retrieval terms -> full
    prod_s = min(prod / 3.0, 1.0)
    return 0.70 * retr_s + 0.30 * prod_s


def _domain_score(candidate: dict) -> float:
    """NLP/IR positive; CV/speech/robotics-dominant-without-NLP penalized."""
    text = jd_spec.candidate_text(candidate)
    nlp = jd_spec.count_terms(text, jd_spec.NLP_TERMS)
    cv = jd_spec.count_terms(text, jd_spec.CV_SPEECH_TERMS)
    if nlp >= 1 and nlp >= cv:
        return 1.0
    if cv > nlp and cv >= 2:           # CV/speech is the primary expertise
        return 0.20
    return 0.50                         # neutral / unclear


def _product_score(candidate: dict) -> float:
    """0 if the entire career is at consulting/services firms (JD soft-DQ)."""
    return 0.0 if jd_spec.is_all_consulting(candidate) else 1.0


def fit_components(candidate: dict) -> dict:
    """All structured sub-scores plus the weighted structured total."""
    tc = jd_spec.title_class(candidate)
    title = {"strong": 1.0, "adjacent": 0.5, "other": 0.0}[tc]
    comps = {
        "title": title,
        "career_substance": _career_substance_score(candidate),
        "yoe": _yoe_score(candidate.get("profile", {}).get("years_of_experience", 0) or 0),
        "domain": _domain_score(candidate),
        "product": _product_score(candidate),
    }
    w = jd_spec.FIT_SUBWEIGHTS
    comps["structured"] = sum(comps[k] * w[k] for k in w)
    comps["title_class"] = tc
    return comps


def structured_fit(candidate: dict) -> float:
    return fit_components(candidate)["structured"]


def blended_fit(structured: float, cosine: float) -> float:
    """Combine structured fit with dense cosine similarity to the JD."""
    return (jd_spec.STRUCTURED_WEIGHT * structured
            + jd_spec.COSINE_WEIGHT * cosine)
