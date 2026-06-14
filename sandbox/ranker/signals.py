"""Behavioral / availability modifier from redrob_signals.

The JD: "a perfect-on-paper candidate who hasn't logged in for 6 months and has
a 5% recruiter response rate is, for hiring purposes, not actually available.
Down-weight them." So this is a BOUNDED MULTIPLIER on fit, never a filter, with
a floor so a strong-but-quiet candidate isn't annihilated.

Critical sentinel handling: github_activity_score == -1 (64.6% of pool) and
offer_acceptance_rate == -1 (59.6%) and empty skill_assessment_scores (77%) mean
"no data" -> treated as NEUTRAL, never as a low value. Absence is no-signal, not
a penalty.

Returns:
  availability_modifier(c) -> float in [FLOOR, CEIL]
  location_factor(c)       -> float in (0, 1]
"""

from __future__ import annotations

from datetime import date

from . import jd_spec

# Reference "today" for recency (the challenge's stated current date).
TODAY = date(2026, 6, 13)

FLOOR = 0.50   # strong-but-quiet candidate keeps at least half credit
CEIL = 1.10    # highly-engaged candidate gets a small boost

# Availability sub-weights (sum to 1.0).
AVAIL_WEIGHTS = {
    "recency": 0.40,
    "response": 0.30,
    "open_to_work": 0.20,
    "verified": 0.10,
}

LOCATION_NON_INDIA = 0.95  # JD: outside India case-by-case, no visa sponsorship


def _parse_date(s):
    if not s:
        return None
    try:
        y, m, d = (int(x) for x in s[:10].split("-"))
        return date(y, m, d)
    except (ValueError, AttributeError):
        return None


def _recency_score(sig: dict) -> float:
    la = _parse_date(sig.get("last_active_date"))
    if la is None:
        return 0.5  # unknown -> neutral
    days = (TODAY - la).days
    if days <= 30:
        return 1.0
    if days <= 90:
        return 0.85
    if days <= 180:
        return 0.65
    if days <= 365:
        return 0.40
    return 0.20


def availability_modifier(candidate: dict) -> float:
    sig = candidate.get("redrob_signals", {}) or {}

    recency = _recency_score(sig)

    rr = sig.get("recruiter_response_rate")
    response = rr if isinstance(rr, (int, float)) and rr >= 0 else 0.5  # neutral if missing

    open_to_work = 1.0 if sig.get("open_to_work_flag") else 0.0

    ve = sig.get("verified_email")
    vp = sig.get("verified_phone")
    verified = ((1.0 if ve else 0.0) + (1.0 if vp else 0.0)) / 2.0

    w = AVAIL_WEIGHTS
    a = (w["recency"] * recency + w["response"] * response
         + w["open_to_work"] * open_to_work + w["verified"] * verified)

    # Map availability score a in [0,1] -> [FLOOR, CEIL].
    return FLOOR + (CEIL - FLOOR) * a


def location_factor(candidate: dict) -> float:
    p = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {}) or {}
    country = (p.get("country") or "").strip()
    if country == jd_spec.PREFERRED_COUNTRY:
        return 1.0
    # Outside India: soft penalty. Willingness to relocate softens it slightly,
    # but visa reality (JD: no sponsorship) keeps it a real penalty.
    if sig.get("willing_to_relocate"):
        return (LOCATION_NON_INDIA + 1.0) / 2.0  # ~0.975
    return LOCATION_NON_INDIA


def notice_factor(candidate: dict) -> float:
    """JD: 'sub-30-day notice ideal; we can buy out up to 30 days; 30+ day notice
    candidates are still in scope but the bar gets higher.' Mild, bounded penalty
    so a long notice nudges rather than buries an otherwise strong candidate."""
    n = (candidate.get("redrob_signals", {}) or {}).get("notice_period_days")
    if n is None:
        return 1.0
    if n <= 30:
        return 1.00
    if n <= 60:
        return 0.98
    if n <= 90:
        return 0.95
    return 0.90


def external_validation_factor(candidate: dict) -> float:
    """JD lists open-source contributions as a plus and 'no external validation
    (papers, talks, open-source)' as a negative. github_activity_score is our
    only observable proxy. ASYMMETRIC: high activity earns a small bonus; the
    sentinel -1 (no GitHub linked, 64% of pool) is NEUTRAL, never a penalty —
    absence of a linked GitHub is not evidence of closed-source-only work."""
    g = (candidate.get("redrob_signals", {}) or {}).get("github_activity_score", -1)
    if g is None or g < 0:
        return 1.0
    return 1.0 + 0.05 * min(g, 100) / 100.0   # up to +5%


def stability_factor(candidate: dict) -> float:
    """JD explicitly does NOT want title-chasers 'switching companies every 1.5
    years'. Mild penalty for clear job-hopping (avg tenure < 18 months)."""
    tenures = [h.get("duration_months", 0) or 0
               for h in candidate.get("career_history", [])]
    if len(tenures) < 2:
        return 1.0
    avg = sum(tenures) / len(tenures)
    short = sum(1 for x in tenures if 0 < x < 18)
    if avg < 18 and short >= 3:
        return 0.90
    if avg < 18:
        return 0.95
    return 1.0


# Desirability / reliability bonus, kept MILD on purpose.
# Empirically these three signals track candidate quality (their correlation with
# career substance, even WITHIN the strong-title pool, exceeds recruiter_response
# _rate) and are not redundant with our other signals. But saves/views are partly
# a *popularity* proxy and the JD distrusts "perfect-on-paper" signal-chasing, so
# this is a gentle within-tier tiebreaker, never a top-10 gamble.
DESIRABILITY_LO, DESIRABILITY_HI = 0.95, 1.07
DESIRABILITY_WEIGHTS = {"saved": 0.45, "views": 0.30, "interview": 0.25}


def desirability_factor(candidate: dict) -> float:
    sig = candidate.get("redrob_signals", {}) or {}
    saved = sig.get("saved_by_recruiters_30d")
    views = sig.get("profile_views_received_30d")
    icr = sig.get("interview_completion_rate")

    saved_n = min((saved or 0) / 40.0, 1.0)
    views_n = min((views or 0) / 150.0, 1.0)
    # interview_completion ranges ~0.3-1.0; missing -> neutral, never a penalty.
    icr_n = 0.5 if not isinstance(icr, (int, float)) else max(0.0, min((icr - 0.3) / 0.7, 1.0))

    w = DESIRABILITY_WEIGHTS
    d = w["saved"] * saved_n + w["views"] * views_n + w["interview"] * icr_n
    return DESIRABILITY_LO + (DESIRABILITY_HI - DESIRABILITY_LO) * d
