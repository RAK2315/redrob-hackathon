"""Honeypot / impossible-profile detection.

The dataset seeds ~80 honeypots with subtly impossible profiles (e.g. 8 years
tenure at a company that can't support it; "expert" in many skills with 0 months
of use). They are forced to relevance tier 0 in the ground truth, and a
submission with >10% honeypots in the top 100 is DISQUALIFIED at Stage 3.

Strategy: hard-reject any candidate matching a high-precision impossibility
rule BEFORE ranking. These rules are deterministic and cheap. We deliberately
keep them tight (impossibility, not mere "unusual") to avoid rejecting genuine
strong candidates. Measured on the full 100k pool: ~107 candidates flagged.
"""

from __future__ import annotations


def honeypot_reasons(candidate: dict) -> list[str]:
    """Return a list of impossibility reasons (empty == not a honeypot)."""
    reasons: list[str] = []
    prof = candidate.get("profile", {})
    yoe = prof.get("years_of_experience", 0) or 0
    skills = candidate.get("skills", [])
    sig = candidate.get("redrob_signals", {}) or {}
    assess = sig.get("skill_assessment_scores", {}) or {}

    # 1. Total claimed tenure across roles exceeds plausible career length.
    total_months = sum(h.get("duration_months", 0) or 0
                       for h in candidate.get("career_history", []))
    if total_months > yoe * 12 + 24:
        reasons.append(
            f"career tenure {total_months}mo >> {yoe}yr experience"
        )

    # 2. A single role longer than the whole stated career (+grace).
    for h in candidate.get("career_history", []):
        if (h.get("duration_months", 0) or 0) > yoe * 12 + 18:
            reasons.append("single role longer than total experience")
            break

    # 3. Expert/advanced proficiency in a skill used for 0 months.
    for s in skills:
        if s.get("proficiency") in ("expert", "advanced") and \
                (s.get("duration_months", 1) or 0) == 0:
            reasons.append(
                f"{s.get('proficiency')} in '{s.get('name')}' with 0 months use"
            )
            break

    # 4. Implausible breadth: many expert skills, paired with weak corroboration
    #    (short durations or low assessment scores) -> fabricated profile.
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 8:
        median_dur = sorted((s.get("duration_months", 0) or 0)
                            for s in expert_skills)[len(expert_skills) // 2]
        low_assess = any(
            assess.get(s.get("name"), 100) < 20 for s in expert_skills
        )
        if median_dur < 18 or low_assess:
            reasons.append(f"{len(expert_skills)} expert skills, weakly corroborated")

    # 5. Expert proficiency directly contradicted by a near-zero assessment.
    for s in skills:
        if s.get("proficiency") == "expert":
            sc = assess.get(s.get("name"))
            if sc is not None and sc < 15:
                reasons.append(
                    f"expert '{s.get('name')}' but assessment {sc:.0f}/100"
                )
                break

    return reasons


def is_honeypot(candidate: dict) -> bool:
    return len(honeypot_reasons(candidate)) > 0


def find_honeypots(candidates: list[dict]) -> set[str]:
    """Return the set of candidate_ids flagged as impossible."""
    return {c["candidate_id"] for c in candidates if is_honeypot(c)}
