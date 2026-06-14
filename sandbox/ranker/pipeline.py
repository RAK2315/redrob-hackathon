"""Single source of truth for the final combined score.

Everything that needs a candidate's final score (rank.py, build_reasoning.py,
audits) calls final_score() here, so the formula can never drift between the
shipped ranking and the precomputed reasoning shortlist.

  final = blended_fit(structured, cosine)         # JD-fit (title + career substance + dense)
          x stuffer_factor                         # keyword-stuffer penalty
          x availability_modifier                  # engagement/recency (bounded, floor 0.5)
          x location_factor                        # India vs relocate vs abroad
          x notice_factor                          # notice-period friction (JD: 30+ raises bar)
          x external_validation_factor             # GitHub/OSS bonus (asymmetric, sentinel-safe)
          x stability_factor                        # anti job-hopping (JD: no title-chasers)
          x desirability_factor                     # mild recruiter-interest/reliability tiebreaker
"""

from __future__ import annotations

from . import fit_score, stuffer, signals


def final_score(candidate: dict, fc: dict, cosine: float) -> float:
    blended = fit_score.blended_fit(fc["structured"], cosine)
    return (
        blended
        * stuffer.stuffer_factor(candidate)
        * signals.availability_modifier(candidate)
        * signals.location_factor(candidate)
        * signals.notice_factor(candidate)
        * signals.external_validation_factor(candidate)
        * signals.stability_factor(candidate)
        * signals.desirability_factor(candidate)
    )
