"""Redrob Ranker — HuggingFace Space demo.

Runs the exact ranking engine (ranker/) on a <=100-candidate sample and shows the
ranked output with grounded reasoning, plus the honeypots it auto-rejected. A
reviewer can also upload their own small .jsonl. Numpy-only at inference time —
embeddings are precomputed (artifacts/), so no model runs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

from ranker.fit_score import fit_components
from ranker.pipeline import final_score
from ranker.honeypot import is_honeypot, honeypot_reasons
from ranker.dense import load_cosine_map
from ranker.combine import order_candidates
from ranker.reasoning import template_reasoning

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "sample_100.jsonl"
COSINE = load_cosine_map(HERE / "artifacts") or {}


def _load(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rank(candidates: list[dict]):
    by_id = {c["candidate_id"]: c for c in candidates}
    rejected = [(c["candidate_id"], "; ".join(honeypot_reasons(c)))
                for c in candidates if is_honeypot(c)]
    scored, comps = [], {}
    for c in candidates:
        if is_honeypot(c):
            continue
        fc = fit_components(c)
        scored.append((c["candidate_id"], final_score(c, fc, COSINE.get(c["candidate_id"], 0.0))))
        comps[c["candidate_id"]] = fc

    ordered = order_candidates(scored)[:100]
    rows = []
    for rk, (cid, score) in enumerate(ordered, 1):
        c = by_id[cid]
        rows.append({
            "rank": rk,
            "candidate_id": cid,
            "title": c["profile"]["current_title"],
            "yoe": c["profile"]["years_of_experience"],
            "score": round(score, 4),
            "reasoning": template_reasoning(c, comps[cid], rk),
        })
    ranked_df = pd.DataFrame(rows)
    rej_df = pd.DataFrame(rejected, columns=["candidate_id", "honeypot_reason"]) \
        if rejected else pd.DataFrame(columns=["candidate_id", "honeypot_reason"])
    summary = (f"Ranked {len(rows)} candidates. "
               f"Auto-rejected {len(rejected)} impossible/honeypot profiles before ranking.")
    return summary, ranked_df, rej_df


def run_sample():
    return rank(_load(SAMPLE))


def run_upload(file):
    if file is None:
        return "Upload a .jsonl with up to 100 candidate records (schema as in candidate_schema.json).", None, None
    return rank(_load(Path(file.name)))


with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown(
        "# Redrob Candidate Ranker — live demo\n"
        "Interpretable multi-stage ranker for the Senior AI Engineer JD. "
        "Ranks by **title + career substance** (not keyword count), blends dense "
        "similarity, applies behavioral/availability modifiers, and **hard-rejects "
        "impossible/honeypot profiles**. Reasoning is grounded in each profile.\n\n"
        "Repo: https://github.com/RAK2315/redrob-hackathon"
    )
    with gr.Row():
        btn = gr.Button("Run on bundled 100-candidate sample", variant="primary")
        up = gr.File(label="…or upload your own .jsonl (<=100)", file_types=[".jsonl"])
    status = gr.Markdown()
    gr.Markdown("### Ranked candidates")
    out = gr.Dataframe(wrap=True)
    gr.Markdown("### Auto-rejected (honeypots / impossible profiles)")
    rej = gr.Dataframe(wrap=True)

    btn.click(run_sample, outputs=[status, out, rej])
    up.change(run_upload, inputs=up, outputs=[status, out, rej])

if __name__ == "__main__":
    demo.launch()
