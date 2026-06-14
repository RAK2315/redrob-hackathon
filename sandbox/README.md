---
title: Redrob Candidate Ranker
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Redrob Candidate Ranker — Sandbox

Live demo of the interpretable multi-stage candidate ranker for the Redrob
"India Runs" challenge (Senior AI Engineer JD).

- Click **Run on bundled 100-candidate sample** to rank a curated slice (strong
  fits + keyword stuffers + honeypots) and watch the engine separate them.
- Or upload your own `.jsonl` (≤100 records, schema per `candidate_schema.json`).

Numpy-only at inference time — embeddings are precomputed (`artifacts/`), so no
model runs in the Space. Full code & methodology:
https://github.com/RAK2315/redrob-hackathon
