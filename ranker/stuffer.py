"""Keyword-stuffer detection -> multiplicative penalty.

The dataset's largest trap (~4,280 profiles, 4.3%): a non-AI title (Marketing
Manager, Accountant, ...) padded with a long list of AI skills. The JD is
explicit that these are NOT fits "no matter how perfect the skill list looks".

Two penalties:
  1. AI-skills-with-non-AI-title: strong down-weight, so embedding similarity to
     the JD (driven by the stuffed skill names) can't promote a non-AI title.
  2. claim-vs-assessment contradiction: a skill claimed expert/advanced but with
     a low Redrob assessment score is unreliable -> mild down-weight.

Returns a factor in (0, 1]; 1.0 means no penalty.
"""

from __future__ import annotations

from . import jd_spec

# Canonical AI/ML/IR skill names (lowercased) used to detect stuffing.
AI_SKILL_NAMES = {
    "nlp", "natural language processing", "fine-tuning llms", "lora", "qlora",
    "peft", "rag", "embeddings", "sentence-transformers", "vector database",
    "vector search", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "elasticsearch", "opensearch", "bm25", "learning-to-rank", "xgboost",
    "transformers", "pytorch", "tensorflow", "bert", "llm", "llms", "retrieval",
    "ranking", "recommendation systems", "information retrieval",
    "semantic search", "hugging face", "huggingface", "image classification",
    "speech recognition", "tts", "gans", "statistical modeling", "mlops",
    "bge", "e5", "deep learning", "machine learning", "computer vision",
    "fine-tuning", "llm fine-tuning", "prompt engineering", "langchain",
}

STUFFER_FACTOR = 0.30        # non-AI title + many AI skills
CONTRADICTION_FACTOR = 0.85  # expert/advanced claim vs low assessment
MIN_AI_SKILLS_FOR_STUFFER = 4


def _n_ai_skills(candidate: dict) -> int:
    return sum(1 for s in candidate.get("skills", [])
               if (s.get("name", "") or "").strip().lower() in AI_SKILL_NAMES)


def stuffer_factor(candidate: dict) -> float:
    factor = 1.0
    tclass = jd_spec.title_class(candidate)

    # 1. Stuffed AI skills under a non-AI/non-adjacent title.
    if tclass == "other" and _n_ai_skills(candidate) >= MIN_AI_SKILLS_FOR_STUFFER:
        factor *= STUFFER_FACTOR

    # 2. Claimed expertise contradicted by a low platform assessment.
    assess = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    contradicted = False
    for s in candidate.get("skills", []):
        if s.get("proficiency") in ("expert", "advanced"):
            sc = assess.get(s.get("name"))
            if sc is not None and sc < 40:
                contradicted = True
                break
    if contradicted:
        factor *= CONTRADICTION_FACTOR

    return factor
