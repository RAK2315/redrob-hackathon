"""JD requirements encoded as explicit, inspectable config.

The core thesis from the JD (verbatim): the right answer is NOT "most AI
keywords". The signal lives in the CURRENT TITLE and the CAREER-HISTORY
SUBSTANCE, not the skills list. A "Marketing Manager" with a perfect AI skill
list is not a fit; a candidate who built a recommendation system at a product
company is a fit even without buzzwords.

Everything here is data, not logic, so a reviewer can read and challenge the
rubric directly. Scoring logic lives in fit_score.py.
"""

from __future__ import annotations

# --- Title classes -------------------------------------------------------
# Strong: directly the role family the JD hires for.
AI_TITLES_STRONG = {
    "ml engineer", "machine learning engineer", "senior machine learning engineer",
    "junior ml engineer", "ai engineer", "data scientist", "applied scientist",
    "applied ml engineer", "research engineer", "nlp engineer", "mlops engineer",
    "search engineer", "search relevance engineer", "research scientist",
}
# Adjacent: plausible transfer (the JD's "data-infra hybrid building toward ML").
ADJACENT_TITLES = {
    "software engineer", "senior software engineer", "backend engineer",
    "full stack developer", "data engineer", "analytics engineer",
    "platform engineer", "devops engineer", "cloud engineer",
}
# Everything else (Marketing Manager, HR Manager, Accountant, ...) = non-tech.

# --- Career-substance evidence (matched in descriptions + summary) -------
# Presence of these IN PROSE is the tier-5 differentiator: it shows the
# candidate actually built retrieval/ranking/recsys/search systems.
RETRIEVAL_TERMS = [
    "retrieval", "ranking", "rank ", "re-rank", "rerank", "recommendation",
    "recommender", "recsys", "search", "semantic search", "information retrieval",
    "embedding", "embeddings", "vector", "vector database", "vector search",
    "bm25", "elasticsearch", "opensearch", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "ndcg", "mrr", "mean average precision", "learning to rank",
    "learning-to-rank", "ltr", "nearest neighbor", "ann ",
]
# Production / scale signals (the JD wants shipped-to-real-users, not research).
PRODUCTION_TERMS = [
    "production", "deployed", "deploy", "real-time", "real time", "at scale",
    "scale", "latency", "throughput", "a/b test", "ab test", "users", "pipeline",
    "served", "serving", "online", "p99", "qps",
]
# NLP / IR positive domain anchors.
NLP_TERMS = [
    "nlp", "natural language", "language model", "llm", "transformer", "bert",
    "text", "retrieval", "search", "ranking", "information retrieval", "rag",
    "fine-tune", "fine-tuning", "embedding", "semantic",
]
# CV/speech/robotics: JD explicitly does NOT want these as PRIMARY expertise
# absent NLP/IR exposure ("you'd be re-learning fundamentals here").
CV_SPEECH_TERMS = [
    "computer vision", "image classification", "object detection", "segmentation",
    "speech recognition", "speech-to-text", "text-to-speech", "tts", "asr",
    "robotics", "gan", "gans", "image generation", "diffusion", "ocr",
    "pose estimation", "video",
]

# --- Employers -----------------------------------------------------------
# JD: candidates whose ENTIRE career is at these is a soft-DQ. Substring match.
CONSULTING = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "ltimindtree", "ibm",
    "deloitte", "pwc", "kpmg", "ernst", "mphasis", "hexaware", "birlasoft",
    "persistent systems",
}

# --- Experience band -----------------------------------------------------
YOE_MIN, YOE_MAX = 5.0, 9.0          # JD "5-9 years" — soft, not a hard filter.
YOE_IDEAL_LO, YOE_IDEAL_HI = 6.0, 8.0  # JD "ideal: 6-8 years".

# --- Location ------------------------------------------------------------
PREFERRED_COUNTRY = "India"
PREFERRED_CITIES = {  # JD: Pune/Noida preferred; Hyderabad/Mumbai/Delhi NCR welcome.
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
    "bangalore", "bengaluru", "ncr", "new delhi",
}

# --- Structured fit sub-weights (sum to 1.0) -----------------------------
# Reported as uncertain knobs; chosen to be defensible mid-range, with title +
# career-substance dominating (the JD's explicit signal).
FIT_SUBWEIGHTS = {
    "title": 0.35,
    "career_substance": 0.30,
    "yoe": 0.15,
    "domain": 0.10,        # NLP/IR vs CV/speech-only
    "product": 0.10,       # product company vs all-consulting
}

# Blend of structured fit vs dense cosine (structured dominates so a non-AI
# title cannot be promoted by embedding similarity alone).
STRUCTURED_WEIGHT = 0.70
COSINE_WEIGHT = 0.30


# --- Shared field extraction helpers (used across modules) --------------

def title_class(candidate: dict) -> str:
    """Return 'strong' | 'adjacent' | 'other' for the current title."""
    t = (candidate.get("profile", {}).get("current_title") or "").strip().lower()
    if t in AI_TITLES_STRONG:
        return "strong"
    if t in ADJACENT_TITLES:
        return "adjacent"
    return "other"


def candidate_text(candidate: dict) -> str:
    """Concatenated prose used for term matching and embeddings: headline,
    summary, current title, career-history descriptions, and skill names."""
    p = candidate.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    for h in candidate.get("career_history", []):
        parts.append(h.get("title", ""))
        parts.append(h.get("description", ""))
    parts.append(" ".join(s.get("name", "") for s in candidate.get("skills", [])))
    return " ".join(x for x in parts if x).lower()


def career_text(candidate: dict) -> str:
    """Only the career-history descriptions + summary (substance, not skills)."""
    p = candidate.get("profile", {})
    parts = [p.get("summary", "")]
    for h in candidate.get("career_history", []):
        parts.append(h.get("description", ""))
    return " ".join(x for x in parts if x).lower()


def count_terms(text: str, terms) -> int:
    return sum(1 for term in terms if term in text)


def is_all_consulting(candidate: dict) -> bool:
    hist = candidate.get("career_history", [])
    if not hist:
        return False
    def consulting(co: str) -> bool:
        c = (co or "").lower()
        return any(k in c for k in CONSULTING)
    return all(consulting(h.get("company", "")) for h in hist)
