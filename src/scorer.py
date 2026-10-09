"""
Eligibility & Scoring Module (Phase 2 — Pure Python, No LLM)
=============================================================
Takes the structured CandidateInfo from Phase 1 and:
  1. Applies hard eligibility rules (Python evidence + AI evidence)
  2. Assigns deterministic base scores using keyword matching and project analysis
  3. Returns a ScoredCandidate ready for Phase 3 (LLM review & penalty adjustment)
"""

from src.models import (
    CandidateInfo,
    ScoreBreakdown,
)
from config import PYTHON_KEYWORDS, AI_KEYWORDS, SCORING_WEIGHTS
from pydantic import BaseModel, Field
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Phase 2 Output Model
# ──────────────────────────────────────────────────────────────

class EligibilityVerdict(BaseModel):
    eligible: bool
    python_evidence: list[str] = Field(default_factory=list)
    ai_evidence: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)


class ScoredCandidate(BaseModel):
    """Output of Phase 2: deterministic scoring. Sent to Phase 3 for LLM review."""
    source_file: str
    candidate_info: CandidateInfo
    eligibility: EligibilityVerdict
    base_score_breakdown: ScoreBreakdown
    base_total_score: float
    matched_skills: list[str] = Field(default_factory=list)
    scoring_rationale: list[str] = Field(default_factory=list)  # human-readable log of why points were given
    github_summary: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# 1. Eligibility Check (Hard Filter)
# ──────────────────────────────────────────────────────────────

def check_eligibility(info: CandidateInfo) -> EligibilityVerdict:
    """
    Rule-based hard filter. A candidate MUST have:
      - Python evidence (in skills, project tech, or work tech)
      - AI/Agentic evidence (at least one AI project or framework)
    """
    # Collect all text sources to search through
    all_skills_lower = [s.lower() for s in info.skills]

    # Also collect technologies from projects and work experience
    project_techs = []
    for p in info.projects:
        project_techs.extend([t.lower() for t in p.technologies])

    work_techs = []
    for w in info.work_experience:
        work_techs.extend([t.lower() for t in w.technologies])

    all_evidence_pool = set(all_skills_lower + project_techs + work_techs)

    # --- Python Evidence ---
    python_evidence = []
    for kw in PYTHON_KEYWORDS:
        for item in all_evidence_pool:
            if kw in item:
                python_evidence.append(item)
                break
    python_evidence = list(set(python_evidence))

    # --- AI/Agentic Evidence ---
    ai_evidence = []
    for kw in AI_KEYWORDS:
        for item in all_evidence_pool:
            if kw in item:
                ai_evidence.append(item)
                break
    # Also check project-level is_ai_related flag from Phase 1 LLM
    for p in info.projects:
        if p.is_ai_related:
            ai_evidence.append(f"Project: {p.title}")
    ai_evidence = list(set(ai_evidence))

    # --- Verdict ---
    rejection_reasons = []
    if not python_evidence:
        rejection_reasons.append("No evidence of Python stack in skills, projects, or work experience")
    if not ai_evidence:
        rejection_reasons.append("No AI/agentic/LLM/RAG project or framework evidence found")

    return EligibilityVerdict(
        eligible=len(rejection_reasons) == 0,
        python_evidence=python_evidence,
        ai_evidence=ai_evidence,
        rejection_reasons=rejection_reasons,
    )


# ──────────────────────────────────────────────────────────────
# 2. Deterministic Base Scoring (Python Script)
# ──────────────────────────────────────────────────────────────

def compute_base_scores(info: CandidateInfo, github_data: dict = None) -> tuple[ScoreBreakdown, list[str]]:
    """
    Assigns deterministic base scores using keyword matching and project analysis.
    Returns (ScoreBreakdown, rationale_log).
    
    This is the "generous" first pass. Phase 3 (LLM) will deduct penalties.
    """
    github_data = github_data or {}
    rationale = []

    # ── AI / Agentic / RAG Project Depth (max 40) ──
    ai_score = 0.0
    ai_projects = [p for p in info.projects if p.is_ai_related]

    if ai_projects:
        for p in ai_projects:
            ai_score += 20.0
            rationale.append(f"+20 AI project found: '{p.title}' (Phase 3 will review for thin wrapper penalties)")
    ai_score = min(ai_score, 40.0)

    # ── Python & Backend Engineering (max 30) ──
    python_score = 0.0
    all_skills_lower = {s.lower() for s in info.skills}
    all_techs = set()
    for p in info.projects:
        all_techs.update(t.lower() for t in p.technologies)
    for w in info.work_experience:
        all_techs.update(t.lower() for t in w.technologies)
    combined = all_skills_lower | all_techs

    python_backend_keywords = {
        "python": 5, "fastapi": 5, "django": 4, "flask": 4,
        "asyncio": 3, "celery": 3, "sqlalchemy": 3,
        "postgresql": 3, "redis": 3, "pydantic": 2,
        "pytest": 2, "uvicorn": 1,
    }
    for kw, pts in python_backend_keywords.items():
        if any(kw in item for item in combined):
            python_score += pts
            rationale.append(f"+{pts} Python/Backend: '{kw}' found")
    
    # Bonus: Python used in actual projects (not just skill list)
    python_in_projects = any("python" in t.lower() for p in info.projects for t in p.technologies)
    if python_in_projects:
        python_score += 3
        rationale.append("+3 Python/Backend: Python used in actual project technologies")

    python_score = min(python_score, 30.0)

    # ── Cloud / Deployment / Full Stack (max 15) ──
    cloud_score = 0.0
    cloud_keywords = {
        "docker": 3, "kubernetes": 3, "gcp": 3, "aws": 3, "azure": 2,
        "ci/cd": 2, "github actions": 2, "terraform": 2,
        "react": 2, "next.js": 2, "nextjs": 2, "vercel": 1,
        "nginx": 1, "linux": 1,
    }
    for kw, pts in cloud_keywords.items():
        if any(kw in item for item in combined):
            cloud_score += pts
            rationale.append(f"+{pts} Cloud/Fullstack: '{kw}' found")
    cloud_score = min(cloud_score, 15.0)

    # ── GitHub (max 10) ──
    github_score = github_data.get("github_score", 0.0)
    if github_score > 0:
        rationale.append(f"+{github_score} GitHub: {github_data.get('github_summary')}")
    else:
        rationale.append("0 GitHub: No activity or URL provided")

    # ── Engineering Depth Signals (max 5) ──
    eng_score = 0.0
    eng_keywords = {
        "testing": 1, "pytest": 1, "unit test": 1,
        "caching": 1, "redis": 1, "queue": 1,
        "observability": 1, "monitoring": 1, "logging": 1,
        "concurrency": 1, "async": 1, "microservice": 1,
        "architecture": 1, "design pattern": 1,
    }
    for kw, pts in eng_keywords.items():
        if any(kw in item for item in combined):
            eng_score += pts
            rationale.append(f"+{pts} Engineering: '{kw}' found")
    eng_score = min(eng_score, 5.0)

    breakdown = ScoreBreakdown(
        ai_project_depth=ai_score,
        python_backend=python_score,
        cloud_fullstack=cloud_score,
        github=github_score,
        engineering_depth=eng_score,
    )

    return breakdown, rationale


# ──────────────────────────────────────────────────────────────
# 3. Full Phase 2 Pipeline
# ──────────────────────────────────────────────────────────────

def score_candidate(info: CandidateInfo, source_file: str, github_data: dict = None) -> ScoredCandidate:
    """
    Complete Phase 2: eligibility check + deterministic base scoring.
    Returns a ScoredCandidate ready for Phase 3 LLM review.
    """
    eligibility = check_eligibility(info)
    breakdown, rationale = compute_base_scores(info, github_data)

    total = (
        breakdown.ai_project_depth
        + breakdown.python_backend
        + breakdown.cloud_fullstack
        + breakdown.github
        + breakdown.engineering_depth
    )

    # Collect matched skills (union of python + ai evidence)
    matched = list(set(eligibility.python_evidence + eligibility.ai_evidence))

    return ScoredCandidate(
        source_file=source_file,
        candidate_info=info,
        eligibility=eligibility,
        base_score_breakdown=breakdown,
        base_total_score=round(total, 1),
        matched_skills=matched,
        scoring_rationale=rationale,
        github_summary=github_data.get("github_summary")
    )
