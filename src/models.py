"""
Pydantic Data Models
====================
Defines the structured schema for all data passed between modules
and the expected JSON output format from the LLM.
"""

from typing import Optional
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────
# Extraction Models (Output of LLM Extractor)
# ──────────────────────────────────────────────────────────────

class ProjectDetail(BaseModel):
    title: str = Field(description="Title of the project or role")
    description: str = Field(description="Short summary of what was built and the business logic")
    technologies: list[str] = Field(description="List of technologies, languages, and frameworks used in this specific project")
    is_ai_related: bool = Field(description="True if the project uses AI, LLMs, RAG, agents, or vector databases")

class WorkExperience(BaseModel):
    company: str
    role: str
    technologies: list[str] = Field(description="Technologies used during this work experience")

class CandidateInfo(BaseModel):
    """The structured data extracted from the resume markdown by the LLM."""
    name: str = Field(description="Candidate's full name. If not found in text, extract from the filename.")
    name_inferred_from_filename: bool = Field(default=False, description="Must be true ONLY if the name was completely missing from the resume text and you had to rely on the filename.")
    email: Optional[str] = None
    phone: Optional[str] = None
    github_url: Optional[str] = Field(default=None, description="Candidate's GitHub profile URL if present")
    linkedin_url: Optional[str] = None
    skills: list[str] = Field(default_factory=list, description="Comprehensive list of all technical skills, languages, and tools mentioned")
    projects: list[ProjectDetail] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Eligibility & Scoring Models (Internal Pipeline State)
# ──────────────────────────────────────────────────────────────

class EligibilityResult(BaseModel):
    candidate: str
    eligible: bool
    rejection_reasons: list[str]
    matched_skills: list[str]
    python_evidence: list[str]
    ai_evidence: list[str]

class ScoreBreakdown(BaseModel):
    ai_project_depth: float     # /40
    python_backend: float       # /30
    cloud_fullstack: float      # /15
    github: float               # /10
    engineering_depth: float    # /5

class GitHubEnrichment(BaseModel):
    username: str
    recent_activity_score: float   # 0-5
    repo_relevance_score: float    # 0-5
    total_public_repos: int
    recent_commits: int
    relevant_repos: list[str]
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# Final Output Models
# ──────────────────────────────────────────────────────────────

class CandidateResult(BaseModel):
    """Final output object per candidate."""
    rank: Optional[int] = None
    candidate_name: str
    source_file: str
    eligible: bool
    total_score: Optional[float] = None
    score_breakdown: Optional[ScoreBreakdown] = None
    matched_skills: list[str]
    project_summary: Optional[str] = None
    github_summary: Optional[str] = None
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)

class BatchSummary(BaseModel):
    total_resumes: int
    successfully_parsed: int
    eligible: int
    rejected: int
    failed_unreadable: int

class FinalOutput(BaseModel):
    batch_summary: BatchSummary
    ranked_candidates: list[CandidateResult]
    rejected_candidates: list[CandidateResult]
    failed_resumes: list[dict]
