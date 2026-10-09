"""
LLM Score Reviewer (Phase 3)
=============================
Takes the ScoredCandidate from Phase 2 (deterministic Python scoring)
and sends it to the LLM along with the penalty rules.
The LLM reviews the scores, applies deductions, and returns the final CandidateResult.
"""

import asyncio
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

from src.scorer import ScoredCandidate
from src.models import ScoreBreakdown, CandidateResult
from config import LLM_API_KEY, LLM_MODEL


# ──────────────────────────────────────────────────────────────
# Phase 3 LLM Output Schema
# ──────────────────────────────────────────────────────────────

class LLMReviewResult(BaseModel):
    """What the LLM returns after reviewing the base scores."""
    adjusted_score_breakdown: ScoreBreakdown
    adjusted_total_score: float
    project_summary: str = Field(description="1-2 sentence summary of the candidate's most notable project work")
    strengths: list[str] = Field(description="Key strengths of this candidate for the SDE Intern role")
    concerns: list[str] = Field(description="Concerns or weaknesses found in the resume")
    penalty_log: list[str] = Field(default_factory=list, description="List of penalties applied with reasons, e.g. '-10: AI project is a thin wrapper around OpenAI API'")


REVIEW_PROMPT = """
You are a senior technical hiring reviewer for an SDE Intern position requiring Python + AI/Agentic skills.

You are given a candidate's resume extraction and their BASE SCORES assigned by a deterministic scoring script.
Your job is to REVIEW these scores and APPLY PENALTIES where appropriate.

## PENALTY RULES (you MUST apply these):
1. Deduct 5-15 points from ai_project_depth when an "AI project" is only a thin wrapper around an LLM/API call 
   with no meaningful workflow, data processing, retrieval, state management, backend logic, evaluation, or product logic.
2. Deduct points for tutorial-style projects listed without implementation details or evidence of ownership.
3. Do NOT award full points just because a framework name appears in the skills section. 
   Prefer evidence showing HOW it was used in real projects.
4. A candidate with strong Python but NO meaningful AI project should NOT rank near the top.

## IMPORTANT RULES:
- You may ONLY REDUCE scores, never increase them above the base scores.
- Each category has a maximum: ai_project_depth=/40, python_backend=/30, cloud_fullstack=/15, github=/10, engineering_depth=/5
- Provide a clear penalty_log explaining every deduction you made.
- If no penalties apply, return the base scores unchanged.

## CANDIDATE DATA:
Name: {name}
Source File: {source_file}

### Extracted Skills:
{skills}

### Projects:
{projects}

### Work Experience:
{work_experience}

### Base Scores (from deterministic script):
{base_scores}

### Scoring Rationale (how base scores were assigned):
{rationale}

Review and adjust the scores now.
"""


def get_client() -> genai.Client:
    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is not set in the environment or .env file.")
    return genai.Client(api_key=LLM_API_KEY)


async def review_candidate(scored: ScoredCandidate, custom_prompt: str = None, max_retries: int = 3) -> CandidateResult:
    """
    Phase 3: Send the scored candidate to the LLM for penalty review.
    Returns the final CandidateResult.
    """
    # If not eligible, skip LLM review entirely — no need to waste API calls
    if not scored.eligibility.eligible:
        return CandidateResult(
            candidate_name=scored.candidate_info.name,
            source_file=scored.source_file,
            eligible=False,
            total_score=0,
            score_breakdown=ScoreBreakdown(
                ai_project_depth=0, python_backend=0,
                cloud_fullstack=0, github=0, engineering_depth=0
            ),
            matched_skills=scored.matched_skills,
            rejection_reasons=scored.eligibility.rejection_reasons,
            github_summary=scored.github_summary,
        )

    client = get_client()

    # Format project details for the prompt
    projects_text = ""
    for p in scored.candidate_info.projects:
        projects_text += f"- {p.title}: {p.description}\n"
        projects_text += f"  Tech: {', '.join(p.technologies)}\n"
        projects_text += f"  AI Related: {p.is_ai_related}\n"

    work_text = ""
    for w in scored.candidate_info.work_experience:
        work_text += f"- {w.role} at {w.company} (Tech: {', '.join(w.technologies)})\n"

    base_prompt = custom_prompt if custom_prompt else REVIEW_PROMPT
    prompt = base_prompt.format(
        name=scored.candidate_info.name,
        source_file=scored.source_file,
        skills=", ".join(scored.candidate_info.skills),
        projects=projects_text or "No projects listed.",
        work_experience=work_text or "No work experience listed.",
        base_scores=scored.base_score_breakdown.model_dump_json(indent=2),
        rationale="\n".join(scored.scoring_rationale) or "No rationale available.",
    )

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=LLMReviewResult,
        temperature=0.1,
    )

    current_prompt = prompt

    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=LLM_MODEL,
                contents=current_prompt,
                config=config,
            )

            if hasattr(response, 'parsed') and response.parsed is not None:
                review = response.parsed
            else:
                review = LLMReviewResult.model_validate_json(response.text)

            # Ensure LLM didn't inflate scores above the base
            final_breakdown = ScoreBreakdown(
                ai_project_depth=min(review.adjusted_score_breakdown.ai_project_depth, scored.base_score_breakdown.ai_project_depth),
                python_backend=min(review.adjusted_score_breakdown.python_backend, scored.base_score_breakdown.python_backend),
                cloud_fullstack=min(review.adjusted_score_breakdown.cloud_fullstack, scored.base_score_breakdown.cloud_fullstack),
                github=min(review.adjusted_score_breakdown.github, scored.base_score_breakdown.github),
                engineering_depth=min(review.adjusted_score_breakdown.engineering_depth, scored.base_score_breakdown.engineering_depth),
            )

            final_total = round(
                final_breakdown.ai_project_depth
                + final_breakdown.python_backend
                + final_breakdown.cloud_fullstack
                + final_breakdown.github
                + final_breakdown.engineering_depth, 1
            )

            return CandidateResult(
                candidate_name=scored.candidate_info.name,
                source_file=scored.source_file,
                eligible=True,
                total_score=final_total,
                score_breakdown=final_breakdown,
                matched_skills=scored.matched_skills,
                project_summary=review.project_summary,
                strengths=review.strengths,
                concerns=review.concerns + review.penalty_log,
                github_summary=scored.github_summary,
            )

        except ValidationError as e:
            print(f"⚠️ Review validation error attempt {attempt + 1}/{max_retries} for {scored.source_file}. Retrying...")
            if attempt == max_retries - 1:
                # On final failure, return base scores without review
                return CandidateResult(
                    candidate_name=scored.candidate_info.name,
                    source_file=scored.source_file,
                    eligible=True,
                    total_score=scored.base_total_score,
                    score_breakdown=scored.base_score_breakdown,
                    matched_skills=scored.matched_skills,
                    concerns=["LLM review failed after retries; using base scores."],
                    github_summary=scored.github_summary,
                )
            current_prompt = prompt + f"\n\nYOUR PREVIOUS OUTPUT FAILED VALIDATION:\n{e}\nPlease fix the JSON."

        except Exception as e:
            error_msg = str(e)
            if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                raise Exception("Resource exhausted in API key. Please check your quota or try again later.")
                
            if attempt == max_retries - 1:
                # For non-quota API errors that persist, fall back to base scores gracefully
                print(f"⚠️ LLM review API error for {scored.source_file}: {e}")
                return CandidateResult(
                    candidate_name=scored.candidate_info.name,
                    source_file=scored.source_file,
                    eligible=True,
                    total_score=scored.base_total_score,
                    score_breakdown=scored.base_score_breakdown,
                    matched_skills=scored.matched_skills,
                    concerns=[f"LLM review skipped due to API error: {str(e)}"],
                    github_summary=scored.github_summary,
                )
            import asyncio
            delay = 3
            print(f"⚠️ API/Traffic error on attempt {attempt + 1}/{max_retries} for {scored.source_file}: {e}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
