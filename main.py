"""
FastAPI Backend — Step-by-Step Pipeline
=======================================
Each phase is a separate endpoint so the user can verify output before proceeding.
  POST /api/phase0  → Parse resumes
  POST /api/phase1  → LLM Extract (takes phase0 output)
  POST /api/phase2  → Python Score (takes phase1 output)
  POST /api/phase3  → LLM Review  (takes phase2 output)
"""

import shutil
import tempfile
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

class Phase3Request(BaseModel):
    custom_prompt: str | None = None

from src.ingestion import ingest_all_resumes
from src.extractor import extract_candidate_info
from src.scorer import score_candidate
from src.reviewer import review_candidate
from src.models import CandidateInfo
from config import MAX_CONCURRENT_LLM_CALLS

app = FastAPI(title="AI Resume Screener API")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store phase results in memory (keyed by a simple session approach)
phase_store = {}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ── Phase 0: Parse ──
@app.post("/api/phase0")
async def run_phase0(
    files: list[UploadFile] = File(...),
    parser_mode: str = Form("auto")
):
    temp_dir = Path(tempfile.mkdtemp(prefix="resumes_"))
    try:
        for file in files:
            safe_filename = Path(file.filename).name
            file_path = temp_dir / safe_filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        loop = asyncio.get_event_loop()
        batch_result = await loop.run_in_executor(
            None,
            lambda: ingest_all_resumes(input_dir=temp_dir, use_cache=False, parser_mode=parser_mode)
        )

        # Store for next phase
        phase_store["phase0"] = batch_result

        return {
            "phase": 0,
            "status": "success",
            "summary": batch_result["summary"],
            "results": [
                {
                    "file_name": r["file_name"],
                    "parser_used": r["parser_used"],
                    "duration_sec": r["duration_sec"],
                    "markdown_preview": (r["markdown"] or "")[:500] + ("..." if len(r.get("markdown", "") or "") > 500 else ""),
                    "markdown_length": len(r.get("markdown", "") or ""),
                }
                for r in batch_result["successful"]
            ],
            "failed": batch_result["failed"],
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Phase 1: LLM Extraction ──
@app.post("/api/phase1")
async def run_phase1():
    batch_result = phase_store.get("phase0")
    if not batch_result:
        raise HTTPException(status_code=400, detail="Phase 0 has not been run yet.")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

    async def extract_one(res):
        async with semaphore:
            try:
                info = await extract_candidate_info(res["markdown"], res["file_name"])
                res["candidate_info"] = info.model_dump()
                res["extraction_error"] = None
            except Exception as e:
                res["candidate_info"] = None
                res["extraction_error"] = str(e)

    tasks = [extract_one(r) for r in batch_result["successful"]]
    if tasks:
        await asyncio.gather(*tasks)

    phase_store["phase0"] = batch_result  # update with extraction data

    return {
        "phase": 1,
        "status": "success",
        "results": [
            {
                "file_name": r["file_name"],
                "candidate_info": r.get("candidate_info"),
                "extraction_error": r.get("extraction_error"),
            }
            for r in batch_result["successful"]
        ],
    }


# ── Phase 2: Python Scoring ──
@app.post("/api/phase2")
async def run_phase2():
    batch_result = phase_store.get("phase0")
    if not batch_result:
        raise HTTPException(status_code=400, detail="Phase 0 has not been run yet.")

    from src.github import fetch_github_enrichment

    # Concurrently fetch GitHub data for all successful candidates
    async def fetch_github_for_candidate(res):
        info_dict = res.get("candidate_info")
        if not info_dict:
            return {"github_summary": "No extraction data", "github_score": 0.0}
        
        info = CandidateInfo(**info_dict)
        if info.github_url:
            return await fetch_github_enrichment(info.github_url)
        return {"github_summary": "No GitHub profile found in resume", "github_score": 0.0}

    github_tasks = [fetch_github_for_candidate(res) for res in batch_result["successful"]]
    github_results = await asyncio.gather(*github_tasks) if github_tasks else []

    results = []
    for i, res in enumerate(batch_result["successful"]):
        if res.get("candidate_info") is None:
            results.append({
                "file_name": res["file_name"],
                "scored": None,
                "error": res.get("extraction_error", "No extraction data"),
            })
            continue

        info = CandidateInfo(**res["candidate_info"])
        github_data = github_results[i]
        
        scored = score_candidate(info, res["file_name"], github_data)
        res["scored"] = scored.model_dump()
        
        results.append({
            "file_name": res["file_name"],
            "eligibility": scored.eligibility.model_dump(),
            "base_score_breakdown": scored.base_score_breakdown.model_dump(),
            "base_total_score": scored.base_total_score,
            "matched_skills": scored.matched_skills,
            "scoring_rationale": scored.scoring_rationale,
            "github_summary": github_data["github_summary"]
        })

    phase_store["phase0"] = batch_result

    return {
        "phase": 2,
        "status": "success",
        "results": results,
    }


# ── Phase 3: LLM Review ──
@app.post("/api/phase3")
async def run_phase3(req: Phase3Request | None = None):
    batch_result = phase_store.get("phase0")
    if not batch_result:
        raise HTTPException(status_code=400, detail="Phase 0 has not been run yet.")

    custom_prompt = req.custom_prompt if req else None

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    scored_candidates = []

    for res in batch_result["successful"]:
        if res.get("scored") is None:
            continue
        from src.scorer import ScoredCandidate
        scored = ScoredCandidate(**res["scored"])
        scored_candidates.append((res, scored))

    async def review_one(item):
        res, scored = item
        async with semaphore:
            result = await review_candidate(scored, custom_prompt)
            res["final_result"] = result.model_dump()

    review_tasks = [review_one(item) for item in scored_candidates]
    if review_tasks:
        await asyncio.gather(*review_tasks)

    # Rank
    eligible = []
    rejected = []
    for res in batch_result["successful"]:
        final = res.get("final_result")
        if final is None:
            continue
        if final["eligible"]:
            eligible.append(final)
        else:
            rejected.append(final)

    eligible.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    for i, c in enumerate(eligible):
        c["rank"] = i + 1

    return {
        "phase": 3,
        "status": "success",
        "ranked_candidates": eligible,
        "rejected_candidates": rejected,
    }


if __name__ == "__main__":
    import sys
    # If the user provides command-line arguments (e.g. --input), run the CLI batch script.
    # Otherwise, spin up the FastAPI interactive wizard.
    if len(sys.argv) > 1:
        import cli
        cli.main()
    else:
        import uvicorn
        print("Starting FastAPI Interactive Wizard...")
        uvicorn.run(app, host="127.0.0.1", port=8000)
