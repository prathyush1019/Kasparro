"""
AI Resume Screening — CLI Entry Point
======================================
Usage:
    python main.py --input ./resumes
    python main.py --input ./resumes --no-cache
    python main.py                      # Opens a folder picker dialog

The script ingests resumes from the selected folder, parses them
using Docling (GPU) or PyMuPDF (fallback), and outputs parsed
markdown files to the cache directory.
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
import asyncio
import json

# Load environment variables from .env file if present
load_dotenv()


def setup_logging() -> None:
    """Configure structured logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def select_folder_interactive() -> Path:
    """
    Open a native Windows folder picker dialog if no --input argument
    is provided. Falls back to a terminal prompt if tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select Resume Folder")
        root.destroy()

        if not folder:
            print("No folder selected. Exiting.")
            sys.exit(1)

        return Path(folder)

    except ImportError:
        # Tkinter not available — fall back to terminal input
        folder = input("Enter the path to the resume folder: ").strip().strip('"')
        if not folder:
            print("No folder provided. Exiting.")
            sys.exit(1)
        return Path(folder)


def validate_input_dir(input_dir: Path) -> None:
    """Validate the input directory exists and contains resume files."""
    if not input_dir.exists():
        print(f"Error: Directory does not exist: {input_dir}")
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Error: Path is not a directory: {input_dir}")
        sys.exit(1)


def main() -> None:
    """Main entry point for the resume screening pipeline."""
    setup_logging()
    logger = logging.getLogger("main")

    # ── Parse CLI Arguments ──
    parser = argparse.ArgumentParser(
        description="AI Resume Screening & Ranking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --input ./resumes\n"
            "  python main.py --input ./resumes --no-cache\n"
            "  python main.py  (opens folder picker)\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the directory containing resume files (PDF/DOCX/TXT)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/results.json"),
        help="Path for the output results JSON file (default: output/results.json)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-parsing all resumes, ignoring cached results",
    )
    parser.add_argument(
        "--parser",
        type=str,
        choices=["auto", "accurate", "fast"],
        default="auto",
        help=(
            "Parser mode: "
            "'auto' = Docling (GPU) with PyMuPDF fallback, "
            "'accurate' = Docling only (best layout accuracy), "
            "'fast' = PyMuPDF only (10x faster on CPU). "
            "Default: auto"
        ),
    )

    args = parser.parse_args()

    # ── Select Input Folder ──
    if args.input is not None:
        input_dir = args.input
    else:
        logger.info("No --input argument provided. Opening folder picker...")
        input_dir = select_folder_interactive()

    validate_input_dir(input_dir)
    logger.info(f"Input directory: {input_dir.resolve()}")

    # ── Ensure Output Directory Exists ──
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ── Run Ingestion Pipeline ──
    from src.ingestion import ingest_all_resumes, print_ingestion_summary

    use_cache = not args.no_cache
    parser_mode = args.parser
    logger.info(f"Parser mode: {parser_mode.upper()}")
    batch_result = ingest_all_resumes(input_dir, use_cache=use_cache, parser_mode=parser_mode)

    # ── Print Summary ──
    print_ingestion_summary(batch_result)

    # ── Report Status ──
    total_parsed = batch_result["summary"]["successfully_parsed"] + batch_result["summary"]["loaded_from_cache"]
    logger.info(f"Ingestion complete. {total_parsed} resumes ready for screening.")

    if batch_result["failed"]:
        logger.warning(f"{len(batch_result['failed'])} resumes failed to parse.")

    # ── Phase 1: LLM Extraction ──
    logger.info("Starting Phase 1: LLM Extraction...")
    from src.extractor import extract_candidate_info
    
    extracted_candidates = []
    # Using asyncio.run for the async components
    async def run_phase1():
        semaphore = asyncio.Semaphore(5)
        async def extract_one(res):
            async with semaphore:
                try:
                    info = await extract_candidate_info(res["markdown"], res["file_name"])
                    res["candidate_info"] = info.model_dump()
                except Exception as e:
                    res["extraction_error"] = str(e)
            return res
        tasks = [extract_one(res) for res in batch_result["successful"]]
        return await asyncio.gather(*tasks)

    batch_result["successful"] = asyncio.run(run_phase1())
    logger.info("Phase 1 complete.")

    # ── Phase 2: Python Scoring & GitHub ──
    logger.info("Starting Phase 2: Scoring and GitHub enrichment...")
    from src.scorer import score_candidate
    from src.github import fetch_github_enrichment
    from src.models import CandidateInfo

    async def run_phase2():
        async def fetch_github_for_candidate(res):
            info_dict = res.get("candidate_info")
            if not info_dict:
                return {"github_summary": "No extraction data", "github_score": 0.0}
            info = CandidateInfo(**info_dict)
            if info.github_url:
                return await fetch_github_enrichment(info.github_url)
            return {"github_summary": "No GitHub profile found in resume", "github_score": 0.0}

        github_tasks = [fetch_github_for_candidate(res) for res in batch_result["successful"]]
        return await asyncio.gather(*github_tasks) if github_tasks else []

    github_results = asyncio.run(run_phase2())
    
    scored_candidates = []
    for i, res in enumerate(batch_result["successful"]):
        if not res.get("candidate_info"):
            continue
        info = CandidateInfo(**res["candidate_info"])
        github_data = github_results[i]
        scored = score_candidate(info, res["file_name"], github_data)
        res["scored"] = scored.model_dump()
        scored_candidates.append((res, scored))
    
    logger.info("Phase 2 complete.")

    # ── Phase 3: LLM Review ──
    logger.info("Starting Phase 3: LLM Review & Penalties...")
    from src.reviewer import review_candidate
    
    async def run_phase3():
        semaphore = asyncio.Semaphore(5)
        async def review_one(item):
            res, scored = item
            async with semaphore:
                result = await review_candidate(scored, None)
                res["final_result"] = result.model_dump()
                return result
        tasks = [review_one(item) for item in scored_candidates]
        return await asyncio.gather(*tasks)

    final_results = asyncio.run(run_phase3())
    logger.info("Phase 3 complete.")

    # ── Generate Output JSON ──
    ranked_candidates = []
    rejected_candidates = []
    for r in final_results:
        if r.eligible:
            ranked_candidates.append(r)
        else:
            rejected_candidates.append(r)
            
    ranked_candidates.sort(key=lambda x: x.total_score, reverse=True)
    
    # Assign ranks
    for i, c in enumerate(ranked_candidates):
        c.rank = i + 1

    final_output = {
        "batch_summary": {
            "total_resumes": batch_result["summary"]["total_files"],
            "successfully_parsed": batch_result["summary"]["successfully_parsed"] + batch_result["summary"]["loaded_from_cache"],
            "eligible": len(ranked_candidates),
            "rejected": len(rejected_candidates),
            "failed_unreadable": len(batch_result["failed"])
        },
        "ranked_candidates": [c.model_dump() for c in ranked_candidates],
        "rejected_candidates": [c.model_dump() for c in rejected_candidates],
        "failed_resumes": batch_result["failed"]
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"Pipeline finished! Results successfully saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
