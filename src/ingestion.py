"""
Resume Ingestion Module
=======================
Parses resumes from a folder using Docling (GPU-accelerated) with
PyMuPDF as a fallback. Supports PDF, DOCX, and TXT files.

Features:
- Docling (GPU) → PyMuPDF fallback hierarchy
- Per-file error isolation (one bad file never crashes the batch)
- Local markdown caching for fast re-runs
- Embedded hyperlink extraction (for GitHub URLs hidden behind icons)
- Progress tracking with tqdm
"""

import logging
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config import SUPPORTED_EXTENSIONS, CACHE_DIR, USE_CACHE, MAX_FILE_SIZE_MB, PARSER_MODE

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Docling Availability Check
# ──────────────────────────────────────────────────────────────
DOCLING_AVAILABLE = False
_docling_converter = None

try:
    from docling.document_converter import DocumentConverter
    _docling_converter = DocumentConverter()
    DOCLING_AVAILABLE = True

    # Check GPU status
    try:
        import torch
        gpu_status = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    except ImportError:
        gpu_status = "CPU (torch not found)"

    logger.info(f"Docling initialized successfully ({gpu_status})")
except ImportError:
    logger.info("Docling not installed. Using PyMuPDF fallback for all files.")
except Exception as e:
    logger.warning(f"Docling initialization failed: {e}. Using PyMuPDF fallback.")


# ──────────────────────────────────────────────────────────────
# File Scanner
# ──────────────────────────────────────────────────────────────
def scan_resume_folder(input_dir: Path) -> list[Path]:
    """
    Scan the input directory for supported resume files.
    Returns a sorted list of file paths.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files = []
    for f in sorted(input_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            # Check file size
            size_mb = f.stat().st_size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                logger.warning(f"Skipping {f.name}: file size ({size_mb:.1f}MB) exceeds limit ({MAX_FILE_SIZE_MB}MB)")
                continue
            files.append(f)

    if not files:
        logger.warning(f"No supported resume files found in {input_dir}")

    # Log file type breakdown
    ext_counts: dict[str, int] = {}
    for f in files:
        ext = f.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    breakdown = ", ".join(f"{count} {ext.upper()}" for ext, count in sorted(ext_counts.items()))
    logger.info(f"Found {len(files)} resume files ({breakdown})")

    return files


# ──────────────────────────────────────────────────────────────
# Cache Management
# ──────────────────────────────────────────────────────────────
def _get_cache_path(file_path: Path) -> Path:
    """Get the cache file path for a given resume file."""
    return CACHE_DIR / f"{file_path.stem}.md"


def _read_from_cache(file_path: Path) -> Optional[str]:
    """Read cached markdown if it exists and caching is enabled."""
    if not USE_CACHE:
        return None
    cache_path = _get_cache_path(file_path)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return None


def _write_to_cache(file_path: Path, markdown: str) -> None:
    """Save parsed markdown to cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_cache_path(file_path)
    cache_path.write_text(markdown, encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Docling Parser
# ──────────────────────────────────────────────────────────────
def _parse_with_docling(file_path: Path) -> Optional[str]:
    """
    Parse a resume using Docling. Returns markdown text or None on failure.
    Handles PDF, DOCX, and other formats natively.
    """
    if not DOCLING_AVAILABLE or _docling_converter is None:
        return None

    try:
        result = _docling_converter.convert(str(file_path))
        markdown = result.document.export_to_markdown()
        if markdown and markdown.strip():
            return markdown.strip()
        logger.warning(f"Docling returned empty text for {file_path.name}")
        return None
    except Exception as e:
        logger.warning(f"Docling failed for {file_path.name}: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# PyMuPDF Fallback Parser
# ──────────────────────────────────────────────────────────────
def _parse_with_pymupdf(file_path: Path) -> Optional[tuple[str, list[str]]]:
    """
    Parse a PDF using PyMuPDF (fitz) as a fallback.
    Returns (markdown_text, list_of_embedded_urls) or None on failure.
    Only supports PDF files.
    """
    if file_path.suffix.lower() != ".pdf":
        return None

    try:
        import fitz  # PyMuPDF

        text_parts = []
        links = []

        with fitz.open(str(file_path)) as doc:
            for page in doc:
                # Block-level extraction with reading-order sorting
                blocks = page.get_text("blocks", sort=True)
                for block in blocks:
                    # block = (x0, y0, x1, y1, text, block_no, block_type)
                    block_text = block[4].strip()
                    if block_text:
                        text_parts.append(block_text)

                # Extract embedded hyperlinks (GitHub URLs behind icons)
                for link in page.get_links():
                    uri = link.get("uri")
                    if uri:
                        links.append(uri)

        full_text = "\n\n".join(text_parts)

        # Append embedded links so the LLM extractor can see them
        if links:
            unique_links = list(dict.fromkeys(links))  # Deduplicate, preserve order
            full_text += "\n\n## Embedded Links Found in Resume\n"
            full_text += "\n".join(f"- {link}" for link in unique_links)

        if full_text.strip():
            return full_text.strip(), links

        logger.warning(f"PyMuPDF returned empty text for {file_path.name}")
        return None

    except Exception as e:
        logger.warning(f"PyMuPDF failed for {file_path.name}: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# TXT Fallback Parser
# ──────────────────────────────────────────────────────────────
def _parse_txt_file(file_path: Path) -> Optional[str]:
    """Parse a plain text resume file."""
    try:
        text = file_path.read_text(encoding="utf-8")
        if text.strip():
            return text.strip()
        return None
    except UnicodeDecodeError:
        try:
            text = file_path.read_text(encoding="latin-1")
            if text.strip():
                return text.strip()
        except Exception:
            pass
        return None
    except Exception as e:
        logger.warning(f"Failed to read text file {file_path.name}: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# Single File Parser (Orchestrates Fallback Hierarchy)
# ──────────────────────────────────────────────────────────────
def parse_single_resume(file_path: Path, parser_mode: str = None) -> dict:
    """
    Parse a single resume file using the selected parser mode:

    Modes:
      - "auto"     : Docling first → PyMuPDF fallback → plaintext fallback
      - "accurate" : Docling only (best layout accuracy, GPU recommended)
      - "fast"     : PyMuPDF only (10x faster, good for most resumes)

    Returns a structured result dict:
    {
        "file_name": str,
        "file_path": str,
        "file_type": str,
        "status": "success" | "cached" | "failed",
        "parser_used": str,
        "markdown": str | None,
        "links": list[str],
        "error": str | None,
        "duration_sec": float
    }
    """
    # Resolve parser mode: argument > config > default
    mode = (parser_mode or PARSER_MODE).lower()
    if mode not in ("auto", "accurate", "fast"):
        logger.warning(f"Unknown parser mode '{mode}', falling back to 'auto'")
        mode = "auto"

    start = time.perf_counter()
    ext = file_path.suffix.lower()

    result = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_type": ext,
        "status": "failed",
        "parser_used": "none",
        "markdown": None,
        "links": [],
        "error": None,
        "duration_sec": 0.0,
    }

    # ── Step 1: Check cache (all modes) ──
    cached = _read_from_cache(file_path)
    if cached is not None:
        result["status"] = "cached"
        result["parser_used"] = "cache"
        result["markdown"] = cached
        result["duration_sec"] = time.perf_counter() - start
        return result

    # ── Step 2: Parse based on mode ──

    if mode == "fast":
        # ─── FAST MODE: PyMuPDF only, skip Docling entirely ───
        if ext == ".pdf":
            pymupdf_result = _parse_with_pymupdf(file_path)
            if pymupdf_result:
                text, links = pymupdf_result
                result["status"] = "success"
                result["parser_used"] = "pymupdf"
                result["markdown"] = text
                result["links"] = links
                _write_to_cache(file_path, text)
                result["duration_sec"] = time.perf_counter() - start
                return result
        elif ext == ".txt":
            text = _parse_txt_file(file_path)
            if text:
                result["status"] = "success"
                result["parser_used"] = "plaintext"
                result["markdown"] = text
                _write_to_cache(file_path, text)
                result["duration_sec"] = time.perf_counter() - start
                return result
        elif ext in {".docx", ".doc"}:
            # PyMuPDF can't handle DOCX — try Docling even in fast mode
            markdown = _parse_with_docling(file_path)
            if markdown:
                result["status"] = "success"
                result["parser_used"] = "docling"
                result["markdown"] = markdown
                _write_to_cache(file_path, markdown)
                result["duration_sec"] = time.perf_counter() - start
                return result

    elif mode == "accurate":
        # ─── ACCURATE MODE: Docling only, no fallback ───
        if ext in {".pdf", ".docx", ".doc"}:
            markdown = _parse_with_docling(file_path)
            if markdown:
                result["status"] = "success"
                result["parser_used"] = "docling"
                result["markdown"] = markdown
                _write_to_cache(file_path, markdown)
                result["duration_sec"] = time.perf_counter() - start
                return result
            else:
                result["error"] = (
                    f"Docling failed for {file_path.name} and no fallback is "
                    f"allowed in 'accurate' mode. Install docling or use --parser auto."
                )
                result["duration_sec"] = time.perf_counter() - start
                return result
        elif ext == ".txt":
            text = _parse_txt_file(file_path)
            if text:
                result["status"] = "success"
                result["parser_used"] = "plaintext"
                result["markdown"] = text
                _write_to_cache(file_path, text)
                result["duration_sec"] = time.perf_counter() - start
                return result

    else:
        # ─── AUTO MODE: Docling → PyMuPDF → plaintext ───
        if ext in {".pdf", ".docx", ".doc"}:
            markdown = _parse_with_docling(file_path)
            if markdown:
                result["status"] = "success"
                result["parser_used"] = "docling"
                result["markdown"] = markdown
                _write_to_cache(file_path, markdown)
                result["duration_sec"] = time.perf_counter() - start
                return result

        # Docling failed or not available — try PyMuPDF for PDFs
        if ext == ".pdf":
            pymupdf_result = _parse_with_pymupdf(file_path)
            if pymupdf_result:
                text, links = pymupdf_result
                result["status"] = "success"
                result["parser_used"] = "pymupdf"
                result["markdown"] = text
                result["links"] = links
                _write_to_cache(file_path, text)
                result["duration_sec"] = time.perf_counter() - start
                return result

        # Plaintext fallback for .txt files
        if ext == ".txt":
            text = _parse_txt_file(file_path)
            if text:
                result["status"] = "success"
                result["parser_used"] = "plaintext"
                result["markdown"] = text
                _write_to_cache(file_path, text)
                result["duration_sec"] = time.perf_counter() - start
                return result

    # ── All parsers failed for this mode ──
    if not result["error"]:
        result["error"] = f"No parser could extract text from {file_path.name} ({ext}) in '{mode}' mode"
    result["duration_sec"] = time.perf_counter() - start
    return result


# ──────────────────────────────────────────────────────────────
# Batch Processor
# ──────────────────────────────────────────────────────────────
def ingest_all_resumes(
    input_dir: Path,
    use_cache: bool = True,
    parser_mode: str = None,
) -> dict:
    """
    Parse all resumes in the input directory.

    Args:
        input_dir: Path to the folder containing resume files.
        use_cache: Whether to use cached parsed results.
        parser_mode: "auto", "accurate", or "fast". Defaults to config.PARSER_MODE.

    Returns a structured batch result:
    {
        "successful": list[dict],    # Parsed resume results
        "failed": list[dict],        # Failed resume results
        "summary": {
            "total_files": int,
            "successfully_parsed": int,
            "loaded_from_cache": int,
            "failed": int,
            "total_time_sec": float,
            "avg_time_per_file_sec": float,
            "parser_mode": str,
            "parser_breakdown": dict[str, int],
            "file_type_breakdown": dict[str, int],
        }
    }
    """
    # Override global cache setting if requested
    import config
    original_cache = config.USE_CACHE
    config.USE_CACHE = use_cache

    batch_start = time.perf_counter()

    # Scan for resume files
    files = scan_resume_folder(input_dir)
    if not files:
        return {
            "successful": [],
            "failed": [],
            "summary": {
                "total_files": 0,
                "successfully_parsed": 0,
                "loaded_from_cache": 0,
                "failed": 0,
                "total_time_sec": 0.0,
                "avg_time_per_file_sec": 0.0,
                "parser_breakdown": {},
                "file_type_breakdown": {},
            },
        }

    successful = []
    failed = []
    parser_counts: dict[str, int] = {}
    ext_counts: dict[str, int] = {}

    # Process each file with progress tracking
    active_mode = (parser_mode or PARSER_MODE).lower()
    logger.info(f"Parser mode: {active_mode.upper()}")

    for file_path in tqdm(files, desc=f"Parsing resumes [{active_mode}]", unit="file"):
        try:
            result = parse_single_resume(file_path, parser_mode=active_mode)
        except Exception as e:
            # Absolute last-resort catch — should never reach here
            logger.error(f"Unexpected error parsing {file_path.name}: {e}")
            result = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower(),
                "status": "failed",
                "parser_used": "none",
                "markdown": None,
                "links": [],
                "error": f"Unexpected error: {str(e)}",
                "duration_sec": 0.0,
            }

        # Categorize result
        if result["status"] in ("success", "cached"):
            successful.append(result)
        else:
            failed.append(result)

        # Track parser usage
        parser = result["parser_used"]
        parser_counts[parser] = parser_counts.get(parser, 0) + 1

        # Track file type
        ext = result["file_type"]
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    batch_time = time.perf_counter() - batch_start
    total = len(files)
    cached_count = sum(1 for r in successful if r["status"] == "cached")

    summary = {
        "total_files": total,
        "successfully_parsed": len(successful) - cached_count,
        "loaded_from_cache": cached_count,
        "failed": len(failed),
        "total_time_sec": round(batch_time, 2),
        "avg_time_per_file_sec": round(batch_time / total, 2) if total > 0 else 0.0,
        "parser_mode": active_mode,
        "parser_breakdown": parser_counts,
        "file_type_breakdown": ext_counts,
    }

    # Restore original cache setting
    config.USE_CACHE = original_cache

    return {
        "successful": successful,
        "failed": failed,
        "summary": summary,
    }


# ──────────────────────────────────────────────────────────────
# Pretty Print Summary
# ──────────────────────────────────────────────────────────────
def print_ingestion_summary(batch_result: dict) -> None:
    """Print a formatted ingestion summary to the terminal."""
    s = batch_result["summary"]

    print()
    print("=" * 56)
    print("           RESUME INGESTION SUMMARY")
    print("=" * 56)
    print(f"  Total Files Found     : {s['total_files']}")
    print(f"  Parser Mode           : {s.get('parser_mode', 'auto').upper()}")

    # File type breakdown
    for ext, count in sorted(s["file_type_breakdown"].items()):
        print(f"    {ext.upper():8s}            : {count}")

    print(f"  Successfully Parsed   : {s['successfully_parsed']}")
    print(f"  Loaded from Cache     : {s['loaded_from_cache']}")
    print(f"  Failed / Corrupt      : {s['failed']}")

    # List failed files
    for f in batch_result["failed"]:
        print(f"    └─ {f['file_name']} : {f['error']}")

    # Parser breakdown
    print(f"  Parser Breakdown      :")
    for parser, count in sorted(s["parser_breakdown"].items()):
        print(f"    {parser:12s}        : {count}")

    print(f"  Total Parse Time      : {s['total_time_sec']}s"
          f" (avg {s['avg_time_per_file_sec']}s/file)")
    print("=" * 56)
    print()
