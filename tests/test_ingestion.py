"""
Unit Tests for Resume Ingestion Module
=======================================
Tests the file scanner, parser fallback hierarchy, caching, and
batch resilience (one bad file must not crash the batch).
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ingestion import (
    scan_resume_folder,
    parse_single_resume,
    _read_from_cache,
    _write_to_cache,
    _get_cache_path,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def sample_pdf_dir(temp_dir):
    """Create a temp dir with a few dummy files of various types."""
    (temp_dir / "resume1.pdf").write_bytes(b"%PDF-1.4 dummy")
    (temp_dir / "resume2.pdf").write_bytes(b"%PDF-1.4 dummy2")
    (temp_dir / "resume3.docx").write_bytes(b"PK dummy docx")
    (temp_dir / "resume4.txt").write_text("John Doe\nPython Developer\nSkills: Python, FastAPI", encoding="utf-8")
    (temp_dir / "photo.jpg").write_bytes(b"\xff\xd8\xff dummy jpg")
    (temp_dir / "notes.xlsx").write_bytes(b"PK dummy xlsx")
    return temp_dir


# ──────────────────────────────────────────────
# Scanner Tests
# ──────────────────────────────────────────────
class TestScanResumeFolder:
    def test_finds_pdf_docx_txt(self, sample_pdf_dir):
        """Scanner should find .pdf, .docx, and .txt files."""
        files = scan_resume_folder(sample_pdf_dir)
        names = {f.name for f in files}
        assert "resume1.pdf" in names
        assert "resume2.pdf" in names
        assert "resume3.docx" in names
        assert "resume4.txt" in names

    def test_ignores_unsupported_extensions(self, sample_pdf_dir):
        """Scanner should skip .jpg, .xlsx and other unsupported formats."""
        files = scan_resume_folder(sample_pdf_dir)
        names = {f.name for f in files}
        assert "photo.jpg" not in names
        assert "notes.xlsx" not in names

    def test_empty_folder(self, temp_dir):
        """Scanner should return empty list for an empty folder."""
        files = scan_resume_folder(temp_dir)
        assert files == []

    def test_nonexistent_folder(self):
        """Scanner should raise FileNotFoundError for missing dirs."""
        with pytest.raises(FileNotFoundError):
            scan_resume_folder(Path("C:/nonexistent/path/that/does/not/exist"))

    def test_returns_sorted(self, sample_pdf_dir):
        """Scanner should return files in alphabetical order."""
        files = scan_resume_folder(sample_pdf_dir)
        names = [f.name for f in files]
        assert names == sorted(names)


# ──────────────────────────────────────────────
# Single File Parser Tests
# ──────────────────────────────────────────────
class TestParseSingleResume:
    def test_txt_file_parses_successfully(self, sample_pdf_dir):
        """Plain text resume should parse with plaintext parser."""
        result = parse_single_resume(sample_pdf_dir / "resume4.txt")
        assert result["status"] == "success"
        assert result["parser_used"] == "plaintext"
        assert "Python" in result["markdown"]
        assert "John Doe" in result["markdown"]

    def test_corrupt_pdf_does_not_crash(self, temp_dir):
        """A corrupt PDF should return status='failed', not raise an exception."""
        corrupt = temp_dir / "corrupt.pdf"
        corrupt.write_bytes(b"this is not a real pdf")
        result = parse_single_resume(corrupt)
        assert result["status"] == "failed"
        assert result["error"] is not None

    def test_empty_file_fails_gracefully(self, temp_dir):
        """An empty file should return status='failed'."""
        empty = temp_dir / "empty.pdf"
        empty.write_bytes(b"")
        result = parse_single_resume(empty)
        assert result["status"] == "failed"

    def test_result_structure(self, sample_pdf_dir):
        """Parser result should contain all required fields."""
        result = parse_single_resume(sample_pdf_dir / "resume4.txt")
        required_keys = {
            "file_name", "file_path", "file_type", "status",
            "parser_used", "markdown", "links", "error", "duration_sec",
        }
        assert required_keys.issubset(result.keys())


# ──────────────────────────────────────────────
# Cache Tests
# ──────────────────────────────────────────────
class TestCaching:
    def test_cache_write_and_read(self, temp_dir):
        """Cached markdown should be retrievable after writing."""
        test_file = temp_dir / "test_resume.pdf"
        test_file.write_bytes(b"dummy")

        with patch("src.ingestion.CACHE_DIR", temp_dir / "cache"):
            _write_to_cache(test_file, "# Cached Resume\n\nSkills: Python")
            cached = _read_from_cache(test_file)
            assert cached is not None
            assert "Python" in cached

    def test_cache_miss_returns_none(self, temp_dir):
        """Reading from cache for a file that hasn't been cached should return None."""
        test_file = temp_dir / "never_cached.pdf"
        test_file.write_bytes(b"dummy")

        with patch("src.ingestion.CACHE_DIR", temp_dir / "cache"):
            result = _read_from_cache(test_file)
            assert result is None


# ──────────────────────────────────────────────
# Batch Resilience Tests
# ──────────────────────────────────────────────
class TestBatchResilience:
    def test_batch_continues_after_corrupt_file(self, temp_dir):
        """One corrupt file must not stop the batch from processing others."""
        from src.ingestion import ingest_all_resumes

        # Create a good .txt and a corrupt .pdf
        (temp_dir / "good.txt").write_text("Jane Doe\nPython, LangChain", encoding="utf-8")
        (temp_dir / "corrupt.pdf").write_bytes(b"not a pdf at all")

        with patch("src.ingestion.CACHE_DIR", temp_dir / "cache"):
            with patch("src.ingestion.USE_CACHE", False):
                batch = ingest_all_resumes(temp_dir, use_cache=False)

        # Good file should succeed
        assert len(batch["successful"]) >= 1
        good_names = {r["file_name"] for r in batch["successful"]}
        assert "good.txt" in good_names

        # Summary should reflect both
        assert batch["summary"]["total_files"] == 2
