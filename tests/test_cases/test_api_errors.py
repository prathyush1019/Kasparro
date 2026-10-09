import pytest
from fastapi.testclient import TestClient
from api import app, phase_store

# Create a test client for the FastAPI app
client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_store():
    """Reset the phase_store before every test to ensure clean state."""
    phase_store.clear()

def test_1_phase1_without_phase0_returns_400():
    """Test calling Phase 1 before parsing returns HTTP 400"""
    response = client.post("/api/phase1")
    assert response.status_code == 400
    assert "Phase 0 has not been run yet" in response.json()["detail"]

def test_2_phase2_without_phase0_returns_400():
    """Test calling Phase 2 before parsing returns HTTP 400"""
    response = client.post("/api/phase2")
    assert response.status_code == 400
    assert "Phase 0 has not been run yet" in response.json()["detail"]

def test_3_phase3_without_phase0_returns_400():
    """Test calling Phase 3 before parsing returns HTTP 400"""
    response = client.post("/api/phase3")
    assert response.status_code == 400
    assert "Phase 0 has not been run yet" in response.json()["detail"]

def test_4_phase0_without_files_returns_422():
    """Test calling Phase 0 with no files (FastAPI returns 422 Unprocessable Entity)"""
    response = client.post("/api/phase0", data={"parser_mode": "fast"})
    # FastAPI automatically throws 422 if required file fields are missing
    assert response.status_code == 422

def test_5_phase0_with_empty_file_list_returns_422():
    """Test calling Phase 0 with an empty files list"""
    response = client.post("/api/phase0", files=[], data={"parser_mode": "fast"})
    assert response.status_code == 422

def test_6_phase0_unsupported_file_extension():
    """Test calling Phase 0 with an unsupported file extension (.exe). 
    The API should return 200 but the file should be listed in 'failed'."""
    files = [("files", ("malicious.exe", b"fake content", "application/x-msdownload"))]
    response = client.post("/api/phase0", files=files, data={"parser_mode": "fast"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["failed"]) == 1
    assert "unsupported format" in data["failed"][0]["error"].lower()

def test_7_phase0_corrupt_pdf():
    """Test calling Phase 0 with a completely corrupt PDF.
    It should not crash the batch (HTTP 200) but should list it as failed."""
    files = [("files", ("corrupt.pdf", b"this is not a real pdf", "application/pdf"))]
    response = client.post("/api/phase0", files=files, data={"parser_mode": "fast"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["failed"]) == 1
    assert len(data["results"]) == 0

def test_8_phase0_mixed_batch():
    """Test calling Phase 0 with one good file and one corrupt file.
    The batch should succeed, processing the good one and logging the bad one."""
    files = [
        ("files", ("corrupt.pdf", b"corrupt", "application/pdf")),
        ("files", ("good.txt", b"Name: John Doe\nSkills: Python, AI\nExperience: Built a LangChain app.", "text/plain"))
    ]
    response = client.post("/api/phase0", files=files, data={"parser_mode": "fast"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["failed"]) == 1
    assert len(data["results"]) == 1

def test_9_phase1_with_corrupt_markdown():
    """Test if Phase 1 handles markdown that causes LLM extraction to fail gracefully."""
    # Fake Phase 0 store with empty/meaningless markdown
    phase_store["phase0"] = {
        "summary": {},
        "successful": [{"file_name": "empty.pdf", "markdown": "just noise", "parser_used": "fast", "duration_sec": 0.1}],
        "failed": []
    }
    response = client.post("/api/phase1")
    assert response.status_code == 200
    data = response.json()
    # If the LLM throws an error (e.g. ValidationError from Pydantic), it sets extraction_error
    assert len(data["results"]) == 1

def test_10_invalid_parser_mode_fallback():
    """Test Phase 0 handles invalid parser mode string gracefully."""
    files = [("files", ("good.txt", b"Content", "text/plain"))]
    # 'invalid_mode' should just default to 'fast' in ingestion.py
    response = client.post("/api/phase0", files=files, data={"parser_mode": "invalid_mode"})
    assert response.status_code == 200
    assert response.json()["summary"]["total_files"] == 1
