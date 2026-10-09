"""
Configuration constants for the AI Resume Screening system.
All tunables are defined here, separated from business logic.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────
# File Ingestion
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
CACHE_DIR = Path("output/parsed_markdown")
USE_CACHE = True
ENABLE_OCR = True
MAX_FILE_SIZE_MB = 50

# Parser Mode:
#   "auto"     — Use Docling (GPU) first, fall back to PyMuPDF if unavailable/fails
#   "accurate" — Force Docling only (best layout accuracy, requires GPU for speed)
#   "fast"     — Force PyMuPDF only (10x faster on CPU, good enough for most resumes)
PARSER_MODE = "auto"

# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────
OUTPUT_DIR = Path("output")
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "results.json"

# ──────────────────────────────────────────────
# Scoring Weights (must sum to 100)
# ──────────────────────────────────────────────
SCORING_WEIGHTS = {
    "ai_project_depth": 40,
    "python_backend": 30,
    "cloud_fullstack": 15,
    "github": 10,
    "engineering_depth": 5,
}

# ──────────────────────────────────────────────
# LLM Configuration
# ──────────────────────────────────────────────
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# ──────────────────────────────────────────────
# GitHub Enrichment
# ──────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)

# ──────────────────────────────────────────────
# Concurrency
# ──────────────────────────────────────────────
MAX_CONCURRENT_LLM_CALLS = 5
MAX_CONCURRENT_GITHUB_CALLS = 10

# ──────────────────────────────────────────────
# Eligibility Keywords (used by rule-based filter)
# ──────────────────────────────────────────────
PYTHON_KEYWORDS = [
    "python", "fastapi", "django", "flask", "streamlit",
    "pandas", "numpy", "scipy", "pytest", "celery",
    "sqlalchemy", "pydantic", "uvicorn", "asyncio",
]

AI_KEYWORDS = [
    "langchain", "langgraph", "llamaindex", "llama_index",
    "rag", "retrieval augmented generation", "embeddings",
    "vector search", "vector database", "pinecone", "weaviate",
    "chromadb", "chroma", "faiss", "openai", "gpt",
    "gemini", "claude", "anthropic", "huggingface", "transformers",
    "llm", "large language model", "agent", "agentic",
    "tool calling", "function calling", "multi-agent",
    "autogen", "crewai", "google adk", "a2a",
    "fine-tuning", "lora", "qlora", "prompt engineering",
    "semantic search", "evaluation pipeline",
]
