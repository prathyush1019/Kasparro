# AI Resume Screening System

A production-minded, hybrid AI/deterministic pipeline for screening software engineering resumes. It ingests resumes (PDF/DOCX/TXT), extracts structured data using Gemini 2.0 Flash, applies strict deterministic eligibility filters, enriches with public GitHub API data, and finally uses the LLM to review and penalize low-quality projects.

## Architecture

This project uses a beautifully unified, dual-interface architecture inside a single `main.py` entry point:
1. **FastAPI Web Wizard (No args):** An interactive UI allowing step-by-step verification of the 4-phase pipeline (Parse ➔ Extract ➔ Score ➔ Review), including a "One-Click" auto-run button.
2. **CLI Mode (Args provided):** Headless execution for batch processing natively through the terminal.

## Setup Instructions

1. **Clone the repository** and open the directory.
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up Environment Variables:**
   Create a `.env` file in the root directory (you can copy `.env.example`).
   ```env
   LLM_API_KEY=your_gemini_api_key_here
   # Optional: GITHUB_TOKEN=your_github_token
   ```

## Run Instructions

### Option 1: FastAPI Interactive Wizard (Recommended for visualization)
Run the web server to access the step-by-step UI:
```bash
python main.py
```
Then open `http://127.0.0.1:8000` in your browser. You can click the **"🚀 Run Full Pipeline"** button to automatically execute all 4 phases in one click!

### Option 2: CLI Batch Mode
Run the pipeline headlessly via the command line:
```bash
python main.py --input ./resumes --output ./output/results.json
```

### Running Tests
To run the automated test suite (verifying API error handling and graceful fallbacks):
```bash
pytest tests/test_cases/test_api_errors.py
```

---

## Design Decisions

### Filtering & Scoring Strategy (Hybrid Approach)
To ensure absolute reliability and prevent the LLM from hallucinating high scores for unqualified candidates, the pipeline uses a **Hybrid Architecture**:
1. **Deterministic Base Scoring (Layer 2):** A strict Python script checks the `skills`, `projects`, and `work_experience` arrays extracted by the LLM. It applies hard eligibility filters (must have Python + AI evidence). If eligible, it deterministically awards up to 100 base points based on keyword and tech-stack mapping. 
2. **Semantic Review & Penalties (Layer 3):** The LLM is strictly used as a reviewer. It receives the base scores and the resume text, and is instructed that it may *only reduce* scores. It applies penalties for "thin wrappers", tutorial projects, and lack of implementation depth.

### LLM Usage
- **Model:** `gemini-2.0-flash` (via the new `google-genai` SDK).
- **Structured Output:** We strictly enforce Pydantic JSON schemas (`CandidateInfo`, `LLMReviewResult`) natively through the Gemini API to guarantee reliable machine-readable output.
- **Resiliency:** API calls are wrapped in bounded concurrency (`asyncio.Semaphore`) and include retry loops. If the LLM repeatedly fails validation for a specific resume, the pipeline gracefully falls back to the deterministic base scores without crashing the batch.

### GitHub Scoring
- We extract the GitHub username from the resume URL.
- We hit the public `api.github.com/users/{username}/repos` endpoint asynchronously during Phase 2.
- **Scoring Logic:** Up to 5 points are awarded for having recent public repositories, and up to 5 bonus points are awarded if those repositories primarily use Python or mention AI/LLM concepts.
- **Resiliency:** The HTTP client has a 5-second timeout. If the API rate limits us or fails, the candidate simply receives 0 GitHub points, and the batch continues successfully.

---

## If I Had More Time

Given more time, I would implement the following extra features:

1. **Dynamic Job Description (JD) Parsing:** I would add a feature to pass a custom JD to the LLM at runtime. The LLM would dynamically parse the JD to generate the strict Python scoring rules and keyword weights, rather than having them hardcoded. This would make the pipeline completely generic for *any* role or use case, rather than just this specific Python/AI intern role.
2. **Deep GitHub Code Review:** I would implement a GitHub reviewer agent. If a candidate mentions a specific project on their resume, the agent would pull the actual repository code from GitHub, evaluate the code quality, architecture, and implementation depth, and adjust the scores accordingly based on real code evidence.
3. **LinkedIn Experience Verification:** I would add a web-scraping or API module to check the candidate's LinkedIn handle and cross-verify if the internships and work experiences listed on the resume actually match their public LinkedIn profile, flagging discrepancies.
