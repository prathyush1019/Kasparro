import httpx
import re
from typing import Optional, Dict, Any

async def fetch_github_enrichment(github_url: str) -> Dict[str, Any]:
    """
    Given a GitHub URL, extract the username and fetch their public repos.
    Returns a dict with a summary string and a score out of 10.
    """
    if not github_url:
        return {"github_summary": "No GitHub profile provided", "github_score": 0.0}

    # Extract username (e.g., https://github.com/username or github.com/username)
    match = re.search(r"github\.com/([^/]+)", github_url.lower())
    if not match:
        return {"github_summary": "Invalid GitHub URL format", "github_score": 0.0}
    
    username = match.group(1)

    try:
        # Use a short timeout so we don't hold up the pipeline
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Fetch repos (sorted by recently updated)
            response = await client.get(
                f"https://api.github.com/users/{username}/repos?sort=updated&per_page=5",
                headers={"Accept": "application/vnd.github.v3+json"}
            )
            
            # Handle rate limits or missing users gracefully
            if response.status_code != 200:
                return {
                    "github_summary": f"Failed to fetch GitHub data (HTTP {response.status_code})",
                    "github_score": 0.0
                }
            
            repos = response.json()
            if not repos:
                return {"github_summary": "GitHub profile exists but has no public repositories.", "github_score": 2.0}

            # Simple logic: Give points for having recent public repos, and bonus if they relate to Python/AI
            ai_python_repos = 0
            for r in repos:
                lang = (r.get("language") or "").lower()
                desc = (r.get("description") or "").lower()
                if lang == "python" or "ai" in desc or "llm" in desc or "gpt" in desc:
                    ai_python_repos += 1

            total_repos = len(repos)
            
            # Max 10 points: 
            # - Base 5 points for having active repos
            # - Up to 5 bonus points for Python/AI related repos
            score = 5.0 + min(5.0, ai_python_repos * 2.5)
            
            summary = f"Found {total_repos} recent public repos. {ai_python_repos} appear to be Python/AI related."
            return {"github_summary": summary, "github_score": score}

    except Exception as e:
        # Catch network timeouts, DNS errors, etc. without crashing the batch
        return {"github_summary": f"GitHub API error: {str(e)}", "github_score": 0.0}
