"""
LLM Extraction Module
=====================
Sends parsed resume Markdown to the Gemini LLM to extract structured CandidateInfo.
Uses the new google-genai SDK for Pydantic structured output.
"""

import asyncio
from google import genai
from google.genai import types
from pydantic import ValidationError

from src.models import CandidateInfo
from config import LLM_API_KEY, LLM_MODEL

# Define the instruction prompt
EXTRACTION_PROMPT = """
You are an expert technical recruiter analyzing a software engineering candidate's resume.
Extract the structured information from the provided resume text according to the schema.

CRITICAL INSTRUCTION FOR NAME: 
If the candidate's name is missing, obscured, or replaced by an image tag in the text, 
you MUST infer their name from this filename: "{filename}". 
From this filename, ONLY extract the candidate's name and absolutely nothing else. Do NOT leave the name blank.

Resume Markdown:
{markdown_text}
"""

def get_client() -> genai.Client:
    """Initialize and return the Gemini API client."""
    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is not set in the environment or .env file.")
    return genai.Client(api_key=LLM_API_KEY)

async def extract_candidate_info(markdown_text: str, filename: str, max_retries: int = 3) -> CandidateInfo:
    """
    Asynchronously extracts structured CandidateInfo from raw resume markdown.
    Includes a retry mechanism if the LLM output fails Pydantic validation.
    """
    client = get_client()
    base_prompt = EXTRACTION_PROMPT.format(filename=filename, markdown_text=markdown_text)
    
    # Configure the API to strictly return our Pydantic schema as JSON
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CandidateInfo,
        temperature=0.1, # Low temperature for more deterministic extraction
    )
    
    current_prompt = base_prompt
    
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=LLM_MODEL,
                contents=current_prompt,
                config=config
            )
            
            # The new SDK parses it into response.parsed if a schema is provided
            if hasattr(response, 'parsed') and response.parsed is not None:
                return response.parsed
                
            # Fallback to manual JSON parsing if the SDK didn't auto-parse
            return CandidateInfo.model_validate_json(response.text)
            
        except ValidationError as e:
            print(f"⚠️ Pydantic Validation Error on attempt {attempt + 1}/{max_retries} for {filename}. Retrying...")
            if attempt == max_retries - 1:
                raise e
            # Tell the LLM what it did wrong so it can fix it
            current_prompt = base_prompt + f"\n\nYOUR PREVIOUS OUTPUT FAILED VALIDATION WITH THIS ERROR:\n{e}\n\nPlease fix the JSON structure."
            
        except Exception as e:
            error_msg = str(e)
            if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                raise Exception("Resource exhausted in API key. Please check your quota or try again later.")
                
            if attempt == max_retries - 1:
                raise e
            
            import asyncio
            delay = 3 # Standard 3s delay for generic errors
            print(f"⚠️ API/Traffic error on attempt {attempt + 1}/{max_retries} for {filename}: {e}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
