import asyncio
import json
from pathlib import Path
from src.extractor import extract_candidate_info

async def main():
    print("Testing LLM Extractor...")
    
    # 1. Read the test resume we created earlier
    test_file = Path("test_resumes/sample.txt")
    if not test_file.exists():
        print("Test resume not found. Please ensure test_resumes/sample.txt exists.")
        return
        
    markdown_text = test_file.read_text(encoding="utf-8")
    filename = test_file.name
    
    print(f"Sending '{filename}' to Gemini API for structured extraction...")
    
    try:
        # 2. Call the LLM extractor
        result = await extract_candidate_info(markdown_text, filename)
        
        # 3. Print the resulting Pydantic model as beautifully formatted JSON
        print("\n" + "="*50)
        print("✨ LLM EXTRACTION SUCCESS ✨")
        print("="*50)
        print(result.model_dump_json(indent=4))
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        print("Make sure you have added a valid Gemini API key to your .env file!")

if __name__ == "__main__":
    asyncio.run(main())
