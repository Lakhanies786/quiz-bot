from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re

app = FastAPI()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
SERPER_KEY = os.getenv("SERPER_KEY", "")

class Question(BaseModel):
    text: str

def clean_ocr_text(raw: str) -> str:
    """Clean OCR for this specific quiz format"""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    
    # Reconstruct malformed options (A on separate line from text)
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # If line is just a letter (A, B, C, D) and next line exists, merge them
        if line in ['A', 'B', 'C', 'D'] and i + 1 < len(lines):
            cleaned_lines.append(f"{line}) {lines[i+1]}")
            i += 2
        else:
            cleaned_lines.append(line)
            i += 1
    
    # Filter junk
    junk = {"reply with your answer", "ask another", "send", "correct!", "incorrect!"}
    final = [line for line in cleaned_lines if not any(j in line.lower() for j in junk)]
    
    return "\n".join(final)

async def web_search(query: str) -> str:
    """Search for current/2026 questions"""
    if not SERPER_KEY:
        return ""
    
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": 3},
            )
            r.raise_for_status()
            data = r.json()
            
            snippets = []
            if "answerBox" in data:
                if "answer" in data["answerBox"]:
                    snippets.append(data["answerBox"]["answer"])
            
            for item in data.get("organic", [])[:2]:
                if "snippet" in item:
                    snippets.append(item["snippet"])
            
            return " | ".join(snippets)
    except:
        return ""

@app.get("/")
async def health():
    return {
        "status": "QuizBot running",
        "model": ANTHROPIC_MODEL,
        "api_key_loaded": bool(ANTHROPIC_API_KEY),
        "search_enabled": bool(SERPER_KEY)
    }

@app.post("/answer")
async def answer(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty")
    
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="No API key")

    cleaned = clean_ocr_text(q.text)
    
    if not cleaned.strip():
        return {"answer": ""}
    
    # Get search context for 2026/current questions
    search_context = ""
    if any(kw in cleaned.lower() for kw in ["2026", "2025", "current", "latest"]):
        first_line = cleaned.split("\n")[0]
        search_context = await web_search(first_line)
    
    # Build simple, clear prompt
    if search_context:
        prompt = f"""Based on this information: {search_context}

Pick the correct answer from these options:
{cleaned}

Reply ONLY with the exact option text. Nothing else."""
    else:
        prompt = f"""Pick the correct answer:
{cleaned}

Reply ONLY with the exact option text. Nothing else."""
    
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 40,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Error: {r.status_code}")
            
            data = r.json()
            answer_text = data["content"][0]["text"].strip()
            
            # Clean up letter prefix if any
            answer_text = re.sub(r'^[A-D]\)\s*', '', answer_text).strip()
            
            return {"answer": answer_text}

    except httpx.HTTPStatusError:
        raise HTTPException(status_code=502, detail="API error")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error: {str(e)}")
