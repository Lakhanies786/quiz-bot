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
    """Clean OCR for quiz format"""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    cleaned_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        if line in ['A', 'B', 'C', 'D'] and i + 1 < len(lines):
            cleaned_lines.append(f"{line}) {lines[i+1]}")
            i += 2
        else:
            cleaned_lines.append(line)
            i += 1
    
    junk = {"reply", "ask another", "send", "correct", "incorrect"}
    final = [line for line in cleaned_lines if not any(j in line.lower() for j in junk)]
    return "\n".join(final)

async def web_search(query: str) -> str:
    """Search Google"""
    if not SERPER_KEY:
        return ""
    
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": 5},
            )
            r.raise_for_status()
            data = r.json()
            
            snippets = []
            if "answerBox" in data and "answer" in data["answerBox"]:
                snippets.append(data["answerBox"]["answer"])
            if "knowledgeGraph" in data and "description" in data["knowledgeGraph"]:
                snippets.append(data["knowledgeGraph"]["description"])
            
            for item in data.get("organic", [])[:3]:
                if "snippet" in item:
                    snippets.append(item["snippet"])
            
            return " ".join(snippets)
    except:
        return ""

def extract_options(cleaned: str) -> list:
    """Extract A, B, C, D options"""
    options = []
    for line in cleaned.split('\n'):
        match = re.match(r'^([A-D])\)\s*(.+)', line)
        if match:
            options.append({
                'letter': match.group(1),
                'text': match.group(2).strip()
            })
    return options

def find_best_match(search_context: str, options: list) -> str:
    """Find which option matches search results best"""
    if not search_context or not options:
        return ""
    
    search_lower = search_context.lower()
    best_match = None
    best_score = 0
    
    for opt in options:
        text_lower = opt['text'].lower()
        
        # Exact match or contained in search
        if text_lower in search_lower or search_lower in text_lower:
            return opt['text']
        
        # Word matching
        words = set(text_lower.split())
        search_words = set(search_lower.split())
        overlap = len(words & search_words)
        
        if overlap > best_score:
            best_score = overlap
            best_match = opt['text']
    
    return best_match if best_score >= 2 else ""

@app.get("/")
async def health():
    return {
        "status": "SmartBot running",
        "model": ANTHROPIC_MODEL,
        "api_key_loaded": bool(ANTHROPIC_API_KEY),
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
    
    options = extract_options(cleaned)
    
    if not options:
        return {"answer": ""}
    
    # For 2026 questions, ALWAYS search and match
    if any(kw in cleaned.lower() for kw in ["2026", "2025"]):
        first_line = cleaned.split("\n")[0]
        search_context = await web_search(first_line)
        
        if search_context:
            # Find which option matches search results
            matched = find_best_match(search_context, options)
            if matched:
                return {"answer": matched}
    
    # Fallback: Ask Claude to pick (for older questions)
    prompt = f"""Pick the correct answer from these options:
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
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Error")
            
            data = r.json()
            answer_text = data["content"][0]["text"].strip()
            answer_text = re.sub(r'^[A-D]\)\s*', '', answer_text).strip()
            
            return {"answer": answer_text}

    except:
        raise HTTPException(status_code=502, detail="Error")
