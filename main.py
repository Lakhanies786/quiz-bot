from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re

app = FastAPI()

# Anthropic API Key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# Optional: web search (costs $0.01 per search). Set to False to disable.
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
SERPER_KEY = os.getenv("SERPER_KEY", "")


class Question(BaseModel):
    text: str


def clean_ocr_text(raw: str) -> str:
    """Strip stale overlay text and junk"""
    lines = raw.splitlines()
    cleaned = []
    found_option = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip low-quality noise
        if line.lower() in {"send", "reply with a, b, c, or d", "choose the correct answer", 
                             "correct!", "incorrect!", "reply with your answer"}:
            continue
        
        is_option = bool(re.match(r'^[A-D][.)]\s', line))
        if is_option:
            found_option = True
        
        # Before first option, skip short junk (stale answer bubbles)
        if not found_option and not is_option and len(line) <= 20 and "?" not in line:
            continue
        
        cleaned.append(line)
    
    return "\n".join(cleaned)


async def web_search(query: str) -> str:
    """Optional web search. Costs $0.01 per call."""
    if not ENABLE_WEB_SEARCH or not SERPER_KEY:
        return ""
    
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": 3},
            )
            r.raise_for_status()
            data = r.json()
            snippets = []
            
            if "answerBox" in data and "answer" in data["answerBox"]:
                snippets.append(data["answerBox"]["answer"])
            
            for item in data.get("organic", [])[:2]:
                if "snippet" in item:
                    snippets.append(item["snippet"])
            
            return " ".join(snippets[:2])
    except Exception:
        return ""


@app.get("/")
async def health():
    return {
        "status": "QuizBot running",
        "model": ANTHROPIC_MODEL,
        "web_search_enabled": ENABLE_WEB_SEARCH,
        "api_key_loaded": bool(ANTHROPIC_API_KEY),
    }


@app.post("/answer")
async def answer(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="No ANTHROPIC_API_KEY set")

    cleaned = clean_ocr_text(q.text)
    
    # Get search context only if enabled (costs $0.01)
    search_context = ""
    if ENABLE_WEB_SEARCH:
        first_line = cleaned.split("\n")[0]
        search_context = await web_search(first_line)

    # Build minimal prompt
    if search_context:
        prompt = f"Search: {search_context}\n\nPick the correct answer. Reply ONLY the exact option text, no letter.\n\n{cleaned}"
    else:
        prompt = f"Pick the correct answer from the options. Reply ONLY the exact text of the correct option, no letter or explanation.\n\n{cleaned}"

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 30,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            r.raise_for_status()
            data = r.json()
            answer_text = data["content"][0]["text"].strip()
            
            # Remove accidental letter prefix if present
            answer_text = re.sub(r'^[A-D][.)]\s*', '', answer_text).strip()
            
            return {"answer": answer_text}

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"API error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed: {e}")
