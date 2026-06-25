from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re

app = FastAPI()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

class Question(BaseModel):
    text: str

def clean_and_reconstruct_ocr(raw: str) -> str:
    """
    Fix malformed OCR text where A, B, C, D are on separate lines.
    Reconstruct it into: A) Option Text format
    """
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    
    if not lines:
        return ""
    
    # Find where options start (usually after question)
    option_lines = []
    question_lines = []
    option_labels = ['A', 'B', 'C', 'D', '1', '2', '3', '4']
    
    for i, line in enumerate(lines):
        # If this line is JUST a letter, and next line exists, merge them
        if line in option_labels and i + 1 < len(lines):
            option_lines.append(f"{line}) {lines[i+1]}")
        elif line not in option_labels or i == 0:
            # If it's a label but first line, or not a label, add to question
            if question_lines or not line in option_labels:
                question_lines.append(line)
    
    # If we didn't find properly formatted options, just return original
    if not option_lines:
        return raw
    
    # Reconstruct
    result = "\n".join(question_lines) + "\n" + "\n".join(option_lines)
    return result.strip()

def clean_ocr_text(raw: str) -> str:
    """Clean OCR text"""
    # First try to fix malformed options
    fixed = clean_and_reconstruct_ocr(raw)
    
    lines = fixed.splitlines()
    cleaned = []
    found_option = False
    
    junk_phrases = {
        "reply with your answer", "ask another", "send", "choose the correct answer",
        "correct!", "incorrect!", "reply with a, b, c, or d"
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if any(junk in line.lower() for junk in junk_phrases):
            continue
        
        is_option = bool(re.match(r'^[A-D\d][.)]\s', line))
        if is_option:
            found_option = True
        
        if not found_option and not is_option and len(line) < 15 and "?" not in line:
            continue
        
        cleaned.append(line)
    
    return "\n".join(cleaned)

@app.get("/")
async def health():
    return {
        "status": "QuizBot running",
        "model": ANTHROPIC_MODEL,
        "api_key_loaded": bool(ANTHROPIC_API_KEY),
    }

@app.post("/answer")
async def answer(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="No ANTHROPIC_API_KEY set")

    cleaned = clean_ocr_text(q.text)
    
    if not cleaned.strip():
        return {"answer": ""}
    
    prompt = f"""You are a quiz answer bot. Your job is to pick the ONLY correct answer from the given options.

IMPORTANT RULES:
1. Reply with ONLY the exact text of the correct answer option
2. Do NOT include the letter (A, B, C, D) or number (1, 2, 3)
3. Do NOT include any explanation
4. Just the answer text

Question and options:
{cleaned}"""
    
    try:
        async with httpx.AsyncClient(timeout=15) as client:
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
                raise HTTPException(status_code=502, detail=f"API error: {r.status_code}")
            
            data = r.json()
            answer_text = data["content"][0]["text"].strip()
            
            # Strip letter prefix if present
            answer_text = re.sub(r'^[A-D\d][.)]\s*', '', answer_text).strip()
            
            return {"answer": answer_text}

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"API error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed: {str(e)}")
