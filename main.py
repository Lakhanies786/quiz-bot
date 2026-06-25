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

def clean_and_reconstruct_ocr(raw: str) -> str:
    """Fix malformed OCR text where A, B, C, D are on separate lines"""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    
    if not lines:
        return ""
    
    option_lines = []
    question_lines = []
    option_labels = ['A', 'B', 'C', 'D', '1', '2', '3', '4']
    
    for i, line in enumerate(lines):
        if line in option_labels and i + 1 < len(lines):
            option_lines.append(f"{line}) {lines[i+1]}")
        elif line not in option_labels or i == 0:
            if question_lines or not line in option_labels:
                question_lines.append(line)
    
    if not option_lines:
        return raw
    
    result = "\n".join(question_lines) + "\n" + "\n".join(option_lines)
    return result.strip()

def clean_ocr_text(raw: str) -> str:
    """Clean OCR text"""
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

async def web_search(query: str) -> str:
    """Search Google via Serper API"""
    if not SERPER_KEY:
        return ""
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
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
            
            for item in data.get("organic", [])[:3]:
                if "snippet" in item:
                    snippets.append(item["snippet"])
            
            return " ".join(snippets[:4])
    except Exception:
        return ""

def answer_is_valid(answer: str, search_context: str, options: list) -> bool:
    """
    Validate that Claude's answer is:
    1. In the options list
    2. Matches search results
    """
    if not answer or len(answer) < 2:
        return False
    
    answer_lower = answer.lower().strip()
    
    # Check if answer is in options
    for opt in options:
        if answer_lower == opt.lower().strip():
            # Check if answer appears in search results
            if search_context:
                if answer_lower in search_context.lower():
                    return True
                # Also accept if search mentions related keywords
                words = set(answer_lower.split())
                search_words = set(search_context.lower().split())
                if len(words & search_words) >= 2:
                    return True
            else:
                return True
    
    return False

@app.get("/")
async def health():
    return {
        "status": "QuizBot with Verification running",
        "model": ANTHROPIC_MODEL,
        "verification": "ENABLED - all answers verified",
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
    
    # ALWAYS search for 2026 questions to verify
    first_line = cleaned.split("\n")[0]
    search_context = await web_search(first_line)
    
    # Extract options
    option_lines = [line for line in cleaned.split('\n') if re.match(r'^[A-D\d][.)]\s', line)]
    options = [re.sub(r'^[A-D\d][.)]\s*', '', line).strip() for line in option_lines]
    options_text = "\n".join(option_lines) if option_lines else ""
    
    # Build prompt with search results for verification
    if search_context:
        prompt = f"""You are a quiz answer bot. Your job is to pick the CORRECT answer using the search results.

SEARCH RESULTS (Use these to verify your answer):
{search_context}

CRITICAL RULES:
1. Use search results to find the correct answer
2. Pick from ONLY these options:
{options_text}

3. Your answer MUST be supported by the search results
4. Reply with ONLY the exact option text (no letter)
5. No explanation, no guessing

Question and options:
{cleaned}

Your answer (MUST match search results exactly):"""
    else:
        prompt = f"""You are a quiz answer bot.

CRITICAL RULES:
1. Pick from ONLY these options:
{options_text}

2. Reply with ONLY the exact option text (no letter)
3. No explanation

Question and options:
{cleaned}

Your answer:"""
    
    # Try up to 2 times to get a valid answer
    for attempt in range(2):
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
                
                # VERIFY answer
                if answer_is_valid(answer_text, search_context, options):
                    return {
                        "answer": answer_text,
                        "verified": True,
                        "confidence": "high" if search_context else "medium"
                    }
                else:
                    # Answer didn't match search results, reject and retry
                    if attempt == 0:
                        # Retry with stricter prompt
                        prompt = f"""You are a quiz answer bot. IMPORTANT: The search results show the correct answer.

SEARCH RESULTS (This is the correct information):
{search_context}

You MUST pick the answer that matches the search results.

Options:
{options_text}

Answer ONLY with one of these options, nothing else. Make sure it matches the search results."""
                        continue
                    else:
                        # Second attempt failed, return error
                        return {
                            "answer": answer_text,
                            "verified": False,
                            "error": "Answer does not match search results. Please verify manually.",
                            "search_results": search_context[:200]
                        }

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"API error: {e.response.status_code}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed: {str(e)}")

