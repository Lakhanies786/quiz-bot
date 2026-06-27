from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re
import asyncio

app = FastAPI()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPER_KEY = os.getenv("SERPER_KEY", "")

MODEL_REASON = "claude-sonnet-4-6"


class Question(BaseModel):
    text: str


def parse_options(text: str) -> tuple:
    """Returns (options dict, cleaned text, question line)"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    options = {}
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Handle bare letter on its own line (A, B, C, D) followed by answer
        if line in ['A', 'B', 'C', 'D'] and i + 1 < len(lines):
            options[line] = lines[i + 1].strip()
            cleaned_lines.append(f"{line}) {lines[i+1].strip()}")
            i += 2
        else:
            m = re.match(r'^([A-D])[.)]\s*(.+)', line)
            if m:
                options[m.group(1)] = m.group(2).strip()
            cleaned_lines.append(line)
            i += 1

    cleaned = "\n".join(cleaned_lines)
    question_line = next((l for l in cleaned_lines if "?" in l), cleaned_lines[0] if cleaned_lines else "")
    return options, cleaned, question_line


async def web_search(query: str) -> str:
    if not SERPER_KEY or not query:
        return ""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": 5},
            )
            r.raise_for_status()
            data = r.json()
            snippets = []
            if "answerBox" in data:
                ab = data["answerBox"]
                snippets.append(ab.get("answer") or ab.get("snippet") or "")
            if "knowledgeGraph" in data:
                snippets.append(data["knowledgeGraph"].get("description", ""))
            for item in data.get("organic", [])[:3]:
                if "snippet" in item:
                    snippets.append(item["snippet"])
            return " ".join(s for s in snippets if s)[:600]
    except Exception:
        return ""


async def claude_reason(cleaned: str, search_context: str) -> str:
    """Single Claude Sonnet call — returns just the letter."""
    prompt = (
        ("Reference facts:\n" + search_context + "\n\n" if search_context else "") +
        "Text:\n" + cleaned + "\n\n"
        "Which single option (A, B, C, or D) is factually correct?\n"
        "Reply with ONLY the letter. Nothing else."
    )
    async with httpx.AsyncClient(timeout=12) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_REASON,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()


@app.get("/")
async def health():
    return {"status": "QuizBot running", "api_key_loaded": bool(ANTHROPIC_API_KEY)}


@app.post("/answer")
async def answer(q: Question):
    """
    Receives OCR text from ML Kit (on-device).
    Runs Serper search + Claude Sonnet reasoning in parallel.
    Returns letter + full answer text.
    """
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    options, cleaned, question_line = parse_options(q.text)

    if not options:
        raise HTTPException(status_code=400, detail=f"No options found in: {q.text[:100]}")

    # Run Serper search and Claude reasoning in parallel
    search_result, reason_result = await asyncio.gather(
        web_search(question_line),
        claude_reason(cleaned, ""),   # start immediately without search
        return_exceptions=True
    )

    search_context = search_result if isinstance(search_result, str) else ""
    raw_letter = reason_result if isinstance(reason_result, str) else ""

    # If search returned data, do one refined call with context
    if search_context and isinstance(reason_result, str):
        try:
            refined = await claude_reason(cleaned, search_context)
            raw_letter = refined
        except Exception:
            pass  # keep original

    letter_match = re.search(r'\b([A-D])\b', raw_letter.upper())
    if not letter_match:
        raise HTTPException(status_code=502, detail=f"No letter in: '{raw_letter}'")

    letter = letter_match.group(1)
    answer_text = options.get(letter, "")
    return {"letter": letter, "answer": answer_text}
