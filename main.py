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


def clean_ocr(text: str) -> str:
    """Remove obvious junk lines, keep question + all option texts."""
    junk = [
        "reply", "time is up", "see score", "points", "next question",
        "ask another", "send", "correct!", "incorrect!", "question of"
    ]
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip standalone single letters A-D and X (OCR artifacts from option circles)
        if re.match(r'^[A-DX]$', line):
            continue
        # Skip junk
        if any(j in line.lower() for j in junk):
            continue
        lines.append(line)
    return "\n".join(lines)


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


async def claude_answer(cleaned: str, search_context: str) -> str:
    """Ask Claude to return the correct answer text directly — no letter."""
    prompt = (
        ("Reference facts from web search:\n" + search_context + "\n\n" if search_context else "") +
        "Below is text extracted from a quiz question screen.\n"
        "Identify the question and the answer options.\n"
        "Reply with ONLY the exact text of the correct answer option — nothing else.\n"
        "No letter prefix, no explanation, just the answer words as they appear.\n\n"
        + cleaned
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
                "max_tokens": 30,
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
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    cleaned = clean_ocr(q.text)

    if not cleaned:
        raise HTTPException(status_code=400, detail="No usable text")

    # Extract question line for search
    question_line = next(
        (l for l in cleaned.splitlines() if "?" in l),
        cleaned.splitlines()[0] if cleaned.splitlines() else ""
    )

    # Run search and reasoning in parallel
    search_result, reason_result = await asyncio.gather(
        web_search(question_line),
        claude_answer(cleaned, ""),
        return_exceptions=True
    )

    search_context = search_result if isinstance(search_result, str) else ""
    answer_text = reason_result if isinstance(reason_result, str) else ""

    # If search returned data, refine
    if search_context:
        try:
            answer_text = await claude_answer(cleaned, search_context)
        except Exception:
            pass

    # Reject obvious refusals
    if not answer_text or any(x in answer_text.lower() for x in [
        "i cannot", "i'm sorry", "as an ai", "i don't", "i am unable"
    ]):
        raise HTTPException(status_code=502, detail=f"Bad response: {answer_text}")

    return {"answer": answer_text}
