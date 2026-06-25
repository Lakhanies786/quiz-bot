from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re
import asyncio

app = FastAPI()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPER_KEY = os.getenv("SERPER_KEY", "")


class Question(BaseModel):
    text: str

class ImageQuestion(BaseModel):
    image: str  # base64 JPEG


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
            return " ".join(s for s in snippets if s)
    except Exception:
        return ""


async def claude_call(messages: list, max_tokens: int = 10) -> str:
    async with httpx.AsyncClient(timeout=7) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": max_tokens,
                "messages": messages,
            },
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()


def parse_options(lines: list) -> tuple[dict, list]:
    """Returns (options dict, cleaned lines list)"""
    options = {}
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line in ['A', 'B', 'C', 'D'] and i + 1 < len(lines):
            options[line] = lines[i + 1].strip()
            cleaned_lines.append(f"{line}) {lines[i+1].strip()}")
            i += 2
        else:
            match = re.match(r'^([A-D])[.)]\s*(.+)', line)
            if match:
                options[match.group(1)] = match.group(2).strip()
            cleaned_lines.append(line)
            i += 1
    return options, cleaned_lines


@app.get("/")
async def health():
    return {"status": "QuizBot running", "api_key_loaded": bool(ANTHROPIC_API_KEY)}


@app.post("/answer-image")
async def answer_image(q: ImageQuestion):
    if not q.image:
        raise HTTPException(status_code=400, detail="Empty image")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    # ── Single Claude call: OCR + identify correct option in one shot ──────
    # Serper needs the question text first, so we do OCR first (fast),
    # then fire Serper + fact-check in parallel.

    # Step 1: OCR only — fast, just extract text (no reasoning yet)
    ocr_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": q.image,
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Extract all text from this image exactly as it appears. "
                        "List each line separately. Output only the extracted text, nothing else."
                    )
                }
            ]
        }
    ]

    try:
        extracted_text = await claude_call(ocr_messages, max_tokens=300)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR failed: {e}")

    # Parse options and question line
    lines = [l.strip() for l in extracted_text.splitlines() if l.strip()]
    options, cleaned_lines = parse_options(lines)
    cleaned = "\n".join(cleaned_lines)

    if not options:
        raise HTTPException(status_code=400, detail="No options parsed")

    question_line = next((l for l in cleaned_lines if "?" in l), cleaned_lines[0] if cleaned_lines else "")

    # Step 2: Serper search + Claude fact-check IN PARALLEL
    async def fact_check(search_context: str) -> str:
        if search_context:
            prompt = (
                "Reference data:\n" + search_context + "\n\n"
                "From the text below, identify which single option (A, B, C, or D) "
                "is factually correct based on the reference data.\n"
                "Reply with ONLY the letter. Nothing else.\n\n"
                + cleaned
            )
        else:
            prompt = (
                "From the text below, identify which single option (A, B, C, or D) "
                "is factually correct.\n"
                "Reply with ONLY the letter. Nothing else.\n\n"
                + cleaned
            )
        return await claude_call(
            [{"role": "user", "content": prompt}],
            max_tokens=5
        )

    # Run Serper search and fact-check simultaneously
    search_context, raw_letter = await asyncio.gather(
        web_search(question_line),
        fact_check("")  # start immediately without search
    )

    # If search returned results and letter seems uncertain, re-check with context
    # (only if search finished fast enough — which it usually does)
    letter_match = re.search(r'\b([A-D])\b', raw_letter.upper())
    if not letter_match:
        raise HTTPException(status_code=502, detail=f"No letter in response: {raw_letter}")

    letter = letter_match.group(1)

    # If we got search context, do one more fast check with it
    # This adds ~1s but only when serper has useful data
    if search_context:
        try:
            refined = await fact_check(search_context)
            refined_match = re.search(r'\b([A-D])\b', refined.upper())
            if refined_match:
                letter = refined_match.group(1)
        except Exception:
            pass  # keep original letter

    answer_text = options.get(letter, letter)
    return {"letter": letter, "answer": answer_text, "model": "claude-haiku"}


@app.post("/answer")
async def answer(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    lines = [l.strip() for l in q.text.splitlines() if l.strip()]
    options, cleaned_lines = parse_options(lines)
    cleaned = "\n".join(cleaned_lines)

    if not options:
        raise HTTPException(status_code=400, detail="No options found")

    question_line = next((l for l in cleaned_lines if "?" in l), cleaned_lines[0] if cleaned_lines else "")
    search_context = await web_search(question_line)

    if search_context:
        prompt = (
            "Reference data:\n" + search_context + "\n\n"
            "From the text below, identify which option (A/B/C/D) is factually correct.\n"
            "Reply with ONLY the letter.\n\n" + cleaned
        )
    else:
        prompt = (
            "From the text below, identify which option (A/B/C/D) is factually correct.\n"
            "Reply with ONLY the letter.\n\n" + cleaned
        )

    try:
        raw = await claude_call([{"role": "user", "content": prompt}], max_tokens=5)
        letter_match = re.search(r'\b([A-D])\b', raw.upper())
        if not letter_match:
            raise HTTPException(status_code=502, detail="No letter in response")
        letter = letter_match.group(1)
        return {"answer": options.get(letter, letter), "letter": letter}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
