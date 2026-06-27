from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re
import asyncio

app = FastAPI()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPER_KEY = os.getenv("SERPER_KEY", "")

# Correct model strings
MODEL_OCR = "claude-haiku-4-5-20251001"
MODEL_REASON = "claude-sonnet-4-6"


class Question(BaseModel):
    text: str

class ImageQuestion(BaseModel):
    image: str


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


async def anthropic_call(model: str, messages: list, max_tokens: int) -> str:
    async with httpx.AsyncClient(timeout=12) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            },
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()


def parse_options(lines: list) -> tuple:
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

    # Step 1: OCR with Haiku — extract text from image
    try:
        extracted_text = await anthropic_call(
            model=MODEL_OCR,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": q.image}
                    },
                    {
                        "type": "text",
                        "text": "Extract all text from this image exactly as it appears. List each line separately. Output only the extracted text, nothing else."
                    }
                ]
            }],
            max_tokens=300
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR failed: {e}")

    # Parse options from extracted text
    lines = [l.strip() for l in extracted_text.splitlines() if l.strip()]
    options, cleaned_lines = parse_options(lines)
    cleaned = "\n".join(cleaned_lines)

    if not options:
        raise HTTPException(status_code=400, detail=f"No options in: {extracted_text[:200]}")

    # Step 2: Run web search and reasoning TRULY in parallel
    question_line = next((l for l in cleaned_lines if "?" in l), cleaned_lines[0] if cleaned_lines else "")

    def make_reason_prompt(ctx: str) -> str:
        base = "Text extracted from image:\n" + cleaned + "\n\nWhich single option (A, B, C, or D) is factually correct?\nReply with ONLY the letter. Nothing else."
        if ctx:
            return "Reference facts:\n" + ctx + "\n\n" + base
        return base

    # Run both simultaneously — search for context, reason without it
    search_result, reason_result = await asyncio.gather(
        web_search(question_line),
        anthropic_call(
            model=MODEL_REASON,
            messages=[{"role": "user", "content": make_reason_prompt("")}],
            max_tokens=5
        ),
        return_exceptions=True
    )

    # Use search-enhanced reasoning only if search returned useful data
    # and reasoning without search gave a valid letter
    search_context = search_result if isinstance(search_result, str) else ""
    raw_letter = reason_result if isinstance(reason_result, str) else ""

    # If search has data, do one more quick reasoning call with context
    # (Sonnet is fast enough at max_tokens=5)
    if search_context:
        try:
            refined = await anthropic_call(
                model=MODEL_REASON,
                messages=[{"role": "user", "content": make_reason_prompt(search_context)}],
                max_tokens=5
            )
            raw_letter = refined  # prefer search-informed answer
        except Exception:
            pass  # keep original raw_letter

    letter_match = re.search(r'\b([A-D])\b', raw_letter.upper())
    if not letter_match:
        raise HTTPException(status_code=502, detail=f"No letter in response: '{raw_letter}'")

    letter = letter_match.group(1)
    answer_text = options.get(letter, "")
    return {"letter": letter, "answer": answer_text, "model": f"{MODEL_OCR}+{MODEL_REASON}"}


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

    prompt = ("Reference facts:\n" + search_context + "\n\n" if search_context else "") + \
             "Which single option (A, B, C, or D) is factually correct?\nReply with ONLY the letter.\n\n" + cleaned

    try:
        raw = await anthropic_call(MODEL_REASON, [{"role": "user", "content": prompt}], 5)
        m = re.search(r'\b([A-D])\b', raw.upper())
        if not m:
            raise HTTPException(status_code=502, detail=f"No letter: {raw}")
        letter = m.group(1)
        return {"answer": options.get(letter, letter), "letter": letter}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
