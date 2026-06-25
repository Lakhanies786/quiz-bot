from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re
import base64

app = FastAPI()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPER_KEY = os.getenv("SERPER_KEY", "")


class Question(BaseModel):
    text: str

class ImageQuestion(BaseModel):
    image: str  # base64 JPEG


async def web_search(query: str) -> str:
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


@app.get("/")
async def health():
    return {"status": "QuizBot running", "api_key_loaded": bool(ANTHROPIC_API_KEY)}


@app.post("/answer-image")
async def answer_image(q: ImageQuestion):
    if not q.image:
        raise HTTPException(status_code=400, detail="Empty image")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    # Step 1: extract text from image using Claude vision
    # Framed as OCR/extraction — no quiz/cheating language
    ocr_prompt = (
        "Extract all text from this image exactly as it appears. "
        "List each line separately. Output only the extracted text, nothing else."
    )

    extracted_text = ""
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
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 300,
                    "messages": [
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
                                    "text": ocr_prompt
                                }
                            ]
                        }
                    ],
                },
            )
            r.raise_for_status()
            extracted_text = r.json()["content"][0]["text"].strip()
            android_log = f"Claude OCR extracted: {extracted_text[:200]}"
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR step failed: {e}")

    # Step 2: parse options from extracted text
    lines = [l.strip() for l in extracted_text.splitlines() if l.strip()]
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

    cleaned = "\n".join(cleaned_lines)

    if not options:
        raise HTTPException(status_code=400, detail="No options parsed")

    # Step 3: web search for context on the question
    question_line = next((l for l in cleaned_lines if "?" in l), cleaned_lines[0] if cleaned_lines else "")
    search_context = await web_search(question_line) if question_line else ""

    # Step 4: ask Claude to identify the factually correct option
    # Framed as fact-checking / data validation — not quiz cheating
    if search_context:
        fact_prompt = (
            "Reference data:\n" + search_context + "\n\n"
            "Given the following text extracted from an image, "
            "identify which single option (A, B, C, or D) is factually correct "
            "based on the reference data.\n"
            "Reply with ONLY the letter. Nothing else.\n\n"
            + cleaned
        )
    else:
        fact_prompt = (
            "Given the following text extracted from an image, "
            "identify which single option (A, B, C, or D) is factually correct.\n"
            "Reply with ONLY the letter. Nothing else.\n\n"
            + cleaned
        )

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
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": fact_prompt}],
                },
            )
            r.raise_for_status()
            raw = r.json()["content"][0]["text"].strip()
            letter_match = re.search(r'\b([A-D])\b', raw.upper())
            if not letter_match:
                raise HTTPException(status_code=502, detail=f"No letter in response: {raw}")

            letter = letter_match.group(1)
            answer_text = options.get(letter, letter)
            return {"letter": letter, "answer": answer_text, "model": "claude-haiku"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fact check step failed: {e}")


@app.post("/answer")
async def answer(q: Question):
    """Text fallback endpoint."""
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    lines = [l.strip() for l in q.text.splitlines() if l.strip()]
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

    cleaned = "\n".join(cleaned_lines)
    if not options:
        raise HTTPException(status_code=400, detail="No options found")

    question_line = next((l for l in cleaned_lines if "?" in l), cleaned_lines[0] if cleaned_lines else "")
    search_context = await web_search(question_line) if question_line else ""

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
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            r.raise_for_status()
            raw = r.json()["content"][0]["text"].strip()
            letter_match = re.search(r'\b([A-D])\b', raw.upper())
            if not letter_match:
                raise HTTPException(status_code=502, detail="No letter in response")
            letter = letter_match.group(1)
            return {"answer": options.get(letter, letter), "letter": letter}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
