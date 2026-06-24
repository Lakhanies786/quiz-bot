from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re

app = FastAPI()

API_KEY = os.getenv("API_KEY")


class Question(BaseModel):
    text: str


def clean_ocr_text(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        alnum_ratio = sum(c.isalnum() or c.isspace() for c in line) / len(line)
        if alnum_ratio < 0.5:
            continue
        junk = {
            "reply with your answer", "ask another", "+ reply to chatgpt",
            "n90", "send", "reply with a, b, c, or d", "reply with a b c or d",
            "choose the correct answer", "select one", "pick one"
        }
        if line.lower() in junk:
            continue
        if line.lower().startswith("reply with"):
            continue
        if re.fullmatch(r'\d{1,2}:\d{2}', line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


@app.get("/")
async def health():
    return {"status": "QuizBot backend running"}


@app.post("/answer")
async def answer(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")

    cleaned = clean_ocr_text(q.text)

    system_prompt = (
        "You are a quiz answer bot. "
        "Output only the correct answer option text — nothing else. "
        "No letter prefix (A/B/C/D), no explanation. "
        "Just the exact words of the correct option as written."
    )

    user_prompt = (
        "Read the question and options carefully. "
        "Reply with ONLY the text of the correct option, no letter prefix.\n\n"
        + cleaned
    )

    # Primary: accurate model. Fallback: lite (for when primary times out)
    models = [
        "google/gemini-2.5-flash",
        "google/gemini-2.0-flash-lite",
    ]

    last_error = None
    for model in models:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0,
                        "max_tokens": 50,
                        "stream": False,
                    },
                )
                r.raise_for_status()
                data = r.json()
                answer_text = data["choices"][0]["message"]["content"].strip()
                answer_text = re.sub(r'^[\(\[]?[A-Da-d][\)\]\.]?\s*', '', answer_text).strip()
                if answer_text:
                    return {"answer": answer_text, "model": model}
        except Exception as e:
            last_error = e
            continue

    raise HTTPException(status_code=502, detail=f"All models failed: {last_error}")
