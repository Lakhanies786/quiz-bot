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

    prompt = """You are a precise quiz answer bot with strong general knowledge.
Read the question and ALL options very carefully before answering.
Think step by step about which option is factually correct.
Then reply with ONLY the exact text of the correct answer option — copied exactly from the options.
No letter prefix (no A/B/C/D), no explanation, just the answer words.

Important: For sports, geography, and current events questions — trust the most recent known facts.

Question and options:
""" + cleaned

    # gemini-2.5-flash first (fast + accurate), lite as fallback only
    models = [
        "google/gemini-2.5-flash",
        "google/gemini-2.0-flash-lite",
    ]

    last_error = None
    for model in models:
        try:
            async with httpx.AsyncClient(timeout=4) as client:  # reduced from 5s
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 30,   # reduced from 40 — answer is short
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
