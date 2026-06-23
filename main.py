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
        if line.lower() in {"reply with your answer", "ask another",
                            "+ reply to chatgpt", "n90", "send"}:
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

    prompt = """You are a quiz solver. Read the question and answer options below carefully.
Identify the single correct answer and respond with ONLY the answer text itself.
Do NOT include the option letter (A/B/C/D). Do NOT include any explanation.
Keep it short - just the answer words, nothing else.

Example:
Question: What is the capital of France?
A) London  B) Berlin  C) Paris  D) Rome
Your response: Paris

Now answer this:
""" + cleaned

    try:
        async with httpx.AsyncClient(timeout=6) as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "google/gemini-2.5-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 30,
                },
            )
            r.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="OpenRouter timeout")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {e.response.status_code}")

    data = r.json()

    try:
        answer_text = data["choices"][0]["message"]["content"].strip()
        # Strip any leading letter the model still emits e.g. "C) Paris" -> "Paris"
        answer_text = re.sub(r'^[\(\[]?[A-Da-d][\)\]\.]\s*', '', answer_text).strip()
        if not answer_text:
            answer_text = "?"
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Unexpected OpenRouter response")

    return {"answer": answer_text}
