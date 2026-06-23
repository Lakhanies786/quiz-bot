from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

API_KEY = os.getenv("API_KEY")  # Set this in Railway → Variables


class Question(BaseModel):
    text: str


@app.get("/")
async def health():
    return {"status": "QuizBot backend running"}


@app.post("/answer")
async def answer(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")

    prompt = f"""You are a football quiz solver. Below is raw OCR text captured from a
quiz screen. It contains a question followed by multiple-choice options
(they may be labelled A/B/C/D, or just listed as separate lines/numbers,
or run together — OCR is imperfect, so infer the structure yourself).

Read the OCR text, figure out the question and the options, and pick the
correct option.

Respond with ONLY the single letter A, B, C, or D corresponding to the
correct option's position (1st option = A, 2nd = B, 3rd = C, 4th = D).
No explanation. No punctuation. No words. Just the one letter.

OCR TEXT:
{q.text}
"""

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
                    "max_tokens": 10,
                },
            )
            r.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="OpenRouter timeout")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {e.response.status_code}")

    data = r.json()

    try:
        raw = data["choices"][0]["message"]["content"].strip().upper()
        # Sanitise — scan for the first A/B/C/D character anywhere in the
        # response, since the model may prepend whitespace/punctuation
        letter = next((ch for ch in raw if ch in "ABCD"), "?")
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Unexpected OpenRouter response")

    return {"answer": letter}
