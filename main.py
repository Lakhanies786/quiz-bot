from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

API_KEY = os.getenv("API_KEY")

class Question(BaseModel):
    text: str

@app.post("/answer")
async def answer(q: Question):

    prompt = f"""
You are a quiz solver.

Return ONLY one letter:
A, B, C, or D.

No explanation.

Question:
{q.text}
"""

    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 2
            }
        )

    data = r.json()

    return {
        "answer": data["choices"][0]["message"]["content"].strip()
    }