from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re

app = FastAPI()

GEMINI_KEY = os.getenv("GEMINI_KEY")
SERPER_KEY = os.getenv("SERPER_KEY")


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


def extract_question_only(cleaned: str) -> str:
    lines = cleaned.splitlines()
    question_lines = []
    for line in lines:
        if re.match(r'^[A-D][.)]\s', line) or re.match(r'^\([A-D]\)', line):
            break
        question_lines.append(line)
    return " ".join(question_lines).strip()


def is_valid_answer(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    refusals = [
        "i'm sorry", "i am sorry", "i cannot", "i can't", "i don't",
        "the question", "as an ai", "unfortunately", "i'm unable",
        "i need more", "not enough", "unclear", "i'm not able",
        "based on the", "it appears", "i would need",
    ]
    for r in refusals:
        if r in low:
            return False
    if len(text) > 80:
        return False
    return True


async def web_search(query: str) -> str:
    if not SERPER_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": SERPER_KEY,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": 5},
            )
            r.raise_for_status()
            data = r.json()
            snippets = []
            if "answerBox" in data:
                ab = data["answerBox"]
                if "answer" in ab:
                    snippets.append(ab["answer"])
                elif "snippet" in ab:
                    snippets.append(ab["snippet"])
            if "knowledgeGraph" in data:
                kg = data["knowledgeGraph"]
                if "description" in kg:
                    snippets.append(kg["description"])
            for item in data.get("organic", [])[:3]:
                if "snippet" in item:
                    snippets.append(item["snippet"])
            return "\n".join(snippets[:4])
    except Exception:
        return ""


@app.get("/")
async def health():
    return {"status": "QuizBot backend running"}


@app.post("/answer")
async def answer(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_KEY not set")

    cleaned = clean_ocr_text(q.text)
    question_only = extract_question_only(cleaned)

    search_context = await web_search(question_only)

    if search_context:
        prompt = (
            "You are a quiz answer bot.\n"
            "Search results to help you:\n"
            + search_context
            + "\n\nUsing the search results, pick the correct answer option.\n"
            "Reply with ONLY the exact text of the correct option — no letter, no explanation.\n\n"
            "Question and options:\n"
            + cleaned
        )
    else:
        prompt = (
            "You are a precise quiz answer bot.\n"
            "Reply with ONLY the exact text of the correct answer option — no letter prefix, no explanation.\n\n"
            "Question and options:\n"
            + cleaned
        )

    models = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest",
    ]

    last_error = None
    for model in models:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0,
                            "maxOutputTokens": 40,
                        },
                    },
                )
                print(f"[{model}] HTTP {r.status_code}: {r.text[:300]}", flush=True)
                r.raise_for_status()
                data = r.json()
                answer_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                answer_text = re.sub(r'^[\(\[]?[A-Da-d][\)\]\.]?\s*', '', answer_text).strip()

                if is_valid_answer(answer_text):
                    return {"answer": answer_text, "model": model}
        except Exception as e:
            print(f"[{model}] Exception: {e}", flush=True)
            last_error = e
            continue

    raise HTTPException(status_code=502, detail=f"All models failed: {last_error}")
