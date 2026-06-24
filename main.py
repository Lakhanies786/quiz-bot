from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re
import itertools

app = FastAPI()

# ---------------------------------------------------------------------------
# API Keys — supports multiple Gemini keys for quota rotation
# Set GEMINI_KEY_1, GEMINI_KEY_2, GEMINI_KEY_3 in Railway variables
# At minimum, GEMINI_KEY_1 must be set (your new key goes here)
# ---------------------------------------------------------------------------
_raw_keys = [
    os.getenv("GEMINI_KEY_1"),
    os.getenv("GEMINI_KEY_2"),
    os.getenv("GEMINI_KEY_3"),
]
GEMINI_KEYS = [k for k in _raw_keys if k]

# Fallback: also accept old variable name API_KEY
if not GEMINI_KEYS:
    old_key = os.getenv("API_KEY")
    if old_key:
        GEMINI_KEYS = [old_key]

_key_cycle = itertools.cycle(GEMINI_KEYS) if GEMINI_KEYS else None

SERPER_KEY = os.getenv("SERPER_KEY")

# ---------------------------------------------------------------------------
# Working Gemini models as of 2026 (2.0-flash retired March 2026)
# ---------------------------------------------------------------------------
MODELS = [
    "gemini-2.5-flash-lite",  # 1000 RPD free — use first
    "gemini-2.5-flash",       # 250 RPD free  — fallback
]


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
        "i'm sorry", "i am sorry", "i cannot", "i can't",
        "as an ai", "unfortunately", "i'm unable",
        "i need more", "not enough", "i'm not able",
    ]
    for r in refusals:
        if r in low:
            return False
    if len(text) > 200:   # relaxed from 80 — short answers still win
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
    key_count = len(GEMINI_KEYS)
    return {"status": "QuizBot backend running", "gemini_keys_loaded": key_count}


@app.post("/answer")
async def answer(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    if not GEMINI_KEYS:
        raise HTTPException(status_code=500, detail="No Gemini API key set. Add GEMINI_KEY_1 in Railway variables.")

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

    last_error = None
    for model in MODELS:
        gemini_key = next(_key_cycle)  # rotate through keys each attempt
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    params={"key": gemini_key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0,
                            "maxOutputTokens": 60,
                        },
                    },
                )
                r.raise_for_status()
                data = r.json()
                answer_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                answer_text = re.sub(r'^[\(\[]?[A-Da-d][\)\]\.]?\s*', '', answer_text).strip()

                if is_valid_answer(answer_text):
                    return {"answer": answer_text, "model": model}

        except Exception as e:
            last_error = e
            continue

    raise HTTPException(status_code=502, detail=f"All models failed: {last_error}")
