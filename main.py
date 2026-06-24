from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re

app = FastAPI()

API_KEY = os.getenv("API_KEY")
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
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")

    cleaned = clean_ocr_text(q.text)
    question_only = extract_question_only(cleaned)

    # Search Google for context (runs in parallel with nothing — just await it)
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
        "google/gemini-2.5-flash",
        "google/gemini-2.0-flash-lite",
        "google/gemini-flash-1.5-8b",
    ]

    last_error = None
    for model in models:
        try:
            async with httpx.AsyncClient(timeout=6) as client:
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
                        "max_tokens": 40,
                    },
                )
                r.raise_for_status()
                data = r.json()
                # Safe content extraction — handle both string and list
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    answer_text = " ".join(
                        b.get("text", "") for b in content if isinstance(b, dict)
                    ).strip()
                else:
                    answer_text = str(content).strip()

                answer_text = re.sub(r'^[\(\[]?[A-Da-d][\)\]\.]?\s*', '', answer_text).strip()

                if is_valid_answer(answer_text):
                    return {"answer": answer_text, "model": model}
                # If invalid (refusal etc), try next model
        except Exception as e:
            last_error = e
            continue

    raise HTTPException(status_code=502, detail=f"All models failed: {last_error}")
