from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import re

app = FastAPI()

API_KEY = os.getenv("API_KEY")
SERPER_KEY = os.getenv("SERPER_KEY")  # free at serper.dev — 2500 searches/month free


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
    """Extract just the question sentence for search query, without options."""
    lines = cleaned.splitlines()
    question_lines = []
    for line in lines:
        u = line.upper()
        if re.match(r'^[A-D][.)]\s', line) or re.match(r'^\([A-D]\)', line):
            break  # stop before options
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
    """Search Google via Serper API and return top snippets as context."""
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
            # Answer box (best source — direct answer)
            if "answerBox" in data:
                ab = data["answerBox"]
                if "answer" in ab:
                    snippets.append(ab["answer"])
                elif "snippet" in ab:
                    snippets.append(ab["snippet"])
            # Knowledge graph
            if "knowledgeGraph" in data:
                kg = data["knowledgeGraph"]
                if "description" in kg:
                    snippets.append(kg["description"])
            # Organic results
            for item in data.get("organic", [])[:3]:
                if "snippet" in item:
                    snippets.append(item["snippet"])
            return "\n".join(snippets[:4])
    except Exception:
        return ""


async def call_model(prompt: str, model: str, timeout: int = 6) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
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
        text = data["choices"][0]["message"]["content"].strip()
        text = re.sub(r'^[\(\[]?[A-Da-d][\)\]\.]?\s*', '', text).strip()
        return text


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

    # Step 1: search Google for the answer context
    search_context = await web_search(question_only)

    # Step 2: build prompt — with search context if available, without if not
    if search_context:
        prompt = f"""You are a precise quiz answer bot.
You have been given real search results to help answer the question accurately.

Search results:
{search_context}

Using the search results above, identify the correct answer option.
Reply with ONLY the exact text of the correct option — no letter prefix, no explanation.

Question and options:
{cleaned}"""
    else:
        prompt = f"""You are a precise quiz answer bot with strong general knowledge.
Read the question and ALL options very carefully before answering.
Reply with ONLY the exact text of the correct answer option — no letter prefix, no explanation.

Question and options:
{cleaned}"""

    # Step 3: try models in order
    models = [
        "google/gemini-2.5-flash",
        "google/gemini-2.0-flash-lite",
        "google/gemini-flash-1.5-8b",
    ]

    last_error = None
    for model in models:
        try:
            answer_text = await call_model(prompt, model)
            if is_valid_answer(answer_text):
                return {"answer": answer_text, "model": model}
        except Exception as e:
            last_error = e
            continue

    raise HTTPException(status_code=502, detail=f"All models failed: {last_error}")
