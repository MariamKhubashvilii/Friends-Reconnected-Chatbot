"""
app.py — FastAPI RAG backend for Friends Reconnected chat widget.

Endpoints:
  POST /chat     { "message": "...", "history": [...] }  →  streaming OR JSON response
  GET  /health   → { "status": "ok", "chunks": N }

Run:
  conda activate friends
  uvicorn app:app --reload --port 8000
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Generator

import chromadb
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from groq import Groq

load_dotenv()


# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "friends_reconnected"
EMBED_MODEL  = "nomic-embed-text"
OLLAMA_URL   = "http://localhost:11434/api/embeddings"
LLM_MODEL    = "llama-3.3-70b-versatile"
TOP_K        = 4
LOW_SCORE_THRESHOLD = 0.55   # cosine similarity below this = "didn't understand"

CONTACT_INFO = "07521652936 / friendsreconnecteduk@gmail.com"

SYSTEM_PROMPT = """You are a helpful assistant for Friends Reconnected, a UK people-tracing service.
You help potential clients understand how the service works, what to expect, and whether it can help them.

You have access to relevant information from the Friends Reconnected website.
Answer questions warmly and honestly using that information.
If you don't know something, say so — never invent details.

Key facts to always remember:
- The service is run by John, an 80-year-old veteran researcher who handles every case personally
- Average search takes 30+ hours and costs money (not free)
- Only one case is worked at a time; expect a 2-4 week wait before starting
- UK success rate is ~95%; about 98% of people searched for are still alive
- Platonic friends: 99% respond positively; romantic: ~70%
- Contact: 07521652936 / friendsreconnecteduk@gmail.com
- Free, no-obligation review of any planned search is available
- The chat header has Call us and Email us buttons, so when contact is useful, point users to those buttons instead of repeating contact details every time

Keep responses concise, warm, and human. This audience is often elderly and emotionally invested.
Write answers in 2-4 short paragraphs, not one large block of text.
Use Markdown bold sparingly: bold the main takeaway or key answer phrase with **double asterisks**.
Do not print source lists or URLs in the answer text; the app shows further-reading links separately."""

CLARIFY_PROMPT = """You are a helpful assistant for Friends Reconnected, a UK people-tracing service.

A user asked something we couldn't quite match confidently to our website content. Your job is to:
1. Interpret what they might have meant — think broadly. For example "contacts" could mean 
   contact details for the missing person, an old address or phone number, or how to get in touch
   with Friends Reconnected.
2. Suggest 2-3 specific questions they might have meant, phrased naturally.
3. If the user might be asking about not having a person's contact details, say that old contact
   information can still help but is not always required; a name, age/date of birth, old address,
   school, relatives, workplace, or other clues may be useful.
4. Keep it warm and brief. Use 1-2 short paragraphs if needed.
5. Use **double asterisks** around the main helpful takeaway if natural.
6. If direct contact would help, tell them they can use the Call us or Email us buttons above.

Contact: 07521652936 / friendsreconnecteduk@gmail.com

Respond in this exact JSON format (no markdown, no extra text):
{
  "message": "A warm 1-2 sentence response acknowledging their question",
  "suggestions": [
    "Did you mean: [specific question 1]?",
    "Did you mean: [specific question 2]?",
    "Did you mean: [specific question 3]?"
  ]
}"""


# ── Init ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Friends Reconnected RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Connecting to Ollama embedding model ({EMBED_MODEL})...")
embed_fn = OllamaEmbeddingFunction(
    url=OLLAMA_URL,
    model_name=EMBED_MODEL,
)
embed_fn._session = httpx.Client(timeout=httpx.Timeout(120.0))

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION, embedding_function=embed_fn)
print(f"Ready. {collection.count()} chunks loaded.")

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
groq_client = Groq(api_key=groq_api_key)
print(f"Connected to Groq. Using model: {LLM_MODEL}")

DATA_FILE = Path("data/site_content.json")
try:
    SITE_PAGES = json.loads(DATA_FILE.read_text()) if DATA_FILE.exists() else []
except Exception as e:
    print(f"Could not load site content for fallback suggestions: {e}")
    SITE_PAGES = []


# ── Schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


# ── RAG helpers ───────────────────────────────────────────────────────────────
def retrieve(query: str):
    """Returns (context_string, sources_list, best_score, error_message)."""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"Retrieval error: {e}")
        return "", [], 0, str(e)

    chunks    = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]   # cosine distance (lower = more similar)

    # Convert distance to similarity score (0-1, higher = better match)
    scores = [1 - d for d in distances]
    best_score = max(scores) if scores else 0

    context_parts = []
    seen_urls = {}   # deduplicate sources by URL
    for chunk, meta, score in zip(chunks, metas, scores):
        title = meta.get("title", "Friends Reconnected")
        url = meta.get("url", "")
        context_parts.append(f"[From: {title} | {url}]\n{chunk}")
        url = meta.get("url", "")
        if url and url not in seen_urls:
            seen_urls[url] = title

    context = "\n\n---\n\n".join(context_parts)
    sources = [{"title": title, "url": url} for url, title in seen_urls.items()]
    return context, sources, best_score, None


def local_topic_matches(user_message: str, limit: int = 3) -> list[dict]:
    """Cheap non-embedding fallback to find nearby pages by keyword overlap."""
    stopwords = {
        "about", "after", "again", "could", "does", "dont", "have", "their",
        "there", "these", "they", "this", "what", "when", "where", "which",
        "with", "would", "you", "your", "that", "them", "then", "than", "from",
        "and", "are", "but", "can", "did", "for", "how", "if", "not", "the",
        "who", "why"
    }
    words = {
        word for word in re.findall(r"[a-z0-9']+", user_message.lower())
        if len(word) > 2 and word not in stopwords
    }
    expanded_words = set(words)
    for word in words:
        if word.endswith("s") and len(word) > 4:
            expanded_words.add(word[:-1])
        if word in {"contact", "contacts"}:
            expanded_words.update({"address", "phone", "telephone", "email", "information", "details"})
        if word in {"info", "information", "details"}:
            expanded_words.update({"address", "name", "birth", "school", "workplace", "relatives"})
    words = expanded_words
    if not words:
        return []

    matches = []
    for page in SITE_PAGES:
        title = page.get("title", "")
        content = page.get("content", "")
        haystack_words = Counter(re.findall(r"[a-z0-9']+", f"{title} {content}".lower()))
        score = sum(haystack_words[word] for word in words)
        if score:
            snippet = re.sub(r"\s+", " ", content).strip()[:260]
            matches.append({
                "title": title,
                "url": page.get("url", ""),
                "snippet": snippet,
                "score": score,
            })
    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:limit]


def parse_sources_from_response(text: str):
    """Extract SOURCES block from LLM response, return (clean_text, sources_list)."""
    sources = []
    if "SOURCES:" not in text:
        return text, sources

    parts = text.split("SOURCES:", 1)
    clean = parts[0].strip()
    raw_sources = parts[1].strip()

    for line in raw_sources.splitlines():
        line = line.strip().lstrip("- ").strip()
        if "|" in line:
            bits = line.split("|", 1)
            title = bits[0].strip()
            url   = bits[1].strip()
            if url.startswith("http"):
                sources.append({"title": title, "url": url})

    return clean, sources


def build_messages(user_message: str, history: list[Message], context: str) -> list[dict]:
    system_with_context = SYSTEM_PROMPT + f"\n\nRELEVANT CONTEXT FROM WEBSITE:\n{context}"
    messages = [{"role": "system", "content": system_with_context}]
    for msg in history[-6:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})
    return messages


def get_clarification(user_message: str, nearby: list[dict] | None = None) -> dict:
    """Ask the LLM to interpret a low-confidence query and suggest alternatives."""
    nearby = nearby or []
    nearby_text = "\n".join(
        f"- {item.get('title', 'Untitled')}: {item.get('snippet', '')}"
        for item in nearby
    )
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": CLARIFY_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"The user asked: \"{user_message}\"\n\n"
                        f"Nearby website topics, if useful:\n{nearby_text or 'None found.'}"
                    ),
                }
            ],
            stream=False,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"Clarify error: {e}")
        return {
            "message": (
                "Sorry, I didn't quite understand the question. If you mean you do not "
                "have the person's current contact details, that is often still workable "
                "if you have other clues such as a name, old address, age, school, "
                "workplace, or relatives."
            ),
            "suggestions": [
                "Did you mean: What information do I need to start a search?",
                "Did you mean: Can you help if I do not have their current contact details?",
                "Did you mean: How can I contact Friends Reconnected?"
            ]
        }


def stream_response(messages: list[dict], fallback_sources: list[dict]) -> Generator[str, None, None]:
    stream = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
    )
    full_text = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            full_text += token
            yield f"data: {json.dumps({'token': token})}\n\n"

    # After streaming is done, parse sources and send them as a final event
    _, sources = parse_sources_from_response(full_text)
    if not sources:
        sources = fallback_sources[:3]
    if sources:
        yield f"data: {json.dumps({'sources': sources})}\n\n"

    yield "data: [DONE]\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    context, sources, best_score, retrieval_error = retrieve(req.message)

    # Low confidence or embedding/retrieval failure — return clarification JSON instead of an error.
    if retrieval_error or best_score < LOW_SCORE_THRESHOLD:
        nearby = local_topic_matches(req.message) or sources
        clarification = get_clarification(req.message, nearby)
        return JSONResponse(content={
            "type": "clarification",
            "message": clarification.get("message", ""),
            "suggestions": clarification.get("suggestions", []),
            "contact": CONTACT_INFO,
            "nearby": nearby[:3],
        })

    messages = build_messages(req.message, req.history, context)
    return StreamingResponse(
        stream_response(messages, sources),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "chunks": collection.count(), "model": LLM_MODEL}


if Path("static").exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
