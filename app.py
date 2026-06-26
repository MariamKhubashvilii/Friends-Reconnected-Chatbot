"""
app.py — FastAPI RAG backend for Friends Reconnected chat widget.

Endpoints:
  POST /chat     { "message": "...", "history": [...] }  →  streaming response
  GET  /health   → { "status": "ok", "chunks": N }

Run:
  uvicorn app:app --reload --port 8000
"""

import os
import json
from pathlib import Path
from typing import Generator

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import ollama


# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "friends_reconnected"
EMBED_MODEL  = "all-MiniLM-L6-v2"
LLM_MODEL    = "llama3.2"          # change to "mistral" if you prefer
TOP_K        = 4                    # chunks to retrieve

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

Keep responses concise, warm, and human. This audience is often elderly and emotionally invested."""


# ── Init ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Friends Reconnected RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION)
print(f"Ready. {collection.count()} chunks loaded.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


# ── RAG helpers ───────────────────────────────────────────────────────────────
def retrieve(query: str) -> str:
    """Embed query, fetch top-K chunks, return as context string."""
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )
    chunks = results["documents"][0]
    metas  = results["metadatas"][0]

    context_parts = []
    for chunk, meta in zip(chunks, metas):
        context_parts.append(f"[From: {meta['title']}]\n{chunk}")

    return "\n\n---\n\n".join(context_parts)


def build_messages(user_message: str, history: list[Message]) -> list[dict]:
    """Build the full message list for Ollama."""
    context = retrieve(user_message)

    # Inject context into system prompt
    system_with_context = (
        SYSTEM_PROMPT
        + f"\n\nRELEVANT CONTEXT FROM WEBSITE:\n{context}"
    )

    messages = [{"role": "system", "content": system_with_context}]

    # Add conversation history (last 6 turns max to stay within context)
    for msg in history[-6:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": user_message})
    return messages


def stream_response(messages: list[dict]) -> Generator[str, None, None]:
    """Stream tokens from Ollama as SSE."""
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            # SSE format
            yield f"data: {json.dumps({'token': token})}\n\n"
    yield "data: [DONE]\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    messages = build_messages(req.message, req.history)
    return StreamingResponse(
        stream_response(messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "chunks": collection.count(), "model": LLM_MODEL}


# ── Serve frontend ────────────────────────────────────────────────────────────
if Path("static").exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
