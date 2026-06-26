"""
ingest.py — Build (or update) the ChromaDB vector store.

INCREMENTAL UPDATES
-------------------
ChromaDB uses a persistent directory (./chroma_db by default).
Every document chunk gets a deterministic ID based on its URL + chunk index.

To add a new page later:
  1. Add a new entry to data/site_content.json  (or point to a new JSON file)
  2. Run:  python ingest.py
  
Only NEW or CHANGED chunks will be upserted — existing unchanged chunks
are left untouched. Nothing gets re-embedded unnecessarily.

To remove a page: delete its entry from the JSON and run:
  python ingest.py --rebuild
  (This wipes the DB and rebuilds from scratch.)
"""

import json
import hashlib
import argparse
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


# ── Config ──────────────────────────────────────────────────────────────────
DATA_FILE   = "data/site_content.json"
CHROMA_DIR  = "./chroma_db"
COLLECTION  = "friends_reconnected"
MODEL_NAME  = "all-MiniLM-L6-v2"   # fast, good quality, ~80MB
CHUNK_SIZE  = 400   # characters
CHUNK_OVERLAP = 80


# ── Chunking ─────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            last_period = text.rfind(". ", start, end)
            if last_period != -1 and last_period > start + overlap:
                end = last_period + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 30]   # drop tiny tail chunks


def make_chunk_id(url: str, chunk_index: int, chunk_text: str) -> str:
    """Deterministic ID: url + index + hash of content."""
    h = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
    safe_url = url.replace("https://", "").replace("/", "_")[:60]
    return f"{safe_url}__c{chunk_index}__{h}"


# ── Main ─────────────────────────────────────────────────────────────────────
def main(rebuild: bool = False):
    print(f"Loading content from {DATA_FILE}...")
    with open(DATA_FILE) as f:
        pages = json.load(f)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild and client.list_collections():
        print("--rebuild flag set: deleting existing collection...")
        client.delete_collection(COLLECTION)

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    all_ids, all_docs, all_metas, all_embeddings = [], [], [], []

    for page in pages:
        url   = page["url"]
        title = page["title"]
        chunks = chunk_text(page["content"])
        print(f"  {title}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(url, i, chunk)
            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_metas.append({
                "url": url,
                "title": title,
                "chunk_index": i,
                "scraped_at": page.get("scraped_at", ""),
                "word_count": page.get("word_count", 0),
            })

    print(f"\nEmbedding {len(all_docs)} chunks...")
    embeddings = model.encode(all_docs, show_progress_bar=True).tolist()

    # Upsert: insert new, update changed, skip identical
    print("Upserting into ChromaDB...")
    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        collection.upsert(
            ids=all_ids[i:i+batch_size],
            documents=all_docs[i:i+batch_size],
            metadatas=all_metas[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
        )

    print(f"\nDone! Collection '{COLLECTION}' now has {collection.count()} chunks.")
    print(f"ChromaDB persisted to: {CHROMA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the collection and rebuild from scratch (use when removing pages)"
    )
    args = parser.parse_args()
    main(rebuild=args.rebuild)
