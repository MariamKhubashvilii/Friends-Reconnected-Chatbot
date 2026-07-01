"""
ingest.py — Build (or update) the ChromaDB vector store.
Memory-conservative version: processes in small batches with explicit
cleanup between each, suitable for 8GB RAM machines.
"""

import json
import hashlib
import argparse
import gc
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import httpx


# ── Config ──────────────────────────────────────────────────────────────────
DATA_FILE     = "data/site_content.json"
CHROMA_DIR    = "./chroma_db"
COLLECTION    = "friends_reconnected"
EMBED_MODEL   = "nomic-embed-text"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
BATCH_SIZE    = 4     # very small batches to keep memory flat


# ── Chunking ─────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple, fast fixed-size chunking with overlap. No backward scanning."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, size // 6 - overlap // 6)  # approx words per chunk, with overlap
    words_per_chunk = max(20, size // 6)
    i = 0
    while i < len(words):
        chunk_words = words[i:i + words_per_chunk]
        chunk = " ".join(chunk_words)
        if len(chunk) > 30:
            chunks.append(chunk)
        i += step
    return chunks


def make_chunk_id(url: str, chunk_index: int, chunk_text: str) -> str:
    h = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
    safe_url = url.replace("https://", "").replace("/", "_")[:60]
    return f"{safe_url}__c{chunk_index}__{h}"


# ── Main ─────────────────────────────────────────────────────────────────────
def main(rebuild: bool = False):
    print(f"Loading content from {DATA_FILE}...")
    with open(DATA_FILE) as f:
        pages = json.load(f)
    print(f"{len(pages)} pages loaded")

    print(f"Loading embedding model: {EMBED_MODEL}...")
    embed_fn = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=EMBED_MODEL,
    )
    # Increase timeout — default is too short for CPU-bound embedding calls
    embed_fn._session = httpx.Client(timeout=httpx.Timeout(120.0))

    print("Connecting to ChromaDB (persistent local)...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild:
        existing = [c.name for c in client.list_collections()]
        if COLLECTION in existing:
            print("--rebuild flag: deleting existing collection...")
            client.delete_collection(COLLECTION)

    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    existing_count = collection.count()
    if existing_count > 0:
        print(f"  {existing_count} chunks already in DB — upsert will skip unchanged ones")

    # --- Check which chunks already exist in ChromaDB ---
    existing_ids = set()
    if collection.count() > 0:
        existing = collection.get(include=[])   # only fetch IDs, not documents/embeddings
        existing_ids = set(existing['ids'])
        print(f"  {len(existing_ids)} chunks already in DB – skipping them")
    else:
        print("  No existing chunks – embedding all.")

    # Build chunk list (only NEW chunks – cheap, no embedding yet)
    all_ids, all_docs, all_metas = [], [], []
    new_count = 0
    for page in pages:
        url, title = page["url"], page["title"]
        chunks = chunk_text(page["content"])
        print(f"  {title}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            cid = make_chunk_id(url, i, chunk)
            if cid in existing_ids:
                continue          # skip already stored chunks
            all_ids.append(cid)
            all_docs.append(chunk)
            all_metas.append({
                "url": url,
                "title": title,
                "chunk_index": i,
                "scraped_at": page.get("scraped_at", ""),
                "word_count": page.get("word_count", 0),
            })
            new_count += 1

    total = len(all_ids)
    print(f"\nNew chunks to embed: {total} (out of {len(existing_ids) + total} total)")
    print(f"Embedding in small batches of {BATCH_SIZE} (memory-conservative)...\n")

    num_batches = (total - 1) // BATCH_SIZE + 1
    failed_batches = []
    for i in range(0, total, BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        ids_batch   = all_ids[i:i+BATCH_SIZE]
        docs_batch  = all_docs[i:i+BATCH_SIZE]
        metas_batch = all_metas[i:i+BATCH_SIZE]

        for attempt in range(3):
            try:
                collection.upsert(
                    ids=ids_batch,
                    documents=docs_batch,
                    metadatas=metas_batch,
                )
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  [{batch_num}/{num_batches}] retry {attempt+1}/2 after error: {e}", flush=True)
                else:
                    print(f"  [{batch_num}/{num_batches}] FAILED after 3 attempts, skipping: {e}", flush=True)
                    failed_batches.append(batch_num)

        print(f"  [{batch_num}/{num_batches}] {min(i+BATCH_SIZE, total)}/{total} chunks done", flush=True)
        if batch_num % 5 == 0:
            gc.collect()

    if failed_batches:
        print(f"\n  {len(failed_batches)} batches failed: {failed_batches}")
        print("  Run 'python ingest.py' again to retry — already-embedded chunks will be skipped.")

    print(f"\nDone! Collection '{COLLECTION}' now has {collection.count()} chunks.")
    print(f"ChromaDB saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe and rebuild from scratch")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
