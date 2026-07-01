"""
parse_xml.py — Convert WordPress XML exports into site_content.json for RAG.

Reads:  data/pages.xml  +  data/posts.xml
Writes: data/site_content.json  (replaces the old scraped version)

Filters:
  - Only published content (skips drafts, private, attachments)
  - Only items with 100+ words (skips thin pages like "Payments", "Contacts")
  - Skips purely structural pages (Blog index, Payments, etc.)

Run:
  python parse_xml.py
"""

import xml.etree.ElementTree as ET
import json
import re
import html
import hashlib
from datetime import datetime
from pathlib import Path

NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "wp":      "http://wordpress.org/export/1.2/",
}

# Pages to skip even if published (structural/boilerplate)
SKIP_TITLES = {
    "blog", "payments", "contacts", "newsletter", "press enquires",
    "search for a person", "slider", "archive content",
}

MIN_WORDS = 100  # skip anything shorter than this


def strip_html(raw: str) -> str:
    """Remove HTML tags and decode entities."""
    text = html.unescape(raw or "")
    # Remove shortcodes like [caption id="..."]
    text = re.sub(r"\[/?[a-z_]+[^\]]*\]", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_file(filepath: str) -> list[dict]:
    tree = ET.parse(filepath)
    root = tree.getroot()
    results = []

    for item in root.findall(".//item"):
        status = item.findtext("wp:status", "", NS)
        ptype  = item.findtext("wp:post_type", "", NS)

        # Only published pages and posts
        if status != "publish" or ptype not in ("page", "post"):
            continue

        title = item.findtext("title", "").strip()

        # Skip boilerplate pages
        if title.lower() in SKIP_TITLES:
            continue

        url = item.findtext("link", "").strip()

        encoded = item.find("content:encoded", NS)
        raw = encoded.text if encoded is not None and encoded.text else ""
        content = strip_html(raw)

        word_count = len(content.split())
        if word_count < MIN_WORDS:
            continue

        fingerprint = hashlib.md5(content.encode()).hexdigest()

        results.append({
            "url":        url,
            "title":      title,
            "content":    content,
            "type":       ptype,
            "scraped_at": datetime.utcnow().isoformat(),
            "word_count": word_count,
            "fingerprint": fingerprint,
        })

    return results


def main():
    all_pages = []

    for fname in ["data/pages.xml", "data/posts.xml"]:
        if not Path(fname).exists():
            print(f"  Skipping {fname} (not found)")
            continue
        items = parse_file(fname)
        print(f"  {fname}: {len(items)} usable items")
        all_pages.extend(items)

    # Deduplicate by URL (in case same page appears in both exports)
    seen = set()
    deduped = []
    for p in all_pages:
        if p["url"] not in seen:
            seen.add(p["url"])
            deduped.append(p)

    total_words = sum(p["word_count"] for p in deduped)
    print(f"\nTotal: {len(deduped)} pages/posts, {total_words:,} words")

    # Print summary
    print("\n--- Pages ---")
    for p in deduped:
        if p["type"] == "page":
            print(f"  {p['word_count']:>5}w  {p['title'][:70]}")
    print("\n--- Posts ---")
    for p in deduped:
        if p["type"] == "post":
            print(f"  {p['word_count']:>5}w  {p['title'][:70]}")

    Path("data").mkdir(exist_ok=True)
    with open("data/site_content.json", "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to data/site_content.json")
    print("Next: python ingest.py --rebuild")


if __name__ == "__main__":
    main()
