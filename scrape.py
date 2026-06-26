"""
scrape.py — Playwright-powered scraper for Friends Reconnected (or any site).

Usage examples:
  # Scrape a list of URLs
  python scrape.py --urls https://friendsreconnected.co.uk/faq/ https://friendsreconnected.co.uk/testimonials/

  # Scrape from a text file (one URL per line)
  python scrape.py --file urls.txt

  # Auto-discover pages from sitemap
  python scrape.py --sitemap https://friendsreconnected.co.uk/sitemap.xml

  # Crawl from a root URL (follows internal links automatically)
  python scrape.py --crawl https://friendsreconnected.co.uk/ --max-pages 30

  # Add new pages to existing data (won't re-scrape already-scraped URLs)
  python scrape.py --urls https://friendsreconnected.co.uk/new-page/ --update

Output:
  data/site_content.json  — ready for ingest.py
"""

import asyncio
import json
import argparse
import re
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import print as rprint

console = Console()

DATA_FILE = Path("data/site_content.json")

# ── Selectors to REMOVE from pages (nav, footer, boilerplate) ─────────────────
REMOVE_SELECTORS = [
    "header", "nav", "footer",
    ".site-header", ".site-footer", ".site-navigation",
    "#wpadminbar",
    ".wp-block-navigation",
    ".comment-form", ".comments-area",
    ".widget_recent_entries",   # "Recent Posts" sidebar
    ".widget_text",
    '[class*="sidebar"]',
    '[class*="widget"]',
    '[id*="sidebar"]',
    ".breadcrumb", ".breadcrumbs",
    "script", "style", "noscript",
    ".cookie-notice", ".gdpr",
    '[class*="social"]',
    ".share-buttons",
    "form",                     # contact forms
    ".navigation.post-navigation",
    ".wp-pagenavi",
]

# ── Domains to stay within when crawling ─────────────────────────────────────
STAY_ON_DOMAIN = True


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Normalise whitespace, remove boilerplate fragments."""
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Remove common WP boilerplate lines
    boilerplate = [
        r"Skip to content",
        r"Open Menu",
        r"Close Menu",
        r"WhatsApp us",
        r"Pay Invoice",
        r"8\.00am.*Mon.*Sat",
        r"Visits By Appointment",
        r"Call Us Now",
        r"Email Us",
        r"Recent Posts",
        r"About Us",
        r"Contact Us",
        r"Company info",
        r"Subscribe to our newsletter",
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_content(html: str, url: str) -> dict:
    """Parse HTML → structured content dict."""
    soup = BeautifulSoup(html, "lxml")

    # Remove noise elements
    for selector in REMOVE_SELECTORS:
        for el in soup.select(selector):
            el.decompose()

    # Title: prefer h1, fall back to <title>
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""
    if not title:
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True).split("–")[0].strip() if title_tag else url

    # Meta description (useful context)
    meta_desc = ""
    meta = soup.find("meta", attrs={"name": "description"}) or \
           soup.find("meta", property="og:description")
    if meta:
        meta_desc = meta.get("content", "")

    # Main content area — try common WP content selectors first
    content_el = (
        soup.select_one(".entry-content") or
        soup.select_one("article") or
        soup.select_one("main") or
        soup.select_one("#content") or
        soup.select_one(".page-content") or
        soup.find("body")
    )

    raw_text = content_el.get_text(separator="\n") if content_el else ""
    content = clean_text(raw_text)

    # Prepend meta description if it adds context not already in content
    if meta_desc and meta_desc[:50] not in content[:200]:
        content = meta_desc + "\n\n" + content

    # Content fingerprint for change detection
    fingerprint = hashlib.md5(content.encode()).hexdigest()

    return {
        "url": url,
        "title": title,
        "content": content,
        "scraped_at": datetime.utcnow().isoformat(),
        "word_count": len(content.split()),
        "fingerprint": fingerprint,
    }


# ── Sitemap parser ────────────────────────────────────────────────────────────
async def fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    """Extract all <loc> URLs from an XML sitemap (handles sitemap index too)."""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(sitemap_url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml-xml")
    urls = []

    # Sitemap index → recurse
    for sitemap in soup.find_all("sitemap"):
        loc = sitemap.find("loc")
        if loc:
            urls.extend(await fetch_sitemap_urls(loc.text.strip()))

    # Regular sitemap
    for loc in soup.find_all("loc"):
        u = loc.text.strip()
        if not u.endswith(".xml"):
            urls.append(u)

    return urls


# ── Link crawler ──────────────────────────────────────────────────────────────
async def crawl_links(page, root_url: str, max_pages: int) -> list[str]:
    """Follow internal links from root_url up to max_pages."""
    domain = urlparse(root_url).netloc
    visited, queue = set(), [root_url]
    found = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Crawling links..."),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("crawl", total=max_pages)

        while queue and len(found) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                html = await page.content()
                soup = BeautifulSoup(html, "lxml")

                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"])
                    parsed = urlparse(href)
                    # Stay on same domain, skip anchors/query strings/files
                    if (
                        parsed.netloc == domain
                        and not parsed.fragment
                        and not parsed.query
                        and not href.endswith((".pdf", ".jpg", ".png", ".zip"))
                        and href not in visited
                        and href not in queue
                    ):
                        queue.append(href)

                found.append(url)
                progress.advance(task)
            except Exception as e:
                console.print(f"[yellow]  ⚠ Skipped {url}: {e}[/yellow]")

    return found


# ── Core scraper ──────────────────────────────────────────────────────────────
async def scrape_urls(urls: list[str], existing: dict) -> list[dict]:
    """Scrape a list of URLs with Playwright. Skip already-scraped unchanged pages."""
    results = []
    skipped = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (compatible; FriendsReconnectedBot/1.0)"
        )
        page = await context.new_page()

        # Block images/fonts to speed up scraping
        await page.route("**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf}", 
                         lambda route: route.abort())

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Scraping[/bold green] {task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} pages"),
            console=console,
        ) as progress:
            task = progress.add_task("", total=len(urls))

            for url in urls:
                progress.update(task, description=url.split("/")[-2] or url)

                # Skip if already scraped (fingerprint check happens after fetch)
                try:
                    await page.goto(url, wait_until="networkidle", timeout=20000)
                    html = await page.content()
                    extracted = extract_content(html, url)

                    # If we already have this URL, check if content changed
                    if url in existing:
                        if existing[url]["fingerprint"] == extracted["fingerprint"]:
                            console.print(f"  [dim]↷ Unchanged: {extracted['title']}[/dim]")
                            results.append(existing[url])  # keep old entry
                            skipped += 1
                            progress.advance(task)
                            continue

                    if extracted["word_count"] < 50:
                        console.print(f"  [yellow]⚠ Thin content ({extracted['word_count']} words): {url}[/yellow]")
                    else:
                        console.print(f"  [green]✓[/green] {extracted['title']} ({extracted['word_count']} words)")

                    results.append(extracted)

                except Exception as e:
                    console.print(f"  [red]✗ Failed {url}: {e}[/red]")

                progress.advance(task)

        await browser.close()

    console.print(f"\n[dim]  {skipped} pages unchanged (skipped re-scraping)[/dim]")
    return results


# ── Save / load ───────────────────────────────────────────────────────────────
def load_existing() -> dict:
    """Load existing scraped data, keyed by URL."""
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            data = json.load(f)
        return {item["url"]: item for item in data}
    return {}


def save(data: list[dict]):
    DATA_FILE.parent.mkdir(exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(results: list[dict]):
    table = Table(title="Scraped Pages", show_lines=True)
    table.add_column("Title", style="bold")
    table.add_column("Words", justify="right")
    table.add_column("URL", style="dim")

    total_words = 0
    for r in results:
        table.add_row(r["title"][:50], str(r["word_count"]), r["url"][:60])
        total_words += r["word_count"]

    console.print(table)
    console.print(f"\n[bold green]Total:[/bold green] {len(results)} pages, {total_words:,} words")
    console.print(f"[bold green]Saved to:[/bold green] {DATA_FILE}\n")
    console.print("[dim]Next step: run  python ingest.py  to embed into ChromaDB[/dim]")


# ── CLI ───────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        description="Scrape website content for RAG ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--urls", nargs="+", metavar="URL", help="Explicit list of URLs")
    group.add_argument("--file", metavar="FILE", help="Text file with one URL per line")
    group.add_argument("--sitemap", metavar="URL", help="Sitemap XML URL")
    group.add_argument("--crawl", metavar="URL", help="Root URL to crawl from")

    parser.add_argument("--max-pages", type=int, default=30,
                        help="Max pages when crawling (default: 30)")
    parser.add_argument("--update", action="store_true",
                        help="Merge with existing data (skip unchanged pages)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore existing data and re-scrape everything")

    args = parser.parse_args()

    console.rule("[bold]Friends Reconnected Scraper[/bold]")

    # Load existing data
    existing = {} if args.overwrite else load_existing()
    if existing:
        console.print(f"[dim]Found {len(existing)} existing pages in {DATA_FILE}[/dim]\n")

    # Collect URLs
    if args.urls:
        urls = args.urls
    elif args.file:
        urls = [u.strip() for u in open(args.file).readlines() if u.strip()]
    elif args.sitemap:
        console.print(f"Fetching sitemap: {args.sitemap}")
        urls = await fetch_sitemap_urls(args.sitemap)
        console.print(f"Found {len(urls)} URLs in sitemap\n")
    elif args.crawl:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            urls = await crawl_links(page, args.crawl, args.max_pages)
            await browser.close()
        console.print(f"Discovered {len(urls)} pages via crawl\n")

    # Deduplicate
    urls = list(dict.fromkeys(urls))
    console.print(f"Scraping [bold]{len(urls)}[/bold] pages...\n")

    # Scrape
    results = await scrape_urls(urls, existing)

    # If --update, merge with existing pages not in current URL list
    if args.update and existing:
        current_urls = {r["url"] for r in results}
        for url, page_data in existing.items():
            if url not in current_urls:
                results.append(page_data)
        console.print(f"[dim]Merged with {len(existing)} existing pages[/dim]")

    # Save and summarise
    save(results)
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
