import xml.etree.ElementTree as ET
import re
import html

def strip_tags(text):
    text = html.unescape(text or "")
    return re.sub(r"<[^>]+>", " ", text).strip()

NS = {"content": "http://purl.org/rss/1.0/modules/content/",
      "wp": "http://wordpress.org/export/1.2/"}

for fname in ["data/pages.xml", "data/posts.xml"]:
    tree = ET.parse(fname)
    root = tree.getroot()
    items = root.findall(".//item")
    print(f"\n{'='*60}")
    print(f"{fname}: {len(items)} items")
    print(f"{'='*60}")
    for item in items:
        title  = item.findtext("title", "").strip()
        status = item.findtext("wp:status", "", NS)
        ptype  = item.findtext("wp:post_type", "", NS)
        encoded = item.find("content:encoded", NS)
        raw = encoded.text or "" if encoded is not None else ""
        text = strip_tags(raw)
        words = len(text.split())
        print(f"  [{status}] [{ptype}] {words:>5}w  {title[:70]}")
