"""
Download recent arXiv abstracts (title + abstract only) as one .md file per paper.

    python scripts/fetch_arxiv.py                 # 2500 per category, 4 categories
    python scripts/fetch_arxiv.py --per-cat 500   # smaller test run

Uses the public arXiv API (https://info.arxiv.org/help/api/), one request every
3 seconds as the arXiv terms ask. Output: data/arxiv/<id>.md
Categories: cs.DC distributed/parallel, cs.PF performance, cs.AR hardware, cs.OS operating systems.
"""

import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "arxiv")
CATS = ["cs.DC", "cs.PF", "cs.AR", "cs.OS"]
PAGE = 500
MIN_WORDS = 60


def parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for e in root.findall("a:entry", NS):
        url = (e.findtext("a:id", "", NS) or "").strip()
        pid = url.rsplit("/abs/", 1)[-1].replace("/", "_")
        title = " ".join((e.findtext("a:title", "", NS) or "").split())
        abstract = " ".join((e.findtext("a:summary", "", NS) or "").split())
        if pid and title and len(abstract.split()) >= MIN_WORDS:
            papers.append({"id": pid, "title": title, "abstract": abstract})
    return papers


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "FastFind-hackathon/0.3 (student project)"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  retrying after error: {e}")
            time.sleep(5 * (attempt + 1))


def main(per_cat: int) -> None:
    os.makedirs(OUT, exist_ok=True)
    seen = set(f[:-3] for f in os.listdir(OUT) if f.endswith(".md"))
    for cat in CATS:
        got = 0
        for start in range(0, per_cat, PAGE):
            q = urllib.parse.urlencode({
                "search_query": f"cat:{cat}", "start": start,
                "max_results": min(PAGE, per_cat - start),
                "sortBy": "submittedDate", "sortOrder": "descending",
            })
            papers = parse_feed(get(f"{API}?{q}"))
            for p in papers:
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                with open(os.path.join(OUT, p["id"] + ".md"), "w", encoding="utf-8") as f:
                    f.write(f"# {p['title']}\n\n{p['abstract']}\n")
                got += 1
            print(f"{cat}: {got} new papers so far", flush=True)
            time.sleep(3)
            if not papers:
                break
    print(f"Done. {len(seen)} papers in {os.path.abspath(OUT)}")


if __name__ == "__main__":
    n = 2500
    if "--per-cat" in sys.argv:
        n = int(sys.argv[sys.argv.index("--per-cat") + 1])
    main(n)
