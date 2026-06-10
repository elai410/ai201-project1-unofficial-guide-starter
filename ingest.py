#!/usr/bin/env python3
"""
Stages 1 + 2: Document ingestion, cleaning, and chunking
Yale Residential College Unofficial Guide

Pipeline per document:
  check documents/saved/<slug>.txt for manually-saved content  ← new
    OR fetch URL via requests
  → strip tags → save documents/raw/<slug>.txt  (raw, before cleaning)
  → decode entities + remove boilerplate         (cleaning)
  → tiktoken chunk 500 tok / 75 tok overlap      (chunking)
  → documents/chunks.json                        (output for Stage 3)

HANDLING BLOCKED SOURCES (YDN 429, Medium 403, etc.)
─────────────────────────────────────────────────────
1. Open the URL in your browser and read the article normally.
2. Select all text (Cmd+A), copy, paste into a plain .txt file.
   OR use File → Save Page As → Web Page, Complete (.html).
3. Drop the file into documents/saved/ named with any slug you like,
   e.g.  documents/saved/ydn-dining-ranking.txt
4. Add a "saved_path" key to the matching entry in SOURCES (see examples).
5. Re-run python ingest.py — it loads from disk instead of fetching.

Run:
    pip install -r requirements.txt
    python ingest.py
"""

import html as html_lib
import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
import tiktoken
from bs4 import BeautifulSoup


# ──────────────────────────────────────────────────────────────────────────────
# Sources  (planning.md § Documents, all 15 entries)
#
# Add  "saved_path": "documents/saved/some-file.txt"  to any source whose
# URL is blocked, and drop the saved text file there.  The script will load
# from disk instead of fetching.
# ──────────────────────────────────────────────────────────────────────────────

SOURCES = [
    {
        "title": "Yale Herald Residential College Rankings",
        "url": "https://medium.com/the-yale-herald/yale-herald-best-residential-colleges-official-rankings-f4fe5c515a1e",
        "saved_path": "documents/saved/yale-herald.txt",
    },
    {
        "title": 'Yale Daily News — "A Very Reliable Ranking" (2023)',
        "url": "https://yaledailynews.com/blog/2023/08/31/a-very-reliable-ranking-of-the-residential-colleges/",
        "saved_path": "documents/saved/ydn-source-2.txt",
    },
    {
        "title": "Yale Daily News — Best and Worst of Yale Dining (2025)",
        "url": "https://yaledailynews.com/blog/2025/02/09/data-the-best-and-worst-of-yale-dining/",
        "saved_path": "documents/saved/ydn-source-3.txt",
    },
    {
        "title": "Yale Daily News — Buttery Prices Rise (2024)",
        "url": "https://yaledailynews.com/blog/2024/10/08/prices-rise-at-some-residential-college-butteries/",
        "saved_path": "documents/saved/ydn-source-4.txt",
    },
    {
        "title": "Yale Daily News — Housing Luck of the Draw",
        "url": "https://yaledailynews.com/articles/in-housing-the-luck-of-the-draw",
        "saved_path": "documents/saved/ydn-source-5.txt",
    },
    {
        "title": "Yale Daily News — 72 Residential College Transfer Requests (2025)",
        "url": "https://yaledailynews.com/blog/2025/02/09/deans-office-receives-72-residential-college-transfer-requests-approves-nearly-three-quarters/",
        "saved_path": "documents/saved/ydn-source-6.txt",
    },
    {
        "title": "College Confidential — Best Residential Colleges Ranked",
        "url": "https://talk.collegeconfidential.com/t/best-residential-colleges-ranked/2093609",
        "saved_path": "documents/saved/collegeconfidential-source-7.txt",
    },
    {
        "title": "College Confidential — Which College Is the Best?",
        "url": "https://talk.collegeconfidential.com/t/objectively-speaking-which-residential-college-is-the-best/718108",
        "saved_path": "documents/saved/collegeconfidential-source-8.txt",
    },
    {
        "title": "Quora — What Is the Best Residential College at Yale?",
        "url": "https://www.quora.com/What-is-the-best-residential-college-at-Yale",
        "_js_only": True,
        "saved_path": "documents/saved/quora-source-9.txt",
    },
    {
        "title": "Quora — What Is the Worst Residential College at Yale?",
        "url": "https://www.quora.com/What-is-the-worst-residential-college-at-Yale",
        "_js_only": True,
        "saved_path": "documents/saved/quora-source-10.txt",
    },
    {
        "title": "Roomsurf — Yale Dorm Reviews",
        "url": "https://www.roomsurf.com/dorm-reviews/yale",
    },
    {
        "title": "Forward Pathway — Yale's Late-Night Butteries",
        "url": "https://www.forwardpathway.us/yales-late-night-butteries-a-unique-student-run-culinary-and-social-hub-in-residential-colleges",
        "saved_path": "documents/saved/forward-pathway-source-12.txt",
    },
    {
        "title": "Yale Admissions Blog — Residential Colleges Debunked",
        "url": "https://admissions.yale.edu/bulldogs-blogs/bernice/2022/03/31/residential-colleges-yale-debunked",
        # "saved_path": "documents/saved/yale-admissions-debunked.txt",
    },
    {
        "title": "Wikipedia — Residential Colleges of Yale University",
        "url": "https://en.wikipedia.org/wiki/Residential_colleges_of_Yale_University",
    },
    {
        "title": "Yale Housing — Room Draw FAQs",
        "url": "https://housing.yale.edu/undergraduate-housing/frequently-asked-questions/room-draw-faqs",
        # "saved_path": "documents/saved/yale-housing-faq.txt",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

RAW_DIR    = Path("documents/raw")
SAVED_DIR  = Path("documents/saved")
CHUNKS_OUT = Path("documents/chunks.json")
MIN_CHARS  = 200

# Entire tag subtrees to drop before text extraction
_DROP_TAGS = [
    "nav", "footer", "header", "aside", "script", "style",
    "noscript", "form", "iframe", "figure",
]

# Lone-quote pattern built with chr() to avoid literal curly-quote chars in source.
# Matches lines that contain only a quote mark — e.g. Roomsurf review delimiters.
_LONE_QUOTE_RE = re.compile(
    r"^\s*[" + chr(0x22) + chr(0x27) + chr(0x201C) + chr(0x201D)
    + chr(0x2018) + chr(0x2019) + r"]\s*$",
    re.MULTILINE,
)

# Wikipedia appendix sections whose content we don't want
_WIKI_SKIP_SECTIONS = {
    "See also", "References", "External links", "Notes",
    "Further reading", "Bibliography", "Citations",
}

# CSS selectors tried in order — first match wins
_CONTENT_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    ".post-content",
    ".article-body",
    ".entry-content",
    ".article__body",
    ".story-body",
    "#mw-content-text",   # Wikipedia
    "#bodyContent",        # Wikipedia fallback
    ".content",
    "#content",
]


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1a — extract raw text (HTML tag removal only, no text cleaning yet)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_raw_from_html(html_str: str) -> str:
    """
    Strip HTML tags and return newline-separated text.
    Does NOT decode entities or apply pattern cleaning.
    This output is saved to documents/raw/ before any cleaning decisions.
    """
    soup = BeautifulSoup(html_str, "html.parser")

    for tag in soup(_DROP_TAGS):
        tag.decompose()

    # Drop Wikipedia reference/appendix sections.
    # Two strategies combined, because Wikipedia structures references in two ways:
    #   (a) As content after a heading — remove via heading detection
    #   (b) As <ol class="references"> / <div class="reflist"> elements — remove directly

    # Strategy (a): heading-based removal
    # Wikipedia headings contain mw-editsection spans; strip those before comparing.
    for heading in soup.find_all(["h2", "h3"]):
        for edit_span in heading.find_all(class_=re.compile(r"mw-editsection")):
            edit_span.decompose()
        heading_text = heading.get_text(strip=True)
        if heading_text in _WIKI_SKIP_SECTIONS:
            for sibling in list(heading.find_next_siblings()):
                if sibling.name in ["h2", "h3"]:
                    break
                sibling.decompose()
            heading.decompose()

    # Strategy (b): drop reference list elements directly
    for ref_el in soup.find_all(class_=re.compile(
        r"reflist|references|mw-references-wrap|citation|footnotes", re.I
    )):
        ref_el.decompose()

    for selector in _CONTENT_SELECTORS:
        el = soup.select_one(selector)
        if el:
            return el.get_text(separator="\n", strip=True)

    body = soup.find("body")
    return (body or soup).get_text(separator="\n", strip=True)


def _url_to_slug(url: str) -> str:
    slug = re.sub(r"https?://", "", url)
    slug = re.sub(r"[^\w]+", "-", slug).strip("-")
    return slug[:80]


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1b — clean text
# ──────────────────────────────────────────────────────────────────────────────

# Applied after html.unescape(), in order.
# Use re.DOTALL where a pattern may span newlines (e.g. Wikipedia citations).
_CLEAN_PATTERNS = [
    # ── Wikipedia ────────────────────────────────────────────────────────────
    # Citation brackets that may span lines: [ \n 70 \n ] or [70] or [ a ]
    (re.compile(r"\[\s*\d+\s*\]",                re.DOTALL), ""),
    (re.compile(r"\[\s*[a-zA-Z]\s*\]",           re.DOTALL), ""),
    (re.compile(r"\[\s*edit\s*\]",               re.DOTALL | re.IGNORECASE), ""),
    (re.compile(r"\[\s*citation\s+needed\s*\]",  re.DOTALL | re.IGNORECASE), ""),

    # ── Roomsurf / review-site boilerplate ───────────────────────────────────
    # "See all reviews →" and variants
    (re.compile(r"See\s+all\s+reviews\s*[→>]?",          re.IGNORECASE), ""),
    # "See Yale Dorms Ranked"
    (re.compile(r"See\s+Yale\s+Dorms?\s+Ranked",          re.IGNORECASE), ""),
    # "Need a Roommate? Get Started Here." (may span lines)
    (re.compile(r"Need a Roommate\?.*?Get Started.*?$",    re.IGNORECASE | re.MULTILINE | re.DOTALL), ""),
    # "Get Started" on its own line
    (re.compile(r"^\s*Get Started\s*$",                    re.MULTILINE | re.IGNORECASE), ""),
    # Lone "more" truncation links
    (re.compile(r"^\s*more\s*$",                           re.MULTILINE | re.IGNORECASE), ""),
    # Wikipedia footnote anchor lines: lines that are just "^" or "^ a b c"
    (re.compile(r"^\s*\^[\s\w]*$",                         re.MULTILINE), ""),
    # Lines containing only "Archived from the original" (wiki reference boilerplate)
    (re.compile(r"^.*Archived from the original.*$",        re.MULTILINE | re.IGNORECASE), ""),
    # Lines containing only "Retrieved <date>" (wiki reference boilerplate)
    (re.compile(r"^.*Retrieved\s+\w.*$",                   re.MULTILINE | re.IGNORECASE), ""),
    # Lone decorative/curly quote marks used as visual dividers (e.g. Roomsurf)
    (_LONE_QUOTE_RE, ""),

    # ── YDN footer boilerplate (appears in manually-pasted text) ─────────────
    # Matches from "YDN Logo" or the address block onward
    (re.compile(
        r"YDN Logo\s*The Yale Daily News is.*",
        re.IGNORECASE | re.DOTALL,
    ), ""),
    # Fallback: address block anchor
    (re.compile(
        r"Yale Daily News Publishing Company.*",
        re.IGNORECASE | re.DOTALL,
    ), ""),

    # ── Medium / publication footer boilerplate ───────────────────────────────
    # "Help\nStatus\nAbout\nCareers\nPress\nBlog\nStore\nPrivacy\nRules\nTerms"
    (re.compile(
        r"\bHelp\s+Status\s+About\s+Careers.*",
        re.IGNORECASE | re.DOTALL,
    ), ""),
    # "Published in The Yale Herald\n236 followers" block and similar
    (re.compile(
        r"Published in .+?\n\d+ followers.*",
        re.IGNORECASE | re.DOTALL,
    ), ""),
    # "Written by <Name>\n<N> followers\n<N> following\nFollow" block
    (re.compile(
        r"Written by .+?\n\d+ followers\s*\n\d+ following.*",
        re.IGNORECASE | re.DOTALL,
    ), ""),

    # ── Emoji — strip emoji bodies and their variation selectors ─────────────
    (re.compile(
        "["
        "\U0001F000-\U0001FFFF"  # all SMP emoji / pictographs (broad range)
        "\U00002700-\U000027BF"  # Dingbats block
        "\U00002600-\U000026FF"  # Miscellaneous symbols
        "\U0000FE00-\U0000FE0F"  # Variation selectors (️ residuals after emoji strip)
        "\U00020000-\U0002FA1F"  # CJK / supplementary ideographs (safety net)
        "]+",
        flags=re.UNICODE,
    ), ""),

    # ── Generic web boilerplate (lone lines) ─────────────────────────────────
    (re.compile(
        r"^\s*(Share|Tweet|Print|Email|Save|Pin|Copy\s*link|"
        r"Facebook|Twitter|LinkedIn|WhatsApp|Reddit|Flipboard)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ), ""),
    (re.compile(r"\bRead\s+more[:\s]*",    re.IGNORECASE), ""),
    (re.compile(r"\bRead\s+Next[:\s]*",    re.IGNORECASE), ""),
    (re.compile(
        r"^\s*(Subscribe|Sign\s*in|Sign\s*up|Log\s*in|Register|"
        r"Already a subscriber\?|Get unlimited access|"
        r"Create a free account|Continue reading)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ), ""),
    (re.compile(r"\d+\s+comments?\b",       re.IGNORECASE), ""),
    (re.compile(r"Comments?\s*\(\d+\)",     re.IGNORECASE), ""),
    (re.compile(r"https?://\S+"),                           ""),  # bare URLs

    # ── Whitespace normalization ──────────────────────────────────────────────
    (re.compile(r"\xa0"),            " "),   # non-breaking space → regular space
    (re.compile(r"[ \t]{2,}"),       " "),   # runs of spaces/tabs → one space
    (re.compile(r" \n"),             "\n"),  # trailing space before newline
    (re.compile(r"\n "),             "\n"),  # leading space after newline
    (re.compile(r"\n{3,}"),          "\n\n"),  # 3+ blank lines → one blank line
]


def clean_text(raw: str) -> str:
    """
    1. html.unescape() — &amp; → &   &nbsp; → space   &#39; → '
    2. Apply _CLEAN_PATTERNS to remove boilerplate and normalize whitespace.
    """
    text = html_lib.unescape(raw)
    for pattern, replacement in _CLEAN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion driver
# ──────────────────────────────────────────────────────────────────────────────

def _load_saved(path: str) -> Optional[str]:
    """Load a manually-saved .txt or .html file and return its raw text."""
    p = Path(path)
    if not p.exists():
        return None
    content = p.read_text(encoding="utf-8", errors="replace")
    # If it looks like HTML, extract text; otherwise treat as plain text
    if content.lstrip().startswith("<"):
        return _extract_raw_from_html(content)
    return content


def ingest_documents(sources: list[dict], delay: float = 1.5) -> list[dict]:
    """
    For each source:
      1. If "saved_path" is set and the file exists → load from disk
         Else if "_js_only" → skip with instructions
         Else → fetch URL
      2. Extract raw text (HTML tags stripped, entities still encoded)
      3. Save raw text to documents/raw/<slug>.txt for inspection
      4. Skip if too short (bot-wall, empty page)
      5. Clean text and return

    Returns list of {text, source_url, source_title, raw_chars, clean_chars}.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    docs = []

    for source in sources:
        url   = source["url"]
        title = source["title"]
        slug  = _url_to_slug(url)

        # ── Option A: load from manually-saved file ───────────────────────────
        saved_path = source.get("saved_path")
        if saved_path:
            raw = _load_saved(saved_path)
            if raw is None:
                print(f"  [MISS ] {title}")
                print(f"           saved_path set but file not found: {saved_path}")
                continue
            print(f"  [FILE ] {title}")
            print(f"           loaded from {saved_path}  ({len(raw):,} chars)")
        # ── Option B: skip JS-only sources (unless they have saved_path) ──────
        elif source.get("_js_only"):
            print(f"  [SKIP ] {title}")
            print(f"           JavaScript-rendered. To ingest:")
            print(f"           1. Open URL in browser, copy all text, save to documents/saved/")
            print(f"           2. Add  \"saved_path\": \"documents/saved/<file>.txt\"  to SOURCES")
            continue
        # ── Option C: fetch URL ───────────────────────────────────────────────
        else:
            try:
                resp = requests.get(url, headers=HEADERS, timeout=20)
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                code = e.response.status_code
                print(f"  [FAIL ] {title}  (HTTP {code})")
                if code == 429:
                    print(f"           Rate-limited. Save the page manually → documents/saved/")
                elif code in (403, 401):
                    print(f"           Bot-wall/auth required. Save manually → documents/saved/")
                continue
            except Exception as e:
                print(f"  [FAIL ] {title}  ({type(e).__name__}: {e})")
                continue
            raw = _extract_raw_from_html(resp.text)

        # ── Save raw before cleaning ──────────────────────────────────────────
        raw_path = RAW_DIR / f"{slug}.txt"
        raw_path.write_text(raw, encoding="utf-8")

        if len(raw) < MIN_CHARS:
            print(f"  [WARN ] {title}  — only {len(raw)} chars after extraction")
            print(f"           Likely a bot-wall page. Raw saved to {raw_path.name}")
            continue

        # ── Clean ─────────────────────────────────────────────────────────────
        cleaned = clean_text(raw)
        removed = (1 - len(cleaned) / max(len(raw), 1)) * 100

        docs.append({
            "text":         cleaned,
            "source_url":   url,
            "source_title": title,
            "raw_chars":    len(raw),
            "clean_chars":  len(cleaned),
        })
        print(f"  [OK   ] {title}")
        print(f"           raw {len(raw):,} → cleaned {len(cleaned):,} chars  "
              f"(-{removed:.0f}%)  saved: raw/{raw_path.name}")

        if not saved_path:
            time.sleep(delay)   # polite delay only when fetching live

    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — chunking  (spec: 500 tokens, 75 overlap, tiktoken cl100k_base)
# ──────────────────────────────────────────────────────────────────────────────

def chunk_document(doc: dict, chunk_size: int = 500, overlap: int = 75) -> list[dict]:
    """
    Sliding-window token split.
    Window advances by (chunk_size - overlap) = 425 tokens each step.
    Every chunk carries its source metadata so retrieval results are self-contained.
    """
    enc    = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(doc["text"])

    if not tokens:
        return []

    step   = chunk_size - overlap
    chunks = []
    start  = 0

    while start < len(tokens):
        end  = min(start + chunk_size, len(tokens))
        toks = tokens[start:end]
        chunks.append({
            "chunk_text":   enc.decode(toks),
            "source_url":   doc["source_url"],
            "source_title": doc["source_title"],
            "chunk_index":  len(chunks),
            "token_count":  len(toks),
        })
        if end >= len(tokens):
            break
        start += step

    return chunks


def chunk_all(docs: list[dict], chunk_size: int = 500, overlap: int = 75) -> list[dict]:
    result = []
    for doc in docs:
        result.extend(chunk_document(doc, chunk_size, overlap))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Verification output
# ──────────────────────────────────────────────────────────────────────────────

def print_one_document(docs: list[dict]) -> None:
    """
    Print the full cleaned text of the first document.
    Read it manually: HTML tags? &amp; / &#39;? Nav text? Emoji?
    Add patterns to _CLEAN_PATTERNS and re-run if you see anything that doesn't belong.
    """
    doc = docs[0]
    print(f"  Source : {doc['source_title']}")
    print(f"  URL    : {doc['source_url']}")
    print(f"  Length : {doc['clean_chars']:,} chars  (raw was {doc['raw_chars']:,})")
    print()
    cap     = 5_000
    preview = doc["text"][:cap]
    print(preview)
    if len(doc["text"]) > cap:
        print(f"\n  ... [{len(doc['text']) - cap:,} more chars — full file in documents/raw/]")


def print_five_chunks(all_chunks: list[dict]) -> None:
    """
    Print the middle chunk from each of the first 5 sources.
    Middle chunks are more representative than index-0 chunks (which often
    capture title/header text rather than body content).
    """
    by_source: dict[str, list[dict]] = {}
    for chunk in all_chunks:
        by_source.setdefault(chunk["source_title"], []).append(chunk)

    for i, (src, chunks) in enumerate(list(by_source.items())[:5], 1):
        mid = chunks[len(chunks) // 2]
        print(f"\n{'─' * 64}")
        print(f"CHUNK {i} / 5")
        print(f"{'─' * 64}")
        print(f"Source : {mid['source_title']}")
        print(f"Tokens : {mid['token_count']}  |  chunk_index={mid['chunk_index']}")
        print(f"{'─' * 64}")
        print(mid["chunk_text"])
        print(f"{'─' * 64}")
        print("CHECK: standalone? no HTML? no &amp;/&#39;/emoji? domain-relevant content?")


def print_chunk_stats(all_chunks: list[dict]) -> None:
    if not all_chunks:
        print("  No chunks produced.")
        return

    tok_counts = [c["token_count"] for c in all_chunks]
    total      = len(tok_counts)
    avg        = sum(tok_counts) / total
    max_tok    = max(tok_counts)
    over_500   = sum(1 for t in tok_counts if t > 500)

    by_source: dict[str, int] = {}
    for c in all_chunks:
        by_source[c["source_title"]] = by_source.get(c["source_title"], 0) + 1

    print(f"  Total chunks   : {total}")
    print(f"  Sources used   : {len(by_source)}")
    print(f"  Avg tokens/chk : {avg:.1f}")
    print(f"  Max tokens     : {max_tok}  {'OK' if max_tok <= 500 else 'EXCEEDS 500'}")
    print(f"  Chunks > 500   : {over_500}  (should be 0)")
    print()
    print("  Chunks per source:")
    for src, n in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"    {n:4d}  {src}")

    print()
    if total < 50:
        print(f"  !! {total} chunks — BELOW 50. Need more sources or smaller chunk_size.")
        print(f"     Each chunk covers too much ground; specific queries won't match.")
        print(f"     → Add saved documents for blocked sources (see SOURCES comments).")
    elif total > 2_000:
        print(f"  !! {total} chunks — ABOVE 2,000. Chunks may be too small.")
        print(f"     Each embedding carries too little meaning for reliable similarity search.")
    else:
        print(f"  OK  {total} chunks — healthy range (50–2,000).")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> list[dict]:
    js_only   = sum(1 for s in SOURCES if s.get("_js_only") and not s.get("saved_path"))
    with_file = sum(1 for s in SOURCES if s.get("saved_path"))
    fetchable = len(SOURCES) - js_only

    print("=" * 64)
    print("STAGE 1 — Ingestion + Cleaning")
    print(f"{len(SOURCES)} sources  |  {js_only} JS-only (skipped)  "
          f"|  {with_file} from saved files  |  {fetchable - with_file} to fetch")
    print("=" * 64)
    docs = ingest_documents(SOURCES)
    failed = fetchable - len(docs)
    print(f"\nResult: {len(docs)} docs ingested  |  {failed} failed/too-short\n")

    if not docs:
        print("No documents ingested.")
        print("For blocked URLs: save page text to documents/saved/ and add saved_path to SOURCES.")
        return []

    # ── Manual cleaning check ─────────────────────────────────────────────────
    print("=" * 64)
    print("CLEANING CHECK — read this output carefully")
    print("Look for: HTML tags  |  &amp; &#39; &nbsp;  |  emoji  |  nav text")
    print("Any found? Add a pattern to _CLEAN_PATTERNS and re-run.")
    print("=" * 64)
    print_one_document(docs)

    # ── Chunk ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("STAGE 2 — Chunking  (chunk_size=500, overlap=75 tokens)")
    print("=" * 64)
    all_chunks = chunk_all(docs)
    print_chunk_stats(all_chunks)

    # ── 5 representative chunks ───────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("CHUNK INSPECTION — middle chunk from each of 5 sources")
    print("=" * 64)
    print_five_chunks(all_chunks)

    # ── Save ──────────────────────────────────────────────────────────────────
    CHUNKS_OUT.parent.mkdir(exist_ok=True)
    with open(CHUNKS_OUT, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print("\n" + "=" * 64)
    print(f"Saved  {len(all_chunks)} chunks  →  {CHUNKS_OUT}")
    print(f"Saved  {len(docs)} raw files  →  {RAW_DIR}/")
    print(f"Next:  python embed.py   (Stage 3 — embedding + ChromaDB)")
    print("=" * 64)

    return all_chunks


if __name__ == "__main__":
    main()
