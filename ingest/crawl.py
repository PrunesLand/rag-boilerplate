from collections import deque
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse

import trafilatura
from langchain_core.documents import Document

import config

EXCLUDE_HINTS = ("/login", "/signin", "/calendar", "/search", ".pdf", ".jpg", ".png", ".zip")

# Cache one robots parser per domain.
_robots_cache = {}


def _allowed_by_robots(url):
    """Respect robots.txt when RESPECT_ROBOTS_TXT is on. Fail open on errors."""
    if not config.RESPECT_ROBOTS_TXT:
        return True
    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    if root not in _robots_cache:
        rp = RobotFileParser()
        rp.set_url(f"{root}/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None  # unreachable robots.txt -> allow
        _robots_cache[root] = rp
    rp = _robots_cache[root]
    return rp.can_fetch("*", url) if rp else True


def _read_sources(path):
    urls = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def _same_domain(a, b):
    return urlparse(a).netloc == urlparse(b).netloc


def _should_skip(url):
    low = url.lower()
    return any(h in low for h in EXCLUDE_HINTS)


def _extract(html, url):
    """Return (clean_text, title) or (None, None) if extraction failed."""
    text = trafilatura.extract(html, include_comments=False, include_tables=True, url=url)
    if not text:
        return None, None
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else url)
    return text, title


def crawl(sources_file=None, max_depth=None):
    """Fetch source pages as list[Document], dispatching on config.CRAWL_SOURCE."""
    sources_file = sources_file or config.SOURCES_FILE
    if config.CRAWL_SOURCE == "commoncrawl":
        return _crawl_commoncrawl(sources_file)
    return _crawl_live(sources_file, max_depth)


def _crawl_commoncrawl(sources_file):
    """Pull archived pages from Common Crawl's CDX index, one query per pattern."""
    import cdx_toolkit

    patterns = [_to_pattern(s) for s in _read_sources(sources_file)]
    crawl_sel = config.CC_CRAWL or _recent_cc_crawls(config.CC_RECENT_CRAWLS)
    print(f"  [cc] snapshot(s): {crawl_sel}")
    cdx = cdx_toolkit.CDXFetcher(source="cc", crawl=crawl_sel)

    seen = set()
    docs = []
    for pattern in patterns:
        _query_pattern(cdx, pattern, seen, docs)

    print(f"Fetched {len(docs)} pages from Common Crawl.")
    return docs


def _query_pattern(cdx, pattern, seen, docs):
    """Query one CDX pattern, appending kept Documents to `docs`.

    Retries with backoff: cdx_toolkit raises a fatal ValueError on a transient
    index blip. `seen` keeps retries idempotent.
    """
    import time

    for attempt in range(1, config.CC_RETRIES + 1):
        print(f"  [cc] querying index for {pattern}"
              + (f" (attempt {attempt})" if attempt > 1 else ""))
        found = 0
        try:
            for obj in cdx.iter(
                pattern,
                limit=config.CC_LIMIT_PER_PATTERN,
                filter=["status:200", "mime:text/html"],
            ):
                url = obj["url"]
                if url in seen or _should_skip(url):
                    continue
                seen.add(url)

                if config.CC_LANGUAGES:
                    langs = (obj.get("languages") or "").split(",")
                    if not any(l in config.CC_LANGUAGES for l in langs):
                        continue

                html = _decode(obj.content)
                if not html:
                    continue
                text, title = _extract(html, url)
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"url": url, "title": title, "captured": obj["timestamp"]},
                    ))
                    found += 1
                    print(f"  [ok] {title[:60]} ({len(text)} chars) {url}")
            print(f"  [cc] {found} pages kept for {pattern}")
            return
        except (ValueError, ConnectionError) as e:
            wait = min(5 * attempt, 30)
            if attempt == config.CC_RETRIES:
                print(f"  [cc] giving up on {pattern} after {attempt} attempts: {e}")
                return
            print(f"  [cc] transient error on {pattern}: {e}; retrying in {wait}s")
            time.sleep(wait)


def _recent_cc_crawls(n):
    """Return the ids of the newest n Common Crawl snapshots (lexical == chrono)."""
    import re
    import cdx_toolkit

    endpoints = cdx_toolkit.CDXFetcher(source="cc").raw_index_list
    ids = sorted({
        m.group(0) for u in endpoints
        if (m := re.search(r"CC-MAIN-\d{4}-\d{2}", u))
    })
    return ids[-n:]


def _to_pattern(source):
    """Normalise a sources.txt line to a CDX pattern: drop scheme, add "/*" if
    it carries no wildcard (so "https://example.org/" -> "example.org/*")."""
    s = source.strip()
    for scheme in ("https://", "http://"):
        if s.startswith(scheme):
            s = s[len(scheme):]
            break
    s = s.rstrip("/")
    if "*" not in s:
        s += "/*"
    return s


def _decode(content):
    """CDX WARC payloads arrive as bytes; hand trafilatura a str."""
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return content


def _crawl_live(sources_file, max_depth):
    """BFS crawl from seeds, same-domain only, returns list[Document]."""
    max_depth = config.CRAWL_MAX_DEPTH if max_depth is None else max_depth

    seeds = _read_sources(sources_file)
    seen = set()
    queue = deque((u, 0) for u in seeds)
    docs = []

    while queue:
        url, depth = queue.popleft()
        if url in seen or _should_skip(url):
            continue
        seen.add(url)

        if not _allowed_by_robots(url):
            print(f"  [robots] disallowed {url}")
            continue

        html = trafilatura.fetch_url(url)
        if not html:
            print(f"  [skip] could not fetch {url}")
            continue

        text, title = _extract(html, url)
        if text:
            docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
            print(f"  [ok] {title[:60]} ({len(text)} chars) {url}")

        if depth < max_depth:
            for link in _discover_links(html, url):
                if link not in seen and _same_domain(link, url):
                    queue.append((link, depth + 1))

    print(f"Crawled {len(docs)} pages.")
    return docs


def _discover_links(html, base_url):
    """Cheap link discovery so we can crawl one or two levels deep."""
    import re

    links = set()
    for match in re.findall(r'href=["\']([^"\']+)["\']', html):
        joined = urljoin(base_url, match.split("#")[0])
        if joined.startswith("http") and not _should_skip(joined):
            links.add(joined)
    return links


if __name__ == "__main__":
    crawl()
