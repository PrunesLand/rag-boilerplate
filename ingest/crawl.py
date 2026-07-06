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
    """BFS crawl from seeds, same-domain only, returns list[Document]."""
    sources_file = sources_file or config.SOURCES_FILE
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
