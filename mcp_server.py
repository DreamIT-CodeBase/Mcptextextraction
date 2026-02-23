# mcp_server.py
import os
import re
import time
from typing import Any, List, Dict, Optional
from urllib.parse import urljoin, urlparse

import httpx
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from mcp.server.fastmcp import FastMCP
from requests import RequestException

# -------- CONFIG --------
DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))
USER_AGENT = "Mozilla/5.0 (compatible; BusinessWebsiteFinder/1.0)"
SCRAPE_TIMEOUT = 15
PAGE_TEXT_LIMIT = int(os.getenv("PAGE_TEXT_LIMIT", "5000"))  # words per page
MAX_PAGES_TO_SCRAPE = int(os.getenv("MAX_PAGES_TO_SCRAPE", "10"))
MIN_TEXT_BLOCK_CHARS = int(os.getenv("MIN_TEXT_BLOCK_CHARS", "20"))
DEFAULT_SPECIFIED_PAGES = ["/", "/about-us", "/services", "/industries", "/insights", "/contact"]

# Simple in-memory cache to reduce repeated searches/scrapes in short time
_SIMPLE_CACHE: Dict[str, Dict] = {}
CACHE_TTL = 300  # seconds


def cache_get(key: str) -> Optional[Dict]:
    entry = _SIMPLE_CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > CACHE_TTL:
        del _SIMPLE_CACHE[key]
        return None
    return entry["value"]


def cache_set(key: str, value: Dict):
    _SIMPLE_CACHE[key] = {"ts": time.time(), "value": value}


# -------- MCP SERVER --------
mcp = FastMCP("Company Research Tools (search+scrape)")

# -------- Helper: DuckDuckGo search (synchronous) --------
def search_duckduckgo(query: str, max_results: int = MAX_RESULTS) -> List[str]:
    """
    Minimal DuckDuckGo HTML scrape to return result URLs. Synchronous by design
    so it can be reused in sync contexts; usage inside MCP tool is fine.
    """
    headers = {"User-Agent": USER_AGENT}
    params = {"q": query}

    try:
        resp = requests.post(
            DUCKDUCKGO_URL, data=params, headers=headers, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
    except RequestException as exc:
        raise HTTPException(status_code=502, detail="Upstream search provider unavailable") from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[str] = []
    for link in soup.select("a.result__a"):
        href = link.get("href")
        if href and href.startswith("http"):
            results.append(href)
            if len(results) >= max_results:
                break
    return results


# -------- MCP TOOL 1: find_company_website --------
@mcp.tool()
async def find_company_website(
    company_name: str,
    max_results: int = 1,
    include_candidates: bool = False,
) -> dict:
    """
    Input: company_name (string)
    Output: dict { business_name, official_website }
    Optional fields when include_candidates=True:
      - candidates
      - searched_results_count
    """
    safe_max_results = max(1, min(int(max_results), MAX_RESULTS))
    key = f"search:{company_name.lower()}:{safe_max_results}:{int(include_candidates)}"
    cached = cache_get(key)
    if cached:
        return cached

    query = f"{company_name.strip()} official website"
    results = search_duckduckgo(query, max_results=safe_max_results)

    official_site = results[0] if results else None
    payload = {
        "business_name": company_name,
        "official_website": official_site,
    }
    if include_candidates:
        payload["candidates"] = results
        payload["searched_results_count"] = len(results)
    cache_set(key, payload)
    return payload


def _safe_page_key(page_url: str) -> str:
    parsed = urlparse(page_url)
    path = (parsed.path or "").strip("/")
    if not path:
        return "homepage"
    key = re.sub(r"[^a-zA-Z0-9_/-]+", "_", path).replace("/", "_").strip("_")
    return key or "page"


def _truncate_words(text: str, max_words: Optional[int]) -> str:
    if max_words is None:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def extract_page_details(
    page_url: str,
    html: str,
    text_limit: Optional[int] = PAGE_TEXT_LIMIT,
    include_html: bool = False,
) -> Dict[str, Any]:
    """Extract full page details (metadata + structured text + links)."""
    original_soup = BeautifulSoup(html, "html.parser")

    title_tag = original_soup.find("title")
    meta_desc_tag = original_soup.find("meta", attrs={"name": "description"})
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    meta_description = (meta_desc_tag.get("content", "") or "").strip() if meta_desc_tag else ""

    links: List[str] = []
    for a in original_soup.find_all("a", href=True):
        full = urljoin(page_url, a["href"])
        if full not in links:
            links.append(full)

    # Separate soup for visible text extraction.
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        tag.decompose()

    content_root = soup.find("main") or soup.find("article") or soup.body or soup
    headings = {
        "h1": [h.get_text(" ", strip=True) for h in content_root.find_all("h1")],
        "h2": [h.get_text(" ", strip=True) for h in content_root.find_all("h2")],
        "h3": [h.get_text(" ", strip=True) for h in content_root.find_all("h3")],
    }
    paragraphs = [
        re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        for p in content_root.find_all("p")
        if len(p.get_text(" ", strip=True)) >= MIN_TEXT_BLOCK_CHARS
    ]
    list_items = [
        re.sub(r"\s+", " ", li.get_text(" ", strip=True)).strip()
        for li in content_root.find_all("li")
        if len(li.get_text(" ", strip=True)) >= MIN_TEXT_BLOCK_CHARS
    ]

    # Full visible text from the page body/content root.
    merged = re.sub(r"\s+", " ", content_root.get_text(" ", strip=True)).strip()
    full_text = _truncate_words(merged, text_limit)

    payload: Dict[str, Any] = {
        "url": page_url,
        "title": title,
        "meta_description": meta_description,
        "headings": headings,
        "paragraphs": paragraphs,
        "list_items": list_items,
        "links": links,
        "text": full_text,
        "text_length": len(full_text),
        "full_text_length": len(merged),
        "html_length": len(html),
    }
    if include_html:
        payload["html"] = html
    return payload


def _remove_image_urls(text: str) -> str:
    cleaned = re.sub(
        r"https?://\S+\.(?:png|jpg|jpeg|gif|webp|svg|avif|bmp|ico)(?:\?\S*)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"/\S+\.(?:png|jpg|jpeg|gif|webp|svg|avif|bmp|ico)(?:\?\S*)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def to_text_only_pages(page_details: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Convert rich per-page payload into plain full text per page key."""
    text_pages: Dict[str, str] = {}
    for key, details in page_details.items():
        text = details.get("text")
        if isinstance(text, str) and text:
            text_pages[key] = _remove_image_urls(text)
    return text_pages


def _normalize_host(url_or_host: str) -> str:
    parsed = urlparse(url_or_host)
    host = (parsed.netloc or parsed.path or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _same_domain(base_url: str, candidate_url: str) -> bool:
    return _normalize_host(base_url) == _normalize_host(candidate_url)


def _strip_url_noise(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    # Remove fragments and query to avoid duplicate pages with tracking params.
    return parsed._replace(path=path, query="", fragment="").geturl()


def _extract_internal_links(current_url: str, html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        full = urljoin(current_url, href)
        parsed = urlparse(full)
        if parsed.scheme not in ("http", "https"):
            continue
        if not _same_domain(base_url, full):
            continue
        cleaned = _strip_url_noise(full)
        if cleaned not in links:
            links.append(cleaned)
    return links


async def scrape_important_pages_async(
    base_url: str,
    text_limit: Optional[int] = PAGE_TEXT_LIMIT,
    max_pages: Optional[int] = None,
    page_urls: Optional[List[str]] = None,
    timeout_seconds: Optional[float] = None,
    include_html: bool = False,
    crawl_all_internal: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Fetch full details for page(s), optionally crawling all internal links."""
    safe_text_limit = PAGE_TEXT_LIMIT if text_limit is None else max(1, min(int(text_limit), PAGE_TEXT_LIMIT))
    safe_max_pages: Optional[int] = None
    if max_pages is not None:
        safe_max_pages = max(1, min(int(max_pages), 5000))
    safe_timeout: Optional[float]
    if timeout_seconds is None:
        safe_timeout = None
    else:
        safe_timeout = max(0.1, float(timeout_seconds))

    # Normalize base URL
    parsed = urlparse(base_url)
    if not parsed.scheme:
        base_url = "https://" + base_url  # assume https if missing

    page_urls_key = ""
    if page_urls:
        page_urls_key = ",".join(page_urls)
    cache_key = (
        f"scrape:{base_url}:{safe_text_limit}:{safe_max_pages}:{safe_timeout}:"
        f"{int(include_html)}:{int(crawl_all_internal)}:{page_urls_key}"
    )
    cached = cache_get(cache_key)
    if cached:
        return cached

    results: Dict[str, Dict[str, Any]] = {}
    seed_urls: List[str] = []
    if page_urls:
        for raw in page_urls:
            if not raw or not raw.strip():
                continue
            full = _strip_url_noise(urljoin(base_url, raw.strip()))
            if not _same_domain(base_url, full):
                continue
            if full not in seed_urls:
                seed_urls.append(full)
    if not seed_urls:
        seed_urls = [_strip_url_noise(base_url)]

    to_visit: List[str] = seed_urls[:]
    visited: List[str] = []

    async with httpx.AsyncClient(timeout=safe_timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        while to_visit:
            if safe_max_pages is not None and len(visited) >= safe_max_pages:
                break
            page_url = to_visit.pop(0)
            if page_url in visited:
                continue
            visited.append(page_url)

            key = _safe_page_key(page_url)
            final_key = key
            suffix = 2
            while final_key in results:
                final_key = f"{key}_{suffix}"
                suffix += 1

            try:
                r = await client.get(page_url)
                if r.status_code >= 400:
                    results[final_key] = {
                        "url": page_url,
                        "status_code": r.status_code,
                        "final_url": str(r.url),
                        "error": f"HTTP {r.status_code} {r.reason_phrase}",
                    }
                    continue

                details = extract_page_details(
                    page_url=str(r.url),
                    html=r.text,
                    text_limit=safe_text_limit,
                    include_html=include_html,
                )
                details["status_code"] = r.status_code
                details["final_url"] = str(r.url)
                results[final_key] = details

                if crawl_all_internal:
                    for link in _extract_internal_links(str(r.url), r.text, base_url):
                        if link not in visited and link not in to_visit:
                            to_visit.append(link)
            except Exception as exc:
                results[final_key] = {
                    "url": page_url,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    cache_set(cache_key, results)
    return results


# -------- MCP TOOL 2: scrape_site --------
@mcp.tool()
async def scrape_site(
    url: str,
    text_limit: Optional[int] = PAGE_TEXT_LIMIT,
    max_pages: Optional[int] = None,
    page_urls: Optional[List[str]] = None,
    no_timeout: bool = True,
    timeout_seconds: Optional[float] = None,
    include_html: bool = False,
    crawl_all_internal: bool = True,
    text_only: bool = True,
) -> dict:
    """
    Input: url (string)
    Output: dict { website: url, pages: { homepage: text, about: text, ... } }
    """
    effective_page_urls = page_urls or DEFAULT_SPECIFIED_PAGES
    effective_crawl = crawl_all_internal if page_urls else False
    effective_timeout = None if no_timeout else timeout_seconds
    pages = await scrape_important_pages_async(
        url,
        text_limit=text_limit,
        max_pages=max_pages,
        page_urls=effective_page_urls,
        timeout_seconds=effective_timeout,
        include_html=include_html,
        crawl_all_internal=effective_crawl,
    )
    if text_only:
        text_pages = to_text_only_pages(pages)
        return {"website": url, "pages": text_pages, "page_count": len(text_pages)}
    return {"website": url, "pages": pages, "page_count": len(pages)}


# -------- FASTAPI (REST wrapper) --------
app = FastAPI(title="Company Research MCP Server (search + scrape)")

@app.get("/")
async def root():
    return {"status": "running", "mcp_endpoint": "/mcp"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# REST proxy to MCP tool: find website
@app.get("/find-website")
async def find_website_api(
    business_name: str = Query(..., min_length=1),
    include_candidates: bool = Query(False),
    max_results: int = Query(1, ge=1, le=MAX_RESULTS),
):
    effective_max_results = max_results if include_candidates else 1
    return await find_company_website(
        business_name,
        max_results=effective_max_results,
        include_candidates=include_candidates,
    )

# REST proxy to MCP tool: scrape site
@app.get("/scrape-site")
async def scrape_site_api(
    url: str = Query(..., min_length=4),
    text_limit: int = Query(PAGE_TEXT_LIMIT, ge=1, le=PAGE_TEXT_LIMIT),
    max_pages: Optional[int] = Query(None, ge=1, le=5000),
    page: List[str] = Query(default=[]),
    no_timeout: bool = Query(True),
    timeout_seconds: Optional[float] = Query(None, gt=0),
    include_html: bool = Query(False),
    crawl_all_internal: Optional[bool] = Query(None),
    text_only: bool = Query(True),
):
    specified_pages: List[str] = []
    for p in page:
        parts = [x.strip() for x in p.split(",") if x.strip()]
        specified_pages.extend(parts)

    effective_crawl = crawl_all_internal
    if effective_crawl is None:
        # Default behavior: fetch only the fixed set of specified pages.
        effective_crawl = False

    effective_pages = specified_pages or DEFAULT_SPECIFIED_PAGES

    return await scrape_site(
        url,
        text_limit=text_limit,
        max_pages=max_pages,
        page_urls=effective_pages,
        no_timeout=no_timeout,
        timeout_seconds=timeout_seconds,
        include_html=include_html,
        crawl_all_internal=effective_crawl,
        text_only=text_only,
    )

# REST proxy: extract any URLs from a free-text string (agent could use find_company_website OR extract URLs from text)
@app.post("/extract-urls")
async def extract_urls_api(payload: dict):
    text = payload.get("text", "") or ""
    pattern = r"https?://[^\s\"']+"
    found = re.findall(pattern, text)
    return {"urls": found}


# Mount MCP SSE endpoint (for MCP clients)
app.mount("/mcp", mcp.sse_app)
