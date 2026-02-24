# mcp_server.py
"""
Production-ready MCP server (HTTP transport) for:
 - find_company_website(company_name, max_results=1, include_candidates=False)
 - scrape_site(url, ...) -> returns text-only pages by default

Key improvements:
 - HTTP MCP transport (streamable_http_app) for Copilot Studio compatibility
 - logging, robots.txt respect, per-host throttle, robust error handling
 - safe caching (no huge HTML stored), environment-configurable limits
"""
import os
import re
import time
import logging
from typing import Any, List, Dict, Optional
from urllib.parse import urljoin, urlparse

import httpx
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from mcp.server.fastmcp import FastMCP
from requests import RequestException
import urllib.robotparser as robotparser

# -------------------------
# Basic configuration
# -------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mcp_server")

DUCKDUCKGO_URL = os.getenv("DUCKDUCKGO_URL", "https://html.duckduckgo.com/html/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
USER_AGENT = os.getenv("USER_AGENT", "BusinessWebsiteFinder/1.0 (+https://example.com)")
SCRAPE_TIMEOUT = float(os.getenv("SCRAPE_TIMEOUT", "15"))
PAGE_TEXT_LIMIT = int(os.getenv("PAGE_TEXT_LIMIT", "5000"))  # words
MAX_PAGES_TO_SCRAPE = int(os.getenv("MAX_PAGES_TO_SCRAPE", "50"))
MIN_TEXT_BLOCK_CHARS = int(os.getenv("MIN_TEXT_BLOCK_CHARS", "20"))
DEFAULT_SPECIFIED_PAGES = ["/", "/about", "/about-us", "/services", "/industries", "/insights", "/contact", "/team", "/careers"]

# Throttle (seconds) minimum interval between requests to same host
HOST_THROTTLE_SECONDS = float(os.getenv("HOST_THROTTLE_SECONDS", "0.5"))

# In-memory caches (simple - suitable for single instance / dev)
_SIMPLE_CACHE: Dict[str, Dict] = {}
_CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # seconds
_LAST_REQUEST_AT: Dict[str, float] = {}  # per-host last request timestamp


def cache_get(key: str) -> Optional[Dict]:
    entry = _SIMPLE_CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > _CACHE_TTL:
        del _SIMPLE_CACHE[key]
        return None
    return entry["value"]


def cache_set(key: str, value: Dict):
    # avoid storing large HTML in cache -> store only essential payloads
    _SIMPLE_CACHE[key] = {"ts": time.time(), "value": value}


# -------------------------
# MCP server and helpers
# -------------------------
mcp = FastMCP("Company Research Tools (search+scrape)")

# -------------------------
# DuckDuckGo helper (sync)
# -------------------------
def search_duckduckgo(query: str, max_results: int = MAX_RESULTS) -> List[str]:
    """
    Return list of result URLs from DuckDuckGo HTML endpoint (synchronous).
    Keep it simple: perform POST and parse result anchors.
    """
    headers = {"User-Agent": USER_AGENT}
    params = {"q": query}
    try:
        logger.debug("DuckDuckGo query: %s", query)
        resp = requests.post(DUCKDUCKGO_URL, data=params, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except RequestException as exc:
        logger.exception("DuckDuckGo request failed")
        raise HTTPException(status_code=502, detail="Upstream search provider unavailable") from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[str] = []
    for link in soup.select("a.result__a"):
        href = link.get("href")
        if href and href.startswith("http"):
            results.append(href)
            if len(results) >= max_results:
                break
    logger.debug("DuckDuckGo returned %d results", len(results))
    return results


# -------------------------
# Utility helpers
# -------------------------
def _normalize_host(url_or_host: str) -> str:
    parsed = urlparse(url_or_host)
    host = (parsed.netloc or parsed.path or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _enforce_host_throttle(url: str):
    host = _normalize_host(url)
    last = _LAST_REQUEST_AT.get(host)
    if last:
        since = time.time() - last
        if since < HOST_THROTTLE_SECONDS:
            wait = HOST_THROTTLE_SECONDS - since
            logger.debug("Throttling: sleeping %.3fs for host %s", wait, host)
            time.sleep(wait)
    _LAST_REQUEST_AT[host] = time.time()


def _strip_url_noise(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return parsed._replace(path=path, query="", fragment="").geturl()


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


def _remove_image_urls(text: str) -> str:
    cleaned = re.sub(
        r"https?://\S+\.(?:png|jpg|jpeg|gif|webp|svg|avif|bmp|ico)(?:\?\S*)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"/\S+\.(?:png|jpg|jpeg|gif|webp|svg|avif|bmp|ico)(?:\?\S*)?", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


# -------------------------
# Extract page details
# -------------------------
def extract_page_details(page_url: str, html: str, text_limit: Optional[int] = PAGE_TEXT_LIMIT, include_html: bool = False) -> Dict[str, Any]:
    """Extract structured text and metadata from a page HTML (safe and robust)."""
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

    # Remove heavy/unwanted tags for text extraction
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


def to_text_only_pages(page_details: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Return plain text per page key (removing image urls)."""
    text_pages: Dict[str, str] = {}
    for key, details in page_details.items():
        text = details.get("text")
        if isinstance(text, str) and text:
            text_pages[key] = _remove_image_urls(text)
    return text_pages


# -------------------------
# Robots.txt check
# -------------------------
def is_allowed_by_robots(base_url: str, user_agent: str = USER_AGENT) -> bool:
    """
    Check robots.txt for a site. If robots.txt not reachable, assume allowed.
    This uses urllib.robotparser which will fetch robots.txt.
    """
    try:
        parsed = urlparse(base_url)
        scheme = parsed.scheme or "https"
        host = parsed.netloc or parsed.path
        robots_url = f"{scheme}://{host}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        # urllib robotparser uses blocking urlopen; give it a small timeout via requests
        # fallback: read manually using requests and parse lines
        try:
            r = requests.get(robots_url, timeout=min(REQUEST_TIMEOUT, 5), headers={"User-Agent": user_agent})
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            else:
                # if robots not 200, assume allowed
                return True
        except Exception:
            return True
        return rp.can_fetch(user_agent, base_url)
    except Exception:
        return True


# -------------------------
# Crawling & scraping (async)
# -------------------------
async def _fetch_page(client: httpx.AsyncClient, page_url: str) -> Optional[httpx.Response]:
    try:
        _enforce_host_throttle(page_url)
        resp = await client.get(page_url)
        return resp
    except Exception as exc:
        logger.warning("Fetch failed for %s: %s", page_url, exc)
        return None


async def scrape_important_pages_async(
    base_url: str,
    text_limit: Optional[int] = PAGE_TEXT_LIMIT,
    max_pages: Optional[int] = None,
    page_urls: Optional[List[str]] = None,
    timeout_seconds: Optional[float] = None,
    include_html: bool = False,
    crawl_all_internal: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Crawl and fetch pages. Returns mapping page_key -> details dict.
    - page_urls if provided should be path segments or full URLs (will be normalized).
    - crawl_all_internal will discover internal links and follow them (bounded by max_pages).
    """
    safe_text_limit = PAGE_TEXT_LIMIT if text_limit is None else max(1, min(int(text_limit), PAGE_TEXT_LIMIT))
    safe_max_pages: Optional[int] = None
    if max_pages is not None:
        safe_max_pages = max(1, min(int(max_pages), MAX_PAGES_TO_SCRAPE))

    safe_timeout = None
    if timeout_seconds is not None:
        safe_timeout = max(0.1, float(timeout_seconds))

    # Normalize base URL (ensure scheme)
    parsed = urlparse(base_url)
    if not parsed.scheme:
        base_url = "https://" + base_url
    base_url = _strip_url_noise(base_url)

    cache_key = f"scrape:{base_url}:{safe_text_limit}:{safe_max_pages}:{int(include_html)}:{int(crawl_all_internal)}:{','.join(page_urls or [])}"
    cached = cache_get(cache_key)
    if cached:
        logger.debug("Returning cached scrape for %s", base_url)
        return cached

    # Check robots
    try:
        if not is_allowed_by_robots(base_url):
            logger.info("Robots disallow crawling for %s", base_url)
            cache_set(cache_key, {})
            return {}
    except Exception:
        # If robots check fails, continue carefully
        logger.debug("Robots check failed (continuing): %s", base_url)

    # Seed URLs
    seed_urls: List[str] = []
    if page_urls:
        for raw in page_urls:
            if not raw or not raw.strip():
                continue
            full = _strip_url_noise(urljoin(base_url, raw.strip()))
            if _normalize_host(full) != _normalize_host(base_url):
                continue
            if full not in seed_urls:
                seed_urls.append(full)
    if not seed_urls:
        seed_urls = [_strip_url_noise(base_url)]

    results: Dict[str, Dict[str, Any]] = {}
    to_visit: List[str] = seed_urls[:]
    visited: List[str] = []

    async with httpx.AsyncClient(timeout=safe_timeout or SCRAPE_TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        while to_visit:
            if safe_max_pages is not None and len(visited) >= safe_max_pages:
                logger.debug("Reached max_pages limit (%s)", safe_max_pages)
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
                resp = await _fetch_page(client, page_url)
                if resp is None:
                    results[final_key] = {"url": page_url, "error": "fetch_failed"}
                    continue
                if resp.status_code >= 400:
                    results[final_key] = {
                        "url": page_url,
                        "status_code": resp.status_code,
                        "final_url": str(resp.url),
                        "error": f"HTTP {resp.status_code}",
                    }
                    continue

                details = extract_page_details(page_url=str(resp.url), html=resp.text, text_limit=safe_text_limit, include_html=include_html)
                details["status_code"] = resp.status_code
                details["final_url"] = str(resp.url)
                results[final_key] = details

                # discover internal links only if crawling requested
                if crawl_all_internal:
                    # extract internal links
                    soup_links = []
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for a in soup.find_all("a", href=True):
                        href = (a.get("href") or "").strip()
                        if not href:
                            continue
                        full = _strip_url_noise(urljoin(str(resp.url), href))
                        if urlparse(full).scheme not in ("http", "https"):
                            continue
                        if _normalize_host(full) != _normalize_host(base_url):
                            continue
                        if full not in visited and full not in to_visit and full not in soup_links:
                            soup_links.append(full)
                    # extend to_visit but keep overall ordering disciplined
                    to_visit.extend(soup_links)

            except Exception as exc:
                logger.exception("Exception while processing %s", page_url)
                results[final_key] = {"url": page_url, "error": f"{type(exc).__name__}: {exc}"}

    # store a trimmed cache (no raw html)
    cache_set(cache_key, results)
    return results


# -------------------------
# MCP tools
# -------------------------
@mcp.tool()
async def find_company_website(company_name: str, max_results: int = 1, include_candidates: bool = False) -> Dict[str, Any]:
    """
    Return the top official website for a company (string), with optional candidates.
    Output (default):
      {"business_name": "...", "official_website": "https://..."}
    If include_candidates=True, also returns "candidates" and "searched_results_count".
    """
    safe_max = max(1, min(int(max_results), MAX_RESULTS))
    key = f"search:{company_name.lower()}:{safe_max}:{int(include_candidates)}"
    cached = cache_get(key)
    if cached:
        return cached

    query = f"{company_name.strip()} official website"
    results = search_duckduckgo(query, max_results=safe_max)

    official_site = None
    # filter out common aggregator/social results in preference
    blacklist = ["linkedin.com", "facebook.com", "instagram.com", "twitter.com", "wikipedia.org", "glassdoor.com", "indeed.com", "crunchbase.com"]
    for r in results:
        if not any(b in r.lower() for b in blacklist):
            official_site = r
            break
    if not official_site and results:
        official_site = results[0]

    payload: Dict[str, Any] = {"business_name": company_name, "official_website": official_site or ""}
    if include_candidates:
        payload["candidates"] = results
        payload["searched_results_count"] = len(results)
    cache_set(key, payload)
    return payload


@mcp.tool()
async def scrape_site(
    url: str,
    text_limit: Optional[int] = PAGE_TEXT_LIMIT,
    max_pages: Optional[int] = None,
    page_urls: Optional[List[str]] = None,
    no_timeout: bool = True,
    timeout_seconds: Optional[float] = None,
    include_html: bool = False,
    crawl_all_internal: bool = False,
    text_only: bool = True,
) -> Dict[str, Any]:
    """
    Scrape important pages. By default returns text-only small payload to feed LLMs:
      {"website": url, "pages": {"homepage": "text...", "about": "...", ...}, "page_count": N}
    If text_only=False, returns full structured page dicts.
    """
    # determine effective pages to request
    effective_pages = page_urls or DEFAULT_SPECIFIED_PAGES
    effective_crawl = bool(crawl_all_internal)
    effective_timeout = None if no_timeout else timeout_seconds

    pages = await scrape_important_pages_async(
        url,
        text_limit=text_limit,
        max_pages=max_pages,
        page_urls=effective_pages,
        timeout_seconds=effective_timeout,
        include_html=include_html,
        crawl_all_internal=effective_crawl,
    )

    if text_only:
        text_pages = to_text_only_pages(pages)
        return {"website": url, "pages": text_pages, "page_count": len(text_pages)}
    return {"website": url, "pages": pages, "page_count": len(pages)}


# -------------------------
# REST wrapper (for Copilot Studio and manual testing)
# -------------------------
app = FastAPI(title="Company Research MCP Server (search + scrape)")

@app.get("/")
async def root():
    return {"status": "running", "mcp_endpoint": "/mcp"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/find-website")
async def find_website_api(
    business_name: str = Query(..., min_length=1),
    include_candidates: bool = Query(False),
    max_results: int = Query(1, ge=1, le=MAX_RESULTS),
):
    effective_max = max_results if include_candidates else 1
    return await find_company_website(business_name, max_results=effective_max, include_candidates=include_candidates)

@app.get("/scrape-site")
async def scrape_site_api(
    url: str = Query(..., min_length=4),
    text_limit: int = Query(PAGE_TEXT_LIMIT, ge=1, le=PAGE_TEXT_LIMIT),
    max_pages: Optional[int] = Query(None, ge=1, le=MAX_PAGES_TO_SCRAPE),
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
    effective_crawl = False if crawl_all_internal is None else bool(crawl_all_internal)
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

@app.post("/extract-urls")
async def extract_urls_api(payload: Dict[str, Any]):
    text = payload.get("text", "") or ""
    pattern = r"https?://[^\s\"']+"
    found = re.findall(pattern, text)
    return {"urls": found}

# -------------------------
# Mount MCP using HTTP transport (Copilot Studio friendly)
# -------------------------
# For mcp>=1.x FastMCP exposes streamable_http_app()/sse_app directly.
app.mount("/mcp", mcp.streamable_http_app())
