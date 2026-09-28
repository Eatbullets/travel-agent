"""Firecrawl markdown scrape and menu-link discovery.

Used only for restaurant sites (homepage, then up to two extra pages if needed) under the free-tier cap.
"""

from __future__ import annotations

import os
import re
from urllib.parse import urljoin, urlparse

from firecrawl import Firecrawl

PRICE_RE = re.compile(
    r"(€\s?\d|[0-9]+[.,][0-9]{2}\s?€|\bEUR\b|\bSEK\b|\bNOK\b|\d+\s?kr\b|\bkr\s?\d|:-)"
)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+|/?[^)]+)\)", re.I)
HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.I)
PDF_URL_RE = re.compile(r"https?://[^\s\]\)\"'<>]+\.pdf(?:\?[^\s\]\)\"'<>]*)?", re.I)


def _client() -> Firecrawl:
    key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("FIRECRAWL_API_KEY is missing.")
    return Firecrawl(api_key=key)


def scrape_markdown(url: str) -> str:
    firecrawl = _client()
    scrape = getattr(firecrawl, "scrape", None) or getattr(firecrawl, "scrape_url")
    kwargs = {"formats": ["markdown"]}
    try:
        doc = scrape(url, wait_for=4000, **kwargs)
    except TypeError:
        doc = scrape(url, **kwargs)
    except Exception:
        doc = scrape(url, **kwargs)
    if isinstance(doc, dict):
        return doc.get("markdown") or doc.get("data", {}).get("markdown") or ""
    markdown = getattr(doc, "markdown", None)
    if markdown:
        return markdown
    data = getattr(doc, "data", None)
    if isinstance(data, dict):
        return data.get("markdown") or ""
    return ""


def page_key(url: str) -> str:
    parsed = urlparse(url)
    path = (parsed.path or "/").rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc.lower()}{path}"


def looks_like_menu(markdown: str) -> bool:
    if not markdown:
        return False
    return len(PRICE_RE.findall(markdown)) >= 4


def is_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    blob = f"{parsed.path} {parsed.query}".lower()
    return ".pdf" in blob


def collect_page_links(homepage_url: str, markdown: str) -> list[tuple[str, str]]:
    if not markdown:
        return []
    base = homepage_url.rstrip("/") + "/"
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    raw: list[tuple[str, str]] = []
    for label, href in MD_LINK_RE.findall(markdown):
        raw.append((label.strip(), href.strip()))
    for href in HREF_RE.findall(markdown):
        raw.append(("", href.strip()))
    for href in PDF_URL_RE.findall(markdown):
        raw.append(("pdf", href.strip()))
    for label, href in raw:
        if not href or href.startswith("mailto:") or href.startswith("#"):
            continue
        absolute = urljoin(base, href.split()[0].rstrip(").,]"))
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        if not is_pdf_url(absolute) and page_key(absolute) == page_key(homepage_url):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        out.append((label, absolute))
    return out


def first_pdf_url(links: list[tuple[str, str]]) -> str | None:
    for _label, url in links:
        if is_pdf_url(url):
            return url
    return None
