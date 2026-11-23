"""Startups + Products.

Startup directory: the open `yc-oss` dataset of Y Combinator companies (static JSON, no scraping risk,
~5k companies). Fields used are deterministic (name, team_size, website, YC URL) -> NO LLM, no hallucination.
Products: each startup's own website/pricing page is fetched; pricing is classified by rules, then LLM fallback,
and kept ONLY if the evidence quote exists verbatim on the page.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..dates import iso
from ..entity import EntityResolver
from ..extract import classify_pricing, html_to_text, is_grounded, llm_pricing, page_title
from ..http import FetchError, Fetcher
from ..llm import AllTiersFailed, LLMOrchestrator
from ..schemas import (Product, ProductContent, Source, Startup, StartupContent, StartupData)

log = logging.getLogger("graphone.startups")
YC_ALL = "https://yc-oss.github.io/api/companies/all.json"
YC_ALL_RAW = "https://raw.githubusercontent.com/yc-oss/api/main/companies/all.json"  # fallback mirror


async def load_startup_directory(fetcher: Fetcher, ai_only: bool = True) -> list[dict]:
    try:
        rows = (await fetcher.get(YC_ALL, robots=False, retries=1)).json()
    except FetchError:
        rows = (await fetcher.get(YC_ALL_RAW, robots=False)).json()
    if ai_only:
        keep = [r for r in rows if re.search(r"\b(ai|artificial intelligence|machine learning|ml|llm|generative)\b",
                                             " ".join([*(r.get("tags") or []), r.get("industry") or "", r.get("subindustry") or "",
                                                       r.get("one_liner") or ""]), re.I)]
        rows = keep if len(keep) >= 1000 else rows
    return rows


def startup_record(row: dict, resolver: EntityResolver) -> Startup | None:
    name, url = (row.get("name") or "").strip(), row.get("url") or row.get("website")
    if not name or not url:
        return None
    res = resolver.resolve(name, row.get("website"))
    size = row.get("team_size")
    return Startup(source=Source(name="Y Combinator (yc-oss dataset)", url=url),
                   content=StartupContent(entityName=res.canonical, rawName=name,
                                          data=StartupData(employeeCount=int(size) if isinstance(size, int) and size > 0 else None,
                                                           website=row.get("website"), description=row.get("one_liner"))),
                   collectedAt=iso(datetime.now(timezone.utc)))


async def product_record(row: dict, resolver: EntityResolver, fetcher: Fetcher, llm: LLMOrchestrator | None) -> Product | None:
    site = row.get("website")
    if not site or not site.startswith("http"):
        return None
    try:
        home = await fetcher.get(site, retries=2)
    except FetchError:
        return None
    soup = BeautifulSoup(home.text, "html.parser")
    pricing_url = next((urljoin(home.url, a["href"]) for a in soup.find_all("a", href=True)
                        if re.search(r"pricing|plans", a["href"], re.I) and urlparse(urljoin(home.url, a["href"])).netloc == urlparse(home.url).netloc), None)
    text, src_url = html_to_text(home.text), home.url
    if pricing_url:
        try:
            pp = await fetcher.get(pricing_url, retries=1)
            text, src_url = html_to_text(pp.text) + "\n\n" + text, pp.url
        except FetchError:
            pass
    verdict = classify_pricing(text)
    method = "rules"
    if verdict is None and llm is not None and len(text) > 200:
        try:
            got = await llm_pricing(llm, text)
        except AllTiersFailed:
            got = None
        verdict, method = (got[:2], "llm") if got else (None, "rules")
    if verdict is None or not is_grounded(verdict[1], text, min_len=3):
        return None
    canon = resolver.resolve(row["name"], site).canonical
    return Product(source=Source(name="Company website", url=src_url),
                   content=ProductContent(startupName=canon, productName=page_title(home.text) or canon, pricingModel=verdict[0],
                                          evidence=verdict[1][:200], extractionMethod=method),
                   collectedAt=iso(datetime.now(timezone.utc)))
