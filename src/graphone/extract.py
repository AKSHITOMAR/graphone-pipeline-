"""Anti-hallucination layer. Nothing the LLM says is stored unless it is quoted verbatim from the source text."""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from .llm import LLMOrchestrator
from .schemas import PricingModel


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).casefold().strip()


def is_grounded(evidence: str | None, source_text: str, min_len: int = 6) -> bool:
    """Evidence must be a verbatim (whitespace/case-insensitive) substring of the page text."""
    return bool(evidence) and len(evidence.strip()) >= min_len and _norm(evidence) in _norm(source_text)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        t.decompose()
    for sel in ("nav", "footer", "header"):
        for t in soup.find_all(sel):
            t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks = [b.get_text(" ", strip=True) for b in root.find_all(["h1", "h2", "h3", "p", "li", "td", "div"])
              if not b.find(["p", "li", "div"])]
    return "\n\n".join(b for b in blocks if len(b) > 2)


def page_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for key in ("og:site_name", "og:title"):
        t = soup.find("meta", attrs={"property": key})
        if t and t.get("content"):
            return t["content"].strip()
    return soup.title.string.strip() if soup.title and soup.title.string else None


_PRICE = re.compile(r"[$€£]\s?\d[\d,.]*\s*(/|per|a)?\s*(mo|month|user|seat|year|yr|1k|million)?", re.I)
_FREE_TIER = re.compile(r"(free (plan|tier|forever|version)|start (for )?free|get started for free|try (it )?free|freemium|forever free)", re.I)
_ALL_FREE = re.compile(r"(100% free|completely free|totally free|free and open[- ]source|free for everyone|always free)", re.I)
_ENT = re.compile(r"(contact (our )?sales|talk to (our )?sales|request a demo|book a demo|custom pricing|enterprise plan)", re.I)


def classify_pricing(text: str) -> tuple[PricingModel, str] | None:
    """Rule-based tier-0 (cheap, deterministic). Returns (model, verbatim evidence) or None if undecidable."""
    price, free_tier, all_free, ent = (_PRICE.search(text), _FREE_TIER.search(text),
                                       _ALL_FREE.search(text), _ENT.search(text))
    if free_tier and price:
        return PricingModel.FREEMIUM, free_tier.group(0)
    if all_free and not price:
        return PricingModel.FREE, all_free.group(0)
    if price:
        return PricingModel.PAID, price.group(0).strip()
    if ent:
        return PricingModel.ENTERPRISE, ent.group(0)
    return None


PRICING_INSTRUCTION = (
    'Task: classify the pricing model of the product described in TEXT. '
    'Return JSON {"pricingModel": "FREE"|"FREEMIUM"|"PAID"|"ENTERPRISE"|null, "evidence": "<verbatim quote from TEXT, max 200 chars>"}. '
    "FREE=no paid plan; FREEMIUM=free tier plus paid plans; PAID=paid only/trial; ENTERPRISE=sales-led, no public prices. "
    "If TEXT does not state pricing, return pricingModel=null.")


async def llm_pricing(llm: LLMOrchestrator, text: str) -> tuple[PricingModel, str, str] | None:
    res = await llm.extract(PRICING_INSTRUCTION, text)
    pm, ev = res.data.get("pricingModel"), res.data.get("evidence")
    if pm not in PricingModel.__members__ or not is_grounded(ev, text):
        return None                                 # ungrounded => discard (never store a guess)
    return PricingModel[pm], ev, res.tier
