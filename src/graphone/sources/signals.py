"""24-hour fresh news (RSS + full-text crawl) and jobs (public ATS/board JSON APIs)."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import feedparser

from ..dates import extract_date_from_html, is_fresh, iso, parse_date
from ..extract import html_to_text
from ..freshness import SeenStore, canonical_url, undated_is_new
from ..http import FetchError, Fetcher
from ..schemas import Job, JobContent, News, NewsContent, Source

log = logging.getLogger("graphone.signals")

NEWS_FEEDS = {  # 5 AI news sources (RSS => stable, polite, no anti-bot friction)
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "MIT Technology Review AI": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
}

# 5 job sources. Public JSON APIs of ATS/boards; board tokens are editable config (404s are logged & skipped).
GREENHOUSE = ["anthropic", "databricks", "scaleai", "stabilityai", "runwayml"]
LEVER = ["mistral", "palantir"]
ASHBY = ["openai", "cohere", "perplexity-ai", "anysphere"]

ROLE_RULES = [("Engineering", r"engineer|developer|sre|devops|software|backend|frontend|full.?stack|infrastructure"),
              ("Research", r"research|scientist|applied ai|machine learning|\bml\b"),
              ("Data", r"data (engineer|analyst|science)|analytics"),
              ("Product", r"product manager|product design|\bpm\b"),
              ("Design", r"design"), ("Sales", r"sales|account exec|solutions|customer success|gtm"),
              ("Marketing", r"marketing|content|brand|communications"),
              ("Operations", r"operations|recruit|people|finance|legal|policy|security|admin")]


def role_family(title: str) -> str:
    t = title.lower()
    return next((fam for fam, rx in ROLE_RULES if re.search(rx, t)), "Other")


def _remote(*vals) -> bool:
    return any(re.search(r"\bremote\b", str(v or ""), re.I) for v in vals)


def mk_job(source: str, company: str, title: str, url: str, dt: datetime | None, how: str, remote: bool, now: datetime) -> Job | None:
    if not (title and url and is_fresh(dt, now)):
        return None
    return Job(source=Source(name=source, url=url),
               content=JobContent(company=company, title=title, url=url, date=iso(dt), date_source=how,
                                  is_remote=remote, role_family=role_family(title)), collectedAt=iso(now))


async def greenhouse(f: Fetcher, token: str, now) -> list[Job]:
    j = (await f.get(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs", params={"content": "false"}, robots=False)).json()
    out = []
    for x in j.get("jobs", []):
        dt = parse_date(x.get("first_published") or x.get("updated_at"), now)   # first_published = true publish date
        how = "api:first_published" if x.get("first_published") else "api:updated_at"
        job = mk_job(f"Greenhouse:{token}", token.title(), x.get("title"), x.get("absolute_url"), dt, how,
                     _remote((x.get("location") or {}).get("name"), x.get("title")), now)
        out.append(job) if job else None
    return out


async def lever(f: Fetcher, token: str, now) -> list[Job]:
    out = []
    for x in (await f.get(f"https://api.lever.co/v0/postings/{token}", params={"mode": "json"}, robots=False)).json():
        job = mk_job(f"Lever:{token}", token.title(), x.get("text"), x.get("hostedUrl"), parse_date(x.get("createdAt"), now),
                     "api:createdAt", (x.get("workplaceType") == "remote") or _remote((x.get("categories") or {}).get("location")), now)
        out.append(job) if job else None
    return out


async def ashby(f: Fetcher, token: str, now) -> list[Job]:
    j = (await f.get(f"https://api.ashbyhq.com/posting-api/job-board/{token}", robots=False)).json()
    out = []
    for x in j.get("jobs", []):
        job = mk_job(f"Ashby:{token}", token.title(), x.get("title"), x.get("jobUrl"), parse_date(x.get("publishedAt"), now),
                     "api:publishedAt", bool(x.get("isRemote")) or _remote(x.get("location")), now)
        out.append(job) if job else None
    return out


async def remotive(f: Fetcher, now) -> list[Job]:
    j = (await f.get("https://remotive.com/api/remote-jobs", params={"category": "software-dev", "limit": 300}, robots=False)).json()
    out = []
    for x in j.get("jobs", []):
        job = mk_job("Remotive", x.get("company_name", ""), x.get("title"), x.get("url"), parse_date(x.get("publication_date"), now),
                     "api:publication_date", True, now)
        out.append(job) if job else None
    return out


async def collect_jobs(f: Fetcher, now: datetime) -> list[Job]:
    import asyncio
    tasks = [greenhouse(f, t, now) for t in GREENHOUSE] + [lever(f, t, now) for t in LEVER] + \
            [ashby(f, t, now) for t in ASHBY] + [remotive(f, now)]
    jobs: list[Job] = []
    for r in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(r, Exception):
            log.warning("job source failed: %s", r)
        else:
            jobs.extend(r)
    return jobs


async def collect_news(f: Fetcher, store: SeenStore, now: datetime) -> list[News]:
    import asyncio
    out: list[News] = []

    async def one(name: str, feed_url: str):
        feed = feedparser.parse((await f.get(feed_url)).text)
        first_run = not store.has_baseline(f"news:{name}")
        for e in feed.entries[:60]:
            url = e.get("link")
            if not url:
                continue
            dt, how = parse_date(e.get("published") or e.get("updated"), now), "feed"
            html = None
            if dt is None or (is_fresh(dt, now) and True):          # fresh (or unknown) => crawl full text
                try:
                    html = (await f.get(url, retries=2)).text
                except FetchError as ex:
                    log.info("skip %s: %s", url, ex)
                    continue
            if dt is None:
                dt, how = extract_date_from_html(html, now)
            if dt is not None:
                if not is_fresh(dt, now) or not store.claim(f"news|{canonical_url(url)}"):
                    continue
            else:                                                     # no date anywhere => heuristic
                ok, how = undated_is_new(store, f"news:{name}", url, html or "")
                if not ok:
                    continue
                dt = now
            text = html_to_text(html)
            if len(text) < 200:
                continue
            out.append(News(source=Source(name=name, url=url), collectedAt=iso(now),
                            content=NewsContent(title=(e.get("title") or "").strip(), url=url, published=iso(dt), date_source=how, text=text[:20000])))
        store.mark_baseline(f"news:{name}")

    for r in await asyncio.gather(*(one(n, u) for n, u in NEWS_FEEDS.items()), return_exceptions=True):
        if isinstance(r, Exception):
            log.warning("news source failed: %s", r)
    return out
