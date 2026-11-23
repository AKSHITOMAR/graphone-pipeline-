"""Research papers: arXiv Atom API (official, polite: 1 req / 3s) + GitHub repo/stars.

GitHub link discovery order: (1) arXiv abstract/comment/links, (2) Hugging Face papers API (successor of
Papers with Code, which now redirects there). Stars come from the GitHub REST API (authenticated: 5k req/h;
batch through GraphQL for 100 repos/request at scale)."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import AsyncIterator

import feedparser

from ..dates import iso
from ..http import FetchError, Fetcher
from ..schemas import PaperContent, ResearchPaper, Source

log = logging.getLogger("graphone.papers")
ARXIV = "http://export.arxiv.org/api/query"
GH_RE = re.compile(r"https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+)", re.I)
CATS = ["cs.LG", "cs.CL", "cs.CV", "cs.AI", "stat.ML"]


def find_github(*texts: str) -> tuple[str, str] | None:
    for t in texts:
        m = GH_RE.search(t or "")
        if m:
            owner, repo = m.group(1), re.sub(r"(\.git|[.,;)]+)$", "", m.group(2))
            if repo and owner.lower() not in {"orgs", "topics", "sponsors"}:
                return owner, repo
    return None


async def arxiv_entries(fetcher: Fetcher, page_size: int = 100, max_pages: int = 200) -> AsyncIterator[dict]:
    q = " OR ".join(f"cat:{c}" for c in CATS)
    for page in range(max_pages):
        feed = feedparser.parse((await fetcher.get(ARXIV, params={"search_query": q, "start": page * page_size,
                                                                  "max_results": page_size, "sortBy": "submittedDate",
                                                                  "sortOrder": "descending"}, robots=False)).text)
        if not feed.entries:
            return
        for e in feed.entries:
            yield e


async def hf_repo(fetcher: Fetcher, arxiv_id: str) -> tuple[str, str] | None:
    try:
        j = (await fetcher.get(f"https://huggingface.co/api/papers/{arxiv_id}", retries=1, robots=False)).json()
    except (FetchError, ValueError):
        return None
    return find_github(j.get("githubRepo") or "")


class GitHubStars:
    def __init__(self, fetcher: Fetcher):
        self.f, self.cache = fetcher, {}
        tok = os.environ.get("GITHUB_TOKEN")
        self.headers = {"Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {tok}"} if tok else {})}

    async def stars(self, owner: str, repo: str) -> int | None:
        key = f"{owner}/{repo}".lower()
        if key not in self.cache:
            try:
                j = (await self.f.get(f"https://api.github.com/repos/{owner}/{repo}", headers=self.headers, robots=False)).json()
                self.cache[key] = int(j["stargazers_count"])
            except (FetchError, KeyError, ValueError):
                self.cache[key] = None
        return self.cache[key]


async def paper_record(entry, fetcher: Fetcher, gh: GitHubStars, use_hf: bool = True) -> ResearchPaper | None:
    abs_url = entry.get("id", "")
    arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1])
    links = " ".join(l.get("href", "") for l in entry.get("links", []))
    repo = find_github(entry.get("arxiv_comment", ""), entry.get("summary", ""), links)
    if repo is None and use_hf:
        repo = await hf_repo(fetcher, arxiv_id)
    if repo is None:
        return None                                   # deliverable requires GitHub metrics
    stars = await gh.stars(*repo)
    if stars is None:
        return None
    now = iso(datetime.now(timezone.utc))
    return ResearchPaper(source=Source(name="arXiv", url=abs_url),
                         content=PaperContent(title=re.sub(r"\s+", " ", entry["title"]).strip(),
                                              authors=[a.get("name", "") for a in entry.get("authors", [])],
                                              paper_url=abs_url, github_url=f"https://github.com/{repo[0]}/{repo[1]}",
                                              github_stars=stars, github_stars_checked_at=now,
                                              published_date=iso(datetime(*entry.published_parsed[:6], tzinfo=timezone.utc))),
                         collectedAt=now)
