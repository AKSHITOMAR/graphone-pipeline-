"""python -m graphone.cli --out out --startups 1100 --products 1100 --papers 1100"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .entity import EntityResolver
from .export import export_all, push_to_gsheet
from .freshness import SeenStore
from .http import Fetcher
from .llm import LLMOrchestrator
from .pool import run_pool
from .sources import papers as P
from .sources import signals as S
from .sources import startups as ST

log = logging.getLogger("graphone")


class Sink:
    """Append-only JSONL (crash-safe checkpoints; resume = skip already-written keys)."""
    def __init__(self, path: Path, key=lambda r: r["source"]["url"]):
        self.path, self.key = path, key
        self.seen = set()
        if path.exists():
            self.seen = {key(json.loads(l)) for l in path.read_text().splitlines() if l.strip()}
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, rec) -> None:
        d = rec.model_dump(mode="json")
        k = self.key(d)
        if k in self.seen:
            return
        self.seen.add(k)
        self.fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        self.fh.flush()

    def __len__(self):
        return len(self.seen)


async def main_async(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    resolver, store = EntityResolver(), SeenStore(str(out / "seen.sqlite"))
    now = datetime.now(timezone.utc)
    async with Fetcher(per_host=a.per_host, host_min_interval={"export.arxiv.org": 3.0}) as f:
        llm = LLMOrchestrator()
        rows = await ST.load_startup_directory(f)
        log.info("directory rows: %d", len(rows))
        s_sink = Sink(out / "startups.jsonl", key=lambda r: r["content"]["entityName"].lower())
        for r in rows:
            if len(s_sink) >= a.startups:
                break
            rec = ST.startup_record(r, resolver)
            rec and s_sink(rec)
        log.info("startups: %d", len(s_sink))

        p_sink = Sink(out / "products.jsonl", key=lambda r: r["content"]["startupName"].lower())

        async def prod(row):
            return await ST.product_record(row, resolver, f, llm)
        await run_pool((r for r in rows if r["name"]), prod, p_sink, a.concurrency, stop_after=a.products)
        log.info("products: %d", len(p_sink))

        pa_sink = Sink(out / "papers.jsonl")
        gh = P.GitHubStars(f)

        async def paper(e):
            return await P.paper_record(e, f, gh)
        await run_pool(P.arxiv_entries(f), paper, pa_sink, min(a.concurrency, 8), stop_after=a.papers)
        log.info("papers: %d", len(pa_sink))

        jobs = await S.collect_jobs(f, now)
        j_sink = Sink(out / "jobs.jsonl")
        for j in jobs:
            j_sink(j)
        news = await S.collect_news(f, store, now)
        n_sink = Sink(out / "news.jsonl")
        for n in news:
            n_sink(n)
        log.info("jobs: %d news: %d | llm stats: %s", len(j_sink), len(n_sink), llm.stats)

    with open(out / "entity_mapping.jsonl", "w") as fh:
        for r in resolver.log:
            fh.write(json.dumps({"raw_name": r.raw, "canonical_name": r.canonical, "method": r.method,
                                 "score": r.score, "in_seed_list": r.known}) + "\n")
    tabs = export_all(out)
    log.info("exported %s", {k: len(v) for k, v in tabs.items()})
    if a.gsheet:
        log.info("Google Sheet: %s", push_to_gsheet(tabs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--startups", type=int, default=1100)
    ap.add_argument("--products", type=int, default=1100)
    ap.add_argument("--papers", type=int, default=1100)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--per-host", type=int, default=4)
    ap.add_argument("--gsheet", action="store_true", help="push to Google Sheets (service account)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
