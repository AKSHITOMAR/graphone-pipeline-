"""Polite, resilient async HTTP client.

Design rules:
  * robots.txt is honoured (RFC 9309 semantics) and requests are paced per host.
  * 429/5xx/network errors -> exponential backoff with FULL jitter, Retry-After honoured.
  * Bot-challenge pages (Cloudflare/Datadome/captcha) are DETECTED and surfaced as `Blocked`.
    We do not try to defeat them: the caller falls back to an API/feed/licensed source.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import aiohttp

log = logging.getLogger("graphone.http")
USER_AGENT = "GraphOneBot/1.0 (research crawler; contact: data-team@example.com)"
RETRY_STATUS = {429, 500, 502, 503, 504}
BLOCK_MARKERS = ("just a moment...", "cf-chl", "attention required! | cloudflare",
                 "captcha-delivery.com", "datadome", "px-captcha", "are you a robot")


class FetchError(Exception):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


class Blocked(FetchError):
    """Anti-bot challenge detected. Do not hammer: escalate to an alternative source."""


class RobotsDisallowed(FetchError):
    pass


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Full-jitter exponential backoff: U(0, min(cap, base*2^attempt))."""
    return random.uniform(0, min(cap, base * (2 ** attempt)))


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            return None


def looks_blocked(status: int, headers: Any, body: str) -> bool:
    if str(headers.get("cf-mitigated", "")).lower() == "challenge":
        return True
    if status in (200, 202, 401, 403, 429, 503):
        head = body[:6000].lower()
        if any(m in head for m in BLOCK_MARKERS) and len(body) < 60000:
            return True
    return False


@dataclass
class Page:
    url: str
    status: int
    headers: Any
    text: str

    def json(self) -> Any:
        import json
        return json.loads(self.text)


class Fetcher:
    def __init__(self, *, per_host: int = 4, global_limit: int = 100, timeout: float = 30,
                 user_agent: str = USER_AGENT, respect_robots: bool = True,
                 host_min_interval: dict[str, float] | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.per_host, self.global_limit, self.timeout = per_host, global_limit, timeout
        self.ua, self.respect_robots, self.sleep = user_agent, respect_robots, sleep
        self.min_interval = host_min_interval or {}
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._last: dict[str, float] = defaultdict(float)
        self._pace_locks: dict[str, asyncio.Lock] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "Fetcher":
        conn = aiohttp.TCPConnector(limit=self.global_limit, limit_per_host=self.per_host, ttl_dns_cache=300)
        self.session = aiohttp.ClientSession(connector=conn, headers={"User-Agent": self.ua},
                                             timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, *exc) -> None:
        if self.session:
            await self.session.close()

    async def _allowed(self, url: str) -> bool:
        p = urlparse(url)
        root = f"{p.scheme}://{p.netloc}"
        if root not in self._robots:
            rp: RobotFileParser | None = RobotFileParser()
            try:
                async with self.session.get(root + "/robots.txt") as r:  # type: ignore[union-attr]
                    if r.status >= 500:
                        rp.parse(["User-agent: *", "Disallow: /"])   # RFC 9309: server error => disallow
                    elif r.status >= 400:
                        rp = None                                     # no robots => allowed
                    else:
                        rp.parse((await r.text(errors="replace")).splitlines())
            except Exception:
                rp = None
            self._robots[root] = rp
        rp = self._robots[root]
        return True if rp is None else rp.can_fetch(self.ua, url)

    async def _pace(self, host: str) -> None:
        gap = self.min_interval.get(host, 0.0)
        if gap <= 0:
            return
        lock = self._pace_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._last[host] + gap - time.monotonic()
            if wait > 0:
                await self.sleep(wait)
            self._last[host] = time.monotonic()

    async def get(self, url: str, *, params: dict | None = None, headers: dict | None = None,
                  retries: int = 4, robots: bool | None = None) -> Page:
        assert self.session is not None, "use `async with Fetcher()`"
        if (self.respect_robots if robots is None else robots) and not await self._allowed(url):
            raise RobotsDisallowed(f"robots.txt disallows {url}")
        host = urlparse(url).netloc
        sem = self._sems.setdefault(host, asyncio.Semaphore(self.per_host))
        last: str = "unknown"
        for attempt in range(retries + 1):
            status, hdrs, body = None, None, ""
            async with sem:
                await self._pace(host)
                try:
                    async with self.session.get(url, params=params, headers=headers) as r:
                        status, hdrs, body = r.status, r.headers, await r.text(errors="replace")
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last = f"{type(e).__name__}: {e}"
            if status is None:
                await self.sleep(backoff_delay(attempt))
                continue
            if looks_blocked(status, hdrs, body):
                raise Blocked(f"bot challenge at {host} ({status})", status)
            if status in RETRY_STATUS:
                ra = parse_retry_after(hdrs.get("Retry-After")) or 0
                wait = min(max(ra, backoff_delay(attempt)), 120)
                log.warning("%s -> %s, retry %d in %.1fs", url, status, attempt + 1, wait)
                last = f"HTTP {status}"
                await self.sleep(wait)
                continue
            if status >= 400:
                raise FetchError(f"HTTP {status} for {url}", status)
            return Page(str(url), status, hdrs, body)
        raise FetchError(f"retries exhausted for {url}: {last}")
