"""Exactly-once tracking + the undated-content heuristic.

`SeenStore` is a tiny interface. SQLite (WAL) is used here; in production swap in Redis
(`SET key 1 NX EX <ttl>`) or Postgres (`INSERT .. ON CONFLICT DO NOTHING`) so that every
crawler node shares ONE atomic claim primitive.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

_TRACKING = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|source$)")


def canonical_url(url: str) -> str:
    p = urlparse(url.strip())
    q = sorted((k, v) for k, v in parse_qsl(p.query) if not _TRACKING.match(k))
    path = re.sub(r"/+$", "", p.path) or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower().removeprefix("www."), path, "", urlencode(q), ""))


def content_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()[:32]


class SeenStore:
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS seen(key TEXT PRIMARY KEY, first_seen REAL)")
            self._db.execute("CREATE TABLE IF NOT EXISTS baseline(source TEXT PRIMARY KEY, ts REAL)")
            self._db.commit()

    def claim(self, key: str) -> bool:
        """Atomically claim `key`. True => first time anywhere (caller should process it)."""
        with self._lock:
            cur = self._db.execute("INSERT OR IGNORE INTO seen VALUES(?,?)", (key, time.time()))
            self._db.commit()
            return cur.rowcount == 1

    def has_baseline(self, source: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM baseline WHERE source=?", (source,)).fetchone() is not None

    def mark_baseline(self, source: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO baseline VALUES(?,?)", (source, time.time()))
            self._db.commit()


def undated_is_new(store: SeenStore, source: str, url: str, text: str, *,
                   http_last_modified_fresh: bool = False) -> tuple[bool, str]:
    """Heuristic for sources that expose NO usable date.

    New iff (a) neither canonical URL nor content hash was ever seen (atomic claim), AND
    (b) the source already has a baseline run (so we're diffing a listing we know), OR
        a server signal (sitemap <lastmod>/HTTP Last-Modified) puts it inside the window.
    On a source's first run without any server signal we refuse to guess: nothing is 'fresh',
    everything is recorded as baseline. Returned records carry date_source='first_seen_heuristic'.
    """
    k1, k2 = f"{source}|u|{canonical_url(url)}", f"{source}|h|{content_hash(text)}"
    first_url, first_hash = store.claim(k1), store.claim(k2)
    is_new = first_url and first_hash
    if not is_new:
        return False, "already_seen"
    if store.has_baseline(source) or http_last_modified_fresh:
        return True, "first_seen_heuristic"
    return False, "baseline_run"
