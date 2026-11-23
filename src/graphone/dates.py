"""Publication-date extraction + normalisation (all outputs are tz-aware UTC)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from bs4 import BeautifulSoup
from dateutil import parser as dtparser

UTC = timezone.utc
FRESH_WINDOW = timedelta(hours=24)
_UNITS = [("seconds", 1), ("second", 1), ("secs", 1), ("sec", 1), ("s", 1),
          ("minutes", 60), ("minute", 60), ("mins", 60), ("min", 60), ("m", 60),
          ("hours", 3600), ("hour", 3600), ("hrs", 3600), ("hr", 3600), ("h", 3600),
          ("days", 86400), ("day", 86400), ("d", 86400),
          ("weeks", 604800), ("week", 604800), ("wk", 604800), ("w", 604800),
          ("months", 2592000), ("month", 2592000), ("mo", 2592000),
          ("years", 31536000), ("year", 31536000), ("yr", 31536000), ("y", 31536000)]
_UNIT_MAP = dict(_UNITS)
_REL = re.compile(r"(?P<n>\d+|an?|one)\s*(?P<u>" + "|".join(u for u, _ in _UNITS) + r")\b\.?\s*ago", re.I)


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def parse_date(value, now: datetime | None = None) -> datetime | None:
    """Parse ISO/RFC822/epoch/'2 hours ago'/'yesterday'. Naive timestamps are assumed UTC."""
    now = _utc(now) if now else datetime.now(UTC)
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"\d{9,13}(\.\d+)?", value.strip())):
        x = float(value)
        x = x / 1000 if x > 1e11 else x
        try:
            return datetime.fromtimestamp(x, UTC)
        except (OverflowError, OSError, ValueError):
            return None
    s = str(value).strip()
    low = s.lower()
    if re.search(r"\b(just now|moments? ago|right now)\b", low):
        return now
    m = _REL.search(low)
    if m:
        n = 1 if m["n"] in ("a", "an", "one") else int(m["n"])
        return now - timedelta(seconds=n * _UNIT_MAP[m["u"].lower()])
    if re.fullmatch(r"(posted |published |updated )?today", low):
        return now
    if re.fullmatch(r"(posted |published |updated )?yesterday", low):
        return now - timedelta(days=1)
    try:
        return _utc(dtparser.parse(s, fuzzy=False))
    except (ValueError, OverflowError, TypeError):
        return None


def is_fresh(dt: datetime | None, now: datetime | None = None, window: timedelta = FRESH_WINDOW) -> bool:
    """True iff now-window <= dt <= now+5min (tiny skew allowed; far-future dates rejected)."""
    if dt is None:
        return False
    now = _utc(now) if now else datetime.now(UTC)
    return now - window <= dt <= now + timedelta(minutes=5)


def iso(dt: datetime | None) -> str | None:
    return _utc(dt).isoformat(timespec="seconds").replace("+00:00", "Z") if dt else None


_META_KEYS = ["article:published_time", "og:article:published_time", "datePublished", "date",
              "pubdate", "publishdate", "DC.date.issued", "dc.date", "parsely-pub-date", "sailthru.date"]


def extract_date_from_html(html: str, now: datetime | None = None) -> tuple[datetime | None, str]:
    """Return (datetime, how). Priority: JSON-LD > meta tags > itemprop > <time> > visible 'x ago'."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if node.get("datePublished"):
                    dt = parse_date(node["datePublished"], now)
                    if dt:
                        return dt, "json-ld"
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
            elif isinstance(node, list):
                stack.extend(node)
    for key in _META_KEYS:
        t = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key}) \
            or soup.find("meta", attrs={"itemprop": key})
        if t and t.get("content"):
            dt = parse_date(t["content"], now)
            if dt:
                return dt, f"meta:{key}"
    t = soup.find(attrs={"itemprop": "datePublished"})
    if t:
        dt = parse_date(t.get("content") or t.get("datetime") or t.get_text(strip=True), now)
        if dt:
            return dt, "itemprop"
    for t in soup.find_all("time"):
        dt = parse_date(t.get("datetime") or t.get_text(strip=True), now)
        if dt:
            return dt, "time-tag"
    head_text = soup.get_text(" ", strip=True)[:1500]
    m = _REL.search(head_text.lower())
    if m:
        return parse_date(m.group(0), now), "visible-relative"
    return None, "none"
