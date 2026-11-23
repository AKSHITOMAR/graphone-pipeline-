"""Semantic-density truncation so payloads never trigger 413 while keeping the useful text."""
from __future__ import annotations

import re

_KEYWORDS = re.compile(
    r"\b(pricing|price|plans?|free|freemium|enterprise|contact sales|per (user|seat|month)|/mo|trial|"
    r"employees?|team size|headcount|founded|funding|raised|series [a-e]|github|code|abstract)\b|[$€£]\s?\d", re.I)


def normalize(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _paragraphs(text: str, max_para: int = 1200) -> list[str]:
    out: list[str] = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_para:
            out.append(p)
        else:  # split overly long blocks at sentence boundaries
            buf = ""
            for s in re.split(r"(?<=[.!?])\s+", p):
                if len(buf) + len(s) > max_para and buf:
                    out.append(buf)
                    buf = ""
                buf += (" " if buf else "") + s
            if buf:
                out.append(buf)
    # hard-split anything still oversized (no sentence/paragraph breaks, e.g. minified text)
    final: list[str] = []
    for p in out:
        final.extend(p[i:i + max_para] for i in range(0, len(p), max_para))
    return final


def _density(p: str) -> float:
    digits = sum(c.isdigit() for c in p)
    caps = len(re.findall(r"\b[A-Z][a-z]+\b", p))
    kw = len(_KEYWORDS.findall(p))
    return (0.4 * digits + 0.15 * caps + 3.0 * kw) / (len(p) ** 0.5 + 1)


def smart_truncate(text: str, max_chars: int, head_frac: float = 0.3) -> str:
    """Return <= max_chars. Keeps the lede (head_frac of budget) + highest-density paragraphs, in original order."""
    text = normalize(text)
    if len(text) <= max_chars:
        return text
    paras = _paragraphs(text)
    keep: set[int] = set()
    used = 0
    for i, p in enumerate(paras):                      # 1) lede
        if used + len(p) > max_chars * head_frac:
            break
        keep.add(i)
        used += len(p) + 2
    ranked = sorted((i for i in range(len(paras)) if i not in keep), key=lambda i: -_density(paras[i]))
    for i in ranked:                                   # 2) densest remaining
        if used + len(paras[i]) + 2 <= max_chars:
            keep.add(i)
            used += len(paras[i]) + 2
    out = "\n\n".join(paras[i] for i in sorted(keep))
    return (out or text)[:max_chars]


def est_tokens(text: str) -> int:
    return len(text) // 4 + 1
