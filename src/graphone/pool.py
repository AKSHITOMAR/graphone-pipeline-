"""Bounded producer/consumer pool: constant memory whether there are 1k or 500k items."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterable, Awaitable, Callable, Iterable

log = logging.getLogger("graphone.pool")


async def run_pool(items: Iterable[Any] | AsyncIterable[Any], worker: Callable[[Any], Awaitable[Any]],
                   sink: Callable[[Any], None], concurrency: int = 32, stop_after: int | None = None) -> dict:
    q: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 2)
    stats = {"in": 0, "ok": 0, "skipped": 0, "errors": 0}
    stop = asyncio.Event()

    async def producer():
        if hasattr(items, "__aiter__"):
            async for it in items:  # type: ignore[union-attr]
                if stop.is_set():
                    break
                await q.put(it)
        else:
            for it in items:  # type: ignore[union-attr]
                if stop.is_set():
                    break
                await q.put(it)
        for _ in range(concurrency):
            await q.put(None)

    async def consumer():
        while (it := await q.get()) is not None:
            stats["in"] += 1
            try:
                r = await worker(it)
                if r is None:
                    stats["skipped"] += 1
                else:
                    sink(r)
                    stats["ok"] += 1
                    if stop_after and stats["ok"] >= stop_after:
                        stop.set()
            except Exception as e:                      # one bad item must never kill the run
                stats["errors"] += 1
                log.warning("item failed: %s: %s", type(e).__name__, e)
            if stats["in"] % 200 == 0:
                log.info("progress %s", stats)

    await asyncio.gather(producer(), *(consumer() for _ in range(concurrency)))
    return stats
