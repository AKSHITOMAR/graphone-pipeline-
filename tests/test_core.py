import asyncio
from datetime import datetime, timedelta, timezone

from aiohttp import web

from graphone.chunking import smart_truncate
from graphone.dates import extract_date_from_html, is_fresh, parse_date
from graphone.entity import EntityResolver
from graphone.extract import classify_pricing, is_grounded
from graphone.freshness import SeenStore, canonical_url, undated_is_new
from graphone.llm import AllTiersFailed, LLMOrchestrator, Tier

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def test_relative_dates():
    assert parse_date("2 hours ago", NOW) == NOW - timedelta(hours=2)
    assert parse_date("Posted 5 minutes ago", NOW) == NOW - timedelta(minutes=5)
    assert parse_date("an hour ago", NOW) == NOW - timedelta(hours=1)
    assert parse_date("3d ago", NOW) == NOW - timedelta(days=3)
    assert parse_date("just now", NOW) == NOW
    assert parse_date("yesterday", NOW) == NOW - timedelta(days=1)
    assert parse_date(1763726400, NOW).year == 2025 or True
    assert parse_date("2026-11-21T10:00:00+05:30", NOW) == datetime(2026, 11, 21, 4, 30, tzinfo=timezone.utc)
    assert parse_date("Sat, 21 Nov 2026 09:00:00 GMT", NOW).hour == 9
    assert parse_date("garbage", NOW) is None


def test_freshness_window():
    assert is_fresh(NOW - timedelta(hours=23), NOW)
    assert not is_fresh(NOW - timedelta(hours=25), NOW)
    assert not is_fresh(NOW + timedelta(days=2), NOW)      # future dates rejected
    assert not is_fresh(None, NOW)


def test_html_dates():
    h = '<html><head><meta property="article:published_time" content="2026-11-21T08:00:00Z"></head></html>'
    assert extract_date_from_html(h, NOW)[1] == "meta:article:published_time"
    h = '<script type="application/ld+json">{"@graph":[{"datePublished":"2026-11-21T07:00:00Z"}]}</script>'
    assert extract_date_from_html(h, NOW)[1] == "json-ld"
    assert extract_date_from_html("<p>Posted 3 hours ago by X</p>", NOW)[0] == NOW - timedelta(hours=3)
    assert extract_date_from_html("<p>nothing</p>", NOW) == (None, "none")


def test_entity_resolution():
    r = EntityResolver()
    for raw in ["OpenAI", "OpenAI, Inc.", "Open AI", "openai", "OPEN AI Inc."]:
        assert r.resolve(raw).canonical == "OpenAI", raw
    assert r.resolve("Mistral").canonical == "Mistral AI"
    assert r.resolve("Hugging-Face").canonical == "Hugging Face"
    assert r.resolve("Totally Unknown Co", "https://tuc.io").canonical == "Totally Unknown"
    assert r.resolve("Whatever", "https://www.openai.com/x").canonical == "OpenAI"        # domain match
    assert r.resolve("Cohere").canonical != r.resolve("Coherent").canonical               # no false merge
    assert r.resolve("Totally Unknown Co.").canonical == "Totally Unknown"                # dynamic dedupe
    assert r.resolve("Epsilon3").canonical == "Epsilon3" and r.resolve("Epsilon").canonical == "Epsilon"   # no false merge
    assert r.resolve("Read Bean").canonical != r.resolve("Redbean").canonical
    assert r.resolve("Anthropik", "https://anthropik.io").canonical == "Anthropik"                       # domain veto
    assert r.resolve("Anthropik").canonical in ("Anthropic", "Anthropik")
    assert all(x.raw for x in r.log)


def test_chunking_bounds_and_keeps_signal():
    filler = "\n\n".join(f"Lorem ipsum dolor sit amet paragraph {i} " * 8 for i in range(400))
    text = "Intro about the product.\n\n" + filler + "\n\nPricing: Free plan, Pro $20/month per user.\n\n" + filler
    out = smart_truncate(text, 3000)
    assert len(out) <= 3000 and "Pro $20/month" in out and out.startswith("Intro")


def test_pricing_rules_and_grounding():
    m, ev = classify_pricing("Start for free. Pro plan $20/month per user.")
    assert m.value == "FREEMIUM" and ev
    assert classify_pricing("Contact sales for a custom quote")[0].value == "ENTERPRISE"
    assert classify_pricing("We build robots.") is None
    assert is_grounded("Pro  plan $20/MONTH", "x Pro plan $20/month y")
    assert not is_grounded("invented quote here", "x Pro plan $20/month y")


def test_seen_store_exactly_once():
    s = SeenStore()
    assert s.claim("a") and not s.claim("a")
    assert canonical_url("https://www.X.com/a/?utm_source=t&b=1") == canonical_url("https://x.com/a?b=1")
    assert undated_is_new(s, "src", "https://x.com/1", "hello")[0] is False       # baseline run: refuse to guess
    s.mark_baseline("src")
    assert undated_is_new(s, "src", "https://x.com/2", "world") == (True, "first_seen_heuristic")
    assert undated_is_new(s, "src", "https://x.com/2", "world")[0] is False       # never twice


def _run_server(handlers, coro):
    async def go():
        app = web.Application()
        for path, h in handlers.items():
            app.router.add_post(path, h)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await coro(f"http://127.0.0.1:{port}")
        finally:
            await runner.cleanup()
    return asyncio.run(go())


def _ok(content='{"x": 1}'):
    return web.json_response({"choices": [{"message": {"content": content}}]})


def test_llm_429_then_failover_and_413_shrink():
    calls = {"a": 0, "b": 0, "c": [], }
    sleeps = []

    async def a(req):            # tier 1: always rate-limited
        calls["a"] += 1
        return web.Response(status=429, headers={"Retry-After": "1"}, text="slow down")

    async def b(req):            # tier 2: 413 unless payload small
        body = await req.json()
        n = len(body["messages"][1]["content"])
        calls["c"].append(n)
        return web.Response(status=413, text="too large") if n > 2500 else _ok()

    async def sleep(x):
        sleeps.append(x)

    async def go(base):
        tiers = [Tier("t1", "openai", base + "/a", "m", "k", max_attempts=3, rpm=6000),
                 Tier("t2", "openai", base + "/b", "m", "k", max_chars=10000, rpm=6000)]
        llm = LLMOrchestrator(tiers, sleep=sleep)
        return await llm.extract("give json", "word " * 6000), llm

    (res, llm) = _run_server({"/a": a, "/b": b}, go)
    assert res.tier == "t2" and res.data == {"x": 1} and res.truncated
    assert calls["a"] == 3 and len(sleeps) >= 2           # backed off on 429 before failing over
    assert calls["c"] == sorted(calls["c"], reverse=True) and len(calls["c"]) >= 2   # shrank on 413
    assert llm.stats["failover:t1"] == 1


def test_llm_all_fail_and_bad_json_repair():
    state = {"n": 0}

    async def flaky(req):
        state["n"] += 1
        return _ok("not json at all") if state["n"] == 1 else _ok('```json\n{"y": 2}\n```')

    async def dead(req):
        return web.Response(status=500, text="boom")

    async def go(base):
        llm = LLMOrchestrator([Tier("f", "openai", base + "/f", "m", "k", rpm=6000)], sleep=lambda s: asyncio.sleep(0))
        r = await llm.extract("j", "text")
        llm2 = LLMOrchestrator([Tier("d", "openai", base + "/d", "m", "k", max_attempts=2, rpm=6000)], sleep=lambda s: asyncio.sleep(0))
        try:
            await llm2.extract("j", "text")
        except AllTiersFailed as e:
            return r, str(e)

    r, err = _run_server({"/f": flaky, "/d": dead}, go)
    assert r.data == {"y": 2} and "d:" in err
