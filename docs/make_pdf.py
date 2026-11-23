from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["Normal"], fontSize=8.8, leading=11.4, spaceAfter=3)
H = ParagraphStyle("h", parent=ss["Heading2"], fontSize=11.5, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1f3a5f"))
T = ParagraphStyle("t", parent=ss["Title"], fontSize=16, spaceAfter=2)
BL = ParagraphStyle("bl", parent=B, leftIndent=10, bulletIndent=0)
C = ParagraphStyle("c", parent=B, fontSize=7.8, leading=9.6)

def bl(items): return [Paragraph(i, BL, bulletText="•") for i in items]

s = [Paragraph("GraphOne Ingestion Pipeline: Architecture & Production Design", T),
     Paragraph("Design principle: <b>deterministic first, LLM last, nothing stored without a verifiable source URL.</b> "
               "Structured APIs/feeds/dumps are always preferred to scraping; LLMs only structure text that is "
               "genuinely unstructured, and their output is accepted only if its evidence quote appears verbatim in the fetched page.", B),
     Paragraph("Pipeline", H)]
flow = [["Scheduler / seed frontier", "Fetch layer (async, per-host limits, robots, backoff)", "Raw store (S3: HTML/JSON, zstd)"],
        ["Parsers (deterministic)", "LLM tier chain (only if parser fails)", "Grounding check"],
        ["Entity resolver (seed + domain + fuzzy)", "Dedup / exactly-once claim", "Postgres + pgvector + graph projection"]]
t = Table(flow, colWidths=[165, 190, 165])
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3f9")),
                       ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
s += [t, Spacer(1, 4)]

s += [Paragraph("1. Scale strategy: 500,000+ startups, products and papers without manual intervention", H)]
s += bl([
 "<b>Source hierarchy.</b> (a) bulk dumps/APIs (arXiv OAI-PMH + API, GitHub GraphQL, YC/Crunchbase-style licensed exports, sitemap indexes), "
 "(b) per-entity enrichment from the company's own site, (c) browser rendering only for JS-only pages. Papers scale via arXiv OAI-PMH harvesting "
 "(resumption tokens) rather than search pagination; each paper's repo link is regex-mined, then confirmed through the HF papers API.",
 "<b>Frontier, not loops.</b> URLs/entities are enqueued in a durable queue (SQS/Redis Streams/Kafka) keyed by (source, entity-id). Stateless worker pods "
 "pull batches; horizontal scale = more pods (KEDA autoscaling on queue depth). Code never changes between 1k and 500k: only worker count and rate budgets. "
 "In-process, <font face='Courier'>run_pool</font> uses a bounded queue so memory is constant.",
 "<b>Discovery loop.</b> Enrichment emits new entities (competitors, founders, GitHub orgs, job-board company slugs) back into the frontier; the "
 "resolver dedupes before enqueue, so growth is bounded by real-world entities, not link count.",
 "<b>Throughput maths.</b> 500k pages at 150 req/s aggregate (spread across thousands of hosts, 2-4 per host) is about 1 hour of fetch; "
 "the binding constraints are per-host politeness, GitHub quota (GraphQL batches 100 repos/call = 5k calls for 500k repos) and LLM quota, hence rules-first parsing "
 "(LLM touches roughly 10-20% of pages).",
 "<b>Self-healing.</b> Idempotent tasks, at-least-once delivery, JSONL/Parquet checkpoints, dead-letter queue with reason codes, "
 "per-source health metrics (success rate, parse-rate, freshness lag) with automatic source quarantine when parse-rate drops (template change).",
 "<b>Anti-bot / protected domains.</b> We detect challenges (cf-mitigated header, Cloudflare/Datadome/captcha markers) and raise <font face='Courier'>Blocked</font>; we do <b>not</b> "
 "solve captchas or spoof fingerprints. Order of escalation: official API/RSS/sitemap -> licensed data partner or bulk export -> low-rate, robots-compliant "
 "fetch with an honest identified User-Agent, conditional requests (ETag / If-Modified-Since), caching and jittered pacing so challenges are never triggered -> real-browser rendering "
 "(Playwright async, one context per host, cookies persisted, human-scale rates) only for legitimately JS-rendered pages -> written permission/allow-listing request to the site owner. "
 "Blocked sources go to a cool-down queue and are reported rather than hammered; this protects IP reputation for the whole fleet."])

s += [Paragraph("2. Handling 413s and 429s across thousands of concurrent extractions", H)]
s += bl([
 "<b>413 / context overflow (prevention + cure).</b> Every prompt passes through <font face='Courier'>smart_truncate</font>: keep the lede (30% of budget) plus the highest "
 "<i>information-density</i> paragraphs (digits, capitalised entities, pricing/team/funding keywords, normalised by length), restored to original order, hard-capped per tier "
 "(Gemini 60k chars, Groq 20k, DeepSeek 40k). If a provider still answers 413 (or 400 with a context-length message) the budget shrinks x0.6 and the SAME tier retries (max 3 shrinks), "
 "then fails over. Oversized single blocks are sentence- then hard-split so truncation can never return empty text.",
 "<b>429.</b> Layered defence: (1) a per-tier <i>token bucket</i> sized to the provider's RPM so most 429s never happen; "
 "(2) on 429, sleep <i>max(Retry-After, full-jitter backoff U(0, min(cap, 2<super>n</super>)))</i> - full jitter prevents thundering-herd retries from thousands of workers; "
 "(3) if Retry-After exceeds a latency budget (30 s) or attempts are exhausted, <b>fail over</b> to the next tier immediately rather than stall; "
 "(4) a per-tier circuit breaker opens after 5 consecutive failed calls for 60 s so a degraded tier costs zero latency.",
 "<b>Fleet-wide coordination.</b> With many pods the token bucket is held in Redis (Lua script, one key per tier/API key) so aggregate RPM is respected globally; "
 "keys are pooled per tier. Requests are queued by priority (fresh news/jobs > bulk backfill) and bulk work is shed first under pressure. "
 "Results are cached by (content-hash, prompt-version) so retries and re-crawls never pay twice.",
 "<b>Correctness guards.</b> temperature 0 + JSON mode; invalid JSON gets one repair retry then failover; page text is wrapped as untrusted data (prompt-injection hardening); "
 "enum values validated by pydantic; evidence quote must be a verbatim substring of the source else the field is dropped."])

s += [Paragraph("3. Freshness tracking: never process the same article/job twice across distributed nodes", H)]
s += bl([
 "<b>Canonical identity.</b> key = sha(source | canonical_url) where canonical_url lower-cases the host, drops www, fragments, trailing slashes and tracking params (utm_*, fbclid, gclid...); "
 "a second key = sha(normalised content). Both are claimed so syndicated copies and URL variants collapse.",
 "<b>Atomic claim primitive.</b> One shared store, one operation: Redis <font face='Courier'>SET key NX EX 7d</font> (or Postgres <font face='Courier'>INSERT ... ON CONFLICT DO NOTHING</font>; SQLite is used in this repo). "
 "Exactly one node wins the claim; others skip. Claim happens <i>before</i> expensive work; on failure the key is released with a lease (visibility timeout) so crashes don't lose items.",
 "<b>Date precedence.</b> API field (publishedAt/createdAt/first_published) > JSON-LD datePublished > meta tags (article:published_time, etc.) > itemprop/&lt;time&gt; > visible relative text ('2 hours ago', "
 "resolved against fetch time, UTC). Future dates and dates outside [now-24h, now+5min] are rejected. Each record stores <font face='Courier'>date_source</font> for auditability.",
 "<b>Undated sources.</b> A first-seen heuristic: new iff the URL and content hash were never claimed AND the source has a completed baseline run (we are diffing a listing we know) or a server signal "
 "(sitemap lastmod / HTTP Last-Modified) is inside the window. On a source's first ever run we record a baseline instead of guessing, so we never mislabel old content as fresh. Output is flagged <font face='Courier'>first_seen_heuristic</font>."])

s += [Paragraph("4. Storage strategy", H)]
s += bl([
 "<b>Primary: PostgreSQL</b> (partitioned by recordType/month). Reasons: ACID upserts for idempotent ingestion, JSONB for versioned canonical schemas, unique indexes for dedupe, "
 "mature ops/replication, and one engine for relational joins, full-text and (with <b>pgvector</b>) embeddings, so entity-resolution candidates (name trigram + embedding ANN) run in-database. "
 "Raw HTML/JSON lives in S3/object storage (cheap, replayable: re-parse without re-crawling); the exactly-once store lives in Redis.",
 "<b>Relationships.</b> Startups, products, people, papers, repos and jobs are nodes; FOUNDED_BY, BUILDS, AUTHORED, IMPLEMENTS, HIRES_FOR, MENTIONED_IN are edges. At 500k nodes / low millions of edges "
 "this is comfortably served by Postgres edge tables + recursive CTEs. Multi-hop analytics (investor co-investment paths, talent flow) are projected nightly into a graph engine "
 "(Neo4j or Apache AGE on the same Postgres) so the system of record stays single and consistent.",
 "<b>Vectors.</b> pgvector (HNSW) for semantic search/entity candidate retrieval; move to a dedicated store (Qdrant) only beyond ~10M vectors. "
 "Embeddings are derived data and always rebuildable.",
 "<b>Entity resolution.</b> Deterministic cascade: domain match > exact on normalised key (legal suffixes, punctuation, spacing removed) > fuzzy (RapidFuzz ratio >= 93 with ambiguity gap) > new entity. "
 "Precision-first: below threshold we create a new entity rather than risk a false merge; every decision is written to the mapping log (raw, canonical, method, score) and is reproducible without an LLM."])

s += [Paragraph("Observability &amp; operations", H)]
s += bl(["Structured JSON logs with run-id/source/url; Prometheus metrics (req/s, 429 rate per tier, failover count, 413 shrinks, parse-rate, freshness lag p95); "
         "alerts on parse-rate or freshness regression; cost dashboard per tier. Schema versioning ('schemaVersion') allows replay/migrations from the raw store."])

doc = SimpleDocTemplate("docs/architecture.pdf", pagesize=A4, leftMargin=36, rightMargin=36, topMargin=30, bottomMargin=30,
                        title="GraphOne Ingestion Pipeline - Architecture")
doc.build(s)
