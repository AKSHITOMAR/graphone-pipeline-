# GraphOne / FrontierAtlas – AI Engineer demo task

Async, fault-tolerant ingestion pipeline: bulk entity acquisition, 24h-fresh news/jobs, multi-tier LLM extraction,
deterministic entity resolution. **Zero-hallucination rule:** every record carries a real `source.url`; deterministic
APIs/feeds are preferred over LLMs, and any LLM output is stored only if its evidence quote appears verbatim on the page.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=... GROQ_API_KEY=... DEEPSEEK_API_KEY=...   # any subset works; tiers without keys are skipped
export GITHUB_TOKEN=...                                           # 5k req/h instead of 60
PYTHONPATH=src python -m graphone.cli --out out --startups 1100 --products 1100 --papers 1100
pytest -q                                                         # offline unit tests (mock LLM servers included)
```
Outputs in `out/`: `*.jsonl` (checkpoints, resumable), per-tab CSVs, `graphone_output.xlsx` (6 tabs; File > Import in
Google Sheets) or `--gsheet` to push directly (service account, `pip install gspread`).
Run it again within 24h and the exactly-once store (`seen.sqlite`) guarantees no news item is re-emitted.

## Layout
| file | role |
|---|---|
| `src/graphone/http.py` | async client: robots.txt, per-host pacing, backoff+full jitter, Retry-After, bot-challenge *detection* |
| `src/graphone/llm.py` | Gemini Flash -> Groq Llama 3 -> DeepSeek; 413 shrink, 429 backoff/failover, circuit breaker, JSON repair |
| `src/graphone/chunking.py` | density-aware truncation (lede + highest-signal paragraphs, hard cap) |
| `src/graphone/extract.py` | rules-first pricing classifier, LLM fallback, **grounding check** |
| `src/graphone/entity.py` | 50-entity seed, domain/exact/fuzzy (precision-first) resolution + mapping log |
| `src/graphone/dates.py`, `freshness.py` | date normalisation (relative, meta, JSON-LD, epoch), 24h window, exactly-once store |
| `src/graphone/sources/` | YC directory (startups/products), arXiv+GitHub (papers), RSS news, ATS/board job APIs |
| `src/graphone/pool.py` | bounded queue/worker pool: constant memory at 1k or 500k items |

## Data sources (all public/documented)
Startups: `yc-oss` YC dataset. Products: each startup's own site/pricing page. Papers: arXiv API (3s pacing) +
GitHub REST stars (+ Hugging Face papers API as Papers-with-Code successor). News: 5 RSS feeds. Jobs: Greenhouse,
Lever, Ashby, Remotive public JSON. Board tokens / feed URLs live at the top of `sources/signals.py`; verify them
at run time (dead ones are logged and skipped).

## Honest limitations
* Written and unit-tested offline; **live-source adapters (field names/endpoints) must be smoke-tested on your machine**.
* Anti-bot: the client detects Cloudflare/Datadome challenges and escalates instead of evading them (see architecture.pdf).
* Datasets are only as large as what the sources expose; counts are logged, never padded.
