# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Setup & Commands

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run research (CLI -- results persist in SQLite, viewable in UI later)
researcher "Your research question" --time 30

# Launch web UI (starts API on :8080 + Next.js on :3000, opens browser)
researcher ui

# CLI options
researcher "topic" --time 60          # Time limit in minutes (default: 60)
researcher "topic" --no-clarify        # Skip pre-research clarification
researcher "topic" --autonomous        # No user interaction
researcher "topic" --db my.db          # Custom SQLite database path
researcher "topic" --timeout 30        # Timeout for mid-research questions

# UI options
researcher ui                          # Launch UI + API servers
researcher ui abc123e                  # Open specific session
researcher ui --port 9000              # Custom API port
researcher ui --no-browser             # Don't auto-open browser

# Linting (configured in pyproject.toml)
ruff check src/
ruff format src/
```

No formal test suite exists. Use short autonomous runs for integration testing:
```bash
researcher "Test query" --autonomous --time 5
```

## Architecture

Hierarchical 3-tier multi-agent deep research system. Not a summarizer -- it searches, extracts discrete findings, builds a knowledge graph, detects contradictions, verifies claims, and synthesizes reports with inline citations.

### Agent Hierarchy

```
Director (Sonnet) -> Manager (Opus + Extended Thinking) -> Intern Pool (3x parallel, Sonnet)
```

- **Director** (`src/agents/director.py`): User interface layer. Session management, pre-research clarification, progress display.
- **Manager** (`src/agents/manager.py`): Strategic brain. Decomposes goals, directs interns, critiques findings, uses KG gaps to guide research, synthesizes reports. Extended thinking for planning/synthesis.
- **Intern** (`src/agents/intern.py`): Web search + academic search (Semantic Scholar, arXiv), query expansion, finding extraction. Runs in parallel pool of 3.
- **BaseAgent** (`src/agents/base.py`): ReAct (Reason->Act->Observe) loop. Contains `ModelRouter` routing tasks to Haiku/Sonnet/Opus by complexity.

### Data Flow

1. Director clarifies goal -> creates session in SQLite
2. Manager decomposes into topics -> queues work
3. Parallel interns search web + academic APIs -> extract findings -> save to DB
4. Knowledge graph builds incrementally (spaCy NER + LLM entity/relation extraction)
5. Manager uses KG gaps/contradictions to guide follow-up research
6. Verification pipeline (CoVe + CRITIC) runs on findings
7. Synthesis: `DeepReportWriter` plans sections dynamically, selects findings per-section, generates with inline citations

### Key Subsystems

- **Reports** (`src/reports/writer.py`): Dynamic section planning. Per-section finding selection (keyword relevance + type affinity + verification weighting). Inline `[N]` citations. KG context injection. Source index mapping.
- **Hybrid Retrieval** (`src/retrieval/`): BM25 + ChromaDB vector search + Reciprocal Rank Fusion + cross-encoder reranking. Entry point: `create_retriever()`.
- **Knowledge Graph** (`src/knowledge/`): Real-time incremental graph using NetworkX + SQLite. Contradiction detection. Visualized as interactive HTML in UI.
- **Memory** (`src/memory/`): Hybrid buffer+compression for long sessions without token overflow.
- **Verification** (`src/verification/`): Chain-of-Verification + CRITIC pipeline. Results feed back into report confidence language.
- **Decision Logging** (`src/audit/`): Async fire-and-forget logging of all agent decisions for auditability.

### Web UI (`ui/`)

Next.js app with these pages:
- `/` -- Session list
- `/session/[id]` -- Session detail with tabs: Overview, Findings, Report, Sources
- `/session/[id]/graph` -- Interactive knowledge graph (vis-network)
- `/session/[id]/agents` -- Agent decision audit trail
- `/session/[id]/verify` -- Verification results
- `/session/[id]/sources` -- Source credibility analysis

API server: `api/server.py` (FastAPI on :8080). WebSocket endpoint for real-time progress.

### Model Routing

- **Haiku**: Classification, simple extraction (fast/cheap)
- **Sonnet**: Web search, finding extraction, analysis, report section generation
- **Opus**: Strategic planning, synthesis, critique (expensive -- minimize usage)

### Conventions

- Everything is `async/await` -- all LLM, DB, and web calls are non-blocking
- Agents receive `llm_callback` functions rather than direct model access (dependency injection)
- Decision logging uses `asyncio.create_task()` (fire-and-forget, never blocks research)
- Token estimation: `len(content) // 4` (rough but O(1))
- Session IDs are 7-character hex strings
- Output saved to `output/{topic_slug}_{session_id}/` with `report.md`, `findings.json`

### Web Search & Scraping (Bright Data)

General web search and page scraping goes through Bright Data (`src/tools/web_search.py`):

- **SERP API**: Google search via `parsed_light` format -- returns structured results (title, URL, snippet) from `organic` field. Endpoint: `https://api.brightdata.com/request` with `zone` and `data_format: "parsed_light"`.
- **Web Unlocker**: Full page scraping with `data_format: "markdown"` -- bypasses bot detection, CAPTCHAs, anti-scraping. Returns clean markdown for LLM consumption.
- **Retry logic**: 3 attempts with exponential backoff + jitter on both search and scrape.
- **Zone**: Configurable via `BRIGHT_DATA_ZONE` env var (defaults to `mcp_unlocker`).

### Academic Search (Direct APIs -- No Bright Data)

Academic paper search uses free APIs directly, **not** Bright Data (`src/tools/academic_search.py`):

- **Semantic Scholar**: 200M+ papers. Search, citation graphs, TLDRs, paper recommendations. Free tier: 100 requests per 5 minutes (no API key needed). Endpoint: `https://api.semanticscholar.org/graph/v1`.
- **arXiv**: 2.4M+ preprints. Category-filtered search, full-text PDF access. Free, no key required. Endpoint: `https://export.arxiv.org/api/query`.
- **Auto-detection**: Interns automatically query academic APIs in parallel with web search when the research topic contains academic indicators (e.g., "research", "study", "clinical trial").
- **Integration**: Results are converted to `SearchResult` objects for compatibility with the existing pipeline. Academic results are prioritized at the top of merged search results.

### Dependencies Requiring Setup

- **Claude Code CLI** -- **Required**. Must be installed and authenticated (`claude` command must work in terminal). This is the LLM backbone -- all three agent tiers (Director, Manager, Interns) use Claude models through it. Install: https://docs.anthropic.com/en/docs/claude-code
- **`BRIGHT_DATA_API_TOKEN`** env var -- **Required**. Powers general web search (SERP API) and page scraping (Web Unlocker). Academic search (Semantic Scholar, arXiv) does **not** require this. Get a token from https://brightdata.com/. Optionally set `BRIGHT_DATA_ZONE` (defaults to `mcp_unlocker`).
- **spaCy model**: `python -m spacy download en_core_web_sm`
- **ChromaDB**: persists vector embeddings locally
- **BGE embeddings**: `BAAI/bge-large-en-v1.5` downloaded on first use via sentence-transformers
- **Node.js 18+**: for the Next.js UI (auto npm-install on first `researcher ui`)
