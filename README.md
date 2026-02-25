# Claude Deep Researcher

A deep research tool for people who actually need answers, not paragraphs of fluff.

Most "AI research" tools give you a wall of vaguely-sourced text that sounds authoritative but says nothing specific. This one deploys a hierarchy of AI agents that independently search the web, extract discrete findings, build a knowledge graph, detect contradictions between sources, verify claims, and synthesize everything into a report with inline citations you can trace back to the original source.

The result: **100-300+ findings** from **50-150+ sources**, a live knowledge graph with entity relationships and contradiction detection, and a report where every claim cites its source.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [How Web Search Works](#how-web-search-works-bright-data)
- [How Academic Search Works](#how-academic-search-works)
- [What Makes This Different](#what-makes-this-different)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Web UI](#web-ui)
- [Key Systems](#key-systems)
- [Output](#output)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/thtskaran/claude-researcher.git
cd claude-researcher
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Download spaCy model (required for knowledge graph NER)
python -m spacy download en_core_web_sm

# Set up your Bright Data API token
cp .env.example .env
# Edit .env and add your BRIGHT_DATA_API_TOKEN

# Run research from CLI
researcher "Your research question" --iterations 10

# Or launch the web UI
researcher ui
```

Two commands to get started: `pip install -e .`, then `researcher`.

---

## Prerequisites

1. **[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)** -- must be installed and authenticated. The system uses Claude's API through the CLI, so the `claude` command must work in your terminal. This is the backbone that powers all three agent tiers.

2. **[Bright Data](https://brightdata.com/) API token** -- set `BRIGHT_DATA_API_TOKEN` in your `.env` file. This powers web search and page scraping (see [How Web Search Works](#how-web-search-works-bright-data) below). Academic search (Semantic Scholar, arXiv) does **not** require this token. Optionally set `BRIGHT_DATA_ZONE` (defaults to `mcp_unlocker`).

3. **Python 3.10+**

4. **Node.js 18+** (for the web UI only -- auto-installs npm deps on first `researcher ui`)

---

## How Web Search Works (Bright Data)

Most AI research tools hit a wall the moment a website blocks their scraper. claude-researcher doesn't have that problem.

General web search and page scraping goes through [Bright Data](https://brightdata.com/)'s infrastructure:

**SERP API** -- Google search results returned as structured data (titles, URLs, snippets) via Bright Data's `parsed_light` format. No HTML parsing, no breaking when Google changes their layout. Clean, structured search results every time.

**Web Unlocker** -- Full page scraping that bypasses bot detection, CAPTCHAs, and anti-scraping measures. Pages are returned as clean markdown, ready for the LLM to process. This means the system can read paywalled research blogs, JavaScript-heavy SPAs, and sites that would return a 403 to a normal `requests.get()`.

Why this matters:
- **Access**: Sites that block scrapers (news sites, research platforms, corporate blogs) are readable
- **Reliability**: Proxy rotation and automatic retries mean searches don't fail randomly
- **Clean data**: Markdown output means no HTML parsing bugs
- **Scale**: 50-150+ sources per session without getting rate-limited or blocked

---

## How Academic Search Works

For academic and scholarly research, interns also query free academic APIs **directly** -- these calls do **not** go through Bright Data:

**[Semantic Scholar](https://api.semanticscholar.org/)** -- 200M+ papers with citation graphs, TLDRs, and paper recommendations. Free tier (100 requests per 5 minutes, no API key needed).

**[arXiv](https://info.arxiv.org/help/api/)** -- 2.4M+ preprints across physics, math, CS, biology, economics, and more. Free, no API key required. Supports category-filtered search and full-text PDF access.

When the research topic contains academic indicators (e.g., "research", "study", "clinical trial"), interns automatically query these APIs **in parallel** with web search.

---

## What Makes This Different

### vs ChatGPT Deep Research / Gemini Deep Research / Perplexity

Same topic: **"AI-powered penetration testing systems"** (10 iterations)

| | claude-researcher | ChatGPT Deep Research |
|---|---|---|
| **Sources analyzed** | **127** | ~25 |
| **Discrete findings** | **303** | Not tracked |
| **Knowledge graph** | **2,688 entities, contradiction detection** | None |
| **Inline citations** | **Every claim cites [N]** | Vague attribution |
| **Verification** | **Chain-of-Verification pipeline** | None |
| **Latest sources** | **Jan 2026** | Older |

Things claude-researcher found that ChatGPT missed entirely:
- OWASP Top 10 for Agentic Applications 2026 (brand new framework)
- 94.4% of LLM agents vulnerable to prompt injection (specific statistic with source)
- IBM Bob exploitation case study (Jan 2026)
- US Federal Register RFI on AI agent security
- Hexstrike-AI's 150-agent orchestration architecture

The difference isn't just "more sources." The system builds a knowledge graph as it researches, detects when sources contradict each other, identifies gaps in what it knows, and directs follow-up research to fill those gaps. It doesn't just search and summarize -- it reasons about what it's learning.

---

## Usage

### CLI

```bash
# Basic research (default 5 iterations)
researcher "Your research question here"

# More iterations for deeper research
researcher "What is WebAssembly?" --iterations 10

# Quick shallow pass
researcher "Comprehensive overview of quantum computing" -n 3

# Fully autonomous (no interaction)
researcher "AI safety" --autonomous

# Skip clarification questions
researcher "AI safety" --no-clarify

# Custom database (isolate projects)
researcher "ML in healthcare" --db ml_research.db
```

Every research session saves to `output/{topic}_{session-id}/` with:
- `report.md` -- Full narrative report with inline citations
- `findings.json` -- All structured findings

All sessions are also persisted in SQLite, so you can view any past session in the UI at any time.

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--iterations`, `-n` | Number of research iterations (1-30) | 5 |
| `--db`, `-d` | SQLite database path | research.db |
| `--no-clarify` | Skip pre-research clarification | False |
| `--autonomous`, `-a` | No user interaction at all | False |
| `--timeout` | Timeout for mid-research questions (seconds) | 60 |

---

## How It Works

### Agent Hierarchy

```
Director (Sonnet) -- user interface, session management
    |
Manager (Opus + Extended Thinking) -- strategy, critique, synthesis
    |
Intern Pool (3x Sonnet, parallel) -- web + academic search, finding extraction
```

The Director talks to you. The Manager thinks. The Interns search.

### Research Flow

1. **Clarification** -- Director asks 2-4 questions to refine your goal (skip with `--no-clarify`)
2. **Decomposition** -- Manager breaks your question into 3+ parallel research threads
3. **Parallel search** -- 3 interns simultaneously search the web + academic APIs, extract findings, save to DB
4. **Knowledge graph** -- Entities and relations extracted in real-time (spaCy NER + LLM)
5. **Critique loop** -- Manager reviews findings, identifies gaps and contradictions, queues follow-up research
6. **Deep dive** -- Iterative cycles of search/critique/follow-up for the configured number of iterations
7. **Verification** -- Chain-of-Verification + CRITIC pipeline on high-stakes claims
8. **Synthesis** -- Report generated with section-relevant finding selection and inline citations

### Research Depth

```
Depth 0: "What are the latest AI safety research directions?"  <-- Your question
         |
         +- Depth 1: "Mechanistic interpretability 2026"       <-- Decomposed aspects
         |           |
         |           +- Depth 2: "Sparse autoencoders scaling" <-- Follow-up from gaps
         |
         +- Depth 1: "AI alignment techniques"
         |
         +- Depth 1: "AI governance frameworks"
```

The system goes up to depth 5 by default. Each level is driven by what the knowledge graph says is missing.

### Interactive Features

**Mid-research guidance**: While research runs, type a message and press Enter to steer the agents:
- "Focus on practical implementations, not theory"
- "Look into company X specifically"
- "Skip historical background, go deeper on current research"

**Mid-research questions**: The Manager may ask you questions during research (e.g., "Should I investigate pricing models?"). Times out after 60s and continues autonomously.

| Mode | Clarification | Mid-research input | Questions |
|------|---------------|-------------------|-----------|
| **Default** | Yes | Yes | Yes |
| `--no-clarify` | No | Yes | Yes |
| `--autonomous` | No | No | No |

---

## Web UI

```bash
# Launch UI (starts API on :8080 + Next.js on :3000, opens browser)
researcher ui

# Open a specific session
researcher ui abc123e

# Custom API port
researcher ui --port 9000

# Start without opening browser
researcher ui --no-browser
```

The web UI lets you:

- **Start new research** directly from the browser with configurable iterations
- **Watch live progress** as agents search, extract, and analyze in real-time via WebSocket
- **Read the full report** with working table of contents, inline citations, tables, and formatting
- **Browse findings** with confidence scores, verification status, and source links
- **Explore the knowledge graph** -- interactive visualization with entity types, relations, and contradiction highlighting
- **Review sources** with credibility scores and metadata
- **Inspect verification results** from the CoVe + CRITIC pipeline
- **Audit agent decisions** -- full trail of every decision every agent made

---

## Key Systems

### Knowledge Graph

Built incrementally as research progresses:
- **Entities**: Concepts, claims, evidence, methods, metrics, organizations
- **Relations**: supports, contradicts, causes, cites, implements, outperforms
- **Gap detection**: Graph analysis identifies under-researched areas
- **Contradiction detection**: Flags conflicting claims across sources with severity levels

The Manager uses KG insights to decide what to research next. The report writer uses KG context to enrich section content.

### Report Generation

Reports use **dynamic AI-driven section planning** -- the AI analyzes your findings and decides what sections are needed rather than using a fixed template.

| Section Type | When Used |
|------|-----------|
| **TL;DR** | Always first -- bottom-line answer in 2-3 sentences |
| **Flash Numbers** | Key statistics when quantitative data exists |
| **Stats Table** | Markdown table when comparing items |
| **Comparison** | Side-by-side analysis of approaches |
| **Timeline** | When temporal progression exists |
| **Narrative** | Core thematic sections (3-5 of these) |
| **Analysis** | Cross-cutting patterns and insights |
| **Gaps** | Open questions and contradictions |
| **Conclusions** | Always last -- recommendations and next steps |

Every section gets findings **selected specifically for that section** (keyword relevance + type affinity + verification status), not just the same top-N findings repeated. Claims cite their sources inline using `[N]` notation.

### Fact Verification

Chain-of-Verification (CoVe) + CRITIC pipeline:
- Verified findings marked `[VERIFIED]` -- cited with confidence in the report
- Flagged findings marked `[FLAGGED]` -- hedged language in the report
- Rejected findings deprioritized
- KG corroboration score factors into confidence

### Hybrid Retrieval

BM25 + ChromaDB vector search + Reciprocal Rank Fusion + cross-encoder reranking. 15-30% better recall than single-method retrieval. All models run locally.

---

## Output

### Generated Files

```
output/
+-- ai-safety-research_a1b2c3d/     # {slug}_{session-id}/
    +-- report.md                    # Narrative report with inline citations
    +-- findings.json                # All findings + cost data
```

### Cost Tracking

The system tracks API costs and shows an estimate at the end:

| Model | Input/M tokens | Output/M tokens |
|-------|-------|--------|
| Opus | $15 | $75 |
| Sonnet | $3 | $15 |
| Haiku | $0.80 | $4 |
| Web Search | $0.01/search | |

A typical 5-iteration session costs $2-5. A 10+ iteration deep dive costs $5-15 depending on topic complexity.

---

## Configuration

### Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```env
# Required: Bright Data API token for web search and scraping
BRIGHT_DATA_API_TOKEN=your_token_here

# Optional: Bright Data zone (default: mcp_unlocker)
BRIGHT_DATA_ZONE=mcp_unlocker

# Optional: Anthropic API key (falls back to Claude Code CLI credentials)
ANTHROPIC_API_KEY=your_key_here

# Optional: API key to protect the web UI API (no auth in local dev by default)
CLAUDE_RESEARCHER_API_KEY=your_secret_key

# Optional: Allowed CORS origins for the API (default: localhost:3000)
CLAUDE_RESEARCHER_CORS_ORIGINS=http://localhost:3000
```

---

## Project Structure

```
claude-researcher/
+-- src/
|   +-- agents/           # Director, Manager, Intern, ReAct loop, parallel pool
|   +-- knowledge/        # KG construction, gap detection, contradiction detection
|   +-- reports/          # Dynamic section planning, inline citations
|   +-- retrieval/        # Hybrid search (BM25 + vector + reranker)
|   +-- verification/     # CoVe + CRITIC verification pipeline
|   +-- memory/           # Hybrid buffer + compression for long sessions
|   +-- interaction/      # User clarification, mid-research guidance
|   +-- costs/            # API cost tracking
|   +-- audit/            # Agent decision logging
|   +-- tools/            # Web search (Bright Data) + academic (Semantic Scholar, arXiv)
|   +-- models/           # Pydantic data models
|   +-- storage/          # SQLite persistence
|   +-- events/           # WebSocket event system
|   +-- main.py           # CLI entry point
+-- api/                  # FastAPI backend for the web UI
|   +-- server.py         # FastAPI app, middleware, WebSocket endpoint
|   +-- routes/           # REST API endpoints
+-- ui/                   # Next.js 16 + React 19 frontend
|   +-- app/              # Pages (sessions list, detail, graph, verify, sources)
|   +-- components/       # React components (findings browser, activity feed, etc.)
+-- output/               # Generated research reports
+-- pyproject.toml        # Python project config
+-- .env.example          # Environment variable template
```

---

## Troubleshooting

**"claude: command not found"** -- Install and authenticate [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code). The `claude` command must work in your terminal.

**Web searches failing** -- Check that `BRIGHT_DATA_API_TOKEN` is set in your `.env` file. Optionally set `BRIGHT_DATA_ZONE` (defaults to `mcp_unlocker`).

**spaCy model missing** -- Run `python -m spacy download en_core_web_sm`.

**UI won't start** -- Make sure Node.js 18+ is installed. `researcher ui` runs `npm install` automatically on first launch.

**Knowledge graph slow with 1000+ nodes** -- In the graph view, disable physics from the bottom toolbar.

**Port already in use** -- `researcher ui` will offer to restart existing servers. Use `--port` to pick a different API port.

---

## Contributing

Contributions are welcome! Here are some ways to help:

- Report bugs by [opening an issue](https://github.com/thtskaran/claude-researcher/issues)
- Submit pull requests for bug fixes or new features
- Improve documentation
- Share your research results and feedback

### Development Setup

```bash
git clone https://github.com/thtskaran/claude-researcher.git
cd claude-researcher
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm

# Lint
ruff check src/
ruff format src/

# Integration test (short autonomous run)
researcher "Test query" --autonomous --iterations 2
```

---

## Credits

Built with [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code), [Claude Agent SDK](https://docs.anthropic.com/), [Bright Data](https://brightdata.com/), [Semantic Scholar API](https://api.semanticscholar.org/), [arXiv API](https://info.arxiv.org/help/api/), [Rich](https://github.com/Textualize/rich), [NetworkX](https://networkx.org/), [Sentence Transformers](https://sbert.net/), [ChromaDB](https://www.trychroma.com/), [FastAPI](https://fastapi.tiangolo.com/), [Next.js](https://nextjs.org/).

Inspired by [Gemini Deep Research](https://gemini.google/overview/deep-research/), [Perplexity](https://www.perplexity.ai/), [GPT Researcher](https://github.com/assafelovic/gpt-researcher), [Stanford STORM](https://arxiv.org/abs/2402.14207).

---

## Author

**Karan Prasad**

- Blog & more: [karanprasad.com](https://karanprasad.com)
- Email: [hello@karanprasad.com](mailto:hello@karanprasad.com)

---

## License

[MIT](LICENSE)
