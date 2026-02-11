# Claude Deep Researcher

A deep research tool for people who actually need answers, not paragraphs of fluff.

Most "AI research" tools give you a wall of vaguely-sourced text that sounds authoritative but says nothing specific. This one deploys a hierarchy of AI agents that independently search the web, extract discrete findings, build a knowledge graph, detect contradictions between sources, verify claims, and then synthesize everything into a report with inline citations you can trace back to the original source.

The result: 100-300+ findings from 50-150+ sources, a live knowledge graph with entity relationships and contradiction detection, and a report where every claim cites its source.

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd claude-researcher
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run research from CLI (results viewable in UI later)
researcher "Your research question" --time 30

# Or launch the web UI directly
researcher ui
```

That's it. Two commands: `pip install -e .`, then `researcher`.

### Prerequisites

1. **[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)** -- must be installed and authenticated. The system uses Claude's API through the CLI, so the `claude` command must work in your terminal. This is the backbone that powers all three agent tiers (Director, Manager, Interns).

2. **[Bright Data](https://brightdata.com/) API token** -- set the `BRIGHT_DATA_API_TOKEN` environment variable. This powers general web search and page scraping (see [How Web Search Works](#how-web-search-works) below). Academic search (Semantic Scholar, arXiv) does **not** require this token. Optionally set `BRIGHT_DATA_ZONE` (defaults to `mcp_unlocker`).

3. **Python 3.10+**

4. **Node.js 18+** (for the web UI -- auto-installs deps on first `researcher ui`)

---

## How Web Search Works (Bright Data)

Most AI research tools hit a wall the moment a website blocks their scraper. claude-researcher doesn't have that problem.

General web search and page scraping goes through [Bright Data](https://brightdata.com/)'s infrastructure, which gives the system two capabilities that generic HTTP requests can't match:

**SERP API** -- Google search results returned as structured data (titles, URLs, snippets) via Bright Data's `parsed_light` format. No HTML parsing, no breaking when Google changes their layout. The system gets clean, structured search results every time.

**Web Unlocker** -- Full page scraping that bypasses bot detection, CAPTCHAs, and anti-scraping measures. Pages are returned as clean markdown, ready for the LLM to process. This means the system can read paywalled research blogs, JavaScript-heavy SPAs, and sites that would return a 403 to a normal `requests.get()`. Bright Data handles proxy rotation, browser fingerprinting, and CAPTCHA solving transparently.

Why this matters for research quality:
- **Access**: Sites that block scrapers (news sites, research platforms, corporate blogs) are readable
- **Reliability**: Proxy rotation and automatic retries mean searches don't fail randomly
- **Clean data**: Markdown output means no HTML parsing bugs, no broken extractors
- **Scale**: 50-150+ sources per session without getting rate-limited or blocked

Without Bright Data, you'd be limited to whatever Google's basic search API returns and whatever sites don't block your IP. With it, the interns can actually read the pages they find.

## How Academic Search Works (Direct APIs -- No Bright Data)

For academic and scholarly research, interns also query free academic APIs **directly** -- these calls do **not** go through Bright Data:

**[Semantic Scholar](https://api.semanticscholar.org/)** -- 200M+ papers with citation graphs, TLDRs, and paper recommendations. Free tier (100 requests per 5 minutes, no API key needed). Returns structured metadata: titles, abstracts, authors, citation counts, venues, DOIs, and open-access PDF links.

**[arXiv](https://info.arxiv.org/help/api/)** -- 2.4M+ preprints across physics, math, CS, biology, economics, and more. Free, no API key required. Supports category-filtered search and provides full-text PDF access.

When the research topic contains academic indicators (e.g., "research", "study", "clinical trial", "algorithm"), interns automatically query these APIs **in parallel** with the Bright Data web search. Academic results are prioritized at the top of merged search results and converted to `SearchResult` objects for seamless integration with the existing pipeline.

---

## What Makes This Different

### vs ChatGPT Deep Research / Gemini Deep Research / Perplexity

Same topic: **"AI-powered penetration testing systems"** (1 hour)

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

The difference isn't just "more sources." It's that the system builds a knowledge graph as it researches, detects when sources contradict each other, identifies gaps in what it knows, and directs follow-up research to fill those gaps. It doesn't just search and summarize -- it reasons about what it's learning.

---

## Usage

### CLI (start research, view later in UI)

```bash
# Basic research (default 60 min)
researcher "Your research question here"

# Quick research
researcher "What is WebAssembly?" --time 5

# Long deep dive
researcher "Comprehensive overview of quantum computing" --time 120

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
- `knowledge_graph.html` -- Interactive graph visualization

All sessions are also persisted in SQLite, so you can view any past session in the UI at any time.

### Web UI (view everything)

```bash
researcher ui
```

This starts the API server (port 8080) and Next.js frontend (port 3000), then opens your browser. From the UI you can:

- **Sessions list** -- See all past and active research sessions
- **Live progress** -- Watch agents work in real-time via WebSocket
- **Report** -- Read the full report with working table of contents, inline citations, tables, and proper formatting
- **Findings** -- Browse all extracted findings with confidence scores and verification status
- **Knowledge Graph** -- Interactive visualization with entity types, relation labels, contradiction highlighting, and node detail panels
- **Sources** -- See every source analyzed with credibility scores
- **Verification** -- Review the Chain-of-Verification results
- **Agent Decisions** -- Full audit trail of every decision every agent made

```bash
# Open UI for a specific session
researcher ui abc123e

# Custom API port
researcher ui --port 9000

# Start without opening browser
researcher ui --no-browser
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--time`, `-t` | Time limit in minutes | 60 |
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
3. **Parallel search** -- 3 interns simultaneously search the web + academic APIs (Semantic Scholar, arXiv), extract findings, save to DB
4. **Knowledge graph** -- Entities and relations extracted in real-time (spaCy NER + LLM)
5. **Critique loop** -- Manager reviews findings, identifies gaps and contradictions, queues follow-up research
6. **Deep dive** -- Iterative cycles of search/critique/follow-up until time runs out
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

## Key Systems

### Knowledge Graph

Built incrementally as research progresses:
- **Entities**: Concepts, claims, evidence, methods, metrics, organizations
- **Relations**: supports, contradicts, causes, cites, implements, outperforms
- **Gap detection**: Graph analysis identifies under-researched areas
- **Contradiction detection**: Flags conflicting claims across sources with severity

The Manager uses KG insights to decide what to research next. The report writer uses KG data (key concepts, contradictions, gaps) to enrich section content.

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

Every section gets findings **selected specifically for that section** (keyword relevance + type affinity + verification status), not just the same top-N findings repeated.

Claims cite their sources inline using `[N]` notation that maps to the numbered reference list.

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
    +-- knowledge_graph.html         # Interactive graph visualization
```

### Cost Tracking

The system tracks API costs and shows an estimate at the end:

| Model | Input/M tokens | Output/M tokens |
|-------|-------|--------|
| Opus | $15 | $75 |
| Sonnet | $3 | $15 |
| Haiku | $0.80 | $4 |
| Web Search | $0.01/search | |

A typical 30-min session costs $2-5. A 60-min deep dive costs $5-15 depending on topic complexity.

---

## Project Structure

```
claude-researcher/
+-- src/
|   +-- agents/           # Director, Manager, Intern, ReAct loop
|   +-- knowledge/        # KG construction, gap detection, contradiction detection
|   +-- reports/          # Dynamic section planning, inline citations
|   +-- retrieval/        # Hybrid search (BM25 + vector + reranker)
|   +-- verification/     # CoVe + CRITIC verification pipeline
|   +-- memory/           # Hybrid buffer + compression for long sessions
|   +-- interaction/      # User clarification, mid-research guidance
|   +-- costs/            # API cost tracking
|   +-- audit/            # Agent decision logging
|   +-- tools/            # Web search (Bright Data) + academic search (Semantic Scholar, arXiv)
|   +-- models/           # Pydantic data models
|   +-- storage/          # SQLite persistence
|   +-- main.py           # CLI entry point
+-- api/                  # FastAPI backend for the web UI
+-- ui/                   # Next.js frontend
+-- output/               # Generated reports
+-- pyproject.toml
+-- CLAUDE.md             # AI coding assistant instructions
+-- README.md
```

---

## Troubleshooting

**"claude: command not found"** -- Install and authenticate [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code).

**Web searches failing** -- Set `BRIGHT_DATA_API_TOKEN` env var. Optionally set `BRIGHT_DATA_ZONE` (defaults to `mcp_unlocker`).

**spaCy model missing** -- Run `python -m spacy download en_core_web_sm`.

**UI won't start** -- Make sure Node.js 18+ is installed. `researcher ui` runs `npm install` automatically on first launch.

**Knowledge graph slow with 1000+ nodes** -- In the graph view, disable physics from the bottom toolbar.

---

## Credits

Built with [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code), [Claude Agent SDK](https://docs.anthropic.com/), [Bright Data](https://brightdata.com/) (SERP API + Web Unlocker), [Semantic Scholar API](https://api.semanticscholar.org/), [arXiv API](https://info.arxiv.org/help/api/), [Rich](https://github.com/Textualize/rich), [NetworkX](https://networkx.org/), [Sentence Transformers](https://sbert.net/), [ChromaDB](https://www.trychroma.com/), [FastAPI](https://fastapi.tiangolo.com/), [Next.js](https://nextjs.org/).

Inspired by [Gemini Deep Research](https://gemini.google/overview/deep-research/), [Perplexity](https://www.perplexity.ai/), [GPT Researcher](https://github.com/assafelovic/gpt-researcher), [Stanford STORM](https://arxiv.org/abs/2402.14207).

## License

MIT
