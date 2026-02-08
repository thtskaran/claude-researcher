# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode
pip install -e .

# Run research
researcher "Your research question" --time 30

# CLI options
researcher "topic" --time 60          # Time limit in minutes (default: 60)
researcher "topic" --no-clarify        # Skip pre-research clarification
researcher "topic" --autonomous        # No user interaction
researcher "topic" --db my.db          # Custom SQLite database path
researcher "topic" --timeout 30        # Timeout for mid-research questions

# Linting (configured in pyproject.toml)
ruff check src/
ruff format src/
```

No formal test suite exists. Use short autonomous runs for integration testing:
```bash
researcher "Test query" --autonomous --time 5
```

## Architecture

This is a **hierarchical 3-tier multi-agent research system** that performs autonomous deep web research.

### Agent Hierarchy

```
Director (Sonnet 4.5) → Manager (Opus 4.5 + Extended Thinking) → Intern Pool (3x parallel, Sonnet 4.5)
```

- **Director** (`src/agents/director.py`): User interface layer. Handles session management, pre-research clarification, and progress display.
- **Manager** (`src/agents/manager.py`): Strategic brain. Decomposes goals, directs interns, critiques findings, synthesizes reports. Uses extended thinking for planning/synthesis.
- **Intern** (`src/agents/intern.py`): Executes web searches and extracts findings. Runs in parallel pool of 3.
- **BaseAgent** (`src/agents/base.py`): Implements ReAct (Reason→Act→Observe) loop. Contains `ModelRouter` that routes tasks to Haiku/Sonnet/Opus based on complexity.

### Data Flow

1. Director clarifies goal → creates session in SQLite
2. Manager decomposes into topics → queues work
3. Parallel interns search web → extract findings → save to DB
4. Knowledge graph builds incrementally from findings (entity/relation extraction via spaCy NER + LLM)
5. Manager uses KG gaps/contradictions to guide follow-up research
6. Synthesis phase: Manager with extended thinking → `DeepReportWriter` generates dynamic sections

### Key Subsystems

- **Hybrid Retrieval** (`src/retrieval/`): BM25 + ChromaDB vector search + Reciprocal Rank Fusion + cross-encoder reranking. Entry point: `create_retriever()`.
- **Knowledge Graph** (`src/knowledge/`): Real-time incremental graph using NetworkX + SQLite. Detects contradictions between sources. Visualized as interactive HTML.
- **Memory** (`src/memory/`): Hybrid buffer+compression to handle long sessions without token overflow.
- **Verification** (`src/verification/`): Chain-of-Verification + CRITIC pipeline for high-stakes claims.
- **Decision Logging** (`src/audit/`): Async fire-and-forget logging of all agent decisions for auditability.

### Model Routing

- **Haiku 4.5**: Classification, simple extraction (fast/cheap)
- **Sonnet 4.5**: Web search, finding extraction, analysis, report writing
- **Opus 4.5**: Strategic planning, synthesis, critique (expensive — minimize usage)

### Conventions

- Everything is `async/await` — all LLM, DB, and web calls are non-blocking
- Agents receive `llm_callback` functions rather than direct model access (dependency injection pattern)
- Decision logging uses `asyncio.create_task()` (fire-and-forget, never blocks research)
- Token estimation: `len(content) // 4 ≈ tokens` (rough but O(1))
- Session IDs are 7-character hex strings
- Output saved to `output/{topic_slug}_{session_id}/` with `report.md`, `findings.json`, `knowledge_graph.html`

### Dependencies Requiring Setup

- **Claude Code CLI** must be authenticated (`claude` command available) — API keys pulled from it
- **`BRIGHT_DATA_API_TOKEN`** env var required for web search/scraping (Bright Data API). Optionally set `BRIGHT_DATA_ZONE` (defaults to `mcp_unlocker`)
- **spaCy model**: required for fast NER (`python -m spacy download en_core_web_sm` or similar)
- **ChromaDB**: persists vector embeddings locally
- **BGE embeddings**: `BAAI/bge-large-en-v1.5` downloaded on first use via sentence-transformers
