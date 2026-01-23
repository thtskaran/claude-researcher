# Claude Deep Researcher

A hierarchical multi-agent research system that performs autonomous deep research on any topic. Inspired by Gemini Deep Research, Perplexity, and GPT Researcher.

## Quick Start

```bash
# Install
pip install -e .

# Run (requires Claude Code CLI authenticated)
researcher "What are the latest AI safety research directions?" --time 10
```

Reports are saved to `output/research_{id}.md`.

---

## Key Concepts

### Research Depth

Depth = how many levels deep the research goes from your original question.

```
Depth 0: "What are the latest AI safety research directions?"  ← Your question
         │
         ├─ Depth 1: "Mechanistic interpretability 2026"       ← Decomposed aspects
         │           │
         │           └─ Depth 2: "Sparse autoencoders scaling" ← Follow-up topics
         │
         ├─ Depth 1: "AI alignment techniques"
         │
         └─ Depth 1: "AI governance frameworks"
```

- **Depth 0**: Your original question
- **Depth 1**: Sub-topics the system breaks your question into (parallel phase)
- **Depth 2+**: Follow-up topics discovered during research
- **Max Depth** (default: 5): Prevents infinite rabbit holes

### Research Phases

1. **Parallel Initial Phase**: Decomposes your goal into 3 aspects, researches them simultaneously
2. **Deep Dive Phase**: ReAct loop - manager critiques findings, identifies gaps, directs follow-up research
3. **Synthesis Phase**: Generates narrative report with extended thinking

### Agent Hierarchy

| Agent | Model | Role |
|-------|-------|------|
| **Director** | Sonnet | User interface, session management |
| **Manager** | Opus | Strategy, critique, synthesis (uses extended thinking) |
| **Intern** | Sonnet | Web searches, finding extraction |

### Finding Types

| Type | What it means |
|------|---------------|
| `FACT` | Verified specific information |
| `INSIGHT` | Analysis or interpretation |
| `CONNECTION` | Links between topics |
| `SOURCE` | Valuable primary source |
| `QUESTION` | Unanswered question for follow-up |
| `CONTRADICTION` | Conflicting information across sources |

---

## Installation

```bash
git clone <repo-url>
cd claude-researcher
pip install -e .
```

### Requirements

- Python 3.10+
- Claude Code CLI installed and authenticated (`claude` command works)
- API credentials (uses Claude Code's credentials automatically)

---

## Usage

```bash
# Basic research (default 60 min)
researcher "Your research question here"

# Quick research
researcher "What is WebAssembly?" --time 5

# Long deep research
researcher "Comprehensive overview of quantum computing" --time 120

# Custom database (isolate projects)
researcher "ML in healthcare" --db ml_research.db
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--time`, `-t` | Time limit in minutes | 60 |
| `--db`, `-d` | SQLite database path | research.db |

---

## Output

### Console Output

You'll see real-time progress:
- Search queries and results
- Extracted findings with confidence scores
- Manager critiques
- Knowledge graph statistics
- Follow-up topics being added

### Generated Files

Each research session creates a dedicated folder:

```
output/
└── ai-safety-research_a1b2c3d/     # {slug}_{session-id}/
    ├── report.md                    # Narrative research report
    ├── findings.json                # Structured findings data
    └── knowledge_graph.html         # Interactive visualization
```

**Naming convention:**
- `slug` - AI-generated from your research question (e.g., "ai-safety-research")
- `session-id` - Unique 7-character hex ID (e.g., "a1b2c3d")

### Session Statistics

At the end of research, you'll see:

| Stat | Meaning |
|------|---------|
| **Topics Explored** | Number of sub-topics researched |
| **Total Findings** | Facts, insights, etc. extracted |
| **Unique Searches** | Web searches performed |
| **Max Depth** | Deepest level reached (see above) |
| **Time Used** | Actual research duration |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DIRECTOR (Sonnet)                       │
│  Session management, progress display, report export        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              MANAGER (Opus + Extended Thinking)             │
│  • Decomposes goal into parallel research threads           │
│  • Critiques findings, identifies gaps                      │
│  • Steers research using knowledge graph insights           │
│  • Synthesizes final report                                 │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌────────────────┐
│  INTERN POOL     │ │  KNOWLEDGE GRAPH │ │    MEMORY      │
│  (Parallel)      │ │                  │ │                │
│  • Web searches  │ │  • Entities      │ │  • Hybrid      │
│  • Query expand  │ │  • Relations     │ │    buffer      │
│  • Extract facts │ │  • Gaps          │ │  • Compression │
│                  │ │  • Contradictions│ │  • External DB │
└──────────────────┘ └──────────────────┘ └────────────────┘
```

### Parallel Execution

The system uses an **intern pool** (default: 3 parallel interns) for:
- Initial research phase: 3 aspects researched simultaneously
- Queued topics: Multiple topics processed in parallel when available

This significantly speeds up research compared to sequential execution.

### Knowledge Graph

Built in real-time as research progresses:
- **Entities**: Concepts, claims, evidence, methods extracted from findings
- **Relations**: How entities connect (causes, supports, contradicts)
- **Gap Detection**: Graph analysis identifies missing knowledge
- **Contradiction Detection**: Flags conflicting claims for investigation

The manager uses KG insights to decide what to research next.

### Hybrid Memory

For long research sessions:
- **Recent Buffer**: Full recent messages (high fidelity)
- **Compressed Summary**: Older context summarized to save tokens
- **External Store**: Findings persisted to SQLite for retrieval

---

## Report Structure

Generated reports include:

| Section | Content |
|---------|---------|
| Executive Summary | 3-4 paragraph overview |
| Introduction | Background and scope |
| Thematic Sections | 4-6 AI-identified themes with narrative |
| Analysis & Insights | Cross-cutting patterns |
| Conclusions | Direct answers, recommendations |
| References | All sources with URLs |
| Appendix | Methodology, stats, KG analysis |

Reports use **Opus with extended thinking** for deep synthesis.

---

## Configuration

### Adjust Research Depth

In `src/agents/manager.py`:
```python
self.max_depth: int = 5  # How deep to follow threads
```

### Adjust Parallel Pool Size

```python
manager = ManagerAgent(
    db, intern, config,
    pool_size=3,        # Number of parallel interns
    use_parallel=True,  # Enable/disable parallel execution
)
```

### Model Selection

The system automatically routes tasks to appropriate models:

| Task | Model |
|------|-------|
| Classification, simple extraction | Haiku |
| Web search, query expansion | Sonnet |
| Strategic planning, synthesis, critique | Opus |

---

## Database

All research persists to SQLite:

| Table | Contents |
|-------|----------|
| `sessions` | Research sessions (7-char hex ID, goal, slug) |
| `findings` | Extracted findings with sources |
| `topics` | Research topics with depth/status |
| `kg_entities` | Knowledge graph entities |
| `kg_relations` | Entity relationships |
| `kg_contradictions` | Detected conflicts |

Files created:
- `research.db` - Main database
- `research_kg.db` - Knowledge graph
- `research_memory.db` - External memory store

Session IDs are unique 7-character hexadecimal strings (e.g., `a1b2c3d`).

---

## Troubleshooting

### "Topics Explored: 0"
Fixed in latest version. Update and reinstall: `pip install -e .`

### Research hangs during KG processing
KG now uses batch processing (5 findings per LLM call). Should be fast.

### Report generation fails
Now uses direct Anthropic API instead of subprocess. Ensure `anthropic` package installed.

### API errors
Check Claude Code is authenticated: `claude --version`

---

## Project Structure

```
claude-researcher/
├── src/
│   ├── agents/
│   │   ├── base.py       # ReAct loop, model routing
│   │   ├── intern.py     # Web search, query expansion
│   │   ├── manager.py    # Strategy, critique, KG integration
│   │   ├── director.py   # User interface
│   │   └── parallel.py   # Parallel intern pool
│   ├── knowledge/
│   │   ├── graph.py      # Incremental KG construction
│   │   ├── store.py      # NetworkX + SQLite hybrid
│   │   ├── query.py      # Gap detection interface
│   │   ├── credibility.py # Source scoring
│   │   └── visualize.py  # Pyvis, Mermaid output
│   ├── memory/
│   │   ├── hybrid.py     # Buffer + summary compression
│   │   └── external.py   # SQLite external store
│   ├── reports/
│   │   └── writer.py     # Narrative report generator
│   ├── models/
│   │   └── findings.py   # Data models
│   ├── storage/
│   │   └── database.py   # SQLite persistence
│   └── main.py           # CLI entry point
├── output/               # Generated reports
├── pyproject.toml
└── README.md
```

---

## Credits

Built with:
- [Claude Agent SDK](https://docs.anthropic.com/) - Agent framework
- [Anthropic API](https://docs.anthropic.com/) - Direct API for reports
- [Rich](https://github.com/Textualize/rich) - Terminal output
- [NetworkX](https://networkx.org/) - Graph algorithms
- [Pyvis](https://pyvis.readthedocs.io/) - Graph visualization

Inspired by:
- [Gemini Deep Research](https://gemini.google/overview/deep-research/)
- [Perplexity](https://www.perplexity.ai/)
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher)
- [Stanford STORM](https://arxiv.org/abs/2402.14207)

## License

MIT
