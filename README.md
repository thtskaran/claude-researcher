# Claude Deep Researcher

A hierarchical multi-agent research system that performs autonomous deep research on any topic. Built with real-time knowledge graphs, contradiction detection, and extended thinking.

## Quick Start

```bash
# Install/update Python dependencies
pip install -e .

# Launch the web UI (auto-starts API + UI)
researcher ui
```

Reports saved to `output/{topic}_{session-id}/`

---

## Real-World Comparison: claude-researcher vs ChatGPT Deep Research

Same topic: **"AI-powered penetration testing systems"** (1 hour research)

| Metric | claude-researcher | ChatGPT Deep Research |
|--------|------------------|----------------------|
| **Sources** | **127** | ~25 |
| **Findings** | **303** | Not tracked |
| **Knowledge Graph** | **2,688 entities** | None |
| **Contradiction Detection** | **Yes** | No |
| **Latest Sources** | **Jan 2026** | Older |

### What claude-researcher found that ChatGPT missed:
- OWASP Top 10 for Agentic Applications 2026 (brand new framework)
- 94.4% of LLM agents vulnerable to prompt injection (specific statistic)
- IBM Bob exploitation case study (Jan 2026)
- US Federal Register RFI on AI agent security
- Hexstrike-AI's 150-agent orchestration architecture

See full comparison: [`examples/`](./examples/)

---

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

# Skip clarification questions
researcher "AI safety" --no-clarify

# Fully autonomous mode (no interaction)
researcher "AI safety" --autonomous
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--time`, `-t` | Time limit in minutes | 60 |
| `--db`, `-d` | SQLite database path | research.db |
| `--no-clarify` | Skip pre-research clarification questions | False |
| `--autonomous`, `-a` | Run fully autonomous (no user interaction) | False |
| `--timeout` | Timeout in seconds for mid-research questions | 60 |

---

## Interactive Features

The researcher supports three interactive features to help focus and guide research:

### 1. Pre-Research Clarification

Before research begins, the system asks 2-4 questions to refine the scope:

```
┌─────────────────────────────────────────────────────────────────┐
│ Before we begin, a few quick questions to focus the research:  │
│ Press Enter to skip any question, or type 's' to skip all.     │
└─────────────────────────────────────────────────────────────────┘

What time period are you most interested in?
Options: Recent (2024-2025) / Historical / All time
> Recent (2024-2025)

What's your primary use case?
Options: Academic research / Business application / General learning
> Business application
```

The system then refines your goal:
```
┌─────────────────────────────────────────────────────────────────┐
│ Original: Machine learning in customer support                  │
│                                                                 │
│ Refined: Recent (2024-2025) machine learning applications in   │
│ customer support for business implementation, focusing on       │
│ practical deployment and ROI.                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Skip with:** `--no-clarify` flag or type `s` during questions.

### 2. Mid-Research Guidance Injection

While research is running, you can inject guidance to steer the direction:

1. **Type your message** (text won't be visible due to the spinner)
2. **Press Enter**
3. **The spinner pauses** and shows what you typed:
   ```
   ━━━ User Guidance ━━━
   You typed: focus more on chatbots and automated responses
   Press Enter to send, or type more to replace: _
   ```
4. **Press Enter** to confirm or type a replacement
5. **Spinner resumes** and your guidance is used in the next iteration

**Example guidance messages:**
- "Focus on practical implementations, not theory"
- "Look into company X specifically"
- "Skip historical background, go deeper on current research"
- "The AI findings seem outdated, search for 2024 sources"

The Manager agent will incorporate your guidance in its next thinking cycle:
```
[USER GUIDANCE] Received 1 message(s)
  → focus more on chatbots and automated responses
```

### 3. Mid-Research Questions (Async)

The Manager may occasionally ask you questions during research:

```
┌─────────────────────────────────────────────────────────────────┐
│ Question                                                        │
├─────────────────────────────────────────────────────────────────┤
│ Should I investigate pricing models in more depth?              │
│ We found conflicting information about enterprise costs.        │
│                                                                 │
│ Options: Yes, investigate / No, continue / Focus on SMB only    │
│ (60s timeout - research will continue if no response)           │
└─────────────────────────────────────────────────────────────────┘
> Yes, investigate
```

If you don't respond within the timeout, research continues autonomously.

### Interaction Modes

| Mode | Clarification | Mid-Research Input | Questions |
|------|---------------|-------------------|-----------|
| **Default** | Yes | Yes | Yes |
| `--no-clarify` | No | Yes | Yes |
| `--autonomous` | No | No | No |

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

### Cost Tracking

The system tracks API costs in real-time and displays an estimate at the end:

```
                        API Cost Estimate
┌────────────┬──────┬──────────┬──────────┬──────────┬──────────┐
│ Model      │ Calls│ Input    │ Output   │ Thinking │ Cost     │
├────────────┼──────┼──────────┼──────────┼──────────┼──────────┤
│ Sonnet 4.5 │ 45   │ 125,000  │ 45,000   │ 0        │ $1.0500  │
│ Opus 4.5   │ 12   │ 80,000   │ 35,000   │ 50,000   │ $2.5250  │
│ Haiku 4.5  │ 8    │ 15,000   │ 8,000    │ 0        │ $0.0550  │
│ Web        │ 23   │ 23 srch  │ 0 fetch  │ -        │ $0.2300  │
├────────────┼──────┼──────────┼──────────┼──────────┼──────────┤
│ TOTAL      │ 65   │ 220,000  │ 88,000   │ 50,000   │ $3.8600  │
└────────────┴──────┴──────────┴──────────┴──────────┴──────────┘
```

**Current Pricing (per million tokens):**

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| Opus 4.5 | $5 | $25 | Extended thinking tokens billed as output |
| Sonnet 4.5 | $3 | $15 | Extended thinking tokens billed as output |
| Haiku 4.5 | $1 | $5 | Extended thinking tokens billed as output |
| Web Search | - | - | $0.01 per search |

Cost data is also saved to `findings.json` for tracking across sessions.

---

## Architecture

```
                         ┌──────────────────────┐
                         │        USER          │
                         │  • Clarification     │
                         │  • Mid-research input│
                         └──────────┬───────────┘
                                    │
┌─────────────────────────────────────────────────────────────┐
│                     DIRECTOR (Sonnet)                       │
│  Session management, progress display, report export        │
│  + UserInteraction handler, InputListener                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              MANAGER (Opus + Extended Thinking)             │
│  • Decomposes goal into parallel research threads           │
│  • Critiques findings, identifies gaps                      │
│  • Steers research using knowledge graph insights           │
│  • Incorporates user guidance from message queue            │
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

**Tip**: If the knowledge graph HTML lags with large graphs (1000+ nodes), click the physics button at the bottom and disable physics to stop the simulation.

### Hybrid Memory

For long research sessions:
- **Recent Buffer**: Full recent messages (high fidelity)
- **Compressed Summary**: Older context summarized to save tokens
- **External Store**: Findings persisted to SQLite for retrieval

---

## Report Structure

Reports use **dynamic AI-driven section planning**. Instead of a fixed template, the AI analyzes findings and decides what sections are needed.

### How It Works

1. **Structure Planning** - AI analyzes findings and outputs a JSON plan of sections
2. **Section Generation** - Each section generated according to its type with specialized formatting

### Available Section Types

| Type | Format | When Used |
|------|--------|-----------|
| **TL;DR** | 2-3 sentence blockquote | Always first - bottom-line answer |
| **Flash Numbers** | `**94.4%** - description` | When quantitative data exists |
| **Stats Table** | Markdown comparison table | When comparing multiple items |
| **Comparison** | Side-by-side analysis | When evaluating approaches/systems |
| **Timeline** | Chronological progression | When temporal data exists |
| **Narrative** | Standard prose (4-6 paragraphs) | Core thematic sections |
| **Analysis** | Deep synthesis | Patterns and insights |
| **Gaps** | Open questions | Uncertainties and unknowns |
| **Conclusions** | Recommendations | Always near end |

### Example Output

```markdown
## 1. TL;DR
> AI-powered pentesting tools can generate working exploits in 10-15 minutes,
> with 94.4% of LLM agents vulnerable to prompt injection attacks.

## 2. Key Numbers
**94.4%** - LLM agents vulnerable to prompt injection
**10-15 min** - Time to generate working CVE exploits
**150+** - Tools orchestrated by Hexstrike-AI

## 3. Current Threat Landscape
[narrative prose...]

## 4. Framework Comparison
| Framework | Focus | Release | Tools |
|-----------|-------|---------|-------|
| PentestGPT | ... | ... | ... |

## 5. Analysis & Patterns
[synthesis...]
```

Reports use **Sonnet** for section generation with specialized prompts per section type.

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

## Hybrid Retrieval System

The system includes a high-quality hybrid retrieval module combining semantic and lexical search:

### Components

| Component | Purpose |
|-----------|---------|
| **BGE Embeddings** | State-of-the-art semantic embeddings (bge-large-en-v1.5) |
| **ChromaDB** | Local persistent vector database |
| **BM25** | Lexical search for exact keyword matching |
| **Cross-Encoder** | Reranking for final quality boost (bge-reranker-large) |

### Quality Benefits

- **15-30% better recall** than single-method retrieval
- **Hybrid fusion** catches both semantic and keyword matches
- **Reranking** provides cross-encoder accuracy on top candidates
- **Local inference** - all models run on-device

### Usage

```python
from src.retrieval import HybridRetriever, create_retriever

# Quick setup with best-quality defaults
retriever = create_retriever()

# Add documents
retriever.add_texts([
    "Quantum computing uses qubits for parallel computation",
    "Machine learning models learn from data patterns",
    "Neural networks are inspired by biological neurons",
])

# Search
results = retriever.search("how do quantum computers work")
for r in results:
    print(f"{r.score:.3f}: {r.content[:50]}...")

# For research findings specifically
from src.retrieval import FindingsRetriever
findings_retriever = FindingsRetriever()
findings_retriever.add_finding(finding, session_id)
results = findings_retriever.search("AI safety research")
```

### Configuration

```python
from src.retrieval import HybridConfig, EmbeddingConfig, RerankerConfig

config = HybridConfig(
    persist_directory=".retrieval",
    embedding=EmbeddingConfig(
        model_name="BAAI/bge-large-en-v1.5",  # Best quality
        device="auto",  # auto-detect GPU/MPS/CPU
    ),
    reranker=RerankerConfig(
        model_name="BAAI/bge-reranker-large",
    ),
    semantic_weight=0.5,  # Balance semantic vs BM25
    use_reranker=True,    # Enable cross-encoder reranking
)

retriever = HybridRetriever(config)
```

### Model Options

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `BAAI/bge-large-en-v1.5` | 1.3GB | Best | Slower |
| `BAAI/bge-base-en-v1.5` | 440MB | Good | Fast |
| `BAAI/bge-small-en-v1.5` | 130MB | OK | Fastest |

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
│   │   ├── director.py   # User interface, session management
│   │   └── parallel.py   # Parallel intern pool
│   ├── interaction/      # User interaction features
│   │   ├── models.py     # ClarifiedGoal, UserMessage, etc.
│   │   ├── config.py     # InteractionConfig
│   │   ├── handler.py    # UserInteraction class
│   │   └── listener.py   # Background input listener
│   ├── retrieval/        # Hybrid retrieval system
│   │   ├── embeddings.py # BGE embedding service
│   │   ├── vectorstore.py # ChromaDB vector store
│   │   ├── bm25.py       # BM25 lexical search
│   │   ├── reranker.py   # Cross-encoder reranking
│   │   ├── hybrid.py     # Hybrid retriever (RRF fusion)
│   │   ├── findings.py   # Research findings retriever
│   │   └── memory_integration.py # Semantic memory store
│   ├── costs/            # API cost tracking
│   │   └── tracker.py    # CostTracker, CostSummary, pricing
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
│   │   └── writer.py     # Dynamic section planning, specialized generators
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
- [Sentence Transformers](https://sbert.net/) - BGE embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database

Inspired by:
- [Gemini Deep Research](https://gemini.google/overview/deep-research/)
- [Perplexity](https://www.perplexity.ai/)
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher)
- [Stanford STORM](https://arxiv.org/abs/2402.14207)

## License

MIT
