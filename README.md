# Claude Deep Researcher

A hierarchical multi-agent research system built on the Claude Agent SDK. Inspired by [Gemini Deep Research](https://gemini.google/overview/deep-research/), [Perplexity](https://www.perplexity.ai/), and [GPT Researcher](https://github.com/assafelovic/gpt-researcher), this tool performs autonomous, deep research on any topic by coordinating multiple AI agents that search the web, analyze findings, critique results, and synthesize comprehensive narrative reports.

## Architecture

The system uses a three-tier agent hierarchy with specialized models:

```
┌─────────────────────────────────────────────────────────────┐
│                     DIRECTOR (Sonnet)                       │
│  - User interface & session management                      │
│  - Progress display & report export                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              MANAGER (Opus + Extended Thinking)             │
│  - Research strategy & topic decomposition                  │
│  - Critical evaluation of findings                          │
│  - Gap identification & follow-up planning                  │
│  - Final narrative synthesis                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      INTERN (Sonnet)                        │
│  - Web searches via Claude WebSearch                        │
│  - Finding extraction & categorization                      │
│  - Source identification & follow-up suggestions            │
└─────────────────────────────────────────────────────────────┘
```

Each agent implements a **ReAct (Reason + Act) loop**:
1. **Think** - Reason about current state and decide next action
2. **Act** - Execute the action (search, analyze, critique)
3. **Observe** - Process results and update state

## Features

### Research Capabilities
- **Autonomous Research**: Set a time limit (1 min to 8 hours) and let the system research independently
- **Deep Diving**: Automatically follows promising threads to configurable depth (default: 5 levels)
- **Real Web Search**: Uses Claude's WebSearch tool for current, up-to-date information
- **Dynamic Date Awareness**: Automatically uses current year in search queries for latest results
- **Finding Extraction**: Categorizes findings as facts, insights, connections, sources, questions, or contradictions
- **Manager Critique**: Each research batch is critically evaluated using Opus with extended thinking (10K+ thinking tokens)

### Verbose Console Output
See exactly what's happening during research:
- Search queries being executed
- Full search summaries from Claude
- Individual search results with URLs and snippets
- Extracted findings with type, source, and confidence scores
- Manager critiques and reasoning
- Follow-up topics identified

### Deep Narrative Reports
Generates comprehensive reports in the style of Gemini Deep Research and Perplexity:

| Section | Description |
|---------|-------------|
| **Executive Summary** | 3-4 paragraph overview of key findings |
| **Table of Contents** | Linked navigation to all sections |
| **Introduction** | Background context and research scope |
| **Thematic Sections** | 4-6 AI-identified themes with narrative prose |
| **Analysis & Insights** | Cross-cutting patterns and connections |
| **Conclusions** | Direct answers and recommendations |
| **References** | All sources with single retrieval date |
| **Appendix** | Methodology and statistics |

Reports are synthesized using **Opus with 16K extended thinking tokens** for deep analysis.

### Persistence
- **SQLite Database**: All findings stored for later analysis
- **Session Management**: Resume or review past research sessions

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd claude-researcher

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .
```

### Requirements

- Python 3.10+
- Claude Code CLI installed and authenticated (`claude` command available)
- Valid Anthropic API credentials (automatically uses Claude Code's credentials)

## Usage

### Basic Research

```bash
# Research a topic for 60 minutes (default)
researcher "What are the latest advances in fusion energy?"

# Set a custom time limit
researcher "History of quantum computing" --time 30

# Short research session (good for testing)
researcher "Current AI safety research directions" --time 5
```

### Command Options

```
researcher [OPTIONS] GOAL

Arguments:
  GOAL                  The research goal or question to investigate

Options:
  -t, --time INTEGER    Time limit in minutes (default: 60, max: 480)
  -d, --db PATH         Path to SQLite database file (default: research.db)
  -e, --export FORMAT   Export results to file (json or markdown)
  --help                Show this message and exit
```

### Examples

```bash
# Quick 5-minute research
researcher "What is WebAssembly?" --time 5

# Long deep research session
researcher "Comprehensive overview of renewable energy technologies" --time 120

# Use a specific database for project isolation
researcher "Machine learning in healthcare" --db ml_research.db

# Export to JSON instead of markdown
researcher "Quantum computing applications" --time 30 --export json
```

## Output

### Generated Files

Reports are saved to the `output/` folder (auto-created):

```
output/
├── research_1.md    # Deep narrative report
├── research_2.md
└── research_3.json  # If --export json used
```

### Sample Report Structure

```markdown
# What are the latest AI safety research directions?

*Deep Research Report*

**Generated:** January 23, 2026 at 06:02
**Session ID:** 15

## Table of Contents
- 1. Executive Summary
- 2. Introduction
- 3. Global Institutionalization of AI Safety
- 4. Mechanistic Interpretability Advances
- 5. AI Control Research
- ...

## 1. Executive Summary

The latest AI safety research is coalescing around two primary
technical directions—alignment and interpretability—while
simultaneously experiencing unprecedented international coordination...

## References

*All sources accessed on January 23, 2026.*

[1] International AI Safety Report 2025. *internationalaisafetyreport.org*.
    https://internationalaisafetyreport.org/publication/...

[2] Anthropic Alignment Recommendations. *alignment.anthropic.com*.
    https://alignment.anthropic.com/2025/recommended-directions/
```

### Database Schema

All research is persisted to SQLite:

| Table | Contents |
|-------|----------|
| `sessions` | Research sessions with goals and timestamps |
| `findings` | Individual findings with type, content, source, confidence |
| `topics` | Research topics with depth and status |
| `messages` | Inter-agent communication logs |

## Configuration

### Agent Configuration

```python
from src.agents.base import AgentConfig

config = AgentConfig(
    model="opus",              # sonnet, opus, or haiku
    max_turns=10,              # Max turns per Claude call
    max_iterations=100,        # Max ReAct loop iterations
    max_thinking_tokens=10000, # Extended thinking tokens
    allowed_tools=["WebSearch", "WebFetch"],
)
```

### Research Parameters

The Manager agent controls research depth:
- `max_depth=5` - How deep to follow topic threads
- `time_limit_minutes` - Overall session time limit
- Manager automatically uses Opus for deep reasoning

## Project Structure

```
claude-researcher/
├── src/
│   ├── agents/
│   │   ├── base.py       # Base agent with ReAct loop
│   │   ├── intern.py     # Web search agent (Sonnet)
│   │   ├── manager.py    # Research coordinator (Opus + extended thinking)
│   │   └── director.py   # User interface agent
│   ├── models/
│   │   └── findings.py   # Data models (Finding, Session, etc.)
│   ├── reports/
│   │   └── writer.py     # Deep narrative report generator (Opus)
│   ├── storage/
│   │   └── database.py   # SQLite persistence layer
│   ├── tools/
│   │   └── web_search.py # WebSearch tool wrapper
│   └── main.py           # CLI entry point
├── output/               # Generated reports (auto-created)
├── research.db           # SQLite database (auto-created)
├── pyproject.toml
└── README.md
```

## How It Works

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. DIRECTOR creates session, delegates to Manager       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 2. MANAGER (Opus) analyzes goal, creates search plan    │
│    Uses extended thinking for strategic planning        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 3. INTERN executes web searches                         │
│    - Shows search summaries                             │
│    - Displays results with URLs                         │
│    - Extracts and categorizes findings                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 4. MANAGER critiques findings                           │
│    - Identifies gaps and contradictions                 │
│    - Creates follow-up directives                       │
│    - Decides: go deeper or synthesize?                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼ (repeat 3-4 until time limit or sufficient coverage)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 5. REPORT WRITER (Opus + 16K thinking)                  │
│    - Generates executive summary                        │
│    - Identifies thematic sections                       │
│    - Writes narrative prose for each section            │
│    - Synthesizes analysis and conclusions               │
│    - Compiles references                                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 6. OUTPUT: output/research_{id}.md                      │
└─────────────────────────────────────────────────────────┘
```

## Finding Types

| Type | Description | Example |
|------|-------------|---------|
| **FACT** | Verified, specific information | "GPT-4 was released on March 14, 2023" |
| **INSIGHT** | Analysis or interpretation | "The trend suggests increasing adoption..." |
| **CONNECTION** | Links between topics | "This relates to earlier findings on..." |
| **SOURCE** | Valuable primary source | "The official documentation at..." |
| **QUESTION** | Unanswered question | "It remains unclear whether..." |
| **CONTRADICTION** | Conflicting information | "Source A claims X, but Source B states Y" |

## Inspiration

This project draws inspiration from:

- **[Gemini Deep Research](https://gemini.google/overview/deep-research/)** - Google's multi-page research reports with narrative synthesis
- **[Perplexity Deep Research](https://www.perplexity.ai/)** - Multi-pass querying with source confidence ratings
- **[GPT Researcher](https://github.com/assafelovic/gpt-researcher)** - Planner + execution agent architecture
- **[OpenAI Deep Research](https://openai.com/index/introducing-deep-research/)** - o3-powered autonomous research

## License

MIT

## Credits

Built with:
- [Claude Agent SDK](https://docs.anthropic.com/) - Anthropic's SDK for building on Claude
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal output
- [Typer](https://github.com/tiangolo/typer) - CLI framework
- [aiosqlite](https://github.com/omnilib/aiosqlite) - Async SQLite
