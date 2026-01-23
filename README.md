# Claude Deep Researcher

A hierarchical multi-agent research system built on the Claude Agent SDK. This tool performs autonomous, deep research on any topic by coordinating multiple AI agents that search the web, analyze findings, critique results, and dive deeper into promising threads.

## Architecture

The system uses a three-tier agent hierarchy:

```
┌─────────────────────────────────────────────────────────────┐
│                        DIRECTOR                             │
│  - User interface                                           │
│  - Session management                                       │
│  - Progress display                                         │
│  - Report generation                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        MANAGER                              │
│  - Research strategy                                        │
│  - Topic decomposition                                      │
│  - Critique findings                                        │
│  - Identify gaps                                            │
│  - Synthesize results                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         INTERN                              │
│  - Web searches                                             │
│  - Extract findings                                         │
│  - Identify sources                                         │
│  - Suggest follow-ups                                       │
└─────────────────────────────────────────────────────────────┘
```

Each agent implements a **ReAct (Reason + Act) loop**:
1. **Think** - Reason about current state and decide next action
2. **Act** - Execute the action (search, analyze, critique)
3. **Observe** - Process results and update state

## Features

- **Autonomous Research**: Set a time limit and let the system research independently
- **Deep Diving**: Automatically follows promising threads to configurable depth
- **Real Web Search**: Uses Claude's WebSearch tool for current information
- **Finding Extraction**: Categorizes findings as facts, insights, connections, sources, questions, or contradictions
- **Manager Critique**: Each research batch is critically evaluated for quality and gaps (powered by Opus with extended thinking)
- **Verbose Output**: See exactly what's being searched, found, and analyzed
- **Deep Narrative Reports**: Generates Gemini/Perplexity-style research reports with:
  - Executive summary
  - Table of contents
  - Narrative synthesis sections organized by theme
  - Analysis and key insights
  - Conclusions and recommendations
  - APA-style references at the end
- **SQLite Persistence**: All findings stored for later analysis

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
- Valid Anthropic API credentials (via Claude Code login)

## Usage

### Basic Research

```bash
# Research a topic for 60 minutes (default)
researcher "What are the latest advances in fusion energy?"

# Set a custom time limit
researcher "History of quantum computing" --time 30

# Short research session
researcher "Current AI safety research directions" --time 5
```

### Options

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
# Quick 5-minute research with markdown export
researcher "What is WebAssembly?" --time 5 --export markdown

# Long research session
researcher "Comprehensive overview of renewable energy technologies" --time 120

# Use a specific database
researcher "Machine learning in healthcare" --db ml_research.db
```

## Output

### Console Output

The system shows detailed progress including:
- Search queries being executed
- Search summaries from Claude
- Individual search results with URLs
- Extracted findings with confidence scores
- Manager critiques
- Follow-up topics identified

### Markdown Report

A comprehensive `research_{session_id}.md` file is auto-generated in the style of Gemini Deep Research and Perplexity:

**Report Structure:**
1. **Title and Metadata** - Research question, date, session info
2. **Table of Contents** - Linked navigation to all sections
3. **Executive Summary** - 3-4 paragraph overview of key findings
4. **Introduction** - Background context and research scope
5. **Main Narrative Sections** - 4-6 thematic sections with synthesized prose (not bullet points)
6. **Analysis and Key Insights** - Cross-cutting patterns and connections
7. **Conclusions and Recommendations** - Direct answers and next steps
8. **References** - All sources in APA format at the end
9. **Appendix** - Research methodology and statistics

The report is generated using Opus with extended thinking for deep narrative synthesis.

### Database

All research is persisted to SQLite with tables for:
- `sessions` - Research sessions with goals and timestamps
- `findings` - Individual findings with type, content, source, confidence
- `topics` - Research topics with depth and status
- `messages` - Inter-agent communication

## Configuration

### Agent Configuration

Agents can be configured via `AgentConfig`:

```python
from src.agents.base import AgentConfig

config = AgentConfig(
    model="sonnet",        # sonnet, opus, or haiku
    max_turns=10,          # Max turns per Claude call
    max_iterations=100,    # Max ReAct loop iterations
    allowed_tools=["WebSearch", "WebFetch"],
)
```

### Research Parameters

The Manager agent controls research depth:
- `max_depth=5` - How deep to follow topic threads
- `time_limit_minutes` - Overall session time limit

## Project Structure

```
claude-researcher/
├── src/
│   ├── agents/
│   │   ├── base.py       # Base agent with ReAct loop
│   │   ├── intern.py     # Web search agent
│   │   ├── manager.py    # Research coordinator (Opus + extended thinking)
│   │   └── director.py   # User interface agent
│   ├── models/
│   │   └── findings.py   # Data models (Finding, Session, etc.)
│   ├── reports/
│   │   └── writer.py     # Deep narrative report generator
│   ├── storage/
│   │   └── database.py   # SQLite persistence layer
│   ├── tools/
│   │   └── web_search.py # WebSearch tool wrapper
│   └── main.py           # CLI entry point
├── pyproject.toml
└── README.md
```

## How It Works

1. **User provides a research goal** via CLI
2. **Director** creates a session and delegates to Manager
3. **Manager** analyzes the goal and creates search directives
4. **Intern** executes web searches and extracts findings
5. **Manager** critiques findings and identifies gaps
6. **Manager** creates follow-up directives for deeper research
7. Steps 4-6 repeat until time limit or sufficient coverage
8. **Manager** synthesizes all findings into a final report
9. **Director** displays results and exports markdown

## Finding Types

| Type | Description |
|------|-------------|
| FACT | Verified, specific information (dates, numbers, events) |
| INSIGHT | Analysis or interpretation from sources |
| CONNECTION | Links between topics or concepts |
| SOURCE | A valuable primary source to investigate further |
| QUESTION | An unanswered question worth investigating |
| CONTRADICTION | Conflicting information that needs resolution |

## License

MIT

## Credits

Built with:
- [Claude Agent SDK](https://github.com/anthropics/claude-code) - Anthropic's SDK for building on Claude Code
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal output
- [Typer](https://github.com/tiangolo/typer) - CLI framework
