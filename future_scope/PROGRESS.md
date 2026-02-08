# Progress Audit

A concise done/not-done checklist for each future scope document.

---

## 1. improvements-research.md

**Focus:** State-of-the-art multi-agent research systems, search strategies, credibility, knowledge synthesis, report quality, cost optimization, memory management.

### Implemented ✅
| Feature | Location |
|---------|----------|
| Director-Manager-Intern hierarchy | `src/agents/director.py`, `manager.py`, `intern.py` |
| Parallel research execution | `src/agents/parallel.py` (asyncio.gather) |
| Knowledge graph construction | `src/knowledge/graph.py` |
| Contradiction detection | `src/knowledge/graph.py` (_check_contradiction) |
| Credibility scoring | `src/knowledge/credibility.py` |
| Hybrid memory (buffer + summary) | `src/memory/hybrid.py` |
| External memory storage | `src/memory/external.py` |
| Hybrid retrieval (BM25 + semantic) | `src/retrieval/hybrid.py`, `bm25.py` |
| Reranking | `src/retrieval/reranker.py` |
| Cost tracking | `src/costs/tracker.py` |
| Deep report writer with Opus | `src/reports/writer.py` |

### Remaining 🔴
| Feature | Priority | Notes |
|---------|----------|-------|
| **Query expansion** (semantic, contextual) | Medium | Only basic queries, no LLM-generated variations |
| **Self-RAG iterative refinement loop** | High | No gap identification → refined query loop |
| **Diverse source layer** (Academic DBs, News APIs) | High | Only web search (`tools/web_search.py`) |
| **Cross-validation triangulation** | Medium | Contradiction detection exists but no multi-source validation |
| **Model routing (Haiku/Sonnet/Opus)** | Medium | Fixed model usage, no dynamic routing |
| **Prompt caching** | Low | Not implemented |
| **Dynamic toolset selection** | Low | N/A (no MCP tools) |
| **MemGPT-style hierarchical memory** | Low | Hybrid exists, but no 3-tier archival system |
| **Co-STORM mind map pattern** | Medium | KG exists but not hierarchical mind map |

**Status: ~70% Complete**

---

## 2. user-interaction-design.md

**Focus:** Pre-research clarification, async mid-research questions, user message queue.

### Implemented ✅
| Feature | Location |
|---------|----------|
| Pre-research clarification | `src/interaction/handler.py` (clarify_research_goal) |
| Async mid-research questions | `src/interaction/handler.py` (ask_with_timeout) |
| User message queue | `src/interaction/handler.py` (inject_message, get_pending_messages) |
| InputListener for background input | `src/interaction/listener.py` |
| InteractionConfig | `src/interaction/config.py` |
| Director integration | `src/agents/director.py` (clarify_research_goal) |
| Manager integration | `src/agents/manager.py` (_maybe_prompt_user) |

### Remaining 🔴
| Feature | Priority | Notes |
|---------|----------|-------|
| **Windows support** for input listener | Low | Uses `select.select()` - Unix only |
| **Web UI** (WebSocket-based) | Future | CLI-only currently |
| **Question quality tuning** | Low | Basic prompts, could be improved |

**Status: ~95% Complete**

---

## 3. data-sources.md

**Focus:** Unified data access via Bright Data (replaces all individual API integrations)

### Implemented ✅
| Feature | Location |
|---------|----------|
| SERP search (Google) | `src/tools/web_search.py` → Bright Data API |
| Page scraping (bot-bypass) | `src/tools/web_search.py` → `fetch_page()` |

### Remaining 🔴
| Feature | Priority | Notes |
|---------|----------|-------|
| **Intern scrapes top SERP results** | High | Full content vs snippets only |
| **Structured extractors in pipeline** | Medium | GitHub/Reddit/YouTube Bright Data datasets |
| **PDF extraction from arXiv** | Medium | `fetch_page()` returns markdown; needs wiring |

**Status: ~40% Complete**

---

## 4. classical-ml-integration.md

**Focus:** Hybrid retrieval, LambdaMART ranking, fast NER, deduplication, topic modeling.

### Implemented ✅
| Feature | Location |
|---------|----------|
| Hybrid retrieval (BM25 + semantic) | `src/retrieval/hybrid.py` |
| Embedding service | `src/retrieval/embeddings.py` |
| BM25 index | `src/retrieval/bm25.py` |
| Reranking | `src/retrieval/reranker.py` |
| Vector store (ChromaDB) | `src/retrieval/vectorstore.py` |
| **MinHash LSH deduplication** | `src/retrieval/deduplication.py` |
| **spaCy fast NER** with LLM fallback | `src/knowledge/fast_ner.py` |

### Remaining 🔴
| Feature | Priority | Notes |
|---------|----------|-------|
| **LambdaMART ranking** (XGBoost) | Medium | No learning-to-rank |
| **BERTopic clustering** | Medium | No automatic topic discovery |
| **Text classification** (distilled models) | Low | Finding type via LLM only |
| **Novelty detection** (Isolation Forest) | Low | Not implemented |

**Status: ~60% Complete**

---

## 5. power-user-features-research.md

**Focus:** Comprehensive feature wishlist - academic workflows, AI patterns, integrations, exports, collaboration.

### Implemented ✅
| Feature | Location |
|---------|----------|
| Interactive research plan (partial) | Clarification in director.py |
| Contradiction detection | `src/knowledge/graph.py` |
| Knowledge graph construction | `src/knowledge/graph.py` |
| Credibility scoring | `src/knowledge/credibility.py` |
| KG visualization | `src/knowledge/visualize.py` |
| JSON/Markdown export | `src/agents/director.py` (export_findings) |
| Confidence scoring | `src/verification/confidence.py` |
| CoVe verification | `src/verification/cove.py` |
| CRITIC verification | `src/verification/critic.py` |
| Verification pipeline | `src/verification/pipeline.py` |

### Remaining 🔴 (MUST-HAVE)
| Feature | Category | Notes |
|---------|----------|-------|
| **Research Checkpoint System** | Power User | No pause/redirect mid-research |
| **Source Decision Audit Trail** | Power User | Why sources selected not exposed |
| **Persistent Cross-Session Memory** | Power User | Memory is session-only |
| **Semantic Paper Network Builder** | Academic | No citation graph visualization |
| **PRISMA-Compliant Review Assistant** | Academic | No systematic review support |
| **Smart Citation Analyzer** | Academic | No supporting/contradicting citation context |
| **Adversarial Verification Debates** | AI Patterns | Single-agent verification only |
| **RAG Sufficiency Detector** | AI Patterns | No "enough context?" checking |
| **Dynamic Retrieval Orchestration** | AI Patterns | Static retrieval pipeline |
| **Bidirectional Obsidian Sync** | Integration | No PKM integrations |
| **Zotero Library Integration** | Integration | No reference manager support |
| **API-First Architecture** | Integration | CLI only, no REST API |
| **Source Type Classifier** | Source | No primary/secondary classification |
| **Multi-Dimensional Bias Analyzer** | Source | Basic credibility only |
| **Source Chain Verification** | Source | No citation chain tracing |
| **Academic Export (LaTeX/BibTeX)** | Output | Markdown only |
| **Contradiction Resolution Agent** | Differentiator | Detection only, no resolution |
| **Explainable Agent Reasoning** | Differentiator | Partial (KG visible, not agent decisions) |

### Remaining 🔴 (SHOULD-HAVE)
| Feature | Category | Notes |
|---------|----------|-------|
| Output Length/Depth Sliders | Power User | No customization |
| Usage Dashboard | Power User | Cost tracking exists but no dashboard |
| Hypothesis Tree Builder | Academic | Not implemented |
| Multi-Agent Reflexion (MAR) | AI Patterns | Not implemented |
| Notion Database Integration | Integration | Not implemented |
| Research-to-Slides Pipeline | Output | Not implemented |
| Structured Data Export (CSV/JSON) | Output | Basic JSON only |
| Knowledge Graph Export (RDF/GraphML) | Output | HTML visualization only |
| Team Research Workspaces | Collaboration | Single-user only |
| Shared Knowledge Base | Collaboration | Not implemented |
| Hypothesis Generation Engine | Differentiator | Not implemented |
| Temporal Knowledge Tracking | Differentiator | Not implemented |
| Research Audit Trail | Differentiator | Partial (database logs) |

**Status: ~25% Complete**

---

## Overall Progress Summary

| Document | % Complete | Key Gaps |
|----------|------------|----------|
| improvements-research.md | ~75% | Self-RAG loop, model routing |
| user-interaction-design.md | ~95% | Windows support, Web UI |
| data-sources.md | ~40% | Full-content scraping and structured extractors not wired |
| classical-ml-integration.md | ~60% | LambdaMART ranking, topic modeling |
| power-user-features-research.md | ~25% | Checkpoints, memory persistence, integrations, exports |

---

## Priority

### Quick Wins (Low effort, high value)
1. ~~**MinHash LSH deduplication**~~ ✅ `src/retrieval/deduplication.py`
2. ~~**spaCy fast NER**~~ ✅ `src/knowledge/fast_ner.py`
3. **Source Decision Audit Trail** - Expose existing credibility reasoning
4. **Explainable Agent Reasoning** - Log agent decisions to DB

### High Impact (Medium effort)
1. **Intern scrapes top SERP results** - Full content via Bright Data `fetch_page()`
2. **Research Checkpoint System** - Pause/redirect mid-research
3. **Persistent Cross-Session Memory** - Project-level memory
4. **Bright Data structured extractors** - GitHub/Reddit/YouTube datasets in pipeline

### Strategic (High effort, transformative)
1. **API-First Architecture** - Enable integrations
2. **Obsidian/Zotero Integration** - Capture academic users
3. **LaTeX/BibTeX Export** - Academic workflow support
4. **Adversarial Verification Debates** - Higher accuracy

---

## Screens in future_scope/screens/

HTML mockups for a potential web UI:
- `agent_thinking_transparency_panel/code.html`
- `cove_verification_pipeline/code.html`
- `knowledge_graph_visualization/code.html`
- `new_research_session_setup/code.html`
- `real-time_research_activity_feed/code.html`
- `research_findings_browser/code.html`
- `research_report_preview/code.html`
- `research_session_dashboard/code.html`
- `research_sources_index/code.html`

**Status:** Design mockups only - no implementation.