# Technologies & Tools for PhD-Level Research Enhancement

*A comprehensive, actionable guide to cutting-edge technologies that can elevate claude-researcher into a real PhD-level scholar research tool.*

*Last Updated: February 2026*

---

## Table of Contents

1. [Competitive Landscape Analysis](#1-competitive-landscape-analysis)
2. [Academic Search APIs (Tier S: Critical)](#2-academic-search-apis-tier-s-critical)
3. [Advanced Retrieval Architectures](#3-advanced-retrieval-architectures)
4. [Knowledge Graph Enhancements](#4-knowledge-graph-enhancements)
5. [Verification & Fact-Checking](#5-verification--fact-checking)
6. [Memory & Cross-Session Intelligence](#6-memory--cross-session-intelligence)
7. [Report Generation Upgrades](#7-report-generation-upgrades)
8. [Academic Export & Integration](#8-academic-export--integration)
9. [Web Scraping & Data Acquisition](#9-web-scraping--data-acquisition)
10. [Multi-Agent Debate & Reasoning](#10-multi-agent-debate--reasoning)
11. [UI/UX Enhancements](#11-uiux-enhancements)
12. [Implementation Priority Matrix](#12-implementation-priority-matrix)

---

## 1. Competitive Landscape Analysis

### Direct Competitors on GitHub

| Tool | Stars | Key Tech | What We Can Learn |
|------|-------|----------|-------------------|
| [**GPT-Researcher**](https://github.com/assafelovic/gpt-researcher) | 25K+ | Multi-agent, Tavily search, MCP server | MCP server pattern, multi-source search, plugin architecture |
| [**DeerFlow**](https://github.com/bytedance/deer-flow) (ByteDance) | 19K+ | LangGraph, Python execution, podcast gen | LangGraph workflows, code execution in research, podcast/audio output |
| [**Alibaba DeepResearch**](https://github.com/Alibaba-NLP/DeepResearch) | 18K+ | Web agent, iterative search, Qwen-based | Iterative deepening strategy, GAIA benchmark approach |
| [**LightRAG**](https://github.com/HKUDS/LightRAG) | 28K+ | Graph-based RAG, dual-level retrieval | Graph+vector dual retrieval, entity-level and relation-level queries |
| [**Khoj**](https://github.com/khoj-ai/khoj) | 32K+ | Obsidian plugin, self-hosted, WhatsApp | Obsidian integration pattern, self-hosting, multi-platform access |
| [**Deep-Searcher**](https://github.com/zilliztech/deep-searcher) | 7K+ | Private data RAG, Milvus vectors | Private document research, vector DB scaling |
| [**MiroThinker**](https://github.com/MiroMindAI/MiroThinker) | 6K+ | GAIA benchmark 80.8%, prediction | Benchmark optimization, prediction capabilities |
| [**Local Deep Research**](https://github.com/LearningCircuit/local-deep-research) | 4K+ | Local LLMs, arXiv/PubMed, 95% SimpleQA | Academic API integration (arXiv, PubMed), local model support |
| [**Agentset**](https://github.com/agentset-ai/agentset) | 1.8K+ | RAG platform, 22+ file formats, MCP | File format support, MCP server, built-in citations |

### Key Differentiators We Already Have
- ✅ 3-tier agent hierarchy (Director→Manager→Intern) - unique architecture
- ✅ Real-time knowledge graph with contradiction detection
- ✅ Hybrid retrieval (BM25 + semantic + reranking)
- ✅ Credibility scoring with domain authority
- ✅ Chain-of-Verification + CRITIC verification pipeline
- ✅ Decision audit logging

### What We're Missing vs. Competitors
- ❌ Academic paper APIs (arXiv, PubMed, Semantic Scholar)
- ❌ MCP server for external tool integration
- ❌ Multi-format document ingestion (PDF, DOCX, etc.)
- ❌ Academic export formats (BibTeX, LaTeX, RIS)
- ❌ Code execution during research
- ❌ Graph-based RAG (GraphRAG / LightRAG pattern)
- ❌ Local/private model support
- ❌ Cross-session persistent memory

---

## 2. Academic Search APIs (Tier S: Critical)

**The single biggest gap** separating us from a PhD-level tool. A real research tool MUST access academic databases directly, not just Google.

### 2.1 Semantic Scholar API (Free, No Key Required)

**What:** AI2's academic search engine covering 200M+ papers with citation graphs, abstracts, and TLDRs.

**Why critical:** Provides structured metadata (citations, references, authors, venues, TLDRs) that web scraping cannot reliably extract. The citation graph enables building semantic paper networks.

**GitHub:** [allenai/s2-folks](https://github.com/allenai/s2-folks) (API docs + community)

**Integration point:** `src/tools/web_search.py` → Add `academic_search()` method

```python
# Key endpoints (all free, no API key needed for basic tier)
BASE = "https://api.semanticscholar.org/graph/v1"

# Search papers
GET /paper/search?query={query}&limit=100&fields=title,abstract,year,citationCount,
    referenceCount,tldr,authors,venue,openAccessPdf,externalIds

# Get paper details + citation graph
GET /paper/{paper_id}?fields=citations,references,citationCount,influentialCitationCount

# Batch paper lookup (up to 500 at once)
POST /paper/batch?fields=title,abstract,citationCount

# Author search
GET /author/search?query={name}

# Recommendation
GET /recommendations/v1/papers/forpaper/{paper_id}
```

**What it enables:**
- Citation count-weighted source credibility (a paper cited 5000 times vs a blog post)
- Semantic paper network builder (graph of citations + AI summaries per cluster)
- Finding foundational vs. emerging papers (citation velocity)
- Author expertise verification
- "Related papers" recommendations for deeper research

### 2.2 arXiv API (Free, No Key Required)

**What:** Open access repository for 2.4M+ preprints in physics, math, CS, biology, economics, etc.

**GitHub MCP Server:** [blazickjp/arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server) (2K+ stars)

**Integration point:** `src/tools/web_search.py` → Add `arxiv_search()` method

```python
# OAI-PMH API for metadata
BASE = "http://export.arxiv.org/api/query"

# Search with category filtering
GET ?search_query=all:{query}&start=0&max_results=50&sortBy=relevance

# Get specific paper
GET ?id_list=2301.00001

# Category-specific search
GET ?search_query=cat:cs.AI+AND+all:knowledge+graph&sortBy=lastUpdatedDate
```

**What it enables:**
- Access to cutting-edge preprints before journal publication
- Full-text PDF access for deep extraction
- Category-aware search (cs.AI, cs.CL, stat.ML, etc.)
- Version tracking (v1→v2→v3 of same paper)

### 2.3 PubMed / NCBI E-utilities (Free, API Key for Higher Rate)

**What:** 37M+ biomedical literature citations. Essential for any health/medical/biology research.

```python
# Search PubMed
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Search
GET /esearch.fcgi?db=pubmed&term={query}&retmax=100&sort=relevance

# Fetch abstracts
GET /efetch.fcgi?db=pubmed&id={pmid_list}&rettype=abstract&retmode=xml

# Related articles
GET /elink.fcgi?dbfrom=pubmed&id={pmid}&cmd=neighbor_score
```

**What it enables:**
- MeSH term-aware medical literature search
- Structured clinical trial data
- MEDLINE metadata (journal impact factor, study type)
- Related articles scoring for literature network building

### 2.4 OpenAlex API (Free, No Key Required)

**What:** Open catalog of 250M+ scholarly works, replacing Microsoft Academic Graph.

```python
# Rich filtering and faceting
GET https://api.openalex.org/works?search={query}
    &filter=publication_year:2023-2026,cited_by_count:>50,type:journal-article
    &sort=cited_by_count:desc
    &per-page=50

# Concept-based search (maps to Wikipedia concepts)
GET https://api.openalex.org/concepts/{concept_id}

# Institution and author disambiguation
GET https://api.openalex.org/authors/{author_id}
```

**What it enables:**
- Concept-level research mapping (what topics connect?)
- Institution-level analysis (which labs are leading?)
- Open access status checking
- Richer metadata than Semantic Scholar for some fields

### 2.5 Crossref API (Free, Polite Pool with Email)

**What:** DOI-based metadata for 150M+ scholarly works.

```python
# Search works with rich filtering
GET https://api.crossref.org/works?query={query}
    &filter=from-pub-date:2023,type:journal-article
    &sort=relevance&rows=50
    &mailto=your@email.com  # Polite pool (faster)

# Get work by DOI
GET https://api.crossref.org/works/{doi}
```

**What it enables:**
- DOI resolution for exact citation matching
- Reference counting and citation metadata
- Publisher and journal information
- Funder acknowledgment data (who funded this research?)

### Recommended Implementation

Create a new module `src/tools/academic_search.py`:

```python
class AcademicSearchProvider:
    """Unified academic search across multiple APIs."""

    async def search_papers(self, query: str, sources: list[str] = None,
                           min_citations: int = 0, year_range: tuple = None,
                           max_results: int = 50) -> list[AcademicPaper]:
        """Search across Semantic Scholar, arXiv, PubMed, OpenAlex."""

    async def get_citation_graph(self, paper_id: str,
                                depth: int = 2) -> CitationGraph:
        """Build citation network from seed paper."""

    async def find_related_papers(self, paper_id: str,
                                 limit: int = 20) -> list[AcademicPaper]:
        """Get AI-powered paper recommendations."""

    async def get_author_profile(self, author_name: str) -> AuthorProfile:
        """Author h-index, paper count, institution, top papers."""
```

---

## 3. Advanced Retrieval Architectures

### 3.1 GraphRAG (Microsoft Pattern)

**What:** Knowledge graph-augmented retrieval that uses community detection to answer global queries that vector search fails on.

**GitHub:** [microsoft/graphrag](https://github.com/microsoft/graphrag) (Original paper: "From Local to Global: A GraphRAG Approach")

**Why it matters:** Our current hybrid retrieval (BM25 + vector) is excellent for local queries ("What is X?") but weak for global queries ("What are the main themes across all findings?"). GraphRAG solves this by building community-level summaries.

**Integration point:** `src/retrieval/` → Enhance with graph-based retrieval from KG

```
Current: Query → BM25 + Vector → RRF → Rerank → Results

Enhanced: Query → Route → Local path (BM25 + Vector + RRF + Rerank)
                       → Global path (KG communities → summaries → answer)
                       → Hybrid path (both, merged)
```

**Key concept:** Use Leiden algorithm for community detection on our existing knowledge graph, generate community summaries at multiple levels, then use those summaries for answering global/thematic questions.

**Implementation sketch in existing code:**
- `src/knowledge/graph.py` already has a NetworkX graph — add community detection
- `src/retrieval/hybrid.py` already has query routing — add global query detection
- Cost: Minimal additional LLM calls (community summaries cached per session)

### 3.2 RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

**What:** Hierarchical retrieval where chunks are recursively clustered and summarized, creating a tree structure that enables multi-level abstraction queries.

**GitHub:** [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor) (1.5K+ stars, Stanford)

**Why it matters:** Our findings are flat. RAPTOR would let us build a tree of findings → themes → meta-themes, enabling queries at any abstraction level.

**Integration point:** `src/retrieval/findings.py`

```
Current:  [Finding1] [Finding2] [Finding3] ... [FindingN] → flat vector search

RAPTOR:   Meta-Theme: "AI in Healthcare"
          ├── Theme: "Diagnosis Applications"
          │   ├── Finding: "CNN accuracy 94%..."
          │   └── Finding: "FDA approved 3..."
          └── Theme: "Drug Discovery"
              ├── Finding: "AlphaFold reduces..."
              └── Finding: "Clinical trials show..."
```

### 3.3 Corrective RAG (CRAG) & Self-RAG

**What:** Retrieval systems that self-evaluate and self-correct. CRAG checks if retrieved docs are relevant and triggers web search as fallback. Self-RAG generates and critiques its own responses.

**Why it matters:** Our interns search the web but don't self-evaluate if their findings actually answer the research question. Adding retrieval self-assessment would reduce noise.

**Integration point:** `src/agents/intern.py` → Add post-retrieval relevance evaluation

```python
# After retrieving findings, evaluate relevance
relevance = await self._evaluate_retrieval_quality(query, findings)
if relevance < 0.6:
    # Reformulate query and retry with different strategy
    reformulated = await self._reformulate_query(query, findings)
    findings = await self._search_with_strategy(reformulated, "deep_dive")
```

### 3.4 ColBERT v2 / Late Interaction Reranking

**What:** Token-level interaction model that provides much more nuanced reranking than cross-encoder models.

**GitHub:** [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT)

**Why it matters:** Our current reranker uses a cross-encoder. ColBERT's late interaction pattern is both faster and often more accurate for academic text.

**Integration point:** `src/retrieval/reranker.py` → Alternative reranking backend

---

## 4. Knowledge Graph Enhancements

### 4.1 LightRAG-Style Dual-Level Retrieval

**What:** Query the knowledge graph at both entity level AND relation level, then merge results.

**GitHub:** [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (28K+ stars, EMNLP 2025)

**Why it matters:** Our KG currently supports entity-based queries. LightRAG shows that relation-level queries ("What causes X?", "What enables Y?") retrieve fundamentally different and complementary information.

**Integration point:** `src/knowledge/query.py`

```python
# Entity-level: "Find all entities related to CRISPR"
entity_results = kg.query_entities("CRISPR", depth=2)

# Relation-level: "Find all CAUSES relationships in gene editing"
relation_results = kg.query_relations("causes", domain="gene editing")

# Merge for comprehensive answer
merged = reciprocal_rank_fusion(entity_results, relation_results)
```

### 4.2 Community Detection for Thematic Clustering

**What:** Use Leiden/Louvain algorithm on the knowledge graph to find natural topic clusters.

**Library:** `networkx` (already a dependency) + `leidenalg` or `cdlib`

**Why it matters:** Automatically identifies research themes/subtopics from the knowledge graph structure. The report can then be organized by these natural clusters.

**Integration point:** `src/knowledge/graph.py`

```python
import community  # python-louvain, already compatible with networkx

def detect_communities(self) -> dict[int, list[str]]:
    """Detect topic communities in the knowledge graph."""
    partition = community.best_partition(self.graph.to_undirected())
    communities = {}
    for node, community_id in partition.items():
        communities.setdefault(community_id, []).append(node)
    return communities
```

### 4.3 Temporal Knowledge Graph

**What:** Add timestamps to KG edges to track how knowledge evolves over time.

**Why it matters:** Research findings can become outdated. A temporal KG lets us identify which claims are based on recent vs. old evidence, detect when newer research supersedes older findings.

**Integration point:** `src/knowledge/graph.py` → Add `timestamp` and `supersedes` edge attributes

### 4.4 Neo4j Export for Persistent Graph

**What:** Export KG to Neo4j for persistent storage and complex graph queries (Cypher).

**Library:** `neo4j` Python driver

**Why it matters:** NetworkX is in-memory. For cross-session knowledge accumulation, a persistent graph database is essential. Neo4j also enables complex pattern queries impossible with NetworkX.

---

## 5. Verification & Fact-Checking

### 5.1 Multi-Agent Debate (DMAD Pattern)

**What:** Deploy Prosecution, Defense, and Judge agents to debate high-stakes claims.

**Research:** "Diverse Multi-Agent Debate" (ICLR 2025) — 82.7% accuracy vs 76.2% single-agent

**Integration point:** `src/verification/` → New `debate.py` module

```python
class VerificationDebate:
    """Multi-agent debate for high-stakes claim verification."""

    async def debate(self, claim: str, evidence: list[Finding],
                    rounds: int = 3) -> DebateResult:
        prosecution = await self._argue_against(claim, evidence)
        defense = await self._argue_for(claim, evidence)
        verdict = await self._judge(claim, prosecution, defense)
        return DebateResult(
            verdict=verdict,
            confidence=verdict.confidence,
            prosecution_args=prosecution,
            defense_args=defense,
            rounds_needed=rounds
        )
```

### 5.2 Academic Source Cross-Referencing

**What:** Verify claims by checking if they appear in peer-reviewed literature using Semantic Scholar.

**Why it matters:** Our current CRITIC verifier searches the web. Cross-referencing against academic databases is far more reliable for scholarly claims.

**Integration point:** `src/verification/critic.py` → Add academic search fallback

### 5.3 Claim Decomposition Verification

**What:** Break complex claims into atomic sub-claims, verify each independently.

**Research:** Based on FActScore methodology (fine-grained claim verification)

**Integration point:** `src/verification/cove.py`

```python
async def decompose_and_verify(self, complex_claim: str) -> list[SubClaimResult]:
    """Break claim into atomic facts, verify each independently."""
    sub_claims = await self._decompose(complex_claim)
    # e.g., "CRISPR cured sickle cell in 94% of patients in 2024"
    # → ["CRISPR was used to treat sickle cell disease",
    #    "The treatment had a 94% success rate",
    #    "This occurred in 2024"]
    results = await asyncio.gather(*[self._verify_atomic(sc) for sc in sub_claims])
    return results
```

---

## 6. Memory & Cross-Session Intelligence

### 6.1 Project-Level Memory with Semantic Search

**What:** Store findings, sources, and insights across sessions with semantic retrieval.

**Why it matters:** Currently each research session starts from scratch. A PhD researcher builds on months/years of accumulated knowledge.

**Integration point:** `src/memory/external.py` → Enhance with vector embeddings

```python
class ProjectMemory:
    """Cross-session memory with semantic search."""

    async def store_session_insights(self, session_id: str, findings: list[Finding],
                                     report_summary: str, kg_snapshot: dict):
        """Persist session knowledge for future retrieval."""

    async def recall_relevant(self, query: str, project_id: str,
                             limit: int = 20) -> list[MemoryItem]:
        """Semantically search across all sessions in a project."""

    async def get_knowledge_growth(self, project_id: str) -> KnowledgeTimeline:
        """Track how understanding evolved across sessions."""
```

### 6.2 User Preference Learning

**What:** Learn and store user preferences (trusted sources, citation style, expertise areas).

**Why it matters:** A PhD researcher doesn't explain their field every time. The tool should remember.

**Integration point:** `src/memory/` → New `preferences.py`

```python
class UserPreferences:
    source_preferences: dict  # {"arxiv.org": "always_include", "medium.com": "deprioritize"}
    citation_style: str  # "APA" | "IEEE" | "Chicago" | "Harvard"
    expertise_areas: list[str]  # ["machine learning", "NLP"]
    preferred_depth: str  # "academic" | "general" | "executive"
    language: str  # "en"
```

### 6.3 Mem0-Style Memory Architecture

**What:** Three-tier memory: short-term (session), episodic (project), semantic (user knowledge).

**GitHub:** [mem0ai/mem0](https://github.com/mem0ai/mem0) — 26% accuracy boost documented

**Integration point:** Layer on top of existing `HybridMemory` + `ExternalMemoryStore`

---

## 7. Report Generation Upgrades

### 7.1 Evidence Mapping & Inline Confidence

**What:** Every claim in the report gets a visual confidence indicator and evidence chain.

**Integration point:** `src/reports/writer.py`

```markdown
## Key Finding

CRISPR-Cas9 has achieved a 94% success rate in sickle cell treatment trials.
[██████████░░ 85% confidence | 4 sources agree | 1 contradicts | newest: 2025]

Evidence chain:
├── [1] Nature Medicine (2025) — Primary clinical trial results
├── [2] NIH Clinical Trials (2024) — Phase III data confirming
├── [3] The Lancet (2024) — Independent replication study
├── [4] Science (2023) — Earlier Phase II with 89% rate
└── ⚠️ [5] bioRxiv preprint (2025) — Reports only 78% in different population
```

### 7.2 Dynamic Section Planning from KG Communities

**What:** Use knowledge graph community detection to automatically determine report sections.

**Integration point:** `src/reports/writer.py` → Use communities from KG for section planning

### 7.3 Multi-Format Export

**What:** Export reports in multiple formats beyond Markdown.

**Formats to support:**
- **LaTeX/BibTeX** — For academic papers (use `pylatex` library)
- **DOCX** — For Word users (use `python-docx` library)
- **PDF** — Direct PDF generation (use `weasyprint` or `pdfkit`)
- **HTML** — Styled report with interactive elements
- **RIS/BibTeX** — Citation-only export for reference managers

**Integration point:** `src/reports/` → New `exporters.py`

### 7.4 PRISMA Flow Diagram Generation

**What:** Auto-generate PRISMA-compliant systematic review flow diagrams.

**Why it matters:** PhD students doing systematic reviews spend weeks on PRISMA compliance. Auto-generating the flow diagram from search metadata is a massive time-saver.

**Integration point:** `src/reports/` → New `prisma.py`

---

## 8. Academic Export & Integration

### 8.1 Zotero Integration

**What:** Bidirectional sync with Zotero reference manager.

**API:** Zotero Web API v3 (requires API key)

```python
# Add sources to Zotero
POST https://api.zotero.org/users/{user_id}/items
Content-Type: application/json

# Import from Zotero library as research context
GET https://api.zotero.org/users/{user_id}/items?q={query}&format=json
```

**Integration point:** `src/tools/` → New `integrations/zotero.py`

### 8.2 Obsidian Vault Export

**What:** Export research as Obsidian-compatible markdown with wikilinks and frontmatter.

**Integration point:** `src/reports/` → New `obsidian_export.py`

```markdown
---
title: "CRISPR Gene Editing Research"
date: 2026-02-11
tags: [crispr, gene-editing, medicine]
session_id: abc123e
sources: 47
findings: 124
---

## Summary
...

## Key Findings
- [[Finding: CRISPR Success Rate]] — 94% in clinical trials
- [[Finding: FDA Approval Timeline]] — Expected 2027

## Knowledge Graph
![[knowledge_graph_abc123e.html]]

## Sources
- [[Source: Nature Medicine 2025]]
- [[Source: NIH Clinical Trials]]
```

### 8.3 BibTeX Generation

**What:** Auto-generate BibTeX entries from discovered sources with DOI resolution.

**Library:** `bibtexparser` for parsing, Crossref API for DOI→metadata

**Integration point:** `src/reports/writer.py` → Add `generate_bibtex()` method

```bibtex
@article{smith2025crispr,
  title={CRISPR-Cas9 Clinical Trial Results for Sickle Cell Disease},
  author={Smith, John and Doe, Jane},
  journal={Nature Medicine},
  volume={31},
  number={2},
  pages={145--152},
  year={2025},
  doi={10.1038/s41591-025-1234-5},
  note={Cited in sections 2.1, 3.4. Credibility: 0.95}
}
```

---

## 9. Web Scraping & Data Acquisition

### 9.1 Firecrawl Integration (Alternative/Complement to Bright Data)

**What:** AI-optimized web scraping that returns clean markdown from any website.

**GitHub:** [firecrawl/firecrawl](https://github.com/firecrawl/firecrawl) (81K+ stars)

**Why consider:** While Bright Data handles bot detection well, Firecrawl is specifically optimized for LLM consumption and offers:
- Cleaner markdown output (less noise)
- Site mapping/crawling capabilities
- Structured data extraction
- JavaScript rendering
- Self-hostable

**Integration point:** `src/tools/web_search.py` → Add as alternative scraping backend

### 9.2 PDF & Document Processing

**What:** Extract structured text from PDFs (papers, reports, books).

**Libraries:**
- `pymupdf` (fitz) — Fast PDF text extraction
- `marker-pdf` — High-quality PDF→markdown with layout preservation
- `unstructured` — Multi-format document processing (PDF, DOCX, PPTX, HTML)

**Why it matters:** Academic papers are primarily PDFs. Being able to ingest PDFs directly (from arXiv, user uploads, etc.) is essential for PhD-level research.

**Integration point:** `src/tools/` → New `document_processor.py`

### 9.3 Tavily Search API (Alternative Search Provider)

**What:** AI-optimized search API designed specifically for LLM agents.

**GitHub:** [tavily-ai/tavily-python](https://github.com/tavily-ai/tavily-python) (1K+ stars)

**Why consider:** Tavily provides:
- Pre-filtered, relevant results (less noise than raw Google)
- Built-in content extraction
- Fact-checking mode
- Research-optimized ranking

**Integration point:** `src/tools/web_search.py` → Add as alternative search provider alongside Bright Data

---

## 10. Multi-Agent Debate & Reasoning

### 10.1 STORM-Style Perspective-Guided Research

**What:** Research a topic from multiple expert perspectives (like Wikipedia's editorial process).

**Research:** Stanford's STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking)

**How it works:**
1. Identify relevant expert perspectives for the topic
2. Simulate conversations between experts and a topic expert
3. Use the diverse questions to guide comprehensive research
4. Synthesize into a well-structured report

**Integration point:** `src/agents/manager.py` → Enhance goal decomposition

```python
# Current: Decompose goal into topics
# Enhanced: Decompose goal into perspectives × topics

perspectives = [
    "As a clinician, what are the practical implications?",
    "As a geneticist, what are the mechanism details?",
    "As an ethicist, what are the societal concerns?",
    "As a patient advocate, what are the access issues?"
]
```

### 10.2 Tree-of-Thought Reasoning for Complex Questions

**What:** Explore multiple reasoning paths for complex research questions, evaluate each, and select the best.

**Why it matters:** Some research questions have multiple valid approaches. ToT ensures we explore diverse angles rather than fixating on the first approach.

**Integration point:** `src/agents/manager.py` → Use for research strategy planning

### 10.3 Iterative Refinement Loop

**What:** After initial research, evaluate completeness and trigger focused follow-up rounds.

**Inspired by:** Alibaba DeepResearch's iterative deepening approach (GAIA benchmark 80.8%)

**Integration point:** `src/agents/manager.py`

```python
async def iterative_research(self, goal: str, max_rounds: int = 3):
    for round in range(max_rounds):
        findings = await self._research_round(goal, round)
        gaps = await self._identify_gaps(findings, goal)
        if not gaps or self._is_sufficient(findings):
            break
        goal = self._refine_goal(goal, gaps)
```

---

## 11. UI/UX Enhancements

### 11.1 Glass-Box Research Dashboard

**What:** Real-time visualization of the research process showing all agent activity.

**Technologies:**
- **Server-Sent Events (SSE)** or **WebSocket** (already have WebSocket) for real-time updates
- **Framer Motion** for smooth animations
- **D3.js** or **vis-network** (already have vis-network) for graph visualization

**Key panels:**
```
┌─────────────────────────────────────────────────────────┐
│ Research Progress                          ████░░ 60%   │
├──────────────────┬──────────────────────────────────────┤
│ Agent Activity   │ Knowledge Graph (live)               │
│                  │                                      │
│ 🔵 Director:    │     [Entity A]──causes──[Entity B]   │
│   Planning next  │        │                    │        │
│   research phase │    part_of            contradicts    │
│                  │        │                    │        │
│ 🟢 Manager:     │     [Entity C]       [Entity D]     │
│   Analyzing 12   │                                      │
│   findings       ├──────────────────────────────────────┤
│                  │ Findings Feed                        │
│ 🟡 Intern 1:    │                                      │
│   Searching      │ ✅ FACT: CRISPR success rate 94%    │
│   "CRISPR 2025"  │    Confidence: 85% | Sources: 4     │
│                  │                                      │
│ 🟡 Intern 2:    │ ⚠️ CONTRADICTION: Earlier studies    │
│   Extracting     │    show only 78% in adults           │
│   from Nature.com│                                      │
│                  │ 💡 INSIGHT: Population-dependent     │
│ 🔴 Intern 3:    │    outcomes suggest genetic factors   │
│   Idle           │                                      │
└──────────────────┴──────────────────────────────────────┘
```

### 11.2 Interactive Knowledge Graph Explorer

**What:** Upgrade the current static KG visualization to an interactive explorer.

**Technologies:** `vis-network` (already installed) or `cytoscape.js` (more powerful)

**Features:**
- Click node → see all related findings
- Click edge → see evidence for relationship
- Color by community/theme
- Size by importance (citation count, centrality)
- Filter by confidence level
- Time slider for temporal view

### 11.3 Research Checkpoint UI

**What:** Visual checkpoints where users can pause, review, and redirect research.

**Integration point:** `ui/components/` → New `ResearchCheckpoint.tsx`

### 11.4 Confidence Visualization

**What:** Visual confidence bars, evidence chains, and contradiction highlights throughout the report.

**Integration point:** `ui/components/ReportPreview.tsx` → Enhance with confidence UI elements

---

## 12. Implementation Priority Matrix

### Phase 1: Scholar Foundation (Weeks 1-4)
*Transform from "web search tool" to "academic research tool"*

| Priority | Feature | Effort | Impact | Dependencies |
|----------|---------|--------|--------|-------------|
| 🔴 P0 | Semantic Scholar API integration | 3 days | 🔥🔥🔥🔥🔥 | None |
| 🔴 P0 | arXiv API integration | 2 days | 🔥🔥🔥🔥🔥 | None |
| 🔴 P0 | Citation count in credibility scoring | 1 day | 🔥🔥🔥🔥 | Semantic Scholar |
| 🟡 P1 | PubMed API integration | 2 days | 🔥🔥🔥🔥 | None |
| 🟡 P1 | OpenAlex API integration | 2 days | 🔥🔥🔥 | None |
| 🟡 P1 | BibTeX auto-generation from DOIs | 2 days | 🔥🔥🔥🔥 | Crossref API |

### Phase 2: Intelligence Upgrade (Weeks 5-8)
*Dramatically improve accuracy and depth*

| Priority | Feature | Effort | Impact | Dependencies |
|----------|---------|--------|--------|-------------|
| 🔴 P0 | GraphRAG community detection | 3 days | 🔥🔥🔥🔥🔥 | python-louvain |
| 🔴 P0 | Multi-agent verification debate | 4 days | 🔥🔥🔥🔥🔥 | None |
| 🟡 P1 | Claim decomposition verification | 2 days | 🔥🔥🔥🔥 | None |
| 🟡 P1 | CRAG self-assessment for retrieval | 2 days | 🔥🔥🔥🔥 | None |
| 🟢 P2 | RAPTOR hierarchical findings | 4 days | 🔥🔥🔥 | Clustering lib |
| 🟢 P2 | Temporal KG with supersession | 2 days | 🔥🔥🔥 | None |

### Phase 3: Memory & Export (Weeks 9-12)
*Cross-session intelligence and academic output*

| Priority | Feature | Effort | Impact | Dependencies |
|----------|---------|--------|--------|-------------|
| 🔴 P0 | Cross-session project memory | 4 days | 🔥🔥🔥🔥🔥 | Vector store |
| 🔴 P0 | LaTeX/BibTeX export | 3 days | 🔥🔥🔥🔥 | pylatex |
| 🟡 P1 | Obsidian vault export | 2 days | 🔥🔥🔥🔥 | None |
| 🟡 P1 | PDF document ingestion | 3 days | 🔥🔥🔥🔥 | pymupdf |
| 🟡 P1 | User preference learning | 2 days | 🔥🔥🔥 | None |
| 🟢 P2 | Zotero bidirectional sync | 3 days | 🔥🔥🔥 | Zotero API key |

### Phase 4: Glass-Box UI (Weeks 13-16)
*Trust-building transparency features*

| Priority | Feature | Effort | Impact | Dependencies |
|----------|---------|--------|--------|-------------|
| 🔴 P0 | Real-time research dashboard | 5 days | 🔥🔥🔥🔥🔥 | WebSocket (exists) |
| 🔴 P0 | Inline confidence visualization | 3 days | 🔥🔥🔥🔥🔥 | None |
| 🟡 P1 | Interactive KG explorer | 3 days | 🔥🔥🔥🔥 | vis-network (exists) |
| 🟡 P1 | Research checkpoint UI | 3 days | 🔥🔥🔥🔥 | None |
| 🟢 P2 | PRISMA flow diagram generation | 3 days | 🔥🔥🔥 | mermaid.js |
| 🟢 P2 | Source decision audit UI | 2 days | 🔥🔥🔥 | Decision log (exists) |

---

## Quick Wins (< 1 Day Each)

These can be implemented immediately with minimal effort:

1. **Add Semantic Scholar to credibility scoring** — Boost credibility score for sources that appear in Semantic Scholar with high citation counts. Modify `src/knowledge/credibility.py`.

2. **Add arXiv URL detection** — Auto-detect arXiv URLs in findings and extract paper metadata. Modify `src/tools/web_search.py`.

3. **Export findings as JSON-LD** — Structured data export for knowledge graph interoperability. Add to `src/reports/writer.py`.

4. **Add DOI detection and resolution** — Regex-detect DOIs in source URLs/text and resolve via Crossref for clean citations.

5. **Community detection in existing KG** — Add `python-louvain` and compute communities on the existing NetworkX graph for thematic organization.

6. **Source type classification enhancement** — Distinguish between journal articles, preprints, conference papers, books, reports, news, blogs. Enhance `src/knowledge/credibility.py`.

---

## Technology Stack Summary

### APIs to Integrate (All Free Tier Available)
| API | Purpose | Rate Limit (Free) | Key Required? |
|-----|---------|-------------------|---------------|
| Semantic Scholar | Paper search, citations, TLDRs | 100/5min | No (recommended) |
| arXiv | Preprints, full-text access | 1 req/3sec | No |
| PubMed (NCBI) | Biomedical literature | 3/sec (10 w/ key) | Optional |
| OpenAlex | Scholarly metadata, concepts | 100K/day | No |
| Crossref | DOI resolution, metadata | 50/sec (polite) | No (email) |
| Google Fact Check | Claim verification | 10K/day | Yes (free) |

### Python Libraries to Add
| Library | Purpose | Size | Already Have? |
|---------|---------|------|---------------|
| `python-louvain` | Community detection in KG | Tiny | ❌ |
| `pylatex` | LaTeX document generation | Small | ❌ |
| `bibtexparser` | BibTeX parsing/generation | Tiny | ❌ |
| `pymupdf` | PDF text extraction | Medium | ❌ |
| `python-docx` | DOCX export | Small | ❌ |

### Existing Dependencies to Leverage More
| Dependency | Currently Used For | Can Also Use For |
|------------|-------------------|------------------|
| `networkx` | Knowledge graph | Community detection, centrality, paths |
| `chromadb` | Vector store | Cross-session memory, finding embeddings |
| `sentence-transformers` | Embeddings | Semantic dedup, query-finding similarity |
| `spacy` | NER | Claim decomposition, entity linking |
| `httpx` | Bright Data API | All new API integrations |

---

## References

### Key Research Papers
1. **GraphRAG**: "From Local to Global: A GraphRAG Approach" (Microsoft, 2024)
2. **RAPTOR**: "Recursive Abstractive Processing for Tree-Organized Retrieval" (Stanford, 2024)
3. **DMAD**: "Diverse Multi-Agent Debate" (ICLR 2025) — 82.7% vs 76.2% accuracy
4. **LightRAG**: "Simple and Fast Retrieval-Augmented Generation" (EMNLP 2025)
5. **STORM**: "Assisting in Writing Wikipedia-like Articles From Scratch" (Stanford, 2024)
6. **Self-RAG**: "Learning to Retrieve, Generate, and Critique" (2023)
7. **CRAG**: "Corrective Retrieval Augmented Generation" (2024)
8. **FActScore**: "Fine-grained Atomic Evaluation of Factual Precision" (2023)
9. **ColBERT v2**: "Effective and Efficient Multi-Stage Retrieval" (Stanford, 2022)

### GitHub Repositories Referenced
- [microsoft/graphrag](https://github.com/microsoft/graphrag)
- [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
- [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher)
- [bytedance/deer-flow](https://github.com/bytedance/deer-flow)
- [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)
- [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor)
- [khoj-ai/khoj](https://github.com/khoj-ai/khoj)
- [blazickjp/arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server)
- [firecrawl/firecrawl](https://github.com/firecrawl/firecrawl)
- [tavily-ai/tavily-python](https://github.com/tavily-ai/tavily-python)
- [LearningCircuit/local-deep-research](https://github.com/LearningCircuit/local-deep-research)
- [mem0ai/mem0](https://github.com/mem0ai/mem0)
