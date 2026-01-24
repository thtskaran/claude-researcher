# Claude-Researcher Improvements: State-of-the-Art Multi-Agent Research Systems

*Research compiled: January 2026*

## Executive Summary

This document synthesizes research on improving hierarchical multi-agent research systems based on analysis of GPT Researcher, Stanford STORM, Gemini Deep Research, Perplexity, and Anthropic's own multi-agent research system. The recommendations are organized into seven key areas: architecture patterns, search strategies, source credibility, knowledge synthesis, report quality, cost optimization, and memory management.

---

## 1. State-of-the-Art Multi-Agent Research Systems

### 1.1 Anthropic's Multi-Agent Research System

Anthropic's internal research system provides the most relevant reference architecture:

**Architecture Pattern: Orchestrator-Worker**
- Lead agent (Claude Opus 4) coordinates specialized subagents (Claude Sonnet 4)
- Subagents operate in parallel, exploring different aspects simultaneously
- CitationAgent processes findings to attribute sources

**Key Performance Findings (BrowseComp Evaluation)**
- Token usage explains **80%** of performance variance
- Number of tool calls and model choice explain the remaining 15%
- Multi-agent system with Opus lead + Sonnet subagents outperformed single-agent Opus by **90.2%**

**Subagent Requirements**
Each subagent needs:
1. Clear objective
2. Defined output format
3. Guidance on tools and sources
4. Clear task boundaries
5. Effort scaling rules (1 agent for simple queries; 10+ for complex)

**Source**: [Anthropic Engineering Blog - Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)

### 1.2 GPT Researcher

**Architecture Components**
- **Planner Agent**: Generates research questions, aggregates findings into final report
- **Execution Agents**: Seek relevant information for each research question in parallel
- **Chief Editor**: Oversees research process and manages the team using LangGraph

**Key Techniques**
- Parallelization via `asyncio.gather()` for concurrent research
- Hybrid research combining web sources with local documents
- Multi-format output (PDF, Docx, Markdown)

**Benchmark**: Outperformed Perplexity, OpenAI, OpenDeepSearch in citation quality, report quality, and information coverage

**Source**: [GPT Researcher Official](https://gptr.dev), [GitHub Discussion](https://github.com/assafelovic/gpt-researcher/discussions/467)

### 1.3 Stanford STORM

STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking) offers a unique pre-writing methodology:

**Two-Stage Process**
1. **Pre-writing Stage**: Internet research + outline generation
2. **Writing Stage**: Full article generation with citations

**Key Innovations**
- **Perspective-Guided Question Asking**: Includes specific perspective in prompts to provide focus
- **Simulated Conversations**: LLM simulates conversation between writer and retrieval-augmented QA component
- **Automatic Perspective Mining**: Discovers diverse perspectives from related Wikipedia articles

**Co-STORM Enhancement**
- Maintains dynamic **mind map** organizing collected information hierarchically
- Reduces mental load during long, in-depth research sessions

**Implementation**: Built with DSPy framework (`pip install knowledge-storm`)

**Source**: [Stanford STORM Project](https://storm-project.stanford.edu/research/storm/), [GitHub](https://github.com/stanford-oval/storm)

### 1.4 Gemini Deep Research

**Architecture**
- Reasoning core uses Gemini 3 Pro (optimized for factual accuracy)
- Multi-step reinforcement learning for search navigation
- Maximum 60-minute research time (most tasks complete in 20 minutes)

**Key Features**
- User-editable research plans before execution
- Thinking panel shows what model has learned and intends to do
- Integrates Gmail, Drive, Chat for personal context
- Iterative planning: queries, reads results, identifies gaps, searches again

**Benchmark**: 46.4% on Humanity's Last Exam, 66.1% on DeepSearchQA, 59.2% on BrowseComp

**Source**: [Gemini Deep Research](https://gemini.google/overview/deep-research/), [Google AI Developers](https://ai.google.dev/gemini-api/docs/deep-research)

### 1.5 Perplexity Architecture

**Infrastructure**
- Uses **Vespa.ai** for massive-scale RAG (hundreds of billions of webpages indexed)
- Unified engine combining vector search, lexical search, structured filtering, ML ranking

**RAG Pipeline**
- Multi-stage ranking progressively refines results
- Dense retrieval (vector search) for semantic matching
- Hybrid retrieval combines lexical and semantic signals
- On-demand crawling avoids stale static indexes

**Source**: [Vespa.ai - Perplexity Case Study](https://vespa.ai/perplexity/), [ByteByteGo Analysis](https://blog.bytebytego.com/p/how-perplexity-built-an-ai-google)

---

## 2. Search Strategy Improvements

### 2.1 Query Expansion Techniques

**Core Approaches**
1. **Retriever-based expansion**: Use initial retrieval results to expand queries
2. **Generation-based expansion**: Use LLMs to generate additional query terms

**Recommended Techniques**
- **Semantic Query Expansion**: Use word embeddings (BERT-based) to find semantically similar terms
- **Contextual Term Addition**: Add terms based on initial retrieval context
- **WordNet/Thesaurus Expansion**: Add synonyms for domain-specific queries

**RAG-Specific Expansion**
```python
# Example: Multi-query generation for RAG
def expand_query(original_query: str, llm) -> list[str]:
    """Generate multiple query variations for better retrieval"""
    prompt = f"""Generate 3-5 alternative search queries that would help answer: "{original_query}"

    Include:
    - A more specific version
    - A broader version
    - Queries using synonyms
    - Queries focusing on different aspects"""

    return llm.generate(prompt).split('\n')
```

**Source**: [PMC - Semantic Query Expansion Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC11935759/)

### 2.2 Iterative Search Refinement (Self-RAG Pattern)

**FAIR-RAG Architecture**
1. Initial routing assesses query complexity
2. Simple queries answered directly; complex ones enter iterative loop
3. Loop: adaptive query generation → hybrid retrieval → filtering → evidence assessment
4. **Structured Evidence Assessment (SEA)**: Deconstructs query into checklist, verifies findings, identifies "Remaining Gaps"
5. Loop continues until sufficiency achieved

**Self-Correcting RAG Implementation**
```python
class SelfCorrectingRAG:
    def __init__(self, retriever, llm, max_iterations=5):
        self.retriever = retriever
        self.llm = llm
        self.max_iterations = max_iterations

    def research(self, query: str) -> dict:
        context = []
        for i in range(self.max_iterations):
            # Retrieve new information
            new_docs = self.retriever.search(query)
            context.extend(new_docs)

            # Evaluate sufficiency
            evaluation = self.llm.evaluate_context(query, context)

            if evaluation['sufficient']:
                break

            # Generate refined query based on gaps
            query = self.llm.generate_refined_query(
                original_query=query,
                current_context=context,
                knowledge_gaps=evaluation['gaps']
            )

        return {'context': context, 'iterations': i + 1}
```

**Source**: [FAIR-RAG Paper](https://arxiv.org/html/2510.22344v1), [Agentic RAG Survey](https://arxiv.org/html/2501.09136v3)

### 2.3 Source Diversity Strategy

**Recommended Source Mix**
| Source Type | Purpose | Examples |
|-------------|---------|----------|
| Academic Databases | Peer-reviewed research | Semantic Scholar, PubMed, Scopus |
| Web Search | Current information | Google, Bing, DuckDuckGo |
| News APIs | Recent events | NewsAPI, GDELT |
| Domain Databases | Specialized knowledge | JSTOR (humanities), IEEE (engineering) |
| Government Sources | Official statistics | Science.gov, data.gov |
| Open Access | Free full-text | DOAJ, CORE, BASE |

**Implementation Pattern**
```python
class DiverseRetriever:
    def __init__(self, sources: list[SearchSource]):
        self.sources = sources

    async def search(self, query: str, source_weights: dict = None) -> list[Document]:
        """Search multiple sources in parallel with configurable weights"""
        tasks = [source.search(query) for source in self.sources]
        results = await asyncio.gather(*tasks)

        # Deduplicate and merge with source diversity scoring
        merged = self.merge_with_diversity(results, source_weights)
        return merged

    def merge_with_diversity(self, results: list, weights: dict) -> list:
        """Ensure representation from multiple source types"""
        seen_urls = set()
        diverse_results = []

        # Round-robin selection to ensure diversity
        for round_num in range(10):
            for source_results in results:
                if source_results and round_num < len(source_results):
                    doc = source_results[round_num]
                    if doc.url not in seen_urls:
                        seen_urls.add(doc.url)
                        diverse_results.append(doc)

        return diverse_results
```

**Source**: [Paperpile - Academic Search Engines](https://paperpile.com/g/academic-search-engines/)

---

## 3. Source Credibility and Verification

### 3.1 Automated Credibility Scoring

**Credibility Signals Framework**
Modern credibility assessment aggregates multiple signals:

| Signal | Description | Weight |
|--------|-------------|--------|
| Factuality | Accuracy of claims | High |
| Source Authority | Domain expertise, institutional backing | High |
| Bias Detection | Political/commercial bias indicators | Medium |
| Recency | Publication date relevance | Medium |
| Citation Network | How often cited by other sources | Medium |
| Author Credentials | Author expertise and track record | Medium |
| Persuasion Techniques | Presence of manipulation patterns | Low (negative) |

**Implementation**
```python
class CredibilityScorer:
    def score_source(self, document: Document) -> CredibilityScore:
        signals = {
            'domain_authority': self.score_domain(document.url),
            'factual_claims': self.verify_claims(document.content),
            'bias_indicator': self.detect_bias(document.content),
            'citation_score': self.get_citation_metrics(document),
            'recency': self.score_recency(document.published_date),
            'author_credentials': self.verify_author(document.author)
        }

        # Weighted aggregation
        weights = {'domain_authority': 0.25, 'factual_claims': 0.30,
                   'bias_indicator': 0.15, 'citation_score': 0.15,
                   'recency': 0.10, 'author_credentials': 0.05}

        final_score = sum(signals[k] * weights[k] for k in weights)
        return CredibilityScore(score=final_score, signals=signals)

    def score_domain(self, url: str) -> float:
        """Score based on domain reputation"""
        domain = extract_domain(url)

        high_credibility = ['.edu', '.gov', 'nature.com', 'science.org',
                           'pubmed.ncbi.nlm.nih.gov', 'arxiv.org']
        medium_credibility = ['wikipedia.org', 'britannica.com',
                              'reuters.com', 'apnews.com']

        for pattern in high_credibility:
            if pattern in domain:
                return 0.9
        for pattern in medium_credibility:
            if pattern in domain:
                return 0.7
        return 0.5  # Unknown domain baseline
```

**Source**: [Automated Source Credibility Scoring](https://www.sourcely.net/resources/what-is-automated-source-credibility-scoring), [Veracity Open-Source Fact-Checking](https://arxiv.org/html/2506.15794v1)

### 3.2 Cross-Validation Through Triangulation

**Types of Triangulation for AI Research**

1. **Data Triangulation**: Validate findings across different sources
2. **Methodological Triangulation**: Use multiple retrieval methods (semantic + lexical)
3. **Temporal Triangulation**: Verify claims across different time periods

**Implementation Pattern**
```python
class FactTriangulator:
    def validate_claim(self, claim: str) -> TriangulationResult:
        """Validate a claim across multiple independent sources"""

        # Search for supporting/contradicting evidence
        search_queries = self.generate_validation_queries(claim)
        evidence = []

        for query in search_queries:
            results = self.diverse_search(query)
            for doc in results:
                stance = self.determine_stance(claim, doc.content)
                evidence.append({
                    'source': doc.url,
                    'stance': stance,  # 'supports', 'contradicts', 'neutral'
                    'excerpt': doc.relevant_excerpt,
                    'credibility': doc.credibility_score
                })

        # Calculate triangulation score
        supporting = sum(1 for e in evidence if e['stance'] == 'supports')
        contradicting = sum(1 for e in evidence if e['stance'] == 'contradicts')

        confidence = self.calculate_confidence(
            supporting=supporting,
            contradicting=contradicting,
            total=len(evidence),
            credibility_weighted=True
        )

        return TriangulationResult(
            claim=claim,
            confidence=confidence,
            evidence=evidence,
            verdict=self.determine_verdict(confidence, contradicting)
        )
```

**Source**: [PMC - Methodological Triangulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9714985/)

---

## 4. Knowledge Synthesis Patterns

### 4.1 Knowledge Graph Construction

**LLM-Driven Knowledge Graph Approaches**

1. **Generative Knowledge Modeling**: Synthesize structured representations from unstructured text
2. **Semantic Unification**: Integrate heterogeneous sources through natural language grounding
3. **Entity Resolution**: Normalize variations in naming, tense, plurality

**KGGen Approach**
```python
class KnowledgeGraphBuilder:
    def __init__(self, llm):
        self.llm = llm

    def extract_entities_and_relations(self, text: str) -> list[Triple]:
        """Extract knowledge graph triples from text"""
        prompt = """Extract structured knowledge from this text as triples.
        Format: (subject, relation, object)

        Text: {text}

        Rules:
        - Normalize entity names (consistent capitalization, singular form)
        - Use canonical relation types where possible
        - Include confidence scores"""

        raw_triples = self.llm.generate(prompt.format(text=text))
        return self.parse_and_normalize(raw_triples)

    def merge_graphs(self, graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
        """Merge multiple KGs with conflict resolution"""
        merged = KnowledgeGraph()

        for graph in graphs:
            for triple in graph.triples:
                existing = merged.find_similar(triple)
                if existing:
                    # Conflict resolution via consensus
                    merged.resolve_conflict(existing, triple)
                else:
                    merged.add(triple)

        return merged
```

**Source**: [KGGen Paper](https://arxiv.org/html/2502.09956v1), [LLM-empowered KG Construction Survey](https://arxiv.org/html/2510.20345v1)

### 4.2 Contradiction Resolution

**Multi-LLM Consensus Validation**
```python
class ContradictionResolver:
    def __init__(self, models: list[LLM]):
        self.models = models

    def resolve(self, claim_a: str, claim_b: str, context: str) -> Resolution:
        """Resolve contradictions using multi-model consensus"""

        prompt = f"""Two sources provide contradictory information:

        Claim A: {claim_a}
        Claim B: {claim_b}

        Context: {context}

        Analyze:
        1. Are these truly contradictory or can both be true in different contexts?
        2. Which claim has stronger evidence?
        3. What additional information would resolve this?

        Provide your assessment with confidence level."""

        # Get assessments from multiple models
        assessments = [model.generate(prompt) for model in self.models]

        # Determine consensus
        return self.find_consensus(assessments)

    def find_consensus(self, assessments: list) -> Resolution:
        """Determine majority position with uncertainty quantification"""
        positions = [self.extract_position(a) for a in assessments]

        # Count agreement
        position_counts = Counter(positions)
        majority_position, count = position_counts.most_common(1)[0]

        confidence = count / len(positions)

        return Resolution(
            verdict=majority_position,
            confidence=confidence,
            requires_human_review=(confidence < 0.7),
            reasoning=assessments
        )
```

**Source**: [Clinical KG Construction with Multi-LLMs](https://arxiv.org/html/2601.01844)

### 4.3 Co-STORM Mind Map Pattern

**Dynamic Hierarchical Organization**
```python
class DynamicMindMap:
    def __init__(self):
        self.root = ConceptNode("Research Topic")
        self.concept_index = {}

    def add_finding(self, finding: Finding):
        """Add finding to appropriate place in hierarchy"""
        # Determine best parent concept
        parent = self.find_best_parent(finding)

        # Create or update concept node
        concept_key = self.extract_concept(finding)
        if concept_key in self.concept_index:
            self.concept_index[concept_key].add_evidence(finding)
        else:
            new_node = ConceptNode(concept_key, finding)
            parent.add_child(new_node)
            self.concept_index[concept_key] = new_node

    def get_summary(self, max_depth: int = 3) -> str:
        """Generate hierarchical summary for context window"""
        return self.root.to_summary(max_depth=max_depth)

    def identify_gaps(self) -> list[str]:
        """Identify under-researched areas based on concept coverage"""
        gaps = []
        for concept, node in self.concept_index.items():
            if node.evidence_count < 2:
                gaps.append(f"Need more sources for: {concept}")
            if node.has_contradictions:
                gaps.append(f"Resolve contradictions in: {concept}")
        return gaps
```

**Source**: [Stanford Co-STORM](https://storm-project.stanford.edu/research/storm/)

---

## 5. Report Quality

### 5.1 Report Structure Best Practices

**Recommended Structure for Deep Research Reports**

```markdown
# [Research Topic]

## Executive Summary
- 150-300 words
- Key findings (3-5 bullet points)
- Primary recommendations
- Confidence assessment

## Table of Contents

## 1. Introduction
- Research question/objective
- Scope and methodology
- Key terms defined

## 2. Background
- Context and history
- Current state of knowledge
- Why this matters now

## 3. Methodology
- Search strategy used
- Sources consulted (with counts)
- Limitations acknowledged

## 4. Findings
### 4.1 [Theme 1]
- Key finding with evidence
- Supporting sources with inline citations
- Confidence level

### 4.2 [Theme 2]
...

## 5. Analysis
- Synthesis across themes
- Patterns and trends identified
- Contradictions and their resolution
- Knowledge gaps identified

## 6. Conclusions
- Summary of key insights
- Implications
- Recommendations
- Future research directions

## References
- Full citations with URLs
- Access dates for web sources

## Appendix
- Detailed methodology
- Raw data/evidence tables
- Search queries used
```

### 5.2 Narrative Flow Techniques

**Problem-Solution-Benefits Framework**
1. Establish clear, well-defined problem
2. Present compelling solution
3. Highlight tangible benefits

**Information Flow Principles**
- Move from familiar to new information in each section
- Most important details first, less important afterward
- Main argument flows coherently without interruption

**Storytelling Integration**
- Narrativize stakes to create empathetic connection
- Focus on people affected by the research topic
- Chronicle impact over time

**Source**: [Kittelson - Storytelling in Technical Reports](https://www.kittelson.com/ideas/the-plot-thickens-elevating-technical-reports-through-storytelling-techniques/)

### 5.3 Visualization and Knowledge Translation

**Infographic Integration**
- Use infographics to illustrate complex relationships
- Visualize data patterns and trends
- Create summary cards for key findings

**Recommended Tools**
- Flourish for interactive data visualizations
- Mermaid diagrams for architecture/flow visualization
- Tables for comparative analysis

**Example: Findings Summary Card**
```
+------------------------------------------+
|  FINDING: [Key Finding Title]            |
+------------------------------------------+
|  Confidence: ████████░░ 80%              |
|  Sources: 7 supporting, 1 contradicting  |
+------------------------------------------+
|  Key Evidence:                           |
|  • Point 1 (Source A, B)                 |
|  • Point 2 (Source C, D, E)              |
+------------------------------------------+
|  Implications: [Brief statement]         |
+------------------------------------------+
```

**Source**: [CHI 2024 - Data Storytelling](https://dl.acm.org/doi/10.1145/3613904.3643022)

---

## 6. Cost Optimization

### 6.1 Model Selection and Routing

**Tiered Model Strategy**
| Task Type | Recommended Model | Cost Factor |
|-----------|-------------------|-------------|
| Query classification | Small/Fast (GPT-4.1-nano, Haiku) | 1x |
| Simple retrieval | Medium (Sonnet) | 5x |
| Deep reasoning | Large (Opus) with extended thinking | 25x |
| Final synthesis | Large (Opus) | 25x |

**Dynamic Routing Implementation**
```python
class ModelRouter:
    def __init__(self):
        self.models = {
            'fast': 'claude-3-5-haiku-20241022',
            'balanced': 'claude-sonnet-4-20250514',
            'powerful': 'claude-opus-4-20250514'
        }

    def route(self, task: Task) -> str:
        """Select appropriate model based on task complexity"""

        if task.type == 'classification':
            return self.models['fast']

        if task.type == 'retrieval_evaluation':
            return self.models['balanced']

        if task.type in ['deep_analysis', 'synthesis', 'reasoning']:
            return self.models['powerful']

        # Default to balanced
        return self.models['balanced']
```

**Anthropic Finding**: Upgrading to Claude Sonnet 4 yields larger gains than doubling token budget on older versions.

### 6.2 Caching Strategies

**Prompt Caching**
- Cached tokens are **75% cheaper**
- Stack unchanging context (system prompts, tool definitions) at the beginning
- Only dynamic content triggers reprocessing

**Semantic Caching**
```python
class SemanticCache:
    def __init__(self, vector_db, similarity_threshold=0.92):
        self.db = vector_db
        self.threshold = similarity_threshold

    def get_or_compute(self, query: str, compute_fn: Callable) -> str:
        """Return cached result if semantically similar query exists"""

        query_embedding = self.embed(query)
        similar = self.db.find_similar(query_embedding, threshold=self.threshold)

        if similar:
            return similar.cached_response

        # Compute and cache
        result = compute_fn(query)
        self.db.store(query_embedding, result)
        return result
```

### 6.3 Tool Management

**Dynamic Toolset Pattern**
- Filter tools based on query relevance before passing to model
- Reduces token usage by up to **96%** for inputs
- Up to **160x token reduction** vs static toolsets

```python
class DynamicToolSelector:
    def __init__(self, all_tools: list[Tool]):
        self.tools = all_tools
        self.tool_embeddings = {t.name: embed(t.description) for t in all_tools}

    def select_relevant_tools(self, query: str, max_tools: int = 5) -> list[Tool]:
        """Select only relevant tools for the query"""
        query_embedding = embed(query)

        # Score tools by relevance
        scores = []
        for tool in self.tools:
            score = cosine_similarity(query_embedding, self.tool_embeddings[tool.name])
            scores.append((tool, score))

        # Return top-k most relevant
        scores.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in scores[:max_tools]]
```

**Source**: [Speakeasy - Reducing MCP Token Usage](https://www.speakeasy.com/blog/how-we-reduced-token-usage-by-100x-dynamic-toolsets-v2)

### 6.4 Context Compression

**Prompt Compression**
- Tools like LLMLingua can compress prompts by up to **20x**
- 800-token prompt → 40 tokens while preserving semantic meaning

**Output Token Control**
- Output tokens cost ~4x more than input tokens
- Control response length explicitly
- Use structured output formats

### 6.5 Cost Optimization Benchmarks

| Technique | Token Reduction | Quality Impact |
|-----------|-----------------|----------------|
| Model routing | 30-50% | Minimal |
| Prompt caching | 50-75% | None |
| Dynamic toolsets | 90-96% | None |
| Prompt compression | 80-95% | Small |
| Fine-tuning smaller models | 50-85% | Varies |

**Source**: [Kosmoy - LLM Cost Management](https://www.kosmoy.com/post/llm-cost-management-stop-burning-money-on-tokens), [Uptech - Reduce LLM Costs](https://www.uptech.team/blog/how-to-reduce-llm-costs)

---

## 7. Memory and Context Management

### 7.1 Memory Architecture Options

**Comparison of Approaches**

| Approach | Token Reduction | Reliability | Best For |
|----------|-----------------|-------------|----------|
| Observation Masking | 50%+ | High | Tool-heavy workflows |
| LLM Summarization | 60-70% | Medium | Conversational |
| Hybrid (Buffer + Summary) | 70%+ | High | Long sessions |
| MemGPT Virtual Memory | 80%+ | Medium | Extended research |

**Research Finding**: Observation masking outperforms LLM summarization in efficiency and reliability.

### 7.2 Hybrid Memory Pattern (Recommended)

```python
class HybridMemory:
    def __init__(self, max_recent_tokens: int = 8000, summary_model: str = 'haiku'):
        self.recent_buffer = []  # Verbatim recent messages
        self.summary = ""  # Compressed older context
        self.max_recent_tokens = max_recent_tokens
        self.summary_model = summary_model

    def add_message(self, message: Message):
        """Add message, compressing old messages if needed"""
        self.recent_buffer.append(message)

        # Check if buffer exceeds limit
        buffer_tokens = count_tokens(self.recent_buffer)

        if buffer_tokens > self.max_recent_tokens:
            self.compress_old_messages()

    def compress_old_messages(self):
        """Move oldest messages to summary"""
        # Keep most recent half
        cutoff = len(self.recent_buffer) // 2
        old_messages = self.recent_buffer[:cutoff]
        self.recent_buffer = self.recent_buffer[cutoff:]

        # Summarize and append to existing summary
        new_summary = self.summarize(old_messages)
        self.summary = self.merge_summaries(self.summary, new_summary)

    def get_context(self) -> str:
        """Get full context for LLM"""
        return f"""## Previous Research Summary
{self.summary}

## Recent Activity
{format_messages(self.recent_buffer)}"""
```

**Source**: [JetBrains Research - Efficient Context Management](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)

### 7.3 MemGPT-Inspired Virtual Memory

**Hierarchical Memory Tiers**
1. **Main Context** (working memory): Current task focus
2. **Archival Memory**: Long-term storage, searchable
3. **Recall Memory**: Recently accessed, quick retrieval

**Memory Management Actions** (exposed as tools)
```python
memory_tools = [
    Tool(
        name="store_to_archive",
        description="Store important finding to long-term memory",
        parameters={"content": str, "tags": list[str]}
    ),
    Tool(
        name="search_archive",
        description="Search archival memory for relevant past findings",
        parameters={"query": str}
    ),
    Tool(
        name="summarize_and_archive",
        description="Summarize current context and archive before context reset",
        parameters={"summary_prompt": str}
    )
]
```

**Source**: [MemGPT Paper](https://arxiv.org/abs/2310.08560)

### 7.4 External Memory Storage

**Anthropic's Approach**
- Store plan details in external memory when context approaches 200,000 tokens
- Subagent outputs stored to filesystem to minimize information loss

**Implementation Pattern**
```python
class ExternalMemory:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.index = {}  # In-memory index for quick lookup

    def store_finding(self, finding: Finding) -> str:
        """Store finding to disk, return reference ID"""
        finding_id = generate_id()
        file_path = self.storage_path / f"{finding_id}.json"

        with open(file_path, 'w') as f:
            json.dump(finding.to_dict(), f)

        # Index for retrieval
        self.index[finding_id] = {
            'summary': finding.summary,
            'tags': finding.tags,
            'timestamp': finding.timestamp
        }

        return finding_id

    def get_context_summary(self, max_findings: int = 10) -> str:
        """Get summary of stored findings for context"""
        summaries = []
        for fid, meta in list(self.index.items())[-max_findings:]:
            summaries.append(f"[{fid}] {meta['summary']}")

        return "Previous findings:\n" + "\n".join(summaries)

    def retrieve_finding(self, finding_id: str) -> Finding:
        """Retrieve full finding from disk"""
        file_path = self.storage_path / f"{finding_id}.json"
        with open(file_path, 'r') as f:
            return Finding.from_dict(json.load(f))
```

### 7.5 Context Window Strategies

**Recommendations by Use Case**

| Research Session Length | Strategy |
|------------------------|----------|
| < 20 exchanges | Full context, no compression |
| 20-50 exchanges | Hybrid buffer + summary |
| 50-100 exchanges | Aggressive summarization + external storage |
| 100+ exchanges | MemGPT-style hierarchical memory |

**Key Finding**: Long-context LLMs demonstrate significant hallucinations. RAG offers a balanced compromise, combining accuracy of short-context LLMs with comprehension of wide-context LLMs.

---

## 8. Recommended Architecture for Claude-Researcher

### 8.1 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DIRECTOR (Claude Opus)                        │
│  • Query analysis and complexity assessment                      │
│  • Research plan generation (user-reviewable)                    │
│  • Effort scaling (1-10+ managers based on complexity)           │
│  • Extended thinking for strategic planning                      │
└─────────────────────────────────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   MANAGER 1     │  │   MANAGER 2     │  │   MANAGER N     │
│ (Claude Opus)   │  │ (Claude Opus)   │  │ (Claude Opus)   │
│ Extended Think  │  │ Extended Think  │  │ Extended Think  │
│ Aspect: [X]     │  │ Aspect: [Y]     │  │ Aspect: [Z]     │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    ▼         ▼          ▼         ▼          ▼         ▼
┌───────┐ ┌───────┐  ┌───────┐ ┌───────┐  ┌───────┐ ┌───────┐
│Intern1│ │Intern2│  │Intern3│ │Intern4│  │Intern5│ │Intern6│
│Sonnet │ │Sonnet │  │Sonnet │ │Sonnet │  │Sonnet │ │Sonnet │
└───┬───┘ └───┬───┘  └───┬───┘ └───┬───┘  └───┬───┘ └───┬───┘
    │         │          │         │          │         │
    └────┬────┘          └────┬────┘          └────┬────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DIVERSE SEARCH LAYER                          │
│  • Web Search (Google, Bing, DuckDuckGo)                        │
│  • Academic (Semantic Scholar, PubMed, arXiv)                   │
│  • News APIs                                                     │
│  • Domain-specific databases                                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  KNOWLEDGE SYNTHESIS LAYER                       │
│  • Knowledge graph construction                                  │
│  • Contradiction detection and resolution                        │
│  • Cross-source triangulation                                    │
│  • Credibility scoring                                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 DEEP REPORT WRITER (Claude Opus)                 │
│  • Narrative synthesis                                           │
│  • Theme identification                                          │
│  • Gap analysis                                                  │
│  • Citation integration                                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FINAL REPORT                               │
│  Executive Summary | Thematic Sections | Analysis |              │
│  Conclusions | References | Confidence Scores                    │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Key Implementation Recommendations

1. **Adopt Anthropic's Subagent Pattern**
   - Director spawns 3-5 Managers in parallel
   - Each Manager coordinates 2-3 Interns
   - Use asyncio.gather() for parallelization

2. **Implement Iterative Search Refinement**
   - Self-RAG loop with gap identification
   - SEA (Structured Evidence Assessment) checklist
   - Maximum 5 refinement iterations per subtask

3. **Add Credibility Scoring**
   - Score each source on 6-signal framework
   - Weight findings by source credibility in synthesis
   - Flag low-confidence claims for additional verification

4. **Build Dynamic Mind Map**
   - Organize findings hierarchically as they arrive
   - Use for gap identification and research steering
   - Include in final report as visual summary

5. **Implement Cost Controls**
   - Model routing: Haiku for classification, Sonnet for search, Opus for reasoning
   - Dynamic toolset selection
   - Aggressive caching with semantic similarity

6. **Add Memory Management**
   - Hybrid buffer + summary for sessions under 50 exchanges
   - External filesystem storage for findings
   - Summarize before context window fills

---

## References

### Multi-Agent Architectures
- [Anthropic Engineering - Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [GPT Researcher Official](https://gptr.dev)
- [Stanford STORM Project](https://storm-project.stanford.edu/research/storm/)
- [Gemini Deep Research](https://gemini.google/overview/deep-research/)
- [Vespa.ai - Perplexity Case Study](https://vespa.ai/perplexity/)

### Search and Retrieval
- [PMC - Semantic Query Expansion Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC11935759/)
- [FAIR-RAG Paper](https://arxiv.org/html/2510.22344v1)
- [Agentic RAG Survey](https://arxiv.org/html/2501.09136v3)

### Source Credibility
- [Sourcely - Automated Credibility Scoring](https://www.sourcely.net/resources/what-is-automated-source-credibility-scoring)
- [Veracity Open-Source Fact-Checking](https://arxiv.org/html/2506.15794v1)
- [PMC - Methodological Triangulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9714985/)

### Knowledge Synthesis
- [KGGen Paper](https://arxiv.org/html/2502.09956v1)
- [LLM-empowered KG Construction Survey](https://arxiv.org/html/2510.20345v1)
- [Clinical KG with Multi-LLMs](https://arxiv.org/html/2601.01844)

### Report Quality
- [CHI 2024 - Data Storytelling](https://dl.acm.org/doi/10.1145/3613904.3643022)
- [Kittelson - Storytelling in Technical Reports](https://www.kittelson.com/ideas/the-plot-thickens-elevating-technical-reports-through-storytelling-techniques/)

### Cost Optimization
- [Kosmoy - LLM Cost Management](https://www.kosmoy.com/post/llm-cost-management-stop-burning-money-on-tokens)
- [Speakeasy - Token Usage Reduction](https://www.speakeasy.com/blog/how-we-reduced-token-usage-by-100x-dynamic-toolsets-v2)
- [Uptech - Reduce LLM Costs](https://www.uptech.team/blog/how-to-reduce-llm-costs)

### Memory Management
- [JetBrains Research - Efficient Context Management](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
- [MemGPT Paper](https://arxiv.org/abs/2310.08560)
- [Claude Extended Thinking Documentation](https://docs.claude.com/en/docs/build-with-claude/extended-thinking)

### LangGraph and Orchestration
- [LangGraph Multi-Agent Workflows](https://www.blog.langchain.com/langgraph-multi-agent-workflows/)
- [langgraph-supervisor Package](https://github.com/langchain-ai/langgraph-supervisor-py)

---

## 9. Real-Time Self-Building Knowledge Graph

*Research compiled: January 2026*

This section details how to implement a real-time, self-building knowledge graph for multi-agent research systems. The graph builds incrementally AS findings stream in (not as a post-processing step), enables gap detection to inform search strategies, detects contradictions, tracks entity relationships across sources, and is queryable by the Manager agent.

### 9.1 Real-Time Knowledge Graph Construction Patterns

**The Incremental Construction Paradigm**

Traditional knowledge graph construction processes all documents in batch. For a research agent, we need incremental construction where new information is integrated without reprocessing the entire graph. This approach supports:
- Updates, deletions, and schema evolution as data streams in
- Efficient change detection
- Real-time querying during the research process

**Key Frameworks and Approaches**

| Framework | Approach | Key Feature | Performance |
|-----------|----------|-------------|-------------|
| [iText2KG](https://github.com/AuvaLab/itext2kg) | Four-module pipeline | Async/await for non-blocking LLM I/O | Incremental entity resolution |
| [KGGen](https://arxiv.org/html/2502.09956v1) | Two-step extraction + clustering | LLM-as-Judge validation | 66% accuracy (vs 48% GraphRAG) |
| [Graphiti](https://github.com/getzep/graphiti) | Real-time agent memory | Temporal modeling | High concurrency support |
| [ATOM](https://github.com/AuvaLab/itext2kg) | Parallel atomic TKG construction | 93.8% latency reduction | Dual-time modeling |

**Recommended Architecture for Claude-Researcher**

```
┌────────────────────────────────────────────────────────────────┐
│                    FINDINGS STREAM                              │
│   (Interns report findings as they complete searches)           │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│               DOCUMENT DISTILLER (Async)                        │
│  • Chunk findings into semantic blocks                          │
│  • Extract atomic facts (<400 tokens each)                      │
│  • Add source attribution and timestamps                        │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│           ENTITY EXTRACTOR (Async, Parallel)                    │
│  • Extract entities from atomic facts                           │
│  • Match against global entity set (cosine similarity ≥ 0.7)    │
│  • Coreference resolution for entity unification                │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│           RELATION EXTRACTOR (Async, Parallel)                  │
│  • Extract (subject, predicate, object) triples                 │
│  • Use resolved entities as context                             │
│  • Normalize predicates to canonical forms                      │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│               GRAPH INTEGRATOR                                  │
│  • Merge new triples with existing graph                        │
│  • Detect contradictions                                        │
│  • Update betweenness centrality for gap detection              │
└────────────────────────────────────────────────────────────────┘
```

**Implementation Pattern: Async Incremental Processing**

```python
import asyncio
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Entity:
    id: str
    name: str
    entity_type: str
    embedding: Optional[np.ndarray] = None
    aliases: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class Relation:
    id: str
    subject_id: str
    predicate: str
    object_id: str
    source_id: str
    confidence: float = 1.0
    timestamp: Optional[str] = None

@dataclass
class Finding:
    id: str
    content: str
    source_url: str
    source_title: str
    timestamp: str
    credibility_score: float

class IncrementalKnowledgeGraph:
    """Real-time knowledge graph that builds as findings stream in."""

    def __init__(self, llm, embeddings_model, similarity_threshold: float = 0.7):
        self.llm = llm
        self.embeddings = embeddings_model
        self.similarity_threshold = similarity_threshold

        # Core graph storage
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.findings: dict[str, Finding] = {}

        # Indexes for fast lookup
        self.entity_embeddings: dict[str, np.ndarray] = {}
        self.entity_by_type: dict[str, list[str]] = {}
        self.relations_by_subject: dict[str, list[Relation]] = {}
        self.relations_by_object: dict[str, list[Relation]] = {}

        # Contradiction tracking
        self.contradictions: list[tuple[Relation, Relation]] = []

        # Processing queue for async handling
        self._processing_lock = asyncio.Lock()

    async def add_finding(self, finding: Finding) -> dict:
        """
        Process a new finding and integrate it into the knowledge graph.
        Returns extracted entities and relations.
        """
        async with self._processing_lock:
            # Store the finding
            self.findings[finding.id] = finding

            # Step 1: Extract atomic facts
            atomic_facts = await self._extract_atomic_facts(finding)

            # Step 2: Extract entities (parallel processing)
            entity_tasks = [
                self._extract_entities(fact, finding.id)
                for fact in atomic_facts
            ]
            entity_results = await asyncio.gather(*entity_tasks)

            # Step 3: Resolve and merge entities
            new_entities = []
            for entities in entity_results:
                for entity in entities:
                    resolved = await self._resolve_entity(entity)
                    new_entities.append(resolved)

            # Step 4: Extract relations using resolved entities
            relation_tasks = [
                self._extract_relations(fact, new_entities, finding.id)
                for fact in atomic_facts
            ]
            relation_results = await asyncio.gather(*relation_tasks)

            # Step 5: Integrate relations and check for contradictions
            new_relations = []
            for relations in relation_results:
                for relation in relations:
                    contradiction = self._check_contradiction(relation)
                    if contradiction:
                        self.contradictions.append((relation, contradiction))
                    self.relations.append(relation)
                    new_relations.append(relation)

                    # Update indexes
                    self.relations_by_subject.setdefault(
                        relation.subject_id, []
                    ).append(relation)
                    self.relations_by_object.setdefault(
                        relation.object_id, []
                    ).append(relation)

            return {
                'entities': new_entities,
                'relations': new_relations,
                'contradictions_found': len([
                    c for c in self.contradictions
                    if c[0] in new_relations or c[1] in new_relations
                ])
            }

    async def _extract_atomic_facts(self, finding: Finding) -> list[str]:
        """Break finding into minimal, self-contained atomic facts."""
        prompt = f"""Extract atomic facts from this research finding.
Each atomic fact should be:
- Self-contained (understandable without context)
- Minimal (single piece of information)
- Factual (not opinion unless attributed)

Finding: {finding.content}

Return as JSON array of strings."""

        response = await self.llm.agenerate(prompt)
        return self._parse_json_array(response)

    async def _extract_entities(
        self, text: str, source_id: str
    ) -> list[Entity]:
        """Extract entities from text using LLM."""
        prompt = f"""Extract key entities from this text.

Text: {text}

For each entity provide:
- name: The canonical name (singular, consistent capitalization)
- type: One of [CONCEPT, PERSON, ORGANIZATION, CLAIM, TECHNOLOGY,
        METHOD, METRIC, LOCATION, DATE, SOURCE]
- aliases: Other names/mentions that refer to this entity

Return as JSON array."""

        response = await self.llm.agenerate(prompt)
        entities_data = self._parse_json_array(response)

        entities = []
        for e in entities_data:
            entity = Entity(
                id=self._generate_id(),
                name=e['name'],
                entity_type=e['type'],
                aliases=e.get('aliases', []),
                sources=[source_id]
            )
            # Generate embedding for similarity matching
            entity.embedding = await self.embeddings.aembed(entity.name)
            entities.append(entity)

        return entities

    async def _resolve_entity(self, entity: Entity) -> Entity:
        """
        Resolve entity against existing global set.
        Uses cosine similarity with threshold for matching.
        """
        if not self.entity_embeddings:
            # First entity - add directly
            self.entities[entity.id] = entity
            self.entity_embeddings[entity.id] = entity.embedding
            self.entity_by_type.setdefault(entity.entity_type, []).append(entity.id)
            return entity

        # Find best matching existing entity
        best_match_id = None
        best_similarity = 0.0

        for existing_id, existing_embedding in self.entity_embeddings.items():
            existing = self.entities[existing_id]
            # Only match within same type
            if existing.entity_type != entity.entity_type:
                continue

            similarity = self._cosine_similarity(entity.embedding, existing_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = existing_id

        if best_similarity >= self.similarity_threshold:
            # Merge with existing entity
            existing = self.entities[best_match_id]
            existing.aliases.extend([entity.name] + entity.aliases)
            existing.aliases = list(set(existing.aliases))
            existing.sources.extend(entity.sources)
            existing.sources = list(set(existing.sources))
            return existing
        else:
            # Add as new entity
            self.entities[entity.id] = entity
            self.entity_embeddings[entity.id] = entity.embedding
            self.entity_by_type.setdefault(entity.entity_type, []).append(entity.id)
            return entity

    async def _extract_relations(
        self, text: str, entities: list[Entity], source_id: str
    ) -> list[Relation]:
        """Extract relations using resolved entities as context."""
        entity_context = "\n".join([
            f"- {e.name} ({e.entity_type})" for e in entities
        ])

        prompt = f"""Extract relationships from this text.

Text: {text}

Known entities:
{entity_context}

For each relationship provide:
- subject: The subject entity name (must be from known entities)
- predicate: The relationship (1-3 words max, e.g., "causes", "is part of")
- object: The object entity name (must be from known entities)
- confidence: 0.0-1.0 confidence score

Return as JSON array."""

        response = await self.llm.agenerate(prompt)
        relations_data = self._parse_json_array(response)

        # Map entity names to IDs
        name_to_id = {e.name.lower(): e.id for e in entities}
        for e in entities:
            for alias in e.aliases:
                name_to_id[alias.lower()] = e.id

        relations = []
        for r in relations_data:
            subject_id = name_to_id.get(r['subject'].lower())
            object_id = name_to_id.get(r['object'].lower())

            if subject_id and object_id:
                relation = Relation(
                    id=self._generate_id(),
                    subject_id=subject_id,
                    predicate=self._normalize_predicate(r['predicate']),
                    object_id=object_id,
                    source_id=source_id,
                    confidence=r.get('confidence', 0.8)
                )
                relations.append(relation)

        return relations

    def _check_contradiction(self, new_relation: Relation) -> Optional[Relation]:
        """Check if new relation contradicts existing relations."""
        # Look for relations with same subject and object but different predicate
        for existing in self.relations:
            if (existing.subject_id == new_relation.subject_id and
                existing.object_id == new_relation.object_id):

                # Check if predicates are contradictory
                if self._predicates_contradict(
                    existing.predicate, new_relation.predicate
                ):
                    return existing

        return None

    def _predicates_contradict(self, pred1: str, pred2: str) -> bool:
        """Determine if two predicates are contradictory."""
        # Simple contradiction patterns
        contradiction_pairs = [
            ('increases', 'decreases'),
            ('causes', 'prevents'),
            ('supports', 'contradicts'),
            ('is', 'is not'),
            ('has', 'lacks'),
            ('enables', 'blocks'),
            ('improves', 'worsens'),
        ]

        pred1_lower = pred1.lower()
        pred2_lower = pred2.lower()

        for p1, p2 in contradiction_pairs:
            if (p1 in pred1_lower and p2 in pred2_lower) or \
               (p2 in pred1_lower and p1 in pred2_lower):
                return True

        return False

    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate to canonical form."""
        # Lowercase, strip, limit to 3 words
        words = predicate.lower().strip().split()[:3]
        return ' '.join(words)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _generate_id(self) -> str:
        """Generate unique ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def _parse_json_array(self, text: str) -> list:
        """Parse JSON array from LLM response."""
        import json
        import re
        # Extract JSON from potential markdown code blocks
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
```

**Source**: [iText2KG](https://arxiv.org/html/2409.03284v1), [Graphiti](https://github.com/getzep/graphiti), [KGGen](https://arxiv.org/html/2502.09956v1)

---

### 9.2 Knowledge Graph Schema for Research

**Recommended Entity Types**

For a research-focused knowledge graph, the schema should capture both domain knowledge and meta-knowledge about the research process:

```python
ENTITY_TYPES = {
    # Domain Knowledge Entities
    'CONCEPT': 'Abstract idea or topic being researched',
    'CLAIM': 'Specific assertion that can be true or false',
    'EVIDENCE': 'Data, study, or observation supporting/refuting claims',
    'METHOD': 'Technique, algorithm, or approach',
    'METRIC': 'Quantitative measurement or statistic',
    'TECHNOLOGY': 'Tool, system, or implementation',

    # Meta-Knowledge Entities
    'SOURCE': 'Document, paper, or website providing information',
    'AUTHOR': 'Person or organization authoring a source',
    'QUOTE': 'Direct quotation from a source',

    # Standard Named Entities
    'PERSON': 'Individual mentioned in research',
    'ORGANIZATION': 'Company, institution, or group',
    'LOCATION': 'Geographic location',
    'DATE': 'Temporal reference',
}

RELATION_TYPES = {
    # Epistemic Relations (knowledge about claims)
    'supports': 'Evidence supports a claim',
    'contradicts': 'Evidence contradicts a claim',
    'qualifies': 'Evidence adds conditions to a claim',
    'cites': 'Source cites another source',

    # Semantic Relations (domain knowledge)
    'is_a': 'Taxonomic relationship',
    'part_of': 'Compositional relationship',
    'causes': 'Causal relationship',
    'correlates_with': 'Statistical relationship',
    'enables': 'Functional dependency',
    'implements': 'Realization relationship',

    # Comparative Relations
    'outperforms': 'Performance comparison',
    'similar_to': 'Similarity relationship',
    'alternative_to': 'Alternative approaches',

    # Attribution Relations
    'authored_by': 'Authorship',
    'published_in': 'Publication venue',
    'mentioned_in': 'Reference in source',
}
```

**Schema Definition with Constraints**

```python
@dataclass
class KnowledgeGraphSchema:
    """Schema definition for research knowledge graph."""

    # Entity type definitions with validation
    entity_types: dict[str, dict] = field(default_factory=lambda: {
        'CLAIM': {
            'required_properties': ['statement', 'confidence'],
            'optional_properties': ['scope', 'conditions'],
        },
        'EVIDENCE': {
            'required_properties': ['description', 'evidence_type'],
            'optional_properties': ['sample_size', 'methodology'],
        },
        'SOURCE': {
            'required_properties': ['url', 'title'],
            'optional_properties': ['author', 'date', 'credibility_score'],
        },
    })

    # Relation constraints
    relation_constraints: dict[str, dict] = field(default_factory=lambda: {
        'supports': {
            'valid_subject_types': ['EVIDENCE', 'SOURCE'],
            'valid_object_types': ['CLAIM'],
        },
        'contradicts': {
            'valid_subject_types': ['EVIDENCE', 'SOURCE', 'CLAIM'],
            'valid_object_types': ['CLAIM'],
        },
        'causes': {
            'valid_subject_types': ['CONCEPT', 'METHOD', 'TECHNOLOGY'],
            'valid_object_types': ['CONCEPT', 'METRIC'],
        },
    })

    def validate_relation(
        self,
        subject_type: str,
        predicate: str,
        object_type: str
    ) -> bool:
        """Validate that a relation follows schema constraints."""
        if predicate not in self.relation_constraints:
            return True  # Allow unconstrained relations

        constraints = self.relation_constraints[predicate]
        return (
            subject_type in constraints['valid_subject_types'] and
            object_type in constraints['valid_object_types']
        )
```

**Source**: [GraphRAG Schema Research](https://drops.dagstuhl.de/entities/document/10.4230/TGDK.3.2.3), [IBM GraphRAG](https://www.ibm.com/think/topics/graphrag)

---

### 9.3 LLM-Based Entity and Relation Extraction

**Best Prompts for Extraction**

Research shows that a two-step extraction approach (entities first, then relations) outperforms single-pass extraction by ensuring consistency between identified entities and their relationships.

**Entity Extraction Prompt**

```python
ENTITY_EXTRACTION_PROMPT = """Extract key entities from the following research finding.

## Finding
{text}

## Entity Types to Extract
- CONCEPT: Abstract ideas or topics (e.g., "machine learning", "knowledge graphs")
- CLAIM: Specific assertions that can be verified (e.g., "GPT-4 outperforms GPT-3")
- EVIDENCE: Supporting data, studies, or observations
- METHOD: Techniques, algorithms, or approaches
- METRIC: Quantitative measurements (e.g., "95% accuracy", "2x speedup")
- TECHNOLOGY: Tools, systems, or implementations
- ORGANIZATION: Companies, institutions, groups
- PERSON: Individuals mentioned

## Rules
1. Use canonical names (singular form, consistent capitalization)
2. Include confidence scores (0.0-1.0) based on how clearly stated
3. Capture aliases (different ways the same entity is mentioned)
4. Do not create entities for vague references

## Output Format (JSON)
[
  {{
    "name": "Entity canonical name",
    "type": "ENTITY_TYPE",
    "confidence": 0.95,
    "aliases": ["alias1", "alias2"],
    "context": "Brief context from text"
  }}
]

Extract entities:"""

RELATION_EXTRACTION_PROMPT = """Extract relationships between these entities from the text.

## Text
{text}

## Entities (use these exact names)
{entities_list}

## Relationship Types
- supports/contradicts/qualifies: Epistemic (evidence <-> claims)
- is_a/part_of: Taxonomic relationships
- causes/correlates_with: Causal/correlational
- implements/enables: Functional dependencies
- outperforms/similar_to: Comparative

## Rules
1. Subject and object MUST be from the entities list above
2. Predicates should be 1-3 words maximum
3. Include confidence based on how explicitly stated
4. Capture direction correctly (subject -> predicate -> object)

## Output Format (JSON)
[
  {{
    "subject": "Entity name",
    "predicate": "relationship",
    "object": "Entity name",
    "confidence": 0.9,
    "evidence": "Quote from text supporting this"
  }}
]

Extract relationships:"""
```

**Extract-Critique-Refine (ECR) Pipeline**

The ECR pattern achieves +16% improvement in triple extraction:

```python
class ECRExtractor:
    """Extract-Critique-Refine pipeline for high-quality extraction."""

    def __init__(self, llm):
        self.llm = llm

    async def extract_with_refinement(
        self, text: str, max_iterations: int = 2
    ) -> dict:
        """Extract entities and relations with self-critique."""

        # Step 1: Initial extraction
        entities = await self._extract_entities(text)
        relations = await self._extract_relations(text, entities)

        for _ in range(max_iterations):
            # Step 2: Critique
            critique = await self._critique_extraction(
                text, entities, relations
            )

            if critique['is_complete'] and critique['is_accurate']:
                break

            # Step 3: Refine based on critique
            if not critique['is_complete']:
                missing = await self._extract_missing(
                    text, entities, critique['missing_aspects']
                )
                entities.extend(missing['entities'])
                relations.extend(missing['relations'])

            if not critique['is_accurate']:
                entities, relations = await self._fix_errors(
                    entities, relations, critique['errors']
                )

        return {'entities': entities, 'relations': relations}

    async def _critique_extraction(
        self, text: str, entities: list, relations: list
    ) -> dict:
        """LLM critiques its own extraction."""
        prompt = f"""Review this knowledge extraction for completeness and accuracy.

## Original Text
{text}

## Extracted Entities
{self._format_entities(entities)}

## Extracted Relations
{self._format_relations(relations)}

## Critique Checklist
1. Are all key concepts captured?
2. Are all claims and evidence identified?
3. Are entity types correct?
4. Are relationships accurate and complete?
5. Are there any hallucinated entities/relations not in the text?

## Output (JSON)
{{
  "is_complete": true/false,
  "is_accurate": true/false,
  "missing_aspects": ["what's missing"],
  "errors": ["what's wrong"],
  "suggestions": ["how to improve"]
}}"""

        response = await self.llm.agenerate(prompt)
        return self._parse_json(response)
```

**Coreference Resolution for Entity Unification**

The LINK-KG framework reduces node duplication by 45% through type-specific coreference resolution:

```python
class CoreferenceResolver:
    """Resolve entity mentions to canonical forms."""

    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        # Type-specific prompt cache (LINK-KG pattern)
        self.prompt_cache: dict[str, list[str]] = {}

    async def resolve_mentions(
        self, text: str, entity_type: str
    ) -> dict[str, str]:
        """
        Resolve all mentions of a specific entity type.
        Returns mapping of mention -> canonical name.
        """
        # Get existing entities of this type
        existing = self.prompt_cache.get(entity_type, [])

        prompt = f"""Identify all mentions of {entity_type} entities in this text
and map them to canonical names.

## Text
{text}

## Known Entities (if a mention refers to one of these, use that name)
{chr(10).join(f'- {e}' for e in existing) if existing else 'None yet'}

## Rules
1. Different surface forms of the same entity should map to ONE canonical name
2. Handle: abbreviations (AI -> Artificial Intelligence)
3. Handle: pronouns (it, they -> the entity they refer to)
4. Handle: role shifts (the company, the researchers -> specific entity)
5. Handle: plural/singular variations

## Output (JSON)
{{
  "mention in text": "canonical name",
  "another mention": "same or different canonical name"
}}"""

        response = await self.llm.agenerate(prompt)
        resolution_map = self._parse_json(response)

        # Update prompt cache with new canonical names
        new_canonicals = set(resolution_map.values()) - set(existing)
        self.prompt_cache.setdefault(entity_type, []).extend(new_canonicals)

        return resolution_map
```

**Source**: [LINK-KG](https://arxiv.org/abs/2510.26486), [CORE-KG](https://arxiv.org/html/2510.26512v1), [Neo4j Best Practices](https://neo4j.com/blog/developer/knowledge-graph-extraction-challenges/)

---

### 9.4 Graph Storage Options

**Comparison for Claude-Researcher**

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **In-Memory (dict/NetworkX)** | Fast, simple, no dependencies | Lost on restart, memory limits | Prototyping, small graphs |
| **SQLite** | Persistent, ACID, already used | No native graph queries | Hybrid with in-memory index |
| **NetworkX + SQLite Hybrid** | Best of both worlds | Complexity | Production recommendation |
| **FalkorDBLite** | Embedded graph DB, Cypher queries | New dependency | If need native graph queries |

**Recommended: Hybrid NetworkX + SQLite Approach**

```python
import sqlite3
import pickle
from pathlib import Path
import networkx as nx
from typing import Optional
import json

class HybridKnowledgeGraphStore:
    """
    Hybrid storage: NetworkX for in-memory graph algorithms,
    SQLite for persistence and metadata queries.
    """

    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = Path(db_path)
        self.graph = nx.DiGraph()  # In-memory graph for algorithms
        self._init_db()
        self._load_from_db()

    def _init_db(self):
        """Initialize SQLite schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                properties TEXT,  -- JSON
                embedding BLOB,   -- Serialized numpy array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Relations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_id TEXT NOT NULL,
                source_id TEXT,
                confidence REAL DEFAULT 1.0,
                properties TEXT,  -- JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES entities(id),
                FOREIGN KEY (object_id) REFERENCES entities(id)
            )
        """)

        # Findings/Sources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_url TEXT,
                source_title TEXT,
                credibility_score REAL,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Contradictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contradictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relation1_id TEXT NOT NULL,
                relation2_id TEXT NOT NULL,
                resolution TEXT,  -- How it was resolved
                resolved_at TIMESTAMP,
                FOREIGN KEY (relation1_id) REFERENCES relations(id),
                FOREIGN KEY (relation2_id) REFERENCES relations(id)
            )
        """)

        # Indexes for fast queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_predicate ON relations(predicate)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_subject ON relations(subject_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_object ON relations(object_id)"
        )

        conn.commit()
        conn.close()

    def _load_from_db(self):
        """Load graph from SQLite into NetworkX."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load entities as nodes
        cursor.execute("SELECT id, name, entity_type, properties FROM entities")
        for row in cursor.fetchall():
            entity_id, name, entity_type, properties = row
            props = json.loads(properties) if properties else {}
            self.graph.add_node(
                entity_id,
                name=name,
                entity_type=entity_type,
                **props
            )

        # Load relations as edges
        cursor.execute("""
            SELECT id, subject_id, predicate, object_id, confidence, properties
            FROM relations
        """)
        for row in cursor.fetchall():
            rel_id, subj, pred, obj, conf, properties = row
            props = json.loads(properties) if properties else {}
            self.graph.add_edge(
                subj, obj,
                relation_id=rel_id,
                predicate=pred,
                confidence=conf,
                **props
            )

        conn.close()

    def add_entity(self, entity: Entity) -> str:
        """Add entity to both NetworkX and SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Serialize embedding
        embedding_blob = pickle.dumps(entity.embedding) if entity.embedding is not None else None
        properties = json.dumps({
            'aliases': entity.aliases,
            'sources': entity.sources,
            'confidence': entity.confidence
        })

        cursor.execute("""
            INSERT OR REPLACE INTO entities (id, name, entity_type, properties, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (entity.id, entity.name, entity.entity_type, properties, embedding_blob))

        conn.commit()
        conn.close()

        # Add to NetworkX
        self.graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            aliases=entity.aliases,
            sources=entity.sources,
            confidence=entity.confidence
        )

        return entity.id

    def add_relation(self, relation: Relation) -> str:
        """Add relation to both NetworkX and SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        properties = json.dumps({
            'timestamp': relation.timestamp
        })

        cursor.execute("""
            INSERT INTO relations
            (id, subject_id, predicate, object_id, source_id, confidence, properties)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            relation.id, relation.subject_id, relation.predicate,
            relation.object_id, relation.source_id, relation.confidence,
            properties
        ))

        conn.commit()
        conn.close()

        # Add to NetworkX
        self.graph.add_edge(
            relation.subject_id,
            relation.object_id,
            relation_id=relation.id,
            predicate=relation.predicate,
            confidence=relation.confidence,
            source_id=relation.source_id
        )

        return relation.id

    def query_by_entity_type(self, entity_type: str) -> list[dict]:
        """Query entities by type using SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, name, properties FROM entities WHERE entity_type = ?",
            (entity_type,)
        )

        results = []
        for row in cursor.fetchall():
            entity_id, name, properties = row
            props = json.loads(properties) if properties else {}
            results.append({
                'id': entity_id,
                'name': name,
                'type': entity_type,
                **props
            })

        conn.close()
        return results

    def get_entity_relations(self, entity_id: str) -> dict:
        """Get all relations for an entity using NetworkX."""
        if entity_id not in self.graph:
            return {'outgoing': [], 'incoming': []}

        outgoing = []
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            outgoing.append({
                'predicate': data['predicate'],
                'target_id': target,
                'target_name': self.graph.nodes[target].get('name', target),
                'confidence': data.get('confidence', 1.0)
            })

        incoming = []
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            incoming.append({
                'predicate': data['predicate'],
                'source_id': source,
                'source_name': self.graph.nodes[source].get('name', source),
                'confidence': data.get('confidence', 1.0)
            })

        return {'outgoing': outgoing, 'incoming': incoming}

    def save_graph_pickle(self, path: str):
        """Save NetworkX graph to pickle for fast reload."""
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)

    def load_graph_pickle(self, path: str):
        """Load NetworkX graph from pickle."""
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
```

**Source**: [NetworkX + SQLite Hybrid Pattern](https://danielkliewer.com/blog/2025-10-19-building-a-local-llm-powered-knowledge-graph), [FalkorDBLite](https://www.falkordb.com/blog/falkordblite-embedded-python-graph-database/)

---

### 9.5 Querying the Graph for Research Steering

The Manager agent needs to query the knowledge graph to understand current knowledge state and identify gaps. This enables intelligent research steering.

**Manager Query Interface**

```python
class ManagerQueryInterface:
    """
    Interface for Manager agent to query knowledge graph.
    Answers questions like:
    - "What do I know about X?"
    - "What's missing?"
    - "What contradictions exist?"
    - "What are the key concepts?"
    """

    def __init__(self, kg_store: HybridKnowledgeGraphStore):
        self.store = kg_store

    def what_do_i_know_about(self, topic: str) -> dict:
        """
        Get all knowledge related to a topic.
        Returns entities, claims, evidence, and relationships.
        """
        # Find entities matching the topic
        matching_entities = []
        for node_id, data in self.store.graph.nodes(data=True):
            name = data.get('name', '').lower()
            aliases = data.get('aliases', [])
            if topic.lower() in name or any(
                topic.lower() in a.lower() for a in aliases
            ):
                matching_entities.append(node_id)

        if not matching_entities:
            return {
                'found': False,
                'message': f"No knowledge about '{topic}' found yet."
            }

        # Gather all related information
        knowledge = {
            'found': True,
            'entities': [],
            'claims': [],
            'evidence': [],
            'relations': [],
            'sources': set()
        }

        for entity_id in matching_entities:
            data = self.store.graph.nodes[entity_id]
            entity_info = {
                'id': entity_id,
                'name': data.get('name'),
                'type': data.get('entity_type'),
                'confidence': data.get('confidence', 1.0)
            }

            if data.get('entity_type') == 'CLAIM':
                knowledge['claims'].append(entity_info)
            elif data.get('entity_type') == 'EVIDENCE':
                knowledge['evidence'].append(entity_info)
            else:
                knowledge['entities'].append(entity_info)

            # Get relations
            relations = self.store.get_entity_relations(entity_id)
            knowledge['relations'].extend(relations['outgoing'])
            knowledge['relations'].extend(relations['incoming'])

            # Track sources
            knowledge['sources'].update(data.get('sources', []))

        knowledge['sources'] = list(knowledge['sources'])
        knowledge['summary'] = self._generate_summary(knowledge)

        return knowledge

    def identify_gaps(self) -> list[dict]:
        """
        Identify knowledge gaps using graph structure analysis.
        Uses betweenness centrality and structural hole detection.
        """
        gaps = []

        # 1. Find concepts with few supporting claims/evidence
        claims = self.store.query_by_entity_type('CLAIM')
        for claim in claims:
            # Count supporting evidence
            evidence_count = sum(
                1 for _, _, d in self.store.graph.in_edges(claim['id'], data=True)
                if d.get('predicate') in ['supports', 'evidence_for']
            )

            if evidence_count < 2:
                gaps.append({
                    'type': 'insufficient_evidence',
                    'entity': claim['name'],
                    'current_evidence': evidence_count,
                    'recommendation': f"Find more evidence for: {claim['name']}"
                })

        # 2. Detect structural holes (disconnected clusters)
        if len(self.store.graph) > 5:
            # Find weakly connected components
            components = list(nx.weakly_connected_components(self.store.graph))
            if len(components) > 1:
                gaps.append({
                    'type': 'disconnected_topics',
                    'components': len(components),
                    'recommendation': (
                        "Research connections between: " +
                        ", ".join(
                            self._get_component_label(c)
                            for c in components[:3]
                        )
                    )
                })

        # 3. Find concepts with high betweenness but low degree
        # (Important bridging concepts that need more exploration)
        if len(self.store.graph) > 3:
            betweenness = nx.betweenness_centrality(self.store.graph)
            degree = dict(self.store.graph.degree())

            for node_id, bc in betweenness.items():
                if bc > 0.3 and degree[node_id] < 4:
                    data = self.store.graph.nodes[node_id]
                    gaps.append({
                        'type': 'bridging_concept',
                        'entity': data.get('name'),
                        'betweenness': bc,
                        'degree': degree[node_id],
                        'recommendation': (
                            f"'{data.get('name')}' bridges important concepts. "
                            "Explore it further."
                        )
                    })

        # 4. Find entity types with low representation
        type_counts = {}
        for _, data in self.store.graph.nodes(data=True):
            etype = data.get('entity_type', 'UNKNOWN')
            type_counts[etype] = type_counts.get(etype, 0) + 1

        expected_types = ['CLAIM', 'EVIDENCE', 'METHOD', 'METRIC']
        for etype in expected_types:
            if type_counts.get(etype, 0) < 2:
                gaps.append({
                    'type': 'missing_entity_type',
                    'entity_type': etype,
                    'current_count': type_counts.get(etype, 0),
                    'recommendation': f"Find more {etype.lower()}s in the research."
                })

        return gaps

    def get_contradictions(self) -> list[dict]:
        """Get all unresolved contradictions."""
        conn = sqlite3.connect(self.store.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT c.id, c.relation1_id, c.relation2_id,
                   r1.predicate as pred1, r2.predicate as pred2,
                   e1.name as subj_name, e2.name as obj_name
            FROM contradictions c
            JOIN relations r1 ON c.relation1_id = r1.id
            JOIN relations r2 ON c.relation2_id = r2.id
            JOIN entities e1 ON r1.subject_id = e1.id
            JOIN entities e2 ON r1.object_id = e2.id
            WHERE c.resolved_at IS NULL
        """)

        contradictions = []
        for row in cursor.fetchall():
            contradictions.append({
                'id': row[0],
                'claim1': f"{row[5]} {row[3]} {row[6]}",
                'claim2': f"{row[5]} {row[4]} {row[6]}",
                'recommendation': (
                    f"Resolve: Does '{row[5]}' {row[3]} or {row[4]} '{row[6]}'?"
                )
            })

        conn.close()
        return contradictions

    def get_key_concepts(self, top_n: int = 10) -> list[dict]:
        """Get most important concepts by centrality."""
        if len(self.store.graph) == 0:
            return []

        # Use PageRank for importance
        pagerank = nx.pagerank(self.store.graph)

        # Sort by importance
        sorted_nodes = sorted(
            pagerank.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        key_concepts = []
        for node_id, score in sorted_nodes:
            data = self.store.graph.nodes[node_id]
            key_concepts.append({
                'id': node_id,
                'name': data.get('name'),
                'type': data.get('entity_type'),
                'importance': round(score, 4),
                'connections': self.store.graph.degree(node_id)
            })

        return key_concepts

    def get_research_summary(self) -> str:
        """Generate a summary of current knowledge state for the Manager."""
        summary_parts = []

        # Stats
        num_entities = self.store.graph.number_of_nodes()
        num_relations = self.store.graph.number_of_edges()

        summary_parts.append(f"## Knowledge Graph Status")
        summary_parts.append(f"- Entities: {num_entities}")
        summary_parts.append(f"- Relations: {num_relations}")

        # Key concepts
        key_concepts = self.get_key_concepts(5)
        if key_concepts:
            summary_parts.append(f"\n## Key Concepts")
            for c in key_concepts:
                summary_parts.append(f"- {c['name']} ({c['type']}, importance: {c['importance']})")

        # Gaps
        gaps = self.identify_gaps()
        if gaps:
            summary_parts.append(f"\n## Knowledge Gaps ({len(gaps)} identified)")
            for g in gaps[:5]:
                summary_parts.append(f"- {g['recommendation']}")

        # Contradictions
        contradictions = self.get_contradictions()
        if contradictions:
            summary_parts.append(f"\n## Contradictions ({len(contradictions)} unresolved)")
            for c in contradictions[:3]:
                summary_parts.append(f"- {c['recommendation']}")

        return "\n".join(summary_parts)

    def _get_component_label(self, component: set) -> str:
        """Get a label for a graph component."""
        if not component:
            return "empty"
        sample_id = next(iter(component))
        if sample_id in self.store.graph:
            return self.store.graph.nodes[sample_id].get('name', sample_id)
        return str(sample_id)

    def _generate_summary(self, knowledge: dict) -> str:
        """Generate natural language summary of knowledge."""
        parts = []

        if knowledge['entities']:
            entity_names = [e['name'] for e in knowledge['entities'][:5]]
            parts.append(f"Related concepts: {', '.join(entity_names)}")

        if knowledge['claims']:
            parts.append(f"Claims found: {len(knowledge['claims'])}")

        if knowledge['evidence']:
            parts.append(f"Evidence pieces: {len(knowledge['evidence'])}")

        parts.append(f"From {len(knowledge['sources'])} sources")

        return ". ".join(parts) + "."
```

**Source**: [InfraNodus Gap Detection](https://infranodus.com/docs/network-analysis), [Betweenness Centrality](https://infranodus.com/use-case/text-network-analysis)

---

### 9.6 Contradiction Detection via Graph Structure

**Multi-Level Contradiction Detection**

Contradictions can occur at multiple levels:
1. **Direct predicate contradictions**: Same subject-object pair with opposing predicates
2. **Transitive contradictions**: A→B→C contradicts A→C relationship
3. **Temporal contradictions**: Claims that were true at different times
4. **Scope contradictions**: Claims true in different contexts

```python
class ContradictionDetector:
    """Detect contradictions in knowledge graph."""

    # Predicate pairs that are contradictory
    CONTRADICTORY_PREDICATES = {
        ('increases', 'decreases'),
        ('causes', 'prevents'),
        ('supports', 'contradicts'),
        ('enables', 'blocks'),
        ('improves', 'worsens'),
        ('is', 'is not'),
        ('has', 'lacks'),
        ('before', 'after'),
        ('greater than', 'less than'),
        ('outperforms', 'underperforms'),
    }

    def __init__(self, kg_store: HybridKnowledgeGraphStore, llm=None):
        self.store = kg_store
        self.llm = llm  # For semantic contradiction detection

    def detect_all_contradictions(self) -> list[dict]:
        """Run all contradiction detection methods."""
        contradictions = []

        # Level 1: Direct predicate contradictions
        contradictions.extend(self._detect_direct_contradictions())

        # Level 2: Transitive contradictions
        contradictions.extend(self._detect_transitive_contradictions())

        # Level 3: Semantic contradictions (requires LLM)
        if self.llm:
            contradictions.extend(self._detect_semantic_contradictions())

        return contradictions

    def _detect_direct_contradictions(self) -> list[dict]:
        """Find direct predicate contradictions between same entity pairs."""
        contradictions = []

        # Group edges by (subject, object) pairs
        edge_groups: dict[tuple, list] = {}
        for u, v, data in self.store.graph.edges(data=True):
            key = (u, v)
            edge_groups.setdefault(key, []).append(data)

        for (subj, obj), edges in edge_groups.items():
            if len(edges) < 2:
                continue

            predicates = [e.get('predicate', '') for e in edges]

            # Check each pair of predicates
            for i, pred1 in enumerate(predicates):
                for pred2 in predicates[i+1:]:
                    if self._are_contradictory(pred1, pred2):
                        subj_name = self.store.graph.nodes[subj].get('name', subj)
                        obj_name = self.store.graph.nodes[obj].get('name', obj)

                        contradictions.append({
                            'type': 'direct',
                            'subject': subj_name,
                            'object': obj_name,
                            'predicate1': pred1,
                            'predicate2': pred2,
                            'severity': 'high',
                            'description': (
                                f"'{subj_name}' is claimed to both "
                                f"'{pred1}' and '{pred2}' '{obj_name}'"
                            )
                        })

        return contradictions

    def _detect_transitive_contradictions(self) -> list[dict]:
        """Find contradictions through transitive relationships."""
        contradictions = []

        # Look for A->B->C paths where A->C exists with contradictory meaning
        for node_a in self.store.graph.nodes():
            # Get A's direct outgoing relations
            a_relations = {
                (v, d.get('predicate')): d
                for _, v, d in self.store.graph.out_edges(node_a, data=True)
            }

            # Get A's 2-hop relations through B
            for node_b in self.store.graph.successors(node_a):
                pred_ab = self.store.graph.edges[node_a, node_b].get('predicate')

                for node_c in self.store.graph.successors(node_b):
                    pred_bc = self.store.graph.edges[node_b, node_c].get('predicate')

                    # Check if A->C exists
                    if (node_c, ) in [(k[0],) for k in a_relations.keys()]:
                        for (target, pred_ac), data in a_relations.items():
                            if target == node_c:
                                # Check if transitive implication contradicts direct
                                implied = self._get_transitive_implication(pred_ab, pred_bc)
                                if implied and self._are_contradictory(implied, pred_ac):
                                    contradictions.append({
                                        'type': 'transitive',
                                        'path': f"{node_a} -[{pred_ab}]-> {node_b} -[{pred_bc}]-> {node_c}",
                                        'direct': f"{node_a} -[{pred_ac}]-> {node_c}",
                                        'severity': 'medium',
                                        'description': (
                                            f"Transitive path implies '{implied}' but "
                                            f"direct relation says '{pred_ac}'"
                                        )
                                    })

        return contradictions

    async def _detect_semantic_contradictions(self) -> list[dict]:
        """Use LLM to detect semantic contradictions between claims."""
        contradictions = []

        # Get all claims
        claims = [
            (node_id, data)
            for node_id, data in self.store.graph.nodes(data=True)
            if data.get('entity_type') == 'CLAIM'
        ]

        # Compare claims pairwise (expensive, so limit)
        for i, (id1, claim1) in enumerate(claims):
            for id2, claim2 in claims[i+1:]:
                # Quick filter: only check claims about similar topics
                if not self._potentially_related(claim1, claim2):
                    continue

                # Ask LLM to evaluate
                prompt = f"""Do these two claims contradict each other?

Claim 1: {claim1.get('name', '')}
Claim 2: {claim2.get('name', '')}

Consider:
- Could both be true in different contexts?
- Are they about the same thing?
- Is one a qualification of the other?

Respond with JSON:
{{
  "contradicts": true/false,
  "explanation": "why or why not",
  "confidence": 0.0-1.0
}}"""

                response = await self.llm.agenerate(prompt)
                result = self._parse_json(response)

                if result.get('contradicts') and result.get('confidence', 0) > 0.7:
                    contradictions.append({
                        'type': 'semantic',
                        'claim1': claim1.get('name'),
                        'claim2': claim2.get('name'),
                        'explanation': result.get('explanation'),
                        'confidence': result.get('confidence'),
                        'severity': 'high' if result.get('confidence', 0) > 0.9 else 'medium'
                    })

        return contradictions

    def _are_contradictory(self, pred1: str, pred2: str) -> bool:
        """Check if two predicates are contradictory."""
        p1 = pred1.lower().strip()
        p2 = pred2.lower().strip()

        for contra_pair in self.CONTRADICTORY_PREDICATES:
            if (contra_pair[0] in p1 and contra_pair[1] in p2) or \
               (contra_pair[1] in p1 and contra_pair[0] in p2):
                return True

        return False

    def _get_transitive_implication(self, pred1: str, pred2: str) -> Optional[str]:
        """Get the implied predicate from A->B->C."""
        # Simplified transitive rules
        transitive_rules = {
            ('causes', 'causes'): 'causes',
            ('causes', 'increases'): 'increases',
            ('increases', 'causes'): 'increases',
            ('part_of', 'part_of'): 'part_of',
            ('is_a', 'is_a'): 'is_a',
        }

        return transitive_rules.get((pred1.lower(), pred2.lower()))

    def _potentially_related(self, claim1: dict, claim2: dict) -> bool:
        """Quick check if claims might be related."""
        # Check if they share any sources or connected entities
        sources1 = set(claim1.get('sources', []))
        sources2 = set(claim2.get('sources', []))

        if sources1 & sources2:  # Shared sources
            return True

        # Check for shared words (simple heuristic)
        words1 = set(claim1.get('name', '').lower().split())
        words2 = set(claim2.get('name', '').lower().split())
        common = words1 & words2 - {'the', 'a', 'an', 'is', 'are', 'was', 'were'}

        return len(common) >= 2

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response."""
        import json
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}
```

**Resolution Strategies**

```python
class ContradictionResolver:
    """Strategies for resolving detected contradictions."""

    def __init__(self, llm, kg_store: HybridKnowledgeGraphStore):
        self.llm = llm
        self.store = kg_store

    async def resolve(self, contradiction: dict) -> dict:
        """Attempt to resolve a contradiction."""

        resolution = {
            'contradiction': contradiction,
            'resolution_type': None,
            'action': None,
            'confidence': 0.0
        }

        # Strategy 1: Check source credibility
        source_resolution = self._resolve_by_credibility(contradiction)
        if source_resolution['confidence'] > 0.8:
            return source_resolution

        # Strategy 2: Check temporal scope
        temporal_resolution = self._resolve_by_time(contradiction)
        if temporal_resolution['confidence'] > 0.8:
            return temporal_resolution

        # Strategy 3: Check contextual scope
        context_resolution = await self._resolve_by_context(contradiction)
        if context_resolution['confidence'] > 0.7:
            return context_resolution

        # Strategy 4: Flag for human review
        return {
            'contradiction': contradiction,
            'resolution_type': 'needs_human_review',
            'action': 'Add to research questions for further investigation',
            'confidence': 0.0,
            'research_question': self._generate_research_question(contradiction)
        }

    def _resolve_by_credibility(self, contradiction: dict) -> dict:
        """Prefer claim from more credible source."""
        # Get source credibility scores
        # ... implementation based on source tracking
        return {'confidence': 0.0}

    def _resolve_by_time(self, contradiction: dict) -> dict:
        """Check if contradiction is due to temporal change."""
        # ... check timestamps
        return {'confidence': 0.0}

    async def _resolve_by_context(self, contradiction: dict) -> dict:
        """Use LLM to check if claims are true in different contexts."""
        prompt = f"""Analyze if these contradictory claims could both be true in different contexts:

Claim 1: {contradiction.get('claim1', contradiction.get('predicate1', ''))}
Claim 2: {contradiction.get('claim2', contradiction.get('predicate2', ''))}

Consider:
1. Different time periods
2. Different geographic regions
3. Different conditions or assumptions
4. Different definitions of key terms

If both can be true, explain the contexts. If truly contradictory, which is more likely correct?

Respond with JSON:
{{
  "both_true_possible": true/false,
  "context_for_claim1": "when claim 1 is true",
  "context_for_claim2": "when claim 2 is true",
  "if_contradictory": "which is more likely correct and why",
  "confidence": 0.0-1.0
}}"""

        response = await self.llm.agenerate(prompt)
        result = self._parse_json(response)

        if result.get('both_true_possible'):
            return {
                'resolution_type': 'contextual',
                'action': 'Add context qualifiers to both claims',
                'context1': result.get('context_for_claim1'),
                'context2': result.get('context_for_claim2'),
                'confidence': result.get('confidence', 0.5)
            }
        else:
            return {
                'resolution_type': 'one_correct',
                'action': 'Mark less supported claim as disputed',
                'preferred': result.get('if_contradictory'),
                'confidence': result.get('confidence', 0.5)
            }

    def _generate_research_question(self, contradiction: dict) -> str:
        """Generate a research question to resolve the contradiction."""
        if contradiction['type'] == 'direct':
            return (
                f"What determines whether {contradiction['subject']} "
                f"{contradiction['predicate1']} or {contradiction['predicate2']} "
                f"{contradiction['object']}?"
            )
        return f"Investigate: {contradiction.get('description', 'Unresolved contradiction')}"
```

**Source**: [KG Inconsistency Survey](https://arxiv.org/html/2502.19023v1), [Detect-Then-Resolve](https://www.mdpi.com/2227-7390/12/15/2318), [ACM Fact Checking Survey](https://dl.acm.org/doi/10.1145/3749838)

---

### 9.7 Visualization

**Pyvis for Interactive HTML Graphs**

```python
from pyvis.network import Network
import networkx as nx
from pathlib import Path

class KnowledgeGraphVisualizer:
    """Visualize knowledge graph using Pyvis."""

    # Color scheme by entity type
    ENTITY_COLORS = {
        'CONCEPT': '#6366f1',      # Indigo
        'CLAIM': '#f59e0b',        # Amber
        'EVIDENCE': '#10b981',     # Emerald
        'METHOD': '#8b5cf6',       # Violet
        'METRIC': '#ef4444',       # Red
        'TECHNOLOGY': '#3b82f6',   # Blue
        'SOURCE': '#6b7280',       # Gray
        'PERSON': '#ec4899',       # Pink
        'ORGANIZATION': '#14b8a6', # Teal
        'DEFAULT': '#9ca3af',      # Gray
    }

    def __init__(self, kg_store: HybridKnowledgeGraphStore):
        self.store = kg_store

    def create_interactive_graph(
        self,
        output_path: str = "knowledge_graph.html",
        height: str = "800px",
        width: str = "100%",
        show_physics_controls: bool = True
    ) -> str:
        """Create interactive Pyvis visualization."""

        # Create Pyvis network
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=False,
            bgcolor="#ffffff",
            font_color="#333333"
        )

        # Configure physics
        net.barnes_hut(
            gravity=-80000,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=0.001
        )

        if show_physics_controls:
            net.show_buttons(filter_=['physics'])

        # Add nodes
        for node_id, data in self.store.graph.nodes(data=True):
            entity_type = data.get('entity_type', 'DEFAULT')
            color = self.ENTITY_COLORS.get(entity_type, self.ENTITY_COLORS['DEFAULT'])

            # Size based on degree centrality
            size = 10 + (self.store.graph.degree(node_id) * 3)

            # Build hover title
            title = self._build_node_title(node_id, data)

            net.add_node(
                node_id,
                label=data.get('name', node_id)[:30],  # Truncate long names
                title=title,
                color=color,
                size=size,
                shape='dot' if entity_type != 'CLAIM' else 'diamond'
            )

        # Add edges
        for u, v, data in self.store.graph.edges(data=True):
            predicate = data.get('predicate', '')
            confidence = data.get('confidence', 1.0)

            # Color edges by type
            edge_color = self._get_edge_color(predicate)

            net.add_edge(
                u, v,
                title=predicate,
                label=predicate[:20] if len(predicate) <= 20 else predicate[:17] + '...',
                color=edge_color,
                width=1 + (confidence * 2),
                arrows='to'
            )

        # Save to file
        net.save_graph(output_path)
        return output_path

    def create_focused_subgraph(
        self,
        center_entity_id: str,
        depth: int = 2,
        output_path: str = "subgraph.html"
    ) -> str:
        """Create visualization centered on a specific entity."""

        # Get ego graph (subgraph around center node)
        subgraph = nx.ego_graph(
            self.store.graph,
            center_entity_id,
            radius=depth,
            undirected=True
        )

        # Create temporary store with subgraph
        temp_store = HybridKnowledgeGraphStore.__new__(HybridKnowledgeGraphStore)
        temp_store.graph = subgraph
        temp_store.db_path = self.store.db_path

        # Visualize
        temp_viz = KnowledgeGraphVisualizer(temp_store)
        return temp_viz.create_interactive_graph(output_path)

    def create_mermaid_diagram(self, max_nodes: int = 30) -> str:
        """Generate Mermaid diagram syntax for embedding in reports."""
        lines = ["```mermaid", "graph TD"]

        # Get most important nodes
        if len(self.store.graph) > max_nodes:
            pagerank = nx.pagerank(self.store.graph)
            top_nodes = sorted(
                pagerank.items(),
                key=lambda x: x[1],
                reverse=True
            )[:max_nodes]
            included_nodes = {n[0] for n in top_nodes}
        else:
            included_nodes = set(self.store.graph.nodes())

        # Add nodes with styling
        for node_id in included_nodes:
            data = self.store.graph.nodes[node_id]
            name = data.get('name', node_id).replace('"', "'")
            entity_type = data.get('entity_type', 'DEFAULT')

            # Mermaid node syntax
            safe_id = node_id.replace('-', '_')
            if entity_type == 'CLAIM':
                lines.append(f'    {safe_id}{{"{name}"}}')  # Diamond
            elif entity_type == 'EVIDENCE':
                lines.append(f'    {safe_id}[("{name}")]')  # Stadium
            else:
                lines.append(f'    {safe_id}["{name}"]')    # Rectangle

        # Add edges
        for u, v, data in self.store.graph.edges(data=True):
            if u in included_nodes and v in included_nodes:
                predicate = data.get('predicate', '')
                safe_u = u.replace('-', '_')
                safe_v = v.replace('-', '_')
                lines.append(f'    {safe_u} -->|{predicate}| {safe_v}')

        # Add styling
        lines.extend([
            "",
            "    classDef concept fill:#6366f1,color:white",
            "    classDef claim fill:#f59e0b,color:white",
            "    classDef evidence fill:#10b981,color:white",
        ])

        lines.append("```")
        return "\n".join(lines)

    def create_summary_stats_card(self) -> str:
        """Generate a text summary card for reports."""
        stats = {
            'nodes': self.store.graph.number_of_nodes(),
            'edges': self.store.graph.number_of_edges(),
            'components': nx.number_weakly_connected_components(self.store.graph),
        }

        # Count by type
        type_counts = {}
        for _, data in self.store.graph.nodes(data=True):
            etype = data.get('entity_type', 'UNKNOWN')
            type_counts[etype] = type_counts.get(etype, 0) + 1

        # Density
        if stats['nodes'] > 1:
            stats['density'] = nx.density(self.store.graph)
        else:
            stats['density'] = 0

        card = f"""
┌─────────────────────────────────────────┐
│          KNOWLEDGE GRAPH STATS          │
├─────────────────────────────────────────┤
│  Entities: {stats['nodes']:<28}│
│  Relations: {stats['edges']:<27}│
│  Components: {stats['components']:<26}│
│  Density: {stats['density']:.3f}                         │
├─────────────────────────────────────────┤
│  By Type:                               │"""

        for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            card += f"\n│    {etype}: {count:<30}│"

        card += "\n└─────────────────────────────────────────┘"
        return card

    def _build_node_title(self, node_id: str, data: dict) -> str:
        """Build HTML hover title for a node."""
        parts = [
            f"<b>{data.get('name', node_id)}</b>",
            f"Type: {data.get('entity_type', 'Unknown')}",
            f"Confidence: {data.get('confidence', 1.0):.2f}",
        ]

        aliases = data.get('aliases', [])
        if aliases:
            parts.append(f"Aliases: {', '.join(aliases[:3])}")

        sources = data.get('sources', [])
        if sources:
            parts.append(f"Sources: {len(sources)}")

        return "<br>".join(parts)

    def _get_edge_color(self, predicate: str) -> str:
        """Get edge color based on predicate type."""
        pred_lower = predicate.lower()

        if any(p in pred_lower for p in ['supports', 'evidence', 'confirms']):
            return '#10b981'  # Green
        elif any(p in pred_lower for p in ['contradicts', 'refutes', 'disputes']):
            return '#ef4444'  # Red
        elif any(p in pred_lower for p in ['causes', 'leads to', 'results']):
            return '#f59e0b'  # Amber
        elif any(p in pred_lower for p in ['is_a', 'type of', 'instance']):
            return '#6366f1'  # Indigo
        else:
            return '#9ca3af'  # Gray
```

**Source**: [Pyvis Documentation](https://pyvis.readthedocs.io/en/latest/), [BurnyCoder KG Visualization](https://github.com/BurnyCoder/knowledge-graph-visualization)

---

### 9.8 Complete Integration Example

**Putting It All Together for Claude-Researcher**

```python
import asyncio
from typing import AsyncIterator

class ResearchKnowledgeGraphSystem:
    """
    Complete knowledge graph system for claude-researcher.
    Integrates incremental construction, gap detection, contradiction detection,
    and Manager querying.
    """

    def __init__(
        self,
        llm,
        embeddings_model,
        db_path: str = "research_kg.db"
    ):
        # Initialize components
        self.kg = IncrementalKnowledgeGraph(
            llm=llm,
            embeddings_model=embeddings_model
        )
        self.store = HybridKnowledgeGraphStore(db_path=db_path)
        self.contradiction_detector = ContradictionDetector(
            kg_store=self.store, llm=llm
        )
        self.query_interface = ManagerQueryInterface(self.store)
        self.visualizer = KnowledgeGraphVisualizer(self.store)

        # Sync internal graph reference
        self.kg.entities = {
            n: Entity(id=n, name=d.get('name', n), entity_type=d.get('entity_type', 'UNKNOWN'))
            for n, d in self.store.graph.nodes(data=True)
        }

    async def process_finding_stream(
        self,
        findings: AsyncIterator[Finding]
    ) -> AsyncIterator[dict]:
        """
        Process a stream of findings from Interns.
        Yields processing results for each finding.
        """
        async for finding in findings:
            # Process and integrate finding
            result = await self.kg.add_finding(finding)

            # Persist to storage
            for entity in result['entities']:
                if entity.id not in self.store.graph:
                    self.store.add_entity(entity)

            for relation in result['relations']:
                self.store.add_relation(relation)

            # Check for new contradictions
            if result['contradictions_found'] > 0:
                result['contradiction_details'] = self.kg.contradictions[-result['contradictions_found']:]

            yield result

    def get_manager_briefing(self) -> str:
        """
        Generate a briefing for the Manager agent.
        Called periodically or on-demand.
        """
        return self.query_interface.get_research_summary()

    def get_next_research_directions(self) -> list[str]:
        """
        Suggest next research directions based on graph analysis.
        """
        directions = []

        # From gaps
        gaps = self.query_interface.identify_gaps()
        for gap in gaps:
            directions.append(gap['recommendation'])

        # From contradictions
        contradictions = self.query_interface.get_contradictions()
        for c in contradictions:
            directions.append(c['recommendation'])

        return directions[:10]  # Top 10 most important

    def export_for_report(self, output_dir: str = ".") -> dict:
        """
        Export graph visualizations and summaries for final report.
        """
        from pathlib import Path
        output_path = Path(output_dir)

        # Interactive HTML visualization
        html_path = self.visualizer.create_interactive_graph(
            str(output_path / "knowledge_graph.html")
        )

        # Mermaid diagram for embedding
        mermaid = self.visualizer.create_mermaid_diagram()

        # Stats card
        stats = self.visualizer.create_summary_stats_card()

        # Key concepts
        key_concepts = self.query_interface.get_key_concepts(10)

        return {
            'html_visualization': html_path,
            'mermaid_diagram': mermaid,
            'stats_card': stats,
            'key_concepts': key_concepts,
            'total_entities': self.store.graph.number_of_nodes(),
            'total_relations': self.store.graph.number_of_edges()
        }


# Usage Example
async def research_session_example():
    """Example of using the knowledge graph in a research session."""

    from langchain_anthropic import ChatAnthropic
    from langchain_openai import OpenAIEmbeddings

    # Initialize
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    embeddings = OpenAIEmbeddings()

    kg_system = ResearchKnowledgeGraphSystem(
        llm=llm,
        embeddings_model=embeddings,
        db_path="my_research.db"
    )

    # Simulate findings stream from Interns
    async def mock_findings_stream():
        findings = [
            Finding(
                id="f1",
                content="GraphRAG reduces hallucinations by 90% compared to traditional RAG.",
                source_url="https://example.com/graphrag",
                source_title="GraphRAG Study",
                timestamp="2026-01-15",
                credibility_score=0.85
            ),
            Finding(
                id="f2",
                content="Knowledge graphs improve RAG accuracy but increase latency by 2x.",
                source_url="https://example.com/kg-study",
                source_title="KG Performance Analysis",
                timestamp="2026-01-14",
                credibility_score=0.80
            ),
        ]
        for f in findings:
            yield f

    # Process findings
    async for result in kg_system.process_finding_stream(mock_findings_stream()):
        print(f"Processed finding: {len(result['entities'])} entities, {len(result['relations'])} relations")

        if result.get('contradictions_found', 0) > 0:
            print(f"  ⚠️  Contradictions detected: {result['contradictions_found']}")

    # Manager queries the graph
    print("\n--- Manager Briefing ---")
    print(kg_system.get_manager_briefing())

    print("\n--- Suggested Research Directions ---")
    for direction in kg_system.get_next_research_directions():
        print(f"  • {direction}")

    # Export for final report
    exports = kg_system.export_for_report("./output")
    print(f"\n--- Exports ---")
    print(f"HTML Graph: {exports['html_visualization']}")
    print(exports['stats_card'])


if __name__ == "__main__":
    asyncio.run(research_session_example())
```

---

### 9.9 Implementation Roadmap

**Phase 1: Core Infrastructure (Week 1)**
- [ ] Implement `IncrementalKnowledgeGraph` class
- [ ] Implement `HybridKnowledgeGraphStore` with SQLite persistence
- [ ] Basic entity and relation extraction prompts
- [ ] Unit tests for core functionality

**Phase 2: Entity Resolution (Week 2)**
- [ ] Embedding-based entity matching
- [ ] Coreference resolution with type-specific prompts
- [ ] Entity deduplication on merge
- [ ] Alias tracking

**Phase 3: Manager Integration (Week 3)**
- [ ] `ManagerQueryInterface` implementation
- [ ] Gap detection algorithms
- [ ] Research steering recommendations
- [ ] Integration with Manager agent loop

**Phase 4: Contradiction Detection (Week 4)**
- [ ] Direct predicate contradiction detection
- [ ] Semantic contradiction detection with LLM
- [ ] Contradiction resolution strategies
- [ ] Human review flagging

**Phase 5: Visualization & Reporting (Week 5)**
- [ ] Pyvis interactive visualization
- [ ] Mermaid diagram generation
- [ ] Stats cards for reports
- [ ] Integration with report writer

---

### Knowledge Graph References

### Incremental KG Construction
- [iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models](https://arxiv.org/html/2409.03284v1)
- [iText2KG GitHub](https://github.com/AuvaLab/itext2kg)
- [KGGen: Extracting Knowledge Graphs from Plain Text](https://arxiv.org/html/2502.09956v1)
- [Graphiti: Build Real-Time Knowledge Graphs for AI Agents](https://github.com/getzep/graphiti)
- [Emergent Mind - Incremental KG Construction](https://www.emergentmind.com/topics/incremental-knowledge-graph-construction)

### Entity Extraction & Resolution
- [LINK-KG: LLM-Driven Coreference-Resolved Knowledge Graphs](https://arxiv.org/abs/2510.26486)
- [CORE-KG: Structured Prompting and Coreference Resolution](https://arxiv.org/html/2510.26512v1)
- [Neo4j Knowledge Graph Extraction Challenges](https://neo4j.com/blog/developer/knowledge-graph-extraction-challenges/)
- [PingCAP: Using LLM to Extract KG Entities](https://www.pingcap.com/article/using-llm-extract-knowledge-graph-entities-and-relationships/)

### Schema & GraphRAG
- [GraphRAG on Technical Documents - Schema Impact](https://drops.dagstuhl.de/entities/document/10.4230/TGDK.3.2.3)
- [IBM: What is GraphRAG?](https://www.ibm.com/think/topics/graphrag)
- [Meilisearch: What is GraphRAG Complete Guide](https://www.meilisearch.com/blog/graph-rag)

### Contradiction Detection
- [Dealing with Inconsistency for Reasoning over Knowledge Graphs: A Survey](https://arxiv.org/html/2502.19023v1)
- [Detect-Then-Resolve: Enhancing KG Conflict Resolution with LLM](https://www.mdpi.com/2227-7390/12/15/2318)
- [ACM: Fact Checking Knowledge Graphs Survey](https://dl.acm.org/doi/10.1145/3749838)

### Gap Detection
- [InfraNodus: Text Network Analysis](https://infranodus.com/docs/network-analysis)
- [InfraNodus MCP Server](https://github.com/infranodus/mcp-server-infranodus)
- [KARMA: Multi-Agent LLMs for KG Enrichment](https://arxiv.org/html/2502.06472v1)

### Storage & Visualization
- [Local LLM-Powered Knowledge Graph (NetworkX + SQLite)](https://danielkliewer.com/blog/2025-10-19-building-a-local-llm-powered-knowledge-graph)
- [FalkorDBLite: Embedded Python Graph Database](https://www.falkordb.com/blog/falkordblite-embedded-python-graph-database/)
- [Pyvis Documentation](https://pyvis.readthedocs.io/en/latest/)
- [BurnyCoder KG Visualization](https://github.com/BurnyCoder/knowledge-graph-visualization)
- [kglab PyPI](https://pypi.org/project/kglab/)
