


import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from ..events import emit_finding
from .base import AgentConfig, BaseAgent, DecisionType


def _get_current_year() -> int:
    """Get the current year for search queries."""
    return datetime.now().year


import asyncio

from rich.console import Console

from ..models.findings import (
    AgentRole,
    Finding,
    FindingType,
    InternReport,
    ManagerDirective,
)
from ..retrieval.deduplication import FindingDeduplicator, get_deduplicator
from ..retrieval.query_expansion import (
    QueryExpander,
    QueryExpansionConfig,
    QueryExpansionResult,
    merge_search_results,
)
from ..storage.database import ResearchDatabase
from ..tools.web_search import SearchResult, WebSearchTool

if TYPE_CHECKING:
    from ..verification import VerificationPipeline


class InternAgent(BaseAgent):
    """The Intern agent searches the web and reports findings to the Manager.

    Responsibilities:
    - Execute web searches based on Manager directives
    - Extract relevant information from search results
    - Identify key facts, insights, and connections
    - Suggest follow-up topics for deeper investigation
    - Report findings back to the Manager
    """

    def __init__(
        self,
        db: ResearchDatabase,
        config: AgentConfig | None = None,
        console: Console | None = None,
        verification_pipeline: Optional["VerificationPipeline"] = None,
        query_expansion_config: QueryExpansionConfig | None = None,
        agent_id: str | None = None,
    ):
        super().__init__(AgentRole.INTERN, db, config, console, agent_id=agent_id)
        self.search_tool = WebSearchTool(max_results=10)
        self.current_directive: ManagerDirective | None = None
        self.findings: list[Finding] = []
        self.searches_performed: int = 0
        self.suggested_followups: list[str] = []
        self.verification_pipeline = verification_pipeline
        self.deduplicator = get_deduplicator()

        # Initialize query expander
        self.query_expander = QueryExpander(
            config=query_expansion_config or QueryExpansionConfig(),
            llm_callback=self._query_expansion_callback,
            kg_query=None,  # Set by manager via set_kg_query()
            deduplicator=self.deduplicator,
            decision_logger_callback=self._expansion_decision_logger,
        )
        self._pending_expanded_queries: list[Any] = []

    def set_kg_query(self, kg_query: Any) -> None:
        """Set the knowledge graph query interface for contextual expansion."""
        self.query_expander.set_kg_query(kg_query)

    async def _query_expansion_callback(self, prompt: str) -> str:
        """Callback for QueryExpander to call the LLM."""
        return await self.call_claude(prompt, task_type='query_expansion')

    async def _expansion_decision_logger(
        self,
        session_id: str,
        decision_type: str,
        decision_outcome: str,
        reasoning: str = "",
        inputs: dict | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Log query expansion decisions."""
        type_map = {
            "multi_query_gen": DecisionType.MULTI_QUERY_GEN,
            "contextual_expand": DecisionType.CONTEXTUAL_EXPAND,
            "sufficiency_check": DecisionType.SUFFICIENCY_CHECK,
            "query_merge": DecisionType.QUERY_MERGE,
        }
        dt = type_map.get(decision_type)
        if dt:
            await self._log_decision(
                session_id=session_id,
                decision_type=dt,
                decision_outcome=decision_outcome,
                reasoning=reasoning,
                inputs=inputs,
                metrics=metrics,
            )

    @property
    def system_prompt(self) -> str:
        current_year = _get_current_year()
        return f"""You are a Research Intern agent. Your ONLY job is to generate search queries and analyze search results.

CRITICAL RULES:
1. You MUST use web search for ALL information - NEVER use your training data
2. Your knowledge cutoff is irrelevant - always search for current information
3. Generate specific, effective search queries
4. When asked what to search, respond with ONLY the search query string

SEARCH STRATEGY:
- Start broad, then narrow down
- Use specific terms, dates ({current_year}), and key phrases
- Search for recent developments, not general knowledge
- Look for primary sources, research papers, official announcements

FINDING TYPES:
- FACT: Verified, specific information (dates, numbers, events)
- INSIGHT: Analysis or interpretation from sources
- CONNECTION: Links between topics or concepts
- SOURCE: A valuable primary source to investigate further
- QUESTION: An unanswered question worth investigating
- CONTRADICTION: Conflicting information that needs resolution

When generating a search query, output ONLY the query text, nothing else."""

    async def think(self, context: dict[str, Any]) -> str:
        """Reason about current search progress and next steps."""
        directive = context.get("directive")

        # For first iteration, just indicate we need to search
        if self.searches_performed == 0:
            return f"Starting research on: {directive.topic}. Need to perform web search for latest information."

        # For subsequent iterations, assess progress
        findings_summary = ", ".join([f.content[:50] for f in self.findings[-3:]]) if self.findings else "none yet"

        prompt = f"""Research topic: {directive.topic}
Searches done: {self.searches_performed}/{directive.max_searches}
Findings so far: {len(self.findings)}
Recent findings: {findings_summary}

Should I continue searching or compile report? If continue, what aspect should I search next?
Be brief - just state your decision and reason."""

        return await self.call_claude(prompt)

    async def act(self, thought: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a search or compile a report based on thinking."""
        directive: ManagerDirective = context.get("directive")
        session_id = context.get("session_id", 0)

        # Check if we should stop
        if self._should_stop_searching(thought, directive):
            # Log stop searching decision
            await self._log_decision(
                session_id=session_id,
                decision_type=DecisionType.STOP_SEARCHING,
                decision_outcome="stop",
                reasoning=thought[:500],
                inputs={"topic": directive.topic, "max_searches": directive.max_searches},
                metrics={"searches_done": self.searches_performed, "findings_count": len(self.findings)},
            )
            return {
                "action": "compile_report",
                "report": await self._compile_report(directive.topic, session_id),
            }

        # Use query expansion for better coverage
        results, search_summary, queries_used = await self._search_with_expansion(
            directive.topic, session_id
        )

        # Check for early stop due to sufficiency
        if not queries_used:
            self._log("[Sufficiency] Gathered enough information", style="bold magenta")
            return {
                "action": "compile_report",
                "report": await self._compile_report(directive.topic, session_id),
            }

        self.searches_performed += 1

        # Show search summary
        if search_summary:
            self._log("─" * 60, style="dim")
            self._log("[Search Summary]", style="bold cyan")
            summary_preview = search_summary[:1500] + "..." if len(search_summary) > 1500 else search_summary
            self.console.print(summary_preview)
            self._log("─" * 60, style="dim")

        # Show search results
        if results:
            self._log(f"[Search Results: {len(results)} found]", style="bold yellow")
            for i, r in enumerate(results[:5], 1):
                self._log(f"  {i}. {r.title}", style="yellow")
                if r.url:
                    self._log(f"     URL: {r.url}", style="dim")
                if r.snippet:
                    snippet = r.snippet[:200] + "..." if len(r.snippet) > 200 else r.snippet
                    self._log(f"     {snippet}", style="dim")

        # Process results and extract findings (use primary query for logging)
        primary_query = queries_used[0] if queries_used else directive.topic
        new_findings = await self._process_search_results(
            results, primary_query, session_id, search_summary
        )

        # Show extracted findings
        if new_findings:
            self._log(f"[Extracted {len(new_findings)} Findings]", style="bold green")
            for f in new_findings:
                self._log(f"  [{f.finding_type.value.upper()}] {f.content[:150]}...", style="green")
                if f.source_url:
                    self._log(f"    Source: {f.source_url}", style="dim")
                self._log(f"    Confidence: {f.confidence:.0%}", style="dim")

        return {
            "action": "search",
            "query": primary_query,
            "queries_used": queries_used,
            "results_count": len(results),
            "findings_extracted": len(new_findings),
            "results": results,
            "summary": search_summary,
        }

    async def observe(self, action_result: dict[str, Any]) -> str:
        """Process the result of a search action."""
        action = action_result.get("action")

        if action == "compile_report":
            report: InternReport = action_result.get("report")
            return f"Report compiled: {len(report.findings)} findings, {len(report.suggested_followups)} follow-up suggestions"

        if action == "search":
            query = action_result.get("query")
            results_count = action_result.get("results_count", 0)
            findings_count = action_result.get("findings_extracted", 0)

            if results_count == 0:
                return f"Search for '{query}' returned no results. Consider rephrasing or trying a different angle."

            return f"Search for '{query}' returned {results_count} results, extracted {findings_count} findings. Total findings now: {len(self.findings)}"

        return "Unknown action result"

    def is_done(self, context: dict[str, Any]) -> bool:
        """Check if the intern has completed the current directive."""
        directive: ManagerDirective = context.get("directive")
        if not directive:
            return True

        # Stop if we hit max searches
        if self.searches_performed >= directive.max_searches:
            return True

        # Stop if action was to compile report
        last_action = context.get("last_action", {})
        if last_action.get("action") == "compile_report":
            return True

        # Stop if directive says to stop
        if directive.action == "stop":
            return True

        return False

    def _should_stop_searching(self, thought: str, directive: ManagerDirective) -> bool:
        """Determine if we should stop searching based on the thought."""
        thought_lower = thought.lower()
        stop_indicators = [
            "should stop",
            "enough information",
            "compile report",
            "ready to report",
            "sufficient findings",
            "covered the topic",
        ]
        return any(indicator in thought_lower for indicator in stop_indicators)

    async def _search_with_expansion(
        self, topic: str, session_id: str
    ) -> tuple[list[SearchResult], str, list[str]]:
        """Search using expanded queries for better coverage.

        Uses QueryExpander to generate multiple query variations, executes them
        in parallel, and merges results using Reciprocal Rank Fusion.

        Returns:
            Tuple of (merged_results, combined_summary, queries_used)
            Empty queries_used indicates sufficiency was reached.
        """
        expansion = await self.query_expander.expand(
            query=topic,
            session_id=session_id,
            previous_findings=self.findings,
            search_iteration=self.searches_performed,
        )

        # Early stop if sufficient information gathered
        if expansion.is_sufficient:
            self._log(
                f"[Query Expansion] Sufficiency reached: {expansion.sufficiency_score:.0%}",
                style="magenta"
            )
            return [], "Sufficient information gathered", []

        queries = [eq.query for eq in expansion.expanded_queries]
        if not queries:
            queries = [topic]

        self._log(f"[Query Expansion] Executing {len(queries)} queries", style="cyan")
        for i, q in enumerate(queries, 1):
            self._log(f"  {i}. {q[:80]}...", style="dim") if len(q) > 80 else self._log(f"  {i}. {q}", style="dim")

        # Execute searches in parallel
        tasks = [self.search_tool.search(q) for q in queries]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results using RRF
        merged_results, merge_summary = merge_search_results(
            queries=queries,
            results_list=results_list,
            k=60,
            max_results=15,
        )

        # Log query merge decision
        await self._log_decision(
            session_id=session_id,
            decision_type=DecisionType.QUERY_MERGE,
            decision_outcome=f"merged_{len(queries)}_queries",
            reasoning=merge_summary,
            inputs={"queries": queries[:3]},
            metrics={
                "query_count": len(queries),
                "total_results": len(merged_results),
                "kg_gaps_used": len(expansion.kg_gaps_used),
            },
        )

        # Combine summaries from all searches
        summaries = []
        for r in results_list:
            if isinstance(r, tuple) and len(r) > 1 and r[1]:
                summaries.append(r[1])

        combined_summary = "\n\n---\n\n".join(summaries[:2]) if summaries else ""

        return merged_results, combined_summary, queries

    async def _extract_search_query(
        self, thought: str, directive: ManagerDirective, session_id: str = ""
    ) -> str | None:
        """Extract a search query from the agent's thought."""
        # Check if thought indicates we should stop
        if self._should_stop_searching(thought, directive):
            return None

        # Generate a search query based on the directive and progress
        current_year = _get_current_year()
        if self.searches_performed == 0:
            # First search - use expanded query for broader coverage
            return await self._expand_query(directive.topic, current_year, session_id)

        # Subsequent searches - use diverse query expansion
        prompt = f"""Topic: {directive.topic}
Searches done: {self.searches_performed}
Previous findings: {len(self.findings)}

Recent findings summary:
{self._get_findings_summary()}

Generate ONE specific search query to find NEW information not covered by existing findings.
Focus on:
- Different angles or perspectives
- Recent developments ({current_year})
- Specific subtopics not yet explored
- Primary sources (research papers, official docs)

Output ONLY the search query, nothing else."""

        response = await self.call_claude(prompt)
        query = response.strip().strip('"').strip("'")

        # Clean up the query - remove any preamble
        if ":" in query and len(query.split(":")[0]) < 20:
            query = query.split(":", 1)[1].strip()

        # Don't search for error messages or meta-text
        if "error" in query.lower() or len(query) > 200:
            return f"{directive.topic} recent research {current_year}"

        return query if query and query.upper() != "STOP" else None

    async def _expand_query(self, topic: str, year: int, session_id: str = "") -> str:
        """Expand a query to improve search coverage.

        Uses query expansion techniques:
        - Synonym expansion
        - Temporal scoping
        - Specificity adjustment
        """
        prompt = f"""Expand this research topic into an effective web search query.

Topic: {topic}

Create a search query that:
1. Includes specific keywords and synonyms
2. Targets recent information ({year})
3. Avoids overly generic terms
4. Is optimized for finding authoritative sources

Output ONLY the search query (15-25 words max), nothing else."""

        response = await self.call_claude(prompt, task_type='query_expansion')
        query = response.strip().strip('"').strip("'")

        # Determine expansion strategy
        used_fallback = False
        if not query or len(query) > 200 or "error" in query.lower():
            query = f"{topic} {year} latest research developments"
            used_fallback = True

        # Log query expansion decision
        if session_id:
            await self._log_decision(
                session_id=session_id,
                decision_type=DecisionType.QUERY_EXPAND,
                decision_outcome="expanded" if not used_fallback else "fallback",
                reasoning=f"Expanded '{topic}' -> '{query[:100]}'",
                inputs={"original_topic": topic, "year": year},
                metrics={"search_number": self.searches_performed + 1, "used_fallback": used_fallback},
            )

        return query

    def _get_findings_summary(self) -> str:
        """Get a brief summary of recent findings to avoid duplicate searches."""
        if not self.findings:
            return "None yet"

        recent = self.findings[-5:]  # Last 5 findings
        summaries = []
        for f in recent:
            content = f.content[:80] + "..." if len(f.content) > 80 else f.content
            summaries.append(f"- {content}")
        return "\n".join(summaries)

    async def _process_search_results(
        self,
        results: list[SearchResult],
        query: str,
        session_id: str,
        search_summary: str = "",
    ) -> list[Finding]:
        """Process search results and extract findings."""
        if not results and not search_summary:
            return []

        # Format results for Claude to analyze
        results_text = "\n\n".join([
            f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}"
            for r in results[:10]
        ])

        # Include the search summary if available
        summary_section = ""
        if search_summary:
            summary_section = f"\n\nSearch Summary:\n{search_summary}\n"

        prompt = f"""Analyze these search results for the query: "{query}"

{results_text}
{summary_section}
Extract key findings. For each finding, provide:
1. The finding content (1-2 sentences)
2. Type: FACT, INSIGHT, CONNECTION, SOURCE, QUESTION, or CONTRADICTION
3. Source URL
4. Confidence score (0.0-1.0)

Also suggest 2-3 follow-up search queries that could deepen this research.

Format your response as JSON:
{{
    "findings": [
        {{"content": "...", "type": "FACT", "url": "...", "confidence": 0.8}},
        ...
    ],
    "followups": ["query1", "query2", "query3"]
}}"""

        response = await self.call_claude(prompt)

        findings = []

        # Try to parse JSON response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])

                for f in data.get("findings", []):
                    content = f.get("content", "")

                    # Check for duplicates before processing
                    if self.deduplicator.enabled:
                        dedup_result = self.deduplicator.check(content)
                        if dedup_result.is_duplicate:
                            self._log(
                                f"[DEDUP] Skipping duplicate ({dedup_result.match_type}, "
                                f"sim={dedup_result.similarity:.0%})",
                                style="dim"
                            )
                            # Log dedup skip decision
                            await self._log_decision(
                                session_id=session_id,
                                decision_type=DecisionType.DEDUP_SKIP,
                                decision_outcome="skipped",
                                reasoning=f"Content matched existing finding: {content[:100]}",
                                inputs={"match_type": dedup_result.match_type},
                                metrics={"similarity": dedup_result.similarity},
                            )
                            continue

                    source_url = f.get("url")
                    finding = Finding(
                        session_id=session_id,
                        content=content,
                        finding_type=FindingType(f.get("type", "fact").lower()),
                        source_url=source_url,
                        confidence=f.get("confidence", 0.7),
                        search_query=query,
                    )

                    # Run streaming verification if pipeline is available
                    if self.verification_pipeline:
                        try:
                            result = await self.verification_pipeline.verify_intern_finding(
                                finding, session_id
                            )
                            # Update finding with verification results
                            finding.original_confidence = finding.confidence
                            finding.confidence = result.verified_confidence
                            finding.verification_status = result.verification_status.value
                            finding.verification_method = result.verification_method.value
                            finding.kg_support_score = result.kg_support_score
                        except Exception as e:
                            self._log(f"[VERIFY] Error: {e}", style="dim")

                    await self.db.save_finding(finding)
                    findings.append(finding)
                    self.findings.append(finding)

                    # Emit finding event for WebSocket
                    if self.session_id:
                        await emit_finding(
                            session_id=self.session_id,
                            agent=self.role.value,
                            content=content[:300],  # Truncate for display
                            source=source_url,
                            confidence=finding.confidence
                        )

                    # Add to deduplication index after saving
                    if self.deduplicator.enabled:
                        finding_id = str(finding.id) if finding.id else f"{session_id}_{len(self.findings)}"
                        self.deduplicator.add(finding_id, content)

                for followup in data.get("followups", []):
                    # Filter out meta-questions/clarifying questions that aren't real topics
                    followup_lower = followup.lower()
                    is_meta_question = any(phrase in followup_lower for phrase in [
                        "please provide",
                        "what information",
                        "could you clarify",
                        "what are you looking for",
                        "what topic",
                        "what subject",
                        "what would you like",
                        "can you specify",
                        "please specify",
                        "more details",
                    ])
                    if not is_meta_question and followup not in self.suggested_followups:
                        self.suggested_followups.append(followup)

        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: if no JSON findings but we have search results, create findings from them
        if not findings and results:
            for r in results[:5]:
                if r.snippet:
                    content = r.snippet[:500]

                    # Check for duplicates
                    if self.deduplicator.enabled:
                        dedup_result = self.deduplicator.check(content)
                        if dedup_result.is_duplicate:
                            # Log dedup skip decision (fallback path)
                            await self._log_decision(
                                session_id=session_id,
                                decision_type=DecisionType.DEDUP_SKIP,
                                decision_outcome="skipped_fallback",
                                reasoning=f"Fallback content matched existing: {content[:100]}",
                                inputs={"match_type": dedup_result.match_type},
                                metrics={"similarity": dedup_result.similarity},
                            )
                            continue

                    finding = Finding(
                        session_id=session_id,
                        content=content,
                        finding_type=FindingType.FACT,
                        source_url=r.url,
                        confidence=0.6,
                        search_query=query,
                    )

                    # Run streaming verification if pipeline is available
                    if self.verification_pipeline:
                        try:
                            result = await self.verification_pipeline.verify_intern_finding(
                                finding, session_id
                            )
                            finding.original_confidence = finding.confidence
                            finding.confidence = result.verified_confidence
                            finding.verification_status = result.verification_status.value
                            finding.verification_method = result.verification_method.value
                            finding.kg_support_score = result.kg_support_score
                        except Exception:
                            pass

                    await self.db.save_finding(finding)
                    findings.append(finding)
                    self.findings.append(finding)

                    # Add to deduplication index
                    if self.deduplicator.enabled:
                        finding_id = str(finding.id) if finding.id else f"{session_id}_{len(self.findings)}"
                        self.deduplicator.add(finding_id, content)

        return findings

    async def _compile_report(self, topic: str, session_id: str) -> InternReport:
        """Compile findings into a report for the Manager."""
        return InternReport(
            topic=topic,
            findings=self.findings.copy(),
            searches_performed=self.searches_performed,
            suggested_followups=self.suggested_followups.copy(),
            blockers=[],
        )

    def reset(self) -> None:
        """Reset state for a new directive."""
        self.current_directive = None
        self.findings = []
        self.searches_performed = 0
        self.suggested_followups = []
        self.search_tool.reset_count()
        self.state = type(self.state)()
        self._pending_expanded_queries = []

    async def execute_directive(
        self, directive: ManagerDirective, session_id: str
    ) -> InternReport:
        """Execute a directive from the Manager and return a report."""
        self.reset()
        self.current_directive = directive
        # Ensure WebSocket event emission uses the correct session ID
        self.session_id = session_id

        context = {
            "directive": directive,
            "session_id": session_id,
        }

        await self.run(context)

        return await self._compile_report(directive.topic, session_id)
