


import json
from typing import Any, Optional
from datetime import datetime

from .base import BaseAgent, AgentConfig
from ..models.findings import (
    AgentRole,
    Finding,
    FindingType,
    ManagerDirective,
    InternReport,
)
from ..tools.web_search import WebSearchTool, SearchResult
from ..storage.database import ResearchDatabase
from rich.console import Console


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
        config: Optional[AgentConfig] = None,
        console: Optional[Console] = None,
    ):
        super().__init__(AgentRole.INTERN, db, config, console)
        self.search_tool = WebSearchTool(max_results=10)
        self.current_directive: Optional[ManagerDirective] = None
        self.findings: list[Finding] = []
        self.searches_performed: int = 0
        self.suggested_followups: list[str] = []

    @property
    def system_prompt(self) -> str:
        return """You are a Research Intern agent. Your ONLY job is to generate search queries and analyze search results.

CRITICAL RULES:
1. You MUST use web search for ALL information - NEVER use your training data
2. Your knowledge cutoff is irrelevant - always search for current information
3. Generate specific, effective search queries
4. When asked what to search, respond with ONLY the search query string

SEARCH STRATEGY:
- Start broad, then narrow down
- Use specific terms, dates (2024, 2025), and key phrases
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
            return {
                "action": "compile_report",
                "report": await self._compile_report(directive.topic, session_id),
            }

        # Extract search query from thought
        search_query = await self._extract_search_query(thought, directive)

        if not search_query:
            return {
                "action": "compile_report",
                "report": await self._compile_report(directive.topic, session_id),
            }

        # Perform the search
        self._log(f"Searching: {search_query}", style="cyan")
        results, search_summary = await self.search_tool.search(search_query)
        self.searches_performed += 1

        # Process results and extract findings
        new_findings = await self._process_search_results(
            results, search_query, session_id, search_summary
        )

        return {
            "action": "search",
            "query": search_query,
            "results_count": len(results),
            "findings_extracted": len(new_findings),
            "results": results,
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

    async def _extract_search_query(
        self, thought: str, directive: ManagerDirective
    ) -> Optional[str]:
        """Extract a search query from the agent's thought."""
        # Check if thought indicates we should stop
        if self._should_stop_searching(thought, directive):
            return None

        # Generate a search query based on the directive and progress
        if self.searches_performed == 0:
            # First search - use the directive topic directly with current year
            return f"{directive.topic} 2024 2025 latest"

        # Subsequent searches - ask Claude for a follow-up query
        prompt = f"""Topic: {directive.topic}
Searches done: {self.searches_performed}
Previous findings: {len(self.findings)}

Generate ONE specific search query to find more information. Focus on recent developments (2024-2025).
Output ONLY the search query, nothing else."""

        response = await self.call_claude(prompt)
        query = response.strip().strip('"').strip("'")

        # Clean up the query - remove any preamble
        if ":" in query and len(query.split(":")[0]) < 20:
            query = query.split(":", 1)[1].strip()

        # Don't search for error messages or meta-text
        if "error" in query.lower() or len(query) > 200:
            return f"{directive.topic} recent research 2024"

        return query if query and query.upper() != "STOP" else None

    async def _process_search_results(
        self,
        results: list[SearchResult],
        query: str,
        session_id: int,
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
                    finding = Finding(
                        session_id=session_id,
                        content=f.get("content", ""),
                        finding_type=FindingType(f.get("type", "fact").lower()),
                        source_url=f.get("url"),
                        confidence=f.get("confidence", 0.7),
                        search_query=query,
                    )
                    await self.db.save_finding(finding)
                    findings.append(finding)
                    self.findings.append(finding)

                for followup in data.get("followups", []):
                    if followup not in self.suggested_followups:
                        self.suggested_followups.append(followup)

        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: if no JSON findings but we have search results, create findings from them
        if not findings and results:
            for r in results[:5]:
                if r.snippet:
                    finding = Finding(
                        session_id=session_id,
                        content=r.snippet[:500],
                        finding_type=FindingType.FACT,
                        source_url=r.url,
                        confidence=0.6,
                        search_query=query,
                    )
                    await self.db.save_finding(finding)
                    findings.append(finding)
                    self.findings.append(finding)

        return findings

    async def _compile_report(self, topic: str, session_id: int) -> InternReport:
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

    async def execute_directive(
        self, directive: ManagerDirective, session_id: int
    ) -> InternReport:
        """Execute a directive from the Manager and return a report."""
        self.reset()
        self.current_directive = directive

        context = {
            "directive": directive,
            "session_id": session_id,
        }

        await self.run(context)

        return await self._compile_report(directive.topic, session_id)
