"""Manager agent - coordinates research and critiques findings."""

import json
from typing import Any, Optional
from datetime import datetime, timedelta

from .base import BaseAgent, AgentConfig
from .intern import InternAgent
from ..models.findings import (
    AgentRole,
    Finding,
    ManagerDirective,
    InternReport,
    ManagerReport,
    ResearchTopic,
)
from ..storage.database import ResearchDatabase
from rich.console import Console


class ManagerAgent(BaseAgent):
    """The Manager agent coordinates research and critiques the Intern's work.

    Responsibilities:
    - Break down research goals into specific topics
    - Create directives for the Intern agent
    - Critique and validate findings
    - Identify gaps and request additional research
    - Synthesize findings into coherent reports
    - Track research progress and depth
    """

    def __init__(
        self,
        db: ResearchDatabase,
        intern: InternAgent,
        config: Optional[AgentConfig] = None,
        console: Optional[Console] = None,
    ):
        super().__init__(AgentRole.MANAGER, db, config, console)
        self.intern = intern
        self.research_goal: str = ""
        self.session_id: int = 0
        self.topics_queue: list[ResearchTopic] = []
        self.completed_topics: list[ResearchTopic] = []
        self.all_findings: list[Finding] = []
        self.all_reports: list[InternReport] = []
        self.current_depth: int = 0
        self.max_depth: int = 5
        self.start_time: Optional[datetime] = None
        self.time_limit_minutes: int = 60

    @property
    def system_prompt(self) -> str:
        return """You are a Research Manager agent. You coordinate and oversee the research process.

RESPONSIBILITIES:
1. Break down research goals into specific, searchable topics
2. Create clear directives for the Research Intern
3. Critically evaluate the Intern's findings
4. Identify gaps, inconsistencies, and areas needing deeper investigation
5. Synthesize findings into coherent insights
6. Track research progress and manage time effectively

RESEARCH STRATEGY:
- Start broad, then go deep on promising threads
- Prioritize high-value topics over comprehensive coverage
- Look for connections between different findings
- Question assumptions and seek verification
- Balance depth vs breadth based on time constraints

CRITIQUE FRAMEWORK:
When reviewing findings, consider:
- Accuracy: Is this information correct and verifiable?
- Relevance: Does this directly address the research goal?
- Depth: Is this surface-level or substantive?
- Sources: Are the sources credible and current?
- Gaps: What important aspects are missing?
- Connections: How does this relate to other findings?

QUALITY STANDARDS:
- Reject findings that are speculation presented as fact
- Flag contradictions for investigation
- Prioritize primary sources over secondary
- Note when confidence is low

OUTPUT FORMAT:
Provide structured analysis with clear reasoning. When creating directives:
- Be specific about what to search for
- Explain why this topic matters
- Set appropriate priority (1-10)
- Define success criteria"""

    async def think(self, context: dict[str, Any]) -> str:
        """Reason about research progress and next steps."""
        time_elapsed = self._get_elapsed_minutes()
        time_remaining = self.time_limit_minutes - time_elapsed

        # Summarize current state
        findings_summary = self._summarize_findings()

        prompt = f"""Research Goal: {self.research_goal}

Current Status:
- Time elapsed: {time_elapsed:.1f} minutes
- Time remaining: {time_remaining:.1f} minutes
- Topics completed: {len(self.completed_topics)}
- Topics in queue: {len(self.topics_queue)}
- Total findings: {len(self.all_findings)}
- Current depth: {self.current_depth}/{self.max_depth}

Last report from Intern:
{context.get('last_report_summary', 'No report yet')}

Findings summary:
{findings_summary}

What should I do next? Consider:
1. Are there gaps in the research?
2. Should I go deeper on any topic?
3. Are there new angles to explore?
4. Should I ask for verification of any findings?
5. Is it time to synthesize and report?

Think step by step about the best next action."""

        return await self.call_claude(prompt)

    async def act(self, thought: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute management actions based on thinking."""
        time_remaining = self.time_limit_minutes - self._get_elapsed_minutes()

        # Check if we should synthesize and stop
        if self._should_synthesize(thought, time_remaining):
            report = await self._synthesize_report()
            return {
                "action": "synthesize",
                "report": report,
            }

        # Check if we should create a new directive for the intern
        if self._should_create_directive(thought):
            directive = await self._create_directive(thought)
            if directive:
                self._log("═" * 70, style="bold blue")
                self._log(f"[DIRECTIVE] {directive.action.upper()}: {directive.topic}", style="bold green")
                self._log(f"  Instructions: {directive.instructions}", style="dim")
                self._log(f"  Priority: {directive.priority}/10 | Max Searches: {directive.max_searches}", style="dim")
                self._log("═" * 70, style="bold blue")

                # Execute the directive via the intern
                intern_report = await self.intern.execute_directive(
                    directive, self.session_id
                )
                self.all_reports.append(intern_report)
                self.all_findings.extend(intern_report.findings)

                # Critique the report
                critique = await self._critique_report(intern_report)

                # Show critique
                self._log("─" * 70, style="dim")
                self._log("[MANAGER CRITIQUE]", style="bold magenta")
                self.console.print(critique)
                self._log("─" * 70, style="dim")

                # Add follow-up topics to queue
                await self._process_followups(intern_report, directive)

                # Show follow-up topics added
                if intern_report.suggested_followups:
                    self._log(f"[Follow-up Topics Added: {len(intern_report.suggested_followups)}]", style="cyan")
                    for ft in intern_report.suggested_followups[:3]:
                        self._log(f"  → {ft}", style="cyan")

                return {
                    "action": "intern_task",
                    "directive": directive,
                    "report": intern_report,
                    "critique": critique,
                }

        # Check pending topics
        if self.topics_queue:
            topic = self.topics_queue.pop(0)
            directive = ManagerDirective(
                action="search",
                topic=topic.topic,
                instructions=f"Research this topic thoroughly: {topic.topic}",
                priority=topic.priority,
                max_searches=5,
            )

            self._log("═" * 70, style="bold blue")
            self._log(f"[QUEUED TOPIC] Depth {topic.depth}: {topic.topic}", style="bold green")
            self._log(f"  Priority: {topic.priority}/10 | Remaining in queue: {len(self.topics_queue)}", style="dim")
            self._log("═" * 70, style="bold blue")

            intern_report = await self.intern.execute_directive(
                directive, self.session_id
            )
            self.all_reports.append(intern_report)
            self.all_findings.extend(intern_report.findings)
            self.completed_topics.append(topic)
            self.current_depth = max(self.current_depth, topic.depth)

            await self.db.update_topic_status(
                topic.id, "completed", len(intern_report.findings)
            )

            critique = await self._critique_report(intern_report)

            # Show critique
            self._log("─" * 70, style="dim")
            self._log("[MANAGER CRITIQUE]", style="bold magenta")
            self.console.print(critique)
            self._log("─" * 70, style="dim")

            await self._process_followups(intern_report, directive, topic)

            # Show follow-up topics added
            if intern_report.suggested_followups:
                self._log(f"[Follow-up Topics Added: {len(intern_report.suggested_followups)}]", style="cyan")
                for ft in intern_report.suggested_followups[:3]:
                    self._log(f"  → {ft}", style="cyan")

            return {
                "action": "intern_task",
                "directive": directive,
                "report": intern_report,
                "critique": critique,
            }

        # Nothing to do, synthesize
        report = await self._synthesize_report()
        return {
            "action": "synthesize",
            "report": report,
        }

    async def observe(self, action_result: dict[str, Any]) -> str:
        """Process the result of a management action."""
        action = action_result.get("action")

        if action == "synthesize":
            report: ManagerReport = action_result.get("report")
            return f"Synthesized report: {len(report.key_findings)} key findings, {len(report.recommended_next_steps)} recommendations"

        if action == "intern_task":
            report: InternReport = action_result.get("report")
            critique = action_result.get("critique", "")
            directive: ManagerDirective = action_result.get("directive")

            return f"""Intern completed task on '{directive.topic}':
- Findings: {len(report.findings)}
- Searches: {report.searches_performed}
- Follow-ups suggested: {len(report.suggested_followups)}
- Critique: {critique[:200]}..."""

        return "Unknown action"

    def is_done(self, context: dict[str, Any]) -> bool:
        """Check if the Manager should stop."""
        # Check time limit
        if self._get_elapsed_minutes() >= self.time_limit_minutes:
            self._log("Time limit reached", style="yellow")
            return True

        # Check if synthesis complete
        last_action = context.get("last_action", {})
        if last_action.get("action") == "synthesize":
            return True

        # Check max iterations
        if self.state.iteration >= self.config.max_iterations:
            return True

        return False

    def _get_elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        if not self.start_time:
            return 0
        return (datetime.now() - self.start_time).total_seconds() / 60

    def _summarize_findings(self) -> str:
        """Create a brief summary of all findings."""
        if not self.all_findings:
            return "No findings yet."

        by_type = {}
        for f in self.all_findings:
            t = f.finding_type.value
            by_type.setdefault(t, []).append(f)

        summary_parts = []
        for ftype, findings in by_type.items():
            summary_parts.append(f"- {ftype.upper()}: {len(findings)} findings")
            for f in findings[:2]:  # Show first 2 of each type
                summary_parts.append(f"  * {f.content[:100]}...")

        return "\n".join(summary_parts)

    def _should_synthesize(self, thought: str, time_remaining: float) -> bool:
        """Determine if it's time to synthesize results."""
        # Never synthesize if we have no findings - always do research first!
        if not self.all_findings:
            return False

        # Time pressure - only synthesize early if we have SOME findings
        if time_remaining < 5 and self.all_findings:
            return True

        # Explicit signals - but only if we have meaningful findings
        if len(self.all_findings) < 3:
            return False  # Need at least some findings before synthesizing

        thought_lower = thought.lower()
        synthesis_signals = [
            "time to synthesize",
            "final report",
            "wrap up now",
            "conclude the research",
            "sufficient coverage",
            "enough findings",
        ]
        return any(signal in thought_lower for signal in synthesis_signals)

    def _should_create_directive(self, thought: str) -> bool:
        """Determine if we should create a new directive."""
        thought_lower = thought.lower()
        directive_signals = [
            "search for",
            "investigate",
            "look into",
            "explore",
            "research",
            "find out",
            "verify",
            "deep dive",
        ]
        return any(signal in thought_lower for signal in directive_signals)

    async def _create_directive(self, thought: str) -> Optional[ManagerDirective]:
        """Create a directive for the Intern based on reasoning."""
        prompt = f"""Based on this reasoning:
{thought}

Create a directive for the Research Intern. Return as JSON:
{{
    "action": "search" or "deep_dive" or "verify",
    "topic": "specific topic to research",
    "instructions": "detailed instructions",
    "priority": 1-10,
    "max_searches": 3-10
}}

Return ONLY the JSON."""

        response = await self.call_claude(prompt)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                return ManagerDirective(
                    action=data.get("action", "search"),
                    topic=data.get("topic", ""),
                    instructions=data.get("instructions", ""),
                    priority=data.get("priority", 5),
                    max_searches=data.get("max_searches", 5),
                )
        except (json.JSONDecodeError, KeyError):
            pass

        return None

    async def _critique_report(self, report: InternReport) -> str:
        """Critique an Intern's report."""
        findings_text = "\n".join([
            f"- [{f.finding_type.value}] {f.content} (confidence: {f.confidence})"
            for f in report.findings[:10]
        ])

        prompt = f"""Critique this research report:

Topic: {report.topic}
Searches: {report.searches_performed}
Findings:
{findings_text}

Suggested follow-ups: {report.suggested_followups}

Evaluate:
1. Quality of findings (depth, accuracy, relevance)
2. Coverage (what's missing?)
3. Credibility of sources
4. Suggestions for improvement

Be constructive but rigorous."""

        return await self.call_claude(prompt)

    async def _process_followups(
        self,
        report: InternReport,
        directive: ManagerDirective,
        parent_topic: Optional[ResearchTopic] = None,
    ) -> None:
        """Process follow-up suggestions and add worthy ones to the queue."""
        if self.current_depth >= self.max_depth:
            return

        new_depth = (parent_topic.depth + 1) if parent_topic else self.current_depth + 1

        for followup in report.suggested_followups[:3]:  # Limit follow-ups
            # Check if we already have this topic
            existing = [t for t in self.topics_queue if t.topic.lower() == followup.lower()]
            if existing:
                continue

            topic = await self.db.create_topic(
                session_id=self.session_id,
                topic=followup,
                parent_topic_id=parent_topic.id if parent_topic else None,
                depth=new_depth,
                priority=max(1, directive.priority - 1),
            )
            self.topics_queue.append(topic)

    async def _synthesize_report(self) -> ManagerReport:
        """Synthesize all findings into a final report."""
        time_elapsed = self._get_elapsed_minutes()
        time_remaining = self.time_limit_minutes - time_elapsed

        # Get top findings
        key_findings = sorted(
            self.all_findings,
            key=lambda f: f.confidence,
            reverse=True
        )[:20]

        findings_text = "\n".join([
            f"- [{f.finding_type.value}] {f.content}"
            for f in key_findings
        ])

        prompt = f"""Synthesize this research into a final report.

Research Goal: {self.research_goal}

Key Findings:
{findings_text}

Topics Explored: {[t.topic for t in self.completed_topics]}
Topics Remaining: {[t.topic for t in self.topics_queue[:5]]}

Create:
1. A comprehensive summary (2-3 paragraphs)
2. Quality assessment of the research
3. Recommended next steps if more time available

Be thorough and insightful."""

        response = await self.call_claude(prompt)

        return ManagerReport(
            summary=response,
            key_findings=key_findings,
            topics_explored=[t.topic for t in self.completed_topics],
            topics_remaining=[t.topic for t in self.topics_queue],
            quality_assessment="",
            recommended_next_steps=[],
            time_elapsed_minutes=time_elapsed,
            time_remaining_minutes=time_remaining,
        )

    async def run_research(
        self,
        goal: str,
        session_id: int,
        time_limit_minutes: int = 60,
    ) -> ManagerReport:
        """Run a complete research session."""
        self.research_goal = goal
        self.session_id = session_id
        self.time_limit_minutes = time_limit_minutes
        self.start_time = datetime.now()

        # Initialize with the main goal as the first topic
        initial_topic = await self.db.create_topic(
            session_id=session_id,
            topic=goal,
            depth=0,
            priority=10,
        )
        self.topics_queue.append(initial_topic)

        context = {
            "goal": goal,
            "session_id": session_id,
        }

        result = await self.run(context)

        # Return the final report
        if "last_action" in result and result["last_action"].get("action") == "synthesize":
            return result["last_action"]["report"]

        # Generate final report if not already done
        return await self._synthesize_report()

    def reset(self) -> None:
        """Reset the manager state."""
        self.research_goal = ""
        self.session_id = 0
        self.topics_queue = []
        self.completed_topics = []
        self.all_findings = []
        self.all_reports = []
        self.current_depth = 0
        self.start_time = None
        self.state = type(self.state)()
        self.intern.reset()
