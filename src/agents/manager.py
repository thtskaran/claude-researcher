"""Manager agent - coordinates research and critiques findings."""

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from rich.console import Console

from ..audit import init_decision_logger
from ..knowledge import (
    CredibilityScorer,
    HybridKnowledgeGraphStore,
    IncrementalKnowledgeGraph,
    KGFinding,
    ManagerQueryInterface,
)
from ..memory import ExternalMemoryStore, HybridMemory
from ..models.findings import (
    AgentRole,
    Finding,
    InternReport,
    ManagerDirective,
    ManagerReport,
    ResearchSession,
    ResearchTopic,
)
from ..retrieval import get_findings_retriever
from ..storage.database import ResearchDatabase
from ..verification import (
    BatchVerificationResult,
    VerificationConfig,
    VerificationPipeline,
)
from .base import AgentConfig, BaseAgent, DecisionType
from .intern import InternAgent
from .parallel import ParallelInternPool

if TYPE_CHECKING:
    from ..interaction import UserInteraction


class ManagerAgent(BaseAgent):
    """The Manager agent coordinates research and critiques the Intern's work.

    Responsibilities:
    - Break down research goals into specific topics
    - Create directives for the Intern agent
    - Critique and validate findings
    - Identify gaps and request additional research
    - Synthesize findings into coherent reports
    - Track research progress and depth

    Uses Opus model with extended thinking for deep reasoning.
    """

    def __init__(
        self,
        db: ResearchDatabase,
        intern: InternAgent,
        config: AgentConfig | None = None,
        console: Console | None = None,
        pool_size: int = 3,
        use_parallel: bool = True,
        interaction: Optional["UserInteraction"] = None,
    ):
        # Force Opus model for manager's deep reasoning
        if config is None:
            config = AgentConfig()
        config.model = "opus"  # Use Opus for heavy reasoning
        super().__init__(AgentRole.MANAGER, db, config, console)
        self.intern = intern
        self.research_goal: str = ""
        self.session_id: str = ""
        self.topics_queue: list[ResearchTopic] = []
        self.completed_topics: list[ResearchTopic] = []
        self.all_findings: list[Finding] = []
        self.all_reports: list[InternReport] = []
        self.current_depth: int = 0
        self.max_depth: int = 5
        self.start_time: datetime | None = None
        self.time_limit_minutes: int = 0  # Kept for stats only, not used for stopping

        # Locks for thread-safe state access (prevents race conditions in parallel execution)
        self._state_lock = asyncio.Lock()  # Protects topics_queue, all_findings, all_reports

        # User interaction support
        self.interaction = interaction

        # Knowledge graph integration
        self.kg_store = HybridKnowledgeGraphStore(db_path=str(db.db_path).replace(".db", "_kg.db"))
        self.knowledge_graph = IncrementalKnowledgeGraph(
            llm_callback=self._kg_llm_callback,
            store=self.kg_store,
            credibility_audit_callback=self._save_credibility_audit,
            session_id=self.session_id,
        )
        self.kg_query = ManagerQueryInterface(self.kg_store)
        self.credibility_scorer = CredibilityScorer()

        # Parallel execution pool
        self.use_parallel = use_parallel
        self.pool_size = pool_size
        self.intern_pool = (
            ParallelInternPool(
                db=db,
                pool_size=pool_size,
                config=config,
                console=console,
            )
            if use_parallel
            else None
        )

        # Hybrid memory for long research sessions
        self.memory = HybridMemory(
            max_recent_tokens=8000,
            summary_threshold=0.8,
            llm_callback=self._memory_llm_callback,
        )
        self.external_memory = ExternalMemoryStore(
            db_path=str(db.db_path).replace(".db", "_memory.db")
        )

        # Hybrid retrieval for semantic search over findings
        self.findings_retriever = get_findings_retriever(
            persist_dir=str(db.db_path).replace(".db", "_retrieval"),
            use_reranker=True,  # Quality is priority
        )

        # Verification pipeline for hallucination reduction
        self.verification_config = VerificationConfig()
        self.verification_pipeline = VerificationPipeline(
            llm_callback=self._verification_llm_callback,
            knowledge_graph=self.knowledge_graph,
            search_callback=self._verification_search_callback,
            config=self.verification_config,
        )
        # Pass pipeline to intern and intern pool
        self.intern.verification_pipeline = self.verification_pipeline
        if self.intern_pool:
            self.intern_pool.set_verification_pipeline(self.verification_pipeline)

        # Track batch verification results for reports
        self.last_batch_verification: BatchVerificationResult | None = None

    async def _kg_llm_callback(
        self,
        prompt: str,
        **kwargs,
    ) -> str | dict | list:
        """LLM callback for knowledge graph extraction (uses faster model)."""
        original_model = self.config.model
        self.config.model = "sonnet"
        try:
            return await self.call_claude(prompt, **kwargs)
        finally:
            self.config.model = original_model

    async def _save_credibility_audit(self, audit_data: dict) -> None:
        """Save credibility audit to database (fire-and-forget)."""
        try:
            await self.db.save_credibility_audit(
                session_id=self.session_id,
                finding_id=audit_data.get("finding_id"),
                url=audit_data.get("url", ""),
                domain=audit_data.get("domain", ""),
                final_score=audit_data.get("final_score", 0.0),
                domain_authority_score=audit_data.get("domain_authority_score", 0.0),
                recency_score=audit_data.get("recency_score", 0.5),
                source_type_score=audit_data.get("source_type_score", 0.6),
                https_score=audit_data.get("https_score", 0.5),
                path_depth_score=audit_data.get("path_depth_score", 0.8),
                credibility_label=audit_data.get("credibility_label", "Medium"),
            )
        except Exception as e:
            # Log error but don't stop processing
            self._log(f"[Credibility Audit Error] Failed to save audit: {e}", style="bold red")
            import traceback

            traceback.print_exc()

    async def _memory_llm_callback(self, prompt: str) -> str:
        """LLM callback for memory summarization (uses faster model)."""
        original_model = self.config.model
        self.config.model = "haiku"  # Fast model for summarization
        try:
            return await self.call_claude(prompt)
        finally:
            self.config.model = original_model

    async def _verification_llm_callback(
        self,
        prompt: str,
        model: str = "sonnet",
        output_format: dict | None = None,
    ) -> str | dict | list:
        """LLM callback for verification (model specified by verification pipeline).

        Args:
            prompt: The prompt to send
            model: Model to use (e.g. "haiku", "sonnet")
            output_format: Optional JSON schema for structured output
        """
        original_model = self.config.model
        self.config.model = model
        try:
            return await self.call_claude(prompt, output_format=output_format)
        finally:
            self.config.model = original_model

    async def _verification_search_callback(self, query: str) -> list[dict]:
        """Web search callback for CoVe/CRITIC independent verification.

        Uses the intern's search tool (Bright Data) to fetch evidence so
        the verification pipeline can ground answers in real web data
        instead of relying solely on parametric knowledge.

        Scrapes the top result's full page content for richer evidence
        when available, so the LLM can evaluate actual source material
        rather than just search snippets.
        """
        try:
            results, _ = await self.intern.search_tool.search(query)
            if not results:
                return []

            output = []
            for r in results[:5]:
                output.append(
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                    }
                )

            # Scrape the top result for richer evidence context
            try:
                page_content = await self.intern.search_tool.fetch_page(results[0].url)
                if page_content and len(page_content) > 100:
                    output[0]["content"] = page_content[:1500]
            except Exception:
                pass  # Snippet fallback is fine

            return output
        except Exception:
            return []

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
        iterations_remaining = self.config.max_iterations - self.state.iteration

        # Check for user guidance messages
        user_guidance = ""
        if self.interaction:
            messages = self.interaction.get_pending_messages()
            if messages:
                guidance_texts = [m.content for m in messages]
                user_guidance = (
                    "USER GUIDANCE (please incorporate this into your research):\n"
                    + "\n".join(f"- {g}" for g in guidance_texts)
                )
                self._log(
                    f"[USER GUIDANCE] Received {len(messages)} message(s)", style="bold yellow"
                )
                for m in messages:
                    self._log(
                        f"  → {m.content[:100]}{'...' if len(m.content) > 100 else ''}",
                        style="yellow",
                    )

        # Summarize current state
        findings_summary = self._summarize_findings()

        # Get knowledge graph insights
        kg_summary = self.kg_query.get_research_summary()
        research_directions = self.kg_query.get_next_research_directions()

        # Use hybrid retrieval to find semantically relevant past findings
        relevant_findings_text = ""
        if self.findings_retriever.count() > 0:
            try:
                # Search for findings relevant to the research goal
                relevant = self.findings_retriever.search(
                    query=self.research_goal,
                    limit=5,
                    session_id=self.session_id,
                )
                if relevant:
                    relevant_findings_text = "Most relevant findings (via semantic search):\n"
                    for r in relevant:
                        relevant_findings_text += f"- [{r.finding.finding_type.value}] {r.finding.content[:200]}... (score: {r.score:.2f})\n"
            except Exception as e:
                self._log(f"[RETRIEVAL] Search error: {e}", style="dim")

        # Get memory context for continuity
        memory_context = self.memory.get_context_for_prompt(max_tokens=2000)

        prompt = f"""Research Goal: {self.research_goal}

{user_guidance}

Current Status:
- Iteration: {self.state.iteration}/{self.config.max_iterations} ({iterations_remaining} remaining)
- Time elapsed: {time_elapsed:.1f} minutes
- Topics completed: {len(self.completed_topics)}
- Topics in queue: {len(self.topics_queue)}
- Total findings: {len(self.all_findings)}
- Current depth: {self.current_depth}/{self.max_depth}

Last report from Intern:
{context.get("last_report_summary", "No report yet")}

Findings summary:
{findings_summary}

{relevant_findings_text}

{kg_summary}

Suggested research directions from knowledge analysis:
{chr(10).join(["- " + d for d in research_directions[:5]]) if research_directions else "None yet"}

{f"Session Memory Context:{chr(10)}{memory_context}" if memory_context else ""}

What should I do next? Consider:
1. Are there gaps in the research identified by knowledge graph analysis?
2. Should I go deeper on any topic?
3. Are there contradictions that need resolution?
4. Should I ask for verification of any findings?
5. Is it time to synthesize and report?

Think step by step about the best next action."""

        thought = await self.call_claude(prompt, use_thinking=True)

        # Track in memory
        await self.memory.add_message(
            role="assistant",
            content=f"Reasoning: {thought[:500]}...",
            metadata={"type": "thought", "iteration": self.state.iteration},
        )

        # Compress memory if needed
        await self.memory.maybe_compress()

        return thought

    async def act(self, thought: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute management actions based on thinking."""
        # Check if pause was requested before starting any long operation
        if self._pause_requested:
            return {"action": "paused"}

        # Periodic checkpoint for crash recovery
        await self._maybe_periodic_checkpoint()

        # Check if we should synthesize and stop
        if self._should_synthesize(thought):
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
                self._log(
                    f"[DIRECTIVE] {directive.action.upper()}: {directive.topic}", style="bold green"
                )
                self._log(f"  Instructions: {directive.instructions}", style="dim")
                self._log(
                    f"  Priority: {directive.priority}/10 | Max Searches: {directive.max_searches}",
                    style="dim",
                )
                self._log("═" * 70, style="bold blue")

                # Check pause before long intern operation
                if self._pause_requested:
                    return {"action": "paused"}

                # Execute the directive via the intern
                intern_report = await self.intern.execute_directive(directive, self.session_id)
                async with self._state_lock:
                    self.all_reports.append(intern_report)
                    self.all_findings.extend(intern_report.findings)

                # Process findings into knowledge graph
                await self._process_findings_to_kg(intern_report.findings)

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
                    self._log(
                        f"[Follow-up Topics Added: {len(intern_report.suggested_followups)}]",
                        style="cyan",
                    )
                    for ft in intern_report.suggested_followups[:3]:
                        self._log(f"  → {ft}", style="cyan")

                return {
                    "action": "intern_task",
                    "directive": directive,
                    "report": intern_report,
                    "critique": critique,
                }

        # Check pending topics - use parallel execution if multiple topics queued
        if self.topics_queue:
            # Use parallel execution if we have multiple topics and pool is available
            if len(self.topics_queue) >= 2 and self.intern_pool:
                # Pop multiple topics for parallel execution (with lock for thread safety)
                async with self._state_lock:
                    topics_to_run = []
                    for _ in range(min(self.pool_size, len(self.topics_queue))):
                        if self.topics_queue:
                            topics_to_run.append(self.topics_queue.pop(0))

                # Log topic selection decision
                await self._log_decision(
                    session_id=self.session_id,
                    decision_type=DecisionType.TOPIC_SELECTION,
                    decision_outcome="parallel_execution",
                    reasoning=f"Selected {len(topics_to_run)} topics for parallel research",
                    inputs={
                        "queue_size": len(self.topics_queue) + len(topics_to_run),
                        "selected_topics": [t.topic for t in topics_to_run],
                        "depths": [t.depth for t in topics_to_run],
                    },
                    metrics={
                        "findings_count": len(self.all_findings),
                        "completed_topics": len(self.completed_topics),
                    },
                )

                # Check pause before long parallel operation
                if self._pause_requested:
                    # Put topics back in queue
                    async with self._state_lock:
                        self.topics_queue = topics_to_run + self.topics_queue
                    return {"action": "paused"}

                self._log("═" * 70, style="bold blue")
                self._log(
                    f"[PARALLEL TOPICS] Running {len(topics_to_run)} topics in parallel",
                    style="bold green",
                )
                for t in topics_to_run:
                    self._log(f"  • {t.topic}", style="dim")
                self._log("═" * 70, style="bold blue")

                await self._run_parallel_topics(topics_to_run, max_parallel=self.pool_size)

                # Track in memory
                await self.memory.add_message(
                    role="system",
                    content=f"Completed parallel research on {len(topics_to_run)} topics",
                    metadata={"topics": [t.topic for t in topics_to_run]},
                )

                return {
                    "action": "parallel_topics",
                    "topics": topics_to_run,
                    "findings_count": len(self.all_findings),
                }

            # Single topic - use regular intern
            async with self._state_lock:
                topic = self.topics_queue.pop(0)

            # Log topic selection decision
            await self._log_decision(
                session_id=self.session_id,
                decision_type=DecisionType.TOPIC_SELECTION,
                decision_outcome="single_topic",
                reasoning=f"Selected topic '{topic.topic}' from queue",
                inputs={
                    "queue_size": len(self.topics_queue) + 1,
                    "selected_topic": topic.topic,
                    "depth": topic.depth,
                    "priority": topic.priority,
                },
                metrics={
                    "findings_count": len(self.all_findings),
                    "completed_topics": len(self.completed_topics),
                },
            )

            directive = ManagerDirective(
                action="search",
                topic=topic.topic,
                instructions=f"Research this topic thoroughly: {topic.topic}",
                priority=topic.priority,
                max_searches=5,
            )

            # Check pause before long intern operation
            if self._pause_requested:
                # Put topic back in queue
                async with self._state_lock:
                    self.topics_queue.insert(0, topic)
                return {"action": "paused"}

            self._log("═" * 70, style="bold blue")
            self._log(f"[QUEUED TOPIC] Depth {topic.depth}: {topic.topic}", style="bold green")
            self._log(
                f"  Priority: {topic.priority}/10 | Remaining in queue: {len(self.topics_queue)}",
                style="dim",
            )
            self._log("═" * 70, style="bold blue")

            intern_report = await self.intern.execute_directive(directive, self.session_id)
            async with self._state_lock:
                self.all_reports.append(intern_report)
                self.all_findings.extend(intern_report.findings)
                self.completed_topics.append(topic)
                self.current_depth = max(self.current_depth, topic.depth)

            # Process findings into knowledge graph
            await self._process_findings_to_kg(intern_report.findings)

            # Track in memory
            await self.memory.add_message(
                role="system",
                content=f"Completed research on: {topic.topic} - {len(intern_report.findings)} findings",
                metadata={"topic": topic.topic, "findings": len(intern_report.findings)},
            )

            await self.db.update_topic_status(topic.id, "completed", len(intern_report.findings))

            critique = await self._critique_report(intern_report)

            # Show critique
            self._log("─" * 70, style="dim")
            self._log("[MANAGER CRITIQUE]", style="bold magenta")
            self.console.print(critique)
            self._log("─" * 70, style="dim")

            await self._process_followups(intern_report, directive, topic)

            # Show follow-up topics added
            if intern_report.suggested_followups:
                self._log(
                    f"[Follow-up Topics Added: {len(intern_report.suggested_followups)}]",
                    style="cyan",
                )
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

    async def _maybe_periodic_checkpoint(self) -> None:
        """Fire-and-forget checkpoint every 2 iterations for crash recovery."""
        if self.state.iteration % 2 == 0 and self.session_id:
            try:
                session = await self.db.get_session(self.session_id)
                if session:
                    asyncio.create_task(self._periodic_checkpoint(session))
            except Exception:
                pass

    async def observe(self, action_result: dict[str, Any]) -> str:
        """Process the result of a management action."""
        action = action_result.get("action")

        if action == "paused":
            return "Research paused by user request"

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

        if action == "parallel_topics":
            topics = action_result.get("topics", [])
            findings_count = action_result.get("findings_count", 0)
            return f"""Parallel research completed:
- Topics researched: {len(topics)}
- Total findings so far: {findings_count}
- Topics: {[t.topic for t in topics]}"""

        return "Unknown action"

    def is_done(self, context: dict[str, Any]) -> bool:
        """Check if the Manager should stop (iteration-based)."""
        # Check if synthesis complete
        last_action = context.get("last_action", {})
        if last_action.get("action") == "synthesize":
            return True

        # Check max iterations (the only stopping condition now)
        if self.state.iteration >= self.config.max_iterations:
            self._log(
                f"Iteration limit reached ({self.state.iteration}/{self.config.max_iterations})",
                style="yellow",
            )
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

    async def _maybe_ask_user(
        self,
        question: str,
        context: str = "",
        options: list[str] | None = None,
    ) -> str | None:
        """Ask the user a question during research if interaction is enabled.

        This is non-blocking with a timeout - if the user doesn't respond,
        research continues autonomously.

        Args:
            question: The question to ask
            context: Context about why this is being asked
            options: Optional suggested answers

        Returns:
            User's response, or None if no interaction or timeout
        """
        if not self.interaction:
            return None

        response = await self.interaction.ask_with_timeout(
            question=question,
            context=context,
            options=options,
        )

        if response:
            # Log that we got a response
            self._log(
                f"[USER RESPONSE] {response[:100]}{'...' if len(response) > 100 else ''}",
                style="bold green",
            )

            # Add to memory for context
            await self.memory.add_message(
                role="user",
                content=f"User guidance: {response}",
                metadata={"type": "mid_research_response"},
            )

        return response

    async def _process_findings_to_kg(self, findings: list[Finding]) -> None:
        """Process findings into the knowledge graph and hybrid retrieval index.

        Uses batch processing for speed (multiple findings per LLM call)
        while still building the full KG that agents can query during research.
        Also indexes findings for semantic search via hybrid retrieval.

        Args:
            findings: List of findings to process
        """
        if not findings:
            return

        self._log(f"[KG] Processing {len(findings)} findings into knowledge graph", style="dim")

        # Index findings for hybrid retrieval (semantic + lexical search)
        try:
            self.findings_retriever.add_findings(
                findings=findings,
                session_id=self.session_id,
            )
            self._log(
                f"[RETRIEVAL] Indexed {len(findings)} findings for semantic search", style="dim"
            )
        except Exception as e:
            self._log(f"[RETRIEVAL] Error indexing findings: {e}", style="yellow")

        # Convert all findings to KGFindings
        kg_findings = []
        for finding in findings:
            try:
                kg_finding = KGFinding(
                    id=str(finding.id or hash(finding.content)),
                    content=finding.content,
                    source_url=finding.source_url or "",
                    source_title=finding.source_url.split("/")[-1] if finding.source_url else "",
                    timestamp=finding.created_at.isoformat(),
                    credibility_score=finding.confidence,
                    finding_type=finding.finding_type.value,
                    search_query=finding.search_query,
                )
                kg_findings.append(kg_finding)
            except Exception as e:
                self._log(f"[KG] Error converting finding: {e}", style="dim")

        # Use batch processing for speed (5 findings per LLM call)
        if len(kg_findings) > 3:
            result = await self.knowledge_graph.add_findings_batch(kg_findings, batch_size=5)
            self._log(
                f"[KG] Extracted {result['total_entities']} entities, "
                f"{result['total_relations']} relations",
                style="dim",
            )
            if result["total_contradictions"] > 0:
                self._log(
                    f"[KG] Contradictions detected: {result['total_contradictions']}",
                    style="yellow",
                )
        else:
            # Process individually for small batches
            for kg_finding in kg_findings:
                try:
                    result = await self.knowledge_graph.add_finding(kg_finding, fast_mode=True)
                    if result.get("contradictions_found", 0) > 0:
                        self._log(
                            f"[KG] Contradiction detected: {result['contradictions_found']} conflicts",
                            style="yellow",
                        )
                except Exception as e:
                    self._log(f"[KG] Error processing finding: {e}", style="dim")

    def _should_synthesize(self, thought: str) -> bool:
        """Determine if it's time to synthesize results (iteration-based)."""
        # Never synthesize if we have no findings
        if not self.all_findings:
            return False

        iterations_remaining = self.config.max_iterations - self.state.iteration

        # On last iteration, always synthesize if we have findings
        if iterations_remaining <= 0 and self.all_findings:
            asyncio.create_task(
                self._log_decision(
                    session_id=self.session_id,
                    decision_type=DecisionType.SYNTHESIS_TRIGGER,
                    decision_outcome="triggered_last_iteration",
                    reasoning=f"Last iteration with {len(self.all_findings)} findings",
                    inputs={"findings_count": len(self.all_findings)},
                    metrics={
                        "iteration": self.state.iteration,
                        "max_iterations": self.config.max_iterations,
                    },
                )
            )
            return True

        # Don't allow early synthesis until at least 80% of iterations are done
        iteration_pct = (self.state.iteration / self.config.max_iterations) * 100
        if iteration_pct < 80:
            return False

        # Need at least some findings before synthesizing
        if len(self.all_findings) < 3:
            return False

        # Check explicit signals from LLM
        thought_lower = thought.lower()
        synthesis_signals = [
            "time to synthesize",
            "final report",
            "wrap up now",
            "conclude the research",
            "sufficient coverage",
            "enough findings",
        ]
        should_synthesize = any(signal in thought_lower for signal in synthesis_signals)

        if should_synthesize:
            asyncio.create_task(
                self._log_decision(
                    session_id=self.session_id,
                    decision_type=DecisionType.SYNTHESIS_TRIGGER,
                    decision_outcome="triggered_explicit_signal",
                    reasoning=thought[:500],
                    inputs={"findings_count": len(self.all_findings)},
                    metrics={
                        "iteration": self.state.iteration,
                        "max_iterations": self.config.max_iterations,
                        "topics_completed": len(self.completed_topics),
                    },
                )
            )

        return should_synthesize

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

    async def _create_directive(self, thought: str) -> ManagerDirective | None:
        """Create a directive for the Intern based on reasoning."""
        prompt = (
            f"Based on this reasoning:\n{thought}\n\nCreate a directive for the Research Intern."
        )

        schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "deep_dive", "verify"],
                    },
                    "topic": {"type": "string"},
                    "instructions": {"type": "string"},
                    "priority": {"type": "integer"},
                    "max_searches": {"type": "integer"},
                },
                "required": [
                    "action",
                    "topic",
                    "instructions",
                    "priority",
                    "max_searches",
                ],
            },
        }

        try:
            response = await self.call_claude(
                prompt,
                output_format=schema,
            )

            if isinstance(response, dict):
                data = response
            else:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start == -1 or end <= start:
                    return None
                data = json.loads(response[start:end])

            directive = ManagerDirective(
                action=data.get("action", "search"),
                topic=data.get("topic", ""),
                instructions=data.get("instructions", ""),
                priority=data.get("priority", 5),
                max_searches=data.get("max_searches", 5),
            )

            # Log directive creation decision
            await self._log_decision(
                session_id=self.session_id,
                decision_type=DecisionType.DIRECTIVE_CREATE,
                decision_outcome=directive.action,
                reasoning=thought[:500],
                inputs={
                    "action": directive.action,
                    "topic": directive.topic,
                    "priority": directive.priority,
                    "max_searches": directive.max_searches,
                },
                metrics={
                    "findings_count": len(self.all_findings),
                    "iterations_remaining": self.config.max_iterations - self.state.iteration,
                },
            )

            return directive
        except (json.JSONDecodeError, KeyError):
            pass

        return None

    async def _critique_report(self, report: InternReport) -> str:
        """Critique an Intern's report with batch verification."""
        # Run batch verification on findings if not already verified
        unverified = [f for f in report.findings if not f.verification_status]
        if unverified and self.verification_config.enable_batch_verification:
            self._log(
                f"[VERIFY] Running batch verification on {len(unverified)} findings...", style="dim"
            )
            batch_result = await self.verification_pipeline.verify_batch(
                unverified, self.session_id
            )
            self.last_batch_verification = batch_result

            # Update findings with verification results
            for result in batch_result.results:
                for f in report.findings:
                    if str(f.id or hash(f.content)) == result.finding_id:
                        f.original_confidence = f.confidence
                        f.confidence = result.verified_confidence
                        f.verification_status = result.verification_status.value
                        f.verification_method = result.verification_method.value
                        f.kg_support_score = result.kg_support_score

                        # Update in database
                        if f.id:
                            await self.db.update_finding_verification(
                                finding_id=f.id,
                                verification_status=f.verification_status,
                                verification_method=f.verification_method,
                                kg_support_score=f.kg_support_score,
                                original_confidence=f.original_confidence,
                                new_confidence=f.confidence,
                            )

                            # Save detailed verification result for UI
                            await self.db.save_verification_result(
                                session_id=self.session_id,
                                finding_id=f.id,
                                result_dict={
                                    "original_confidence": result.original_confidence,
                                    "verified_confidence": result.verified_confidence,
                                    "verification_status": result.verification_status.value,
                                    "verification_method": result.verification_method.value,
                                    "consistency_score": result.consistency_score,
                                    "kg_support_score": result.kg_support_score,
                                    "kg_entity_matches": result.kg_entity_matches,
                                    "kg_supporting_relations": result.kg_supporting_relations,
                                    "critic_iterations": result.critic_iterations,
                                    "corrections_made": result.corrections_made,
                                    "questions_asked": [
                                        {
                                            "question": q.question,
                                            "aspect": q.aspect,
                                            "independent_answer": q.independent_answer,
                                            "supports_original": q.supports_original,
                                            "confidence": q.confidence,
                                        }
                                        for q in result.questions_asked
                                    ],
                                    "external_verification_used": result.external_verification_used,
                                    "contradictions": result.contradictions,
                                    "verification_time_ms": result.verification_time_ms,
                                    "error": result.error,
                                },
                            )
                        break

            # Log verification summary
            self._log(
                f"[VERIFY] Results: {batch_result.verified_count} verified, "
                f"{batch_result.flagged_count} flagged, {batch_result.rejected_count} rejected",
                style="dim",
            )

        # Separate findings by verification status
        verified = [f for f in report.findings if f.verification_status == "verified"]
        flagged = [f for f in report.findings if f.verification_status == "flagged"]
        rejected = [f for f in report.findings if f.verification_status == "rejected"]

        findings_text = "\n".join(
            [
                f"- [{f.finding_type.value}] {f.content} (confidence: {f.confidence:.0%}, status: {f.verification_status or 'pending'})"
                for f in report.findings[:10]
            ]
        )

        verification_summary = ""
        if verified or flagged or rejected:
            verification_summary = f"""
Verification Summary:
- Verified (high confidence): {len(verified)}
- Flagged (needs review): {len(flagged)}
- Rejected (low confidence): {len(rejected)}
"""

        prompt = f"""Critique this research report:

Topic: {report.topic}
Searches: {report.searches_performed}
{verification_summary}
Findings:
{findings_text}

Suggested follow-ups: {report.suggested_followups}

Evaluate:
1. Quality of findings (depth, accuracy, relevance)
2. Verification status - pay special attention to flagged and rejected findings
3. Coverage (what's missing?)
4. Credibility of sources
5. Suggestions for improvement

Be constructive but rigorous. Flag any rejected findings that should be re-researched."""

        return await self.call_claude(prompt)

    async def _process_followups(
        self,
        report: InternReport,
        directive: ManagerDirective,
        parent_topic: ResearchTopic | None = None,
    ) -> None:
        """Process follow-up suggestions and add worthy ones to the queue."""
        if self.current_depth >= self.max_depth:
            return

        new_depth = (parent_topic.depth + 1) if parent_topic else self.current_depth + 1

        for followup in report.suggested_followups[:3]:  # Limit follow-ups
            # Filter out meta-questions/clarifying questions
            followup_lower = followup.lower()
            is_meta_question = any(
                phrase in followup_lower
                for phrase in [
                    "please provide",
                    "what information",
                    "could you clarify",
                    "what are you looking for",
                    "what topic",
                    "what subject",
                    "what would you like",
                    "can you specify",
                    "please specify",
                ]
            )
            if is_meta_question:
                continue

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
            async with self._state_lock:
                self.topics_queue.append(topic)

    async def _synthesize_report(self) -> ManagerReport:
        """Synthesize all findings into a final report with verification awareness."""
        time_elapsed = self._get_elapsed_minutes()

        # Run batch verification on findings that haven't had thorough verification.
        # Includes: unverified findings, streaming-only verified findings (lightweight
        # CoVe only), and findings that were rejected/flagged by streaming and deserve
        # a second look with the full pipeline (CoVe batch + KG + CRITIC).
        _streaming_methods = {"streaming", ""}
        needs_batch = [
            f
            for f in self.all_findings
            if not f.verification_status or (f.verification_method or "") in _streaming_methods
        ]
        if needs_batch and self.verification_config.enable_batch_verification:
            self._log(f"[VERIFY] Batch verification on {len(needs_batch)} findings...", style="dim")
            batch_result = await self.verification_pipeline.verify_batch(
                needs_batch, self.session_id
            )
            self.last_batch_verification = batch_result

            # Update findings with verification results and save to verification_results table
            for result in batch_result.results:
                for f in self.all_findings:
                    if str(f.id or hash(f.content)) == result.finding_id:
                        f.original_confidence = f.confidence
                        f.confidence = result.verified_confidence
                        f.verification_status = result.verification_status.value
                        f.verification_method = result.verification_method.value
                        f.kg_support_score = result.kg_support_score

                        # Update in database
                        if f.id:
                            await self.db.update_finding_verification(
                                finding_id=f.id,
                                verification_status=f.verification_status,
                                verification_method=f.verification_method,
                                kg_support_score=f.kg_support_score,
                                original_confidence=f.original_confidence,
                                new_confidence=f.confidence,
                            )

                            # Save detailed verification result for UI
                            await self.db.save_verification_result(
                                session_id=self.session_id,
                                finding_id=f.id,
                                result_dict={
                                    "original_confidence": result.original_confidence,
                                    "verified_confidence": result.verified_confidence,
                                    "verification_status": result.verification_status.value,
                                    "verification_method": result.verification_method.value,
                                    "consistency_score": result.consistency_score,
                                    "kg_support_score": result.kg_support_score,
                                    "kg_entity_matches": result.kg_entity_matches,
                                    "kg_supporting_relations": result.kg_supporting_relations,
                                    "critic_iterations": result.critic_iterations,
                                    "corrections_made": result.corrections_made,
                                    "questions_asked": [
                                        {
                                            "question": q.question,
                                            "aspect": q.aspect,
                                            "independent_answer": q.independent_answer,
                                            "supports_original": q.supports_original,
                                            "confidence": q.confidence,
                                        }
                                        for q in result.questions_asked
                                    ],
                                    "external_verification_used": result.external_verification_used,
                                    "contradictions": result.contradictions,
                                    "verification_time_ms": result.verification_time_ms,
                                    "error": result.error,
                                },
                            )
                        break

        # Separate findings by verification status
        verified_findings = [f for f in self.all_findings if f.verification_status == "verified"]
        flagged_findings = [f for f in self.all_findings if f.verification_status == "flagged"]
        rejected_findings = [f for f in self.all_findings if f.verification_status == "rejected"]
        other_findings = [
            f
            for f in self.all_findings
            if f.verification_status not in ["verified", "flagged", "rejected"]
        ]

        # Priority: verified > flagged > unverified > rejected
        # Weight by calibrated confidence
        priority_findings = (
            sorted(verified_findings, key=lambda f: f.confidence, reverse=True)
            + sorted(flagged_findings, key=lambda f: f.confidence, reverse=True)
            + sorted(other_findings, key=lambda f: f.confidence, reverse=True)
        )
        key_findings = priority_findings[:20]

        findings_text = "\n".join(
            [
                f"- [{f.finding_type.value}] {f.content} (verified: {f.verification_status or 'pending'}, confidence: {f.confidence:.0%})"
                for f in key_findings
            ]
        )

        # Verification context for synthesis
        verification_context = ""
        if verified_findings or flagged_findings or rejected_findings:
            verification_context = f"""
Verification Summary:
- High confidence (verified): {len(verified_findings)} findings
- Medium confidence (flagged for review): {len(flagged_findings)} findings
- Low confidence (rejected): {len(rejected_findings)} findings
- Unverified: {len(other_findings)} findings

Note: Prioritize verified findings in your synthesis. Flagged findings may need additional context.
Rejected findings ({len(rejected_findings)}) have low confidence and should not be primary conclusions.
"""

        prompt = f"""Synthesize this research into a final report.

Research Goal: {self.research_goal}
{verification_context}
Key Findings (sorted by verification confidence):
{findings_text}

Topics Explored: {[t.topic for t in self.completed_topics]}
Topics Remaining: {[t.topic for t in self.topics_queue[:5]]}

Create:
1. A comprehensive summary (2-3 paragraphs) - base conclusions on verified/high-confidence findings
2. Quality assessment of the research (including note on verification rates)
3. Recommended next steps if more time available

Be thorough and insightful. Note where findings have lower confidence."""

        response = await self.call_claude(prompt, use_thinking=True)

        return ManagerReport(
            summary=response,
            key_findings=key_findings,
            topics_explored=[t.topic for t in self.completed_topics],
            topics_remaining=[t.topic for t in self.topics_queue],
            quality_assessment="",
            recommended_next_steps=[],
            time_elapsed_minutes=time_elapsed,
            iterations_completed=self.state.iteration,
        )

    async def _run_parallel_initial_research(self, goal: str, max_aspects: int = 3) -> None:
        """Run parallel initial research to quickly gather broad findings.

        Decomposes the goal into distinct aspects and researches them in parallel
        using the intern pool.

        Args:
            goal: The main research goal
            max_aspects: Maximum number of parallel research threads
        """
        if not self.intern_pool:
            return

        self._log("=" * 70, style="bold cyan")
        self._log("[PARALLEL RESEARCH] Starting parallel initial research phase", style="bold cyan")
        self._log("=" * 70, style="bold cyan")

        # Record in memory
        await self.memory.add_message(
            role="system",
            content=f"Starting parallel research on: {goal}",
            metadata={"phase": "parallel_init"},
        )

        # Decompose goal into aspects first
        aspects = await self.intern_pool._decompose_goal(
            goal=goal,
            llm_callback=self._kg_llm_callback,
            max_aspects=max_aspects,
        )

        self._log(f"[PARALLEL] Decomposed into {len(aspects)} aspects:", style="cyan")
        for aspect in aspects:
            self._log(f"  • {aspect}", style="dim")

        # Create topics in database for tracking
        aspect_topics = []
        for aspect in aspects:
            topic = await self.db.create_topic(
                session_id=self.session_id,
                topic=aspect,
                depth=1,
                priority=9,
            )
            aspect_topics.append(topic)

        # Create directives from aspects
        directives = [
            ManagerDirective(
                action="search",
                topic=aspect,
                instructions=f"Research this specific aspect thoroughly: {aspect}",
                priority=8,
                max_searches=5,
            )
            for aspect in aspects
        ]

        # Check pause before long parallel operation
        if self._pause_requested:
            return

        # Execute in parallel
        result = await self.intern_pool.research_parallel(directives, self.session_id)

        # Process results (with lock for thread safety)
        async with self._state_lock:
            self.all_findings.extend(result.total_findings)
            self.all_reports.extend(result.reports)

            # Mark topics as completed and track them
            for topic in aspect_topics:
                self.completed_topics.append(topic)

        # Update DB status outside lock
        for topic in aspect_topics:
            await self.db.update_topic_status(
                topic.id, "completed", len(result.total_findings) // len(aspects)
            )

        # Update depth tracking
        self.current_depth = max(self.current_depth, 1)

        # Process all findings into KG in real-time (fast mode)
        await self._process_findings_to_kg(result.total_findings)

        # Store findings summary in external memory for later retrieval
        if result.total_findings:
            findings_summary = "\n".join(
                [f"- {f.content[:200]}" for f in result.total_findings[:20]]
            )
            await self.external_memory.store(
                session_id=self.session_id,
                content=f"Parallel research findings:\n{findings_summary}",
                memory_type="finding",
                tags=["parallel", "initial"],
                metadata={"count": len(result.total_findings)},
            )

        # Record completion in memory
        await self.memory.add_message(
            role="system",
            content=f"Parallel research complete: {len(result.total_findings)} findings from {result.total_searches} searches in {result.execution_time_seconds:.1f}s",
            metadata={"phase": "parallel_complete", "findings_count": len(result.total_findings)},
        )

        self._log("=" * 70, style="bold cyan")
        self._log(
            f"[PARALLEL RESEARCH] Complete: {len(result.total_findings)} findings, "
            f"{result.total_searches} searches, {result.execution_time_seconds:.1f}s",
            style="bold green",
        )
        if result.errors:
            self._log(f"  Errors: {len(result.errors)}", style="yellow")
        self._log("=" * 70, style="bold cyan")

    async def _run_parallel_topics(
        self, topics: list[ResearchTopic], max_parallel: int = 3
    ) -> None:
        """Run multiple queued topics in parallel.

        Args:
            topics: List of topics to research in parallel
            max_parallel: Maximum number to run at once
        """
        if not self.intern_pool or not topics:
            return

        # Check pause before starting
        if self._pause_requested:
            return

        # Create directives from topics
        from ..models.findings import ManagerDirective

        directives = [
            ManagerDirective(
                action="search",
                topic=topic.topic,
                instructions=f"Research this topic thoroughly: {topic.topic}",
                priority=topic.priority,
                max_searches=5,
            )
            for topic in topics[:max_parallel]
        ]

        self._log(f"[PARALLEL] Running {len(directives)} topics in parallel", style="cyan")

        result = await self.intern_pool.research_parallel(directives, self.session_id)

        # Process results (with lock for thread safety)
        async with self._state_lock:
            self.all_findings.extend(result.total_findings)
            self.all_reports.extend(result.reports)

            # Mark topics as completed
            for topic in topics[:max_parallel]:
                self.completed_topics.append(topic)
                self.current_depth = max(self.current_depth, topic.depth)

        # Update database outside lock
        findings_per_topic = len(result.total_findings) // max(len(topics[:max_parallel]), 1)
        for topic in topics[:max_parallel]:
            await self.db.update_topic_status(topic.id, "completed", findings_per_topic)

        # Process findings to KG
        await self._process_findings_to_kg(result.total_findings)

        # Compress memory if needed
        await self.memory.maybe_compress()

    async def checkpoint_state(self, session: ResearchSession) -> None:
        """Save Manager orchestration state to DB for pause/crash recovery."""
        elapsed = self._get_elapsed_minutes() * 60  # Convert to seconds
        session.elapsed_seconds = elapsed
        session.iteration_count = self.state.iteration
        session.phase = self._current_phase
        session.paused_at = datetime.now()
        session.status = "paused"
        await self.db.update_session(session)
        self._log(
            f"[CHECKPOINT] Saved state: elapsed={elapsed:.0f}s, "
            f"iteration={self.state.iteration}, phase={self._current_phase}"
        )

    async def _periodic_checkpoint(self, session: ResearchSession) -> None:
        """Update elapsed time and iteration in DB (fire-and-forget)."""
        try:
            elapsed = self._get_elapsed_minutes() * 60
            session.elapsed_seconds = elapsed
            session.iteration_count = self.state.iteration
            session.phase = self._current_phase
            await self.db.update_session(session)
        except Exception:
            pass  # Non-critical

    async def restore_state(self, session: ResearchSession) -> None:
        """Rebuild Manager state from DB for resume after pause/crash."""
        self._log("[RESTORE] Rebuilding Manager state from database...")

        # Reload all findings
        self.all_findings = await self.db.get_session_findings(session.id)
        self._log(f"[RESTORE] Loaded {len(self.all_findings)} findings")

        # Reset in_progress topics to pending (they were mid-execution)
        await self.db.reset_in_progress_topics(session.id)

        # Reconstruct topics from DB
        all_topics = await self.db.get_all_topics(session.id)
        self.topics_queue = [t for t in all_topics if t.status == "pending"]
        self.completed_topics = [t for t in all_topics if t.status == "completed"]
        self._log(
            f"[RESTORE] Queue: {len(self.topics_queue)} pending, "
            f"{len(self.completed_topics)} completed"
        )

        # Restore timing: set start_time so _get_elapsed_minutes() returns correct total
        from datetime import timedelta

        self.start_time = datetime.now() - timedelta(seconds=session.elapsed_seconds)

        # Restore iteration count
        self.state.iteration = session.iteration_count

        # Compute current depth from completed topics
        self.current_depth = max((t.depth for t in self.completed_topics), default=0)

        # Re-initialize decision logger
        await init_decision_logger(self.db)

        # Re-initialize memory context
        await self.memory.add_message(
            role="system",
            content=(
                f"Resumed research session. "
                f"{len(self.all_findings)} findings loaded, "
                f"{len(self.topics_queue)} topics pending."
            ),
            metadata={"type": "resume", "session_id": session.id},
        )

        # Index existing findings for retrieval
        if self.all_findings:
            try:
                self.findings_retriever.add_findings(
                    findings=self.all_findings,
                    session_id=session.id,
                )
            except Exception:
                pass  # Non-critical

        # KG auto-loads from its SQLite DB, just set session_id
        self.knowledge_graph.session_id = session.id

        # Clear pause flags
        self._pause_requested = False
        self.intern._pause_requested = False
        if self.intern_pool:
            self.intern_pool._pause_requested = False

        self._log("[RESTORE] State restoration complete")

    async def run_research(
        self,
        goal: str,
        session_id: str,
        max_iterations: int = 5,
        use_parallel_init: bool = True,
        resume: bool = False,
        session: ResearchSession | None = None,
    ) -> ManagerReport:
        """Run a complete research session.

        Args:
            goal: The main research goal
            session_id: Session ID for persistence (7-char hex)
            max_iterations: Number of manager ReAct loop iterations (controls depth)
            use_parallel_init: If True, start with parallel decomposition phase
            resume: If True, restore state from DB and continue
            session: Existing session object (required if resume=True)
        """
        self.research_goal = goal
        self.session_id = session_id
        self.knowledge_graph.session_id = session_id
        self.config.max_iterations = max_iterations
        self._current_phase = "init"

        # Eagerly initialize external memory DB
        try:
            await self.external_memory._ensure_initialized()
        except Exception:
            pass  # Non-critical

        if resume and session:
            # Resume from checkpoint
            await self.restore_state(session)
            session.paused_at = None
            session.status = "running"
            await self.db.update_session(session)
            self._current_phase = "react_loop"
        else:
            # Fresh start
            self.start_time = datetime.now()

            # Initialize decision logger for audit trail
            await init_decision_logger(self.db)

            # Initialize memory for this session
            await self.memory.add_message(
                role="user", content=f"Research goal: {goal}", metadata={"session_id": session_id}
            )

            # Phase 1: Parallel initial research (if enabled and pool available)
            if use_parallel_init and self.intern_pool:
                self._current_phase = "parallel_init"
                await self._run_parallel_initial_research(goal, max_aspects=self.pool_size)

            # Initialize with the main goal as the first topic (if not enough findings yet)
            if len(self.all_findings) < 5:
                initial_topic = await self.db.create_topic(
                    session_id=session_id,
                    topic=goal,
                    depth=0,
                    priority=10,
                )
                async with self._state_lock:
                    self.topics_queue.append(initial_topic)

        context = {
            "goal": goal,
            "session_id": session_id,
        }

        # Phase 2: ReAct loop for deeper research
        self._current_phase = "react_loop"

        # Load session for periodic checkpoints
        if not session:
            session = await self.db.get_session(session_id)

        result = await self.run(context, resume=resume)

        # Check if we paused
        if result.get("paused"):
            self._current_phase = "react_loop"
            await self.checkpoint_state(session)
            # Return partial report
            return ManagerReport(
                summary="Research paused. Progress has been saved and can be resumed.",
                key_findings=self.all_findings[:20],
                topics_explored=[t.topic for t in self.completed_topics],
                topics_remaining=[t.topic for t in self.topics_queue],
                quality_assessment="",
                recommended_next_steps=["Resume research to continue"],
                time_elapsed_minutes=self._get_elapsed_minutes(),
                iterations_completed=self.state.iteration,
            )

        self._current_phase = "done"

        # Store final summary in external memory
        await self.external_memory.store(
            session_id=self.session_id,
            content=f"Research completed on: {goal}\nTotal findings: {len(self.all_findings)}\nTopics explored: {len(self.completed_topics)}",
            memory_type="summary",
            tags=["final", "session_complete"],
            metadata={
                "findings_count": len(self.all_findings),
                "topics_count": len(self.completed_topics),
                "time_minutes": self._get_elapsed_minutes(),
            },
        )

        # Return the final report
        if "last_action" in result and result["last_action"].get("action") == "synthesize":
            return result["last_action"]["report"]

        # Generate final report if not already done
        return await self._synthesize_report()

    def reset(self, clear_knowledge_graph: bool = False, clear_memory: bool = False) -> None:
        """Reset the manager state.

        Args:
            clear_knowledge_graph: If True, also clear the knowledge graph data
            clear_memory: If True, also clear hybrid memory
        """
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

        if self.intern_pool:
            self.intern_pool.reset_all()

        if clear_knowledge_graph:
            self.kg_store.clear()

        if clear_memory:
            self.memory.clear()
            # Also clear the findings retrieval index for fresh start
            try:
                self.findings_retriever.clear()
            except Exception:
                pass

    def search_past_research(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """Search past research sessions for relevant findings.

        Uses hybrid semantic + lexical search for high-quality retrieval.

        Args:
            query: Search query
            limit: Maximum results
            min_confidence: Minimum confidence threshold

        Returns:
            List of dicts with finding info and relevance scores
        """
        results = self.findings_retriever.search(
            query=query,
            limit=limit,
            min_confidence=min_confidence,
            use_reranker=True,  # Use reranker for best quality
        )

        return [
            {
                "content": r.finding.content,
                "finding_type": r.finding.finding_type.value,
                "confidence": r.finding.confidence,
                "source_url": r.finding.source_url,
                "score": r.score,
                "reranked": r.reranked,
            }
            for r in results
        ]

    def get_retrieval_stats(self) -> dict:
        """Get statistics about the hybrid retrieval system."""
        return self.findings_retriever.stats()

    def get_knowledge_graph_exports(self, output_dir: str = ".") -> dict:
        """Get knowledge graph visualizations and summaries for reports.

        Args:
            output_dir: Directory to save visualizations

        Returns:
            Dict with visualization paths and summary data
        """
        from ..knowledge import KnowledgeGraphVisualizer

        visualizer = KnowledgeGraphVisualizer(self.kg_store)

        exports = {
            "stats": self.kg_store.get_stats(),
            "key_concepts": self.kg_query.get_key_concepts(10),
            "gaps": [g.to_dict() for g in self.kg_query.identify_gaps()],
            "contradictions": self.kg_query.get_contradictions(),
            "mermaid_diagram": visualizer.create_mermaid_diagram(max_nodes=20),
            "stats_card": visualizer.create_summary_stats_card(),
        }

        return exports
