"""Base agent class with ReAct loop implementation using Claude Agent SDK."""

import asyncio
import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query
from rich.console import Console

from ..audit import DecisionLogger, DecisionType, get_decision_logger
from ..costs.tracker import get_cost_tracker
from ..events import emit_action, emit_agent_event, emit_error, emit_thinking
from ..models.findings import AgentMessage, AgentRole
from ..storage.database import ResearchDatabase


def _get_api_key() -> str | None:
    """Get the API key from Claude Code's config or environment."""
    # First check environment
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        return api_key

    # Try Claude Code's get-api-key.sh script
    script_path = Path.home() / ".claude" / "get-api-key.sh"
    if script_path.exists():
        try:
            result = subprocess.run(
                ["bash", str(script_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

    return None


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    model: str = "sonnet"  # sonnet, opus, haiku
    max_turns: int = 10
    max_iterations: int = 100
    allowed_tools: list[str] = field(default_factory=lambda: ["WebSearch", "WebFetch"])
    max_thinking_tokens: int = 10000  # Extended thinking for deep reasoning


class ModelRouter:
    """Routes tasks to appropriate models based on complexity.

    Model selection strategy:
    - Haiku: Quick classification, simple extraction, yes/no decisions
    - Sonnet: Web search analysis, finding extraction, moderate reasoning
    - Opus: Strategic planning, deep synthesis, complex reasoning
    """

    # Task to model mapping
    TASK_MODELS = {
        # Haiku tasks (fast, cheap)
        'classify': 'haiku',
        'extract_simple': 'haiku',
        'yes_no': 'haiku',
        'format': 'haiku',

        # Sonnet tasks (balanced)
        'search': 'sonnet',
        'extract_findings': 'sonnet',
        'summarize': 'sonnet',
        'analyze_results': 'sonnet',
        'query_expansion': 'sonnet',

        # Opus tasks (deep reasoning)
        'strategic_planning': 'opus',
        'synthesis': 'opus',
        'critique': 'opus',
        'deep_analysis': 'opus',
        'report_writing': 'opus',
    }

    @classmethod
    def get_model_for_task(cls, task: str, default: str = 'sonnet') -> str:
        """Get the appropriate model for a task.

        Args:
            task: The task type
            default: Default model if task not found

        Returns:
            Model name (haiku, sonnet, or opus)
        """
        return cls.TASK_MODELS.get(task, default)

    @classmethod
    def should_use_thinking(cls, task: str) -> bool:
        """Determine if a task should use extended thinking.

        Extended thinking is best for:
        - Complex synthesis
        - Strategic planning
        - Deep analysis
        """
        thinking_tasks = {
            'strategic_planning',
            'synthesis',
            'critique',
            'deep_analysis',
            'report_writing',
        }
        return task in thinking_tasks


@dataclass
class AgentState:
    """Current state of an agent."""
    iteration: int = 0
    total_actions: int = 0
    last_action: str | None = None
    last_observation: str | None = None
    is_complete: bool = False
    error: str | None = None
    history: list[dict] = field(default_factory=list)


class BaseAgent(ABC):
    """Base class for all research agents implementing ReAct loop.

    ReAct = Reason + Act
    Uses Claude Agent SDK for all LLM calls.
    """

    def __init__(
        self,
        role: AgentRole,
        db: ResearchDatabase,
        config: AgentConfig | None = None,
        console: Console | None = None,
        session_id: str | None = None,
    ):
        self.role = role
        self.db = db
        self.config = config or AgentConfig()
        self.console = console or Console()
        self.state = AgentState()
        self._stop_requested = False
        self._callbacks: list[Callable] = []
        self.session_id = session_id  # For WebSocket events

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining the agent's role and behavior."""
        pass

    @abstractmethod
    async def think(self, context: dict[str, Any]) -> str:
        """Reason about the current state and decide what to do next."""
        pass

    @abstractmethod
    async def act(self, thought: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute an action based on the thought."""
        pass

    @abstractmethod
    async def observe(self, action_result: dict[str, Any]) -> str:
        """Process the result of an action."""
        pass

    @abstractmethod
    def is_done(self, context: dict[str, Any]) -> bool:
        """Check if the agent has completed its task."""
        pass

    async def run(self, initial_context: dict[str, Any]) -> dict[str, Any]:
        """Run the ReAct loop until completion or max iterations."""
        context = initial_context.copy()
        self.state = AgentState()

        self._log(f"Starting {self.role.value} agent")

        while (
            not self.is_done(context)
            and not self._stop_requested
            and self.state.iteration < self.config.max_iterations
        ):
            self.state.iteration += 1
            self._log(f"Iteration {self.state.iteration}", style="dim")

            try:
                # Think
                thought = await self.think(context)
                self._log(f"[Think] {thought}")
                self.state.history.append({"type": "thought", "content": thought})

                # Emit thinking event
                if self.session_id:
                    await emit_thinking(
                        session_id=self.session_id,
                        agent=self.role.value,
                        thought=thought[:500]  # Truncate for display
                    )

                if self._stop_requested:
                    break

                # Act
                action_result = await self.act(thought, context)
                self.state.total_actions += 1
                self.state.last_action = str(action_result.get("action", "unknown"))
                self._log(f"[Act] {self.state.last_action}")
                self.state.history.append({"type": "action", "content": action_result})

                # Emit action event
                if self.session_id:
                    await emit_action(
                        session_id=self.session_id,
                        agent=self.role.value,
                        action=self.state.last_action,
                        details={"iteration": self.state.iteration}
                    )

                if self._stop_requested:
                    break

                # Observe
                observation = await self.observe(action_result)
                self.state.last_observation = observation
                self._log(f"[Observe] {observation}")
                self.state.history.append({"type": "observation", "content": observation})

                # Update context
                context["last_thought"] = thought
                context["last_action"] = action_result
                context["last_observation"] = observation
                context["iteration"] = self.state.iteration

                # Fire callbacks
                for callback in self._callbacks:
                    await self._safe_callback(callback, context)

            except Exception as e:
                self.state.error = str(e)
                self._log(f"[Error] {e}", style="red")

                # Emit error event
                if self.session_id:
                    await emit_error(
                        session_id=self.session_id,
                        agent=self.role.value,
                        error=str(e),
                        recoverable=False
                    )

                break

        self.state.is_complete = self.is_done(context)
        self._log(f"Agent completed: iterations={self.state.iteration}, actions={self.state.total_actions}")

        return context

    def stop(self) -> None:
        """Request the agent to stop."""
        self._stop_requested = True
        self._log("Stop requested")

    def add_callback(self, callback: Callable) -> None:
        """Add a callback to be called after each iteration."""
        self._callbacks.append(callback)

    async def _safe_callback(self, callback: Callable, context: dict) -> None:
        """Safely execute a callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(self, context)
            else:
                callback(self, context)
        except Exception as e:
            self._log(f"Callback error: {e}", style="yellow")

    async def send_message(
        self,
        to_agent: AgentRole,
        message_type: str,
        content: str,
        session_id: str,
        metadata: dict | None = None,
    ) -> AgentMessage:
        """Send a message to another agent."""
        message = AgentMessage(
            session_id=session_id,
            from_agent=self.role,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
        )
        return await self.db.save_message(message)

    async def call_claude(
        self,
        prompt: str,
        tools: list[str] | None = None,
        use_thinking: bool = False,
        task_type: str | None = None,
    ) -> str:
        """Call Claude via the Agent SDK.

        Args:
            prompt: The prompt to send
            tools: Optional list of tools to enable (default: none for pure reasoning)
            use_thinking: Enable extended thinking for deep reasoning (Opus recommended)
            task_type: Optional task type for automatic model routing

        Returns:
            The text response from Claude
        """
        # Build environment with API key
        env = {}
        if api_key := _get_api_key():
            env["ANTHROPIC_API_KEY"] = api_key

        # Use model routing if task_type specified
        model = self.config.model
        if task_type:
            model = ModelRouter.get_model_for_task(task_type, default=self.config.model)
            # Also check if we should use thinking for this task
            if not use_thinking and ModelRouter.should_use_thinking(task_type):
                use_thinking = True

        options = ClaudeAgentOptions(
            model=model,
            max_turns=1,  # Single turn for reasoning
            allowed_tools=tools or [],
            system_prompt=self.system_prompt,
            env=env,
            max_thinking_tokens=self.config.max_thinking_tokens if use_thinking else None,
        )

        response_text = ""
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
        except Exception as e:
            self._log(f"Error calling Claude: {e}", style="red")
            response_text = f"Error: {e}"

        # Track costs
        tracker = get_cost_tracker()
        tracker.track_call(
            model=model,
            input_text=prompt,
            output_text=response_text,
            system_prompt=self.system_prompt,
        )

        return response_text

    async def call_claude_with_tools(
        self,
        prompt: str,
        tools: list[str],
    ) -> tuple[str, list[dict]]:
        """Call Claude with tools enabled via the Agent SDK.

        Args:
            prompt: The prompt to send
            tools: List of tools to enable (e.g., ["WebSearch", "WebFetch"])

        Returns:
            Tuple of (text_response, tool_results)
        """
        from claude_agent_sdk import ToolResultBlock, ToolUseBlock

        # Build environment with API key
        env = {}
        if api_key := _get_api_key():
            env["ANTHROPIC_API_KEY"] = api_key

        options = ClaudeAgentOptions(
            model=self.config.model,
            max_turns=self.config.max_turns,
            allowed_tools=tools,
            system_prompt=self.system_prompt,
            env=env,
        )

        response_text = ""
        tool_results = []

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                        elif isinstance(block, ToolUseBlock):
                            tool_results.append({
                                "tool": block.name,
                                "id": block.id,
                                "input": block.input,
                            })
                        elif isinstance(block, ToolResultBlock):
                            # Find matching tool use and add result
                            for tr in tool_results:
                                if tr.get("id") == block.tool_use_id:
                                    tr["result"] = block.content
                                    tr["is_error"] = block.is_error
        except Exception as e:
            self._log(f"Error calling Claude with tools: {e}", style="red")
            response_text = f"Error: {e}"

        # Track costs
        tracker = get_cost_tracker()
        tracker.track_call(
            model=self.config.model,
            input_text=prompt,
            output_text=response_text,
            system_prompt=self.system_prompt,
        )

        # Track web searches and fetches
        for tr in tool_results:
            tool_name = tr.get("tool", "").lower()
            if "websearch" in tool_name or "web_search" in tool_name:
                tracker.track_web_search()
            elif "webfetch" in tool_name or "web_fetch" in tool_name:
                tracker.track_web_fetch()

        return response_text, tool_results

    def _log(self, message: str, style: str | None = None) -> None:
        """Log a message to the console."""
        prefix = f"[{self.role.value.upper()}]"
        if style:
            self.console.print(f"{prefix} {message}", style=style)
        else:
            self.console.print(f"{prefix} {message}")

        # Optionally mirror logs to the UI as "system" events for verbosity
        if (
            self.session_id
            and os.environ.get("CLAUDE_RESEARCHER_DISABLE_LOG_EVENTS") != "1"
        ):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.create_task(
                emit_agent_event(
                    session_id=self.session_id,
                    event_type="system",
                    agent=self.role.value,
                    data={"message": message},
                )
            )

    async def _log_decision(
        self,
        session_id: str,
        decision_type: DecisionType,
        decision_outcome: str,
        reasoning: str | None = None,
        inputs: dict | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Log an agent decision for audit trail.

        This is a fire-and-forget async operation that won't block
        the main agent loop. Decisions are batched and written to DB
        in the background.

        Args:
            session_id: Current research session ID
            decision_type: Type of decision (from DecisionType enum)
            decision_outcome: The outcome/choice made
            reasoning: Agent's reasoning (truncated to 500 chars in DB)
            inputs: Key inputs that influenced the decision
            metrics: Metrics at decision time (e.g., time_remaining, findings_count)
        """
        logger = get_decision_logger(self.db)
        if logger is None:
            return

        try:
            await logger.log_decision(
                session_id=session_id,
                agent_role=self.role.value,
                decision_type=decision_type,
                decision_outcome=decision_outcome,
                reasoning=reasoning,
                inputs=inputs,
                metrics=metrics,
                iteration=self.state.iteration,
            )
        except Exception:
            # Don't let logging errors affect agent operation
            pass
