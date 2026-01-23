"""User interaction handler for research sessions."""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Optional, Awaitable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .models import (
    ClarificationQuestion,
    ClarifiedGoal,
    PendingQuestion,
    UserMessage,
)
from .config import InteractionConfig


class UserInteraction:
    """Handles user interaction during research sessions.

    Provides three main features:
    1. Pre-research clarification questions
    2. Async mid-research questions with timeout
    3. User message queue for injecting guidance anytime
    """

    def __init__(
        self,
        config: Optional[InteractionConfig] = None,
        console: Optional[Console] = None,
        llm_callback: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        """Initialize the interaction handler.

        Args:
            config: Interaction configuration
            console: Rich console for output
            llm_callback: Async function to call the LLM for generating questions
        """
        self.config = config or InteractionConfig()
        self.console = console or Console()
        self.llm_callback = llm_callback

        # State for async questions
        self._pending_question: Optional[PendingQuestion] = None
        self._response_event = asyncio.Event()
        self._response_value: Optional[str] = None

        # Message queue for user guidance
        self._message_queue: asyncio.Queue[UserMessage] = asyncio.Queue()

        # Track questions asked during session
        self._questions_asked: int = 0

        # Callbacks for pausing/resuming progress display
        self._on_pause: Optional[Callable[[], None]] = None
        self._on_resume: Optional[Callable[[], None]] = None

    def set_progress_callbacks(
        self,
        on_pause: Optional[Callable[[], None]] = None,
        on_resume: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set callbacks for pausing/resuming the progress display.

        Args:
            on_pause: Called when interaction needs to pause the spinner
            on_resume: Called when interaction is done and spinner can resume
        """
        self._on_pause = on_pause
        self._on_resume = on_resume

    async def clarify_research_goal(self, goal: str) -> ClarifiedGoal:
        """Ask clarification questions before starting research.

        Args:
            goal: The original research goal from the user

        Returns:
            ClarifiedGoal with enriched context from user answers
        """
        if not self.config.enable_clarification or self.config.autonomous_mode:
            return ClarifiedGoal(
                original=goal,
                enriched_context=goal,
                skipped=True,
            )

        # Generate clarification questions using LLM
        questions = await self._generate_clarification_questions(goal)

        if not questions:
            return ClarifiedGoal(
                original=goal,
                enriched_context=goal,
                skipped=True,
            )

        # Display intro
        self.console.print()
        self.console.print(Panel(
            "[bold]Before we begin, a few quick questions to focus the research:[/bold]\n"
            "[dim]Press Enter to skip any question, or type 's' to skip all and start immediately.[/dim]",
            border_style="cyan",
        ))
        self.console.print()

        # Ask each question
        clarifications: dict[int, str] = {}

        for q in questions[:self.config.max_clarification_questions]:
            answer = await self._ask_clarification(q)

            if answer and answer.lower() == 's':
                # Skip remaining questions
                self.console.print("[dim]Skipping remaining questions...[/dim]")
                break

            if answer:
                clarifications[q.id] = answer

        # Generate enriched context from answers
        enriched = await self._enrich_goal(goal, questions, clarifications)

        return ClarifiedGoal(
            original=goal,
            clarifications=clarifications,
            enriched_context=enriched,
            skipped=False,
        )

    async def _generate_clarification_questions(
        self, goal: str
    ) -> list[ClarificationQuestion]:
        """Generate clarification questions using the LLM."""
        if not self.llm_callback:
            return []

        prompt = f"""Given this research goal, generate 2-4 brief clarification questions that would help focus the research.

Research Goal: {goal}

Consider asking about:
- Specific focus areas or aspects
- Time period of interest (recent vs historical)
- Depth vs breadth preference
- Any specific use case or application
- Geographic or domain scope

Return as JSON array:
[
    {{"id": 1, "question": "...", "options": ["option1", "option2"], "category": "scope"}},
    ...
]

Keep questions brief and options concise. Return ONLY the JSON array."""

        try:
            response = await self.llm_callback(prompt)

            # Parse JSON from response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                return [ClarificationQuestion(**item) for item in data]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return []

    async def _ask_clarification(
        self, question: ClarificationQuestion
    ) -> Optional[str]:
        """Ask a single clarification question."""
        # Format question with options if available
        prompt_text = f"[bold cyan]{question.question}[/bold cyan]"

        if question.options:
            options_text = " / ".join(question.options)
            prompt_text += f"\n[dim]Options: {options_text}[/dim]"

        self.console.print(prompt_text)

        # Get input with timeout
        try:
            loop = asyncio.get_event_loop()
            answer = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: Prompt.ask("[cyan]>[/cyan]", default="")),
                timeout=self.config.clarification_timeout,
            )
            return answer.strip() if answer else None
        except asyncio.TimeoutError:
            self.console.print("[dim]Timeout - skipping question[/dim]")
            return None

    async def _enrich_goal(
        self,
        original_goal: str,
        questions: list[ClarificationQuestion],
        answers: dict[int, str],
    ) -> str:
        """Generate an enriched research goal from clarification answers."""
        if not answers or not self.llm_callback:
            return original_goal

        # Format Q&A pairs
        qa_pairs = []
        for q in questions:
            if q.id in answers:
                qa_pairs.append(f"Q: {q.question}\nA: {answers[q.id]}")

        if not qa_pairs:
            return original_goal

        prompt = f"""Combine the original research goal with the user's clarifications into an enriched, focused research goal.

Original Goal: {original_goal}

User Clarifications:
{chr(10).join(qa_pairs)}

Write a single enriched research goal that incorporates these clarifications.
Keep it concise but complete. Do not add any preamble or explanation - just output the enriched goal."""

        try:
            enriched = await self.llm_callback(prompt)
            return enriched.strip()
        except Exception:
            return original_goal

    async def ask_with_timeout(
        self,
        question: str,
        context: str = "",
        options: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Ask a question during research with a timeout.

        If the user doesn't respond within the timeout, returns None
        and research continues autonomously.

        Args:
            question: The question to ask
            context: Optional context about why this question is being asked
            options: Optional list of suggested answers

        Returns:
            User's response, or None if timeout/skipped
        """
        if not self.config.enable_async_questions or self.config.autonomous_mode:
            return None

        if self._questions_asked >= self.config.max_questions_per_session:
            return None

        self._questions_asked += 1

        # Pause the progress spinner if callback is set
        if self._on_pause:
            self._on_pause()

        # Create pending question
        self._pending_question = PendingQuestion(
            text=question,
            context=context,
            options=options or [],
            timeout_seconds=self.config.question_timeout,
        )
        self._response_event.clear()
        self._response_value = None

        # Display question
        self.console.print()
        panel_content = f"[bold]{question}[/bold]"
        if context:
            panel_content += f"\n[dim]{context}[/dim]"
        if options:
            panel_content += f"\n[cyan]Options: {' / '.join(options)}[/cyan]"
        panel_content += f"\n[dim]({self.config.question_timeout}s timeout - research will continue if no response)[/dim]"

        self.console.print(Panel(
            panel_content,
            title="[yellow]Question[/yellow]",
            border_style="yellow",
        ))

        try:
            # Get input directly with visible typing (since spinner is paused)
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: Prompt.ask("[yellow]>[/yellow]", default="")
                ),
                timeout=self.config.question_timeout,
            )
            response = response.strip() if response else None
        except asyncio.TimeoutError:
            self.console.print("[dim]No response - continuing research...[/dim]")
            response = None

        self._pending_question = None

        # Resume the progress spinner
        if self._on_resume:
            self._on_resume()

        return response

    def respond(self, response: str) -> bool:
        """Handle a user response to a pending question.

        Called by the InputListener when user types during a pending question.

        Args:
            response: The user's response

        Returns:
            True if there was a pending question to respond to
        """
        if self._pending_question:
            self._response_value = response
            self._response_event.set()
            return True
        return False

    def has_pending_question(self) -> bool:
        """Check if there's a pending question waiting for response."""
        return self._pending_question is not None

    def inject_message(self, text: str) -> None:
        """Inject a guidance message into the queue.

        Messages will be picked up by the Manager agent on its next iteration.

        Args:
            text: The guidance message from the user
        """
        if not self.config.enable_message_queue or self.config.autonomous_mode:
            return

        message = UserMessage(content=text)
        self._message_queue.put_nowait(message)
        self.console.print(f"[bold green]Guidance queued[/bold green] [dim](will be used in next research iteration)[/dim]")

    def get_pending_messages(self) -> list[UserMessage]:
        """Get all pending messages from the queue (non-blocking).

        Returns:
            List of user messages, empty if none pending
        """
        messages = []
        while True:
            try:
                msg = self._message_queue.get_nowait()
                msg.processed = True
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    def reset(self) -> None:
        """Reset interaction state for a new session."""
        self._pending_question = None
        self._response_event.clear()
        self._response_value = None
        self._questions_asked = 0

        # Clear message queue
        while True:
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
