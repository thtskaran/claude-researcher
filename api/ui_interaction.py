"""
UI-compatible interaction handler.

This adapts the CLI's UserInteraction system to work with WebSocket communication.
"""
from collections.abc import Callable
from typing import Any

from api.question_manager import get_question_manager
from src.interaction.config import InteractionConfig
from src.interaction.models import UserMessage


class UIInteraction:
    """
    Interaction handler for UI that uses WebSocket instead of terminal.

    This implements the same interface as UserInteraction but routes
    questions through WebSocket and the QuestionManager.
    """

    def __init__(
        self,
        session_id: str,
        config: InteractionConfig,
        llm_callback: Callable[[str], Any] | None = None,
    ):
        self.session_id = session_id
        self.config = config
        self.llm_callback = llm_callback
        self._questions_asked = 0
        self._message_queue: list[UserMessage] = []
        self.question_manager = get_question_manager()

    async def ask_with_timeout(
        self,
        question: str,
        context: str = "",
        options: list[str] | None = None,
    ) -> str | None:
        """
        Ask a mid-research question via WebSocket.

        Returns:
            User's response, or None if timeout/disabled
        """
        # Check if questions are enabled
        if not self.config.enable_async_questions or self.config.autonomous_mode:
            return None

        # Check question limit
        if self._questions_asked >= self.config.max_questions_per_session:
            return None

        self._questions_asked += 1

        # Ask via question manager (sends WebSocket event and waits)
        response = await self.question_manager.ask_question(
            session_id=self.session_id,
            question=question,
            context=context,
            options=options,
            timeout=self.config.question_timeout,
        )

        return response

    def inject_message(self, message: str) -> None:
        """Inject a guidance message into the queue."""
        self._message_queue.append(UserMessage(content=message))

    def get_pending_messages(self) -> list[UserMessage]:
        """Get and clear all pending guidance messages."""
        messages = self._message_queue.copy()
        self._message_queue.clear()
        return messages

    def has_pending_question(self) -> bool:
        """Check if there's a question waiting for response."""
        pending = self.question_manager.get_pending_question(self.session_id)
        return pending is not None

    # Dummy methods for CLI compatibility (not used in UI)
    def set_progress_callbacks(self, on_pause=None, on_resume=None): pass
    async def start_listener(self): pass
    async def stop_listener(self): pass
    async def _generate_clarification_questions(self, goal: str): return []
    async def _enrich_goal(self, goal: str, questions: list, answers: dict): return goal
