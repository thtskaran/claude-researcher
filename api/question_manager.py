"""
Question manager for handling mid-research questions in the UI.

This bridges the CLI's interaction system with WebSocket communication.
"""
import asyncio
from dataclasses import dataclass

from api.events import emit_event


@dataclass
class PendingQuestion:
    """A question waiting for user response."""
    question_id: str
    session_id: str
    question: str
    context: str
    options: list[str]
    timeout: int
    response_event: asyncio.Event
    response: str | None = None


class QuestionManager:
    """Manages mid-research questions for UI sessions."""

    def __init__(self):
        self._pending_questions: dict[str, PendingQuestion] = {}
        self._question_counter = 0

    async def ask_question(
        self,
        session_id: str,
        question: str,
        context: str = "",
        options: list[str] | None = None,
        timeout: int = 60,
    ) -> str | None:
        """
        Ask a question and wait for user response via WebSocket.

        Returns:
            User's response text, or None if timeout
        """
        # Generate unique question ID
        self._question_counter += 1
        question_id = f"q_{self._question_counter}"

        # Create pending question
        response_event = asyncio.Event()
        pending = PendingQuestion(
            question_id=question_id,
            session_id=session_id,
            question=question,
            context=context,
            options=options or [],
            timeout=timeout,
            response_event=response_event,
        )
        self._pending_questions[question_id] = pending

        # Emit question event via WebSocket
        await emit_event(
            session_id=session_id,
            event_type="question_asked",
            agent="manager",
            data={
                "question_id": question_id,
                "question": question,
                "context": context,
                "options": options or [],
                "timeout": timeout,
            }
        )

        try:
            # Wait for response with timeout
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
            response = pending.response

            # Emit response received event
            await emit_event(
                session_id=session_id,
                event_type="question_answered",
                agent="user",
                data={
                    "question_id": question_id,
                    "response": response,
                }
            )

            return response

        except TimeoutError:
            # Emit timeout event
            await emit_event(
                session_id=session_id,
                event_type="question_timeout",
                agent="system",
                data={
                    "question_id": question_id,
                    "message": "No response - continuing research autonomously",
                }
            )
            return None

        finally:
            # Clean up
            if question_id in self._pending_questions:
                del self._pending_questions[question_id]

    def answer_question(self, question_id: str, response: str) -> bool:
        """
        Provide an answer to a pending question.

        Returns:
            True if question was found and answered, False otherwise
        """
        if question_id not in self._pending_questions:
            return False

        pending = self._pending_questions[question_id]
        pending.response = response
        pending.response_event.set()
        return True

    def get_pending_question(self, session_id: str) -> PendingQuestion | None:
        """Get the current pending question for a session, if any."""
        for pending in self._pending_questions.values():
            if pending.session_id == session_id:
                return pending
        return None


# Global question manager instance
_question_manager: QuestionManager | None = None


def get_question_manager() -> QuestionManager:
    """Get or create the global question manager."""
    global _question_manager
    if _question_manager is None:
        _question_manager = QuestionManager()
    return _question_manager
