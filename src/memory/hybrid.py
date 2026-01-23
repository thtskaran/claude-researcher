"""Hybrid memory management with buffer + summary pattern."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Any
import json


@dataclass
class MemoryMessage:
    """A message stored in memory."""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_estimate: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'token_estimate': self.token_estimate,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryMessage':
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            token_estimate=data.get('token_estimate', 0),
            metadata=data.get('metadata', {}),
        )


class HybridMemory:
    """Hybrid memory pattern combining recent buffer with compressed summary.

    Based on JetBrains Research findings on efficient context management.

    Memory tiers:
    1. Recent buffer - Verbatim recent messages (high fidelity)
    2. Summary - Compressed older context (lower fidelity, lower tokens)

    When buffer exceeds threshold, oldest messages are summarized and
    compressed into the summary.
    """

    def __init__(
        self,
        max_recent_tokens: int = 8000,
        summary_threshold: float = 0.8,
        llm_callback: Optional[Callable[[str], Any]] = None,
    ):
        """Initialize hybrid memory.

        Args:
            max_recent_tokens: Maximum tokens in recent buffer before summarization
            summary_threshold: Trigger summarization at this % of max (default: 80%)
            llm_callback: Async function to call LLM for summarization
        """
        self.max_recent_tokens = max_recent_tokens
        self.summary_threshold = summary_threshold
        self.llm_callback = llm_callback

        # Memory tiers
        self.recent_buffer: list[MemoryMessage] = []
        self.summary: str = ""
        self.summary_updated_at: Optional[datetime] = None

        # Statistics
        self.total_messages_processed: int = 0
        self.summarization_count: int = 0
        self.current_buffer_tokens: int = 0

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a message to memory.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional metadata
        """
        # Estimate tokens (rough: 4 chars per token)
        token_estimate = len(content) // 4

        message = MemoryMessage(
            role=role,
            content=content,
            token_estimate=token_estimate,
            metadata=metadata or {},
        )

        self.recent_buffer.append(message)
        self.current_buffer_tokens += token_estimate
        self.total_messages_processed += 1

    async def maybe_compress(self) -> bool:
        """Check if compression is needed and perform it.

        Returns:
            True if compression was performed
        """
        threshold = int(self.max_recent_tokens * self.summary_threshold)

        if self.current_buffer_tokens < threshold:
            return False

        if not self.llm_callback:
            # No LLM available, just truncate
            self._truncate_buffer()
            return True

        # Compress older messages into summary
        await self._compress_to_summary()
        return True

    async def _compress_to_summary(self) -> None:
        """Compress older buffer messages into summary."""
        if len(self.recent_buffer) < 4:
            return  # Need at least some messages to compress

        # Split buffer: keep recent 25%, compress the rest
        split_point = max(2, len(self.recent_buffer) // 4)
        to_compress = self.recent_buffer[:-split_point]
        to_keep = self.recent_buffer[-split_point:]

        # Format messages for summarization
        messages_text = self._format_messages(to_compress)

        # Generate summary using LLM
        prompt = f"""Summarize this research conversation, preserving key findings and decisions.

Previous summary:
{self.summary if self.summary else 'None'}

New messages to incorporate:
{messages_text}

Create a concise summary (max 500 words) that captures:
1. Key research findings and facts discovered
2. Important decisions made
3. Current research direction and goals
4. Any contradictions or gaps identified

Output ONLY the summary, no preamble."""

        try:
            new_summary = await self.llm_callback(prompt)
            self.summary = new_summary
            self.summary_updated_at = datetime.now()
            self.summarization_count += 1
        except Exception:
            # If summarization fails, just truncate
            pass

        # Update buffer
        self.recent_buffer = to_keep
        self.current_buffer_tokens = sum(m.token_estimate for m in to_keep)

    def _truncate_buffer(self) -> None:
        """Truncate buffer without summarization."""
        if len(self.recent_buffer) < 4:
            return

        # Keep most recent 50%
        split_point = len(self.recent_buffer) // 2
        self.recent_buffer = self.recent_buffer[split_point:]
        self.current_buffer_tokens = sum(m.token_estimate for m in self.recent_buffer)

    def get_context(self) -> str:
        """Get full context for LLM consumption.

        Returns:
            Formatted context string with summary and recent messages
        """
        parts = []

        if self.summary:
            parts.append("## Previous Research Summary")
            parts.append(self.summary)
            parts.append("")

        if self.recent_buffer:
            parts.append("## Recent Activity")
            parts.append(self._format_messages(self.recent_buffer))

        return "\n".join(parts)

    def get_context_for_prompt(self, max_tokens: int = 4000) -> str:
        """Get context optimized for a specific token budget.

        Args:
            max_tokens: Maximum tokens to use

        Returns:
            Context string within token budget
        """
        context = self.get_context()
        estimated_tokens = len(context) // 4

        if estimated_tokens <= max_tokens:
            return context

        # Need to truncate - prioritize recent buffer
        parts = []

        # Always include recent buffer (up to half budget)
        recent_budget = max_tokens // 2
        recent_text = self._format_messages(self.recent_buffer)
        if len(recent_text) // 4 > recent_budget:
            # Truncate recent buffer
            recent_text = recent_text[-(recent_budget * 4):]

        # Use remaining budget for summary
        summary_budget = max_tokens - (len(recent_text) // 4)
        if self.summary and summary_budget > 100:
            summary_text = self.summary[:summary_budget * 4]
            parts.append("## Previous Research Summary")
            parts.append(summary_text)
            parts.append("")

        parts.append("## Recent Activity")
        parts.append(recent_text)

        return "\n".join(parts)

    def _format_messages(self, messages: list[MemoryMessage]) -> str:
        """Format messages as text."""
        lines = []
        for msg in messages:
            role_prefix = msg.role.upper()
            lines.append(f"[{role_prefix}] {msg.content}")
        return "\n\n".join(lines)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            'buffer_messages': len(self.recent_buffer),
            'buffer_tokens': self.current_buffer_tokens,
            'max_buffer_tokens': self.max_recent_tokens,
            'has_summary': bool(self.summary),
            'summary_length': len(self.summary) if self.summary else 0,
            'total_processed': self.total_messages_processed,
            'summarization_count': self.summarization_count,
            'summary_updated_at': self.summary_updated_at.isoformat() if self.summary_updated_at else None,
        }

    def clear(self) -> None:
        """Clear all memory."""
        self.recent_buffer = []
        self.summary = ""
        self.summary_updated_at = None
        self.current_buffer_tokens = 0

    def save_state(self) -> dict:
        """Save memory state for persistence."""
        return {
            'recent_buffer': [m.to_dict() for m in self.recent_buffer],
            'summary': self.summary,
            'summary_updated_at': self.summary_updated_at.isoformat() if self.summary_updated_at else None,
            'total_messages_processed': self.total_messages_processed,
            'summarization_count': self.summarization_count,
        }

    def load_state(self, state: dict) -> None:
        """Load memory state from persistence."""
        self.recent_buffer = [
            MemoryMessage.from_dict(m) for m in state.get('recent_buffer', [])
        ]
        self.summary = state.get('summary', '')
        if state.get('summary_updated_at'):
            self.summary_updated_at = datetime.fromisoformat(state['summary_updated_at'])
        self.total_messages_processed = state.get('total_messages_processed', 0)
        self.summarization_count = state.get('summarization_count', 0)
        self.current_buffer_tokens = sum(m.token_estimate for m in self.recent_buffer)
