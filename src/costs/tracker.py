"""Cost tracking for API usage during research sessions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class ModelPricing:
    """Anthropic API pricing per million tokens.

    Official pricing from https://platform.claude.com/docs/en/about-claude/pricing
    Prices in USD per million tokens.
    Last updated: January 2026

    Note: Extended thinking tokens are billed as output tokens at the same rate.
    """

    # Claude Opus 4.5 (latest)
    OPUS_INPUT = 5.00       # $5 / MTok
    OPUS_OUTPUT = 25.00     # $25 / MTok (includes thinking tokens)

    # Claude Sonnet 4.5
    SONNET_INPUT = 3.00     # $3 / MTok
    SONNET_OUTPUT = 15.00   # $15 / MTok (includes thinking tokens)

    # Claude Haiku 4.5
    HAIKU_INPUT = 1.00      # $1 / MTok
    HAIKU_OUTPUT = 5.00     # $5 / MTok (includes thinking tokens)

    # Web search: $10 per 1,000 searches
    WEB_SEARCH_COST = 0.01  # $0.01 per search

    # Web fetch: No additional cost (just tokens)
    WEB_FETCH_COST = 0.00

    # Prompt caching multipliers (for reference)
    CACHE_WRITE_5MIN_MULTIPLIER = 1.25  # 5-minute cache writes
    CACHE_WRITE_1HR_MULTIPLIER = 2.0    # 1-hour cache writes
    CACHE_READ_MULTIPLIER = 0.1         # Cache reads

    @classmethod
    def get_input_price(cls, model: str) -> float:
        """Get input token price per million for a model."""
        model_lower = model.lower()
        if "opus" in model_lower:
            return cls.OPUS_INPUT
        elif "haiku" in model_lower:
            return cls.HAIKU_INPUT
        else:  # Default to sonnet
            return cls.SONNET_INPUT

    @classmethod
    def get_output_price(cls, model: str) -> float:
        """Get output token price per million for a model.

        Note: Extended thinking tokens are billed as output tokens.
        """
        model_lower = model.lower()
        if "opus" in model_lower:
            return cls.OPUS_OUTPUT
        elif "haiku" in model_lower:
            return cls.HAIKU_OUTPUT
        else:  # Default to sonnet
            return cls.SONNET_OUTPUT


@dataclass
class ModelUsage:
    """Track usage for a specific model."""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0  # Extended thinking tokens (billed as output)
    calls: int = 0

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens including thinking."""
        return self.output_tokens + self.thinking_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.total_output_tokens


@dataclass
class CostSummary:
    """Summary of costs for a research session."""
    # Per-model breakdown
    sonnet_usage: ModelUsage = field(default_factory=ModelUsage)
    opus_usage: ModelUsage = field(default_factory=ModelUsage)
    haiku_usage: ModelUsage = field(default_factory=ModelUsage)

    # Web searches
    web_searches: int = 0
    web_fetches: int = 0

    # Timing
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @property
    def total_input_tokens(self) -> int:
        return (
            self.sonnet_usage.input_tokens +
            self.opus_usage.input_tokens +
            self.haiku_usage.input_tokens
        )

    @property
    def total_output_tokens(self) -> int:
        return (
            self.sonnet_usage.total_output_tokens +
            self.opus_usage.total_output_tokens +
            self.haiku_usage.total_output_tokens
        )

    @property
    def total_thinking_tokens(self) -> int:
        return (
            self.sonnet_usage.thinking_tokens +
            self.opus_usage.thinking_tokens +
            self.haiku_usage.thinking_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_calls(self) -> int:
        return (
            self.sonnet_usage.calls +
            self.opus_usage.calls +
            self.haiku_usage.calls
        )

    @property
    def sonnet_cost(self) -> float:
        """Cost for Sonnet usage in USD (thinking tokens billed as output)."""
        input_cost = (self.sonnet_usage.input_tokens / 1_000_000) * ModelPricing.SONNET_INPUT
        output_cost = (self.sonnet_usage.total_output_tokens / 1_000_000) * ModelPricing.SONNET_OUTPUT
        return input_cost + output_cost

    @property
    def opus_cost(self) -> float:
        """Cost for Opus usage in USD (thinking tokens billed as output)."""
        input_cost = (self.opus_usage.input_tokens / 1_000_000) * ModelPricing.OPUS_INPUT
        output_cost = (self.opus_usage.total_output_tokens / 1_000_000) * ModelPricing.OPUS_OUTPUT
        return input_cost + output_cost

    @property
    def haiku_cost(self) -> float:
        """Cost for Haiku usage in USD (thinking tokens billed as output)."""
        input_cost = (self.haiku_usage.input_tokens / 1_000_000) * ModelPricing.HAIKU_INPUT
        output_cost = (self.haiku_usage.total_output_tokens / 1_000_000) * ModelPricing.HAIKU_OUTPUT
        return input_cost + output_cost

    @property
    def search_cost(self) -> float:
        """Cost for web searches in USD."""
        return self.web_searches * ModelPricing.WEB_SEARCH_COST

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return self.sonnet_cost + self.opus_cost + self.haiku_cost + self.search_cost

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "models": {
                "sonnet": {
                    "input_tokens": self.sonnet_usage.input_tokens,
                    "output_tokens": self.sonnet_usage.output_tokens,
                    "thinking_tokens": self.sonnet_usage.thinking_tokens,
                    "calls": self.sonnet_usage.calls,
                    "cost_usd": round(self.sonnet_cost, 4),
                },
                "opus": {
                    "input_tokens": self.opus_usage.input_tokens,
                    "output_tokens": self.opus_usage.output_tokens,
                    "thinking_tokens": self.opus_usage.thinking_tokens,
                    "calls": self.opus_usage.calls,
                    "cost_usd": round(self.opus_cost, 4),
                },
                "haiku": {
                    "input_tokens": self.haiku_usage.input_tokens,
                    "output_tokens": self.haiku_usage.output_tokens,
                    "thinking_tokens": self.haiku_usage.thinking_tokens,
                    "calls": self.haiku_usage.calls,
                    "cost_usd": round(self.haiku_cost, 4),
                },
            },
            "web": {
                "searches": self.web_searches,
                "fetches": self.web_fetches,
                "cost_usd": round(self.search_cost, 4),
            },
            "totals": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "thinking_tokens": self.total_thinking_tokens,
                "total_tokens": self.total_tokens,
                "api_calls": self.total_calls,
                "total_cost_usd": round(self.total_cost, 4),
            },
            "pricing_reference": {
                "opus_4.5": {"input": "$5/MTok", "output": "$25/MTok"},
                "sonnet_4.5": {"input": "$3/MTok", "output": "$15/MTok"},
                "haiku_4.5": {"input": "$1/MTok", "output": "$5/MTok"},
                "web_search": "$0.01/search",
                "note": "Thinking tokens are billed as output tokens",
            },
        }


class CostTracker:
    """Tracks API costs during a research session.

    Uses token estimation since the SDK doesn't expose exact counts.
    Estimation: ~4 characters per token (rough average for English text).
    """

    # Characters per token (rough estimate)
    CHARS_PER_TOKEN = 4

    def __init__(self):
        self._summary = CostSummary()
        self._summary.started_at = datetime.now()

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Estimate token count from text.

        Uses a simple heuristic of ~4 characters per token.
        This is a rough estimate - actual tokenization varies.
        """
        if not text:
            return 0
        return max(1, len(text) // cls.CHARS_PER_TOKEN)

    def track_call(
        self,
        model: str,
        input_text: str,
        output_text: str,
        system_prompt: str = "",
        thinking_text: str = "",
    ) -> None:
        """Track an API call.

        Args:
            model: Model name (sonnet, opus, haiku)
            input_text: The prompt sent to the model
            output_text: The response from the model
            system_prompt: Optional system prompt (counted as input)
            thinking_text: Optional thinking/reasoning text (billed as output)
        """
        # Estimate tokens
        input_tokens = self.estimate_tokens(input_text) + self.estimate_tokens(system_prompt)
        output_tokens = self.estimate_tokens(output_text)
        thinking_tokens = self.estimate_tokens(thinking_text)

        # Get the right usage tracker
        model_lower = model.lower()
        if "opus" in model_lower:
            usage = self._summary.opus_usage
        elif "haiku" in model_lower:
            usage = self._summary.haiku_usage
        else:
            usage = self._summary.sonnet_usage

        # Update counts
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.thinking_tokens += thinking_tokens
        usage.calls += 1

    def track_web_search(self, count: int = 1) -> None:
        """Track web search(es)."""
        self._summary.web_searches += count

    def track_web_fetch(self, count: int = 1) -> None:
        """Track web fetch(es)."""
        self._summary.web_fetches += count

    def get_summary(self) -> CostSummary:
        """Get the current cost summary."""
        self._summary.ended_at = datetime.now()
        return self._summary

    def reset(self) -> None:
        """Reset the tracker for a new session."""
        self._summary = CostSummary()
        self._summary.started_at = datetime.now()


# Global tracker instance
_global_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_cost_tracker() -> None:
    """Reset the global cost tracker."""
    global _global_tracker
    _global_tracker = CostTracker()
