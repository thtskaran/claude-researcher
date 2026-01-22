"""Web search tool using Claude Agent SDK's built-in WebSearch capability."""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock


def _get_api_key() -> Optional[str]:
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
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None


class WebSearchTool:
    """Web search tool that uses Claude Agent SDK's WebSearch.

    This tool calls Claude with WebSearch enabled, which performs actual
    web searches and returns results with citations.
    """

    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self._search_count = 0

    async def search(self, query_text: str) -> tuple[list[SearchResult], str]:
        """Perform a web search and return results.

        Uses Claude Agent SDK with WebSearch tool enabled.

        Args:
            query_text: The search query

        Returns:
            Tuple of (list of SearchResult, summary text from Claude)
        """
        self._search_count += 1

        prompt = f"""Search the web for: {query_text}

After searching, provide:
1. A summary of what you found (2-3 paragraphs)
2. Key facts and findings with their sources

Focus on recent, authoritative sources. Include URLs in your response."""

        # Build environment with API key
        env = {}
        if api_key := _get_api_key():
            env["ANTHROPIC_API_KEY"] = api_key

        options = ClaudeAgentOptions(
            model="sonnet",
            max_turns=5,  # Allow multiple search iterations
            allowed_tools=["WebSearch"],
            env=env,
        )

        results = []
        summary_text = ""

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            summary_text += block.text

                            # Extract URLs from the text to create SearchResults
                            urls = self._extract_urls(block.text)
                            for url in urls[:self.max_results]:
                                if not any(r.url == url for r in results):
                                    results.append(SearchResult(
                                        title=self._extract_title_near_url(block.text, url),
                                        url=url,
                                        snippet=self._extract_context_near_url(block.text, url),
                                    ))

        except Exception as e:
            summary_text = f"Search error: {e}"

        # If no URLs found but we have text, create a result from the summary
        if not results and summary_text:
            results.append(SearchResult(
                title=query_text,
                url="",
                snippet=summary_text[:500],
            ))

        return results, summary_text

    async def search_and_summarize(self, query_text: str) -> str:
        """Perform a web search and return a summary.

        Args:
            query_text: The search query

        Returns:
            Summary text from Claude including search findings
        """
        _, summary = await self.search(query_text)
        return summary

    async def fetch_page(self, url: str, extract_prompt: str) -> Optional[str]:
        """Fetch a web page and extract information using Claude.

        Args:
            url: The URL to fetch
            extract_prompt: What to extract from the page

        Returns:
            Extracted content or None on error
        """
        prompt = f"""Fetch and analyze this URL: {url}

Extract and summarize: {extract_prompt}

Be concise but comprehensive. Include specific facts, dates, and details."""

        # Build environment with API key
        env = {}
        if api_key := _get_api_key():
            env["ANTHROPIC_API_KEY"] = api_key

        options = ClaudeAgentOptions(
            model="sonnet",
            max_turns=3,
            allowed_tools=["WebFetch"],
            env=env,
        )

        result_text = ""
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result_text += block.text
        except Exception:
            return None

        return result_text if result_text else None

    def _extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text."""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\])]*'
        urls = re.findall(url_pattern, text)
        # Clean up URLs (remove trailing punctuation)
        cleaned = []
        for url in urls:
            url = url.rstrip('.,;:!?)')
            if url and url not in cleaned:
                cleaned.append(url)
        return cleaned

    def _extract_title_near_url(self, text: str, url: str) -> str:
        """Try to extract a title near a URL in text."""
        # Look for text in brackets or quotes before the URL
        import re
        # Try to find [title](url) markdown pattern
        pattern = rf'\[([^\]]+)\]\({re.escape(url)}\)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # Try to find title before URL on same line
        lines = text.split('\n')
        for line in lines:
            if url in line:
                # Get text before URL, clean it up
                before = line.split(url)[0].strip()
                # Remove common prefixes
                before = re.sub(r'^[-*•]\s*', '', before)
                before = re.sub(r'\[|\]|\(|\)', '', before)
                if before and len(before) < 200:
                    return before[:100]

        return url.split('/')[2] if '/' in url else url[:50]

    def _extract_context_near_url(self, text: str, url: str) -> str:
        """Extract context around a URL in text."""
        # Find the URL and get surrounding context
        idx = text.find(url)
        if idx == -1:
            return ""

        # Get 200 chars before and after
        start = max(0, idx - 200)
        end = min(len(text), idx + len(url) + 200)
        context = text[start:end]

        # Clean up
        context = context.replace(url, '').strip()
        context = ' '.join(context.split())  # Normalize whitespace

        return context[:300] if context else ""

    @property
    def search_count(self) -> int:
        """Number of searches performed."""
        return self._search_count

    def reset_count(self) -> None:
        """Reset the search counter."""
        self._search_count = 0
