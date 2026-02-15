"""Web search and scraping using Bright Data API."""

import asyncio
import os
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

from ..logging_config import get_logger

logger = get_logger(__name__)

# Load environment variables from .env file
dotenv_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    content: str | None = None


class WebSearchTool:
    """Web search and scraping via Bright Data's SERP and Web Unlocker APIs.

    Requires BRIGHT_DATA_API_TOKEN env var. Optionally set BRIGHT_DATA_ZONE
    (defaults to 'mcp_unlocker').

    For each search query this makes one API call to Bright Data's SERP endpoint
    and returns structured results. Pages can be scraped individually for full
    content via fetch_page().
    """

    _API_ENDPOINT = "https://api.brightdata.com/request"

    def __init__(
        self,
        api_token: str | None = None,
        zone: str | None = None,
        max_results: int = 10,
    ):
        self.api_token = api_token or os.environ.get("BRIGHT_DATA_API_TOKEN", "")
        self.zone = zone or os.environ.get("BRIGHT_DATA_ZONE", "mcp_unlocker")
        self.max_results = max_results
        self._search_count = 0

        if not self.api_token:
            raise ValueError(
                "Bright Data API token required. Set BRIGHT_DATA_API_TOKEN env var."
            )

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    async def search(self, query_text: str) -> tuple[list[SearchResult], str]:
        """Search Google via Bright Data SERP API.

        Returns:
            Tuple of (list of SearchResult, summary string)
        """
        import httpx

        self._search_count += 1
        logger.info("Web search: query=%s", query_text[:200])
        search_url = f"https://www.google.com/search?q={quote(query_text)}"

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        self._API_ENDPOINT,
                        headers=self._headers,
                        json={
                            "url": search_url,
                            "zone": self.zone,
                            "format": "raw",
                            "data_format": "parsed_light",
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                results = self._parse_google_results(data)
                summary = self._build_summary(query_text, results)
                return results[: self.max_results], summary

            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep((2**attempt) + random.uniform(0, 0.5))
                else:
                    logger.error("Web search failed: %s", e, exc_info=True)
                    return [], f"Search error: {e}"

        return [], ""

    async def search_and_summarize(self, query_text: str) -> str:
        """Search and return summary text."""
        _, summary = await self.search(query_text)
        return summary

    async def fetch_page(self, url: str, extract_prompt: str = "") -> str | None:
        """Scrape a page via Bright Data Web Unlocker (bypasses bot detection).

        Returns full page content as markdown, or None on error.
        """
        import httpx

        logger.debug("Fetching page: %s", url[:200])
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self._API_ENDPOINT,
                        headers=self._headers,
                        json={
                            "url": url,
                            "zone": self.zone,
                            "format": "raw",
                            "data_format": "markdown",
                        },
                    )
                    response.raise_for_status()
                    return response.text

            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep((2**attempt) + random.uniform(0, 0.5))
                else:
                    logger.warning("Page fetch failed: %s", url[:100], exc_info=True)

        return None

    def _parse_google_results(self, data: dict) -> list[SearchResult]:
        """Parse Bright Data's parsed_light Google SERP response."""
        results = []
        for entry in data.get("organic", []):
            link = entry.get("link", "").strip()
            title = entry.get("title", "").strip()
            if not link or not title:
                continue
            results.append(SearchResult(
                title=title,
                url=link,
                snippet=entry.get("description", "").strip(),
            ))
        return results

    def _build_summary(self, query: str, results: list[SearchResult]) -> str:
        """Build a text summary from search results."""
        if not results:
            return f"No results found for: {query}"

        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results[:5], 1):
            lines.append(f"{i}. {r.title}")
            lines.append(f"   {r.url}")
            if r.snippet:
                lines.append(f"   {r.snippet}")
            lines.append("")
        return "\n".join(lines)

    @property
    def search_count(self) -> int:
        """Number of searches performed."""
        return self._search_count

    def reset_count(self) -> None:
        """Reset the search counter."""
        self._search_count = 0
