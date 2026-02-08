# Data Sources for claude-researcher

*Bright Data as unified data layer — replaces all individual API integrations*

---

## Overview

All external data access goes through the Bright Data API (`src/tools/web_search.py`).
This eliminates per-source auth, rate limit handling, and parsing logic.

**Single endpoint:** `POST https://api.brightdata.com/request`
**Auth:** `BRIGHT_DATA_API_TOKEN` env var
**Zone:** `mcp_unlocker` (default)

---

## Capabilities via Bright Data

### Search
- Google/Bing/Yandex SERP — structured JSON results (`data_format: parsed_light`)
- Already implemented in `WebSearchTool.search()`

### Scraping (bot-bypass)
- Any URL as markdown — `WebSearchTool.fetch_page(url)`
- Works on: GitHub, arXiv, HN, SEC EDGAR, academic sites, paywalled content

### Structured Extractors (use `mcp__brightData__web_data_*`)
Available as direct Bright Data dataset APIs for high-value structured data:

| Source | What it provides | Bright Data tool |
|--------|-----------------|------------------|
| GitHub repos | Stars, forks, README, issues | `web_data_github_repository_file` |
| LinkedIn companies | Company info, employee count | `web_data_linkedin_company_profile` |
| Crunchbase | Funding, investors, team | `web_data_crunchbase_company` |
| Reddit posts | Discussions, community signals | `web_data_reddit_posts` |
| YouTube videos | Transcripts, metadata | `web_data_youtube_videos` |

For sites without a dedicated extractor, use `fetch_page(url)` which returns full markdown.

---

## Replacing Individual APIs

| Was planned | Now replaced by |
|------------|-----------------|
| Semantic Scholar API | `fetch_page("https://api.semanticscholar.org/...")` or scrape paper pages |
| GitHub API | `web_data_github_repository_file` or `fetch_page(github_url)` |
| Hacker News API | `fetch_page("https://hn.algolia.com/api/...")` or scrape HN pages |
| arXiv API | `fetch_page("https://arxiv.org/abs/...")` — full paper content |
| SEC EDGAR | `fetch_page(edgar_url)` — filings as markdown |
| Wikidata SPARQL | `fetch_page(wikidata_url)` |
| npm/PyPI stats | `fetch_page("https://pypistats.org/...")` |

---

## Implementation Status

### Done ✅
- SERP search (Google) via `WebSearchTool.search()`
- Page scraping via `WebSearchTool.fetch_page()`

### Remaining 🔴
| Feature | Notes |
|---------|-------|
| **Intern uses `fetch_page` for top results** | Currently only uses SERP snippets; scraping top URLs would give full content |
| **Structured extractors wired into intern** | GitHub/Reddit/YouTube data available but not called from research pipeline |
| **PDF extraction** | Bright Data returns markdown from PDFs — needs parsing for arXiv papers |

**Status: ~40% Complete** (core search done; full-content scraping and structured extractors not yet wired in)
