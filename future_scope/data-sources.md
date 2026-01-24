# Data Sources for claude-researcher

*External APIs and tools to enhance research quality*

---

## Overview

This document catalogs external data sources that can be integrated into claude-researcher to improve research depth, verify facts, and provide real-time signals. Each source is evaluated for rate limits, cost, reliability, and implementation complexity.

---

## 1. Academic/Paper Access

| Tool | Rate Limit | Cost | Reliability | Notes |
|------|------------|------|-------------|-------|
| **arXiv API** | 1 req/3 sec (20/min) | Free | High | Max 30k results, PDF downloads need slower rate |
| **Semantic Scholar** | 1 req/sec (auth), shared pool (unauth) | Free | High | Can request higher limits. Batch endpoints available |
| **CrossRef** | 50 req/sec (polite pool with email) | Free | High | Add email to User-Agent for polite pool |

### arXiv API
- **Endpoint:** `http://export.arxiv.org/api/query`
- **Auth:** None required
- **Best for:** Fetching paper abstracts, metadata, author info
- **Limitations:** Recent issues with pagination >1000 results
- **Docs:** https://info.arxiv.org/help/api/index.html

### Semantic Scholar API
- **Endpoint:** `https://api.semanticscholar.org/graph/v1/`
- **Auth:** Optional API key (recommended)
- **Best for:** Citation graphs, finding influential papers, related work
- **MCP Server:** https://glama.ai/mcp/servers/@fegizii/SemanticScholarMCP
- **Docs:** https://www.semanticscholar.org/product/api

### CrossRef API
- **Endpoint:** `https://api.crossref.org/`
- **Auth:** None, add email to User-Agent for higher limits
- **Best for:** DOI resolution, publication metadata, citation counts

**Recommendation:** Semantic Scholar highest value - citation graphs help identify authoritative sources.

---

## 2. Structured Data Sources

| Tool | Rate Limit | Cost | Reliability | Notes |
|------|------------|------|-------------|-------|
| **Wikidata SPARQL** | 60 sec query time/min | Free | Medium | 429 errors lead to escalating bans |
| **Wolfram Alpha** | 2,000/month (free) | Free tier, then $25/mo | High | Great for computational facts |
| **data.gov** | Varies by dataset | Free | High | US government data |

### Wikidata SPARQL
- **Endpoint:** `https://query.wikidata.org/sparql`
- **Auth:** None
- **Best for:** Entity verification, structured facts, relationships
- **Caution:** HTTP 429 → 1 hour ban, escalates with repeated violations
- **Docs:** https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service

### Wolfram Alpha API
- **Endpoint:** `https://api.wolframalpha.com/v2/query`
- **Auth:** API key required
- **Pricing:** Free 2,000/month, then $25/month
- **Docs:** https://products.wolframalpha.com/api

**Recommendation:** Wikidata most useful for entity verification in CoVe pipeline.

---

## 3. Code/Tech Intelligence

| Tool | Rate Limit | Cost | Reliability | Notes |
|------|------------|------|-------------|-------|
| **GitHub API** | 5,000 req/hr (auth), 60/hr (unauth) | Free | Very High | Personal access token recommended |
| **npm Registry** | 5,000 req/hr (auth), 1,000/hr (unauth) | Free | High | Download stats limited to 18 months |
| **PyPI Stats** | ~100 req/min | Free | High | Via pypistats.org or BigQuery |

### GitHub API
- **Endpoint:** `https://api.github.com/`
- **Auth:** Personal access token (recommended)
- **Best for:** Repository popularity, real adoption signals, contributor analysis
- **Docs:** https://docs.github.com/en/rest

### npm Registry API
- **Endpoint:** `https://registry.npmjs.org/`
- **Stats:** `https://api.npmjs.org/downloads/`
- **Limitation:** Download stats capped at 18 months

### PyPI Stats
- **Endpoint:** `https://pypistats.org/api/`
- **Alternative:** Google BigQuery public dataset

**Recommendation:** GitHub API highest value - shows what developers actually use vs what gets hyped.

---

## 4. Source Credibility

| Tool | Rate Limit | Cost | Reliability | Notes |
|------|------------|------|-------------|-------|
| **Wayback Machine** | 60 req/min | Free | Medium | Ban escalation on violations |
| **Google Fact Check API** | Not documented | Free | Medium | Limited coverage |
| **WHOIS Lookup** | Varies | Free-$50/mo | High | Many free options |

### Wayback Machine (Internet Archive)
- **CDX Endpoint:** `https://web.archive.org/cdx/search/cdx`
- **Availability:** `https://archive.org/wayback/available`
- **Best for:** Verifying sources haven't been modified, finding historical versions
- **Caution:** 60 req/min, violations → 1hr ban (doubles each time)
- **Docs:** https://archive.org/developers/wayback-cdx-server.html

### Google Fact Check Tools API
- **Endpoint:** `https://factchecktools.googleapis.com/v1alpha1/claims:search`
- **Auth:** API key required
- **Limitation:** Limited database coverage
- **Docs:** https://developers.google.com/fact-check/tools/api

**Recommendation:** Wayback Machine most useful but handle rate limits carefully.

---

## 5. Real-time Signals

| Tool | Rate Limit | Cost | Reliability | Notes |
|------|------------|------|-------------|-------|
| **Reddit API** | 100 req/min (OAuth) | Free (non-commercial) | Medium | 1000 post cap per subreddit |
| **Hacker News (Firebase)** | **No limit** | Free | Very High | Best deal available |
| **HN Algolia Search** | 10,000 req/hr | Free | Very High | Better for search |

### Hacker News API (Firebase)
- **Endpoint:** `https://hacker-news.firebaseio.com/v0/`
- **Auth:** None
- **Rate Limit:** None documented - very generous
- **Best for:** Tech community pulse, trending discussions
- **Docs:** https://github.com/HackerNews/API

### HN Algolia Search API
- **Endpoint:** `https://hn.algolia.com/api/v1/`
- **Rate Limit:** 10,000 requests/hour per IP
- **Best for:** Searching HN history, finding topic discussions
- **Docs:** https://hn.algolia.com/api

### Reddit API
- **Endpoint:** `https://oauth.reddit.com/`
- **Auth:** OAuth 2.0 required
- **Pricing:** Free (non-commercial), $0.24/1k calls (commercial)
- **Limitation:** 1000 post cap per subreddit
- **Docs:** https://www.reddit.com/dev/api/

**Recommendation:** Hacker News is gold - no limits, free, high-quality tech discussions.

---

## 6. Document Processing

| Tool | Rate Limit | Cost | Reliability | Notes |
|------|------------|------|-------------|-------|
| **PyMuPDF** | N/A (local) | Free | High | Fast PDF parsing |
| **pdfplumber** | N/A (local) | Free | High | Good table extraction |
| **Marker** | N/A (local) | Free | High | Best for academic PDFs |
| **Tesseract OCR** | N/A (local) | Free | Medium | Open-source OCR |

### Installation

```bash
# PDF Processing
pip install pymupdf pdfplumber marker-pdf

# OCR
sudo apt install tesseract-ocr
pip install pytesseract
```

### Usage
- **PyMuPDF:** Fast text extraction, metadata
- **pdfplumber:** Table extraction, structured content
- **Marker:** Academic papers, multi-column layouts, equations
- **Tesseract:** Scanned documents, images with text

**Recommendation:** All local - no API limits or costs. Use Marker for academic PDFs.

---

## 7. Business/Economic Data

| Tool | Rate Limit | Cost | Reliability | Notes |
|------|------------|------|-------------|-------|
| **SEC EDGAR** | 10 req/sec | Free | Very High | Requires User-Agent |
| **Crunchbase** | 200 req/min (free) | Free basic, $99+/mo pro | High | Limited free tier |
| **Glassdoor** | N/A | N/A | Low | No official API |

### SEC EDGAR
- **Endpoint:** `https://data.sec.gov/`
- **Auth:** None, but User-Agent required
- **User-Agent Format:** `Company Name admin@company.com`
- **Best for:** Company financials, insider trading, executive compensation
- **Docs:** https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data

### Crunchbase API
- **Endpoint:** `https://api.crunchbase.com/v4/`
- **Free Tier:** 200 req/min, limited fields, non-commercial only
- **Paid:** $99-199/month
- **Docs:** https://data.crunchbase.com/docs

**Recommendation:** SEC EDGAR excellent for public companies - free, reliable, authoritative.

---

## Implementation Priority

### Tier 1: Implement First
| Tool | Reason |
|------|--------|
| **Semantic Scholar** | Citation graphs for source authority, MCP server exists |
| **GitHub API** | Real adoption signals, generous limits |
| **Hacker News API** | No limits, tech pulse, free |

### Tier 2: High Value
| Tool | Reason |
|------|--------|
| **SEC EDGAR** | Authoritative business data, 10 req/sec |
| **arXiv API** | Academic papers, free |
| **Wikidata** | Structured facts for CoVe verification |

### Tier 3: Situational
| Tool | Reason |
|------|--------|
| **Wayback Machine** | Useful but fragile rate limits |
| **npm/PyPI Stats** | Good for tech research specifically |
| **Reddit** | Valuable but restrictive ToS |

### Skip for Now
| Tool | Reason |
|------|--------|
| **Glassdoor** | No API |
| **Crunchbase** | Limited free tier |
| **Google Fact Check** | Limited database coverage |

---

## Cost Summary

| Tool | Monthly Cost |
|------|-------------|
| arXiv | $0 |
| Semantic Scholar | $0 |
| CrossRef | $0 |
| Wikidata | $0 |
| GitHub | $0 |
| npm/PyPI | $0 |
| Hacker News | $0 |
| SEC EDGAR | $0 |
| Wayback Machine | $0 |
| Wolfram Alpha | $0-25 |
| Reddit | $0 (non-commercial) |
| Crunchbase | $0-99 |

**Total for Tier 1+2 tools: $0/month**

---

## Rate Limit Handling

```python
import time
from functools import wraps

def rate_limited(max_per_second):
    """Decorator for rate limiting API calls."""
    min_interval = 1.0 / max_per_second
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait = min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

def exponential_backoff(func, max_retries=5, base_delay=1):
    """Retry with exponential backoff on 429 errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
```

---

## Integration as Intern Tools

```python
TOOLS = [
    {
        "name": "semantic_scholar_search",
        "description": "Search academic papers and get citation graphs",
        "parameters": {"query": "string", "limit": "int"}
    },
    {
        "name": "github_repo_stats",
        "description": "Get repository statistics to gauge real adoption",
        "parameters": {"repo": "owner/repo"}
    },
    {
        "name": "hn_search",
        "description": "Search Hacker News discussions on a topic",
        "parameters": {"query": "string", "sort": "date|points"}
    },
    {
        "name": "wikidata_entity",
        "description": "Verify entity facts from Wikidata",
        "parameters": {"entity": "string"}
    }
]
```

---

## References

- arXiv API: https://info.arxiv.org/help/api/index.html
- Semantic Scholar: https://www.semanticscholar.org/product/api
- Wikidata: https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service
- GitHub API: https://docs.github.com/en/rest
- Hacker News API: https://github.com/HackerNews/API
- SEC EDGAR: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
- Wayback Machine: https://archive.org/developers/wayback-cdx-server.html
