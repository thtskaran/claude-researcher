"""Source credibility scoring system."""

from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse
from typing import Optional


@dataclass
class CredibilityScore:
    """Credibility score with component signals."""
    score: float  # 0.0 to 1.0
    signals: dict[str, float]
    domain: str
    url: str


class CredibilityScorer:
    """Automated source credibility scoring.

    Aggregates multiple signals to assess source reliability:
    - Domain authority (institutional, academic, government)
    - Recency of publication
    - Domain reputation patterns
    """

    # High credibility domains
    HIGH_CREDIBILITY_PATTERNS = [
        '.edu',
        '.gov',
        'nature.com',
        'science.org',
        'pubmed.ncbi.nlm.nih.gov',
        'arxiv.org',
        'scholar.google',
        'ieee.org',
        'acm.org',
        'springer.com',
        'wiley.com',
        'sciencedirect.com',
        'pnas.org',
        'cell.com',
        'thelancet.com',
        'nejm.org',
        'bmj.com',
        'nih.gov',
        'cdc.gov',
        'who.int',
    ]

    # Medium credibility domains
    MEDIUM_CREDIBILITY_PATTERNS = [
        'wikipedia.org',
        'britannica.com',
        'reuters.com',
        'apnews.com',
        'bbc.com',
        'bbc.co.uk',
        'nytimes.com',
        'washingtonpost.com',
        'theguardian.com',
        'economist.com',
        'forbes.com',
        'wired.com',
        'techcrunch.com',
        'arstechnica.com',
        'github.com',
        'stackoverflow.com',
        'medium.com',
        'substack.com',
    ]

    # Low credibility patterns (to be penalized)
    LOW_CREDIBILITY_PATTERNS = [
        'reddit.com',
        'twitter.com',
        'x.com',
        'facebook.com',
        'tiktok.com',
        'pinterest.com',
        'quora.com',
    ]

    # Credibility weights
    WEIGHTS = {
        'domain_authority': 0.35,
        'recency': 0.20,
        'source_type': 0.25,
        'https': 0.10,
        'path_depth': 0.10,
    }

    def score_source(self, url: str, publication_date: Optional[str] = None) -> CredibilityScore:
        """Score a source's credibility.

        Args:
            url: The source URL
            publication_date: Optional publication date (ISO format)

        Returns:
            CredibilityScore with overall score and component signals
        """
        signals = {}

        # Parse URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace('www.', '')
        except Exception:
            domain = url[:50]
            return CredibilityScore(score=0.3, signals={'error': 1.0}, domain=domain, url=url)

        # Domain authority
        signals['domain_authority'] = self._score_domain_authority(domain)

        # Source type based on URL patterns
        signals['source_type'] = self._score_source_type(url, domain)

        # Recency
        signals['recency'] = self._score_recency(publication_date)

        # HTTPS
        signals['https'] = 1.0 if parsed.scheme == 'https' else 0.5

        # Path depth (very deep paths can indicate less authoritative content)
        path_parts = [p for p in parsed.path.split('/') if p]
        if len(path_parts) <= 3:
            signals['path_depth'] = 1.0
        elif len(path_parts) <= 5:
            signals['path_depth'] = 0.8
        else:
            signals['path_depth'] = 0.6

        # Calculate weighted score
        final_score = sum(
            signals[k] * self.WEIGHTS.get(k, 0)
            for k in signals
            if k in self.WEIGHTS
        )

        # Clamp to valid range
        final_score = max(0.0, min(1.0, final_score))

        return CredibilityScore(
            score=final_score,
            signals=signals,
            domain=domain,
            url=url,
        )

    def score_source_with_audit(
        self, url: str, publication_date: Optional[str] = None
    ) -> tuple["CredibilityScore", dict]:
        """Score a source's credibility and return audit data.

        This method extends score_source() to also return a detailed audit
        dictionary with all signal scores and weighted contributions for
        persistence to the credibility_audit table.

        Args:
            url: The source URL
            publication_date: Optional publication date (ISO format)

        Returns:
            Tuple of (CredibilityScore, audit_dict)
            audit_dict contains:
                - domain_authority_score
                - recency_score
                - source_type_score
                - https_score
                - path_depth_score
                - weighted_contributions (how each signal contributed to final)
                - credibility_label
        """
        score = self.score_source(url, publication_date)

        # Calculate weighted contributions
        weighted_contributions = {
            k: score.signals.get(k, 0) * self.WEIGHTS.get(k, 0)
            for k in self.WEIGHTS
        }

        audit = {
            "url": url,
            "domain": score.domain,
            "final_score": score.score,
            "domain_authority_score": score.signals.get("domain_authority", 0.0),
            "recency_score": score.signals.get("recency", 0.5),
            "source_type_score": score.signals.get("source_type", 0.6),
            "https_score": score.signals.get("https", 0.5),
            "path_depth_score": score.signals.get("path_depth", 0.8),
            "credibility_label": self.get_credibility_label(score.score),
            "weighted_contributions": weighted_contributions,
        }

        return score, audit

    def _score_domain_authority(self, domain: str) -> float:
        """Score domain based on authority patterns."""
        domain_lower = domain.lower()

        # Check high credibility patterns
        for pattern in self.HIGH_CREDIBILITY_PATTERNS:
            if pattern in domain_lower:
                return 0.95

        # Check medium credibility patterns
        for pattern in self.MEDIUM_CREDIBILITY_PATTERNS:
            if pattern in domain_lower:
                return 0.75

        # Check low credibility patterns
        for pattern in self.LOW_CREDIBILITY_PATTERNS:
            if pattern in domain_lower:
                return 0.35

        # Default for unknown domains
        return 0.5

    def _score_source_type(self, url: str, domain: str) -> float:
        """Score based on inferred source type."""
        url_lower = url.lower()

        # Academic papers
        if any(x in url_lower for x in ['/paper/', '/article/', '/pdf/', '.pdf', '/doi/']):
            return 0.9

        # Documentation
        if any(x in url_lower for x in ['/docs/', '/documentation/', '/api/', '/reference/']):
            return 0.85

        # Blog posts (can be high or low quality)
        if any(x in url_lower for x in ['/blog/', '/post/', '/article/']):
            return 0.65

        # News
        if any(x in url_lower for x in ['/news/', '/press/', '/release/']):
            return 0.7

        # Default
        return 0.6

    def _score_recency(self, publication_date: Optional[str]) -> float:
        """Score based on how recent the source is."""
        if not publication_date:
            return 0.5  # Unknown date, neutral score

        try:
            # Try to parse the date
            if 'T' in publication_date:
                pub_date = datetime.fromisoformat(publication_date.replace('Z', '+00:00'))
            else:
                pub_date = datetime.fromisoformat(publication_date)

            days_old = (datetime.now() - pub_date.replace(tzinfo=None)).days

            if days_old <= 30:
                return 1.0  # Very recent
            elif days_old <= 90:
                return 0.9
            elif days_old <= 365:
                return 0.8
            elif days_old <= 730:  # 2 years
                return 0.7
            elif days_old <= 1825:  # 5 years
                return 0.5
            else:
                return 0.3  # Old source
        except Exception:
            return 0.5  # Could not parse, neutral score

    def get_credibility_label(self, score: float) -> str:
        """Get a human-readable credibility label."""
        if score >= 0.85:
            return "High"
        elif score >= 0.65:
            return "Medium-High"
        elif score >= 0.5:
            return "Medium"
        elif score >= 0.35:
            return "Low-Medium"
        else:
            return "Low"
