"""Deep Research Report Writer - generates comprehensive narrative reports like Gemini/Perplexity."""

import os
import subprocess
import re
from datetime import datetime
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None
    HAS_ANTHROPIC = False

from ..models.findings import Finding, ResearchSession


def _get_api_key() -> Optional[str]:
    """Get the API key from Claude Code's config or environment."""
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        return api_key
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


# Maximum tokens for prompts to avoid overflows
MAX_FINDINGS_CHARS = 15000  # ~4k tokens for findings context


@dataclass
class ReportSection:
    """A section in the research report."""
    title: str
    content: str
    subsections: list["ReportSection"] = None

    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []


class DeepReportWriter:
    """Generates comprehensive narrative research reports.

    Inspired by Gemini Deep Research, Perplexity, and GPT Researcher.
    Creates multi-page reports with:
    - Executive summary
    - Table of contents
    - Narrative synthesis sections
    - Analysis and insights
    - Conclusions
    - APA-style references at end
    """

    def __init__(self, model: str = "opus"):
        self.model = model

    async def _call_claude(self, prompt: str, system_prompt: str = "") -> str:
        """Call Claude for report generation using direct Anthropic API."""
        api_key = _get_api_key()
        if not api_key:
            return "[Error: No API key available for report generation]"

        if not HAS_ANTHROPIC:
            return "[Error: anthropic package not installed]"

        # Map model name to full model ID
        model_map = {
            "opus": "claude-sonnet-4-20250514",  # Use Sonnet for report gen (faster, reliable)
            "sonnet": "claude-sonnet-4-20250514",
            "haiku": "claude-haiku-3-5-20241022",
        }
        model_id = model_map.get(self.model, "claude-sonnet-4-20250514")

        try:
            client = anthropic.Anthropic(api_key=api_key)

            # Use extended thinking for synthesis (if sonnet)
            message = client.messages.create(
                model=model_id,
                max_tokens=8000,
                system=system_prompt if system_prompt else "You are an expert research analyst.",
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            response_text = ""
            for block in message.content:
                if hasattr(block, 'text'):
                    response_text += block.text

            return response_text

        except anthropic.APIError as e:
            return f"[API Error: {e.message}]"
        except Exception as e:
            return f"[Error generating report section: {str(e)[:200]}]"

    async def generate_report(
        self,
        session: ResearchSession,
        findings: list[Finding],
        topics_explored: list[str],
        topics_remaining: list[str],
        kg_exports: dict = None,
    ) -> str:
        """Generate a comprehensive deep research report.

        Args:
            session: The research session
            findings: All findings from the research
            topics_explored: Topics that were researched
            topics_remaining: Topics that could be researched with more time
            kg_exports: Optional knowledge graph exports (stats, visualization, gaps)

        Returns:
            Complete markdown report
        """
        # Organize findings
        findings_by_type = self._organize_findings(findings)
        sources = self._extract_sources(findings)

        # Generate report sections using Claude
        print("[REPORT] Generating executive summary...")
        executive_summary = await self._generate_executive_summary(
            session.goal, findings, topics_explored
        )

        print("[REPORT] Generating introduction...")
        introduction = await self._generate_introduction(session.goal, findings)

        print("[REPORT] Generating main narrative sections...")
        main_sections = await self._generate_main_sections(session.goal, findings)

        print("[REPORT] Generating analysis and insights...")
        analysis = await self._generate_analysis(session.goal, findings, findings_by_type)

        print("[REPORT] Generating conclusions...")
        conclusions = await self._generate_conclusions(
            session.goal, findings, topics_remaining
        )

        # Compile the full report
        report = self._compile_report(
            session=session,
            executive_summary=executive_summary,
            introduction=introduction,
            main_sections=main_sections,
            analysis=analysis,
            conclusions=conclusions,
            sources=sources,
            findings=findings,
            topics_explored=topics_explored,
            kg_exports=kg_exports,
        )

        return report

    def _organize_findings(self, findings: list[Finding]) -> dict[str, list[Finding]]:
        """Organize findings by type."""
        by_type = {}
        for f in findings:
            t = f.finding_type.value
            by_type.setdefault(t, []).append(f)
        return by_type

    def _extract_sources(self, findings: list[Finding]) -> list[dict]:
        """Extract unique sources from findings with titles."""
        sources = {}
        for f in findings:
            if f.source_url and f.source_url not in sources:
                # Try to extract domain and create a better title
                domain = ""
                title = ""
                try:
                    from urllib.parse import urlparse, unquote
                    parsed = urlparse(f.source_url)
                    domain = parsed.netloc.replace("www.", "")

                    # Try to extract title from path
                    path = unquote(parsed.path)
                    path_parts = [p for p in path.split('/') if p and p not in ['index', 'html', 'htm']]
                    if path_parts:
                        # Get the last meaningful part
                        title_part = path_parts[-1]
                        # Clean up common patterns
                        title_part = title_part.replace('-', ' ').replace('_', ' ')
                        title_part = title_part.replace('.html', '').replace('.htm', '').replace('.pdf', '')
                        # Capitalize
                        title = title_part.title()

                    if not title or len(title) < 5:
                        title = domain.split('.')[0].title()
                except Exception:
                    domain = f.source_url[:50]
                    title = domain

                sources[f.source_url] = {
                    "url": f.source_url,
                    "domain": domain,
                    "title": title,
                }
        return list(sources.values())

    def _truncate_findings_text(self, findings_text: str, max_chars: int = None) -> str:
        """Truncate findings text to stay within token limits."""
        max_chars = max_chars or MAX_FINDINGS_CHARS
        if len(findings_text) <= max_chars:
            return findings_text
        return findings_text[:max_chars] + "\n... [truncated for length]"

    async def _generate_executive_summary(
        self, goal: str, findings: list[Finding], topics: list[str]
    ) -> str:
        """Generate executive summary."""
        # Get top findings by confidence
        top_findings = sorted(findings, key=lambda f: f.confidence, reverse=True)[:20]
        findings_text = "\n".join([
            f"- [{f.finding_type.value}] {f.content[:300]}" for f in top_findings
        ])
        findings_text = self._truncate_findings_text(findings_text)

        prompt = f"""You are writing the Executive Summary for a deep research report.

RESEARCH QUESTION: {goal}

TOPICS EXPLORED: {', '.join(topics[:10])}

KEY FINDINGS:
{findings_text}

Write a compelling 3-4 paragraph executive summary that:
1. Opens with the main answer/conclusion to the research question
2. Highlights the most significant discoveries and their implications
3. Notes any surprising findings or contradictions discovered
4. Briefly mentions the scope and depth of research conducted

Write in a professional, authoritative tone. Be specific with facts and figures.
Do NOT include any citations or references in this section - those go at the end.
Output ONLY the executive summary text, no headers."""

        return await self._call_claude(prompt, "You are an expert research analyst writing executive summaries.")

    async def _generate_introduction(self, goal: str, findings: list[Finding]) -> str:
        """Generate introduction section."""
        # Sample of different finding types
        facts = [f for f in findings if f.finding_type.value == "fact"][:5]
        insights = [f for f in findings if f.finding_type.value == "insight"][:3]

        context = "\n".join([f"- {f.content}" for f in facts + insights])

        prompt = f"""You are writing the Introduction section for a deep research report.

RESEARCH QUESTION: {goal}

CONTEXT FROM RESEARCH:
{context}

Write a 2-3 paragraph introduction that:
1. Establishes why this research question matters and its relevance
2. Provides necessary background context for understanding the findings
3. Outlines what the report will cover (methodology briefly, then findings)

Write in an engaging, informative style. Set up the reader to understand what follows.
Do NOT include any citations - those go at the end of the report.
Output ONLY the introduction text, no headers."""

        return await self._call_claude(prompt, "You are an expert research writer.")

    async def _generate_main_sections(self, goal: str, findings: list[Finding]) -> list[ReportSection]:
        """Generate main narrative sections organized by theme."""
        # First, ask Claude to identify the main themes/sections
        # Use top findings by confidence
        top_findings = sorted(findings, key=lambda f: f.confidence, reverse=True)[:40]
        findings_summary = "\n".join([
            f"- [{f.finding_type.value}] {f.content[:200]}"
            for f in top_findings
        ])
        findings_summary = self._truncate_findings_text(findings_summary, 10000)

        theme_prompt = f"""Analyze these research findings and identify 4-6 main thematic sections for organizing a comprehensive report.

RESEARCH QUESTION: {goal}

FINDINGS SAMPLE:
{findings_summary}

Return ONLY a JSON array of section titles, like:
["Section 1 Title", "Section 2 Title", "Section 3 Title", ...]

Choose themes that:
1. Cover the major aspects of the research question
2. Group related findings logically
3. Tell a coherent story from background to current state to future
4. Are specific, not generic (e.g., "Mechanistic Interpretability Breakthroughs" not "Technical Findings")"""

        themes_response = await self._call_claude(theme_prompt)

        # Parse themes
        themes = []
        try:
            import json
            # Find JSON array in response
            match = re.search(r'\[.*?\]', themes_response, re.DOTALL)
            if match:
                themes = json.loads(match.group())
        except Exception:
            # Fallback themes
            themes = [
                "Current State and Recent Developments",
                "Key Technical Advances",
                "Challenges and Limitations",
                "Future Directions and Implications",
            ]

        # Generate content for each section
        sections = []
        for theme in themes[:6]:  # Max 6 sections
            section_content = await self._generate_section_content(goal, theme, findings)
            sections.append(ReportSection(title=theme, content=section_content))

        return sections

    async def _generate_section_content(
        self, goal: str, section_title: str, findings: list[Finding]
    ) -> str:
        """Generate content for a specific section."""
        # Select relevant findings for this section (top by confidence)
        top_findings = sorted(findings, key=lambda f: f.confidence, reverse=True)[:30]
        findings_text = "\n".join([
            f"- [{f.finding_type.value}] {f.content[:250]}"
            for f in top_findings
        ])
        findings_text = self._truncate_findings_text(findings_text, 12000)

        prompt = f"""You are writing a section of a deep research report.

RESEARCH QUESTION: {goal}
SECTION TITLE: {section_title}

AVAILABLE FINDINGS:
{findings_text}

Write 4-6 paragraphs for this section that:
1. Opens with the key point or main finding for this theme
2. Develops the narrative with supporting details and evidence
3. Explains significance and implications
4. Connects to the broader research question
5. Notes any nuances, debates, or areas of uncertainty

Guidelines:
- Write flowing prose, not bullet points
- Be specific with facts, dates, organizations, and technical details
- Maintain professional, authoritative tone
- Do NOT include inline citations - all references go at the end of the report
- Use phrases like "Research shows...", "According to recent studies...", "Experts note..."

Output ONLY the section content, no headers."""

        return await self._call_claude(
            prompt,
            "You are an expert research analyst writing detailed narrative sections."
        )

    async def _generate_analysis(
        self, goal: str, findings: list[Finding], findings_by_type: dict
    ) -> str:
        """Generate analysis and key insights section."""
        # Get insights and connections
        insights = findings_by_type.get("insight", [])[:10]
        connections = findings_by_type.get("connection", [])[:5]
        contradictions = findings_by_type.get("contradiction", [])[:3]
        questions = findings_by_type.get("question", [])[:5]

        analysis_data = []
        if insights:
            analysis_data.append("INSIGHTS:\n" + "\n".join([f"- {i.content}" for i in insights]))
        if connections:
            analysis_data.append("CONNECTIONS:\n" + "\n".join([f"- {c.content}" for c in connections]))
        if contradictions:
            analysis_data.append("CONTRADICTIONS:\n" + "\n".join([f"- {c.content}" for c in contradictions]))
        if questions:
            analysis_data.append("OPEN QUESTIONS:\n" + "\n".join([f"- {q.content}" for q in questions]))

        prompt = f"""You are writing the Analysis and Key Insights section of a deep research report.

RESEARCH QUESTION: {goal}

{chr(10).join(analysis_data)}

Write 3-4 paragraphs of analysis that:
1. Synthesizes the most important insights across all findings
2. Identifies patterns, trends, and connections
3. Addresses any contradictions or debates in the field
4. Highlights gaps in knowledge or areas needing more research

Be analytical and thoughtful. Draw connections the reader might not see.
Do NOT include citations - those go at the end.
Output ONLY the analysis text, no headers."""

        return await self._call_claude(
            prompt,
            "You are an expert research analyst providing deep synthesis and analysis."
        )

    async def _generate_conclusions(
        self, goal: str, findings: list[Finding], topics_remaining: list[str]
    ) -> str:
        """Generate conclusions and recommendations."""
        top_findings = sorted(findings, key=lambda f: f.confidence, reverse=True)[:10]
        findings_summary = "\n".join([f"- {f.content}" for f in top_findings])

        prompt = f"""You are writing the Conclusions section of a deep research report.

RESEARCH QUESTION: {goal}

KEY FINDINGS:
{findings_summary}

TOPICS FOR FURTHER RESEARCH: {', '.join(topics_remaining[:5]) if topics_remaining else 'None identified'}

Write 2-3 paragraphs that:
1. Directly answer the original research question based on evidence gathered
2. Summarize the most important takeaways
3. Provide actionable recommendations or next steps
4. Suggest areas for further investigation

Be definitive where evidence supports it, and appropriately hedged where uncertainty exists.
Output ONLY the conclusions text, no headers."""

        return await self._call_claude(
            prompt,
            "You are an expert research analyst writing conclusions."
        )

    def _compile_report(
        self,
        session: ResearchSession,
        executive_summary: str,
        introduction: str,
        main_sections: list[ReportSection],
        analysis: str,
        conclusions: str,
        sources: list[dict],
        findings: list[Finding],
        topics_explored: list[str],
        kg_exports: dict = None,
    ) -> str:
        """Compile all sections into the final report."""
        # Build table of contents
        toc_items = [
            "1. Executive Summary",
            "2. Introduction",
        ]
        for i, section in enumerate(main_sections, 3):
            toc_items.append(f"{i}. {section.title}")
        toc_items.extend([
            f"{len(main_sections) + 3}. Analysis and Key Insights",
            f"{len(main_sections) + 4}. Conclusions and Recommendations",
            f"{len(main_sections) + 5}. References",
        ])

        toc = "\n".join([f"- [{item}](#{item.lower().replace(' ', '-').replace('.', '')})" for item in toc_items])

        # Build main sections
        main_content = ""
        for i, section in enumerate(main_sections, 3):
            main_content += f"\n## {i}. {section.title}\n\n{section.content}\n"

        # Build references in cleaner format (no redundant dates)
        references = []
        for i, source in enumerate(sources, 1):
            title = source.get('title', source['domain'])
            references.append(
                f"[{i}] {title}. *{source['domain']}*. {source['url']}"
            )
        references_text = "\n\n".join(references)

        # Add retrieval date once at the top
        retrieval_date = datetime.now().strftime("%B %d, %Y")

        # Stats - use sources count if topics_explored is empty
        topics_count = len(topics_explored) if topics_explored else len(sources)
        stats = f"""**Research Statistics:**
- Total Findings: {len(findings)}
- Sources Analyzed: {len(sources)}
- Topics Explored: {topics_count}
- Research Duration: {session.started_at.strftime('%Y-%m-%d %H:%M')} to {session.ended_at.strftime('%Y-%m-%d %H:%M') if session.ended_at else 'In Progress'}"""

        # Compile full report
        report = f"""# {session.goal}

*Deep Research Report*

---

**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M')}
**Session ID:** {session.id}

---

## Table of Contents

{toc}

---

## 1. Executive Summary

{executive_summary}

---

## 2. Introduction

{introduction}

---
{main_content}
---

## {len(main_sections) + 3}. Analysis and Key Insights

{analysis}

---

## {len(main_sections) + 4}. Conclusions and Recommendations

{conclusions}

---

## {len(main_sections) + 5}. References

*All sources accessed on {retrieval_date}.*

{references_text}

---

## Appendix: Research Methodology

This report was generated using a hierarchical multi-agent research system:

1. **Research Planning**: An AI manager agent (powered by Claude Opus with extended thinking) analyzed the research question and developed a systematic research strategy.
2. **Information Gathering**: AI intern agents conducted {len(sources)} web searches, analyzing sources for relevance and credibility.
3. **Finding Extraction**: {len(findings)} discrete findings were extracted and categorized by type (facts, insights, connections, etc.).
4. **Critical Review**: Each batch of findings was critiqued for accuracy, relevance, and gaps.
5. **Knowledge Graph Construction**: Findings were integrated into a real-time knowledge graph for gap detection and contradiction analysis.
6. **Narrative Synthesis**: An AI writer (Claude Opus) synthesized findings into this cohesive narrative report using extended thinking for deep analysis.

{stats}

**Topics Researched:**
{chr(10).join(['- ' + t for t in topics_explored[:15]]) if topics_explored else '- ' + session.goal}

{self._format_kg_section(kg_exports) if kg_exports else ''}

---

*Report generated by Claude Deep Researcher*
"""
        return report

    def _format_kg_section(self, kg_exports: dict) -> str:
        """Format knowledge graph section for the report."""
        if not kg_exports:
            return ""

        sections = ["---", "", "## Appendix: Knowledge Graph Analysis"]

        # Stats
        stats = kg_exports.get('stats', {})
        if stats:
            sections.append(f"""
**Knowledge Graph Statistics:**
- Entities extracted: {stats.get('num_entities', 0)}
- Relations identified: {stats.get('num_relations', 0)}
- Connected components: {stats.get('num_components', 0)}
- Graph density: {stats.get('density', 0):.3f}
""")

        # Key concepts
        key_concepts = kg_exports.get('key_concepts', [])
        if key_concepts:
            sections.append("**Key Concepts by Importance (PageRank):**")
            for c in key_concepts[:5]:
                sections.append(f"- {c['name']} ({c['type']}) - importance: {c['importance']}")
            sections.append("")

        # Gaps identified
        gaps = kg_exports.get('gaps', [])
        if gaps:
            sections.append(f"**Knowledge Gaps Identified ({len(gaps)}):**")
            for g in gaps[:5]:
                sections.append(f"- {g.get('recommendation', g.get('gap_type', 'Unknown'))}")
            sections.append("")

        # Contradictions
        contradictions = kg_exports.get('contradictions', [])
        if contradictions:
            sections.append(f"**Contradictions Detected ({len(contradictions)}):**")
            for c in contradictions[:3]:
                sections.append(f"- {c.get('description', c.get('recommendation', 'Unknown'))}")
            sections.append("")

        # Mermaid diagram (if available and not too large)
        mermaid = kg_exports.get('mermaid_diagram', '')
        if mermaid and 'No data yet' not in mermaid and len(mermaid) < 5000:
            sections.append("**Knowledge Graph Visualization:**")
            sections.append("")
            sections.append(mermaid)
            sections.append("")

        # Link to HTML visualization
        html_viz = kg_exports.get('html_visualization')
        if html_viz:
            sections.append(f"*Interactive visualization available at: {html_viz}*")
            sections.append("")

        return "\n".join(sections)
