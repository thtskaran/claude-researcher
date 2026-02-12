"""Deep Research Report Writer - generates comprehensive narrative reports like Gemini/Perplexity."""

import asyncio
import json
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

from ..models.findings import Finding, ResearchSession


class SectionType(str, Enum):
    """Types of sections that can appear in a dynamic report."""

    TLDR = "tldr"  # 2-3 sentence summary
    FLASH_NUMBERS = "flash_numbers"  # Key metrics callouts
    STATS_TABLE = "stats_table"  # Tabular comparisons
    COMPARISON = "comparison"  # Side-by-side analysis
    TIMELINE = "timeline"  # Chronological view
    NARRATIVE = "narrative"  # Standard prose section
    ANALYSIS = "analysis"  # Deep synthesis
    GAPS = "gaps"  # Open questions
    CONCLUSIONS = "conclusions"
    REFERENCES = "references"


@dataclass
class PlannedSection:
    """A planned section in the dynamic report structure."""

    section_type: SectionType
    title: str
    description: str
    priority: int = 5
    content: str = ""


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

    async def _call_claude(
        self, prompt: str, system_prompt: str = "", model: str | None = None
    ) -> str:
        """Call Claude for report generation using Claude Agent SDK.

        Uses claude_agent_sdk.query() which works with both API keys and OAuth
        authentication (normal Claude Code accounts).
        Falls back to cheaper models if the primary model fails (e.g. rate limits).
        """
        # Combine system prompt with user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        use_model = model or self.model

        options = ClaudeAgentOptions(
            model=use_model,
            max_turns=1,  # Single turn for report generation
            allowed_tools=[],  # No tools needed for text generation
        )

        response_text = ""
        max_retries = 5
        base_delay = 5.0  # Start with 5s delay for rate limit recovery

        for attempt in range(max_retries):
            try:
                async for message in query(prompt=full_prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                response_text += block.text
                break  # Success, exit retry loop

            except Exception as e:
                error_str = str(e).lower()
                print(f"[REPORT] Attempt {attempt + 1}/{max_retries} failed ({use_model}): {e}")

                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) + random.uniform(0, 2.0)
                    print(f"[REPORT] Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    response_text = ""  # Reset for retry
                else:
                    # All retries exhausted with primary model — try fallback
                    if use_model == "opus":
                        print(
                            f"[REPORT] Opus failed after {max_retries} attempts, falling back to sonnet..."
                        )
                        return await self._call_claude(prompt, system_prompt, model="sonnet")
                    return f"[Error generating report section: {str(e)[:200]}]"

        return response_text

    async def generate_report(
        self,
        session: ResearchSession,
        findings: list[Finding],
        topics_explored: list[str],
        topics_remaining: list[str],
        kg_exports: dict = None,
        dynamic: bool = True,
        verification_metrics: dict = None,
        progress_callback: Callable[[str, int], Awaitable[None]] | None = None,
    ) -> str:
        """Generate a comprehensive deep research report.

        Args:
            session: The research session
            findings: All findings from the research
            topics_explored: Topics that were researched
            topics_remaining: Topics that could be researched with more time
            kg_exports: Optional knowledge graph exports (stats, visualization, gaps)
            dynamic: If True, use AI-driven dynamic section planning
            verification_metrics: Optional verification metrics from the pipeline

        Returns:
            Complete markdown report
        """
        # Use dynamic report generation by default
        if dynamic:
            return await self.generate_dynamic_report(
                session=session,
                findings=findings,
                topics_explored=topics_explored,
                topics_remaining=topics_remaining,
                kg_exports=kg_exports,
                verification_metrics=verification_metrics,
                progress_callback=progress_callback,
            )

        # Fallback to legacy fixed structure
        findings_by_type = self._organize_findings(findings)
        sources = self._extract_sources(findings)

        # Generate report sections using Claude
        await _emit_progress(progress_callback, "Generating executive summary...", 10)
        print("[REPORT] Generating executive summary...")
        executive_summary = await self._generate_executive_summary(
            session.goal, findings, topics_explored
        )

        await _emit_progress(progress_callback, "Generating introduction...", 25)
        print("[REPORT] Generating introduction...")
        introduction = await self._generate_introduction(session.goal, findings)

        await _emit_progress(progress_callback, "Generating main sections...", 45)
        print("[REPORT] Generating main narrative sections...")
        main_sections = await self._generate_main_sections(session.goal, findings)

        await _emit_progress(progress_callback, "Generating analysis and insights...", 70)
        print("[REPORT] Generating analysis and insights...")
        analysis = await self._generate_analysis(session.goal, findings, findings_by_type)

        await _emit_progress(progress_callback, "Generating conclusions...", 85)
        print("[REPORT] Generating conclusions...")
        conclusions = await self._generate_conclusions(session.goal, findings, topics_remaining)

        # Compile the full report
        await _emit_progress(progress_callback, "Compiling report...", 95)
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

        await _emit_progress(progress_callback, "Report complete", 100)
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
                    from urllib.parse import unquote, urlparse

                    parsed = urlparse(f.source_url)
                    domain = parsed.netloc.replace("www.", "")

                    # Try to extract title from path
                    path = unquote(parsed.path)
                    path_parts = [
                        p for p in path.split("/") if p and p not in ["index", "html", "htm"]
                    ]
                    if path_parts:
                        # Get the last meaningful part
                        title_part = path_parts[-1]
                        # Clean up common patterns
                        title_part = title_part.replace("-", " ").replace("_", " ")
                        title_part = (
                            title_part.replace(".html", "").replace(".htm", "").replace(".pdf", "")
                        )
                        # Capitalize
                        title = title_part.title()

                    if not title or len(title) < 5:
                        title = domain.split(".")[0].title()
                except Exception:
                    domain = f.source_url[:50]
                    title = domain

                sources[f.source_url] = {
                    "url": f.source_url,
                    "domain": domain,
                    "title": title,
                }
        return list(sources.values())

    def _format_finding_for_prompt(self, f: Finding) -> str:
        """Format a single finding for inclusion in LLM prompts.

        Includes verification badge, extended content (500 chars), and source reference.
        """
        # Verification badge
        vstatus = f.verification_status or "unverified"
        if vstatus == "verified":
            badge = "[VERIFIED]"
        elif vstatus == "flagged":
            badge = "[FLAGGED]"
        elif vstatus == "rejected":
            badge = "[REJECTED]"
        else:
            badge = "[UNVERIFIED]"

        # Source reference with citation number
        source_ref = ""
        if f.source_url and hasattr(self, "_source_index"):
            ref_num = self._source_index.get(f.source_url)
            if ref_num:
                try:
                    from urllib.parse import urlparse

                    domain = urlparse(f.source_url).netloc.replace("www.", "")
                except Exception:
                    domain = f.source_url[:40]
                source_ref = f" [Source: [{ref_num}] {domain}]"

        content = f.content[:500]
        return f"- [{f.finding_type.value.upper()}] {badge} {content}{source_ref}"

    def _format_findings_block(self, findings: list[Finding], max_chars: int = None) -> str:
        """Format a list of findings for prompt inclusion, with truncation."""
        max_chars = max_chars or MAX_FINDINGS_CHARS
        lines = [self._format_finding_for_prompt(f) for f in findings]
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated for length]"
        return text

    def _select_findings_for_section(
        self,
        findings: list[Finding],
        section_title: str,
        section_description: str = "",
        section_type: SectionType = SectionType.NARRATIVE,
        max_findings: int = 20,
    ) -> list[Finding]:
        """Select the most relevant findings for a given section.

        Scores each finding by keyword match, type affinity, verification status, and confidence.
        """
        # Build keywords from section title + description
        stop_words = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "and",
            "or",
            "is",
            "are",
            "this",
            "that",
            "with",
            "from",
        }
        raw_keywords = (section_title + " " + section_description).lower().split()
        keywords = [w for w in raw_keywords if w not in stop_words and len(w) > 2]

        # Type affinity: which finding types are preferred by each section type
        type_affinity: dict[SectionType, set[str]] = {
            SectionType.ANALYSIS: {"insight", "connection", "contradiction"},
            SectionType.GAPS: {"question", "contradiction"},
            SectionType.CONCLUSIONS: {"insight", "fact", "connection"},
            SectionType.NARRATIVE: {"fact", "insight"},
            SectionType.COMPARISON: {"fact", "connection"},
            SectionType.TIMELINE: {"fact"},
            SectionType.FLASH_NUMBERS: {"fact"},
            SectionType.STATS_TABLE: {"fact"},
            SectionType.TLDR: {"fact", "insight"},
        }
        preferred_types = type_affinity.get(section_type, {"fact", "insight"})

        # Verification weight map
        verification_weights = {
            "verified": 3.0,
            "unverified": 1.5,
            "flagged": 1.0,
            "rejected": -2.0,
        }

        scored: list[tuple[float, Finding]] = []
        for f in findings:
            score = 0.0

            # 1. Keyword match (0-3 pts)
            text = (f.content + " " + (f.search_query or "")).lower()
            kw_hits = sum(1 for kw in keywords if kw in text)
            score += min(3.0, kw_hits)

            # 2. Type affinity (0-2 pts)
            if f.finding_type.value in preferred_types:
                score += 2.0

            # 3. Verification-weighted confidence (0-3 pts)
            vstatus = f.verification_status or "unverified"
            score += verification_weights.get(vstatus, 1.5)
            score += (f.kg_support_score or 0.0) * 1.0

            # 4. Raw confidence as tiebreaker (0-1 pts)
            score += f.confidence

            scored.append((score, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:max_findings]]

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
        findings_text = "\n".join(
            [f"- [{f.finding_type.value}] {f.content[:300]}" for f in top_findings]
        )
        findings_text = self._truncate_findings_text(findings_text)

        prompt = f"""You are writing the Executive Summary for a deep research report.

RESEARCH QUESTION: {goal}

TOPICS EXPLORED: {", ".join(topics[:10])}

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

        return await self._call_claude(
            prompt, "You are an expert research analyst writing executive summaries."
        )

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

    async def _generate_main_sections(
        self, goal: str, findings: list[Finding]
    ) -> list[ReportSection]:
        """Generate main narrative sections organized by theme."""
        # First, ask Claude to identify the main themes/sections
        # Use top findings by confidence
        top_findings = sorted(findings, key=lambda f: f.confidence, reverse=True)[:40]
        findings_summary = "\n".join(
            [f"- [{f.finding_type.value}] {f.content[:200]}" for f in top_findings]
        )
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
            # Find JSON array in response
            match = re.search(r"\[.*\]", themes_response, re.DOTALL)
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
        """Generate content for a specific section (legacy path)."""
        selected = self._select_findings_for_section(
            findings,
            section_title,
            "",
            section_type=SectionType.NARRATIVE,
            max_findings=20,
        )
        findings_text = self._format_findings_block(selected, 12000)

        prompt = f"""You are writing a section of a deep research report.

RESEARCH QUESTION: {goal}
SECTION TITLE: {section_title}

AVAILABLE FINDINGS:
{findings_text}

CITATION INSTRUCTIONS:
- Each finding includes a source like [Source: [N] domain].
- Cite claims inline using [N] notation, e.g., "accuracy improved 40% [3]."
- Aim for 3-8 citations per section. Do NOT list references at the end.

Write 4-6 paragraphs for this section that:
1. Opens with the key point or main finding for this theme
2. Develops the narrative with supporting details and evidence
3. Explains significance and implications
4. Connects to the broader research question
5. Notes any nuances, debates, or areas of uncertainty

Guidelines:
- Write flowing prose, not bullet points
- Name specific sources, organizations, or tools when available
- Use exact numbers, dates, and details from findings

Output ONLY the section content, no headers."""

        return await self._call_claude(
            prompt,
            "You are writing a sourced, evidence-based research report. Every major claim must cite its source using [N].",
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
            analysis_data.append(
                "CONNECTIONS:\n" + "\n".join([f"- {c.content}" for c in connections])
            )
        if contradictions:
            analysis_data.append(
                "CONTRADICTIONS:\n" + "\n".join([f"- {c.content}" for c in contradictions])
            )
        if questions:
            analysis_data.append(
                "OPEN QUESTIONS:\n" + "\n".join([f"- {q.content}" for q in questions])
            )

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
            prompt, "You are an expert research analyst providing deep synthesis and analysis."
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

TOPICS FOR FURTHER RESEARCH: {", ".join(topics_remaining[:5]) if topics_remaining else "None identified"}

Write 2-3 paragraphs that:
1. Directly answer the original research question based on evidence gathered
2. Summarize the most important takeaways
3. Provide actionable recommendations or next steps
4. Suggest areas for further investigation

Be definitive where evidence supports it, and appropriately hedged where uncertainty exists.
Output ONLY the conclusions text, no headers."""

        return await self._call_claude(
            prompt, "You are an expert research analyst writing conclusions."
        )

    def _get_representative_findings(
        self, findings: list[Finding], max_total: int = 80
    ) -> list[Finding]:
        """Get a stratified sample of findings grouped by search_query.

        Instead of taking top-N by confidence (which skews to one topic),
        take top 2-3 per search query group to give the planner a complete picture.
        """
        # Group by search_query
        groups: dict[str, list[Finding]] = {}
        for f in findings:
            key = f.search_query or "__no_query__"
            groups.setdefault(key, []).append(f)

        # Sort each group by confidence
        for key in groups:
            groups[key].sort(key=lambda f: f.confidence, reverse=True)

        # Take top 2-3 per group
        per_group = max(2, min(3, max_total // max(1, len(groups))))
        selected: list[Finding] = []
        for key in groups:
            selected.extend(groups[key][:per_group])

        # Pad to max_total from remaining if needed
        if len(selected) < max_total:
            selected_ids = {id(f) for f in selected}
            remaining = sorted(
                [f for f in findings if id(f) not in selected_ids],
                key=lambda f: f.confidence,
                reverse=True,
            )
            selected.extend(remaining[: max_total - len(selected)])

        return selected[:max_total]

    async def _plan_report_structure(
        self, goal: str, findings: list[Finding], topics_explored: list[str]
    ) -> list[PlannedSection]:
        """Have AI analyze findings and plan what sections the report needs."""
        # Stratified sampling: representative findings across all topics
        representative = self._get_representative_findings(findings, max_total=80)
        findings_summary = self._format_findings_block(representative, 12000)

        # Count finding types for context
        type_counts = {}
        for f in findings:
            t = f.finding_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        prompt = f"""Analyze these research findings and plan the optimal report structure.

RESEARCH QUESTION: {goal}

FINDING COUNTS BY TYPE: {json.dumps(type_counts)}

TOPICS EXPLORED: {", ".join(topics_explored[:10]) if topics_explored else "Various"}

SAMPLE FINDINGS:
{findings_summary}

Your task: Decide what sections this report needs based on the content. Choose from these section types:

- tldr: A 2-3 sentence bottom-line answer (always include this first)
- flash_numbers: Key metrics/statistics callouts if quantitative data exists
- stats_table: Tabular comparison if comparing multiple items
- comparison: Side-by-side analysis if comparing approaches/systems
- timeline: Chronological progression if temporal data exists
- narrative: Standard prose section for a specific theme
- analysis: Deep synthesis of patterns and insights
- gaps: Open questions and unknowns
- conclusions: Final takeaways and recommendations (always include near end)

Return a JSON array of sections in the order they should appear. Example format:
[
  {{"type": "tldr", "title": "TL;DR", "description": "Bottom-line answer to the research question", "priority": 10}},
  {{"type": "flash_numbers", "title": "Key Numbers", "description": "Critical metrics from the research", "priority": 9}},
  {{"type": "narrative", "title": "Current Landscape", "description": "Overview of the current state", "priority": 8}},
  {{"type": "comparison", "title": "Framework Comparison", "description": "Side-by-side analysis of major frameworks", "priority": 7}},
  {{"type": "narrative", "title": "Technical Deep Dive", "description": "Detailed technical analysis", "priority": 6}},
  {{"type": "analysis", "title": "Patterns & Insights", "description": "Cross-cutting analysis", "priority": 5}},
  {{"type": "gaps", "title": "Open Questions", "description": "Knowledge gaps and uncertainties", "priority": 4}},
  {{"type": "conclusions", "title": "Conclusions", "description": "Final recommendations", "priority": 3}}
]

Guidelines:
- ALWAYS start with tldr
- Include flash_numbers ONLY if significant quantitative data exists
- Include stats_table or comparison ONLY if comparing multiple distinct items
- Include timeline ONLY if clear temporal progression exists
- Use 3-5 narrative sections with SPECIFIC titles (not generic like "Overview")
- ALWAYS end with conclusions
- Total sections should be 6-10

Return ONLY the JSON array, no explanation."""

        response = await self._call_claude(
            prompt, "You are an expert research analyst planning report structure."
        )

        # Parse the response
        planned_sections = []
        try:
            # Find JSON array in response
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                sections_data = json.loads(match.group())
                for s in sections_data:
                    try:
                        section_type = SectionType(s.get("type", "narrative"))
                    except ValueError:
                        section_type = SectionType.NARRATIVE
                    planned_sections.append(
                        PlannedSection(
                            section_type=section_type,
                            title=s.get("title", "Section"),
                            description=s.get("description", ""),
                            priority=s.get("priority", 5),
                        )
                    )
        except (json.JSONDecodeError, KeyError):
            # Fallback to basic structure
            planned_sections = [
                PlannedSection(SectionType.TLDR, "TL;DR", "Bottom-line answer"),
                PlannedSection(SectionType.NARRATIVE, "Background", "Context and background"),
                PlannedSection(SectionType.NARRATIVE, "Key Findings", "Main discoveries"),
                PlannedSection(SectionType.ANALYSIS, "Analysis", "Synthesis and insights"),
                PlannedSection(SectionType.CONCLUSIONS, "Conclusions", "Recommendations"),
            ]

        return planned_sections

    def _format_kg_context(self, kg_exports: dict | None, section_type: SectionType) -> str:
        """Extract relevant KG data for a given section type."""
        if not kg_exports:
            return ""

        parts: list[str] = []

        # Key concepts for narrative/analysis/conclusions
        if section_type in (SectionType.NARRATIVE, SectionType.ANALYSIS, SectionType.CONCLUSIONS):
            key_concepts = kg_exports.get("key_concepts", [])
            if key_concepts:
                concepts_str = ", ".join(f"{c['name']} ({c['type']})" for c in key_concepts[:5])
                parts.append(f"Key concepts (by importance): {concepts_str}")

        # Contradictions for analysis/gaps/narrative
        if section_type in (SectionType.ANALYSIS, SectionType.GAPS, SectionType.NARRATIVE):
            contradictions = kg_exports.get("contradictions", [])
            if contradictions:
                contras = []
                for c in contradictions[:3]:
                    desc = c.get("description", c.get("recommendation", "Unknown"))
                    severity = c.get("severity", "unknown")
                    contras.append(f"- {desc} (severity: {severity})")
                parts.append("Contradictions detected:\n" + "\n".join(contras))

        # Knowledge gaps for gaps section
        if section_type == SectionType.GAPS:
            gaps = kg_exports.get("gaps", [])
            if gaps:
                gap_items = []
                for g in gaps[:5]:
                    rec = g.get("recommendation", g.get("gap_type", "Unknown"))
                    gap_items.append(f"- {rec}")
                parts.append("Knowledge gaps identified:\n" + "\n".join(gap_items))

        if not parts:
            return ""
        return "\nKNOWLEDGE GRAPH CONTEXT:\n" + "\n\n".join(parts) + "\n"

    async def _generate_dynamic_section(
        self,
        section: PlannedSection,
        goal: str,
        findings: list[Finding],
        kg_exports: dict | None = None,
    ) -> str:
        """Generate content for a planned section based on its type."""
        generators = {
            SectionType.TLDR: self._gen_tldr,
            SectionType.FLASH_NUMBERS: self._gen_flash_numbers,
            SectionType.STATS_TABLE: self._gen_stats_table,
            SectionType.COMPARISON: self._gen_comparison,
            SectionType.TIMELINE: self._gen_timeline,
            SectionType.GAPS: self._gen_gaps,
            SectionType.NARRATIVE: self._gen_narrative,
            SectionType.ANALYSIS: self._gen_analysis_section,
            SectionType.CONCLUSIONS: self._gen_conclusions_section,
        }

        generator = generators.get(section.section_type, self._gen_narrative)
        return await generator(section, goal, findings, kg_exports=kg_exports)

    async def _gen_tldr(
        self,
        _section: PlannedSection,
        goal: str,
        findings: list[Finding],
        kg_exports: dict | None = None,
    ) -> str:
        """Generate a 2-3 sentence TL;DR summary."""
        # Filter out any meta-questions or placeholder content from findings
        valid_findings = [
            f
            for f in findings
            if not any(
                phrase in f.content.lower()
                for phrase in [
                    "please provide",
                    "what information",
                    "could you clarify",
                    "what are you looking for",
                    "template or placeholder",
                ]
            )
        ]
        selected = self._select_findings_for_section(
            valid_findings,
            "TL;DR summary",
            "Bottom-line answer",
            section_type=SectionType.TLDR,
            max_findings=15,
        )
        findings_text = self._format_findings_block(selected)

        prompt = f"""Write a TL;DR (Too Long; Didn't Read) summary for this research.

RESEARCH QUESTION: {goal}

KEY FINDINGS (focus on these substantive findings):
{findings_text}

Write 2-3 sentences that directly answer the research question with the most important takeaways.
Be specific and definitive — use exact numbers, names, and details from findings.
Do NOT say the research is incomplete or a placeholder — summarize what was actually found.

Format as a blockquote (start each line with >).
Output ONLY the blockquote, nothing else."""

        return await self._call_claude(
            prompt,
            "You are writing a sourced, evidence-based research report. Distill findings into a precise summary.",
        )

    async def _gen_flash_numbers(
        self,
        _section: PlannedSection,
        goal: str,
        findings: list[Finding],
        kg_exports: dict | None = None,
    ) -> str:
        """Generate key metrics callouts."""
        # Find findings with numbers
        numeric_findings = [f for f in findings if any(c.isdigit() for c in f.content)][:20]

        if not numeric_findings:
            return ""

        findings_text = self._format_findings_block(numeric_findings)

        prompt = f"""Extract the most impactful numbers/statistics from these research findings.

RESEARCH QUESTION: {goal}

FINDINGS WITH DATA:
{findings_text}

Format each key metric as:
**[NUMBER/STAT]** — [Brief description of what it means] [N]

Where [N] is the source reference number from the finding.

Example:
**94.4%** — LLM agents vulnerable to prompt injection attacks [3]
**10-15 min** — Time to generate working CVE exploits with AI [7]

Select 3-6 of the most compelling, relevant statistics.
Output ONLY the formatted metrics, one per line. No introductory text."""

        return await self._call_claude(
            prompt,
            "You are writing a sourced, evidence-based research report. Highlight key statistics with their sources.",
        )

    async def _gen_stats_table(
        self,
        section: PlannedSection,
        goal: str,
        findings: list[Finding],
        kg_exports: dict | None = None,
    ) -> str:
        """Generate a markdown comparison table."""
        selected = self._select_findings_for_section(
            findings,
            section.title,
            section.description,
            section_type=SectionType.STATS_TABLE,
            max_findings=25,
        )
        findings_text = self._format_findings_block(selected)

        prompt = f"""Create a comparison table from these research findings.

RESEARCH QUESTION: {goal}
SECTION DESCRIPTION: {section.description}

FINDINGS:
{findings_text}

Create a markdown table comparing the key items/options/approaches found in the research.
Choose appropriate column headers based on what's being compared.
Include a "Source" column with [N] references where data comes from specific findings.

Output ONLY the markdown table. No introductory text."""

        return await self._call_claude(
            prompt,
            "You are writing a sourced, evidence-based research report. Create precise comparison tables.",
        )

    async def _gen_comparison(
        self,
        section: PlannedSection,
        goal: str,
        findings: list[Finding],
        kg_exports: dict | None = None,
    ) -> str:
        """Generate side-by-side comparison analysis."""
        selected = self._select_findings_for_section(
            findings,
            section.title,
            section.description,
            section_type=SectionType.COMPARISON,
            max_findings=20,
        )
        findings_text = self._format_findings_block(selected)
        kg_context = self._format_kg_context(kg_exports, SectionType.COMPARISON)

        prompt = f"""Write a side-by-side comparison analysis.

RESEARCH QUESTION: {goal}
SECTION TITLE: {section.title}
SECTION DESCRIPTION: {section.description}

AVAILABLE FINDINGS:
{findings_text}
{kg_context}
CITATION INSTRUCTIONS:
- Each finding includes a source like [Source: [N] domain].
- Cite claims inline using [N] notation, e.g., "accuracy improved 40% [3]."
- Aim for 3-8 citations per section. Do NOT list references at the end.

Write 3-4 paragraphs that:
1. Identify the key items/approaches being compared
2. Analyze strengths and weaknesses of each
3. Highlight key differentiators
4. Provide guidance on when to use each

Name specific sources, organizations, or tools when available.
Use exact numbers, dates, and details from findings.
Use subheadings if helpful.
Output ONLY the comparison content."""

        return await self._call_claude(
            prompt,
            "You are writing a sourced, evidence-based research report. Every major claim must cite its source using [N]. When findings disagree, present both sides.",
        )

    async def _gen_timeline(
        self,
        section: PlannedSection,
        goal: str,
        findings: list[Finding],
        kg_exports: dict | None = None,
    ) -> str:
        """Generate chronological timeline view."""
        # Find findings with dates/years
        # [HARDENED] CONF-003: Generate year range dynamically
        current_year = datetime.now().year
        year_keywords = [str(y) for y in range(current_year - 5, current_year + 1)]
        temporal_keywords = [
            *year_keywords,
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "released",
            "launched",
            "announced",
            "introduced",
        ]

        temporal_findings = [
            f for f in findings if any(kw in f.content.lower() for kw in temporal_keywords)
        ][:25]

        if not temporal_findings:
            temporal_findings = findings[:20]

        findings_text = self._format_findings_block(temporal_findings)

        prompt = f"""Create a chronological timeline from these research findings.

RESEARCH QUESTION: {goal}
SECTION TITLE: {section.title}

FINDINGS:
{findings_text}

CITATION INSTRUCTIONS:
- Each finding includes a source like [Source: [N] domain].
- Cite claims inline using [N] notation, e.g., "Released in March 2024 [3]."

Format as a timeline with clear dates/periods:

**[Date/Period]**: [Event/Development] [N]
- Key details

Order from earliest to most recent.
Output ONLY the timeline content."""

        return await self._call_claude(
            prompt,
            "You are writing a sourced, evidence-based research report. Every major claim must cite its source using [N].",
        )

    async def _gen_gaps(
        self,
        _section: PlannedSection,
        goal: str,
        findings: list[Finding],
        kg_exports: dict | None = None,
    ) -> str:
        """Generate open questions and knowledge gaps section."""
        selected = self._select_findings_for_section(
            findings,
            "Open questions gaps unknowns",
            "Knowledge gaps and uncertainties",
            section_type=SectionType.GAPS,
            max_findings=20,
        )
        findings_text = self._format_findings_block(selected)
        kg_context = self._format_kg_context(kg_exports, SectionType.GAPS)

        prompt = f"""Identify knowledge gaps and open questions from this research.

RESEARCH QUESTION: {goal}

FINDINGS:
{findings_text}
{kg_context}
CITATION INSTRUCTIONS:
- Each finding includes a source like [Source: [N] domain].
- When referencing a specific claim or contradiction, cite it inline using [N].

Write 2-3 paragraphs covering:
1. What important questions remain unanswered
2. Areas where more research is needed
3. Any contradictions or debates that aren't resolved
4. Limitations of current knowledge

If findings are marked [VERIFIED], note where confidence is high.
If [UNVERIFIED] or [FLAGGED], note these as areas needing further investigation.
Be specific about what we don't know yet.
Output ONLY the gaps content."""

        return await self._call_claude(
            prompt,
            "You are writing a sourced, evidence-based research report. Every major claim must cite its source using [N]. When findings disagree, present both sides.",
        )

    async def _gen_narrative(
        self,
        section: PlannedSection,
        goal: str,
        findings: list[Finding],
        kg_exports: dict | None = None,
    ) -> str:
        """Generate standard narrative prose section."""
        selected = self._select_findings_for_section(
            findings,
            section.title,
            section.description,
            section_type=SectionType.NARRATIVE,
            max_findings=20,
        )
        findings_text = self._format_findings_block(selected, 12000)
        kg_context = self._format_kg_context(kg_exports, SectionType.NARRATIVE)

        prompt = f"""Write a section of a deep research report.

RESEARCH QUESTION: {goal}
SECTION TITLE: {section.title}
SECTION DESCRIPTION: {section.description}

AVAILABLE FINDINGS:
{findings_text}
{kg_context}
CITATION INSTRUCTIONS:
- Each finding includes a source like [Source: [N] domain].
- Cite claims inline using [N] notation, e.g., "accuracy improved 40% [3]."
- Aim for 3-8 citations per section. Do NOT list references at the end.

Write 4-6 paragraphs that:
1. Open with the key point for this theme
2. Develop with supporting details and evidence
3. Explain significance and implications
4. Connect to the broader research question

Guidelines:
- Write flowing prose, not bullet points
- Name specific sources, organizations, or tools when available
- Use exact numbers, dates, and details from findings
- If marked [VERIFIED], state with confidence. If [UNVERIFIED], hedge appropriately.

Output ONLY the section content, no headers."""

        return await self._call_claude(
            prompt,
            "You are writing a sourced, evidence-based research report. Every major claim must cite its source using [N]. When findings disagree, present both sides.",
        )

    async def _gen_analysis_section(
        self,
        section: PlannedSection,
        goal: str,
        findings: list[Finding],
        kg_exports: dict | None = None,
    ) -> str:
        """Generate analysis and insights section."""
        selected = self._select_findings_for_section(
            findings,
            section.title,
            section.description,
            section_type=SectionType.ANALYSIS,
            max_findings=20,
        )
        findings_text = self._format_findings_block(selected)
        kg_context = self._format_kg_context(kg_exports, SectionType.ANALYSIS)

        prompt = f"""Write an analysis section synthesizing research findings.

RESEARCH QUESTION: {goal}
SECTION TITLE: {section.title}
SECTION DESCRIPTION: {section.description}

AVAILABLE FINDINGS:
{findings_text}
{kg_context}
CITATION INSTRUCTIONS:
- Each finding includes a source like [Source: [N] domain].
- Cite claims inline using [N] notation, e.g., "accuracy improved 40% [3]."
- Aim for 3-8 citations per section. Do NOT list references at the end.

Write 3-4 paragraphs that:
1. Synthesize the most important insights
2. Identify patterns, trends, and connections
3. Address contradictions or debates
4. Draw conclusions the reader might not see

Name specific sources, organizations, or tools when available.
If marked [VERIFIED], state with confidence. If [UNVERIFIED], hedge appropriately.
Output ONLY the analysis content."""

        return await self._call_claude(
            prompt,
            "You are writing a sourced, evidence-based research report. Every major claim must cite its source using [N]. When findings disagree, present both sides.",
        )

    async def _gen_conclusions_section(
        self,
        _section: PlannedSection,
        goal: str,
        findings: list[Finding],
        kg_exports: dict | None = None,
    ) -> str:
        """Generate conclusions section."""
        selected = self._select_findings_for_section(
            findings,
            "Conclusions recommendations takeaways",
            "Final answer and recommendations",
            section_type=SectionType.CONCLUSIONS,
            max_findings=20,
        )
        findings_text = self._format_findings_block(selected)
        kg_context = self._format_kg_context(kg_exports, SectionType.CONCLUSIONS)

        prompt = f"""Write the conclusions section of a research report.

RESEARCH QUESTION: {goal}

KEY FINDINGS:
{findings_text}
{kg_context}
CITATION INSTRUCTIONS:
- Each finding includes a source like [Source: [N] domain].
- Cite key claims inline using [N] notation.

Write 2-3 paragraphs that:
1. Directly answer the research question
2. Summarize the most important takeaways with specific evidence
3. Provide actionable recommendations
4. Suggest areas for further investigation

Be definitive where evidence is [VERIFIED], hedged where [UNVERIFIED].
Output ONLY the conclusions content."""

        return await self._call_claude(
            prompt,
            "You are writing a sourced, evidence-based research report. Every major claim must cite its source using [N].",
        )

    async def generate_dynamic_report(
        self,
        session: ResearchSession,
        findings: list[Finding],
        topics_explored: list[str],
        topics_remaining: list[str],  # noqa: ARG002 - kept for API compatibility
        kg_exports: dict = None,
        verification_metrics: dict = None,
        progress_callback: Callable[[str, int], Awaitable[None]] | None = None,
    ) -> str:
        """Generate a comprehensive report with AI-driven dynamic structure.

        Args:
            session: The research session
            findings: All findings from the research
            topics_explored: Topics that were researched
            topics_remaining: Topics that could be researched with more time (unused in dynamic mode)
            kg_exports: Optional knowledge graph exports
            verification_metrics: Optional verification metrics from the pipeline

        Returns:
            Complete markdown report
        """
        del topics_remaining  # Unused in dynamic mode, but kept for API compatibility
        sources = self._extract_sources(findings)

        # Build source index for inline citations [N]
        self._source_index: dict[str, int] = {}
        for i, src in enumerate(sources, 1):
            url = src["url"]
            self._source_index[url] = i

        # Phase 1: Plan the report structure
        await _emit_progress(progress_callback, "Planning report structure...", 5)
        print("[REPORT] Planning report structure...")
        planned_sections = await self._plan_report_structure(
            session.goal, findings, topics_explored
        )
        await _emit_progress(progress_callback, "Planned report structure", 10)
        print(
            f"[REPORT] Planned {len(planned_sections)} sections: {[s.title for s in planned_sections]}"
        )

        # Phase 2: Generate each section (with pacing to avoid rate limits)
        for i, section in enumerate(planned_sections):
            progress = 10 + int(((i + 1) / max(1, len(planned_sections))) * 80)
            await _emit_progress(
                progress_callback,
                f"Generating section {i + 1}/{len(planned_sections)}: {section.title}",
                progress,
            )
            print(
                f"[REPORT] Generating section {i + 1}/{len(planned_sections)}: {section.title}..."
            )
            section.content = await self._generate_dynamic_section(
                section, session.goal, findings, kg_exports=kg_exports
            )
            # Brief pause between sections to reduce rate limit pressure
            if i < len(planned_sections) - 1:
                await asyncio.sleep(2)

        # Compile the report
        await _emit_progress(progress_callback, "Compiling report...", 95)
        return self._compile_dynamic_report(
            session=session,
            planned_sections=planned_sections,
            sources=sources,
            findings=findings,
            topics_explored=topics_explored,
            kg_exports=kg_exports,
            verification_metrics=verification_metrics,
        )

    def _compile_dynamic_report(
        self,
        session: ResearchSession,
        planned_sections: list[PlannedSection],
        sources: list[dict],
        findings: list[Finding],
        topics_explored: list[str],
        kg_exports: dict = None,
        verification_metrics: dict = None,
    ) -> str:
        """Compile dynamically planned sections into final report."""
        # Build table of contents
        toc_items = []
        for i, section in enumerate(planned_sections, 1):
            toc_items.append(f"{i}. {section.title}")
        toc_items.append(f"{len(planned_sections) + 1}. References")

        toc = "\n".join(
            [
                f"- [{item}](#{item.lower().replace(' ', '-').replace('.', '')})"
                for item in toc_items
            ]
        )

        # Build main content with type-specific formatting
        main_content = ""
        for i, section in enumerate(planned_sections, 1):
            main_content += f"\n## {i}. {section.title}\n\n"

            # Type-specific formatting
            if section.section_type == SectionType.TLDR:
                # Ensure blockquote formatting
                content = section.content.strip()
                if not content.startswith(">"):
                    content = "> " + content.replace("\n", "\n> ")
                main_content += f"{content}\n"
            elif section.section_type == SectionType.FLASH_NUMBERS:
                main_content += f"{section.content}\n"
            elif section.section_type in (SectionType.STATS_TABLE, SectionType.COMPARISON):
                main_content += f"{section.content}\n"
            else:
                main_content += f"{section.content}\n"

            main_content += "\n---\n"

        # Build references
        references = []
        for i, source in enumerate(sources, 1):
            title = source.get("title", source["domain"])
            references.append(f"[{i}] {title}. *{source['domain']}*. {source['url']}")
        references_text = "\n\n".join(references)
        retrieval_date = datetime.now().strftime("%B %d, %Y")

        # Stats
        topics_count = len(topics_explored) if topics_explored else len(sources)
        stats = f"""**Research Statistics:**
- Total Findings: {len(findings)}
- Sources Analyzed: {len(sources)}
- Topics Explored: {topics_count}
- Research Duration: {session.started_at.strftime("%Y-%m-%d %H:%M")} to {session.ended_at.strftime("%Y-%m-%d %H:%M") if session.ended_at else "In Progress"}"""

        # Compile full report
        report = f"""# {session.goal}

*Deep Research Report*

---

**Generated:** {datetime.now().strftime("%B %d, %Y at %H:%M")}
**Session ID:** {session.id}

---

## Table of Contents

{toc}

---
{main_content}
## {len(planned_sections) + 1}. References

*All sources accessed on {retrieval_date}.*

{references_text}

---

## Appendix: Research Methodology

This report was generated using a hierarchical multi-agent research system:

1. **Research Planning**: An AI manager agent analyzed the research question and developed a systematic research strategy.
2. **Information Gathering**: AI intern agents conducted {len(sources)} web searches, analyzing sources for relevance and credibility.
3. **Finding Extraction**: {len(findings)} discrete findings were extracted and categorized by type.
4. **Report Structure Planning**: AI analyzed findings to determine optimal report sections (TL;DR, statistics, comparisons, narratives, etc.).
5. **Narrative Synthesis**: Each section was generated according to its type with specialized formatting.
6. **Fact Verification**: Findings were verified using Chain-of-Verification (CoVe) to reduce hallucinations.

{stats}

**Topics Researched:**
{chr(10).join(["- " + t for t in topics_explored[:15]]) if topics_explored else "- " + session.goal}

{self._format_verification_section(verification_metrics, findings) if verification_metrics else ""}

{self._format_kg_section(kg_exports) if kg_exports else ""}

---

*Report generated by Claude Deep Researcher*
"""
        return report

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
        toc_items.extend(
            [
                f"{len(main_sections) + 3}. Analysis and Key Insights",
                f"{len(main_sections) + 4}. Conclusions and Recommendations",
                f"{len(main_sections) + 5}. References",
            ]
        )

        toc = "\n".join(
            [
                f"- [{item}](#{item.lower().replace(' ', '-').replace('.', '')})"
                for item in toc_items
            ]
        )

        # Build main sections
        main_content = ""
        for i, section in enumerate(main_sections, 3):
            main_content += f"\n## {i}. {section.title}\n\n{section.content}\n"

        # Build references in cleaner format (no redundant dates)
        references = []
        for i, source in enumerate(sources, 1):
            title = source.get("title", source["domain"])
            references.append(f"[{i}] {title}. *{source['domain']}*. {source['url']}")
        references_text = "\n\n".join(references)

        # Add retrieval date once at the top
        retrieval_date = datetime.now().strftime("%B %d, %Y")

        # Stats - use sources count if topics_explored is empty
        topics_count = len(topics_explored) if topics_explored else len(sources)
        stats = f"""**Research Statistics:**
- Total Findings: {len(findings)}
- Sources Analyzed: {len(sources)}
- Topics Explored: {topics_count}
- Research Duration: {session.started_at.strftime("%Y-%m-%d %H:%M")} to {session.ended_at.strftime("%Y-%m-%d %H:%M") if session.ended_at else "In Progress"}"""

        # Compile full report
        report = f"""# {session.goal}

*Deep Research Report*

---

**Generated:** {datetime.now().strftime("%B %d, %Y at %H:%M")}
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
{chr(10).join(["- " + t for t in topics_explored[:15]]) if topics_explored else "- " + session.goal}

{self._format_kg_section(kg_exports) if kg_exports else ""}

---

*Report generated by Claude Deep Researcher*
"""
        return report

    def _format_verification_section(
        self, verification_metrics: dict, findings: list[Finding]
    ) -> str:
        """Format verification section for the report."""
        if not verification_metrics:
            return ""

        sections = ["---", "", "## Appendix: Fact Verification Analysis"]

        # Overall verification stats
        status = verification_metrics.get("status", {})
        verified = status.get("verified", 0)
        flagged = status.get("flagged", 0)
        rejected = status.get("rejected", 0)
        total = verification_metrics.get("total_verifications", 0)

        if total > 0:
            sections.append(f"""
**Verification Summary:**
- Total Findings Verified: {total}
- High Confidence (Verified, >85%): {verified} ({verified / total * 100:.1f}%)
- Medium Confidence (Flagged, 50-85%): {flagged} ({flagged / total * 100:.1f}%)
- Low Confidence (Rejected, <50%): {rejected} ({rejected / total * 100:.1f}%)
""")

        # Confidence calibration
        confidence = verification_metrics.get("confidence", {})
        avg_delta = confidence.get("avg_delta", 0)
        if avg_delta != 0:
            direction = "increased" if avg_delta > 0 else "decreased"
            sections.append(
                f"**Confidence Calibration:** Average confidence {direction} by {abs(avg_delta) * 100:.1f}% after verification.\n"
            )

        # Contradictions
        contradictions = verification_metrics.get("contradictions", {})
        total_contradictions = contradictions.get("total", 0)
        if total_contradictions > 0:
            sections.append(f"""
**Contradictions Detected:** {total_contradictions}
These findings had conflicting information that may require further investigation.
""")

        # KG integration stats
        kg_int = verification_metrics.get("kg_integration", {})
        kg_matches = kg_int.get("matches", 0)
        if kg_matches > 0:
            sections.append(
                f"**Knowledge Graph Corroboration:** {kg_matches} findings were corroborated by the knowledge graph.\n"
            )

        # Latency stats
        latency = verification_metrics.get("latency", {})
        streaming_avg = latency.get("streaming_avg_ms", 0)
        batch_avg = latency.get("batch_avg_ms", 0)
        if streaming_avg > 0 or batch_avg > 0:
            sections.append(f"""
**Verification Performance:**
- Streaming verification: {streaming_avg:.0f}ms avg (target <500ms)
- Batch verification: {batch_avg:.0f}ms avg (target <2000ms)
""")

        # Findings by verification status (with badges)
        verified_findings = [f for f in findings if f.verification_status == "verified"][:5]
        flagged_findings = [f for f in findings if f.verification_status == "flagged"][:3]
        rejected_findings = [f for f in findings if f.verification_status == "rejected"][:3]

        if verified_findings:
            sections.append("\n**Sample Verified Findings (High Confidence):**")
            for f in verified_findings:
                badge = f"[{f.confidence * 100:.0f}%]"
                sections.append(f"- {badge} {f.content[:150]}...")

        if flagged_findings:
            sections.append("\n**Flagged Findings (Needs Review):**")
            for f in flagged_findings:
                badge = f"[{f.confidence * 100:.0f}%]"
                sections.append(f"- {badge} {f.content[:150]}...")

        if rejected_findings:
            sections.append("\n**Rejected Findings (Low Confidence):**")
            for f in rejected_findings:
                badge = f"[{f.confidence * 100:.0f}%]"
                sections.append(f"- {badge} {f.content[:150]}...")

        sections.append("")
        return "\n".join(sections)

    def _format_kg_section(self, kg_exports: dict) -> str:
        """Format knowledge graph section for the report."""
        if not kg_exports:
            return ""

        sections = ["---", "", "## Appendix: Knowledge Graph Analysis"]

        # Stats
        stats = kg_exports.get("stats", {})
        if stats:
            sections.append(f"""
**Knowledge Graph Statistics:**
- Entities extracted: {stats.get("num_entities", 0)}
- Relations identified: {stats.get("num_relations", 0)}
- Connected components: {stats.get("num_components", 0)}
- Graph density: {stats.get("density", 0):.3f}
""")

        # Key concepts
        key_concepts = kg_exports.get("key_concepts", [])
        if key_concepts:
            sections.append("**Key Concepts by Importance (PageRank):**")
            for c in key_concepts[:5]:
                sections.append(f"- {c['name']} ({c['type']}) - importance: {c['importance']}")
            sections.append("")

        # Gaps identified
        gaps = kg_exports.get("gaps", [])
        if gaps:
            sections.append(f"**Knowledge Gaps Identified ({len(gaps)}):**")
            for g in gaps[:5]:
                sections.append(f"- {g.get('recommendation', g.get('gap_type', 'Unknown'))}")
            sections.append("")

        # Contradictions
        contradictions = kg_exports.get("contradictions", [])
        if contradictions:
            sections.append(f"**Contradictions Detected ({len(contradictions)}):**")
            for c in contradictions[:3]:
                sections.append(f"- {c.get('description', c.get('recommendation', 'Unknown'))}")
            sections.append("")

        # Link to HTML visualization (Mermaid diagram removed - HTML visualization is preferred)
        html_viz = kg_exports.get("html_visualization")
        if html_viz:
            sections.append(f"*Interactive visualization available at: {html_viz}*")
            sections.append("")

        return "\n".join(sections)


async def _emit_progress(
    callback: Callable[[str, int], Awaitable[None]] | None,
    message: str,
    progress: int,
) -> None:
    if callback is None:
        return
    try:
        await callback(message, progress)
    except Exception:
        return
