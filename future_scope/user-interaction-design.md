# User Interaction Design

*Interactive clarification and async guidance for claude-researcher*

---

## Overview

Three features to make research more interactive without blocking autonomy:

1. **Pre-research clarification** - Refine scope before starting (like Gemini/ChatGPT Deep Research)
2. **Async mid-research questions** - Ask user, wait 60s, continue if no reply
3. **User message queue** - Inject guidance that gets picked up next iteration

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      UserInteraction                         │
├─────────────────────────────────────────────────────────────┤
│  user_queue: asyncio.Queue       (user → system)            │
│  pending_question: Optional[Question]                        │
│  response_timeout: int = 60                                  │
│  _response_event: asyncio.Event                              │
├─────────────────────────────────────────────────────────────┤
│  async ask_with_timeout(question, timeout) → Optional[str]  │
│  inject_message(text) → None                                 │
│  get_pending_messages() → list[str]                          │
│  respond(text) → None                                        │
│  async pre_research_clarify(goal) → ClarifiedGoal           │
└─────────────────────────────────────────────────────────────┘
```

**Data flow:**
```
User types → input_listener() → is there pending_question?
                                    ├─ Yes → respond() → unblocks ask_with_timeout()
                                    └─ No  → inject_message() → queued for next iteration
```

---

## Feature 1: Pre-Research Clarification

### Flow

```
User: "What are the latest advances in fusion energy?"
     ↓
Director.clarify_research_goal()
     ↓
LLM generates 2-4 clarifying questions
     ↓
Display to user with defaults
     ↓
User answers or presses Enter to skip
     ↓
Build enriched goal context
     ↓
Start research with refined scope
```

### Implementation

```python
@dataclass
class ClarificationQuestion:
    id: int
    question: str
    options: list[str]  # Empty if free-form
    default: Optional[str] = None

@dataclass
class ClarifiedGoal:
    original: str
    clarifications: dict[int, str]  # question_id → answer
    enriched_context: str


class Director:
    async def clarify_research_goal(self, goal: str) -> ClarifiedGoal:
        """Generate and ask clarifying questions before research."""

        prompt = f"""Analyze this research goal and generate 2-4 clarifying questions
that would help focus the research.

Goal: {goal}

Consider asking about:
- Scope (broad overview vs deep technical dive)
- Time period (recent developments vs historical)
- Perspective (academic, practical, business)
- Specific aspects to prioritize or skip
- Output format preferences (comprehensive vs concise)
- Any known constraints or context

Return as JSON:
{{
    "interpretation": "your understanding of what they want",
    "questions": [
        {{
            "id": 1,
            "question": "What depth of coverage do you need?",
            "options": ["High-level overview", "Technical deep-dive", "Both"],
            "default": "Both"
        }}
    ]
}}"""

        response = await self.call_claude(prompt)
        questions = self._parse_clarification_questions(response)

        # Ask user (with skip option)
        responses = await self.interaction.ask_clarifications(
            questions,
            skip_allowed=True,
            timeout=120  # 2 min to answer clarifications
        )

        # Build enriched context
        enriched = self._build_enriched_context(goal, questions, responses)

        return ClarifiedGoal(
            original=goal,
            clarifications=responses,
            enriched_context=enriched,
        )

    def _build_enriched_context(
        self,
        goal: str,
        questions: list[ClarificationQuestion],
        responses: dict[int, str]
    ) -> str:
        """Build enriched goal context from clarifications."""

        parts = [f"Research Goal: {goal}", "", "User Preferences:"]

        for q in questions:
            answer = responses.get(q.id, q.default or "not specified")
            parts.append(f"- {q.question} → {answer}")

        return "\n".join(parts)
```

### UX

```
┌─────────────────────────────────────────────────────────────┐
│ Before I start, a few quick questions:                      │
│                                                             │
│ 1. Focus area?                                              │
│    [1] All aspects (default)                                │
│    [2] Tokamaks only                                        │
│    [3] Laser fusion                                         │
│    [4] Startup landscape                                    │
│                                                             │
│ 2. Time period?                                             │
│    [1] 2024-2025 only (default)                             │
│    [2] Last 5 years                                         │
│    [3] Full historical context                              │
│                                                             │
│ 3. Depth?                                                   │
│    [1] Overview for general audience                        │
│    [2] Technical deep-dive (default)                        │
│                                                             │
│ Enter choices (e.g., "2,1,2") or press Enter for defaults:  │
└─────────────────────────────────────────────────────────────┘
> 2,1,2

Starting research on tokamaks (2024-2025, technical)...
```

---

## Feature 2: Async Mid-Research Questions

### Flow

```
Manager hits decision point needing user input
     ↓
ask_with_timeout(question, 60s)
     ↓
Display question to user
     ↓
Start 60s timer
     ↓
User responds? ──Yes──→ Return response, incorporate into decision
     │
     └──Timeout──→ Return None, continue with best guess
                   Log: "Proceeding without user input on: {question}"
```

### Implementation

```python
@dataclass
class PendingQuestion:
    text: str
    context: str
    asked_at: datetime
    options: list[str] = field(default_factory=list)


class UserInteraction:
    def __init__(self, response_timeout: int = 60):
        self.response_timeout = response_timeout
        self.user_queue: asyncio.Queue = asyncio.Queue()
        self.pending_question: Optional[PendingQuestion] = None
        self._response_event = asyncio.Event()
        self._last_response: Optional[str] = None
        self.console = Console()

    async def ask_with_timeout(
        self,
        question: str,
        context: str = "",
        options: list[str] = None,
    ) -> Optional[str]:
        """Ask user a question, wait up to timeout, continue if no reply.

        Args:
            question: The question to ask
            context: Additional context for the question
            options: Optional list of suggested answers

        Returns:
            User's response, or None if timeout
        """
        self.pending_question = PendingQuestion(
            text=question,
            context=context,
            asked_at=datetime.now(),
            options=options or [],
        )
        self._response_event.clear()
        self._last_response = None

        # Display to user
        self._display_question()

        try:
            await asyncio.wait_for(
                self._response_event.wait(),
                timeout=self.response_timeout
            )
            response = self._last_response
            self._log_response(response)
            return response

        except asyncio.TimeoutError:
            self.console.print(
                f"[dim]No response after {self.response_timeout}s, continuing...[/dim]"
            )
            return None

        finally:
            self.pending_question = None

    def respond(self, response: str) -> bool:
        """User responds to pending question.

        Returns:
            True if there was a pending question, False otherwise
        """
        if self.pending_question:
            self._last_response = response.strip()
            self._response_event.set()
            return True
        return False

    def _display_question(self) -> None:
        """Display pending question to user."""
        q = self.pending_question
        if not q:
            return

        self.console.print()
        self.console.print("─" * 60, style="yellow")
        self.console.print(
            f"[bold yellow]QUICK QUESTION[/bold yellow] "
            f"[dim](respond in {self.response_timeout}s or I'll continue)[/dim]"
        )
        self.console.print()
        self.console.print(f"  {q.text}")

        if q.context:
            self.console.print(f"  [dim]Context: {q.context}[/dim]")

        if q.options:
            self.console.print()
            for i, opt in enumerate(q.options, 1):
                self.console.print(f"  [{i}] {opt}")

        self.console.print("─" * 60, style="yellow")
        self.console.print("[dim]> [/dim]", end="")
```

### Manager Integration

```python
class ManagerAgent:
    def __init__(self, ..., interaction: Optional[UserInteraction] = None):
        self.interaction = interaction or UserInteraction()

    async def _maybe_ask_user(self, question: str, context: str = "") -> Optional[str]:
        """Ask user if interaction is available, otherwise return None."""
        if self.interaction:
            return await self.interaction.ask_with_timeout(question, context)
        return None

    async def act(self, thought: str, context: dict[str, Any]) -> dict[str, Any]:
        # ... existing code ...

        # Example: Ask about contradictions
        if self._found_contradiction(thought):
            response = await self._maybe_ask_user(
                "I found conflicting information about X. Which source should I trust more?",
                context="Source A says X, Source B says Y",
                options=["Trust Source A", "Trust Source B", "Research both further"]
            )

            if response:
                # Incorporate user preference
                self._update_source_preference(response)
            else:
                # No response, use credibility scores
                self._use_credibility_heuristic()
```

### When to Ask

Manager should ask about:
- **Contradictions** - Which source to trust
- **Scope decisions** - Go deeper on topic A or pivot to topic B
- **Ambiguity** - Multiple interpretations of a finding
- **Resource allocation** - Time running low, what to prioritize

Manager should NOT ask about:
- Routine decisions (next search query)
- Things it can resolve with existing data
- Low-stakes choices

---

## Feature 3: User Message Queue

### Flow

```
User types message while research is running
     ↓
No pending question? → inject_message()
     ↓
Message queued
     ↓
Next Manager.think() iteration
     ↓
get_pending_messages() → Returns ["also look into X", "focus on Y"]
     ↓
Messages incorporated into prompt context
     ↓
Manager adjusts research direction
```

### Implementation

```python
@dataclass
class UserMessage:
    content: str
    injected_at: datetime


class UserInteraction:
    # ... previous code ...

    def inject_message(self, text: str) -> None:
        """User injects a message to be seen next iteration.

        Use this to provide guidance without responding to a specific question.
        """
        msg = UserMessage(
            content=text.strip(),
            injected_at=datetime.now(),
        )
        self.user_queue.put_nowait(msg)
        self.console.print(f"[green]Queued:[/green] {text[:50]}...")

    def get_pending_messages(self) -> list[UserMessage]:
        """Get all pending user messages (non-blocking).

        Called by Manager at start of each iteration.
        """
        messages = []
        while True:
            try:
                msg = self.user_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    def has_pending_messages(self) -> bool:
        """Check if there are pending messages without consuming them."""
        return not self.user_queue.empty()
```

### Manager Integration

```python
class ManagerAgent:
    async def think(self, context: dict[str, Any]) -> str:
        """Reason about research progress and next steps."""

        # Check for user injected messages
        user_messages = self.interaction.get_pending_messages()

        user_guidance = ""
        if user_messages:
            self._log(f"[USER INPUT] Received {len(user_messages)} message(s)", style="green")

            guidance_parts = ["USER GUIDANCE (just received):"]
            for msg in user_messages:
                guidance_parts.append(f"- {msg.content}")
                self._log(f"  → {msg.content}", style="green")

            guidance_parts.append("")
            guidance_parts.append("Consider this input when deciding next steps.")
            user_guidance = "\n".join(guidance_parts)

        prompt = f"""Research Goal: {self.research_goal}

{user_guidance}

Current Status:
- Time elapsed: {time_elapsed:.1f} minutes
- Findings: {len(self.all_findings)}
...
"""

        return await self.call_claude(prompt, use_thinking=True)
```

---

## CLI Input Listener

```python
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor


class InputListener:
    """Background listener for user input during research."""

    def __init__(self, interaction: UserInteraction):
        self.interaction = interaction
        self.running = False
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def start(self) -> None:
        """Start listening for input in background."""
        self.running = True
        asyncio.create_task(self._listen_loop())

    def stop(self) -> None:
        """Stop listening."""
        self.running = False

    async def _listen_loop(self) -> None:
        """Main input loop."""
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                # Non-blocking read from stdin
                line = await loop.run_in_executor(
                    self._executor,
                    self._read_line
                )

                if line is None:
                    await asyncio.sleep(0.1)
                    continue

                line = line.strip()
                if not line:
                    continue

                # Route input
                if self.interaction.pending_question:
                    # Response to question
                    self.interaction.respond(line)
                else:
                    # Inject as guidance
                    self.interaction.inject_message(line)

            except Exception as e:
                # Don't crash on input errors
                await asyncio.sleep(0.1)

    def _read_line(self) -> Optional[str]:
        """Read line with timeout (runs in thread)."""
        import select

        # Check if input available (Unix only)
        if select.select([sys.stdin], [], [], 0.1)[0]:
            return sys.stdin.readline()
        return None
```

### Main Integration

```python
# In main.py

async def run():
    global _harness

    async with ResearchHarness(db_path) as harness:
        _harness = harness

        # Create interaction handler
        interaction = UserInteraction(response_timeout=60)
        harness.director.interaction = interaction
        harness.director.manager.interaction = interaction

        # Start input listener
        listener = InputListener(interaction)
        await listener.start()

        try:
            # Pre-research clarification
            clarified = await harness.director.clarify_research_goal(goal)

            # Run research with clarified goal
            report = await harness.research(
                clarified.enriched_context,
                time_limit
            )
            return report

        finally:
            listener.stop()
```

---

## Complete UX Example

```
$ researcher "What are the latest advances in fusion energy?"

┌─────────────────────────────────────────────────────────────┐
│ Before I start, a few quick questions:                      │
│                                                             │
│ 1. Focus area? [all/tokamaks/laser/startups] (default: all) │
│ 2. Time period? [2024-2025/last 5 years/historical]         │
│ 3. Depth? [overview/technical deep-dive]                    │
│                                                             │
│ Press Enter to skip and use defaults, or answer:            │
└─────────────────────────────────────────────────────────────┘
> tokamaks, 2024-2025, technical

═══════════════════════════════════════════════════════════════
 Starting research: Tokamak fusion advances (2024-2025)
 Time limit: 60 minutes
═══════════════════════════════════════════════════════════════

[PARALLEL RESEARCH] Running 3 topics in parallel
  • Recent tokamak experimental results 2024-2025
  • ITER construction and timeline updates
  • Private tokamak startups progress

[Iteration 2 | Findings: 23]

────────────────────────────────────────────────────────────────
QUICK QUESTION (respond in 60s or I'll continue)

  I found conflicting reports on ITER's first plasma date.
  Official ITER says 2035, but some analysts say 2037+.
  Which perspective should I prioritize?

  [1] Official ITER timeline
  [2] Independent analyst estimates
  [3] Cover both perspectives equally

────────────────────────────────────────────────────────────────
> 3

[Noted: Will cover both perspectives]

[Iteration 3 | Findings: 31]

# User types unprompted:
> also look into Commonwealth Fusion's SPARC project

[green]Queued: also look into Commonwealth Fusion's SPARC...[/green]

[Iteration 4 | Findings: 38]
[USER INPUT] Received 1 message(s)
  → also look into Commonwealth Fusion's SPARC project

[DIRECTIVE] SEARCH: Commonwealth Fusion SPARC tokamak 2024 progress
...

═══════════════════════════════════════════════════════════════
 Research complete: 67 findings in 23 minutes
 Report saved to: output/tokamak-fusion_a3f2b1c/
═══════════════════════════════════════════════════════════════
```

---

## Configuration

```python
@dataclass
class InteractionConfig:
    # Pre-research
    enable_clarification: bool = True
    max_clarification_questions: int = 4
    clarification_timeout: int = 120  # 2 min

    # Mid-research
    enable_async_questions: bool = True
    question_timeout: int = 60  # 1 min
    max_questions_per_session: int = 5  # Don't annoy user

    # Message queue
    enable_message_queue: bool = True

    # Skip all interaction (fully autonomous)
    autonomous_mode: bool = False
```

### CLI flags

```
researcher "goal" --no-clarify        # Skip pre-research questions
researcher "goal" --autonomous        # No interaction at all
researcher "goal" --timeout 30        # 30s for mid-research questions
```

---

## Implementation Priority

1. **UserInteraction class** - Core queue and event handling
2. **Pre-research clarification** - Highest UX impact
3. **Message queue** - Simple, high value
4. **Async questions** - More complex, lower priority
5. **InputListener** - Platform-specific challenges

---

## Open Questions

1. **Windows support** - `select.select()` on stdin doesn't work on Windows. Need `msvcrt.kbhit()` or similar.

2. **Web UI** - If we add a web interface, this becomes WebSocket-based instead of stdin.

3. **Question frequency** - How often should Manager ask? Too many = annoying. Too few = misses opportunities.

4. **Question quality** - Need good prompts to generate actually useful questions, not trivial ones.
