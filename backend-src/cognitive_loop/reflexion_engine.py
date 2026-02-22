"""
Reflexion Engine - Self-Critique and Learning

Implements the Reflexion pattern for:
- Self-critique of decisions and actions
- Updating causal models based on outcomes
- Recording learnings to AI Memory (via SolutionTracker)
- Identifying patterns across multiple cycles

FIXED: Now uses SolutionTracker directly instead of fake REST API
"""

import os
import sys
import json
import aiohttp
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add cerebro-mcp to path for SolutionTracker access
MCP_SRC = Path(os.environ.get("CEREBRO_MCP_SRC", os.path.expanduser("~/cerebro-mcp/src")))
if str(MCP_SRC) not in sys.path:
    sys.path.insert(0, str(MCP_SRC))

import logging

logger = logging.getLogger(__name__)

from .ollama_client import OllamaClient, ChatMessage
from .thought_journal import ThoughtJournal, Thought, ThoughtPhase, ThoughtType

# Import the real AI Memory solution tracker
try:
    from solution_tracker import SolutionTracker
    SOLUTION_TRACKER_AVAILABLE = True
except ImportError as _st_err:
    SOLUTION_TRACKER_AVAILABLE = False
    logger.critical(
        "[ReflexionEngine] SolutionTracker import FAILED - learnings will NOT persist to AI Memory! Error: %s",
        _st_err
    )


@dataclass
class Critique:
    """Self-critique of an action."""
    action_type: str
    assessment: str  # positive, negative, neutral
    what_worked: List[str]
    what_failed: List[str]
    improvements: List[str]
    confidence_adjustment: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CausalUpdate:
    """Update to the causal model."""
    cause: str
    effect: str
    observed_outcome: str
    strengthen: bool  # True = reinforce, False = weaken
    evidence: str


@dataclass
class Learning:
    """A learning to record."""
    problem: str
    solution: Optional[str]
    what_failed: Optional[str]
    context: str
    tags: List[str]
    learning_type: str  # solution, failure, antipattern


class ReflexionEngine:
    """
    Self-reflection and learning engine.

    Analyzes outcomes and:
    - Provides self-critique
    - Updates causal understanding
    - Records learnings to AI Memory (via SolutionTracker)
    - Identifies recurring patterns
    """

    # Garbage patterns to filter out before saving
    GARBAGE_PATTERNS = [
        "BLOCKED:",
        "doesn't allow",
        "Autonomy level",
        "[...truncated",
        "high risk actions",
        "Unknown parameter",
    ]

    # Keywords that indicate valuable research content
    RESEARCH_KEYWORDS = [
        "options trading", "stock", "trading", "investing", "finance",
        "programming", "python", "javascript", "api", "database",
        "machine learning", "ai", "neural", "deep learning",
    ]

    CRITIQUE_PROMPT = """You are a reflective AI analyzing your own performance.

Action taken: {action_type}
Target: {target}
Description: {description}
Expected outcome: {expected}

Actual result:
- Success: {success}
- Output: {output}
- Error: {error}
- Duration: {duration_ms}ms

Analyze this outcome critically:

1. Assessment (positive/negative/neutral)
2. What worked well (list)
3. What didn't work (list)
4. Specific improvements for next time (list)
5. Confidence adjustment (-0.2 to +0.2)

Be honest and specific. Focus on actionable improvements."""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        thought_journal: Optional[ThoughtJournal] = None
    ):
        self.ollama = ollama_client or OllamaClient()
        self.journal = thought_journal or ThoughtJournal()

        # Initialize SolutionTracker for persisting learnings to AI Memory
        if SOLUTION_TRACKER_AVAILABLE:
            self.solution_tracker = SolutionTracker()
            print("[ReflexionEngine] SolutionTracker connected to AI Memory")
        else:
            self.solution_tracker = None

        # Track recent outcomes for pattern detection
        self._recent_outcomes: List[Dict[str, Any]] = []
        self._max_recent = 50

        # Track what we've already saved to avoid duplicates
        self._saved_hashes: set = set()

    async def self_critique(
        self,
        action_type: str,
        target: Optional[str],
        description: str,
        expected: str,
        success: bool,
        output: Any,
        error: Optional[str],
        duration_ms: float
    ) -> Critique:
        """
        Generate self-critique of an action outcome.

        Uses LLM to honestly assess what worked and what didn't.
        """
        prompt = self.CRITIQUE_PROMPT.format(
            action_type=action_type,
            target=target or "N/A",
            description=description,
            expected=expected,
            success=success,
            output=str(output)[:500] if output else "None",
            error=error or "None",
            duration_ms=duration_ms
        )

        messages = [
            ChatMessage(role="user", content=prompt)
        ]

        response = await self.ollama.chat(messages, thinking=True)

        # Parse critique from response
        critique = self._parse_critique(response.content, action_type)

        # Track outcome
        self._track_outcome({
            "action_type": action_type,
            "target": target,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Log thought
        thought = Thought.create(
            phase=ThoughtPhase.REFLECT,
            type=ThoughtType.REFLECTION,
            content=f"Self-critique: {critique.assessment}",
            reasoning=response.thinking,
            confidence=0.5 + critique.confidence_adjustment,
            what_worked=critique.what_worked,
            what_failed=critique.what_failed
        )
        await self.journal.log_thought(thought)

        return critique

    async def update_causal_model(
        self,
        cause: str,
        effect: str,
        observed: bool,
        evidence: str
    ) -> CausalUpdate:
        """
        Update the causal model based on observed outcome.

        If we predicted cause → effect and it happened, strengthen.
        If it didn't happen, weaken the link.
        """
        update = CausalUpdate(
            cause=cause,
            effect=effect,
            observed_outcome="observed" if observed else "not observed",
            strengthen=observed,
            evidence=evidence
        )

        # Call MCP bridge to update causal model
        try:
            async with aiohttp.ClientSession() as session:
                action = "reinforce" if observed else "weaken"
                async with session.post(
                    f"{self.MCP_BRIDGE_URL}/api/causal/{action}",
                    json={
                        "cause": cause,
                        "effect": effect,
                        "evidence": evidence
                    }
                ) as resp:
                    if resp.status == 200:
                        pass  # Successfully updated
        except Exception:
            pass  # Non-critical, log and continue

        # Log thought
        thought = Thought.create(
            phase=ThoughtPhase.REFLECT,
            type=ThoughtType.LEARNING,
            content=f"Causal update: {cause} → {effect} ({'strengthened' if observed else 'weakened'})",
            confidence=0.7 if observed else 0.5,
            cause=cause,
            effect=effect,
            observed=observed
        )
        await self.journal.log_thought(thought)

        return update

    async def record_learning(
        self,
        problem: str,
        solution: Optional[str] = None,
        what_failed: Optional[str] = None,
        context: str = "",
        tags: Optional[List[str]] = None
    ) -> Learning:
        """
        Record a learning to AI Memory via SolutionTracker.

        Can be:
        - solution: Problem + working solution
        - failure: Problem + what didn't work
        - antipattern: Pattern to avoid

        VALIDATES data before saving to prevent garbage in AI Memory.
        """
        # === VALIDATION STEP 1: Check for garbage patterns ===
        all_content = f"{problem} {solution or ''} {what_failed or ''} {context}"
        for garbage in self.GARBAGE_PATTERNS:
            if garbage.lower() in all_content.lower():
                print(f"[ReflexionEngine] FILTERED garbage: contains '{garbage}'")
                # Return without saving - this is garbage
                return Learning(
                    problem=problem,
                    solution=solution,
                    what_failed=what_failed,
                    context=context,
                    tags=tags or [],
                    learning_type="filtered"
                )

        # === VALIDATION STEP 2: Minimum content length ===
        if len(problem) < 10:
            print(f"[ReflexionEngine] FILTERED: Problem too short ({len(problem)} chars)")
            return Learning(problem=problem, solution=solution, what_failed=what_failed,
                          context=context, tags=tags or [], learning_type="filtered")

        # === VALIDATION STEP 3: Deduplication ===
        import hashlib
        content_hash = hashlib.md5(f"{problem}:{solution or ''}".encode()).hexdigest()[:16]
        if content_hash in self._saved_hashes:
            print(f"[ReflexionEngine] FILTERED: Duplicate content (hash: {content_hash})")
            return Learning(problem=problem, solution=solution, what_failed=what_failed,
                          context=context, tags=tags or [], learning_type="duplicate")

        # Determine learning type
        if solution:
            learning_type = "solution"
        elif what_failed:
            learning_type = "failure" if problem else "antipattern"
        else:
            learning_type = "observation"

        learning = Learning(
            problem=problem,
            solution=solution,
            what_failed=what_failed,
            context=context,
            tags=tags or [],
            learning_type=learning_type
        )

        # === SAVE TO AI MEMORY via SolutionTracker ===
        saved_to_memory = False
        if self.solution_tracker:
            try:
                # Categorize with proper tags
                auto_tags = self._auto_categorize(problem, solution or "", context)
                all_tags = list(set((tags or []) + auto_tags + ["cognitive_loop", "cerebro"]))

                if learning_type == "solution":
                    result = self.solution_tracker.record_solution(
                        problem=problem,
                        solution=solution,
                        context=context,
                        tags=all_tags
                    )
                    saved_to_memory = True
                    print(f"[ReflexionEngine] SAVED SOLUTION to AI Memory: {result.get('id', 'unknown')}")

                elif learning_type in ["failure", "antipattern"]:
                    result = self.solution_tracker.record_antipattern(
                        what_not_to_do=solution or problem,
                        why_it_failed=what_failed or "Unknown failure reason",
                        original_problem=problem,
                        tags=all_tags
                    )
                    saved_to_memory = True
                    print(f"[ReflexionEngine] SAVED ANTIPATTERN to AI Memory: {result.get('id', 'unknown')}")

                # Mark as saved to prevent duplicates
                self._saved_hashes.add(content_hash)

            except Exception as e:
                print(f"[ReflexionEngine] ERROR saving to AI Memory: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[ReflexionEngine] WARNING: SolutionTracker not available - learning NOT persisted!")

        # Log thought
        thought = Thought.create(
            phase=ThoughtPhase.REFLECT,
            type=ThoughtType.LEARNING,
            content=f"{'✓' if saved_to_memory else '✗'} Recorded {learning_type}: {problem[:100]}",
            confidence=0.8 if saved_to_memory else 0.3,
            learning_type=learning_type,
            has_solution=solution is not None,
            saved_to_memory=saved_to_memory
        )
        await self.journal.log_thought(thought)

        return learning

    def _auto_categorize(self, problem: str, solution: str, context: str) -> List[str]:
        """
        Automatically categorize content with appropriate tags.
        This ensures learnings are properly organized in AI Memory.
        """
        all_text = f"{problem} {solution} {context}".lower()
        tags = []

        # Topic detection
        if any(kw in all_text for kw in ["trading", "stock", "option", "invest", "finance"]):
            tags.append("trading")
            tags.append("finance")
        if any(kw in all_text for kw in ["python", "javascript", "code", "programming", "api"]):
            tags.append("programming")
        if any(kw in all_text for kw in ["machine learning", "ai", "neural", "model"]):
            tags.append("ai")
        if any(kw in all_text for kw in ["web search", "research", "learn about"]):
            tags.append("research")
        if any(kw in all_text for kw in ["error", "bug", "fix", "issue"]):
            tags.append("debugging")

        # Source detection
        if "investopedia" in all_text or "nerdwallet" in all_text:
            tags.append("educational_source")
        if "reddit" in all_text or "youtube" in all_text:
            tags.append("community_source")

        return tags

    async def identify_patterns(self) -> List[Dict[str, Any]]:
        """
        Identify recurring patterns in recent outcomes.

        Looks for:
        - Repeated failures on same action type
        - Consistent successes
        - Time-based patterns
        """
        patterns = []

        if len(self._recent_outcomes) < 5:
            return patterns

        # Group by action type
        by_action: Dict[str, List[Dict]] = {}
        for outcome in self._recent_outcomes:
            action = outcome.get("action_type", "unknown")
            if action not in by_action:
                by_action[action] = []
            by_action[action].append(outcome)

        # Look for patterns
        for action, outcomes in by_action.items():
            if len(outcomes) >= 3:
                successes = sum(1 for o in outcomes if o.get("success"))
                failures = len(outcomes) - successes

                # High failure rate
                if failures >= 3 and failures / len(outcomes) > 0.6:
                    patterns.append({
                        "type": "repeated_failure",
                        "action": action,
                        "failure_rate": failures / len(outcomes),
                        "suggestion": f"Consider reviewing {action} implementation or reducing its usage"
                    })

                # Consistent success
                elif successes >= 3 and successes / len(outcomes) > 0.8:
                    patterns.append({
                        "type": "reliable_action",
                        "action": action,
                        "success_rate": successes / len(outcomes),
                        "suggestion": f"{action} is reliable, can increase confidence"
                    })

        return patterns

    async def generate_insights(self) -> List[str]:
        """
        Generate insights from accumulated learnings.

        Uses LLM to synthesize patterns into actionable insights.
        """
        # Get recent thoughts
        recent_thoughts = await self.journal.get_recent(limit=20)

        if len(recent_thoughts) < 5:
            return ["Insufficient data for insights"]

        # Build summary for LLM
        thought_summary = "\n".join([
            f"- [{t.phase}] {t.content} (confidence: {t.confidence:.2f})"
            for t in recent_thoughts[:15]
        ])

        prompt = f"""Analyze these recent cognitive loop thoughts and generate 3-5 actionable insights:

Recent Activity:
{thought_summary}

Provide:
1. Key patterns observed
2. Recommendations for improvement
3. Potential opportunities

Be specific and actionable."""

        messages = [
            ChatMessage(role="user", content=prompt)
        ]

        response = await self.ollama.chat(messages, thinking=True)

        # Parse insights
        insights = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                insights.append(line.lstrip('-•0123456789.').strip())

        # Log insight generation
        thought = Thought.create(
            phase=ThoughtPhase.REFLECT,
            type=ThoughtType.INSIGHT,
            content=f"Generated {len(insights)} insights from recent activity",
            reasoning=response.thinking,
            confidence=0.7,
            insight_count=len(insights)
        )
        await self.journal.log_thought(thought)

        return insights[:5]

    def _track_outcome(self, outcome: Dict[str, Any]):
        """Track an outcome for pattern detection."""
        self._recent_outcomes.append(outcome)
        if len(self._recent_outcomes) > self._max_recent:
            self._recent_outcomes = self._recent_outcomes[-self._max_recent:]

    def _parse_critique(self, content: str, action_type: str) -> Critique:
        """Parse LLM response into Critique."""
        content_lower = content.lower()

        # Determine assessment
        if 'positive' in content_lower:
            assessment = 'positive'
        elif 'negative' in content_lower:
            assessment = 'negative'
        else:
            assessment = 'neutral'

        # Extract lists
        what_worked = []
        what_failed = []
        improvements = []

        current_section = None
        for line in content.split('\n'):
            line_lower = line.lower()
            if 'worked' in line_lower or 'success' in line_lower:
                current_section = 'worked'
            elif 'failed' in line_lower or 'didn\'t' in line_lower or 'did not' in line_lower:
                current_section = 'failed'
            elif 'improve' in line_lower or 'next time' in line_lower:
                current_section = 'improvements'
            elif line.strip().startswith('-') or line.strip().startswith('•'):
                item = line.strip().lstrip('-•').strip()
                if item:
                    if current_section == 'worked':
                        what_worked.append(item)
                    elif current_section == 'failed':
                        what_failed.append(item)
                    elif current_section == 'improvements':
                        improvements.append(item)

        # Extract confidence adjustment
        confidence_adjustment = 0.0
        import re
        conf_match = re.search(r'confidence[:\s]+([+-]?[0-9.]+)', content_lower)
        if conf_match:
            try:
                confidence_adjustment = float(conf_match.group(1))
                confidence_adjustment = max(-0.2, min(0.2, confidence_adjustment))
            except:
                pass

        # Default adjustment based on assessment
        if confidence_adjustment == 0.0:
            if assessment == 'positive':
                confidence_adjustment = 0.1
            elif assessment == 'negative':
                confidence_adjustment = -0.1

        return Critique(
            action_type=action_type,
            assessment=assessment,
            what_worked=what_worked[:5],
            what_failed=what_failed[:5],
            improvements=improvements[:5],
            confidence_adjustment=confidence_adjustment
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get reflexion engine statistics."""
        # Analyze recent outcomes
        if not self._recent_outcomes:
            return {
                "total_outcomes": 0,
                "success_rate": 0,
                "patterns": []
            }

        successes = sum(1 for o in self._recent_outcomes if o.get("success"))
        total = len(self._recent_outcomes)

        patterns = await self.identify_patterns()

        return {
            "total_outcomes": total,
            "success_rate": successes / total if total > 0 else 0,
            "recent_successes": successes,
            "recent_failures": total - successes,
            "patterns": patterns,
            "by_action": self._count_by_action()
        }

    def _count_by_action(self) -> Dict[str, Dict[str, int]]:
        """Count outcomes by action type."""
        by_action: Dict[str, Dict[str, int]] = {}
        for outcome in self._recent_outcomes:
            action = outcome.get("action_type", "unknown")
            if action not in by_action:
                by_action[action] = {"success": 0, "failure": 0}
            if outcome.get("success"):
                by_action[action]["success"] += 1
            else:
                by_action[action]["failure"] += 1
        return by_action


class GoalReflexionEngine(ReflexionEngine):
    """
    Extended Reflexion Engine for Goal-Level Learning.

    Implements Reflexion pattern specifically for goal pursuit:
    - Verbal reinforcement on subtask completion/failure
    - Approach-level learning (what works, what doesn't)
    - Escalation after repeated failures
    - Pattern detection across goals
    """

    SUBTASK_REFLECTION_PROMPT = """Reflect on this subtask outcome for Reflexion learning.

Goal: {goal_description}
Milestone: {milestone_description}
Subtask: {subtask_description}
Agent Type: {agent_type}

Result:
- Success: {success}
- Output: {output}
- Error: {error}
- Attempts: {attempts}/{max_attempts}

Previous Failed Approaches (avoid these):
{failed_approaches}

REFLEXION ANALYSIS:
1. What SPECIFICALLY worked or didn't work?
2. What should I try DIFFERENTLY next time?
3. Is this a pattern I'm seeing? (same failure mode as before?)
4. Should I escalate to Professor? (after 3 failures)

Provide VERBAL REINFORCEMENT - a short phrase I'll remember:
- For success: "X works well for Y because Z"
- For failure: "Don't try X when Y - do Z instead"

Output JSON:
{{
    "assessment": "success" | "partial" | "failure",
    "verbal_reinforcement": "Short memorable lesson",
    "pattern_detected": true | false,
    "pattern_description": "If pattern detected",
    "should_escalate": true | false,
    "escalation_reason": "If should escalate",
    "alternative_approach": "Suggested next approach"
}}"""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        thought_journal: Optional[ThoughtJournal] = None
    ):
        super().__init__(ollama_client, thought_journal)

        # Track subtask outcomes per goal for pattern detection
        self._goal_outcomes: Dict[str, List[Dict]] = {}

    async def reflect_on_subtask(
        self,
        subtask: 'Subtask',
        goal: 'Goal',
        milestone: 'Milestone',
        success: bool,
        output: str,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform Reflexion learning after subtask completion.

        Args:
            subtask: The subtask that was executed
            goal: The parent goal
            milestone: The parent milestone
            success: Whether the subtask succeeded
            output: The agent's output
            error: Error message if failed

        Returns:
            Reflexion result with verbal reinforcement and recommendations
        """
        # Build prompt
        prompt = self.SUBTASK_REFLECTION_PROMPT.format(
            goal_description=goal.description[:200],
            milestone_description=milestone.description[:200],
            subtask_description=subtask.description[:300],
            agent_type=subtask.agent_type,
            success=success,
            output=str(output)[:500] if output else "None",
            error=error or "None",
            attempts=subtask.attempts,
            max_attempts=subtask.max_attempts,
            failed_approaches="\n".join(f"- {a}" for a in goal.failed_approaches[-5:]) or "None"
        )

        try:
            messages = [ChatMessage(role="user", content=prompt)]
            response = await self.ollama.chat(messages, thinking=True)

            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {
                    "assessment": "success" if success else "failure",
                    "verbal_reinforcement": f"{'Completed' if success else 'Failed'}: {subtask.description[:50]}",
                    "pattern_detected": False,
                    "should_escalate": subtask.attempts >= subtask.max_attempts
                }

            # Record the outcome
            self._track_goal_outcome(goal.goal_id, {
                "subtask_id": subtask.subtask_id,
                "success": success,
                "agent_type": subtask.agent_type,
                "verbal_reinforcement": result.get("verbal_reinforcement", ""),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Record to AI Memory if significant
            if result.get("verbal_reinforcement"):
                if success:
                    await self.record_learning(
                        problem=subtask.description[:200],
                        solution=result.get("verbal_reinforcement", ""),
                        context=f"Goal: {goal.description[:100]}",
                        tags=["goal_pursuit", "subtask", subtask.agent_type, "success"]
                    )
                else:
                    await self.record_learning(
                        problem=subtask.description[:200],
                        what_failed=result.get("verbal_reinforcement", ""),
                        context=f"Goal: {goal.description[:100]}",
                        tags=["goal_pursuit", "subtask", subtask.agent_type, "failure"]
                    )

            # Log thought
            thought = Thought.create(
                phase=ThoughtPhase.REFLECT,
                type=ThoughtType.LEARNING,
                content=f"Reflexion: {result.get('verbal_reinforcement', 'No insight')}",
                reasoning=response.thinking,
                confidence=0.8 if success else 0.5,
                subtask_id=subtask.subtask_id,
                goal_id=goal.goal_id,
                should_escalate=result.get("should_escalate", False)
            )
            await self.journal.log_thought(thought)

            return result

        except Exception as e:
            print(f"[GoalReflexion] Error in subtask reflection: {e}")
            return {
                "assessment": "success" if success else "failure",
                "verbal_reinforcement": "Reflection failed - continuing",
                "error": str(e)
            }

    def _track_goal_outcome(self, goal_id: str, outcome: Dict):
        """Track outcomes per goal for pattern detection."""
        if goal_id not in self._goal_outcomes:
            self._goal_outcomes[goal_id] = []
        self._goal_outcomes[goal_id].append(outcome)

        # Keep only last 20 outcomes per goal
        if len(self._goal_outcomes[goal_id]) > 20:
            self._goal_outcomes[goal_id] = self._goal_outcomes[goal_id][-20:]

    def get_goal_patterns(self, goal_id: str) -> List[Dict]:
        """Identify patterns in goal outcomes."""
        outcomes = self._goal_outcomes.get(goal_id, [])
        if len(outcomes) < 3:
            return []

        patterns = []

        # Check failure rate by agent type
        by_type: Dict[str, Dict[str, int]] = {}
        for o in outcomes:
            atype = o.get("agent_type", "unknown")
            if atype not in by_type:
                by_type[atype] = {"success": 0, "failure": 0}
            if o.get("success"):
                by_type[atype]["success"] += 1
            else:
                by_type[atype]["failure"] += 1

        for atype, counts in by_type.items():
            total = counts["success"] + counts["failure"]
            if total >= 3:
                failure_rate = counts["failure"] / total
                if failure_rate >= 0.6:
                    patterns.append({
                        "type": "high_failure_rate",
                        "agent_type": atype,
                        "failure_rate": failure_rate,
                        "suggestion": f"Consider alternative approach for {atype} tasks"
                    })
                elif failure_rate <= 0.2:
                    patterns.append({
                        "type": "reliable_agent",
                        "agent_type": atype,
                        "success_rate": 1 - failure_rate,
                        "suggestion": f"{atype} agents are working well - continue using"
                    })

        return patterns

    async def suggest_approach(self, goal: 'Goal', failed_subtask: 'Subtask') -> Optional[str]:
        """
        Suggest an alternative approach after subtask failure.

        Uses Reflexion pattern to generate verbal guidance.
        """
        prompt = f"""The following subtask FAILED for goal "{goal.description[:100]}":

Subtask: {failed_subtask.description}
Agent Type: {failed_subtask.agent_type}
Attempts: {failed_subtask.attempts}
Learnings from this subtask: {failed_subtask.learnings}

Previously failed approaches for this goal:
{chr(10).join(f"- {a}" for a in goal.failed_approaches[-5:]) or "None"}

Suggest ONE alternative approach that might work better.
Be specific and actionable. Output only the suggestion, no preamble."""

        try:
            messages = [ChatMessage(role="user", content=prompt)]
            response = await self.ollama.chat(messages, thinking=False)
            return response.content.strip()[:500]
        except Exception:
            return None


# Singleton instance for GoalReflexionEngine
_goal_reflexion_instance: Optional[GoalReflexionEngine] = None


def get_goal_reflexion_engine() -> GoalReflexionEngine:
    """Get or create the goal reflexion engine instance."""
    global _goal_reflexion_instance
    if _goal_reflexion_instance is None:
        _goal_reflexion_instance = GoalReflexionEngine()
    return _goal_reflexion_instance
