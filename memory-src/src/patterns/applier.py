"""
Pattern Applier - Apply validated patterns to context injection.

Part of Phase 4 Enhancement in the All-Knowing Brain PRD.
Takes validated patterns and formats them for injection into Claude's context.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class AppliedPattern:
    """A pattern formatted for injection."""
    pattern_type: str
    content: str
    confidence: float
    priority: int  # Lower = higher priority


# Pattern type priorities (lower = injected first)
PATTERN_PRIORITIES = {
    "preference": 1,      # Preferences are most important
    "workflow": 2,        # Workflows guide behavior
    "problem": 3,         # Problems provide warnings
    "knowledge_gap": 4,   # Gaps suggest what to explain
    "topic": 5,           # Topics are informational
}


class PatternApplier:
    """
    Applies validated patterns to context injection.
    Formats patterns for Claude to understand and act on.
    """

    def __init__(self, max_patterns: int = 6, max_tokens: int = 300):
        """
        Initialize the applier.

        Args:
            max_patterns: Maximum patterns to inject
            max_tokens: Approximate token budget for patterns
        """
        self.max_patterns = max_patterns
        self.max_tokens = max_tokens

    def format_topic_pattern(self, pattern_data: Dict[str, Any],
                             confidence: float) -> AppliedPattern:
        """
        Format a topic pattern for injection.

        Args:
            pattern_data: Pattern data from validator
            confidence: Confidence score

        Returns:
            AppliedPattern ready for injection
        """
        topic = pattern_data.get("topic", "unknown")
        occurrences = pattern_data.get("occurrences", 0)
        frequency = pattern_data.get("frequency_percent", 0)

        content = f"User frequently works on: {topic} ({occurrences} sessions, {frequency}% of work)"

        return AppliedPattern(
            pattern_type="topic",
            content=content,
            confidence=confidence,
            priority=PATTERN_PRIORITIES["topic"]
        )

    def format_problem_pattern(self, pattern_data: Dict[str, Any],
                                confidence: float) -> AppliedPattern:
        """
        Format a problem pattern for injection.

        Args:
            pattern_data: Pattern data from validator
            confidence: Confidence score

        Returns:
            AppliedPattern ready for injection
        """
        problem = pattern_data.get("problem", "unknown")
        occurrences = pattern_data.get("occurrences", 0)

        # Clean up the problem description
        problem_clean = problem.strip()[:80]
        if problem_clean.endswith("..."):
            problem_clean = problem_clean[:-3].strip()

        content = f"RECURRING ISSUE ({occurrences}x): {problem_clean}"

        return AppliedPattern(
            pattern_type="problem",
            content=content,
            confidence=confidence,
            priority=PATTERN_PRIORITIES["problem"]
        )

    def format_knowledge_gap(self, pattern_data: Dict[str, Any],
                              confidence: float) -> AppliedPattern:
        """
        Format a knowledge gap pattern for injection.

        Args:
            pattern_data: Pattern data from validator
            confidence: Confidence score

        Returns:
            AppliedPattern ready for injection
        """
        concept = pattern_data.get("concept", "unknown")
        times = pattern_data.get("times_explained", 0)

        content = f"Previously explained {times}x: {concept} (consider adding to persistent context)"

        return AppliedPattern(
            pattern_type="knowledge_gap",
            content=content,
            confidence=confidence,
            priority=PATTERN_PRIORITIES["knowledge_gap"]
        )

    def format_workflow_pattern(self, pattern_data: Dict[str, Any],
                                 confidence: float) -> AppliedPattern:
        """
        Format a workflow pattern for injection.

        Args:
            pattern_data: Pattern data from validator
            confidence: Confidence score

        Returns:
            AppliedPattern ready for injection
        """
        sequence = pattern_data.get("sequence", [])
        occurrences = pattern_data.get("occurrences", 0)

        if sequence:
            workflow_str = " → ".join(sequence)
            content = f"USER WORKFLOW: {workflow_str} (observed {occurrences}x)"
        else:
            content = f"Workflow pattern with {occurrences} observations"

        return AppliedPattern(
            pattern_type="workflow",
            content=content,
            confidence=confidence,
            priority=PATTERN_PRIORITIES["workflow"]
        )

    def apply_patterns(self, validated_patterns: Dict[str, Any]) -> List[AppliedPattern]:
        """
        Apply all validated patterns, formatting them for injection.

        Args:
            validated_patterns: Output from PatternValidator.get_validated_patterns()

        Returns:
            List of AppliedPatterns, sorted by priority
        """
        applied = []

        # Format topics
        for result in validated_patterns.get("topics", []):
            if hasattr(result, "pattern_data"):
                # It's a ValidationResult
                pattern = self.format_topic_pattern(result.pattern_data, result.confidence)
            else:
                # It's a dict from cache
                pattern = self.format_topic_pattern(result.get("data", {}), result.get("confidence", 0.5))
            applied.append(pattern)

        # Format problems
        for result in validated_patterns.get("problems", []):
            if hasattr(result, "pattern_data"):
                pattern = self.format_problem_pattern(result.pattern_data, result.confidence)
            else:
                pattern = self.format_problem_pattern(result.get("data", {}), result.get("confidence", 0.5))
            applied.append(pattern)

        # Format knowledge gaps
        for result in validated_patterns.get("knowledge_gaps", []):
            if hasattr(result, "pattern_data"):
                pattern = self.format_knowledge_gap(result.pattern_data, result.confidence)
            else:
                pattern = self.format_knowledge_gap(result.get("data", {}), result.get("confidence", 0.5))
            applied.append(pattern)

        # Sort by priority, then confidence (for same priority)
        applied.sort(key=lambda p: (p.priority, -p.confidence))

        return applied[:self.max_patterns]

    def format_for_injection(self, patterns: List[AppliedPattern]) -> str:
        """
        Format applied patterns into a string for context injection.

        Args:
            patterns: List of AppliedPatterns

        Returns:
            Formatted string for injection
        """
        if not patterns:
            return ""

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        total_chars = 0
        max_chars = self.max_tokens * 4

        lines = []
        for pattern in patterns:
            line = f"• {pattern.content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        if not lines:
            return ""

        result = "[LEARNED PATTERNS]\n"
        result += "\n".join(lines)
        result += "\n[/LEARNED PATTERNS]"

        return result

    def get_injection_context(self, validated_patterns: Dict[str, Any] = None) -> str:
        """
        Get the complete injection context for patterns.

        Args:
            validated_patterns: Pre-validated patterns (optional, will fetch if not provided)

        Returns:
            Formatted string ready for injection
        """
        if validated_patterns is None:
            from .validator import get_validated_patterns
            validated_patterns = get_validated_patterns(max_per_type=3)

        applied = self.apply_patterns(validated_patterns)
        return self.format_for_injection(applied)


def apply_patterns_to_context(context: str,
                               validated_patterns: Dict[str, Any] = None,
                               max_patterns: int = 6) -> str:
    """
    Convenience function to apply patterns to existing context.

    Args:
        context: Existing context string
        validated_patterns: Pre-validated patterns (optional)
        max_patterns: Maximum patterns to inject

    Returns:
        Context with patterns appended
    """
    applier = PatternApplier(max_patterns=max_patterns)
    pattern_context = applier.get_injection_context(validated_patterns)

    if pattern_context:
        if context:
            return context + "\n\n" + pattern_context
        return pattern_context

    return context


if __name__ == "__main__":
    # Test the applier
    from validator import get_validated_patterns

    print("=== Pattern Applier Test ===")

    # Get validated patterns
    validated = get_validated_patterns(max_per_type=3)

    print(f"\nValidation stats: {validated.get('stats', {})}")

    # Apply patterns
    applier = PatternApplier(max_patterns=6)
    applied = applier.apply_patterns(validated)

    print(f"\nApplied {len(applied)} patterns:")
    for pattern in applied:
        print(f"  [{pattern.priority}] [{pattern.confidence:.2f}] {pattern.pattern_type}: {pattern.content[:60]}...")

    # Get injection context
    injection = applier.format_for_injection(applied)

    print("\n=== Injection Context ===")
    print(injection)
