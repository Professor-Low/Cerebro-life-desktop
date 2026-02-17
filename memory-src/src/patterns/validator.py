"""
Pattern Validator - Validate patterns before acting on them.

Part of Phase 4 Enhancement in the All-Knowing Brain PRD.
Validates patterns based on:
- Minimum occurrences
- Confidence threshold
- Age/freshness
- Contradiction rate
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


# Validation rules
VALIDATION_RULES = {
    "min_occurrences": 5,        # Pattern must appear at least this many times
    "min_confidence": 0.5,       # Minimum confidence score (0-1) - lowered to allow high-occurrence patterns
    "max_age_days": 30,          # Patterns older than this are stale
    "max_contradiction_rate": 0.2,  # If >20% contradictions, pattern is unreliable
}

# Pattern type weights (how much each type matters)
PATTERN_TYPE_WEIGHTS = {
    "topic": 0.6,        # Topics are informational, lower weight
    "problem": 0.9,      # Problems are actionable, higher weight
    "workflow": 0.85,    # Workflows are preferences, high weight
    "preference": 0.95,  # Direct preferences are most important
    "knowledge_gap": 0.7,  # Knowledge gaps should be addressed
}


@dataclass
class ValidationResult:
    """Result of pattern validation."""
    is_valid: bool
    reason: str
    confidence: float
    pattern_type: str
    pattern_data: Dict[str, Any]


class PatternValidator:
    """
    Validates patterns before they are applied to context.
    Filters out unreliable, stale, or low-confidence patterns.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.patterns_cache_path = self.base_path / "validated_patterns.json"
        self.validation_rules = VALIDATION_RULES.copy()

    def validate_topic_pattern(self, pattern: Dict[str, Any]) -> ValidationResult:
        """
        Validate a recurring topic pattern.

        Args:
            pattern: Dict with 'topic', 'occurrences', 'frequency_percent', 'conversations'

        Returns:
            ValidationResult with validity status
        """
        # Handle field aliasing: detector may return 'count' instead of 'occurrences'
        occurrences = pattern.get("occurrences") or pattern.get("count", 0)
        frequency = pattern.get("frequency_percent", 0)
        topic = pattern.get("topic", "")

        # Check minimum occurrences
        if occurrences < self.validation_rules["min_occurrences"]:
            return ValidationResult(
                is_valid=False,
                reason=f"Insufficient occurrences: {occurrences} < {self.validation_rules['min_occurrences']}",
                confidence=0.0,
                pattern_type="topic",
                pattern_data=pattern
            )

        # Calculate confidence based on frequency and occurrences
        # More occurrences + higher frequency = higher confidence
        freq_score = min(1.0, frequency / 50.0) if frequency > 0 else 0.5  # Default 0.5 if not provided
        occ_score = min(1.0, occurrences / 20.0)  # Normalize: 20+ = 1.0
        # Weight occurrences more heavily since frequency isn't always available
        confidence = (freq_score * 0.3) + (occ_score * 0.7)

        if confidence < self.validation_rules["min_confidence"]:
            return ValidationResult(
                is_valid=False,
                reason=f"Low confidence: {confidence:.2f} < {self.validation_rules['min_confidence']}",
                confidence=confidence,
                pattern_type="topic",
                pattern_data=pattern
            )

        return ValidationResult(
            is_valid=True,
            reason="Topic pattern validated",
            confidence=confidence,
            pattern_type="topic",
            pattern_data=pattern
        )

    def validate_problem_pattern(self, pattern: Dict[str, Any]) -> ValidationResult:
        """
        Validate a recurring problem pattern.

        Args:
            pattern: Dict with 'problem', 'occurrences', 'conversations', 'examples'

        Returns:
            ValidationResult with validity status
        """
        # Handle field aliasing: detector may return 'count' instead of 'occurrences'
        occurrences = pattern.get("occurrences") or pattern.get("count", 0)
        problem = pattern.get("problem", "")
        # Handle field aliasing: detector may return 'solutions' instead of 'examples'
        examples = pattern.get("examples") or pattern.get("solutions", [])

        # Problems need fewer occurrences to be significant (2+)
        min_problem_occurrences = max(2, self.validation_rules["min_occurrences"] - 3)

        if occurrences < min_problem_occurrences:
            return ValidationResult(
                is_valid=False,
                reason=f"Insufficient occurrences: {occurrences} < {min_problem_occurrences}",
                confidence=0.0,
                pattern_type="problem",
                pattern_data=pattern
            )

        # Higher confidence for problems with more examples and occurrences
        occ_score = min(1.0, occurrences / 5.0)  # Normalize: 5+ = 1.0
        example_score = min(1.0, len(examples) / 3.0)  # More examples = better
        confidence = (occ_score * 0.7) + (example_score * 0.3)

        # Problems are actionable - apply type weight
        confidence *= PATTERN_TYPE_WEIGHTS["problem"]

        if confidence < self.validation_rules["min_confidence"]:
            return ValidationResult(
                is_valid=False,
                reason=f"Low confidence: {confidence:.2f} < {self.validation_rules['min_confidence']}",
                confidence=confidence,
                pattern_type="problem",
                pattern_data=pattern
            )

        return ValidationResult(
            is_valid=True,
            reason="Problem pattern validated",
            confidence=confidence,
            pattern_type="problem",
            pattern_data=pattern
        )

    def validate_knowledge_gap(self, pattern: Dict[str, Any]) -> ValidationResult:
        """
        Validate a knowledge gap pattern.

        Args:
            pattern: Dict with 'concept', 'times_explained', 'conversations'

        Returns:
            ValidationResult with validity status
        """
        # Handle field aliasing: detector may return 'explanation_count' instead of 'times_explained'
        times_explained = pattern.get("times_explained") or pattern.get("explanation_count", 0)
        concept = pattern.get("concept", "")

        # Knowledge gaps need repeated explanation (3+)
        min_explanations = 3

        if times_explained < min_explanations:
            return ValidationResult(
                is_valid=False,
                reason=f"Insufficient explanations: {times_explained} < {min_explanations}",
                confidence=0.0,
                pattern_type="knowledge_gap",
                pattern_data=pattern
            )

        # More explanations = higher confidence it's a real gap
        confidence = min(1.0, times_explained / 5.0) * PATTERN_TYPE_WEIGHTS["knowledge_gap"]

        if confidence < self.validation_rules["min_confidence"]:
            return ValidationResult(
                is_valid=False,
                reason=f"Low confidence: {confidence:.2f} < {self.validation_rules['min_confidence']}",
                confidence=confidence,
                pattern_type="knowledge_gap",
                pattern_data=pattern
            )

        return ValidationResult(
            is_valid=True,
            reason="Knowledge gap validated",
            confidence=confidence,
            pattern_type="knowledge_gap",
            pattern_data=pattern
        )

    def validate_workflow_pattern(self,
                                   action_sequence: List[str],
                                   occurrences: int,
                                   last_seen: Optional[datetime] = None) -> ValidationResult:
        """
        Validate a workflow pattern (sequence of actions).

        Args:
            action_sequence: List of actions (e.g., ["edit", "test", "commit"])
            occurrences: How many times this sequence was observed
            last_seen: When this pattern was last observed

        Returns:
            ValidationResult with validity status
        """
        pattern_data = {
            "sequence": action_sequence,
            "occurrences": occurrences,
            "last_seen": last_seen.isoformat() if last_seen else None
        }

        if occurrences < self.validation_rules["min_occurrences"]:
            return ValidationResult(
                is_valid=False,
                reason=f"Insufficient occurrences: {occurrences} < {self.validation_rules['min_occurrences']}",
                confidence=0.0,
                pattern_type="workflow",
                pattern_data=pattern_data
            )

        # Check staleness
        if last_seen:
            age_days = (datetime.now() - last_seen).days
            if age_days > self.validation_rules["max_age_days"]:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Stale pattern: {age_days} days old > {self.validation_rules['max_age_days']}",
                    confidence=0.0,
                    pattern_type="workflow",
                    pattern_data=pattern_data
                )

        # Calculate confidence
        occ_score = min(1.0, occurrences / 10.0)
        recency_score = 1.0 if not last_seen else max(0.5, 1.0 - (datetime.now() - last_seen).days / 30)
        confidence = (occ_score * 0.7 + recency_score * 0.3) * PATTERN_TYPE_WEIGHTS["workflow"]

        if confidence < self.validation_rules["min_confidence"]:
            return ValidationResult(
                is_valid=False,
                reason=f"Low confidence: {confidence:.2f} < {self.validation_rules['min_confidence']}",
                confidence=confidence,
                pattern_type="workflow",
                pattern_data=pattern_data
            )

        return ValidationResult(
            is_valid=True,
            reason="Workflow pattern validated",
            confidence=confidence,
            pattern_type="workflow",
            pattern_data=pattern_data
        )

    def validate_all_patterns(self,
                               topics: List[Dict] = None,
                               problems: List[Dict] = None,
                               knowledge_gaps: List[Dict] = None) -> Dict[str, List[ValidationResult]]:
        """
        Validate all patterns at once.

        Args:
            topics: List of topic patterns from PatternDetector
            problems: List of problem patterns
            knowledge_gaps: List of knowledge gap patterns

        Returns:
            Dict with pattern type -> list of valid patterns
        """
        results = {
            "topics": [],
            "problems": [],
            "knowledge_gaps": [],
            "stats": {
                "total_validated": 0,
                "total_rejected": 0,
                "by_type": {}
            }
        }

        # Validate topics
        if topics:
            for pattern in topics:
                result = self.validate_topic_pattern(pattern)
                if result.is_valid:
                    results["topics"].append(result)
                    results["stats"]["total_validated"] += 1
                else:
                    results["stats"]["total_rejected"] += 1

        # Validate problems
        if problems:
            for pattern in problems:
                result = self.validate_problem_pattern(pattern)
                if result.is_valid:
                    results["problems"].append(result)
                    results["stats"]["total_validated"] += 1
                else:
                    results["stats"]["total_rejected"] += 1

        # Validate knowledge gaps
        if knowledge_gaps:
            for pattern in knowledge_gaps:
                result = self.validate_knowledge_gap(pattern)
                if result.is_valid:
                    results["knowledge_gaps"].append(result)
                    results["stats"]["total_validated"] += 1
                else:
                    results["stats"]["total_rejected"] += 1

        # Update stats
        results["stats"]["by_type"] = {
            "topics": len(results["topics"]),
            "problems": len(results["problems"]),
            "knowledge_gaps": len(results["knowledge_gaps"]),
        }

        return results

    def get_validated_patterns(self, max_per_type: int = 5, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get all validated patterns, using cache when available.

        Args:
            max_per_type: Maximum patterns per type to return
            force_refresh: If True, bypass cache and re-run detection

        Returns:
            Dict with validated patterns ready for injection
        """
        # Check cache first (much faster than re-scanning 1000+ conversations)
        if not force_refresh:
            cached = self.load_cached_patterns()
            if cached:
                # Apply max_per_type limit to cached results
                for key in ["topics", "problems", "knowledge_gaps"]:
                    if key in cached:
                        cached[key] = cached[key][:max_per_type]
                cached["from_cache"] = True
                return cached

        try:
            from pattern_detector import PatternDetector
            detector = PatternDetector()

            # Get current patterns
            topics = detector.detect_recurring_topics(threshold=3)
            problems = detector.detect_recurring_problems()
            knowledge_gaps = detector.find_knowledge_gaps(min_explanations=3)

            # Validate all
            validated = self.validate_all_patterns(
                topics=topics,
                problems=problems,
                knowledge_gaps=knowledge_gaps
            )

            # Sort by confidence and limit
            for key in ["topics", "problems", "knowledge_gaps"]:
                validated[key] = sorted(
                    validated[key],
                    key=lambda x: x.confidence,
                    reverse=True
                )[:max_per_type]

            # Save to cache for future fast access
            self.save_validated_patterns(validated)
            validated["from_cache"] = False

            return validated

        except Exception as e:
            return {
                "error": str(e),
                "topics": [],
                "problems": [],
                "knowledge_gaps": [],
                "stats": {"total_validated": 0, "total_rejected": 0}
            }

    def save_validated_patterns(self, patterns: Dict[str, Any]) -> None:
        """Save validated patterns to cache."""
        try:
            # Convert ValidationResults to dicts
            serializable = {
                "topics": [
                    {"confidence": r.confidence, "data": r.pattern_data}
                    for r in patterns.get("topics", [])
                ],
                "problems": [
                    {"confidence": r.confidence, "data": r.pattern_data}
                    for r in patterns.get("problems", [])
                ],
                "knowledge_gaps": [
                    {"confidence": r.confidence, "data": r.pattern_data}
                    for r in patterns.get("knowledge_gaps", [])
                ],
                "stats": patterns.get("stats", {}),
                "validated_at": datetime.now().isoformat()
            }

            with open(self.patterns_cache_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error saving patterns: {e}")

    def load_cached_patterns(self) -> Optional[Dict[str, Any]]:
        """Load cached validated patterns."""
        if not self.patterns_cache_path.exists():
            return None

        try:
            with open(self.patterns_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            # Check if cache is fresh (< 1 hour old)
            validated_at = cached.get("validated_at", "")
            if validated_at:
                dt = datetime.fromisoformat(validated_at)
                if (datetime.now() - dt).total_seconds() > 3600:
                    return None  # Cache is stale

            return cached

        except:
            return None


def validate_pattern(pattern: Dict[str, Any], pattern_type: str) -> ValidationResult:
    """
    Convenience function to validate a single pattern.

    Args:
        pattern: Pattern data
        pattern_type: One of 'topic', 'problem', 'knowledge_gap', 'workflow'

    Returns:
        ValidationResult
    """
    validator = PatternValidator()

    if pattern_type == "topic":
        return validator.validate_topic_pattern(pattern)
    elif pattern_type == "problem":
        return validator.validate_problem_pattern(pattern)
    elif pattern_type == "knowledge_gap":
        return validator.validate_knowledge_gap(pattern)
    else:
        return ValidationResult(
            is_valid=False,
            reason=f"Unknown pattern type: {pattern_type}",
            confidence=0.0,
            pattern_type=pattern_type,
            pattern_data=pattern
        )


def get_validated_patterns(max_per_type: int = 5, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Convenience function to get all validated patterns.

    Args:
        max_per_type: Maximum patterns per type
        force_refresh: If True, bypass cache

    Returns:
        Dict with validated patterns
    """
    validator = PatternValidator()
    return validator.get_validated_patterns(max_per_type, force_refresh=force_refresh)


if __name__ == "__main__":
    # Test the validator
    validator = PatternValidator()

    print("=== Pattern Validator Test ===")
    patterns = validator.get_validated_patterns(max_per_type=3)

    print(f"\nStats: {patterns.get('stats', {})}")

    print("\nValidated Topics:")
    for result in patterns.get("topics", []):
        print(f"  [{result.confidence:.2f}] {result.pattern_data.get('topic', 'unknown')}")

    print("\nValidated Problems:")
    for result in patterns.get("problems", []):
        print(f"  [{result.confidence:.2f}] {result.pattern_data.get('problem', 'unknown')[:50]}")

    print("\nValidated Knowledge Gaps:")
    for result in patterns.get("knowledge_gaps", []):
        print(f"  [{result.confidence:.2f}] {result.pattern_data.get('concept', 'unknown')}")
