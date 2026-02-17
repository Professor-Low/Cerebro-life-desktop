"""
Patterns Module - Phase 4 of All-Knowing Brain PRD.

Provides:
- PatternValidator: Validate patterns before acting on them
- PatternApplier: Apply validated patterns to context injection
"""

from .validator import PatternValidator, validate_pattern, get_validated_patterns
from .applier import PatternApplier, apply_patterns_to_context

__all__ = [
    # Pattern validation
    "PatternValidator",
    "validate_pattern",
    "get_validated_patterns",
    # Pattern application
    "PatternApplier",
    "apply_patterns_to_context",
]
