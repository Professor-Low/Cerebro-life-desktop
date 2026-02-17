"""
Personality Evolution Module - Phase 6 Enhancement

Tracks how user preferences and traits evolve over time.
Provides consistency checking and feedback-driven evolution.
"""

from .trait_tracker import (
    TraitTracker,
    Trait,
    TraitChange,
    get_trait_history,
    record_trait_change
)

from .evolution_engine import (
    PersonalityEvolutionEngine,
    evolve_from_correction,
    evolve_from_feedback,
    get_evolution_summary
)

from .consistency_checker import (
    ConsistencyChecker,
    check_trait_consistency,
    get_contradictions
)

__all__ = [
    # Trait tracking
    "TraitTracker",
    "Trait",
    "TraitChange",
    "get_trait_history",
    "record_trait_change",
    # Evolution
    "PersonalityEvolutionEngine",
    "evolve_from_correction",
    "evolve_from_feedback",
    "get_evolution_summary",
    # Consistency
    "ConsistencyChecker",
    "check_trait_consistency",
    "get_contradictions",
]
