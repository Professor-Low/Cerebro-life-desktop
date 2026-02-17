"""
Consistency Checker - Detect contradictions in personality traits.

Part of Phase 6 Enhancement in the All-Knowing Brain PRD.
Identifies conflicting traits and suggests resolutions.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict


# Known contradictory trait pairs
CONTRADICTORY_PAIRS = [
    # Communication contradictions
    ("comm_concise_responses", "comm_detailed_explanations"),
    ("comm_skip_basics", "comm_detailed_explanations"),

    # Workflow contradictions
    ("work_confirm_before_action", "work_autonomous_action"),
    ("work_action_oriented", "work_confirm_before_action"),

    # Technical contradictions
    ("tech_prefer_python", "tech_prefer_javascript"),  # Mild contradiction
]

# Semantic similarity groups (traits that shouldn't both be "dislikes")
SEMANTIC_GROUPS = {
    "explanation_level": [
        "comm_concise_responses",
        "comm_detailed_explanations",
        "comm_skip_basics",
    ],
    "autonomy_level": [
        "work_confirm_before_action",
        "work_autonomous_action",
        "work_action_oriented",
    ],
}


class Contradiction:
    """Represents a contradiction between traits."""

    def __init__(self, trait1: str, trait2: str, severity: str,
                 description: str, suggested_resolution: str = None):
        self.trait1 = trait1
        self.trait2 = trait2
        self.severity = severity  # "high", "medium", "low"
        self.description = description
        self.suggested_resolution = suggested_resolution
        self.detected_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trait1": self.trait1,
            "trait2": self.trait2,
            "severity": self.severity,
            "description": self.description,
            "suggested_resolution": self.suggested_resolution,
            "detected_at": self.detected_at
        }


class ConsistencyChecker:
    """
    Checks for consistency in personality traits.
    Detects contradictions and suggests resolutions.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.personality_dir = self.base_path / "personality"
        self.contradictions_file = self.personality_dir / "contradictions.json"

        # Lazy load trait tracker
        self._trait_tracker = None

    @property
    def trait_tracker(self):
        """Lazy load trait tracker."""
        if self._trait_tracker is None:
            try:
                from .trait_tracker import TraitTracker
            except ImportError:
                from trait_tracker import TraitTracker
            self._trait_tracker = TraitTracker(str(self.base_path))
        return self._trait_tracker

    def check_all_traits(self) -> Dict[str, Any]:
        """
        Check all traits for contradictions.

        Returns:
            Dict with contradictions and analysis
        """
        all_traits = self.trait_tracker.get_all_traits()
        contradictions = []

        # Get all traits as a flat list
        flat_traits = {}
        for category, data in all_traits.get("by_category", {}).items():
            for trait in data.get("prefers", []):
                flat_traits[trait["name"]] = trait
            for trait in data.get("dislikes", []):
                flat_traits[trait["name"]] = trait

        # Check known contradictory pairs
        for trait1_name, trait2_name in CONTRADICTORY_PAIRS:
            if trait1_name in flat_traits and trait2_name in flat_traits:
                trait1 = flat_traits[trait1_name]
                trait2 = flat_traits[trait2_name]

                # Only flag if both are strong enough
                if trait1["strength"] >= 0.4 and trait2["strength"] >= 0.4:
                    # Determine severity based on strength
                    avg_strength = (trait1["strength"] + trait2["strength"]) / 2
                    if avg_strength >= 0.7:
                        severity = "high"
                    elif avg_strength >= 0.5:
                        severity = "medium"
                    else:
                        severity = "low"

                    contradiction = Contradiction(
                        trait1=trait1_name,
                        trait2=trait2_name,
                        severity=severity,
                        description=f"'{trait1['value'][:50]}' conflicts with '{trait2['value'][:50]}'",
                        suggested_resolution=self._suggest_resolution(trait1, trait2)
                    )
                    contradictions.append(contradiction)

        # Check semantic groups for "both disliked" contradictions
        for group_name, traits in SEMANTIC_GROUPS.items():
            dislikes_in_group = []
            for trait_name in traits:
                if trait_name in flat_traits:
                    trait = flat_traits[trait_name]
                    if not trait["positive"] and trait["strength"] >= 0.5:
                        dislikes_in_group.append(trait)

            if len(dislikes_in_group) >= 2:
                # Multiple dislikes in same semantic group
                contradiction = Contradiction(
                    trait1=dislikes_in_group[0]["name"],
                    trait2=dislikes_in_group[1]["name"],
                    severity="medium",
                    description=f"Multiple conflicting dislikes in {group_name} group",
                    suggested_resolution=f"Review {group_name} preferences - can't dislike all options"
                )
                contradictions.append(contradiction)

        # Save contradictions
        self._save_contradictions(contradictions)

        return {
            "total_traits": len(flat_traits),
            "contradictions_found": len(contradictions),
            "contradictions": [c.to_dict() for c in contradictions],
            "consistency_score": self._calculate_consistency_score(flat_traits, contradictions),
            "checked_at": datetime.now().isoformat()
        }

    def _suggest_resolution(self, trait1: Dict, trait2: Dict) -> str:
        """Suggest how to resolve a contradiction."""
        # Prefer the stronger trait
        if trait1["strength"] > trait2["strength"] + 0.2:
            return f"Keep '{trait1['name']}' (stronger) and weaken/remove '{trait2['name']}'"
        elif trait2["strength"] > trait1["strength"] + 0.2:
            return f"Keep '{trait2['name']}' (stronger) and weaken/remove '{trait1['name']}'"

        # Prefer the more recently observed
        if trait1["last_observed"] > trait2["last_observed"]:
            return f"Keep '{trait1['name']}' (more recent) and re-evaluate '{trait2['name']}'"
        elif trait2["last_observed"] > trait1["last_observed"]:
            return f"Keep '{trait2['name']}' (more recent) and re-evaluate '{trait1['name']}'"

        # Default: ask user
        return "Conflicting traits of similar strength - clarify preference with user"

    def _calculate_consistency_score(self, traits: Dict,
                                      contradictions: List[Contradiction]) -> float:
        """
        Calculate an overall consistency score.

        Returns:
            Score from 0.0 (very inconsistent) to 1.0 (fully consistent)
        """
        if not traits:
            return 1.0  # No traits = no contradictions

        # Base score starts at 1.0
        score = 1.0

        # Deduct for each contradiction based on severity
        for contradiction in contradictions:
            if contradiction.severity == "high":
                score -= 0.15
            elif contradiction.severity == "medium":
                score -= 0.1
            else:
                score -= 0.05

        return max(0.0, min(1.0, score))

    def _save_contradictions(self, contradictions: List[Contradiction]) -> None:
        """Save detected contradictions to file."""
        try:
            self.personality_dir.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "1.0",
                "checked_at": datetime.now().isoformat(),
                "contradictions": [c.to_dict() for c in contradictions]
            }

            with open(self.contradictions_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception:
            pass

    def check_new_trait(self, trait_name: str, trait_value: str,
                        trait_positive: bool, trait_category: str) -> Dict[str, Any]:
        """
        Check if a new trait would create contradictions.

        Args:
            trait_name: Name of the potential new trait
            trait_value: Value of the trait
            trait_positive: Whether it's a preference (True) or dislike (False)
            trait_category: Category of the trait

        Returns:
            Dict with potential contradictions
        """
        potential_contradictions = []

        all_traits = self.trait_tracker.get_all_traits()
        flat_traits = {}
        for category, data in all_traits.get("by_category", {}).items():
            for trait in data.get("prefers", []) + data.get("dislikes", []):
                flat_traits[trait["name"]] = trait

        # Check against known contradictory pairs
        for pair in CONTRADICTORY_PAIRS:
            other_trait_name = None
            if trait_name == pair[0]:
                other_trait_name = pair[1]
            elif trait_name == pair[1]:
                other_trait_name = pair[0]

            if other_trait_name and other_trait_name in flat_traits:
                other_trait = flat_traits[other_trait_name]
                potential_contradictions.append({
                    "conflicting_trait": other_trait_name,
                    "conflicting_value": other_trait["value"],
                    "severity": "medium",
                    "message": f"Would conflict with existing trait '{other_trait['value'][:50]}'"
                })

        # Check semantic groups
        for group_name, group_traits in SEMANTIC_GROUPS.items():
            if trait_name in group_traits:
                for other_name in group_traits:
                    if other_name != trait_name and other_name in flat_traits:
                        other_trait = flat_traits[other_name]
                        # Flag if both would be dislikes
                        if not trait_positive and not other_trait["positive"]:
                            potential_contradictions.append({
                                "conflicting_trait": other_name,
                                "conflicting_value": other_trait["value"],
                                "severity": "low",
                                "message": f"Multiple dislikes in {group_name} group"
                            })

        return {
            "trait_name": trait_name,
            "would_conflict": len(potential_contradictions) > 0,
            "potential_contradictions": potential_contradictions,
            "recommendation": "Safe to add" if not potential_contradictions else "Review conflicts first"
        }

    def get_resolution_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get suggestions for resolving current contradictions.

        Returns:
            List of actionable suggestions
        """
        result = self.check_all_traits()
        suggestions = []

        for contradiction in result.get("contradictions", []):
            suggestion = {
                "issue": contradiction["description"],
                "severity": contradiction["severity"],
                "resolution": contradiction["suggested_resolution"],
                "traits_involved": [contradiction["trait1"], contradiction["trait2"]],
                "action_type": self._determine_action_type(contradiction)
            }
            suggestions.append(suggestion)

        # Sort by severity (high first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: severity_order.get(x["severity"], 3))

        return suggestions

    def _determine_action_type(self, contradiction: Dict) -> str:
        """Determine what type of action is needed."""
        if "weaken" in contradiction.get("suggested_resolution", "").lower():
            return "auto_resolvable"
        elif "clarify" in contradiction.get("suggested_resolution", "").lower():
            return "needs_user_input"
        else:
            return "review_needed"

    def auto_resolve_contradictions(self, max_resolutions: int = 3) -> Dict[str, Any]:
        """
        Automatically resolve obvious contradictions.
        Only resolves when one trait is clearly stronger/newer.

        Args:
            max_resolutions: Maximum number of auto-resolutions

        Returns:
            Result of auto-resolution
        """
        result = self.check_all_traits()
        resolutions = []

        for contradiction in result.get("contradictions", [])[:max_resolutions]:
            trait1_name = contradiction["trait1"]
            trait2_name = contradiction["trait2"]

            trait1 = self.trait_tracker.get_trait(trait1_name)
            trait2 = self.trait_tracker.get_trait(trait2_name)

            if not trait1 or not trait2:
                continue

            # Auto-resolve if strength difference is significant
            strength_diff = abs(trait1["strength"] - trait2["strength"])
            if strength_diff >= 0.3:
                weaker_trait = trait1_name if trait1["strength"] < trait2["strength"] else trait2_name

                # Weaken the weaker trait
                weaken_result = self.trait_tracker.weaken_trait(
                    weaker_trait, 0.2, "auto_contradiction_resolution"
                )

                if weaken_result:
                    resolutions.append({
                        "contradiction": contradiction["description"],
                        "action": f"Weakened '{weaker_trait}'",
                        "result": weaken_result
                    })

        return {
            "contradictions_found": result.get("contradictions_found", 0),
            "auto_resolved": len(resolutions),
            "resolutions": resolutions,
            "remaining_contradictions": result.get("contradictions_found", 0) - len(resolutions),
            "timestamp": datetime.now().isoformat()
        }


# Convenience functions

def check_trait_consistency() -> Dict[str, Any]:
    """Check all traits for consistency."""
    checker = ConsistencyChecker()
    return checker.check_all_traits()


def get_contradictions() -> List[Dict[str, Any]]:
    """Get current contradictions."""
    checker = ConsistencyChecker()
    result = checker.check_all_traits()
    return result.get("contradictions", [])


if __name__ == "__main__":
    # Test the consistency checker
    print("=== Consistency Checker Test ===\n")

    checker = ConsistencyChecker()

    # Check all traits
    print("1. Checking trait consistency...")
    result = checker.check_all_traits()
    print(f"   Total traits: {result['total_traits']}")
    print(f"   Contradictions found: {result['contradictions_found']}")
    print(f"   Consistency score: {result['consistency_score']:.2f}")

    if result["contradictions"]:
        print("\n   Contradictions:")
        for c in result["contradictions"]:
            print(f"   - [{c['severity']}] {c['description']}")

    # Get resolution suggestions
    print("\n2. Getting resolution suggestions...")
    suggestions = checker.get_resolution_suggestions()
    if suggestions:
        for s in suggestions[:3]:
            print(f"   - {s['issue']}")
            print(f"     Resolution: {s['resolution']}")
    else:
        print("   No resolutions needed")

    # Test checking a new trait
    print("\n3. Checking potential new trait...")
    check_result = checker.check_new_trait(
        trait_name="comm_detailed_explanations",
        trait_value="Prefers detailed explanations",
        trait_positive=True,
        trait_category="communication"
    )
    print(f"   Would conflict: {check_result['would_conflict']}")
    if check_result["potential_contradictions"]:
        for pc in check_result["potential_contradictions"]:
            print(f"   - {pc['message']}")

    print("\n=== Test Complete ===")
