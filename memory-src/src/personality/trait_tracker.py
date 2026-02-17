"""
Trait Tracker - Track personality traits and their changes over time.

Part of Phase 6 Enhancement in the All-Knowing Brain PRD.
Tracks traits across categories and records how they evolve.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum


class TraitCategory(Enum):
    """Categories of personality traits."""
    COMMUNICATION = "communication"      # How user prefers to communicate
    TECHNICAL = "technical"              # Technical preferences
    WORKFLOW = "workflow"                # Work style preferences
    EMOTIONAL = "emotional"              # Emotional patterns
    LEARNING = "learning"                # How user learns/problem-solves


class TraitStrength(Enum):
    """Strength/confidence of a trait."""
    INFERRED = "inferred"       # Inferred from behavior (0.3-0.5)
    OBSERVED = "observed"       # Observed multiple times (0.5-0.7)
    STATED = "stated"           # User explicitly stated (0.7-0.9)
    CONFIRMED = "confirmed"     # Repeatedly confirmed (0.9-1.0)


@dataclass
class Trait:
    """A personality trait."""
    name: str
    category: str
    value: str
    strength: float  # 0.0 to 1.0
    positive: bool   # True = prefers, False = dislikes
    source: str      # How this trait was learned
    first_observed: str
    last_observed: str
    observation_count: int = 1
    related_corrections: List[str] = field(default_factory=list)  # Correction IDs
    notes: Optional[str] = None


@dataclass
class TraitChange:
    """A change in a trait over time."""
    trait_name: str
    category: str
    change_type: str  # "added", "removed", "strengthened", "weakened", "modified"
    old_value: Optional[str]
    new_value: str
    old_strength: Optional[float]
    new_strength: float
    timestamp: str
    reason: str
    source: str  # What triggered this change (correction_id, conversation_id, etc.)


class TraitTracker:
    """
    Tracks personality traits and their evolution over time.
    Maintains history of changes for analysis.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.personality_dir = self.base_path / "personality"
        self.traits_file = self.personality_dir / "traits.json"
        self.history_file = self.personality_dir / "trait_history.json"

        # Ensure directory exists
        self.personality_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.traits: Dict[str, Trait] = {}
        self.history: List[TraitChange] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load traits and history from files."""
        # Load traits
        if self.traits_file.exists():
            try:
                with open(self.traits_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for name, trait_data in data.get("traits", {}).items():
                    self.traits[name] = Trait(**trait_data)
            except Exception:
                pass

        # Load history
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for change_data in data.get("changes", []):
                    self.history.append(TraitChange(**change_data))
            except Exception:
                pass

        # Initialize from existing preferences if empty
        if not self.traits:
            self._initialize_from_existing()

    def _initialize_from_existing(self) -> None:
        """Initialize traits from existing preference data."""
        now = datetime.now().isoformat()

        # Try to load from user profile
        profile_file = self.base_path / "user" / "profile.json"
        if profile_file.exists():
            try:
                with open(profile_file, "r", encoding="utf-8") as f:
                    profile = json.load(f)

                # Extract communication style
                comm_style = profile.get("communication_style", {})
                for pref in comm_style.get("prefers", []):
                    self._add_trait(
                        name=f"comm_prefer_{self._slugify(pref[:30])}",
                        category=TraitCategory.COMMUNICATION.value,
                        value=pref,
                        strength=0.7,
                        positive=True,
                        source="profile_import",
                        timestamp=now
                    )
                for pref in comm_style.get("dislikes", []):
                    self._add_trait(
                        name=f"comm_dislike_{self._slugify(pref[:30])}",
                        category=TraitCategory.COMMUNICATION.value,
                        value=pref,
                        strength=0.7,
                        positive=False,
                        source="profile_import",
                        timestamp=now
                    )

                # Extract personal preferences
                prefs = profile.get("preferences", {})
                for pref in prefs.get("personal", []):
                    self._add_trait(
                        name=f"personal_{self._slugify(pref[:30])}",
                        category=TraitCategory.WORKFLOW.value,
                        value=pref,
                        strength=0.7,
                        positive=True,
                        source="profile_import",
                        timestamp=now
                    )
                for pref in prefs.get("dislikes", []):
                    self._add_trait(
                        name=f"dislike_{self._slugify(pref[:30])}",
                        category=TraitCategory.WORKFLOW.value,
                        value=pref,
                        strength=0.7,
                        positive=False,
                        source="profile_import",
                        timestamp=now
                    )

                # Extract emotional patterns
                emotions = profile.get("emotional_patterns", {})
                for frustration in emotions.get("frustrations", []):
                    trigger = frustration.get("trigger", "")
                    if trigger:
                        self._add_trait(
                            name=f"frustration_{self._slugify(trigger[:30])}",
                            category=TraitCategory.EMOTIONAL.value,
                            value=trigger,
                            strength=0.6,
                            positive=False,
                            source="profile_import",
                            timestamp=now,
                            notes=frustration.get("resolution")
                        )

            except Exception:
                pass

        # Try to load from preference manager
        prefs_file = self.base_path / "preferences" / "user_preferences.json"
        if prefs_file.exists():
            try:
                with open(prefs_file, "r", encoding="utf-8") as f:
                    prefs = json.load(f)

                for category in ["communication_style", "workflow", "technical"]:
                    cat_data = prefs.get(category, {})
                    trait_category = {
                        "communication_style": TraitCategory.COMMUNICATION.value,
                        "workflow": TraitCategory.WORKFLOW.value,
                        "technical": TraitCategory.TECHNICAL.value
                    }.get(category, TraitCategory.WORKFLOW.value)

                    for pref in cat_data.get("prefers", []):
                        self._add_trait(
                            name=f"{category[:4]}_prefer_{self._slugify(pref[:30])}",
                            category=trait_category,
                            value=pref,
                            strength=0.6,
                            positive=True,
                            source="preferences_import",
                            timestamp=now
                        )
                    for pref in cat_data.get("dislikes", []):
                        self._add_trait(
                            name=f"{category[:4]}_dislike_{self._slugify(pref[:30])}",
                            category=trait_category,
                            value=pref,
                            strength=0.6,
                            positive=False,
                            source="preferences_import",
                            timestamp=now
                        )

            except Exception:
                pass

        self._save_data()

    def _slugify(self, text: str) -> str:
        """Convert text to a slug-like identifier."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9]+', '_', text)
        text = text.strip('_')
        return text[:30]

    def _add_trait(self, name: str, category: str, value: str, strength: float,
                   positive: bool, source: str, timestamp: str,
                   notes: str = None) -> Trait:
        """Add a new trait (internal method)."""
        trait = Trait(
            name=name,
            category=category,
            value=value,
            strength=strength,
            positive=positive,
            source=source,
            first_observed=timestamp,
            last_observed=timestamp,
            observation_count=1,
            related_corrections=[],
            notes=notes
        )
        self.traits[name] = trait
        return trait

    def _save_data(self) -> None:
        """Save traits and history to files."""
        try:
            # Save traits
            traits_data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "traits": {name: asdict(trait) for name, trait in self.traits.items()}
            }
            with open(self.traits_file, "w", encoding="utf-8") as f:
                json.dump(traits_data, f, indent=2, ensure_ascii=False)

            # Save history (keep last 500 changes)
            history_data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "changes": [asdict(change) for change in self.history[-500:]]
            }
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error saving trait data: {e}")

    def add_trait(self, name: str, category: str, value: str,
                  strength: float = 0.5, positive: bool = True,
                  source: str = "manual", notes: str = None) -> Dict[str, Any]:
        """
        Add or update a personality trait.

        Args:
            name: Unique identifier for the trait
            category: Category (communication, technical, workflow, emotional, learning)
            value: The trait value/description
            strength: Confidence level 0.0-1.0
            positive: True for preferences, False for dislikes
            source: How this was learned

        Returns:
            Result dict with trait info
        """
        now = datetime.now().isoformat()

        if name in self.traits:
            # Update existing trait
            old_trait = self.traits[name]
            change_type = "modified" if old_trait.value != value else "strengthened"

            # Record change
            change = TraitChange(
                trait_name=name,
                category=category,
                change_type=change_type,
                old_value=old_trait.value,
                new_value=value,
                old_strength=old_trait.strength,
                new_strength=strength,
                timestamp=now,
                reason=f"Updated from {source}",
                source=source
            )
            self.history.append(change)

            # Update trait
            old_trait.value = value
            old_trait.strength = min(1.0, strength)
            old_trait.last_observed = now
            old_trait.observation_count += 1
            if notes:
                old_trait.notes = notes

            self._save_data()
            return {
                "action": "updated",
                "trait": asdict(old_trait),
                "change": asdict(change)
            }
        else:
            # Add new trait
            trait = self._add_trait(
                name=name,
                category=category,
                value=value,
                strength=strength,
                positive=positive,
                source=source,
                timestamp=now,
                notes=notes
            )

            # Record change
            change = TraitChange(
                trait_name=name,
                category=category,
                change_type="added",
                old_value=None,
                new_value=value,
                old_strength=None,
                new_strength=strength,
                timestamp=now,
                reason=f"New trait from {source}",
                source=source
            )
            self.history.append(change)

            self._save_data()
            return {
                "action": "added",
                "trait": asdict(trait),
                "change": asdict(change)
            }

    def strengthen_trait(self, name: str, amount: float = 0.1,
                         source: str = "observation") -> Optional[Dict[str, Any]]:
        """
        Strengthen an existing trait's confidence.

        Args:
            name: Trait name
            amount: How much to strengthen (default 0.1)
            source: What triggered the strengthening

        Returns:
            Result dict or None if trait not found
        """
        if name not in self.traits:
            return None

        trait = self.traits[name]
        old_strength = trait.strength
        new_strength = min(1.0, trait.strength + amount)

        trait.strength = new_strength
        trait.last_observed = datetime.now().isoformat()
        trait.observation_count += 1

        # Record change
        change = TraitChange(
            trait_name=name,
            category=trait.category,
            change_type="strengthened",
            old_value=trait.value,
            new_value=trait.value,
            old_strength=old_strength,
            new_strength=new_strength,
            timestamp=datetime.now().isoformat(),
            reason=f"Confirmed by {source}",
            source=source
        )
        self.history.append(change)

        self._save_data()
        return {
            "action": "strengthened",
            "trait": name,
            "old_strength": old_strength,
            "new_strength": new_strength,
            "observation_count": trait.observation_count
        }

    def weaken_trait(self, name: str, amount: float = 0.1,
                     source: str = "contradiction") -> Optional[Dict[str, Any]]:
        """
        Weaken an existing trait's confidence.

        Args:
            name: Trait name
            amount: How much to weaken (default 0.1)
            source: What triggered the weakening

        Returns:
            Result dict or None if trait not found
        """
        if name not in self.traits:
            return None

        trait = self.traits[name]
        old_strength = trait.strength
        new_strength = max(0.0, trait.strength - amount)

        # If strength drops too low, mark as removed
        if new_strength < 0.1:
            change_type = "removed"
            del self.traits[name]
        else:
            change_type = "weakened"
            trait.strength = new_strength
            trait.last_observed = datetime.now().isoformat()

        # Record change
        change = TraitChange(
            trait_name=name,
            category=trait.category,
            change_type=change_type,
            old_value=trait.value,
            new_value=trait.value if change_type != "removed" else None,
            old_strength=old_strength,
            new_strength=new_strength,
            timestamp=datetime.now().isoformat(),
            reason=f"Weakened by {source}",
            source=source
        )
        self.history.append(change)

        self._save_data()
        return {
            "action": change_type,
            "trait": name,
            "old_strength": old_strength,
            "new_strength": new_strength
        }

    def link_correction(self, trait_name: str, correction_id: str) -> bool:
        """Link a correction to a trait."""
        if trait_name not in self.traits:
            return False

        trait = self.traits[trait_name]
        if correction_id not in trait.related_corrections:
            trait.related_corrections.append(correction_id)
            self._save_data()
        return True

    def get_trait(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific trait."""
        if name in self.traits:
            return asdict(self.traits[name])
        return None

    def get_traits_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all traits in a category."""
        return [
            asdict(trait)
            for trait in self.traits.values()
            if trait.category == category
        ]

    def get_all_traits(self) -> Dict[str, Any]:
        """Get all traits organized by category."""
        by_category = {}
        for trait in self.traits.values():
            if trait.category not in by_category:
                by_category[trait.category] = {"prefers": [], "dislikes": []}

            trait_dict = asdict(trait)
            if trait.positive:
                by_category[trait.category]["prefers"].append(trait_dict)
            else:
                by_category[trait.category]["dislikes"].append(trait_dict)

        return {
            "total_traits": len(self.traits),
            "by_category": by_category,
            "strongest_traits": self._get_strongest_traits(5),
            "updated_at": datetime.now().isoformat()
        }

    def _get_strongest_traits(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the strongest/most confident traits."""
        sorted_traits = sorted(
            self.traits.values(),
            key=lambda t: t.strength,
            reverse=True
        )
        return [asdict(t) for t in sorted_traits[:limit]]

    def get_trait_history(self, trait_name: str = None,
                          days: int = 30) -> List[Dict[str, Any]]:
        """
        Get history of trait changes.

        Args:
            trait_name: Optional filter by specific trait
            days: Number of days to look back

        Returns:
            List of changes
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        changes = [
            asdict(c)
            for c in self.history
            if c.timestamp > cutoff
        ]

        if trait_name:
            changes = [c for c in changes if c["trait_name"] == trait_name]

        return changes

    def get_evolution_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get a summary of how traits have evolved.

        Args:
            days: Number of days to analyze

        Returns:
            Summary of evolution
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        recent_changes = [c for c in self.history if c.timestamp > cutoff]

        # Count by type
        by_type = {
            "added": 0,
            "removed": 0,
            "strengthened": 0,
            "weakened": 0,
            "modified": 0
        }
        for change in recent_changes:
            if change.change_type in by_type:
                by_type[change.change_type] += 1

        # Find most evolved traits
        trait_changes = {}
        for change in recent_changes:
            if change.trait_name not in trait_changes:
                trait_changes[change.trait_name] = 0
            trait_changes[change.trait_name] += 1

        most_evolved = sorted(
            trait_changes.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "period_days": days,
            "total_changes": len(recent_changes),
            "changes_by_type": by_type,
            "most_evolved_traits": most_evolved,
            "net_traits_added": by_type["added"] - by_type["removed"],
            "average_changes_per_day": len(recent_changes) / max(days, 1)
        }


# Convenience functions

def get_trait_history(trait_name: str = None, days: int = 30) -> List[Dict[str, Any]]:
    """Get history of trait changes."""
    tracker = TraitTracker()
    return tracker.get_trait_history(trait_name, days)


def record_trait_change(name: str, category: str, value: str,
                        strength: float = 0.5, positive: bool = True,
                        source: str = "manual") -> Dict[str, Any]:
    """Record a trait change."""
    tracker = TraitTracker()
    return tracker.add_trait(name, category, value, strength, positive, source)


if __name__ == "__main__":
    # Test the trait tracker
    print("=== Trait Tracker Test ===\n")

    tracker = TraitTracker()

    # Show loaded traits
    all_traits = tracker.get_all_traits()
    print(f"Total traits loaded: {all_traits['total_traits']}")

    print("\nTraits by category:")
    for cat, data in all_traits.get("by_category", {}).items():
        print(f"  {cat}: {len(data.get('prefers', []))} prefers, {len(data.get('dislikes', []))} dislikes")

    print("\nStrongest traits:")
    for trait in all_traits.get("strongest_traits", []):
        print(f"  - {trait['name']}: {trait['value'][:50]}... (strength={trait['strength']:.2f})")

    # Test adding a trait
    print("\n--- Testing trait operations ---")
    result = tracker.add_trait(
        name="test_direct_communication",
        category="communication",
        value="Prefers direct, no-fluff communication",
        strength=0.8,
        positive=True,
        source="test"
    )
    print(f"Added trait: {result['action']}")

    # Test strengthening
    result = tracker.strengthen_trait("test_direct_communication", 0.1, "test_observation")
    if result:
        print(f"Strengthened: {result['old_strength']:.2f} -> {result['new_strength']:.2f}")

    # Get evolution summary
    print("\nEvolution summary (30 days):")
    summary = tracker.get_evolution_summary(30)
    print(f"  Total changes: {summary['total_changes']}")
    print(f"  Changes by type: {summary['changes_by_type']}")

    print("\n=== Test Complete ===")
