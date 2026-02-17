"""
Improvement Engine - Generate Proposals from AI Memory Patterns

Analyzes AI Memory patterns and learnings to generate improvement proposals:
- Read quick_facts.json for promoted patterns
- Convert patterns to improvement opportunities
- Filter by confidence threshold
- Create proposals from opportunities
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class OpportunityType(str, Enum):
    """Types of improvement opportunities."""
    HOOK_OPTIMIZATION = "hook_optimization"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE = "performance"
    CODE_PATTERN = "code_pattern"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    FEATURE = "feature"


class OpportunityPriority(str, Enum):
    """Priority levels for opportunities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ImprovementOpportunity:
    """Represents a potential improvement opportunity."""
    id: str
    type: OpportunityType
    title: str
    description: str
    priority: OpportunityPriority
    confidence: float  # 0.0 to 1.0
    source_pattern: str  # The pattern that generated this
    affected_files: List[str] = field(default_factory=list)
    suggested_changes: Dict[str, str] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    requires_approval: bool = True

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "confidence": self.confidence,
            "source_pattern": self.source_pattern,
            "affected_files": self.affected_files,
            "suggested_changes": self.suggested_changes,
            "evidence": self.evidence,
            "created_at": self.created_at,
            "requires_approval": self.requires_approval
        }


class ImprovementEngine:
    """
    Analyzes AI Memory patterns to generate improvement opportunities.

    Reads promoted patterns, learnings, and corrections from AI Memory
    to identify potential improvements to the Cerebro system.
    """

    # Minimum confidence to generate an opportunity
    MIN_CONFIDENCE = 0.6

    # Pattern keywords that map to opportunity types
    PATTERN_TYPE_MAPPING = {
        "hook": OpportunityType.HOOK_OPTIMIZATION,
        "error": OpportunityType.ERROR_HANDLING,
        "exception": OpportunityType.ERROR_HANDLING,
        "slow": OpportunityType.PERFORMANCE,
        "performance": OpportunityType.PERFORMANCE,
        "latency": OpportunityType.PERFORMANCE,
        "pattern": OpportunityType.CODE_PATTERN,
        "config": OpportunityType.CONFIGURATION,
        "setting": OpportunityType.CONFIGURATION,
        "docs": OpportunityType.DOCUMENTATION,
        "feature": OpportunityType.FEATURE,
    }

    def __init__(self, memory_path: Path, repo_path: Path = None):
        """
        Initialize the improvement engine.

        Args:
            memory_path: Path to AI Memory (configured via AI_MEMORY_PATH env var)
            repo_path: Path to Cerebro repository
        """
        self.memory_path = Path(memory_path)
        self.repo_path = repo_path or self.memory_path / "projects" / "digital_companion" / "cerebro"
        self.quick_facts_path = self.memory_path / "quick_facts.json"
        self._opportunities: List[ImprovementOpportunity] = []

    def _generate_id(self, title: str) -> str:
        """Generate unique opportunity ID."""
        ts = datetime.now().isoformat()
        return f"opp_{hashlib.sha256(f'{title}_{ts}'.encode()).hexdigest()[:10]}"

    def _load_quick_facts(self) -> Dict:
        """Load quick_facts.json."""
        if not self.quick_facts_path.exists():
            return {}
        try:
            return json.loads(self.quick_facts_path.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"Error loading quick_facts: {e}")
            return {}

    def _detect_type(self, pattern_text: str) -> OpportunityType:
        """
        Detect opportunity type from pattern text.

        Args:
            pattern_text: Text describing the pattern

        Returns:
            Best matching OpportunityType
        """
        pattern_lower = pattern_text.lower()

        for keyword, opp_type in self.PATTERN_TYPE_MAPPING.items():
            if keyword in pattern_lower:
                return opp_type

        return OpportunityType.CODE_PATTERN

    def _assess_priority(self, pattern: Dict) -> OpportunityPriority:
        """
        Assess priority based on pattern characteristics.

        Args:
            pattern: Pattern data

        Returns:
            Priority level
        """
        # High frequency patterns get higher priority
        occurrences = pattern.get("occurrences", pattern.get("count", 1))

        if occurrences >= 10:
            return OpportunityPriority.HIGH
        elif occurrences >= 5:
            return OpportunityPriority.MEDIUM
        else:
            return OpportunityPriority.LOW

    def _calculate_confidence(self, pattern: Dict) -> float:
        """
        Calculate confidence score for a pattern.

        Args:
            pattern: Pattern data

        Returns:
            Confidence score 0.0-1.0
        """
        base_confidence = 0.5

        # More occurrences = higher confidence
        occurrences = pattern.get("occurrences", pattern.get("count", 1))
        occurrence_boost = min(occurrences * 0.05, 0.3)

        # If explicitly promoted, higher confidence
        if pattern.get("promoted", False):
            base_confidence = 0.7

        # If it's from corrections, higher confidence (user-validated)
        if pattern.get("from_correction", False):
            base_confidence = 0.8

        return min(base_confidence + occurrence_boost, 0.95)

    async def analyze_patterns(self) -> List[ImprovementOpportunity]:
        """
        Analyze AI Memory patterns and generate opportunities.

        Returns:
            List of improvement opportunities
        """
        opportunities = []
        quick_facts = self._load_quick_facts()

        # Analyze promoted patterns
        promoted = quick_facts.get("promoted_patterns", [])
        for pattern in promoted:
            opp = self._pattern_to_opportunity(pattern)
            if opp and opp.confidence >= self.MIN_CONFIDENCE:
                opportunities.append(opp)

        # Analyze top corrections
        corrections = quick_facts.get("top_corrections", [])
        for correction in corrections:
            opp = self._correction_to_opportunity(correction)
            if opp and opp.confidence >= self.MIN_CONFIDENCE:
                opportunities.append(opp)

        # Analyze recent learnings
        learnings = quick_facts.get("recent_learnings_summary", {})
        if learnings:
            learnings.get("top_keywords", [])
            solutions = learnings.get("recent_solutions", [])
            for solution in solutions:
                opp = self._solution_to_opportunity(solution)
                if opp and opp.confidence >= self.MIN_CONFIDENCE:
                    opportunities.append(opp)

        # Sort by confidence and priority
        opportunities.sort(
            key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.priority.value],
                x.confidence
            ),
            reverse=True
        )

        self._opportunities = opportunities
        return opportunities

    def _pattern_to_opportunity(self, pattern: Dict) -> Optional[ImprovementOpportunity]:
        """
        Convert a promoted pattern to an opportunity.

        Args:
            pattern: Pattern data from quick_facts

        Returns:
            ImprovementOpportunity or None
        """
        title = pattern.get("pattern", pattern.get("title", ""))
        if not title:
            return None

        opp_type = self._detect_type(title)
        priority = self._assess_priority(pattern)
        confidence = self._calculate_confidence(pattern)

        description = pattern.get("description", f"Pattern detected: {title}")

        # Generate suggested changes based on pattern type
        suggested_changes = self._generate_suggested_changes(title, opp_type, pattern)

        return ImprovementOpportunity(
            id=self._generate_id(title),
            type=opp_type,
            title=title[:100],
            description=description[:500],
            priority=priority,
            confidence=confidence,
            source_pattern=title,
            affected_files=pattern.get("files", []),
            suggested_changes=suggested_changes,
            evidence=[f"Seen {pattern.get('occurrences', 1)} times"]
        )

    def _correction_to_opportunity(self, correction: Dict) -> Optional[ImprovementOpportunity]:
        """
        Convert a correction to an opportunity.

        Args:
            correction: Correction data

        Returns:
            ImprovementOpportunity or None
        """
        topic = correction.get("topic", "")
        content = correction.get("content", correction.get("correction", ""))

        if not content:
            return None

        return ImprovementOpportunity(
            id=self._generate_id(content[:50]),
            type=OpportunityType.ERROR_HANDLING,
            title=f"Apply correction: {topic}" if topic else "Apply learned correction",
            description=content[:500],
            priority=OpportunityPriority.HIGH,  # Corrections are high priority
            confidence=0.85,  # User-validated
            source_pattern=content,
            evidence=[f"User correction on topic: {topic}"]
        )

    def _solution_to_opportunity(self, solution: Dict) -> Optional[ImprovementOpportunity]:
        """
        Convert a solution to an opportunity.

        Args:
            solution: Solution data

        Returns:
            ImprovementOpportunity or None
        """
        problem = solution.get("problem", "")
        sol = solution.get("solution", "")

        if not problem or not sol:
            return None

        return ImprovementOpportunity(
            id=self._generate_id(problem[:50]),
            type=self._detect_type(problem),
            title=f"Apply solution: {problem[:60]}",
            description=f"Problem: {problem}\n\nSolution: {sol}",
            priority=OpportunityPriority.MEDIUM,
            confidence=self._calculate_confidence(solution),
            source_pattern=problem,
            evidence=[f"Solution confirmed: {sol[:100]}"]
        )

    def _generate_suggested_changes(
        self,
        title: str,
        opp_type: OpportunityType,
        pattern: Dict
    ) -> Dict[str, str]:
        """
        Generate suggested code changes for an opportunity.

        This is a placeholder - in production, this would use more
        sophisticated analysis to generate actual code changes.

        Args:
            title: Pattern title
            opp_type: Opportunity type
            pattern: Pattern data

        Returns:
            Dict mapping file paths to suggested changes
        """
        # For now, return empty - the actual changes would need
        # more context and potentially Claude Code analysis
        return {}

    async def create_proposal_from_opportunity(
        self,
        opportunity: ImprovementOpportunity,
        self_mod_manager=None
    ) -> Optional[Dict]:
        """
        Create a modification proposal from an opportunity.

        Args:
            opportunity: The opportunity to convert
            self_mod_manager: SelfModificationManager instance

        Returns:
            Proposal dict or None
        """
        if not self_mod_manager:
            return None

        if not opportunity.suggested_changes:
            return None

        # For each suggested change, create a proposal
        for file_path, new_content in opportunity.suggested_changes.items():
            try:
                # Read current content
                full_path = self.repo_path / file_path
                if full_path.exists():
                    old_content = full_path.read_text(encoding='utf-8')
                else:
                    old_content = ""

                # Map opportunity type to modification type
                from self_modification import ModificationType

                mod_type_map = {
                    OpportunityType.HOOK_OPTIMIZATION: ModificationType.HOOK_UPDATE,
                    OpportunityType.CONFIGURATION: ModificationType.CONFIG_TWEAK,
                    OpportunityType.CODE_PATTERN: ModificationType.PROMPT_IMPROVEMENT,
                }

                mod_type = mod_type_map.get(
                    opportunity.type,
                    ModificationType.CONFIG_TWEAK
                )

                proposal = await self_mod_manager.create_proposal(
                    mod_type=mod_type,
                    description=opportunity.title,
                    file_path=str(full_path),
                    old_content=old_content,
                    new_content=new_content,
                    reason=f"Generated from AI Memory pattern: {opportunity.source_pattern}"
                )

                return proposal.to_dict()

            except Exception as e:
                print(f"Error creating proposal: {e}")
                return None

        return None

    def get_opportunities(
        self,
        min_confidence: float = None,
        type_filter: OpportunityType = None,
        priority_filter: OpportunityPriority = None,
        limit: int = 20
    ) -> List[ImprovementOpportunity]:
        """
        Get filtered opportunities.

        Args:
            min_confidence: Minimum confidence threshold
            type_filter: Filter by opportunity type
            priority_filter: Filter by priority
            limit: Maximum number to return

        Returns:
            Filtered list of opportunities
        """
        result = self._opportunities.copy()

        if min_confidence is not None:
            result = [o for o in result if o.confidence >= min_confidence]

        if type_filter:
            result = [o for o in result if o.type == type_filter]

        if priority_filter:
            result = [o for o in result if o.priority == priority_filter]

        return result[:limit]

    def get_opportunity_by_id(self, opp_id: str) -> Optional[ImprovementOpportunity]:
        """Get a specific opportunity by ID."""
        for opp in self._opportunities:
            if opp.id == opp_id:
                return opp
        return None

    async def refresh_opportunities(self) -> int:
        """
        Refresh the opportunities list from AI Memory.

        Returns:
            Number of opportunities found
        """
        opportunities = await self.analyze_patterns()
        return len(opportunities)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of current opportunities.

        Returns:
            Summary dict with counts and breakdown
        """
        by_type = {}
        by_priority = {}

        for opp in self._opportunities:
            by_type[opp.type.value] = by_type.get(opp.type.value, 0) + 1
            by_priority[opp.priority.value] = by_priority.get(opp.priority.value, 0) + 1

        return {
            "total": len(self._opportunities),
            "by_type": by_type,
            "by_priority": by_priority,
            "avg_confidence": (
                sum(o.confidence for o in self._opportunities) / len(self._opportunities)
                if self._opportunities else 0
            ),
            "high_confidence_count": sum(
                1 for o in self._opportunities if o.confidence >= 0.8
            )
        }


# Singleton instance
_engine_instance: Optional[ImprovementEngine] = None


def get_improvement_engine(
    memory_path: Path = None,
    repo_path: Path = None
) -> Optional[ImprovementEngine]:
    """Get or create the improvement engine singleton."""
    global _engine_instance

    if _engine_instance is None and memory_path:
        _engine_instance = ImprovementEngine(memory_path, repo_path)

    return _engine_instance
