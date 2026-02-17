"""
Predictive Interrupt Service - Warn Users Before They Hit Known Failures

This service analyzes user messages/actions and surfaces warnings
BEFORE they execute potentially dangerous or error-prone operations.

Uses:
- MCP Bridge for causal model and predictions
- Pattern matching for known dangerous operations
- Learning history for past failures
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PredictionWarning:
    """Represents a single prediction warning."""
    pattern: str
    warning: str
    confidence: float
    severity: str = "medium"  # low, medium, high, critical
    category: str = "general"
    mechanism: Optional[str] = None
    suggested_alternative: Optional[str] = None


@dataclass
class PredictionResult:
    """Result of analyzing a message for potential issues."""
    has_warnings: bool
    warnings: List[PredictionWarning] = field(default_factory=list)
    suggested_alternatives: List[str] = field(default_factory=list)
    relevant_learnings: List[Dict] = field(default_factory=list)
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "has_warnings": self.has_warnings,
            "warnings": [
                {
                    "pattern": w.pattern,
                    "warning": w.warning,
                    "confidence": w.confidence,
                    "severity": w.severity,
                    "category": w.category,
                    "mechanism": w.mechanism,
                    "suggested_alternative": w.suggested_alternative
                }
                for w in self.warnings
            ],
            "suggested_alternatives": self.suggested_alternatives,
            "relevant_learnings": self.relevant_learnings,
            "analyzed_at": self.analyzed_at
        }


class PredictiveInterruptService:
    """
    Analyzes messages and surfaces warnings before execution.

    Warning Threshold: Only warn if confidence > threshold to avoid noise.
    """

    # Dangerous pattern definitions
    DANGER_PATTERNS = [
        # File System Dangers
        {
            "patterns": ["rm -rf", "rmdir /s", "del /f /s", "remove-item -recurse -force"],
            "warning": "Recursive force delete can permanently remove entire directory trees",
            "confidence": 0.95,
            "severity": "critical",
            "category": "file_system",
            "alternative": "Consider using trash/recycle bin or creating a backup first"
        },
        {
            "patterns": ["rm -r /", "rm -rf /", "del c:\\*"],
            "warning": "This could delete your entire system!",
            "confidence": 0.99,
            "severity": "critical",
            "category": "file_system",
            "alternative": "Never run this command - it will destroy your system"
        },
        {
            "patterns": ["format ", ":format", "diskpart", "fdisk"],
            "warning": "Disk formatting operations are irreversible and will erase all data",
            "confidence": 0.9,
            "severity": "critical",
            "category": "file_system",
            "alternative": "Ensure you have complete backups before proceeding"
        },

        # Git Dangers
        {
            "patterns": ["git push -f", "git push --force"],
            "warning": "Force pushing can overwrite remote history and affect other collaborators",
            "confidence": 0.85,
            "severity": "high",
            "category": "git",
            "alternative": "Consider using --force-with-lease for safer force push"
        },
        {
            "patterns": ["git reset --hard"],
            "warning": "Hard reset discards all uncommitted changes permanently",
            "confidence": 0.9,
            "severity": "high",
            "category": "git",
            "alternative": "Stash changes first with 'git stash' or commit them"
        },
        {
            "patterns": ["git clean -f", "git clean -fd"],
            "warning": "This permanently deletes untracked files",
            "confidence": 0.85,
            "severity": "high",
            "category": "git",
            "alternative": "Run 'git clean -n' first to preview what will be deleted"
        },
        {
            "patterns": ["git checkout ."],
            "warning": "This discards all unstaged changes in tracked files",
            "confidence": 0.8,
            "severity": "medium",
            "category": "git",
            "alternative": "Use 'git stash' to save changes for later"
        },

        # Database Dangers
        {
            "patterns": ["drop table", "drop database", "truncate table"],
            "warning": "This will permanently delete data from the database",
            "confidence": 0.95,
            "severity": "critical",
            "category": "database",
            "alternative": "Create a backup first, and test on staging environment"
        },
        {
            "patterns": ["delete from", "delete *"],
            "warning": "Mass deletion without WHERE clause will remove all rows",
            "confidence": 0.85,
            "severity": "high",
            "category": "database",
            "alternative": "Always use a WHERE clause and test on a small dataset first"
        },

        # System/Security Dangers
        {
            "patterns": ["sudo rm", "sudo del", "runas"],
            "warning": "Running destructive commands with elevated privileges is especially dangerous",
            "confidence": 0.85,
            "severity": "high",
            "category": "security",
            "alternative": "Double-check the command before running with elevated privileges"
        },
        {
            "patterns": ["chmod 777", "chmod -R 777"],
            "warning": "Setting world-writable permissions is a security risk",
            "confidence": 0.8,
            "severity": "medium",
            "category": "security",
            "alternative": "Use more restrictive permissions like 755 or 644"
        },
        {
            "patterns": ["--no-verify", "--skip-hooks"],
            "warning": "Skipping verification/hooks bypasses safety checks",
            "confidence": 0.7,
            "severity": "medium",
            "category": "general",
            "alternative": "Fix the underlying issue rather than bypassing checks"
        },

        # Network Dangers
        {
            "patterns": ["curl | bash", "wget | sh", "curl | sh"],
            "warning": "Piping remote scripts directly to shell is dangerous - you can't review the code first",
            "confidence": 0.9,
            "severity": "high",
            "category": "security",
            "alternative": "Download the script first, review it, then execute"
        },

        # Common Mistakes
        {
            "patterns": ["overwrite", "replace all"],
            "warning": "Mass replacement operations should be reviewed carefully",
            "confidence": 0.6,
            "severity": "low",
            "category": "general",
            "alternative": "Preview changes before applying, or work on a copy"
        },
    ]

    def __init__(self, mcp_bridge=None, warning_threshold: float = 0.6):
        """
        Initialize the predictive interrupt service.

        Args:
            mcp_bridge: MCP Bridge instance for causal/learning lookups
            warning_threshold: Minimum confidence to surface a warning (0.0-1.0)
        """
        self.mcp_bridge = mcp_bridge
        self.warning_threshold = warning_threshold
        self._pattern_cache = {}  # Cache compiled patterns

    async def analyze(self, message: str) -> PredictionResult:
        """
        Analyze a message for potential issues.

        Args:
            message: The user's message or command to analyze

        Returns:
            PredictionResult with any warnings found
        """
        warnings = []
        alternatives = []

        # 1. Check static danger patterns
        pattern_warnings = self._check_danger_patterns(message)
        warnings.extend(pattern_warnings)

        # Collect alternatives from pattern matches
        for w in pattern_warnings:
            if w.suggested_alternative:
                alternatives.append(w.suggested_alternative)

        # 2. Check causal model for known failure patterns (if MCP bridge available)
        if self.mcp_bridge:
            try:
                causal_result = await self.mcp_bridge.predict(
                    "anticipate_failures",
                    context=message
                )
                if causal_result.get("success") and causal_result.get("warnings"):
                    for cw in causal_result["warnings"]:
                        # Avoid duplicates
                        if not any(w.pattern == cw.get("pattern") for w in warnings):
                            if cw.get("confidence", 0.5) >= self.warning_threshold:
                                warnings.append(PredictionWarning(
                                    pattern=cw.get("pattern", "causal"),
                                    warning=cw.get("warning", "Potential issue detected"),
                                    confidence=cw.get("confidence", 0.5),
                                    severity=cw.get("severity", "medium"),
                                    category="causal_model",
                                    mechanism=cw.get("mechanism")
                                ))

                # Get preventive actions
                preventive = await self.mcp_bridge.predict(
                    "preventive_actions",
                    context=message
                )
                if preventive.get("success") and preventive.get("actions"):
                    alternatives.extend(preventive["actions"])

            except Exception as e:
                print(f"[PredictiveInterrupt] Causal lookup failed: {e}")

        # 3. Check learning history for past failures
        relevant_learnings = []
        if self.mcp_bridge:
            try:
                learnings_result = await self.mcp_bridge.learning(
                    "get_antipatterns",
                    context=message
                )
                if learnings_result.get("success") and learnings_result.get("antipatterns"):
                    relevant_learnings = learnings_result["antipatterns"][:3]

                    # Add antipattern warnings
                    for ap in relevant_learnings:
                        warnings.append(PredictionWarning(
                            pattern="learned_antipattern",
                            warning=f"Past failure: {ap.get('problem', 'Similar action failed before')[:100]}",
                            confidence=0.75,
                            severity="medium",
                            category="learned",
                            suggested_alternative=ap.get("solution")
                        ))

            except Exception as e:
                print(f"[PredictiveInterrupt] Learning lookup failed: {e}")

        # Filter warnings by threshold
        warnings = [w for w in warnings if w.confidence >= self.warning_threshold]

        # Sort by severity and confidence
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        warnings.sort(key=lambda w: (severity_order.get(w.severity, 2), -w.confidence))

        # Dedupe alternatives
        alternatives = list(dict.fromkeys(alternatives))[:5]

        return PredictionResult(
            has_warnings=len(warnings) > 0,
            warnings=warnings,
            suggested_alternatives=alternatives,
            relevant_learnings=relevant_learnings
        )

    def _check_danger_patterns(self, message: str) -> List[PredictionWarning]:
        """Check message against known dangerous patterns."""
        warnings = []
        message_lower = message.lower()

        for danger in self.DANGER_PATTERNS:
            for pattern in danger["patterns"]:
                if pattern.lower() in message_lower:
                    if danger["confidence"] >= self.warning_threshold:
                        warnings.append(PredictionWarning(
                            pattern=pattern,
                            warning=danger["warning"],
                            confidence=danger["confidence"],
                            severity=danger["severity"],
                            category=danger["category"],
                            suggested_alternative=danger.get("alternative")
                        ))
                    break  # Only match once per danger definition

        return warnings

    async def get_quick_check(self, message: str) -> Optional[Dict]:
        """
        Quick synchronous check for critical dangers only.
        Use this for fast feedback before full analysis.
        """
        message_lower = message.lower()

        for danger in self.DANGER_PATTERNS:
            if danger["severity"] in ("critical", "high"):
                for pattern in danger["patterns"]:
                    if pattern.lower() in message_lower:
                        return {
                            "has_critical_warning": True,
                            "pattern": pattern,
                            "warning": danger["warning"],
                            "severity": danger["severity"]
                        }

        return {"has_critical_warning": False}


# Singleton instance
_service_instance = None


def get_predictive_service(mcp_bridge=None) -> PredictiveInterruptService:
    """Get or create the predictive interrupt service singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictiveInterruptService(mcp_bridge)
    elif mcp_bridge and not _service_instance.mcp_bridge:
        _service_instance.mcp_bridge = mcp_bridge
    return _service_instance
