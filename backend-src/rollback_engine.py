"""
Rollback Engine - Automatic Recovery from Failed Changes

Provides automatic and manual rollback capabilities:
- Auto-rollback on test failure
- Auto-rollback on health check failure
- Git reset to previous commit
- Safe restart after rollback
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class RollbackReason(str, Enum):
    """Reason for rollback."""
    TEST_FAILURE = "test_failure"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    MANUAL = "manual"
    TIMEOUT = "timeout"
    ERROR = "error"
    PERFORMANCE_REGRESSION = "performance_regression"


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    success: bool
    reason: RollbackReason
    target_commit: Optional[str] = None
    previous_commit: Optional[str] = None
    message: str = ""
    rolled_back_at: str = field(default_factory=lambda: datetime.now().isoformat())
    restart_success: Optional[bool] = None
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "reason": self.reason.value,
            "target_commit": self.target_commit,
            "previous_commit": self.previous_commit,
            "message": self.message,
            "rolled_back_at": self.rolled_back_at,
            "restart_success": self.restart_success,
            "details": self.details
        }


class RollbackEngine:
    """
    Handles automatic and manual rollback of failed changes.

    Integrates with GitManager for version control rollback and
    SafeRestart for graceful server restart after rollback.
    """

    # Maximum number of rollback attempts
    MAX_ROLLBACK_ATTEMPTS = 3

    def __init__(
        self,
        git_manager=None,
        safe_restart=None,
        health_monitor=None,
        audit_logger=None
    ):
        """
        Initialize the rollback engine.

        Args:
            git_manager: GitManager instance for version control
            safe_restart: SafeRestart instance for server restart
            health_monitor: HealthMonitor instance for health verification
            audit_logger: AuditLogger instance for logging actions
        """
        self.git_manager = git_manager
        self.safe_restart = safe_restart
        self.health_monitor = health_monitor
        self.audit_logger = audit_logger
        self._rollback_history: List[RollbackResult] = []
        self._safe_commits: List[str] = []  # Known good commits

    def mark_commit_as_safe(self, commit_hash: str):
        """
        Mark a commit as known safe/good.

        Args:
            commit_hash: Commit hash to mark as safe
        """
        if commit_hash and commit_hash not in self._safe_commits:
            self._safe_commits.append(commit_hash)
            # Keep only last 20 safe commits
            self._safe_commits = self._safe_commits[-20:]

    def get_last_safe_commit(self) -> Optional[str]:
        """Get the most recent known safe commit."""
        return self._safe_commits[-1] if self._safe_commits else None

    async def rollback_on_failure(
        self,
        test_failed: bool = False,
        health_failed: bool = False,
        reason: str = None,
        target_commit: str = None,
        proposal_id: str = None
    ) -> RollbackResult:
        """
        Perform automatic rollback due to failure.

        Args:
            test_failed: Whether tests failed
            health_failed: Whether health check failed
            reason: Human-readable reason
            target_commit: Specific commit to rollback to (default: last safe)
            proposal_id: ID of proposal that caused the failure

        Returns:
            RollbackResult with operation details
        """
        # Determine rollback reason
        if test_failed:
            rollback_reason = RollbackReason.TEST_FAILURE
        elif health_failed:
            rollback_reason = RollbackReason.HEALTH_CHECK_FAILURE
        else:
            rollback_reason = RollbackReason.ERROR

        # Get current commit before rollback
        current_commit = None
        if self.git_manager:
            current_commit = self.git_manager.get_current_commit()

        # Determine target commit
        if not target_commit:
            target_commit = self.get_last_safe_commit()

        if not target_commit:
            # No safe commit known, try to get parent commit
            if self.git_manager:
                commits = self.git_manager.get_commit_history(limit=2)
                if len(commits) >= 2:
                    target_commit = commits[1]["hash"]

        if not target_commit:
            return RollbackResult(
                success=False,
                reason=rollback_reason,
                message="No target commit available for rollback",
                previous_commit=current_commit
            )

        # Perform the rollback
        result = await self._perform_rollback(
            target_commit=target_commit,
            reason=rollback_reason,
            previous_commit=current_commit,
            details={
                "test_failed": test_failed,
                "health_failed": health_failed,
                "reason_text": reason,
                "proposal_id": proposal_id
            }
        )

        # Log to audit trail
        if self.audit_logger:
            await self.audit_logger.log_action(
                action="rollback",
                proposal_id=proposal_id,
                details={
                    "reason": rollback_reason.value,
                    "target_commit": target_commit,
                    "success": result.success
                }
            )

        # Add to history
        self._rollback_history.append(result)

        return result

    async def _perform_rollback(
        self,
        target_commit: str,
        reason: RollbackReason,
        previous_commit: str = None,
        details: Dict = None
    ) -> RollbackResult:
        """
        Execute the actual rollback operation.

        Args:
            target_commit: Commit to rollback to
            reason: Reason for rollback
            previous_commit: Commit being rolled back from
            details: Additional details

        Returns:
            RollbackResult with operation status
        """
        result = RollbackResult(
            success=False,
            reason=reason,
            target_commit=target_commit,
            previous_commit=previous_commit,
            details=details or {}
        )

        # Step 1: Git rollback
        if not self.git_manager:
            result.message = "Git manager not available"
            return result

        git_result = self.git_manager.rollback_to_commit(target_commit)
        if not git_result.success:
            result.message = f"Git rollback failed: {git_result.message}"
            return result

        result.details["git_rollback"] = "success"

        # Step 2: Restart server
        if self.safe_restart:
            restart_result = await self.safe_restart.restart_with_verification(
                max_wait_seconds=60
            )
            result.restart_success = restart_result.get("success", False)
            result.details["restart"] = restart_result

            if not result.restart_success:
                result.message = f"Rollback complete but restart failed: {restart_result.get('error')}"
                result.success = True  # Git rollback succeeded
                return result

        # Step 3: Verify health
        if self.health_monitor:
            health_report = await self.health_monitor.wait_for_healthy(
                timeout_seconds=30
            )
            result.details["health_check"] = health_report.to_dict()

            if not health_report.healthy:
                result.message = "Rollback complete but health check failed"
                result.success = True  # Git rollback succeeded
                return result

        result.success = True
        result.message = f"Successfully rolled back to {target_commit[:8]}"

        return result

    async def rollback_to_commit(
        self,
        commit_hash: str,
        reason: str = "Manual rollback"
    ) -> RollbackResult:
        """
        Manually rollback to a specific commit.

        Args:
            commit_hash: Commit to rollback to
            reason: Reason for manual rollback

        Returns:
            RollbackResult with operation details
        """
        current_commit = None
        if self.git_manager:
            current_commit = self.git_manager.get_current_commit()

        result = await self._perform_rollback(
            target_commit=commit_hash,
            reason=RollbackReason.MANUAL,
            previous_commit=current_commit,
            details={"reason_text": reason}
        )

        # Log to audit trail
        if self.audit_logger:
            await self.audit_logger.log_action(
                action="manual_rollback",
                details={
                    "target_commit": commit_hash,
                    "reason": reason,
                    "success": result.success
                }
            )

        self._rollback_history.append(result)

        return result

    async def rollback_last_change(self, reason: str = None) -> RollbackResult:
        """
        Rollback to the commit before the last change.

        Args:
            reason: Reason for rollback

        Returns:
            RollbackResult with operation details
        """
        if not self.git_manager:
            return RollbackResult(
                success=False,
                reason=RollbackReason.MANUAL,
                message="Git manager not available"
            )

        # Get commit history
        commits = self.git_manager.get_commit_history(limit=2)
        if len(commits) < 2:
            return RollbackResult(
                success=False,
                reason=RollbackReason.MANUAL,
                message="Not enough commits to rollback"
            )

        # Rollback to parent commit
        return await self.rollback_to_commit(
            commit_hash=commits[1]["hash"],
            reason=reason or "Rollback last change"
        )

    def get_rollback_history(self, limit: int = 20) -> List[Dict]:
        """
        Get recent rollback history.

        Args:
            limit: Maximum entries to return

        Returns:
            List of rollback result dicts
        """
        return [r.to_dict() for r in self._rollback_history[-limit:]]

    def get_safe_commits(self) -> List[str]:
        """Get list of known safe commits."""
        return self._safe_commits.copy()

    async def verify_rollback_capability(self) -> Dict[str, Any]:
        """
        Check if rollback is possible.

        Returns:
            Dict with rollback capability status
        """
        result = {
            "can_rollback": False,
            "git_available": self.git_manager is not None,
            "safe_commits_available": len(self._safe_commits),
            "last_safe_commit": self.get_last_safe_commit(),
            "current_commit": None,
            "commits_available": 0
        }

        if not self.git_manager:
            result["error"] = "Git manager not available"
            return result

        result["current_commit"] = self.git_manager.get_current_commit()

        commits = self.git_manager.get_commit_history(limit=5)
        result["commits_available"] = len(commits)

        result["can_rollback"] = (
            result["git_available"] and
            (result["safe_commits_available"] > 0 or result["commits_available"] > 1)
        )

        return result

    async def emergency_rollback(self) -> RollbackResult:
        """
        Emergency rollback to last known safe state.

        This is a fast-path rollback that skips some verification steps
        for emergency recovery.

        Returns:
            RollbackResult with operation details
        """
        # Find target commit
        target = self.get_last_safe_commit()

        if not target and self.git_manager:
            # Try to get a recent commit
            commits = self.git_manager.get_commit_history(limit=5)
            for commit in commits[1:]:  # Skip current
                target = commit["hash"]
                break

        if not target:
            return RollbackResult(
                success=False,
                reason=RollbackReason.ERROR,
                message="No target commit for emergency rollback"
            )

        current = self.git_manager.get_current_commit() if self.git_manager else None

        # Fast rollback - just git reset, minimal verification
        if self.git_manager:
            git_result = self.git_manager.rollback_to_commit(target)
            if not git_result.success:
                return RollbackResult(
                    success=False,
                    reason=RollbackReason.ERROR,
                    target_commit=target,
                    previous_commit=current,
                    message=f"Emergency rollback failed: {git_result.message}"
                )

        # Quick restart if available
        if self.safe_restart:
            await self.safe_restart.restart_with_verification(max_wait_seconds=30)

        result = RollbackResult(
            success=True,
            reason=RollbackReason.ERROR,
            target_commit=target,
            previous_commit=current,
            message=f"Emergency rollback to {target[:8]} complete"
        )

        # Log to audit trail
        if self.audit_logger:
            await self.audit_logger.log_action(
                action="emergency_rollback",
                details={
                    "target_commit": target,
                    "success": True
                }
            )

        self._rollback_history.append(result)

        return result


# Singleton instance
_engine_instance: Optional[RollbackEngine] = None


def get_rollback_engine(
    git_manager=None,
    safe_restart=None,
    health_monitor=None,
    audit_logger=None
) -> RollbackEngine:
    """Get or create the rollback engine singleton."""
    global _engine_instance

    if _engine_instance is None:
        _engine_instance = RollbackEngine(
            git_manager=git_manager,
            safe_restart=safe_restart,
            health_monitor=health_monitor,
            audit_logger=audit_logger
        )

    return _engine_instance
