"""
Self-Modification Manager - Allow Cerebro to Propose Improvements

This module manages proposals for Cerebro to improve itself through:
- Hook updates (modify hook scripts)
- Schedule changes (add/modify schedules)
- Prompt improvements (improve agent prompts)
- Config tweaks (non-security config changes)

SECURITY:
- All changes require explicit user approval
- Forbidden file patterns block security-sensitive modifications
- Automatic backup before any change
- One-click rollback always available
- Full audit log of all proposals and decisions
"""

import os
import json
import shutil
import difflib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import fnmatch


class ModificationType(str, Enum):
    """Types of allowed modifications."""
    HOOK_UPDATE = "hook_update"
    SCHEDULE_CHANGE = "schedule_change"
    PROMPT_IMPROVEMENT = "prompt_improvement"
    CONFIG_TWEAK = "config_tweak"


class RiskLevel(str, Enum):
    """Risk levels for modifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProposalStatus(str, Enum):
    """Status of a modification proposal."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class Proposal:
    """Represents a modification proposal."""
    id: str
    mod_type: ModificationType
    description: str
    file_path: str
    old_content: str
    new_content: str
    diff: str
    reason: str
    risk_level: RiskLevel
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None
    decision_reason: Optional[str] = None
    backup_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.mod_type.value if isinstance(self.mod_type, ModificationType) else self.mod_type,
            "description": self.description,
            "file_path": self.file_path,
            "diff": self.diff,
            "reason": self.reason,
            "risk_level": self.risk_level.value if isinstance(self.risk_level, RiskLevel) else self.risk_level,
            "status": self.status.value if isinstance(self.status, ProposalStatus) else self.status,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
            "decision_reason": self.decision_reason,
            "backup_path": self.backup_path,
            "error": self.error
        }


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


class SelfModificationManager:
    """
    Manages proposals for Cerebro to improve itself.
    All changes require explicit user approval.
    """

    # Patterns for forbidden files (never modify these)
    FORBIDDEN_PATTERNS = [
        "*.key",
        "*.pem",
        "*.crt",
        "*.env",
        "*.secrets",
        "*password*",
        "*secret*",
        "*credential*",
        "*token*",
        "*.ssh/*",
        "id_rsa*",
        "authorized_keys",
        "known_hosts",
        ".git/config",
        ".gitconfig",
    ]

    # Paths that are explicitly allowed
    ALLOWED_PATHS = [
        os.path.expanduser("~/.claude/hooks/*"),
        os.path.join(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data")), "schedules", "*"),
        os.path.join(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data")), "projects", "*"),
        os.path.join(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data")), "cerebro", "*"),
    ]

    def __init__(self, storage_path: Path):
        """
        Initialize the self-modification manager.

        Args:
            storage_path: Base path for storing proposals and backups
        """
        self.storage_path = storage_path
        self.proposals_file = storage_path / "cerebro" / "proposals.json"
        self.history_file = storage_path / "cerebro" / "modification_history.json"
        self.backups_dir = storage_path / "cerebro" / "backups"

        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories."""
        self.proposals_file.parent.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def _generate_id(self) -> str:
        """Generate unique proposal ID."""
        import hashlib
        ts = datetime.now().isoformat()
        return f"prop_{hashlib.sha256(ts.encode()).hexdigest()[:10]}"

    def _is_forbidden(self, file_path: str) -> bool:
        """Check if a file path matches forbidden patterns."""
        path_lower = file_path.lower()

        for pattern in self.FORBIDDEN_PATTERNS:
            # Convert glob pattern to work with fnmatch
            if fnmatch.fnmatch(path_lower, pattern.lower()):
                return True

            # Also check if pattern appears anywhere in path
            pattern_simple = pattern.replace("*", "").lower()
            if pattern_simple and pattern_simple in path_lower:
                return True

        return False

    def _is_allowed_path(self, file_path: str) -> bool:
        """Check if file path is in an allowed location."""
        for pattern in self.ALLOWED_PATHS:
            if fnmatch.fnmatch(file_path, pattern):
                return True

            # Also check as prefix
            prefix = pattern.replace("*", "")
            if file_path.startswith(prefix):
                return True

        return False

    def _assess_risk(self, mod_type: ModificationType, file_path: str) -> RiskLevel:
        """Assess the risk level of a modification."""
        # Hook updates are medium risk (can affect all sessions)
        if mod_type == ModificationType.HOOK_UPDATE:
            return RiskLevel.MEDIUM

        # Config tweaks can be high risk
        if mod_type == ModificationType.CONFIG_TWEAK:
            if "main.py" in file_path or "config" in file_path.lower():
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM

        # Prompt improvements are generally low risk
        if mod_type == ModificationType.PROMPT_IMPROVEMENT:
            return RiskLevel.LOW

        # Schedule changes are low risk
        if mod_type == ModificationType.SCHEDULE_CHANGE:
            return RiskLevel.LOW

        return RiskLevel.MEDIUM

    def _create_diff(self, old_content: str, new_content: str) -> str:
        """Create a unified diff between old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile='original',
            tofile='modified',
            lineterm=''
        )

        return ''.join(diff)

    async def create_proposal(
        self,
        mod_type: ModificationType,
        description: str,
        file_path: str,
        old_content: str,
        new_content: str,
        reason: str
    ) -> Proposal:
        """
        Create a new modification proposal.

        Args:
            mod_type: Type of modification
            description: Human-readable description
            file_path: Path to the file being modified
            old_content: Current content of the file
            new_content: Proposed new content
            reason: Why this modification is being proposed

        Returns:
            The created Proposal

        Raises:
            SecurityError: If the file is protected
        """
        # Security checks
        if self._is_forbidden(file_path):
            raise SecurityError(f"Cannot modify protected file: {file_path}")

        if not self._is_allowed_path(file_path):
            raise SecurityError(f"File is not in an allowed location: {file_path}")

        # Create the proposal
        proposal = Proposal(
            id=self._generate_id(),
            mod_type=mod_type,
            description=description,
            file_path=file_path,
            old_content=old_content,
            new_content=new_content,
            diff=self._create_diff(old_content, new_content),
            reason=reason,
            risk_level=self._assess_risk(mod_type, file_path)
        )

        # Save to pending proposals
        await self._save_proposal(proposal)

        return proposal

    async def _save_proposal(self, proposal: Proposal):
        """Save a proposal to storage."""
        proposals = await self._load_proposals()
        proposals.append(proposal.to_dict())

        with open(self.proposals_file, 'w', encoding='utf-8') as f:
            json.dump({"proposals": proposals, "updated_at": datetime.now().isoformat()}, f, indent=2)

    async def _load_proposals(self) -> List[Dict]:
        """Load all proposals from storage."""
        if self.proposals_file.exists():
            try:
                with open(self.proposals_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("proposals", [])
            except:
                pass
        return []

    async def _update_proposal(self, proposal_id: str, updates: Dict):
        """Update a proposal in storage."""
        proposals = await self._load_proposals()

        for i, p in enumerate(proposals):
            if p.get("id") == proposal_id:
                proposals[i].update(updates)
                break

        with open(self.proposals_file, 'w', encoding='utf-8') as f:
            json.dump({"proposals": proposals, "updated_at": datetime.now().isoformat()}, f, indent=2)

    async def get_proposal(self, proposal_id: str) -> Optional[Dict]:
        """Get a specific proposal by ID."""
        proposals = await self._load_proposals()
        for p in proposals:
            if p.get("id") == proposal_id:
                return p
        return None

    async def list_proposals(self, status: Optional[str] = None) -> List[Dict]:
        """List proposals, optionally filtered by status."""
        proposals = await self._load_proposals()

        if status:
            proposals = [p for p in proposals if p.get("status") == status]

        return sorted(proposals, key=lambda p: p.get("created_at", ""), reverse=True)

    async def _backup_file(self, file_path: str) -> str:
        """Create a backup of a file before modification."""
        source = Path(file_path)
        if not source.exists():
            return ""

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source.stem}_{timestamp}{source.suffix}"
        backup_path = self.backups_dir / backup_name

        shutil.copy2(source, backup_path)
        return str(backup_path)

    async def approve_proposal(self, proposal_id: str, user: str = "professor") -> Dict:
        """
        Approve and apply a proposal.

        Args:
            proposal_id: ID of the proposal to approve
            user: User who approved

        Returns:
            Dict with success status and details
        """
        proposal = await self.get_proposal(proposal_id)
        if not proposal:
            return {"success": False, "error": "Proposal not found"}

        if proposal.get("status") != ProposalStatus.PENDING.value:
            return {"success": False, "error": f"Proposal is not pending (status: {proposal.get('status')})"}

        # Create backup
        backup_path = await self._backup_file(proposal["file_path"])

        # Apply the change
        try:
            file_path = Path(proposal["file_path"])

            # Verify current content matches expected (prevent race conditions)
            if file_path.exists():
                current = file_path.read_text(encoding='utf-8')
                if current != proposal.get("old_content", ""):
                    return {
                        "success": False,
                        "error": "File has been modified since proposal was created"
                    }

            # Write new content
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(proposal["new_content"], encoding='utf-8')

            # Update proposal status
            await self._update_proposal(proposal_id, {
                "status": ProposalStatus.APPLIED.value,
                "decided_at": datetime.now().isoformat(),
                "decided_by": user,
                "backup_path": backup_path
            })

            # Log to history
            await self._log_modification(proposal_id, "applied", user, backup_path)

            return {
                "success": True,
                "proposal_id": proposal_id,
                "backup": backup_path
            }

        except Exception as e:
            # Restore from backup if available
            if backup_path and Path(backup_path).exists():
                shutil.copy2(backup_path, proposal["file_path"])

            await self._update_proposal(proposal_id, {
                "status": ProposalStatus.FAILED.value,
                "error": str(e)
            })

            return {"success": False, "error": str(e)}

    async def reject_proposal(self, proposal_id: str, reason: str, user: str = "professor") -> Dict:
        """Reject a proposal."""
        proposal = await self.get_proposal(proposal_id)
        if not proposal:
            return {"success": False, "error": "Proposal not found"}

        await self._update_proposal(proposal_id, {
            "status": ProposalStatus.REJECTED.value,
            "decided_at": datetime.now().isoformat(),
            "decided_by": user,
            "decision_reason": reason
        })

        await self._log_modification(proposal_id, "rejected", user, reason=reason)

        return {"success": True, "proposal_id": proposal_id}

    async def rollback(self, proposal_id: str, user: str = "professor") -> Dict:
        """Rollback a previously applied proposal."""
        proposal = await self.get_proposal(proposal_id)
        if not proposal:
            return {"success": False, "error": "Proposal not found"}

        if proposal.get("status") != ProposalStatus.APPLIED.value:
            return {"success": False, "error": "Proposal was not applied"}

        backup_path = proposal.get("backup_path")
        if not backup_path or not Path(backup_path).exists():
            return {"success": False, "error": "No backup available for rollback"}

        try:
            # Restore from backup
            shutil.copy2(backup_path, proposal["file_path"])

            await self._update_proposal(proposal_id, {
                "status": ProposalStatus.ROLLED_BACK.value,
                "decided_at": datetime.now().isoformat(),
                "decided_by": user
            })

            await self._log_modification(proposal_id, "rolled_back", user)

            return {"success": True, "proposal_id": proposal_id, "rolled_back": True}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _log_modification(
        self,
        proposal_id: str,
        action: str,
        user: str,
        backup_path: str = None,
        reason: str = None
    ):
        """Log a modification action to history."""
        history = []
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                history = data.get("history", [])
            except:
                pass

        history.insert(0, {
            "proposal_id": proposal_id,
            "action": action,
            "user": user,
            "backup_path": backup_path,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })

        # Keep last 100 entries
        history = history[:100]

        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump({"history": history, "updated_at": datetime.now().isoformat()}, f, indent=2)

    async def get_history(self, limit: int = 20) -> List[Dict]:
        """Get modification history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("history", [])[:limit]
            except:
                pass
        return []

    async def create_from_learning(self, learning: Dict) -> Optional[Proposal]:
        """
        Create a proposal from a promoted learning pattern.

        When a pattern is confirmed 3+ times, this generates a proposal
        to codify the learning (e.g., as a hook modification).
        """
        # For now, just create a notification about the learning
        # Full implementation would analyze the learning and generate
        # appropriate modifications

        # Example: If learning is about a command that always needs a flag,
        # we could propose a hook that adds the flag automatically

        return None  # Placeholder for future implementation

    async def apply_with_git(
        self,
        proposal_id: str,
        git_manager=None,
        user: str = "professor",
        commit_message: str = None
    ) -> Dict:
        """
        Approve and apply a proposal with git commit.

        This integrates with the git manager to commit changes after applying,
        providing version control for all modifications.

        Args:
            proposal_id: ID of the proposal to apply
            git_manager: GitManager instance for version control
            user: User performing the action
            commit_message: Custom commit message (optional)

        Returns:
            Dict with success status, proposal details, and commit info
        """
        # First, apply the proposal using standard method
        result = await self.approve_proposal(proposal_id, user)

        if not result.get("success"):
            return result

        # If git manager provided, commit the changes
        if git_manager:
            proposal = await self.get_proposal(proposal_id)
            if proposal:
                # Generate commit message
                if not commit_message:
                    mod_type = proposal.get("type", "modification")
                    description = proposal.get("description", "Applied modification")[:50]
                    commit_message = f"[Cerebro Self-Improvement] {mod_type}: {description}"

                # Commit the changed file
                files = [proposal.get("file_path")] if proposal.get("file_path") else None
                git_result = git_manager.commit_changes(
                    message=commit_message,
                    files=files,
                    author="Cerebro <cerebro@professors-nas.local>"
                )

                if git_result.success:
                    result["commit_hash"] = git_result.commit_hash
                    result["committed"] = True
                else:
                    result["commit_error"] = git_result.message
                    result["committed"] = False
            else:
                result["committed"] = False
                result["commit_error"] = "Proposal not found after apply"
        else:
            result["committed"] = False
            result["commit_error"] = "Git manager not provided"

        return result


# Singleton instance
_manager_instance = None


def get_self_mod_manager(storage_path: Path = None) -> Optional[SelfModificationManager]:
    """Get or create the self-modification manager singleton."""
    global _manager_instance

    if _manager_instance is None and storage_path:
        _manager_instance = SelfModificationManager(storage_path)

    return _manager_instance
