"""
Audit Logger - Immutable Change History with Checksums

Provides tamper-evident logging for self-improvement actions:
- Append-only JSONL logs per month
- Checksum for integrity verification
- Track: propose, approve, test, deploy, rollback
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class AuditAction(str, Enum):
    """Types of auditable actions."""
    PROPOSE = "propose"
    APPROVE = "approve"
    REJECT = "reject"
    STAGE = "stage"
    TEST = "test"
    DEPLOY = "deploy"
    ROLLBACK = "rollback"
    MANUAL_ROLLBACK = "manual_rollback"
    EMERGENCY_ROLLBACK = "emergency_rollback"
    HEALTH_CHECK = "health_check"
    RESTART = "restart"


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: str
    action: AuditAction
    actor: str  # Who performed the action
    proposal_id: Optional[str] = None
    session_id: Optional[str] = None
    details: Dict = field(default_factory=dict)
    checksum: Optional[str] = None  # Hash of this entry
    prev_checksum: Optional[str] = None  # Hash chain link

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "action": self.action.value,
            "actor": self.actor,
            "proposal_id": self.proposal_id,
            "session_id": self.session_id,
            "details": self.details,
            "checksum": self.checksum,
            "prev_checksum": self.prev_checksum
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AuditEntry':
        return cls(
            timestamp=data["timestamp"],
            action=AuditAction(data["action"]),
            actor=data["actor"],
            proposal_id=data.get("proposal_id"),
            session_id=data.get("session_id"),
            details=data.get("details", {}),
            checksum=data.get("checksum"),
            prev_checksum=data.get("prev_checksum")
        )


class AuditLogger:
    """
    Provides immutable, tamper-evident audit logging for Cerebro.

    Uses append-only JSONL files with hash chains for integrity verification.
    Logs are organized by month for easy archival and retrieval.
    """

    def __init__(self, audit_path: Path, actor: str = "cerebro"):
        """
        Initialize the audit logger.

        Args:
            audit_path: Base path for audit logs
            actor: Default actor name for log entries
        """
        self.audit_path = Path(audit_path)
        self.default_actor = actor
        self._last_checksum: Optional[str] = None
        self._ensure_directory()

    def _ensure_directory(self):
        """Create audit log directory if needed."""
        self.audit_path.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self, date: datetime = None) -> Path:
        """
        Get the log file path for a given date.

        Args:
            date: Date to get log file for (default: now)

        Returns:
            Path to the JSONL log file
        """
        date = date or datetime.now()
        filename = f"audit_{date.strftime('%Y-%m')}.jsonl"
        return self.audit_path / filename

    def _compute_checksum(self, entry: Dict, prev_checksum: str = None) -> str:
        """
        Compute checksum for an entry.

        Args:
            entry: Entry data (without checksum fields)
            prev_checksum: Previous entry's checksum for chain

        Returns:
            SHA256 checksum
        """
        # Create a copy without checksum fields for hashing
        hash_data = {k: v for k, v in entry.items() if k not in ('checksum', 'prev_checksum')}
        hash_data['prev_checksum'] = prev_checksum or ""

        # Deterministic JSON serialization
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _get_last_checksum(self, log_file: Path) -> Optional[str]:
        """
        Get the checksum of the last entry in a log file.

        Args:
            log_file: Path to the log file

        Returns:
            Last entry's checksum or None
        """
        if not log_file.exists():
            return None

        last_line = None
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        last_line = line
        except Exception:
            return None

        if last_line:
            try:
                entry = json.loads(last_line)
                return entry.get('checksum')
            except json.JSONDecodeError:
                pass

        return None

    async def log_action(
        self,
        action: str,
        proposal_id: str = None,
        session_id: str = None,
        details: Dict = None,
        actor: str = None
    ) -> AuditEntry:
        """
        Log an action to the audit trail.

        Args:
            action: Action type (string or AuditAction)
            proposal_id: Associated proposal ID
            session_id: Associated staging session ID
            details: Additional details
            actor: Actor performing the action

        Returns:
            The created AuditEntry
        """
        # Convert string action to enum if needed
        if isinstance(action, str):
            try:
                action = AuditAction(action)
            except ValueError:
                # Use as-is if not a known action
                pass

        timestamp = datetime.now().isoformat()
        log_file = self._get_log_file()

        # Get previous checksum for chain
        prev_checksum = self._last_checksum or self._get_last_checksum(log_file)

        # Create entry
        entry = AuditEntry(
            timestamp=timestamp,
            action=action if isinstance(action, AuditAction) else AuditAction.PROPOSE,
            actor=actor or self.default_actor,
            proposal_id=proposal_id,
            session_id=session_id,
            details=details or {},
            prev_checksum=prev_checksum
        )

        # Compute checksum
        entry_dict = entry.to_dict()
        entry.checksum = self._compute_checksum(entry_dict, prev_checksum)
        entry_dict['checksum'] = entry.checksum

        # Append to log file
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry_dict, default=str) + '\n')

            self._last_checksum = entry.checksum

        except Exception as e:
            print(f"Error writing audit log: {e}")

        return entry

    async def verify_integrity(self, log_file: Path = None) -> Dict[str, Any]:
        """
        Verify the integrity of an audit log file.

        Checks that all checksums are valid and the chain is unbroken.

        Args:
            log_file: Log file to verify (default: current month)

        Returns:
            Dict with verification results
        """
        log_file = log_file or self._get_log_file()

        result = {
            "file": str(log_file),
            "valid": True,
            "entries_count": 0,
            "broken_at": None,
            "errors": []
        }

        if not log_file.exists():
            result["valid"] = True
            result["entries_count"] = 0
            return result

        prev_checksum = None

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as e:
                        result["valid"] = False
                        result["errors"].append(f"Line {line_num}: Invalid JSON - {e}")
                        result["broken_at"] = line_num
                        break

                    result["entries_count"] += 1

                    # Verify chain link
                    if entry.get('prev_checksum') != prev_checksum:
                        result["valid"] = False
                        result["errors"].append(
                            f"Line {line_num}: Chain broken - expected prev_checksum {prev_checksum}"
                        )
                        result["broken_at"] = line_num
                        break

                    # Verify checksum
                    stored_checksum = entry.get('checksum')
                    computed_checksum = self._compute_checksum(entry, prev_checksum)

                    if stored_checksum != computed_checksum:
                        result["valid"] = False
                        result["errors"].append(
                            f"Line {line_num}: Checksum mismatch - entry may have been tampered"
                        )
                        result["broken_at"] = line_num
                        break

                    prev_checksum = stored_checksum

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Error reading file: {e}")

        return result

    async def get_entries(
        self,
        log_file: Path = None,
        action_filter: AuditAction = None,
        proposal_filter: str = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """
        Retrieve audit log entries.

        Args:
            log_file: Log file to read (default: current month)
            action_filter: Filter by action type
            proposal_filter: Filter by proposal ID
            limit: Maximum entries to return

        Returns:
            List of AuditEntry objects (newest first)
        """
        log_file = log_file or self._get_log_file()

        if not log_file.exists():
            return []

        entries = []

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        entry = AuditEntry.from_dict(data)

                        # Apply filters
                        if action_filter and entry.action != action_filter:
                            continue
                        if proposal_filter and entry.proposal_id != proposal_filter:
                            continue

                        entries.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        continue

        except Exception as e:
            print(f"Error reading audit log: {e}")

        # Return newest first, limited
        return entries[-limit:][::-1]

    async def get_proposal_history(self, proposal_id: str) -> List[AuditEntry]:
        """
        Get all audit entries for a specific proposal.

        Args:
            proposal_id: Proposal ID to get history for

        Returns:
            List of related entries (oldest first)
        """
        entries = await self.get_entries(proposal_filter=proposal_id, limit=1000)
        return entries[::-1]  # Oldest first for history view

    def list_log_files(self) -> List[Path]:
        """
        List all audit log files.

        Returns:
            List of log file paths (newest first)
        """
        files = list(self.audit_path.glob("audit_*.jsonl"))
        return sorted(files, reverse=True)

    async def get_recent_actions(self, limit: int = 20) -> List[AuditEntry]:
        """
        Get the most recent audit entries across all files.

        Args:
            limit: Maximum entries to return

        Returns:
            List of recent entries
        """
        all_entries = []

        for log_file in self.list_log_files():
            entries = await self.get_entries(log_file, limit=limit)
            all_entries.extend(entries)

            if len(all_entries) >= limit:
                break

        # Sort by timestamp and limit
        all_entries.sort(key=lambda e: e.timestamp, reverse=True)
        return all_entries[:limit]

    async def export_entries(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        output_path: Path = None
    ) -> str:
        """
        Export audit entries to a JSON file.

        Args:
            start_date: Start of date range
            end_date: End of date range
            output_path: Output file path

        Returns:
            Path to exported file
        """
        entries = []

        for log_file in self.list_log_files():
            file_entries = await self.get_entries(log_file, limit=10000)

            for entry in file_entries:
                entry_date = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))

                if start_date and entry_date < start_date:
                    continue
                if end_date and entry_date > end_date:
                    continue

                entries.append(entry.to_dict())

        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.audit_path / f"export_{timestamp}.json"

        # Write export
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "exported_at": datetime.now().isoformat(),
                "entries_count": len(entries),
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "entries": entries
            }, f, indent=2, default=str)

        return str(output_path)


# Singleton instance
_logger_instance: Optional[AuditLogger] = None


def get_audit_logger(audit_path: Path = None, actor: str = "cerebro") -> Optional[AuditLogger]:
    """Get or create the audit logger singleton."""
    global _logger_instance

    if _logger_instance is None and audit_path:
        _logger_instance = AuditLogger(audit_path, actor)

    return _logger_instance
