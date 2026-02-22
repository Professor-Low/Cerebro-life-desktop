"""
Safety Layer - Action Classification and Risk Management

Ensures autonomous actions are:
1. Classified by risk level
2. Within budget limits
3. Approved when required
4. Killable at any time
"""

import os
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Set
from pathlib import Path
from enum import Enum
import uuid
import re


class RiskLevel(str, Enum):
    """Risk classification for actions."""
    LOW = "low"        # Read-only, non-destructive
    MEDIUM = "medium"  # Limited side effects, reversible
    HIGH = "high"      # Significant effects, needs approval
    CRITICAL = "critical"  # System-level, always needs approval


@dataclass
class ActionBudget:
    """Budget configuration for action types."""
    max_per_hour: int = 100
    max_per_day: int = 1000
    requires_approval: bool = False
    cooldown_seconds: int = 0


@dataclass
class PendingAction:
    """An action waiting for approval."""
    id: str
    action_type: str
    target: Optional[str]
    description: str
    risk_level: str
    reasoning: str
    created_at: str
    expires_at: str
    approved: Optional[bool] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PendingAction':
        return cls(**data)


@dataclass
class SpawnUsage:
    """Tracks spawn usage statistics."""
    daily_spawns: int = 0
    daily_date: str = ""  # YYYY-MM-DD
    hourly_spawns: int = 0
    hourly_timestamp: str = ""  # ISO timestamp
    total_spawns: int = 0
    by_agent_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'SpawnUsage':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Risk level configurations
RISK_CONFIGS: Dict[RiskLevel, ActionBudget] = {
    RiskLevel.LOW: ActionBudget(
        max_per_hour=100,
        max_per_day=2000,
        requires_approval=False,
        cooldown_seconds=0
    ),
    RiskLevel.MEDIUM: ActionBudget(
        max_per_hour=20,
        max_per_day=200,
        requires_approval=False,
        cooldown_seconds=5
    ),
    RiskLevel.HIGH: ActionBudget(
        max_per_hour=5,
        max_per_day=20,
        requires_approval=True,
        cooldown_seconds=30
    ),
    RiskLevel.CRITICAL: ActionBudget(
        max_per_hour=2,
        max_per_day=5,
        requires_approval=True,
        cooldown_seconds=60
    )
}

# Spawn-specific limits (separate from general budgets)
SPAWN_LIMITS = {
    "max_per_hour": 5,
    "max_per_day": 20,
    "cooldown_seconds": 30,
    "alert_threshold": 0.8,  # Alert at 80% usage
}

# Pre-approved agent types (can be modified at runtime)
DEFAULT_PREAPPROVALS: Dict[str, int] = {
    # agent_type -> hours of pre-approval (0 = always needs approval)
    "worker": 0,
    "researcher": 0,
    "coder": 0,
    "analyst": 0,
    "orchestrator": 0,  # Never pre-approve orchestrators by default
}


# Action type to risk level mapping
ACTION_RISK_MAP: Dict[str, RiskLevel] = {
    # Low risk - read-only operations
    "search_memory": RiskLevel.LOW,
    "web_search": RiskLevel.LOW,  # Internet search - just reading, no side effects
    "get_goals": RiskLevel.LOW,
    "check_predictions": RiskLevel.LOW,
    "analyze_context": RiskLevel.LOW,
    "read_file": RiskLevel.LOW,
    "list_files": RiskLevel.LOW,
    "get_status": RiskLevel.LOW,
    "query_database": RiskLevel.LOW,
    "ask_question": RiskLevel.LOW,  # Asking the user a question - no side effects
    "propose_paths": RiskLevel.LOW,  # Proposing strategic paths for approval

    # Medium risk - limited side effects
    "create_suggestion": RiskLevel.MEDIUM,
    "update_goal": RiskLevel.MEDIUM,
    "record_learning": RiskLevel.MEDIUM,
    "send_notification": RiskLevel.MEDIUM,
    "create_task": RiskLevel.MEDIUM,
    "update_causal_model": RiskLevel.MEDIUM,
    "create_plan": RiskLevel.MEDIUM,  # Creating an actionable plan

    # High risk - significant effects
    "spawn_agent": RiskLevel.HIGH,
    "execute_command": RiskLevel.HIGH,
    "write_file": RiskLevel.HIGH,
    "modify_config": RiskLevel.HIGH,
    "send_message": RiskLevel.HIGH,
    "schedule_task": RiskLevel.HIGH,

    # Critical - system-level
    "modify_system": RiskLevel.CRITICAL,
    "delete_file": RiskLevel.CRITICAL,
    "restart_service": RiskLevel.CRITICAL,
    "modify_permissions": RiskLevel.CRITICAL,
}


class SafetyLayer:
    """
    Safety layer for autonomous actions.

    Manages:
    - Risk classification
    - Action budgets
    - Approval workflow
    - Kill switch
    """

    DEFAULT_PATH = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "cognitive_loop"

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        autonomy_level: int = 2
    ):
        self.storage_path = storage_path or self.DEFAULT_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.state_file = self.storage_path / "autonomy_state.json"
        self.pending_file = self.storage_path / "pending_actions.json"
        self.spawn_usage_file = self.storage_path / "spawn_usage.json"

        # Autonomy level (1-5)
        self._autonomy_level = autonomy_level

        # Kill switch
        self._killed = False

        # FULL AUTONOMY MODE - allows spawning Claude agents
        # When OFF: Only thinks and does memory operations (FREE)
        # When ON: Can spawn Claude Code agents to do real work (USES SUBSCRIPTION)
        self._full_autonomy_enabled = False

        # Action tracking
        self._action_counts: Dict[str, List[datetime]] = {}
        self._last_action_time: Dict[str, datetime] = {}

        # Pending approvals
        self._pending_actions: Dict[str, PendingAction] = {}

        # Blocked patterns (user-defined)
        self._blocked_patterns: Set[str] = set()

        # Spawn-specific tracking
        self._spawn_usage = SpawnUsage()
        self._spawn_history: List[Dict] = []  # Recent spawn records

        # Pre-approval system
        self._preapprovals: Dict[str, datetime] = {}  # agent_type -> expiry time

        # Load state
        self._load_state()

    def _load_state(self):
        """Load persisted state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    self._autonomy_level = state.get("autonomy_level", 2)
                    self._killed = state.get("killed", False)
                    self._full_autonomy_enabled = state.get("full_autonomy_enabled", False)
                    self._blocked_patterns = set(state.get("blocked_patterns", []))

                    # Load preapprovals
                    preapprovals_raw = state.get("preapprovals", {})
                    for agent_type, expiry_str in preapprovals_raw.items():
                        try:
                            expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                            if expiry > datetime.now(timezone.utc):
                                self._preapprovals[agent_type] = expiry
                        except (ValueError, AttributeError):
                            pass
            except (json.JSONDecodeError, KeyError):
                pass

        if self.pending_file.exists():
            try:
                with open(self.pending_file) as f:
                    data = json.load(f)
                    for item in data.get("pending", []):
                        action = PendingAction.from_dict(item)
                        self._pending_actions[action.id] = action
            except (json.JSONDecodeError, KeyError):
                pass

        if self.spawn_usage_file.exists():
            try:
                with open(self.spawn_usage_file) as f:
                    data = json.load(f)
                    self._spawn_usage = SpawnUsage.from_dict(data.get("usage", {}))
                    self._spawn_history = data.get("history", [])[-100:]  # Keep last 100
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self):
        """Save state to disk."""
        state = {
            "autonomy_level": self._autonomy_level,
            "killed": self._killed,
            "full_autonomy_enabled": self._full_autonomy_enabled,
            "blocked_patterns": list(self._blocked_patterns),
            "preapprovals": {
                agent_type: expiry.isoformat()
                for agent_type, expiry in self._preapprovals.items()
                if expiry > datetime.now(timezone.utc)
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        pending = {
            "pending": [a.to_dict() for a in self._pending_actions.values()]
        }
        with open(self.pending_file, 'w') as f:
            json.dump(pending, f, indent=2)

    def _save_spawn_usage(self):
        """Save spawn usage to disk."""
        data = {
            "usage": self._spawn_usage.to_dict(),
            "history": self._spawn_history[-100:],  # Keep last 100
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        with open(self.spawn_usage_file, 'w') as f:
            json.dump(data, f, indent=2)

    @property
    def autonomy_level(self) -> int:
        """Get current autonomy level (1-5)."""
        return self._autonomy_level

    @autonomy_level.setter
    def autonomy_level(self, level: int):
        """Set autonomy level."""
        self._autonomy_level = max(1, min(5, level))
        self._save_state()

    @property
    def is_killed(self) -> bool:
        """Check if kill switch is active."""
        return self._killed

    @property
    def full_autonomy_enabled(self) -> bool:
        """Check if full autonomy mode is enabled (allows agent spawning)."""
        return self._full_autonomy_enabled

    @full_autonomy_enabled.setter
    def full_autonomy_enabled(self, enabled: bool):
        """Enable/disable full autonomy mode."""
        self._full_autonomy_enabled = enabled
        self._save_state()

    def set_full_autonomy(self, enabled: bool) -> dict:
        """Enable or disable full autonomy mode (allows spawning Claude agents)."""
        self._full_autonomy_enabled = enabled
        self._save_state()
        return {
            "full_autonomy_enabled": enabled,
            "warning": "Claude Code agents can now be spawned (uses your subscription)" if enabled else None
        }

    def classify_action(self, action_type: str, target: Optional[str] = None) -> RiskLevel:
        """
        Classify an action's risk level.

        Args:
            action_type: The type of action
            target: Optional target (file, service, etc.)

        Returns:
            RiskLevel classification
        """
        # Check blocked patterns
        if self._is_blocked(action_type, target):
            return RiskLevel.CRITICAL  # Force approval for blocked

        # Look up base risk
        risk = ACTION_RISK_MAP.get(action_type, RiskLevel.MEDIUM)

        # Elevate risk for sensitive targets
        if target:
            target_lower = target.lower()
            if any(p in target_lower for p in ['system32', 'windows', '/etc/', 'passwd', '.ssh']):
                risk = RiskLevel.CRITICAL
            elif any(p in target_lower for p in ['config', 'settings', '.env', 'credentials']):
                if risk.value < RiskLevel.HIGH.value:
                    risk = RiskLevel.HIGH

        return risk

    def _is_blocked(self, action_type: str, target: Optional[str]) -> bool:
        """Check if action matches any blocked pattern."""
        check_str = f"{action_type}:{target}" if target else action_type
        for pattern in self._blocked_patterns:
            if re.search(pattern, check_str, re.IGNORECASE):
                return True
        return False

    def can_execute(
        self,
        action_type: str,
        risk_level: RiskLevel
    ) -> tuple[bool, str]:
        """
        Check if an action can be executed based on autonomy level.

        Args:
            action_type: The action type
            risk_level: The risk classification

        Returns:
            (can_execute, reason)
        """
        print(f"[SafetyLayer.can_execute] START: action={action_type}, risk={risk_level.value}, level={self._autonomy_level}, killed={self._killed}, full_autonomy={self._full_autonomy_enabled}")

        # Kill switch takes priority
        if self._killed:
            print("[SafetyLayer.can_execute] BLOCKED: Kill switch active")
            return False, "Kill switch is active"

        # Check if action requires full autonomy mode
        REQUIRES_FULL_AUTONOMY = {"spawn_agent", "execute_command", "write_file", "delete_file"}
        if action_type in REQUIRES_FULL_AUTONOMY:
            if not self._full_autonomy_enabled:
                print(f"[SafetyLayer.can_execute] BLOCKED: Full autonomy required for {action_type}")
                return False, f"Full Autonomy mode required for '{action_type}'. Enable it in controls to allow agent spawning."

        # Get budget config
        config = RISK_CONFIGS[risk_level]

        # Check autonomy level allows this risk
        # Level 1: Observer - no actions
        # Level 2: Assistant - LOW only
        # Level 3: Helper - LOW + MEDIUM
        # Level 4: Partner - LOW + MEDIUM + HIGH (with approval)
        # Level 5: Autonomous - all (HIGH/CRITICAL still need approval)
        level_allows = {
            1: [],
            2: [RiskLevel.LOW],
            3: [RiskLevel.LOW, RiskLevel.MEDIUM],
            4: [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH],
            5: [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        }

        allowed_for_level = level_allows.get(self._autonomy_level, [])
        print(f"[SafetyLayer.can_execute] Level {self._autonomy_level} allows: {[r.value for r in allowed_for_level]}")

        if risk_level not in allowed_for_level:
            print(f"[SafetyLayer.can_execute] BLOCKED: Level {self._autonomy_level} doesn't allow {risk_level.value}")
            return False, f"Autonomy level {self._autonomy_level} doesn't allow {risk_level.value} risk actions"

        # Check requires approval (skip at level 5 with full autonomy)
        if config.requires_approval:
            if self._autonomy_level >= 5 and self._full_autonomy_enabled:
                print(f"[SafetyLayer.can_execute] Level 5 + Full Autonomy: skipping approval for {risk_level.value}")
            else:
                print(f"[SafetyLayer.can_execute] BLOCKED: {risk_level.value} requires approval")
                return False, f"{risk_level.value} risk actions require approval"

        # Check budget
        if not self._check_budget(action_type, risk_level):
            print("[SafetyLayer.can_execute] BLOCKED: Budget exceeded")
            return False, f"Budget exceeded for {risk_level.value} risk actions"

        # Check cooldown
        if not self._check_cooldown(action_type, config.cooldown_seconds):
            print("[SafetyLayer.can_execute] BLOCKED: Cooldown active")
            return False, f"Cooldown active for {action_type}"

        print(f"[SafetyLayer.can_execute] ALLOWED: {action_type} can execute")
        return True, "OK"

    def _check_budget(self, action_type: str, risk_level: RiskLevel) -> bool:
        """Check if action is within budget."""
        config = RISK_CONFIGS[risk_level]
        now = datetime.now(timezone.utc)

        # Clean old entries
        key = f"{risk_level.value}:{action_type}"
        if key not in self._action_counts:
            self._action_counts[key] = []

        # Remove entries older than 1 day
        cutoff = now - timedelta(days=1)
        self._action_counts[key] = [
            t for t in self._action_counts[key] if t > cutoff
        ]

        # Check hourly limit
        hour_ago = now - timedelta(hours=1)
        hour_count = sum(1 for t in self._action_counts[key] if t > hour_ago)
        if hour_count >= config.max_per_hour:
            return False

        # Check daily limit
        day_count = len(self._action_counts[key])
        if day_count >= config.max_per_day:
            return False

        return True

    def _check_cooldown(self, action_type: str, cooldown_seconds: int) -> bool:
        """Check if cooldown has passed."""
        if cooldown_seconds == 0:
            return True

        last_time = self._last_action_time.get(action_type)
        if not last_time:
            return True

        elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
        return elapsed >= cooldown_seconds

    def record_action(self, action_type: str, risk_level: RiskLevel):
        """Record that an action was executed."""
        now = datetime.now(timezone.utc)
        key = f"{risk_level.value}:{action_type}"

        if key not in self._action_counts:
            self._action_counts[key] = []
        self._action_counts[key].append(now)
        self._last_action_time[action_type] = now

    def request_approval(
        self,
        action_type: str,
        target: Optional[str],
        description: str,
        reasoning: str,
        risk_level: RiskLevel,
        expires_in_minutes: int = 30
    ) -> PendingAction:
        """
        Request approval for a high-risk action.

        Returns:
            PendingAction that frontend can display
        """
        action = PendingAction(
            id=f"approval_{uuid.uuid4().hex[:12]}",
            action_type=action_type,
            target=target,
            description=description,
            risk_level=risk_level.value,
            reasoning=reasoning,
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)).isoformat()
        )
        self._pending_actions[action.id] = action
        self._save_state()
        return action

    def approve_action(self, action_id: str, approved_by: str = "user") -> Optional[PendingAction]:
        """Approve a pending action."""
        if action_id not in self._pending_actions:
            return None

        action = self._pending_actions[action_id]
        action.approved = True
        action.approved_by = approved_by
        action.approved_at = datetime.now(timezone.utc).isoformat()
        self._save_state()
        return action

    def reject_action(self, action_id: str, rejected_by: str = "user") -> Optional[PendingAction]:
        """Reject a pending action."""
        if action_id not in self._pending_actions:
            return None

        action = self._pending_actions[action_id]
        action.approved = False
        action.approved_by = rejected_by
        action.approved_at = datetime.now(timezone.utc).isoformat()
        self._save_state()
        return action

    def get_pending_actions(self) -> List[PendingAction]:
        """Get all pending actions that haven't expired."""
        now = datetime.now(timezone.utc)
        valid = []
        for action in self._pending_actions.values():
            if action.approved is None:  # Not yet decided
                expires = datetime.fromisoformat(action.expires_at.replace('Z', '+00:00'))
                if expires > now:
                    valid.append(action)
        return valid

    def is_action_approved(self, action_id: str) -> Optional[bool]:
        """Check if an action has been approved."""
        action = self._pending_actions.get(action_id)
        if action:
            return action.approved
        return None

    def kill_switch(self, reason: str = "User triggered"):
        """
        Activate the kill switch - stops all autonomous actions.

        This is the emergency stop that immediately halts the cognitive loop.
        """
        self._killed = True
        self._save_state()
        return {
            "killed": True,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def reset_kill_switch(self):
        """Reset the kill switch to allow operations again."""
        self._killed = False
        self._save_state()
        return {"killed": False}

    def add_blocked_pattern(self, pattern: str):
        """Add a pattern to block."""
        self._blocked_patterns.add(pattern)
        self._save_state()

    def remove_blocked_pattern(self, pattern: str):
        """Remove a blocked pattern."""
        self._blocked_patterns.discard(pattern)
        self._save_state()

    def get_status(self) -> dict:
        """Get current safety status."""
        return {
            "autonomy_level": self._autonomy_level,
            "killed": self._killed,
            "full_autonomy_enabled": self._full_autonomy_enabled,
            "pending_approvals": len(self.get_pending_actions()),
            "blocked_patterns": list(self._blocked_patterns),
            "budgets": {
                level.value: {
                    "max_per_hour": config.max_per_hour,
                    "max_per_day": config.max_per_day,
                    "requires_approval": config.requires_approval
                }
                for level, config in RISK_CONFIGS.items()
            },
            "spawn_usage": self.get_spawn_usage()
        }

    # ========== Spawn-Specific Methods ==========

    def can_spawn(self, agent_type: str = "worker") -> tuple[bool, str]:
        """
        Check if a spawn is allowed based on spawn-specific limits.

        Returns:
            (can_spawn, reason)
        """
        if self._killed:
            return False, "Kill switch is active"

        if not self._full_autonomy_enabled:
            return False, "Full Autonomy mode required for spawning"

        # Update usage counters for current period
        self._update_spawn_counters()

        # Check hourly limit
        if self._spawn_usage.hourly_spawns >= SPAWN_LIMITS["max_per_hour"]:
            return False, f"Hourly spawn limit reached ({SPAWN_LIMITS['max_per_hour']}/hour)"

        # Check daily limit
        if self._spawn_usage.daily_spawns >= SPAWN_LIMITS["max_per_day"]:
            return False, f"Daily spawn limit reached ({SPAWN_LIMITS['max_per_day']}/day)"

        # Check cooldown
        if "spawn_agent" in self._last_action_time:
            elapsed = (datetime.now(timezone.utc) - self._last_action_time["spawn_agent"]).total_seconds()
            if elapsed < SPAWN_LIMITS["cooldown_seconds"]:
                remaining = int(SPAWN_LIMITS["cooldown_seconds"] - elapsed)
                return False, f"Spawn cooldown active ({remaining}s remaining)"

        return True, "OK"

    def _update_spawn_counters(self):
        """Update spawn counters for current period."""
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        current_hour = now.replace(minute=0, second=0, microsecond=0).isoformat()

        # Reset daily counter if new day
        if self._spawn_usage.daily_date != today:
            self._spawn_usage.daily_spawns = 0
            self._spawn_usage.daily_date = today

        # Reset hourly counter if new hour
        if self._spawn_usage.hourly_timestamp != current_hour:
            self._spawn_usage.hourly_spawns = 0
            self._spawn_usage.hourly_timestamp = current_hour

    def record_spawn(self, agent_type: str, agent_id: str, task: str):
        """Record that a spawn occurred."""
        now = datetime.now(timezone.utc)

        self._update_spawn_counters()

        # Update counters
        self._spawn_usage.hourly_spawns += 1
        self._spawn_usage.daily_spawns += 1
        self._spawn_usage.total_spawns += 1

        # Track by agent type
        if agent_type not in self._spawn_usage.by_agent_type:
            self._spawn_usage.by_agent_type[agent_type] = 0
        self._spawn_usage.by_agent_type[agent_type] += 1

        # Record action time for cooldown
        self._last_action_time["spawn_agent"] = now

        # Add to history
        self._spawn_history.append({
            "agent_id": agent_id,
            "agent_type": agent_type,
            "task": task[:200],
            "timestamp": now.isoformat()
        })

        # Save to disk
        self._save_spawn_usage()

    def get_spawn_usage(self) -> dict:
        """Get current spawn usage statistics."""
        self._update_spawn_counters()

        hourly_remaining = max(0, SPAWN_LIMITS["max_per_hour"] - self._spawn_usage.hourly_spawns)
        daily_remaining = max(0, SPAWN_LIMITS["max_per_day"] - self._spawn_usage.daily_spawns)

        # Calculate usage percentages
        hourly_pct = self._spawn_usage.hourly_spawns / SPAWN_LIMITS["max_per_hour"]
        daily_pct = self._spawn_usage.daily_spawns / SPAWN_LIMITS["max_per_day"]

        return {
            "hourly": {
                "used": self._spawn_usage.hourly_spawns,
                "limit": SPAWN_LIMITS["max_per_hour"],
                "remaining": hourly_remaining,
                "percentage": round(hourly_pct * 100, 1)
            },
            "daily": {
                "used": self._spawn_usage.daily_spawns,
                "limit": SPAWN_LIMITS["max_per_day"],
                "remaining": daily_remaining,
                "percentage": round(daily_pct * 100, 1)
            },
            "total_spawns": self._spawn_usage.total_spawns,
            "by_agent_type": self._spawn_usage.by_agent_type,
            "cooldown_seconds": SPAWN_LIMITS["cooldown_seconds"],
            "at_alert_threshold": hourly_pct >= SPAWN_LIMITS["alert_threshold"] or daily_pct >= SPAWN_LIMITS["alert_threshold"],
            "recent_spawns": self._spawn_history[-10:]  # Last 10
        }

    # ========== Pre-approval System ==========

    def set_preapproval(self, agent_type: str, hours: int) -> dict:
        """
        Pre-approve an agent type for X hours.

        Args:
            agent_type: Type of agent (worker, researcher, coder, analyst, orchestrator)
            hours: Hours of pre-approval (0 to remove)

        Returns:
            Status dict
        """
        if agent_type == "orchestrator" and hours > 0:
            # Extra warning for orchestrators
            print(f"[Safety] WARNING: Pre-approving orchestrator agents for {hours} hours")

        if hours <= 0:
            # Remove pre-approval
            self._preapprovals.pop(agent_type, None)
            self._save_state()
            return {
                "agent_type": agent_type,
                "preapproved": False,
                "message": f"Pre-approval removed for {agent_type}"
            }

        # Set expiry
        expiry = datetime.now(timezone.utc) + timedelta(hours=hours)
        self._preapprovals[agent_type] = expiry
        self._save_state()

        return {
            "agent_type": agent_type,
            "preapproved": True,
            "expires_at": expiry.isoformat(),
            "hours": hours,
            "message": f"Pre-approved {agent_type} agents for {hours} hours"
        }

    def is_preapproved(self, agent_type: str) -> bool:
        """Check if an agent type is currently pre-approved."""
        if agent_type not in self._preapprovals:
            return False

        expiry = self._preapprovals[agent_type]
        if expiry > datetime.now(timezone.utc):
            return True

        # Expired - remove it
        del self._preapprovals[agent_type]
        self._save_state()
        return False

    def get_preapprovals(self) -> dict:
        """Get all current pre-approvals."""
        now = datetime.now(timezone.utc)
        active = {}

        for agent_type, expiry in list(self._preapprovals.items()):
            if expiry > now:
                remaining = (expiry - now).total_seconds() / 3600
                active[agent_type] = {
                    "expires_at": expiry.isoformat(),
                    "hours_remaining": round(remaining, 1)
                }
            else:
                # Clean up expired
                del self._preapprovals[agent_type]

        self._save_state()
        return active

    def clear_preapprovals(self):
        """Clear all pre-approvals."""
        self._preapprovals.clear()
        self._save_state()
        return {"cleared": True}
