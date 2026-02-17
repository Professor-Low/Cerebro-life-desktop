"""
Proactive Agent Manager - Autonomous Agent Spawning

This service monitors system state and spawns agents proactively
based on:
- Goals with actionable blockers
- Scheduled maintenance
- System health issues
- Pattern-detected opportunities

All autonomous actions are logged and can be paused by the user.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum


class ProactiveActionType(str, Enum):
    """Types of proactive actions."""
    GOAL_BLOCKER = "goal_blocker"
    MAINTENANCE = "maintenance"
    HEALTH_FIX = "health_fix"
    LEARNING_APPLICATION = "learning_application"
    SCHEDULED = "scheduled"


@dataclass
class ProactiveAction:
    """Represents a proactive action taken or proposed."""
    action_id: str
    action_type: ProactiveActionType
    description: str
    reason: str
    agent_id: Optional[str] = None
    status: str = "proposed"  # proposed, approved, executing, completed, failed, rejected
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    executed_at: Optional[str] = None
    result: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value if isinstance(self.action_type, ProactiveActionType) else self.action_type,
            "description": self.description,
            "reason": self.reason,
            "agent_id": self.agent_id,
            "status": self.status,
            "created_at": self.created_at,
            "executed_at": self.executed_at,
            "result": self.result
        }


class ProactiveAgentManager:
    """
    Monitors system state and spawns agents proactively.

    Safety Features:
    - All actions are logged
    - User can pause/disable proactive actions
    - Actions can require approval before execution
    - Rate limiting prevents spam
    """

    def __init__(
        self,
        mcp_bridge,
        create_agent_func: Callable[..., Awaitable[str]],
        notify_func: Callable[..., Awaitable[None]],
        storage_path: Path,
        check_interval: int = 300,  # 5 minutes
        require_approval: bool = False
    ):
        """
        Initialize the proactive agent manager.

        Args:
            mcp_bridge: MCP Bridge for goals/causal/predictions
            create_agent_func: Async function to create agents
            notify_func: Async function to send notifications
            storage_path: Path to store action history
            check_interval: Seconds between proactive checks
            require_approval: Whether to require user approval
        """
        self.mcp = mcp_bridge
        self.create_agent = create_agent_func
        self.notify = notify_func
        self.storage_path = storage_path
        self.check_interval = check_interval
        self.require_approval = require_approval

        self.enabled = True
        self.paused_until = None
        self.actions: List[ProactiveAction] = []
        self._running = False
        self._task = None

        # Rate limiting
        self._last_actions = {}  # action_type -> last_action_time
        self._min_intervals = {
            ProactiveActionType.GOAL_BLOCKER: 600,  # 10 min between goal actions
            ProactiveActionType.MAINTENANCE: 3600,  # 1 hour between maintenance
            ProactiveActionType.HEALTH_FIX: 300,  # 5 min between health fixes
            ProactiveActionType.LEARNING_APPLICATION: 1800,  # 30 min
        }

        self._ensure_storage()

    def _ensure_storage(self):
        """Create storage directories."""
        proactive_path = self.storage_path / "proactive"
        proactive_path.mkdir(parents=True, exist_ok=True)

    def _generate_id(self) -> str:
        """Generate unique action ID."""
        import hashlib
        ts = datetime.now().isoformat()
        return f"pa_{hashlib.sha256(ts.encode()).hexdigest()[:10]}"

    async def start_monitoring(self):
        """Start the proactive monitoring loop."""
        if self._running:
            return

        self._running = True
        print("[ProactiveAgent] Starting monitoring loop")

        while self._running:
            try:
                if self.enabled and not self._is_paused():
                    await self._check_and_act()
            except Exception as e:
                print(f"[ProactiveAgent] Error in monitoring loop: {e}")

            await asyncio.sleep(self.check_interval)

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._running = False
        print("[ProactiveAgent] Monitoring stopped")

    def pause(self, minutes: int = 60):
        """Pause proactive actions for specified minutes."""
        self.paused_until = datetime.now() + timedelta(minutes=minutes)
        print(f"[ProactiveAgent] Paused until {self.paused_until}")

    def resume(self):
        """Resume proactive actions."""
        self.paused_until = None
        print("[ProactiveAgent] Resumed")

    def _is_paused(self) -> bool:
        """Check if currently paused."""
        if self.paused_until is None:
            return False
        if datetime.now() > self.paused_until:
            self.paused_until = None
            return False
        return True

    def _can_take_action(self, action_type: ProactiveActionType) -> bool:
        """Check if enough time has passed since last action of this type."""
        last = self._last_actions.get(action_type)
        if not last:
            return True

        min_interval = self._min_intervals.get(action_type, 300)
        elapsed = (datetime.now() - last).total_seconds()
        return elapsed >= min_interval

    async def _check_and_act(self):
        """Main check loop - examine state and take proactive actions."""
        print(f"[ProactiveAgent] Running check at {datetime.now().isoformat()}")

        # 1. Check goals with actionable blockers
        await self._check_goal_blockers()

        # 2. Check for due maintenance
        await self._check_maintenance()

        # 3. Check system health
        await self._check_health()

    async def _check_goal_blockers(self):
        """Check for goals with blockers that might be resolvable."""
        if not self._can_take_action(ProactiveActionType.GOAL_BLOCKER):
            return

        try:
            result = await self.mcp.goals("list_active")
            if not result.get("success"):
                return

            goals = result.get("goals", [])

            for goal in goals:
                blockers = goal.get("known_blockers", [])
                if not blockers:
                    continue

                # Check if any blocker seems actionable (not waiting on external factors)
                for blocker in blockers:
                    if self._is_actionable_blocker(blocker):
                        await self._propose_blocker_resolution(goal, blocker)
                        return  # One at a time

        except Exception as e:
            print(f"[ProactiveAgent] Error checking goal blockers: {e}")

    def _is_actionable_blocker(self, blocker: str) -> bool:
        """Determine if a blocker can be acted upon autonomously."""
        blocker_lower = blocker.lower()

        # Blockers that need human action
        non_actionable = [
            "waiting for", "need to ask", "depends on",
            "budget", "approval", "meeting", "call",
            "someone else", "external", "third party"
        ]

        if any(phrase in blocker_lower for phrase in non_actionable):
            return False

        # Blockers that might be resolvable
        actionable = [
            "need to check", "need to find", "need to configure",
            "need to set up", "need to install", "need to fix",
            "error", "failing", "broken", "investigate"
        ]

        return any(phrase in blocker_lower for phrase in actionable)

    async def _propose_blocker_resolution(self, goal: Dict, blocker: str):
        """Propose or execute blocker resolution."""
        action = ProactiveAction(
            action_id=self._generate_id(),
            action_type=ProactiveActionType.GOAL_BLOCKER,
            description=f"Resolve blocker for goal: {goal['description'][:50]}...",
            reason=f"Blocker identified: {blocker[:100]}"
        )

        self.actions.append(action)
        self._save_action(action)

        if self.require_approval:
            # Notify user and wait for approval
            await self.notify(
                notif_type="proactive_proposal",
                title="Cerebro Can Help With Your Goal",
                message=f"I noticed a blocker I might be able to resolve: {blocker[:60]}...",
                link=f"/proactive/{action.action_id}"
            )
            action.status = "proposed"
        else:
            # Execute directly
            await self._execute_blocker_resolution(action, goal, blocker)

        self._last_actions[ProactiveActionType.GOAL_BLOCKER] = datetime.now()

    async def _execute_blocker_resolution(self, action: ProactiveAction, goal: Dict, blocker: str):
        """Execute blocker resolution by spawning an agent."""
        action.status = "executing"
        action.executed_at = datetime.now().isoformat()

        try:
            task = f"""I'm helping Professor resolve a blocker for one of their goals.

GOAL: {goal['description']}
BLOCKER: {blocker}

Please investigate this blocker and try to resolve it. If you need more information,
explain what you found and what's needed next. Be thorough but efficient.

Context: Goal ID is {goal['goal_id']}"""

            agent_id = await self.create_agent(
                task=task,
                agent_type="researcher",
                context=f"Proactive blocker resolution for goal: {goal['goal_id']}"
            )

            action.agent_id = agent_id
            action.status = "executing"
            self._save_action(action)

            await self.notify(
                notif_type="proactive_action",
                title="Cerebro is Working on Your Goal",
                message=f"Attempting to resolve: {blocker[:50]}...",
                link=f"/agents/{agent_id}"
            )

        except Exception as e:
            action.status = "failed"
            action.result = str(e)
            self._save_action(action)
            print(f"[ProactiveAgent] Failed to spawn agent: {e}")

    async def _check_maintenance(self):
        """Check for scheduled maintenance tasks."""
        if not self._can_take_action(ProactiveActionType.MAINTENANCE):
            return

        # Check for common maintenance needs
        maintenance_tasks = []

        try:
            # Check FAISS index age (if > 24 hours, might need rebuild)
            faiss_index = self.storage_path / "faiss" / "combined.index"
            if faiss_index.exists():
                age_hours = (datetime.now() - datetime.fromtimestamp(
                    faiss_index.stat().st_mtime
                )).total_seconds() / 3600

                if age_hours > 48:
                    maintenance_tasks.append({
                        "task": "Rebuild FAISS index",
                        "reason": f"Index is {age_hours:.0f} hours old"
                    })

            # Check for orphaned agent files
            agents_dir = self.storage_path / "agents"
            if agents_dir.exists():
                # Count failed agents in last 24 hours
                failed_count = 0
                for f in agents_dir.glob("*/*.json"):
                    try:
                        data = json.loads(f.read_text())
                        if data.get("status") == "failed":
                            created = datetime.fromisoformat(data.get("created_at", "2000-01-01"))
                            if (datetime.now() - created).days < 1:
                                failed_count += 1
                    except:
                        pass

                if failed_count > 5:
                    maintenance_tasks.append({
                        "task": "Review failed agents",
                        "reason": f"{failed_count} agents failed in last 24 hours"
                    })

        except Exception as e:
            print(f"[ProactiveAgent] Error checking maintenance: {e}")

        # Execute first maintenance task if any
        if maintenance_tasks:
            task = maintenance_tasks[0]
            await self._spawn_maintenance_agent(task)
            self._last_actions[ProactiveActionType.MAINTENANCE] = datetime.now()

    async def _spawn_maintenance_agent(self, task: Dict):
        """Spawn agent for maintenance task."""
        action = ProactiveAction(
            action_id=self._generate_id(),
            action_type=ProactiveActionType.MAINTENANCE,
            description=task["task"],
            reason=task["reason"]
        )

        if self.require_approval:
            action.status = "proposed"
            self.actions.append(action)
            self._save_action(action)

            await self.notify(
                notif_type="proactive_proposal",
                title="Maintenance Suggested",
                message=f"{task['task']}: {task['reason']}"
            )
        else:
            # For maintenance, always notify but don't auto-execute
            action.status = "proposed"
            self.actions.append(action)
            self._save_action(action)

            await self.notify(
                notif_type="maintenance_suggestion",
                title="Maintenance Recommended",
                message=f"{task['task']}: {task['reason']}"
            )

    async def _check_health(self):
        """Check system health and propose fixes."""
        if not self._can_take_action(ProactiveActionType.HEALTH_FIX):
            return

        try:
            # Check MCP health
            health = await self.mcp.mcp_health() if hasattr(self.mcp, 'mcp_health') else None

            if health and health.get("issues"):
                for issue in health["issues"][:1]:  # One at a time
                    action = ProactiveAction(
                        action_id=self._generate_id(),
                        action_type=ProactiveActionType.HEALTH_FIX,
                        description=f"Fix health issue: {issue}",
                        reason="Detected during health check"
                    )

                    action.status = "proposed"
                    self.actions.append(action)
                    self._save_action(action)

                    await self.notify(
                        notif_type="health_issue",
                        title="System Health Issue Detected",
                        message=issue[:100]
                    )

                    self._last_actions[ProactiveActionType.HEALTH_FIX] = datetime.now()

        except Exception as e:
            print(f"[ProactiveAgent] Error checking health: {e}")

    def _save_action(self, action: ProactiveAction):
        """Save action to history."""
        try:
            history_file = self.storage_path / "proactive" / "history.json"

            if history_file.exists():
                data = json.loads(history_file.read_text())
            else:
                data = {"actions": []}

            # Update or add
            found = False
            for i, a in enumerate(data["actions"]):
                if a.get("action_id") == action.action_id:
                    data["actions"][i] = action.to_dict()
                    found = True
                    break

            if not found:
                data["actions"].insert(0, action.to_dict())

            # Keep last 100 actions
            data["actions"] = data["actions"][:100]
            data["updated_at"] = datetime.now().isoformat()

            history_file.write_text(json.dumps(data, indent=2))

        except Exception as e:
            print(f"[ProactiveAgent] Failed to save action: {e}")

    async def approve_action(self, action_id: str) -> Optional[ProactiveAction]:
        """Approve and execute a proposed action."""
        for action in self.actions:
            if action.action_id == action_id and action.status == "proposed":
                # Execute based on type
                if action.action_type == ProactiveActionType.GOAL_BLOCKER:
                    # Would need to re-fetch goal info - simplified here
                    action.status = "approved"
                    self._save_action(action)
                    return action

        return None

    async def reject_action(self, action_id: str, reason: str = None) -> Optional[ProactiveAction]:
        """Reject a proposed action."""
        for action in self.actions:
            if action.action_id == action_id and action.status == "proposed":
                action.status = "rejected"
                action.result = reason or "User rejected"
                self._save_action(action)
                return action

        return None

    def get_pending_actions(self) -> List[Dict]:
        """Get all pending/proposed actions."""
        return [a.to_dict() for a in self.actions if a.status == "proposed"]

    def get_action_history(self, limit: int = 20) -> List[Dict]:
        """Get action history."""
        try:
            history_file = self.storage_path / "proactive" / "history.json"
            if history_file.exists():
                data = json.loads(history_file.read_text())
                return data.get("actions", [])[:limit]
        except:
            pass
        return []

    def get_status(self) -> Dict:
        """Get current status of proactive system."""
        return {
            "enabled": self.enabled,
            "paused": self._is_paused(),
            "paused_until": self.paused_until.isoformat() if self.paused_until else None,
            "running": self._running,
            "require_approval": self.require_approval,
            "pending_actions": len(self.get_pending_actions()),
            "check_interval": self.check_interval
        }


# Singleton instance
_manager_instance = None


def get_proactive_manager(
    mcp_bridge=None,
    create_agent_func=None,
    notify_func=None,
    storage_path=None
) -> Optional[ProactiveAgentManager]:
    """Get or create the proactive manager singleton."""
    global _manager_instance

    if _manager_instance is None and all([mcp_bridge, create_agent_func, notify_func, storage_path]):
        _manager_instance = ProactiveAgentManager(
            mcp_bridge=mcp_bridge,
            create_agent_func=create_agent_func,
            notify_func=notify_func,
            storage_path=storage_path
        )

    return _manager_instance
