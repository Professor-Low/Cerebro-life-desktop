"""
Goal Pursuit System - Persistent Goal Tracking

Enables Cerebro to persistently pursue long-term goals across sessions:
- Track goals with quantifiable targets and deadlines
- Auto-decompose goals into milestones and subtasks
- Monitor progress with pacing metrics (behind/on-track/ahead)
- Learn from failures using Reflexion-style verbal reinforcement

Based on:
- Plan-and-Execute (separate planning from execution)
- HTN Planning (hierarchical task decomposition)
- OKR Scoring (0.0-1.0 progress, pacing calculations)
"""

import os
import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum


class GoalType(str, Enum):
    """Type of goal."""
    OUTCOME = "outcome"  # Result-based (e.g., "Make $2000")
    PROCESS = "process"  # Activity-based (e.g., "Exercise daily")


class GoalStatus(str, Enum):
    """Status of a goal."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class MilestoneStatus(str, Enum):
    """Status of a milestone."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class SubtaskStatus(str, Enum):
    """Status of a subtask."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class AgentType(str, Enum):
    """Types of agents for subtask execution."""
    CODER = "coder"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WORKER = "worker"


@dataclass
class GoalProgress:
    """Progress snapshot for a goal."""
    goal_id: str
    timestamp: str
    current_value: float
    target_value: float
    progress_percentage: float
    pacing_score: float  # actual/ideal (1.0 = on track, >1 = ahead, <1 = behind)
    velocity: float  # rolling 7-day average of daily progress
    projected_completion: Optional[str] = None  # ISO date
    risk_level: str = "low"  # low, medium, high, critical
    days_remaining: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Subtask:
    """An atomic unit of work within a milestone."""
    subtask_id: str
    milestone_id: str
    description: str
    agent_type: str = "worker"  # coder, researcher, analyst, worker
    depends_on: List[str] = field(default_factory=list)  # Other subtask IDs
    status: str = "pending"
    attempts: int = 0
    max_attempts: int = 3
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0
    learnings: List[str] = field(default_factory=list)  # Reflexion learnings

    # Skill integration fields (Adaptive Browser Learning)
    skill_id: Optional[str] = None           # Use existing skill
    skill_parameters: Dict[str, str] = field(default_factory=dict)  # Skill params
    exploration_goal: Optional[str] = None   # Or explore to create skill
    generated_skill_id: Optional[str] = None # Skill created by exploration

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Subtask':
        # Handle backwards compatibility
        valid_fields = {f.name for f in __import__('dataclasses').fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def is_ready(self, completed_ids: set) -> bool:
        """Check if all dependencies are completed."""
        return set(self.depends_on).issubset(completed_ids)

    def start(self):
        """Mark subtask as in progress."""
        self.status = SubtaskStatus.IN_PROGRESS.value
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.attempts += 1

    def complete(self, result: str):
        """Mark subtask as completed."""
        self.status = SubtaskStatus.COMPLETED.value
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.result = result
        if self.started_at:
            start = datetime.fromisoformat(self.started_at.replace('Z', '+00:00'))
            end = datetime.fromisoformat(self.completed_at.replace('Z', '+00:00'))
            self.duration_ms = (end - start).total_seconds() * 1000

    def fail(self, error: str):
        """Mark subtask as failed."""
        if self.attempts >= self.max_attempts:
            self.status = SubtaskStatus.FAILED.value
        else:
            self.status = SubtaskStatus.PENDING.value  # Can retry
        self.error = error
        self.completed_at = datetime.now(timezone.utc).isoformat()


@dataclass
class Milestone:
    """A checkpoint on the way to achieving a goal."""
    milestone_id: str
    goal_id: str
    description: str
    order: int = 0  # Order within goal (0 = first milestone)
    target_value: Optional[float] = None  # For quantifiable milestones
    target_percentage: float = 0.0  # Percentage of goal (e.g., 0.25 for first 25%)
    current_value: float = 0.0
    subtasks: List[str] = field(default_factory=list)  # Subtask IDs
    status: str = "pending"
    pacing_score: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    expanded: bool = False  # True if subtasks have been generated (lazy expansion)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Milestone':
        return cls(**data)

    def get_progress_percentage(self) -> float:
        """Calculate milestone progress percentage."""
        if self.target_value and self.target_value > 0:
            return min(self.current_value / self.target_value, 1.0)
        return 0.0

    def is_complete(self) -> bool:
        """Check if milestone is complete."""
        if self.target_value and self.target_value > 0:
            return self.current_value >= self.target_value
        return self.status == MilestoneStatus.COMPLETED.value


@dataclass
class Goal:
    """
    Enhanced Goal with targeting, progress tracking, and hierarchy.

    Supports:
    - Quantifiable targets (e.g., $2000 by end of month)
    - Process goals (e.g., exercise 3x/week)
    - Hierarchical decomposition into milestones and subtasks
    - Progress history and learning from failures
    """
    goal_id: str
    description: str
    goal_type: str = "outcome"  # outcome or process

    # Targeting
    target_value: Optional[float] = None  # e.g., 2000
    target_unit: str = ""  # e.g., "dollars"
    deadline: Optional[str] = None  # ISO datetime

    # Progress
    current_value: float = 0.0
    progress_history: List[Dict] = field(default_factory=list)  # [{timestamp, value, delta, note}]

    # Hierarchy
    milestones: List[str] = field(default_factory=list)  # Milestone IDs

    # Learning (Reflexion pattern)
    relevant_learnings: List[str] = field(default_factory=list)  # Learning IDs from AI Memory
    failed_approaches: List[str] = field(default_factory=list)  # What didn't work
    successful_approaches: List[str] = field(default_factory=list)  # What did work

    # Status
    status: str = "active"
    priority: str = "medium"  # low, medium, high

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None

    # Context
    inferred_from: str = "explicit"  # explicit, inferred, directive
    source_directive_id: Optional[str] = None  # Link to directive if created from one

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Goal':
        # Filter out unknown fields (backwards compatibility with old format)
        valid_fields = {f.name for f in __import__('dataclasses').fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def get_progress_percentage(self) -> float:
        """Calculate overall goal progress as percentage."""
        if self.target_value and self.target_value > 0:
            return min(self.current_value / self.target_value, 1.0)
        return 0.0

    def update_progress(self, new_value: float, note: str = ""):
        """Update progress and record in history."""
        delta = new_value - self.current_value
        self.progress_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "value": new_value,
            "delta": delta,
            "note": note
        })
        self.current_value = new_value
        self.updated_at = datetime.now(timezone.utc).isoformat()

        # Check if goal is complete
        if self.target_value and new_value >= self.target_value:
            self.status = GoalStatus.COMPLETED.value
            self.completed_at = datetime.now(timezone.utc).isoformat()

    def add_failed_approach(self, approach: str):
        """Record a failed approach for Reflexion learning."""
        if approach not in self.failed_approaches:
            self.failed_approaches.append(approach)
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_successful_approach(self, approach: str):
        """Record a successful approach for Reflexion learning."""
        if approach not in self.successful_approaches:
            self.successful_approaches.append(approach)
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_days_remaining(self) -> Optional[int]:
        """Get days until deadline."""
        if not self.deadline:
            return None
        try:
            deadline_dt = datetime.fromisoformat(self.deadline.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            delta = deadline_dt - now
            return max(0, delta.days)
        except (ValueError, TypeError):
            return None


class GoalPursuitEngine:
    r"""
    Engine for managing persistent goal pursuit.

    Responsibilities:
    - CRUD operations for goals, milestones, subtasks
    - Progress tracking and pacing calculations
    - Identifying ready subtasks
    - Persistence to $AI_MEMORY_PATH/goals/
    """

    DEFAULT_PATH = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "goals"

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or self.DEFAULT_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Index file for quick access
        self.active_file = self.storage_path / "active.json"

        # In-memory caches
        self._goals: Dict[str, Goal] = {}
        self._milestones: Dict[str, Milestone] = {}
        self._subtasks: Dict[str, Subtask] = {}

        # Load existing data
        self._load_all()

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{prefix}_{timestamp}_{len(self._goals)}"
        return f"{prefix}_{hashlib.md5(hash_input.encode()).hexdigest()[:10]}"

    def _get_goal_dir(self, goal_id: str) -> Path:
        """Get directory for a specific goal."""
        goal_dir = self.storage_path / goal_id
        goal_dir.mkdir(parents=True, exist_ok=True)
        return goal_dir

    def _load_all(self):
        """Load all goals and their data."""
        # Load active goals list
        if self.active_file.exists():
            try:
                data = json.loads(self.active_file.read_text())
                for goal_data in data.get("goals", []):
                    goal = Goal.from_dict(goal_data)
                    self._goals[goal.goal_id] = goal

                    # Load milestones and subtasks for this goal
                    self._load_goal_data(goal.goal_id)
            except Exception as e:
                print(f"[GoalPursuit] Error loading active goals: {e}")

    def _load_goal_data(self, goal_id: str):
        """Load milestones and subtasks for a goal."""
        goal_dir = self._get_goal_dir(goal_id)

        # Load milestones
        milestones_file = goal_dir / "milestones.json"
        if milestones_file.exists():
            try:
                data = json.loads(milestones_file.read_text())
                for m_data in data.get("milestones", []):
                    milestone = Milestone.from_dict(m_data)
                    self._milestones[milestone.milestone_id] = milestone

                for s_data in data.get("subtasks", []):
                    subtask = Subtask.from_dict(s_data)
                    self._subtasks[subtask.subtask_id] = subtask
            except Exception as e:
                print(f"[GoalPursuit] Error loading goal data for {goal_id}: {e}")

    def _save_active_goals(self):
        """Save active goals list."""
        active_goals = [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE.value]
        data = {
            "goals": [g.to_dict() for g in active_goals],
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        self.active_file.write_text(json.dumps(data, indent=2))

    def _save_goal(self, goal_id: str):
        """Save a goal and its data."""
        goal = self._goals.get(goal_id)
        if not goal:
            return

        goal_dir = self._get_goal_dir(goal_id)

        # Save goal data
        goal_file = goal_dir / "goal.json"
        goal_file.write_text(json.dumps(goal.to_dict(), indent=2))

        # Save milestones and subtasks
        goal_milestones = [m for m in self._milestones.values() if m.goal_id == goal_id]
        milestone_ids = [m.milestone_id for m in goal_milestones]
        goal_subtasks = [s for s in self._subtasks.values() if s.milestone_id in milestone_ids]

        data = {
            "milestones": [m.to_dict() for m in goal_milestones],
            "subtasks": [s.to_dict() for s in goal_subtasks]
        }
        milestones_file = goal_dir / "milestones.json"
        milestones_file.write_text(json.dumps(data, indent=2))

        # Save progress history
        progress_file = goal_dir / "progress.json"
        progress_data = {
            "history": goal.progress_history,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        progress_file.write_text(json.dumps(progress_data, indent=2))

        # Update active goals list
        self._save_active_goals()

    # ========== Goal CRUD ==========

    def create_goal(
        self,
        description: str,
        target_value: Optional[float] = None,
        target_unit: str = "",
        deadline: Optional[str] = None,
        goal_type: str = "outcome",
        priority: str = "medium",
        source_directive_id: Optional[str] = None
    ) -> Goal:
        """Create a new goal."""
        goal_id = self._generate_id("goal")

        # Normalize deadline to ISO string
        deadline_str = None
        if deadline is not None:
            if isinstance(deadline, datetime):
                deadline_str = deadline.isoformat()
            else:
                deadline_str = str(deadline)

        # Normalize goal_type to string value
        goal_type_str = goal_type.value if isinstance(goal_type, GoalType) else str(goal_type)

        goal = Goal(
            goal_id=goal_id,
            description=description,
            goal_type=goal_type_str,
            target_value=target_value,
            target_unit=target_unit,
            deadline=deadline_str,
            priority=priority,
            source_directive_id=source_directive_id,
            inferred_from="directive" if source_directive_id else "explicit"
        )

        self._goals[goal_id] = goal
        self._save_goal(goal_id)

        print(f"[GoalPursuit] Created goal: {goal_id} - {description[:50]}")
        return goal

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals."""
        return [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE.value]

    def update_goal_progress(
        self,
        goal_id: str,
        new_value: Optional[float] = None,
        delta: Optional[float] = None,
        note: str = ""
    ) -> Optional[Goal]:
        """Update progress on a goal."""
        goal = self._goals.get(goal_id)
        if not goal:
            return None

        if new_value is not None:
            goal.update_progress(new_value, note)
        elif delta is not None:
            goal.update_progress(goal.current_value + delta, note)

        self._save_goal(goal_id)
        return goal

    def complete_goal(self, goal_id: str) -> Optional[Goal]:
        """Mark a goal as completed."""
        goal = self._goals.get(goal_id)
        if not goal:
            return None

        goal.status = GoalStatus.COMPLETED.value
        goal.completed_at = datetime.now(timezone.utc).isoformat()
        goal.updated_at = datetime.now(timezone.utc).isoformat()

        self._save_goal(goal_id)
        return goal

    def pause_goal(self, goal_id: str) -> Optional[Goal]:
        """Pause a goal."""
        goal = self._goals.get(goal_id)
        if not goal:
            return None

        goal.status = GoalStatus.PAUSED.value
        goal.updated_at = datetime.now(timezone.utc).isoformat()

        self._save_goal(goal_id)
        return goal

    # ========== Milestone CRUD ==========

    def add_milestone(
        self,
        goal_id: str,
        description: str,
        target_value: Optional[float] = None,
        target_percentage: float = 0.0,
        order: int = 0
    ) -> Optional[Milestone]:
        """Add a milestone to a goal."""
        goal = self._goals.get(goal_id)
        if not goal:
            return None

        milestone_id = self._generate_id("milestone")
        milestone = Milestone(
            milestone_id=milestone_id,
            goal_id=goal_id,
            description=description,
            target_value=target_value,
            target_percentage=target_percentage,
            order=order
        )

        self._milestones[milestone_id] = milestone
        goal.milestones.append(milestone_id)

        self._save_goal(goal_id)
        return milestone

    def get_milestones_for_goal(self, goal_id: str) -> List[Milestone]:
        """Get all milestones for a goal."""
        return sorted(
            [m for m in self._milestones.values() if m.goal_id == goal_id],
            key=lambda m: m.order
        )

    def get_active_milestone(self, goal_id: str) -> Optional[Milestone]:
        """Get the currently active milestone for a goal."""
        milestones = self.get_milestones_for_goal(goal_id)
        for m in milestones:
            if m.status in [MilestoneStatus.ACTIVE.value, MilestoneStatus.PENDING.value]:
                return m
        return None

    def complete_milestone(self, milestone_id: str) -> Optional[Milestone]:
        """Mark a milestone as completed."""
        milestone = self._milestones.get(milestone_id)
        if not milestone:
            return None

        milestone.status = MilestoneStatus.COMPLETED.value
        milestone.completed_at = datetime.now(timezone.utc).isoformat()

        # Activate next milestone
        goal_milestones = self.get_milestones_for_goal(milestone.goal_id)
        for m in goal_milestones:
            if m.order > milestone.order and m.status == MilestoneStatus.PENDING.value:
                m.status = MilestoneStatus.ACTIVE.value
                m.started_at = datetime.now(timezone.utc).isoformat()
                break

        self._save_goal(milestone.goal_id)
        return milestone

    # ========== Subtask CRUD ==========

    def add_subtask(
        self,
        milestone_id: str,
        description: str,
        agent_type: str = "worker",
        depends_on: Optional[List[str]] = None,
        skill_id: Optional[str] = None,
        skill_parameters: Optional[Dict[str, str]] = None,
        exploration_goal: Optional[str] = None
    ) -> Optional[Subtask]:
        """Add a subtask to a milestone."""
        milestone = self._milestones.get(milestone_id)
        if not milestone:
            return None

        subtask_id = self._generate_id("subtask")
        subtask = Subtask(
            subtask_id=subtask_id,
            milestone_id=milestone_id,
            description=description,
            agent_type=agent_type,
            depends_on=depends_on or [],
            skill_id=skill_id,
            skill_parameters=skill_parameters or {},
            exploration_goal=exploration_goal
        )

        self._subtasks[subtask_id] = subtask
        milestone.subtasks.append(subtask_id)

        # Mark milestone as expanded
        milestone.expanded = True

        self._save_goal(milestone.goal_id)
        return subtask

    def get_subtasks_for_milestone(self, milestone_id: str) -> List[Subtask]:
        """Get all subtasks for a milestone."""
        return [s for s in self._subtasks.values() if s.milestone_id == milestone_id]

    def get_ready_subtasks(self, goal_id: str) -> List[Subtask]:
        """
        Get subtasks that are ready to execute (dependencies met).

        Returns subtasks where:
        - Status is pending or (failed with attempts < max)
        - All dependencies are completed
        """
        # Get all completed subtask IDs for this goal
        goal_milestones = self.get_milestones_for_goal(goal_id)
        milestone_ids = {m.milestone_id for m in goal_milestones}

        completed_ids = {
            s.subtask_id for s in self._subtasks.values()
            if s.milestone_id in milestone_ids and s.status == SubtaskStatus.COMPLETED.value
        }

        ready = []
        for subtask in self._subtasks.values():
            if subtask.milestone_id not in milestone_ids:
                continue

            # Check if pending (or failed with retries available)
            can_execute = (
                subtask.status == SubtaskStatus.PENDING.value or
                (subtask.status == SubtaskStatus.FAILED.value and subtask.attempts < subtask.max_attempts)
            )

            if can_execute and subtask.is_ready(completed_ids):
                ready.append(subtask)

        return ready

    def get_next_subtask(self, goal_id: str) -> Optional[Subtask]:
        """Get the highest-priority ready subtask for a goal."""
        ready = self.get_ready_subtasks(goal_id)
        if not ready:
            return None

        # Prioritize by milestone order, then by creation time
        def sort_key(s):
            milestone = self._milestones.get(s.milestone_id)
            return (milestone.order if milestone else 999, s.created_at)

        return min(ready, key=sort_key)

    def start_subtask(self, subtask_id: str) -> Optional[Subtask]:
        """Mark a subtask as in progress."""
        subtask = self._subtasks.get(subtask_id)
        if not subtask:
            return None

        subtask.start()

        milestone = self._milestones.get(subtask.milestone_id)
        if milestone:
            self._save_goal(milestone.goal_id)

        return subtask

    def complete_subtask(self, subtask_id: str, result: str) -> Optional[Subtask]:
        """Mark a subtask as completed."""
        subtask = self._subtasks.get(subtask_id)
        if not subtask:
            return None

        subtask.complete(result)

        milestone = self._milestones.get(subtask.milestone_id)
        if milestone:
            # Check if all subtasks in milestone are complete
            milestone_subtasks = self.get_subtasks_for_milestone(milestone.milestone_id)
            all_complete = all(
                s.status == SubtaskStatus.COMPLETED.value for s in milestone_subtasks
            )

            if all_complete:
                self.complete_milestone(milestone.milestone_id)

            self._save_goal(milestone.goal_id)

        return subtask

    def fail_subtask(self, subtask_id: str, error: str, learning: str = "") -> Optional[Subtask]:
        """Mark a subtask as failed and record learning."""
        subtask = self._subtasks.get(subtask_id)
        if not subtask:
            return None

        subtask.fail(error)

        if learning:
            subtask.learnings.append(learning)

        milestone = self._milestones.get(subtask.milestone_id)
        if milestone:
            self._save_goal(milestone.goal_id)

        return subtask

    # ========== Query Methods ==========

    def get_goals_with_progress(self) -> List[Dict[str, Any]]:
        """Get all active goals with their progress metrics."""
        from .progress_tracker import ProgressTracker
        tracker = ProgressTracker()

        results = []
        for goal in self.get_active_goals():
            progress = tracker.calculate_pacing(goal)
            ready_subtasks = self.get_ready_subtasks(goal.goal_id)
            active_milestone = self.get_active_milestone(goal.goal_id)

            results.append({
                "goal": goal.to_dict(),
                "progress": progress.to_dict(),
                "ready_subtasks": [s.to_dict() for s in ready_subtasks],
                "active_milestone": active_milestone.to_dict() if active_milestone else None,
                "total_subtasks": len([
                    s for s in self._subtasks.values()
                    if s.milestone_id in goal.milestones
                ]),
                "completed_subtasks": len([
                    s for s in self._subtasks.values()
                    if s.milestone_id in goal.milestones and s.status == SubtaskStatus.COMPLETED.value
                ])
            })

        # Sort by priority then pacing (lower pacing = more urgent)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        results.sort(key=lambda x: (
            priority_order.get(x["goal"]["priority"], 1),
            x["progress"]["pacing_score"]
        ))

        return results

    def get_blocked_goals(self) -> List[Goal]:
        """Get goals that are blocked (no ready subtasks)."""
        blocked = []
        for goal in self.get_active_goals():
            ready = self.get_ready_subtasks(goal.goal_id)
            if not ready and goal.milestones:  # Has milestones but nothing ready
                blocked.append(goal)
        return blocked


# Singleton instance
_engine_instance: Optional[GoalPursuitEngine] = None


def get_goal_pursuit_engine() -> GoalPursuitEngine:
    """Get or create the goal pursuit engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = GoalPursuitEngine()
    return _engine_instance
