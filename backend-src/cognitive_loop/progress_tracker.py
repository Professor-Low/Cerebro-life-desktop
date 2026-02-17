"""
Progress Tracker - Pacing and Velocity Calculations

Tracks goal progress with:
- Pacing score (actual/ideal progress)
- Velocity (rolling 7-day average)
- Projected completion date
- Risk assessment

Based on OKR scoring patterns (0.0-1.0 scale).
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, List

from .goal_pursuit import Goal, GoalProgress


@dataclass
class PacingMetrics:
    """Detailed pacing metrics for a goal."""
    days_elapsed: int
    days_remaining: int
    days_total: int
    ideal_progress: float  # What progress should be at this point
    actual_progress: float  # What progress actually is
    pacing_score: float  # actual / ideal (1.0 = on track)
    velocity: float  # progress per day (7-day rolling average)
    projected_days_to_complete: Optional[int]
    projected_completion_date: Optional[str]
    risk_level: str  # low, medium, high, critical
    risk_factors: List[str]


class ProgressTracker:
    """
    Calculates pacing, velocity, and risk for goal progress.

    Key formulas:
    - pacing_score = actual_cumulative / ideal_cumulative
    - velocity = rolling_7_day_average(daily_progress)
    - projected_completion = today + (remaining / velocity)
    """

    # Velocity calculation window (days)
    VELOCITY_WINDOW = 7

    def calculate_pacing(self, goal: Goal) -> GoalProgress:
        """
        Calculate comprehensive pacing metrics for a goal.

        Returns GoalProgress with:
        - progress_percentage: How much is done (0-1)
        - pacing_score: actual/ideal (1.0 = on track, >1 = ahead, <1 = behind)
        - velocity: Rolling 7-day average of progress per day
        - projected_completion: Estimated completion date
        - risk_level: low/medium/high/critical
        """
        now = datetime.now(timezone.utc)
        progress_pct = goal.get_progress_percentage()

        # Default values for goals without targets
        if not goal.target_value or goal.target_value <= 0:
            return GoalProgress(
                goal_id=goal.goal_id,
                timestamp=now.isoformat(),
                current_value=goal.current_value,
                target_value=goal.target_value or 0,
                progress_percentage=0.0,
                pacing_score=1.0,
                velocity=0.0,
                projected_completion=None,
                risk_level="low",
                days_remaining=0
            )

        # Calculate time-based metrics
        days_remaining = goal.get_days_remaining()
        days_elapsed = 0
        days_total = 0

        if goal.deadline:
            try:
                deadline_dt = datetime.fromisoformat(goal.deadline.replace('Z', '+00:00'))
                created_dt = datetime.fromisoformat(goal.created_at.replace('Z', '+00:00'))
                days_total = max(1, (deadline_dt - created_dt).days)
                days_elapsed = max(0, (now - created_dt).days)
            except (ValueError, TypeError):
                pass

        # Calculate ideal progress (linear assumption)
        if days_total > 0 and days_elapsed >= 0:
            ideal_progress = min(1.0, days_elapsed / days_total)
        else:
            ideal_progress = 0.5  # Default if no timeline

        # Calculate pacing score
        if ideal_progress > 0:
            pacing_score = progress_pct / ideal_progress
        else:
            pacing_score = 1.0 if progress_pct > 0 else 0.0

        # Calculate velocity from progress history
        velocity = self._calculate_velocity(goal)

        # Project completion
        projected_completion = None
        if velocity > 0 and goal.target_value:
            remaining = goal.target_value - goal.current_value
            if remaining > 0:
                days_to_complete = remaining / velocity
                projected_dt = now + timedelta(days=days_to_complete)
                projected_completion = projected_dt.isoformat()

        # Assess risk
        risk_level = self._assess_risk(pacing_score, days_remaining, velocity, goal)

        return GoalProgress(
            goal_id=goal.goal_id,
            timestamp=now.isoformat(),
            current_value=goal.current_value,
            target_value=goal.target_value,
            progress_percentage=progress_pct,
            pacing_score=round(pacing_score, 3),
            velocity=round(velocity, 3),
            projected_completion=projected_completion,
            risk_level=risk_level,
            days_remaining=days_remaining or 0
        )

    def _calculate_velocity(self, goal: Goal) -> float:
        """
        Calculate rolling 7-day average velocity.

        Velocity = sum(daily_deltas) / days_with_progress
        """
        if not goal.progress_history:
            return 0.0

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.VELOCITY_WINDOW)

        # Get recent progress entries
        recent_deltas = []
        for entry in goal.progress_history:
            try:
                timestamp = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                if timestamp >= cutoff:
                    delta = entry.get("delta", 0)
                    if delta > 0:  # Only count positive progress
                        recent_deltas.append(delta)
            except (ValueError, TypeError, KeyError):
                continue

        if not recent_deltas:
            # Fall back to overall velocity
            if goal.progress_history and len(goal.progress_history) >= 2:
                try:
                    first = goal.progress_history[0]
                    last = goal.progress_history[-1]
                    first_dt = datetime.fromisoformat(first["timestamp"].replace('Z', '+00:00'))
                    last_dt = datetime.fromisoformat(last["timestamp"].replace('Z', '+00:00'))
                    days = max(1, (last_dt - first_dt).days)
                    return goal.current_value / days
                except (ValueError, TypeError, KeyError):
                    pass
            return 0.0

        # Calculate average daily velocity
        return sum(recent_deltas) / self.VELOCITY_WINDOW

    def _assess_risk(
        self,
        pacing_score: float,
        days_remaining: Optional[int],
        velocity: float,
        goal: Goal
    ) -> str:
        """
        Assess risk level based on pacing and trajectory.

        Risk Levels:
        - low: On track or ahead (pacing >= 1.0)
        - medium: Slightly behind (pacing 0.8-1.0)
        - high: Significantly behind (pacing 0.5-0.8)
        - critical: Severely behind (pacing < 0.5) or deadline imminent

        Additional factors:
        - Zero velocity with remaining work
        - Deadline within 3 days with significant remaining work
        - Multiple failed subtasks
        """
        # Base risk from pacing
        if pacing_score >= 1.0:
            risk = "low"
        elif pacing_score >= 0.8:
            risk = "medium"
        elif pacing_score >= 0.5:
            risk = "high"
        else:
            risk = "critical"

        # Escalate if deadline is imminent
        if days_remaining is not None:
            progress_pct = goal.get_progress_percentage()
            remaining_work = 1.0 - progress_pct

            # Critical: Less than 3 days with more than 30% work remaining
            if days_remaining <= 3 and remaining_work > 0.3:
                risk = "critical"

            # High: Less than 7 days with more than 50% work remaining
            elif days_remaining <= 7 and remaining_work > 0.5:
                if risk not in ["critical"]:
                    risk = "high"

        # Escalate if velocity is zero but work remains
        if velocity == 0 and goal.current_value < (goal.target_value or 0):
            if risk == "low":
                risk = "medium"
            elif risk == "medium":
                risk = "high"

        # Escalate if many failed approaches
        if len(goal.failed_approaches) >= 3:
            if risk not in ["critical"]:
                risk = "high"

        return risk

    def get_detailed_metrics(self, goal: Goal) -> PacingMetrics:
        """Get detailed pacing metrics with risk factors."""
        now = datetime.now(timezone.utc)
        progress_pct = goal.get_progress_percentage()

        # Time calculations
        days_remaining = goal.get_days_remaining() or 0
        days_elapsed = 0
        days_total = 0

        if goal.deadline:
            try:
                deadline_dt = datetime.fromisoformat(goal.deadline.replace('Z', '+00:00'))
                created_dt = datetime.fromisoformat(goal.created_at.replace('Z', '+00:00'))
                days_total = max(1, (deadline_dt - created_dt).days)
                days_elapsed = max(0, (now - created_dt).days)
            except (ValueError, TypeError):
                pass

        # Ideal progress
        if days_total > 0:
            ideal_progress = min(1.0, days_elapsed / days_total)
        else:
            ideal_progress = 0.5

        # Pacing
        pacing_score = progress_pct / ideal_progress if ideal_progress > 0 else 1.0

        # Velocity
        velocity = self._calculate_velocity(goal)

        # Projection
        projected_days = None
        projected_date = None
        if velocity > 0 and goal.target_value:
            remaining = goal.target_value - goal.current_value
            if remaining > 0:
                projected_days = int(remaining / velocity)
                projected_dt = now + timedelta(days=projected_days)
                projected_date = projected_dt.isoformat()

        # Risk assessment with factors
        risk_level = self._assess_risk(pacing_score, days_remaining, velocity, goal)
        risk_factors = self._get_risk_factors(
            pacing_score, days_remaining, velocity, goal, progress_pct
        )

        return PacingMetrics(
            days_elapsed=days_elapsed,
            days_remaining=days_remaining,
            days_total=days_total,
            ideal_progress=round(ideal_progress, 3),
            actual_progress=round(progress_pct, 3),
            pacing_score=round(pacing_score, 3),
            velocity=round(velocity, 3),
            projected_days_to_complete=projected_days,
            projected_completion_date=projected_date,
            risk_level=risk_level,
            risk_factors=risk_factors
        )

    def _get_risk_factors(
        self,
        pacing_score: float,
        days_remaining: Optional[int],
        velocity: float,
        goal: Goal,
        progress_pct: float
    ) -> List[str]:
        """Identify specific risk factors affecting the goal."""
        factors = []

        # Pacing issues
        if pacing_score < 0.5:
            factors.append(f"Severely behind pace ({pacing_score:.0%} of ideal)")
        elif pacing_score < 0.8:
            factors.append(f"Behind pace ({pacing_score:.0%} of ideal)")

        # Velocity issues
        if velocity == 0 and progress_pct < 1.0:
            factors.append("No recent progress (velocity = 0)")
        elif velocity > 0 and goal.target_value:
            remaining = goal.target_value - goal.current_value
            if remaining > 0 and days_remaining:
                needed_velocity = remaining / days_remaining
                if velocity < needed_velocity * 0.5:
                    factors.append(f"Velocity too slow (need {needed_velocity:.1f}/day, at {velocity:.1f}/day)")

        # Deadline issues
        if days_remaining is not None:
            if days_remaining <= 0:
                factors.append("Deadline passed")
            elif days_remaining <= 3 and progress_pct < 0.7:
                factors.append(f"Only {days_remaining} days left with {(1-progress_pct)*100:.0f}% remaining")
            elif days_remaining <= 7 and progress_pct < 0.5:
                factors.append(f"Less than a week left with {(1-progress_pct)*100:.0f}% remaining")

        # Learning issues
        if len(goal.failed_approaches) >= 3:
            factors.append(f"{len(goal.failed_approaches)} failed approaches recorded")
        elif len(goal.failed_approaches) >= 1:
            factors.append(f"Some approaches have failed ({len(goal.failed_approaches)})")

        # No milestones
        if not goal.milestones:
            factors.append("Goal not decomposed into milestones")

        return factors

    def suggest_interventions(self, goal: Goal) -> List[str]:
        """Suggest interventions for at-risk goals."""
        metrics = self.get_detailed_metrics(goal)
        suggestions = []

        if metrics.risk_level == "critical":
            suggestions.append("URGENT: Escalate to Professor for guidance")
            suggestions.append("Consider reducing scope or extending deadline")

        if "No recent progress" in str(metrics.risk_factors):
            suggestions.append("Identify and remove blockers")
            suggestions.append("Consider re-decomposing stuck milestones")

        if "Velocity too slow" in str(metrics.risk_factors):
            suggestions.append("Increase focus/allocation to this goal")
            suggestions.append("Look for parallelizable subtasks")

        if not goal.milestones:
            suggestions.append("Decompose goal into milestones for better tracking")

        if len(goal.failed_approaches) >= 2:
            suggestions.append("Review failed approaches and try fundamentally different strategy")
            suggestions.append("Search AI Memory for similar problems and their solutions")

        if not suggestions:
            if metrics.pacing_score >= 1.0:
                suggestions.append("On track - maintain current pace")
            else:
                suggestions.append("Monitor closely - small adjustments may help")

        return suggestions


# Singleton instance
_tracker_instance: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """Get or create the progress tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ProgressTracker()
    return _tracker_instance
