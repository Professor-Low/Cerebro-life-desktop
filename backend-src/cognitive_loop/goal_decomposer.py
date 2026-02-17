"""
Goal Decomposer - Hierarchical Task Network Decomposition

Uses LLM (Qwen3-32B on DGX Spark) to break goals into:
- Milestones (3-5 checkpoints, each ~20-30% of goal)
- Subtasks for active milestone (atomic, 15-60 min tasks)

Key patterns:
- HTN Planning (hierarchical task decomposition)
- Lazy Expansion (only decompose active milestones)
- Agent-aware tasks (coder, researcher, analyst, worker)
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from .goal_pursuit import (
    Goal, Milestone, GoalPursuitEngine,
    MilestoneStatus, SubtaskStatus, get_goal_pursuit_engine
)
from .ollama_client import OllamaClient, ChatMessage


@dataclass
class DecompositionResult:
    """Result of goal decomposition."""
    success: bool
    goal_id: str
    milestones_created: int
    subtasks_created: int
    error: Optional[str] = None


class GoalDecomposer:
    """
    Decomposes goals into milestones and subtasks using LLM.

    Strategy:
    1. When a goal is created, generate 3-5 milestones
    2. Only generate subtasks for the FIRST milestone (lazy expansion)
    3. When a milestone is 80% complete, expand the next milestone
    4. If decomposition fails, allow manual re-decomposition
    """

    MILESTONE_PROMPT = '''You are decomposing a goal into milestones for an autonomous AI agent (Cerebro).

Goal: "{description}"
Target: {target_value} {target_unit}
Deadline: {deadline}
Type: {goal_type}

Context about the user:
- Developer with coding skills (Python, JavaScript, FastAPI)
- Has NAS storage system and DGX Spark AI server
- Looking for practical, achievable paths

Break this goal into 3-5 MILESTONES. Each milestone should:
1. Represent 20-30% of the total goal
2. Be a concrete checkpoint (measurable)
3. Build toward the final goal progressively
4. Be achievable in 1-2 weeks max

For financial goals, milestones might be:
- Research & validate approach ($0 target, 0% of goal)
- Set up infrastructure ($0-100, 5-10% of goal)
- First revenue milestone ($X, 25% of goal)
- Scaling milestone ($X, 50% of goal)
- Target milestone (100%)

RESPOND WITH ONLY JSON in this exact format:
{{
    "milestones": [
        {{
            "description": "Research and validate 3 income approaches",
            "target_percentage": 0.0,
            "target_value": 0,
            "order": 0
        }},
        {{
            "description": "Set up chosen approach (tools, accounts, first steps)",
            "target_percentage": 0.1,
            "target_value": 200,
            "order": 1
        }},
        {{
            "description": "First $500 milestone",
            "target_percentage": 0.25,
            "target_value": 500,
            "order": 2
        }},
        {{
            "description": "Scale to $1000/month",
            "target_percentage": 0.5,
            "target_value": 1000,
            "order": 3
        }},
        {{
            "description": "Reach target of $2000/month",
            "target_percentage": 1.0,
            "target_value": 2000,
            "order": 4
        }}
    ]
}}'''

    SUBTASK_PROMPT = '''You are generating subtasks for a milestone. These will be executed by autonomous AI agents.

Goal: "{goal_description}"
Milestone: "{milestone_description}"
Milestone target: {target_value} {target_unit} ({target_percentage:.0%} of goal)

Previous milestones completed: {completed_milestones}
Failed approaches to avoid: {failed_approaches}

AVAILABLE SKILLS (reusable browser automations):
{available_skills}

Generate 3-7 SUBTASKS for this milestone. Each subtask should:
1. Be atomic (15-60 minutes of work)
2. Have clear success criteria
3. Specify the agent type needed:
   - "coder": Write/modify code, create scripts
   - "researcher": Search web, gather information
   - "analyst": Analyze data, create reports
   - "worker": General tasks, file operations

For tasks that involve BROWSER AUTOMATION (navigating websites, filling forms, clicking buttons):
- If an existing skill matches, use it with: "skill_id": "<skill_id>", "skill_parameters": {{...}}
- If no skill exists, specify: "exploration_goal": "<what to learn to do on the website>"
  The system will explore the website, learn the workflow, and create a reusable skill.

RESPOND WITH ONLY JSON in this exact format:
{{
    "subtasks": [
        {{
            "description": "Search web for top 10 income approaches",
            "agent_type": "researcher",
            "depends_on": []
        }},
        {{
            "description": "Create Upwork freelancer profile",
            "agent_type": "worker",
            "exploration_goal": "Navigate to Upwork signup, complete freelancer registration form",
            "depends_on": ["0"]
        }},
        {{
            "description": "Post first job listing on platform",
            "agent_type": "worker",
            "skill_id": "skill_post_job",
            "skill_parameters": {{"job_title": "B2B Lead Generation", "budget": "500"}},
            "depends_on": ["1"]
        }},
        {{
            "description": "Write recommendation report",
            "agent_type": "coder",
            "depends_on": ["2"]
        }}
    ]
}}

NOTE: depends_on uses INDEX positions (0, 1, 2...) of other subtasks in this list.'''

    SUBTASK_PROMPT_NO_SKILLS = '''You are generating subtasks for a milestone. These will be executed by autonomous AI agents.

Goal: "{goal_description}"
Milestone: "{milestone_description}"
Milestone target: {target_value} {target_unit} ({target_percentage:.0%} of goal)

Previous milestones completed: {completed_milestones}
Failed approaches to avoid: {failed_approaches}

Generate 3-7 SUBTASKS for this milestone. Each subtask should:
1. Be atomic (15-60 minutes of work)
2. Have clear success criteria
3. Specify the agent type needed:
   - "coder": Write/modify code, create scripts
   - "researcher": Search web, gather information
   - "analyst": Analyze data, create reports
   - "worker": General tasks, file operations

For the FIRST milestone (research/validation), subtasks might be:
- Research online for {topic} options
- Analyze the user's existing skills/resources
- Compare 3-5 specific approaches
- Create a recommendation summary

RESPOND WITH ONLY JSON in this exact format:
{{
    "subtasks": [
        {{
            "description": "Search web for top 10 ways to earn $2000/month with coding skills",
            "agent_type": "researcher",
            "depends_on": []
        }},
        {{
            "description": "Analyze existing business for expansion opportunities",
            "agent_type": "analyst",
            "depends_on": []
        }},
        {{
            "description": "Research freelancing platforms and their earning potential",
            "agent_type": "researcher",
            "depends_on": []
        }},
        {{
            "description": "Create comparison matrix of income approaches",
            "agent_type": "analyst",
            "depends_on": ["0", "1", "2"]
        }},
        {{
            "description": "Write recommendation report with top 3 paths for Professor to choose",
            "agent_type": "coder",
            "depends_on": ["3"]
        }}
    ]
}}

NOTE: depends_on uses INDEX positions (0, 1, 2...) of other subtasks in this list.'''

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        goal_engine: Optional[GoalPursuitEngine] = None,
        skill_generator=None
    ):
        self.ollama = ollama_client or OllamaClient()
        self.goal_engine = goal_engine or get_goal_pursuit_engine()
        self.skill_generator = skill_generator  # Optional: for skill-aware decomposition

    def _get_available_skills_summary(self) -> str:
        """Get summary of available skills for the prompt."""
        if not self.skill_generator:
            return "No skills available (skill generator not configured)"

        try:
            from .skill_generator import SkillStatus
            skills = self.skill_generator.list_skills(status=SkillStatus.VERIFIED)

            if not skills:
                skills = self.skill_generator.list_skills()[:10]  # Get any skills

            if not skills:
                return "No skills available yet"

            lines = []
            for skill in skills[:15]:  # Limit to 15 skills
                params = f" (params: {', '.join(skill.parameters)})" if skill.parameters else ""
                lines.append(f"- {skill.id}: {skill.name} - {skill.description[:80]}{params}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error loading skills: {e}"

    async def decompose_goal(self, goal_id: str) -> DecompositionResult:
        """
        Decompose a goal into milestones and initial subtasks.

        Steps:
        1. Generate milestones for the entire goal
        2. Generate subtasks for ONLY the first milestone (lazy expansion)
        3. Mark first milestone as active
        """
        goal = self.goal_engine.get_goal(goal_id)
        if not goal:
            return DecompositionResult(
                success=False,
                goal_id=goal_id,
                milestones_created=0,
                subtasks_created=0,
                error="Goal not found"
            )

        # Check if already decomposed
        existing_milestones = self.goal_engine.get_milestones_for_goal(goal_id)
        if existing_milestones:
            # Already has milestones - expand next milestone instead
            return await self._expand_next_milestone(goal, existing_milestones)

        try:
            # Step 1: Generate milestones
            milestones = await self._generate_milestones(goal)
            if not milestones:
                return DecompositionResult(
                    success=False,
                    goal_id=goal_id,
                    milestones_created=0,
                    subtasks_created=0,
                    error="Failed to generate milestones"
                )

            # Step 2: Create milestones in storage
            created_milestones = []
            for m_data in milestones:
                milestone = self.goal_engine.add_milestone(
                    goal_id=goal_id,
                    description=m_data["description"],
                    target_value=m_data.get("target_value"),
                    target_percentage=m_data.get("target_percentage", 0.0),
                    order=m_data.get("order", len(created_milestones))
                )
                if milestone:
                    created_milestones.append(milestone)

            if not created_milestones:
                return DecompositionResult(
                    success=False,
                    goal_id=goal_id,
                    milestones_created=0,
                    subtasks_created=0,
                    error="Failed to create milestones"
                )

            # Step 3: Activate first milestone
            first_milestone = created_milestones[0]
            first_milestone.status = MilestoneStatus.ACTIVE.value
            first_milestone.started_at = datetime.now(timezone.utc).isoformat()

            # Step 4: Generate subtasks for first milestone only (lazy expansion)
            subtasks_created = 0
            subtasks = await self._generate_subtasks(goal, first_milestone, [])
            if subtasks:
                for s_data in subtasks:
                    subtask = self.goal_engine.add_subtask(
                        milestone_id=first_milestone.milestone_id,
                        description=s_data["description"],
                        agent_type=s_data.get("agent_type", "worker"),
                        depends_on=s_data.get("depends_on", []),
                        skill_id=s_data.get("skill_id"),
                        skill_parameters=s_data.get("skill_parameters", {}),
                        exploration_goal=s_data.get("exploration_goal")
                    )
                    if subtask:
                        subtasks_created += 1

            print(f"[GoalDecomposer] Decomposed goal {goal_id}: {len(created_milestones)} milestones, {subtasks_created} subtasks")

            return DecompositionResult(
                success=True,
                goal_id=goal_id,
                milestones_created=len(created_milestones),
                subtasks_created=subtasks_created
            )

        except Exception as e:
            print(f"[GoalDecomposer] Error decomposing goal: {e}")
            import traceback
            traceback.print_exc()
            return DecompositionResult(
                success=False,
                goal_id=goal_id,
                milestones_created=0,
                subtasks_created=0,
                error=str(e)
            )

    async def _expand_next_milestone(
        self,
        goal: Goal,
        existing_milestones: List[Milestone]
    ) -> DecompositionResult:
        """
        Expand the next pending milestone with subtasks.

        Called when:
        - Current milestone is 80%+ complete
        - Explicitly requested
        """
        # Find first unexpanded milestone
        next_milestone = None
        for m in existing_milestones:
            if not m.expanded and m.status in [MilestoneStatus.PENDING.value, MilestoneStatus.ACTIVE.value]:
                next_milestone = m
                break

        if not next_milestone:
            return DecompositionResult(
                success=True,
                goal_id=goal.goal_id,
                milestones_created=0,
                subtasks_created=0,
                error="All milestones already expanded"
            )

        # Get completed milestones for context
        completed = [m.description for m in existing_milestones if m.status == MilestoneStatus.COMPLETED.value]

        # Generate subtasks
        subtasks = await self._generate_subtasks(goal, next_milestone, completed)
        subtasks_created = 0

        if subtasks:
            for s_data in subtasks:
                subtask = self.goal_engine.add_subtask(
                    milestone_id=next_milestone.milestone_id,
                    description=s_data["description"],
                    agent_type=s_data.get("agent_type", "worker"),
                    depends_on=s_data.get("depends_on", []),
                    skill_id=s_data.get("skill_id"),
                    skill_parameters=s_data.get("skill_parameters", {}),
                    exploration_goal=s_data.get("exploration_goal")
                )
                if subtask:
                    subtasks_created += 1

        print(f"[GoalDecomposer] Expanded milestone {next_milestone.milestone_id}: {subtasks_created} subtasks")

        return DecompositionResult(
            success=True,
            goal_id=goal.goal_id,
            milestones_created=0,
            subtasks_created=subtasks_created
        )

    async def _generate_milestones(self, goal: Goal) -> List[Dict[str, Any]]:
        """Use LLM to generate milestones for a goal."""
        prompt = self.MILESTONE_PROMPT.format(
            description=goal.description,
            target_value=goal.target_value or "N/A",
            target_unit=goal.target_unit or "",
            deadline=goal.deadline or "No specific deadline",
            goal_type=goal.goal_type
        )

        try:
            messages = [ChatMessage(role="user", content=prompt)]
            response = await self.ollama.chat(messages, thinking=False)

            # Parse JSON from response
            milestones = self._parse_json_response(response.content, "milestones")
            if milestones:
                return milestones

        except Exception as e:
            print(f"[GoalDecomposer] Error generating milestones: {e}")

        return []

    async def _generate_subtasks(
        self,
        goal: Goal,
        milestone: Milestone,
        completed_milestones: List[str],
        skill_aware: bool = True
    ) -> List[Dict[str, Any]]:
        """Use LLM to generate subtasks for a milestone."""
        # Use skill-aware prompt if skill generator is available
        if skill_aware and self.skill_generator:
            available_skills = self._get_available_skills_summary()
            prompt = self.SUBTASK_PROMPT.format(
                goal_description=goal.description,
                milestone_description=milestone.description,
                target_value=milestone.target_value or goal.target_value or "N/A",
                target_unit=goal.target_unit or "",
                target_percentage=milestone.target_percentage,
                completed_milestones=", ".join(completed_milestones) if completed_milestones else "None",
                failed_approaches=", ".join(goal.failed_approaches) if goal.failed_approaches else "None",
                available_skills=available_skills
            )
        else:
            # Fallback to basic prompt without skills
            prompt = self.SUBTASK_PROMPT_NO_SKILLS.format(
                goal_description=goal.description,
                milestone_description=milestone.description,
                target_value=milestone.target_value or goal.target_value or "N/A",
                target_unit=goal.target_unit or "",
                target_percentage=milestone.target_percentage,
                completed_milestones=", ".join(completed_milestones) if completed_milestones else "None",
                failed_approaches=", ".join(goal.failed_approaches) if goal.failed_approaches else "None"
            )

        try:
            messages = [ChatMessage(role="user", content=prompt)]
            response = await self.ollama.chat(messages, thinking=False)

            # Parse JSON from response
            subtasks = self._parse_json_response(response.content, "subtasks")
            if subtasks:
                # Convert index-based depends_on to subtask IDs
                # For now, we'll use temporary IDs that will be replaced during creation
                return subtasks

        except Exception as e:
            print(f"[GoalDecomposer] Error generating subtasks: {e}")

        return []

    def _parse_json_response(self, content: str, key: str) -> List[Dict[str, Any]]:
        """Parse JSON array from LLM response."""
        try:
            # Try to find JSON block
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                if key in data and isinstance(data[key], list):
                    return data[key]
        except json.JSONDecodeError:
            pass

        # Try to find JSON array directly
        try:
            array_match = re.search(r'\[[\s\S]*\]', content)
            if array_match:
                return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

        return []

    async def should_expand_next(self, goal_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if the next milestone should be expanded.

        Returns (should_expand, milestone_id or reason)

        Expand when:
        - Current milestone is 80%+ complete
        - Current milestone has no pending subtasks
        """
        goal = self.goal_engine.get_goal(goal_id)
        if not goal:
            return False, "Goal not found"

        active_milestone = self.goal_engine.get_active_milestone(goal_id)
        if not active_milestone:
            return False, "No active milestone"

        # Check if already expanded
        if active_milestone.expanded:
            # Check completion percentage
            subtasks = self.goal_engine.get_subtasks_for_milestone(active_milestone.milestone_id)
            if not subtasks:
                return False, "Milestone has no subtasks"

            completed = sum(1 for s in subtasks if s.status == SubtaskStatus.COMPLETED.value)
            total = len(subtasks)
            completion_pct = completed / total if total > 0 else 0

            if completion_pct >= 0.8:
                # Find next unexpanded milestone
                milestones = self.goal_engine.get_milestones_for_goal(goal_id)
                for m in milestones:
                    if m.order > active_milestone.order and not m.expanded:
                        return True, m.milestone_id

        return False, "Current milestone not ready for expansion"

    async def redecompose_milestone(self, milestone_id: str) -> DecompositionResult:
        """
        Re-decompose a milestone (when original decomposition failed or needs revision).

        Clears existing subtasks and generates new ones.
        """
        milestone = self.goal_engine._milestones.get(milestone_id)
        if not milestone:
            return DecompositionResult(
                success=False,
                goal_id="",
                milestones_created=0,
                subtasks_created=0,
                error="Milestone not found"
            )

        goal = self.goal_engine.get_goal(milestone.goal_id)
        if not goal:
            return DecompositionResult(
                success=False,
                goal_id=milestone.goal_id,
                milestones_created=0,
                subtasks_created=0,
                error="Goal not found"
            )

        # Get existing subtasks (to include their learnings as failed approaches)
        existing_subtasks = self.goal_engine.get_subtasks_for_milestone(milestone_id)
        failed_learnings = []
        for s in existing_subtasks:
            if s.status == SubtaskStatus.FAILED.value and s.learnings:
                failed_learnings.extend(s.learnings)

        # Add to goal's failed approaches
        for learning in failed_learnings:
            goal.add_failed_approach(learning)

        # Clear existing subtasks
        milestone.subtasks.clear()
        milestone.expanded = False

        # Get completed milestones for context
        milestones = self.goal_engine.get_milestones_for_goal(goal.goal_id)
        completed = [m.description for m in milestones if m.status == MilestoneStatus.COMPLETED.value]

        # Generate new subtasks
        subtasks = await self._generate_subtasks(goal, milestone, completed)
        subtasks_created = 0

        if subtasks:
            for s_data in subtasks:
                subtask = self.goal_engine.add_subtask(
                    milestone_id=milestone_id,
                    description=s_data["description"],
                    agent_type=s_data.get("agent_type", "worker"),
                    depends_on=s_data.get("depends_on", []),
                    skill_id=s_data.get("skill_id"),
                    skill_parameters=s_data.get("skill_parameters", {}),
                    exploration_goal=s_data.get("exploration_goal")
                )
                if subtask:
                    subtasks_created += 1

        print(f"[GoalDecomposer] Re-decomposed milestone {milestone_id}: {subtasks_created} subtasks")

        return DecompositionResult(
            success=True,
            goal_id=goal.goal_id,
            milestones_created=0,
            subtasks_created=subtasks_created
        )


# Singleton instance
_decomposer_instance: Optional[GoalDecomposer] = None


def get_goal_decomposer() -> GoalDecomposer:
    """Get or create the goal decomposer instance."""
    global _decomposer_instance
    if _decomposer_instance is None:
        _decomposer_instance = GoalDecomposer()
    return _decomposer_instance
