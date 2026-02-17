"""
Learning Injector - Apply Learnings to Agent Prompts

This module injects relevant learnings into agent prompts:
- Solutions that have worked before
- Antipatterns to avoid
- Context-specific tips

This creates a feedback loop where Cerebro gets smarter over time.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class LearningInjector:
    """
    Injects relevant learnings into agent prompts.

    How it works:
    1. When an agent is created, search for relevant learnings
    2. Inject "What has worked before" and "What to avoid" sections
    3. Track whether injected learnings were useful
    4. Update learning confidence based on outcomes
    """

    def __init__(self, mcp_bridge, storage_path: Path):
        """
        Initialize the learning injector.

        Args:
            mcp_bridge: MCP Bridge for searching learnings
            storage_path: Path to AI Memory storage
        """
        self.mcp = mcp_bridge
        self.storage_path = storage_path
        self.injections_log = storage_path / "cerebro" / "learning_injections.json"
        self.injections_log.parent.mkdir(parents=True, exist_ok=True)

    async def get_relevant_learnings(self, task: str, agent_type: str = None) -> Dict[str, List]:
        """
        Find learnings relevant to a task.

        Args:
            task: The task description
            agent_type: Type of agent (optional, for filtering)

        Returns:
            Dict with "apply" (solutions) and "avoid" (antipatterns) lists
        """
        result = {
            "apply": [],
            "avoid": [],
            "tips": []
        }

        try:
            # Search for solutions
            solutions = await self.mcp.learning(
                "find",
                problem=task,
                limit=5
            )

            if solutions.get("success") and solutions.get("learnings"):
                for learning in solutions["learnings"]:
                    if learning.get("type") == "solution":
                        result["apply"].append({
                            "id": learning.get("id"),
                            "problem": learning.get("problem", "")[:100],
                            "solution": learning.get("solution", "")[:200],
                            "confidence": learning.get("confidence", 0.7)
                        })

            # Search for antipatterns
            antipatterns = await self.mcp.learning(
                "get_antipatterns",
                context=task
            )

            if antipatterns.get("success") and antipatterns.get("antipatterns"):
                for ap in antipatterns["antipatterns"]:
                    result["avoid"].append({
                        "id": ap.get("id"),
                        "what_not_to_do": ap.get("what_not_to_do", ap.get("problem", ""))[:100],
                        "why": ap.get("why_it_failed", "")[:150],
                        "confidence": ap.get("confidence", 0.6)
                    })

            # Add agent-type specific tips
            if agent_type:
                tips = self._get_agent_tips(agent_type)
                result["tips"].extend(tips)

        except Exception as e:
            print(f"[LearningInjector] Error fetching learnings: {e}")

        return result

    def _get_agent_tips(self, agent_type: str) -> List[Dict]:
        """Get tips specific to an agent type."""
        tips = {
            "researcher": [
                {"tip": "Cite sources and file paths when reporting findings", "priority": "high"},
                {"tip": "If unsure, investigate before assuming", "priority": "medium"},
            ],
            "coder": [
                {"tip": "Always read existing code before modifying", "priority": "high"},
                {"tip": "Keep changes minimal and focused", "priority": "medium"},
                {"tip": "Test after making changes", "priority": "high"},
            ],
            "worker": [
                {"tip": "Break complex tasks into smaller steps", "priority": "medium"},
                {"tip": "Verify completion of each step", "priority": "medium"},
            ],
            "analyst": [
                {"tip": "Present data clearly with context", "priority": "medium"},
                {"tip": "Consider edge cases and exceptions", "priority": "medium"},
            ],
            "orchestrator": [
                {"tip": "Wait for child agents to complete before synthesizing", "priority": "high"},
                {"tip": "Provide clear, specific tasks to child agents", "priority": "high"},
            ]
        }

        return tips.get(agent_type, [])

    def inject_into_prompt(
        self,
        base_prompt: str,
        learnings: Dict[str, List],
        task: str
    ) -> str:
        """
        Add learnings section to an agent prompt.

        Args:
            base_prompt: The original agent prompt
            learnings: Dict from get_relevant_learnings()
            task: The task (for logging)

        Returns:
            Modified prompt with learnings injected
        """
        apply_list = learnings.get("apply", [])
        avoid_list = learnings.get("avoid", [])
        tips_list = learnings.get("tips", [])

        if not apply_list and not avoid_list and not tips_list:
            return base_prompt

        additions = "\n\n## LEARNINGS FROM PAST EXPERIENCE\n"
        additions += "The following insights come from previous similar tasks:\n"

        if apply_list:
            additions += "\n### What has worked before:\n"
            for learning in apply_list[:3]:
                problem = learning.get("problem", "").strip()
                solution = learning.get("solution", "").strip()
                if problem and solution:
                    additions += f"- **Problem**: {problem}\n"
                    additions += f"  **Solution**: {solution}\n\n"

        if avoid_list:
            additions += "\n### What to AVOID (known failures):\n"
            for ap in avoid_list[:3]:
                what = ap.get("what_not_to_do", "").strip()
                why = ap.get("why", "").strip()
                if what:
                    additions += f"- **DON'T**: {what}"
                    if why:
                        additions += f" — *{why}*"
                    additions += "\n"

        if tips_list:
            additions += "\n### Agent Tips:\n"
            for tip in tips_list[:2]:
                additions += f"- {tip.get('tip', '')}\n"

        # Log the injection for tracking
        self._log_injection(task, learnings)

        return base_prompt + additions

    def _log_injection(self, task: str, learnings: Dict):
        """Log which learnings were injected."""
        try:
            log_data = []
            if self.injections_log.exists():
                try:
                    with open(self.injections_log, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    log_data = data.get("injections", [])
                except:
                    pass

            # Extract learning IDs
            applied_ids = [l.get("id") for l in learnings.get("apply", []) if l.get("id")]
            avoided_ids = [l.get("id") for l in learnings.get("avoid", []) if l.get("id")]

            if applied_ids or avoided_ids:
                log_data.insert(0, {
                    "task_preview": task[:100],
                    "applied_learnings": applied_ids,
                    "avoided_learnings": avoided_ids,
                    "timestamp": datetime.now().isoformat(),
                    "outcome": None  # Will be updated when agent completes
                })

                # Keep last 200 entries
                log_data = log_data[:200]

                with open(self.injections_log, 'w', encoding='utf-8') as f:
                    json.dump({
                        "injections": log_data,
                        "updated_at": datetime.now().isoformat()
                    }, f, indent=2)

        except Exception as e:
            print(f"[LearningInjector] Failed to log injection: {e}")

    async def record_outcome(
        self,
        agent_id: str,
        success: bool,
        learnings_helped: bool = None
    ):
        """
        Record the outcome of an agent that had learnings injected.

        This allows us to track learning effectiveness and update confidence.

        Args:
            agent_id: The agent ID
            success: Whether the agent completed successfully
            learnings_helped: Whether the injected learnings helped (user feedback)
        """
        try:
            if not self.injections_log.exists():
                return

            with open(self.injections_log, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Find recent injection without outcome
            injections = data.get("injections", [])
            for inj in injections[:10]:  # Check recent injections
                if inj.get("outcome") is None:
                    inj["outcome"] = {
                        "success": success,
                        "learnings_helped": learnings_helped,
                        "recorded_at": datetime.now().isoformat()
                    }

                    # If learnings helped, reinforce them
                    if learnings_helped and self.mcp:
                        for learning_id in inj.get("applied_learnings", []):
                            try:
                                await self.mcp.record_learning(
                                    type="confirm",
                                    solution_id=learning_id
                                )
                            except:
                                pass

                    break

            with open(self.injections_log, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"[LearningInjector] Failed to record outcome: {e}")

    def get_stats(self) -> Dict:
        """Get statistics about learning injections."""
        stats = {
            "total_injections": 0,
            "successful_outcomes": 0,
            "learnings_helped_count": 0,
            "most_applied": [],
            "most_avoided": []
        }

        try:
            if not self.injections_log.exists():
                return stats

            with open(self.injections_log, 'r', encoding='utf-8') as f:
                data = json.load(f)

            injections = data.get("injections", [])
            stats["total_injections"] = len(injections)

            # Count outcomes
            learning_counts = {}
            for inj in injections:
                outcome = inj.get("outcome")
                if outcome:
                    if outcome.get("success"):
                        stats["successful_outcomes"] += 1
                    if outcome.get("learnings_helped"):
                        stats["learnings_helped_count"] += 1

                # Count learning usage
                for lid in inj.get("applied_learnings", []):
                    learning_counts[lid] = learning_counts.get(lid, 0) + 1

            # Sort by most used
            sorted_learnings = sorted(
                learning_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            stats["most_applied"] = sorted_learnings[:5]

        except Exception as e:
            print(f"[LearningInjector] Error getting stats: {e}")

        return stats


# Singleton instance
_injector_instance = None


def get_learning_injector(mcp_bridge=None, storage_path: Path = None) -> Optional[LearningInjector]:
    """Get or create the learning injector singleton."""
    global _injector_instance

    if _injector_instance is None and mcp_bridge and storage_path:
        _injector_instance = LearningInjector(mcp_bridge, storage_path)
    elif mcp_bridge and _injector_instance and not _injector_instance.mcp:
        _injector_instance.mcp = mcp_bridge

    return _injector_instance
