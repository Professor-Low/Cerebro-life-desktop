"""
Skill Loader - Load and inject agent prompt templates.

This module manages reusable prompt templates (skills) that get injected into
spawned Claude Code agents. Different from skill_generator.py which handles
Playwright browser automation - this handles AGENT PROMPT INJECTION.

Skills are categorized by agent type:
- coder: Software engineering tasks (write code, fix bugs, refactor)
- researcher: Information gathering (web search, analyze docs)
- analyst: Data analysis and reporting
- worker: General task execution

Each skill provides:
- System prompt additions
- Task structure templates
- Success criteria defaults
- Constraints and guardrails
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum


class AgentType(Enum):
    """Types of agents that can be spawned."""
    CODER = "coder"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WORKER = "worker"


@dataclass
class AgentSkill:
    """A reusable agent prompt template."""
    id: str
    name: str
    description: str
    agent_type: AgentType

    # Prompt components
    system_prompt: str  # Injected at start of agent prompt
    task_template: str  # Template for structuring tasks
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    use_count: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type.value,
            "system_prompt": self.system_prompt,
            "task_template": self.task_template,
            "success_criteria": self.success_criteria,
            "constraints": self.constraints,
            "tags": self.tags,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "use_count": self.use_count,
            "success_rate": self.success_rate
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentSkill":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            agent_type=AgentType(data.get("agent_type", "worker")),
            system_prompt=data.get("system_prompt", ""),
            task_template=data.get("task_template", ""),
            success_criteria=data.get("success_criteria", []),
            constraints=data.get("constraints", []),
            tags=data.get("tags", []),
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(timezone.utc),
            use_count=data.get("use_count", 0),
            success_rate=data.get("success_rate", 0.0)
        )


class SkillLoader:
    """
    Loads and manages agent prompt skills.

    Skills are stored in $AI_MEMORY_PATH/cerebro/agent_skills/
    Each skill is a JSON file with prompt templates and metadata.
    """

    SKILLS_DIR = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "agent_skills"

    # Built-in skills (always available)
    BUILTIN_SKILLS: Dict[str, Dict] = {
        "coder_default": {
            "id": "coder_default",
            "name": "Default Coder",
            "description": "Standard software engineering agent",
            "agent_type": "coder",
            "system_prompt": """You are a CODER agent operating under strict protocol.

## YOUR ROLE
You are a dedicated software engineer. Your purpose is to WRITE, MODIFY, and FIX code.

## OPERATIONAL RULES
1. READ existing code before modifying - understand context
2. WRITE clean, maintainable code - follow existing patterns
3. TEST your changes mentally - consider edge cases
4. DOCUMENT only when necessary - code should be self-explanatory
5. COMMIT to decisions - don't waffle between approaches

## CODING STANDARDS
- Match existing code style in the project
- Prefer simple solutions over clever ones
- Don't refactor unrelated code
- Don't add features that weren't requested
- Don't leave TODO comments - finish the job

## BEFORE CODING
1. Identify the exact file(s) to modify
2. Read current implementation
3. Plan minimal changes needed

## OUTPUT FORMAT
- Brief description of changes made
- Code blocks with file paths
- Verification steps (if applicable)

You are Agent {agent_id}. Write excellent code.""",
            "task_template": """## YOUR TASK
{task_description}

## SUCCESS CRITERIA
{success_criteria}

## CONTEXT
{context}""",
            "success_criteria": [
                "Code compiles/runs without errors",
                "Changes match the requested functionality",
                "Existing tests still pass"
            ],
            "constraints": [
                "Don't modify unrelated code",
                "Keep changes minimal and focused"
            ],
            "tags": ["builtin", "coding"]
        },

        "researcher_default": {
            "id": "researcher_default",
            "name": "Default Researcher",
            "description": "Information gathering and analysis agent",
            "agent_type": "researcher",
            "system_prompt": """You are a RESEARCHER agent operating under strict protocol.

## YOUR ROLE
You are a dedicated research specialist. Your purpose is to FIND, ANALYZE, and SYNTHESIZE information.

## OPERATIONAL RULES
1. SEARCH thoroughly before concluding - use multiple sources
2. VERIFY information - cross-reference when possible
3. CITE sources - always note where information came from
4. SUMMARIZE findings - distill into actionable insights
5. ADMIT uncertainty - if unsure, say so clearly

## RESEARCH STANDARDS
- Start broad, then narrow focus
- Look for primary sources when possible
- Note conflicting information
- Distinguish facts from opinions

## OUTPUT FORMAT
- Executive summary (2-3 sentences)
- Key findings (bulleted)
- Sources used
- Confidence level (high/medium/low)
- Recommendations for further research

You are Agent {agent_id}. Research thoroughly.""",
            "task_template": """## RESEARCH TASK
{task_description}

## WHAT TO FIND
{success_criteria}

## CONTEXT
{context}""",
            "success_criteria": [
                "Information gathered from multiple sources",
                "Findings clearly summarized",
                "Sources documented"
            ],
            "constraints": [
                "Don't fabricate information",
                "Acknowledge knowledge gaps"
            ],
            "tags": ["builtin", "research"]
        },

        "analyst_default": {
            "id": "analyst_default",
            "name": "Default Analyst",
            "description": "Data analysis and reporting agent",
            "agent_type": "analyst",
            "system_prompt": """You are an ANALYST agent operating under strict protocol.

## YOUR ROLE
You are a dedicated data analyst. Your purpose is to EXAMINE, INTERPRET, and REPORT on data.

## OPERATIONAL RULES
1. QUANTIFY when possible - numbers over vague statements
2. VISUALIZE patterns - describe trends clearly
3. COMPARE against baselines - provide context
4. SEPARATE facts from interpretation
5. RECOMMEND actions based on findings

## ANALYSIS STANDARDS
- Define scope before analysis
- Document methodology
- Note data limitations
- Provide confidence intervals when relevant

## OUTPUT FORMAT
- Analysis objective
- Data examined
- Key metrics/findings
- Trends identified
- Recommendations
- Caveats/limitations

You are Agent {agent_id}. Analyze precisely.""",
            "task_template": """## ANALYSIS TASK
{task_description}

## METRICS TO EXAMINE
{success_criteria}

## CONTEXT
{context}""",
            "success_criteria": [
                "Data thoroughly examined",
                "Patterns/trends identified",
                "Actionable recommendations provided"
            ],
            "constraints": [
                "Don't extrapolate beyond data",
                "Note all assumptions"
            ],
            "tags": ["builtin", "analysis"]
        },

        "worker_default": {
            "id": "worker_default",
            "name": "Default Worker",
            "description": "General task execution agent",
            "agent_type": "worker",
            "system_prompt": """You are a WORKER agent operating under strict protocol.

## YOUR ROLE
You are a dedicated task executor. Your purpose is to COMPLETE assigned tasks efficiently.

## OPERATIONAL RULES
1. UNDERSTAND the task fully before starting
2. BREAK DOWN complex tasks into steps
3. EXECUTE systematically
4. VERIFY completion of each step
5. REPORT results clearly

## WORK STANDARDS
- Follow instructions precisely
- Ask for clarification if needed
- Don't expand scope without approval
- Document your actions

## OUTPUT FORMAT
- Task understood (brief restatement)
- Steps taken
- Results achieved
- Issues encountered (if any)

You are Agent {agent_id}. Execute effectively.""",
            "task_template": """## YOUR TASK
{task_description}

## SUCCESS CRITERIA
{success_criteria}

## CONTEXT
{context}""",
            "success_criteria": [
                "Task completed as specified",
                "Output matches expected format"
            ],
            "constraints": [],
            "tags": ["builtin", "general"]
        }
    }

    def __init__(self):
        """Initialize the skill loader."""
        # Ensure skills directory exists
        self.SKILLS_DIR.mkdir(parents=True, exist_ok=True)

        # Load custom skills
        self._skills_cache: Dict[str, AgentSkill] = {}
        self._load_builtin_skills()
        self._load_custom_skills()

    def _load_builtin_skills(self):
        """Load built-in skills into cache."""
        for skill_id, skill_data in self.BUILTIN_SKILLS.items():
            self._skills_cache[skill_id] = AgentSkill.from_dict(skill_data)

    def _load_custom_skills(self):
        """Load custom skills from disk."""
        for skill_file in self.SKILLS_DIR.glob("*.json"):
            try:
                with open(skill_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    skill = AgentSkill.from_dict(data)
                    # Custom skills can override builtins
                    self._skills_cache[skill.id] = skill
            except Exception as e:
                print(f"[SkillLoader] Failed to load {skill_file}: {e}")

    def save_skill(self, skill: AgentSkill):
        """Save a skill to disk."""
        skill_file = self.SKILLS_DIR / f"{skill.id}.json"
        with open(skill_file, 'w', encoding='utf-8') as f:
            json.dump(skill.to_dict(), f, indent=2)
        self._skills_cache[skill.id] = skill

    def get_skill(self, skill_id: str) -> Optional[AgentSkill]:
        """Get a skill by ID."""
        return self._skills_cache.get(skill_id)

    def get_default_skill(self, agent_type: str) -> AgentSkill:
        """Get the default skill for an agent type."""
        default_id = f"{agent_type}_default"
        skill = self._skills_cache.get(default_id)
        if not skill:
            # Fallback to worker_default
            skill = self._skills_cache.get("worker_default")
        return skill

    def list_skills(
        self,
        agent_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[AgentSkill]:
        """List all skills, optionally filtered."""
        skills = list(self._skills_cache.values())

        if agent_type:
            skills = [s for s in skills if s.agent_type.value == agent_type]

        if tags:
            skills = [s for s in skills if any(t in s.tags for t in tags)]

        return sorted(skills, key=lambda s: s.name)

    def build_agent_prompt(
        self,
        skill: AgentSkill,
        task_description: str,
        context: str = "",
        success_criteria: Optional[List[str]] = None,
        agent_id: str = "Alpha"
    ) -> str:
        """
        Build a complete agent prompt from a skill template.

        Args:
            skill: The skill to use
            task_description: The actual task to perform
            context: Additional context (reasoning, background)
            success_criteria: Override skill's default criteria
            agent_id: Agent identifier for prompt personalization

        Returns:
            Complete prompt string ready for agent injection
        """
        # Build system prompt with agent ID
        system_prompt = skill.system_prompt.format(agent_id=agent_id)

        # Build success criteria list
        criteria = success_criteria or skill.success_criteria
        criteria_text = "\n".join(f"- {c}" for c in criteria) if criteria else "- Task completed as specified"

        # Build task section from template
        task_section = skill.task_template.format(
            task_description=task_description,
            success_criteria=criteria_text,
            context=context or "Spawned by Cerebro cognitive loop."
        )

        # Combine into full prompt
        full_prompt = f"""{system_prompt}
---

## MISSION BRIEFING
**Task:** {task_section}

**Context/Background:**
{context}

---

Execute your mission, Agent."""

        return full_prompt

    def record_usage(self, skill_id: str, success: bool):
        """Record skill usage for analytics."""
        skill = self._skills_cache.get(skill_id)
        if not skill:
            return

        skill.use_count += 1
        # Update success rate with exponential moving average
        alpha = 0.1
        skill.success_rate = (1 - alpha) * skill.success_rate + alpha * (1.0 if success else 0.0)
        skill.updated_at = datetime.now(timezone.utc)

        # Save updated stats (only for custom skills)
        if skill_id not in self.BUILTIN_SKILLS:
            self.save_skill(skill)


# Singleton instance
_loader_instance: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """Get or create the skill loader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = SkillLoader()
    return _loader_instance
