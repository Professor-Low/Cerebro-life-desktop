"""
Adaptive Explorer for LLM-Guided Browser Exploration.

This module provides capabilities for autonomous website exploration
using LLM guidance, learning from successful explorations to create
reusable skills.

Part of the Adaptive Browser Learning System for Cerebro.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from pathlib import Path
import json
import uuid
import asyncio

from .page_understanding import PageUnderstanding, PageState
from .element_fingerprint import SelfHealingLocator, FingerprintGenerator
from .ollama_client import ChatMessage


@dataclass
class ExplorationStep:
    """Records a single step in an exploration session."""
    step_number: int
    timestamp: str
    page_url: str
    page_state_summary: str          # Compressed page state
    action: str                       # click, fill, navigate, wait, done, stuck
    selector: Optional[str] = None
    value: Optional[str] = None       # For fill actions
    reasoning: str = ""               # LLM's reasoning for this action
    result: str = ""                  # success, failed, unexpected
    error: Optional[str] = None
    fingerprint: Optional[Dict] = None  # Element fingerprint if applicable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "timestamp": self.timestamp,
            "page_url": self.page_url,
            "page_state_summary": self.page_state_summary,
            "action": self.action,
            "selector": self.selector,
            "value": self.value,
            "reasoning": self.reasoning,
            "result": self.result,
            "error": self.error,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExplorationStep":
        return cls(**data)


@dataclass
class ExplorationSession:
    """
    Records a complete exploration session.
    """
    session_id: str
    goal: str
    start_url: str
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: Optional[str] = None
    steps: List[ExplorationStep] = field(default_factory=list)
    status: str = "exploring"         # exploring, succeeded, failed, stuck, timeout
    final_result: Optional[Any] = None
    parameters_discovered: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "start_url": self.start_url,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "final_result": self.final_result,
            "parameters_discovered": self.parameters_discovered,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExplorationSession":
        steps = [ExplorationStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            session_id=data["session_id"],
            goal=data["goal"],
            start_url=data["start_url"],
            started_at=data.get("started_at", datetime.utcnow().isoformat()),
            ended_at=data.get("ended_at"),
            steps=steps,
            status=data.get("status", "exploring"),
            final_result=data.get("final_result"),
            parameters_discovered=data.get("parameters_discovered", {}),
            metadata=data.get("metadata", {}),
        )

    def add_step(self, step: ExplorationStep):
        self.steps.append(step)

    def get_recent_steps(self, n: int = 5) -> List[ExplorationStep]:
        return self.steps[-n:]

    def get_failed_approaches(self) -> List[str]:
        """Get actions that failed for reflexion."""
        return [
            f"{s.action} on {s.selector}: {s.error}"
            for s in self.steps
            if s.result == "failed" and s.error
        ]


class AdaptiveExplorer:
    """
    LLM-guided browser exploration that learns and creates skills.

    Uses a Stagehand-inspired approach: observe page state, let LLM decide
    next action, execute with retry/heal, record for skill creation.
    """

    # LLM prompt template for deciding next action
    EXPLORATION_PROMPT = '''You are exploring a website to accomplish a goal.

GOAL: {goal}

CURRENT PAGE:
URL: {url}
Title: {title}
Type: {page_type}

{page_state}

EXPLORATION HISTORY (last {history_count} steps):
{history}

FAILED APPROACHES TO AVOID:
{failed_approaches}

Based on the current page state and your goal, decide the next action.

Respond with a JSON object:
{{
    "action": "click" | "fill" | "navigate" | "select" | "wait" | "scroll" | "done" | "stuck",
    "element_index": <number if click/fill/select>,
    "selector": "<CSS selector if action targets element>",
    "value": "<value for fill/select actions or URL for navigate>",
    "reasoning": "<brief explanation of why this action helps achieve the goal>",
    "goal_progress": "<what progress has been made toward the goal>",
    "is_goal_achieved": <true if goal is fully achieved, false otherwise>
}}

Rules:
- Use element_index to reference elements from the INTERACTABLE ELEMENTS list
- For "fill", always provide a value (can be a parameter placeholder like {{username}})
- For "navigate", provide the full URL in value
- Use "wait" if page is loading or you expect content to appear
- Use "done" ONLY when the goal is fully achieved
- Use "stuck" if you cannot proceed and need human help
- Avoid repeating failed approaches
- After submitting a search or query, you MUST scroll down and read the results before saying "done"
- For search/lookup goals: scroll the results page, then set is_goal_achieved=true and put the top results (titles + brief info) as a JSON array in goal_progress like: [{{"title": "...", "detail": "..."}}, ...]
- NEVER say "done" immediately after a search submission — always scroll at least once first
- If the page shows a list of results (videos, products, articles), extract the top 5-8 items
'''

    def __init__(
        self,
        ollama_client,
        page_understanding: Optional[PageUnderstanding] = None,
        fingerprint_locator: Optional[SelfHealingLocator] = None,
        storage_path: Optional[Path] = None,
        reflexion_engine=None,
        model: str = "qwen3:32b",
    ):
        """
        Initialize the adaptive explorer.

        Args:
            ollama_client: Ollama client for LLM calls
            page_understanding: PageUnderstanding instance
            fingerprint_locator: SelfHealingLocator instance
            storage_path: Path to store exploration sessions
            reflexion_engine: Optional ReflexionEngine for learning from failures
            model: LLM model to use for exploration decisions
        """
        self.ollama = ollama_client
        self.page_understanding = page_understanding or PageUnderstanding()
        self.locator = fingerprint_locator or SelfHealingLocator()
        self.fingerprint_gen = FingerprintGenerator(self.locator)
        self.storage_path = storage_path or Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "explorations"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.reflexion = reflexion_engine
        self.model = model

    async def explore(
        self,
        page,
        goal: str,
        start_url: str,
        max_steps: int = 20,
        timeout_seconds: int = 300,
        on_step: Optional[Callable] = None,
    ) -> ExplorationSession:
        """
        DEPRECATED (Cerebro v2.0): Website exploration is now handled by Claude Code
        browser agents that control the shared Chrome via HTTP endpoints.
        This method's LLM loop (Qwen) is no longer used.

        The browser agent prompt + HTTP control API replaces this entirely.
        PageUnderstanding is still used by the HTTP endpoints.

        Args:
            page: Playwright page object (browser should already be launched)
            goal: What to accomplish
            start_url: URL to start exploration from
            max_steps: Maximum exploration steps before giving up
            timeout_seconds: Total timeout for exploration
            on_step: Optional callback called after each step

        Returns:
            ExplorationSession with full history and results
        """
        session = ExplorationSession(
            session_id=str(uuid.uuid4())[:8],
            goal=goal,
            start_url=start_url,
        )

        start_time = datetime.utcnow()

        try:
            # Navigate to start URL
            await page.goto(start_url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(1)  # Brief settle time

            step_number = 0
            while step_number < max_steps:
                # Check timeout
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    session.status = "timeout"
                    break

                # Analyze current page
                page_state = await self.page_understanding.analyze_page(page)
                compressed_state = self.page_understanding.compress_for_llm(page_state)

                # Get LLM decision
                decision = await self._get_next_action(
                    goal=goal,
                    page_state=page_state,
                    compressed_state=compressed_state,
                    session=session,
                )

                if not decision:
                    session.status = "stuck"
                    break

                # Create step record
                step = ExplorationStep(
                    step_number=step_number,
                    timestamp=datetime.utcnow().isoformat(),
                    page_url=page.url,
                    page_state_summary=compressed_state[:500],
                    action=decision.get("action", "unknown"),
                    selector=decision.get("selector"),
                    value=decision.get("value"),
                    reasoning=decision.get("reasoning", ""),
                )

                # Check if goal achieved
                if decision.get("is_goal_achieved") or decision.get("action") == "done":
                    step.result = "success"
                    session.add_step(step)
                    session.status = "succeeded"
                    session.final_result = decision.get("goal_progress")
                    break

                # Check if stuck
                if decision.get("action") == "stuck":
                    step.result = "stuck"
                    session.add_step(step)
                    session.status = "stuck"
                    break

                # Execute the action
                try:
                    await self._execute_action(page, decision, page_state, step)
                    step.result = "success"

                    # Capture fingerprint for successful interactions
                    if step.selector and step.action in ("click", "fill", "select"):
                        fp = await self.locator.capture_fingerprint(page, step.selector)
                        if fp:
                            step.fingerprint = fp.to_dict()

                except Exception as e:
                    step.result = "failed"
                    step.error = str(e)

                session.add_step(step)

                # Callback
                if on_step:
                    await on_step(session, step) if asyncio.iscoroutinefunction(on_step) else on_step(session, step)

                # Wait for page to settle after action
                await asyncio.sleep(0.5)
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass  # Timeout is OK, page might not have network activity

                step_number += 1

            if step_number >= max_steps and session.status == "exploring":
                session.status = "failed"

        except Exception as e:
            session.status = "failed"
            session.metadata["error"] = str(e)

        session.ended_at = datetime.utcnow().isoformat()

        # Save session
        self._save_session(session)

        # Learn from failure if reflexion engine available
        if session.status == "failed" and self.reflexion:
            await self._reflect_on_failure(session)

        return session

    async def _get_next_action(
        self,
        goal: str,
        page_state: PageState,
        compressed_state: str,
        session: ExplorationSession,
    ) -> Optional[Dict]:
        """Get next action from LLM."""

        # Build history summary
        recent = session.get_recent_steps(5)
        history_lines = []
        for s in recent:
            history_lines.append(f"Step {s.step_number}: {s.action}")
            if s.selector:
                history_lines.append(f"  Target: {s.selector}")
            if s.value:
                history_lines.append(f"  Value: {s.value}")
            history_lines.append(f"  Result: {s.result}")
            if s.error:
                history_lines.append(f"  Error: {s.error}")

        history = "\n".join(history_lines) if history_lines else "No steps taken yet"

        # Build failed approaches
        failed = session.get_failed_approaches()
        failed_text = "\n".join(f"- {f}" for f in failed[-5:]) if failed else "None"

        prompt = self.EXPLORATION_PROMPT.format(
            goal=goal,
            url=page_state.url,
            title=page_state.title,
            page_type=page_state.page_type,
            page_state=compressed_state,
            history_count=len(recent),
            history=history,
            failed_approaches=failed_text,
        )

        try:
            # Use the correct OllamaClient API with ChatMessage objects
            messages = [ChatMessage(role="user", content=prompt)]
            print(f"[AdaptiveExplorer] Asking LLM for next action (goal: {goal[:60]}...)")
            response = await self.ollama.chat(messages=messages, thinking=False)

            # Extract content from OllamaResponse
            content = response.content if hasattr(response, 'content') else str(response)
            print(f"[AdaptiveExplorer] LLM raw response ({len(content)} chars): {content[:200]}...")

            # Parse JSON response - handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Also try to extract JSON object if response has extra text
            if not content.startswith("{"):
                json_start = content.find("{")
                if json_start >= 0:
                    content = content[json_start:]
                    # Find matching closing brace
                    depth = 0
                    for i, c in enumerate(content):
                        if c == "{":
                            depth += 1
                        elif c == "}":
                            depth -= 1
                            if depth == 0:
                                content = content[:i+1]
                                break

            parsed = json.loads(content)
            print(f"[AdaptiveExplorer] Parsed action: {parsed.get('action')} | element: {parsed.get('element_index')} | value: {str(parsed.get('value', ''))[:50]}")
            return parsed

        except json.JSONDecodeError as e:
            print(f"[AdaptiveExplorer] Failed to parse LLM response: {e}")
            print(f"[AdaptiveExplorer] Raw content was: {content[:300] if content else 'EMPTY'}")
            return None
        except Exception as e:
            print(f"[AdaptiveExplorer] LLM error: {e}")
            return None

    async def _execute_action(
        self,
        page,
        decision: Dict,
        page_state: PageState,
        step: ExplorationStep,
    ):
        """Execute the decided action on the page."""
        action = decision.get("action")
        selector = decision.get("selector")
        value = decision.get("value")
        element_index = decision.get("element_index")

        # Resolve selector from element index if needed
        if element_index is not None and not selector:
            for el in page_state.interactable_elements:
                if el.index == element_index:
                    selector = el.selector
                    step.selector = selector
                    break

        if action == "click":
            if not selector:
                raise ValueError("Click action requires selector")
            await page.click(selector, timeout=10000)

        elif action == "fill":
            if not selector or not value:
                raise ValueError("Fill action requires selector and value")
            await page.fill(selector, value, timeout=10000)

        elif action == "select":
            if not selector or not value:
                raise ValueError("Select action requires selector and value")
            await page.select_option(selector, value, timeout=10000)

        elif action == "navigate":
            if not value:
                raise ValueError("Navigate action requires URL in value")
            await page.goto(value, wait_until="networkidle", timeout=30000)

        elif action == "wait":
            wait_time = int(value) if value and value.isdigit() else 2000
            await asyncio.sleep(wait_time / 1000)

        elif action == "scroll":
            direction = value or "down"
            if direction == "down":
                await page.evaluate("window.scrollBy(0, 500)")
            elif direction == "up":
                await page.evaluate("window.scrollBy(0, -500)")

        elif action in ("done", "stuck"):
            pass  # No execution needed

        else:
            raise ValueError(f"Unknown action: {action}")

    async def convert_to_skill(
        self,
        session: ExplorationSession,
        skill_name: str,
        skill_description: str,
        parameterize: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Convert a successful exploration session into a reusable skill.

        Args:
            session: Completed exploration session
            skill_name: Name for the new skill
            skill_description: Description of what the skill does
            parameterize: List of values to turn into parameters

        Returns:
            Skill dictionary compatible with skill_generator.py
        """
        if session.status != "succeeded":
            raise ValueError(f"Cannot convert failed session to skill: {session.status}")

        parameterize = parameterize or []

        # Convert steps to skill steps
        skill_steps = []
        param_map = {}  # value -> param name

        for step in session.steps:
            if step.action in ("wait", "done", "stuck"):
                continue  # Skip non-actionable steps

            skill_step = {
                "action": step.action,
                "selector": step.selector,
                "description": step.reasoning,
            }

            # Handle parameterization
            if step.value:
                # Check if this value should be parameterized
                for param_value in parameterize:
                    if param_value in (step.value or ""):
                        # Generate param name
                        param_name = self._generate_param_name(param_value, param_map)
                        param_map[param_value] = param_name
                        skill_step["value"] = f"{{{param_name}}}"
                        break
                else:
                    skill_step["value"] = step.value

            # Include fingerprint for self-healing
            if step.fingerprint:
                skill_step["fingerprint"] = step.fingerprint

            skill_steps.append(skill_step)

        # Build skill object
        skill = {
            "skill_id": f"skill_{session.session_id}_{skill_name.lower().replace(' ', '_')}",
            "name": skill_name,
            "description": skill_description,
            "start_url": session.start_url,
            "steps": skill_steps,
            "parameters": list(param_map.values()),
            "parameter_descriptions": {v: f"Value for {v}" for v in param_map.values()},
            "created_from_exploration": session.session_id,
            "created_at": datetime.utcnow().isoformat(),
            "status": "unverified",  # Needs verification before trusted
            "version": 1,
        }

        # Save skill
        skill_path = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "skills" / f"{skill['skill_id']}.json"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        with open(skill_path, "w") as f:
            json.dump(skill, f, indent=2)

        return skill

    def _generate_param_name(self, value: str, existing: Dict[str, str]) -> str:
        """Generate a parameter name from a value."""
        # Simple heuristics
        value_lower = value.lower()

        if "@" in value:
            name = "email"
        elif any(p in value_lower for p in ["password", "pass", "pwd"]):
            name = "password"
        elif any(p in value_lower for p in ["user", "name", "login"]):
            name = "username"
        elif value.isdigit():
            name = "number"
        else:
            name = "input"

        # Ensure uniqueness
        base_name = name
        counter = 1
        while name in existing.values():
            name = f"{base_name}_{counter}"
            counter += 1

        return name

    def _save_session(self, session: ExplorationSession):
        """Save exploration session to storage."""
        filepath = self.storage_path / f"{session.session_id}.json"
        with open(filepath, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def load_session(self, session_id: str) -> Optional[ExplorationSession]:
        """Load exploration session from storage."""
        filepath = self.storage_path / f"{session_id}.json"
        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            data = json.load(f)
        return ExplorationSession.from_dict(data)

    async def _reflect_on_failure(self, session: ExplorationSession):
        """Use reflexion engine to learn from failed exploration."""
        if not self.reflexion:
            return

        # Create failure summary for reflexion
        failure_summary = {
            "goal": session.goal,
            "start_url": session.start_url,
            "steps_taken": len(session.steps),
            "final_status": session.status,
            "failed_steps": [
                {"step": s.step_number, "action": s.action, "error": s.error}
                for s in session.steps if s.result == "failed"
            ],
            "last_page_url": session.steps[-1].page_url if session.steps else session.start_url,
        }

        # Call reflexion engine (interface depends on implementation)
        try:
            await self.reflexion.reflect_on_exploration_failure(
                session_id=session.session_id,
                failure_summary=failure_summary,
            )
        except Exception as e:
            print(f"[AdaptiveExplorer] Reflexion failed: {e}")


class ExplorationManager:
    """
    High-level manager for exploration sessions.

    Handles skill lookup, exploration triggering, and caching.
    """

    def __init__(
        self,
        explorer: AdaptiveExplorer,
        skill_storage_path: Optional[Path] = None,
    ):
        self.explorer = explorer
        self.skill_storage = skill_storage_path or Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "skills"

    def find_existing_skill(self, goal: str, url_pattern: str = None) -> Optional[Dict]:
        """
        Find an existing skill that matches the goal.

        Args:
            goal: The goal to accomplish
            url_pattern: Optional URL pattern to match

        Returns:
            Skill dict if found, None otherwise
        """
        if not self.skill_storage.exists():
            return None

        goal_lower = goal.lower()
        best_match = None
        best_score = 0

        for skill_file in self.skill_storage.glob("*.json"):
            try:
                with open(skill_file, "r") as f:
                    skill = json.load(f)

                # Skip unverified or failed skills
                if skill.get("status") not in ("verified", "unverified"):
                    continue

                # Score based on name/description match
                name = skill.get("name", "").lower()
                desc = skill.get("description", "").lower()

                score = 0
                for word in goal_lower.split():
                    if word in name:
                        score += 2
                    if word in desc:
                        score += 1

                # Bonus for URL match
                if url_pattern and url_pattern in skill.get("start_url", ""):
                    score += 3

                if score > best_score:
                    best_score = score
                    best_match = skill

            except Exception:
                continue

        return best_match if best_score >= 2 else None

    async def explore_or_use_skill(
        self,
        page,
        goal: str,
        start_url: str,
        parameters: Optional[Dict[str, str]] = None,
        force_explore: bool = False,
    ) -> Dict[str, Any]:
        """
        Either use an existing skill or explore to create one.

        Args:
            page: Playwright page
            goal: Goal to accomplish
            start_url: URL to start from
            parameters: Parameters for skill execution
            force_explore: Force new exploration even if skill exists

        Returns:
            Result with skill_used, session, or execution result
        """
        result = {
            "method": None,
            "success": False,
            "skill": None,
            "session": None,
            "error": None,
        }

        # Try to find existing skill
        if not force_explore:
            existing_skill = self.find_existing_skill(goal, start_url)
            if existing_skill:
                result["method"] = "existing_skill"
                result["skill"] = existing_skill
                # Skill execution would happen here via skill_generator
                # For now, return skill for caller to execute
                return result

        # No skill found, explore
        session = await self.explorer.explore(page, goal, start_url)
        result["method"] = "exploration"
        result["session"] = session.to_dict()

        if session.status == "succeeded":
            result["success"] = True
            # Auto-convert to skill for future use
            try:
                skill = await self.explorer.convert_to_skill(
                    session,
                    skill_name=f"Auto: {goal[:50]}",
                    skill_description=f"Automatically generated skill for: {goal}",
                )
                result["skill"] = skill

                # ============================================================
                # Wire Skill Verification After Creation
                # ============================================================
                if skill and skill.get("skill_id"):
                    try:
                        from .skill_verifier import SkillVerifier
                        from .skill_generator import get_skill_generator

                        skill_gen = get_skill_generator()
                        verifier = SkillVerifier(
                            skill_generator=skill_gen,
                            self_healing_locator=self.explorer.locator if hasattr(self.explorer, 'locator') else None
                        )

                        # Use the exploration page if available for full verification
                        verify_page = page if page else None
                        verification = await verifier.verify_skill(
                            skill_id=skill["skill_id"],
                            page=verify_page,
                            dry_run=(verify_page is None),  # Full verify with page, dry run without
                            auto_heal=True  # Attempt to heal broken selectors
                        )

                        result["skill_verified"] = verification.success
                        result["verification_result"] = {
                            "success": verification.success,
                            "steps_passed": verification.steps_passed,
                            "steps_total": verification.steps_total,
                            "healed_selectors": len(verification.healed_selectors) if verification.healed_selectors else 0
                        }

                        # Update skill status based on verification
                        if verification.success:
                            skill_obj = skill_gen.get_skill(skill["skill_id"])
                            if skill_obj:
                                from .skill_generator import SkillStatus
                                skill_obj.status = SkillStatus.VERIFIED
                                skill_gen._save_skill(skill_obj)

                    except Exception as verify_error:
                        print(f"[AdaptiveExplorer] Skill verification failed for {skill.get('skill_id', '?')}: {verify_error}")
                        result["verification_error"] = str(verify_error)

            except Exception as e:
                result["skill_creation_error"] = str(e)
        else:
            result["error"] = f"Exploration {session.status}: {session.metadata.get('error', 'Unknown error')}"

        return result

    async def explore_or_execute(
        self,
        goal: str,
        start_url: str,
        skill_name_hint: Optional[str] = None,
        parameters: Optional[Dict[str, str]] = None,
        force_explore: bool = False,
        create_skill: bool = True,
    ) -> Dict[str, Any]:
        """
        High-level method that manages its own browser session.

        This is the main entry point for the OODA engine to use.

        Args:
            goal: Goal to accomplish
            start_url: URL to start from
            skill_name_hint: Optional name for the skill to create
            parameters: Parameters for skill execution
            force_explore: Force new exploration even if skill exists
            create_skill: Whether to create a reusable skill

        Returns:
            Result dict with success, result, skill_created, etc.
        """
        from playwright.async_api import async_playwright

        result = {
            "success": False,
            "result": None,
            "skill_created": False,
            "skill_id": None,
            "steps_taken": 0,
            "error": None,
        }

        async with async_playwright() as p:
            try:
                # Launch browser
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = await context.new_page()

                # Check for existing skill first
                if not force_explore:
                    existing_skill = self.find_existing_skill(goal, start_url)
                    if existing_skill:
                        # Use existing skill via skill_generator
                        try:
                            from .skill_generator import get_skill_generator
                            skill_gen = get_skill_generator()
                            exec_result = await skill_gen.execute_skill(
                                existing_skill["skill_id"],
                                page=page,
                                parameters=parameters or {},
                            )
                            result["success"] = exec_result.get("success", False)
                            result["result"] = exec_result.get("result")
                            result["skill_id"] = existing_skill["skill_id"]
                            result["method"] = "existing_skill"
                            await browser.close()
                            return result
                        except Exception as e:
                            # Skill execution failed, fall through to exploration
                            print(f"[ExplorationManager] Existing skill failed: {e}, exploring...")

                # Explore the website
                session = await self.explorer.explore(page, goal, start_url)
                result["steps_taken"] = len(session.steps)

                if session.status == "succeeded":
                    result["success"] = True
                    result["result"] = session.final_result

                    # Create skill if requested
                    if create_skill:
                        try:
                            skill_name = skill_name_hint or f"Auto: {goal[:50]}"
                            skill = await self.explorer.convert_to_skill(
                                session,
                                skill_name=skill_name,
                                skill_description=f"Automatically generated: {goal}",
                            )
                            result["skill_created"] = True
                            result["skill_id"] = skill.get("skill_id")

                            # Wire Skill Verification After Creation
                            if skill.get("skill_id"):
                                try:
                                    from .skill_verifier import SkillVerifier
                                    from .skill_generator import get_skill_generator, SkillStatus

                                    skill_gen = get_skill_generator()
                                    verifier = SkillVerifier(
                                        skill_generator=skill_gen,
                                        self_healing_locator=self.explorer.locator if hasattr(self.explorer, 'locator') else None
                                    )

                                    verification = await verifier.verify_skill(
                                        skill_id=skill["skill_id"],
                                        page=page,  # Use existing page for verification
                                        dry_run=True,
                                        auto_heal=True
                                    )

                                    result["skill_verified"] = verification.success
                                    if verification.healed_selectors:
                                        result["healed_selectors"] = len(verification.healed_selectors)

                                    # Update skill status based on verification
                                    if verification.success:
                                        skill_obj = skill_gen.get_skill(skill["skill_id"])
                                        if skill_obj:
                                            skill_obj.status = SkillStatus.VERIFIED
                                            skill_gen._save_skill(skill_obj)

                                except Exception as verify_error:
                                    result["verification_error"] = str(verify_error)

                        except Exception as e:
                            result["skill_creation_error"] = str(e)
                else:
                    result["error"] = f"Exploration {session.status}"
                    if session.metadata.get("error"):
                        result["error"] += f": {session.metadata['error']}"

                await browser.close()

            except Exception as e:
                result["error"] = str(e)

        return result


# Singleton instance
_exploration_manager: Optional[ExplorationManager] = None


def get_exploration_manager() -> ExplorationManager:
    """
    Get or create the singleton ExplorationManager.

    Lazily initializes all required components.
    """
    global _exploration_manager

    if _exploration_manager is None:
        # Initialize dependencies
        from .page_understanding import PageUnderstanding
        from .element_fingerprint import SelfHealingLocator
        from .ollama_client import OllamaClient
        from .reflexion_engine import get_goal_reflexion_engine

        page_understanding = PageUnderstanding()
        fingerprint_locator = SelfHealingLocator()
        ollama = OllamaClient()
        reflexion = get_goal_reflexion_engine()

        # Create explorer with correct parameter names
        explorer = AdaptiveExplorer(
            ollama_client=ollama,
            page_understanding=page_understanding,
            fingerprint_locator=fingerprint_locator,
            reflexion_engine=reflexion,
        )

        # Create manager
        _exploration_manager = ExplorationManager(explorer=explorer)

    return _exploration_manager
