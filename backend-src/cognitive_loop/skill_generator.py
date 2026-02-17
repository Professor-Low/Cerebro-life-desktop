"""
Skill Generator - Cerebro learns to automate tasks using Playwright.

Based on Voyager pattern: OBSERVE → GENERATE → VERIFY → STORE
Cerebro can watch what you do, generate automation scripts, and save them as reusable skills.

Uses Playwright for browser automation, integrated with the OODA cognitive loop.
"""

import os
import json
import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum

# Playwright imports - use SYNC API to avoid Windows asyncio issues
try:
    from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Self-healing imports (local)
try:
    from .element_fingerprint import SelfHealingLocator, ElementFingerprint, FingerprintGenerator
    SELF_HEALING_AVAILABLE = True
except ImportError:
    SELF_HEALING_AVAILABLE = False


class SkillStatus(Enum):
    """Status of a generated skill."""
    DRAFT = "draft"           # Just generated, not tested
    TESTING = "testing"       # Currently being tested
    VERIFIED = "verified"     # Passed tests, ready to use
    FAILED = "failed"         # Failed verification
    DEPRECATED = "deprecated" # Replaced by newer version


@dataclass
class SkillStep:
    """A single step in a skill."""
    action: str                          # navigate, click, fill, wait, extract, screenshot
    selector: Optional[str] = None       # CSS/XPath selector or accessibility label
    value: Optional[str] = None          # Value to fill or URL to navigate
    description: str = ""                # Human-readable description
    wait_for: Optional[str] = None       # Condition to wait for after action
    timeout_ms: int = 5000               # Max wait time
    fingerprint: Optional[Dict] = None   # Element fingerprint for self-healing

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "selector": self.selector,
            "value": self.value,
            "description": self.description,
            "wait_for": self.wait_for,
            "timeout_ms": self.timeout_ms,
            "fingerprint": self.fingerprint
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SkillStep":
        return cls(
            action=data.get("action", ""),
            selector=data.get("selector"),
            value=data.get("value"),
            description=data.get("description", ""),
            wait_for=data.get("wait_for"),
            timeout_ms=data.get("timeout_ms", 5000),
            fingerprint=data.get("fingerprint")
        )


@dataclass
class Skill:
    """A reusable automation skill."""
    id: str
    name: str
    description: str
    steps: List[SkillStep]
    parameters: List[str] = field(default_factory=list)  # Parameterized values
    tags: List[str] = field(default_factory=list)
    status: SkillStatus = SkillStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success_count: int = 0
    fail_count: int = 0
    last_used: Optional[datetime] = None
    version: int = 1
    parent_id: Optional[str] = None  # If this is an evolved version

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "parameters": self.parameters,
            "tags": self.tags,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "version": self.version,
            "parent_id": self.parent_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Skill":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            steps=[SkillStep.from_dict(s) for s in data.get("steps", [])],
            parameters=data.get("parameters", []),
            tags=data.get("tags", []),
            status=SkillStatus(data.get("status", "draft")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(timezone.utc),
            success_count=data.get("success_count", 0),
            fail_count=data.get("fail_count", 0),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            version=data.get("version", 1),
            parent_id=data.get("parent_id")
        )

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class ExecutionResult:
    """Result of executing a skill."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    screenshots: List[bytes] = field(default_factory=list)
    duration_ms: float = 0
    steps_completed: int = 0
    total_steps: int = 0


class SkillGenerator:
    """
    Generates and executes browser automation skills.

    Capabilities:
    - Record user workflows via Playwright
    - Generate skills from natural language descriptions
    - Execute skills with parameter substitution
    - Self-heal broken selectors
    - Learn from failures and improve
    """

    SKILLS_DIR = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "skills"

    def __init__(self, ollama_client=None, headless: bool = True, enable_self_healing: bool = True):
        """
        Initialize the skill generator.

        Args:
            ollama_client: OllamaClient instance for LLM calls
            headless: Run browser without visible window
            enable_self_healing: Enable self-healing selector capabilities
        """
        self.ollama_client = ollama_client
        self.headless = headless
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._playwright = None

        # Self-healing setup
        self.enable_self_healing = enable_self_healing and SELF_HEALING_AVAILABLE
        self._self_healing_locator: Optional["SelfHealingLocator"] = None
        self._fingerprint_gen: Optional["FingerprintGenerator"] = None
        if self.enable_self_healing:
            self._self_healing_locator = SelfHealingLocator()
            self._fingerprint_gen = FingerprintGenerator(self._self_healing_locator)

        # Ensure skills directory exists
        self.SKILLS_DIR.mkdir(parents=True, exist_ok=True)

        # Skill cache
        self._skills_cache: Dict[str, Skill] = {}
        self._load_skills()

    def _load_skills(self):
        """Load all skills from disk."""
        for skill_file in self.SKILLS_DIR.glob("*.json"):
            try:
                with open(skill_file) as f:
                    data = json.load(f)
                    skill = Skill.from_dict(data)
                    self._skills_cache[skill.id] = skill
            except Exception as e:
                print(f"[SkillGenerator] Failed to load {skill_file}: {e}")

    def _save_skill(self, skill: Skill):
        """Save a skill to disk."""
        skill_file = self.SKILLS_DIR / f"{skill.id}.json"
        with open(skill_file, "w") as f:
            json.dump(skill.to_dict(), f, indent=2)
        self._skills_cache[skill.id] = skill

    def _generate_skill_id(self, name: str) -> str:
        """Generate a unique skill ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:6]
        return f"skill_{timestamp}_{name_hash}"

    # ========== Browser Management ==========
    # Uses SYNC API + asyncio.to_thread() to avoid Windows asyncio subprocess issues

    def _ensure_browser_sync(self) -> Page:
        """Ensure browser is running and return the page (SYNC version for thread)."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not installed. Run: pip install playwright && playwright install")

        if self._page and not self._page.is_closed():
            return self._page

        if not self._playwright:
            self._playwright = sync_playwright().start()

        if not self._browser or not self._browser.is_connected():
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"]  # Avoid detection
            )

        if not self._context:
            self._context = self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )

        self._page = self._context.new_page()
        return self._page

    async def _ensure_browser(self) -> Page:
        """Ensure browser is running and return the page (async wrapper)."""
        return await asyncio.to_thread(self._ensure_browser_sync)

    def _close_browser_sync(self):
        """Close the browser (SYNC version for thread)."""
        if self._page:
            self._page.close()
            self._page = None
        if self._context:
            self._context.close()
            self._context = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    async def close_browser(self):
        """Close the browser."""
        await asyncio.to_thread(self._close_browser_sync)

    # ========== Skill Execution ==========

    async def execute_skill(
        self,
        skill_id: str,
        parameters: Optional[Dict[str, str]] = None,
        take_screenshots: bool = False
    ) -> ExecutionResult:
        """
        Execute a skill by ID using subprocess to avoid Windows asyncio issues.

        Args:
            skill_id: The skill to execute
            parameters: Parameter values to substitute
            take_screenshots: Capture screenshot after each step

        Returns:
            ExecutionResult with success status and output
        """
        skill = self._skills_cache.get(skill_id)
        if not skill:
            return ExecutionResult(
                success=False,
                error=f"Skill not found: {skill_id}",
                total_steps=0
            )

        # Use subprocess to run Playwright in isolated process (fixes Windows issue)
        return await asyncio.to_thread(self._execute_via_subprocess, skill, parameters or {})

    def _execute_via_subprocess(self, skill: "Skill", parameters: Dict[str, str]) -> ExecutionResult:
        """Execute skill via subprocess to isolate Playwright from uvicorn's event loop."""
        import subprocess
        import sys

        # Get path to executor script
        executor_path = Path(__file__).parent / "skill_executor.py"

        # Serialize skill and params
        skill_json = json.dumps(skill.to_dict())
        params_json = json.dumps(parameters)

        try:
            # Run in subprocess
            result = subprocess.run(
                [sys.executable, str(executor_path), skill_json, params_json],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip())

                # Update skill stats
                if data.get("success"):
                    skill.success_count += 1
                else:
                    skill.fail_count += 1
                skill.last_used = datetime.now(timezone.utc)
                self._save_skill(skill)

                return ExecutionResult(
                    success=data.get("success", False),
                    output=data.get("output"),
                    error=data.get("error"),
                    steps_completed=data.get("steps_completed", 0),
                    total_steps=data.get("total_steps", len(skill.steps)),
                    duration_ms=data.get("duration_ms", 0)
                )
            else:
                error_msg = result.stderr or "Subprocess failed with no output"
                return ExecutionResult(
                    success=False,
                    error=error_msg[:500],
                    total_steps=len(skill.steps)
                )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error="Skill execution timed out (120s)",
                total_steps=len(skill.steps)
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                total_steps=len(skill.steps)
            )

    def _execute_steps_sync(
        self,
        skill: Skill,
        parameters: Dict[str, str],
        take_screenshots: bool
    ) -> ExecutionResult:
        """Execute the steps of a skill (SYNC version for thread)."""
        start_time = datetime.now(timezone.utc)
        screenshots = []
        steps_completed = 0
        output = None

        try:
            page = self._ensure_browser_sync()

            for i, step in enumerate(skill.steps):
                try:
                    result = self._execute_step_sync(
                        page, step, parameters,
                        skill_id=skill.id,
                        step_index=i
                    )
                    if result:
                        output = result
                    steps_completed += 1

                    if take_screenshots:
                        screenshot = page.screenshot()
                        screenshots.append(screenshot)

                except Exception as e:
                    # Step failed
                    duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    skill.fail_count += 1
                    skill.last_used = datetime.now(timezone.utc)
                    self._save_skill(skill)

                    return ExecutionResult(
                        success=False,
                        error=f"Step {i+1} failed: {step.description or step.action} - {str(e)}",
                        screenshots=screenshots,
                        duration_ms=duration,
                        steps_completed=steps_completed,
                        total_steps=len(skill.steps)
                    )

            # All steps completed
            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            skill.success_count += 1
            skill.last_used = datetime.now(timezone.utc)
            self._save_skill(skill)

            return ExecutionResult(
                success=True,
                output=output,
                screenshots=screenshots,
                duration_ms=duration,
                steps_completed=steps_completed,
                total_steps=len(skill.steps)
            )

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return ExecutionResult(
                success=False,
                error=str(e),
                screenshots=screenshots,
                duration_ms=duration,
                steps_completed=steps_completed,
                total_steps=len(skill.steps)
            )

    async def _execute_steps(
        self,
        skill: Skill,
        parameters: Dict[str, str],
        take_screenshots: bool
    ) -> ExecutionResult:
        """Execute the steps of a skill (async wrapper)."""
        return await asyncio.to_thread(
            self._execute_steps_sync, skill, parameters, take_screenshots
        )

    def _execute_step_sync(
        self,
        page: Page,
        step: SkillStep,
        parameters: Dict[str, str],
        skill_id: Optional[str] = None,
        step_index: Optional[int] = None
    ) -> Optional[Any]:
        """Execute a single step (SYNC version for thread)."""
        import time
        # Substitute parameters in selector and value
        selector = self._substitute_params(step.selector, parameters) if step.selector else None
        value = self._substitute_params(step.value, parameters) if step.value else None

        action = step.action.lower()

        # For actions that need selectors, try self-healing if primary selector fails
        if action in ("click", "fill", "type", "select", "hover") and selector:
            try:
                # Try primary selector first
                page.wait_for_selector(selector, timeout=2000, state="visible")
            except Exception:
                # Primary selector failed, try self-healing
                if self.enable_self_healing and step.fingerprint:
                    healed_selector = self._try_heal_selector_sync(page, step, skill_id, step_index)
                    if healed_selector:
                        selector = healed_selector
                    else:
                        # Re-raise the original error
                        raise

        if action == "navigate":
            page.goto(value, wait_until="domcontentloaded", timeout=step.timeout_ms)

        elif action == "click":
            page.click(selector, timeout=step.timeout_ms)

        elif action == "fill" or action == "type":
            page.fill(selector, value, timeout=step.timeout_ms)

        elif action == "press":
            page.press(selector or "body", value, timeout=step.timeout_ms)

        elif action == "wait":
            if step.wait_for:
                page.wait_for_selector(step.wait_for, timeout=step.timeout_ms)
            else:
                time.sleep(step.timeout_ms / 1000)

        elif action == "extract":
            if selector:
                element = page.query_selector(selector)
                if element:
                    return element.text_content()
            else:
                return page.content()

        elif action == "screenshot":
            return page.screenshot()

        elif action == "select":
            page.select_option(selector, value, timeout=step.timeout_ms)

        elif action == "scroll":
            if selector:
                page.locator(selector).scroll_into_view_if_needed()
            else:
                page.evaluate(f"window.scrollBy(0, {value or 500})")

        elif action == "hover":
            page.hover(selector, timeout=step.timeout_ms)

        else:
            raise ValueError(f"Unknown action: {action}")

        # Wait for condition after action if specified
        if step.wait_for and action not in ["wait"]:
            try:
                page.wait_for_selector(step.wait_for, timeout=2000)
            except Exception:
                pass  # Optional wait, don't fail

        return None

    def _try_heal_selector_sync(
        self,
        page: Page,
        step: SkillStep,
        skill_id: Optional[str],
        step_index: Optional[int]
    ) -> Optional[str]:
        """Try to heal a broken selector (SYNC stub - returns None for now)."""
        # TODO: Implement sync version of self-healing
        return None

    async def _try_heal_selector(
        self,
        page: Page,
        step: SkillStep,
        skill_id: Optional[str],
        step_index: Optional[int]
    ) -> Optional[str]:
        """
        Try to heal a broken selector using fingerprint.

        Returns the healed selector if successful, None otherwise.
        """
        if not self._self_healing_locator or not step.fingerprint:
            return None

        try:
            # Convert dict to ElementFingerprint
            fingerprint = ElementFingerprint.from_dict(step.fingerprint)

            # Try to find element using fingerprint
            locator = await self._self_healing_locator.locate(page, fingerprint)

            if locator and await locator.count() > 0:
                # Found it! Capture new fingerprint
                new_fp = await self._self_healing_locator.heal_selector(page, fingerprint)

                if new_fp:
                    # Update the step's selector and fingerprint
                    new_selector = new_fp.css_selector

                    # Update skill on disk if we have the skill ID
                    if skill_id and step_index is not None:
                        self._update_healed_step(skill_id, step_index, new_selector, new_fp.to_dict())

                    print(f"[SkillGenerator] Self-healed selector: {step.selector} → {new_selector}")
                    return new_selector

        except Exception as e:
            print(f"[SkillGenerator] Self-healing failed: {e}")

        return None

    def _update_healed_step(
        self,
        skill_id: str,
        step_index: int,
        new_selector: str,
        new_fingerprint: Dict
    ):
        """Update a skill's step after successful healing."""
        skill = self._skills_cache.get(skill_id)
        if not skill or step_index >= len(skill.steps):
            return

        # Update step
        skill.steps[step_index].selector = new_selector
        skill.steps[step_index].fingerprint = new_fingerprint

        # Increment version
        skill.version += 1
        skill.updated_at = datetime.now(timezone.utc)

        # Save
        self._save_skill(skill)

    def _substitute_params(self, text: str, parameters: Dict[str, str]) -> str:
        """Substitute {{param}} placeholders with values."""
        if not text:
            return text
        for key, value in parameters.items():
            text = text.replace(f"{{{{{key}}}}}", value)
        return text

    # ========== Skill Generation ==========

    async def generate_skill_from_description(
        self,
        name: str,
        description: str,
        example_url: Optional[str] = None
    ) -> Skill:
        """
        Generate a skill from a natural language description.

        Uses the LLM to understand the task and generate steps.
        """
        if not self.ollama_client:
            raise RuntimeError("OllamaClient required for skill generation")

        prompt = f"""You are a browser automation expert. Generate a SIMPLE Playwright skill.

Task: {description}
{f'Starting URL: {example_url}' if example_url else ''}

IMPORTANT RULES:
1. Keep it SIMPLE - use 3-5 steps maximum
2. This is BROWSER automation only - navigate, click, extract from the page
3. DO NOT set up APIs, OAuth, or install packages - just browser actions
4. Prefer public pages that don't require login
5. For Reddit: use reddit.com/r/SUBREDDIT (public view), NOT /prefs/apps
6. For data extraction: navigate to the page, wait for content, extract with CSS selectors

Available actions: navigate, click, fill, press, wait, extract, screenshot, select, scroll, hover

JSON format:
```json
{{
  "steps": [
    {{"action": "navigate", "value": "https://site.com/page", "description": "Go to page"}},
    {{"action": "wait", "wait_for": ".content", "timeout_ms": 5000, "description": "Wait for content"}},
    {{"action": "extract", "selector": ".item-title", "description": "Extract titles"}}
  ],
  "parameters": []
}}
```

Generate ONLY the JSON (no explanation):"""

        # Call LLM
        from .ollama_client import ChatMessage
        ollama_response = await self.ollama_client.chat(
            messages=[ChatMessage(role="user", content=prompt)],
            thinking=False
        )
        response = ollama_response.content if ollama_response else ""

        print(f"[SkillGenerator] LLM response length: {len(response) if response else 0}")

        # Check for empty response
        if not response or not response.strip():
            print("[SkillGenerator] Empty LLM response, creating basic navigation skill")
            # Fallback: create a simple navigation-only skill
            data = {
                "steps": [
                    {"action": "navigate", "value": example_url or "https://example.com", "description": f"Navigate to {example_url}"},
                    {"action": "wait", "timeout_ms": 2000, "description": "Wait for page to load"},
                    {"action": "extract", "selector": "body", "description": "Extract page content"}
                ],
                "parameters": []
            }
        else:
            # Parse JSON from response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                # Try to find raw JSON object
                json_obj_match = re.search(r'\{[\s\S]*\}', response)
                if json_obj_match:
                    data = json.loads(json_obj_match.group(0))
                else:
                    print(f"[SkillGenerator] Could not parse JSON from: {response[:200]}...")
                    # Fallback
                    data = {
                        "steps": [
                            {"action": "navigate", "value": example_url or "https://example.com", "description": "Navigate to URL"},
                            {"action": "extract", "selector": "body", "description": "Extract content"}
                        ],
                        "parameters": []
                    }

        # Create skill
        skill = Skill(
            id=self._generate_skill_id(name),
            name=name,
            description=description,
            steps=[SkillStep.from_dict(s) for s in data.get("steps", [])],
            parameters=data.get("parameters", []),
            tags=["generated", "untested"],
            status=SkillStatus.DRAFT
        )

        self._save_skill(skill)
        return skill

    async def record_skill(
        self,
        name: str,
        description: str,
        start_url: str
    ) -> Tuple[Skill, Page]:
        """
        Start recording a new skill.

        Returns the skill (in draft mode) and the page for interaction.
        User can then interact with the page, and we'll record the actions.
        """
        skill = Skill(
            id=self._generate_skill_id(name),
            name=name,
            description=description,
            steps=[
                SkillStep(action="navigate", value=start_url, description=f"Navigate to {start_url}")
            ],
            tags=["recorded"],
            status=SkillStatus.DRAFT
        )

        # Start browser and navigate
        page = await self._ensure_browser()
        await page.goto(start_url)

        # Set up event listeners to record actions
        # (In practice, Playwright codegen does this better)

        return skill, page

    # ========== Skill Library ==========

    def list_skills(
        self,
        status: Optional[SkillStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[Skill]:
        """List all skills, optionally filtered."""
        skills = list(self._skills_cache.values())

        if status:
            skills = [s for s in skills if s.status == status]

        if tags:
            skills = [s for s in skills if any(t in s.tags for t in tags)]

        # Safe sorting that handles mixed timezone-aware and naive datetimes
        def safe_sort_key(s):
            dt = s.updated_at
            if dt is None:
                return datetime.min
            # Convert to naive UTC for consistent comparison
            if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                return dt.replace(tzinfo=None)
            return dt

        return sorted(skills, key=safe_sort_key, reverse=True)

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self._skills_cache.get(skill_id)

    def search_skills(self, query: str) -> List[Skill]:
        """Search skills by name or description."""
        query_lower = query.lower()
        return [
            s for s in self._skills_cache.values()
            if query_lower in s.name.lower() or query_lower in s.description.lower()
        ]

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill."""
        if skill_id not in self._skills_cache:
            return False

        skill_file = self.SKILLS_DIR / f"{skill_id}.json"
        if skill_file.exists():
            skill_file.unlink()

        del self._skills_cache[skill_id]
        return True

    # ========== Page Observation ==========

    async def get_page_state(self) -> Dict[str, Any]:
        """Get the current state of the page for LLM analysis."""
        if not self._page or self._page.is_closed():
            return {"error": "No page open"}

        page = self._page

        # Get accessibility tree (like Playwright MCP does)
        try:
            accessibility_tree = await page.accessibility.snapshot()
        except Exception:
            accessibility_tree = None

        # Get clickable elements
        clickable = await page.evaluate("""() => {
            const elements = document.querySelectorAll('a, button, input, select, textarea, [onclick], [role="button"]');
            return Array.from(elements).slice(0, 50).map((el, i) => ({
                index: i,
                tag: el.tagName.toLowerCase(),
                type: el.type || null,
                text: el.innerText?.slice(0, 100) || el.value?.slice(0, 100) || '',
                placeholder: el.placeholder || null,
                name: el.name || null,
                id: el.id || null,
                href: el.href || null
            }));
        }""")

        return {
            "url": page.url,
            "title": await page.title(),
            "accessibility_tree": accessibility_tree,
            "clickable_elements": clickable
        }

    async def take_screenshot(self) -> Optional[bytes]:
        """Take a screenshot of the current page."""
        if not self._page or self._page.is_closed():
            return None
        return await self._page.screenshot()

    # ========== Integration with OODA ==========

    def get_tool_definitions(self) -> str:
        """Get tool definitions for the cognitive loop."""
        return """
### browser_navigate
Navigate to a URL in the browser.
Parameters:
  - url (required): URL to navigate to
Example: <tool_call>{"tool": "browser_navigate", "params": {"url": "https://example.com"}}</tool_call>

### browser_click
Click an element on the page.
Parameters:
  - selector (required): CSS selector or "text=Button Text" for text matching
Example: <tool_call>{"tool": "browser_click", "params": {"selector": "button#submit"}}</tool_call>

### browser_fill
Fill a text input.
Parameters:
  - selector (required): Input selector
  - value (required): Text to fill
Example: <tool_call>{"tool": "browser_fill", "params": {"selector": "#email", "value": "test@example.com"}}</tool_call>

### browser_state
Get the current page state (URL, title, clickable elements).
Parameters: none
Example: <tool_call>{"tool": "browser_state", "params": {}}</tool_call>

### browser_screenshot
Take a screenshot of the current page.
Parameters: none
Example: <tool_call>{"tool": "browser_screenshot", "params": {}}</tool_call>

### execute_skill
Execute a saved automation skill.
Parameters:
  - skill_id (required): ID of the skill to execute
  - parameters (optional): Parameter values for the skill
Example: <tool_call>{"tool": "execute_skill", "params": {"skill_id": "skill_123", "parameters": {"username": "john"}}}</tool_call>

### list_skills
List available automation skills.
Parameters:
  - status (optional): Filter by status (draft, verified, etc.)
Example: <tool_call>{"tool": "list_skills", "params": {}}</tool_call>

### generate_skill
Generate a new skill from a description.
Parameters:
  - name (required): Name for the skill
  - description (required): What the skill should do
  - url (optional): Starting URL
Example: <tool_call>{"tool": "generate_skill", "params": {"name": "login_github", "description": "Log into GitHub", "url": "https://github.com/login"}}</tool_call>
"""


# Singleton instance
_generator_instance: Optional[SkillGenerator] = None

def get_skill_generator(ollama_client=None, headless: bool = True) -> SkillGenerator:
    """Get or create the skill generator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = SkillGenerator(ollama_client, headless)
    elif ollama_client:
        _generator_instance.ollama_client = ollama_client
    return _generator_instance
