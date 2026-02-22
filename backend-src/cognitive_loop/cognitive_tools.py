"""
Cognitive Tools - MCP Tool Wrappers for Autonomous Thinking

Gives the local LLM (Qwen3-32B) the same tools I (Claude) have access to.
The LLM can call these tools during thinking to gather information and take action.
"""

import os
import json
import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Awaitable
from pathlib import Path
from datetime import datetime, timezone


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None


@dataclass
class ToolCall:
    """A parsed tool call from LLM output."""
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


# MCP Server URL (direct to MCP, not through Cerebro backend)
MCP_URL = os.environ.get("MCP_URL", "http://localhost:58700")
CEREBRO_URL = os.environ.get("CEREBRO_URL", "http://localhost:59000")

# Feature flag for agent spawning from cognitive loop
# Set CEREBRO_COGNITIVE_SPAWN=true to enable (disabled by default for safety)
COGNITIVE_SPAWN_ENABLED = os.environ.get("CEREBRO_COGNITIVE_SPAWN", "false").lower() == "true"


class CognitiveTools:
    """
    Tool system for autonomous cognitive loop.

    Provides tools that mirror MCP capabilities:
    - search_memory: Semantic search across all memory
    - get_user_profile: User preferences, identity, goals
    - get_corrections: Known mistakes to avoid
    - find_learning: Find proven solutions
    - record_learning: Save new learning
    - get_goals: Active goals
    - check_predictions: Anticipate issues
    - causal_query: Query causal model
    """

    # Auth token for internal API calls (loaded from file or env)
    _auth_token: Optional[str] = None

    # Tool definitions for the LLM
    TOOL_DEFINITIONS = """
## Available Tools

You can call tools by outputting them in this format:
<tool_call>
{"tool": "tool_name", "params": {"param1": "value1"}}
</tool_call>

### web_search
Search the internet for current, up-to-date information. USE THIS for research tasks!
Parameters:
  - query (required): What to search for on the web
  - max_results (optional): Number of results (default 5, max 10)
Example: <tool_call>{"tool": "web_search", "params": {"query": "walrus habitat and behavior facts"}}</tool_call>
Example: <tool_call>{"tool": "web_search", "params": {"query": "futures trading strategies 2024", "max_results": 8}}</tool_call>

### search_memory
Search the AI Memory system for relevant context (things the user has saved before).
Parameters:
  - query (required): What to search for
  - limit (optional): Max results (default 5)
Example: <tool_call>{"tool": "search_memory", "params": {"query": "NAS configuration issues"}}</tool_call>

### get_user_profile
Get the user's profile - preferences, identity, projects, goals.
Parameters:
  - category (optional): "all", "identity", "preferences", "goals", "technical_environment"
Example: <tool_call>{"tool": "get_user_profile", "params": {"category": "preferences"}}</tool_call>

### get_corrections
Get known mistakes and corrections to avoid repeating them.
Parameters:
  - topic (optional): Filter by topic
  - limit (optional): Max results (default 10)
Example: <tool_call>{"tool": "get_corrections", "params": {"topic": "network"}}</tool_call>

### find_learning
Find proven solutions to problems.
Parameters:
  - problem (required): Problem description to find solutions for
  - limit (optional): Max results (default 5)
Example: <tool_call>{"tool": "find_learning", "params": {"problem": "PowerShell parsing errors"}}</tool_call>

### record_learning
Record a new learning, solution, or antipattern.
Parameters:
  - type (required): "solution", "failure", or "antipattern"
  - problem (required): Problem description
  - solution (for solution): What worked
  - what_failed (for antipattern): What NOT to do
  - tags (optional): List of tags
Example: <tool_call>{"tool": "record_learning", "params": {"type": "solution", "problem": "X fails", "solution": "Do Y instead"}}</tool_call>

### get_goals
Get active user goals.
Parameters:
  - status (optional): "active", "completed", "all"
Example: <tool_call>{"tool": "get_goals", "params": {"status": "active"}}</tool_call>

### check_predictions
Check for predictions about potential issues.
Parameters:
  - context (required): Context to check predictions for
Example: <tool_call>{"tool": "check_predictions", "params": {"context": "deploying new service"}}</tool_call>

### causal_query
Query the causal model for cause-effect relationships.
Parameters:
  - query (required): What to query
  - action (optional): "find_causes", "find_effects", "what_if"
Example: <tool_call>{"tool": "causal_query", "params": {"query": "NAS disconnection", "action": "find_causes"}}</tool_call>

### get_quick_facts
Get quick facts about system configuration and current state.
Parameters: none
Example: <tool_call>{"tool": "get_quick_facts", "params": {}}</tool_call>

### browser_navigate
Navigate to a URL in the browser (launches browser if needed).
Parameters:
  - url (required): URL to navigate to
Example: <tool_call>{"tool": "browser_navigate", "params": {"url": "https://example.com"}}</tool_call>

### browser_click
Click an element on the page.
Parameters:
  - selector (required): CSS selector or "text=Button Text" for text matching
Example: <tool_call>{"tool": "browser_click", "params": {"selector": "button#submit"}}</tool_call>

### browser_fill
Fill a text input field.
Parameters:
  - selector (required): Input selector
  - value (required): Text to fill
Example: <tool_call>{"tool": "browser_fill", "params": {"selector": "#email", "value": "test@example.com"}}</tool_call>

### browser_state
Get the current page state (URL, title, clickable elements).
Parameters: none
Example: <tool_call>{"tool": "browser_state", "params": {}}</tool_call>

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
Generate a new automation skill from a description.
Parameters:
  - name (required): Name for the skill
  - description (required): What the skill should do
  - url (optional): Starting URL
Example: <tool_call>{"tool": "generate_skill", "params": {"name": "login_github", "description": "Log into GitHub", "url": "https://github.com/login"}}</tool_call>

### spawn_agent
**USE THIS when directives require CREATING FILES, WRITING CODE, or EXECUTING tasks!**
Deploy a Claude Code agent to execute a complex task. This is YOUR HANDS - the way you actually DO things in the real world.

**WHEN TO USE (mandatory for these directive types):**
  - "Create a script/file" → USE spawn_agent (agent_type="coder")
  - "Build/write code" → USE spawn_agent (agent_type="coder")
  - "Set up/configure something" → USE spawn_agent (agent_type="worker")
  - "Analyze code" → USE spawn_agent (agent_type="analyst")
  Explaining HOW to do something is NOT doing it. You MUST spawn an agent for real work!

**REQUIREMENTS:**
  - Full Autonomy mode must be ON
  - CEREBRO_COGNITIVE_SPAWN feature flag must be true
  - Uses the user's Anthropic subscription - use wisely!
  - Max 10 spawns per cognitive session, 5/hour, 20/day
Parameters:
  - task (required): What the agent should accomplish
  - agent_type (optional): "worker", "researcher", "coder", "analyst", "orchestrator" (default: "worker")
  - context (optional): Additional context for the agent
  - expected_output (optional): What output format you expect
  - priority (optional): "low", "normal", "high" (default: "normal")
Example: <tool_call>{"tool": "spawn_agent", "params": {"task": "Create a Python script that backs up the AI Memory database", "agent_type": "coder"}}</tool_call>

### wait_for_agent
Wait for a spawned agent to complete. Blocks until agent finishes or timeout.
Parameters:
  - agent_id (required): The agent ID returned from spawn_agent
  - timeout_minutes (optional): Max wait time (default: 30, max: 60)
Example: <tool_call>{"tool": "wait_for_agent", "params": {"agent_id": "Alpha-1"}}</tool_call>

### check_agent_status
Check the current status of an agent without blocking.
Parameters:
  - agent_id (required): The agent ID to check
Example: <tool_call>{"tool": "check_agent_status", "params": {"agent_id": "Alpha-1"}}</tool_call>

### get_active_goals
**GOAL PURSUIT SYSTEM** - Get all active goals with progress metrics and ready subtasks.
This is MY mission control - shows what I'm working toward and what I can do next.
Parameters:
  - include_ready_subtasks (optional): Include list of ready subtasks (default: true)
Example: <tool_call>{"tool": "get_active_goals", "params": {}}</tool_call>

### update_goal_progress
Update progress on a goal. Use after completing subtasks or achieving milestones.
Parameters:
  - goal_id (required): The goal to update
  - delta (optional): Progress increase (e.g., 100 for $100 earned)
  - new_value (optional): Absolute new value
  - note (optional): Description of what caused the progress
Example: <tool_call>{"tool": "update_goal_progress", "params": {"goal_id": "goal_abc123", "delta": 100, "note": "First freelance payment"}}</tool_call>

### complete_subtask
Mark a subtask as completed and record the result.
Parameters:
  - subtask_id (required): The subtask to complete
  - result (required): What was accomplished (will be saved for future reference)
Example: <tool_call>{"tool": "complete_subtask", "params": {"subtask_id": "subtask_xyz", "result": "Created comparison matrix with 5 approaches"}}</tool_call>

### fail_subtask
Mark a subtask as failed with the reason. Will trigger Reflexion learning.
Parameters:
  - subtask_id (required): The subtask that failed
  - error (required): What went wrong
  - learning (optional): What to avoid next time (Reflexion pattern)
Example: <tool_call>{"tool": "fail_subtask", "params": {"subtask_id": "subtask_xyz", "error": "API rate limited", "learning": "Use batch requests instead of individual calls"}}</tool_call>

### decompose_goal
Decompose a goal into milestones and subtasks using HTN planning.
Call this when a new goal is created or when stuck on a goal.
Parameters:
  - goal_id (required): The goal to decompose
Example: <tool_call>{"tool": "decompose_goal", "params": {"goal_id": "goal_abc123"}}</tool_call>

### get_next_subtask
Get the highest-priority subtask that's ready to execute.
Returns the next actionable task with its agent type and context.
Parameters:
  - goal_id (optional): Specific goal to get subtask from (default: highest priority goal)
Example: <tool_call>{"tool": "get_next_subtask", "params": {}}}</tool_call>

### explore_website
**ADAPTIVE BROWSER LEARNING** - Explore a website to accomplish a goal.
Uses LLM-guided exploration to navigate unknown websites, then creates a reusable skill.
Parameters:
  - goal (required): What to accomplish (e.g., "Login to Upwork and create freelancer profile")
  - start_url (required): URL to start exploration from
  - max_steps (optional): Max exploration steps (default 20)
  - create_skill (optional): Convert successful exploration to skill (default true)
  - skill_name (optional): Name for generated skill
Example: <tool_call>{"tool": "explore_website", "params": {"goal": "Sign up for Upwork freelancer account", "start_url": "https://upwork.com"}}</tool_call>

### execute_subtask_with_skill
Execute a subtask using its associated skill or exploration.
- If subtask has skill_id: executes the existing skill
- If subtask has exploration_goal: explores to accomplish goal and creates skill
Parameters:
  - subtask_id (required): The subtask to execute
Example: <tool_call>{"tool": "execute_subtask_with_skill", "params": {"subtask_id": "subtask_abc123"}}</tool_call>

### verify_skill
Verify a skill works correctly and auto-heal broken selectors.
Parameters:
  - skill_id (required): ID of skill to verify
  - dry_run (optional): Only check selectors exist, don't execute (default true)
Example: <tool_call>{"tool": "verify_skill", "params": {"skill_id": "skill_abc123", "dry_run": true}}</tool_call>

### record_action
Start or stop recording browser actions for skill creation.
Parameters:
  - action (required): "start" or "stop"
  - name (required for start): Name for the recording
  - description (optional): Description of what's being recorded
Example: <tool_call>{"tool": "record_action", "params": {"action": "start", "name": "linkedin_login"}}</tool_call>

---
After receiving tool results, continue your reasoning. You can call multiple tools before reaching a conclusion.
When done reasoning, output your final answer WITHOUT tool calls.
"""

    # Session spawn limits
    MAX_SPAWNS_PER_SESSION = 10
    MAX_SPAWN_OUTPUT_CHARS = 5000
    MAX_CONCURRENT_AGENTS = 3  # The user's computer can handle max 3 agents at once

    def __init__(self, debug_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None):
        self._debug_callback = debug_callback
        self._session: Optional[aiohttp.ClientSession] = None

        # Quick facts cache
        self._quick_facts_cache: Optional[Dict] = None
        self._quick_facts_time: Optional[datetime] = None

        # Agent spawn tracking (per session)
        self._session_spawn_count: int = 0
        self._spawned_agents: Dict[str, Dict] = {}  # agent_id -> spawn info

        # Load auth token for internal API calls
        self._load_auth_token()

    def _load_auth_token(self):
        """Load auth token from file or environment."""
        # Try environment first
        token = os.environ.get("CEREBRO_AUTH_TOKEN")
        if token:
            CognitiveTools._auth_token = token
            return

        # Try token file
        token_file = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "auth_token.txt"
        try:
            if token_file.exists():
                CognitiveTools._auth_token = token_file.read_text().strip()
        except Exception:
            pass

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with auth token for API calls."""
        headers = {"Content-Type": "application/json"}
        if CognitiveTools._auth_token:
            headers["Authorization"] = f"Bearer {CognitiveTools._auth_token}"
        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _emit_debug(self, event_type: str, data: Dict[str, Any]):
        """Emit debug event."""
        if self._debug_callback:
            debug_event = {
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **data
            }
            try:
                await self._debug_callback(debug_event)
            except Exception:
                pass

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def parse_tool_calls(self, llm_output: str) -> List[ToolCall]:
        """Parse tool calls from LLM output."""
        import re
        tool_calls = []

        # Method 1: Look for <tool_call>...</tool_call> blocks (preferred format)
        pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
        matches = re.findall(pattern, llm_output, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                if data.get("tool"):
                    tool_calls.append(ToolCall(
                        tool_name=data.get("tool", ""),
                        parameters=data.get("params", {})
                    ))
            except json.JSONDecodeError:
                continue

        # Method 2: Look for raw JSON tool calls (Qwen often outputs without tags)
        # Pattern: {"tool": "name", "params": {...}}
        if not tool_calls:
            raw_pattern = r'\{["\']tool["\']\s*:\s*["\'](\w+)["\']\s*,\s*["\']params["\']\s*:\s*(\{[^}]*\})\}'
            raw_matches = re.findall(raw_pattern, llm_output, re.DOTALL)

            for tool_name, params_str in raw_matches:
                try:
                    params = json.loads(params_str)
                    tool_calls.append(ToolCall(
                        tool_name=tool_name,
                        parameters=params
                    ))
                except json.JSONDecodeError:
                    # Try to parse the whole match as JSON
                    continue

            # Method 3: Try to find complete JSON objects with "tool" key
            if not tool_calls:
                # Find all JSON-like objects
                json_pattern = r'\{"tool"\s*:\s*"[^"]+"\s*,\s*"params"\s*:\s*\{[^}]+\}\}'
                json_matches = re.findall(json_pattern, llm_output)

                for json_str in json_matches:
                    try:
                        data = json.loads(json_str)
                        if data.get("tool"):
                            tool_calls.append(ToolCall(
                                tool_name=data.get("tool", ""),
                                parameters=data.get("params", {})
                            ))
                    except json.JSONDecodeError:
                        continue

        return tool_calls

    def has_tool_calls(self, llm_output: str) -> bool:
        """Check if output contains tool calls."""
        import re
        # Check for tagged format
        if "<tool_call>" in llm_output:
            return True
        # Check for raw JSON format (Qwen often outputs without tags)
        # Pattern matches: {"tool": "...", "params": ...}
        raw_pattern = r'\{"tool"\s*:\s*"[^"]+"'
        if re.search(raw_pattern, llm_output):
            return True
        return False

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        await self._emit_debug("tool_call", {
            "tool": tool_call.tool_name,
            "params": tool_call.parameters
        })

        try:
            # Route to appropriate handler
            handler = getattr(self, f"_tool_{tool_call.tool_name}", None)
            if handler:
                result = await handler(tool_call.parameters)
                await self._emit_debug("tool_result", {
                    "tool": tool_call.tool_name,
                    "success": True,
                    "result_preview": str(result)[:200]
                })
                return ToolResult(
                    tool_name=tool_call.tool_name,
                    success=True,
                    result=result
                )
            else:
                return ToolResult(
                    tool_name=tool_call.tool_name,
                    success=False,
                    result=None,
                    error=f"Unknown tool: {tool_call.tool_name}"
                )
        except Exception as e:
            await self._emit_debug("tool_error", {
                "tool": tool_call.tool_name,
                "error": str(e)
            })
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                result=None,
                error=str(e)
            )

    async def execute_all_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute all tool calls and return results."""
        results = []
        for tc in tool_calls:
            result = await self.execute_tool(tc)
            results.append(result)
        return results

    def format_tool_results(self, results: List[ToolResult]) -> str:
        """Format tool results for injection back into LLM context."""
        if not results:
            return ""

        parts = ["## Tool Results\n"]
        for r in results:
            parts.append(f"### {r.tool_name}")
            if r.success:
                # Format result nicely
                if isinstance(r.result, dict):
                    parts.append(f"```json\n{json.dumps(r.result, indent=2, default=str)[:1000]}\n```")
                elif isinstance(r.result, list):
                    parts.append(f"```json\n{json.dumps(r.result, indent=2, default=str)[:1000]}\n```")
                else:
                    parts.append(str(r.result)[:1000])
            else:
                parts.append(f"**Error**: {r.error}")
            parts.append("")

        return "\n".join(parts)

    # ========== Tool Implementations ==========

    async def _tool_search_memory(self, params: Dict) -> Any:
        """Search AI Memory."""
        query = params.get("query", "")
        limit = params.get("limit", 5)

        if not query:
            return {"error": "query parameter required"}

        session = await self._get_session()

        # Try MCP bridge search endpoint
        try:
            url = f"{CEREBRO_URL}/memory/search"
            async with session.get(url, params={"q": query, "limit": limit}, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass

        # Fallback: read from quick_facts or return empty
        return {"results": [], "query": query, "note": "Search endpoint not available"}

    async def _tool_get_user_profile(self, params: Dict) -> Any:
        """Get user profile from AI Memory."""
        category = params.get("category", "all")

        session = await self._get_session()

        try:
            url = f"{CEREBRO_URL}/api/user-profile"
            async with session.get(url, params={"category": category}, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass

        # Fallback: return basic info from quick_facts
        facts = await self._get_quick_facts()
        return {
            "identity": {
                "name": os.environ.get("CEREBRO_USER_NAME", ""),
            },
            "preferences": facts.get("preferences", {}),
            "note": "Full profile endpoint not available"
        }

    async def _tool_get_corrections(self, params: Dict) -> Any:
        """Get known corrections to avoid mistakes."""
        topic = params.get("topic")
        limit = params.get("limit", 10)

        session = await self._get_session()

        try:
            url = f"{CEREBRO_URL}/api/corrections"
            query_params = {"limit": limit}
            if topic:
                query_params["topic"] = topic
            async with session.get(url, params=query_params, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass

        # Fallback from quick_facts
        facts = await self._get_quick_facts()
        raw = facts.get("top_corrections", [])
        if isinstance(raw, dict):
            raw = raw.get("most_common", [])
        return {
            "corrections": raw if isinstance(raw, list) else [],
            "note": "From quick_facts cache"
        }

    async def _tool_find_learning(self, params: Dict) -> Any:
        """Find proven solutions."""
        problem = params.get("problem", "")
        limit = params.get("limit", 5)

        if not problem:
            return {"error": "problem parameter required"}

        session = await self._get_session()

        try:
            url = f"{CEREBRO_URL}/api/learnings/search"
            async with session.get(url, params={"problem": problem, "limit": limit}, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass

        return {"solutions": [], "problem": problem, "note": "Search endpoint not available"}

    async def _tool_record_learning(self, params: Dict) -> Any:
        """Record a new learning."""
        learning_type = params.get("type", "solution")
        problem = params.get("problem", "")
        solution = params.get("solution", "")
        what_failed = params.get("what_failed", "")
        tags = params.get("tags", [])

        if not problem:
            return {"error": "problem parameter required"}

        session = await self._get_session()

        try:
            url = f"{CEREBRO_URL}/api/learnings"
            payload = {
                "type": learning_type,
                "problem": problem,
                "solution": solution,
                "what_not_to_do": what_failed,
                "tags": tags,
                "source": "cognitive_loop"
            }
            async with session.post(url, json=payload, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"error": f"Failed: {resp.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def _tool_get_goals(self, params: Dict) -> Any:
        """Get active goals."""
        status = params.get("status", "active")

        session = await self._get_session()

        try:
            url = f"{CEREBRO_URL}/api/goals"
            async with session.get(url, params={"status": status}, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass

        # Fallback from quick_facts
        facts = await self._get_quick_facts()
        return {
            "goals": facts.get("active_goals", []),
            "note": "From quick_facts cache"
        }

    async def _tool_check_predictions(self, params: Dict) -> Any:
        """Check predictions for potential issues."""
        context = params.get("context", "")

        if not context:
            return {"error": "context parameter required"}

        session = await self._get_session()

        try:
            url = f"{CEREBRO_URL}/api/predictions"
            async with session.get(url, params={"context": context}, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass

        return {"predictions": [], "context": context}

    async def _tool_causal_query(self, params: Dict) -> Any:
        """Query the causal model."""
        query = params.get("query", "")
        action = params.get("action", "find_causes")

        if not query:
            return {"error": "query parameter required"}

        session = await self._get_session()

        try:
            url = f"{CEREBRO_URL}/api/causal"
            async with session.get(url, params={"query": query, "action": action}, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass

        return {"results": [], "query": query, "action": action}

    async def _tool_get_quick_facts(self, params: Dict) -> Any:
        """Get quick facts about system state."""
        return await self._get_quick_facts()

    # ========== Browser/Skill Tools ==========

    def _get_skill_generator(self):
        """Lazy-load skill generator to avoid circular imports."""
        from .skill_generator import get_skill_generator
        return get_skill_generator(headless=False)  # Show browser for the user to see

    async def _tool_browser_navigate(self, params: Dict) -> Any:
        """Navigate to a URL in the browser."""
        url = params.get("url", "")
        if not url:
            return {"error": "url parameter required"}

        try:
            gen = self._get_skill_generator()
            page = await gen._ensure_browser()
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return {
                "success": True,
                "url": page.url,
                "title": await page.title()
            }
        except Exception as e:
            return {"error": str(e)}

    async def _tool_browser_click(self, params: Dict) -> Any:
        """Click an element on the page."""
        selector = params.get("selector", "")
        if not selector:
            return {"error": "selector parameter required"}

        try:
            gen = self._get_skill_generator()
            if not gen._page or gen._page.is_closed():
                return {"error": "No browser page open. Use browser_navigate first."}

            await gen._page.click(selector, timeout=5000)
            return {"success": True, "clicked": selector}
        except Exception as e:
            return {"error": str(e)}

    async def _tool_browser_fill(self, params: Dict) -> Any:
        """Fill a text input."""
        selector = params.get("selector", "")
        value = params.get("value", "")
        if not selector:
            return {"error": "selector parameter required"}

        try:
            gen = self._get_skill_generator()
            if not gen._page or gen._page.is_closed():
                return {"error": "No browser page open. Use browser_navigate first."}

            await gen._page.fill(selector, value, timeout=5000)
            return {"success": True, "filled": selector, "value_length": len(value)}
        except Exception as e:
            return {"error": str(e)}

    async def _tool_browser_state(self, params: Dict) -> Any:
        """Get the current page state."""
        try:
            gen = self._get_skill_generator()
            return await gen.get_page_state()
        except Exception as e:
            return {"error": str(e)}

    async def _tool_execute_skill(self, params: Dict) -> Any:
        """Execute a saved automation skill."""
        skill_id = params.get("skill_id", "")
        parameters = params.get("parameters", {})

        if not skill_id:
            return {"error": "skill_id parameter required"}

        try:
            gen = self._get_skill_generator()
            result = await gen.execute_skill(skill_id, parameters)
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "steps_completed": result.steps_completed,
                "total_steps": result.total_steps,
                "duration_ms": result.duration_ms
            }
        except Exception as e:
            return {"error": str(e)}

    async def _tool_list_skills(self, params: Dict) -> Any:
        """List available automation skills."""
        try:
            gen = self._get_skill_generator()
            skills = gen.list_skills()
            return {
                "count": len(skills),
                "skills": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "description": s.description[:100],
                        "status": s.status.value,
                        "success_rate": f"{s.success_rate:.0%}"
                    }
                    for s in skills[:20]  # Limit response size
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    async def _tool_generate_skill(self, params: Dict) -> Any:
        """Generate a new automation skill from description."""
        name = params.get("name", "")
        description = params.get("description", "")
        url = params.get("url")

        if not name or not description:
            return {"error": "name and description parameters required"}

        try:
            # Need ollama client for generation
            from .ollama_client import get_ollama_client
            gen = self._get_skill_generator()
            gen.ollama_client = get_ollama_client()

            skill = await gen.generate_skill_from_description(name, description, url)
            return {
                "success": True,
                "skill_id": skill.id,
                "name": skill.name,
                "steps_count": len(skill.steps),
                "parameters": skill.parameters,
                "status": skill.status.value
            }
        except Exception as e:
            return {"error": str(e)}

    async def _tool_web_search(self, params: Dict) -> Any:
        """
        Search the internet using DuckDuckGo.
        This is the PRIMARY tool for research tasks - use it to find current information!
        """
        query = params.get("query", "")
        max_results = min(params.get("max_results", 5), 10)  # Cap at 10

        if not query:
            return {"error": "Query is required", "results": []}

        try:
            # Import new ddgs package (old duckduckgo_search is deprecated)
            from ddgs import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })

            return {
                "query": query,
                "results_count": len(results),
                "results": results
            }

        except ImportError:
            # Fallback to old package name if new one not installed
            try:
                from duckduckgo_search import DDGS
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", "")
                        })
                return {
                    "query": query,
                    "results_count": len(results),
                    "results": results
                }
            except ImportError:
                return {
                    "error": "ddgs package not installed. Run: pip install ddgs",
                    "results": []
                }
        except Exception as e:
            return {
                "error": f"Web search failed: {str(e)}",
                "query": query,
                "results": []
            }

    async def _get_quick_facts(self) -> Dict:
        """Load quick_facts.json (cached for 5 minutes)."""
        now = datetime.now(timezone.utc)

        # Check cache
        if self._quick_facts_cache and self._quick_facts_time:
            age = (now - self._quick_facts_time).total_seconds()
            if age < 300:  # 5 minute cache
                return self._quick_facts_cache

        # Load from file
        quick_facts_path = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "quick_facts.json"
        try:
            if quick_facts_path.exists():
                with open(quick_facts_path) as f:
                    self._quick_facts_cache = json.load(f)
                    self._quick_facts_time = now
                    return self._quick_facts_cache
        except Exception:
            pass

        return {}

    # ========== Agent Spawning Tools ==========

    async def _tool_spawn_agent(self, params: Dict) -> Any:
        """
        Spawn a Claude Code agent to execute a task.

        This uses the user's Anthropic subscription - requires Full Autonomy mode.
        Has session limit of 10 spawns to prevent runaway.
        """
        # Log that spawn_agent was called (for debugging)
        print(f"[SPAWN_AGENT] Tool called with params: {params}")
        await self._emit_debug("spawn_agent_called", {
            "params": params,
            "feature_flag_enabled": COGNITIVE_SPAWN_ENABLED
        })

        # Check feature flag first
        if not COGNITIVE_SPAWN_ENABLED:
            return {
                "error": "Agent spawning from cognitive loop is disabled",
                "message": "Set CEREBRO_COGNITIVE_SPAWN=true environment variable to enable",
                "feature_flag": "CEREBRO_COGNITIVE_SPAWN"
            }

        task = params.get("task", "")
        agent_type = params.get("agent_type", "worker")
        context = params.get("context", "")
        expected_output = params.get("expected_output", "")
        priority = params.get("priority", "normal")

        if not task:
            return {"error": "task parameter required"}

        # Check session spawn limit
        if self._session_spawn_count >= self.MAX_SPAWNS_PER_SESSION:
            return {
                "error": f"Session spawn limit reached ({self.MAX_SPAWNS_PER_SESSION}). "
                         "This prevents runaway spawning. Reset by restarting the cognitive loop.",
                "spawns_this_session": self._session_spawn_count,
                "limit": self.MAX_SPAWNS_PER_SESSION
            }

        # Check with safety layer before spawning
        session = await self._get_session()

        # Check concurrent agent limit and task deduplication (including COMPLETED agents)
        try:
            agents_url = f"{CEREBRO_URL}/agents"
            async with session.get(agents_url, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    agents_data = await resp.json()
                    all_agents = agents_data.get("agents", [])

                    # Filter for running agents only
                    running_agents = [a for a in all_agents if a.get("status") in ["running", "pending", "spawned"]]

                    # Check concurrent limit
                    if len(running_agents) >= self.MAX_CONCURRENT_AGENTS:
                        return {
                            "error": f"Concurrent agent limit reached ({self.MAX_CONCURRENT_AGENTS}). "
                                     "Wait for existing agents to complete before spawning new ones.",
                            "running_agents": len(running_agents),
                            "limit": self.MAX_CONCURRENT_AGENTS,
                            "active_agents": [{"id": a.get("agent_id"), "task": a.get("task", "")[:50]} for a in running_agents],
                            "tip": "Use check_agent_status to monitor existing agents, or wait_for_agent to wait for completion"
                        }

                    # Check task deduplication - is there already an agent working on a similar task?
                    task_lower = task.lower()
                    for agent in running_agents:
                        agent_task = agent.get("task", "").lower()
                        overlap = self._calculate_task_overlap(task_lower, agent_task)
                        if overlap > 0.4:  # 40% overlap threshold
                            return {
                                "already_running": True,
                                "agent_id": agent.get("agent_id"),
                                "message": "An agent is already working on a similar task. Use check_agent_status to monitor it.",
                                "existing_task": agent.get("task", "")[:200],
                                "tip": f"Call check_agent_status with agent_id='{agent.get('agent_id')}' to get progress"
                            }

                    # NEW: Check COMPLETED agents from last 24 hours for redundant spawning
                    from datetime import timedelta
                    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                    completed_agents = []
                    for a in all_agents:
                        if a.get("status") == "completed":
                            try:
                                created_str = a.get("created_at", "")
                                if created_str:
                                    created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                                    if created > cutoff:
                                        completed_agents.append(a)
                            except (ValueError, TypeError):
                                pass

                    # Check completed agents for similar tasks
                    similar_completed = []
                    for agent in completed_agents:
                        agent_task = agent.get("task", "").lower()
                        overlap = self._calculate_task_overlap(task_lower, agent_task)
                        if overlap > 0.4:  # 40% overlap threshold
                            similar_completed.append(agent)

                    # Check if this is a refresh request (bypasses deduplication)
                    is_refresh = self._is_refresh_request(task)

                    if is_refresh and similar_completed:
                        print("[SPAWN_AGENT] Refresh request detected - bypassing completed agent deduplication")
                        # Skip deduplication for refresh requests
                    elif similar_completed:
                        # Diminishing returns detection - HARD BLOCK if 3+ similar agents already ran (spam protection)
                        if len(similar_completed) >= 3:
                            return {
                                "error": "Diminishing returns detected",
                                "message": f"{len(similar_completed)} similar agents already ran in last 24h. "
                                           "Spawning more is unlikely to help.",
                                "similar_agents": [a.get("agent_id") or a.get("id") for a in similar_completed[:5]],
                                "suggestion": "Report findings to the user or try a completely different approach.",
                                "tip": "The same task type has been attempted multiple times. Consider: "
                                       "(1) Reviewing existing results, (2) Reporting to the user, or "
                                       "(3) Trying a fundamentally different approach."
                            }

                        # For 1-2 similar tasks: ASK the user instead of silently blocking
                        most_recent = similar_completed[0]
                        return {
                            "needs_confirmation": True,
                            "similar_agent_id": most_recent.get("agent_id") or most_recent.get("id"),
                            "similar_task": most_recent.get("task", "")[:200],
                            "completed_at": most_recent.get("completed_at"),
                            "message": "A similar task was completed recently. Ask the user if they want to run it again.",
                            "suggested_question": "I already researched something similar recently. Do you want me to do this again for updated results, or should I use the previous findings?"
                        }

        except Exception as e:
            # Non-critical - continue with spawn if we can't check
            print(f"[SPAWN_AGENT] Warning: Could not check concurrent agents: {e}")

        try:
            # First, check if Full Autonomy is enabled via safety layer
            safety_url = f"{CEREBRO_URL}/api/safety/status"
            async with session.get(safety_url, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    safety_status = await resp.json()
                    if not safety_status.get("full_autonomy_enabled", False):
                        return {
                            "error": "Full Autonomy mode is OFF. Cannot spawn agents.",
                            "message": "Enable Full Autonomy in Cerebro controls to allow agent spawning.",
                            "current_mode": "thinking_only"
                        }
                    if safety_status.get("killed", False):
                        return {
                            "error": "Kill switch is active. All actions blocked.",
                            "message": "Reset the kill switch in Cerebro controls."
                        }
        except Exception as e:
            # If we can't check safety, assume not allowed
            return {
                "error": f"Cannot verify safety status: {str(e)}",
                "message": "Safety check required before spawning."
            }

        # Build the spawn request
        spawn_request = {
            "task": task,
            "agent_type": agent_type,
            "context": f"Autonomously spawned by Cerebro cognitive loop.\n{context}".strip(),
            "expected_output": expected_output,
            "priority": priority
        }

        try:
            url = f"{CEREBRO_URL}/agents"
            async with session.post(url, json=spawn_request, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    agent_id = result.get("agent_id")

                    # Track this spawn locally
                    self._session_spawn_count += 1
                    self._spawned_agents[agent_id] = {
                        "agent_id": agent_id,
                        "task": task[:500],
                        "agent_type": agent_type,
                        "spawned_at": datetime.now(timezone.utc).isoformat(),
                        "status": "running"
                    }

                    # Also record in safety layer for persistent tracking
                    # (fire-and-forget, don't block on it)
                    try:
                        record_url = f"{CEREBRO_URL}/api/safety/record-spawn"
                        asyncio.create_task(
                            session.post(record_url, json={
                                "agent_id": agent_id,
                                "agent_type": agent_type,
                                "task": task[:500]
                            }, headers=self._get_headers())
                        )
                    except Exception:
                        pass  # Non-critical

                    await self._emit_debug("agent_spawned", {
                        "agent_id": agent_id,
                        "task": task[:100],
                        "agent_type": agent_type,
                        "session_spawn_count": self._session_spawn_count
                    })

                    return {
                        "success": True,
                        "agent_id": agent_id,
                        "message": f"Agent {agent_id} spawned successfully",
                        "task": task[:500],
                        "agent_type": agent_type,
                        "session_spawns": self._session_spawn_count,
                        "spawns_remaining": self.MAX_SPAWNS_PER_SESSION - self._session_spawn_count,
                        "tip": "Use wait_for_agent to wait for completion, or check_agent_status for progress"
                    }
                else:
                    error_text = await resp.text()
                    return {
                        "success": False,
                        "error": f"Failed to spawn agent: {resp.status}",
                        "details": error_text[:200]
                    }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error spawning agent: {str(e)}"
            }

    async def _tool_wait_for_agent(self, params: Dict) -> Any:
        """
        Wait for a spawned agent to complete.

        Polls the agent status every 5 seconds until completion or timeout.
        Returns compressed output (max 5000 chars) to prevent context bloat.
        """
        agent_id = params.get("agent_id", "")
        timeout_minutes = min(params.get("timeout_minutes", 30), 60)  # Cap at 60 min

        if not agent_id:
            return {"error": "agent_id parameter required"}

        session = await self._get_session()
        url = f"{CEREBRO_URL}/agents/{agent_id}"
        start_time = datetime.now(timezone.utc)
        timeout_seconds = timeout_minutes * 60
        poll_interval = 5  # seconds

        await self._emit_debug("agent_wait_start", {
            "agent_id": agent_id,
            "timeout_minutes": timeout_minutes
        })

        while True:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

            if elapsed > timeout_seconds:
                return {
                    "success": False,
                    "agent_id": agent_id,
                    "error": f"Timeout waiting for agent after {timeout_minutes} minutes",
                    "elapsed_seconds": int(elapsed),
                    "tip": "Use check_agent_status to see current progress"
                }

            try:
                async with session.get(url, headers=self._get_headers()) as resp:
                    if resp.status == 200:
                        agent = await resp.json()
                        status = agent.get("status", "unknown")

                        if status in ["completed", "failed"]:
                            # Agent finished - return compressed output
                            output = agent.get("output", "")
                            if len(output) > self.MAX_SPAWN_OUTPUT_CHARS:
                                # Truncate but keep start and end
                                half = self.MAX_SPAWN_OUTPUT_CHARS // 2
                                output = (
                                    output[:half] +
                                    f"\n\n... [TRUNCATED {len(output) - self.MAX_SPAWN_OUTPUT_CHARS} chars] ...\n\n" +
                                    output[-half:]
                                )

                            # Update our tracking
                            if agent_id in self._spawned_agents:
                                self._spawned_agents[agent_id]["status"] = status

                            await self._emit_debug("agent_wait_complete", {
                                "agent_id": agent_id,
                                "status": status,
                                "elapsed_seconds": int(elapsed)
                            })

                            return {
                                "success": status == "completed",
                                "agent_id": agent_id,
                                "status": status,
                                "output": output,
                                "tools_used": agent.get("tools_used", []),
                                "error": agent.get("error"),
                                "elapsed_seconds": int(elapsed)
                            }

                        # Still running - wait and poll again
                        await self._emit_debug("agent_wait_polling", {
                            "agent_id": agent_id,
                            "status": status,
                            "elapsed_seconds": int(elapsed)
                        })

                    elif resp.status == 404:
                        return {
                            "success": False,
                            "agent_id": agent_id,
                            "error": "Agent not found"
                        }

            except Exception as e:
                # Log error but keep polling
                await self._emit_debug("agent_wait_error", {
                    "agent_id": agent_id,
                    "error": str(e)
                })

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    async def _tool_check_agent_status(self, params: Dict) -> Any:
        """
        Check the current status of an agent without blocking.

        Quick status check - use wait_for_agent if you need the full output.
        """
        agent_id = params.get("agent_id", "")

        if not agent_id:
            return {"error": "agent_id parameter required"}

        session = await self._get_session()
        url = f"{CEREBRO_URL}/agents/{agent_id}"

        try:
            async with session.get(url, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    agent = await resp.json()

                    # Get output preview (first 500 chars)
                    output = agent.get("output", "")
                    output_preview = output[:500] + "..." if len(output) > 500 else output

                    return {
                        "agent_id": agent_id,
                        "status": agent.get("status", "unknown"),
                        "type": agent.get("type", "worker"),
                        "task": agent.get("task", "")[:200],
                        "output_preview": output_preview,
                        "output_length": len(output),
                        "tools_used": agent.get("tools_used", []),
                        "error": agent.get("error"),
                        "created_at": agent.get("created_at"),
                        "started_at": agent.get("started_at"),
                        "completed_at": agent.get("completed_at")
                    }

                elif resp.status == 404:
                    return {
                        "agent_id": agent_id,
                        "error": "Agent not found"
                    }
                else:
                    return {
                        "agent_id": agent_id,
                        "error": f"Failed to get status: {resp.status}"
                    }

        except Exception as e:
            return {
                "agent_id": agent_id,
                "error": f"Error checking status: {str(e)}"
            }

    def get_spawn_stats(self) -> Dict:
        """Get current spawn statistics for this session."""
        return {
            "session_spawn_count": self._session_spawn_count,
            "max_spawns_per_session": self.MAX_SPAWNS_PER_SESSION,
            "spawns_remaining": self.MAX_SPAWNS_PER_SESSION - self._session_spawn_count,
            "spawned_agents": list(self._spawned_agents.values())
        }

    def reset_session_spawns(self):
        """Reset session spawn counter (called when cognitive loop restarts)."""
        self._session_spawn_count = 0
        self._spawned_agents.clear()

    def _calculate_task_overlap(self, task1: str, task2: str) -> float:
        """Calculate word overlap between two tasks. Returns 0-1."""
        stopwords = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'will', 'have', 'been',
                     'are', 'was', 'were', 'been', 'being', 'has', 'had', 'having', 'does', 'did',
                     'doing', 'would', 'could', 'should', 'may', 'might', 'must', 'shall'}

        def get_words(text):
            return set(w for w in text.lower().split() if len(w) > 3 and w not in stopwords)

        words1 = get_words(task1)
        words2 = get_words(task2)

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _is_refresh_request(self, task: str) -> bool:
        """Detect if this is a legitimate refresh/update request that should bypass deduplication."""
        refresh_keywords = {
            'update', 'latest', 'new', 'today', 'current', 'refresh', 'again',
            'recent', 'now', 'fresh', 'updated', 'newer', 'morning', 'evening',
            'tonight', 'this week', 'this month', 'news', 'daily'
        }
        task_lower = task.lower()
        task_words = set(task_lower.split())

        # Check for refresh keywords
        if task_words & refresh_keywords:
            return True

        # Check for phrases that indicate refresh intent
        refresh_phrases = [
            'what is new', "what's new", 'catch up', 'check again', 'look again',
            'any new', 'any updates', 'what happened', 'today\'s', "today's"
        ]
        for phrase in refresh_phrases:
            if phrase in task_lower:
                return True

        return False

    # ========== Goal Pursuit Tools ==========

    async def _tool_get_active_goals(self, params: Dict) -> Any:
        """
        Get all active goals with progress metrics and ready subtasks.

        This is Cerebro's mission control - shows what goals are being pursued
        and what subtasks are ready for execution.
        """
        include_ready = params.get("include_ready_subtasks", True)

        try:
            from .goal_pursuit import get_goal_pursuit_engine
            from .progress_tracker import get_progress_tracker

            engine = get_goal_pursuit_engine()
            tracker = get_progress_tracker()

            goals_data = []
            for goal in engine.get_active_goals():
                progress = tracker.calculate_pacing(goal)

                goal_info = {
                    "goal_id": goal.goal_id,
                    "description": goal.description,
                    "target_value": goal.target_value,
                    "target_unit": goal.target_unit,
                    "current_value": goal.current_value,
                    "deadline": goal.deadline,
                    "priority": goal.priority,
                    "progress_percentage": progress.progress_percentage,
                    "pacing_score": progress.pacing_score,
                    "velocity": progress.velocity,
                    "risk_level": progress.risk_level,
                    "days_remaining": progress.days_remaining,
                    "milestones_count": len(goal.milestones),
                    "failed_approaches": goal.failed_approaches[:3]  # Top 3 to avoid
                }

                if include_ready:
                    ready = engine.get_ready_subtasks(goal.goal_id)
                    goal_info["ready_subtasks"] = [
                        {
                            "subtask_id": s.subtask_id,
                            "description": s.description,
                            "agent_type": s.agent_type,
                            "attempts": s.attempts
                        }
                        for s in ready[:5]  # Top 5 ready subtasks
                    ]

                    # Also get active milestone info
                    active_milestone = engine.get_active_milestone(goal.goal_id)
                    if active_milestone:
                        goal_info["active_milestone"] = {
                            "milestone_id": active_milestone.milestone_id,
                            "description": active_milestone.description,
                            "target_percentage": active_milestone.target_percentage,
                            "expanded": active_milestone.expanded
                        }

                goals_data.append(goal_info)

            # Sort by priority and pacing (urgent goals first)
            priority_order = {"high": 0, "medium": 1, "low": 2}
            goals_data.sort(key=lambda g: (
                priority_order.get(g["priority"], 1),
                g["pacing_score"]  # Lower pacing = more urgent
            ))

            return {
                "success": True,
                "goals_count": len(goals_data),
                "goals": goals_data
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get active goals: {str(e)}",
                "goals": []
            }

    async def _tool_update_goal_progress(self, params: Dict) -> Any:
        """
        Update progress on a goal.

        Call this after completing subtasks or achieving milestones.
        """
        goal_id = params.get("goal_id", "")
        delta = params.get("delta")
        new_value = params.get("new_value")
        note = params.get("note", "")

        if not goal_id:
            return {"success": False, "error": "goal_id is required"}

        if delta is None and new_value is None:
            return {"success": False, "error": "Either delta or new_value is required"}

        try:
            from .goal_pursuit import get_goal_pursuit_engine

            engine = get_goal_pursuit_engine()
            goal = engine.update_goal_progress(
                goal_id=goal_id,
                delta=delta,
                new_value=new_value,
                note=note
            )

            if goal:
                return {
                    "success": True,
                    "goal_id": goal.goal_id,
                    "new_value": goal.current_value,
                    "progress_percentage": goal.get_progress_percentage(),
                    "status": goal.status,
                    "note": note
                }
            else:
                return {"success": False, "error": f"Goal not found: {goal_id}"}

        except Exception as e:
            return {"success": False, "error": f"Failed to update progress: {str(e)}"}

    async def _tool_complete_subtask(self, params: Dict) -> Any:
        """
        Mark a subtask as completed.

        This triggers:
        - Milestone progress check
        - Potential milestone completion
        - Lazy expansion of next milestone if needed
        """
        subtask_id = params.get("subtask_id", "")
        result = params.get("result", "")

        if not subtask_id:
            return {"success": False, "error": "subtask_id is required"}
        if not result:
            return {"success": False, "error": "result is required"}

        try:
            from .goal_pursuit import get_goal_pursuit_engine
            from .goal_decomposer import get_goal_decomposer

            engine = get_goal_pursuit_engine()
            decomposer = get_goal_decomposer()

            subtask = engine.complete_subtask(subtask_id, result)

            if subtask:
                # Check if we should expand the next milestone
                milestone = engine._milestones.get(subtask.milestone_id)
                if milestone:
                    should_expand, next_id = await decomposer.should_expand_next(milestone.goal_id)
                    if should_expand and next_id:
                        await decomposer._expand_next_milestone(
                            engine.get_goal(milestone.goal_id),
                            engine.get_milestones_for_goal(milestone.goal_id)
                        )

                return {
                    "success": True,
                    "subtask_id": subtask.subtask_id,
                    "status": subtask.status,
                    "duration_ms": subtask.duration_ms,
                    "result_recorded": True
                }
            else:
                return {"success": False, "error": f"Subtask not found: {subtask_id}"}

        except Exception as e:
            return {"success": False, "error": f"Failed to complete subtask: {str(e)}"}

    async def _tool_fail_subtask(self, params: Dict) -> Any:
        """
        Mark a subtask as failed and record Reflexion learning.

        After 3 failures, the subtask is marked as permanently failed
        and the learning is recorded as an antipattern.
        """
        subtask_id = params.get("subtask_id", "")
        error = params.get("error", "")
        learning = params.get("learning", "")

        if not subtask_id:
            return {"success": False, "error": "subtask_id is required"}
        if not error:
            return {"success": False, "error": "error description is required"}

        try:
            from .goal_pursuit import get_goal_pursuit_engine, SubtaskStatus

            engine = get_goal_pursuit_engine()
            subtask = engine.fail_subtask(subtask_id, error, learning)

            if subtask:
                # If max attempts reached, escalate
                if subtask.status == SubtaskStatus.FAILED.value:
                    escalation_msg = "Max attempts reached - escalating to the user"
                else:
                    escalation_msg = f"Will retry ({subtask.max_attempts - subtask.attempts} attempts remaining)"

                return {
                    "success": True,
                    "subtask_id": subtask.subtask_id,
                    "status": subtask.status,
                    "attempts": subtask.attempts,
                    "max_attempts": subtask.max_attempts,
                    "learning_recorded": bool(learning),
                    "escalation": escalation_msg
                }
            else:
                return {"success": False, "error": f"Subtask not found: {subtask_id}"}

        except Exception as e:
            return {"success": False, "error": f"Failed to mark subtask as failed: {str(e)}"}

    async def _tool_decompose_goal(self, params: Dict) -> Any:
        """
        Decompose a goal into milestones and subtasks using HTN planning.

        This uses the LLM to break down the goal into:
        - 3-5 milestones (each ~20-30% of goal)
        - Subtasks for the first milestone (lazy expansion)
        """
        goal_id = params.get("goal_id", "")

        if not goal_id:
            return {"success": False, "error": "goal_id is required"}

        try:
            from .goal_decomposer import get_goal_decomposer

            decomposer = get_goal_decomposer()
            result = await decomposer.decompose_goal(goal_id)

            return {
                "success": result.success,
                "goal_id": result.goal_id,
                "milestones_created": result.milestones_created,
                "subtasks_created": result.subtasks_created,
                "error": result.error
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to decompose goal: {str(e)}"}

    async def _tool_get_next_subtask(self, params: Dict) -> Any:
        """
        Get the highest-priority subtask that's ready to execute.

        Returns the next actionable task along with context
        for spawning an appropriate agent.
        """
        goal_id = params.get("goal_id")

        try:
            from .goal_pursuit import get_goal_pursuit_engine
            from .progress_tracker import get_progress_tracker

            engine = get_goal_pursuit_engine()
            tracker = get_progress_tracker()

            # If no goal specified, pick from highest priority goal
            if not goal_id:
                goals = engine.get_goals_with_progress()
                if not goals:
                    return {
                        "success": True,
                        "has_task": False,
                        "message": "No active goals with subtasks"
                    }
                # Get first goal with ready subtasks
                for g in goals:
                    if g["ready_subtasks"]:
                        goal_id = g["goal"]["goal_id"]
                        break

            if not goal_id:
                return {
                    "success": True,
                    "has_task": False,
                    "message": "No goals have ready subtasks"
                }

            subtask = engine.get_next_subtask(goal_id)

            if not subtask:
                return {
                    "success": True,
                    "has_task": False,
                    "goal_id": goal_id,
                    "message": "No ready subtasks for this goal"
                }

            # Get context for the agent
            goal = engine.get_goal(goal_id)
            milestone = engine._milestones.get(subtask.milestone_id)
            progress = tracker.calculate_pacing(goal) if goal else None

            return {
                "success": True,
                "has_task": True,
                "subtask": {
                    "subtask_id": subtask.subtask_id,
                    "description": subtask.description,
                    "agent_type": subtask.agent_type,
                    "attempts": subtask.attempts,
                    "max_attempts": subtask.max_attempts,
                    "depends_on": subtask.depends_on
                },
                "context": {
                    "goal_id": goal_id,
                    "goal_description": goal.description if goal else "",
                    "milestone_description": milestone.description if milestone else "",
                    "pacing_score": progress.pacing_score if progress else 1.0,
                    "risk_level": progress.risk_level if progress else "low",
                    "failed_approaches": goal.failed_approaches if goal else []
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to get next subtask: {str(e)}"}


    # ========== Adaptive Browser Learning Tools ==========

    async def _tool_explore_website(self, params: Dict) -> Any:
        """
        Explore a website to accomplish a goal and potentially create a skill.

        Uses LLM-guided exploration to navigate unknown websites,
        then optionally converts the successful exploration into a reusable skill.
        """
        goal = params.get("goal", "")
        start_url = params.get("start_url", "")
        max_steps = params.get("max_steps", 20)
        create_skill = params.get("create_skill", True)
        skill_name = params.get("skill_name", "")

        if not goal:
            return {"error": "goal parameter required"}
        if not start_url:
            return {"error": "start_url parameter required"}

        try:
            from .adaptive_explorer import AdaptiveExplorer
            from .page_understanding import PageUnderstanding
            from .element_fingerprint import SelfHealingLocator
            from .ollama_client import get_ollama_client

            gen = self._get_skill_generator()
            page = await gen._ensure_browser()

            # Initialize explorer
            ollama = get_ollama_client()
            explorer = AdaptiveExplorer(
                ollama_client=ollama,
                page_understanding=PageUnderstanding(),
                fingerprint_locator=SelfHealingLocator(),
            )

            # Run exploration
            session = await explorer.explore(
                page=page,
                goal=goal,
                start_url=start_url,
                max_steps=max_steps,
            )

            result = {
                "success": session.status == "succeeded",
                "session_id": session.session_id,
                "status": session.status,
                "steps_taken": len(session.steps),
                "final_url": session.steps[-1].page_url if session.steps else start_url,
            }

            # Convert to skill if successful and requested
            if session.status == "succeeded" and create_skill:
                try:
                    skill = await explorer.convert_to_skill(
                        session,
                        skill_name=skill_name or f"Auto: {goal[:40]}",
                        skill_description=f"Automatically generated: {goal}",
                    )
                    result["skill_created"] = True
                    result["skill_id"] = skill["skill_id"]
                except Exception as e:
                    result["skill_created"] = False
                    result["skill_error"] = str(e)

            return result

        except Exception as e:
            return {"error": f"Exploration failed: {str(e)}"}

    async def _tool_execute_subtask_with_skill(self, params: Dict) -> Any:
        """
        Execute a subtask using its associated skill or exploration.

        If subtask has skill_id: execute the skill
        If subtask has exploration_goal: explore, create skill, then use it
        """
        subtask_id = params.get("subtask_id", "")

        if not subtask_id:
            return {"error": "subtask_id parameter required"}

        try:
            from .goal_pursuit import get_goal_pursuit_engine
            from .adaptive_explorer import AdaptiveExplorer
            from .page_understanding import PageUnderstanding
            from .element_fingerprint import SelfHealingLocator
            from .ollama_client import get_ollama_client

            engine = get_goal_pursuit_engine()
            subtask = engine._subtasks.get(subtask_id)

            if not subtask:
                return {"error": f"Subtask not found: {subtask_id}"}

            # Mark subtask as in progress
            engine.start_subtask(subtask_id)

            gen = self._get_skill_generator()

            # Case 1: Subtask has an existing skill
            if subtask.skill_id:
                result = await gen.execute_skill(
                    subtask.skill_id,
                    subtask.skill_parameters
                )

                if result.success:
                    engine.complete_subtask(subtask_id, f"Skill executed: {result.output}")
                    return {
                        "success": True,
                        "method": "existing_skill",
                        "skill_id": subtask.skill_id,
                        "result": result.output,
                        "steps_completed": result.steps_completed,
                    }
                else:
                    engine.fail_subtask(subtask_id, result.error or "Skill execution failed")
                    return {
                        "success": False,
                        "method": "existing_skill",
                        "skill_id": subtask.skill_id,
                        "error": result.error,
                    }

            # Case 2: Subtask has exploration goal - explore and create skill
            elif subtask.exploration_goal:
                page = await gen._ensure_browser()
                ollama = get_ollama_client()

                explorer = AdaptiveExplorer(
                    ollama_client=ollama,
                    page_understanding=PageUnderstanding(),
                    fingerprint_locator=SelfHealingLocator(),
                )

                # Determine start URL from exploration goal or current page
                start_url = page.url if page.url and page.url != "about:blank" else "https://www.google.com"

                session = await explorer.explore(
                    page=page,
                    goal=subtask.exploration_goal,
                    start_url=start_url,
                    max_steps=20,
                )

                if session.status == "succeeded":
                    # Convert to skill for future use
                    try:
                        skill = await explorer.convert_to_skill(
                            session,
                            skill_name=f"Skill: {subtask.description[:40]}",
                            skill_description=subtask.exploration_goal,
                        )
                        subtask.generated_skill_id = skill["skill_id"]
                    except Exception:
                        pass  # Skill creation optional

                    engine.complete_subtask(subtask_id, f"Exploration succeeded: {session.final_result}")
                    return {
                        "success": True,
                        "method": "exploration",
                        "session_id": session.session_id,
                        "steps_taken": len(session.steps),
                        "skill_created": subtask.generated_skill_id is not None,
                        "generated_skill_id": subtask.generated_skill_id,
                    }
                else:
                    engine.fail_subtask(
                        subtask_id,
                        f"Exploration {session.status}",
                        learning=f"Failed approaches: {session.get_failed_approaches()}"
                    )
                    return {
                        "success": False,
                        "method": "exploration",
                        "session_id": session.session_id,
                        "status": session.status,
                        "steps_taken": len(session.steps),
                        "error": f"Exploration {session.status}",
                    }

            # Case 3: No skill or exploration - return info
            else:
                return {
                    "success": False,
                    "error": "Subtask has no skill_id or exploration_goal",
                    "subtask_description": subtask.description,
                    "agent_type": subtask.agent_type,
                    "message": "Use spawn_agent to execute this subtask with the appropriate agent type"
                }

        except Exception as e:
            return {"error": f"Subtask execution failed: {str(e)}"}

    async def _tool_verify_skill(self, params: Dict) -> Any:
        """
        Verify a skill works correctly and auto-heal broken selectors.
        """
        skill_id = params.get("skill_id", "")
        dry_run = params.get("dry_run", True)

        if not skill_id:
            return {"error": "skill_id parameter required"}

        try:
            from .skill_verifier import SkillVerifier
            from .element_fingerprint import SelfHealingLocator

            gen = self._get_skill_generator()
            verifier = SkillVerifier(
                skill_generator=gen,
                self_healing_locator=SelfHealingLocator(),
            )

            result = await verifier.verify_skill(
                skill_id=skill_id,
                dry_run=dry_run,
                auto_heal=True,
            )

            return {
                "success": result.success,
                "skill_id": result.skill_id,
                "steps_total": result.steps_total,
                "steps_passed": result.steps_passed,
                "healed_selectors": result.healed_selectors,
                "failure_step": result.failure_step,
                "error": result.error,
            }

        except Exception as e:
            return {"error": f"Verification failed: {str(e)}"}

    async def _tool_record_action(self, params: Dict) -> Any:
        """
        Start or stop recording browser actions for skill creation.
        """
        action = params.get("action", "")  # start, stop
        name = params.get("name", "")
        description = params.get("description", "")

        if action not in ("start", "stop"):
            return {"error": "action must be 'start' or 'stop'"}

        try:
            from .action_recorder import ActionRecorder

            if not hasattr(self, '_action_recorder'):
                self._action_recorder = ActionRecorder()

            recorder = self._action_recorder

            if action == "start":
                if not name:
                    return {"error": "name parameter required for start"}

                gen = self._get_skill_generator()
                page = await gen._ensure_browser()

                session = await recorder.start_recording(
                    page=page,
                    name=name,
                    description=description or "",
                )

                return {
                    "success": True,
                    "action": "started",
                    "recording_id": session.recording_id,
                    "message": "Recording started. Interact with the browser, then call with action='stop'"
                }

            elif action == "stop":
                session = await recorder.stop_recording()

                return {
                    "success": True,
                    "action": "stopped",
                    "recording_id": session.recording_id,
                    "actions_recorded": len(session.actions),
                    "message": "Recording stopped. Use generate_skill with recording_id to create a skill"
                }

        except Exception as e:
            return {"error": f"Recording action failed: {str(e)}"}


# Singleton instance
_tools_instance: Optional[CognitiveTools] = None

def get_cognitive_tools(debug_callback=None) -> CognitiveTools:
    """Get or create the cognitive tools instance."""
    global _tools_instance
    if _tools_instance is None:
        _tools_instance = CognitiveTools(debug_callback)
    elif debug_callback:
        _tools_instance._debug_callback = debug_callback
    return _tools_instance
