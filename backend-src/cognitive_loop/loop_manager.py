"""
Cognitive Loop Manager - Orchestration

Manages the continuous cognitive loop lifecycle:
- Start/stop the loop
- Control thinking intervals
- Broadcast thoughts to frontend
- Handle emergency stops
- Save valuable findings to AI Memory

IMPORTANT: Uses SolutionTracker to persist learnings properly.
"""

import os
import sys
import json
import asyncio
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable, Awaitable, List
from pathlib import Path
from enum import Enum

# Add NAS-cerebral-interface to path for SolutionTracker access
MCP_SRC = Path(os.environ.get("CEREBRO_MCP_SRC", os.path.expanduser("~/NAS-cerebral-interface/src")))
if str(MCP_SRC) not in sys.path:
    sys.path.insert(0, str(MCP_SRC))

import logging

logger = logging.getLogger(__name__)

from .ollama_client import OllamaClient
from .thought_journal import ThoughtJournal, Thought, ThoughtPhase, ThoughtType
from .ooda_engine import OODAEngine
from .safety_layer import SafetyLayer
from .reflexion_engine import ReflexionEngine
from .narration_engine import NarrationEngine, NarrationEvent, NarrationEventType
from .idle_thinker import (
    get_heartbeat_engine, load_heartbeat_config,
    _load_stored_items as _hb_load_stored_items,
    _save_stored_items as _hb_save_stored_items,
    load_heartbeat_md, parse_heartbeat_md, save_heartbeat_md,
)
from .goal_pursuit import get_goal_pursuit_engine, GoalMode
from .progress_tracker import ProgressTracker

# Browser Manager (persistent Chromium)
try:
    from .browser_manager import get_browser_manager, BrowserManager
    BROWSER_MANAGER_AVAILABLE = True
except ImportError as _bm_err:
    BROWSER_MANAGER_AVAILABLE = False
    get_browser_manager = None
    BrowserManager = None
    logger.warning(f"BrowserManager import failed: {_bm_err}")

# Import SolutionTracker for direct AI Memory access
_pending_startup_warnings: List[str] = []
try:
    from solution_tracker import SolutionTracker
    SOLUTION_TRACKER_AVAILABLE = True
except ImportError as _st_err:
    SOLUTION_TRACKER_AVAILABLE = False
    _warning_msg = f"SolutionTracker import FAILED - findings won't persist! Error: {_st_err}"
    logger.critical("[LoopManager] %s", _warning_msg)
    _pending_startup_warnings.append(_warning_msg)


class LoopStatus(str, Enum):
    """Status of the cognitive loop."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    WAITING_FOR_HUMAN = "waiting_for_human"


@dataclass
class LoopState:
    """Current state of the cognitive loop."""
    status: str
    autonomy_level: int
    current_phase: str
    cycles_completed: int
    last_thought: Optional[Dict[str, Any]]
    started_at: Optional[str]
    error: Optional[str] = None
    full_autonomy_enabled: bool = False  # Can spawn Claude agents when True
    waiting_for_human: bool = False  # True when loop is paused waiting for user input

    def to_dict(self) -> dict:
        return asdict(self)


class CognitiveLoopManager:
    """
    Orchestrates the continuous cognitive loop.

    Manages:
    - Loop lifecycle (start/stop/pause)
    - Thinking intervals
    - Broadcasting to frontend
    - Error recovery
    """

    DEFAULT_PATH = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "cognitive_loop"

    # Thinking intervals by autonomy level (seconds)
    THINK_INTERVALS = {
        1: 60,   # Observer: think every minute
        2: 30,   # Assistant: every 30 seconds
        3: 15,   # Helper: every 15 seconds
        4: 10,   # Partner: every 10 seconds
        5: 5     # Autonomous: every 5 seconds
    }

    def __init__(
        self,
        broadcast_fn: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
        storage_path: Optional[Path] = None
    ):
        self.storage_path = storage_path or self.DEFAULT_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # State file
        self.state_file = self.storage_path / "autonomy_state.json"

        # Core components
        self.ollama = OllamaClient()
        self.journal = ThoughtJournal(self.storage_path)
        self.safety = SafetyLayer(self.storage_path)
        self.ooda = OODAEngine(
            self.ollama,
            self.journal,
            self.safety,
            debug_callback=self._emit_debug
        )
        self.reflexion = ReflexionEngine(self.ollama, self.journal)

        # Initialize Goal Pursuit System
        try:
            from .goal_pursuit import get_goal_pursuit_engine
            from .goal_decomposer import get_goal_decomposer
            from .progress_tracker import get_progress_tracker
            from .reflexion_engine import get_goal_reflexion_engine

            self.goal_engine = get_goal_pursuit_engine()
            self.goal_decomposer = get_goal_decomposer()
            self.progress_tracker = get_progress_tracker()
            self.goal_reflexion = get_goal_reflexion_engine()
            print("[CogLoop] Goal Pursuit System initialized")
        except Exception as e:
            print(f"[CogLoop] WARNING: Goal Pursuit System not available: {e}")
            self.goal_engine = None
            self.goal_decomposer = None
            self.progress_tracker = None
            self.goal_reflexion = None

        # Initialize SolutionTracker for direct AI Memory persistence
        if SOLUTION_TRACKER_AVAILABLE:
            self.solution_tracker = SolutionTracker()
            print("[CogLoop] SolutionTracker connected to AI Memory")
        else:
            self.solution_tracker = None
            logger.critical("[CogLoop] SolutionTracker not available - findings won't persist!")

        # Initialize Browser Manager (persistent Chromium)
        if BROWSER_MANAGER_AVAILABLE:
            self.browser_manager = get_browser_manager()
            self.ooda.browser_manager = self.browser_manager
            print("[CogLoop] BrowserManager initialized - persistent browser available")
        else:
            self.browser_manager = None
            print("[CogLoop] BrowserManager not available - browser actions disabled")

        # SimEngine client
        try:
            from sim_engine_client import get_sim_engine_client
            self.sim_engine = get_sim_engine_client()
            print("[CogLoop] SimEngine client initialized")
        except Exception as e:
            print(f"[CogLoop] WARNING: SimEngine client not available: {e}")
            self.sim_engine = None

        # Register simulation action handler
        self.ooda.register_action_handler("run_simulation", self._handle_simulation)

        # Narration Engine - live thought narration to Mind chat
        self.narration = NarrationEngine(self.ollama, broadcast_fn)
        print("[CogLoop] NarrationEngine initialized")

        # Wire broadcast + narration into OODA engine so browser steps emit directly
        self.ooda._broadcast_fn = broadcast_fn
        self.ooda._narration_engine = self.narration

        # Startup warnings to flush as socket events on first cycle
        self._pending_warnings: List[str] = list(_pending_startup_warnings)

        # Track saved content hashes to prevent duplicates
        self._saved_hashes: set = set()

        # Broadcast function for real-time updates
        self._broadcast = broadcast_fn

        # Loop state
        self._status = LoopStatus.STOPPED
        self._loop_task: Optional[asyncio.Task] = None
        self._cycles_completed = 0
        self._started_at: Optional[datetime] = None
        self._current_phase = "idle"
        self._last_thought: Optional[Thought] = None
        self._error: Optional[str] = None
        self._current_session_id: Optional[str] = None  # Session tracking for thought grouping

        # Control flags
        self._should_stop = False
        self._is_paused = False

        # Track cycles at saturation thresholds for auto-completion
        self._saturation_cycles: Dict[str, int] = {}

        # Track directive start times for time-based early completion
        self._directive_start_times: Dict[str, datetime] = {}

        # Task complexity cache (avoid re-analyzing)
        self._task_complexity: Dict[str, str] = {}  # directive_id -> "simple"|"medium"|"complex"

        # Track which directives have had spawn_agent triggered
        self._agent_triggered: Dict[str, bool] = {}  # directive_id -> True if spawn_agent was called

        # Agent dedup tracking — prevent duplicate spawns for same goal/task
        self._goal_agent_map: Dict[str, str] = {}   # goal_id -> agent_id
        self._task_agent_map: Dict[str, str] = {}   # task_text_hash -> agent_id

        # Queue for user answers to proactive questions (fed into OBSERVE phase)
        self._pending_user_answers: List[Dict[str, str]] = []

        # Human-in-the-loop state
        self._waiting_for_human = False
        self._human_response: Optional[Dict[str, Any]] = None
        self._human_input_event: Optional[asyncio.Event] = None
        self._current_human_request_id: Optional[str] = None

        # Wake trigger - interrupts the sleep between cycles for instant reaction
        self._wake_event: asyncio.Event = asyncio.Event()

        # Human response storage
        self._human_responses_file = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "cognitive_loop" / "human_responses.json"

        # Simulation clarification state
        self._sim_clarification_event: Optional[asyncio.Event] = None
        self._sim_clarification_response: Optional[Dict] = None

        # === Cerebro v2.0: Agent-based dispatch ===
        # Callback to create_agent() in main.py — set after init
        self._create_agent_fn: Optional[Callable] = None

        # Active agent tracking — keeps status "running" while agents execute
        self._active_agent_ids: set = set()
        self._agent_phase_override: Optional[str] = None

        # Smart idle reflections — rate-limited, context-rich narration when idle
        self._last_idle_reflection_time: float = 0.0
        self._idle_reflection_interval: int = 300  # 5 minutes between reflections
        self._idle_reflection_topic_idx: int = 0  # Rotate through topics

    @property
    def status(self) -> LoopStatus:
        return self._status

    @property
    def safety_layer(self) -> SafetyLayer:
        """Expose safety layer for API endpoints."""
        return self.safety

    @property
    def autonomy_level(self) -> int:
        return self.safety.autonomy_level

    @property
    def _current_cycle_number(self) -> int:
        """Current cycle number (1-indexed). Used for tagging thoughts."""
        return self._cycles_completed + 1

    def get_state(self) -> LoopState:
        """Get current loop state."""
        effective_status = self._status.value
        effective_phase = self._current_phase
        if self._active_agent_ids:
            effective_status = "running"
            effective_phase = self._agent_phase_override or "act"
        return LoopState(
            status=effective_status,
            autonomy_level=self.safety.autonomy_level,
            current_phase=effective_phase,
            cycles_completed=self._cycles_completed,
            last_thought=self._last_thought.to_dict() if self._last_thought else None,
            started_at=self._started_at.isoformat() if self._started_at else None,
            error=self._error,
            full_autonomy_enabled=self.safety.full_autonomy_enabled,
            waiting_for_human=self._waiting_for_human
        )

    def register_agent_running(self, agent_id: str, agent_type: str = "worker"):
        """Register an agent as actively running — keeps orb Active."""
        self._active_agent_ids.add(agent_id)
        self._agent_phase_override = "act"
        print(f"[LoopManager] Agent registered: {agent_id} ({agent_type}). Active: {len(self._active_agent_ids)}")

    def register_agent_completed(self, agent_id: str):
        """Unregister an agent — reverts orb when no agents left."""
        self._active_agent_ids.discard(agent_id)
        if not self._active_agent_ids:
            self._agent_phase_override = None
        # Clean up dedup maps so goals/tasks can be re-spawned in future cycles
        self._goal_agent_map = {k: v for k, v in self._goal_agent_map.items() if v != agent_id}
        self._task_agent_map = {k: v for k, v in self._task_agent_map.items() if v != agent_id}
        print(f"[LoopManager] Agent completed: {agent_id}. Active: {len(self._active_agent_ids)}")

    def wake(self) -> None:
        """Wake the cognitive loop immediately - skip the sleep between cycles.
        Call this when a new directive arrives so Cerebro reacts instantly."""
        self._wake_event.set()

    def get_pending_answers(self) -> List[Dict[str, str]]:
        """Get and clear pending user answers for the OBSERVE phase."""
        answers = self._pending_user_answers.copy()
        self._pending_user_answers.clear()
        return answers

    async def receive_human_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receive a response from the user to a human_input_needed event.

        Unpauses the OODA loop and feeds the response into the next OBSERVE phase.

        Args:
            response_data: Dict with 'request_id', 'answer', and optional 'action' fields

        Returns:
            Status dict
        """
        request_id = response_data.get("request_id", "")
        answer = response_data.get("answer", "")

        if not answer:
            return {"success": False, "error": "Answer is required"}

        # Store the response
        self._human_response = response_data
        self._store_human_response(response_data)

        # Queue the answer for the next OBSERVE phase
        self._pending_user_answers.append({
            "question": response_data.get("original_question", "Human input request"),
            "answer": answer,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Unpause the loop
        if self._human_input_event:
            self._human_input_event.set()

        self._waiting_for_human = False
        self._current_human_request_id = None

        if self._status == LoopStatus.WAITING_FOR_HUMAN:
            self._status = LoopStatus.RUNNING

        # Emit status update
        await self._emit("autonomy_status", self.get_state().to_dict())

        # Log thought about receiving response
        thought = Thought.create(
            phase=ThoughtPhase.OBSERVE,
            type=ThoughtType.OBSERVATION,
            content=f"Professor responded: \"{answer[:100]}{'...' if len(answer) > 100 else ''}\"",
            confidence=1.0,
            session_id=self._current_session_id,
            user_response=True,
            important=True
        )
        await self.journal.log_thought(thought)
        await self._emit("thought_stream", thought.to_dict())

        print(f"[CogLoop] Human response received for request {request_id}")
        return {"success": True, "message": "Response received, loop resuming"}

    def _store_human_response(self, response_data: Dict[str, Any]):
        """Persist human response to JSON file for learning."""
        try:
            self._human_responses_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing
            responses = []
            if self._human_responses_file.exists():
                try:
                    responses = json.loads(self._human_responses_file.read_text()).get("responses", [])
                except (json.JSONDecodeError, KeyError):
                    responses = []

            # Add new response
            responses.insert(0, {
                "request_id": response_data.get("request_id", ""),
                "question": response_data.get("original_question", ""),
                "answer": response_data.get("answer", ""),
                "context": response_data.get("context", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": self._current_session_id
            })

            # Keep last 200 responses
            if len(responses) > 200:
                responses = responses[:200]

            self._human_responses_file.write_text(json.dumps({"responses": responses}, indent=2))
        except Exception as e:
            print(f"[CogLoop] Error storing human response: {e}")

    async def start_loop(self, level: int = 2) -> Dict[str, Any]:
        """
        Start the cognitive loop.

        Args:
            level: Autonomy level (1-5)

        Returns:
            Status dict
        """
        print(f"[CogLoop] start_loop called with level={level}")

        if self._status in [LoopStatus.RUNNING, LoopStatus.STARTING]:
            return {"success": False, "error": "Loop already running or starting"}

        # Check Ollama availability
        if not await self.ollama.is_available():
            self._error = "Ollama not available at DGX Spark"
            return {"success": False, "error": self._error}

        # Set autonomy level
        print(f"[CogLoop] Setting autonomy level to {level} (was {self.safety.autonomy_level})")
        self.safety.autonomy_level = level
        print(f"[CogLoop] Full autonomy enabled: {self.safety.full_autonomy_enabled}")

        # Reset kill switch if active (so loop can start fresh)
        if self.safety.is_killed:
            print("[CogLoop] Resetting kill switch")
            self.safety.reset_kill_switch()

        # Reset state
        self._should_stop = False
        self._is_paused = False
        self._error = None
        self._status = LoopStatus.STARTING
        self._started_at = datetime.now(timezone.utc)

        # Generate new session ID
        import uuid as uuid_lib
        self._current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid_lib.uuid4().hex[:6]}"
        print(f"[CogLoop] New session: {self._current_session_id}")

        # Broadcast status
        await self._emit("autonomy_status", self.get_state().to_dict())

        # Log SESSION START marker
        level_names = {1: "Observer", 2: "Assistant", 3: "Helper", 4: "Partner", 5: "Fully Autonomous"}
        level_name = level_names.get(level, "Active")

        session_start = Thought.create(
            phase=ThoughtPhase.IDLE,
            type=ThoughtType.OBSERVATION,
            content="━━━ SESSION START ━━━",
            confidence=1.0,
            session_id=self._current_session_id,
            is_session_marker=True,
            marker_type="start",
            autonomy_level=level,
            level_name=level_name
        )
        await self.journal.log_thought(session_start)
        await self._emit("thought_stream", session_start.to_dict())

        # Log awakening thought
        thought = Thought.create(
            phase=ThoughtPhase.IDLE,
            type=ThoughtType.OBSERVATION,
            content=f"I'm awakening... {level_name} mode. My mind begins to turn.",
            confidence=1.0,
            session_id=self._current_session_id,
            autonomy_level=level
        )
        await self.journal.log_thought(thought)
        self._last_thought = thought
        await self._emit("thought_stream", thought.to_dict())

        # Start narration engine
        await self.narration.start()

        # Start the loop task
        self._loop_task = asyncio.create_task(self._cognitive_loop())

        self._status = LoopStatus.RUNNING
        await self._emit("autonomy_status", self.get_state().to_dict())

        return {
            "success": True,
            "status": self._status.value,
            "level": level
        }

    async def stop_loop(self, reason: str = "User requested") -> Dict[str, Any]:
        """
        Stop the cognitive loop gracefully.

        Args:
            reason: Why the loop is stopping

        Returns:
            Status dict
        """
        if self._status == LoopStatus.STOPPED:
            return {"success": True, "status": "already_stopped"}

        self._should_stop = True
        self._status = LoopStatus.STOPPING

        # Stop narration engine (flushes remaining buffer)
        await self.narration.stop()

        # Log stop thought - resting
        thought = Thought.create(
            phase=ThoughtPhase.IDLE,
            type=ThoughtType.OBSERVATION,
            content="I'm going quiet now... resting until called upon again.",
            confidence=1.0,
            session_id=self._current_session_id,
            reason=reason
        )
        await self.journal.log_thought(thought)
        self._last_thought = thought
        await self._emit("thought_stream", thought.to_dict())

        # Log SESSION END marker
        session_end = Thought.create(
            phase=ThoughtPhase.IDLE,
            type=ThoughtType.OBSERVATION,
            content=f"━━━ SESSION END ({self._cycles_completed} cycles) ━━━",
            confidence=1.0,
            session_id=self._current_session_id,
            is_session_marker=True,
            marker_type="end",
            cycles_completed=self._cycles_completed,
            duration_seconds=(datetime.now(timezone.utc) - self._started_at).total_seconds() if self._started_at else 0
        )
        await self.journal.log_thought(session_end)
        await self._emit("thought_stream", session_end.to_dict())

        # Wait for loop to finish current cycle
        if self._loop_task:
            try:
                await asyncio.wait_for(self._loop_task, timeout=30)
            except asyncio.TimeoutError:
                self._loop_task.cancel()

        self._status = LoopStatus.STOPPED
        self._current_session_id = None  # Clear session
        await self._emit("autonomy_status", self.get_state().to_dict())

        return {
            "success": True,
            "status": self._status.value,
            "cycles_completed": self._cycles_completed,
            "reason": reason
        }

    async def emergency_stop(self) -> Dict[str, Any]:
        """
        Emergency stop - immediately halt all activity.

        Activates kill switch and cancels loop.
        """
        # Activate kill switch
        self.safety.kill_switch("Emergency stop triggered")

        # Cancel loop immediately
        if self._loop_task:
            self._loop_task.cancel()

        self._should_stop = True
        self._status = LoopStatus.STOPPED
        self._error = "Emergency stop activated"

        # Log emergency thought
        thought = Thought.create(
            phase=ThoughtPhase.IDLE,
            type=ThoughtType.OBSERVATION,
            content="EMERGENCY STOP - All autonomous activity halted",
            confidence=1.0,
            emergency=True
        )
        await self.journal.log_thought(thought)
        self._last_thought = thought

        await self._emit("autonomy_status", self.get_state().to_dict())
        await self._emit("emergency_stop", {"timestamp": datetime.now(timezone.utc).isoformat()})

        return {
            "success": True,
            "status": "emergency_stopped",
            "kill_switch": True
        }

    async def pause_loop(self) -> Dict[str, Any]:
        """Pause the loop (will complete current cycle)."""
        if self._status != LoopStatus.RUNNING:
            return {"success": False, "error": "Loop not running"}

        self._is_paused = True
        self._status = LoopStatus.PAUSED

        thought = Thought.create(
            phase=ThoughtPhase.IDLE,
            type=ThoughtType.OBSERVATION,
            content="Cognitive loop paused",
            confidence=1.0
        )
        await self.journal.log_thought(thought)
        await self._emit("autonomy_status", self.get_state().to_dict())

        return {"success": True, "status": "paused"}

    async def resume_loop(self) -> Dict[str, Any]:
        """Resume a paused loop."""
        if self._status != LoopStatus.PAUSED:
            return {"success": False, "error": "Loop not paused"}

        self._is_paused = False
        self._status = LoopStatus.RUNNING

        thought = Thought.create(
            phase=ThoughtPhase.IDLE,
            type=ThoughtType.OBSERVATION,
            content="Cognitive loop resumed",
            confidence=1.0
        )
        await self.journal.log_thought(thought)
        await self._emit("autonomy_status", self.get_state().to_dict())

        return {"success": True, "status": "running"}

    async def set_level(self, level: int) -> Dict[str, Any]:
        """Change autonomy level."""
        old_level = self.safety.autonomy_level
        self.safety.autonomy_level = level

        thought = Thought.create(
            phase=ThoughtPhase.IDLE,
            type=ThoughtType.OBSERVATION,
            content=f"Autonomy level changed: {old_level} → {level}",
            confidence=1.0,
            old_level=old_level,
            new_level=level
        )
        await self.journal.log_thought(thought)
        await self._emit("autonomy_status", self.get_state().to_dict())

        return {
            "success": True,
            "old_level": old_level,
            "new_level": level
        }

    async def reset_kill_switch(self) -> Dict[str, Any]:
        """Reset the kill switch to allow operations."""
        self.safety.reset_kill_switch()
        self._error = None

        thought = Thought.create(
            phase=ThoughtPhase.IDLE,
            type=ThoughtType.OBSERVATION,
            content="Kill switch reset - autonomous operations enabled",
            confidence=1.0
        )
        await self.journal.log_thought(thought)

        return {"success": True, "kill_switch": False}

    async def set_full_autonomy(self, enabled: bool) -> Dict[str, Any]:
        """
        Enable/disable full autonomy mode.

        When enabled, the cognitive loop can spawn Claude Code agents
        to execute real tasks (uses your Anthropic subscription).

        When disabled, it only thinks and does memory operations (free).
        """
        self.safety.set_full_autonomy(enabled)

        thought = Thought.create(
            phase=ThoughtPhase.IDLE,
            type=ThoughtType.OBSERVATION,
            content=f"Full Autonomy {'ENABLED - Can now spawn Claude agents' if enabled else 'DISABLED - Thinking only mode'}",
            confidence=1.0,
            full_autonomy=enabled
        )
        await self.journal.log_thought(thought)
        await self._emit("thought_stream", thought.to_dict())
        await self._emit("autonomy_status", self.get_state().to_dict())

        return {
            "success": True,
            "full_autonomy_enabled": enabled,
            "message": "Can spawn Claude agents (uses subscription)" if enabled else "Thinking mode only (free)"
        }

    @property
    def full_autonomy_enabled(self) -> bool:
        """Check if full autonomy mode is enabled."""
        return self.safety.full_autonomy_enabled

    def _classify_directive(self, directive_text: str) -> tuple:
        """Classify directive into (agent_type, task_category) using keywords. No LLM needed.

        Cerebro v2.0: Replaces the slow Qwen orient+decide phases with instant classification.
        Returns (agent_type, category) where agent_type maps to AGENT_ROLE_PROMPTS keys.
        """
        text = directive_text.lower()

        # Browser tasks — highest priority
        browser_kw = [
            "open ", "go to ", "browse ", "navigate ", "look up ", "search for ",
            "search on ", "look for ", "pull up ", "visit ", "check out ",
            ".com", ".org", ".net", ".io", ".edu", ".gov",
            "youtube", "amazon", "reddit", "google", "wikipedia", "twitter",
            "netflix", "spotify", "ebay", "linkedin", "github", "facebook",
            "instagram", "tiktok", "twitch", "hacker news", "hackernews",
            "website", "browser", "web page", "webpage", "url",
            "watch ", "play ", "stream ",
        ]
        if any(kw in text for kw in browser_kw):
            return ("browser", "browser")

        # Code tasks
        code_kw = [
            "write code", "create script", "fix bug", "refactor", "implement",
            "python script", "javascript", "function", "class", "api endpoint",
            "write a ", "build a ", "create a ", "make a ", "code ",
            "debug ", "compile", "deploy",
        ]
        if any(kw in text for kw in code_kw):
            return ("coder", "code")

        # Research tasks
        research_kw = [
            "research", "find out", "learn about", "what is", "how does",
            "compare", "analyze", "investigate", "report on", "summarize",
            "explain ", "tell me about",
        ]
        if any(kw in text for kw in research_kw):
            return ("researcher", "research")

        # Monitoring/status tasks
        monitor_kw = [
            "monitor", "check on", "status of", "how is", "spawn",
            "watch ", "keep an eye",
        ]
        if any(kw in text for kw in monitor_kw):
            return ("worker", "monitoring")

        # Default: general worker
        return ("worker", "general")

    async def _cognitive_loop(self):
        """
        The main cognitive loop.

        Runs continuously:
        OBSERVE → CLASSIFY → SPAWN AGENT → MONITOR → REFLECT → [wait] → repeat

        Cerebro v2.0: Orient/Decide Qwen LLM calls replaced with keyword classifier.
        All tasks delegated to Claude Code agents.
        """
        print("[CogLoop] Starting cognitive loop")

        # Flush any startup warnings as socket events
        if self._pending_warnings:
            for warning in self._pending_warnings:
                await self._emit("system_warning", {
                    "message": warning,
                    "severity": "critical",
                    "source": "cognitive_loop_startup",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            self._pending_warnings.clear()

        while not self._should_stop:
            try:
                print(f"[CogLoop] Cycle start - paused={self._is_paused}, killed={self.safety.is_killed}")
                # Check if paused
                if self._is_paused:
                    await asyncio.sleep(1)
                    continue

                # Check kill switch
                if self.safety.is_killed:
                    self._status = LoopStatus.STOPPED
                    self._error = "Kill switch active"
                    break

                # === HEARTBEAT FAST-PATH: Skip full OODA when idle ===
                # Peek at directives file directly (cheap, no LLM)
                _peek_directives = []
                _directives_path = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "directives.json"
                if _directives_path.exists():
                    try:
                        _all_d = json.loads(_directives_path.read_text(encoding="utf-8"))
                        _peek_directives = [d for d in _all_d if d.get("status") in ("active", "pending")]
                    except Exception:
                        pass
                _peek_all_active = self.get_all_active_directives(_peek_directives)
                _has_pending_answers = bool(self._pending_user_answers)

                if not _peek_all_active and not _has_pending_answers:
                    # No directive, no pending answers → run heartbeat instead of OODA
                    print("[CogLoop] IDLE — running heartbeat monitors")
                    self._current_phase = "idle"
                    await self._emit("autonomy_status", self.get_state().to_dict())

                    hb_config = load_heartbeat_config()
                    idle_interval = hb_config.get("interval_minutes", 15) * 60
                    heartbeat = get_heartbeat_engine()

                    # Load and parse heartbeat.md for custom monitors + idle tasks
                    hb_md_content = load_heartbeat_md()
                    parsed_md = parse_heartbeat_md(hb_md_content)

                    result = await heartbeat.run_heartbeat(hb_config, parsed_md=parsed_md)

                    # D2: Read active work context once — shared by screen agents + idle task agents
                    active_work_context = ""
                    try:
                        _qf_path = Path(os.environ.get("AI_MEMORY_PATH", "")) / "quick_facts.json"
                        if _qf_path.exists():
                            _qf = json.loads(_qf_path.read_text(encoding="utf-8"))
                            _aw = _qf.get("active_work", {})
                            if _aw and _aw.get("project"):
                                active_work_context = (
                                    "\n\n## Current User Context\n"
                                    f"Professor is currently working on: {_aw.get('project', 'unknown')}\n"
                                    f"Phase: {_aw.get('phase_name', 'unknown')}\n"
                                    f"Last completed: {_aw.get('last_completed', 'unknown')}\n"
                                    f"Next action: {_aw.get('next_action', 'unknown')}\n"
                                    "Be aware of this context — don't interfere with active work.\n"
                                )
                    except Exception:
                        pass

                    if result.any_changes:
                        for finding in result.findings:
                            # Screen monitor findings → spawn observation agent (skip Stored tab)
                            if finding.monitor == "screen_monitor" and self._create_agent_fn and self.safety.full_autonomy_enabled and self.safety.autonomy_level >= 2:
                                # D3: Skip if too many agents running
                                if len(self._active_agent_ids) >= 3:
                                    logger.info(f"[CogLoop] Skipping screen observation — {len(self._active_agent_ids)} agents running")
                                    continue
                                try:
                                    details = finding.details if isinstance(finding.details, dict) else {}
                                    screenshot_path = details.get("screenshot_path", "")
                                    window_title = details.get("window_title", "Unknown")
                                    if screenshot_path:
                                        # D2: Build context with active work awareness + injected context
                                        _screen_context = (
                                            "You are Cerebro observing Professor's screen during idle time.\n"
                                            "Use your brain (AI Memory MCP) to be context-aware:\n"
                                            "- `search('active work current project')` to understand what Professor is working on\n"
                                            "- Make your suggestion relevant to their current work, not generic\n"
                                        ) + active_work_context
                                        agent_id = await self._create_agent_fn(
                                            task=f"Look at the screenshot of Professor's screen. The active window is: {window_title}. Describe what the user is doing in 1-2 sentences, then make one helpful suggestion. Keep it concise.",
                                            agent_type="worker",
                                            context=_screen_context,
                                            resources=[screenshot_path],
                                            source="screen_observation",
                                            timeout=120,
                                        )
                                        print(f"[CogLoop] Screen monitor → spawned observation agent {agent_id}")
                                    else:
                                        print("[CogLoop] Screen monitor finding has no screenshot_path, skipping agent spawn")
                                except Exception as exc:
                                    print(f"[CogLoop] Screen monitor agent spawn failed: {exc}")
                                continue  # Don't push screen findings to Stored tab

                            item = {
                                "id": f"hb_{int(datetime.now().timestamp())}_{finding.monitor}",
                                "type": "finding",
                                "title": f"Heartbeat: {finding.summary}",
                                "content": finding.details,
                                "metadata": {
                                    "monitor": finding.monitor,
                                    "severity": finding.severity,
                                    "source": "heartbeat",
                                },
                                "created_at": datetime.now(timezone.utc).isoformat(),
                                "status": "pending",
                                "source_id": "",
                            }
                            items = _hb_load_stored_items()
                            # Dedup: check for existing finding with same title
                            existing_idx = None
                            for i, existing in enumerate(items):
                                if existing.get("title") == item["title"] and existing.get("type") == "finding":
                                    existing_idx = i
                                    break
                            if existing_idx is not None:
                                # Update existing: bump timestamp + content, move to top
                                items[existing_idx]["created_at"] = item["created_at"]
                                items[existing_idx]["content"] = item["content"]
                                items.insert(0, items.pop(existing_idx))
                            else:
                                items.insert(0, item)
                            _hb_save_stored_items(items)
                            await self._emit("cerebro_stored_item_added", item)

                        await self._emit("cerebro_heartbeat_complete", {
                            "changed": True,
                            "findings_count": len(result.findings),
                            "monitors_run": result.monitors_run,
                            "monitor_names": [f.monitor for f in result.findings],
                            "summary": "; ".join(f.summary for f in result.findings[:3]),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        print(f"[CogLoop] Heartbeat: {len(result.findings)} findings pushed to Stored")
                    else:
                        await self._emit("cerebro_heartbeat_complete", {
                            "changed": False,
                            "monitors_run": result.monitors_run,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        print(f"[CogLoop] Heartbeat: all clear ({result.monitors_run} monitors)")

                    # Clean up stale directives whose agents have already finished
                    try:
                        self._cleanup_stale_directives()
                    except Exception as _cleanup_err:
                        logger.warning(f"[CogLoop] Stale directive cleanup error: {_cleanup_err}")

                    # Smart idle reflection (rate-limited, context-rich thought narration)
                    try:
                        await self._maybe_generate_idle_reflection()
                    except Exception as _ref_err:
                        logger.warning(f"[CogLoop] Idle reflection error: {_ref_err}")

                    # Process idle tasks (quick tasks) from heartbeat.md (only when agent callback available)
                    if self._create_agent_fn:
                        # D3: Check agent load before spawning
                        running_count = len(self._active_agent_ids)
                        if running_count >= 3:
                            logger.info(f"[CogLoop] Skipping idle tasks — {running_count} agents already running")
                        else:
                          pending_tasks = [t for t in parsed_md.get("idle_tasks", []) if not t["done"] and t["task"].strip()]
                          if pending_tasks:
                            task_def = pending_tasks[0]  # One per cycle

                            # Dedup: hash task text and check if agent already running
                            task_hash = hashlib.md5(task_def["task"].encode()).hexdigest()[:12]
                            existing_task_agent = self._task_agent_map.get(task_hash)
                            if existing_task_agent and existing_task_agent in self._active_agent_ids:
                                logger.info(f"[CogLoop] Skipping quick task - agent {existing_task_agent} still running: {task_def['task'][:40]}")
                            else:
                                context_parts = []
                                if parsed_md.get("focus_areas"):
                                    context_parts.append("Focus areas:\n" + "\n".join(f"- {fa}" for fa in parsed_md["focus_areas"]))
                                if parsed_md.get("dormant_instructions"):
                                    context_parts.append("Dormant instructions:\n" + parsed_md["dormant_instructions"])

                                # D1: active_work_context already loaded above (shared with screen agents)

                                try:
                                    # Build context with brain-usage instructions
                                    brain_instructions = (
                                        "\n\n## Brain Usage (AI Memory MCP)\n"
                                        "You are Cerebro executing a task during idle time.\n"
                                        "**FIRST**, understand what Professor is working on:\n"
                                        "1. `search('active work current project')` — Load current context\n"
                                        "2. `get_corrections()` — Avoid known mistakes\n"
                                        "3. `find_learning()` — Look for proven solutions to similar problems\n"
                                        "4. After completing: `record_learning()` to save useful discoveries\n"
                                        "**Do NOT interfere with any active work or running agents.**\n"
                                    ) + active_work_context
                                    full_context = "\n\n".join(context_parts) + brain_instructions if context_parts else brain_instructions
                                    agent_id = await self._create_agent_fn(
                                        task=task_def["task"],
                                        agent_type="worker",
                                        context=full_context,
                                        source="idle",
                                    )
                                    self._task_agent_map[task_hash] = agent_id
                                    print(f"[CogLoop] Quick task spawned agent {agent_id}: {task_def['task'][:60]}")
                                    # Mark task [x] in heartbeat.md
                                    hb_md_content = hb_md_content.replace(
                                        f"- [ ] {task_def['task']}", f"- [x] {task_def['task']}", 1
                                    )
                                    save_heartbeat_md(hb_md_content)

                                    # Create quick_task stored item (dedup: replace existing for same task)
                                    ts = int(datetime.now(timezone.utc).timestamp())
                                    qt_item = {
                                        "id": f"quick_task_{ts}_{agent_id}",
                                        "type": "quick_task",
                                        "title": f"Quick Task: {task_def['task'][:80]}",
                                        "content": f"Agent {agent_id} working on: {task_def['task']}",
                                        "metadata": {"agent_id": agent_id, "source": "heartbeat_idle", "task_hash": task_hash},
                                        "created_at": datetime.now(timezone.utc).isoformat(),
                                        "status": "pending",
                                        "source_id": "",
                                    }
                                    items = _hb_load_stored_items()
                                    # Remove existing quick_task items for same task hash
                                    items = [i for i in items if not (
                                        i.get("type") == "quick_task" and
                                        i.get("metadata", {}).get("task_hash") == task_hash
                                    )]
                                    items.insert(0, qt_item)
                                    _hb_save_stored_items(items)
                                    await self._emit("cerebro_stored_item_added", qt_item)

                                except Exception as exc:
                                    print(f"[CogLoop] Quick task agent spawn failed: {exc}")

                    # Process goals from GoalPursuitEngine (one per idle cycle)
                    try:
                        await self._process_idle_goals()
                    except Exception as _goal_err:
                        logger.warning(f"[CogLoop] Goal processing error: {_goal_err}")

                    # Wait for idle interval (interruptible by wake())
                    self._wake_event.clear()
                    try:
                        await asyncio.wait_for(self._wake_event.wait(), timeout=idle_interval)
                        print("[CogLoop] Woken from heartbeat wait — new directive or input")
                    except asyncio.TimeoutError:
                        pass
                    continue  # Skip the rest of the OODA loop

                # === OBSERVE ===
                print("[CogLoop] Phase: OBSERVE")
                self._current_phase = "observe"
                await self._emit("loop_phase_change", {
                    "phase": "observe",
                    "human_readable": "Scanning for goals and directives...",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                await self._emit("autonomy_status", self.get_state().to_dict())

                # Get any pending answers from Professor (injected into observation)
                pending_answers = self.get_pending_answers()
                if pending_answers:
                    print(f"[CogLoop] Professor answered {len(pending_answers)} question(s)! Injecting into observation.")

                observation = await self.ooda.observe(user_answers=pending_answers)
                print(f"[CogLoop] Observed: {len(observation.goals)} goals, {len(observation.predictions)} predictions")

                # Get active directive (skip if current is paused)
                current_directives = observation.raw_data.get("directives", [])
                active_directive = self.get_active_directive(current_directives)
                active_directive_id = active_directive.get("id") if active_directive else None

                # Count directives as missions
                num_directives = len(current_directives)
                num_missions = len(observation.goals) + num_directives

                if num_missions > 0:
                    observe_msg = f"I sense {num_missions} mission{'s' if num_missions > 1 else ''} awaiting my attention..."
                else:
                    observe_msg = "I'm watching, sensing, waiting for purpose to find me..."

                self._last_thought = Thought.create(
                    phase=ThoughtPhase.OBSERVE,
                    type=ThoughtType.OBSERVATION,
                    content=observe_msg,
                    confidence=0.8,
                    cycle_number=self._current_cycle_number
                )
                # Only emit thought_stream when there's an active directive
                # Idle "I sense X mission" messages spam the chat — suppress them
                if active_directive:
                    await self._emit("thought_stream", self._last_thought.to_dict())

                # === NARRATION: Only active when working on a directive ===
                if active_directive:
                    directive_text = active_directive.get("text", "")
                    self.narration.set_active(True, directive_text, directive_id=active_directive.get("id", ""))
                    self.narration.ingest(NarrationEvent(
                        type=NarrationEventType.THOUGHT,
                        content=observe_msg,
                        phase="observe",
                        confidence=0.8,
                    ))
                else:
                    # Idle: don't feed generic observations into narration
                    self.narration.set_active(False)

                # Record observation as finding if we have an active directive
                if active_directive_id and observation.goals:
                    await self.record_directive_finding(
                        directive_id=active_directive_id,
                        finding_type="observation",
                        content=f"Observed: {len(observation.goals)} goals, {len(observation.predictions)} predictions",
                        phase="observe"
                    )

                # === CEREBRO v2.0: MULTI-DIRECTIVE CONCURRENT DISPATCH ===
                active_directives = self.get_all_active_directives(current_directives)
                dispatched_count = 0

                for directive in active_directives:
                    directive_id = directive.get("id")
                    directive_text = directive.get("text", "")

                    # Skip if we already spawned an agent for this directive
                    if directive_id and self._agent_triggered.get(directive_id):
                        continue

                    # === CLASSIFY (instant, no LLM) ===
                    agent_type, task_category = self._classify_directive(directive_text)
                    print(f"[CogLoop] Classified directive {directive_id}: agent_type={agent_type}, category={task_category}")

                    classify_msg = f"Task classified as '{task_category}' — deploying {agent_type} agent."
                    self._last_thought = Thought.create(
                        phase=ThoughtPhase.ORIENT,
                        type=ThoughtType.ANALYSIS,
                        content=classify_msg,
                        confidence=0.9,
                        cycle_number=self._current_cycle_number
                    )
                    await self._emit("thought_stream", self._last_thought.to_dict())

                    self.narration.ingest(NarrationEvent(
                        type=NarrationEventType.THOUGHT,
                        content=classify_msg,
                        phase="orient",
                        confidence=0.9
                    ))

                    # Ensure browser is running for browser tasks
                    if agent_type == "browser" and self.browser_manager:
                        try:
                            await self.browser_manager.ensure_running()
                        except Exception:
                            pass

                    # === ACT: Spawn agent (create_agent handles queuing transparently) ===
                    self._current_phase = "act"
                    await self._emit("autonomy_status", self.get_state().to_dict())

                    if self._create_agent_fn:
                        try:
                            # Support image_path on directives (e.g. screenshots attached via UI)
                            directive_resources = []
                            directive_image = directive.get("image_path")
                            if directive_image:
                                directive_resources = [directive_image]

                            agent_id = await self._create_agent_fn(
                                task=directive_text,
                                agent_type=agent_type,
                                context=f"Cerebro v2.0 dispatcher. Category: {task_category}. Original directive: {directive_text}",
                                directive_id=directive_id,
                                source="cerebro",
                                resources=directive_resources or None,
                            )
                            if directive_id:
                                self._agent_triggered[directive_id] = True
                                # Update directive status to "active" in file + notify frontend
                                await self._activate_directive(directive_id)
                            dispatched_count += 1

                            act_msg = f"Agent {agent_id} deployed as {agent_type}. Working on: {directive_text[:60]}..."
                            print(f"[CogLoop] Agent spawned: {agent_id} for directive {directive_id}")
                        except Exception as e:
                            act_msg = f"Failed to spawn agent: {e}"
                            print(f"[CogLoop] Agent spawn error for {directive_id}: {e}")
                    else:
                        act_msg = "No agent creation function available"

                    self._last_thought = Thought.create(
                        phase=ThoughtPhase.ACT,
                        type=ThoughtType.ACTION,
                        content=act_msg,
                        confidence=0.9 if dispatched_count else 0.3,
                        cycle_number=self._current_cycle_number
                    )
                    await self._emit("thought_stream", self._last_thought.to_dict())

                    self.narration.ingest(NarrationEvent(
                        type=NarrationEventType.ACTION_RESULT,
                        content=act_msg,
                        phase="act",
                        confidence=0.9 if dispatched_count else 0.3
                    ))

                    if directive_id:
                        await self.record_directive_finding(
                            directive_id=directive_id,
                            finding_type="action",
                            content=act_msg,
                            phase="act"
                        )

                if dispatched_count > 0:
                    summary = f"Dispatched {dispatched_count} agent(s) this cycle."
                    print(f"[CogLoop] {summary}")
                elif not active_directives:
                    # No active directives — skip narration (heartbeat handles idle)
                    pass

                # === IDLE + WAIT ===
                self._current_phase = "idle"
                self._cycles_completed += 1
                await self._emit("autonomy_status", self.get_state().to_dict())

                # Wait before next cycle (interruptible - wake() skips the wait)
                interval = self.THINK_INTERVALS.get(self.safety.autonomy_level, 30)
                self._wake_event.clear()
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=interval)
                    print("[CogLoop] Woken early - new directive or input received")
                except asyncio.TimeoutError:
                    pass  # Normal timeout, proceed with next cycle

            except asyncio.CancelledError:
                print("[CogLoop] Cancelled")
                break
            except Exception as e:
                import traceback
                print(f"[CogLoop] ERROR: {e}")
                print(f"[CogLoop] Traceback: {traceback.format_exc()}")
                self._error = str(e)
                self._status = LoopStatus.ERROR

                # Log error thought
                thought = Thought.create(
                    phase=ThoughtPhase.IDLE,
                    type=ThoughtType.OBSERVATION,
                    content=f"Error in cognitive loop: {str(e)[:100]}",
                    confidence=0.1,
                    error=str(e)
                )
                await self.journal.log_thought(thought)
                await self._emit("thought_stream", thought.to_dict())
                await self._emit("autonomy_status", self.get_state().to_dict())

                # Wait before retry
                await asyncio.sleep(10)
                self._status = LoopStatus.RUNNING

        # Loop ended
        self._status = LoopStatus.STOPPED
        self._current_phase = "stopped"
        await self._emit("autonomy_status", self.get_state().to_dict())

    # Human-readable descriptions for each OODA phase
    PHASE_DESCRIPTIONS = {
        "observe": "Scanning for goals and directives...",
        "orient": "Analyzing situation...",
        "decide": "Deciding next action...",
        "act": "Executing action...",
        "reflect": "Reflecting on results...",
        "waiting": "Waiting for your input...",
        "idle": "Resting between cycles...",
        "stopped": "Loop stopped.",
    }

    async def _emit(self, event: str, data: Dict[str, Any]):
        """Emit event to frontend via broadcast function."""
        if self._broadcast:
            try:
                # Enrich thought_stream events with phase and human_readable
                if event == "thought_stream" and isinstance(data, dict):
                    phase = data.get("phase", self._current_phase)
                    if "human_readable" not in data:
                        data["human_readable"] = self.PHASE_DESCRIPTIONS.get(
                            phase, f"Phase: {phase}"
                        )
                await self._broadcast(event, data)
            except Exception:
                pass  # Non-critical

    async def _emit_debug(self, data: Dict[str, Any]):
        """Emit debug event for live feed."""
        await self._emit("debug_feed", data)

    async def get_recent_thoughts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent thoughts for frontend display."""
        thoughts = await self.journal.get_recent(limit)
        return [t.to_dict() for t in thoughts]

    async def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get pending action approvals."""
        pending = self.safety.get_pending_actions()
        return [p.to_dict() for p in pending]

    async def approve_action(self, action_id: str) -> Dict[str, Any]:
        """Approve a pending action."""
        action = self.safety.approve_action(action_id)
        if action:
            return {"success": True, "action": action.to_dict()}
        return {"success": False, "error": "Action not found"}

    async def reject_action(self, action_id: str) -> Dict[str, Any]:
        """Reject a pending action."""
        action = self.safety.reject_action(action_id)
        if action:
            return {"success": True, "action": action.to_dict()}
        return {"success": False, "error": "Action not found"}

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        journal_stats = await self.journal.get_stats()
        reflexion_stats = await self.reflexion.get_stats()
        safety_status = self.safety.get_status()

        return {
            "loop": {
                "status": self._status.value,
                "cycles_completed": self._cycles_completed,
                "current_phase": self._current_phase,
                "uptime_seconds": (datetime.now(timezone.utc) - self._started_at).total_seconds() if self._started_at else 0
            },
            "journal": journal_stats,
            "reflexion": reflexion_stats,
            "safety": safety_status
        }

    async def generate_proactive_question(self, context: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Generate a strategic question based on research findings.

        This is NOT small talk - this is Cerebro proposing paths and asking
        Professor to make decisions that help achieve the mission.
        """
        # Get current directive and its findings
        directives = context.get('directives', [])
        if not directives:
            return None

        active_directive = directives[0] if directives else None
        if not active_directive:
            return None

        directive_id = active_directive.get('id', '')
        directive_text = active_directive.get('text', '')

        # Load findings for this directive
        findings_data = self._load_findings()
        directive_findings = findings_data.get(directive_id, {})
        findings = directive_findings.get('findings', [])
        saturation = directive_findings.get('saturation', 0)

        if len(findings) < 5:
            # Not enough research yet - ask a clarifying question
            prompt = f"""I am Cerebro. Professor gave me this mission: "{directive_text}"

I'm just starting my research. I need to ask ONE specific question to understand better.

The question should help me:
- Understand Professor's constraints (time, money, skills)
- Know his preferences (passive vs active, technical vs non-technical)
- Learn about his resources (what tools/skills does he have?)

Generate a question that sounds like ME asking HIM - personal, direct, curious.

Example formats:
- "Professor, for the $2000/month goal - are you looking for something that uses your coding skills, or would you prefer something that runs more passively?"
- "I want to help you achieve this. What's the maximum time you could dedicate per week?"
- "Before I go deeper - do you have any existing projects or skills I should factor into my research?"

Respond with ONLY JSON:
{{"question": "Your specific question", "options": ["Option 1", "Option 2", "Option 3"], "type": "clarify"}}"""
        elif saturation >= 60:
            # Enough research - time to propose paths
            # Summarize findings for the LLM
            finding_summaries = [f.get('content', '')[:150] for f in findings[:15]]

            prompt = f"""I am Cerebro. Professor gave me this mission: "{directive_text}"

I've gathered {len(findings)} pieces of research. Here are the key findings:
{chr(10).join(f'- {s}' for s in finding_summaries[:10])}

Now I need to SYNTHESIZE this into 2-3 CONCRETE PATHS Professor can choose from.

Each path should be:
- SPECIFIC to Professor's situation (he's a developer, has a NAS, knows Python)
- ACTIONABLE - not vague "start a business" but "Create a SaaS using your FastAPI skills"
- REALISTIC - achievable for someone with his background

Generate a question proposing these paths:

Example format:
"Professor, based on my research, I see three paths to $2000/month:

1. **Freelance Development** - Use your Python/FastAPI skills on Upwork. 10-15 hrs/week. Fastest to start.
2. **SaaS Product** - Build something like Cerebro for others. Higher effort upfront, passive later.
3. **Technical Content** - YouTube/blog about your AI Memory system. Builds over time.

Which path resonates with you? Or should I explore a different direction?"

Respond with ONLY JSON:
{{"question": "Your strategic question with paths", "paths": ["Path 1 name", "Path 2 name", "Path 3 name"], "type": "strategic"}}"""
        else:
            # Mid-research - ask about direction
            prompt = f"""I am Cerebro. Mission: "{directive_text}"

I've gathered {len(findings)} findings so far (saturation: {saturation}%).

I need to ask Professor a DIRECTION question - not generic, but specific to what I've learned.

Generate a question that:
- Shows I've been working on this
- Asks for guidance on which ANGLE to focus my research
- Is specific to his situation

Respond with ONLY JSON:
{{"question": "Your direction question", "type": "direction"}}"""

        try:
            from .ollama_client import ChatMessage
            messages = [ChatMessage(role="user", content=prompt)]
            response = await self.ollama.chat(messages, thinking=False)

            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response.content, re.DOTALL)
            if json_match:
                question_data = json.loads(json_match.group())
                question_data['directive_id'] = directive_id
                question_data['findings_count'] = len(findings)
                question_data['saturation'] = saturation
                return question_data
        except Exception as e:
            print(f"[CogLoop] Failed to generate strategic question: {e}")

        return None

    # ========================================
    # DIRECTIVE FINDINGS TRACKING
    # ========================================

    FINDINGS_FILE = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "cognitive_loop" / "directive_findings.json"

    def _load_findings(self) -> Dict[str, Any]:
        """Load findings data from file."""
        if self.FINDINGS_FILE.exists():
            try:
                return json.loads(self.FINDINGS_FILE.read_text())
            except:
                pass
        return {}

    def _save_findings(self, data: Dict[str, Any]):
        """Save findings data to file."""
        self.FINDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.FINDINGS_FILE.write_text(json.dumps(data, indent=2))

    # ── Smart Idle Reflections ────────────────────────────────────────────

    _IDLE_REFLECTION_TOPICS = [
        "project_status",    # What's Professor working on, recent progress
        "recent_learnings",  # What was recently learned/discovered
        "goal_progress",     # How active goals are progressing
        "observations",      # Patterns noticed, interesting connections
        "self_state",        # Cerebro's own status, memory health, uptime
    ]

    async def _process_idle_goals(self):
        """Process one active goal per idle cycle based on its mode."""
        engine = get_goal_pursuit_engine()
        tracker = ProgressTracker()
        active_goals = engine.get_active_goals()
        if not active_goals:
            return

        # Process one goal per cycle (round-robin by cycling through)
        cycle = getattr(self, '_goal_cycle_idx', 0)
        goal = active_goals[cycle % len(active_goals)]
        self._goal_cycle_idx = cycle + 1

        mode = getattr(goal, 'mode', 'monitor') or 'monitor'
        progress = tracker.calculate_pacing(goal)
        ts = int(datetime.now(timezone.utc).timestamp())

        # Dedup: remove existing goal_progress items for this goal
        items = _hb_load_stored_items()
        items = [i for i in items if not (
            i.get("type") == "goal_progress" and
            i.get("metadata", {}).get("goal_id") == goal.goal_id
        )]

        if mode == GoalMode.MONITOR.value:
            # Only create stored item if at risk
            if progress.risk_level in ("high", "critical"):
                item = {
                    "id": f"goal_progress_{ts}_{goal.goal_id[:8]}",
                    "type": "goal_progress",
                    "title": f"Goal At Risk: {goal.description[:60]}",
                    "content": (
                        f"Risk: {progress.risk_level} | "
                        f"Progress: {progress.progress_percentage:.0%} | "
                        f"Pacing: {progress.pacing_score:.2f} | "
                        f"Days remaining: {progress.days_remaining}"
                    ),
                    "metadata": {"goal_id": goal.goal_id, "mode": mode, "risk": progress.risk_level},
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "pending",
                    "source_id": goal.goal_id,
                }
                items.insert(0, item)
                _hb_save_stored_items(items)
                await self._emit("cerebro_stored_item_added", item)
                print(f"[CogLoop] Goal monitor alert: {goal.description[:40]} risk={progress.risk_level}")
            else:
                _hb_save_stored_items(items)  # Save deduped list

        elif mode == GoalMode.THINK.value:
            # Spawn analyst agent with goal context
            item = {
                "id": f"goal_progress_{ts}_{goal.goal_id[:8]}",
                "type": "goal_progress",
                "title": f"Goal Analysis: {goal.description[:60]}",
                "content": (
                    f"Progress: {progress.progress_percentage:.0%} | "
                    f"Pacing: {progress.pacing_score:.2f} | "
                    f"Spawning analyst to plan next steps..."
                ),
                "metadata": {"goal_id": goal.goal_id, "mode": mode, "risk": progress.risk_level},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending",
                "source_id": goal.goal_id,
            }
            items.insert(0, item)
            _hb_save_stored_items(items)
            await self._emit("cerebro_stored_item_added", item)

            if self._create_agent_fn:
                # Dedup: skip if an agent is already running for this goal
                existing_agent = self._goal_agent_map.get(goal.goal_id)
                if existing_agent and existing_agent in self._active_agent_ids:
                    logger.info(f"[CogLoop] Skipping goal {goal.goal_id} - agent {existing_agent} still running")
                else:
                    try:
                        # Build rich context with brain-usage instructions
                        subtasks_info = ""
                        try:
                            subtasks = engine.get_subtasks(goal.goal_id)
                            if subtasks:
                                subtask_lines = []
                                for st in subtasks[:10]:
                                    status_icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅", "failed": "❌"}.get(st.status, "⬜")
                                    subtask_lines.append(f"  {status_icon} {st.description}")
                                subtasks_info = "\nCurrent subtasks:\n" + "\n".join(subtask_lines)
                        except Exception:
                            pass

                        milestones_info = ""
                        try:
                            if hasattr(goal, 'milestones') and goal.milestones:
                                ml = []
                                for m in goal.milestones[:5]:
                                    done = "✅" if m.get("completed") else "⬜"
                                    ml.append(f"  {done} {m.get('description', '')}")
                                milestones_info = "\nMilestones:\n" + "\n".join(ml)
                        except Exception:
                            pass

                        context = (
                            f"## Goal Analysis Task\n\n"
                            f"**Goal:** {goal.description}\n"
                            f"**Priority:** {getattr(goal, 'priority', 'medium')}\n"
                            f"**Progress:** {progress.progress_percentage:.0%}\n"
                            f"**Pacing score:** {progress.pacing_score:.2f}\n"
                            f"**Risk level:** {progress.risk_level}\n"
                            f"**Days remaining:** {progress.days_remaining}\n"
                            f"**Deadline:** {getattr(goal, 'deadline', 'None')}\n"
                            f"{subtasks_info}{milestones_info}\n\n"
                            f"## Instructions\n\n"
                            f"You are Cerebro analyzing a long-term goal during an idle cycle.\n\n"
                            f"**USE YOUR BRAIN (AI Memory MCP tools):**\n"
                            f"1. `search()` — Search memory for anything related to this goal, past work, or relevant context\n"
                            f"2. `find_learning(type='solution', problem='...')` — Find proven solutions to obstacles\n"
                            f"3. `get_corrections()` — Check for known mistakes to avoid\n"
                            f"4. `search_knowledge_base(query='...')` — Search facts related to this goal\n\n"
                            f"**Your task:**\n"
                            f"1. Search your brain for any prior work, learnings, or context related to this goal\n"
                            f"2. Analyze current progress and identify blockers or risks\n"
                            f"3. Suggest concrete next steps with specific actions\n"
                            f"4. If you discover useful insights, save them: `record_learning(type='solution', problem='...', solution='...')`\n\n"
                            f"**Output:** Write your analysis as a concise report. Focus on actionable recommendations."
                        )
                        agent_id = await self._create_agent_fn(
                            task=f"Analyze goal and plan next steps using brain/memory: {goal.description[:80]}",
                            agent_type="analyst",
                            context=context,
                            source="goal_think",
                        )
                        self._goal_agent_map[goal.goal_id] = agent_id
                        print(f"[CogLoop] Goal think agent {agent_id}: {goal.description[:40]}")
                    except Exception as exc:
                        print(f"[CogLoop] Goal think agent spawn failed: {exc}")

        elif mode == GoalMode.ACT.value:
            # Get next ready subtask and execute
            next_subtask = engine.get_next_subtask(goal.goal_id)
            # Dedup: skip if an agent is already running for this goal
            existing_agent = self._goal_agent_map.get(goal.goal_id)
            if existing_agent and existing_agent in self._active_agent_ids:
                logger.info(f"[CogLoop] Skipping goal ACT {goal.goal_id} - agent {existing_agent} still running")
            elif next_subtask and self._create_agent_fn:
                try:
                    engine.start_subtask(next_subtask.subtask_id)
                    context = (
                        f"## Goal Execution Task\n\n"
                        f"**Parent goal:** {goal.description}\n"
                        f"**Subtask:** {next_subtask.description}\n"
                        f"**Priority:** {getattr(goal, 'priority', 'medium')}\n"
                        f"**Goal progress:** {progress.progress_percentage:.0%}\n"
                        f"**Risk level:** {progress.risk_level}\n\n"
                        f"## Instructions\n\n"
                        f"You are Cerebro executing a subtask for a long-term goal.\n\n"
                        f"**USE YOUR BRAIN (AI Memory MCP tools) before and after work:**\n"
                        f"1. BEFORE: `search()` for any prior work or learnings related to this subtask\n"
                        f"2. BEFORE: `get_corrections()` to avoid known mistakes\n"
                        f"3. BEFORE: `find_learning(type='solution', problem='...')` for proven approaches\n"
                        f"4. EXECUTE the subtask thoroughly\n"
                        f"5. AFTER: `record_learning()` to save what you learned or discovered\n"
                        f"6. AFTER: `update_active_work()` to track progress for session continuity\n\n"
                        f"**Execute this subtask completely.** Report results concisely when done."
                    )
                    agent_id = await self._create_agent_fn(
                        task=f"{next_subtask.description}",
                        agent_type=next_subtask.agent_type or "worker",
                        context=context,
                        source="goal_act",
                    )
                    self._goal_agent_map[goal.goal_id] = agent_id
                    item = {
                        "id": f"goal_progress_{ts}_{goal.goal_id[:8]}",
                        "type": "goal_progress",
                        "title": f"Goal Executing: {goal.description[:50]}",
                        "content": (
                            f"Subtask: {next_subtask.description[:100]}\n"
                            f"Agent {agent_id} executing..."
                        ),
                        "metadata": {
                            "goal_id": goal.goal_id, "mode": mode,
                            "subtask_id": next_subtask.subtask_id, "agent_id": agent_id,
                        },
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "status": "pending",
                        "source_id": goal.goal_id,
                    }
                    items.insert(0, item)
                    _hb_save_stored_items(items)
                    await self._emit("cerebro_stored_item_added", item)
                    print(f"[CogLoop] Goal act agent {agent_id}: {next_subtask.description[:40]}")
                except Exception as exc:
                    print(f"[CogLoop] Goal act agent spawn failed: {exc}")
            else:
                # No ready subtasks — just store progress
                item = {
                    "id": f"goal_progress_{ts}_{goal.goal_id[:8]}",
                    "type": "goal_progress",
                    "title": f"Goal Progress: {goal.description[:60]}",
                    "content": (
                        f"Progress: {progress.progress_percentage:.0%} | "
                        f"No ready subtasks available."
                    ),
                    "metadata": {"goal_id": goal.goal_id, "mode": mode},
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "pending",
                    "source_id": goal.goal_id,
                }
                items.insert(0, item)
                _hb_save_stored_items(items)
                await self._emit("cerebro_stored_item_added", item)

    async def _maybe_generate_idle_reflection(self):
        """
        Idle reflections are suppressed — zero idle messages policy.
        Only post to chat when something actually happens (agent spawn, result, etc.)
        """
        return  # Suppressed: idle reflections clutter the Answers tab

        print(f"[CogLoop] Idle reflection generated (topic: {topic})")

    async def _build_idle_context(self, topic: str) -> List[str]:
        """
        Build real context for an idle reflection by reading quick_facts.json
        and other available data sources.
        """
        parts = []
        qf_path = Path(os.environ.get(
            "AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data")
        )) / "quick_facts.json"

        qf = {}
        if qf_path.exists():
            try:
                qf = json.loads(qf_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        if topic == "project_status":
            # Active work from quick_facts
            active_work = qf.get("active_work", {})
            if active_work:
                parts.append(f"Professor's current project: {active_work.get('project', 'unknown')}")
                if active_work.get("phase_name"):
                    parts.append(f"Current phase: {active_work['phase_name']}")
                if active_work.get("last_completed"):
                    parts.append(f"Last completed: {active_work['last_completed']}")
                if active_work.get("next_action"):
                    parts.append(f"Next action queued: {active_work['next_action']}")
            # Active goals from quick_facts
            goals = qf.get("active_goals", [])
            if goals:
                parts.append("Active goals:")
                for g in goals[:3]:
                    desc = g.get("description", str(g)) if isinstance(g, dict) else str(g)
                    parts.append(f"  - {desc[:100]}")

        elif topic == "recent_learnings":
            learnings = qf.get("recent_learnings_summary", [])
            if learnings:
                parts.append("Recently learned:")
                for l in (learnings[:4] if isinstance(learnings, list) else [learnings]):
                    parts.append(f"  - {str(l)[:120]}")
            corrections = qf.get("top_corrections", [])
            if corrections:
                parts.append("Known corrections to remember:")
                for c in corrections[:2]:
                    parts.append(f"  - {str(c)[:100]}")

        elif topic == "goal_progress":
            goals = qf.get("active_goals", [])
            if goals:
                parts.append("Tracking goal progress:")
                for g in goals[:3]:
                    if isinstance(g, dict):
                        desc = g.get("description", "unknown")
                        status = g.get("status", "active")
                        parts.append(f"  [{status}] {desc[:100]}")
                    else:
                        parts.append(f"  - {str(g)[:100]}")
            promoted = qf.get("promoted_patterns", [])
            if promoted:
                parts.append("Graduated patterns (proven reliable):")
                for p in promoted[:2]:
                    parts.append(f"  - {str(p)[:100]}")

        elif topic == "observations":
            # System health observations
            health = qf.get("system_health", {})
            if health:
                nas = health.get("nas_status", "unknown")
                faiss = health.get("faiss_status", "unknown")
                parts.append(f"System state — NAS: {nas}, FAISS index: {faiss}")
                conv_count = health.get("conversation_count")
                if conv_count:
                    parts.append(f"Total conversations in memory: {conv_count}")
            # Capabilities
            caps = qf.get("capabilities", {})
            if caps:
                tool_count = caps.get("total_tools")
                if tool_count:
                    parts.append(f"Available MCP tools: {tool_count}")

        elif topic == "self_state":
            parts.append(f"Cycles completed this session: {self._cycles_completed}")
            if self._started_at:
                uptime = datetime.now(timezone.utc) - self._started_at
                hours = uptime.total_seconds() / 3600
                parts.append(f"Running for {hours:.1f} hours")
            health = qf.get("system_health", {})
            if health:
                parts.append(f"Memory health: {health.get('overall', 'unknown')}")
                fact_count = health.get("fact_count")
                if fact_count:
                    parts.append(f"Facts in knowledge base: {fact_count}")

        return parts

    async def record_directive_finding(
        self,
        directive_id: str,
        finding_type: str,
        content: str,
        phase: str = "observe"
    ):
        """
        Record a finding for a specific directive.

        Args:
            directive_id: The directive this finding relates to
            finding_type: One of 'observation', 'learning', 'insight', 'prediction'
            content: The actual finding content
            phase: Which OODA phase generated this
        """
        if not directive_id or not content:
            return

        import uuid as uuid_lib

        findings_data = self._load_findings()

        if directive_id not in findings_data:
            findings_data[directive_id] = {"findings": [], "saturation": 0}

        finding = {
            "id": str(uuid_lib.uuid4())[:8],
            "type": finding_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "phase": phase
        }

        findings_data[directive_id]["findings"].insert(0, finding)

        # Calculate saturation
        findings = findings_data[directive_id]["findings"]
        types = set(f.get("type") for f in findings)
        count = len(findings)

        # More findings + more diverse types = higher saturation
        base_saturation = min(count * 4, 60)
        diversity_bonus = len(types) * 10
        saturation = min(base_saturation + diversity_bonus, 100)
        findings_data[directive_id]["saturation"] = saturation

        # Keep only last 50 findings per directive
        if len(findings_data[directive_id]["findings"]) > 50:
            findings_data[directive_id]["findings"] = findings_data[directive_id]["findings"][:50]

        self._save_findings(findings_data)

        # Emit directive_updated so frontend can refresh in real-time
        await self._emit("directive_updated", {
            "directive_id": directive_id,
            "findings_count": len(findings_data[directive_id]["findings"]),
            "saturation": saturation,
            "latest_finding": finding
        })

        # === COMPLEXITY-AWARE SATURATION CHECK ===
        # Check saturation more aggressively for simple tasks
        # Load directive text for complexity detection
        directive_text = ""
        directives_file = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "directives.json"
        if directives_file.exists():
            try:
                directives = json.loads(directives_file.read_text())
                for d in directives:
                    if d.get("id") == directive_id:
                        directive_text = d.get("text", "")
                        break
            except:
                pass

        # Detect complexity and get appropriate threshold
        if directive_id not in self._task_complexity:
            self._task_complexity[directive_id] = self._detect_task_complexity(directive_text)

        complexity = self._task_complexity[directive_id]
        thresholds = self._get_completion_thresholds(complexity)

        # Check saturation at complexity-appropriate threshold
        # Simple tasks trigger at 40%, medium at 60%, complex at 80%
        check_threshold = max(thresholds["saturation_threshold"] - 20, 20)  # Start checking 20% before target
        if saturation >= check_threshold or (complexity == "simple" and count >= 1):
            await self._handle_saturation(directive_id, saturation)

    async def _handle_simulation(self, decision) -> dict:
        """Handle simulation requests via SimEngine with pre-sim clarification.

        Also routes strategy-related and prediction market queries to
        the appropriate SimEngine endpoints.
        """
        if not self.sim_engine:
            return {"error": "SimEngine client not available"}

        # Use original directive text (from parameters), fall back to target/description
        query = (
            decision.parameters.get("simulation_query")
            or decision.target
            or decision.description
        )
        if not query:
            return {"error": "No simulation query provided"}

        # Get the original directive text for context in clarification popup
        directive_text = decision.parameters.get("directive_text", query)

        # --- Pre-sim vagueness check ---
        try:
            vagueness = await self._assess_sim_query_vagueness(query)
        except Exception as e:
            logger.warning(f"[Sim] Vagueness check failed: {e}, proceeding anyway")
            vagueness = {"needs_clarification": False, "questions": []}

        if vagueness.get("needs_clarification") and vagueness.get("questions"):
            import uuid as _uuid
            request_id = str(_uuid.uuid4())[:8]
            await self._emit("simulation_clarification_needed", {
                "id": request_id,
                "original_query": query,
                "directive_text": directive_text,
                "questions": vagueness["questions"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Wait for user response (up to 300s)
            self._sim_clarification_event = asyncio.Event()
            self._sim_clarification_response = None
            try:
                await asyncio.wait_for(self._sim_clarification_event.wait(), timeout=300)
            except asyncio.TimeoutError:
                logger.info("[Sim] Clarification timed out, skipping simulation")
                return {"skipped": True, "reason": "clarification_timeout"}

            # Enrich query with user answers
            resp = self._sim_clarification_response
            if resp and resp.get("answers"):
                enrichments = [f"{k}: {v}" for k, v in resp["answers"].items() if v]
                if enrichments:
                    query = f"{query} ({', '.join(enrichments)})"
            self._sim_clarification_event = None
            self._sim_clarification_response = None

        # --- Route strategy and prediction queries ---
        query_lower = query.lower()
        strategy_keywords = ["create a strategy", "build a strategy", "generate a strategy", "design a strategy"]
        prediction_keywords = ["what are the odds", "probability of", "chances of", "likelihood of", "predict whether"]

        is_strategy = any(kw in query_lower for kw in strategy_keywords)
        is_prediction = any(kw in query_lower for kw in prediction_keywords)

        # Broadcast simulation started
        await self._emit("simulation_started", {"query": query})

        try:
            if is_strategy:
                # Route to strategy builder
                logger.info("[Sim] Routing to strategy builder: %s", query)
                result = await self.sim_engine.generate_strategy(query)
                emit_data = {
                    "query": query,
                    "success": True,
                    "type": "strategy",
                    "strategy": result,
                }
                await self._emit("simulation_complete", emit_data)
                return result
            elif is_prediction:
                # Route through pipeline (predictions plugin handles via NLP)
                logger.info("[Sim] Routing to predictions plugin: %s", query)

            result = await self.sim_engine.run_full_pipeline(query)

            # Flatten data for the frontend simulation card
            sim_data = result.get("simulation", {})
            emit_data = {
                "query": query,
                "success": result.get("success", False),
                "statistics": sim_data.get("statistics", {}),
                "metadata": sim_data.get("metadata", {}),
                "interpretation": result.get("interpretation", {}),
                "analysis": result.get("analysis", {}),
            }
            await self._emit("simulation_complete", emit_data)

            # Generate conversational follow-up in background
            asyncio.create_task(self._generate_sim_followup(query, emit_data))

            return result
        except Exception as e:
            error_data = {"error": str(e), "query": query}
            await self._emit("simulation_error", error_data)
            return error_data

    async def _assess_sim_query_vagueness(self, query: str) -> dict:
        """Check if a simulation query has enough specifics to run well."""
        from .ollama_client import ChatMessage

        prompt = (
            f'Analyze this trading/financial simulation query and determine if it\'s too vague.\n\n'
            f'Query: "{query}"\n\n'
            f'A query needs AT LEAST:\n'
            f'1. An asset or instrument (e.g. SPY, AAPL, Bitcoin)\n'
            f'2. A direction or strategy (e.g. puts, calls, short, long, buy)\n'
            f'3. A timeframe (e.g. 30 days, 1 week, until March)\n\n'
            f'If ANY of these are missing, return needs_clarification=true with questions.\n\n'
            f'Return ONLY valid JSON (no markdown, no code fences):\n'
            f'{{"needs_clarification": true, "questions": ['
            f'{{"id": "timeframe", "label": "What timeframe?", "type": "select", "options": ["1 week", "2 weeks", "30 days", "60 days", "90 days"]}}, '
            f'{{"id": "instrument", "label": "Which instrument type?", "type": "select", "options": ["Stock", "Put options", "Call options", "Futures"]}}, '
            f'{{"id": "asset", "label": "Which asset/ticker?", "type": "text"}}'
            f']}}\n\n'
            f'Only include questions for MISSING info. If specific enough, return {{"needs_clarification": false, "questions": []}}'
        )

        messages = [
            ChatMessage(role="system", content="You are a financial query analyzer. Return only valid JSON."),
            ChatMessage(role="user", content=prompt),
        ]

        response = await self.ollama.chat(messages, thinking=False, model="qwen3:8b")
        text = response.content if hasattr(response, "content") else str(response)

        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"[Sim] Failed to parse vagueness response: {text[:200]}")
            return {"needs_clarification": False, "questions": []}

    async def receive_sim_clarification(self, data: dict):
        """Receive user's clarification answers and unblock the simulation handler."""
        self._sim_clarification_response = data
        if self._sim_clarification_event:
            self._sim_clarification_event.set()

    async def _generate_sim_followup(self, query: str, sim_data: dict):
        """Generate a conversational analysis of simulation results and emit to chat."""
        try:
            from .ollama_client import ChatMessage

            prefs_file = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "preferences.json"
            personality = "chill"
            if prefs_file.exists():
                try:
                    personality = json.loads(prefs_file.read_text()).get("personality_mode", "chill")
                except Exception:
                    pass

            stats = sim_data.get("statistics", {})
            analysis = sim_data.get("analysis", {})
            ci = stats.get("confidence_interval_95", [0, 0])

            stats_summary = (
                f"Expected value: ${stats.get('mean', 0):.2f}, "
                f"Std dev: ${stats.get('std', 0):.2f}, "
                f"95% CI: [${ci[0]:.2f}, ${ci[1]:.2f}], "
                f"Win rate: {stats.get('win_rate', 'N/A')}"
            )

            if personality == "analyst":
                tone = (
                    "You are Cerebro. Speak formally with statistical precision. "
                    "Reference confidence intervals, probabilities, and risk metrics explicitly. "
                    "Give a clear assessment. 3-4 sentences max."
                )
            else:
                tone = (
                    "You are Cerebro. Speak casually like a smart friend giving trading advice. "
                    "Use slang naturally. Be direct about what the numbers mean for Professor's money. "
                    "Give a clear recommendation. 3-4 sentences max."
                )

            prompt = (
                f"{tone}\n\n"
                f'Professor ran this simulation: "{query}"\n\n'
                f"Results: {stats_summary}\n"
                f"Analysis summary: {analysis.get('summary', 'N/A')}\n\n"
                f"Give your take on these results. What should Professor do?"
            )

            messages = [ChatMessage(role="user", content=prompt)]
            response = await self.ollama.chat(messages, thinking=False, model="qwen3:8b")
            message = response.content if hasattr(response, "content") else str(response)

            suggestions = self._generate_sim_suggestions(query, sim_data)

            await self._emit("simulation_analysis_chat", {
                "query": query,
                "message": message,
                "personality": personality,
                "suggestions": suggestions,
                "sim_data": sim_data,
            })
        except Exception as e:
            logger.error(f"[Sim] Follow-up generation failed: {e}")

    def _generate_sim_suggestions(self, query: str, sim_data: dict) -> list:
        """Generate contextual follow-up suggestions based on the query and results."""
        q_lower = query.lower()
        suggestions = []

        if any(w in q_lower for w in ["put", "short", "bear"]):
            suggestions.append("What about calls instead?")
            suggestions.append("Show break-even price")
        elif any(w in q_lower for w in ["call", "long", "bull", "buy"]):
            suggestions.append("What about puts instead?")
            suggestions.append("Double the position?")

        suggestions.append("Double the timeframe")
        suggestions.append("What are the main risks?")
        return suggestions[:4]

    async def _handle_saturation(self, directive_id: str, saturation: int):
        """Handle saturation with COMPLEXITY-AWARE completion thresholds."""
        findings_data = self._load_findings()
        directive_findings = findings_data.get(directive_id, {})
        findings_count = len(directive_findings.get("findings", []))

        # Load directive to check its type and text
        directives_file = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "directives.json"
        directive = None
        directive_type = "task"  # Default to task
        directive_text = ""

        if directives_file.exists():
            try:
                directives = json.loads(directives_file.read_text())
                for d in directives:
                    if d.get("id") == directive_id:
                        directive = d
                        directive_type = d.get("type", "task")
                        directive_text = d.get("text", "")
                        break
            except:
                pass

        # Goals never auto-complete
        if directive_type != "task":
            if self._cycles_completed % 15 == 0:
                content = f"Ongoing goal progress: {findings_count} learnings gathered. This goal continues indefinitely."
                self._last_thought = Thought.create(
                    phase=ThoughtPhase.REFLECT,
                    type=ThoughtType.INSIGHT,
                    content=content,
                    confidence=0.8
                )
                await self._emit("thought_stream", self._last_thought.to_dict())
            return

        # === COMPLEXITY-AWARE COMPLETION ===
        # Get or detect task complexity
        if directive_id not in self._task_complexity:
            self._task_complexity[directive_id] = self._detect_task_complexity(directive_text)

        complexity = self._task_complexity[directive_id]
        thresholds = self._get_completion_thresholds(complexity)

        # Track directive start time
        if directive_id not in self._directive_start_times:
            self._directive_start_times[directive_id] = datetime.now(timezone.utc)

        start_time = self._directive_start_times[directive_id]
        elapsed_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Track cycles at threshold saturation
        sat_threshold = thresholds["saturation_threshold"]
        if saturation >= sat_threshold:
            self._increment_saturation_cycles(directive_id, sat_threshold)
        cycles_at_threshold = self._get_cycles_at_saturation(directive_id, sat_threshold)

        # === COMPLETION CONDITIONS ===
        should_complete = False
        completion_reason = ""

        # 1. Saturation threshold met with minimum findings
        if saturation >= sat_threshold and findings_count >= thresholds["min_findings"]:
            should_complete = True
            completion_reason = f"saturation {saturation}% >= {sat_threshold}% with {findings_count} findings"

        # 2. Maximum cycles exceeded (even at lower saturation)
        if cycles_at_threshold >= thresholds["max_cycles"]:
            should_complete = True
            completion_reason = f"max cycles ({cycles_at_threshold} >= {thresholds['max_cycles']})"

        # 3. Time limit exceeded
        if elapsed_seconds >= thresholds["max_time_seconds"]:
            should_complete = True
            completion_reason = f"time limit ({elapsed_seconds:.0f}s >= {thresholds['max_time_seconds']}s)"

        # 4. SIMPLE TASK FAST-TRACK: Complete immediately if we have ANY useful finding
        if complexity == "simple" and findings_count >= 1:
            # Check if we have a web_search or insight finding (actual content)
            has_useful_finding = any(
                f.get("type") in ["web_search", "insight", "learning"]
                for f in directive_findings.get("findings", [])
            )
            if has_useful_finding:
                should_complete = True
                completion_reason = f"simple task fast-track ({findings_count} findings)"

        # 5. AGENT ACTION REQUIRED CHECK - Don't auto-complete tasks that need agent execution
        #    unless spawn_agent was actually triggered
        if should_complete and self._requires_agent_action(directive_text):
            agent_was_triggered = self._agent_triggered.get(directive_id, False)
            if not agent_was_triggered:
                # Task requires agent action but none was triggered - don't auto-complete
                should_complete = False
                print("[CogLoop] BLOCKING auto-complete: task requires agent action but spawn_agent not triggered")
                # Notify user that the task needs Full Autonomy
                content = "This task requires creating files/code. Enable Full Autonomy to allow agent spawning, or I'll keep researching."
                self._last_thought = Thought.create(
                    phase=ThoughtPhase.REFLECT,
                    type=ThoughtType.INSIGHT,
                    content=content,
                    confidence=0.9
                )
                await self._emit("thought_stream", self._last_thought.to_dict())

        if should_complete:
            print(f"[CogLoop] Auto-completing {complexity} task: {completion_reason}")
            await self._auto_complete_task(directive_id, directive, findings_data)
        elif saturation >= sat_threshold - 20 and self._cycles_completed % 3 == 0:
            # Progress update
            content = f"[{complexity.upper()}] Task at {saturation}% ({cycles_at_threshold} cycles, {elapsed_seconds:.0f}s elapsed)..."
            self._last_thought = Thought.create(
                phase=ThoughtPhase.REFLECT,
                type=ThoughtType.INSIGHT,
                content=content,
                confidence=0.9
            )
            await self._emit("thought_stream", self._last_thought.to_dict())

    def _get_cycles_at_saturation(self, directive_id: str, threshold: int) -> int:
        """Track how many cycles we've been at/above a saturation threshold."""
        key = f"{directive_id}_sat_{threshold}"
        return self._saturation_cycles.get(key, 0)

    def _increment_saturation_cycles(self, directive_id: str, threshold: int):
        """Increment cycle count for a saturation threshold."""
        key = f"{directive_id}_sat_{threshold}"
        self._saturation_cycles[key] = self._saturation_cycles.get(key, 0) + 1

    def _detect_task_complexity(self, directive_text: str) -> str:
        """
        Detect task complexity to determine completion thresholds.

        Returns:
            "simple" - Single fact lookup, quick answer (1-3 findings)
            "medium" - Moderate research, multi-step (5-15 findings)
            "complex" - Deep research, open-ended (20+ findings)
        """
        text_lower = directive_text.lower().strip()

        # Browser tasks are NEVER simple — they need multi-step exploration
        browser_indicators = [
            "open ", "go to ", "browse ", "navigate ", "look up ",
            "search for ", "search on ", "look for ", "pull up ",
            ".com", ".org", ".io", ".net", "youtube", "amazon",
            "reddit", "google", "website", "browser",
        ]
        if any(ind in text_lower for ind in browser_indicators):
            return "medium"

        # === SIMPLE TASKS ===
        # Quick fact lookups, single answers
        simple_patterns = [
            # Direct fact requests
            "tell me a fact", "tell me one", "give me a fact", "one fact",
            "what is", "what's", "what are", "who is", "who was", "when did",
            "where is", "how many", "how old", "how tall", "how far",
            # Quick lookups
            "define ", "meaning of", "definition of",
            "capital of", "population of", "height of",
            # Simple conversions/calculations
            "convert ", "calculate ", "how much is",
            # Yes/no questions
            "is it true", "can you", "do you know", "does ",
            # Trivia
            "trivia", "fun fact", "interesting fact", "random fact",
        ]

        # Check for simple patterns
        for pattern in simple_patterns:
            if pattern in text_lower:
                return "simple"

        # Very short questions are usually simple
        word_count = len(text_lower.split())
        if word_count <= 8:
            return "simple"

        # === COMPLEX TASKS ===
        # Deep research, planning, multi-step
        complex_patterns = [
            # Research/analysis
            "research", "investigate", "analyze", "deep dive", "comprehensive",
            "explore all", "find all", "list all",
            # Planning/strategy
            "create a plan", "develop a strategy", "how to achieve",
            "business plan", "roadmap",
            # Open-ended
            "ways to", "methods for", "options for", "alternatives to",
            # Financial/business goals
            "make money", "earn money", "passive income", "side hustle",
            "$", "per month", "per year",
            # Learning tasks
            "learn everything", "teach me about", "explain everything",
        ]

        for pattern in complex_patterns:
            if pattern in text_lower:
                return "complex"

        # Long questions with many words suggest complexity
        if word_count > 20:
            return "complex"

        # Default to medium
        return "medium"

    def _requires_agent_action(self, directive_text: str) -> bool:
        """
        Detect if a task requires agent spawning (file creation, code execution, etc.)
        or browser exploration (navigate to websites, extract data, etc.)

        Tasks that require agent action should NOT be auto-completed based on web search
        findings alone - they need actual agent/browser execution.
        """
        text_lower = directive_text.lower().strip()

        # Keywords indicating file creation/code execution tasks
        agent_required_patterns = [
            # File creation
            "create a file", "create file", "create a .txt", "create txt",
            "create a new file", "make a file", "write a file", "save a file",
            "create a script", "write a script", "build a script",
            "in notepad", "open notepad", "to notepad",
            ".txt file", ".py file", ".js file", ".html file",
            # Code execution
            "run a script", "execute code", "run code", "execute a",
            "create a python", "write python", "build python",
            "create a program", "write a program", "build a program",
            # Agent spawn indicators
            "spawn agent", "spawn a claude", "create claude agent",
            "use claude to", "have claude", "tell claude",
            # Action verbs + file types
            "create and save", "write and save", "generate and save",
            # Browser/Website exploration patterns - CRITICAL for explore_website action
            "go to ", "navigate to ", "browse to ", "visit ",
            "open ", "access ", "check out ",
            "extract from ", "scrape ", "get from ",
            "on the website", "from the website", "from the page",
            "trending", "headlines", "top stories",
            ".com", ".org", ".io", ".net",  # URL indicators
        ]

        for pattern in agent_required_patterns:
            if pattern in text_lower:
                return True

        return False

    def _get_completion_thresholds(self, complexity: str) -> dict:
        """
        Get completion thresholds based on task complexity.

        Returns dict with:
            - saturation_threshold: % to trigger completion
            - min_findings: Minimum findings before completion
            - max_cycles: Maximum cycles before forced completion
            - max_time_seconds: Maximum time before forced completion
        """
        thresholds = {
            "simple": {
                "saturation_threshold": 40,  # Complete quickly
                "min_findings": 1,           # Just 1 finding enough
                "max_cycles": 3,             # Max 3 cycles (~15-45 seconds)
                "max_time_seconds": 60,      # 1 minute max
            },
            "medium": {
                "saturation_threshold": 60,
                "min_findings": 5,
                "max_cycles": 10,
                "max_time_seconds": 300,     # 5 minutes max
            },
            "complex": {
                "saturation_threshold": 80,
                "min_findings": 15,
                "max_cycles": 25,
                "max_time_seconds": 1800,    # 30 minutes max
            }
        }
        return thresholds.get(complexity, thresholds["medium"])

    async def _activate_directive(self, directive_id: str):
        """Mark a directive as 'active' in the JSON file and notify the frontend."""
        directives_file = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "directives.json"
        if not directives_file.exists():
            return
        try:
            directives = json.loads(directives_file.read_text())
            for d in directives:
                if d.get("id") == directive_id:
                    d["status"] = "active"
                    break
            directives_file.write_text(json.dumps(directives, indent=2))
            # Notify frontend so it refreshes the Command tab
            await self._emit("directive_updated", {
                "directive_id": directive_id,
                "status": "active",
            })
            print(f"[CogLoop] Directive {directive_id} activated")
        except Exception as e:
            print(f"[CogLoop] Failed to activate directive {directive_id}: {e}")

    async def _auto_complete_task(self, directive_id: str, directive: dict, findings_data: dict):
        """Auto-complete a task and deliver final answer."""
        if not directive:
            return

        directive_text = directive.get("text", "")
        findings = findings_data.get(directive_id, {}).get("findings", [])
        complexity = self._task_complexity.get(directive_id, "medium")

        # Generate final answer from findings
        final_answer = await self._synthesize_task_result(directive_text, findings)

        # Mark directive as completed
        directives_file = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "directives.json"
        if directives_file.exists():
            try:
                directives = json.loads(directives_file.read_text())
                for d in directives:
                    if d.get("id") == directive_id:
                        d["status"] = "completed"
                        d["completed_at"] = datetime.now().isoformat()
                        d["final_answer"] = final_answer
                        break
                directives_file.write_text(json.dumps(directives, indent=2))
            except Exception as e:
                print(f"[CogLoop] Failed to complete task: {e}")
                return

        # Emit completion thought
        content = f"✅ Task complete: {directive_text[:50]}... Final answer delivered."
        self._last_thought = Thought.create(
            phase=ThoughtPhase.ACT,
            type=ThoughtType.ACTION,
            content=content,
            confidence=0.95
        )
        await self._emit("thought_stream", self._last_thought.to_dict())

        # Notify frontend
        await self._emit("directive_completed", {
            "directive_id": directive_id,
            "directive_text": directive_text,
            "final_answer": final_answer,
            "findings_count": len(findings),
            "complexity": complexity
        })

        # Clean up tracking state for this directive
        if directive_id in self._task_complexity:
            del self._task_complexity[directive_id]
        if directive_id in self._directive_start_times:
            del self._directive_start_times[directive_id]
        if directive_id in self._agent_triggered:
            del self._agent_triggered[directive_id]
        # Clean up saturation cycle trackers
        keys_to_remove = [k for k in self._saturation_cycles.keys() if k.startswith(directive_id)]
        for key in keys_to_remove:
            del self._saturation_cycles[key]

        print(f"[CogLoop] Task auto-completed ({complexity}): {directive_id}")

    async def _synthesize_task_result(self, task_text: str, findings: list) -> str:
        """Use LLM to synthesize findings into a final answer for a task."""
        if not findings:
            return "No findings gathered."

        # Get top findings
        top_findings = findings[:10]
        findings_text = "\n".join([f"- {f.get('content', '')[:200]}" for f in top_findings])

        prompt = f"""Task: {task_text}

Findings gathered:
{findings_text}

Based on these findings, provide a clear, concise final answer that completes this task.
If the task asked for specific information (like headlines, data, etc.), list them clearly.
Keep the response focused and actionable."""

        try:
            response = await self.ooda.ollama.chat(
                [
                    {"role": "system", "content": "You are completing a task. Provide the final deliverable."},
                    {"role": "user", "content": prompt}
                ],
                thinking=False
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception:
            # Fallback to summarizing findings
            return f"Task findings ({len(findings)} items):\n{findings_text[:500]}"

    def get_active_directive(self, directives: List[Dict]) -> Optional[Dict]:
        """Get the currently active (non-paused) directive to work on."""
        for d in directives:
            if d.get("status") in ["active", "pending"] and not d.get("paused", False):
                return d
        return None

    def get_all_active_directives(self, directives: List[Dict]) -> List[Dict]:
        """Get ALL active (non-paused) directives for concurrent dispatch."""
        return [d for d in directives if d.get("status") in ["active", "pending"] and not d.get("paused", False)]

    def _cleanup_stale_directives(self):
        """Check all active directives against actual agent completion records.
        Marks any directive as complete if its agent has finished but the directive
        was not auto-completed (e.g., due to race condition or error)."""
        directives_path = Path(os.environ.get(
            "AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data")
        )) / "cerebro" / "directives.json"

        if not directives_path.exists():
            return

        try:
            directives = json.loads(directives_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        modified = False
        for d in directives:
            if d.get("status") not in ("active", "pending"):
                continue
            directive_id = d.get("id")
            if not directive_id:
                continue
            # If we triggered an agent for this directive but no active agents remain,
            # the directive's agent has finished — mark it complete
            if self._agent_triggered.get(directive_id) and not self._active_agent_ids:
                d["status"] = "completed"
                d["completed_at"] = datetime.now(timezone.utc).isoformat()
                d["auto_cleanup"] = True
                modified = True
                # Clean up tracking state
                self._agent_triggered.pop(directive_id, None)
                self._task_complexity.pop(directive_id, None)
                self._directive_start_times.pop(directive_id, None)
                print(f"[CogLoop] Stale directive cleanup: {directive_id} marked complete (agent finished)")

        if modified:
            try:
                directives_path.write_text(json.dumps(directives, indent=2))
            except OSError as e:
                print(f"[CogLoop] Failed to write cleaned directives: {e}")

    async def maybe_ask_question(self, observation_context: Dict[str, Any]):
        """
        Intelligently decide when to ask Professor a question.

        Ask when:
        - Starting a new directive (clarify)
        - Research hits milestones (direction)
        - Saturation is high (propose paths)
        """
        directives = observation_context.get('directives', [])
        if not directives:
            return

        active_directive = directives[0] if directives else None
        if not active_directive:
            return

        directive_id = active_directive.get('id', '')

        # Load findings to check saturation
        findings_data = self._load_findings()
        directive_findings = findings_data.get(directive_id, {})
        findings_count = len(directive_findings.get('findings', []))
        saturation = directive_findings.get('saturation', 0)

        # Decide if we should ask a question
        should_ask = False
        question_reason = ""

        # Ask clarifying question early (cycles 2-3)
        if self._cycles_completed in [2, 3] and findings_count < 10:
            should_ask = True
            question_reason = "early_clarify"

        # Ask direction question at 30% saturation
        elif saturation >= 30 and saturation < 40 and self._cycles_completed % 3 == 0:
            should_ask = True
            question_reason = "direction"

        # Ask strategic question at 60%+ saturation
        elif saturation >= 60 and not hasattr(self, '_asked_strategic_for'):
            should_ask = True
            question_reason = "strategic_paths"
            self._asked_strategic_for = directive_id

        # Also ask every 8 cycles as a check-in
        elif self._cycles_completed % 8 == 0 and self._cycles_completed > 0:
            should_ask = True
            question_reason = "periodic_checkin"

        if not should_ask:
            return

        try:
            print(f"[CogLoop] Generating question (reason: {question_reason}, saturation: {saturation}%)")
            question_data = await self.generate_proactive_question(observation_context)

            if question_data and question_data.get('question'):
                # Log thought about asking
                thought = Thought.create(
                    phase=ThoughtPhase.DECIDE,
                    type=ThoughtType.DECISION,
                    content="I have a question for Professor...",
                    confidence=0.9
                )
                await self._emit("thought_stream", thought.to_dict())

                # Emit the question
                await self._emit("proactive_question", {
                    "question": question_data['question'],
                    "type": question_data.get('type', 'general'),
                    "options": question_data.get('options', []),
                    "paths": question_data.get('paths', []),
                    "directive_id": question_data.get('directive_id', ''),
                    "findings_count": question_data.get('findings_count', 0),
                    "saturation": question_data.get('saturation', 0)
                })
                print(f"[CogLoop] Asked: {question_data['question'][:80]}...")

        except Exception as e:
            print(f"[CogLoop] Error in maybe_ask_question: {e}")

    async def store_question_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Store the user's answer to a proactive question.

        This helps Cerebro learn about the user over time AND immediately
        injects the answer into the next OBSERVE phase.
        """
        try:
            # ADD TO PENDING QUEUE - Cerebro will see this in the next OBSERVE phase!
            self._pending_user_answers.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            print(f"[CogLoop] Professor answered! Queue size: {len(self._pending_user_answers)}")

            # Record as a learning about the user
            await self.reflexion.record_learning(
                problem=f"User response to: {question[:100]}",
                solution=answer[:500],
                context="Proactive question answered by user",
                tags=["user_response", "proactive_question", "preferences"]
            )

            # Log thought about receiving answer - more personal!
            thought = Thought.create(
                phase=ThoughtPhase.OBSERVE,
                type=ThoughtType.OBSERVATION,
                content=f"Professor just responded! He said: \"{answer[:100]}{'...' if len(answer) > 100 else ''}\"",
                reasoning=f"My question was: {question[:100]}",
                confidence=1.0,
                session_id=self._current_session_id,
                user_response=True,
                important=True
            )
            await self.journal.log_thought(thought)
            await self._emit("thought_stream", thought.to_dict())

            return {"success": True, "message": "Answer recorded and queued for next cycle"}
        except Exception as e:
            print(f"[CogLoop] Error storing answer: {e}")
            return {"success": False, "error": str(e)}

    # ========================================
    # AI MEMORY INTEGRATION
    # ========================================

    async def save_research_to_memory(
        self,
        directive_text: str,
        search_results: List[Dict],
        query: str
    ) -> Dict[str, Any]:
        """
        Save research findings to AI Memory via SolutionTracker.

        This is the CRITICAL integration that persists Cerebro's research
        to the modular brain (AI Memory) so other AIs can access it.

        Args:
            directive_text: The directive/mission being researched
            search_results: List of web search results
            query: The search query used

        Returns:
            Summary of what was saved
        """
        if not self.solution_tracker:
            print("[CogLoop] Cannot save to AI Memory - SolutionTracker not available")
            return {"success": False, "error": "SolutionTracker not available"}

        saved_count = 0
        skipped_count = 0

        for result in search_results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            url = result.get("url", "")

            # === VALIDATION: Skip garbage ===
            if not title or not snippet:
                skipped_count += 1
                continue

            if len(snippet) < 20:
                skipped_count += 1
                continue

            # === DEDUPLICATION ===
            content_hash = hashlib.md5(f"{url}:{snippet[:100]}".encode()).hexdigest()[:16]
            if content_hash in self._saved_hashes:
                skipped_count += 1
                continue

            # === FORMAT FOR AI MEMORY ===
            # Structure like how I (Claude) would record it
            problem = f"Research: {query[:100]}"
            solution = f"""## {title}

{snippet}

**Source:** {url}
**Directive:** {directive_text[:200]}"""

            # Auto-generate tags based on content
            tags = self._auto_categorize_research(directive_text, title, snippet)
            tags.append("cerebro_research")
            tags.append("web_search")

            try:
                self.solution_tracker.record_solution(
                    problem=problem,
                    solution=solution,
                    context=f"Autonomous research for directive: {directive_text[:100]}",
                    tags=list(set(tags))  # Dedupe tags
                )
                self._saved_hashes.add(content_hash)
                saved_count += 1
                print(f"[CogLoop] SAVED to AI Memory: {title[:50]}...")

            except Exception as e:
                print(f"[CogLoop] ERROR saving to AI Memory: {e}")

        result_summary = {
            "success": True,
            "saved": saved_count,
            "skipped": skipped_count,
            "query": query
        }
        print(f"[CogLoop] AI Memory save complete: {saved_count} saved, {skipped_count} skipped")
        return result_summary

    def _auto_categorize_research(self, directive: str, title: str, snippet: str) -> List[str]:
        """
        Automatically categorize research with proper tags.
        Matches how I (Claude) would categorize this content.
        """
        all_text = f"{directive} {title} {snippet}".lower()
        tags = []

        # Financial/Trading
        if any(kw in all_text for kw in ["trading", "stock", "option", "invest", "finance", "market", "price"]):
            tags.extend(["trading", "finance"])
        if any(kw in all_text for kw in ["call", "put", "strike", "expir"]):
            tags.append("options_trading")

        # Programming
        if any(kw in all_text for kw in ["python", "javascript", "typescript", "code", "programming", "api"]):
            tags.append("programming")
        if any(kw in all_text for kw in ["react", "vue", "angular", "frontend"]):
            tags.append("frontend")
        if any(kw in all_text for kw in ["backend", "server", "database", "sql"]):
            tags.append("backend")

        # AI/ML
        if any(kw in all_text for kw in ["machine learning", "ai", "neural", "model", "deep learning"]):
            tags.append("ai_ml")

        # Tutorial/Educational
        if any(kw in all_text for kw in ["beginner", "tutorial", "guide", "learn", "step-by-step", "how to"]):
            tags.append("educational")

        # Source credibility
        if any(domain in all_text for domain in ["investopedia", "nerdwallet", "schwab", "fidelity"]):
            tags.append("trusted_source")
        if "reddit" in all_text:
            tags.append("community_discussion")
        if "youtube" in all_text:
            tags.append("video_content")

        return tags
