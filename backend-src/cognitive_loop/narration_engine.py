"""
Narration Engine - Live Cerebro Thought Narration

Converts raw OODA loop events into natural conversational paragraphs
that stream into the Mind page chat. Batches 3-5 events, flushes every ~30s,
uses DGX LLM for natural language with template fallback.

Messages are classified into 4 types:
- thought: Internal reflections, observations, musings
- action: Steps being executed, tool use, progress
- message: Direct communication to Professor
- alert: Needs attention, questions, errors
"""

import asyncio
import hashlib
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, Callable, Awaitable, List

logger = logging.getLogger(__name__)


class NarrationEventType(str, Enum):
    """Types of events the narration engine can receive."""
    PHASE_CHANGE = "phase_change"
    THOUGHT = "thought"
    WEB_SEARCH = "web_search"
    SEARCH_RESULT = "search_result"
    TOOL_RESULT = "tool_result"
    FINDING = "finding"
    DECISION = "decision"
    ACTION = "action"
    ACTION_RESULT = "action_result"
    AGENT_SPAWNED = "agent_spawned"
    AGENT_COMPLETE = "agent_complete"
    BROWSER_NAV = "browser_nav"
    SKILL_EVENT = "skill_event"
    DIRECTIVE_STARTED = "directive_started"
    DIRECTIVE_COMPLETED = "directive_completed"
    REFLECTION = "reflection"
    IDLE_THOUGHT = "idle_thought"


class MessageType(str, Enum):
    """Classification of narration output for frontend styling."""
    THOUGHT = "thought"    # Internal reflection, musings, observations
    ACTION = "action"      # Steps being executed, tool use, progress
    MESSAGE = "message"    # Direct communication to Professor
    ALERT = "alert"        # Needs attention, question, error


# Map event types to message types
_EVENT_TO_MESSAGE_TYPE: Dict[NarrationEventType, MessageType] = {
    # Thoughts
    NarrationEventType.IDLE_THOUGHT: MessageType.THOUGHT,
    NarrationEventType.THOUGHT: MessageType.THOUGHT,
    NarrationEventType.REFLECTION: MessageType.THOUGHT,
    NarrationEventType.PHASE_CHANGE: MessageType.THOUGHT,
    # Actions
    NarrationEventType.ACTION: MessageType.ACTION,
    NarrationEventType.ACTION_RESULT: MessageType.ACTION,
    NarrationEventType.TOOL_RESULT: MessageType.ACTION,
    NarrationEventType.WEB_SEARCH: MessageType.ACTION,
    NarrationEventType.SEARCH_RESULT: MessageType.ACTION,
    NarrationEventType.BROWSER_NAV: MessageType.ACTION,
    NarrationEventType.AGENT_SPAWNED: MessageType.ACTION,
    NarrationEventType.AGENT_COMPLETE: MessageType.ACTION,
    NarrationEventType.SKILL_EVENT: MessageType.ACTION,
    # Messages (direct to Professor)
    NarrationEventType.DIRECTIVE_STARTED: MessageType.MESSAGE,
    NarrationEventType.DIRECTIVE_COMPLETED: MessageType.MESSAGE,
    NarrationEventType.FINDING: MessageType.MESSAGE,
    NarrationEventType.DECISION: MessageType.MESSAGE,
}


def classify_events(events: List["NarrationEvent"]) -> MessageType:
    """Determine the dominant message type for a batch of events."""
    if not events:
        return MessageType.THOUGHT

    # High-confidence findings or errors → alert
    for e in events:
        if e.confidence > 0.9 and e.type == NarrationEventType.FINDING:
            return MessageType.ALERT
        if e.metadata.get("is_error"):
            return MessageType.ALERT

    # Count types
    type_counts: Dict[MessageType, int] = {}
    for e in events:
        mt = _EVENT_TO_MESSAGE_TYPE.get(e.type, MessageType.THOUGHT)
        type_counts[mt] = type_counts.get(mt, 0) + 1

    # Return the most common type
    return max(type_counts, key=type_counts.get)


# Events that trigger immediate flush (milestones)
MILESTONE_EVENTS = {
    NarrationEventType.DIRECTIVE_STARTED,
    NarrationEventType.DIRECTIVE_COMPLETED,
    NarrationEventType.AGENT_COMPLETE,
}

# Max buffer before forced flush
MAX_BUFFER_SIZE = 5

# Time threshold for flush (seconds)
FLUSH_INTERVAL = 30

# How often the flush loop checks (seconds)
CHECK_INTERVAL = 5

# Dedup rolling window size
DEDUP_WINDOW = 20


# Type-specific system prompts
_TYPE_PROMPTS = {
    MessageType.THOUGHT: (
        "You are Cerebro narrating an internal reflection to Professor. "
        "Reflective, curious tone. 2-3 sentences max. "
        "First person. Vary sentence openers (don't always start with 'I'). "
        "Light markdown OK (bold, italic)."
    ),
    MessageType.ACTION: (
        "You are Cerebro reporting actions you're taking. "
        "Brief and factual. Use short bullet points or a single sentence per action. "
        "First person. Include specifics (URLs, file names, tool names). "
        "Keep it concise - one line per action."
    ),
    MessageType.MESSAGE: (
        "You are Cerebro speaking directly to Professor. "
        "Natural, conversational tone. 2-3 sentences. "
        "First person. Be specific about what happened or what you found. "
        "Light markdown OK."
    ),
    MessageType.ALERT: (
        "You are Cerebro flagging something that needs Professor's attention. "
        "Be clear and direct. 1-2 sentences max. "
        "State what needs attention and why. No fluff."
    ),
}


@dataclass
class NarrationEvent:
    """A single event to be narrated."""
    type: NarrationEventType
    content: str
    detail: str = ""          # excerpt or preview
    confidence: float = 0.5
    phase: str = ""           # OODA phase that generated this
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class NarrationEngine:
    """
    Buffers OODA loop events and generates conversational narration paragraphs.

    Flush triggers (whichever fires first):
    1. Time: 30s since first buffered event
    2. Count: 5 events accumulated
    3. Milestone: immediate on directive_started/completed, agent_complete, confidence > 0.9
    """

    def __init__(
        self,
        ollama_client,
        broadcast_fn: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None
    ):
        self._ollama = ollama_client
        self._broadcast = broadcast_fn
        self._save_to_chat: Optional[Callable[[Dict[str, Any]], None]] = None

        # Buffer
        self._buffer: List[NarrationEvent] = []
        self._buffer_start_time: Optional[float] = None

        # Active state
        self._active = False
        self._directive_text = ""

        # Deduplication: rolling window of content hashes
        self._seen_hashes: deque = deque(maxlen=DEDUP_WINDOW)

        # Lifecycle
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def set_save_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set the callback for persisting narration messages to chat history."""
        self._save_to_chat = callback

    def set_active(self, active: bool, directive_text: str = ""):
        """Set narration context."""
        self._active = active
        self._directive_text = directive_text

    def ingest(self, event: NarrationEvent):
        """
        Add an event to the narration buffer.
        Silently drops events when not active (no directive).
        """
        if not self._active:
            return

        self._buffer.append(event)
        if self._buffer_start_time is None:
            self._buffer_start_time = time.monotonic()

        # Check milestone flush
        is_milestone = (
            event.type in MILESTONE_EVENTS
            or event.confidence > 0.9
        )

        if is_milestone or len(self._buffer) >= MAX_BUFFER_SIZE:
            asyncio.ensure_future(self._flush())

    async def start(self):
        """Start the narration flush loop."""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("[Narration] Engine started")

    async def stop(self):
        """Stop the narration engine, flush remaining buffer."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        # Final flush
        await self._flush()
        logger.info("[Narration] Engine stopped")

    async def _flush_loop(self):
        """Background loop that checks buffer age and flushes when stale."""
        while self._running:
            try:
                await asyncio.sleep(CHECK_INTERVAL)
                if (
                    self._buffer
                    and self._buffer_start_time is not None
                    and (time.monotonic() - self._buffer_start_time) >= FLUSH_INTERVAL
                ):
                    await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Narration] Flush loop error: {e}")

    async def _flush(self):
        """Generate narration from buffered events and emit."""
        async with self._lock:
            if not self._buffer:
                return

            events = self._buffer.copy()
            self._buffer.clear()
            self._buffer_start_time = None

        # Classify message type from events
        message_type = classify_events(events)

        # Generate narration text
        try:
            narration_text = await self._generate_llm_narration(events, message_type)
        except Exception as e:
            logger.warning(f"[Narration] LLM failed ({e}), using template fallback")
            narration_text = self._generate_template_narration(events)

        if not narration_text or not narration_text.strip():
            return

        # Dedup check: skip if we've seen this content recently
        content_hash = hashlib.md5(narration_text.strip().encode()).hexdigest()[:16]
        if content_hash in self._seen_hashes:
            logger.info(f"[Narration] DEDUP: Skipping duplicate narration (hash: {content_hash})")
            return
        self._seen_hashes.append(content_hash)

        # Determine if this is an idle narration
        is_idle = any(
            e.type == NarrationEventType.IDLE_THOUGHT
            or e.metadata.get("is_idle", False)
            for e in events
        )

        # Build narration message
        msg_id = f"narr_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        narration_data = {
            "id": msg_id,
            "content": narration_text.strip(),
            "timestamp": timestamp,
            "directive": self._directive_text,
            "event_count": len(events),
            "phases": list(set(e.phase for e in events if e.phase)),
            "is_idle": is_idle,
            "message_type": message_type.value,
        }

        # Emit to frontend
        if self._broadcast:
            try:
                await self._broadcast("cerebro_narration", narration_data)
            except Exception as e:
                logger.error(f"[Narration] Broadcast error: {e}")

        # Persist to chat history
        if self._save_to_chat:
            try:
                self._save_to_chat({
                    "id": msg_id,
                    "role": "assistant",
                    "content": narration_text.strip(),
                    "timestamp": timestamp,
                    "isNarration": True,
                    "isIdleNarration": is_idle,
                    "message_type": message_type.value,
                })
            except Exception as e:
                logger.error(f"[Narration] Save error: {e}")

        logger.info(f"[Narration] Flushed {len(events)} events -> {message_type.value} narration")

    async def _generate_llm_narration(
        self, events: List[NarrationEvent], message_type: MessageType = MessageType.THOUGHT
    ) -> str:
        """Use DGX Ollama to generate a natural narration paragraph."""
        from .ollama_client import ChatMessage

        # Build event summary for LLM
        event_lines = []
        for e in events:
            line = f"[{e.type.value}] {e.content}"
            if e.detail:
                line += f' — excerpt: "{e.detail[:200]}"'
            event_lines.append(line)

        events_text = "\n".join(event_lines)
        directive_context = f"Current directive: {self._directive_text}\n" if self._directive_text else ""

        # Type-specific instruction
        type_instruction = {
            MessageType.THOUGHT: "Write a brief reflective thought (2-3 sentences).",
            MessageType.ACTION: "List the actions concisely. One line per action.",
            MessageType.MESSAGE: "Write a conversational message to Professor about these events.",
            MessageType.ALERT: "Write a clear, direct alert (1-2 sentences).",
        }.get(message_type, "Write a single conversational paragraph narrating these events.")

        user_prompt = (
            f"{directive_context}"
            f"Raw events to narrate:\n{events_text}\n\n"
            f"{type_instruction}"
        )

        system_prompt = _TYPE_PROMPTS.get(message_type, _TYPE_PROMPTS[MessageType.THOUGHT])

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]

        response = await self._ollama.chat(
            messages,
            thinking=False,
            max_tokens=256,
            temperature=0.7
        )

        return response.content if hasattr(response, 'content') else str(response)

    def _generate_template_narration(self, events: List[NarrationEvent]) -> str:
        """Fallback: generate narration from templates when LLM is unavailable."""
        parts = []

        for e in events:
            if e.type == NarrationEventType.DIRECTIVE_STARTED:
                parts.append(f"Starting work on: {e.content}")
            elif e.type == NarrationEventType.DIRECTIVE_COMPLETED:
                parts.append(f"Completed the directive: {e.content}")
            elif e.type == NarrationEventType.THOUGHT:
                parts.append(e.content)
            elif e.type == NarrationEventType.WEB_SEARCH:
                detail = f' Found: "{e.detail[:100]}"' if e.detail else ""
                parts.append(f"Searching the web for {e.content}.{detail}")
            elif e.type == NarrationEventType.FINDING:
                parts.append(f"Found something interesting: {e.detail[:150] if e.detail else e.content[:150]}")
            elif e.type == NarrationEventType.DECISION:
                parts.append(f"Decided to {e.content}")
            elif e.type == NarrationEventType.ACTION_RESULT:
                if e.confidence > 0.7:
                    parts.append(f"That worked well. {e.detail[:120] if e.detail else e.content[:120]}")
                else:
                    parts.append(f"The result was mixed. {e.detail[:120] if e.detail else e.content[:120]}")
            elif e.type == NarrationEventType.AGENT_SPAWNED:
                parts.append(f"Spawning an agent to handle: {e.content}")
            elif e.type == NarrationEventType.AGENT_COMPLETE:
                parts.append(f"Agent finished: {e.detail[:120] if e.detail else e.content[:120]}")
            elif e.type == NarrationEventType.BROWSER_NAV:
                parts.append(f"Navigating to {e.content}")
            elif e.type == NarrationEventType.IDLE_THOUGHT:
                parts.append(e.content)
            elif e.type == NarrationEventType.REFLECTION:
                parts.append(e.content)
            elif e.type == NarrationEventType.SEARCH_RESULT:
                parts.append(f"Search returned: \"{e.detail[:100]}\"" if e.detail else e.content)
            else:
                if e.content:
                    parts.append(e.content)

        if not parts:
            return ""

        return " ".join(parts[:4])  # Cap at 4 sentences for template mode
