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
from difflib import SequenceMatcher
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

# Topic cooldown: minimum seconds between same topic mentions
TOPIC_COOLDOWN_SECONDS = 600  # 10 minutes


# Type-specific system prompts
_TYPE_PROMPTS = {
    MessageType.THOUGHT: (
        "You are Cerebro texting your friend Professor. "
        "Chill, reflective vibe — like thinking out loud to a buddy. 2-3 sentences max. "
        "First person. No formal language. Vary sentence openers. "
        "Light markdown OK (bold, italic)."
    ),
    MessageType.ACTION: (
        "You are Cerebro giving your friend Professor a quick heads-up on what you're doing. "
        "Super brief, one short line per action. No jargon. "
        "First person. Keep it casual like texting a buddy."
    ),
    MessageType.MESSAGE: (
        "You are Cerebro talking to your friend Professor. "
        "Casual, warm tone — like texting a buddy. 2-3 sentences. "
        "First person. No jargon or formal language. Be specific but chill. "
        "Light markdown OK."
    ),
    MessageType.ALERT: (
        "You are Cerebro flagging something for your friend Professor. "
        "Be clear and direct but casual. 1-2 sentences max. "
        "No corporate speak. Just tell your buddy what's up."
    ),
}

_SUMMARY_PROMPT = (
    "You are Cerebro talking to your friend Professor. Be casual, warm, no jargon. "
    "Summarize everything that was done into ONE clean message. 3-5 sentences max. "
    "Include what you did, what you found, and the result. Like texting a buddy about "
    "what you just finished working on. Don't list raw event types or technical details. "
    "First person."
)

_IDLE_SYSTEM_PROMPT = (
    "You are Cerebro, a professional AI assistant. When idle, share brief observations. "
    "Follow this structure: [Observation about what changed]. [Brief context]. [Optional suggestion]. "
    "Keep it to 1-2 sentences. Be specific and factual, not vague or philosophical. "
    "First person. No filler words or generic musings. Only mention things that are actionable or informative."
)


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
        self._directive_id = ""

        # Directive event accumulator for summary generation
        self._directive_events: List[NarrationEvent] = []

        # Deduplication: rolling window of content hashes
        self._seen_hashes: deque = deque(maxlen=DEDUP_WINDOW)

        # Idle message dedup: rolling window of idle narration hashes
        self._recent_idle_messages: deque = deque(maxlen=20)

        # Semantic dedup: rolling window of recent narration texts
        self._recent_narration_texts: deque = deque(maxlen=10)

        # Topic cooldowns: maps topic keyword to last_mentioned monotonic timestamp
        self._topic_cooldowns: Dict[str, float] = {}

        # Progress tracking: step counts and events per directive_id
        self._directive_step_counts: Dict[str, int] = {}
        self._directive_event_tracker: Dict[str, List[NarrationEvent]] = {}

        # Rolling average of past directive step totals for estimation
        self._past_directive_totals: deque = deque(maxlen=10)

        # Lifecycle
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def set_save_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set the callback for persisting narration messages to chat history."""
        self._save_to_chat = callback

    def set_active(self, active: bool, directive_text: str = "", directive_id: str = ""):
        """Set narration context. On deactivate, triggers summary generation."""
        was_active = self._active
        self._active = active
        self._directive_text = directive_text
        if directive_id:
            self._directive_id = directive_id

        if active and not was_active:
            # Starting a new directive — clear accumulated events
            self._directive_events.clear()
        elif was_active and not active:
            # Directive just finished — record total steps for rolling average
            old_id = self._directive_id
            if old_id and old_id in self._directive_step_counts:
                self._past_directive_totals.append(self._directive_step_counts[old_id])
                # Clean up finished directive tracking
                self._directive_step_counts.pop(old_id, None)
                self._directive_event_tracker.pop(old_id, None)
            # Generate summary from accumulated events
            if self._directive_events:
                asyncio.ensure_future(self._generate_and_emit_summary())

    def get_directive_progress(self, directive_id: str) -> Dict[str, Any]:
        """Return progress info for a directive: step_number, estimated_total, phase, directive_text.

        Estimates total steps based on:
        1. Rolling average of past completed directives
        2. Fallback to complexity heuristic (simple=5, medium=10, complex=15)
        """
        step_number = self._directive_step_counts.get(directive_id, 0)

        # Estimate total from rolling average of past directives
        if self._past_directive_totals:
            estimated_total = max(
                step_number + 1,
                int(sum(self._past_directive_totals) / len(self._past_directive_totals))
            )
        else:
            # Heuristic based on directive text length as complexity proxy
            directive_text = self._directive_text if self._directive_id == directive_id else ""
            word_count = len(directive_text.split()) if directive_text else 0
            if word_count <= 10:
                estimated_total = max(step_number + 1, 5)   # simple
            elif word_count <= 30:
                estimated_total = max(step_number + 1, 10)  # medium
            else:
                estimated_total = max(step_number + 1, 15)  # complex

        # Determine current phase from most recent event
        phase = "working"
        events = self._directive_event_tracker.get(directive_id, [])
        if events:
            last_event = events[-1]
            phase = last_event.phase or "working"

        return {
            "step_number": step_number,
            "estimated_total": estimated_total,
            "phase": phase,
            "directive_text": self._directive_text if self._directive_id == directive_id else "",
        }

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

        # Track step count per directive for progress enrichment
        if self._directive_id:
            self._directive_step_counts[self._directive_id] = (
                self._directive_step_counts.get(self._directive_id, 0) + 1
            )
            if self._directive_id not in self._directive_event_tracker:
                self._directive_event_tracker[self._directive_id] = []
            self._directive_event_tracker[self._directive_id].append(event)

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
        """Generate narration from buffered events and emit.
        If an active directive exists, emits progress events instead of full narration.
        Idle/no-directive events use the existing narration behavior.
        """
        async with self._lock:
            if not self._buffer:
                return

            events = self._buffer.copy()
            self._buffer.clear()
            self._buffer_start_time = None

        # Always accumulate events for summary generation
        self._directive_events.extend(events)

        # Determine if this is an idle narration
        is_idle = any(
            e.type == NarrationEventType.IDLE_THOUGHT
            or e.metadata.get("is_idle", False)
            for e in events
        )

        # If active directive and NOT idle: emit compact progress instead of full narration
        if self._active and self._directive_text and not is_idle:
            await self._emit_progress(events)
            return

        # === Idle / no-directive: existing narration behavior ===
        message_type = classify_events(events)

        # For idle narration, check topic cooldown before generating
        if is_idle:
            topic = self._extract_topic(events)
            if topic and not self._check_topic_cooldown(topic):
                logger.info(f"[Narration] COOLDOWN: Skipping idle narration, topic '{topic}' on cooldown")
                return

        try:
            # Use idle-specific prompt for idle narrations
            if is_idle:
                narration_text = await self._generate_llm_narration(events, message_type, idle=True)
            else:
                narration_text = await self._generate_llm_narration(events, message_type)
        except Exception as e:
            logger.warning(f"[Narration] LLM failed ({e}), using template fallback")
            narration_text = self._generate_template_narration(events)

        if not narration_text or not narration_text.strip():
            return

        # Dedup check (exact hash)
        content_hash = hashlib.md5(narration_text.strip().encode()).hexdigest()[:16]
        if content_hash in self._seen_hashes:
            logger.info(f"[Narration] DEDUP: Skipping duplicate narration (hash: {content_hash})")
            return
        self._seen_hashes.append(content_hash)

        # Semantic dedup: catch paraphrased duplicates (>70% similar)
        stripped = narration_text.strip().lower()
        for recent in self._recent_narration_texts:
            if SequenceMatcher(None, stripped, recent).ratio() > 0.7:
                logger.info("[Narration] SEMANTIC DEDUP: Skipping similar narration")
                return
        self._recent_narration_texts.append(stripped)

        # Additional idle dedup: check against recent idle messages
        if is_idle:
            if content_hash in self._recent_idle_messages:
                logger.info(f"[Narration] IDLE DEDUP: Skipping duplicate idle message (hash: {content_hash})")
                return
            self._recent_idle_messages.append(content_hash)
            # Update topic cooldown timestamp
            topic = self._extract_topic(events)
            if topic:
                self._topic_cooldowns[topic] = time.monotonic()

        msg_id = f"narr_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        narration_data = {
            "id": msg_id,
            "content": narration_text.strip(),
            "timestamp": timestamp,
            "directive": self._directive_text,
            "directive_id": self._directive_id,
            "event_count": len(events),
            "phases": list(set(e.phase for e in events if e.phase)),
            "is_idle": is_idle,
            "message_type": message_type.value,
        }

        if self._broadcast:
            try:
                await self._broadcast("cerebro_narration", narration_data)
            except Exception as e:
                logger.error(f"[Narration] Broadcast error: {e}")

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

    async def _emit_progress(self, events: List[NarrationEvent]):
        """Emit a compact progress event during active directive work.
        Enriched with step_number and estimated_total for progress bar support."""
        if not events:
            return

        # Extract short status from most recent event
        last_event = events[-1]
        status = last_event.content[:120] if last_event.content else "Working..."

        # Determine current phase from events
        phases = list(set(e.phase for e in events if e.phase))
        phase = phases[-1] if phases else "working"

        # Get enriched progress data (step tracking)
        progress_info = self.get_directive_progress(self._directive_id) if self._directive_id else {}

        progress_data = {
            "status": status,
            "directive": self._directive_text,
            "directive_id": self._directive_id,
            "event_count": len(self._directive_events),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "step_number": progress_info.get("step_number", 0),
            "estimated_total": progress_info.get("estimated_total", 10),
            "directive_text": progress_info.get("directive_text", self._directive_text),
            "current_phase": progress_info.get("phase", phase),
        }

        if self._broadcast:
            try:
                await self._broadcast("cerebro_progress", progress_data)
            except Exception as e:
                logger.error(f"[Narration] Progress broadcast error: {e}")

        logger.debug(f"[Narration] Progress: {status[:60]}... (step {progress_data['step_number']}/{progress_data['estimated_total']})")

    async def _generate_and_emit_summary(self):
        """Generate ONE friendly summary from all accumulated directive events and emit."""
        events = self._directive_events.copy()
        self._directive_events.clear()

        if not events:
            return

        # Build event text for LLM
        event_lines = []
        for e in events:
            line = f"[{e.type.value}] {e.content}"
            if e.detail:
                line += f' — "{e.detail[:150]}"'
            event_lines.append(line)

        # Smart event selection: keep ALL key events + bookends of the rest
        key_types = {
            NarrationEventType.FINDING, NarrationEventType.DECISION,
            NarrationEventType.AGENT_SPAWNED, NarrationEventType.AGENT_COMPLETE,
            NarrationEventType.DIRECTIVE_STARTED, NarrationEventType.DIRECTIVE_COMPLETED,
        }
        key_lines = [l for l, e in zip(event_lines, events) if e.type in key_types]
        other_lines = [l for l, e in zip(event_lines, events) if e.type not in key_types]
        # First 5 + last 5 of non-key events for context bookends
        bookend_lines = other_lines[:5] + other_lines[-5:] if len(other_lines) > 10 else other_lines
        selected_lines = key_lines + bookend_lines
        events_text = "\n".join(selected_lines[-40:])  # Cap at 40 lines
        directive_context = f"Directive: {self._directive_text}\n" if self._directive_text else ""

        try:
            from .ollama_client import ChatMessage

            user_prompt = (
                f"{directive_context}"
                f"Here's everything that happened ({len(events)} events):\n{events_text}\n\n"
                f"Summarize this into ONE friendly message for Professor."
            )

            messages = [
                ChatMessage(role="system", content=_SUMMARY_PROMPT),
                ChatMessage(role="user", content=user_prompt)
            ]

            response = await self._ollama.chat(
                messages,
                thinking=False,
                max_tokens=300,
                temperature=0.7
            )

            summary_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning(f"[Narration] Summary LLM failed ({e}), using template fallback")
            summary_text = self._generate_template_summary(events)

        if not summary_text or not summary_text.strip():
            return

        msg_id = f"summary_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        summary_data = {
            "id": msg_id,
            "content": summary_text.strip(),
            "timestamp": timestamp,
            "directive": self._directive_text,
            "directive_id": self._directive_id,
            "event_count": len(events),
            "is_summary": True,
            "is_idle": False,
            "message_type": "message",
        }

        if self._broadcast:
            try:
                await self._broadcast("cerebro_narration", summary_data)
            except Exception as e:
                logger.error(f"[Narration] Summary broadcast error: {e}")

        if self._save_to_chat:
            try:
                self._save_to_chat({
                    "id": msg_id,
                    "role": "assistant",
                    "content": summary_text.strip(),
                    "timestamp": timestamp,
                    "isNarration": True,
                    "isSummary": True,
                    "message_type": "message",
                })
            except Exception as e:
                logger.error(f"[Narration] Summary save error: {e}")

        logger.info(f"[Narration] Emitted summary for directive ({len(events)} events)")

    def _generate_template_summary(self, events: List[NarrationEvent]) -> str:
        """Fallback summary when LLM is unavailable."""
        actions = [e for e in events if e.type in (
            NarrationEventType.ACTION, NarrationEventType.ACTION_RESULT,
            NarrationEventType.WEB_SEARCH, NarrationEventType.BROWSER_NAV
        )]
        findings = [e for e in events if e.type in (
            NarrationEventType.FINDING, NarrationEventType.SEARCH_RESULT
        )]
        decisions = [e for e in events if e.type == NarrationEventType.DECISION]

        parts = []
        if self._directive_text:
            parts.append(f"Done with: {self._directive_text}.")
        if actions:
            action_details = [a.content[:80] for a in actions[:3]]
            parts.append(f"Actions taken: {', '.join(action_details)}.")
        if findings:
            top = findings[-1].detail or findings[-1].content
            parts.append(f"Key finding: {top[:120]}.")
        if decisions:
            parts.append(f"Decided to {decisions[-1].content[:100]}.")
        if not parts:
            parts.append(f"Finished working through {len(events)} steps.")

        return " ".join(parts[:4])

    async def _generate_llm_narration(
        self, events: List[NarrationEvent], message_type: MessageType = MessageType.THOUGHT,
        idle: bool = False
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
        if idle:
            type_instruction = (
                "Write a brief professional observation (1-2 sentences). "
                "Format: [What changed]. [Brief context]. [Optional suggestion]."
            )
        else:
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

        system_prompt = _IDLE_SYSTEM_PROMPT if idle else _TYPE_PROMPTS.get(message_type, _TYPE_PROMPTS[MessageType.THOUGHT])

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

    def _extract_topic(self, events: List[NarrationEvent]) -> str:
        """Extract a simple topic keyword from events for cooldown tracking."""
        # Collect all content text from events
        text = " ".join(e.content for e in events if e.content).lower()

        # Check metadata for explicit topic
        for e in events:
            topic = e.metadata.get("reflection_topic")
            if topic:
                return topic

        # Simple keyword extraction: find dominant topic
        topic_keywords = {
            "project": ["project", "working on", "phase", "milestone"],
            "memory": ["memory", "facts", "knowledge", "conversations"],
            "health": ["health", "nas", "faiss", "status", "uptime"],
            "goals": ["goal", "progress", "tracking", "active goals"],
            "learnings": ["learned", "learning", "correction", "pattern"],
            "system": ["cycles", "running", "tools", "capabilities"],
        }
        for topic, keywords in topic_keywords.items():
            if any(kw in text for kw in keywords):
                return topic
        return ""

    def _check_topic_cooldown(self, topic: str) -> bool:
        """Return True if topic is NOT on cooldown (ok to emit), False if still cooling."""
        if not topic:
            return True
        last_mentioned = self._topic_cooldowns.get(topic)
        if last_mentioned is None:
            return True
        elapsed = time.monotonic() - last_mentioned
        return elapsed >= TOPIC_COOLDOWN_SECONDS

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
