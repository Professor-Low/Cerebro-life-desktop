"""
Thought Journal - Persistent Thought Storage

Stores all cognitive loop thoughts in JSONL format for:
- Real-time streaming to frontend
- Historical analysis
- Learning from past reasoning
"""

import os
import json
import asyncio
import aiofiles
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List, AsyncGenerator, Dict, Any
from pathlib import Path
from enum import Enum
import uuid


class ThoughtPhase(str, Enum):
    """OODA loop phases."""
    OBSERVE = "observe"
    ORIENT = "orient"
    DECIDE = "decide"
    ACT = "act"
    REFLECT = "reflect"
    IDLE = "idle"


class ThoughtType(str, Enum):
    """Types of thoughts."""
    OBSERVATION = "observation"
    ANALYSIS = "analysis"
    HYPOTHESIS = "hypothesis"
    DECISION = "decision"
    ACTION = "action"
    REFLECTION = "reflection"
    LEARNING = "learning"
    QUESTION = "question"
    INSIGHT = "insight"


@dataclass
class Thought:
    """A single thought in the cognitive loop."""
    id: str
    timestamp: str
    phase: str
    type: str
    content: str
    reasoning: Optional[str] = None
    confidence: float = 0.5
    session_id: Optional[str] = None  # Links thoughts to a session
    is_session_marker: bool = False  # True for session start/end markers
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        phase: ThoughtPhase,
        type: ThoughtType,
        content: str,
        reasoning: Optional[str] = None,
        confidence: float = 0.5,
        session_id: Optional[str] = None,
        is_session_marker: bool = False,
        **metadata
    ) -> 'Thought':
        """Create a new thought with auto-generated ID and timestamp."""
        return cls(
            id=f"thought_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            phase=phase.value if isinstance(phase, ThoughtPhase) else phase,
            type=type.value if isinstance(type, ThoughtType) else type,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            session_id=session_id,
            is_session_marker=is_session_marker,
            metadata=metadata
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Thought':
        """Create from dictionary."""
        return cls(**data)


class ThoughtJournal:
    """
    Persistent storage for cognitive loop thoughts.

    Uses JSONL (JSON Lines) format for efficient append-only writing
    and streaming reads.
    """

    DEFAULT_PATH = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "cognitive_loop"

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or self.DEFAULT_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.journal_file = self.storage_path / "thought_journal.jsonl"
        self.action_log = self.storage_path / "action_log.jsonl"
        self._lock = asyncio.Lock()

    async def log_thought(self, thought: Thought) -> str:
        """
        Log a thought to the journal.

        Args:
            thought: The thought to log

        Returns:
            The thought ID
        """
        async with self._lock:
            async with aiofiles.open(self.journal_file, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(thought.to_dict()) + '\n')
        return thought.id

    async def log_action(self, action: dict) -> str:
        """
        Log an action to the action log.

        Args:
            action: Action dictionary with type, target, result, etc.

        Returns:
            Action ID
        """
        if 'id' not in action:
            action['id'] = f"action_{uuid.uuid4().hex[:12]}"
        if 'timestamp' not in action:
            action['timestamp'] = datetime.now(timezone.utc).isoformat()

        async with self._lock:
            async with aiofiles.open(self.action_log, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(action) + '\n')
        return action['id']

    async def get_recent(self, limit: int = 50) -> List[Thought]:
        """
        Get most recent thoughts.

        Args:
            limit: Maximum number of thoughts to return

        Returns:
            List of recent thoughts, newest first
        """
        thoughts = []
        if not self.journal_file.exists():
            return thoughts

        # Read all lines (for small files) or tail for large files
        async with aiofiles.open(self.journal_file, 'r', encoding='utf-8') as f:
            lines = await f.readlines()

        # Parse most recent lines
        for line in reversed(lines[-limit:]):
            try:
                data = json.loads(line.strip())
                thoughts.append(Thought.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue

        return thoughts

    async def get_by_phase(self, phase: ThoughtPhase, limit: int = 20) -> List[Thought]:
        """Get recent thoughts from a specific phase."""
        all_thoughts = await self.get_recent(limit * 5)  # Get more to filter
        return [t for t in all_thoughts if t.phase == phase.value][:limit]

    async def get_recent_actions(self, limit: int = 20) -> List[dict]:
        """Get recent actions from action log."""
        actions = []
        if not self.action_log.exists():
            return actions

        async with aiofiles.open(self.action_log, 'r', encoding='utf-8') as f:
            lines = await f.readlines()

        for line in reversed(lines[-limit:]):
            try:
                actions.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

        return actions

    async def stream_thoughts(
        self,
        since: Optional[datetime] = None
    ) -> AsyncGenerator[Thought, None]:
        """
        Stream thoughts, optionally since a specific time.

        Useful for catching up on missed thoughts.
        """
        if not self.journal_file.exists():
            return

        async with aiofiles.open(self.journal_file, 'r', encoding='utf-8') as f:
            async for line in f:
                try:
                    data = json.loads(line.strip())
                    thought = Thought.from_dict(data)

                    if since:
                        thought_time = datetime.fromisoformat(
                            thought.timestamp.replace('Z', '+00:00')
                        )
                        if thought_time <= since:
                            continue

                    yield thought
                except (json.JSONDecodeError, KeyError):
                    continue

    async def get_stats(self) -> dict:
        """Get journal statistics."""
        thought_count = 0
        action_count = 0
        phases = {}
        types = {}

        if self.journal_file.exists():
            async with aiofiles.open(self.journal_file, 'r', encoding='utf-8') as f:
                async for line in f:
                    try:
                        data = json.loads(line.strip())
                        thought_count += 1
                        phase = data.get('phase', 'unknown')
                        ttype = data.get('type', 'unknown')
                        phases[phase] = phases.get(phase, 0) + 1
                        types[ttype] = types.get(ttype, 0) + 1
                    except json.JSONDecodeError:
                        continue

        if self.action_log.exists():
            async with aiofiles.open(self.action_log, 'r', encoding='utf-8') as f:
                async for line in f:
                    action_count += 1

        return {
            "total_thoughts": thought_count,
            "total_actions": action_count,
            "by_phase": phases,
            "by_type": types,
            "journal_file": str(self.journal_file),
            "action_log": str(self.action_log)
        }

    async def search(self, query: str, limit: int = 20) -> List[Thought]:
        """
        Simple keyword search in thoughts.

        For semantic search, use the AI Memory MCP tools.
        """
        query_lower = query.lower()
        matches = []

        if not self.journal_file.exists():
            return matches

        async with aiofiles.open(self.journal_file, 'r', encoding='utf-8') as f:
            async for line in f:
                try:
                    data = json.loads(line.strip())
                    content = data.get('content', '').lower()
                    reasoning = (data.get('reasoning') or '').lower()

                    if query_lower in content or query_lower in reasoning:
                        matches.append(Thought.from_dict(data))
                        if len(matches) >= limit:
                            break
                except json.JSONDecodeError:
                    continue

        return matches

    async def clear(self, before: Optional[datetime] = None):
        """
        Clear old thoughts.

        Args:
            before: Clear thoughts before this time. If None, clears all.
        """
        if before is None:
            # Clear all
            if self.journal_file.exists():
                self.journal_file.unlink()
            if self.action_log.exists():
                self.action_log.unlink()
            return

        # Filter and rewrite
        kept_thoughts = []
        if self.journal_file.exists():
            async with aiofiles.open(self.journal_file, 'r', encoding='utf-8') as f:
                async for line in f:
                    try:
                        data = json.loads(line.strip())
                        thought_time = datetime.fromisoformat(
                            data['timestamp'].replace('Z', '+00:00')
                        )
                        if thought_time >= before:
                            kept_thoughts.append(line)
                    except (json.JSONDecodeError, KeyError):
                        continue

            async with aiofiles.open(self.journal_file, 'w', encoding='utf-8') as f:
                for line in kept_thoughts:
                    await f.write(line)
