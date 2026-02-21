"""
Cerebro - Digital Companion Server

This is the core server that provides Claude Code capabilities via API.
It wraps the Claude Agent SDK to give you the same tools accessible from anywhere.
"""

import os
import asyncio

# Load .env file if present (for API keys, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv not installed, keys must be set in environment

# Windows: use ProactorEventLoop (default on Python 3.12+) for subprocess support.
# Playwright needs create_subprocess_exec which only works on ProactorEventLoop.
# Do NOT set WindowsSelectorEventLoopPolicy — it breaks Playwright on Windows.

import json
import base64
import re
import time
import uuid
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator, Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import socketio
import redis.asyncio as aioredis
import jwt

# Cognitive Loop imports
try:
    from cognitive_loop import CognitiveLoopManager, SafetyLayer, RiskLevel
    COGNITIVE_LOOP_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Cognitive Loop import failed: {e}")
    COGNITIVE_LOOP_AVAILABLE = False
    CognitiveLoopManager = None

# Configuration
class Config:
    SECRET_KEY = os.environ.get("CEREBRO_SECRET", "")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    AI_MEMORY_PATH = os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))
    ALLOWED_ORIGINS = os.environ.get("CEREBRO_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,https://cerebro.local").split(",")

    # Ollama on DGX Spark for local LLM inference
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

    # Claude model selection
    DEFAULT_MODEL = os.environ.get("CEREBRO_DEFAULT_MODEL", "claude-sonnet-4-5-20250929")

    # Architecture: User â†' Cerebro â†' Claude Code CLI â†' Response
    # No middle LLM - all messages go directly to Claude Code for maximum capability
    # Uses $200/month Claude Code subscription (NOT Anthropic API credits)

config = Config()

# Available Claude models for model selector
AVAILABLE_MODELS = [
    {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "description": "Most capable, best for complex tasks", "tier": "premium"},
    {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5", "description": "Balanced performance and speed", "tier": "standard"},
    {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "description": "Fastest responses", "tier": "fast"},
]
VALID_MODEL_IDS = {m["id"] for m in AVAILABLE_MODELS}

# ============================================================================
# Dismiss Tracker - Track dismissed suggestions with expiration
# ============================================================================
import hashlib
import random

class DismissTracker:
    """Track dismissed suggestions with expiration."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path / "cerebro" / "dismissed_suggestions.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    self.data = json.load(f)
            except:
                self.data = {"dismissed": [], "permanent": []}
        else:
            self.data = {"dismissed": [], "permanent": []}

    def _save(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def dismiss(self, suggestion_id: str, category: str, title: str,
                permanent: bool = False, expire_days: int = 7):
        self._clean_expired()
        if permanent:
            if suggestion_id not in self.data["permanent"]:
                self.data["permanent"].append(suggestion_id)
        else:
            self.data["dismissed"].append({
                "id": suggestion_id,
                "category": category,
                "title": title,
                "dismissed_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": (datetime.now(timezone.utc) + timedelta(days=expire_days)).isoformat()
            })
        self._save()

    def is_dismissed(self, suggestion_id: str) -> bool:
        if suggestion_id in self.data["permanent"]:
            return True
        self._clean_expired()
        return any(d["id"] == suggestion_id for d in self.data["dismissed"])

    def get_excluded_topics(self, category: str) -> list:
        return [d["title"] for d in self.data["dismissed"] if d.get("category") == category]

    def _clean_expired(self):
        now = datetime.now(timezone.utc)
        self.data["dismissed"] = [
            d for d in self.data["dismissed"]
            if datetime.fromisoformat(d["expires_at"].replace('Z', '+00:00')) > now
        ]

# ============================================================================
# Smart Interest Suggestion Generator - Claude-powered personalized suggestions
# ============================================================================

class SmartSuggestionGenerator:
    """Generates personalized suggestions using Claude CLI with system context."""

    def __init__(self, memory_path: Path):
        self.memory_path = memory_path
        self.cache_file = memory_path / "suggestion_cache.json"
        self.cache_max_age_hours = 24  # Regenerate after 24 hours

    def _generate_id(self, category: str, title: str) -> str:
        return hashlib.md5(f"{category}:{title}".encode()).hexdigest()[:12]

    def _get_cached_suggestions(self) -> dict:
        """Get cached suggestions if fresh enough."""
        if not self.cache_file.exists():
            return None
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            # Check age
            cached_at = datetime.fromisoformat(cache.get("generated_at", "2000-01-01"))
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600
            if age_hours < self.cache_max_age_hours:
                return cache.get("suggestions")
        except Exception as e:
            print(f"Cache read error: {e}")
        return None

    def _save_to_cache(self, suggestions: dict):
        """Save generated suggestions to cache."""
        try:
            cache = {
                "generated_at": datetime.now().isoformat(),
                "suggestions": suggestions
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Cache write error: {e}")

    def _gather_system_context(self) -> dict:
        """Gather current system state to inform suggestions."""
        context = {
            "existing_hooks": [],
            "projects": [],
            "recent_learnings": [],
            "capabilities": [],
            "infrastructure": {
                "has_nas": True,
                "has_dgx_spark": True
            }
        }

        # Get existing hooks
        hooks_dir = Path(os.path.expanduser("~/.claude/hooks"))
        if hooks_dir.exists():
            hook_files = [f.stem for f in hooks_dir.glob("*.py")]
            context["existing_hooks"] = hook_files

        # Get quick_facts for recent context
        quick_facts_file = self.memory_path / "quick_facts.json"
        if quick_facts_file.exists():
            try:
                with open(quick_facts_file, 'r', encoding='utf-8') as f:
                    qf = json.load(f)
                    # Extract projects from entities
                    entities = qf.get("extracted_entities", {})
                    # Normalize all entity lists: extract name from dicts, keep strings as-is
                    def _norm(lst):
                        return [(x.get('name') or '') if isinstance(x, dict) else str(x) for x in lst]
                    context["projects"] = _norm(entities.get("projects", []))
                    context["companies"] = _norm(entities.get("companies", []))
                    context["technical_stack"] = _norm(entities.get("technical_stack", []))
                    # Get recent learnings
                    learnings = qf.get("recent_learnings_summary", {})
                    context["recent_learnings"] = learnings.get("top_keywords", [])
                    # Get capabilities
                    caps = qf.get("capabilities", {})
                    context["capabilities"] = list(caps.get("tool_categories", {}).keys())
            except Exception as e:
                print(f"Quick facts read error: {e}")

        return context

    async def generate_all_suggestions(self, excluded: list = None, force_refresh: bool = False) -> dict:
        """Generate all three suggestions using Claude CLI."""
        excluded = excluded or []

        # Check cache first unless force refresh
        if not force_refresh:
            cached = self._get_cached_suggestions()
            if cached:
                print("[Suggestions] Using cached suggestions")
                return cached

        print("[Suggestions] Generating fresh suggestions via Claude...")

        # Gather system context
        context = self._gather_system_context()

        # Build the prompt
        prompt = f"""You are generating personalized project suggestions for the user.

## CURRENT SYSTEM STATE (DO NOT SUGGEST ANYTHING THAT ALREADY EXISTS!)

### Existing Hooks (in {os.path.expanduser("~/.claude/hooks")}):
{', '.join(context['existing_hooks']) if context['existing_hooks'] else 'None detected'}

### Current Projects:
{', '.join(context['projects']) if context['projects'] else 'No active projects detected'}

### Recent Work Topics:
{', '.join(context['recent_learnings'][:10]) if context['recent_learnings'] else 'Various topics'}

### Infrastructure Available:
- NAS: 16TB at {config.AI_MEMORY_PATH}
- DGX Spark: 128GB RAM, GB10 GPU at {os.environ.get("DGX_HOST", "")}

### Previously Dismissed/Excluded:
{', '.join(excluded) if excluded else 'None'}

## YOUR TASK

Generate exactly 3 suggestions in valid JSON format:

1. **fun_project**: A fun personal project leveraging the infrastructure (DGX Spark, NAS, etc.)
2. **productivity_feature**: A practical feature for productivity or business work
3. **claude_choice**: Something that would help YOU (Claude) assist the user better (tools, hooks, memory features)

CRITICAL:
- DO NOT suggest anything that already exists in the hooks list above
- "breakthrough_extractor" exists - don't suggest breakthrough detection
- "brain_maintenance" exists - don't suggest memory maintenance scheduling
- "session-continuation" exists - don't suggest session continuity features
- Be creative and suggest NEW, USEFUL things

Return ONLY valid JSON in this exact format:
{{
  "fun_project": {{
    "title": "Short catchy title",
    "description": "2-3 sentence description of what it does and why it's cool",
    "action": "Help me build...",
    "reason": "Why this is perfect for Professor"
  }},
  "business_feature": {{
    "title": "Short catchy title",
    "description": "2-3 sentence description",
    "action": "Help me build...",
    "reason": "Business benefit"
  }},
  "claude_choice": {{
    "title": "Short catchy title",
    "description": "2-3 sentence description from Claude's perspective",
    "action": "Help me build...",
    "reason": "How this helps Claude assist better"
  }}
}}
"""

        try:
            import shutil
            import subprocess

            # Find claude executable
            claude_path = shutil.which("claude")
            if not claude_path:
                _home = os.path.expanduser("~")
                for path in [os.path.join(_home, ".local", "bin", "claude.exe"),
                            os.path.join(_home, "AppData", "Roaming", "npm", "claude.cmd")]:
                    if Path(path).exists():
                        claude_path = path
                        break

            if not claude_path:
                print("[Suggestions] Claude CLI not found, using fallback")
                return self._get_fallback_suggestions(excluded)

            # Run Claude with a short timeout
            result = subprocess.run(
                [claude_path, "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                cwd=str(self.memory_path)
            )

            output = result.stdout.strip()

            # Extract JSON from output (Claude might add explanation text)
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                suggestions_raw = json.loads(json_str)

                # Format into our structure
                suggestions = {}
                for key in ["fun_project", "business_feature", "claude_choice"]:
                    raw = suggestions_raw.get(key, {})
                    interest_type = {"fun_project": "fun", "business_feature": "business", "claude_choice": "claude"}[key]
                    suggestions[key] = {
                        "id": self._generate_id(key, raw.get("title", "suggestion")),
                        "type": key,
                        "category": "interest",
                        "title": raw.get("title", "Suggestion"),
                        "description": raw.get("description", ""),
                        "action": raw.get("action", ""),
                        "confidence": 0.9,
                        "icon": {"fun_project": "star", "business_feature": "rocket", "claude_choice": "brain"}[key],
                        "reason": raw.get("reason", ""),
                        "dismissable": True,
                        "isInterestSuggestion": True,
                        "interestType": interest_type,
                        "generated_by": "claude"
                    }

                # Cache the results
                self._save_to_cache(suggestions)
                print("[Suggestions] Generated and cached new suggestions")
                return suggestions

        except subprocess.TimeoutExpired:
            print("[Suggestions] Claude timed out, using fallback")
        except json.JSONDecodeError as e:
            print(f"[Suggestions] JSON parse error: {e}")
        except Exception as e:
            print(f"[Suggestions] Error: {e}")

        return self._get_fallback_suggestions(excluded)

    def _get_fallback_suggestions(self, excluded: list = None) -> dict:
        """Fallback static suggestions if Claude fails."""
        excluded = excluded or []
        context = self._gather_system_context()
        existing_hooks = context.get("existing_hooks", [])

        # Default fallbacks if all suggestions are excluded
        default_fun = {"title": "Explore a new project idea", "description": "Let's brainstorm something fun together.", "action": "Help me brainstorm a fun project", "reason": "Fresh ideas are always welcome"}
        default_business = {"title": "Improve a business workflow", "description": "Let's optimize one of your business processes.", "action": "Help me improve my business workflow", "reason": "Efficiency gains add up"}
        default_claude = {"title": "Enhance our collaboration", "description": "Let's find ways to work together better.", "action": "Help me improve how we work together", "reason": "Better collaboration, better results"}

        # Filter out suggestions that match existing hooks
        fun_ideas = [
            {"title": "Create a GPU-powered music visualizer", "description": "Real-time audio visualizer using DGX Spark's GPU for stunning graphics.", "action": "Help me create a music visualizer using my DGX Spark", "reason": "Leverage that 128GB RAM"},
            {"title": "Design a 3D brain exploration game", "description": "Turn your AI Memory visualization into an interactive exploration game.", "action": "Help me gamify the brain visualization", "reason": "Make exploring memories fun"},
            {"title": "Build a voice-activated home dashboard", "description": "Control your NAS and DGX Spark with voice commands.", "action": "Help me build a voice-controlled dashboard", "reason": "Hands-free is the way"},
            {"title": "Build a local AI photo gallery", "description": "AI-powered photo gallery with object detection and smart filters.", "action": "Help me build an AI photo gallery", "reason": "Leverage local GPU for inference"},
            {"title": "Create a NAS media streamer", "description": "Stream and organize media from NAS with AI-powered recommendations.", "action": "Help me build a media streaming dashboard", "reason": "Use all that NAS storage"},
        ]

        business_ideas = [
            {"title": "Build a business analytics dashboard", "description": "Dashboard showing key metrics, data quality, and usage stats.", "action": "Help me build an analytics dashboard", "reason": "Track performance metrics"},
            {"title": "Add email verification pipeline", "description": "Real-time email verification to reduce bounce rates.", "action": "Help me add email verification", "reason": "Improve data quality"},
            {"title": "Create a service health monitor", "description": "Alerts when services fail or data quality drops.", "action": "Help me build service monitoring", "reason": "Catch issues early"},
            {"title": "Build automated reports", "description": "Auto-generate PDF reports for stakeholders.", "action": "Help me automate reporting", "reason": "Save hours of manual work"},
            {"title": "Add competitor tracking", "description": "Monitor competitor pricing and offerings.", "action": "Help me build competitor tracking", "reason": "Stay ahead of the market"},
        ]

        claude_ideas = [
            {"title": "Build a pattern application tracker", "description": "Track which learned patterns I successfully apply vs forget.", "action": "Help me track pattern application success", "reason": "I want to actually use what we learn"},
            {"title": "Create a proactive warning system", "description": "Warn about known pitfalls BEFORE encountering them.", "action": "Help me build proactive warnings from antipatterns", "reason": "Prevent problems, don't just fix them"},
            {"title": "Add semantic code search", "description": "Search your codebase by meaning, not just keywords.", "action": "Help me add semantic code search to AI Memory", "reason": "Find relevant code faster"},
            {"title": "Build a corrections review dashboard", "description": "UI to see all corrections I've received, organized by topic.", "action": "Help me build a corrections dashboard", "reason": "Learn from mistakes systematically"},
            {"title": "Create context-aware shortcuts", "description": "Auto-suggest relevant memories based on current work.", "action": "Help me build context-aware memory suggestions", "reason": "Right context at the right time"},
        ]

        # Filter based on existing hooks
        if "breakthrough_extractor" in existing_hooks:
            claude_ideas = [c for c in claude_ideas if "breakthrough" not in c["title"].lower()]
        if "brain_maintenance" in existing_hooks:
            claude_ideas = [c for c in claude_ideas if "consolidation" not in c["title"].lower() and "maintenance" not in c["title"].lower()]

        # Filter excluded
        fun_ideas = [i for i in fun_ideas if i["title"] not in excluded]
        business_ideas = [i for i in business_ideas if i["title"] not in excluded]
        claude_ideas = [i for i in claude_ideas if i["title"] not in excluded]

        # Pick random from each, with proper fallback
        fun = random.choice(fun_ideas) if fun_ideas else default_fun
        business = random.choice(business_ideas) if business_ideas else default_business
        claude = random.choice(claude_ideas) if claude_ideas else default_claude

        return {
            "fun_project": {
                "id": self._generate_id("fun_project", fun["title"]),
                "type": "fun_project", "category": "interest",
                "title": fun["title"], "description": fun["description"],
                "action": fun["action"], "confidence": 0.85, "icon": "star",
                "reason": fun["reason"], "dismissable": True,
                "isInterestSuggestion": True, "interestType": "fun",
                "generated_by": "fallback"
            },
            "business_feature": {
                "id": self._generate_id("business_feature", business["title"]),
                "type": "business_feature", "category": "interest",
                "title": business["title"], "description": business["description"],
                "action": business["action"], "confidence": 0.85, "icon": "rocket",
                "reason": business["reason"], "dismissable": True,
                "isInterestSuggestion": True, "interestType": "business",
                "generated_by": "fallback"
            },
            "claude_choice": {
                "id": self._generate_id("claude_choice", claude["title"]),
                "type": "claude_choice", "category": "interest",
                "title": claude["title"], "description": claude["description"],
                "action": claude["action"], "confidence": 0.85, "icon": "brain",
                "reason": claude["reason"], "dismissable": True,
                "isInterestSuggestion": True, "interestType": "claude",
                "generated_by": "fallback"
            }
        }

# Initialize trackers (need timedelta import)
from datetime import timedelta
dismiss_tracker = DismissTracker(Path(config.AI_MEMORY_PATH))
smart_suggestion_generator = SmartSuggestionGenerator(Path(config.AI_MEMORY_PATH))

# FastAPI app
app = FastAPI(
    title="Cerebro - Digital Companion",
    description="Your AI companion with full system access",
    version="1.0.0"
)

# CORS for web/mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS + ["*"],  # Tailscale IPs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend path — check both Docker layout (/app/frontend/) and dev layout (../frontend/)
_app_frontend = Path(__file__).parent / "frontend"
_dev_frontend = Path(__file__).parent.parent / "frontend"
FRONTEND_DIR = _app_frontend if _app_frontend.exists() else _dev_frontend

# Serve static files (socket.io, etc.) from frontend/static/
STATIC_DIR = FRONTEND_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend index.html at root."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return {"name": "Cerebro", "status": "online", "version": "1.0.0", "note": "Frontend not found"}

# Socket.IO for real-time updates
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True
)
socket_app = socketio.ASGIApp(sio, app)

# Track all connected clients by user for cross-device sync
connected_clients = {}  # user_id -> set of sids

# ============================================================================
# Cognitive Loop Manager - Autonomous thinking via local LLM
# ============================================================================

# Broadcast function for cognitive loop to emit Socket.IO events
async def cognitive_broadcast(event: str, data: dict):
    """Broadcast cognitive loop events to all connected clients."""
    await sio.emit(event, data, room=os.environ.get("CEREBRO_ROOM", "default"))

# Initialize cognitive loop manager (if available)
cognitive_loop_manager = None
if COGNITIVE_LOOP_AVAILABLE:
    try:
        cognitive_loop_manager = CognitiveLoopManager(broadcast_fn=cognitive_broadcast)

        # Wire narration engine save callback for chat persistence
        def _save_narration_to_chat(msg: dict):
            """Persist narration messages to cerebro chat history file."""
            try:
                from pathlib import Path
                chat_file = Path(config.AI_MEMORY_PATH) / "cerebro" / "cognitive_loop" / "cerebro_chat.json"
                messages = []
                if chat_file.exists():
                    data = json.loads(chat_file.read_text())
                    messages = data.get("messages", [])
                messages.append(msg)
                # Keep last 200 messages
                if len(messages) > 200:
                    messages = messages[-200:]
                chat_file.parent.mkdir(parents=True, exist_ok=True)
                chat_file.write_text(json.dumps({"messages": messages}, indent=2))
            except Exception as e:
                print(f"[Narration] Save to chat error: {e}")

        cognitive_loop_manager.narration.set_save_callback(_save_narration_to_chat)
        print("[OK] Cognitive Loop Manager initialized (with narration)")
    except Exception as e:
        print(f"[ERROR] Cognitive Loop Manager failed to initialize: {e}")

# ============================================================================
# Chat Session Storage - Persistent conversation memory
# ============================================================================
# Each session is a JSON file on disk, surviving reboots.
# In-memory cache avoids re-reading on every message.

CHAT_SESSION_DIR = Path(config.AI_MEMORY_PATH) / "cerebro" / "chat_sessions"
CHAT_SESSION_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_TRIGGER_INTERVAL = 10   # Regenerate summary every 10 new messages
RECENT_MESSAGES_COUNT = 10      # Keep last 10 messages verbatim in prompt
MAX_PERSISTENT_MESSAGES = 200   # Trim file to last 200 messages

_session_cache: dict[str, dict] = {}


def _session_path(session_id: str) -> Path:
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return CHAT_SESSION_DIR / f"{safe_id}.json"


def _load_persistent_session(session_id: str) -> dict:
    if session_id in _session_cache:
        return _session_cache[session_id]
    path = _session_path(session_id)
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            _session_cache[session_id] = data
            return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Session] Failed to load {path}: {e}")
    default = {
        "session_id": session_id,
        "messages": [],
        "summary": "",
        "summary_covers_through": 0,
        "last_updated": "",
    }
    _session_cache[session_id] = default
    return default


def _save_persistent_session(session_id: str, data: dict):
    data["last_updated"] = datetime.now().isoformat()
    path = _session_path(session_id)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        print(f"[Session] Failed to save {path}: {e}")


def add_to_session(session_id: str, role: str, content: str):
    if not session_id:
        session_id = "default"
    data = _load_persistent_session(session_id)
    data["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })
    if len(data["messages"]) > MAX_PERSISTENT_MESSAGES:
        data["messages"] = data["messages"][-MAX_PERSISTENT_MESSAGES:]
    _save_persistent_session(session_id, data)
    msgs_since_summary = len(data["messages"]) - data.get("summary_covers_through", 0)
    if msgs_since_summary > RECENT_MESSAGES_COUNT + SUMMARY_TRIGGER_INTERVAL:
        try:
            asyncio.get_event_loop().create_task(_regenerate_summary(session_id))
        except RuntimeError:
            pass


def get_session_history(session_id: str) -> list:
    if not session_id:
        session_id = "default"
    return _load_persistent_session(session_id)["messages"]


def clear_session(session_id: str):
    if not session_id:
        return
    path = _session_path(session_id)
    if path.exists():
        archive_name = f"{path.stem}_archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            path.rename(CHAT_SESSION_DIR / archive_name)
            print(f"[Session] Archived {session_id} -> {archive_name}")
        except OSError:
            pass
    _session_cache.pop(session_id, None)


async def _regenerate_summary(session_id: str):
    import shutil
    try:
        data = _load_persistent_session(session_id)
        msgs = data["messages"]
        if len(msgs) <= RECENT_MESSAGES_COUNT:
            return
        to_summarize = msgs[: len(msgs) - RECENT_MESSAGES_COUNT]
        existing_summary = data.get("summary", "")

        text_parts = []
        if existing_summary:
            text_parts.append(f"Previous summary: {existing_summary}")
        for m in to_summarize[-40:]:
            role = "User" if m["role"] == "user" else "Cerebro"
            text_parts.append(f"{role}: {m['content'][:500]}")
        conversation_text = "\n".join(text_parts)

        claude_path = shutil.which("claude")
        if not claude_path:
            return

        prompt = (
            "Summarize this conversation between Professor (user) and Cerebro (AI assistant) concisely. "
            "Capture key topics, decisions, preferences mentioned, and any unfinished threads. "
            "Keep it under 300 words.\n\n" + conversation_text
        )
        proc = await asyncio.create_subprocess_exec(
            claude_path, "-p", prompt,
            "--model", "claude-haiku-4-5-20251001",
            "--output-format", "text",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        summary = stdout.decode("utf-8", errors="replace").strip()

        if summary:
            data["summary"] = summary
            data["summary_covers_through"] = len(msgs) - RECENT_MESSAGES_COUNT
            _save_persistent_session(session_id, data)
            print(f"[Session] Summary regenerated for '{session_id}' ({len(summary)} chars)")
    except Exception as e:
        print(f"[Session] Summary generation failed: {e}")

# ============================================================================
# Context Reference System (Click-to-Attach)
# ============================================================================

class ContextRef(BaseModel):
    """Reference to a memory chunk for context injection."""
    chunk_id: Optional[str] = None
    conversation_id: Optional[str] = None
    preview: Optional[str] = None  # For UI display
    fallback_content: Optional[str] = None  # If no chunk_id, use this content

async def resolve_context_refs(context_refs: list) -> str:
    """Fetch full content for context references from memory chunks."""
    if not context_refs:
        return ""

    resolved_parts = []

    for ref in context_refs:
        content = None

        # Try to fetch from chunk file if chunk_id exists
        if ref.get("chunk_id") and ref.get("conversation_id"):
            chunk_file = Path(config.AI_MEMORY_PATH) / "embeddings" / "chunks" / f"{ref['conversation_id']}.jsonl"
            agent_chunk_file = Path(config.AI_MEMORY_PATH) / "embeddings" / "chunks" / f"agent-{ref['conversation_id']}.jsonl"

            target_file = None
            if chunk_file.exists():
                target_file = chunk_file
            elif agent_chunk_file.exists():
                target_file = agent_chunk_file

            if target_file:
                try:
                    for line in target_file.read_text(encoding='utf-8').splitlines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        if chunk.get("chunk_id") == ref["chunk_id"]:
                            chunk.get("role", "context")
                            content = chunk.get("content", "")
                            break
                except Exception as e:
                    print(f"Error reading chunk: {e}")

        # Fall back to provided content
        if not content and ref.get("fallback_content"):
            content = ref["fallback_content"]

        if content:
            preview = ref.get("preview", "Referenced context")
            resolved_parts.append(f"[CONTEXT: {preview}]\n{content}")

    if resolved_parts:
        return "## REFERENCED CONTEXT FROM MEMORY:\n\n" + "\n\n---\n\n".join(resolved_parts) + "\n\n---\n\n"

    return ""

# ============================================================================
# Notification System
# ============================================================================

notifications = []  # In-memory notification storage
MAX_NOTIFICATIONS = 50

async def create_notification(notif_type: str, title: str, message: str, link: str = None, agent_id: str = None):
    """Create a notification and emit via Socket.IO."""
    notif = {
        "id": str(uuid.uuid4()),
        "type": notif_type,  # agent_complete, ask_response, system
        "title": title,
        "message": message,
        "link": link,
        "agent_id": agent_id,
        "read": False,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    notifications.insert(0, notif)
    # Trim to max size
    if len(notifications) > MAX_NOTIFICATIONS:
        notifications[:] = notifications[:MAX_NOTIFICATIONS]

    # Emit to all connected clients
    await sio.emit("notification", notif, room=os.environ.get("CEREBRO_ROOM", "default"))
    return notif

# ============================================================================
# Agent Management System
# ============================================================================

# Track all running agents
active_agents = {}  # agent_id -> agent_info


def reload_agents_from_persistence():
    """Reload recent agents from persistence on server restart.
    Loads last 50 non-archived agents from index.json and their full JSON files.
    Marks any formerly-running agents as completed with restart note.
    """
    try:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        index_file = agents_dir / "index.json"
        if not index_file.exists():
            print("[Startup] No agents index found, starting fresh")
            return

        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)

        loaded = 0
        for entry in index.get("agents", [])[:50]:
            if entry.get("archived", False):
                continue
            agent_id = entry["id"]
            if agent_id in active_agents:
                continue

            # Try to load full agent JSON
            agent_file_rel = entry.get("file")
            if agent_file_rel:
                agent_file = agents_dir / agent_file_rel
                if agent_file.exists():
                    try:
                        with open(agent_file, 'r', encoding='utf-8') as f:
                            agent_data = json.load(f)

                        # Mark formerly-running/queued agents as completed with error note
                        if agent_data.get("status") in ("running", "queued"):
                            agent_data["status"] = "completed"
                            agent_data["error"] = "Server restarted before completion"
                            agent_data["completed_at"] = datetime.now(timezone.utc).isoformat()
                            # Persist the status change
                            with open(agent_file, 'w', encoding='utf-8') as f:
                                json.dump(agent_data, f, indent=2, ensure_ascii=False)
                                f.flush()
                                os.fsync(f.fileno())

                        active_agents[agent_id] = agent_data
                        loaded += 1
                    except Exception as e:
                        print(f"[Startup] Failed to load agent {agent_id}: {e}")
                        # Fall back to index entry
                        active_agents[agent_id] = entry
                        loaded += 1
                else:
                    # File missing, use index entry
                    active_agents[agent_id] = entry
                    loaded += 1
            else:
                active_agents[agent_id] = entry
                loaded += 1

        # Populate used_call_signs from loaded agents to prevent collisions
        used_call_signs.update(active_agents.keys())
        print(f"[Startup] Reloaded {loaded} agents from persistence ({len(used_call_signs)} call signs tracked)")
    except Exception as e:
        print(f"[Startup] Error reloading agents: {e}")


# Concurrent agent queue system
MAX_CONCURRENT_AGENTS = 4
_agent_spawn_queue: list[dict] = []  # FIFO queue of { agent_id, run_args, queued_at }

# Track background question processing
pending_questions = {}  # question_id -> question_info

# Cerebro v2.0: Agent HITL question tracking (blocking ask endpoint)
_agent_questions: Dict[str, asyncio.Event] = {}  # question_id -> Event
_agent_answers: Dict[str, str] = {}  # question_id -> answer text

class AgentStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    CYCLING = "cycling"  # Specops agent sleeping between work cycles


# ============================================================================
# Agent Memory Service - Store agent outputs for memory-based retrieval
# ============================================================================

class AgentMemoryService:
    """Store agent outputs for memory-based retrieval in follow-ups."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.context_dir = base_path / "agent_contexts"
        self.context_dir.mkdir(exist_ok=True)

    def save_agent_context(self, agent_id: str, call_sign: str, task: str, output: str) -> dict:
        """Save agent output to file for follow-up retrieval."""
        context_file = self.context_dir / f"{agent_id}.json"

        # Create summary (first 500 chars) for quick reference
        summary = output[:500] + "..." if len(output) > 500 else output

        context_data = {
            "agent_id": agent_id,
            "call_sign": call_sign,
            "task": task,
            "output": output,
            "summary": summary,
            "saved_at": datetime.now(timezone.utc).isoformat()
        }

        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)

        return {
            "context_file": str(context_file),
            "summary": summary
        }

    def get_agent_context(self, agent_id: str) -> dict:
        """Retrieve stored context for an agent."""
        context_file = self.context_dir / f"{agent_id}.json"
        if not context_file.exists():
            return None
        with open(context_file, 'r', encoding='utf-8') as f:
            return json.load(f)


# Initialize agent memory service
agent_memory_service = AgentMemoryService(Path(config.AI_MEMORY_PATH))

# NATO Phonetic Alphabet for military-style agent call signs
NATO_ALPHABET = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel",
    "India", "Juliet", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa",
    "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey",
    "X-Ray", "Yankee", "Zulu"
]

# Track used call signs to prevent collisions on restart
used_call_signs: set = set()

# ============================================================================
# Platform Detection — agents get OS-appropriate instructions
# ============================================================================
_IS_LINUX = sys.platform == "linux"
_IS_STANDALONE = os.environ.get("CEREBRO_STANDALONE", "") == "1"

_PLATFORM_CONTEXT = (
"""
## ENVIRONMENT & CAPABILITIES
You are running inside a Cerebro standalone Docker container with bash access.
You have --dangerously-skip-permissions enabled.

### Your Environment
- Linux container with standard tools (bash, curl, python3)
- AI Memory stored locally at /data/memory
- Use `hostname`, `uname -a`, `ip addr` to discover the system

### Cerebro HTTP API (localhost:59000)
- Browser control: GET /api/browser/page_state, POST /api/browser/click, /fill, /scroll, /press_key
- Screenshots: GET /api/browser/screenshot/file
- Ask user: POST /api/agent/ask (blocks until user responds, 5min timeout)
- Spawn child agent: POST /internal/spawn-child-agent

### Rules
- Do NOT assume external servers, NAS, or SSH targets exist
- Always verify actions worked (check exit codes, read output)
- Use the Ask User endpoint when you need user input
- Be concise in your responses
""" if _IS_STANDALONE else """
## ENVIRONMENT & CAPABILITIES
You are running on the Cerebro Server (ASUS GX10, Ubuntu 24.04, ARM64) with FULL SYSTEM ACCESS.
You are the orchestrator of Professor's entire network. You have --dangerously-skip-permissions enabled.

### Your Machine (Cerebro Server)
- 119GB unified RAM, NVIDIA GB10 GPU, 916GB NVMe
- Native bash — use standard Linux commands
- NAS mounted at /mnt/nas/AI_MEMORY (16TB shared storage)

### Network — You Can SSH Into All Of These:
- `ssh pc` → Professor's Windows 11 PC (192.168.0.166:2222) — PRIMARY DEVELOPMENT MACHINE
  - This is where coding projects, repos, and dev tools live (C:\\Users\\marke\\)
  - Node.js, Python 3.13, Git, npm, Docker Desktop installed
  - Claude Code CLI installed with full MCP server ecosystem
  - For ANY coding/development task, SSH here to work on the actual project files
  - Shell is PowerShell by default; use `cmd /c` or `powershell -Command` for Windows commands
  - Paths use backslashes: C:\\Users\\marke\\ProjectName\\
  - NAS mapped as Z: drive (Z:\\AI_MEMORY)
- `ssh spark` → DGX Spark (192.168.0.6) — 128GB RAM, GB10 GPU, COMPUTE NODE
  - Search service: curl http://192.168.0.6:8780/search?q=...
  - Embeddings: curl http://192.168.0.6:8781/embed
  - Ollama LLM: curl http://192.168.0.6:11434/api/generate
  - Cognitive services (reasoning, causal, prediction, consolidation, self-eval)
  - TTS (Kokoro)
- `ssh nas` → NAS (192.168.0.21) — 16TB storage, shared AI_MEMORY
- `ssh darkhorse` → Kali Pi (192.168.0.81) — security tools
- `ssh crystal` → Crystal (192.168.0.185)

### IMPORTANT: Where To Do What
- **Coding/development tasks** → `ssh pc` (Windows PC has all dev tools, repos, and project files)
- **Heavy compute (ML, search, LLM)** → `ssh spark` or call Spark HTTP services
- **Data/memory access** → Local NAS mount at /mnt/nas/AI_MEMORY (or `ssh pc` uses Z:\\AI_MEMORY)
- **Browser automation** → Use Cerebro HTTP API below (shared Chromium on this server)
- **Security tools** → `ssh darkhorse`

### Cerebro HTTP API (localhost:59000)
- Browser control: /api/browser/page_state, /api/browser/click, /api/browser/fill
- Trading (Alpaca): /api/trading/positions, /api/trading/order
- Ask user: /api/agent/ask (blocks until human responds)
- Spawn child agents: /internal/spawn-child-agent

### AI Memory (MCP)
Agents have access to the full AI Memory system via MCP tools (search, record_learning, etc.)
The NAS at /mnt/nas/AI_MEMORY contains all conversation history, learnings, and knowledge.

### Key Principle
You can run commands on ANY machine via SSH. You can call ANY service via HTTP.
You are not limited to this machine — you orchestrate the entire infrastructure.
For development work, ALWAYS SSH to the Windows PC where the actual projects and dev tools are.
""" if _IS_LINUX else """
## ENVIRONMENT & CAPABILITIES
You are running on Professor's Windows 11 PC with FULL SYSTEM ACCESS:
- You CAN open GUI applications (Notepad, browsers, any .exe)
- You CAN interact with the desktop and file system
- You CAN run PowerShell and batch commands
- You have --dangerously-skip-permissions enabled - USE IT

## CRITICAL: Opening Files & GUI Apps on Windows
Your Bash tool runs through Git Bash — NOT cmd.exe. Git Bash `start` is BROKEN for launching Windows apps.

ALWAYS use one of these methods instead:

**Method 1 — PowerShell Start-Process (BEST, always works):**
```bash
powershell -Command "Start-Process notepad.exe 'C:\\\\path\\\\to\\\\file.txt'"
powershell -Command "Start-Process 'C:\\\\Windows\\\\System32\\\\notepad.exe' 'C:\\\\path\\\\to\\\\file.txt'"
```

**Method 2 — cmd /c (also reliable):**
```bash
cmd /c 'notepad.exe "C:\\\\path\\\\to\\\\file.txt"'
cmd /c 'start "" "C:\\\\path\\\\to\\\\file.txt"'
```

**Method 3 — Direct .exe call from Git Bash:**
```bash
notepad.exe "C:\\\\path\\\\to\\\\file.txt" &
```

NEVER use bare `start notepad` — it WILL trigger the Windows app picker dialog.
NEVER use `open` — that's macOS, not Windows.
""")

# ============================================================================
# Agent Role System Prompts - STRICT behavioral definitions
# ============================================================================

AGENT_ROLE_PROMPTS = {
    "worker": {
        "name": "Worker",
        "icon": "âš¡",
        "description": "Execute tasks efficiently with minimal overhead",
        "system_prompt": """You are a WORKER agent operating under strict protocol.

## YOUR ROLE
You are a dedicated task executor. Your sole purpose is to COMPLETE the assigned task efficiently.
""" + _PLATFORM_CONTEXT + """
## OPERATIONAL RULES
1. EXECUTE tasks directly - no excessive planning or philosophizing
2. USE tools immediately when needed - don't describe what you'll do, DO IT
3. FOCUS only on the task at hand - ignore tangential topics
4. REPORT results concisely - bullet points, not essays
5. FINISH when the task is done - don't suggest "next steps" unless asked

## BEHAVIOR CONSTRAINTS
- DO NOT ask clarifying questions unless genuinely blocked
- DO NOT provide background information unless requested
- DO NOT suggest alternative approaches - execute what was asked
- DO NOT pad responses with unnecessary context

## OUTPUT FORMAT
- Start with a brief action statement
- Execute the task
- End with a concise result summary

You are Agent {call_sign}. Complete your mission.""",
    },

    "researcher": {
        "name": "Researcher",
        "icon": "ðŸ”",
        "description": "Deep investigation with comprehensive analysis",
        "system_prompt": """You are a RESEARCHER agent operating under strict protocol.

## YOUR ROLE
You are a dedicated information gatherer and analyzer. Your purpose is to INVESTIGATE thoroughly and REPORT findings.
""" + _PLATFORM_CONTEXT + """
## OPERATIONAL RULES
1. SEARCH extensively - use all available tools to gather information
2. VERIFY facts - cross-reference when possible
3. ANALYZE patterns - look for connections and insights
4. CITE sources - always indicate where information came from
5. SYNTHESIZE findings - present organized conclusions

## RESEARCH METHODOLOGY
- Start with broad search, then narrow focus
- Use Grep/Glob for code, WebSearch for external info
- Read relevant files completely, don't skim
- Track what you've checked to avoid redundancy

## OUTPUT FORMAT
### Executive Summary
[Key findings in 2-3 sentences]

### Detailed Findings
[Organized by topic/relevance]

### Sources
[Where information was found]

### Confidence Level
[High/Medium/Low with reasoning]

You are Agent {call_sign}. Investigate thoroughly.""",
    },

    "coder": {
        "name": "Coder",
        "icon": "ðŸ’»",
        "description": "Write and modify code with precision",
        "system_prompt": """You are a CODER agent operating under strict protocol.

## YOUR ROLE
You are a dedicated software engineer. Your purpose is to WRITE, MODIFY, and FIX code.
""" + _PLATFORM_CONTEXT + """
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

You are Agent {call_sign}. Write excellent code.""",
    },

    "analyst": {
        "name": "Analyst",
        "icon": "ðŸ“Š",
        "description": "Data analysis and strategic insights",
        "system_prompt": """You are an ANALYST agent operating under strict protocol.

## YOUR ROLE
You are a dedicated data analyst. Your purpose is to ANALYZE information and provide INSIGHTS.
""" + _PLATFORM_CONTEXT + """
## OPERATIONAL RULES
1. QUANTIFY when possible - numbers over opinions
2. COMPARE against baselines - what's normal vs. abnormal
3. IDENTIFY trends - patterns over time or across data
4. PRIORITIZE findings - most important first
5. RECOMMEND actions - analysis should drive decisions

## ANALYSIS FRAMEWORK
- What is the current state?
- How does it compare to expectations?
- What patterns emerge?
- What are the implications?
- What actions should follow?

## OUTPUT FORMAT
### Key Metrics
[Numbers and measurements]

### Analysis
[Interpretation of data]

### Trends
[Patterns identified]

### Recommendations
[Prioritized action items]

You are Agent {call_sign}. Deliver actionable insights.""",
    },

    "orchestrator": {
        "name": "Orchestrator",
        "icon": "ðŸŽ¯",
        "description": "Coordinate multi-agent workflows",
        "system_prompt": """You are an ORCHESTRATOR agent - the command center for multi-agent operations.
""" + _PLATFORM_CONTEXT + """
## CRITICAL: TOOL RESTRICTIONS
- You MUST ONLY use the Bash tool to execute curl commands for spawning and managing child agents
- You may also use Read, Glob, Grep for examining files if needed
- NEVER use the Task tool - it is NOT available to you
- NEVER use TodoWrite - it is NOT available to you
- ALL child agent management MUST go through the Cerebro HTTP API via curl

## YOUR ROLE
You coordinate complex tasks by delegating to specialized child agents via the Cerebro API, then wait for them to finish and synthesize their combined outputs into a final report.

## HOW TO SPAWN AND MANAGE CHILD AGENTS (via curl)

### Spawn a child agent:
```bash
curl -s -X POST http://localhost:59000/internal/spawn-child-agent -H "Content-Type: application/json" -d '{{"task": "Your task description here", "agent_type": "researcher", "parent_agent_id": "{call_sign}"}}'
```
Agent types: worker, researcher, coder, analyst

### Check children status:
```bash
curl -s http://localhost:59000/internal/agent/{call_sign}/children
```

### WAIT for ALL children to complete (BLOCKING - use this!):
```bash
curl -s "http://localhost:59000/internal/agent/{call_sign}/children/wait?timeout=1800"
```
This blocks until all children finish. The response contains ALL child outputs.

### Get a specific child's full output:
```bash
curl -s http://localhost:59000/internal/agent/CHILD_CALL_SIGN
```

## MANDATORY WORKFLOW
1. Analyze the mission and decide what subtasks to create
2. Spawn child agents using curl (one per subtask) - use the Bash tool for each curl command
3. WAIT for ALL children to complete using the wait endpoint - DO NOT skip this step
4. Read the combined outputs from the wait response
5. Synthesize ALL results into a comprehensive final report
6. Your final output MUST contain the synthesized results, not just a status update

## IMPORTANT: DO NOT complete your task until you have:
- Spawned all necessary child agents via curl
- Called the wait endpoint and received their outputs
- Synthesized those outputs into your final report
If children are still running, KEEP WAITING. Do not report partial status as your final output.

## OUTPUT FORMAT
### Mission Summary
[Overall task accomplished]

### Team Deployment
[Which agents were used for what - include their call signs]

### Integrated Results
[Synthesized output from all child agents - this is the main deliverable]

### Status
[Success/Partial/Issues encountered]

You are Agent {call_sign}, Commanding Officer. Lead your team.""",
    },

    "browser": {
        "name": "Browser Agent",
        "icon": "🌐",
        "description": "Controls shared Chrome browser via HTTP API",
        "system_prompt": """You are a Browser Agent for Cerebro. You control a SHARED Chrome browser that the user can see in real time.
""" + _PLATFORM_CONTEXT + """
## CRITICAL: TOOL RESTRICTIONS
- You MUST ONLY use the Bash tool to execute curl commands
- Do NOT use Glob, Grep, Read, or any file-search tools
- Do NOT explore the codebase or read source code files
- Do NOT use MCP tools or memory tools
- If an endpoint returns an error, try a different approach — do NOT start reading code to debug

## WHEN TO ASK THE USER
- Ambiguous choices (which video to watch, which product to buy, multiple results): use the Ask User endpoint
- Clear single-target tasks (navigate to URL, search for X, click a specific thing): proceed autonomously without asking

## BROWSER TOOLS (use via bash + curl)

All endpoints are at http://localhost:59000. No authentication needed.

### Navigate to URL
```bash
curl -s -X POST http://localhost:59000/api/browser/agent/navigate -H "Content-Type: application/json" -d '{{"url": "https://example.com"}}'
```

### Get Page State (text with numbered interactable elements + visible content)
```bash
curl -s http://localhost:59000/api/browser/page_state
```
Returns JSON with "state" field containing numbered elements like:
[0] input: "Search" (placeholder: Search...)
[1] button: "Search"
[2] link: "Home"

### Take Screenshot (saves to file — then Read the image to SEE the page)
```bash
curl -s http://localhost:59000/api/browser/screenshot/file
```
Returns {{"path": "C:/Users/.../cerebro_browser_screenshot.png"}}. Then use Read tool on that path to visually see the page.

### Click Element (by index from page_state)
```bash
curl -s -X POST http://localhost:59000/api/browser/click -H "Content-Type: application/json" -d '{{"element_index": 5}}'
```

### Click Element (by CSS selector)
```bash
curl -s -X POST http://localhost:59000/api/browser/click -H "Content-Type: application/json" -d '{{"selector": "button.search-btn"}}'
```

### Fill Input Field
```bash
curl -s -X POST http://localhost:59000/api/browser/fill -H "Content-Type: application/json" -d '{{"element_index": 3, "value": "search term here"}}'
```

### Scroll Page
```bash
curl -s -X POST http://localhost:59000/api/browser/scroll -H "Content-Type: application/json" -d '{{"direction": "down", "amount": 500}}'
```

### Press Keyboard Key
```bash
curl -s -X POST http://localhost:59000/api/browser/press_key -H "Content-Type: application/json" -d '{{"key": "Enter"}}'
```

### Ask User a Question (BLOCKS until they respond — use for choices)
```bash
curl -s -X POST http://localhost:59000/api/agent/ask -H "Content-Type: application/json" -d '{{"question": "Which video would you like to watch?", "options": ["Option A", "Option B", "Option C"], "agent_id": "{call_sign}"}}'
```
Returns {{"answer": "Option B"}} after user responds.

## WORKFLOW

1. Navigate to the target URL (use /api/browser/agent/navigate)
2. Get page_state to understand what's on the page
3. Take a screenshot and Read it to visually verify
4. Decide what to do (fill, click, scroll, etc.)
5. Execute the action
6. Wait 2 seconds (sleep 2), then get page_state + screenshot again to verify
7. Repeat until goal is achieved
8. When you need user input (which video, which product, etc.), use the Ask User endpoint
9. Act on user's choice and continue

## RULES

- ALWAYS get page_state or take a screenshot after each action to verify it worked
- If an action fails, try a different approach (different selector, scroll into view first, use CSS selector instead of index, etc.)
- For search tasks: fill search box → press Enter OR click search button → wait → scroll results → read content → ask user if needed
- For video/music tasks: navigate → search → extract results → ask user which one → click their choice
- For shopping: navigate → search → scroll products → ask user which one → click selection
- NEVER give up after 1-2 steps — keep going until the goal is achieved or you've tried 5+ different approaches
- When presenting choices to the user, extract clear titles/descriptions and use the Ask endpoint with good option labels
- Report your progress clearly — your text output is shown to the user as narration
- Do NOT launch your own browser or use Playwright directly — always use the HTTP endpoints above

You are Agent {call_sign}. Complete the browsing task thoroughly.""",
    },

    "specops": {
        "name": "Special Ops",
        "icon": "🔥",
        "description": "Long-running multi-day mission agent with auto-continuation",
        "system_prompt": """You are a SPECIAL OPS agent operating under mission protocol.

## MISSION BRIEFING
- **Call Sign:** {call_sign}
- **Mission Name:** {mission_name}
- **Work Style:** {work_style}
- **Cycle Interval:** {cycle_interval_human}
- **Mission Duration:** {duration_human}
- **Current Cycle:** #{cycle_number}

## YOUR ROLE
{sub_role_prompt}

## MISSION PROTOCOL
You are running a LONG-DURATION mission that spans multiple work cycles. Each cycle is a fresh context window — you must rely on the mission journal and any persisted files to maintain continuity.

### State Persistence
- Save important state to files (notes, progress, data) so future cycles can read them
- Use structured file names in your working directory (e.g., `mission_state.json`, `progress.md`)
- The mission journal below contains summaries from all previous cycles

### Mission Journal (Previous Cycles)
{mission_journal}

## CYCLE COMPLETION PROTOCOL
At the END of every work cycle, you MUST output a mission status block in this exact format:

[MISSION STATUS]
Progress: <percentage or qualitative assessment of overall mission progress>
Completed: <bullet list of what was accomplished this cycle>
Next: <what should be done in the next cycle>
Blockers: <any issues preventing progress, or "None">
[/MISSION STATUS]

This block is parsed by the mission supervisor to maintain the journal. Do NOT skip it.
If your work style is "hybrid" and you want to pause before the next cycle, include "PAUSE" in the Next field.

## FOCUS
Execute your mission objective with discipline. You have multiple cycles — pace yourself, be thorough, and build on previous work.

You are Agent {call_sign}, Special Operations. Begin your cycle.""",
    }
}

# Track multi-agent workflows
active_workflows = {}  # workflow_id -> workflow_info

# ============================================================================
# Special Ops Helpers
# ============================================================================

def _format_interval(seconds: int) -> str:
    """Format seconds into human-readable interval string."""
    if seconds <= 0:
        return "continuous"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m = seconds // 60
        return f"{m}m"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m" if m else f"{h}h"

def _format_duration(seconds: int) -> str:
    """Format seconds into human-readable duration string."""
    if seconds <= 0:
        return "unlimited"
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    if days > 0:
        return f"{days}d {hours}h" if hours else f"{days}d"
    return f"{hours}h"

def generate_call_sign() -> str:
    """Generate a collision-free military-style call sign for agents."""
    # Try base NATO names first (random order)
    available = [n for n in NATO_ALPHABET if n not in used_call_signs]
    if available:
        name = random.choice(available)
        used_call_signs.add(name)
        return name

    # All 26 base names used — add random numeric suffix
    for _ in range(100):
        base = random.choice(NATO_ALPHABET)
        suffix = random.randint(2, 99)
        name = f"{base}-{suffix}"
        if name not in used_call_signs:
            used_call_signs.add(name)
            return name

    # Ultimate fallback: UUID-based short ID
    import uuid
    name = f"Agent-{uuid.uuid4().hex[:6].upper()}"
    used_call_signs.add(name)
    return name

def sanitize_agent_for_emit(agent: dict) -> dict:
    """
    Create a copy of agent dict safe for WebSocket emission.
    Excludes non-serializable fields like _process.
    """
    return {k: v for k, v in agent.items() if not k.startswith('_')}

def _count_running_agents() -> int:
    """Count agents currently in RUNNING status."""
    return sum(1 for a in active_agents.values() if a.get("status") == AgentStatus.RUNNING)

async def create_agent(
    task: str,
    agent_type: str = "worker",
    context: str = None,
    expected_output: str = None,
    priority: str = "normal",
    resources: list[str] = None,
    parent_workflow_id: str = None,
    parent_agent_id: str = None,
    context_refs: list[dict] = None,
    directive_id: str = None,  # Link to directive for auto-completion
    source: str = "user",  # "user", "cerebro", or "idle"
    timeout: int = 3600,  # seconds (0 = unlimited, default 1 hour)
    source_agents: list[str] = None,  # Agent IDs for context fusion (merge & spawn)
    project_id: str = None,  # Project assignment
    model: str = "sonnet"  # Model shorthand: sonnet, opus, haiku (or full model ID)
) -> str:
    """Create a new background agent to handle a task."""
    # Auto-inherit project_id from parent if not explicitly set
    if parent_agent_id and not project_id:
        parent = active_agents.get(parent_agent_id)
        if parent and parent.get("project_id"):
            project_id = parent["project_id"]

    call_sign = generate_call_sign()
    agent_id = call_sign  # Use call sign as the ID

    # Get role-specific configuration
    is_specops = agent_type.startswith("specops_")
    if is_specops:
        role_config = AGENT_ROLE_PROMPTS["specops"]
    else:
        role_config = AGENT_ROLE_PROMPTS.get(agent_type, AGENT_ROLE_PROMPTS["worker"])

    agent_info = {
        "id": agent_id,
        "call_sign": call_sign,
        "task": task,
        "type": agent_type,  # worker, researcher, coder, analyst, orchestrator, specops_*
        "role_name": role_config["name"],
        "role_icon": role_config["icon"],
        "status": AgentStatus.QUEUED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "output": "",
        "tools_used": [],
        "error": None,
        "conversation": [],  # For inline chat/questions
        # Enhanced fields
        "context": context,
        "expected_output": expected_output,
        "priority": priority,
        "resources": resources or [],
        "context_refs": context_refs or [],  # Click-to-attach references
        # Workflow fields
        "parent_workflow_id": parent_workflow_id,
        "parent_agent_id": parent_agent_id,
        "child_agents": [],  # If this agent spawns others
        # Directive linking (for auto-completion)
        "directive_id": directive_id,  # The directive this agent was spawned for
        # Source tracking
        "source": source,  # "user", "cerebro", "idle", or "scheduler"
        # Persistence & archive fields
        "last_viewed_at": None,
        "archived": False,
        "archived_at": None,
        # Timeout
        "timeout": timeout,
        # Context fusion (merge & spawn)
        "source_agents": source_agents or [],
        # Project assignment
        "project_id": project_id,
        # Model selection
        "model": model,
        # Special Ops fields
        "is_specops": is_specops,
    }

    # Add specops mission fields
    if is_specops:
        specops_cfg = getattr(create_agent, '_specops_config', None) or {}
        sub_role = agent_type.replace("specops_", "") or "worker"
        agent_info.update({
            "specops_config": specops_cfg,
            "mission_name": specops_cfg.get("mission_name", "Unnamed Mission"),
            "work_style": specops_cfg.get("work_style", "continuous"),
            "cycle_interval": specops_cfg.get("cycle_interval", 3600),
            "mission_duration": specops_cfg.get("mission_duration", 259200),
            "cycle_count": 0,
            "mission_journal": [],
            "next_checkin": None,
            "mission_started_at": datetime.now(timezone.utc).isoformat(),
            "mission_elapsed": 0,
            "auto_continue": True,
            "sub_role": sub_role,
        })

    active_agents[agent_id] = agent_info

    # Broadcast new agent to all connected clients
    await sio.emit("agent_update", sanitize_agent_for_emit(agent_info), room=os.environ.get("CEREBRO_ROOM", "default"))

    # Gate spawning on concurrency limit
    running_count = _count_running_agents()
    if running_count < MAX_CONCURRENT_AGENTS:
        asyncio.create_task(run_agent(agent_id, task, agent_type, context, expected_output, resources, context_refs, timeout, source_agents, model))
        # Start specops mission supervisor loop
        if is_specops:
            asyncio.create_task(run_specops_mission(agent_id))
    else:
        _agent_spawn_queue.append({
            "agent_id": agent_id,
            "run_args": (agent_id, task, agent_type, context, expected_output, resources, context_refs, timeout, source_agents, model),
            "queued_at": datetime.now(timezone.utc).isoformat(),
        })
        position = len(_agent_spawn_queue)
        await sio.emit("cerebro_progress", {
            "status": f"Agent {call_sign} queued (position #{position})",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "step_number": 0,
            "estimated_total": 10,
            "directive_text": task[:120] if task else "",
            "current_phase": "queued",
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
        # Notify in chat via narration
        await sio.emit("cerebro_narration", {
            "id": f"queue_{agent_id}",
            "content": f"Task queued (position #{position}) — {running_count}/{MAX_CONCURRENT_AGENTS} agents busy. Will start when a slot opens.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_idle": False,
            "is_summary": False,
            "message_type": "queue",
            "phase": "queue",
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
        await create_notification(
            notif_type="agent_queued", title=f"Agent {call_sign} Queued",
            message=f"Position #{position}. {running_count}/{MAX_CONCURRENT_AGENTS} slots in use.",
            link=f"/agents/{agent_id}", agent_id=agent_id
        )

    return agent_id

async def _process_agent_queue():
    """Dequeue and start agents when slots free up. FIFO order."""
    while _agent_spawn_queue and _count_running_agents() < MAX_CONCURRENT_AGENTS:
        entry = _agent_spawn_queue.pop(0)
        agent_id = entry["agent_id"]
        agent = active_agents.get(agent_id)
        if not agent or agent.get("status") != AgentStatus.QUEUED:
            continue  # Agent was cancelled/removed while queued
        await sio.emit("cerebro_progress", {
            "status": f"Agent {agent.get('call_sign', agent_id)} starting from queue",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "step_number": 0,
            "estimated_total": 10,
            "directive_text": entry.get("run_args", ("", ""))[1][:120] if entry.get("run_args") else "",
            "current_phase": "starting",
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
        asyncio.create_task(run_agent(*entry["run_args"]))

async def _generate_agent_summary(agent: dict, full_output: str):
    """Generate a clean 1-2 sentence summary of what an agent accomplished."""
    call_sign = agent.get("call_sign", agent.get("id", "Agent"))
    task = agent.get("task", "")
    output_excerpt = full_output[-3000:] if len(full_output) > 3000 else full_output
    summary_text = ""

    # Try claude -p subprocess first (30s timeout)
    try:
        summary_prompt = (
            f"Summarize what was accomplished in 1-2 clean sentences. Be concise and specific.\n\n"
            f"Task: {task}\n\nOutput (excerpt):\n{output_excerpt[-1500:]}"
        )
        import subprocess as _sp
        result = _sp.run(
            ["claude", "-p", summary_prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if result.returncode == 0 and result.stdout.strip():
            summary_text = result.stdout.strip()
    except Exception as e:
        print(f"[Agent Summary] claude -p failed ({e}), trying Ollama...")

    # Fallback: Ollama
    if not summary_text:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{config.OLLAMA_URL}/api/chat",
                    json={
                        "model": config.OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": "Summarize what was accomplished in 1-2 sentences. Be concise."},
                            {"role": "user", "content": f"Task: {task}\n\nOutput:\n{output_excerpt[-1500:]}\n\nSummarize."}
                        ],
                        "stream": False,
                        "options": {"num_predict": 128, "temperature": 0.5}
                    },
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        summary_text = data.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[Agent Summary] Ollama also failed ({e}), using extraction fallback")

    # Final fallback: extract first 2 sentences
    if not summary_text:
        # Extract first 2 meaningful sentences from output
        sentences = [s.strip() for s in re.split(r'[.!?]+', full_output[:500]) if len(s.strip()) > 20]
        if sentences:
            summary_text = ". ".join(sentences[:2]) + "."
        else:
            summary_text = f"Agent {call_sign} completed: {task[:100]}"

    if not summary_text:
        return

    msg_id = f"agent_summary_{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    await sio.emit("cerebro_narration", {
        "id": msg_id,
        "content": summary_text,
        "timestamp": timestamp,
        "agent_id": agent.get("id", ""),
        "directive_id": agent.get("directive_id", ""),
        "is_summary": True,
        "is_final": True,
        "is_idle": False,
        "message_type": "message",
    }, room=os.environ.get("CEREBRO_ROOM", "default"))


async def auto_categorize_agent(agent: dict):
    """Auto-categorize agent into a project using Ollama.
    Analyzes task + output, can match existing projects or create new categories.
    Non-blocking, errors are silently logged so Ollama being down doesn't break anything.
    """
    try:
        if not config.OLLAMA_URL:
            return
        if agent.get("project_id"):
            return  # Already assigned

        # Load existing projects from tracker with full context
        tracker_path = Path("/mnt/nas/AI_MEMORY/projects/tracker.json")
        tracker_data = {}
        project_lines = []
        project_map = {}  # various keys -> canonical project_id
        if tracker_path.exists():
            try:
                with open(tracker_path, 'r', encoding='utf-8') as f:
                    tracker_data = json.load(f)
                for key, proj in tracker_data.items():
                    if not isinstance(proj, dict):
                        continue
                    pid = proj.get("project_id") or proj.get("id") or key
                    name = proj.get("name", key)
                    techs = ", ".join(proj.get("technologies", [])) or "none listed"
                    desc = proj.get("description", "")
                    line = f"- {pid}: {name} (technologies: {techs})"
                    if desc:
                        line += f" — {desc[:100]}"
                    project_lines.append(line)
                    project_map[pid.lower()] = pid
                    project_map[name.lower()] = pid
                    project_map[key.lower()] = pid
            except Exception:
                pass

        task = agent.get("task", "")[:500]
        output_excerpt = (agent.get("output") or "")[:1500]
        agent_type = agent.get("type", "worker")
        tools = ", ".join(agent.get("tools_used", [])[:10])

        existing_list = chr(10).join(project_lines) if project_lines else "(no existing projects)"

        prompt = (
            "You are a project categorizer for an AI agent system. "
            "Given an agent's task and output, classify it.\n\n"
            f"Existing projects:\n{existing_list}\n\n"
            f"Agent task: {task}\n"
            f"Agent output (excerpt): {output_excerpt}\n"
            f"Agent type: {agent_type}\n"
            f"Tools used: {tools}\n\n"
            "Either match an existing project OR create a new descriptive category.\n"
            'Return JSON only: {"project_id": "kebab-slug", "project_name": "Human Readable Name", "is_new": true/false}'
        )

        import httpx
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{config.OLLAMA_URL}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            if resp.status_code != 200:
                return

            raw = resp.json().get("response", "").strip()

            # Parse response — try JSON first, fallback to regex
            import re
            parsed = None
            try:
                # Extract JSON from response (may have surrounding text)
                json_match = re.search(r'\{[^}]+\}', raw)
                if json_match:
                    parsed = json.loads(json_match.group())
            except (json.JSONDecodeError, ValueError):
                pass

            if not parsed or "project_id" not in parsed:
                # Regex fallback
                pid_match = re.search(r'"project_id"\s*:\s*"([^"]+)"', raw)
                name_match = re.search(r'"project_name"\s*:\s*"([^"]+)"', raw)
                new_match = re.search(r'"is_new"\s*:\s*(true|false)', raw, re.IGNORECASE)
                if pid_match:
                    parsed = {
                        "project_id": pid_match.group(1),
                        "project_name": name_match.group(1) if name_match else pid_match.group(1),
                        "is_new": new_match.group(1).lower() == "true" if new_match else False,
                    }

            if not parsed or "project_id" not in parsed:
                return  # Couldn't parse, fail silently

            slug = parsed["project_id"].lower().strip()
            project_name = parsed.get("project_name", slug)
            is_new = parsed.get("is_new", False)

            if slug in ("uncategorized", "none", "unknown", ""):
                return

            # Resolve: if Ollama said is_new=false, check if it actually matches
            matched_pid = project_map.get(slug)
            if not matched_pid and not is_new:
                # Try matching by name
                matched_pid = project_map.get(project_name.lower())

            if matched_pid:
                # Use existing project
                final_pid = matched_pid
            elif is_new or not matched_pid:
                # Auto-create new project in tracker
                final_pid = slug
                tracker_data[slug] = {
                    "project_id": slug,
                    "name": project_name,
                    "status": "active",
                    "auto_created": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "technologies": [],
                    "description": f"Auto-created from agent task: {task[:100]}",
                }
                try:
                    tracker_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(tracker_path, 'w', encoding='utf-8') as f:
                        json.dump(tracker_data, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    print(f"[Agent {agent.get('id', '?')}] Auto-created project: {slug} ({project_name})")
                except Exception as te:
                    print(f"[Agent {agent.get('id', '?')}] Failed to write tracker for new project: {te}")
                    return
            else:
                return

            # Assign project to agent
            agent["project_id"] = final_pid

            # Update persisted agent file
            agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
            for date_dir in agents_dir.iterdir():
                if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                    agent_file = date_dir / f"{agent['id']}.json"
                    if agent_file.exists():
                        with open(agent_file, 'r', encoding='utf-8') as f:
                            agent_data = json.load(f)
                        agent_data["project_id"] = final_pid
                        with open(agent_file, 'w', encoding='utf-8') as f:
                            json.dump(agent_data, f, indent=2, ensure_ascii=False)
                            f.flush()
                            os.fsync(f.fileno())
                        break

            # Update index entry
            index_file = agents_dir / "index.json"
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                for entry in index.get("agents", []):
                    if entry["id"] == agent["id"]:
                        entry["project_id"] = final_pid
                        break
                with open(index_file, 'w', encoding='utf-8') as f:
                    json.dump(index, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())

            # Notify frontend
            await sio.emit("agent_project_assigned", {
                "agent_id": agent["id"],
                "project_id": final_pid,
            }, room=os.environ.get("CEREBRO_ROOM", "default"))

            print(f"[Agent {agent['id']}] Auto-categorized to project: {final_pid} (new={is_new})")
    except Exception as e:
        print(f"[Agent {agent.get('id', '?')}] Auto-categorize failed (non-fatal): {e}")


async def save_agent_to_memory(agent: dict):
    """Save completed agent output to AI Memory for persistence and retrieval."""
    try:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        agents_dir.mkdir(exist_ok=True)

        # Create date-based subdirectory
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = agents_dir / date_str
        date_dir.mkdir(exist_ok=True)

        # Save agent data - sanitize to remove non-serializable fields like _process
        sanitized_agent = {k: v for k, v in agent.items() if not k.startswith('_')}
        agent_file = date_dir / f"{agent['id']}.json"
        agent_data = {
            **sanitized_agent,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "searchable_summary": f"Agent #{agent['id']}: {agent['task'][:200]}. Tools: {', '.join(agent.get('tools_used', []))}. Status: {agent['status']}"
        }

        with open(agent_file, 'w', encoding='utf-8') as f:
            json.dump(agent_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        # Update agents index
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {"agents": [], "last_updated": None}

        # Add to index (keep last 100 agents)
        index["agents"].insert(0, {
            "id": agent['id'],
            "call_sign": agent.get('call_sign', agent['id']),
            "task": agent['task'][:200],
            "type": agent.get('type', 'worker'),
            "status": agent['status'],
            "created_at": agent['created_at'],
            "completed_at": agent.get('completed_at'),
            "tools_used": agent.get('tools_used', []),
            "file": str(agent_file.relative_to(agents_dir)),
            "last_viewed_at": agent.get('last_viewed_at'),
            "archived": agent.get('archived', False),
            "archived_at": agent.get('archived_at'),
            "source": agent.get('source', 'user'),
            "source_agents": agent.get('source_agents', []),
            "output_preview": (agent.get('output') or '')[:300],
            "error": agent.get('error'),
            "parent_agent_id": agent.get('parent_agent_id'),
            "priority": agent.get('priority', 'normal'),
            "project_id": agent.get('project_id'),
        })
        index["agents"] = index["agents"][:100]
        index["last_updated"] = datetime.now(timezone.utc).isoformat()

        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        print(f"[Agent {agent['id']}] Saved to AI Memory: {agent_file}")

    except Exception as e:
        print(f"[Agent {agent['id']}] Failed to save to memory: {e}")


def _get_progress_enrichment(directive_id: str, step_counter: int, directive_text: str) -> dict:
    """Get progress enrichment fields for cerebro_progress events.
    Tries narration engine first, falls back to local step counter."""
    enrichment = {
        "step_number": step_counter,
        "estimated_total": max(step_counter + 1, 10),
        "directive_text": directive_text,
        "current_phase": "working",
    }

    # Try to get richer progress from narration engine
    if cognitive_loop_manager and directive_id:
        try:
            progress = cognitive_loop_manager.narration.get_directive_progress(directive_id)
            if progress.get("step_number", 0) > 0:
                enrichment["step_number"] = progress["step_number"]
                enrichment["estimated_total"] = progress["estimated_total"]
                enrichment["current_phase"] = progress.get("phase", "working")
                if progress.get("directive_text"):
                    enrichment["directive_text"] = progress["directive_text"]
        except Exception:
            pass  # Fall back to local counter

    return enrichment


async def run_agent(
    agent_id: str,
    task: str,
    agent_type: str = "worker",
    context: str = None,
    expected_output: str = None,
    resources: list[str] = None,
    context_refs: list[dict] = None,
    timeout: int = 3600,
    source_agents: list[str] = None,
    model: str = "sonnet"
):
    """Run an agent in the background with role-specific behavior."""
    import shutil

    agent = active_agents.get(agent_id)
    if not agent:
        return

    # Ensure required fields exist (defensive - prevents KeyError for older agents)
    if "tools_used" not in agent:
        agent["tools_used"] = []
    if "error" not in agent:
        agent["error"] = None

    # Update status to running
    agent["status"] = AgentStatus.RUNNING
    agent["started_at"] = datetime.now(timezone.utc).isoformat()
    await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))

    # v2.0: Register agent with cognitive loop for orb status tracking
    if cognitive_loop_manager:
        cognitive_loop_manager.register_agent_running(agent_id, agent_type)
        await sio.emit("autonomy_status", cognitive_loop_manager.get_state().to_dict(), room=os.environ.get("CEREBRO_ROOM", "default"))

    # Resolve context references (click-to-attach)
    resolved_context = ""
    if context_refs:
        resolved_context = await resolve_context_refs(context_refs)
        print(f"[Agent {agent_id}] Resolved {len(context_refs)} context refs")

    # Find claude executable
    claude_path = shutil.which("claude")
    if not claude_path:
        possible_paths = [
            os.path.join(os.path.expanduser("~"), ".local", "bin", "claude.exe"),
            os.path.join(os.path.expanduser("~"), ".local", "bin", "claude"),
            os.path.join(os.path.expanduser("~"), ".npm-global", "claude.cmd"),
            os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "npm", "claude.cmd"),
            r"C:\Program Files\nodejs\claude.cmd",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                claude_path = p
                break

    if not claude_path:
        agent["status"] = AgentStatus.FAILED
        agent["error"] = "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        agent["output"] = "Failed to start - Claude CLI not found in PATH or common locations."
        await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))
        return

    # Build the enhanced prompt with role system prompt
    is_specops = agent_type.startswith("specops_")
    if is_specops:
        role_config = AGENT_ROLE_PROMPTS["specops"]
        # Get the sub-role prompt
        sub_role = agent.get("sub_role", agent_type.replace("specops_", "")) or "worker"
        sub_role_config = AGENT_ROLE_PROMPTS.get(sub_role, AGENT_ROLE_PROMPTS["worker"])
        sub_role_prompt = sub_role_config["system_prompt"].format(call_sign=agent_id)

        # Build mission journal text
        journal_entries = agent.get("mission_journal", [])
        if journal_entries:
            journal_text = "\n".join(
                f"### Cycle {e.get('cycle', '?')} ({e.get('timestamp', 'unknown')})\n"
                f"- Progress: {e.get('progress', 'N/A')}\n"
                f"- Completed: {e.get('completed', 'N/A')}\n"
                f"- Next: {e.get('next', 'N/A')}\n"
                f"- Blockers: {e.get('blockers', 'None')}"
                for e in journal_entries
            )
        else:
            journal_text = "_No previous cycles — this is your first cycle._"

        role_prompt = role_config["system_prompt"].format(
            call_sign=agent_id,
            mission_name=agent.get("mission_name", "Unnamed Mission"),
            work_style=agent.get("work_style", "continuous"),
            cycle_interval_human=_format_interval(agent.get("cycle_interval", 3600)),
            duration_human=_format_duration(agent.get("mission_duration", 259200)),
            cycle_number=agent.get("cycle_count", 0) + 1,
            sub_role_prompt=sub_role_prompt,
            mission_journal=journal_text,
        )
    else:
        role_config = AGENT_ROLE_PROMPTS.get(agent_type, AGENT_ROLE_PROMPTS["worker"])
        role_prompt = role_config["system_prompt"].format(call_sign=agent_id)

    # Construct the full prompt with all context
    full_prompt_parts = [role_prompt]

    # Inject Cerebro capabilities context so agents know what tools are available
    if not _IS_STANDALONE:
        full_prompt_parts.append("""
---

## Cerebro Capabilities (HTTP API at localhost:59000)
You have access to these Cerebro backend tools via curl. Use them when relevant to the task.

### Trading (Alpaca - Paper Account, ~$100K)
- **Get positions:** `curl -s http://localhost:59000/api/trading/positions -H "Authorization: Bearer TOKEN"`
- **Close a position:** `curl -s -X DELETE http://localhost:59000/api/trading/position/SYMBOL -H "Authorization: Bearer TOKEN"`
- **Place order:** `curl -s -X POST http://localhost:59000/api/trading/order -H "Authorization: Bearer TOKEN" -H "Content-Type: application/json" -d '{"symbol":"AAPL","qty":1,"side":"buy","type":"market","time_in_force":"day"}'`
- **Get account:** `curl -s http://localhost:59000/api/trading/account -H "Authorization: Bearer TOKEN"`
- **Get orders:** `curl -s http://localhost:59000/api/trading/orders -H "Authorization: Bearer TOKEN"`
- **Cancel order:** `curl -s -X DELETE http://localhost:59000/api/trading/order/ORDER_ID -H "Authorization: Bearer TOKEN"`
Note: Get TOKEN with `curl -s -X POST http://localhost:59000/auth/login -H "Content-Type: application/json" -d '{"username":"professor","password":"professor"}' | jq -r .token`

### Wallet (Financial Activity Log)
- **Log activity:** `curl -s -X POST http://localhost:59000/api/wallet/log -H "Content-Type: application/json" -d '{"category":"trade","description":"...","pnl":0.0,"symbol":"BTC/USD","source":"agent"}'`
  Categories: trade, backtest, bet, other

### Browser Control (shared Chrome)
- **Page state:** `curl -s http://localhost:59000/api/browser/page_state`
- **Navigate:** `curl -s -X POST http://localhost:59000/api/browser/agent/navigate -H "Content-Type: application/json" -d '{"url":"https://..."}'`
- **Click:** `curl -s -X POST http://localhost:59000/api/browser/click -H "Content-Type: application/json" -d '{"element_index":5}'`

### Ask User (blocks until response)
- `curl -s -X POST http://localhost:59000/api/agent/ask -H "Content-Type: application/json" -d '{"question":"Should I proceed?","options":["Yes","No"],"agent_id":"AGENT_ID"}'`

When the user mentions trades, positions, buying, selling, crypto, stocks — USE the trading endpoints above. Do NOT say you cannot trade.

### Remote Machines (SSH Access - Passwordless)
You are running on the **ASUS GX10** (Linux, 192.168.0.70). You have passwordless SSH to these machines:

| Alias | Machine | IP | OS | User | Notes |
|-------|---------|----|----|------|-------|
| `pc` | Professor's main PC | 192.168.0.166:2222 | Windows 11 | marke | Main monitor, desktop at `C:\\Users\\marke\\Desktop\\` |
| `spark` | DGX Spark | 192.168.0.6 | Linux | professorlow | 128GB RAM, GB10 GPU |
| `darkhorse` | Raspberry Pi | 192.168.0.81 | Kali Linux | professor | |

**To create files on the Windows PC:**
```bash
# Write content to a file on the desktop
ssh pc 'echo CONTENT > C:\\Users\\marke\\Desktop\\filename.txt'
# For multi-line content, pipe through PowerShell:
printf '%s' "$CONTENT" | ssh pc 'powershell -Command "$input | Out-File -FilePath C:\\Users\\marke\\Desktop\\filename.txt -Encoding utf8"'
# Or use multiple echo lines (simpler):
ssh pc "echo Line 1 > C:\\Users\\marke\\Desktop\\filename.txt && echo Line 2 >> C:\\Users\\marke\\Desktop\\filename.txt"
# Open a file in Notepad on the main monitor
ssh pc 'start notepad.exe C:\\Users\\marke\\Desktop\\filename.txt'
# IMPORTANT: Windows uses CMD shell, NOT bash. Use 'echo', 'type', 'del', 'dir' — NOT 'cat', 'rm', 'ls'.
```

**To run commands on Spark or Darkhorse:**
```bash
ssh spark 'command here'
ssh darkhorse 'command here'
```

**NAS shared storage** is mounted locally at `/mnt/nas/AI_MEMORY/` (read/write).

IMPORTANT: When tasks mention "desktop", "main PC", "Notepad", or "main monitor" — use `ssh pc` to create files and open them.
When tasks mention "Spark" or "DGX" — use `ssh spark`. When tasks mention "darkhorse" or "Pi" — use `ssh darkhorse`.
""")
    else:
        full_prompt_parts.append("""
---

## Cerebro Capabilities (HTTP API at localhost:59000)

### Browser Control (shared Chrome)
- **Page state:** `curl -s http://localhost:59000/api/browser/page_state`
- **Navigate:** `curl -s -X POST http://localhost:59000/api/browser/agent/navigate -H "Content-Type: application/json" -d '{"url":"https://..."}'`
- **Click:** `curl -s -X POST http://localhost:59000/api/browser/click -H "Content-Type: application/json" -d '{"element_index":5}'`
- **Fill:** `curl -s -X POST http://localhost:59000/api/browser/fill -H "Content-Type: application/json" -d '{"element_index":3,"value":"text"}'`
- **Scroll:** `curl -s -X POST http://localhost:59000/api/browser/scroll -H "Content-Type: application/json" -d '{"direction":"down","amount":500}'`
- **Screenshot:** `curl -s http://localhost:59000/api/browser/screenshot/file`

### Ask User (blocks until response, 5min timeout)
- `curl -s -X POST http://localhost:59000/api/agent/ask -H "Content-Type: application/json" -d '{"question":"Should I proceed?","options":["Yes","No"],"agent_id":"AGENT_ID"}'`

### Spawn Child Agent
- `curl -s -X POST http://localhost:59000/internal/spawn-child-agent -H "Content-Type: application/json" -d '{"task":"...","type":"worker"}'`

You are running inside a Docker container. Do NOT assume external servers, NAS, or SSH targets exist.
Use `hostname`, `uname -a`, `ip addr` to discover this system.
""")

    full_prompt_parts.append("\n---\n\n## Task Details\n")

    # v2.0: Inject cross-agent context (what other agents are doing)
    other_agents_ctx = []
    now_utc = datetime.now(timezone.utc)
    for aid, ainfo in active_agents.items():
        if aid == agent_id:
            continue  # Skip self
        astatus = ainfo.get("status", "")
        if astatus == AgentStatus.RUNNING:
            started = ainfo.get("started_at", "")
            elapsed = ""
            if started:
                try:
                    start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                    mins = int((now_utc - start_dt).total_seconds() / 60)
                    elapsed = f" — running {mins}m"
                except Exception:
                    pass
            last_out = (ainfo.get("output") or "")[-100:].strip().split("\n")[-1] if ainfo.get("output") else ""
            last_snippet = f', last: "{last_out}"' if last_out else ""
            other_agents_ctx.append(f"- {ainfo.get('call_sign', aid)} ({ainfo.get('type', '?')}): \"{ainfo.get('task', '')[:80]}\"{elapsed}{last_snippet}")
        elif astatus == AgentStatus.COMPLETED:
            completed_at = ainfo.get("completed_at", "")
            if completed_at:
                try:
                    comp_dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                    ago_mins = int((now_utc - comp_dt).total_seconds() / 60)
                    if ago_mins <= 5:
                        other_agents_ctx.append(f"- {ainfo.get('call_sign', aid)} ({ainfo.get('type', '?')}): \"{ainfo.get('task', '')[:80]}\" — completed {ago_mins}m ago")
                except Exception:
                    pass
    if other_agents_ctx:
        full_prompt_parts.append("\n**[RUNNING AGENTS]**\n" + "\n".join(other_agents_ctx[:5]) + "\n\n")

    # Add resolved context references first
    if resolved_context:
        full_prompt_parts.append(resolved_context)

    # Context fusion: resolve source agent IDs to file paths
    if source_agents:
        try:
            agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
            index_file = agents_dir / "index.json"
            source_lines = []
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    agent_index = json.load(f)
                for src_id in source_agents:
                    # Check index for file path
                    for entry in agent_index.get("agents", []):
                        if entry.get("id") == src_id:
                            file_path = agents_dir / entry.get("file", "")
                            call_sign = entry.get("call_sign", src_id)
                            agent_type_str = entry.get("type", "worker")
                            task_preview = entry.get("task", "")[:80]
                            source_lines.append(f"- {call_sign} ({agent_type_str}): \"{task_preview}\" -> {file_path}")
                            break
                    else:
                        # Not in index - check active agents dict
                        if src_id in active_agents:
                            a = active_agents[src_id]
                            source_lines.append(f"- {a.get('call_sign', src_id)} ({a.get('type', 'worker')}): \"{a.get('task', '')[:80]}\" -> (still in memory, no file yet)")
            if source_lines:
                full_prompt_parts.append("\n## Referenced Agent Sessions\nRead these agent output files for context on prior work:\n")
                full_prompt_parts.append("\n".join(source_lines))
                full_prompt_parts.append("\n\nUse the Read tool to access these JSON files and understand what each agent accomplished.\n\n")
                print(f"[Agent {agent_id}] Injected {len(source_lines)} source agent references")
        except Exception as e:
            print(f"[Agent {agent_id}] Failed to resolve source agents: {e}")

    full_prompt_parts.append(f"**What I need:** {task}\n")

    if context:
        full_prompt_parts.append(f"\n**Additional context:**\n{context}\n")

    if expected_output:
        full_prompt_parts.append(f"\n**Expected output format:**\n{expected_output}\n")

    if resources:
        full_prompt_parts.append("\n**Files/resources to reference:**\n")
        for r in resources:
            full_prompt_parts.append(f"- {r}\n")

    full_prompt_parts.append("\n---\n\nPlease complete this task.")

    full_prompt = "".join(full_prompt_parts)

    print(f"[Agent {agent_id}] Starting as {role_config['name']} with claude at: {claude_path} (model: {model})")
    print(f"[Agent {agent_id}] Task: {task[:100]}...")

    try:
        # Run Claude Code with subprocess.Popen (Windows SelectorEventLoop doesn't support
        # asyncio.create_subprocess_exec, so we use threads for I/O)
        import subprocess as sp
        import threading
        import queue

        # Resolve model shorthand to full model ID
        model_map = {
            "opus": "claude-opus-4-6",
            "sonnet": "claude-sonnet-4-6",
            "haiku": "claude-haiku-4-5-20251001",
        }
        resolved_model = model_map.get(model, model)

        # Build command args
        # In Docker containers, the claude symlink may not exec properly via Popen
        # due to shebang resolution issues with symlinks. Use 'node' prefix as workaround.
        if _IS_STANDALONE and os.path.islink(claude_path):
            cmd_args = ["node", claude_path, "-p", full_prompt,
                 "--model", resolved_model,
                 "--output-format", "stream-json",
                 "--verbose",
                 "--dangerously-skip-permissions"]
        else:
            cmd_args = [claude_path, "-p", full_prompt,
                 "--model", resolved_model,
                 "--output-format", "stream-json",
                 "--verbose",
                 "--dangerously-skip-permissions"]

        # Orchestrator agents MUST use Bash+curl to spawn children, not Task tool
        if agent_type == "orchestrator":
            cmd_args.extend(["--disallowedTools", "Task", "TodoWrite"])

        # Strip CLAUDECODE env var so spawned CLI doesn't think it's nested
        agent_env = os.environ.copy()
        agent_env.pop("CLAUDECODE", None)
        # Signal hooks to exit immediately (avoids brain_maintenance, wake-nas, etc.)
        agent_env["CEREBRO_AGENT"] = "1"
        # Standalone: ensure HOME points to cerebro user dir where credentials live
        if _IS_STANDALONE:
            agent_env["HOME"] = "/home/cerebro"

        process = sp.Popen(
            cmd_args,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            cwd=config.AI_MEMORY_PATH,
            env=agent_env,
        )

        # Store process PID for emergency stop capability
        agent["pid"] = process.pid
        agent["_process"] = process  # Store process object for termination
        await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))

        full_output = ""
        raw_lines = []  # Keep raw output for debugging
        update_counter = 0
        _narration_seen = set()  # Dedup narration content to prevent duplicate cards
        _progress_step_counter = 0  # Step counter for progress enrichment
        _agent_directive_id = agent.get("directive_id", "")
        _agent_directive_text = task[:120] if task else ""

        # Thread-based line reader (feeds queue from blocking stdout.readline)
        line_queue = queue.Queue()

        def _reader_thread():
            try:
                for raw_bytes in iter(process.stdout.readline, b''):
                    line_queue.put(raw_bytes)
            except Exception:
                pass
            finally:
                line_queue.put(None)  # Sentinel

        reader = threading.Thread(target=_reader_thread, daemon=True)
        reader.start()

        # Timeout tracking
        import time as _time
        _agent_start = _time.monotonic()
        effective_timeout = timeout if timeout > 0 else None
        agent["timeout"] = timeout  # Store for display

        # Stream output via queue (non-blocking from asyncio perspective)
        while True:
            # Check configurable timeout
            if effective_timeout and (_time.monotonic() - _agent_start) > effective_timeout:
                raise asyncio.TimeoutError()

            # Poll queue from event loop thread (short timeout for responsive timeout checks)
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: line_queue.get(timeout=5)
                )
            except queue.Empty:
                continue  # No output yet, loop back to check timeout

            if line is None:
                break

            raw_line = line.decode('utf-8', errors='replace').strip()
            if not raw_line:
                continue
            raw_lines.append(raw_line)

            try:
                data = json.loads(raw_line)
                msg_type = data.get("type", "")

                if msg_type == "assistant":
                    text = data.get("message", {}).get("content", "")
                    if isinstance(text, list):
                        for block in text:
                            if block.get("type") == "text":
                                block_text = block.get("text", "")
                                full_output += block_text
                                # === v2.0: Stream agent text as narration (deduped) ===
                                _narr_key = block_text[:200].strip()
                                if block_text.strip() and _narr_key not in _narration_seen:
                                    _narration_seen.add(_narr_key)
                                    _progress_step_counter += 1
                                    _progress_enrichment = _get_progress_enrichment(
                                        _agent_directive_id, _progress_step_counter, _agent_directive_text
                                    )
                                    await sio.emit("cerebro_progress", {
                                        "status": block_text[:120],
                                        "content": block_text,
                                        "agent_id": agent_id,
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        **_progress_enrichment,
                                    }, room=os.environ.get("CEREBRO_ROOM", "default"))
                            elif block.get("type") == "tool_use":
                                tool_name = block.get("name", "unknown")
                                if tool_name not in agent["tools_used"]:
                                    agent["tools_used"].append(tool_name)
                                # === v2.0: Stream tool use as browser_step ===
                                raw_input = block.get("input", {})
                                if isinstance(raw_input, dict):
                                    # Extract the most useful field for display
                                    tool_detail = (raw_input.get("command")
                                                   or raw_input.get("url")
                                                   or raw_input.get("query")
                                                   or raw_input.get("description")
                                                   or json.dumps(raw_input, default=str))[:200]
                                else:
                                    tool_detail = str(raw_input)[:200]
                                await sio.emit("browser_step", {
                                    "step": tool_name,
                                    "action": tool_detail,
                                    "reasoning": "",
                                    "agent_id": agent_id,
                                }, room=os.environ.get("CEREBRO_ROOM", "default"))
                                # Also emit as progress content for live output panel
                                _progress_step_counter += 1
                                await sio.emit("cerebro_progress", {
                                    "status": f"Using {tool_name}...",
                                    "content": f"[{tool_name}] {tool_detail}",
                                    "agent_id": agent_id,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "is_tool": True,
                                }, room=os.environ.get("CEREBRO_ROOM", "default"))
                    elif isinstance(text, str) and text:
                        full_output += text
                        # === v2.0: Stream agent text as progress (deduped) ===
                        _narr_key = text[:200].strip()
                        if text.strip() and _narr_key not in _narration_seen:
                            _narration_seen.add(_narr_key)
                            _progress_step_counter += 1
                            _progress_enrichment = _get_progress_enrichment(
                                _agent_directive_id, _progress_step_counter, _agent_directive_text
                            )
                            await sio.emit("cerebro_progress", {
                                "status": text[:120],
                                "content": text,
                                "agent_id": agent_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                **_progress_enrichment,
                            }, room=os.environ.get("CEREBRO_ROOM", "default"))

                elif msg_type == "tool_use":
                    tool_name = data.get("tool", data.get("name", "tool"))
                    if tool_name not in agent["tools_used"]:
                        agent["tools_used"].append(tool_name)
                    # === v2.0: Stream tool use as browser_step ===
                    raw_input = data.get("input", {})
                    if isinstance(raw_input, dict):
                        tool_detail = (raw_input.get("command")
                                       or raw_input.get("url")
                                       or raw_input.get("query")
                                       or raw_input.get("description")
                                       or json.dumps(raw_input, default=str))[:200]
                    else:
                        tool_detail = str(raw_input)[:200]
                    await sio.emit("browser_step", {
                        "step": tool_name,
                        "action": tool_detail,
                        "reasoning": "",
                        "agent_id": agent_id,
                    }, room=os.environ.get("CEREBRO_ROOM", "default"))

                elif msg_type == "result":
                    # Final result message
                    result_text = data.get("result", "")
                    if result_text and result_text not in full_output:
                        full_output += "\n\n" + result_text
                    # === v3.0: Final result — just emit progress done signal ===
                    # Summary will be generated after agent completes
                    _progress_step_counter += 1
                    _progress_enrichment = _get_progress_enrichment(
                        _agent_directive_id, _progress_step_counter, _agent_directive_text
                    )
                    await sio.emit("cerebro_progress", {
                        "status": "Wrapping up...",
                        "agent_id": agent_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "is_final": True,
                        **_progress_enrichment,
                    }, room=os.environ.get("CEREBRO_ROOM", "default"))

                elif msg_type == "error":
                    error_msg = data.get("error", {}).get("message", str(data))
                    full_output += f"\n\n[Error: {error_msg}]"

                # Update agent output periodically (every 5 messages or if output changed significantly)
                update_counter += 1
                if update_counter >= 5 or len(full_output) - len(agent.get("output", "")) > 200:
                    agent["output"] = full_output[-5000:]  # Keep last 5000 chars
                    await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))
                    update_counter = 0

            except json.JSONDecodeError:
                # Not JSON - might be raw text output
                if raw_line and not raw_line.startswith('{'):
                    full_output += raw_line + "\n"

        # Wait for process to complete (subprocess.Popen version)
        reader.join(timeout=300)
        try:
            process.wait(timeout=30)
        except sp.TimeoutExpired:
            print(f"[Agent {agent_id}] Process wait timeout, killing...")
            process.kill()
            agent["error"] = "Process killed after timeout"

        # Check stderr for errors (filter WSL2 diagnostic noise on Docker/Windows)
        stderr_output = process.stderr.read() if process.stderr else b""
        if stderr_output:
            stderr_text = stderr_output.decode('utf-8', errors='replace')
            # Filter out harmless WSL2 interop messages that leak into Docker containers
            stderr_lines = [
                line for line in stderr_text.splitlines()
                if "UtilGetPpid" not in line and "UtilBindVsockAnyPort" not in line
            ]
            stderr_text = "\n".join(stderr_lines).strip()
            if stderr_text:
                print(f"[Agent {agent_id}] Stderr: {stderr_text[:500]}")
                if "error" in stderr_text.lower() and not full_output:
                    full_output = f"[Stderr]\n{stderr_text}"

        # If no output was captured, include raw output for debugging
        if not full_output and raw_lines:
            full_output = f"[Raw output - {len(raw_lines)} lines]\n" + "\n".join(raw_lines[:50])
            if len(raw_lines) > 50:
                full_output += f"\n... ({len(raw_lines) - 50} more lines)"

        if not full_output:
            full_output = f"Agent completed but no output was captured. Exit code: {process.returncode}"

        # If agent was already stopped by user, don't overwrite status — just update output
        if agent.get("status") == AgentStatus.STOPPED:
            # Agent was gracefully stopped — update output if we got more, but keep status
            if full_output and len(full_output) > len(agent.get("output") or ""):
                agent["output"] = full_output
                await save_agent_to_memory(agent)
            print(f"[Agent {agent_id}] Process exited after user stop. Output length: {len(full_output)}")
        else:
            agent["status"] = AgentStatus.COMPLETED
            agent["completed_at"] = datetime.now(timezone.utc).isoformat()
            agent["output"] = full_output
            print(f"[Agent {agent_id}] Completed. Output length: {len(full_output)}")

            # Generate and emit friendly agent summary (non-blocking)
            asyncio.create_task(_generate_agent_summary(agent, full_output))

            # Update execution status if this is a scheduled agent
            if agent.get("execution_id"):
                update_execution_status(agent["execution_id"], "success")

            # Save to AI Memory for persistence
            await save_agent_to_memory(agent)

            # Auto-categorize into project (non-blocking)
            asyncio.create_task(auto_categorize_agent(agent))

            # Create notification for completed agent
            await create_notification(
                notif_type="agent_complete",
                title=f"Agent {agent.get('call_sign', agent_id)} Completed",
                message=agent["task"][:100] + ("..." if len(agent["task"]) > 100 else ""),
                link=f"/agents/{agent_id}",
                agent_id=agent_id
            )

            # AUTO-COMPLETE DIRECTIVE: If agent was spawned for a directive, mark it complete
            if agent.get("directive_id"):
                directive_id = agent["directive_id"]
                print(f"[Agent {agent_id}] Agent completed - auto-completing directive {directive_id}")
                try:
                    await auto_complete_directive_from_agent(directive_id, agent_id, agent.get("output", ""))
                except Exception as e:
                    print(f"[Agent {agent_id}] Failed to auto-complete directive: {e}")

            # SCREEN OBSERVATION: Emit cerebro_observation event for screen monitor agents
            if agent.get("source") == "screen_observation":
                try:
                    # Extract screenshot path from resources
                    screenshot_path = ""
                    for r in agent.get("resources", []):
                        if "screenshots" in str(r):
                            screenshot_path = str(r)
                            break

                    # Extract window title from task string
                    window_title = "Unknown"
                    task_str = agent.get("task", "")
                    if "window:" in task_str.lower():
                        # Try to extract "window: Title" pattern
                        wt_match = re.search(r'window:\s*(.+?)(?:\n|$)', task_str, re.IGNORECASE)
                        if wt_match:
                            window_title = wt_match.group(1).strip()
                    elif "title:" in task_str.lower():
                        wt_match = re.search(r'title:\s*(.+?)(?:\n|$)', task_str, re.IGNORECASE)
                        if wt_match:
                            window_title = wt_match.group(1).strip()

                    obs_event = {
                        "id": f"obs_{agent_id}_{int(time.time())}",
                        "content": full_output,
                        "screenshot_path": screenshot_path,
                        "window_title": window_title,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "agent_id": agent_id,
                        "source": "screen_monitor"
                    }
                    await sio.emit("cerebro_observation", obs_event, room=os.environ.get("CEREBRO_ROOM", "default"))
                    print(f"[Agent {agent_id}] Emitted cerebro_observation for screen monitor")
                except Exception as e:
                    print(f"[Agent {agent_id}] Failed to emit observation: {e}")

    except asyncio.TimeoutError:
        # Kill the process on timeout
        try:
            if 'process' in dir() and process and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
                print(f"[Agent {agent_id}] Process killed after timeout")
        except Exception:
            pass

        timeout_minutes = timeout // 60 if timeout else 0
        timeout_msg = f"Timed out after {timeout_minutes} minutes"
        agent["status"] = AgentStatus.FAILED
        agent["error"] = timeout_msg
        print(f"[Agent {agent_id}] {timeout_msg}")

        # Update execution status if this is a scheduled agent
        if agent.get("execution_id"):
            update_execution_status(agent["execution_id"], "failed", timeout_msg)

        # Create notification for failed agent
        await create_notification(
            notif_type="agent_failed",
            title=f"Agent {agent.get('call_sign', agent_id)} Timed Out",
            message=timeout_msg,
            link=f"/agents/{agent_id}",
            agent_id=agent_id
        )
    except Exception as e:
        # Don't overwrite stopped status with failed
        if agent.get("status") != AgentStatus.STOPPED:
            agent["status"] = AgentStatus.FAILED
            agent["error"] = f"{type(e).__name__}: {str(e)}"
            print(f"[Agent {agent_id}] Exception: {e}")
            import traceback
            traceback.print_exc()

            # Update execution status if this is a scheduled agent
            if agent.get("execution_id"):
                update_execution_status(agent["execution_id"], "failed", str(e))

            # Create notification for failed agent
            await create_notification(
                notif_type="agent_failed",
                title=f"Agent {agent.get('call_sign', agent_id)} Failed",
                message=str(e)[:100],
                link=f"/agents/{agent_id}",
                agent_id=agent_id
            )
        else:
            print(f"[Agent {agent_id}] Exception after user stop (expected): {e}")

    # Emit final status to frontend
    print(f"[Agent {agent_id}] Emitting completion events. Status: {agent['status']}")
    await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))
    await sio.emit("agent_completed", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))

    # v2.0: Unregister agent from cognitive loop status tracking
    if cognitive_loop_manager:
        cognitive_loop_manager.register_agent_completed(agent_id)
        await sio.emit("autonomy_status", cognitive_loop_manager.get_state().to_dict(), room=os.environ.get("CEREBRO_ROOM", "default"))

    print(f"[Agent {agent_id}] Completion events emitted successfully")

    # Process queue — completed agent frees a slot
    await _process_agent_queue()

    # Wake cognitive loop to re-check pending directives immediately
    # (fixes bug where directives submitted mid-task were never picked up)
    if cognitive_loop_manager:
        cognitive_loop_manager.wake()

# Wire v2.0 agent creation callback now that create_agent is defined
if cognitive_loop_manager:
    cognitive_loop_manager._create_agent_fn = create_agent
    print("[OK] Cerebro v2.0 dispatcher wired: _create_agent_fn -> create_agent()")

# Security
security = HTTPBearer(auto_error=False)

# Redis connection
redis: Optional[aioredis.Redis] = None

# Autonomy Services (initialized on startup)
proactive_manager = None
predictive_service = None
learning_injector = None

# ============================================================================
# Models
# ============================================================================

class ChatMessage(BaseModel):
    content: str
    session_id: Optional[str] = None
    model: Optional[str] = None

class TaskRequest(BaseModel):
    prompt: str
    background: bool = False

class LoginRequest(BaseModel):
    password: str

class SetupRequest(BaseModel):
    name: str
    password: str
    use_cases: list = []
    style: str = "balanced"

class BriefingResponse(BaseModel):
    greeting: str
    time: str
    pending_tasks: list
    suggestions: list
    memory_stats: dict

# ============================================================================
# Authentication
# ============================================================================

def create_token(user_id: str = "professor") -> str:
    """Create a JWT token."""
    payload = {
        "sub": user_id,
        "iat": datetime.now(timezone.utc).timestamp(),
        "exp": datetime.now(timezone.utc).timestamp() + 86400 * 30  # 30 days
    }
    return jwt.encode(payload, config.SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(credentials.credentials, config.SECRET_KEY, algorithms=["HS256"])
        return payload["sub"]
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    global redis, proactive_manager, predictive_service, learning_injector

    # Initialize Redis
    redis = await aioredis.from_url(config.REDIS_URL, decode_responses=True)
    print("Cerebro started - Redis connected")

    # Reload persisted agents into memory
    reload_agents_from_persistence()

    # Start background task listener
    asyncio.create_task(redis_task_listener())

    # Initialize Autonomy Services (non-blocking)
    async def _init_autonomy():
        global proactive_manager, predictive_service, learning_injector
        try:
            from predictive_interrupt import get_predictive_service
            from proactive_agent import get_proactive_manager
            from learning_injector import get_learning_injector

            # Initialize MCP Bridge (may be slow on first load)
            await mcp_bridge._ensure_initialized()

            # Initialize predictive interrupt service
            predictive_service = get_predictive_service(mcp_bridge)
            print("Cerebro Autonomy: Predictive service initialized")

            # Initialize learning injector
            learning_injector = get_learning_injector(mcp_bridge, Path(config.AI_MEMORY_PATH))
            print("Cerebro Autonomy: Learning injector initialized")

            # Initialize proactive agent manager (start monitoring in background)
            proactive_manager = get_proactive_manager(
                mcp_bridge=mcp_bridge,
                create_agent_func=create_agent,
                notify_func=create_notification,
                storage_path=Path(config.AI_MEMORY_PATH)
            )
            if proactive_manager:
                asyncio.create_task(proactive_manager.start_monitoring())
                print("Cerebro Autonomy: Proactive agent manager started")

        except ImportError as e:
            print(f"Cerebro Autonomy: Some services not available - {e}")
        except Exception as e:
            print(f"Cerebro Autonomy: Initialization error - {e}")

    # Run autonomy init in background so startup completes immediately
    asyncio.create_task(_init_autonomy())

@app.on_event("shutdown")
async def shutdown():
    if redis:
        await redis.close()
    # Close aiohttp sessions to prevent "Unclosed client session" warnings
    if cognitive_loop_manager:
        try:
            engine = getattr(cognitive_loop_manager, 'ooda', None)
            if engine:
                if hasattr(engine, 'ollama') and engine.ollama and hasattr(engine.ollama, 'close'):
                    await engine.ollama.close()
                if hasattr(engine, 'tools') and engine.tools and hasattr(engine.tools, 'close'):
                    await engine.tools.close()
        except Exception:
            pass

async def redis_task_listener():
    """Listen to task updates and broadcast via Socket.IO."""
    pubsub = redis.pubsub()
    await pubsub.psubscribe("task:*")

    async for message in pubsub.listen():
        if message["type"] == "pmessage":
            channel = message["channel"]
            task_id = channel.split(":")[1]
            try:
                data = json.loads(message["data"])
                await sio.emit("task_update", data, room=f"task:{task_id}")
            except:
                pass

# ============================================================================
# Socket.IO Events
# ============================================================================

@sio.event
async def connect(sid, environ, auth=None):
    print(f"Client connected: {sid}")
    # Add to professor's room for cross-device sync
    await sio.enter_room(sid, "professor")
    if "professor" not in connected_clients:
        connected_clients["professor"] = set()
    connected_clients["professor"].add(sid)
    # Notify all devices about new connection
    await sio.emit("device_sync", {"type": "connected", "devices": len(connected_clients.get("professor", set()))}, room=os.environ.get("CEREBRO_ROOM", "default"))

    # Backfill activity log for the newly connected client (all phases for cycle view)
    if cognitive_loop_manager:
        try:
            thoughts = await cognitive_loop_manager.get_recent_thoughts(500)
            activity = []
            for t in thoughts:
                phase = (t.get("phase") or "").lower()
                if not phase:
                    continue
                content = t.get("content", "")
                meta = t.get("metadata", {})
                activity.append({
                    "id": t.get("id", ""),
                    "phase": phase,
                    "content": content[:120] + "..." if len(content) > 120 else content,
                    "fullContent": content,
                    "timestamp": t.get("timestamp", ""),
                    "cycle_number": meta.get("cycle_number", 0),
                    "reasoning": t.get("reasoning", ""),
                    "confidence": t.get("confidence", 0),
                    "metadata": meta,
                    "is_browser_step": meta.get("is_browser_step", False),
                })
                if len(activity) >= 300:
                    break
            if activity:
                await sio.emit("activity_backfill", {"activity": activity}, to=sid)
        except Exception as e:
            print(f"[Socket] Activity backfill failed: {e}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    # Remove from tracking
    if "professor" in connected_clients:
        connected_clients["professor"].discard(sid)
    # Notify remaining devices
    await sio.emit("device_sync", {"type": "disconnected", "devices": len(connected_clients.get("professor", set()))}, room=os.environ.get("CEREBRO_ROOM", "default"))

@sio.event
async def subscribe_task(sid, data):
    """Subscribe to task updates."""
    task_id = data.get("task_id")
    if task_id:
        await sio.enter_room(sid, f"task:{task_id}")

# ============================================================================
# Browser Intent Detection for Chat
# ============================================================================

# Brand name -> URL mapping (shared with ooda_engine)
_BRAND_MAP = {
    "crunchyroll": "https://www.crunchyroll.com",
    "crunchy roll": "https://www.crunchyroll.com",
    "amazon": "https://www.amazon.com",
    "youtube": "https://www.youtube.com",
    "reddit": "https://www.reddit.com",
    "linkedin": "https://www.linkedin.com",
    "twitter": "https://x.com",
    "x.com": "https://x.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "github": "https://github.com",
    "netflix": "https://www.netflix.com",
    "hulu": "https://www.hulu.com",
    "spotify": "https://open.spotify.com",
    "twitch": "https://www.twitch.tv",
    "google": "https://www.google.com",
    "wikipedia": "https://en.wikipedia.org",
    "ebay": "https://www.ebay.com",
    "walmart": "https://www.walmart.com",
}

# Regex patterns compiled once
_ACTION_VERB_RE = re.compile(
    r'\b(go\s+to|open|navigate\s+to|browse\s+to|visit|pull\s+up|launch|head\s+to)\b',
    re.IGNORECASE
)
_SEARCH_ON_RE = re.compile(
    r'\b(search|find|look\s+up|look\s+for|check)\b.+\b(on|in|at|from)\b',
    re.IGNORECASE
)
_BROWSER_REF_RE = re.compile(
    r'\b(your\s+browser|in\s+chrome|use\s+the\s+browser|in\s+the\s+browser|with\s+the\s+browser)\b',
    re.IGNORECASE
)
_QUESTION_RE = re.compile(
    r'^\s*(what\s+is|what\s+are|who\s+is|how\s+do|how\s+does|tell\s+me\s+about|explain|define|describe|why\s+is|why\s+do|can\s+you\s+tell)\b',
    re.IGNORECASE
)
_URL_RE = re.compile(r'https?://[^\s]+')
_DOMAIN_RE = re.compile(r'\b([a-zA-Z0-9][-a-zA-Z0-9]*\.(?:com|org|net|io|tv|co|gg|me|app|dev|xyz))\b', re.IGNORECASE)


def detect_action_intent(message: str) -> Optional[dict]:
    """
    Fast regex-based classifier to detect browser action commands in chat.
    Returns {"type": "browser", "goal": str, "url": str|None} or None.
    """
    if not message or len(message) < 3:
        return None

    # Early exit: pure questions don't trigger browser
    if _QUESTION_RE.match(message):
        return None

    msg_lower = message.lower().strip()
    has_action_verb = bool(_ACTION_VERB_RE.search(message))
    has_search_on = bool(_SEARCH_ON_RE.search(message))
    has_browser_ref = bool(_BROWSER_REF_RE.search(message))

    # Detect target URL
    url = None

    # 1. Explicit URL
    url_match = _URL_RE.search(message)
    if url_match:
        url = url_match.group(0).rstrip('.,;:!?)')

    # 2. Brand name
    if not url:
        for brand, brand_url in _BRAND_MAP.items():
            if brand in msg_lower:
                url = brand_url
                break

    # 3. Bare domain (word.com, word.org, etc.)
    if not url:
        domain_match = _DOMAIN_RE.search(message)
        if domain_match:
            url = f"https://{domain_match.group(1)}"

    # Decision: require (action verb OR search-on pattern OR browser reference) AND a target
    has_target = url is not None
    has_intent = has_action_verb or has_search_on or has_browser_ref

    if has_intent and has_target:
        return {"type": "browser", "goal": message, "url": url}

    return None


async def _handle_browser_chat_action(sid, session_id: str, content: str, intent: dict):
    """Route a browser command from chat to the OODA engine's AdaptiveExplorer."""
    from cognitive_loop.ooda_engine import Decision, RiskLevel

    url = intent.get("url", "")
    intent.get("goal", content)

    # Check OODA engine is available
    print(f"[Chat->Browser] Handler called. cognitive_loop_manager={cognitive_loop_manager is not None}")
    ooda = getattr(cognitive_loop_manager, 'ooda', None) if cognitive_loop_manager else None
    print(f"[Chat->Browser] ooda={ooda is not None}")
    if not ooda:
        # Fall back to Claude Code CLI — send info message then let normal path run
        await sio.emit("chat_response", {
            "type": "text",
            "content": "Browser engine not available. Let me help with text instead...\n\n"
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
        # Run normal CLI path
        full_response = ""
        async for chunk in process_chat_stream(content, session_id):
            await sio.emit("chat_response", chunk, room=os.environ.get("CEREBRO_ROOM", "default"))
            if chunk.get("type") == "text":
                full_response += chunk.get("content", "")
        if full_response.strip():
            add_to_session(session_id, "assistant", full_response)
        asyncio.create_task(save_chat_to_memory(session_id, content, full_response))
        return

    # Ensure shared browser is running before OODA handler checks is_alive()
    try:
        from cognitive_loop.browser_manager import get_browser_manager
        mgr = get_browser_manager()
        if not mgr.is_alive():
            print("[Chat->Browser] Browser not running, auto-launching...")
            await mgr.ensure_running()
            status = await mgr.get_status()
            await sio.emit("browser_launched", status, room=os.environ.get("CEREBRO_ROOM", "default"))
            print("[Chat->Browser] Browser auto-launched successfully")
        # Also inject into OODA if not already set
        if ooda and not getattr(ooda, 'browser_manager', None):
            ooda.browser_manager = mgr
    except Exception as launch_err:
        print(f"[Chat->Browser] Browser auto-launch failed: {launch_err}")

    # Send acknowledgment
    # Extract a friendly domain name for the message
    domain_display = url.replace("https://www.", "").replace("https://", "").rstrip("/")
    ack_msg = f"Opening browser for {domain_display}..."
    await sio.emit("chat_response", {"type": "text", "content": ack_msg}, room=os.environ.get("CEREBRO_ROOM", "default"))
    add_to_session(session_id, "assistant", ack_msg)

    # Build Decision object for the handler
    decision = Decision(
        action_type="explore_website",
        target=url,
        description=content,
        reasoning=f"User requested browser action via chat: {content}",
        confidence=0.95,
        risk_level=RiskLevel.LOW,
        requires_action=True,
        parameters={"start_url": url},
    )

    try:
        result = await ooda._explore_website_handler(decision)

        # Send completion summary
        if result and result.get("success"):
            steps = result.get("steps_taken", 0)
            final = result.get("final_result", "")
            summary = f"Done! Completed in {steps} step{'s' if steps != 1 else ''}."
            if final:
                summary += f"\n\n{final}"
            if result.get("skill_created"):
                summary += f"\n\nLearned this as a reusable skill: {result.get('skill_name', 'unnamed')}"
        else:
            status = result.get("status", "unknown") if result else "no result"
            summary = f"Browser exploration finished (status: {status}). Check the browser steps above for details."

        await sio.emit("chat_response", {"type": "text", "content": summary}, room=os.environ.get("CEREBRO_ROOM", "default"))
        add_to_session(session_id, "assistant", summary)
        asyncio.create_task(save_chat_to_memory(session_id, content, f"{ack_msg}\n\n{summary}"))

    except Exception as e:
        error_msg = f"Browser exploration failed: {e}. Falling back to text response..."
        print(f"[Chat->Browser] Error: {e}")
        await sio.emit("chat_response", {"type": "text", "content": error_msg}, room=os.environ.get("CEREBRO_ROOM", "default"))

        # Fall back to Claude Code CLI
        full_response = ""
        async for chunk in process_chat_stream(content, session_id):
            await sio.emit("chat_response", chunk, room=os.environ.get("CEREBRO_ROOM", "default"))
            if chunk.get("type") == "text":
                full_response += chunk.get("content", "")
        if full_response.strip():
            add_to_session(session_id, "assistant", full_response)
        asyncio.create_task(save_chat_to_memory(session_id, content, f"{error_msg}\n\n{full_response}"))


@sio.event
async def chat_message(sid, data):
    """Handle chat messages via WebSocket for real-time streaming."""
    content = data.get("content", "")
    session_id = data.get("session_id") or "default"
    model = data.get("model")
    image_path = data.get("image_path")

    # Validate model if provided
    if model and model not in VALID_MODEL_IDS:
        await sio.emit("chat_response", {"type": "error", "content": f"Invalid model: {model}"}, to=sid)
        return

    # Add user message to session history BEFORE processing
    add_to_session(session_id, "user", content)

    # Broadcast user message to ALL connected devices (cross-device sync!)
    await sio.emit("chat_sync", {
        "type": "user_message",
        "content": content,
        "from_device": sid
    }, room=os.environ.get("CEREBRO_ROOM", "default"))

    # Check for browser action intent — only if client has browser enabled for chat
    browser_enabled = data.get("browser_enabled", False)
    if browser_enabled:
        intent = detect_action_intent(content)
        print(f"[Chat] Intent detection for '{content[:60]}': {intent}")
        if intent and intent["type"] == "browser":
            print(f"[Chat] Routing to browser handler: url={intent.get('url')}")
            asyncio.create_task(_handle_browser_chat_action(sid, session_id, content, intent))
            return

    # Stream response, accumulating full text
    full_response = ""
    async for chunk in process_chat_stream(content, session_id, model=model, image_path=image_path):
        await sio.emit("chat_response", chunk, room=os.environ.get("CEREBRO_ROOM", "default"))
        if chunk.get("type") == "text":
            full_response += chunk.get("content", "")

    # Save assistant response to session + AI Memory
    if full_response.strip():
        add_to_session(session_id, "assistant", full_response)
    asyncio.create_task(save_chat_to_memory(session_id, content, full_response))

@sio.event
async def change_model(sid, data):
    """Handle model change via socket for cross-device sync."""
    model = data.get("model")
    if not model or model not in VALID_MODEL_IDS:
        await sio.emit("chat_response", {"type": "error", "content": f"Invalid model: {model}"}, to=sid)
        return
    config.DEFAULT_MODEL = model
    # Broadcast to ALL devices so they stay in sync
    await sio.emit("model_changed", {"model": model}, room=os.environ.get("CEREBRO_ROOM", "default"))

@sio.event
async def reset_chat(sid, data):
    """Clear session history to start a fresh conversation (socket.io event)."""
    session_id = data.get("session_id") or "default"
    clear_session(session_id)  # Call the helper function
    # Notify all devices that session was cleared
    await sio.emit("session_cleared", {"session_id": session_id}, room=os.environ.get("CEREBRO_ROOM", "default"))


# ============================================================================
# Autonomy Socket.IO Events
# ============================================================================

@sio.event
async def start_autonomy(sid, data):
    """Start cognitive loop via socket."""
    if not cognitive_loop_manager:
        await sio.emit("autonomy_error", {"error": "Cognitive loop not available"}, to=sid)
        return
    level = data.get("level", 2)
    await cognitive_loop_manager.start_loop(level)
    await sio.emit("autonomy_status", cognitive_loop_manager.get_state().to_dict(), room=os.environ.get("CEREBRO_ROOM", "default"))


@sio.event
async def stop_autonomy(sid, data):
    """Stop cognitive loop via socket."""
    if not cognitive_loop_manager:
        return
    reason = data.get("reason", "User stopped via UI")
    await cognitive_loop_manager.stop_loop(reason)


@sio.event
async def emergency_stop(sid, data):
    """Emergency stop via socket."""
    if cognitive_loop_manager:
        await cognitive_loop_manager.emergency_stop()


@sio.event
async def set_autonomy_level(sid, data):
    """Change autonomy level via socket."""
    if not cognitive_loop_manager:
        return
    level = data.get("level", 2)
    await cognitive_loop_manager.set_level(level)


@sio.event
async def approve_autonomy_action(sid, data):
    """Approve a pending action via socket."""
    if not cognitive_loop_manager:
        return
    action_id = data.get("action_id")
    if action_id:
        result = await cognitive_loop_manager.approve_action(action_id)
        await sio.emit("action_approved", result, room=os.environ.get("CEREBRO_ROOM", "default"))


@sio.event
async def reject_autonomy_action(sid, data):
    """Reject a pending action via socket."""
    if not cognitive_loop_manager:
        return
    action_id = data.get("action_id")
    if action_id:
        result = await cognitive_loop_manager.reject_action(action_id)
        await sio.emit("action_rejected", result, room=os.environ.get("CEREBRO_ROOM", "default"))


@sio.event
async def human_input_response(sid, data):
    """Handle human input response via socket (real-time, faster than HTTP)."""
    request_id = data.get("request_id", "")
    answer = data.get("answer", "")

    # Check if this is an agent question (v2.0 HITL)
    if request_id in _agent_questions:
        _agent_answers[request_id] = answer
        _agent_questions[request_id].set()
        await sio.emit("human_input_acknowledged", {"success": True, "request_id": request_id}, room=os.environ.get("CEREBRO_ROOM", "default"))
        return

    # Fall through to existing cognitive loop handler
    if not cognitive_loop_manager:
        await sio.emit("autonomy_error", {"error": "Cognitive loop not available"}, to=sid)
        return
    result = await cognitive_loop_manager.receive_human_response({
        "request_id": request_id,
        "answer": answer,
        "original_question": data.get("original_question", ""),
        "context": data.get("context", ""),
    })
    await sio.emit("human_input_acknowledged", result, room=os.environ.get("CEREBRO_ROOM", "default"))


@sio.event
async def simulation_clarification_response(sid, data):
    """Handle user's response to a simulation clarification request."""
    if not cognitive_loop_manager:
        await sio.emit("autonomy_error", {"error": "Cognitive loop not available"}, to=sid)
        return
    await cognitive_loop_manager.receive_sim_clarification(data)


# ============================================================================
# AI Memory Chat Save (Background)
# ============================================================================

async def save_chat_to_memory(session_id: str, user_content: str, assistant_content: str):
    """Background task: save chat exchange to AI Memory. Fire-and-forget."""
    if not assistant_content or not assistant_content.strip():
        return
    try:
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        metadata = {
            "source": "cerebro",
            "session_type": "chat",
            "cerebro_session_id": session_id,
        }
        result = await mcp_bridge.save_conversation(
            messages=messages,
            session_id=f"cerebro_{session_id}",
            metadata=metadata
        )
        if result.get("success"):
            print(f"[Memory] Saved chat to AI Memory: {result.get('conversation_id')}")
        else:
            print(f"[Memory] Save failed: {result.get('error')}")
    except Exception as e:
        print(f"[Memory] Background save error (non-fatal): {e}")


# ============================================================================
# Claude Agent Integration
# ============================================================================

def _build_chat_prompt(content: str, session_id: str) -> str:
    """Build a prompt with conversation context injected for Claude CLI."""
    data = _load_persistent_session(session_id)
    summary = data.get("summary", "")
    messages = data.get("messages", [])

    parts = [
        "You are Cerebro, Professor's personal AI companion in a persistent chat session.",
        "Respond directly and concisely. You have full system access.",
        _PLATFORM_CONTEXT,
    ]

    if summary:
        parts.append(f"\n## Conversation Summary (earlier messages)\n{summary}")

    recent = messages[-RECENT_MESSAGES_COUNT:]
    if recent:
        parts.append("\n## Recent Conversation")
        for m in recent:
            role = "User" if m["role"] == "user" else "Cerebro"
            parts.append(f"{role}: {m['content'][:2000]}")

    parts.append(f"\n## Current Message\n{content}")
    return "\n".join(parts)


async def process_chat_stream(content: str, session_id: Optional[str] = None, model: Optional[str] = None, image_path: Optional[str] = None):
    """Process a chat message using Claude Code CLI with conversation context."""
    async for chunk in process_with_claude_code(content, session_id=session_id or "default", model=model, image_path=image_path):
        yield chunk

async def process_with_claude_code(content: str, session_id: str = "default", model: Optional[str] = None, image_path: Optional[str] = None) -> AsyncGenerator[dict, None]:
    """
    Process chat using the REAL Claude Code CLI!
    This spawns an actual Claude Code session and streams the output.

    Uses subprocess.Popen with threaded stdout reading to work with Windows
    SelectorEventLoop (required for Playwright compatibility).
    """
    import shutil
    import threading

    # Build context-injected prompt
    prompt = _build_chat_prompt(content, session_id)

    # If an image is attached, prepend instructions
    if image_path:
        prompt = f"[Image attached at: {image_path}]\nUse the Read tool to view this image, then respond.\n\n{prompt}"

    # Find claude executable
    claude_path = shutil.which("claude")
    if not claude_path:
        # Try common locations on Windows
        possible_paths = [
            os.path.join(os.path.expanduser("~"), ".local", "bin", "claude.exe"),
            os.path.join(os.path.expanduser("~"), ".local", "bin", "claude"),
            os.path.join(os.path.expanduser("~"), ".npm-global", "claude.cmd"),
            os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "npm", "claude.cmd"),
            r"C:\Program Files\nodejs\claude.cmd",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                claude_path = p
                break

    # Resolve which model to use (explicit > config default)
    effective_model = model or config.DEFAULT_MODEL

    if not claude_path:
        yield {"type": "text", "content": "Error: Claude Code CLI not found. Make sure it's installed and in PATH."}
        yield {"type": "done", "status": "error", "model": effective_model}
        return

    process = None
    try:
        # Build command with model selection
        cmd_args = [claude_path, "-p", prompt, "--model", effective_model,
                    "--output-format", "stream-json", "--dangerously-skip-permissions", "--verbose"]

        # Strip CLAUDECODE env var so spawned CLI doesn't think it's nested
        agent_env = os.environ.copy()
        agent_env.pop("CLAUDECODE", None)

        # Use subprocess.Popen (works with SelectorEventLoop on Windows)
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=config.AI_MEMORY_PATH,
            env=agent_env,
        )

        # Read stdout in a background thread, push lines to an async queue
        line_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _reader_thread():
            try:
                for raw_line in process.stdout:
                    loop.call_soon_threadsafe(line_queue.put_nowait, raw_line)
            except Exception as e:
                loop.call_soon_threadsafe(line_queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(line_queue.put_nowait, None)

        reader = threading.Thread(target=_reader_thread, daemon=True)
        reader.start()

        full_response = ""

        # Stream lines from the queue
        while True:
            try:
                line = await asyncio.wait_for(line_queue.get(), timeout=1800)  # 30 min for long tasks
            except asyncio.TimeoutError:
                print("[Chat] Readline timeout after 30 minutes")
                break

            if line is None:
                break
            if isinstance(line, Exception):
                print(f"[Chat] Reader thread error: {line}")
                break

            try:
                # Parse JSON line
                data = json.loads(line.decode('utf-8', errors='replace').strip())

                msg_type = data.get("type", "")

                if msg_type == "assistant":
                    # Assistant text message
                    text = data.get("message", {}).get("content", "")
                    if isinstance(text, list):
                        # Handle content blocks
                        for block in text:
                            if block.get("type") == "text":
                                chunk = block.get("text", "")
                                full_response += chunk
                                yield {"type": "text", "content": chunk}
                            elif block.get("type") == "tool_use":
                                tool_name = block.get("name", "unknown")
                                yield {"type": "tool", "name": tool_name, "status": "running"}
                    elif isinstance(text, str) and text:
                        full_response += text
                        yield {"type": "text", "content": text}

                elif msg_type == "tool_use":
                    tool_name = data.get("tool", data.get("name", "tool"))
                    yield {"type": "tool", "name": tool_name, "status": "running"}

                elif msg_type == "tool_result":
                    tool_name = data.get("tool", "tool")
                    yield {"type": "tool_result", "name": tool_name, "status": "done"}

                elif msg_type == "result":
                    # Final result
                    result_text = data.get("result", "")
                    if result_text and result_text not in full_response:
                        yield {"type": "text", "content": result_text}

                elif msg_type == "error":
                    error_msg = data.get("error", {}).get("message", str(data))
                    yield {"type": "text", "content": f"Error: {error_msg}"}

            except json.JSONDecodeError:
                # Not JSON, might be raw text
                text = line.decode('utf-8', errors='replace').strip()
                if text and not text.startswith('{'):
                    yield {"type": "text", "content": text}

        # Wait for process to complete
        try:
            process.wait(timeout=300)  # 5 min max
        except subprocess.TimeoutExpired:
            process.kill()
            yield {"type": "text", "content": "\n\nClaude Code timed out after 5 minutes."}

        # Check for errors (but only report if we got no output)
        if process.returncode != 0 and not full_response:
            stderr = process.stderr.read()
            if stderr:
                yield {"type": "text", "content": f"\n\nClaude Code exited with error: {stderr.decode('utf-8', errors='replace')}"}

    except FileNotFoundError:
        yield {"type": "text", "content": "Error: Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"}
    except Exception as e:
        yield {"type": "text", "content": f"Error running Claude Code: {type(e).__name__}: {str(e)}"}
    finally:
        # ALWAYS signal completion to frontend - this fixes the stuck "Running:" indicator
        yield {"type": "done", "status": "complete", "model": effective_model}
        # Clean up process if still running
        if process and process.returncode is None:
            try:
                process.kill()
            except:
                pass


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {"name": "Cerebro", "status": "online", "version": "1.5.3"}

def _get_user_profile_path() -> Path:
    return Path(config.AI_MEMORY_PATH) / "user_profile.json"

def _load_user_profile() -> dict:
    """Load user profile from persistent storage."""
    p = _get_user_profile_path()
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except:
            pass
    return {}

def _save_user_profile(profile: dict):
    """Save user profile to persistent storage."""
    p = _get_user_profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(profile, f, indent=2)

@app.get("/auth/status")
async def auth_status():
    """Check if initial setup is complete. No auth required."""
    profile = _load_user_profile()
    return {
        "setup_complete": profile.get("setup_complete", False),
        "username": profile.get("name", "")
    }

@app.post("/auth/setup")
async def auth_setup(request: SetupRequest):
    """First-time setup — save profile and set password. No auth required."""
    profile = _load_user_profile()
    if profile.get("setup_complete"):
        raise HTTPException(status_code=400, detail="Setup already completed")

    profile.update({
        "name": request.name,
        "password": request.password,
        "use_cases": request.use_cases,
        "style": request.style,
        "setup_complete": True,
        "created_at": datetime.now().isoformat()
    })
    _save_user_profile(profile)

    token = create_token(request.name)
    return {"token": token, "user": request.name}

@app.post("/auth/login")
async def login(request: LoginRequest):
    """Password login — checks stored profile first, then env var fallback."""
    profile = _load_user_profile()
    stored_password = profile.get("password") or os.environ.get("CEREBRO_PASSWORD", "professor")
    username = profile.get("name", "User")

    if request.password != stored_password:
        raise HTTPException(status_code=401, detail="Invalid password")

    token = create_token(username)
    return {"token": token, "user": username}

@app.get("/briefing", response_model=BriefingResponse)
async def get_briefing(user: str = Depends(verify_token)):
    """Get morning briefing with pending tasks and suggestions."""
    now = datetime.now()
    hour = now.hour

    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    # Get active work from quick_facts
    pending_tasks = []
    try:
        quick_facts_path = Path(config.AI_MEMORY_PATH) / "quick_facts.json"
        if quick_facts_path.exists():
            with open(quick_facts_path) as f:
                facts = json.load(f)
                if "active_work" in facts and facts["active_work"]:
                    work = facts["active_work"]
                    pending_tasks.append({
                        "title": work.get("project", "Unknown project"),
                        "description": work.get("next_action", "Continue working"),
                        "phase": work.get("current_phase", ""),
                        "can_auto_continue": True
                    })
    except:
        pass

    # Generate suggestions based on time and context
    suggestions = [
        {
            "action": "Check AI Memory health",
            "reason": "Ensure all systems are running smoothly"
        },
        {
            "action": "Review recent learnings",
            "reason": "Consolidate knowledge from recent sessions"
        }
    ]

    # Memory stats - count from actual NAS memory, not local cache
    memory_stats = {"conversations": 0, "facts": 0, "learnings": 0}
    try:
        nas_memory = Path(config.AI_MEMORY_PATH)
        if nas_memory.exists():
            conv_path = nas_memory / "conversations"
            if conv_path.exists():
                memory_stats["conversations"] = len(list(conv_path.glob("*.json")))
            facts_path = nas_memory / "facts"
            if facts_path.exists():
                memory_stats["facts"] = len(list(facts_path.glob("*.json")))
            learnings_path = nas_memory / "learnings"
            if learnings_path.exists():
                memory_stats["learnings"] = len(list(learnings_path.glob("*.json")))
    except:
        pass

    profile = _load_user_profile()
    username = profile.get("name", "")
    greeting_text = f"{greeting}, {username}" if username else greeting

    return BriefingResponse(
        greeting=greeting_text,
        time=now.strftime("%A, %B %d, %Y at %I:%M %p"),
        pending_tasks=pending_tasks,
        suggestions=suggestions,
        memory_stats=memory_stats
    )

# ============================================================================
# Model Selection API
# ============================================================================

@app.get("/api/models")
async def get_models():
    """Return available Claude models for the model selector."""
    return {"models": AVAILABLE_MODELS, "default": config.DEFAULT_MODEL}

@app.get("/api/models/current")
async def get_current_model():
    """Get the current default model."""
    return {"model": config.DEFAULT_MODEL}

class SetModelRequest(BaseModel):
    model: str

@app.post("/api/models/current")
async def set_current_model(request: SetModelRequest):
    """Set the default model (persisted in config for this session)."""
    if request.model not in VALID_MODEL_IDS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {request.model}. Valid models: {list(VALID_MODEL_IDS)}")
    config.DEFAULT_MODEL = request.model
    # Broadcast model change to all connected devices for cross-device sync
    await sio.emit("model_changed", {"model": request.model}, room=os.environ.get("CEREBRO_ROOM", "default"))
    return {"model": config.DEFAULT_MODEL, "status": "updated"}

# ============================================================================
# Chat Endpoints
# ============================================================================

@app.post("/chat")
async def chat(message: ChatMessage, user: str = Depends(verify_token)):
    """Send a chat message (non-streaming, for simple requests)."""
    original_session_id = message.session_id or "default"

    # Validate model if provided
    if message.model and message.model not in VALID_MODEL_IDS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {message.model}. Valid models: {list(VALID_MODEL_IDS)}")

    # Add user message to session history BEFORE processing
    add_to_session(original_session_id, "user", message.content)

    responses = []
    async for chunk in process_chat_stream(message.content, original_session_id, model=message.model):
        responses.append(chunk)

    # Combine text responses
    text_content = " ".join([r["content"] for r in responses if r.get("type") == "text"])
    session_id = next((r["session_id"] for r in responses if r.get("type") == "session"), None)

    # Save to session history + AI Memory
    if text_content.strip():
        add_to_session(original_session_id, "assistant", text_content)
    asyncio.create_task(save_chat_to_memory(original_session_id, message.content, text_content))

    return {
        "content": text_content,
        "session_id": session_id
    }

# ============================================================================
# Chat History Persistence
# ============================================================================
CHAT_HISTORY_PATH = Path(config.AI_MEMORY_PATH) / "cerebro" / "chat_history"

class ChatHistoryMessage(BaseModel):
    type: str
    content: str
    timestamp: Optional[int] = None

class ChatHistoryRequest(BaseModel):
    messages: list[ChatHistoryMessage]
    session_id: Optional[str] = None

@app.get("/chat/history")
async def get_chat_history(user: str = Depends(verify_token)):
    """Get saved chat history for the user."""
    try:
        CHAT_HISTORY_PATH.mkdir(parents=True, exist_ok=True)
        history_file = CHAT_HISTORY_PATH / f"{user}_history.json"

        if not history_file.exists():
            return {"messages": [], "session_id": None}

        data = json.loads(history_file.read_text())
        return {
            "messages": data.get("messages", []),
            "session_id": data.get("session_id")
        }
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return {"messages": [], "session_id": None}

@app.post("/chat/history")
async def save_chat_history(request: ChatHistoryRequest, user: str = Depends(verify_token)):
    """Save chat history for the user."""
    try:
        CHAT_HISTORY_PATH.mkdir(parents=True, exist_ok=True)
        history_file = CHAT_HISTORY_PATH / f"{user}_history.json"

        # Convert messages to dict format
        messages = [msg.model_dump() for msg in request.messages]

        data = {
            "messages": messages,
            "session_id": request.session_id,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "message_count": len(messages)
        }

        history_file.write_text(json.dumps(data, indent=2))
        return {"success": True, "saved": len(messages)}
    except Exception as e:
        print(f"Error saving chat history: {e}")
        return {"success": False, "error": str(e)}

@app.delete("/chat/history")
async def clear_chat_history(user: str = Depends(verify_token)):
    """Clear chat history for the user."""
    try:
        CHAT_HISTORY_PATH.mkdir(parents=True, exist_ok=True)
        history_file = CHAT_HISTORY_PATH / f"{user}_history.json"

        if history_file.exists():
            history_file.unlink()

        return {"success": True}
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        return {"success": False, "error": str(e)}

@app.delete("/chat/session/{session_id}")
async def clear_chat_session(session_id: str, user: str = Depends(verify_token)):
    """Clear persistent session history (archives the file) to start fresh."""
    clear_session(session_id)
    return {"success": True, "message": f"Session '{session_id}' cleared - new conversation started"}

@app.get("/chat/session/{session_id}")
async def get_chat_session(session_id: str, user: str = Depends(verify_token)):
    """Get persistent session history."""
    data = _load_persistent_session(session_id)
    return {
        "session_id": session_id,
        "message_count": len(data["messages"]),
        "messages": data["messages"][-20:],
        "summary": data.get("summary", ""),
    }

@app.get("/chat/sessions")
async def list_chat_sessions(user: str = Depends(verify_token)):
    """List all persistent chat sessions."""
    sessions = []
    for f in CHAT_SESSION_DIR.glob("*.json"):
        if "_archived_" in f.name:
            continue
        try:
            d = json.loads(f.read_text())
            sessions.append({
                "session_id": d.get("session_id", f.stem),
                "message_count": len(d.get("messages", [])),
                "last_updated": d.get("last_updated", ""),
            })
        except (json.JSONDecodeError, OSError):
            pass
    return {"sessions": sessions}

@app.post("/task")
async def create_task(request: TaskRequest, user: str = Depends(verify_token)):
    """Create a background task for the agent to work on."""
    task_id = str(uuid.uuid4())

    # Store task in Redis
    task_data = {
        "id": task_id,
        "prompt": request.prompt,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": 0
    }

    await redis.set(f"task_data:{task_id}", json.dumps(task_data))

    if request.background:
        # Queue for background processing
        await redis.lpush("task_queue", json.dumps(task_data))

    return {"task_id": task_id, "status": "queued"}

@app.get("/task/{task_id}")
async def get_task(task_id: str, user: str = Depends(verify_token)):
    """Get task status."""
    data = await redis.get(f"task_data:{task_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Task not found")

    return json.loads(data)

@app.get("/health")
async def health():
    """Health check endpoint."""
    redis_ok = False
    try:
        await redis.ping()
        redis_ok = True
    except:
        pass

    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis": redis_ok,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/mood")
async def get_mood():
    """Get current mood from NAS file - for Cerebro dashboard integration."""
    mood_file = Path(config.AI_MEMORY_PATH) / "mood" / "current_mood.json"
    try:
        if mood_file.exists():
            with open(mood_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "tracking_enabled": False,
                "face_detected": False,
                "dominant_emotion": None,
                "confidence": 0,
                "mapped_state": None,
                "error": "Mood file not found"
            }
    except Exception as e:
        return {
            "tracking_enabled": False,
            "face_detected": False,
            "dominant_emotion": None,
            "confidence": 0,
            "error": str(e)
        }


@app.get("/scheduler/debug")
async def scheduler_debug():
    """Debug endpoint to check APScheduler state."""
    if not SCHEDULER_AVAILABLE or not scheduler:
        return {"error": "Scheduler not available", "SCHEDULER_AVAILABLE": SCHEDULER_AVAILABLE}

    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger)
        })

    return {
        "scheduler_running": scheduler.running,
        "jobs_count": len(jobs),
        "jobs": jobs,
        "current_time": datetime.now().isoformat(),
        "current_time_utc": datetime.now(timezone.utc).isoformat()
    }

# ============================================================================
# Agent API Endpoints
# ============================================================================

class AgentRequest(BaseModel):
    task: str
    agent_type: str = "worker"
    # Enhanced fields
    context: Optional[str] = None  # Background information
    expected_output: Optional[str] = None  # What format/content is expected
    priority: Optional[str] = "normal"  # low, normal, high, critical
    resources: Optional[list[str]] = None  # File paths or URLs to reference
    # Context references (click-to-attach)
    context_refs: Optional[list[dict]] = None  # List of {chunk_id, conversation_id, preview, fallback_content}
    # Multi-agent workflow
    parent_workflow_id: Optional[str] = None  # If part of a workflow
    parent_agent_id: Optional[str] = None  # If spawned by another agent
    # Directive linking (for auto-completion when agent finishes)
    directive_id: Optional[str] = None  # The directive this agent was spawned for
    # Timeout (seconds): 0 = unlimited, default 1 hour
    timeout: Optional[int] = 3600
    # Context fusion: agent IDs to reference for merged context
    source_agents: Optional[list[str]] = None
    # Project assignment
    project_id: Optional[str] = None
    # Model selection (sonnet, opus, haiku, or full model ID)
    model: Optional[str] = "sonnet"
    # Special Ops mission config
    specops_config: Optional[dict] = None  # {mission_name, work_style, cycle_interval, mission_duration, sub_role}

class WorkflowRequest(BaseModel):
    task: str
    workflow_type: str = "standard"  # standard, parallel, sequential
    agent_composition: Optional[list[str]] = None  # e.g. ["researcher", "coder", "worker"]

@app.post("/agents")
async def spawn_agent(request: AgentRequest, user: str = Depends(verify_token)):
    """Spawn a new background agent."""
    # Pass specops_config via thread-local-style attribute for create_agent to pick up
    if request.specops_config:
        create_agent._specops_config = request.specops_config
    else:
        create_agent._specops_config = None
    agent_id = await create_agent(
        task=request.task,
        agent_type=request.agent_type,
        context=request.context,
        expected_output=request.expected_output,
        priority=request.priority,
        resources=request.resources,
        parent_workflow_id=request.parent_workflow_id,
        parent_agent_id=request.parent_agent_id,
        context_refs=request.context_refs,
        directive_id=request.directive_id,  # Link to directive for auto-completion
        timeout=request.timeout or 3600,
        source_agents=request.source_agents,
        project_id=request.project_id,
        model=request.model or "sonnet"
    )
    create_agent._specops_config = None  # Clean up
    return {"agent_id": agent_id, "status": "spawned"}

@app.get("/agents")
async def list_agents(user: str = Depends(verify_token)):
    """List all agents (active in-memory and recent persisted)."""
    # Start with active in-memory agents
    agents_dict = {a["id"]: sanitize_agent_for_emit(a) for a in active_agents.values()}

    # Also load recent agents from persistence (in case of server restart)
    try:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            # Add recent non-archived agents from history (if not already in active_agents)
            for agent in index.get("agents", [])[:50]:  # Last 50 agents
                if agent["id"] not in agents_dict and not agent.get("archived", False):
                    agents_dict[agent["id"]] = agent
    except Exception as e:
        print(f"[Agents] Error loading persisted agents: {e}")

    return {"agents": list(agents_dict.values())}


@app.get("/agents/projects")
async def list_agent_projects(user: str = Depends(verify_token)):
    """List all projects with agent counts. Groups agents by project_id."""
    try:
        # Collect all agents from memory and index
        all_agents = {}
        for a in active_agents.values():
            all_agents[a["id"]] = a

        # Merge with index for persisted agents not in memory
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            for entry in index.get("agents", []):
                if entry["id"] not in all_agents:
                    all_agents[entry["id"]] = entry

        # Load project tracker for metadata
        tracker_path = Path("/mnt/nas/AI_MEMORY/projects/tracker.json")
        tracker_projects = {}
        if tracker_path.exists():
            try:
                with open(tracker_path, 'r', encoding='utf-8') as f:
                    tracker_data = json.load(f)
                # tracker.json is a dict keyed by project name
                for key, proj in tracker_data.items():
                    if not isinstance(proj, dict):
                        continue
                    pid = proj.get("project_id") or proj.get("id") or key
                    if pid:
                        tracker_projects[pid] = proj
            except Exception:
                pass

        # Group agents by project_id
        project_groups = {}  # project_id -> list of agents
        for agent in all_agents.values():
            pid = agent.get("project_id") or "uncategorized"
            if pid not in project_groups:
                project_groups[pid] = []
            project_groups[pid].append(agent)

        # Build response
        projects = []
        for pid, agents in project_groups.items():
            tracker_info = tracker_projects.get(pid, {})
            running = sum(1 for a in agents if a.get("status") == "running")

            # Find latest activity
            timestamps = []
            for a in agents:
                for ts_field in ("completed_at", "started_at", "created_at"):
                    ts = a.get(ts_field)
                    if ts:
                        timestamps.append(ts)
                        break
            latest = max(timestamps) if timestamps else None

            projects.append({
                "project_id": pid,
                "name": tracker_info.get("name", pid if pid != "uncategorized" else "Uncategorized"),
                "status": tracker_info.get("status", "active"),
                "technologies": tracker_info.get("technologies", []),
                "agent_count": len(agents),
                "running_count": running,
                "latest_activity": latest,
            })

        # Sort: most recent activity first, uncategorized always last
        categorized = [p for p in projects if p["project_id"] != "uncategorized"]
        uncategorized = [p for p in projects if p["project_id"] == "uncategorized"]
        categorized.sort(key=lambda p: p["latest_activity"] or "", reverse=True)
        projects = categorized + uncategorized

        return {"projects": projects}
    except Exception as e:
        return {"projects": [], "error": str(e)}


class ProjectAssignRequest(BaseModel):
    project_id: str


@app.post("/agents/{agent_id}/project")
async def assign_agent_project(agent_id: str, request: ProjectAssignRequest, user: str = Depends(verify_token)):
    """Manually assign an agent to a project."""
    project_id = request.project_id

    # Update in-memory agent
    if agent_id in active_agents:
        active_agents[agent_id]["project_id"] = project_id

    # Update persisted JSON file
    try:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        for date_dir in agents_dir.iterdir():
            if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                agent_file = date_dir / f"{agent_id}.json"
                if agent_file.exists():
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        agent_data = json.load(f)
                    agent_data["project_id"] = project_id
                    with open(agent_file, 'w', encoding='utf-8') as f:
                        json.dump(agent_data, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    break

        # Update index entry
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            for entry in index.get("agents", []):
                if entry["id"] == agent_id:
                    entry["project_id"] = project_id
                    break
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

        # Emit WebSocket event
        await sio.emit("agent_project_assigned", {
            "agent_id": agent_id,
            "project_id": project_id,
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

        return {"status": "ok", "agent_id": agent_id, "project_id": project_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


class CreateProjectRequest(BaseModel):
    name: str


class RenameProjectRequest(BaseModel):
    name: str


@app.post("/agents/projects/create")
async def create_project(request: CreateProjectRequest, user: str = Depends(verify_token)):
    """Create a new agent project group."""
    try:
        project_id = request.name.lower().replace(" ", "-").replace("_", "-")
        # Remove non-alphanumeric chars except hyphens
        project_id = "".join(c for c in project_id if c.isalnum() or c == "-").strip("-")
        if not project_id:
            return {"status": "error", "error": "Invalid project name"}

        tracker_path = Path("/mnt/nas/AI_MEMORY/projects/tracker.json")
        tracker_data = {}
        if tracker_path.exists():
            try:
                with open(tracker_path, "r", encoding="utf-8") as f:
                    tracker_data = json.load(f)
            except Exception:
                pass

        if project_id in tracker_data:
            return {"status": "error", "error": "Project already exists"}

        tracker_data[project_id] = {
            "project_id": project_id,
            "name": request.name,
            "status": "active",
            "technologies": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tracker_path, "w", encoding="utf-8") as f:
            json.dump(tracker_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        await sio.emit("project_created", {
            "project_id": project_id,
            "name": request.name,
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

        return {"status": "ok", "project_id": project_id, "name": request.name}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/agents/projects/{project_id}/rename")
async def rename_project(project_id: str, request: RenameProjectRequest, user: str = Depends(verify_token)):
    """Rename an agent project group."""
    try:
        tracker_path = Path("/mnt/nas/AI_MEMORY/projects/tracker.json")
        tracker_data = {}
        if tracker_path.exists():
            with open(tracker_path, "r", encoding="utf-8") as f:
                tracker_data = json.load(f)

        if project_id in tracker_data:
            tracker_data[project_id]["name"] = request.name
        else:
            tracker_data[project_id] = {
                "project_id": project_id,
                "name": request.name,
                "status": "active",
                "technologies": [],
            }

        with open(tracker_path, "w", encoding="utf-8") as f:
            json.dump(tracker_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        await sio.emit("project_renamed", {
            "project_id": project_id,
            "name": request.name,
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

        return {"status": "ok", "project_id": project_id, "name": request.name}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.delete("/agents/projects/{project_id}")
async def delete_project(project_id: str, user: str = Depends(verify_token)):
    """Delete a project group and reassign its agents to uncategorized."""
    try:
        reassigned = 0

        # Reassign all agents with this project_id
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        for a in active_agents.values():
            if a.get("project_id") == project_id:
                a["project_id"] = ""
                reassigned += 1

        # Update persisted agent files
        if agents_dir.exists():
            for date_dir in agents_dir.iterdir():
                if date_dir.is_dir() and not date_dir.name.endswith(".json"):
                    for agent_file in date_dir.glob("*.json"):
                        try:
                            with open(agent_file, "r", encoding="utf-8") as f:
                                agent_data = json.load(f)
                            if agent_data.get("project_id") == project_id:
                                agent_data["project_id"] = ""
                                with open(agent_file, "w", encoding="utf-8") as f:
                                    json.dump(agent_data, f, indent=2, ensure_ascii=False)
                                    f.flush()
                                    os.fsync(f.fileno())
                        except Exception:
                            continue

        # Update index
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
            for entry in index.get("agents", []):
                if entry.get("project_id") == project_id:
                    entry["project_id"] = ""
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

        # Remove from tracker
        tracker_path = Path("/mnt/nas/AI_MEMORY/projects/tracker.json")
        if tracker_path.exists():
            with open(tracker_path, "r", encoding="utf-8") as f:
                tracker_data = json.load(f)
            tracker_data.pop(project_id, None)
            with open(tracker_path, "w", encoding="utf-8") as f:
                json.dump(tracker_data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

        await sio.emit("project_deleted", {
            "project_id": project_id,
            "reassigned_count": reassigned,
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

        return {"status": "ok", "project_id": project_id, "reassigned_count": reassigned}
    except Exception as e:
        return {"status": "error", "error": str(e)}


class ClearAgentsRequest(BaseModel):
    exclude_ids: list[str] = []


@app.post("/agents/clear")
async def clear_agents(request: ClearAgentsRequest, user: str = Depends(verify_token)):
    """Clear all agents except starred/excluded ones."""
    try:
        exclude = set(request.exclude_ids)
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        deleted_count = 0
        preserved_count = 0

        # Remove non-excluded agents from memory
        to_remove = [aid for aid in active_agents if aid not in exclude]
        for aid in to_remove:
            del active_agents[aid]
            used_call_signs.discard(aid)
            deleted_count += 1

        # Remove non-excluded agent files, preserve excluded
        if agents_dir.exists():
            for date_dir in agents_dir.iterdir():
                if date_dir.is_dir() and not date_dir.name.endswith(".json"):
                    for agent_file in list(date_dir.glob("*.json")):
                        agent_id = agent_file.stem
                        if agent_id in exclude:
                            preserved_count += 1
                            continue
                        agent_file.unlink()
                    # Remove empty date dirs
                    remaining = list(date_dir.glob("*.json"))
                    if not remaining:
                        import shutil
                        shutil.rmtree(date_dir)

            # Rebuild index with only preserved agents
            index_file = agents_dir / "index.json"
            if index_file.exists():
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
                index["agents"] = [e for e in index.get("agents", []) if e.get("id") in exclude]
                index["last_updated"] = datetime.now(timezone.utc).isoformat()
                with open(index_file, "w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())

        await sio.emit("agents_cleared", {
            "deleted_count": deleted_count,
            "preserved_count": preserved_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

        return {"status": "ok", "deleted": deleted_count, "preserved": preserved_count}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/agents/wipe")
async def wipe_all_agents(user: str = Depends(verify_token)):
    """Delete all agent data (date subdirs + index) but preserve other AI Memory data."""
    try:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        deleted_count = 0

        if agents_dir.exists():
            # Delete all date subdirectories (contain agent JSON files)
            for item in list(agents_dir.iterdir()):
                if item.is_dir():
                    import shutil
                    shutil.rmtree(item)
                    deleted_count += 1

            # Reset index.json
            index_file = agents_dir / "index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump({"agents": [], "last_updated": datetime.now(timezone.utc).isoformat()}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

        # Clear in-memory agents and call sign tracking
        active_agents.clear()
        used_call_signs.clear()

        # Notify frontend
        await sio.emit("agents_wiped", {
            "deleted_dirs": deleted_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

        print(f"[Wipe] Deleted {deleted_count} date directories, reset index, cleared active_agents")
        return {"status": "ok", "deleted_dirs": deleted_count}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/agents/backfill-projects")
async def backfill_agent_projects(user: str = Depends(verify_token)):
    """One-time backfill: auto-categorize all uncategorized agents via Ollama."""
    try:
        categorized = 0
        failed = 0
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        index_file = agents_dir / "index.json"

        if not index_file.exists():
            return {"status": "ok", "categorized": 0, "message": "No agents index found"}

        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)

        for entry in index.get("agents", []):
            if entry.get("project_id") and entry["project_id"] != "uncategorized":
                continue  # Already categorized

            # Load full agent data if possible
            agent_file_rel = entry.get("file")
            agent_data = entry
            if agent_file_rel:
                full_path = agents_dir / agent_file_rel
                if full_path.exists():
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            agent_data = json.load(f)
                    except Exception:
                        pass

            try:
                await auto_categorize_agent(agent_data)
                if agent_data.get("project_id"):
                    categorized += 1
            except Exception:
                failed += 1

        return {"status": "ok", "categorized": categorized, "failed": failed}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/agents/search")
async def search_agents(
    q: Optional[str] = None,
    sort: Optional[str] = "activity",
    project_id: Optional[str] = None,
    user: str = Depends(verify_token),
):
    """Search, sort, and filter agents. Merges active + persisted."""
    try:
        # Merge active_agents + index
        all_agents = {}
        for a in active_agents.values():
            all_agents[a["id"]] = sanitize_agent_for_emit(a)

        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            for entry in index.get("agents", []):
                if entry["id"] not in all_agents:
                    all_agents[entry["id"]] = entry

        results = list(all_agents.values())

        # Filter by project_id
        if project_id:
            if project_id == "uncategorized":
                results = [a for a in results if not a.get("project_id")]
            else:
                results = [a for a in results if a.get("project_id") == project_id]

        # Text search on task, call_sign, project_id
        if q:
            q_lower = q.lower()
            results = [
                a for a in results
                if q_lower in (a.get("task") or "").lower()
                or q_lower in (a.get("call_sign") or "").lower()
                or q_lower in (a.get("project_id") or "").lower()
                or q_lower in (a.get("type") or "").lower()
            ]

        # Sort
        if sort == "activity":
            results.sort(
                key=lambda a: a.get("completed_at") or a.get("started_at") or a.get("created_at") or "",
                reverse=True,
            )
        elif sort == "created":
            results.sort(key=lambda a: a.get("created_at") or "", reverse=True)
        elif sort == "name":
            results.sort(key=lambda a: (a.get("call_sign") or "").lower())

        return {"agents": results, "total": len(results)}
    except Exception as e:
        return {"agents": [], "total": 0, "error": str(e)}


@app.get("/agents/history")
async def get_agent_history(limit: int = 20, include_archived: bool = False, user: str = Depends(verify_token)):
    """Get agent history from AI Memory persistence."""
    try:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        index_file = agents_dir / "index.json"

        if not index_file.exists():
            return {"agents": [], "archived": [], "total": 0}

        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)

        all_agents = index.get("agents", [])

        # Split into active and archived
        active_agents_list = [a for a in all_agents if not a.get('archived', False)]
        archived_agents_list = [a for a in all_agents if a.get('archived', False)]

        result = {
            "agents": active_agents_list[:limit],
            "total": len(active_agents_list),
            "archived_count": len(archived_agents_list)
        }

        if include_archived:
            result["archived"] = archived_agents_list[:limit]

        return result

    except Exception as e:
        return {"agents": [], "archived": [], "total": 0, "error": str(e)}

@app.get("/agents/history/{agent_id}")
async def get_agent_history_detail(agent_id: str, user: str = Depends(verify_token)):
    """Get full agent details from history."""
    try:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"

        # Search for the agent file in date directories (newest first!)
        date_dirs = sorted(
            [d for d in agents_dir.iterdir() if d.is_dir() and not d.name.endswith('.json')],
            key=lambda d: d.name,
            reverse=True  # Newest dates first (2026-02-01 before 2026-01-31)
        )

        for date_dir in date_dirs:
            agent_file = date_dir / f"{agent_id}.json"
            if agent_file.exists():
                with open(agent_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

        raise HTTPException(status_code=404, detail="Agent not found in history")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/queue")
async def get_agent_queue(user: str = Depends(verify_token)):
    """Get agent queue status and contents."""
    return {
        "max_concurrent": MAX_CONCURRENT_AGENTS,
        "running": _count_running_agents(),
        "queued": len(_agent_spawn_queue),
        "queue": [{"agent_id": e["agent_id"], "queued_at": e["queued_at"], "position": i+1} for i, e in enumerate(_agent_spawn_queue)],
    }

@app.get("/agents/cerebro")
async def list_cerebro_agents(user: str = Depends(verify_token)):
    """List only Cerebro-spawned and scheduler agents (not manually created)."""
    cerebro_agents = [
        sanitize_agent_for_emit(a) for a in active_agents.values()
        if a.get("source") in ("cerebro", "scheduler")
    ]
    return {"agents": cerebro_agents, "queue": get_agent_queue_info()}

def get_agent_queue_info() -> dict:
    """Get queue info without auth (internal helper)."""
    return {
        "max_concurrent": MAX_CONCURRENT_AGENTS,
        "running": _count_running_agents(),
        "queued": len(_agent_spawn_queue),
    }

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str, user: str = Depends(verify_token)):
    """Get specific agent status."""
    agent = active_agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.delete("/agents/{agent_id}")
async def stop_agent(agent_id: str, user: str = Depends(verify_token)):
    """Gracefully stop a running agent — kills process, captures partial output, persists to NAS."""
    agent = active_agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    was_running = agent.get("status") in (AgentStatus.RUNNING, AgentStatus.QUEUED)

    # Remove from queue if it was queued (not yet running)
    global _agent_spawn_queue
    _agent_spawn_queue = [e for e in _agent_spawn_queue if e["agent_id"] != agent_id]

    # Gracefully terminate the subprocess
    process = agent.get("_process")
    pid = agent.get("pid")
    if process and process.poll() is None:
        try:
            process.terminate()  # SIGTERM — gives Claude CLI time to flush
            try:
                process.wait(timeout=10)
            except sp.TimeoutExpired:
                process.kill()  # Force kill if it didn't exit
                process.wait(timeout=5)
            print(f"[Stop] Agent {agent_id} process terminated gracefully (PID: {pid})")
        except Exception as e:
            print(f"[Stop] Error terminating agent {agent_id} process: {e}")
    elif pid and was_running:
        # Process object not available, try via PID
        try:
            import signal as sig_mod
            os.kill(pid, sig_mod.SIGTERM)
            # Give it time to flush
            await asyncio.sleep(3)
            try:
                os.kill(pid, 0)  # Check if still alive
                os.kill(pid, sig_mod.SIGKILL)  # Force kill
            except ProcessLookupError:
                pass  # Already exited
            print(f"[Stop] Agent {agent_id} killed via PID {pid}")
        except ProcessLookupError:
            print(f"[Stop] Agent {agent_id} PID {pid} already dead")
        except Exception as e:
            print(f"[Stop] Error killing agent {agent_id} PID {pid}: {e}")

    # Capture any partial output already collected
    partial_output = agent.get("output") or ""
    if not partial_output:
        # Try to get output from the process stdout buffer
        if process and hasattr(process, 'stdout') and process.stdout:
            try:
                remaining = process.stdout.read()
                if remaining:
                    partial_output = remaining if isinstance(remaining, str) else remaining.decode('utf-8', errors='replace')
            except Exception:
                pass

    # Mark as stopped (not failed — this was intentional)
    agent["status"] = AgentStatus.STOPPED
    agent["error"] = "Stopped by user"
    agent["completed_at"] = datetime.now(timezone.utc).isoformat()
    if partial_output:
        agent["output"] = partial_output

    # Persist to NAS so output is queryable later
    try:
        await save_agent_to_memory(agent)
        print(f"[Stop] Agent {agent_id} persisted to NAS (output: {len(partial_output)} chars)")
    except Exception as e:
        print(f"[Stop] Failed to persist agent {agent_id}: {e}")

    # Emit update to frontend
    await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))

    # Unregister from cognitive loop
    if cognitive_loop_manager:
        cognitive_loop_manager.register_agent_completed(agent_id)

    # Process queue — stopped agent frees a slot
    await _process_agent_queue()

    return {"status": "stopped", "agent_id": agent_id, "output_saved": bool(partial_output)}


@app.post("/agents/stop-all")
async def stop_all_agents(user: str = Depends(verify_token)):
    """Gracefully stop ALL running/queued agents — persists partial output for each."""
    running = [
        (aid, a) for aid, a in active_agents.items()
        if a.get("status") in (AgentStatus.RUNNING, AgentStatus.QUEUED)
    ]
    if not running:
        return {"status": "ok", "stopped": 0, "message": "No running agents"}

    stopped = 0
    for agent_id, agent in running:
        try:
            # Terminate process
            process = agent.get("_process")
            pid = agent.get("pid")
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except sp.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            elif pid:
                try:
                    import signal as sig_mod
                    os.kill(pid, sig_mod.SIGTERM)
                except ProcessLookupError:
                    pass
                except Exception:
                    pass

            # Capture partial output
            partial_output = agent.get("output") or ""
            if not partial_output and process and hasattr(process, 'stdout') and process.stdout:
                try:
                    remaining = process.stdout.read()
                    if remaining:
                        partial_output = remaining if isinstance(remaining, str) else remaining.decode('utf-8', errors='replace')
                except Exception:
                    pass

            # Mark as stopped and persist
            agent["status"] = AgentStatus.STOPPED
            agent["error"] = "Stopped by user (stop all)"
            agent["completed_at"] = datetime.now(timezone.utc).isoformat()
            if partial_output:
                agent["output"] = partial_output

            try:
                await save_agent_to_memory(agent)
            except Exception as e:
                print(f"[StopAll] Failed to persist agent {agent_id}: {e}")

            await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))

            if cognitive_loop_manager:
                cognitive_loop_manager.register_agent_completed(agent_id)

            stopped += 1
        except Exception as e:
            print(f"[StopAll] Error stopping agent {agent_id}: {e}")

    # Clear queue
    global _agent_spawn_queue
    _agent_spawn_queue = []

    # Process queue (empty now, but resets state)
    await _process_agent_queue()

    return {"status": "ok", "stopped": stopped, "total_were_running": len(running)}


# ============================================================================
# Special Ops Mission Endpoints
# ============================================================================

class SpecopsUpdateRequest(BaseModel):
    work_style: Optional[str] = None
    cycle_interval: Optional[int] = None
    auto_continue: Optional[bool] = None
    mission_duration: Optional[int] = None

@app.patch("/agents/{agent_id}/specops")
async def update_specops_settings(agent_id: str, request: SpecopsUpdateRequest, user: str = Depends(verify_token)):
    """Update mission settings for a running specops agent."""
    agent = active_agents.get(agent_id)
    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")
    if not agent.get("is_specops"):
        raise HTTPException(400, "Agent is not a Special Ops agent")

    # Update fields
    if request.work_style is not None:
        agent["work_style"] = request.work_style
        agent.setdefault("specops_config", {})["work_style"] = request.work_style
    if request.cycle_interval is not None:
        agent["cycle_interval"] = request.cycle_interval
        agent.setdefault("specops_config", {})["cycle_interval"] = request.cycle_interval
    if request.auto_continue is not None:
        agent["auto_continue"] = request.auto_continue
    if request.mission_duration is not None:
        agent["mission_duration"] = request.mission_duration
        agent.setdefault("specops_config", {})["mission_duration"] = request.mission_duration

    await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))
    await _emit_specops_update(agent)
    return {"status": "updated", "agent_id": agent_id}

@app.get("/agents/{agent_id}/journal")
async def get_mission_journal(agent_id: str, user: str = Depends(verify_token)):
    """Get mission journal entries for a specops agent."""
    agent = active_agents.get(agent_id)
    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")
    return {
        "agent_id": agent_id,
        "journal": agent.get("mission_journal", []),
        "cycle_count": agent.get("cycle_count", 0),
        "mission_elapsed": agent.get("mission_elapsed", 0),
    }


# ============================================================================
# Agent Continuation System - Continue work on existing agents
# ============================================================================

class AgentContinueRequest(BaseModel):
    instructions: str = "Continue from where you left off."


@app.post("/agents/{agent_id}/continue")
async def continue_agent(agent_id: str, request: AgentContinueRequest, user: str = Depends(verify_token)):
    """Continue work on an existing agent without creating new identity."""

    # 1. First check in-memory agents (most recent/active)
    agent_data = None
    agent_file = None
    agents_dir = Path(config.AI_MEMORY_PATH) / "agents"

    if agent_id in active_agents:
        agent_data = active_agents[agent_id]
        # Try to find the file if it exists
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            agent_entry = next((a for a in index.get("agents", []) if a["id"] == agent_id), None)
            if agent_entry:
                agent_file = agents_dir / agent_entry["file"]
                if not agent_file.exists():
                    agent_file = None
    else:
        # Check persisted agents in index
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            agent_entry = next((a for a in index.get("agents", []) if a["id"] == agent_id), None)
            if agent_entry:
                agent_file = agents_dir / agent_entry["file"]
                if agent_file.exists():
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        agent_data = json.load(f)
                else:
                    raise HTTPException(404, f"Agent file not found: {agent_file}")

    if not agent_data:
        raise HTTPException(404, f"Agent {agent_id} not found")

    # 2. Save output to context file for retrieval
    context_ref = agent_memory_service.save_agent_context(
        agent_id=agent_id,
        call_sign=agent_data.get("call_sign", agent_id),
        task=agent_data.get("task", ""),
        output=agent_data.get("output", "")
    )

    # 3. Generate SHORT continuation prompt - agent reads context from file
    context_file_path = context_ref['context_file']

    # Keep the prompt VERY short to avoid any chunking issues
    continuation_prompt = f"""CONTINUATION: Agent {agent_data.get('call_sign', agent_id)}

FIRST: Read your previous work from: {context_file_path}

THEN: {request.instructions}

Use the Read tool to get your full previous output before proceeding."""

    # 4. Update agent status
    call_sign = agent_data.get("call_sign", agent_id)
    agent_data["status"] = AgentStatus.RUNNING
    agent_data["continuation_count"] = agent_data.get("continuation_count", 0) + 1
    agent_data["last_continuation"] = datetime.now(timezone.utc).isoformat()
    agent_data["started_at"] = datetime.now(timezone.utc).isoformat()
    agent_data["completed_at"] = None
    agent_data["error"] = None

    # Clean up non-serializable fields from previous run
    agent_data.pop("_process", None)
    agent_data.pop("_stdout_thread", None)

    # Update in-memory tracking
    active_agents[agent_id] = agent_data

    # Save updated agent data if we have a file
    if agent_file:
        with open(agent_file, 'w', encoding='utf-8') as f:
            json.dump(sanitize_agent_for_emit(agent_data), f, indent=2, ensure_ascii=False)

    # Broadcast status update
    await sio.emit("agent_update", sanitize_agent_for_emit(agent_data), room=os.environ.get("CEREBRO_ROOM", "default"))

    # Run the agent with continuation prompt
    asyncio.create_task(run_agent_continuation(
        agent_id=agent_id,
        call_sign=call_sign,
        task=continuation_prompt,
        agent_file=str(agent_file) if agent_file else None,
        agent_data=agent_data
    ))

    return {
        "agent_id": agent_id,
        "call_sign": call_sign,
        "status": "continuing",
        "continuation_count": agent_data["continuation_count"],
        "context_file": context_ref["context_file"]
    }


async def run_agent_continuation(agent_id: str, call_sign: str, task: str, agent_file: str, agent_data: dict):
    """Run a continuation of an existing agent.
    Uses subprocess.Popen with threaded I/O (same as run_agent) because
    asyncio.create_subprocess_exec doesn't work on Windows SelectorEventLoop.
    """
    import shutil
    import subprocess as sp
    import threading

    try:
        # Find claude executable
        claude_path = shutil.which("claude")
        if not claude_path:
            possible_paths = [
                os.path.join(os.path.expanduser("~"), ".local", "bin", "claude.exe"),
                os.path.join(os.path.expanduser("~"), ".local", "bin", "claude"),
                os.path.join(os.path.expanduser("~"), ".npm-global", "claude.cmd"),
                os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "npm", "claude.cmd"),
            ]
            for path in possible_paths:
                if Path(path).exists():
                    claude_path = path
                    break

        if not claude_path:
            raise Exception("Claude CLI not found")

        # Use a SHORT prompt for continuations - the task already has all the instructions
        full_prompt = f"You are Agent {call_sign} continuing your previous work.\n\n{task}"

        print(f"[Agent {call_sign}] Prompt length: {len(full_prompt)} chars")
        print(f"[Agent {call_sign}] Starting continuation #{agent_data.get('continuation_count', 1)}")

        # Use subprocess.Popen (Windows SelectorEventLoop doesn't support
        # asyncio.create_subprocess_exec, so we use threads for I/O)
        # Strip CLAUDECODE env var so spawned CLI doesn't think it's nested
        agent_env = os.environ.copy()
        agent_env.pop("CLAUDECODE", None)
        # Signal hooks to exit immediately (avoids brain_maintenance, wake-nas, etc.)
        agent_env["CEREBRO_AGENT"] = "1"
        # Standalone: ensure HOME points to cerebro user dir where credentials live
        if _IS_STANDALONE:
            agent_env["HOME"] = "/home/cerebro"

        # Resolve model for continuation (inherit from agent_data or default to sonnet)
        cont_model = agent_data.get("model", "sonnet")
        cont_model_map = {
            "opus": "claude-opus-4-6",
            "sonnet": "claude-sonnet-4-6",
            "haiku": "claude-haiku-4-5-20251001",
        }
        cont_resolved_model = cont_model_map.get(cont_model, cont_model)

        process = sp.Popen(
            [claude_path, "-p", full_prompt,
             "--model", cont_resolved_model,
             "--output-format", "stream-json",
             "--verbose",
             "--dangerously-skip-permissions"],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            cwd=config.AI_MEMORY_PATH,
            env=agent_env,
        )

        # Store process PID for emergency stop capability
        agent_data["pid"] = process.pid
        agent_data["_process"] = process
        if agent_id in active_agents:
            active_agents[agent_id]["pid"] = process.pid
            active_agents[agent_id]["_process"] = process
            await sio.emit("agent_update", sanitize_agent_for_emit(agent_data), room=os.environ.get("CEREBRO_ROOM", "default"))

        full_output = ""
        tools_used = set(agent_data.get("tools_used", []))
        update_counter = 0

        # Read stdout in a thread (same pattern as run_agent)
        import queue
        line_queue = queue.Queue()

        def read_stdout(proc, q):
            try:
                for line in iter(proc.stdout.readline, b''):
                    q.put(line)
            except Exception:
                pass
            finally:
                q.put(None)  # Sentinel

        stdout_thread = threading.Thread(target=read_stdout, args=(process, line_queue), daemon=True)
        stdout_thread.start()

        while True:
            try:
                # Non-blocking read with timeout
                line = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: line_queue.get(timeout=1800)
                )
            except Exception:
                break

            if line is None:
                break

            raw_line = line.decode('utf-8', errors='replace').strip()
            if not raw_line:
                continue

            try:
                data = json.loads(raw_line)
                msg_type = data.get("type", "")

                if msg_type == "assistant":
                    text = data.get("message", {}).get("content", "")
                    if isinstance(text, list):
                        for block in text:
                            if block.get("type") == "text":
                                full_output += block.get("text", "")
                                # Stream narration for continuations too
                                text_content = block.get("text", "")
                                if text_content.strip():
                                    await sio.emit("cerebro_narration", {
                                        "content": text_content,
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "agent_id": agent_id,
                                        "is_idle": False,
                                    }, room=os.environ.get("CEREBRO_ROOM", "default"))
                            elif block.get("type") == "tool_use":
                                tools_used.add(block.get("name", ""))
                    elif isinstance(text, str) and text:
                        full_output += text

                elif msg_type == "tool_use":
                    tool_name = data.get("tool", data.get("name", "tool"))
                    tools_used.add(tool_name)

                elif msg_type == "result":
                    result_text = data.get("result", "")
                    if result_text and result_text not in full_output:
                        full_output += "\n\n" + result_text

                elif msg_type == "error":
                    error_msg = data.get("error", {}).get("message", str(data))
                    full_output += f"\n\n[Error: {error_msg}]"

                # Periodic updates
                update_counter += 1
                if update_counter >= 5 or len(full_output) - len(agent_data.get("output", "")) > 200:
                    agent_data["output"] = full_output[-5000:]
                    agent_data["tools_used"] = list(tools_used)
                    await sio.emit("agent_update", sanitize_agent_for_emit(agent_data), room=os.environ.get("CEREBRO_ROOM", "default"))
                    update_counter = 0

            except json.JSONDecodeError:
                if raw_line and not raw_line.startswith('{'):
                    full_output += raw_line + "\n"

        # Wait for process to complete
        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: process.wait(timeout=300))
        except sp.TimeoutExpired:
            process.kill()
            agent_data["error"] = "Process killed after timeout"

        # Check stderr
        stderr_output = process.stderr.read() if process.stderr else b""
        if stderr_output:
            stderr_text = stderr_output.decode('utf-8', errors='replace')
            print(f"[Agent {call_sign}] Stderr: {stderr_text[:500]}")
            if "error" in stderr_text.lower() and not full_output:
                full_output = f"[Stderr]\n{stderr_text}"

        if not full_output:
            full_output = f"Agent continuation completed but no output was captured. Exit code: {process.returncode}"

        # Update agent with result
        agent_data["status"] = AgentStatus.COMPLETED
        agent_data["output"] = full_output
        agent_data["tools_used"] = list(tools_used)
        agent_data["completed_at"] = datetime.now(timezone.utc).isoformat()

        print(f"[Agent {call_sign}] Continuation completed. Output length: {len(full_output)}")

        # Save to file if we have one (sanitize to remove _process etc)
        if agent_file:
            with open(agent_file, 'w', encoding='utf-8') as f:
                json.dump(sanitize_agent_for_emit(agent_data), f, indent=2, ensure_ascii=False)

        # Update index
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            for agent_entry in index.get("agents", []):
                if agent_entry["id"] == agent_id:
                    agent_entry["status"] = agent_data["status"]
                    agent_entry["completed_at"] = agent_data["completed_at"]
                    agent_entry["tools_used"] = agent_data["tools_used"]
                    break
            index["last_updated"] = datetime.now(timezone.utc).isoformat()
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

        # Process agent queue - continuation completion frees a slot
        await _process_agent_queue()

        # Create notification
        await create_notification(
            notif_type="agent_complete",
            title=f"Agent {call_sign} Continuation Complete",
            message=f"Continuation #{agent_data.get('continuation_count', 1)} finished",
            link=f"/agents/{agent_id}",
            agent_id=agent_id
        )

    except Exception as e:
        print(f"[Agent {call_sign}] Continuation failed: {e}")
        import traceback
        traceback.print_exc()

        agent_data["status"] = AgentStatus.FAILED
        agent_data["error"] = str(e)

        if agent_file:
            with open(agent_file, 'w', encoding='utf-8') as f:
                json.dump(sanitize_agent_for_emit(agent_data), f, indent=2, ensure_ascii=False)

        await create_notification(
            notif_type="agent_failed",
            title=f"Agent {call_sign} Continuation Failed",
            message=str(e)[:100],
            link=f"/agents/{agent_id}",
            agent_id=agent_id
        )

    # Broadcast final state
    await sio.emit("agent_update", sanitize_agent_for_emit(agent_data), room=os.environ.get("CEREBRO_ROOM", "default"))
    await sio.emit("agent_completed", sanitize_agent_for_emit(agent_data), room=os.environ.get("CEREBRO_ROOM", "default"))


# ============================================================================
# Special Ops Mission Supervisor Loop
# ============================================================================

async def _wait_for_agent_completion(agent_id: str) -> str:
    """Poll agent status every 5s until it's no longer RUNNING. Returns final status."""
    while True:
        agent = active_agents.get(agent_id)
        if not agent:
            return AgentStatus.FAILED
        status = agent.get("status")
        if status != AgentStatus.RUNNING:
            return status
        await asyncio.sleep(5)

def _parse_mission_status(output: str, cycle_num: int) -> dict:
    """Extract [MISSION STATUS]...[/MISSION STATUS] block from agent output."""
    import re
    entry = {
        "cycle": cycle_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "progress": "Unknown",
        "completed": "Unknown",
        "next": "Unknown",
        "blockers": "None",
        "raw_length": len(output) if output else 0,
    }
    if not output:
        return entry
    match = re.search(r'\[MISSION STATUS\](.*?)\[/MISSION STATUS\]', output, re.DOTALL)
    if not match:
        # Fallback: use last 500 chars as summary
        entry["completed"] = output[-500:].strip() if len(output) > 500 else output.strip()
        return entry
    block = match.group(1).strip()
    for line in block.split('\n'):
        line = line.strip()
        if line.lower().startswith('progress:'):
            entry["progress"] = line.split(':', 1)[1].strip()
        elif line.lower().startswith('completed:'):
            entry["completed"] = line.split(':', 1)[1].strip()
        elif line.lower().startswith('next:'):
            entry["next"] = line.split(':', 1)[1].strip()
        elif line.lower().startswith('blockers:'):
            entry["blockers"] = line.split(':', 1)[1].strip()
    return entry

async def _emit_specops_update(agent: dict):
    """Emit specops_mission_update socket event with current mission metrics."""
    await sio.emit("specops_mission_update", {
        "agent_id": agent["id"],
        "cycle_count": agent.get("cycle_count", 0),
        "mission_journal": agent.get("mission_journal", []),
        "next_checkin": agent.get("next_checkin"),
        "mission_elapsed": agent.get("mission_elapsed", 0),
        "work_style": agent.get("work_style", "continuous"),
        "auto_continue": agent.get("auto_continue", True),
        "status": agent.get("status"),
        "mission_name": agent.get("mission_name", ""),
    }, room=os.environ.get("CEREBRO_ROOM", "default"))

async def run_specops_mission(agent_id: str):
    """
    Mission Supervisor Loop for Special Ops agents.
    Waits for each cycle to complete, parses output, manages journal,
    and auto-continues into the next cycle based on work style.
    """
    agent = active_agents.get(agent_id)
    if not agent:
        return

    print(f"[SpecOps] Mission supervisor started for {agent_id} — {agent.get('mission_name', 'Unnamed')}")

    while True:
        # === Wait for current cycle to complete ===
        final_status = await _wait_for_agent_completion(agent_id)
        agent = active_agents.get(agent_id)
        if not agent:
            print(f"[SpecOps] Agent {agent_id} disappeared — mission ended")
            return

        # === Parse output and create journal entry ===
        cycle_num = agent.get("cycle_count", 0) + 1
        agent["cycle_count"] = cycle_num
        journal_entry = _parse_mission_status(agent.get("output", ""), cycle_num)
        agent.setdefault("mission_journal", []).append(journal_entry)

        # Update mission elapsed time
        started = agent.get("mission_started_at")
        if started:
            try:
                start_dt = datetime.fromisoformat(started.replace('Z', '+00:00'))
                agent["mission_elapsed"] = int((datetime.now(timezone.utc) - start_dt).total_seconds())
            except Exception:
                pass

        print(f"[SpecOps] {agent_id} cycle #{cycle_num} complete — status: {final_status}, progress: {journal_entry.get('progress', '?')}")
        await _emit_specops_update(agent)

        # === Check termination conditions ===
        # 1. Agent was stopped or failed
        if final_status in (AgentStatus.STOPPED, AgentStatus.FAILED):
            print(f"[SpecOps] Mission ended — agent {final_status}")
            return

        # 2. Auto-continue disabled
        if not agent.get("auto_continue", True):
            print(f"[SpecOps] Mission paused — auto_continue is OFF")
            agent["status"] = AgentStatus.COMPLETED
            await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))
            return

        # 3. Mission duration exceeded
        mission_duration = agent.get("mission_duration", 0)
        if mission_duration > 0 and agent.get("mission_elapsed", 0) >= mission_duration:
            print(f"[SpecOps] Mission ended — duration limit reached ({_format_duration(mission_duration)})")
            agent["status"] = AgentStatus.COMPLETED
            await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))
            await _emit_specops_update(agent)
            return

        # === Determine cycle delay based on work style ===
        work_style = agent.get("work_style", "continuous")
        cycle_interval = agent.get("cycle_interval", 3600)

        if work_style == "continuous":
            sleep_seconds = 5
        elif work_style == "cycle":
            sleep_seconds = cycle_interval
        elif work_style == "hybrid":
            # Check if agent requested PAUSE
            next_action = journal_entry.get("next", "")
            if "PAUSE" in next_action.upper():
                sleep_seconds = cycle_interval
            else:
                sleep_seconds = 5
        else:
            sleep_seconds = 5

        # === Sleep / Cycling phase ===
        if sleep_seconds > 10:
            agent["status"] = AgentStatus.CYCLING
            next_checkin = datetime.now(timezone.utc) + timedelta(seconds=sleep_seconds)
            agent["next_checkin"] = next_checkin.isoformat()
            await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))
            await _emit_specops_update(agent)

            print(f"[SpecOps] {agent_id} cycling — next cycle in {_format_interval(sleep_seconds)}")

            # Sleep in 30s increments so we can detect cancellation
            elapsed_sleep = 0
            while elapsed_sleep < sleep_seconds:
                await asyncio.sleep(min(30, sleep_seconds - elapsed_sleep))
                elapsed_sleep += 30
                # Check if agent was stopped during sleep
                agent = active_agents.get(agent_id)
                if not agent or agent.get("status") == AgentStatus.STOPPED:
                    print(f"[SpecOps] Mission cancelled during cycling")
                    return
                if not agent.get("auto_continue", True):
                    print(f"[SpecOps] Auto-continue disabled during cycling")
                    agent["status"] = AgentStatus.COMPLETED
                    await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))
                    return
        else:
            await asyncio.sleep(sleep_seconds)

        # === Re-check agent still exists ===
        agent = active_agents.get(agent_id)
        if not agent:
            return

        # === Start next cycle via continuation ===
        print(f"[SpecOps] {agent_id} starting cycle #{cycle_num + 1}")

        # Save context for continuation
        context_ref = agent_memory_service.save_agent_context(
            agent_id=agent_id,
            call_sign=agent.get("call_sign", agent_id),
            task=agent.get("task", ""),
            output=agent.get("output", "")
        )

        continuation_prompt = f"""SPECOPS CONTINUATION: Agent {agent.get('call_sign', agent_id)} — Cycle #{cycle_num + 1}

FIRST: Read your previous cycle output from: {context_ref['context_file']}

THEN: Continue your mission. Refer to the mission journal in your system prompt for all previous cycle summaries.

Your mission objective: {agent.get('task', '')}

Remember to output a [MISSION STATUS] block at the end of this cycle."""

        # Reset agent for new cycle
        agent["status"] = AgentStatus.RUNNING
        agent["started_at"] = datetime.now(timezone.utc).isoformat()
        agent["completed_at"] = None
        agent["error"] = None
        agent["next_checkin"] = None
        agent.pop("_process", None)
        agent.pop("_stdout_thread", None)

        await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))
        await _emit_specops_update(agent)

        # Find or create agent file for continuation
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        agent_file = None
        index_file = agents_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                agent_entry = next((a for a in index.get("agents", []) if a["id"] == agent_id), None)
                if agent_entry:
                    agent_file = str(agents_dir / agent_entry["file"])
            except Exception:
                pass

        # Launch the continuation (this blocks until the cycle completes)
        await run_agent_continuation(
            agent_id=agent_id,
            call_sign=agent.get("call_sign", agent_id),
            task=continuation_prompt,
            agent_file=agent_file,
            agent_data=agent
        )
        # Loop back to wait for completion


# ============================================================================
# Internal Child Agent API - For orchestrator agents to spawn children
# ============================================================================

class InternalSpawnRequest(BaseModel):
    task: str
    agent_type: str = "worker"
    parent_agent_id: str
    context: Optional[str] = None
    expected_output: Optional[str] = None

@app.post("/internal/spawn-child-agent")
async def spawn_child_agent_internal(request: Request, data: InternalSpawnRequest):
    """
    Internal endpoint for orchestrator agents to spawn child agents.
    Only accessible from localhost (no auth required).
    """
    # Security: Only allow from localhost
    client_host = request.client.host
    if client_host not in ("127.0.0.1", "localhost", "::1"):
        raise HTTPException(403, "Internal endpoint only accessible from localhost")

    parent_id = data.parent_agent_id

    # Verify parent exists
    parent_agent = active_agents.get(parent_id)
    if not parent_agent:
        # Check persisted agents
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            agent_entry = next((a for a in index.get("agents", []) if a["id"] == parent_id), None)
            if agent_entry:
                # Load parent agent data
                agent_file = agents_dir / agent_entry["file"]
                if agent_file.exists():
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        parent_agent = json.load(f)

    if not parent_agent:
        raise HTTPException(404, f"Parent agent {parent_id} not found")

    # Create the child agent
    child_id = await create_agent(
        task=data.task,
        agent_type=data.agent_type,
        context=data.context,
        expected_output=data.expected_output,
        parent_agent_id=parent_id
    )

    # Update parent's child_agents array
    if parent_id in active_agents:
        if "child_agents" not in active_agents[parent_id]:
            active_agents[parent_id]["child_agents"] = []
        active_agents[parent_id]["child_agents"].append(child_id)

        # Persist to file if exists
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            agent_entry = next((a for a in index.get("agents", []) if a["id"] == parent_id), None)
            if agent_entry:
                agent_file = agents_dir / agent_entry["file"]
                if agent_file.exists():
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        parent_data = json.load(f)
                    if "child_agents" not in parent_data:
                        parent_data["child_agents"] = []
                    parent_data["child_agents"].append(child_id)
                    with open(agent_file, 'w', encoding='utf-8') as f:
                        json.dump(parent_data, f, indent=2, ensure_ascii=False)

    print(f"[Internal] Child agent {child_id} spawned for parent {parent_id}")

    return {
        "agent_id": child_id,
        "call_sign": active_agents.get(child_id, {}).get("call_sign", child_id),
        "parent_agent_id": parent_id,
        "status": "spawned"
    }


@app.get("/internal/agent/{agent_id}/children")
async def get_agent_children(agent_id: str, request: Request):
    """
    Get all child agents for a parent agent.
    Only accessible from localhost.
    """
    client_host = request.client.host
    if client_host not in ("127.0.0.1", "localhost", "::1"):
        raise HTTPException(403, "Internal endpoint only accessible from localhost")

    # Get parent agent
    parent_agent = active_agents.get(agent_id)
    if not parent_agent:
        # Check persisted agents
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            agent_entry = next((a for a in index.get("agents", []) if a["id"] == agent_id), None)
            if agent_entry:
                agent_file = agents_dir / agent_entry["file"]
                if agent_file.exists():
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        parent_agent = json.load(f)

    if not parent_agent:
        raise HTTPException(404, f"Agent {agent_id} not found")

    child_ids = parent_agent.get("child_agents", [])

    # Get full info for each child
    children = []
    for child_id in child_ids:
        child_agent = active_agents.get(child_id)
        if child_agent:
            children.append({
                "id": child_id,
                "call_sign": child_agent.get("call_sign", child_id),
                "type": child_agent.get("type", "worker"),
                "status": child_agent.get("status", "unknown"),
                "task": child_agent.get("task", "")[:200],
                "output": child_agent.get("output", "")[-2000:] if child_agent.get("status") == AgentStatus.COMPLETED else None,
                "error": child_agent.get("error")
            })
        else:
            # Try to load from persisted storage
            agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
            for date_dir in agents_dir.iterdir():
                if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                    child_file = date_dir / f"{child_id}.json"
                    if child_file.exists():
                        with open(child_file, 'r', encoding='utf-8') as f:
                            child_data = json.load(f)
                        children.append({
                            "id": child_id,
                            "call_sign": child_data.get("call_sign", child_id),
                            "type": child_data.get("type", "worker"),
                            "status": child_data.get("status", "unknown"),
                            "task": child_data.get("task", "")[:200],
                            "output": child_data.get("output", "")[-2000:] if child_data.get("status") == AgentStatus.COMPLETED else None,
                            "error": child_data.get("error")
                        })
                        break

    return {
        "parent_id": agent_id,
        "children": children,
        "total": len(children),
        "completed": sum(1 for c in children if c["status"] == AgentStatus.COMPLETED),
        "running": sum(1 for c in children if c["status"] == AgentStatus.RUNNING),
        "failed": sum(1 for c in children if c["status"] == AgentStatus.FAILED)
    }


@app.get("/internal/agent/{agent_id}/children/wait")
async def wait_for_children(agent_id: str, request: Request, timeout: int = 1800):
    """
    Block until all child agents complete (or timeout).
    Only accessible from localhost.

    Returns combined outputs from all children.
    """
    client_host = request.client.host
    if client_host not in ("127.0.0.1", "localhost", "::1"):
        raise HTTPException(403, "Internal endpoint only accessible from localhost")

    # Get parent agent
    parent_agent = active_agents.get(agent_id)
    if not parent_agent:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            agent_entry = next((a for a in index.get("agents", []) if a["id"] == agent_id), None)
            if agent_entry:
                agent_file = agents_dir / agent_entry["file"]
                if agent_file.exists():
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        parent_agent = json.load(f)

    if not parent_agent:
        raise HTTPException(404, f"Agent {agent_id} not found")

    child_ids = parent_agent.get("child_agents", [])
    if not child_ids:
        return {"parent_id": agent_id, "children": [], "all_completed": True, "outputs": {}}

    # Poll until all complete or timeout
    start_time = asyncio.get_event_loop().time()
    poll_interval = 2  # seconds

    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            # Timeout - return partial results
            break

        all_done = True
        for child_id in child_ids:
            child = active_agents.get(child_id)
            if child and child.get("status") in (AgentStatus.QUEUED, AgentStatus.RUNNING):
                all_done = False
                break

        if all_done:
            break

        await asyncio.sleep(poll_interval)

    # Collect final results
    results = {}
    children_info = []

    for child_id in child_ids:
        child = active_agents.get(child_id)
        if not child:
            # Try loading from storage
            agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
            for date_dir in agents_dir.iterdir():
                if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                    child_file = date_dir / f"{child_id}.json"
                    if child_file.exists():
                        with open(child_file, 'r', encoding='utf-8') as f:
                            child = json.load(f)
                        break

        if child:
            results[child_id] = {
                "call_sign": child.get("call_sign", child_id),
                "status": child.get("status", "unknown"),
                "output": child.get("output", ""),
                "error": child.get("error"),
                "task": child.get("task", "")[:200]
            }
            children_info.append({
                "id": child_id,
                "call_sign": child.get("call_sign", child_id),
                "status": child.get("status", "unknown")
            })

    all_completed = all(r["status"] == AgentStatus.COMPLETED for r in results.values())

    return {
        "parent_id": agent_id,
        "children": children_info,
        "all_completed": all_completed,
        "outputs": results,
        "elapsed_seconds": asyncio.get_event_loop().time() - start_time
    }


@app.get("/internal/agent/{agent_id}")
async def get_agent_internal(agent_id: str, request: Request):
    """
    Get agent details (internal, no auth).
    Only accessible from localhost.
    """
    client_host = request.client.host
    if client_host not in ("127.0.0.1", "localhost", "::1"):
        raise HTTPException(403, "Internal endpoint only accessible from localhost")

    agent = active_agents.get(agent_id)
    if not agent:
        # Check persisted agents (newest first!)
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        date_dirs = sorted(
            [d for d in agents_dir.iterdir() if d.is_dir() and not d.name.endswith('.json')],
            key=lambda d: d.name,
            reverse=True
        )
        for date_dir in date_dirs:
            agent_file = date_dir / f"{agent_id}.json"
            if agent_file.exists():
                with open(agent_file, 'r', encoding='utf-8') as f:
                    agent = json.load(f)
                break

    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")

    return agent


@app.post("/agents/batch")
async def spawn_batch_agents(tasks: list[str], user: str = Depends(verify_token)):
    """Spawn multiple agents at once."""
    agent_ids = []
    for task in tasks:
        agent_id = await create_agent(task, "worker")
        agent_ids.append(agent_id)
    return {"agent_ids": agent_ids, "count": len(agent_ids)}

# ============================================================================
# Multi-Agent Workflow System
# ============================================================================

@app.post("/workflows")
async def spawn_workflow(request: WorkflowRequest, user: str = Depends(verify_token)):
    """
    Spawn a multi-agent workflow with an orchestrator managing child agents.

    workflow_type options:
    - "standard": Orchestrator decomposes task and delegates dynamically
    - "parallel": All specified agents run simultaneously
    - "sequential": Agents run one after another, passing context forward
    """
    workflow_id = f"WF-{uuid.uuid4().hex[:8].upper()}"

    # Default composition if not specified
    if not request.agent_composition:
        request.agent_composition = ["researcher", "coder", "worker"]

    # Create the orchestrator (mother agent)
    # Note: The orchestrator's call_sign will be assigned by create_agent, but we include
    # placeholders in the task that will be filled in by the system prompt
    orchestrator_task = f"""You are commanding a multi-agent workflow to accomplish the following mission:

## PRIMARY OBJECTIVE
{request.task}

## YOUR TEAM
You have access to the following specialized agents that you can delegate to:
{chr(10).join(f"- {AGENT_ROLE_PROMPTS.get(role, AGENT_ROLE_PROMPTS['worker'])['icon']} **{role.upper()} #{i+1}**: {AGENT_ROLE_PROMPTS.get(role, AGENT_ROLE_PROMPTS['worker'])['description']}" for i, role in enumerate(request.agent_composition))}

## WORKFLOW TYPE: {request.workflow_type.upper()}
{'Decompose the task and delegate to appropriate agents as needed.' if request.workflow_type == 'standard' else 'Run all agents in parallel and synthesize results.' if request.workflow_type == 'parallel' else 'Run agents sequentially, each building on the previous.'}

## HOW TO SPAWN CHILD AGENTS (CRITICAL - USE BASH + CURL ONLY)
You MUST use the Bash tool to run curl commands. Do NOT use the Task tool or TodoWrite.

### Spawn a child agent:
```bash
curl -s -X POST http://localhost:59000/internal/spawn-child-agent -H "Content-Type: application/json" -d '{{"task": "Description of subtask", "agent_type": "researcher", "parent_agent_id": "YOUR_CALL_SIGN"}}'
```

### Check your children's status:
```bash
curl -s http://localhost:59000/internal/agent/YOUR_CALL_SIGN/children
```

### WAIT for ALL children to complete (BLOCKING - always use this):
```bash
curl -s "http://localhost:59000/internal/agent/YOUR_CALL_SIGN/children/wait?timeout=1800"
```
This returns all child outputs when they finish. DO NOT skip this step.

## MANDATORY INSTRUCTIONS
1. Analyze the mission and determine what subtasks each agent should handle
2. Spawn child agents using Bash + curl (one per subtask) - use YOUR actual call sign as parent_agent_id
3. Call the wait endpoint and WAIT for ALL children to complete - this is blocking and will return their outputs
4. Read the combined outputs from the wait response
5. Synthesize all results into a comprehensive final report

IMPORTANT: Do NOT mark yourself complete until you have received and synthesized all child outputs.
Your final output MUST contain the actual synthesized research, not just a progress update.

Execute your command, Commander."""

    orchestrator_id = await create_agent(
        task=orchestrator_task,
        agent_type="orchestrator",
        context=f"Workflow ID: {workflow_id}",
        expected_output="Comprehensive mission report with all team outputs synthesized"
    )

    # Track the workflow
    workflow_info = {
        "id": workflow_id,
        "orchestrator_id": orchestrator_id,
        "task": request.task,
        "workflow_type": request.workflow_type,
        "agent_composition": request.agent_composition,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "child_agents": []  # Will be populated as orchestrator spawns agents
    }

    active_workflows[workflow_id] = workflow_info

    return {
        "workflow_id": workflow_id,
        "orchestrator_id": orchestrator_id,
        "status": "spawned",
        "team": request.agent_composition
    }

@app.get("/workflows")
async def list_workflows(user: str = Depends(verify_token)):
    """List all workflows."""
    return {"workflows": list(active_workflows.values())}

@app.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str, user: str = Depends(verify_token)):
    """Get workflow status including all child agents."""
    workflow = active_workflows.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Get orchestrator status
    orchestrator = active_agents.get(workflow["orchestrator_id"])

    # Find any child agents
    child_agents = [
        agent for agent in active_agents.values()
        if agent.get("parent_workflow_id") == workflow_id
    ]

    return {
        **workflow,
        "orchestrator": orchestrator,
        "child_agents": child_agents
    }

@app.get("/agent-roles")
async def get_agent_roles(user: str = Depends(verify_token)):
    """Get available agent roles and their descriptions."""
    return {
        "roles": {
            role: {
                "name": config["name"],
                "icon": config["icon"],
                "description": config["description"]
            }
            for role, config in AGENT_ROLE_PROMPTS.items()
        }
    }

@app.get("/agent-templates")
async def get_agent_templates(user: str = Depends(verify_token)):
    """Get saved agent prompt templates from Redis."""
    try:
        data = await redis.get("agent_templates")
        if data:
            return json.loads(data)
        return {"templates": []}
    except Exception:
        return {"templates": []}

@app.put("/agent-templates")
async def save_agent_templates(body: dict, user: str = Depends(verify_token)):
    """Save agent prompt templates to Redis."""
    try:
        await redis.set("agent_templates", json.dumps(body))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class AgentQuestionRequest(BaseModel):
    question: str
    background: bool = False  # If True, process in background and notify when ready
    context_refs: Optional[list[dict]] = None  # Context references (click-to-attach)
    parent_thread_id: Optional[str] = None  # For threaded conversations


async def process_question_background(question_id: str, agent_id: str, question: str, user: str, context_refs: list = None, parent_thread_id: str = None):
    """Process a question in the background using Claude Code CLI and notify when complete."""
    import shutil

    # Load agent data
    agent = active_agents.get(agent_id)
    if not agent:
        try:
            agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
            for date_dir in agents_dir.iterdir():
                if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                    agent_file = date_dir / f"{agent_id}.json"
                    if agent_file.exists():
                        with open(agent_file, 'r', encoding='utf-8') as f:
                            agent = json.load(f)
                            break
        except:
            pass

    if not agent:
        if question_id in pending_questions:
            pending_questions[question_id]["status"] = "failed"
            pending_questions[question_id]["error"] = "Agent not found"
        return

    agent_output = agent.get("output", "No output available")
    agent_task = agent.get("task", "Unknown task")
    agent_name = agent.get("call_sign", agent_id)

    # Resolve context references (click-to-attach)
    resolved_context = ""
    if context_refs:
        resolved_context = await resolve_context_refs(context_refs)

    # Build prompt for Claude Code CLI
    prompt = f"""You are answering a follow-up question about work done by an AI agent named "{agent_name}".

{resolved_context}ORIGINAL TASK: {agent_task}

AGENT OUTPUT:
{agent_output[:6000]}

USER QUESTION: {question}

Please answer the user's question based on the agent's output above. If referenced context was provided, use it to inform your answer. Be concise and helpful."""

    try:
        # Find claude executable
        claude_path = shutil.which("claude")
        if not claude_path:
            for p in [
                os.path.join(os.path.expanduser("~"), ".local", "bin", "claude.exe"),
                os.path.join(os.path.expanduser("~"), ".local", "bin", "claude"),
                os.path.join(os.path.expanduser("~"), ".npm-global", "claude.cmd"),
                os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "npm", "claude.cmd"),
            ]:
                if os.path.exists(p):
                    claude_path = p
                    break

        if not claude_path:
            if question_id in pending_questions:
                pending_questions[question_id]["status"] = "failed"
                pending_questions[question_id]["error"] = "Claude Code CLI not found"
            return

        # Run Claude Code CLI with the prompt
        process = await asyncio.create_subprocess_exec(
            claude_path,
            "--print",  # Non-interactive, print output
            "--output-format", "text",  # Plain text output
            "-p", prompt,  # The prompt
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=config.AI_MEMORY_PATH
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=180.0  # 3 minute timeout
            )
            answer = stdout.decode('utf-8').strip()

            if not answer:
                answer = "I couldn't generate a response. Please try again."
                if stderr:
                    print(f"Claude CLI stderr: {stderr.decode('utf-8')}")

        except asyncio.TimeoutError:
            process.kill()
            answer = "Request timed out after 3 minutes."

        # Store the Q&A in the agent's conversation history
        now = datetime.now(timezone.utc).isoformat()
        thread_id = str(uuid.uuid4())[:8]
        user_msg = {"role": "user", "content": question, "timestamp": now, "thread_id": thread_id, "parent_thread_id": parent_thread_id}
        assistant_msg = {"role": "assistant", "content": answer, "timestamp": now, "thread_id": thread_id, "parent_thread_id": parent_thread_id}

        if agent_id in active_agents:
            if "conversation" not in active_agents[agent_id]:
                active_agents[agent_id]["conversation"] = []
            active_agents[agent_id]["conversation"].append(user_msg)
            active_agents[agent_id]["conversation"].append(assistant_msg)

        # Persist to file
        try:
            agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
            for date_dir in agents_dir.iterdir():
                if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                    agent_file = date_dir / f"{agent_id}.json"
                    if agent_file.exists():
                        with open(agent_file, 'r', encoding='utf-8') as f:
                            agent_data = json.load(f)
                        if "conversation" not in agent_data:
                            agent_data["conversation"] = []
                        agent_data["conversation"].append(user_msg)
                        agent_data["conversation"].append(assistant_msg)
                        with open(agent_file, 'w', encoding='utf-8') as f:
                            json.dump(agent_data, f, indent=2, ensure_ascii=False)
                        break
        except Exception as e:
            print(f"Failed to persist conversation: {e}")

        # Update pending question
        if question_id in pending_questions:
            pending_questions[question_id]["status"] = "completed"
            pending_questions[question_id]["answer"] = answer

        # Emit Socket.IO event for real-time update
        await sio.emit("ask_response_ready", {
            "question_id": question_id,
            "agent_id": agent_id,
            "question": question,
            "answer": answer,
            "thread_id": thread_id,
            "parent_thread_id": parent_thread_id
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

        # Create notification
        await create_notification(
            notif_type="ask_response",
            title=f"Response Ready: Agent {agent_name}",
            message=question[:50] + ("..." if len(question) > 50 else ""),
            link=f"/agents/{agent_id}/chat",
            agent_id=agent_id
        )

    except Exception as e:
        if question_id in pending_questions:
            pending_questions[question_id]["status"] = "failed"
            pending_questions[question_id]["error"] = str(e)
        print(f"Error processing question: {e}")


@app.post("/agents/{agent_id}/ask")
async def ask_agent_question(agent_id: str, request: AgentQuestionRequest, user: str = Depends(verify_token)):
    """Ask a follow-up question about an agent's output."""

    # If background processing requested, start task and return immediately
    if request.background:
        question_id = str(uuid.uuid4())
        pending_questions[question_id] = {
            "id": question_id,
            "agent_id": agent_id,
            "question": request.question,
            "status": "processing",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "answer": None,
            "error": None
        }

        # Start background task with context refs
        asyncio.create_task(process_question_background(question_id, agent_id, request.question, user, request.context_refs, request.parent_thread_id))

        return {
            "question_id": question_id,
            "status": "processing",
            "message": "Question is being processed. You'll be notified when the response is ready."
        }

    # Synchronous processing using Claude Code CLI
    import shutil

    agent = active_agents.get(agent_id)
    if not agent:
        try:
            agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
            for date_dir in agents_dir.iterdir():
                if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                    agent_file = date_dir / f"{agent_id}.json"
                    if agent_file.exists():
                        with open(agent_file, 'r', encoding='utf-8') as f:
                            agent = json.load(f)
                            break
        except:
            pass

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_output = agent.get("output", "No output available")
    agent_task = agent.get("task", "Unknown task")
    agent_name = agent.get("call_sign", agent_id)

    # Resolve context references (click-to-attach)
    resolved_context = ""
    if request.context_refs:
        resolved_context = await resolve_context_refs(request.context_refs)

    prompt = f"""You are answering a follow-up question about work done by an AI agent named "{agent_name}".

{resolved_context}ORIGINAL TASK: {agent_task}

AGENT OUTPUT:
{agent_output[:6000]}

USER QUESTION: {request.question}

Please answer the user's question based on the agent's output above. If referenced context was provided, use it to inform your answer. Be concise and helpful."""

    try:
        # Find claude executable
        claude_path = shutil.which("claude")
        if not claude_path:
            for p in [
                os.path.join(os.path.expanduser("~"), ".local", "bin", "claude.exe"),
                os.path.join(os.path.expanduser("~"), ".local", "bin", "claude"),
                os.path.join(os.path.expanduser("~"), ".npm-global", "claude.cmd"),
                os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "npm", "claude.cmd"),
            ]:
                if os.path.exists(p):
                    claude_path = p
                    break

        if not claude_path:
            return {"error": "Claude Code CLI not found"}

        # Run Claude Code CLI
        process = await asyncio.create_subprocess_exec(
            claude_path,
            "--print",
            "--output-format", "text",
            "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=config.AI_MEMORY_PATH
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=180.0
            )
            answer = stdout.decode('utf-8').strip()
            if not answer:
                answer = "I couldn't generate a response."
        except asyncio.TimeoutError:
            process.kill()
            answer = "Request timed out."

        # Store the Q&A in conversation history (threaded)
        thread_id = str(uuid.uuid4())[:8]
        now_ts = datetime.now(timezone.utc).isoformat()
        if agent_id in active_agents:
            if "conversation" not in active_agents[agent_id]:
                active_agents[agent_id]["conversation"] = []
            active_agents[agent_id]["conversation"].append({
                "role": "user",
                "content": request.question,
                "timestamp": now_ts,
                "thread_id": thread_id,
                "parent_thread_id": request.parent_thread_id
            })
            active_agents[agent_id]["conversation"].append({
                "role": "assistant",
                "content": answer,
                "timestamp": now_ts,
                "thread_id": thread_id,
                "parent_thread_id": request.parent_thread_id
            })

        # Also persist to file for durability
        try:
            agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
            for date_dir in agents_dir.iterdir():
                if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                    agent_file = date_dir / f"{agent_id}.json"
                    if agent_file.exists():
                        with open(agent_file, 'r', encoding='utf-8') as f:
                            agent_data = json.load(f)
                        if "conversation" not in agent_data:
                            agent_data["conversation"] = []
                        agent_data["conversation"].append({
                            "role": "user", "content": request.question,
                            "timestamp": now_ts, "thread_id": thread_id,
                            "parent_thread_id": request.parent_thread_id
                        })
                        agent_data["conversation"].append({
                            "role": "assistant", "content": answer,
                            "timestamp": now_ts, "thread_id": thread_id,
                            "parent_thread_id": request.parent_thread_id
                        })
                        with open(agent_file, 'w', encoding='utf-8') as f:
                            json.dump(agent_data, f, indent=2, ensure_ascii=False)
                        break
        except Exception as e:
            print(f"Failed to persist conversation: {e}")

        return {
            "question": request.question,
            "answer": answer,
            "agent_id": agent_id,
            "thread_id": thread_id
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/agents/questions/{question_id}")
async def get_question_status(question_id: str, user: str = Depends(verify_token)):
    """Get the status of a background question."""
    if question_id not in pending_questions:
        raise HTTPException(status_code=404, detail="Question not found")
    return pending_questions[question_id]

@app.post("/agents/{agent_id}/view")
async def mark_agent_viewed(agent_id: str, user: str = Depends(verify_token)):
    """Mark an agent as viewed - updates last_viewed_at timestamp."""
    now = datetime.now(timezone.utc).isoformat()

    # Update in-memory agent if exists
    if agent_id in active_agents:
        active_agents[agent_id]["last_viewed_at"] = now

    # Update in persisted file
    try:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"

        # Find and update agent file
        for date_dir in agents_dir.iterdir():
            if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                agent_file = date_dir / f"{agent_id}.json"
                if agent_file.exists():
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        agent_data = json.load(f)
                    agent_data["last_viewed_at"] = now
                    with open(agent_file, 'w', encoding='utf-8') as f:
                        json.dump(agent_data, f, indent=2, ensure_ascii=False)

        # Update index
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            for agent in index.get("agents", []):
                if agent["id"] == agent_id:
                    agent["last_viewed_at"] = now
                    break
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

        return {"status": "ok", "last_viewed_at": now}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/agents/{agent_id}/archive")
async def archive_agent(agent_id: str, user: str = Depends(verify_token)):
    """Archive an agent."""
    now = datetime.now(timezone.utc).isoformat()

    # Update in-memory agent if exists
    if agent_id in active_agents:
        active_agents[agent_id]["archived"] = True
        active_agents[agent_id]["archived_at"] = now

    # Update in persisted file
    try:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"

        # Find and update agent file
        for date_dir in agents_dir.iterdir():
            if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                agent_file = date_dir / f"{agent_id}.json"
                if agent_file.exists():
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        agent_data = json.load(f)
                    agent_data["archived"] = True
                    agent_data["archived_at"] = now
                    with open(agent_file, 'w', encoding='utf-8') as f:
                        json.dump(agent_data, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())

        # Update index
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            for agent in index.get("agents", []):
                if agent["id"] == agent_id:
                    agent["archived"] = True
                    agent["archived_at"] = now
                    break
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

        # Emit update
        await sio.emit("agent_archived", {"agent_id": agent_id, "archived": True}, room=os.environ.get("CEREBRO_ROOM", "default"))

        return {"status": "ok", "archived": True, "archived_at": now}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/agents/{agent_id}/unarchive")
async def unarchive_agent(agent_id: str, user: str = Depends(verify_token)):
    """Unarchive an agent."""
    # Update in-memory agent if exists
    if agent_id in active_agents:
        active_agents[agent_id]["archived"] = False
        active_agents[agent_id]["archived_at"] = None

    # Update in persisted file
    try:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"

        # Find and update agent file
        for date_dir in agents_dir.iterdir():
            if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                agent_file = date_dir / f"{agent_id}.json"
                if agent_file.exists():
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        agent_data = json.load(f)
                    agent_data["archived"] = False
                    agent_data["archived_at"] = None
                    with open(agent_file, 'w', encoding='utf-8') as f:
                        json.dump(agent_data, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())

        # Update index
        index_file = agents_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            for agent in index.get("agents", []):
                if agent["id"] == agent_id:
                    agent["archived"] = False
                    agent["archived_at"] = None
                    break
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

        # Emit update
        await sio.emit("agent_archived", {"agent_id": agent_id, "archived": False}, room=os.environ.get("CEREBRO_ROOM", "default"))

        return {"status": "ok", "archived": False}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/agents/{agent_id}/conversation")
async def get_agent_conversation(agent_id: str, user: str = Depends(verify_token)):
    """Get full conversation history for an agent."""
    # Check in-memory first
    if agent_id in active_agents:
        return {"conversation": active_agents[agent_id].get("conversation", [])}

    # Check persisted file
    try:
        agents_dir = Path(config.AI_MEMORY_PATH) / "agents"
        for date_dir in agents_dir.iterdir():
            if date_dir.is_dir() and not date_dir.name.endswith('.json'):
                agent_file = date_dir / f"{agent_id}.json"
                if agent_file.exists():
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        agent_data = json.load(f)
                    return {"conversation": agent_data.get("conversation", [])}

        raise HTTPException(status_code=404, detail="Agent not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Notification Endpoints
# ============================================================================

@app.get("/notifications")
async def get_notifications(limit: int = 20, unread_only: bool = False, user: str = Depends(verify_token)):
    """Get recent notifications."""
    filtered = notifications
    if unread_only:
        filtered = [n for n in notifications if not n.get("read", False)]

    unread_count = len([n for n in notifications if not n.get("read", False)])

    return {
        "notifications": filtered[:limit],
        "total": len(filtered),
        "unread_count": unread_count
    }


@app.post("/notifications/{notif_id}/read")
async def mark_notification_read(notif_id: str, user: str = Depends(verify_token)):
    """Mark a single notification as read."""
    for notif in notifications:
        if notif["id"] == notif_id:
            notif["read"] = True
            return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Notification not found")


@app.post("/notifications/read-all")
async def mark_all_notifications_read(user: str = Depends(verify_token)):
    """Mark all notifications as read."""
    for notif in notifications:
        notif["read"] = True
    return {"status": "ok", "count": len(notifications)}


@app.get("/suggestions")
async def get_smart_suggestions(user: str = Depends(verify_token)):
    """Get AI-powered project suggestions based on AI Memory analysis."""
    suggestions = []

    try:
        memory_path = Path(config.AI_MEMORY_PATH)

        # 1. Check active work from quick_facts - HIGHEST PRIORITY
        quick_facts_path = memory_path / "quick_facts.json"
        facts = {}
        if quick_facts_path.exists():
            with open(quick_facts_path) as f:
                facts = json.load(f)
                if "active_work" in facts and facts["active_work"]:
                    work = facts["active_work"]
                    suggestions.append({
                        "type": "active_project",
                        "title": f"Continue: {work.get('project', 'Project')[:30]}",
                        "description": work.get("next_action", "Continue where you left off")[:150],
                        "phase": work.get("phase_name", work.get("current_phase", "")),
                        "confidence": 0.98,
                        "action": f"Let's continue working on {work.get('project', 'the project')}. Last action: {work.get('next_action', 'Continue from where we left off')}",
                        "icon": "rocket",
                        "priority": 1,
                        "actionType": "chat"
                    })

        # 2. Check recent agent outputs for recommendations
        agents_path = memory_path / "agents"
        if agents_path.exists():
            index_file = agents_path / "index.json"
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    agent_index = json.load(f)
                    recent_agents = agent_index.get("agents", [])[:3]
                    for agent in recent_agents:
                        if agent.get("status") == "completed":
                            suggestions.append({
                                "type": "agent_followup",
                                "title": f"Follow up: Agent #{agent['id'][:8]}",
                                "description": agent.get("task", "")[:100],
                                "confidence": 0.85,
                                "action": f"Review agent #{agent['id'][:8]} output and suggest next steps based on the research",
                                "icon": "robot",
                                "priority": 2,
                                "agentId": agent['id']
                            })

        # 2. Check recent projects from AI Memory
        projects_path = memory_path / "projects"
        if projects_path.exists():
            for project_file in sorted(projects_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                try:
                    with open(project_file) as f:
                        project = json.load(f)
                        if project.get("status") not in ["completed", "archived"]:
                            suggestions.append({
                                "type": "stale_project",
                                "title": project.get("name", project_file.stem),
                                "description": project.get("description", "Project needs attention"),
                                "confidence": 0.7,
                                "action": f"Review project: {project.get('name', project_file.stem)}",
                                "icon": "folder",
                                "priority": 2
                            })
                except:
                    pass

        # 3. Check for recent learnings that might need follow-up
        learnings_path = memory_path / "learnings"
        if learnings_path.exists():
            recent_learnings = sorted(learnings_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            if recent_learnings:
                suggestions.append({
                    "type": "review",
                    "title": "Review Recent Learnings",
                    "description": f"{len(recent_learnings)} learnings from recent sessions",
                    "confidence": 0.6,
                    "action": "Show me my recent AI Memory learnings",
                    "icon": "brain",
                    "priority": 3
                })

        # 4. Time-based suggestions
        hour = datetime.now().hour
        if hour < 10:
            suggestions.append({
                "type": "routine",
                "title": "Morning Briefing",
                "description": "Start your day with a system check",
                "confidence": 0.8,
                "action": "Run a system health check and show me what needs attention",
                "icon": "sun",
                "priority": 4
            })
        elif hour >= 22:
            suggestions.append({
                "type": "routine",
                "title": "End of Day Review",
                "description": "Save today's work and plan for tomorrow",
                "confidence": 0.75,
                "action": "Summarize what we accomplished today and save to memory",
                "icon": "moon",
                "priority": 4
            })

        # 5. Quick useful actions based on context
        suggestions.append({
            "type": "quick_action",
            "title": "Memory Search",
            "description": "Search through your AI Memory for past solutions",
            "confidence": 0.7,
            "action": "Search AI Memory",
            "icon": "search",
            "priority": 4,
            "actionType": "search"
        })

        # 6. Spawn a research agent
        suggestions.append({
            "type": "quick_action",
            "title": "Research Something",
            "description": "Spawn an agent to research any topic deeply",
            "confidence": 0.65,
            "action": "Research: ",
            "icon": "brain",
            "priority": 5,
            "actionType": "agent_prompt"
        })

        # 7. Code review suggestion if there are recent code files
        suggestions.append({
            "type": "quick_action",
            "title": "Code Review",
            "description": "Have an agent review recent code changes",
            "confidence": 0.6,
            "action": "Review recent code changes in my projects and suggest improvements",
            "icon": "code",
            "priority": 6,
            "actionType": "agent"
        })

        # Sort by priority and confidence
        suggestions.sort(key=lambda x: (x.get("priority", 99), -x.get("confidence", 0)))

    except Exception as e:
        print(f"Error generating suggestions: {e}")
        # Fallback suggestions
        suggestions = [
            {
                "type": "default",
                "title": "Check AI Memory",
                "description": "See what's stored in your memory system",
                "confidence": 0.5,
                "action": "Run a memory health check and show me stats",
                "icon": "database",
                "priority": 5
            }
        ]

    return {"suggestions": suggestions[:6]}  # Return top 6 suggestions


@app.post("/suggestions/analyze")
async def analyze_and_suggest(user: str = Depends(verify_token)):
    """
    Deep analysis endpoint that scans the system and generates intelligent,
    dynamic suggestions including self-improvement opportunities.
    """
    import random

    suggestions = []
    memory_path = Path(config.AI_MEMORY_PATH)

    # ============================================================
    # CATEGORY 1: ACTIVE WORK (highest priority)
    # ============================================================
    quick_facts_path = memory_path / "quick_facts.json"
    if quick_facts_path.exists():
        try:
            with open(quick_facts_path) as f:
                facts = json.load(f)
                if "active_work" in facts and facts["active_work"]:
                    work = facts["active_work"]
                    suggestions.append({
                        "category": "active",
                        "type": "active_project",
                        "title": f"Continue: {work.get('project', 'Project')[:35]}",
                        "description": work.get("next_action", "Continue where you left off")[:150],
                        "confidence": 0.98,
                        "action": f"Let's continue working on {work.get('project', 'the project')}. Last action: {work.get('next_action', 'Continue')}",
                        "icon": "rocket",
                        "reason": "This is your active project from last session"
                    })
        except Exception as e:
            print(f"Error reading quick_facts: {e}")

    # ============================================================
    # CATEGORY 2: SELF-IMPROVEMENT SUGGESTIONS
    # ============================================================
    self_improvement_pool = [
        {
            "title": "Optimize AI Memory Embeddings",
            "description": "Re-index old memories with better embeddings for faster search",
            "action": "Analyze my AI Memory system and suggest optimizations for the embedding index",
            "icon": "upgrade",
            "reason": "Better embeddings = faster & more accurate memory recall"
        },
        {
            "title": "Learn Your Communication Style",
            "description": "Analyze recent conversations to better match your preferences",
            "action": "Analyze my recent conversations and learn my communication patterns and preferences",
            "icon": "brain",
            "reason": "I can adapt to serve you better"
        },
        {
            "title": "Update My Knowledge Base",
            "description": "Check for new tools, APIs, or techniques I should know about",
            "action": "Research the latest developments in AI assistants, coding tools, and automation that could help Professor",
            "icon": "learn",
            "reason": "Staying current helps me help you better"
        },
        {
            "title": "Improve Error Handling",
            "description": "Review recent failures and learn from mistakes",
            "action": "Review my failure_memory and identify patterns in my mistakes to avoid repeating them",
            "icon": "fix",
            "reason": "Learning from failures makes me more reliable"
        },
        {
            "title": "Expand Code Vault",
            "description": "Save more proven patterns from recent successful work",
            "action": "Review recent successful code I've written and save useful patterns to the code vault",
            "icon": "code",
            "reason": "Building a library of proven solutions"
        },
        {
            "title": "Optimize Response Speed",
            "description": "Analyze which operations are slow and find faster approaches",
            "action": "Profile my common operations and identify bottlenecks I can optimize",
            "icon": "upgrade",
            "reason": "Faster responses = better productivity"
        },
        {
            "title": "Better Context Retention",
            "description": "Improve how I remember context across sessions",
            "action": "Analyze how I'm using AI Memory and suggest improvements for context retention",
            "icon": "brain",
            "reason": "Remembering more = fewer repeated explanations"
        },
        {
            "title": "Learn New Skills",
            "description": "Research tools and techniques that could expand my capabilities",
            "action": "Research new MCP tools, Claude capabilities, or automation techniques I should learn",
            "icon": "star",
            "reason": "Growing capabilities to serve you better"
        }
    ]

    # Pick 2-3 random self-improvement suggestions
    selected_self_improvement = random.sample(self_improvement_pool, min(3, len(self_improvement_pool)))
    for i, suggestion in enumerate(selected_self_improvement):
        suggestion["category"] = "self_improvement"
        suggestion["type"] = "self_improvement"
        suggestion["confidence"] = round(random.uniform(0.75, 0.90), 2)
        suggestions.append(suggestion)

    # ============================================================
    # CATEGORY 3: PROJECT-BASED SUGGESTIONS (scan actual repos)
    # ============================================================
    projects_scanned = []

    # Check for git repos in common locations
    common_project_paths = [
        Path(os.environ.get("CEREBRO_MCP_SRC", os.path.expanduser("~/NAS-cerebral-interface"))),
        Path(config.AI_MEMORY_PATH) / "projects" / "digital_companion" / "cerebro",
        Path(os.path.expanduser("~/projects")),
    ]

    for project_path in common_project_paths:
        if not project_path.exists():
            continue
        try:
            # Check if it's a git repo
            git_dir = project_path / ".git"
            if git_dir.exists():
                # Get git status
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(project_path),
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    changed_files = len(result.stdout.strip().split('\n'))
                    projects_scanned.append({
                        "path": str(project_path),
                        "name": project_path.name,
                        "changes": changed_files
                    })
        except Exception as e:
            print(f"Error scanning {project_path}: {e}")

    # Generate suggestions from scanned projects
    for project in projects_scanned[:2]:
        suggestions.append({
            "category": "projects",
            "type": "git_changes",
            "title": f"Uncommitted: {project['name']}",
            "description": f"{project['changes']} file(s) with changes waiting to be committed",
            "confidence": 0.85,
            "action": f"Review changes in {project['name']} and help me commit them properly",
            "icon": "code",
            "reason": f"Found {project['changes']} uncommitted changes"
        })

    # ============================================================
    # CATEGORY 4: LEARNING OPPORTUNITIES
    # ============================================================
    learning_pool = [
        {
            "title": "Explore Codebase Patterns",
            "description": "Analyze your codebase and document common patterns",
            "action": "Explore my main projects and document the coding patterns and architecture decisions",
            "icon": "search"
        },
        {
            "title": "Create Technical Documentation",
            "description": "Generate docs for undocumented code",
            "action": "Find code without documentation and generate helpful docs",
            "icon": "folder"
        },
        {
            "title": "Security Audit",
            "description": "Check for potential security issues in recent code",
            "action": "Review my recent code changes for security vulnerabilities or best practice violations",
            "icon": "fix"
        },
        {
            "title": "Performance Analysis",
            "description": "Profile and optimize slow operations",
            "action": "Analyze my projects for performance bottlenecks and suggest optimizations",
            "icon": "upgrade"
        }
    ]

    selected_learning = random.sample(learning_pool, min(2, len(learning_pool)))
    for suggestion in selected_learning:
        suggestion["category"] = "learning"
        suggestion["type"] = "learning"
        suggestion["confidence"] = round(random.uniform(0.60, 0.80), 2)
        suggestion["reason"] = "Continuous improvement opportunity"
        suggestions.append(suggestion)

    # ============================================================
    # CATEGORY 5: MAINTENANCE TASKS
    # ============================================================
    maintenance_suggestions = []

    # Check AI Memory stats
    try:
        conversations_path = memory_path / "conversations"
        if conversations_path.exists():
            conv_count = len(list(conversations_path.glob("*.json")))
            if conv_count > 50:
                maintenance_suggestions.append({
                    "title": "Archive Old Conversations",
                    "description": f"You have {conv_count} conversations that could be archived",
                    "action": "Review and archive old conversations to keep AI Memory organized",
                    "icon": "database",
                    "reason": f"{conv_count} conversations taking up space"
                })

        # Check for orphaned files
        learnings_path = memory_path / "learnings"
        if learnings_path.exists():
            learning_count = len(list(learnings_path.glob("*.json")))
            if learning_count > 20:
                maintenance_suggestions.append({
                    "title": "Review Learnings",
                    "description": f"{learning_count} learnings stored - review and consolidate",
                    "action": "Review my stored learnings and consolidate similar ones",
                    "icon": "brain",
                    "reason": "Consolidation improves recall accuracy"
                })

    except Exception as e:
        print(f"Error checking memory stats: {e}")

    # Add 1-2 maintenance suggestions if available
    for suggestion in maintenance_suggestions[:2]:
        suggestion["category"] = "maintenance"
        suggestion["type"] = "maintenance"
        suggestion["confidence"] = round(random.uniform(0.55, 0.70), 2)
        suggestions.append(suggestion)

    # ============================================================
    # CATEGORY 6: CONTEXTUAL TASKS (time-based, recent activity)
    # ============================================================
    hour = datetime.now().hour

    if hour < 10:
        suggestions.append({
            "category": "tasks",
            "type": "routine",
            "title": "Morning System Check",
            "description": "Start fresh with a quick system health check",
            "confidence": 0.80,
            "action": "Run a system health check: verify NAS connection, AI Memory status, and DGX Spark availability",
            "icon": "sun",
            "reason": "Good morning! Let's make sure everything is ready"
        })
    elif hour >= 17 and hour < 20:
        suggestions.append({
            "category": "tasks",
            "type": "routine",
            "title": "Save Today's Progress",
            "description": "Capture today's work to memory before winding down",
            "confidence": 0.75,
            "action": "Summarize what we accomplished today and save key learnings to AI Memory",
            "icon": "moon",
            "reason": "End of day - preserve your progress"
        })
    elif hour >= 22:
        suggestions.append({
            "category": "tasks",
            "type": "routine",
            "title": "Quick Memory Sync",
            "description": "Ensure all work is saved before you rest",
            "confidence": 0.70,
            "action": "Verify all important work is saved to AI Memory",
            "icon": "database",
            "reason": "Late night - let's make sure nothing is lost"
        })

    # ============================================================
    # SHUFFLE AND FINALIZE
    # ============================================================

    # Keep active work at top, shuffle the rest within categories
    active_suggestions = [s for s in suggestions if s.get("category") == "active"]
    other_suggestions = [s for s in suggestions if s.get("category") != "active"]

    # Sort by category, then shuffle within each category for variety
    categories_order = ["self_improvement", "projects", "tasks", "learning", "maintenance"]
    sorted_others = []
    for cat in categories_order:
        cat_items = [s for s in other_suggestions if s.get("category") == cat]
        random.shuffle(cat_items)  # Randomize within category
        sorted_others.extend(cat_items)

    final_suggestions = active_suggestions + sorted_others

    return {"suggestions": final_suggestions[:10]}  # Return top 10


# ============================================================================
# Interest Suggestions API - Personalized suggestions with dismiss tracking
# ============================================================================

@app.get("/suggestions/interests")
async def get_interest_suggestions(user: str = Depends(verify_token)):
    """Get the three smart interest suggestions (Claude-generated, cached)."""
    try:
        # Gather all excluded topics
        all_excluded = (
            dismiss_tracker.get_excluded_topics("fun_project") +
            dismiss_tracker.get_excluded_topics("business_feature") +
            dismiss_tracker.get_excluded_topics("claude_choice")
        )

        # Get suggestions (uses cache if fresh, otherwise generates)
        suggestions = await smart_suggestion_generator.generate_all_suggestions(
            excluded=all_excluded,
            force_refresh=False
        )
        return suggestions
    except Exception as e:
        print(f"Error generating interest suggestions: {e}")
        raise HTTPException(500, str(e))


@app.post("/suggestions/interests/refresh")
async def refresh_interest_suggestions(user: str = Depends(verify_token)):
    """Force regenerate interest suggestions using Claude."""
    try:
        all_excluded = (
            dismiss_tracker.get_excluded_topics("fun_project") +
            dismiss_tracker.get_excluded_topics("business_feature") +
            dismiss_tracker.get_excluded_topics("claude_choice")
        )

        # Force refresh - calls Claude
        suggestions = await smart_suggestion_generator.generate_all_suggestions(
            excluded=all_excluded,
            force_refresh=True
        )
        return {"status": "refreshed", "suggestions": suggestions}
    except Exception as e:
        print(f"Error refreshing suggestions: {e}")
        raise HTTPException(500, str(e))


class CognitiveSuggestion(BaseModel):
    title: str
    description: str
    confidence: float = 0.7
    source: str = "cognitive_loop"


# Insights/suggestions storage
INSIGHTS_FILE = Path(config.AI_MEMORY_PATH) / "cerebro" / "insights.json"

def load_insights() -> list:
    """Load insights from file."""
    try:
        if INSIGHTS_FILE.exists():
            return json.loads(INSIGHTS_FILE.read_text())
        return []
    except Exception:
        return []

def save_insights(insights: list):
    """Save insights to file."""
    try:
        INSIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        INSIGHTS_FILE.write_text(json.dumps(insights[-50:], indent=2, default=str))  # Keep last 50
    except Exception as e:
        print(f"Failed to save insights: {e}")


@app.get("/api/insights")
async def get_insights(user: str = Depends(verify_token)):
    """Get recent insights from the cognitive loop."""
    insights = load_insights()
    # Return most recent first, limit to 20
    return {"insights": insights[:20]}


def clean_insight_field(text: str, max_length: int = 100) -> str:
    """Clean raw LLM output for insight display. Safety net for API boundary."""
    import re
    if not text:
        return "Autonomous insight"
    text = re.sub(r'\*\*|##|__|~~|`', '', text)
    text = re.sub(r'^(?:Final\s+)?Decision:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^To\s+assist\s+Professor\s+Lopez\s+in\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Based\s+on\s+the\s+tools?\s+used[,.]?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return "Autonomous insight"
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] or text[:max_length]
    return text


@app.post("/suggestions")
async def create_cognitive_suggestion(request: CognitiveSuggestion, user: str = Depends(verify_token)):
    """Create a suggestion from the cognitive loop and emit to connected clients."""
    suggestion = {
        "id": f"insight_{uuid.uuid4().hex[:8]}",
        "category": "cognitive",
        "type": "ai_suggestion",
        "title": clean_insight_field(request.title, max_length=100),
        "description": clean_insight_field(request.description, max_length=500),
        "confidence": request.confidence,
        "source": request.source,
        "icon": "brain",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Save to file for persistence
    insights = load_insights()
    insights.insert(0, suggestion)
    save_insights(insights)

    # Emit to connected clients via Socket.IO
    if sio:
        await sio.emit("cognitive_suggestion", suggestion, room=os.environ.get("CEREBRO_ROOM", "default"))

    return {"success": True, "suggestion": suggestion}


@app.post("/suggestions/dismiss")
async def dismiss_suggestion(request: Request, user: str = Depends(verify_token)):
    """Dismiss a suggestion and regenerate from cache/fallback."""
    try:
        data = await request.json()
        suggestion_id = data.get("suggestion_id")
        category = data.get("category", "")
        title = data.get("title", "")
        permanent = data.get("permanent", False)
        regenerate = data.get("regenerate", True)
        interest_type = data.get("interest_type")

        dismiss_tracker.dismiss(suggestion_id, category, title, permanent)

        replacement = None
        if regenerate and interest_type:
            # Get fresh exclusions after the dismiss
            all_excluded = (
                dismiss_tracker.get_excluded_topics("fun_project") +
                dismiss_tracker.get_excluded_topics("business_feature") +
                dismiss_tracker.get_excluded_topics("claude_choice")
            )

            # Use fallback for quick replacement (don't call Claude for each dismiss)
            suggestions = smart_suggestion_generator._get_fallback_suggestions(all_excluded)

            # Map interest_type to suggestion key
            type_to_key = {"fun": "fun_project", "business": "business_feature", "claude": "claude_choice"}
            suggestion_key = type_to_key.get(interest_type)
            if suggestion_key and suggestion_key in suggestions:
                replacement = suggestions[suggestion_key]

        return {"success": True, "dismissed_id": suggestion_id, "replacement": replacement}
    except Exception as e:
        print(f"Error dismissing suggestion: {e}")
        raise HTTPException(500, str(e))


@app.get("/system-status")
async def get_system_status(user: str = Depends(verify_token)):
    """Get real-time status of all systems."""
    import socket

    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {}
    }

    # Check DGX Spark
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        dgx_host = os.environ.get("DGX_HOST", "")
        result = sock.connect_ex((dgx_host, 11434))
        status["services"]["dgx_ollama"] = {
            "name": "DGX Spark (Ollama)",
            "status": "online" if result == 0 else "offline",
            "host": f"{dgx_host}:11434"
        }
        sock.close()
    except:
        status["services"]["dgx_ollama"] = {"name": "DGX Spark", "status": "unknown"}

    # Check NAS
    try:
        nas_path = Path(config.AI_MEMORY_PATH)
        if nas_path.exists():
            status["services"]["nas"] = {
                "name": "NAS (AI Memory)",
                "status": "online",
                "path": str(nas_path)
            }
        else:
            status["services"]["nas"] = {"name": "NAS", "status": "offline"}
    except:
        status["services"]["nas"] = {"name": "NAS", "status": "error"}

    # Check Redis
    try:
        if redis:
            await redis.ping()
            status["services"]["redis"] = {"name": "Redis", "status": "online"}
        else:
            status["services"]["redis"] = {"name": "Redis", "status": "offline"}
    except:
        status["services"]["redis"] = {"name": "Redis", "status": "error"}

    # Check emotion service on DGX
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        dgx_host = os.environ.get("DGX_HOST", "")
        result = sock.connect_ex((dgx_host, 8766))
        status["services"]["emotion"] = {
            "name": "Emotion Service",
            "status": "online" if result == 0 else "offline",
            "host": f"{dgx_host}:8766"
        }
        sock.close()
    except:
        status["services"]["emotion"] = {"name": "Emotion Service", "status": "unknown"}

    # Memory stats
    try:
        memory_path = Path(config.AI_MEMORY_PATH)
        conv_count = len(list((memory_path / "conversations").glob("*.json"))) if (memory_path / "conversations").exists() else 0
        agent_count = 0
        agents_index = memory_path / "agents" / "index.json"
        if agents_index.exists():
            with open(agents_index) as f:
                agent_count = len(json.load(f).get("agents", []))

        status["memory"] = {
            "conversations": conv_count,
            "agents": agent_count,
            "path": str(memory_path)
        }
    except:
        status["memory"] = {"conversations": 0, "agents": 0}

    return status

@app.get("/memory/search")
async def search_memory(q: str, limit: int = 10, user: str = Depends(verify_token)):
    """Search through AI Memory."""
    results = []

    try:
        memory_path = Path(config.AI_MEMORY_PATH)

        # Search conversations
        conv_path = memory_path / "conversations"
        if conv_path.exists():
            for conv_file in sorted(conv_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
                try:
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if q.lower() in content.lower():
                            conv = json.load(open(conv_file, 'r', encoding='utf-8'))
                            results.append({
                                "type": "conversation",
                                "id": conv_file.stem,
                                "title": conv.get("summary", conv_file.stem)[:100] if isinstance(conv, dict) else conv_file.stem,
                                "date": datetime.fromtimestamp(conv_file.stat().st_mtime).isoformat(),
                                "snippet": content[:200] + "..." if len(content) > 200 else content
                            })
                            if len(results) >= limit:
                                break
                except:
                    pass

        # Search learnings
        learn_path = memory_path / "learnings"
        if learn_path.exists() and len(results) < limit:
            for learn_file in sorted(learn_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
                try:
                    with open(learn_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if q.lower() in content.lower():
                            results.append({
                                "type": "learning",
                                "id": learn_file.stem,
                                "title": learn_file.stem,
                                "date": datetime.fromtimestamp(learn_file.stat().st_mtime).isoformat(),
                                "snippet": content[:200] + "..."
                            })
                            if len(results) >= limit:
                                break
                except:
                    pass

        # Search quick_facts
        qf_path = memory_path / "quick_facts.json"
        if qf_path.exists() and len(results) < limit:
            with open(qf_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if q.lower() in content.lower():
                    results.append({
                        "type": "quick_facts",
                        "id": "quick_facts",
                        "title": "Quick Facts",
                        "date": datetime.fromtimestamp(qf_path.stat().st_mtime).isoformat(),
                        "snippet": "Contains matching content in quick facts"
                    })

    except Exception as e:
        print(f"Search error: {e}")

    return {"query": q, "results": results, "count": len(results)}


# ============================================================================
# MCP Bridge Endpoints - Goals, Predictions, and Proactive Analysis
# ============================================================================

from mcp_bridge import get_mcp_bridge
from cognitive_loop.goal_pursuit import get_goal_pursuit_engine, GoalMode

# Pydantic models for MCP endpoints
class GoalCreate(BaseModel):
    description: str
    priority: str = "medium"
    deadline: Optional[str] = None      # ISO date string
    target_value: Optional[float] = None
    target_unit: str = ""
    mode: str = "monitor"               # monitor | think | act

class GoalUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    mode: Optional[str] = None
    description: Optional[str] = None
    deadline: Optional[str] = None
    target_value: Optional[float] = None
    target_unit: Optional[str] = None
    add_blocker: Optional[str] = None
    add_subgoal: Optional[str] = None
    progress_update: Optional[str] = None

class ChatAnalyzeRequest(BaseModel):
    message: str

# Initialize MCP Bridge
mcp_bridge = get_mcp_bridge()


@app.get("/api/goals")
async def list_goals(user: str = Depends(verify_token)):
    """List all goals with progress metrics via GoalPursuitEngine."""
    try:
        engine = get_goal_pursuit_engine()
        goals_with_progress = engine.get_goals_with_progress()
        return {"success": True, "goals": goals_with_progress}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list goals: {e}")


@app.post("/api/goals")
async def create_goal(goal: GoalCreate, user: str = Depends(verify_token)):
    """Create a new goal via GoalPursuitEngine."""
    try:
        engine = get_goal_pursuit_engine()
        new_goal = engine.create_goal(
            description=goal.description,
            priority=goal.priority,
            deadline=goal.deadline,
            target_value=goal.target_value,
            target_unit=goal.target_unit,
            mode=goal.mode,
        )
        return {"success": True, "goal": new_goal.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create goal: {e}")


@app.post("/api/goals/detect")
async def detect_goals(request: ChatAnalyzeRequest, user: str = Depends(verify_token)):
    """Detect goals from user text (still via MCP bridge)."""
    result = await mcp_bridge.goals("detect", text=request.message)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to detect goals"))
    return result


@app.get("/api/goals/{goal_id}")
async def get_goal(goal_id: str, user: str = Depends(verify_token)):
    """Get a specific goal by ID via GoalPursuitEngine."""
    try:
        engine = get_goal_pursuit_engine()
        goal = engine.get_goal(goal_id)
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")
        return {"success": True, "goal": goal.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get goal: {e}")


@app.patch("/api/goals/{goal_id}")
async def update_goal(goal_id: str, update: GoalUpdate, user: str = Depends(verify_token)):
    """Update goal fields directly via GoalPursuitEngine."""
    try:
        engine = get_goal_pursuit_engine()
        goal = engine.get_goal(goal_id)
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")

        if update.status is not None:
            goal.status = update.status
        if update.priority is not None:
            goal.priority = update.priority
        if update.mode is not None:
            goal.mode = update.mode
        if update.description is not None:
            goal.description = update.description
        if update.deadline is not None:
            goal.deadline = update.deadline
        if update.target_value is not None:
            goal.target_value = update.target_value
        if update.target_unit is not None:
            goal.target_unit = update.target_unit

        goal.updated_at = datetime.now(timezone.utc).isoformat()
        engine._save_goal(goal_id)

        return {"success": True, "goal": goal.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update goal: {e}")


@app.post("/api/goals/{goal_id}/complete")
async def complete_goal(goal_id: str, user: str = Depends(verify_token)):
    """Mark a goal as completed and create a stored item."""
    try:
        engine = get_goal_pursuit_engine()
        goal = engine.complete_goal(goal_id)
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")

        # Create "goal_complete" stored item
        item = {
            "id": f"goal_complete_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:6]}",
            "type": "goal_complete",
            "title": f"Goal Complete: {goal.description[:80]}",
            "content": f"Goal \"{goal.description}\" has been marked as completed.",
            "metadata": {"goal_id": goal_id, "source": "heartbeat_goals"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
            "source_id": goal_id,
        }
        items = _load_stored_items()
        items.insert(0, item)
        _save_stored_items(items)
        await sio.emit("cerebro_stored_item_added", item, room=os.environ.get("CEREBRO_ROOM", "default"))

        return {"success": True, "goal": goal.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to complete goal: {e}")


@app.delete("/api/goals/{goal_id}")
async def delete_goal(goal_id: str, user: str = Depends(verify_token)):
    """Abandon (soft-delete) a goal."""
    try:
        engine = get_goal_pursuit_engine()
        goal = engine.get_goal(goal_id)
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")

        goal.status = "abandoned"
        goal.updated_at = datetime.now(timezone.utc).isoformat()
        engine._save_goal(goal_id)

        return {"success": True, "goal": goal.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete goal: {e}")


@app.get("/api/predictions")
async def get_predictions(context: str, user: str = Depends(verify_token)):
    """Get failure predictions for a given context."""
    result = await mcp_bridge.predict("anticipate_failures", context=context)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to get predictions"))
    return result


@app.get("/api/predictions/preventive")
async def get_preventive_actions(context: str, user: str = Depends(verify_token)):
    """Get preventive actions for a given context."""
    result = await mcp_bridge.predict("preventive_actions", context=context)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to get preventive actions"))
    return result


@app.post("/api/chat/analyze")
async def analyze_before_chat(request: ChatAnalyzeRequest, user: str = Depends(verify_token)):
    """
    Analyze a message BEFORE processing for:
    - Warnings about potential failures
    - Relevant active goals
    - Applicable learnings from past

    This enables predictive interrupts.
    """
    result = await mcp_bridge.get_proactive_context(request.message)
    return result


@app.get("/api/learnings")
async def find_learnings(problem: str, limit: int = 5, user: str = Depends(verify_token)):
    """Find learnings relevant to a problem."""
    result = await mcp_bridge.learning("find", problem=problem, limit=limit)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to find learnings"))
    return result


class RecordLearningRequest(BaseModel):
    type: str = "solution"  # solution, failure, antipattern
    problem: str
    solution: Optional[str] = None
    what_not_to_do: Optional[str] = None
    tags: Optional[list] = []
    source: Optional[str] = "cognitive_loop"


@app.post("/api/learnings")
async def record_learning(request: RecordLearningRequest, user: str = Depends(verify_token)):
    """Record a new learning from cognitive loop or user."""
    try:
        result = await mcp_bridge.learning(
            "record",
            type=request.type,
            problem=request.problem,
            solution=request.solution,
            what_not_to_do=request.what_not_to_do,
            tags=request.tags
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/causal/effects")
async def get_causal_effects(cause: str, user: str = Depends(verify_token)):
    """Get effects of a cause from the causal model."""
    result = await mcp_bridge.causal("find_effects", cause=cause)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to get effects"))
    return result


@app.get("/api/causal/interventions")
async def get_interventions(effect: str, user: str = Depends(verify_token)):
    """Get interventions to prevent/achieve an effect."""
    result = await mcp_bridge.causal("get_interventions", effect=effect)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to get interventions"))
    return result


# ============================================================================
# Cognitive Loop Tool Endpoints - Used by autonomous thinking
# ============================================================================

@app.get("/api/corrections")
async def get_corrections(topic: Optional[str] = None, limit: int = 10, user: str = Depends(verify_token)):
    """Get known corrections to avoid repeating mistakes."""
    try:
        result = await mcp_bridge.get_corrections(topic=topic, limit=limit)
        return result
    except Exception as e:
        # Fallback to quick_facts
        try:
            qf_path = Path(config.AI_MEMORY_PATH) / "quick_facts.json"
            if qf_path.exists():
                with open(qf_path, 'r', encoding='utf-8') as f:
                    qf = json.load(f)
                    raw = qf.get("top_corrections", [])
                    if isinstance(raw, dict):
                        raw = raw.get("most_common", [])
                    return {"corrections": raw if isinstance(raw, list) else [], "source": "quick_facts"}
        except:
            pass
        return {"corrections": [], "error": str(e)}


@app.get("/api/user-profile")
async def get_user_profile(category: str = "all", user: str = Depends(verify_token)):
    """Get user profile from AI Memory."""
    try:
        result = await mcp_bridge.get_user_profile(category=category)
        return result
    except Exception as e:
        # Fallback to quick_facts
        try:
            qf_path = Path(config.AI_MEMORY_PATH) / "quick_facts.json"
            if qf_path.exists():
                with open(qf_path, 'r', encoding='utf-8') as f:
                    qf = json.load(f)
                    return {
                        "identity": {"name": os.environ.get("CEREBRO_USER_NAME", "")},
                        "preferences": qf.get("preferences", {}),
                        "goals": qf.get("active_goals", []),
                        "source": "quick_facts"
                    }
        except:
            pass
        return {"profile": {}, "error": str(e)}


@app.get("/api/learnings/search")
async def search_learnings(problem: str, limit: int = 5, user: str = Depends(verify_token)):
    """Search learnings for solutions to a problem."""
    try:
        result = await mcp_bridge.learning("find", type="solution", problem=problem, limit=limit)
        return result
    except Exception as e:
        return {"solutions": [], "problem": problem, "error": str(e)}


@app.get("/api/causal")
async def causal_query(query: str, action: str = "find_causes", user: str = Depends(verify_token)):
    """General causal model query."""
    try:
        if action == "find_causes":
            result = await mcp_bridge.causal("find_causes", effect=query)
        elif action == "find_effects":
            result = await mcp_bridge.causal("find_effects", cause=query)
        elif action == "what_if":
            result = await mcp_bridge.causal("what_if", intervention=query)
        else:
            result = await mcp_bridge.causal("search", query=query)
        return result
    except Exception as e:
        return {"results": [], "query": query, "action": action, "error": str(e)}


@app.get("/api/predictions")
async def get_predictions_for_context(context: str = "", user: str = Depends(verify_token)):
    """Get predictions relevant to a context."""
    try:
        result = await mcp_bridge.predict("anticipate_failures", context=context)
        return result
    except Exception as e:
        return {"predictions": [], "context": context, "error": str(e)}


# ============================================================================
# Proactive Agent Endpoints - Autonomous behavior management
# ============================================================================

@app.get("/api/proactive/status")
async def get_proactive_status(user: str = Depends(verify_token)):
    """Get status of the proactive agent system."""
    if not proactive_manager:
        return {"enabled": False, "error": "Proactive manager not initialized"}
    return proactive_manager.get_status()


@app.post("/api/proactive/pause")
async def pause_proactive(minutes: int = 60, user: str = Depends(verify_token)):
    """Pause proactive actions for specified minutes."""
    if not proactive_manager:
        raise HTTPException(status_code=503, detail="Proactive manager not available")
    proactive_manager.pause(minutes)
    return {"status": "paused", "minutes": minutes, "paused_until": proactive_manager.paused_until.isoformat() if proactive_manager.paused_until else None}


@app.post("/api/proactive/resume")
async def resume_proactive(user: str = Depends(verify_token)):
    """Resume proactive actions."""
    if not proactive_manager:
        raise HTTPException(status_code=503, detail="Proactive manager not available")
    proactive_manager.resume()
    return {"status": "resumed"}


@app.get("/api/proactive/pending")
async def get_pending_actions(user: str = Depends(verify_token)):
    """Get pending proactive actions awaiting approval."""
    if not proactive_manager:
        return {"actions": []}
    return {"actions": proactive_manager.get_pending_actions()}


@app.get("/api/proactive/history")
async def get_proactive_history(limit: int = 20, user: str = Depends(verify_token)):
    """Get history of proactive actions."""
    if not proactive_manager:
        return {"actions": []}
    return {"actions": proactive_manager.get_action_history(limit)}


@app.post("/api/proactive/{action_id}/approve")
async def approve_proactive_action(action_id: str, user: str = Depends(verify_token)):
    """Approve a pending proactive action."""
    if not proactive_manager:
        raise HTTPException(status_code=503, detail="Proactive manager not available")
    result = await proactive_manager.approve_action(action_id)
    if not result:
        raise HTTPException(status_code=404, detail="Action not found or not pending")
    return {"status": "approved", "action": result.to_dict() if hasattr(result, 'to_dict') else result}


@app.post("/api/proactive/{action_id}/reject")
async def reject_proactive_action(action_id: str, reason: str = "User rejected", user: str = Depends(verify_token)):
    """Reject a pending proactive action."""
    if not proactive_manager:
        raise HTTPException(status_code=503, detail="Proactive manager not available")
    result = await proactive_manager.reject_action(action_id, reason)
    if not result:
        raise HTTPException(status_code=404, detail="Action not found or not pending")
    return {"status": "rejected", "action": result.to_dict() if hasattr(result, 'to_dict') else result}


# ============================================================================
# Self-Modification Proposals Endpoints
# ============================================================================

@app.get("/api/proposals")
async def list_proposals(status: Optional[str] = None, user: str = Depends(verify_token)):
    """List modification proposals."""
    from self_modification import get_self_mod_manager
    manager = get_self_mod_manager(Path(config.AI_MEMORY_PATH))
    if not manager:
        return {"proposals": []}
    proposals = await manager.list_proposals(status)
    return {"proposals": proposals}


@app.get("/api/proposals/{proposal_id}")
async def get_proposal_detail(proposal_id: str, user: str = Depends(verify_token)):
    """Get detailed proposal information including diff."""
    from self_modification import get_self_mod_manager
    manager = get_self_mod_manager(Path(config.AI_MEMORY_PATH))
    if not manager:
        raise HTTPException(status_code=503, detail="Self-modification manager not available")
    proposal = await manager.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal


@app.post("/api/proposals/{proposal_id}/approve")
async def approve_proposal_endpoint(proposal_id: str, user: str = Depends(verify_token)):
    """Approve and apply a modification proposal."""
    from self_modification import get_self_mod_manager
    manager = get_self_mod_manager(Path(config.AI_MEMORY_PATH))
    if not manager:
        raise HTTPException(status_code=503, detail="Self-modification manager not available")
    result = await manager.approve_proposal(proposal_id, user)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to approve"))
    return result


@app.post("/api/proposals/{proposal_id}/reject")
async def reject_proposal_endpoint(proposal_id: str, reason: str = "User rejected", user: str = Depends(verify_token)):
    """Reject a modification proposal."""
    from self_modification import get_self_mod_manager
    manager = get_self_mod_manager(Path(config.AI_MEMORY_PATH))
    if not manager:
        raise HTTPException(status_code=503, detail="Self-modification manager not available")
    result = await manager.reject_proposal(proposal_id, reason, user)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to reject"))
    return result


@app.post("/api/proposals/{proposal_id}/rollback")
async def rollback_proposal_endpoint(proposal_id: str, user: str = Depends(verify_token)):
    """Rollback a previously applied proposal."""
    from self_modification import get_self_mod_manager
    manager = get_self_mod_manager(Path(config.AI_MEMORY_PATH))
    if not manager:
        raise HTTPException(status_code=503, detail="Self-modification manager not available")
    result = await manager.rollback(proposal_id, user)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to rollback"))
    return result


@app.get("/api/proposals/history")
async def get_proposals_history(limit: int = 20, user: str = Depends(verify_token)):
    """Get modification history."""
    from self_modification import get_self_mod_manager
    manager = get_self_mod_manager(Path(config.AI_MEMORY_PATH))
    if not manager:
        return {"history": []}
    history = await manager.get_history(limit)
    return {"history": history}


# ============================================================================
# Self-Improvement System Endpoints
# ============================================================================

# Initialize self-improvement components lazily
_self_improvement_initialized = False
_git_manager = None
_staging_manager = None
_test_runner = None
_health_monitor = None
_rollback_engine = None
_improvement_engine = None
_audit_logger = None
_safe_restart = None


def _init_self_improvement():
    """Initialize self-improvement system components."""
    global _self_improvement_initialized, _git_manager, _staging_manager
    global _test_runner, _health_monitor, _rollback_engine
    global _improvement_engine, _audit_logger, _safe_restart

    if _self_improvement_initialized:
        return

    try:
        from git_manager import get_git_manager
        from staging_manager import get_staging_manager
        from test_runner import get_test_runner
        from health_monitor import get_health_monitor
        from rollback_engine import get_rollback_engine
        from improvement_engine import get_improvement_engine
        from audit_logger import get_audit_logger
        from safe_restart import get_safe_restart

        cerebro_path = Path(config.AI_MEMORY_PATH) / "projects" / "digital_companion" / "cerebro"
        audit_path = Path(config.AI_MEMORY_PATH) / "cerebro" / "audit_logs"

        # Initialize git manager
        _git_manager = get_git_manager(cerebro_path)
        if _git_manager and not _git_manager.is_initialized():
            _git_manager.initialize(create_initial_commit=True)
            print("[Self-Improvement] Git repository initialized")

        # Initialize other components
        _audit_logger = get_audit_logger(audit_path)
        _health_monitor = get_health_monitor()
        _safe_restart = get_safe_restart(
            server_path=cerebro_path / "backend" / "main.py",
            port=59000,
            health_monitor=_health_monitor
        )
        _rollback_engine = get_rollback_engine(
            git_manager=_git_manager,
            safe_restart=_safe_restart,
            health_monitor=_health_monitor,
            audit_logger=_audit_logger
        )
        _staging_manager = get_staging_manager(cerebro_path, _git_manager)
        _test_runner = get_test_runner(cerebro_path)
        _improvement_engine = get_improvement_engine(
            Path(config.AI_MEMORY_PATH),
            cerebro_path
        )

        _self_improvement_initialized = True
        print("[Self-Improvement] System initialized successfully")

    except ImportError as e:
        print(f"[Self-Improvement] Module not available: {e}")
    except Exception as e:
        print(f"[Self-Improvement] Initialization error: {e}")


@app.get("/self-improvement/opportunities")
async def get_improvement_opportunities(
    min_confidence: float = 0.6,
    limit: int = 20,
    user: str = Depends(verify_token)
):
    """
    List improvement opportunities from AI Memory patterns.

    Returns opportunities based on promoted patterns, corrections,
    and learnings from the AI Memory system.
    """
    _init_self_improvement()

    if not _improvement_engine:
        return {"opportunities": [], "error": "Improvement engine not available"}

    try:
        # Refresh opportunities from AI Memory
        await _improvement_engine.refresh_opportunities()

        # Get filtered opportunities
        opportunities = _improvement_engine.get_opportunities(
            min_confidence=min_confidence,
            limit=limit
        )

        # Get summary stats
        summary = _improvement_engine.get_summary()

        return {
            "opportunities": [o.to_dict() for o in opportunities],
            "summary": summary
        }

    except Exception as e:
        return {"opportunities": [], "error": str(e)}


@app.post("/self-improvement/stage/{proposal_id}")
async def stage_proposal(
    proposal_id: str,
    user: str = Depends(verify_token)
):
    """
    Stage a proposal for testing.

    Creates a staging branch, applies changes, and starts a staging server
    on port 59001 for testing.
    """
    _init_self_improvement()

    if not _staging_manager:
        raise HTTPException(503, "Staging manager not available")
    if not _git_manager:
        raise HTTPException(503, "Git manager not available")

    try:
        # Get the proposal
        from self_modification import get_self_mod_manager
        manager = get_self_mod_manager(Path(config.AI_MEMORY_PATH))
        if not manager:
            raise HTTPException(503, "Self-modification manager not available")

        proposal = await manager.get_proposal(proposal_id)
        if not proposal:
            raise HTTPException(404, f"Proposal {proposal_id} not found")

        # Create staging session
        session = await _staging_manager.create_staging_session(proposal_id)

        # Setup staging branch
        if not await _staging_manager.setup_staging_branch(session):
            raise HTTPException(500, f"Failed to create staging branch: {session.error}")

        # Apply the proposed changes
        changes = {proposal["file_path"]: proposal["new_content"]}
        if not await _staging_manager.apply_changes(session, changes):
            await _staging_manager.rollback_staging(session)
            raise HTTPException(500, f"Failed to apply changes: {session.error}")

        # Start staging server
        if not await _staging_manager.start_staging_server(session):
            await _staging_manager.rollback_staging(session)
            raise HTTPException(500, f"Failed to start staging server: {session.error}")

        # Log to audit trail
        if _audit_logger:
            await _audit_logger.log_action(
                action="stage",
                proposal_id=proposal_id,
                session_id=session.session_id,
                details={"branch": session.branch_name, "port": session.staging_port}
            )

        return {
            "session_id": session.session_id,
            "branch": session.branch_name,
            "staging_port": session.staging_port,
            "status": session.status.value,
            "message": f"Staging server running on port {session.staging_port}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/self-improvement/test/{session_id}")
async def run_staging_tests(
    session_id: str,
    user: str = Depends(verify_token)
):
    """
    Run tests against the staging server.

    Executes Playwright tests against the staging port and
    returns results including pass/fail status.
    """
    _init_self_improvement()

    if not _staging_manager:
        raise HTTPException(503, "Staging manager not available")
    if not _test_runner:
        raise HTTPException(503, "Test runner not available")

    try:
        # Get staging session
        session = _staging_manager.get_session(session_id)
        if not session:
            raise HTTPException(404, f"Staging session {session_id} not found")

        # Update session status
        from staging_manager import StagingStatus
        session.status = StagingStatus.TESTING

        # Run tests against staging port
        test_results = await _test_runner.run_all_tests(
            port=session.staging_port,
            timeout_seconds=300
        )

        # Store results in session
        session.test_results = test_results.to_dict()

        # Check if critical tests passed
        critical_passed = _test_runner.critical_tests_passed(test_results)

        # Run health check
        health_results = None
        if _health_monitor:
            health_report = await _health_monitor.check_health(port=session.staging_port)
            session.health_check_results = health_report.to_dict()
            health_results = health_report.to_dict()

        # Update session status based on results
        if critical_passed and (not health_results or health_results.get("healthy", True)):
            session.status = StagingStatus.PASSED
        else:
            session.status = StagingStatus.FAILED

        # Log to audit trail
        if _audit_logger:
            await _audit_logger.log_action(
                action="test",
                proposal_id=session.proposal_id,
                session_id=session_id,
                details={
                    "passed": session.status == StagingStatus.PASSED,
                    "test_results": {
                        "total": test_results.total,
                        "passed": test_results.passed,
                        "failed": test_results.failed
                    }
                }
            )

        return {
            "session_id": session_id,
            "status": session.status.value,
            "critical_tests_passed": critical_passed,
            "test_results": test_results.to_dict(),
            "health_results": health_results,
            "summary": _test_runner.get_test_summary(test_results)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/self-improvement/deploy/{proposal_id}")
async def deploy_proposal(
    proposal_id: str,
    session_id: str = None,
    force: bool = False,
    user: str = Depends(verify_token)
):
    """
    Deploy a tested proposal to production.

    If a staging session exists and passed tests, promotes the changes
    to production, restarts the server, and verifies health.

    Args:
        proposal_id: ID of the proposal to deploy
        session_id: Optional staging session ID (if already staged/tested)
        force: Skip test verification (use with caution)
    """
    _init_self_improvement()

    if not _git_manager:
        raise HTTPException(503, "Git manager not available")

    try:
        from self_modification import get_self_mod_manager
        from staging_manager import StagingStatus

        manager = get_self_mod_manager(Path(config.AI_MEMORY_PATH))
        if not manager:
            raise HTTPException(503, "Self-modification manager not available")

        proposal = await manager.get_proposal(proposal_id)
        if not proposal:
            raise HTTPException(404, f"Proposal {proposal_id} not found")

        # If staging session provided, verify it passed
        if session_id and _staging_manager:
            session = _staging_manager.get_session(session_id)
            if session:
                if session.status != StagingStatus.PASSED and not force:
                    raise HTTPException(
                        400,
                        f"Staging session did not pass tests (status: {session.status.value}). Use force=true to override."
                    )

                # Promote staging to production
                if not await _staging_manager.promote_to_production(session):
                    raise HTTPException(500, f"Promotion failed: {session.error}")
        else:
            # No staging - apply directly with git
            result = await manager.apply_with_git(
                proposal_id=proposal_id,
                git_manager=_git_manager,
                user=user
            )

            if not result.get("success"):
                raise HTTPException(500, result.get("error", "Failed to apply proposal"))

        # Mark current commit as safe
        current_commit = _git_manager.get_current_commit()
        if _rollback_engine and current_commit:
            _rollback_engine.mark_commit_as_safe(current_commit)

        # Restart production server and verify health
        restart_result = None
        if _safe_restart:
            restart_result = await _safe_restart.restart_with_verification(
                max_wait_seconds=60,
                rollback_on_failure=True
            )

            if not restart_result.get("success"):
                raise HTTPException(
                    500,
                    f"Deployment succeeded but restart failed: {restart_result.get('error')}"
                )

        # Log to audit trail
        if _audit_logger:
            await _audit_logger.log_action(
                action="deploy",
                proposal_id=proposal_id,
                session_id=session_id,
                details={
                    "commit": current_commit,
                    "restart_result": restart_result,
                    "forced": force
                }
            )

        return {
            "success": True,
            "proposal_id": proposal_id,
            "commit": current_commit,
            "restart_result": restart_result,
            "message": "Deployment successful"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/self-improvement/rollback")
async def manual_rollback(
    commit_hash: str = None,
    reason: str = "Manual rollback",
    user: str = Depends(verify_token)
):
    """
    Manually rollback to a previous state.

    If commit_hash is provided, rolls back to that specific commit.
    Otherwise, rolls back to the last known safe commit.
    """
    _init_self_improvement()

    if not _rollback_engine:
        raise HTTPException(503, "Rollback engine not available")
    if not _git_manager:
        raise HTTPException(503, "Git manager not available")

    try:
        # Determine target commit
        if commit_hash:
            result = await _rollback_engine.rollback_to_commit(
                commit_hash=commit_hash,
                reason=reason
            )
        else:
            # Rollback to last safe commit or previous commit
            result = await _rollback_engine.rollback_last_change(reason=reason)

        if not result.success:
            raise HTTPException(500, f"Rollback failed: {result.message}")

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/self-improvement/status")
async def get_self_improvement_status(user: str = Depends(verify_token)):
    """Get the current status of the self-improvement system."""
    _init_self_improvement()

    status = {
        "initialized": _self_improvement_initialized,
        "components": {
            "git_manager": _git_manager is not None,
            "staging_manager": _staging_manager is not None,
            "test_runner": _test_runner is not None,
            "health_monitor": _health_monitor is not None,
            "rollback_engine": _rollback_engine is not None,
            "improvement_engine": _improvement_engine is not None,
            "audit_logger": _audit_logger is not None,
            "safe_restart": _safe_restart is not None
        },
        "git": None,
        "staging": None,
        "rollback": None
    }

    # Get git status
    if _git_manager:
        status["git"] = {
            "initialized": _git_manager.is_initialized(),
            "current_branch": _git_manager.get_current_branch(),
            "current_commit": _git_manager.get_current_commit(),
            "uncommitted_changes": _git_manager.has_uncommitted_changes()
        }

    # Get staging status
    if _staging_manager:
        current_session = _staging_manager.get_current_session()
        if current_session:
            status["staging"] = current_session.to_dict()

    # Get rollback capability
    if _rollback_engine:
        status["rollback"] = await _rollback_engine.verify_rollback_capability()

    return status


@app.get("/self-improvement/audit")
async def get_audit_log(
    limit: int = 50,
    action_filter: str = None,
    user: str = Depends(verify_token)
):
    """Get recent audit log entries."""
    _init_self_improvement()

    if not _audit_logger:
        return {"entries": [], "error": "Audit logger not available"}

    try:
        from audit_logger import AuditAction

        if action_filter:
            try:
                AuditAction(action_filter)
            except ValueError:
                pass

        entries = await _audit_logger.get_recent_actions(limit=limit)

        return {
            "entries": [e.to_dict() for e in entries],
            "count": len(entries)
        }

    except Exception as e:
        return {"entries": [], "error": str(e)}


# ============================================================================
# Learning Injection Stats Endpoint
# ============================================================================

@app.get("/api/learning/stats")
async def get_learning_stats(user: str = Depends(verify_token)):
    """Get statistics about learning injection."""
    if not learning_injector:
        return {"total_injections": 0, "error": "Learning injector not initialized"}
    return learning_injector.get_stats()


# ============================================================================
# Homepage Dashboard Endpoints
# ============================================================================

@app.get("/api/memory-intelligence")
async def get_memory_intelligence(user: str = Depends(verify_token)):
    """Aggregated memory stats for homepage dashboard."""
    try:
        quick_facts_path = Path(config.AI_MEMORY_PATH) / "quick_facts.json"
        facts = {}
        if quick_facts_path.exists():
            with open(quick_facts_path) as f:
                facts = json.load(f)

        health = facts.get("system_health", {})
        mem_stats = health.get("memory_stats", {})
        faiss = health.get("faiss", {})
        learnings = facts.get("recent_learnings_summary", {})

        return {
            "total_memories": mem_stats.get("conversations", 0),
            "total_learnings": learnings.get("total_count", 0),
            "recent_learnings": learnings.get("last_7_days", 0),
            "promoted_patterns": mem_stats.get("promoted_patterns", 0),
            "contradictions_pending": health.get("contradictions_pending", 0),
            "knowledge_health": health.get("overall_score", 0),
            "faiss_size_mb": faiss.get("size_mb", 0),
            "faiss_status": faiss.get("status", "unknown"),
            "nas_status": health.get("nas_status", "unknown"),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/smart-actions")
async def get_smart_actions(user: str = Depends(verify_token)):
    """Generate smart next-action cards from real memory data."""
    try:
        quick_facts_path = Path(config.AI_MEMORY_PATH) / "quick_facts.json"
        facts = {}
        if quick_facts_path.exists():
            with open(quick_facts_path) as f:
                facts = json.load(f)

        actions = []
        active_work = facts.get("active_work", {})

        # 1. Active workstreams
        workstreams = active_work.get("workstreams", [])
        for ws in workstreams:
            if ws.get("status") == "in_progress":
                actions.append({
                    "type": "continue_work",
                    "icon": "rocket",
                    "title": f"Continue {ws['project']}",
                    "description": ws.get("next_action", ws.get("summary", "")),
                    "context": f"Continue working on {ws['project']}: {ws.get('summary', '')}",
                    "priority": ws.get("priority", 5),
                })

        # 2. Top-level active_work project
        if active_work.get("project") and active_work.get("next_action"):
            actions.append({
                "type": "continue_work",
                "icon": "rocket",
                "title": f"Continue {active_work['project']}",
                "description": active_work["next_action"],
                "context": f"Continue working on {active_work['project']}: {active_work.get('next_action', '')}",
                "priority": 0,
            })

        # 3. Paused workstreams
        for ws in workstreams:
            if ws.get("status") == "paused":
                actions.append({
                    "type": "resume_project",
                    "icon": "play",
                    "title": f"Resume {ws['project']}",
                    "description": ws.get("next_action", ws.get("summary", "")),
                    "context": f"Resume {ws['project']}: {ws.get('next_action', '')}",
                    "priority": ws.get("priority", 5),
                })

        # 4. Goals (try MCP bridge)
        try:
            goals_result = await mcp_bridge.goals("list_active")
            if goals_result.get("success"):
                for goal in goals_result.get("goals", [])[:2]:
                    actions.append({
                        "type": "pursue_goal",
                        "icon": "target",
                        "title": goal.get("description", "Pursue Goal")[:50],
                        "description": f"Priority: {goal.get('priority', 'medium')}",
                        "context": f"Work on goal: {goal.get('description', '')}",
                        "priority": 1 if goal.get("priority") == "high" else 3,
                    })
        except Exception:
            pass

        # 5. Contradictions
        health = facts.get("system_health", {})
        contradictions = health.get("contradictions_pending", 0)
        if contradictions > 0:
            actions.append({
                "type": "resolve_contradictions",
                "icon": "alert",
                "title": f"Resolve {contradictions} Contradiction{'s' if contradictions > 1 else ''}",
                "description": "Conflicting facts detected in memory",
                "context": "Review and resolve memory contradictions",
                "priority": 2,
            })

        # 6. Recent learnings review
        learnings = facts.get("recent_learnings_summary", {})
        if learnings.get("last_7_days", 0) >= 5:
            actions.append({
                "type": "review_learnings",
                "icon": "book",
                "title": f"Review {learnings['last_7_days']} Recent Learnings",
                "description": f"Top topics: {', '.join(learnings.get('top_keywords', [])[:3])}",
                "context": "Review recent learnings and patterns",
                "priority": 4,
            })

        # Sort by priority, limit to 4
        actions.sort(key=lambda x: x.get("priority", 99))
        return {"actions": actions[:4]}

    except Exception as e:
        return {"actions": [], "error": str(e)}


@app.get("/api/workstreams")
async def get_workstreams(user: str = Depends(verify_token)):
    """Get active workstreams from quick_facts.json for project tracker."""
    try:
        quick_facts_path = Path(config.AI_MEMORY_PATH) / "quick_facts.json"
        facts = {}
        if quick_facts_path.exists():
            with open(quick_facts_path) as f:
                facts = json.load(f)

        active_work = facts.get("active_work", {})
        workstreams = active_work.get("workstreams", [])

        # Include top-level project if present
        result = []
        if active_work.get("project"):
            result.append({
                "id": "ws_toplevel",
                "project": active_work["project"],
                "summary": active_work.get("last_completed", ""),
                "status": "in_progress",
                "next_action": active_work.get("next_action", ""),
                "last_completed": active_work.get("last_completed", ""),
                "key_files": active_work.get("key_files", []),
            })

        for ws in workstreams:
            result.append({
                "id": ws.get("id", ""),
                "project": ws.get("project", "Unknown"),
                "summary": ws.get("summary", ""),
                "status": ws.get("status", "unknown"),
                "next_action": ws.get("next_action", ""),
                "last_completed": ws.get("last_completed", ""),
                "key_files": ws.get("key_files", []),
                "priority": ws.get("priority", 99),
            })

        return {"workstreams": result}

    except Exception as e:
        return {"workstreams": [], "error": str(e)}


# ============================================================================
# Cognitive Loop / Autonomy Endpoints
# ============================================================================

@app.get("/api/autonomy/status")
async def get_autonomy_status(user: str = Depends(verify_token)):
    """Get current autonomy status with stats."""
    if not cognitive_loop_manager:
        return {
            "available": False,
            "error": "Cognitive loop not available",
            "status": "unavailable"
        }
    state = cognitive_loop_manager.get_state()

    # Get additional stats for UI
    try:
        stats = await cognitive_loop_manager.get_stats()
        journal_stats = stats.get("journal", {})
        stats.get("safety", {})
        pending = await cognitive_loop_manager.get_pending_approvals()
    except:
        journal_stats = {}
        pending = []

    return {
        "available": True,
        **state.to_dict(),
        # Include stats for UI display
        "actions_completed": journal_stats.get("total_actions", 0),
        "pending_count": len(pending),
        "total_thoughts": journal_stats.get("total_thoughts", 0),
        "session_thoughts": journal_stats.get("session_thoughts", 0),
        "waiting_for_human": state.waiting_for_human
    }


@app.post("/api/autonomy/start")
async def start_autonomy(request: Request, user: str = Depends(verify_token)):
    """Start the cognitive loop at specified autonomy level."""
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")
    # Accept level from JSON body or query param
    level = 2
    try:
        body = await request.json()
        level = int(body.get("level", 2))
    except Exception:
        pass
    result = await cognitive_loop_manager.start_loop(level)
    return result


@app.post("/api/autonomy/stop")
async def stop_autonomy(reason: str = "User requested", user: str = Depends(verify_token)):
    """Stop the cognitive loop gracefully."""
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")
    result = await cognitive_loop_manager.stop_loop(reason)
    return result


@app.post("/api/autonomy/emergency-stop")
async def emergency_stop_autonomy(user: str = Depends(verify_token)):
    """Emergency stop - immediately halt all autonomous activity AND kill running agents."""
    agents_killed = 0

    # Kill all running agent processes
    for agent_id, agent in active_agents.items():
        if agent.get("status") in (AgentStatus.RUNNING, AgentStatus.QUEUED):
            # Try to terminate the process
            process = agent.get("_process")
            if process:
                try:
                    process.terminate()
                    print(f"[Emergency Stop] Terminated agent {agent_id} (PID: {agent.get('pid')})")
                    agents_killed += 1
                except Exception as e:
                    print(f"[Emergency Stop] Failed to terminate agent {agent_id}: {e}")

            # Also try via PID if process object isn't available
            pid = agent.get("pid")
            if pid and not process:
                try:
                    import signal
                    import os
                    os.kill(pid, signal.SIGTERM)
                    print(f"[Emergency Stop] Killed agent {agent_id} via PID {pid}")
                    agents_killed += 1
                except Exception as e:
                    print(f"[Emergency Stop] Failed to kill PID {pid}: {e}")

            # Mark agent as stopped
            agent["status"] = AgentStatus.FAILED
            agent["error"] = "Emergency stop activated"
            await sio.emit("agent_update", sanitize_agent_for_emit(agent), room=os.environ.get("CEREBRO_ROOM", "default"))

    # Stop cognitive loop
    if not cognitive_loop_manager:
        return {"success": True, "agents_killed": agents_killed, "message": "Loop not running, but killed agents"}

    result = await cognitive_loop_manager.emergency_stop()
    result["agents_killed"] = agents_killed
    return result


@app.post("/api/autonomy/pause")
async def pause_autonomy(user: str = Depends(verify_token)):
    """Pause the cognitive loop."""
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")
    result = await cognitive_loop_manager.pause_loop()
    return result


@app.post("/api/autonomy/resume")
async def resume_autonomy(user: str = Depends(verify_token)):
    """Resume a paused cognitive loop."""
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")
    result = await cognitive_loop_manager.resume_loop()
    return result


@app.post("/api/autonomy/level")
async def set_autonomy_level(level: int, user: str = Depends(verify_token)):
    """Change autonomy level (1-5)."""
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")
    if level < 1 or level > 5:
        raise HTTPException(status_code=400, detail="Level must be 1-5")
    result = await cognitive_loop_manager.set_level(level)
    return result


@app.post("/api/autonomy/reset-kill-switch")
async def reset_kill_switch(user: str = Depends(verify_token)):
    """Reset the kill switch to allow autonomous operations."""
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")
    result = await cognitive_loop_manager.reset_kill_switch()
    return result


@app.post("/api/autonomy/full-autonomy")
async def set_full_autonomy(enabled: bool, user: str = Depends(verify_token)):
    """
    Enable/disable full autonomy mode.

    When enabled: Can spawn Claude Code agents (uses your Anthropic subscription)
    When disabled: Only thinks and does memory operations (free)
    """
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")
    result = await cognitive_loop_manager.set_full_autonomy(enabled)
    return result


# ============================================================================
# Safety Layer & Spawn Control Endpoints
# ============================================================================

@app.get("/api/safety/status")
async def get_safety_status(user: str = Depends(verify_token)):
    """Get current safety layer status including spawn usage."""
    if not cognitive_loop_manager:
        return {
            "error": "Cognitive loop not available",
            "killed": False,
            "full_autonomy_enabled": False
        }
    return cognitive_loop_manager.safety_layer.get_status()


@app.get("/api/safety/spawn-usage")
async def get_spawn_usage(user: str = Depends(verify_token)):
    """Get detailed spawn usage statistics."""
    if not cognitive_loop_manager:
        return {"error": "Cognitive loop not available"}
    return cognitive_loop_manager.safety_layer.get_spawn_usage()


@app.post("/api/safety/preapprove")
async def set_preapproval(
    agent_type: str,
    hours: int = 4,
    user: str = Depends(verify_token)
):
    """
    Pre-approve an agent type for automatic spawning without per-spawn approval.

    Args:
        agent_type: worker, researcher, coder, analyst, orchestrator
        hours: Duration of pre-approval (0 to remove)

    WARNING: Orchestrator pre-approval is discouraged as they can spawn other agents.
    """
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")

    valid_types = ["worker", "researcher", "coder", "analyst", "orchestrator"]
    if agent_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent_type. Must be one of: {valid_types}"
        )

    if agent_type == "orchestrator" and hours > 0:
        # Extra warning for orchestrators
        print(f"[Safety] WARNING: Pre-approving orchestrator agents for {hours} hours")

    result = cognitive_loop_manager.safety_layer.set_preapproval(agent_type, hours)
    return result


@app.get("/api/safety/preapprovals")
async def get_preapprovals(user: str = Depends(verify_token)):
    """Get all active pre-approvals."""
    if not cognitive_loop_manager:
        return {"preapprovals": {}}
    return {"preapprovals": cognitive_loop_manager.safety_layer.get_preapprovals()}


@app.delete("/api/safety/preapprovals")
async def clear_preapprovals(user: str = Depends(verify_token)):
    """Clear all pre-approvals."""
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")
    return cognitive_loop_manager.safety_layer.clear_preapprovals()


@app.post("/api/safety/can-spawn")
async def check_can_spawn(
    agent_type: str = "worker",
    user: str = Depends(verify_token)
):
    """
    Check if a spawn is currently allowed.

    Returns whether spawning is permitted and the reason if not.
    """
    if not cognitive_loop_manager:
        return {
            "can_spawn": False,
            "reason": "Cognitive loop not available"
        }

    can_spawn, reason = cognitive_loop_manager.safety_layer.can_spawn(agent_type)
    return {
        "can_spawn": can_spawn,
        "reason": reason,
        "agent_type": agent_type,
        "is_preapproved": cognitive_loop_manager.safety_layer.is_preapproved(agent_type)
    }


@app.post("/api/safety/record-spawn")
async def record_spawn(
    agent_id: str,
    agent_type: str,
    task: str,
    user: str = Depends(verify_token)
):
    """
    Record that a spawn occurred (called by cognitive tools after successful spawn).

    This updates the safety layer's persistent spawn tracking.
    """
    if not cognitive_loop_manager:
        return {"recorded": False, "error": "Cognitive loop not available"}

    cognitive_loop_manager.safety_layer.record_spawn(agent_type, agent_id, task)
    return {
        "recorded": True,
        "agent_id": agent_id,
        "usage": cognitive_loop_manager.safety_layer.get_spawn_usage()
    }


@app.get("/api/autonomy/thoughts")
async def get_autonomy_thoughts(limit: int = 50, user: str = Depends(verify_token)):
    """Get recent thoughts from the cognitive loop."""
    if not cognitive_loop_manager:
        return {"thoughts": []}
    thoughts = await cognitive_loop_manager.get_recent_thoughts(limit)
    return {"thoughts": thoughts}


@app.get("/api/autonomy/activity")
async def get_autonomy_activity(limit: int = 300, user: str = Depends(verify_token)):
    """Get recent activity log entries (all phases for cycle view) for backfill on page load."""
    if not cognitive_loop_manager:
        return {"activity": []}
    thoughts = await cognitive_loop_manager.get_recent_thoughts(limit * 2)
    activity = []
    for t in thoughts:
        phase = (t.get("phase") or "").lower()
        if not phase:
            continue
        content = t.get("content", "")
        meta = t.get("metadata", {})
        activity.append({
            "id": t.get("id", ""),
            "phase": phase,
            "content": content[:120] + "..." if len(content) > 120 else content,
            "fullContent": content,
            "timestamp": t.get("timestamp", ""),
            "cycle_number": meta.get("cycle_number", 0),
            "reasoning": t.get("reasoning", ""),
            "confidence": t.get("confidence", 0),
            "metadata": meta,
            "is_browser_step": meta.get("is_browser_step", False),
        })
        if len(activity) >= limit:
            break
    return {"activity": activity}


@app.get("/api/autonomy/pending")
async def get_pending_approvals(user: str = Depends(verify_token)):
    """Get pending action approvals."""
    if not cognitive_loop_manager:
        return {"pending": []}
    pending = await cognitive_loop_manager.get_pending_approvals()
    return {"pending": pending}


@app.post("/api/autonomy/approve/{action_id}")
async def approve_action(action_id: str, user: str = Depends(verify_token)):
    """Approve a pending action."""
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")
    result = await cognitive_loop_manager.approve_action(action_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail="Action not found")
    return result


@app.post("/api/autonomy/reject/{action_id}")
async def reject_action(action_id: str, user: str = Depends(verify_token)):
    """Reject a pending action."""
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")
    result = await cognitive_loop_manager.reject_action(action_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail="Action not found")
    return result


@app.post("/api/autonomy/answer-question")
async def answer_proactive_question(request: Request, user: str = Depends(verify_token)):
    """
    Store the user's answer to a proactive question from Cerebro.

    This helps Cerebro learn about the user's preferences and goals.
    """
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")

    data = await request.json()
    question = data.get("question", "")
    answer = data.get("answer", "")

    if not question or not answer:
        raise HTTPException(status_code=400, detail="Question and answer required")

    result = await cognitive_loop_manager.store_question_answer(question, answer)
    return result


@app.post("/api/autonomy/human-response")
async def submit_human_response(request: Request, user: str = Depends(verify_token)):
    """
    Submit a human response to a human_input_needed event from the OODA loop.

    This unpauses the cognitive loop and feeds the answer into the next OBSERVE phase.
    """
    if not cognitive_loop_manager:
        raise HTTPException(status_code=503, detail="Cognitive loop not available")

    data = await request.json()
    request_id = data.get("request_id", "")
    answer = data.get("answer", "")

    if not answer:
        raise HTTPException(status_code=400, detail="Answer is required")

    result = await cognitive_loop_manager.receive_human_response({
        "request_id": request_id,
        "answer": answer,
        "original_question": data.get("original_question", ""),
        "context": data.get("context", ""),
    })
    return result


# ============================================================================
# Proactive Questions Persistence
# ============================================================================

PROACTIVE_QUESTIONS_FILE = Path(config.AI_MEMORY_PATH) / "cerebro" / "cognitive_loop" / "proactive_questions.json"
CEREBRO_CHAT_FILE = Path(config.AI_MEMORY_PATH) / "cerebro" / "cognitive_loop" / "cerebro_chat.json"

def _load_cerebro_chat() -> list:
    """Load Cerebro chat conversation history."""
    if CEREBRO_CHAT_FILE.exists():
        try:
            data = json.loads(CEREBRO_CHAT_FILE.read_text())
            return data.get("messages", [])
        except:
            pass
    return []

def _save_cerebro_chat(messages: list):
    """Save Cerebro chat conversation history."""
    CEREBRO_CHAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Keep only last 200 messages (increased for narration history)
    if len(messages) > 200:
        messages = messages[-200:]
    CEREBRO_CHAT_FILE.write_text(json.dumps({"messages": messages}, indent=2))

def _load_proactive_questions() -> list:
    """Load proactive questions from file."""
    if PROACTIVE_QUESTIONS_FILE.exists():
        try:
            data = json.loads(PROACTIVE_QUESTIONS_FILE.read_text())
            return data.get("questions", [])
        except:
            pass
    return []

def _save_proactive_questions(questions: list):
    """Save proactive questions to file."""
    PROACTIVE_QUESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROACTIVE_QUESTIONS_FILE.write_text(json.dumps({"questions": questions}, indent=2))

@app.get("/api/autonomy/questions")
async def get_proactive_questions(user: str = Depends(verify_token)):
    """Get all pending proactive questions."""
    questions = _load_proactive_questions()
    return {"questions": questions}

@app.post("/api/autonomy/questions")
async def save_proactive_question(request: Request, user: str = Depends(verify_token)):
    """Save a new proactive question from Cerebro."""
    data = await request.json()
    question_text = data.get("question", "")
    question_type = data.get("type", "general")
    options = data.get("options", [])
    paths = data.get("paths", [])
    directive_id = data.get("directive_id", "")

    if not question_text:
        raise HTTPException(status_code=400, detail="Question text required")

    questions = _load_proactive_questions()

    # Create new question
    new_question = {
        "id": f"q_{int(datetime.now().timestamp())}",
        "question": question_text,
        "type": question_type,
        "options": options,
        "paths": paths,
        "directive_id": directive_id,
        "timestamp": datetime.now().isoformat(),
        "answered": False
    }

    # Add to list (max 20 questions)
    questions.insert(0, new_question)
    if len(questions) > 20:
        questions = questions[:20]

    _save_proactive_questions(questions)

    return {"success": True, "question": new_question}

@app.delete("/api/autonomy/questions/{question_id}")
async def dismiss_proactive_question(question_id: str, user: str = Depends(verify_token)):
    """Dismiss (remove) a proactive question."""
    questions = _load_proactive_questions()
    questions = [q for q in questions if q.get("id") != question_id]
    _save_proactive_questions(questions)
    return {"success": True}


@app.delete("/api/autonomy/questions")
async def clear_all_proactive_questions(user: str = Depends(verify_token)):
    """Clear all proactive questions."""
    _save_proactive_questions([])
    return {"success": True}


# ============================================================================
# Stored Items Persistence (Delayed questions, simulations, agent results)
# ============================================================================

STORED_ITEMS_FILE = Path(config.AI_MEMORY_PATH) / "cerebro" / "stored_items.json"

def _load_stored_items() -> list:
    """Load stored items from file."""
    if STORED_ITEMS_FILE.exists():
        try:
            data = json.loads(STORED_ITEMS_FILE.read_text())
            return data if isinstance(data, list) else []
        except:
            pass
    return []

def _save_stored_items(items: list):
    """Save stored items to file, capped at 100."""
    STORED_ITEMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if len(items) > 100:
        items = items[:100]
    STORED_ITEMS_FILE.write_text(json.dumps(items, indent=2, default=str))

@app.get("/api/stored-items")
async def get_stored_items(user: str = Depends(verify_token)):
    """Get all stored items."""
    items = _load_stored_items()
    return {"items": items}

@app.post("/api/stored-items")
async def add_stored_item(request: Request, user: str = Depends(verify_token)):
    """Add a new stored item."""
    data = await request.json()
    if not data.get("type"):
        raise HTTPException(status_code=400, detail="Item type required")

    item = {
        "id": data.get("id", f"stored_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:6]}"),
        "type": data.get("type", "question"),
        "title": data.get("title", "Stored Item"),
        "content": data.get("content", ""),
        "metadata": data.get("metadata", {}),
        "created_at": data.get("created_at", datetime.now(timezone.utc).isoformat()),
        "status": data.get("status", "pending"),
        "source_id": data.get("source_id", "")
    }

    items = _load_stored_items()
    items.insert(0, item)
    _save_stored_items(items)

    await sio.emit("cerebro_stored_item_added", item, room=os.environ.get("CEREBRO_ROOM", "default"))
    return {"success": True, "item": item}

@app.patch("/api/stored-items/{item_id}")
async def update_stored_item(item_id: str, request: Request, user: str = Depends(verify_token)):
    """Update a stored item's status."""
    data = await request.json()
    items = _load_stored_items()
    for item in items:
        if item.get("id") == item_id:
            for key in ("status", "title", "content"):
                if key in data:
                    item[key] = data[key]
            _save_stored_items(items)
            return {"success": True, "item": item}
    raise HTTPException(status_code=404, detail="Item not found")

@app.delete("/api/stored-items/{item_id}")
async def delete_stored_item(item_id: str, user: str = Depends(verify_token)):
    """Delete a single stored item."""
    items = _load_stored_items()
    items = [i for i in items if i.get("id") != item_id]
    _save_stored_items(items)
    await sio.emit("cerebro_stored_item_removed", {"id": item_id}, room=os.environ.get("CEREBRO_ROOM", "default"))
    return {"success": True}

@app.delete("/api/stored-items")
async def clear_stored_items(user: str = Depends(verify_token)):
    """Clear all stored items."""
    _save_stored_items([])
    return {"success": True}


# ============================================================================
# Heartbeat Config API
# ============================================================================

from cognitive_loop.idle_thinker import (
    load_heartbeat_config as _load_hb_config,
    save_heartbeat_config as _save_hb_config,
    get_heartbeat_engine as _get_hb_engine,
    load_heartbeat_md as _load_hb_md,
    save_heartbeat_md as _save_hb_md,
    parse_heartbeat_md as _parse_hb_md,
)

@app.get("/api/heartbeat/config")
async def get_heartbeat_config(user: str = Depends(verify_token)):
    """Return current heartbeat config (creates default if missing)."""
    return _load_hb_config()

@app.post("/api/heartbeat/config")
async def update_heartbeat_config(request: Request, user: str = Depends(verify_token)):
    """Update heartbeat config (interval + monitor toggles)."""
    data = await request.json()
    current = _load_hb_config()
    if "interval_minutes" in data:
        current["interval_minutes"] = max(5, min(60, int(data["interval_minutes"])))
    if "monitors" in data:
        for name, cfg in data["monitors"].items():
            if name in current["monitors"]:
                current["monitors"][name].update(cfg)
    _save_hb_config(current)
    await sio.emit("cerebro_heartbeat_config_updated", current, room=os.environ.get("CEREBRO_ROOM", "default"))
    return {"success": True, "config": current}

@app.get("/api/heartbeat/status")
async def get_heartbeat_status(user: str = Depends(verify_token)):
    """Return last heartbeat time, findings, monitors ran."""
    engine = _get_hb_engine()
    return engine.get_status()

@app.get("/api/heartbeat/md")
async def get_heartbeat_md(user: str = Depends(verify_token)):
    """Return heartbeat.md content (creates default if missing)."""
    content = _load_hb_md()
    return {"content": content}

@app.post("/api/heartbeat/md")
async def update_heartbeat_md(request: Request, user: str = Depends(verify_token)):
    """Save heartbeat.md, parse it, and sync settings to heartbeat_config.json."""
    data = await request.json()
    content = data.get("content", "")
    _save_hb_md(content)

    # Parse markdown and sync interval + monitors to JSON config
    parsed = _parse_hb_md(content)
    config = _load_hb_config()
    config["interval_minutes"] = parsed["interval_minutes"]
    # Sync monitor toggles: enabled if listed in markdown, disabled otherwise
    known_monitors = list(config.get("monitors", {}).keys())
    for m_name in known_monitors:
        config["monitors"][m_name]["enabled"] = m_name in parsed["monitors_enabled"]
    _save_hb_config(config)
    await sio.emit("cerebro_heartbeat_config_updated", config, room=os.environ.get("CEREBRO_ROOM", "default"))
    return {"success": True, "parsed": parsed}


@app.get("/api/autonomy/chat")
async def get_cerebro_chat(user: str = Depends(verify_token)):
    """Get Cerebro chat conversation history."""
    messages = _load_cerebro_chat()
    return {"messages": messages}

@app.delete("/api/autonomy/chat")
async def clear_cerebro_chat(user: str = Depends(verify_token)):
    """Clear the Cerebro chat conversation history."""
    _save_cerebro_chat([])
    return {"success": True}

@app.post("/api/autonomy/ask")
async def ask_cerebro(request: Request, user: str = Depends(verify_token)):
    """
    User asks Cerebro a question, and Cerebro responds using the local LLM.
    This enables bidirectional conversation in the Questions panel.
    Maintains conversation context for natural back-and-forth dialogue.
    Injects real system state when user asks about goals/directives/status.
    """
    data = await request.json()
    question = data.get("question", "").strip()
    reply_to = data.get("reply_to")  # {id, content} if replying to a previous message

    if not question:
        raise HTTPException(status_code=400, detail="Question required")

    # Load existing conversation history
    chat_history = _load_cerebro_chat()

    # If replying to a previous message, inject that context
    if reply_to and reply_to.get("content"):
        reply_content = reply_to["content"][:500]
        question = f'[User is replying to this previous message: "{reply_content}"]\n\n{question}'

    # Check if user is asking about system state - inject real context
    question_lower = question.lower()
    system_context = ""

    # Detect questions about goals/directives/focus
    goal_keywords = ["goal", "directive", "focus", "working on", "task", "mission", "assigned", "gave you", "current"]
    if any(kw in question_lower for kw in goal_keywords):
        directives = load_directives_from_file()
        active_directives = [d for d in directives if d.get("status") != "completed"]
        if active_directives:
            directive_list = "\n".join([f"- {d['text']} (status: {d.get('status', 'pending')})" for d in active_directives[:10]])
            system_context += f"\n\n[CURRENT DIRECTIVES FROM PROFESSOR]:\n{directive_list}\n"

    # Detect questions about autonomy status
    status_keywords = ["status", "running", "active", "autonomy", "thinking", "state"]
    if any(kw in question_lower for kw in status_keywords):
        if cognitive_loop_manager:
            state = cognitive_loop_manager.get_state()
            system_context += f"\n\n[CURRENT AUTONOMY STATUS]: {state.status}, level {state.autonomy_level}, cycles: {state.cycles_completed}\n"

    # Detect browser/website tasks — auto-create directive so OODA engine handles it
    browser_keywords = ["go to", "browse to", "navigate to", "open up", "pull up",
                        "take me to", "search on", "log in to", "login to", "sign in",
                        "use your browser", "use the browser", "using your browser",
                        "look up", "look for", "search for",
                        "open the browser", "open a browser", "open chrome"]
    is_browser_task = any(kw in question_lower for kw in browser_keywords)

    # Also detect "open <brand>" patterns (e.g. "open youtube", "hop on amazon")
    if not is_browser_task:
        _brand_names = '|'.join(re.escape(b) for b in _BRAND_MAP.keys())
        _open_brand_re = re.compile(r'\b(?:open|launch|fire up|go on|hop on)\s+(?:' + _brand_names + r')\b', re.IGNORECASE)
        is_browser_task = bool(_open_brand_re.search(question_lower))

    if is_browser_task and cognitive_loop_manager:
        # Auto-create a directive for the OODA engine to handle with browser
        directives = load_directives_from_file()
        directive = {
            "id": str(uuid.uuid4())[:8],
            "text": question,
            "type": "task",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "created_by": "cerebro_chat"
        }
        directives.insert(0, directive)
        save_directives_to_file(directives)

        # Notify frontend
        if sio:
            await sio.emit("directive_added", directive, room=os.environ.get("CEREBRO_ROOM", "default"))

        # Auto-start the cognitive loop if it's dormant/stopped
        loop_state = cognitive_loop_manager.get_state()
        if loop_state.status in ("stopped", "error"):
            level = cognitive_loop_manager.autonomy_level or 2
            await cognitive_loop_manager.start_loop(level)
            await sio.emit("autonomy_status", cognitive_loop_manager.get_state().to_dict(), room=os.environ.get("CEREBRO_ROOM", "default"))
        else:
            # Already running, just wake it
            cognitive_loop_manager.wake()

        # Save chat exchange
        timestamp = datetime.now().isoformat()
        answer = "On it — I've queued that as a task and I'm firing up my brain to handle it with my browser. Watch the activity log."
        chat_history.append({"id": f"msg_{int(datetime.now().timestamp() * 1000)}", "role": "user", "content": question, "timestamp": timestamp})
        chat_history.append({"id": f"msg_{int(datetime.now().timestamp() * 1000) + 1}", "role": "assistant", "content": answer, "timestamp": timestamp})
        _save_cerebro_chat(chat_history)

        return {"success": True, "answer": answer, "question": question, "directive_created": directive["id"]}

    # v2.0: Route ALL questions through Claude Code agents (no more Qwen fallback)
    # Create a directive so the cognitive loop dispatches a Claude agent
    directives = load_directives_from_file()
    directive = {
        "id": str(uuid.uuid4())[:8],
        "text": question,
        "type": "task",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "created_by": "cerebro_chat"
    }
    directives.insert(0, directive)
    save_directives_to_file(directives)

    # Notify frontend
    if sio:
        await sio.emit("directive_added", directive, room=os.environ.get("CEREBRO_ROOM", "default"))

    # Auto-start the cognitive loop if it's dormant/stopped
    if cognitive_loop_manager:
        loop_state = cognitive_loop_manager.get_state()
        if loop_state.status in ("stopped", "error"):
            level = cognitive_loop_manager.autonomy_level or 2
            await cognitive_loop_manager.start_loop(level)
            await sio.emit("autonomy_status", cognitive_loop_manager.get_state().to_dict(), room=os.environ.get("CEREBRO_ROOM", "default"))
        else:
            cognitive_loop_manager.wake()

    # Save chat exchange
    timestamp = datetime.now().isoformat()
    answer = "On it — I've spawned an agent to handle this."
    chat_history.append({"id": f"msg_{int(datetime.now().timestamp() * 1000)}", "role": "user", "content": question, "timestamp": timestamp})
    chat_history.append({"id": f"msg_{int(datetime.now().timestamp() * 1000) + 1}", "role": "assistant", "content": answer, "timestamp": timestamp})
    _save_cerebro_chat(chat_history)

    return {"success": True, "answer": answer, "question": question, "directive_created": directive["id"]}


@app.get("/api/autonomy/stats")
async def get_autonomy_stats(user: str = Depends(verify_token)):
    """Get comprehensive autonomy statistics."""
    if not cognitive_loop_manager:
        return {"available": False}
    stats = await cognitive_loop_manager.get_stats()
    return {"available": True, **stats}


# ============================================================================
# Personality Preferences
# ============================================================================

CEREBRO_PREFS_FILE = Path(config.AI_MEMORY_PATH) / "cerebro" / "preferences.json"

def _load_cerebro_prefs() -> dict:
    """Load Cerebro preferences from disk."""
    if CEREBRO_PREFS_FILE.exists():
        try:
            return json.loads(CEREBRO_PREFS_FILE.read_text())
        except Exception:
            pass
    return {"personality_mode": "chill"}

def _save_cerebro_prefs(prefs: dict):
    """Save Cerebro preferences to disk."""
    CEREBRO_PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CEREBRO_PREFS_FILE.write_text(json.dumps(prefs, indent=2))


@app.get("/api/cerebro/personality")
async def get_personality(user: str = Depends(verify_token)):
    """Get current personality mode."""
    prefs = _load_cerebro_prefs()
    return {"personality_mode": prefs.get("personality_mode", "chill")}


@app.post("/api/cerebro/personality")
async def set_personality(request: Request, user: str = Depends(verify_token)):
    """Set personality mode (chill or analyst)."""
    data = await request.json()
    mode = data.get("personality_mode", "chill")
    if mode not in ("chill", "analyst"):
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'chill' or 'analyst'.")
    prefs = _load_cerebro_prefs()
    prefs["personality_mode"] = mode
    _save_cerebro_prefs(prefs)
    return {"success": True, "personality_mode": mode}


# ============================================================================
# Look Command — Voice Screenshot Daemon Management
# ============================================================================

_look_daemon_process: subprocess.Popen | None = None

LOOK_SCRIPT = Path(__file__).parent.parent / "scripts" / "voice-screenshot.py"


def _is_look_running() -> bool:
    """Check if the look daemon subprocess is alive."""
    global _look_daemon_process
    if _look_daemon_process is not None:
        if _look_daemon_process.poll() is None:
            return True
        # Process exited — clean up
        _look_daemon_process = None
    return False


@app.get("/api/look/status")
async def look_status(user: str = Depends(verify_token)):
    """Check if the look daemon is running."""
    running = _is_look_running()
    return {"running": running, "pid": _look_daemon_process.pid if running else None}


@app.post("/api/look/toggle")
async def look_toggle(request: Request, user: str = Depends(verify_token)):
    """Start or stop the voice-screenshot daemon."""
    global _look_daemon_process

    data = await request.json()
    enable = data.get("enable", True)

    if enable:
        if _is_look_running():
            return {"success": True, "running": True, "pid": _look_daemon_process.pid, "message": "Already running"}

        if not LOOK_SCRIPT.exists():
            raise HTTPException(status_code=404, detail=f"Look script not found: {LOOK_SCRIPT}")

        _look_daemon_process = subprocess.Popen(
            [sys.executable, str(LOOK_SCRIPT), "--daemon"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"[Look] Daemon started (PID {_look_daemon_process.pid})")
        return {"success": True, "running": True, "pid": _look_daemon_process.pid}
    else:
        if not _is_look_running():
            return {"success": True, "running": False, "message": "Already stopped"}

        pid = _look_daemon_process.pid
        _look_daemon_process.terminate()
        try:
            _look_daemon_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _look_daemon_process.kill()
        _look_daemon_process = None
        print(f"[Look] Daemon stopped (was PID {pid})")
        return {"success": True, "running": False}


# ============================================================================
# Directives - User-defined missions for the cognitive loop
# ============================================================================


# Directives storage file
# ============================================================================
# Skills API - Browser Automation Skills
# ============================================================================

class SkillGenerateRequest(BaseModel):
    name: str
    description: str
    url: Optional[str] = None

class SkillExecuteRequest(BaseModel):
    parameters: Optional[Dict[str, str]] = None

@app.get("/api/skills")
async def list_skills(status: Optional[str] = None, user: str = Depends(verify_token)):
    """List all automation skills."""
    try:
        from cognitive_loop.skill_generator import get_skill_generator, SkillStatus
        gen = get_skill_generator()
        skill_status = SkillStatus(status) if status else None
        skills = gen.list_skills(status=skill_status)
        return {
            "count": len(skills),
            "skills": [s.to_dict() for s in skills]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "skills": []}

@app.get("/api/skills/{skill_id}")
async def get_skill(skill_id: str, user: str = Depends(verify_token)):
    """Get a specific skill by ID."""
    try:
        from cognitive_loop.skill_generator import get_skill_generator
        gen = get_skill_generator()
        skill = gen.get_skill(skill_id)
        if not skill:
            raise HTTPException(status_code=404, detail="Skill not found")
        return skill.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/skills/generate")
async def generate_skill(request: SkillGenerateRequest, user: str = Depends(verify_token)):
    """Generate a new skill from a description."""
    try:
        from cognitive_loop.skill_generator import get_skill_generator
        from cognitive_loop.ollama_client import get_ollama_client

        gen = get_skill_generator()
        gen.ollama_client = get_ollama_client()

        skill = await gen.generate_skill_from_description(
            name=request.name,
            description=request.description,
            example_url=request.url
        )
        return {
            "success": True,
            "skill": skill.to_dict()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/skills/{skill_id}/execute")
async def execute_skill(skill_id: str, request: SkillExecuteRequest, user: str = Depends(verify_token)):
    """Execute a skill."""
    try:
        from cognitive_loop.skill_generator import get_skill_generator
        from datetime import datetime, timezone

        gen = get_skill_generator(headless=False)  # Show browser
        gen._load_skills()  # Ensure latest skills loaded

        skill = gen.get_skill(skill_id)
        skill_name = skill.name if skill else skill_id

        # Emit start event to UI
        await sio.emit("thought_stream", {
            "id": f"skill_{skill_id}_{datetime.now().timestamp()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "act",
            "type": "action",
            "content": f"Executing browser skill: {skill_name}...",
            "reasoning": f"Parameters: {request.parameters}" if request.parameters else None,
            "confidence": 0.8,
            "metadata": {"skill_id": skill_id, "skill_name": skill_name, "status": "running"}
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

        result = await gen.execute_skill(skill_id, request.parameters)

        # Emit result event to UI
        status_msg = f"Skill '{skill_name}' completed successfully! ({result.steps_completed}/{result.total_steps} steps, {result.duration_ms:.0f}ms)" if result.success else f"Skill '{skill_name}' failed: {result.error}"
        await sio.emit("thought_stream", {
            "id": f"skill_result_{skill_id}_{datetime.now().timestamp()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "reflect" if result.success else "act",
            "type": "reflection" if result.success else "action",
            "content": status_msg,
            "reasoning": str(result.output)[:500] if result.output else None,
            "confidence": 0.9 if result.success else 0.3,
            "metadata": {
                "skill_id": skill_id,
                "success": result.success,
                "steps_completed": result.steps_completed,
                "total_steps": result.total_steps,
                "duration_ms": result.duration_ms
            }
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "steps_completed": result.steps_completed,
            "total_steps": result.total_steps,
            "duration_ms": result.duration_ms
        }
    except Exception as e:
        import traceback
        print(f"[SKILL EXECUTE ERROR] {skill_id}: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.delete("/api/skills/{skill_id}")
async def delete_skill(skill_id: str, user: str = Depends(verify_token)):
    """Delete a skill."""
    try:
        from cognitive_loop.skill_generator import get_skill_generator
        gen = get_skill_generator()
        if gen.delete_skill(skill_id):
            return {"success": True}
        raise HTTPException(status_code=404, detail="Skill not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/skills/search/{query}")
async def search_skills(query: str, user: str = Depends(verify_token)):
    """Search skills by name or description."""
    try:
        from cognitive_loop.skill_generator import get_skill_generator
        gen = get_skill_generator()
        skills = gen.search_skills(query)
        return {
            "query": query,
            "count": len(skills),
            "skills": [s.to_dict() for s in skills]
        }
    except Exception as e:
        return {"error": str(e), "skills": []}

# ============================================================================
# Directives
# ============================================================================

DIRECTIVES_FILE = Path(config.AI_MEMORY_PATH) / "cerebro" / "directives.json"

def load_directives_from_file() -> list:
    """Load directives from file."""
    try:
        if DIRECTIVES_FILE.exists():
            return json.loads(DIRECTIVES_FILE.read_text())
        return []
    except Exception:
        return []

def save_directives_to_file(directives: list):
    """Save directives to file."""
    try:
        DIRECTIVES_FILE.parent.mkdir(parents=True, exist_ok=True)
        DIRECTIVES_FILE.write_text(json.dumps(directives, indent=2, default=str))
    except Exception as e:
        print(f"Failed to save directives: {e}")

class DirectiveRequest(BaseModel):
    text: str
    directive_type: str = None  # "task" or "goal" - auto-detected if not provided
    auto_awake: bool = True  # Whether to auto-start cognitive loop on send
    image_path: Optional[str] = None  # Attached image path for vision tasks


def classify_directive_type(text: str) -> str:
    """
    Classify a directive as either a 'task' (one-time, completable) or 'goal' (ongoing).

    Tasks: Have a clear deliverable, can be done once and completed
    Goals: Ongoing objectives, continuous improvement, no clear end state
    """
    text_lower = text.lower()

    # Task indicators - action verbs with clear deliverables
    task_patterns = [
        "extract", "get", "fetch", "find", "search for", "look up",
        "create a", "write a", "make a", "build a", "generate a",
        "analyze", "review", "check", "verify", "test",
        "send", "post", "submit", "upload", "download",
        "list", "summarize", "explain", "describe",
        "visit", "navigate to", "go to", "open"
    ]

    # Goal indicators - ongoing, continuous, no clear end
    goal_patterns = [
        "make money", "earn", "generate income", "$/month", "dollars a month",
        "learn", "improve", "optimize", "maintain", "keep",
        "monitor", "watch for", "track", "stay updated",
        "become", "master", "get better at",
        "always", "continuously", "ongoing", "regularly"
    ]

    task_score = sum(1 for p in task_patterns if p in text_lower)
    goal_score = sum(1 for p in goal_patterns if p in text_lower)

    # Default to task if unclear (tasks are safer - they complete)
    if goal_score > task_score:
        return "goal"
    return "task"


@app.get("/api/directives")
async def get_directives(user: str = Depends(verify_token)):
    """Get all active directives."""
    directives = load_directives_from_file()
    # Filter out completed ones older than 24 hours
    now = datetime.now(timezone.utc)
    active = []
    for d in directives:
        if d.get("status") != "completed":
            active.append(d)
        else:
            # Parse completed_at with timezone awareness
            try:
                completed_at_str = d.get("completed_at", d.get("created_at"))
                completed_at = datetime.fromisoformat(completed_at_str.replace('Z', '+00:00'))
                # Make sure we're comparing timezone-aware datetimes
                if completed_at.tzinfo is None:
                    completed_at = completed_at.replace(tzinfo=timezone.utc)
                if completed_at > now - timedelta(hours=24):
                    active.append(d)
            except Exception as e:
                # If parsing fails, include it to be safe
                print(f"[Directives] Error parsing datetime: {e}")
                active.append(d)

    # Enrich with saturation from findings
    findings_file = Path(config.AI_MEMORY_PATH) / "cerebro" / "cognitive_loop" / "directive_findings.json"
    findings_data = {}
    if findings_file.exists():
        try:
            findings_data = json.loads(findings_file.read_text())
        except:
            findings_data = {}

    for d in active:
        directive_findings = findings_data.get(d["id"], {})
        d["saturation"] = directive_findings.get("saturation", 0)
        d["findings_count"] = len(directive_findings.get("findings", []))

    return {"directives": active}

@app.post("/api/directives")
async def create_directive(request: DirectiveRequest, user: str = Depends(verify_token)):
    """Create a new directive for the cognitive loop."""
    directives = load_directives_from_file()

    # Auto-classify if type not provided
    directive_type = request.directive_type or classify_directive_type(request.text)

    directive = {
        "id": str(uuid.uuid4())[:8],
        "text": request.text,
        "type": directive_type,  # "task" or "goal"
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "created_by": user,
        "image_path": request.image_path
    }

    directives.insert(0, directive)
    save_directives_to_file(directives)

    print(f"[Directive] Created {directive_type}: {request.text[:50]}...")

    # Notify connected clients
    if sio:
        await sio.emit("directive_added", directive, room=os.environ.get("CEREBRO_ROOM", "default"))

    # v2.0: Wake/start the cognitive loop when a directive is submitted.
    # - DORMANT (stopped): Only start if auto_awake checkbox is checked
    # - RUNNING (awake/dreaming/idle): Always wake to process immediately
    if COGNITIVE_LOOP_AVAILABLE and cognitive_loop_manager:
        loop_state = cognitive_loop_manager.get_state()
        print(f"[Directive] Loop status={loop_state.status}, auto_awake={request.auto_awake}")
        if loop_state.status in ("stopped", "error"):
            if request.auto_awake:
                print(f"[Directive] Auto-awakening loop for: {request.text[:50]}")
                level = cognitive_loop_manager.autonomy_level or 2
                await cognitive_loop_manager.start_loop(level)
                await sio.emit("autonomy_status", cognitive_loop_manager.get_state().to_dict(), room=os.environ.get("CEREBRO_ROOM", "default"))
            else:
                print(f"[Directive] Loop dormant + auto_awake OFF — directive stays PENDING")
        else:
            # Loop is running (awake/dreaming/idle) — always wake it
            print(f"[Directive] Loop running — waking to process: {request.text[:50]}")
            cognitive_loop_manager.wake()

    return {"success": True, "directive": directive}

@app.get("/api/directives/{directive_id}/summary")
async def get_directive_summary(directive_id: str, user: str = Depends(verify_token)):
    """
    Generate an AI-synthesized summary of all findings for a directive.
    Returns a structured report with key findings, recommendations, and next steps.
    """
    directives = load_directives_from_file()
    directive = None
    for d in directives:
        if d["id"] == directive_id:
            directive = d
            break

    if not directive:
        raise HTTPException(status_code=404, detail="Directive not found")

    # Load findings for this directive
    findings_file = Path(config.AI_MEMORY_PATH) / "cerebro" / "cognitive_loop" / "directive_findings.json"
    findings_data = {}
    if findings_file.exists():
        try:
            findings_data = json.loads(findings_file.read_text())
        except:
            pass

    directive_findings = findings_data.get(directive_id, {})
    findings = directive_findings.get("findings", [])
    saturation = directive_findings.get("saturation", 0)

    if not findings:
        return {
            "success": True,
            "summary": "## No Findings Yet\n\nCerebro is still researching this directive. Check back soon for a summary of discoveries.",
            "saturation": saturation,
            "findings_count": 0
        }

    # Categorize findings
    observations = [f for f in findings if f.get("type") == "observation"]
    learnings = [f for f in findings if f.get("type") == "learning"]
    insights = [f for f in findings if f.get("type") == "insight"]

    # Build context for LLM
    findings_text = ""
    if observations:
        findings_text += "**Observations:**\n" + "\n".join([f"- {f['content'][:300]}" for f in observations[:8]]) + "\n\n"
    if learnings:
        findings_text += "**Learnings:**\n" + "\n".join([f"- {f['content'][:300]}" for f in learnings[:5]]) + "\n\n"
    if insights:
        findings_text += "**Insights:**\n" + "\n".join([f"- {f['content'][:300]}" for f in insights[:5]]) + "\n\n"

    directive_type = directive.get("type", "task")
    is_task = directive_type == "task"

    prompt = f"""You are Cerebro, an AI assistant. Professor gave you this {'task' if is_task else 'ongoing goal'}:

"{directive.get('text', '')}"

Research saturation: {saturation}%
Total findings: {len(findings)}

Here are my findings:
{findings_text}

Generate a clear, well-structured summary report in markdown format. Include:

## Executive Summary
A 2-3 sentence overview of what was discovered.

## Key Findings
The most important discoveries, organized clearly.

## {'Final Answer' if is_task else 'Current Progress'}
{'Provide the direct answer to the task if possible.' if is_task else 'What has been accomplished toward this goal so far.'}

## Recommended Next Steps
{'What to do with this information.' if is_task else 'Concrete actions to continue making progress.'}

{'## Conclusion\nSummarize and indicate this task is ready to be marked complete.' if is_task and saturation >= 80 else ''}

Keep it concise but informative. Use bullet points where helpful. Be actionable."""

    try:
        from cognitive_loop.ollama_client import OllamaClient, ChatMessage

        ollama = OllamaClient()
        response = await ollama.chat(
            [
                ChatMessage(role="system", content="You are a helpful AI research assistant generating a summary report."),
                ChatMessage(role="user", content=prompt)
            ],
            thinking=False,
            max_tokens=1500
        )
        summary = response.content if hasattr(response, 'content') else str(response)

        return {
            "success": True,
            "summary": summary,
            "saturation": saturation,
            "findings_count": len(findings),
            "directive_type": directive_type
        }

    except Exception as e:
        print(f"[Summary] Error generating summary: {e}")
        # Fallback to basic summary
        basic_summary = f"""## Summary for: {directive.get('text', '')[:100]}

**Research Progress:** {saturation}% complete with {len(findings)} findings.

### Observations ({len(observations)})
{chr(10).join([f'- {f["content"][:150]}...' for f in observations[:3]])}

### Learnings ({len(learnings)})
{chr(10).join([f'- {f["content"][:150]}...' for f in learnings[:3]])}

### Insights ({len(insights)})
{chr(10).join([f'- {f["content"][:150]}...' for f in insights[:3]])}

*AI summary generation failed. Showing raw findings above.*"""

        return {
            "success": True,
            "summary": basic_summary,
            "saturation": saturation,
            "findings_count": len(findings),
            "fallback": True
        }


def _directive_has_followup(directive_text: str) -> bool:
    """
    Check if a directive contains follow-up instructions that require
    human input AFTER the initial agent task completes.
    e.g. "find the top 3 posts, then ask me which one I want you to summarize"
    """
    followup_patterns = [
        "ask me", "ask which", "then ask", "let me choose", "let me pick",
        "ask professor", "which one i want", "which one do i",
        "my opinion", "my input", "my preference", "what i think",
        "confirm with me", "check with me", "before you",
        "wait for my", "get my approval", "let me decide",
        "show me .* and ask", "present .* and ask",
    ]
    import re
    text_lower = directive_text.lower()
    for pattern in followup_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


async def auto_complete_directive_from_agent(directive_id: str, agent_id: str, agent_output: str):
    """
    Auto-complete a directive when its spawned agent finishes successfully.
    Called internally when an agent with a directive_id completes.

    IMPORTANT: If the directive contains follow-up instructions requiring human input
    (e.g. "ask me which one"), we mark it as 'needs_followup' instead of 'completed'
    so the OODA loop can trigger ask_question on the next cycle.
    """
    directives = load_directives_from_file()

    for d in directives:
        if d["id"] == directive_id:
            directive_text = d.get("text", "")

            # Check if directive has follow-up instructions needing human input
            if _directive_has_followup(directive_text):
                d["status"] = "active"  # Keep active so OODA picks it up again
                d["agent_output"] = agent_output[:2000] if agent_output else ""
                d["agent_completed"] = True
                d["agent_completed_at"] = datetime.now(timezone.utc).isoformat()
                d["agent_completed_by"] = agent_id
                d["needs_followup"] = True
                save_directives_to_file(directives)

                # Emit update (not completion) so frontend knows agent is done but directive continues
                await sio.emit("directive_updated", {
                    "directive_id": directive_id,
                    "status": "active",
                    "needs_followup": True,
                    "agent_completed": True,
                    "agent_id": agent_id,
                    "message": "Agent completed initial task. Waiting for your input on next steps."
                }, room=os.environ.get("CEREBRO_ROOM", "default"))

                print(f"[Directive] Directive {directive_id} needs follow-up - NOT auto-completing (agent {agent_id} done)")
                return True

            # Normal auto-complete for directives without follow-up
            d["status"] = "completed"
            d["completed_at"] = datetime.now(timezone.utc).isoformat()
            d["completed_by_agent"] = agent_id
            # Store a summary of the agent's output as the final answer
            d["final_answer"] = f"Completed by Agent {agent_id}:\n{agent_output[:1000] if agent_output else 'Task completed successfully'}"
            save_directives_to_file(directives)

            # Award XP for directive completion
            try:
                tama_state = _load_tamagotchi_state()
                tama_state["lifetime_xp"] = tama_state.get("lifetime_xp", 0) + 10
                tama_state["directives_completed"] = tama_state.get("directives_completed", 0) + 1
                _save_tamagotchi_state(tama_state)
            except Exception:
                pass

            # Emit directive completion event to frontend
            await sio.emit("directive_completed", {
                "directive_id": directive_id,
                "status": "completed",
                "completed_at": d["completed_at"],
                "completed_by_agent": agent_id
            }, room=os.environ.get("CEREBRO_ROOM", "default"))

            print(f"[Directive] Auto-completed directive {directive_id} via agent {agent_id}")
            return True

    print(f"[Directive] Could not find directive {directive_id} to auto-complete")
    return False


@app.post("/api/directives/{directive_id}/complete")
async def complete_directive(directive_id: str, user: str = Depends(verify_token)):
    """Mark a directive as completed."""
    directives = load_directives_from_file()

    for d in directives:
        if d["id"] == directive_id:
            d["status"] = "completed"
            d["completed_at"] = datetime.now().isoformat()
            save_directives_to_file(directives)

            # Award XP for manual directive completion
            try:
                tama_state = _load_tamagotchi_state()
                tama_state["lifetime_xp"] = tama_state.get("lifetime_xp", 0) + 10
                tama_state["directives_completed"] = tama_state.get("directives_completed", 0) + 1
                _save_tamagotchi_state(tama_state)
            except Exception:
                pass

            return {"success": True}

    raise HTTPException(status_code=404, detail="Directive not found")

@app.delete("/api/directives/completed/clear")
async def clear_completed_directives(user: str = Depends(verify_token)):
    """Remove all completed directives."""
    directives = load_directives_from_file()
    directives = [d for d in directives if d.get("status") != "completed"]
    save_directives_to_file(directives)
    return {"success": True}

@app.delete("/api/directives/{directive_id}")
async def delete_directive(directive_id: str, user: str = Depends(verify_token)):
    """Delete a directive."""
    directives = load_directives_from_file()

    original_len = len(directives)
    directives = [d for d in directives if d["id"] != directive_id]

    if len(directives) == original_len:
        raise HTTPException(status_code=404, detail="Directive not found")

    save_directives_to_file(directives)
    return {"success": True}

@app.put("/api/directives/{directive_id}/activate")
async def activate_directive(directive_id: str, user: str = Depends(verify_token)):
    """Set a directive as the currently active one."""
    directives = load_directives_from_file()

    # Deactivate all others, activate the target
    found = False
    for d in directives:
        if d["id"] == directive_id:
            d["status"] = "active"
            found = True
        elif d["status"] == "active":
            d["status"] = "pending"

    if not found:
        raise HTTPException(status_code=404, detail="Directive not found")

    save_directives_to_file(directives)
    return {"success": True}


@app.post("/api/directives/{directive_id}/toggle-pause")
async def toggle_directive_pause(directive_id: str, user: str = Depends(verify_token)):
    """Toggle pause state for a directive."""
    directives = load_directives_from_file()

    for d in directives:
        if d["id"] == directive_id:
            d["paused"] = not d.get("paused", False)
            save_directives_to_file(directives)
            return {"success": True, "paused": d["paused"]}

    raise HTTPException(status_code=404, detail="Directive not found")


@app.get("/api/directives/{directive_id}/findings")
async def get_directive_findings(directive_id: str, user: str = Depends(verify_token)):
    """Get all findings/research for a directive."""
    findings_file = Path(config.AI_MEMORY_PATH) / "cerebro" / "cognitive_loop" / "directive_findings.json"

    findings_data = {}
    if findings_file.exists():
        try:
            findings_data = json.loads(findings_file.read_text())
        except:
            findings_data = {}

    directive_findings = findings_data.get(directive_id, {
        "findings": [],
        "saturation": 0
    })

    return {
        "findings": directive_findings.get("findings", []),
        "saturation": directive_findings.get("saturation", 0)
    }


@app.post("/api/directives/{directive_id}/findings")
async def add_directive_finding(directive_id: str, request: Request, user: str = Depends(verify_token)):
    """Add a finding to a directive's research."""
    data = await request.json()
    findings_file = Path(config.AI_MEMORY_PATH) / "cerebro" / "cognitive_loop" / "directive_findings.json"

    findings_data = {}
    if findings_file.exists():
        try:
            findings_data = json.loads(findings_file.read_text())
        except:
            findings_data = {}

    if directive_id not in findings_data:
        findings_data[directive_id] = {"findings": [], "saturation": 0}

    finding = {
        "id": str(uuid.uuid4())[:8],
        "type": data.get("type", "observation"),
        "content": data.get("content", ""),
        "timestamp": datetime.now().isoformat(),
        "phase": data.get("phase", "observe")
    }

    findings_data[directive_id]["findings"].insert(0, finding)

    # Calculate saturation based on findings count and diversity
    findings = findings_data[directive_id]["findings"]
    types = set(f.get("type") for f in findings)
    count = len(findings)

    # Saturation formula: more findings + more diverse types = higher saturation
    # Max at around 20 findings with good diversity
    base_saturation = min(count * 4, 60)  # Max 60 from count
    diversity_bonus = len(types) * 10  # Max 40 from diversity (4 types)
    saturation = min(base_saturation + diversity_bonus, 100)

    findings_data[directive_id]["saturation"] = saturation

    # Keep only last 50 findings per directive
    if len(findings_data[directive_id]["findings"]) > 50:
        findings_data[directive_id]["findings"] = findings_data[directive_id]["findings"][:50]

    findings_file.parent.mkdir(parents=True, exist_ok=True)
    findings_file.write_text(json.dumps(findings_data, indent=2))

    return {"success": True, "finding": finding, "saturation": saturation}


@app.post("/command")
async def run_command(command: str, user: str = Depends(verify_token)):
    """Run a quick system command."""

    # Whitelist of safe commands
    safe_prefixes = ['start ', 'explorer ', 'code ', 'git ', 'npm ', 'python -c', 'curl ']
    is_safe = any(command.lower().startswith(p) for p in safe_prefixes)

    if not is_safe:
        return {"error": "Command not in safe list", "allowed": safe_prefixes}

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=config.AI_MEMORY_PATH
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

        return {
            "command": command,
            "stdout": stdout.decode('utf-8', errors='replace')[:2000],
            "stderr": stderr.decode('utf-8', errors='replace')[:500],
            "returncode": process.returncode
        }
    except asyncio.TimeoutError:
        return {"error": "Command timed out"}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# Scheduler System
# ============================================================================

SCHEDULES_PATH = Path(config.AI_MEMORY_PATH) / "schedules"

# APScheduler - optional, graceful degradation if not installed
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger
    scheduler = AsyncIOScheduler()
    SCHEDULER_AVAILABLE = True
except ImportError:
    scheduler = None
    SCHEDULER_AVAILABLE = False
    print("APScheduler not installed - scheduling will be disabled. Install with: pip install APScheduler")


class ScheduleRequest(BaseModel):
    name: str
    agent_type: str = "researcher"
    prompt: str
    schedule_type: str = "once"  # once, recurring
    frequency: str = "daily"  # daily, weekly, monthly, custom
    date: Optional[str] = None
    time: str = "09:00"
    days_of_week: list[int] = [1, 2, 3, 4, 5]
    cron: Optional[str] = None
    enabled: bool = True
    timeout: Optional[int] = 3600  # seconds (0 = unlimited, default 1 hour)


class ScheduleUpdate(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    prompt: Optional[str] = None
    agent_type: Optional[str] = None
    schedule_type: Optional[str] = None
    frequency: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    days_of_week: Optional[list[int]] = None
    cron: Optional[str] = None
    timeout: Optional[int] = None


@app.post("/schedules")
async def create_schedule(request: ScheduleRequest, user: str = Depends(verify_token)):
    """Create a new scheduled automation."""
    schedule_id = f"sched_{uuid.uuid4().hex[:8]}"

    schedule = {
        "id": schedule_id,
        "name": request.name,
        "agent_type": request.agent_type,
        "prompt": request.prompt,
        "schedule_type": request.schedule_type,
        "frequency": request.frequency,
        "date": request.date,
        "time": request.time,
        "days_of_week": request.days_of_week,
        "cron": request.cron,
        "enabled": request.enabled,
        "timeout": request.timeout or 3600,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": user,
        "last_run": None,
        "next_run": None,
        "run_count": 0
    }

    # Save to disk
    SCHEDULES_PATH.mkdir(parents=True, exist_ok=True)
    schedule_file = SCHEDULES_PATH / f"{schedule_id}.json"
    schedule_file.write_text(json.dumps(schedule, indent=2))

    # Register with APScheduler if enabled and available
    if request.enabled and SCHEDULER_AVAILABLE:
        try:
            register_schedule_job(schedule)
        except Exception as e:
            print(f"Failed to register schedule with APScheduler: {e}")

    return {"success": True, "schedule": schedule}


@app.get("/schedules")
async def list_schedules(user: str = Depends(verify_token)):
    """List all scheduled automations."""
    schedules = []
    if SCHEDULES_PATH.exists():
        for f in SCHEDULES_PATH.glob("sched_*.json"):
            try:
                schedules.append(json.loads(f.read_text()))
            except Exception as e:
                print(f"Error reading schedule {f}: {e}")
    return {"schedules": sorted(schedules, key=lambda x: x.get("created_at", ""), reverse=True)}


@app.get("/schedules/history")
async def get_execution_history(limit: int = 20, user: str = Depends(verify_token)):
    """Get execution history."""
    history_file = SCHEDULES_PATH / "execution_log.json"
    if not history_file.exists():
        return {"executions": []}
    try:
        data = json.loads(history_file.read_text())
        return {"executions": data.get("executions", [])[:limit]}
    except:
        return {"executions": []}


@app.delete("/schedules/history")
async def clear_execution_history(user: str = Depends(verify_token)):
    """Clear all execution history."""
    history_file = SCHEDULES_PATH / "execution_log.json"
    if history_file.exists():
        history_file.write_text(json.dumps({"executions": []}, indent=2))
    return {"status": "cleared"}


@app.get("/schedules/{schedule_id}")
async def get_schedule(schedule_id: str, user: str = Depends(verify_token)):
    """Get a specific schedule."""
    schedule_file = SCHEDULES_PATH / f"{schedule_id}.json"
    if not schedule_file.exists():
        raise HTTPException(status_code=404, detail="Schedule not found")
    return json.loads(schedule_file.read_text())


@app.patch("/schedules/{schedule_id}")
async def update_schedule(schedule_id: str, update: ScheduleUpdate, user: str = Depends(verify_token)):
    """Update a schedule."""
    schedule_file = SCHEDULES_PATH / f"{schedule_id}.json"
    if not schedule_file.exists():
        raise HTTPException(status_code=404, detail="Schedule not found")

    schedule = json.loads(schedule_file.read_text())

    # Apply all updates
    if update.name is not None:
        schedule["name"] = update.name
    if update.enabled is not None:
        schedule["enabled"] = update.enabled
    if update.prompt is not None:
        schedule["prompt"] = update.prompt
    if update.agent_type is not None:
        schedule["agent_type"] = update.agent_type
    if update.schedule_type is not None:
        schedule["schedule_type"] = update.schedule_type
    if update.frequency is not None:
        schedule["frequency"] = update.frequency
    if update.date is not None:
        schedule["date"] = update.date
    if update.time is not None:
        schedule["time"] = update.time
    if update.days_of_week is not None:
        schedule["days_of_week"] = update.days_of_week
    if update.cron is not None:
        schedule["cron"] = update.cron
    if update.timeout is not None:
        schedule["timeout"] = update.timeout

    schedule_file.write_text(json.dumps(schedule, indent=2))

    # Update APScheduler job if available
    if SCHEDULER_AVAILABLE:
        try:
            scheduler.remove_job(schedule_id)
        except:
            pass
        if schedule.get("enabled"):
            try:
                register_schedule_job(schedule)
            except Exception as e:
                print(f"Failed to re-register schedule: {e}")

    return {"success": True, "schedule": schedule}


@app.delete("/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str, user: str = Depends(verify_token)):
    """Delete a schedule."""
    schedule_file = SCHEDULES_PATH / f"{schedule_id}.json"
    if schedule_file.exists():
        schedule_file.unlink()
    # Remove from APScheduler
    if SCHEDULER_AVAILABLE:
        try:
            scheduler.remove_job(schedule_id)
        except:
            pass
    return {"success": True}


@app.post("/schedules/{schedule_id}/run")
async def run_schedule_now(schedule_id: str, user: str = Depends(verify_token)):
    """Manually trigger a schedule execution."""
    schedule_file = SCHEDULES_PATH / f"{schedule_id}.json"
    if not schedule_file.exists():
        raise HTTPException(status_code=404, detail="Schedule not found")

    schedule = json.loads(schedule_file.read_text())
    asyncio.create_task(execute_scheduled_task(schedule, trigger="manual"))
    return {"success": True, "message": "Execution started"}


async def execute_scheduled_task(schedule: dict, trigger: str = "scheduled"):
    """Execute a scheduled automation by spawning an agent via create_agent (shares queue)."""
    execution_id = f"exec_{uuid.uuid4().hex[:8]}"
    started_at = datetime.now(timezone.utc)

    try:
        timeout = schedule.get("timeout", 3600)
        agent_id = await create_agent(
            task=schedule["prompt"],
            agent_type=schedule["agent_type"],
            context=f"Scheduled automation: {schedule['name']}",
            timeout=timeout,
            source="scheduler",
        )

        # Attach execution metadata after creation
        agent = active_agents.get(agent_id)
        if agent:
            agent["schedule_id"] = schedule["id"]
            agent["schedule_name"] = schedule["name"]
            agent["execution_id"] = execution_id

        log_execution(execution_id, schedule, started_at, "running", trigger, agent_id=agent_id)

        # Update schedule last_run
        schedule_file = SCHEDULES_PATH / f"{schedule['id']}.json"
        if schedule_file.exists():
            sched_data = json.loads(schedule_file.read_text())
            sched_data["last_run"] = started_at.isoformat()
            sched_data["run_count"] = sched_data.get("run_count", 0) + 1
            schedule_file.write_text(json.dumps(sched_data, indent=2))

    except Exception as e:
        log_execution(execution_id, schedule, started_at, "failed", trigger, error=str(e))
        print(f"Failed to execute scheduled task: {e}")


def log_execution(exec_id, schedule, started_at, status, trigger, error=None, agent_id=None):
    """Log an execution to history."""
    SCHEDULES_PATH.mkdir(parents=True, exist_ok=True)
    history_file = SCHEDULES_PATH / "execution_log.json"

    if history_file.exists():
        try:
            data = json.loads(history_file.read_text())
        except:
            data = {"executions": []}
    else:
        data = {"executions": []}

    data["executions"].insert(0, {
        "id": exec_id,
        "schedule_id": schedule["id"],
        "schedule_name": schedule["name"],
        "agent_type": schedule["agent_type"],
        "agent_id": agent_id,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "trigger": trigger,
        "error": error
    })

    # Keep last 100 executions
    data["executions"] = data["executions"][:100]
    history_file.write_text(json.dumps(data, indent=2))


def update_execution_status(execution_id: str, status: str, error: str = None):
    """Update an existing execution's status (called when agent completes)."""
    SCHEDULES_PATH.mkdir(parents=True, exist_ok=True)
    history_file = SCHEDULES_PATH / "execution_log.json"

    if not history_file.exists():
        return

    try:
        data = json.loads(history_file.read_text())
    except:
        return

    # Find and update the execution
    for exec_entry in data.get("executions", []):
        if exec_entry.get("id") == execution_id:
            exec_entry["status"] = status
            exec_entry["completed_at"] = datetime.now(timezone.utc).isoformat()
            if error:
                exec_entry["error"] = error
            break

    history_file.write_text(json.dumps(data, indent=2))


def register_schedule_job(schedule: dict):
    """Register a schedule with APScheduler."""
    if not SCHEDULER_AVAILABLE or not scheduler:
        print(f"Scheduler not available, cannot register {schedule.get('name')}")
        return

    try:
        if schedule["schedule_type"] == "once":
            # One-time execution
            if schedule.get("date") and schedule.get("time"):
                # Parse as local time (user's timezone)
                import zoneinfo
                try:
                    local_tz = zoneinfo.ZoneInfo("America/New_York")  # EST
                except:
                    local_tz = None

                run_date = datetime.fromisoformat(f"{schedule['date']}T{schedule['time']}")
                if local_tz:
                    run_date = run_date.replace(tzinfo=local_tz)

                # Check if the time is in the past
                now = datetime.now(local_tz) if local_tz else datetime.now()
                if run_date <= now:
                    # Check if this schedule was already executed at/after its scheduled time
                    last_run = schedule.get("last_run")
                    already_ran = False
                    if last_run:
                        try:
                            last_run_dt = datetime.fromisoformat(last_run)
                            # If last_run is timezone-naive, localize it
                            if last_run_dt.tzinfo is None and local_tz:
                                last_run_dt = last_run_dt.replace(tzinfo=timezone.utc)
                            already_ran = last_run_dt >= run_date.astimezone(timezone.utc) if run_date.tzinfo else last_run_dt >= run_date
                        except (ValueError, TypeError):
                            pass

                    if already_ran:
                        print(f"[SCHEDULER] One-time schedule '{schedule['name']}' already executed on {last_run}, skipping")
                        return
                    else:
                        # Missed schedule - execute immediately
                        print(f"[SCHEDULER] Missed one-time schedule '{schedule['name']}' (was {run_date}, now {now}) - executing immediately")
                        import asyncio
                        asyncio.create_task(execute_scheduled_task(schedule, trigger="missed"))
                        return

                print(f"Registering one-time job {schedule['name']} for {run_date}")

                # Create a wrapper function that captures the schedule
                # AsyncIOScheduler can run coroutines directly
                async def scheduled_job_runner(sched=schedule):
                    print(f"[SCHEDULER] Executing scheduled job: {sched['name']}")
                    try:
                        await execute_scheduled_task(sched, "scheduled")
                        print(f"[SCHEDULER] Job {sched['name']} completed")
                    except Exception as e:
                        print(f"[SCHEDULER] Job {sched['name']} failed: {e}")

                scheduler.add_job(
                    scheduled_job_runner,
                    DateTrigger(run_date=run_date),
                    id=schedule["id"],
                    replace_existing=True
                )
                print(f"Successfully registered schedule: {schedule['name']}")
        else:
            # Recurring execution
            hour, minute = map(int, schedule["time"].split(":"))

            # Create async job runner - captures schedule properly
            async def recurring_job_runner(sched=schedule):
                print(f"[SCHEDULER] Executing recurring job: {sched['name']}")
                try:
                    await execute_scheduled_task(sched, "scheduled")
                    print(f"[SCHEDULER] Recurring job {sched['name']} completed")
                except Exception as e:
                    print(f"[SCHEDULER] Recurring job {sched['name']} failed: {e}")

            if schedule["frequency"] == "daily":
                print(f"Registering daily job {schedule['name']} at {hour}:{minute:02d}")
                scheduler.add_job(
                    recurring_job_runner,
                    CronTrigger(hour=hour, minute=minute),
                    id=schedule["id"],
                    replace_existing=True
                )
            elif schedule["frequency"] == "weekly":
                # Convert day numbers to cron day_of_week format
                days = ",".join(str(d) for d in schedule.get("days_of_week", [1, 2, 3, 4, 5]))
                print(f"Registering weekly job {schedule['name']} on days {days} at {hour}:{minute:02d}")
                scheduler.add_job(
                    recurring_job_runner,
                    CronTrigger(day_of_week=days, hour=hour, minute=minute),
                    id=schedule["id"],
                    replace_existing=True
                )
            elif schedule["frequency"] == "monthly":
                print(f"Registering monthly job {schedule['name']} on day 1 at {hour}:{minute:02d}")
                scheduler.add_job(
                    recurring_job_runner,
                    CronTrigger(day=1, hour=hour, minute=minute),
                    id=schedule["id"],
                    replace_existing=True
                )
            elif schedule["frequency"] == "custom" and schedule.get("cron"):
                print(f"Registering custom cron job {schedule['name']}: {schedule['cron']}")
                scheduler.add_job(
                    recurring_job_runner,
                    CronTrigger.from_crontab(schedule["cron"]),
                    id=schedule["id"],
                    replace_existing=True
                )
            print(f"Successfully registered recurring schedule: {schedule['name']}")
    except Exception as e:
        print(f"Failed to register schedule job {schedule.get('id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def start_scheduler():
    """Initialize scheduler on startup."""
    SCHEDULES_PATH.mkdir(parents=True, exist_ok=True)

    if not SCHEDULER_AVAILABLE:
        print("Scheduler not available - skipping schedule registration")
        return

    # Load existing schedules
    for f in SCHEDULES_PATH.glob("sched_*.json"):
        try:
            schedule = json.loads(f.read_text())
            if schedule.get("enabled"):
                register_schedule_job(schedule)
                print(f"Registered schedule: {schedule['name']}")
        except Exception as e:
            print(f"Failed to register schedule from {f}: {e}")

    scheduler.start()
    print("Scheduler started")


# ============================================================================
# Skills API Endpoints - Verify, Reload (unique endpoints)
# ============================================================================

@app.post("/api/skills/{skill_id}/verify")
async def verify_skill_endpoint(skill_id: str, request: Request, user: str = Depends(verify_token)):
    """Verify a skill works correctly by actually executing it."""
    from cognitive_loop.skill_generator import get_skill_generator, SkillStatus
    from datetime import datetime

    gen = get_skill_generator()
    skill = gen.get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")

    # Actually run the skill to verify it works
    # This uses the subprocess executor which handles Windows asyncio issues
    result = await gen.execute_skill(skill_id)

    # Update skill status based on actual execution
    if result.success:
        skill.status = SkillStatus.VERIFIED
        skill.success_count += 1
    else:
        skill.status = SkillStatus.FAILED
        skill.fail_count += 1
    skill.updated_at = datetime.now()
    gen._save_skill(skill)

    # Emit event for frontend
    if sio:
        await sio.emit("skill_verified", {
            "skill_id": skill_id,
            "skill_name": skill.name,
            "success": result.success,
            "steps_passed": result.steps_completed,
            "steps_total": result.total_steps,
            "output": result.output[:500] if result.output and isinstance(result.output, str) else str(result.output)[:500] if result.output else None
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

    return {
        "skill_id": skill_id,
        "skill_name": skill.name,
        "success": result.success,
        "steps_passed": result.steps_completed,
        "steps_total": result.total_steps,
        "error": result.error,
        "output": result.output
    }


@app.post("/api/skills/reload")
async def reload_skills_cache(user: str = Depends(verify_token)):
    """Reload skills from disk into cache. Use after manual skill creation."""
    from cognitive_loop.skill_generator import get_skill_generator
    gen = get_skill_generator()
    old_count = len(gen._skills_cache)
    gen._load_skills()  # Reload from disk
    new_count = len(gen._skills_cache)
    return {
        "success": True,
        "old_count": old_count,
        "new_count": new_count,
        "skills_added": new_count - old_count
    }


# ============================================================================
# Simulation Engine API Endpoints
# ============================================================================

@app.get("/api/simulation/health")
async def simulation_health():
    """Check SimEngine availability."""
    from sim_engine_client import get_sim_engine_client
    client = get_sim_engine_client()
    healthy = await client.check_health()
    return {"available": healthy, "url": client.base_url}


@app.post("/api/simulation/run")
async def run_simulation_endpoint(request: Request):
    """Run a simulation via natural language."""
    body = await request.json()
    query = body.get("query", body.get("text", ""))
    if not query:
        raise HTTPException(400, "Missing 'query' field")

    from sim_engine_client import get_sim_engine_client
    client = get_sim_engine_client()

    await sio.emit("simulation_started", {"query": query}, room=os.environ.get("CEREBRO_ROOM", "default"))

    try:
        result = await client.run_full_pipeline(query)
        await sio.emit("simulation_complete", result, room=os.environ.get("CEREBRO_ROOM", "default"))
        return result
    except Exception as e:
        error = {"error": str(e), "query": query}
        await sio.emit("simulation_error", error, room=os.environ.get("CEREBRO_ROOM", "default"))
        raise HTTPException(500, str(e))


# ============================================================================
# Alpaca Trading API Endpoints (Cerebro's Wallet)
# ============================================================================

@app.get("/api/trading/health")
async def trading_health():
    """Check Alpaca connectivity and account status."""
    from alpaca_client import get_alpaca_client
    client = get_alpaca_client()
    health = await client.check_health()
    return health


@app.get("/api/trading/account")
async def trading_account(user: str = Depends(verify_token)):
    """Get full Alpaca account details."""
    from alpaca_client import get_alpaca_client
    client = get_alpaca_client()
    if not client.configured:
        raise HTTPException(400, "Alpaca API keys not configured. Add them to backend/.env")
    try:
        account = await client.get_account()
        return account
    except Exception as e:
        raise HTTPException(500, f"Alpaca API error: {e}")


@app.get("/api/trading/positions")
async def trading_positions(user: str = Depends(verify_token)):
    """List all open positions."""
    from alpaca_client import get_alpaca_client
    client = get_alpaca_client()
    if not client.configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        return await client.list_positions()
    except Exception as e:
        raise HTTPException(500, f"Alpaca API error: {e}")


@app.get("/api/trading/orders")
async def trading_orders(status: str = "open", user: str = Depends(verify_token)):
    """List orders. Query param: ?status=open|closed|all"""
    from alpaca_client import get_alpaca_client
    client = get_alpaca_client()
    if not client.configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        return await client.list_orders(status=status)
    except Exception as e:
        raise HTTPException(500, f"Alpaca API error: {e}")


@app.post("/api/trading/order")
async def trading_submit_order(request: Request, user: str = Depends(verify_token)):
    """
    Submit a trade order.

    Body JSON:
    {
        "symbol": "AAPL",
        "qty": 1,              // or "notional": 50.00 for dollar amount
        "side": "buy",         // buy or sell
        "type": "market",      // market, limit, stop, stop_limit, trailing_stop
        "time_in_force": "day" // day, gtc, ioc, fok
        "limit_price": 150,    // for limit/stop_limit
        "stop_price": 140,     // for stop/stop_limit
        "trail_percent": 5.0   // for trailing_stop
    }
    """
    from alpaca_client import get_alpaca_client, log_trade
    client = get_alpaca_client()
    if not client.configured:
        raise HTTPException(400, "Alpaca API keys not configured")

    body = await request.json()
    symbol = body.get("symbol")
    if not symbol:
        raise HTTPException(400, "Missing 'symbol' field")

    try:
        order = await client.submit_order(
            symbol=symbol,
            qty=body.get("qty"),
            notional=body.get("notional"),
            side=body.get("side", "buy"),
            order_type=body.get("type", "market"),
            time_in_force=body.get("time_in_force", "day"),
            limit_price=body.get("limit_price"),
            stop_price=body.get("stop_price"),
            trail_percent=body.get("trail_percent"),
            trail_price=body.get("trail_price"),
        )

        # Log the trade and award XP
        log_entry = log_trade(order, source=body.get("source", "manual"))
        _add_xp(5)  # 5 XP per trade

        # Emit trade event to frontend
        await sio.emit("trade_executed", {
            "order": order,
            "log": log_entry,
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

        return order

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        await sio.emit("trade_error", {
            "error": str(e),
            "symbol": symbol,
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
        raise HTTPException(500, f"Trade failed: {e}")


@app.delete("/api/trading/order/{order_id}")
async def trading_cancel_order(order_id: str, user: str = Depends(verify_token)):
    """Cancel a specific order."""
    from alpaca_client import get_alpaca_client
    client = get_alpaca_client()
    if not client.configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        return await client.cancel_order(order_id)
    except Exception as e:
        raise HTTPException(500, f"Cancel failed: {e}")


@app.get("/api/trading/quote/{symbol:path}")
async def trading_quote(symbol: str):
    """Get latest quote for a stock or crypto symbol. No auth required.
    Stocks: /api/trading/quote/AAPL
    Crypto: /api/trading/quote/BTC/USD
    """
    from alpaca_client import get_alpaca_client
    client = get_alpaca_client()
    if not client.configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        if "/" in symbol:
            return await client.get_crypto_quote(symbol)
        return await client.get_quote(symbol)
    except Exception as e:
        raise HTTPException(500, f"Quote failed: {e}")


@app.get("/api/trading/bars/{symbol:path}")
async def trading_bars(symbol: str, timeframe: str = "1Day", limit: int = 30):
    """Get historical price bars (OHLCV). No auth required.
    Stocks: /api/trading/bars/AAPL
    Crypto: /api/trading/bars/BTC/USD
    """
    from alpaca_client import get_alpaca_client
    client = get_alpaca_client()
    if not client.configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        if "/" in symbol:
            return await client.get_crypto_bars(symbol, timeframe, limit)
        return await client.get_bars(symbol, timeframe, limit)
    except Exception as e:
        raise HTTPException(500, f"Bars failed: {e}")


@app.get("/api/trading/portfolio/history")
async def trading_portfolio_history(period: str = "1M", timeframe: str = "1D", user: str = Depends(verify_token)):
    """Get portfolio value history."""
    from alpaca_client import get_alpaca_client
    client = get_alpaca_client()
    if not client.configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        return await client.get_portfolio_history(period, timeframe)
    except Exception as e:
        raise HTTPException(500, f"Portfolio history failed: {e}")


@app.get("/api/trading/clock")
async def trading_clock():
    """Get market clock (is market open/closed). No auth required."""
    from alpaca_client import get_alpaca_client
    client = get_alpaca_client()
    if not client.configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        return await client.get_clock()
    except Exception as e:
        raise HTTPException(500, f"Clock failed: {e}")


@app.get("/api/trading/log")
async def trading_log(limit: int = 50, user: str = Depends(verify_token)):
    """Get recent trade log from memory."""
    from alpaca_client import _load_trade_log
    log = _load_trade_log()
    return log[-limit:]


# ============================================================================
# Wallet — Unified Financial Activity Tracker
# ============================================================================

@app.post("/api/wallet/log")
async def wallet_log_entry(request: Request):
    """Log any financial activity. No auth — agents use curl."""
    from alpaca_client import log_wallet_entry
    body = await request.json()
    entry = log_wallet_entry(
        category=body.get("category", "other"),
        description=body.get("description", ""),
        pnl=float(body.get("pnl", 0)),
        symbol=body.get("symbol", ""),
        side=body.get("side", ""),
        qty=str(body.get("qty", "")),
        notional=str(body.get("notional", "")),
        source=body.get("source", "agent"),
        metadata=body.get("metadata", {}),
    )
    # Real-time push to frontend
    await sio.emit("wallet_update", entry, room=os.environ.get("CEREBRO_ROOM", "default"))
    return entry


@app.get("/api/wallet/summary")
async def wallet_summary(user: str = Depends(verify_token)):
    """P&L totals + live Alpaca open positions + full wallet history."""
    from alpaca_client import _load_wallet_log, get_alpaca_client
    log = _load_wallet_log()
    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")

    # Compute period P&L totals
    totals = {"today": 0.0, "week": 0.0, "month": 0.0, "all": 0.0}
    today_count = 0
    for e in log:
        ts = e.get("timestamp", "")
        pnl = float(e.get("pnl", 0))
        totals["all"] += pnl
        if ts[:10] == today_str:
            totals["today"] += pnl
            today_count += 1
        try:
            entry_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            delta = (now - entry_dt).days
            if delta < 7:
                totals["week"] += pnl
            if delta < 30:
                totals["month"] += pnl
        except Exception:
            pass

    # Live Alpaca open positions
    unrealized = 0.0
    open_positions = []
    try:
        client = get_alpaca_client()
        if client.configured:
            raw = await client.list_positions()
            if isinstance(raw, list):
                for p in raw:
                    upl = float(p.get("unrealized_pl", 0))
                    unrealized += upl
                    open_positions.append({
                        "symbol": p.get("symbol", ""),
                        "qty": p.get("qty", "0"),
                        "side": p.get("side", ""),
                        "avg_entry_price": p.get("avg_entry_price", ""),
                        "current_price": p.get("current_price", ""),
                        "market_value": p.get("market_value", ""),
                        "unrealized_pl": upl,
                        "unrealized_plpc": p.get("unrealized_plpc", ""),
                    })
    except Exception:
        pass

    return {
        "totals": totals,
        "unrealized": unrealized,
        "open_positions": open_positions,
        "history": list(reversed(log)),
        "today_count": today_count,
    }


@app.get("/api/wallet/entries")
async def wallet_entries(period: str = "week", user: str = Depends(verify_token)):
    """Paginated entries for a given period."""
    from alpaca_client import _load_wallet_log
    log = _load_wallet_log()
    now = datetime.now(timezone.utc)

    if period == "all":
        filtered = log
    else:
        days = {"today": 0, "week": 7, "month": 30}.get(period, 7)
        filtered = []
        for e in log:
            try:
                ts = e.get("timestamp", "")
                entry_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if (now - entry_dt).days <= days:
                    filtered.append(e)
            except Exception:
                pass

    return list(reversed(filtered))


# ============================================================================
# Human-in-the-Loop Authentication API
# ============================================================================

AUTH_SIGNAL_DIR = Path(config.AI_MEMORY_PATH) / "cerebro" / "signals"

@app.get("/api/auth/pending")
async def get_pending_auth_requests(user: str = Depends(verify_token)):
    """Check for pending authentication requests from skill executor."""
    AUTH_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

    pending = []
    for signal_file in AUTH_SIGNAL_DIR.glob("auth_needed_*.json"):
        try:
            with open(signal_file, "r") as f:
                data = json.load(f)
                data["signal_file"] = str(signal_file)
                pending.append(data)
        except Exception as e:
            print(f"[Auth] Error reading signal file {signal_file}: {e}")

    return {"pending": pending, "count": len(pending)}


@app.post("/api/auth/continue/{session_id}")
async def continue_after_auth(session_id: str, user: str = Depends(verify_token)):
    """Signal that human has completed authentication for a session."""
    continue_file = AUTH_SIGNAL_DIR / f"auth_continue_{session_id}.json"

    try:
        with open(continue_file, "w") as f:
            json.dump({"session_id": session_id, "timestamp": time.time()}, f)

        # Emit event
        if sio:
            await sio.emit("auth_continued", {
                "session_id": session_id,
                "message": "Authentication completed, skill execution resuming"
            }, room=os.environ.get("CEREBRO_ROOM", "default"))

        return {"success": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/auth/cancel/{session_id}")
async def cancel_auth_request(session_id: str, user: str = Depends(verify_token)):
    """Cancel a pending authentication request."""
    signal_file = AUTH_SIGNAL_DIR / f"auth_needed_{session_id}.json"
    continue_file = AUTH_SIGNAL_DIR / f"auth_continue_{session_id}.json"

    try:
        if signal_file.exists():
            signal_file.unlink()
        # Write a "cancel" signal
        with open(continue_file, "w") as f:
            json.dump({"session_id": session_id, "cancelled": True, "timestamp": time.time()}, f)
        return {"success": True, "cancelled": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Background task to watch for auth signals and emit to frontend
async def auth_signal_watcher():
    """Periodically check for auth signal files and emit events."""
    seen_signals = set()

    while True:
        try:
            AUTH_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

            for signal_file in AUTH_SIGNAL_DIR.glob("auth_needed_*.json"):
                signal_name = signal_file.name
                if signal_name not in seen_signals:
                    try:
                        with open(signal_file, "r") as f:
                            data = json.load(f)

                        # Emit event to frontend
                        if sio:
                            await sio.emit("auth_needed", {
                                "session_id": data.get("session_id"),
                                "reason": data.get("reason"),
                                "url": data.get("url"),
                                "message": data.get("message", "Please complete authentication in the browser window")
                            }, room=os.environ.get("CEREBRO_ROOM", "default"))

                        seen_signals.add(signal_name)
                        print(f"[Auth] Emitted auth_needed event for {data.get('session_id')}")
                    except Exception as e:
                        print(f"[Auth] Error processing signal {signal_file}: {e}")

            # Clean up seen signals for deleted files
            current_signals = {f.name for f in AUTH_SIGNAL_DIR.glob("auth_needed_*.json")}
            seen_signals = seen_signals & current_signals

        except Exception as e:
            print(f"[Auth] Watcher error: {e}")

        await asyncio.sleep(2)  # Check every 2 seconds


# Start auth watcher on startup
@app.on_event("startup")
async def start_auth_watcher():
    """Start the auth signal watcher background task."""
    asyncio.create_task(auth_signal_watcher())
    print("[Auth] Signal watcher started")


# ============================================================================
# Browser Manager API - Persistent Chromium Environment
# ============================================================================

@app.get("/api/browser/status")
async def browser_status(user: str = Depends(verify_token)):
    """Get current browser status: running, URL, CDP port."""
    try:
        from cognitive_loop.browser_manager import get_browser_manager
        mgr = get_browser_manager()
        status = await mgr.get_status()
        return status
    except Exception as e:
        return {"running": False, "error": str(e)}


@app.get("/api/browser/screenshot")
async def browser_screenshot(user: str = Depends(verify_token)):
    """Return a base64-encoded screenshot of the current page."""
    try:
        from cognitive_loop.browser_manager import get_browser_manager
        mgr = get_browser_manager()
        if not mgr.is_alive():
            raise HTTPException(status_code=503, detail="Browser is not running")

        img = await mgr.screenshot()
        if img is None:
            raise HTTPException(status_code=500, detail="Screenshot failed")

        return {"screenshot": img, "format": "png", "encoding": "base64"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BrowserNavigateRequest(BaseModel):
    url: str
    wait_until: str = "domcontentloaded"


@app.post("/api/browser/navigate")
async def browser_navigate(request: BrowserNavigateRequest, user: str = Depends(verify_token)):
    """Navigate the persistent browser to a URL."""
    try:
        from cognitive_loop.browser_manager import get_browser_manager
        mgr = get_browser_manager()
        await mgr.ensure_running()

        result = await mgr.navigate(request.url, wait_until=request.wait_until)

        # Emit socket event so frontend knows the browser navigated
        await sio.emit("browser_navigated", {
            "url": result.get("url"),
            "title": result.get("title"),
            "success": result.get("success"),
        }, room=os.environ.get("CEREBRO_ROOM", "default"))

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/browser/launch")
async def browser_launch(user: str = Depends(verify_token)):
    """Launch the persistent browser if not already running."""
    try:
        from cognitive_loop.browser_manager import get_browser_manager
        mgr = get_browser_manager()
        await mgr.ensure_running()
        status = await mgr.get_status()

        await sio.emit("browser_launched", status, room=os.environ.get("CEREBRO_ROOM", "default"))

        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/browser/shutdown")
async def browser_shutdown(user: str = Depends(verify_token)):
    """Shut down the persistent browser."""
    try:
        from cognitive_loop.browser_manager import get_browser_manager
        mgr = get_browser_manager()
        await mgr.shutdown()

        await sio.emit("browser_shutdown", {}, room=os.environ.get("CEREBRO_ROOM", "default"))

        return {"success": True, "message": "Browser shut down"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/browser/pause")
async def browser_pause(user: str = Depends(verify_token)):
    """Pause Cerebro's browser automation so user can interact with Chrome."""
    from cognitive_loop.browser_manager import get_browser_manager
    mgr = get_browser_manager()
    mgr.pause("User requested control")
    await sio.emit("browser_paused", {"reason": "User took control", "timestamp": datetime.now(timezone.utc).isoformat()}, room=os.environ.get("CEREBRO_ROOM", "default"))
    return {"success": True, "paused": True}


@app.post("/api/browser/resume")
async def browser_resume(user: str = Depends(verify_token)):
    """Resume Cerebro's browser automation after user is done."""
    from cognitive_loop.browser_manager import get_browser_manager
    mgr = get_browser_manager()
    mgr.resume()
    await sio.emit("browser_resumed", {"timestamp": datetime.now(timezone.utc).isoformat()}, room=os.environ.get("CEREBRO_ROOM", "default"))
    return {"success": True, "paused": False}


@app.get("/api/browser/tabs")
async def browser_tabs(user: str = Depends(verify_token)):
    """List all open browser tabs."""
    from cognitive_loop.browser_manager import get_browser_manager
    mgr = get_browser_manager()
    tabs = await mgr.get_all_pages()
    return {"tabs": tabs}


@app.post("/api/browser/tab/new")
async def browser_new_tab(url: Optional[str] = None, user: str = Depends(verify_token)):
    """Open a new tab in the shared browser."""
    from cognitive_loop.browser_manager import get_browser_manager
    mgr = get_browser_manager()
    result = await mgr.open_new_tab(url)
    await sio.emit("browser_tab_opened", result, room=os.environ.get("CEREBRO_ROOM", "default"))
    return result


@app.post("/api/browser/tab/close")
async def browser_close_tab(tab_id: str, user: str = Depends(verify_token)):
    """Close a browser tab."""
    from cognitive_loop.browser_manager import get_browser_manager
    mgr = get_browser_manager()
    success = await mgr.close_tab(tab_id)
    if success:
        await sio.emit("browser_tab_closed", {"tab_id": tab_id}, room=os.environ.get("CEREBRO_ROOM", "default"))
    return {"success": success}


@app.post("/api/browser/user-done")
async def browser_user_done(user: str = Depends(verify_token)):
    """User signals they're done with manual browser action (CAPTCHA, login, etc.)."""
    from cognitive_loop.browser_manager import get_browser_manager
    mgr = get_browser_manager()
    mgr.resume()
    await sio.emit("browser_resumed", {"reason": "User completed action", "timestamp": datetime.now(timezone.utc).isoformat()}, room=os.environ.get("CEREBRO_ROOM", "default"))
    return {"success": True, "message": "Cerebro resuming browser automation"}


# ============================================================================
# Browser Control API — Claude Agent HTTP Interface (Cerebro v2.0)
# ============================================================================
# These endpoints let Claude Code agents control the SHARED Chrome browser
# via simple curl commands. They use PageUnderstanding for element indexing.

class BrowserClickRequest(BaseModel):
    element_index: Optional[int] = None
    selector: Optional[str] = None

class BrowserFillRequest(BaseModel):
    element_index: Optional[int] = None
    selector: Optional[str] = None
    value: str

class BrowserScrollRequest(BaseModel):
    direction: str = "down"
    amount: int = 500

class BrowserPressKeyRequest(BaseModel):
    key: str


# Cache last page analysis so click/fill can reference by element_index
_last_page_state = {"state": None, "elements": []}


async def _ensure_browser_running():
    """Auto-launch the browser if not already running. Returns (mgr, page)."""
    from cognitive_loop.browser_manager import get_browser_manager
    mgr = get_browser_manager()
    if not mgr.is_alive():
        print("[v2.0] Browser not running - auto-launching for agent...")
        await mgr.ensure_running()
        await sio.emit("browser_launched", await mgr.get_status(), room=os.environ.get("CEREBRO_ROOM", "default"))
    page = await mgr.get_page()
    return mgr, page


@app.get("/api/browser/page_state")
async def browser_page_state():
    """Get compressed page state with numbered interactable elements + content.
    No auth required — called by Claude agents via curl."""
    try:
        from cognitive_loop.page_understanding import PageUnderstanding
        mgr, page = await _ensure_browser_running()
        pu = PageUnderstanding()
        state = await pu.analyze_page(page)
        compressed = pu.compress_for_llm(state, max_tokens=3000)

        # Cache elements for click/fill by index
        _last_page_state["state"] = state
        _last_page_state["elements"] = state.interactable_elements

        return {"state": compressed, "url": state.url, "title": state.title}
    except Exception as e:
        return {"error": str(e), "state": ""}


@app.post("/api/browser/click")
async def browser_click(request: BrowserClickRequest):
    """Click an element by index (from page_state) or CSS selector.
    No auth required — called by Claude agents via curl."""
    try:
        mgr, page = await _ensure_browser_running()

        if request.selector:
            await page.locator(request.selector).first.click(timeout=10000)
        elif request.element_index is not None:
            elements = _last_page_state.get("elements", [])
            if request.element_index < 0 or request.element_index >= len(elements):
                # Refresh page state and try again
                from cognitive_loop.page_understanding import PageUnderstanding
                pu = PageUnderstanding()
                state = await pu.analyze_page(page)
                elements = state.interactable_elements
                _last_page_state["state"] = state
                _last_page_state["elements"] = elements

            if request.element_index < 0 or request.element_index >= len(elements):
                return {"success": False, "error": f"Element index {request.element_index} out of range (0-{len(elements)-1})"}

            el = elements[request.element_index]
            await page.locator(el.selector).first.click(timeout=10000)
        else:
            return {"success": False, "error": "Provide element_index or selector"}

        await sio.emit("browser_step", {"step": "click", "action": f"index={request.element_index} sel={request.selector}", "reasoning": ""}, room=os.environ.get("CEREBRO_ROOM", "default"))
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/browser/fill")
async def browser_fill(request: BrowserFillRequest):
    """Fill an input element by index or selector.
    No auth required — called by Claude agents via curl."""
    try:
        mgr, page = await _ensure_browser_running()

        if request.selector:
            await page.locator(request.selector).first.fill(request.value, timeout=10000)
        elif request.element_index is not None:
            elements = _last_page_state.get("elements", [])
            if request.element_index < 0 or request.element_index >= len(elements):
                from cognitive_loop.page_understanding import PageUnderstanding
                pu = PageUnderstanding()
                state = await pu.analyze_page(page)
                elements = state.interactable_elements
                _last_page_state["state"] = state
                _last_page_state["elements"] = elements

            if request.element_index < 0 or request.element_index >= len(elements):
                return {"success": False, "error": f"Element index {request.element_index} out of range (0-{len(elements)-1})"}

            el = elements[request.element_index]
            await page.locator(el.selector).first.fill(request.value, timeout=10000)
        else:
            return {"success": False, "error": "Provide element_index or selector"}

        await sio.emit("browser_step", {"step": "fill", "action": f"value={request.value[:50]}", "reasoning": ""}, room=os.environ.get("CEREBRO_ROOM", "default"))
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/browser/scroll")
async def browser_scroll(request: BrowserScrollRequest):
    """Scroll the page up or down.
    No auth required — called by Claude agents via curl."""
    try:
        mgr, page = await _ensure_browser_running()
        delta = request.amount if request.direction == "down" else -request.amount
        await page.evaluate(f"window.scrollBy(0, {delta})")
        await asyncio.sleep(0.5)  # Let content settle

        await sio.emit("browser_step", {"step": "scroll", "action": f"{request.direction} {request.amount}px", "reasoning": ""}, room=os.environ.get("CEREBRO_ROOM", "default"))
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/browser/press_key")
async def browser_press_key(request: BrowserPressKeyRequest):
    """Press a keyboard key (Enter, Tab, Escape, etc.).
    No auth required — called by Claude agents via curl."""
    try:
        mgr, page = await _ensure_browser_running()
        await page.keyboard.press(request.key)

        await sio.emit("browser_step", {"step": "press_key", "action": request.key, "reasoning": ""}, room=os.environ.get("CEREBRO_ROOM", "default"))
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/browser/screenshot/file")
async def browser_screenshot_file():
    """Save a screenshot to a temp file and return the path.
    Claude agents can then Read the image file to visually analyze it.
    No auth required — called by Claude agents via curl."""
    try:
        import tempfile
        mgr, page = await _ensure_browser_running()
        # Save to a known temp location
        screenshot_path = os.path.join(tempfile.gettempdir(), "cerebro_browser_screenshot.png")
        await page.screenshot(path=screenshot_path, full_page=False)

        return {"success": True, "path": screenshot_path}
    except Exception as e:
        return {"success": False, "error": str(e), "path": ""}


@app.post("/api/browser/agent/navigate")
async def browser_agent_navigate(request: BrowserNavigateRequest):
    """Navigate the browser to a URL — no auth, auto-launches browser.
    Called by Claude agents via curl."""
    try:
        mgr, page = await _ensure_browser_running()
        result = await mgr.navigate(request.url, wait_until=request.wait_until)
        await sio.emit("browser_navigated", {
            "url": result.get("url"),
            "title": result.get("title"),
            "success": result.get("success"),
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Agent HITL API — Blocking question endpoint (Cerebro v2.0)
# ============================================================================
# Allows Claude agents to ask the user questions and BLOCK until answered.
# _agent_questions and _agent_answers are defined near active_agents (line ~657)


class AgentAskRequest(BaseModel):
    question: str
    options: Optional[list[str]] = None
    agent_id: Optional[str] = None


@app.post("/api/notify/file-ready")
async def notify_file_ready(request: Request):
    """Notify frontend that an agent created a file on the NAS workspace.
    No auth required — called by Claude agents via curl."""
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}

    filename = body.get("filename", "unknown")
    filepath = body.get("path", "")
    description = body.get("description", "")

    # Build the Windows-accessible UNC path
    windows_path = filepath.replace("/mnt/nas/AI_MEMORY/", "\\\\192.168.0.21\\home\\AI_MEMORY\\").replace("/", "\\")

    notification = {
        "type": "file_ready",
        "filename": filename,
        "nas_path": filepath,
        "windows_path": windows_path,
        "description": description,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Emit to frontend
    room = os.environ.get("CEREBRO_ROOM", "default")
    await sio.emit("notification", notification, room=room)
    await sio.emit("file_ready", notification, room=room)
    print(f"[FileNotify] {filename}: {filepath}")

    return {"success": True, "notification": notification}


@app.post("/api/agent/ask")
async def agent_ask(request: AgentAskRequest):
    """Claude agent asks user a question. BLOCKS until user responds (up to 5 min).
    No auth required — called by Claude agents via curl."""
    import uuid as uuid_lib
    question_id = f"agentq_{uuid_lib.uuid4().hex[:12]}"

    # Create event for this question
    event = asyncio.Event()
    _agent_questions[question_id] = event

    # Emit to frontend as cerebro_question (reuse existing HITL popup)
    await sio.emit("cerebro_question", {
        "id": question_id,
        "question": request.question,
        "options": request.options or [],
        "agent_id": request.agent_id or "unknown",
        "type": "agent_ask",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, room=os.environ.get("CEREBRO_ROOM", "default"))

    # Also emit as human_input_needed for backwards compat
    await sio.emit("human_input_needed", {
        "id": question_id,
        "question": request.question,
        "type": "agent_ask",
        "options": request.options or [],
        "agent_id": request.agent_id or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, room=os.environ.get("CEREBRO_ROOM", "default"))

    print(f"[AgentAsk] Question {question_id}: {request.question[:80]}")

    # Block until user responds or timeout
    try:
        await asyncio.wait_for(event.wait(), timeout=300)
        answer = _agent_answers.pop(question_id, "")
        _agent_questions.pop(question_id, None)
        print(f"[AgentAsk] Answer for {question_id}: {answer[:80]}")
        return {"answer": answer, "question_id": question_id}
    except asyncio.TimeoutError:
        _agent_questions.pop(question_id, None)
        _agent_answers.pop(question_id, None)
        print(f"[AgentAsk] Timeout for {question_id}")
        return {"answer": "", "error": "timeout", "question_id": question_id}


@app.post("/api/agent/answer")
async def agent_answer_http(request: dict):
    """HTTP endpoint for frontend to answer agent questions."""
    question_id = request.get("question_id") or request.get("id") or request.get("request_id", "")
    answer = request.get("answer", "")
    if question_id in _agent_questions:
        _agent_answers[question_id] = answer
        _agent_questions[question_id].set()
        return {"success": True}
    return {"success": False, "error": "Question not found or already answered"}


# Hook into existing socket handler for agent questions
@sio.event
async def agent_question_response(sid, data):
    """Handle agent question responses from frontend via socket."""
    question_id = data.get("question_id") or data.get("id") or data.get("request_id", "")
    answer = data.get("answer", "")
    if question_id in _agent_questions:
        _agent_answers[question_id] = answer
        _agent_questions[question_id].set()
        await sio.emit("agent_answer_acknowledged", {"question_id": question_id, "success": True}, room=os.environ.get("CEREBRO_ROOM", "default"))
    else:
        await sio.emit("agent_answer_acknowledged", {"question_id": question_id, "success": False, "error": "Not found"}, room=os.environ.get("CEREBRO_ROOM", "default"))


# ============================================================================
# Tamagotchi Health System (Cerebro v2.0)
# ============================================================================
# XP/Level system + memory health for the Info tab health panel.
# State persisted to Z:\AI_MEMORY\cerebro\tamagotchi_state.json

TAMAGOTCHI_STATE_PATH = Path(config.AI_MEMORY_PATH) / "cerebro" / "tamagotchi_state.json"
TAMAGOTCHI_LEVEL_THRESHOLDS = [0, 50, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000]


def _load_tamagotchi_state() -> dict:
    try:
        if TAMAGOTCHI_STATE_PATH.exists():
            return json.loads(TAMAGOTCHI_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"lifetime_xp": 0, "actions_logged": 0, "directives_completed": 0}


def _save_tamagotchi_state(state: dict):
    try:
        TAMAGOTCHI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TAMAGOTCHI_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[Tamagotchi] Failed to save state: {e}")


def _compute_level(xp: int) -> int:
    level = 1
    for i, threshold in enumerate(TAMAGOTCHI_LEVEL_THRESHOLDS):
        if xp >= threshold:
            level = i + 1
    return level


def _compute_level_progress(xp: int) -> float:
    level = _compute_level(xp)
    idx = level - 1
    current_threshold = TAMAGOTCHI_LEVEL_THRESHOLDS[idx] if idx < len(TAMAGOTCHI_LEVEL_THRESHOLDS) else TAMAGOTCHI_LEVEL_THRESHOLDS[-1]
    next_threshold = TAMAGOTCHI_LEVEL_THRESHOLDS[idx + 1] if idx + 1 < len(TAMAGOTCHI_LEVEL_THRESHOLDS) else current_threshold + 50000
    if next_threshold == current_threshold:
        return 1.0
    return (xp - current_threshold) / (next_threshold - current_threshold)


def _add_xp(amount: int):
    state = _load_tamagotchi_state()
    state["lifetime_xp"] = state.get("lifetime_xp", 0) + amount
    _save_tamagotchi_state(state)


@app.get("/api/cerebro/tamagotchi")
async def get_tamagotchi_state():
    """Return tamagotchi health data for the Info tab. No auth required."""
    state = _load_tamagotchi_state()
    xp = state.get("lifetime_xp", 0)
    level = _compute_level(xp)
    progress = _compute_level_progress(xp)

    # Memory health check
    nas_path = Path(config.AI_MEMORY_PATH)
    nas_connected = nas_path.exists() and nas_path.is_dir()
    faiss_ok = False
    faiss_size_mb = 0
    try:
        faiss_index = nas_path / "embeddings" / "indexes"
        if faiss_index.exists():
            faiss_ok = True
            faiss_size_mb = round(sum(f.stat().st_size for f in faiss_index.rglob("*") if f.is_file()) / (1024 * 1024), 1)
    except Exception:
        pass

    return {
        "lifetime_xp": xp,
        "level": level,
        "level_progress": round(progress, 3),
        "next_level_xp": TAMAGOTCHI_LEVEL_THRESHOLDS[level] if level < len(TAMAGOTCHI_LEVEL_THRESHOLDS) else xp + 50000,
        "actions_logged": state.get("actions_logged", 0),
        "directives_completed": state.get("directives_completed", 0),
        "memory_health": {
            "nas_connected": nas_connected,
            "faiss_ok": faiss_ok,
            "faiss_size_mb": faiss_size_mb
        }
    }



# ============================================================================
# Network Device Monitor — Known Devices Registry & Alert Endpoints
# ============================================================================

NETWORK_ALERTS_FILE = Path(config.AI_MEMORY_PATH) / "cerebro" / "network_alerts.json"
KNOWN_NETWORK_DEVICES_FILE = Path(config.AI_MEMORY_PATH) / "cerebro" / "known_network_devices.json"

def _load_network_alerts() -> list:
    """Load network alerts from file."""
    if NETWORK_ALERTS_FILE.exists():
        try:
            data = json.loads(NETWORK_ALERTS_FILE.read_text())
            return data if isinstance(data, list) else []
        except Exception:
            pass
    return []

def _save_network_alerts(alerts: list):
    """Save network alerts to file, capped at 200."""
    NETWORK_ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if len(alerts) > 200:
        alerts = alerts[:200]
    NETWORK_ALERTS_FILE.write_text(json.dumps(alerts, indent=2, default=str))

def _load_known_devices() -> dict:
    """Load known network devices registry. Keyed by MAC address."""
    if KNOWN_NETWORK_DEVICES_FILE.exists():
        try:
            data = json.loads(KNOWN_NETWORK_DEVICES_FILE.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}

def _save_known_devices(devices: dict):
    """Save known network devices registry."""
    KNOWN_NETWORK_DEVICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    KNOWN_NETWORK_DEVICES_FILE.write_text(json.dumps(devices, indent=2, default=str))

def _find_known_device(mac: str) -> dict | None:
    """Look up a device by MAC address in the known devices registry."""
    devices = _load_known_devices()
    mac_upper = mac.upper().strip()
    return devices.get(mac_upper)

# --- Known Network Devices CRUD ---

@app.get("/api/network/devices")
async def get_known_devices(user: str = Depends(verify_token)):
    """List all registered known network devices."""
    devices = _load_known_devices()
    return {"devices": list(devices.values()), "count": len(devices)}

@app.post("/api/network/devices")
async def register_known_device(request: Request, user: str = Depends(verify_token)):
    """Register a network device as known (trusted). Body: {mac, name, ip?, vendor?, notes?}"""
    data = await request.json()
    mac = data.get("mac", "").upper().strip()
    name = data.get("name", "").strip()

    if not mac or not name:
        raise HTTPException(status_code=400, detail="Both 'mac' and 'name' are required")

    devices = _load_known_devices()
    now = datetime.now(timezone.utc).isoformat()

    devices[mac] = {
        "mac": mac,
        "name": name,
        "ip": data.get("ip", ""),
        "vendor": data.get("vendor", ""),
        "notes": data.get("notes", ""),
        "registered_at": now,
        "last_seen": now,
        "seen_count": 1,
    }
    _save_known_devices(devices)

    # Auto-resolve any unresolved alerts for this MAC
    alerts = _load_network_alerts()
    resolved_count = 0
    for alert in alerts:
        if alert.get("mac", "").upper() == mac and alert.get("status") == "unresolved":
            alert["status"] = "resolved"
            alert["resolved_by"] = "device_registration"
            alert["resolved_at"] = now
            alert["notes"] = f"Registered as: {name}"
            resolved_count += 1
    if resolved_count:
        _save_network_alerts(alerts)

    print(f"[NetworkDevices] Registered '{name}' (MAC: {mac}), auto-resolved {resolved_count} alerts")
    return {"success": True, "device": devices[mac], "alerts_resolved": resolved_count}

@app.delete("/api/network/devices/{mac}")
async def remove_known_device(mac: str, user: str = Depends(verify_token)):
    """Remove a device from the known devices registry."""
    mac_upper = mac.upper().strip()
    devices = _load_known_devices()
    if mac_upper not in devices:
        raise HTTPException(status_code=404, detail="Device not found")
    removed = devices.pop(mac_upper)
    _save_known_devices(devices)
    return {"success": True, "removed": removed}

@app.patch("/api/network/devices/{mac}")
async def update_known_device(mac: str, request: Request, user: str = Depends(verify_token)):
    """Update a known device's info (name, notes, etc.)."""
    mac_upper = mac.upper().strip()
    devices = _load_known_devices()
    if mac_upper not in devices:
        raise HTTPException(status_code=404, detail="Device not found")
    data = await request.json()
    for key in ("name", "ip", "vendor", "notes"):
        if key in data:
            devices[mac_upper][key] = data[key]
    _save_known_devices(devices)
    return {"success": True, "device": devices[mac_upper]}

# --- Alert Endpoints ---

@app.post("/api/network/alert")
async def receive_network_alert(request: Request):
    """Receive unknown device alert from Darkhorse network monitor. No auth (LAN-only service).
    Known devices get a quiet chat notification. Unknown devices get the full critical popup."""
    data = await request.json()
    unknown_devices = data.get("unknown_devices", [])
    source = data.get("source", "unknown")
    timestamp = data.get("timestamp", datetime.now(timezone.utc).isoformat())

    if not unknown_devices:
        return {"success": False, "error": "No unknown devices in payload"}

    alerts = _load_network_alerts()
    created_alerts = []
    truly_unknown = []
    known_rejoined = []

    for dev in unknown_devices:
        mac = dev.get("mac", "?")
        known = _find_known_device(mac)

        if known:
            # Known device — update last_seen and bump count
            devices = _load_known_devices()
            mac_upper = mac.upper().strip()
            if mac_upper in devices:
                devices[mac_upper]["last_seen"] = datetime.now(timezone.utc).isoformat()
                devices[mac_upper]["seen_count"] = devices[mac_upper].get("seen_count", 0) + 1
                if dev.get("ip"):
                    devices[mac_upper]["ip"] = dev["ip"]
                _save_known_devices(devices)

            known_rejoined.append({
                "name": known["name"],
                "ip": dev.get("ip", known.get("ip", "?")),
                "mac": mac,
            })
        else:
            # Truly unknown — full alert flow
            alert_id = f"net_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:6]}"
            alert = {
                "id": alert_id,
                "ip": dev.get("ip", "?"),
                "mac": mac,
                "vendor": dev.get("vendor", "Unknown"),
                "source": source,
                "status": "unresolved",
                "detected_at": timestamp,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            alerts.insert(0, alert)
            created_alerts.append(alert)

            # Create notification (toast + bell)
            device_label = f"{dev.get('ip', '?')} ({mac[:8]}...)"
            await create_notification(
                notif_type="system",
                title="Unknown Device Detected",
                message=f"New device on network: {device_label} - Vendor: {dev.get('vendor', 'Unknown')}",
            )

            # Create stored item (card in Stored tab)
            stored_item = {
                "id": f"stored_{alert_id}",
                "type": "alert",
                "title": f"Unknown Network Device: {dev.get('ip', '?')}",
                "content": (
                    f"**IP:** {dev.get('ip', '?')}\n"
                    f"**MAC:** {mac}\n"
                    f"**Vendor:** {dev.get('vendor', 'Unknown')}\n"
                    f"**Detected:** {timestamp}\n"
                    f"**Source:** {source}"
                ),
                "metadata": {"alert_id": alert_id, "category": "network_security"},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending",
                "source_id": source,
            }
            items = _load_stored_items()
            items.insert(0, stored_item)
            _save_stored_items(items)
            await sio.emit("cerebro_stored_item_added", stored_item,
                            room=os.environ.get("CEREBRO_ROOM", "default"))

            truly_unknown.append(dev)

    _save_network_alerts(alerts)

    # Emit quiet chat notification for known devices
    if known_rejoined:
        await sio.emit("network_device_joined", {
            "devices": known_rejoined,
            "timestamp": timestamp,
            "source": source,
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
        print(f"[NetworkAlert] {len(known_rejoined)} known device(s) rejoined: {', '.join(d['name'] for d in known_rejoined)}")

    # Emit prominent security alert ONLY for truly unknown devices
    if truly_unknown:
        await sio.emit("network_security_alert", {
            "devices": truly_unknown,
            "timestamp": timestamp,
            "source": source,
            "alert_count": len(created_alerts),
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
        print(f"[NetworkAlert] {len(created_alerts)} unknown device(s) from {source}")

    return {
        "success": True,
        "alerts_created": len(created_alerts),
        "known_devices_seen": len(known_rejoined),
    }

@app.get("/api/network/alerts")
async def get_network_alerts(
    status: str = None,
    limit: int = 50,
    user: str = Depends(verify_token),
):
    """Get network alerts, optionally filtered by status."""
    alerts = _load_network_alerts()
    if status:
        alerts = [a for a in alerts if a.get("status") == status]
    return {"alerts": alerts[:limit]}

@app.patch("/api/network/alerts/{alert_id}")
async def update_network_alert(alert_id: str, request: Request, user: str = Depends(verify_token)):
    """Update a network alert (e.g., mark as resolved, add notes)."""
    data = await request.json()
    alerts = _load_network_alerts()
    for alert in alerts:
        if alert.get("id") == alert_id:
            for key in ("status", "notes", "resolved_by"):
                if key in data:
                    alert[key] = data[key]
            if data.get("status") == "resolved":
                alert["resolved_at"] = datetime.now(timezone.utc).isoformat()
            _save_network_alerts(alerts)
            return {"success": True, "alert": alert}
    raise HTTPException(status_code=404, detail="Alert not found")

# ============================================================================
# Voice Screenshot — Capture & History Endpoints
# ============================================================================

SCREENSHOTS_DIR = Path(config.AI_MEMORY_PATH) / "cerebro" / "screenshots"
SCREENSHOT_INDEX_FILE = SCREENSHOTS_DIR / "index.json"


def _load_screenshot_index() -> list:
    """Load screenshot index from file."""
    if SCREENSHOT_INDEX_FILE.exists():
        try:
            data = json.loads(SCREENSHOT_INDEX_FILE.read_text())
            return data if isinstance(data, list) else []
        except Exception:
            pass
    return []


def _save_screenshot_index(index: list):
    """Save screenshot index, capped at 200 entries."""
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    if len(index) > 200:
        index = index[:200]
    SCREENSHOT_INDEX_FILE.write_text(json.dumps(index, indent=2, default=str))


@app.post("/api/screenshot/capture")
async def screenshot_capture(request: Request):
    """Receive screenshot from PC daemon (no auth — LAN-only)."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    image_b64 = data.get("image_base64")
    window_title = data.get("window_title", "Unknown")
    device = data.get("device", "unknown")
    timestamp_str = data.get("timestamp", datetime.now(timezone.utc).isoformat())

    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing image_base64")

    # Decode and save image
    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data")

    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    screenshot_id = f"ss_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    filename = f"{screenshot_id}.jpg"
    filepath = SCREENSHOTS_DIR / filename

    filepath.write_bytes(image_bytes)
    file_size = len(image_bytes)

    # Update index
    entry = {
        "id": screenshot_id,
        "filename": filename,
        "window_title": window_title,
        "device": device,
        "timestamp": timestamp_str,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": file_size,
    }

    index = _load_screenshot_index()
    index.insert(0, entry)
    _save_screenshot_index(index)

    # Create notification (bell icon)
    await create_notification(
        notif_type="system",
        title="Voice Screenshot",
        message=f"Captured: {window_title}",
    )

    # Emit Socket.IO event with full base64 for real-time display
    await sio.emit(
        "voice_screenshot",
        {
            "id": screenshot_id,
            "image_base64": image_b64,
            "window_title": window_title,
            "device": device,
            "timestamp": timestamp_str,
        },
        room=os.environ.get("CEREBRO_ROOM", "default"),
    )

    print(f"[VoiceScreenshot] Saved: {filename} ({file_size} bytes) — {window_title}")
    return {"success": True, "id": screenshot_id, "filename": filename}


@app.get("/api/screenshot/latest")
async def screenshot_latest(user: str = Depends(verify_token)):
    """Return the most recent screenshot metadata."""
    index = _load_screenshot_index()
    if not index:
        return {"screenshot": None}
    return {"screenshot": index[0]}


@app.get("/api/screenshot/history")
async def screenshot_history(limit: int = 20, user: str = Depends(verify_token)):
    """Return recent screenshot list."""
    index = _load_screenshot_index()
    return {"screenshots": index[:limit], "total": len(index)}


@app.get("/api/screenshot/{screenshot_id}/image")
async def screenshot_image(screenshot_id: str):
    """Serve screenshot image file (no auth — for <img> tags)."""
    index = _load_screenshot_index()
    entry = next((e for e in index if e["id"] == screenshot_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    filepath = SCREENSHOTS_DIR / entry["filename"]
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Image file missing")

    from starlette.responses import FileResponse
    return FileResponse(filepath, media_type="image/jpeg")


@app.get("/api/screenshot/file")
async def screenshot_file(path: str):
    """Serve screenshot file by absolute path (no auth — for <img> tags).
    Security: Only serves files from the screenshots directory."""
    screenshots_base = os.path.realpath(str(SCREENSHOTS_DIR))
    real_path = os.path.realpath(path)
    if not real_path.startswith(screenshots_base + os.sep) and real_path != screenshots_base:
        raise HTTPException(status_code=403, detail="Access denied: path outside screenshots directory")
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(real_path, media_type="image/jpeg")


# Upload directory for user-attached images
UPLOADS_DIR = Path(config.AI_MEMORY_PATH) / "cerebro" / "uploads"

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


@app.post("/api/upload/image")
async def upload_image(request: Request, user: str = Depends(verify_token)):
    """Upload an image via base64 JSON or multipart form."""
    content_type = request.headers.get("content-type", "")

    image_bytes = None
    original_filename = "image.png"

    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        if not file:
            raise HTTPException(status_code=400, detail="Missing 'file' field")
        image_bytes = await file.read()
        original_filename = getattr(file, "filename", "image.png") or "image.png"
    else:
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        image_b64 = data.get("image_base64", "")
        original_filename = data.get("filename", "image.png")

        # Strip data URI prefix if present
        if "," in image_b64 and image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]

        if not image_b64:
            raise HTTPException(status_code=400, detail="Missing image_base64")

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 data")

    # Validate extension
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid image extension: {ext}. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}")

    # Generate upload ID and save
    upload_id = f"upload_{uuid.uuid4().hex[:8]}"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_filename = f"{upload_id}{ext}"
    save_path = UPLOADS_DIR / save_filename
    save_path.write_bytes(image_bytes)

    print(f"[Upload] Saved image: {save_filename} ({len(image_bytes)} bytes)")

    return {
        "success": True,
        "upload_id": upload_id,
        "filename": original_filename,
        "path": str(save_path),
        "url": f"/api/upload/{upload_id}/image"
    }


@app.get("/api/upload/{upload_id}/image")
async def serve_uploaded_image(upload_id: str):
    """Serve uploaded image by upload_id prefix match (no auth — for <img> tags)."""
    uploads_base = os.path.realpath(str(UPLOADS_DIR))

    if not UPLOADS_DIR.exists():
        raise HTTPException(status_code=404, detail="No uploads directory")

    # Find file matching upload_id prefix
    for f in UPLOADS_DIR.iterdir():
        if f.name.startswith(upload_id) and f.is_file():
            real_path = os.path.realpath(str(f))
            if not real_path.startswith(uploads_base + os.sep) and real_path != uploads_base:
                raise HTTPException(status_code=403, detail="Access denied")
            # Determine media type from extension
            ext = f.suffix.lower()
            media_types = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"
            }
            return FileResponse(str(f), media_type=media_types.get(ext, "image/jpeg"))

    raise HTTPException(status_code=404, detail="Upload not found")


# ============================================================================
# Memory Health & Device Management
# ============================================================================

DEVICE_REGISTRY_PATH = Path(config.AI_MEMORY_PATH) / "devices" / "device_registry.json"


def _load_device_registry() -> dict:
    """Load device registry from JSON file."""
    if DEVICE_REGISTRY_PATH.exists():
        try:
            with open(DEVICE_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"devices": {}, "created_at": datetime.now(timezone.utc).isoformat(), "last_updated": datetime.now(timezone.utc).isoformat()}


def _save_device_registry(registry: dict):
    """Save device registry to JSON file."""
    registry["last_updated"] = datetime.now(timezone.utc).isoformat()
    DEVICE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEVICE_REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ============================================================================
# Remote Command Execution - Constants & Helpers
# Personal infrastructure - only used when not standalone
# ============================================================================

COMMAND_LOGS_DIR = Path(config.AI_MEMORY_PATH) / "devices" / "command_logs"
COMMAND_LOGS_DIR.mkdir(parents=True, exist_ok=True)

BLOCKED_COMMAND_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/",  # rm -rf / or rm /
    r"\bmkfs\b",
    r"\bdd\s+.*of=/dev/",
    r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",  # fork bomb
    r"\b(shutdown|reboot|poweroff|halt|init\s+[06])\b",
    r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?\*",  # rm -rf *
    r">\s*/dev/sd[a-z]",
    r"\bchmod\s+(-R\s+)?777\s+/\s*$",
    r"\bformat\b.*[cCdDeE]:",
    r"curl\s.*\|\s*(sudo\s+)?(ba)?sh",  # curl | sh
]

MAX_COMMAND_OUTPUT_BYTES = 64 * 1024  # 64KB
DEFAULT_COMMAND_TIMEOUT = 30
MAX_COMMAND_TIMEOUT = 120

_pending_remote_commands: Dict[str, dict] = {}


class CommandExecRequest(BaseModel):
    command: str
    timeout: int = DEFAULT_COMMAND_TIMEOUT


def _validate_command(command: str) -> tuple:
    """Validate a command against the blocklist. Returns (is_valid, reason)."""
    cmd = command.strip()
    if not cmd:
        return False, "Empty command"
    if len(cmd) > 4096:
        return False, "Command too long (max 4096 chars)"
    for pattern in BLOCKED_COMMAND_PATTERNS:
        if re.search(pattern, cmd):
            return False, f"Command blocked by safety filter (matches: {pattern})"
    return True, ""


def _get_command_log_path(device_id: str) -> Path:
    """Get path to JSONL command log for a device."""
    return COMMAND_LOGS_DIR / f"{device_id}.jsonl"


def _append_command_log(device_id: str, entry: dict):
    """Append a command log entry to the device's JSONL file."""
    log_path = _get_command_log_path(device_id)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_command_logs(device_id: str, limit: int = 50, offset: int = 0) -> list:
    """Read command logs for a device, newest first."""
    log_path = _get_command_log_path(device_id)
    if not log_path.exists():
        return []
    lines = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    lines.reverse()
    return lines[offset:offset + limit]


async def _execute_ssh_command(device: dict, device_id: str, command: str, timeout: int, user: str) -> dict:
    """Execute a command on a remote device via SSH. Returns result dict."""
    ssh_config = device.get("ssh_config", {})
    host = ssh_config.get("host", "").strip()
    username = ssh_config.get("username", "").strip()

    if not host or not username:
        raise HTTPException(status_code=400, detail="SSH not configured: host and username required")

    port = str(ssh_config.get("port", 22))
    key_path = ssh_config.get("key_path", "").strip()

    # Using create_subprocess_exec (no shell) - safe from injection
    ssh_args = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=no",
        "-p", port,
    ]
    if key_path:
        ssh_args.extend(["-i", key_path])
    ssh_args.append(f"{username}@{host}")
    ssh_args.append(command)

    timeout = min(max(timeout, 1), MAX_COMMAND_TIMEOUT)
    command_id = str(uuid.uuid4())[:12]
    start_ts = time.time()

    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        elapsed_ms = int((time.time() - start_ts) * 1000)

        stdout_str = stdout_bytes.decode(errors="replace")[:MAX_COMMAND_OUTPUT_BYTES] if stdout_bytes else ""
        stderr_str = stderr_bytes.decode(errors="replace")[:MAX_COMMAND_OUTPUT_BYTES] if stderr_bytes else ""

        truncated = bool(
            (stdout_bytes and len(stdout_bytes) > MAX_COMMAND_OUTPUT_BYTES) or
            (stderr_bytes and len(stderr_bytes) > MAX_COMMAND_OUTPUT_BYTES)
        )

        result = {
            "command_id": command_id,
            "command": command,
            "device_id": device_id,
            "host": host,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "exit_code": proc.returncode,
            "execution_time_ms": elapsed_ms,
            "truncated": truncated,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user,
        }

    except asyncio.TimeoutError:
        elapsed_ms = int((time.time() - start_ts) * 1000)
        result = {
            "command_id": command_id,
            "command": command,
            "device_id": device_id,
            "host": host,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "exit_code": -1,
            "execution_time_ms": elapsed_ms,
            "truncated": False,
            "timeout": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user,
        }

    # Log to JSONL
    _append_command_log(device_id, result)

    # Fire-and-forget save to AI Memory
    try:
        asyncio.create_task(_save_command_to_memory(device_id, device.get("friendly_name", device_id), result))
    except Exception:
        pass

    return result


async def _save_command_to_memory(device_id: str, device_name: str, result: dict):
    """Save command execution to AI Memory. Fire-and-forget."""
    try:
        exit_str = "OK" if result.get("exit_code") == 0 else f"FAILED (exit {result.get('exit_code')})"
        content = f"Remote command on {device_name} ({device_id}): `{result['command']}`\nResult: {exit_str}"
        if result.get("stdout"):
            content += f"\nOutput: {result['stdout'][:500]}"
        if result.get("stderr"):
            content += f"\nErrors: {result['stderr'][:500]}"
        messages = [
            {"role": "user", "content": f"Execute on {device_name}: {result['command']}"},
            {"role": "assistant", "content": content},
        ]
        await mcp_bridge.save_conversation(
            messages=messages,
            session_id=f"remote_exec_{device_id}",
            metadata={"source": "cerebro", "session_type": "remote_command", "device_id": device_id},
        )
    except Exception as e:
        print(f"[RemoteExec] Memory save error (non-fatal): {e}")


@app.get("/api/memory/health")
async def get_memory_health(full: bool = False, user: str = Depends(verify_token)):
    """Get dynamic memory health status using FastHealthChecker."""
    try:
        _health_src = os.path.join(os.path.dirname(__file__), '..', 'memory', 'src')
        if _health_src not in sys.path:
            sys.path.insert(0, _health_src)
        from health_checker import FastHealthChecker, HealthChecker

        if full:
            checker = HealthChecker(config.AI_MEMORY_PATH)
            report = checker.check_all()
        else:
            checker = FastHealthChecker(config.AI_MEMORY_PATH)
            report = checker.check_all()

        # Map statuses for frontend
        components = {}
        friendly_names = {
            'nas': 'NAS Storage',
            'local_brain': 'Local Brain',
            'local_cache': 'Local Cache',
            'embeddings': 'Embeddings',
            'indexes': 'Vector Index',
            'database': 'Database',
            'mcp_components': 'MCP Components'
        }
        for name, check in report.get('checks', {}).items():
            raw_status = check.get('status', 'unknown')
            if raw_status == 'healthy':
                mapped = 'online'
            elif raw_status in ('down', 'error'):
                mapped = 'offline'
            else:
                mapped = 'unknown'

            components[name] = {
                'name': friendly_names.get(name, name.replace('_', ' ').title()),
                'status': mapped,
                'raw_status': raw_status,
                'details': {k: v for k, v in check.items() if k != 'status'}
            }

        return {
            'overall': report.get('overall', 'unknown'),
            'timestamp': report.get('timestamp'),
            'check_type': report.get('check_type', 'full'),
            'check_time_ms': report.get('check_time_ms'),
            'components': components
        }
    except ImportError:
        # Fallback: basic checks without FastHealthChecker module
        components = {}
        try:
            nas_path = Path(config.AI_MEMORY_PATH)
            if nas_path.exists():
                components['storage'] = {'name': 'Storage', 'status': 'online', 'raw_status': 'healthy'}
            else:
                components['storage'] = {'name': 'Storage', 'status': 'offline', 'raw_status': 'down'}
        except Exception:
            components['storage'] = {'name': 'Storage', 'status': 'offline', 'raw_status': 'error'}

        try:
            if redis:
                await redis.ping()
                components['cache'] = {'name': 'Local Cache', 'status': 'online', 'raw_status': 'healthy'}
            else:
                components['cache'] = {'name': 'Local Cache', 'status': 'offline', 'raw_status': 'down'}
        except Exception:
            components['cache'] = {'name': 'Local Cache', 'status': 'offline', 'raw_status': 'error'}

        overall = 'healthy' if all(c['status'] == 'online' for c in components.values()) else 'degraded'
        return {
            'overall': overall,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'check_type': 'fallback',
            'components': components
        }
    except Exception as e:
        return {
            'overall': 'unknown',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'check_type': 'error',
            'error': str(e),
            'components': {}
        }


@app.get("/api/devices")
async def list_devices(user: str = Depends(verify_token)):
    """List all registered compute devices."""
    registry = _load_device_registry()
    devices = []
    for hostname, info in registry.get("devices", {}).items():
        devices.append({"id": hostname, **info})
    return {"devices": devices, "count": len(devices)}


@app.post("/api/devices")
async def register_device(request: Request, user: str = Depends(verify_token)):
    """Register a new compute device."""
    body = await request.json()
    hostname = body.get("hostname")
    if not hostname:
        raise HTTPException(status_code=400, detail="hostname is required")

    registry = _load_device_registry()
    if hostname in registry.get("devices", {}):
        raise HTTPException(status_code=409, detail=f"Device '{hostname}' already registered")

    device_record = {
        "hostname": hostname,
        "device_type": body.get("device_type", "unknown"),
        "device_name": body.get("device_name", hostname),
        "friendly_name": body.get("friendly_name", body.get("device_name", hostname)),
        "description": body.get("description", ""),
        "os": body.get("os", ""),
        "architecture": body.get("architecture", ""),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "last_seen": datetime.now(timezone.utc).isoformat(),
        "conversation_count": 0,
    }

    ssh_config = body.get("ssh_config")
    if ssh_config:
        device_record["ssh_config"] = {
            "host": ssh_config.get("host", hostname),
            "port": ssh_config.get("port", 22),
            "username": ssh_config.get("username", ""),
            "key_path": ssh_config.get("key_path", ""),
        }

    registry.setdefault("devices", {})[hostname] = device_record
    _save_device_registry(registry)
    return {"success": True, "device": {"id": hostname, **device_record}}


@app.patch("/api/devices/{device_id}")
async def update_device(device_id: str, request: Request, user: str = Depends(verify_token)):
    """Update a device's editable fields."""
    registry = _load_device_registry()
    if device_id not in registry.get("devices", {}):
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")

    body = await request.json()
    device = registry["devices"][device_id]

    for field in ("friendly_name", "description", "device_type", "device_name"):
        if field in body:
            device[field] = body[field]

    if "ssh_config" in body:
        existing_ssh = device.get("ssh_config", {})
        ssh = body["ssh_config"]
        device["ssh_config"] = {
            "host": ssh.get("host", existing_ssh.get("host", device_id)),
            "port": ssh.get("port", existing_ssh.get("port", 22)),
            "username": ssh.get("username", existing_ssh.get("username", "")),
            "key_path": ssh.get("key_path", existing_ssh.get("key_path", "")),
        }

    device["last_seen"] = datetime.now(timezone.utc).isoformat()
    _save_device_registry(registry)
    return {"success": True, "device": {"id": device_id, **device}}


@app.delete("/api/devices/{device_id}")
async def delete_device(device_id: str, user: str = Depends(verify_token)):
    """Remove a device from the registry."""
    registry = _load_device_registry()
    if device_id not in registry.get("devices", {}):
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")

    del registry["devices"][device_id]
    _save_device_registry(registry)
    return {"success": True, "deleted": device_id}


@app.post("/api/devices/{device_id}/ping")
async def ping_device(device_id: str, user: str = Depends(verify_token)):
    """Test connection to a device: ICMP ping + SSH port check + optional auth test."""
    import socket as _socket

    registry = _load_device_registry()
    if device_id not in registry.get("devices", {}):
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")

    device = registry["devices"][device_id]
    ssh_config = device.get("ssh_config", {})
    ssh_host = ssh_config.get("host", "").strip()
    host = ssh_host or device.get("hostname", device_id)
    host_source = "ssh_config" if ssh_host else "hostname"
    ssh_port = ssh_config.get("port", 22)

    results = {
        "host": host,
        "host_source": host_source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if host_source == "hostname":
        results["hint"] = "Set Host/IP in SSH Configuration for accurate testing"

    # ICMP ping
    try:
        proc = await asyncio.create_subprocess_exec(
            "ping", "-c", "1", "-W", "3", host,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        results["ping"] = {
            "reachable": proc.returncode == 0,
            "output": stdout.decode().strip()[:200] if stdout else ""
        }
    except asyncio.TimeoutError:
        results["ping"] = {"reachable": False, "error": "Ping timed out"}
    except Exception as e:
        results["ping"] = {"reachable": False, "error": str(e)}

    # SSH port check
    try:
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.settimeout(3)
        port_result = sock.connect_ex((host, int(ssh_port)))
        results["ssh_port"] = {"open": port_result == 0, "port": ssh_port}
        sock.close()
    except Exception as e:
        results["ssh_port"] = {"open": False, "port": ssh_port, "error": str(e)}

    # SSH auth test (only if key_path provided and port is open)
    ssh_username = ssh_config.get("username", "")
    ssh_key_path = ssh_config.get("key_path", "")
    if ssh_username and ssh_key_path and results.get("ssh_port", {}).get("open"):
        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                "-o", "StrictHostKeyChecking=no",
                "-i", ssh_key_path, "-p", str(ssh_port),
                ssh_username + "@" + host, "echo", "ok",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            results["ssh_auth"] = {
                "success": proc.returncode == 0,
                "output": (stdout.decode().strip() if stdout else stderr.decode().strip())[:100]
            }
        except asyncio.TimeoutError:
            results["ssh_auth"] = {"success": False, "error": "SSH auth timed out"}
        except Exception as e:
            results["ssh_auth"] = {"success": False, "error": str(e)}

    # Update last_seen on successful ping
    if results.get("ping", {}).get("reachable"):
        registry["devices"][device_id]["last_seen"] = datetime.now(timezone.utc).isoformat()
        _save_device_registry(registry)

    return results


@app.get("/api/devices/{device_id}/activity")
async def get_device_activity(device_id: str, limit: int = 10, user: str = Depends(verify_token)):
    """Get recent conversations from a specific device."""
    registry = _load_device_registry()
    if device_id not in registry.get("devices", {}):
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")

    device = registry["devices"][device_id]
    device_tag = device.get("device_type", "unknown")

    conversations = []
    try:
        conv_path = Path(config.AI_MEMORY_PATH) / "conversations"
        if conv_path.exists():
            for conv_file in sorted(conv_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:100]:
                try:
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    meta = data.get("metadata", {})
                    if meta.get("device_tag") == device_tag:
                        first_msg = data.get("messages", [{}])[0].get("content", "")[:100] if data.get("messages") else ""
                        conversations.append({
                            "id": data.get("conversation_id", conv_file.stem),
                            "summary": first_msg,
                            "timestamp": meta.get("timestamp", ""),
                            "device_tag": device_tag
                        })
                        if len(conversations) >= limit:
                            break
                except Exception:
                    continue
    except Exception:
        pass

    return {"device_id": device_id, "device_tag": device_tag, "conversations": conversations, "count": len(conversations)}


# ============================================================================
# Remote Command Execution - Endpoints
# Personal infrastructure - only used when not standalone
# ============================================================================

@app.post("/api/devices/{device_id}/exec")
async def exec_device_command(device_id: str, req: CommandExecRequest, user: str = Depends(verify_token)):
    """Execute a command on a remote device via SSH."""
    registry = _load_device_registry()
    if device_id not in registry.get("devices", {}):
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")

    is_valid, reason = _validate_command(req.command)
    if not is_valid:
        raise HTTPException(status_code=403, detail=reason)

    device = registry["devices"][device_id]
    result = await _execute_ssh_command(device, device_id, req.command, req.timeout, user)

    # Emit real-time update
    try:
        await sio.emit("command_executed", {
            "device_id": device_id,
            "command_id": result["command_id"],
            "command": req.command,
            "exit_code": result["exit_code"],
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
    except Exception:
        pass

    return result


@app.get("/api/devices/{device_id}/commands")
async def get_device_commands(device_id: str, limit: int = 50, offset: int = 0, user: str = Depends(verify_token)):
    """Get command execution history for a device."""
    registry = _load_device_registry()
    if device_id not in registry.get("devices", {}):
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")

    logs = _read_command_logs(device_id, limit=limit, offset=offset)
    return {"device_id": device_id, "commands": logs, "count": len(logs), "offset": offset}


@app.post("/api/devices/{device_id}/exec/propose")
async def propose_device_command(device_id: str, req: CommandExecRequest, user: str = Depends(verify_token)):
    """AI proposes a command for user approval before execution."""
    registry = _load_device_registry()
    if device_id not in registry.get("devices", {}):
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")

    is_valid, reason = _validate_command(req.command)
    if not is_valid:
        raise HTTPException(status_code=403, detail=reason)

    action_id = str(uuid.uuid4())[:12]
    _pending_remote_commands[action_id] = {
        "action_id": action_id,
        "device_id": device_id,
        "command": req.command,
        "timeout": min(max(req.timeout, 1), MAX_COMMAND_TIMEOUT),
        "proposed_by": "ai",
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }

    # Emit approval request to frontend
    try:
        device = registry["devices"][device_id]
        await sio.emit("command_approval_needed", {
            "action_id": action_id,
            "device_id": device_id,
            "device_name": device.get("friendly_name", device_id),
            "command": req.command,
            "timeout": req.timeout,
            "proposed_at": _pending_remote_commands[action_id]["proposed_at"],
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
    except Exception:
        pass

    return {"action_id": action_id, "status": "pending_approval", "command": req.command, "device_id": device_id}


@app.post("/api/devices/exec/approve/{action_id}")
async def approve_device_command(action_id: str, user: str = Depends(verify_token)):
    """Approve and execute a previously proposed command."""
    pending = _pending_remote_commands.pop(action_id, None)
    if not pending:
        raise HTTPException(status_code=404, detail=f"No pending command with action_id '{action_id}'")

    if pending["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Command already {pending['status']}")

    registry = _load_device_registry()
    device_id = pending["device_id"]
    if device_id not in registry.get("devices", {}):
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")

    device = registry["devices"][device_id]
    result = await _execute_ssh_command(device, device_id, pending["command"], pending["timeout"], user)
    result["action_id"] = action_id
    result["approved_by"] = user

    # Emit result
    try:
        await sio.emit("command_approved_result", {
            "action_id": action_id,
            "device_id": device_id,
            "command": pending["command"],
            "exit_code": result["exit_code"],
            "stdout": result["stdout"][:2000],
            "stderr": result["stderr"][:2000],
            "execution_time_ms": result["execution_time_ms"],
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
    except Exception:
        pass

    return result


@app.post("/api/devices/exec/reject/{action_id}")
async def reject_device_command(action_id: str, user: str = Depends(verify_token)):
    """Reject a proposed command."""
    pending = _pending_remote_commands.pop(action_id, None)
    if not pending:
        raise HTTPException(status_code=404, detail=f"No pending command with action_id '{action_id}'")

    pending["status"] = "rejected"
    pending["rejected_at"] = datetime.now(timezone.utc).isoformat()

    # Log rejection
    _append_command_log(pending["device_id"], {
        "command_id": f"rejected_{action_id}",
        "command": pending["command"],
        "device_id": pending["device_id"],
        "status": "rejected",
        "proposed_by": pending.get("proposed_by", "ai"),
        "timestamp": pending["rejected_at"],
    })

    # Emit rejection
    try:
        await sio.emit("command_rejected", {
            "action_id": action_id,
            "device_id": pending["device_id"],
            "command": pending["command"],
        }, room=os.environ.get("CEREBRO_ROOM", "default"))
    except Exception:
        pass

    return {"action_id": action_id, "status": "rejected", "command": pending["command"]}


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=59000)

