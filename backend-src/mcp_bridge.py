"""
MCP Bridge - Connect Cerebro to AI Memory MCP Tools

This module bridges the Cerebro backend to the AI Memory system's capabilities:
- Goals: Track and manage user goals
- Causal: Understand cause-effect relationships
- Predict: Anticipate failures and suggest mitigations
- Learning: Apply and record learnings

Instead of going through MCP protocol, we import the modules directly
for maximum performance and reliability.
"""

import os
import sys
import json
import asyncio
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Add MCP modules to path for imports
# Standalone Docker: /app/mcp_modules (bundled in image)
# Dev: ~/cerebro-mcp/src or CEREBRO_MCP_SRC env var
_default_mcp_path = "/app/mcp_modules" if os.environ.get("CEREBRO_STANDALONE") == "1" else os.path.expanduser("~/cerebro-mcp/src")
MCP_PATH = Path(os.environ.get("CEREBRO_MCP_SRC") or _default_mcp_path)
if str(MCP_PATH) not in sys.path:
    sys.path.insert(0, str(MCP_PATH))

# Configuration
AI_MEMORY_PATH = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data")))

# Thread pool for blocking I/O
_executor = ThreadPoolExecutor(max_workers=4)


class MCPBridge:
    """
    Bridge between Cerebro and AI Memory MCP tools.

    Provides async wrappers around the synchronous AI Memory modules.
    """

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or AI_MEMORY_PATH)
        self._goal_tracker = None
        self._causal_manager = None
        self._predictor = None
        self._learning_extractor = None
        self._memory_service = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Lazy initialization of modules."""
        if self._initialized:
            return

        loop = asyncio.get_event_loop()

        # Initialize modules in thread pool to avoid blocking
        def init_modules():
            try:
                from goal_tracker import GoalTracker
                from causal_model import CausalModelManager
                from predictor import Predictor
                from learning_extractor import LearningExtractor
                from ai_memory_ultimate import UltimateMemoryService

                bp = str(self.base_path)
                return {
                    'goal_tracker': GoalTracker(bp),
                    'causal_manager': CausalModelManager(bp),
                    'predictor': Predictor(bp),
                    'learning_extractor': LearningExtractor(bp),
                    'memory_service': UltimateMemoryService(bp)
                }
            except ImportError as e:
                print(f"[MCP Bridge] Import error: {e}")
                return None
            except Exception as e:
                print(f"[MCP Bridge] Init error: {e}")
                return None

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(_executor, init_modules),
                timeout=20
            )
        except asyncio.TimeoutError:
            print("[MCP Bridge] TIMEOUT: Init took >20s, will retry lazily")
            result = None
        if result:
            self._goal_tracker = result['goal_tracker']
            self._causal_manager = result['causal_manager']
            self._predictor = result['predictor']
            self._learning_extractor = result['learning_extractor']
            self._memory_service = result.get('memory_service')
            self._initialized = True
            print("[MCP Bridge] Initialized successfully")
        else:
            print("[MCP Bridge] Failed to initialize - modules not available")

    # =========================================================================
    # GOALS API
    # =========================================================================

    async def goals(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Goal management API.

        Actions:
            - list_active: List all active goals
            - detect: Detect goals from text
            - add: Add a new goal manually
            - get: Get a specific goal
            - update: Update a goal's status, blockers, subgoals
            - complete: Mark a goal as completed
            - find_relevant: Find goals relevant to context
        """
        await self._ensure_initialized()

        if not self._goal_tracker:
            return {"error": "Goal tracker not available", "success": False}

        loop = asyncio.get_event_loop()

        try:
            if action == "list_active":
                def _list():
                    data = self._goal_tracker._load_active()
                    return data.get("goals", [])
                goals = await loop.run_in_executor(_executor, _list)
                return {"goals": goals, "count": len(goals), "success": True}

            elif action == "detect":
                text = kwargs.get("text", "")
                def _detect():
                    detected = self._goal_tracker.detect_goals(text)
                    return [g.to_dict() for g in detected]
                goals = await loop.run_in_executor(_executor, _detect)
                return {"detected": goals, "count": len(goals), "success": True}

            elif action == "add":
                description = kwargs.get("description", "")
                priority = kwargs.get("priority", "medium")
                def _add():
                    from goal_tracker import Goal
                    goal = Goal(
                        goal_id=self._goal_tracker._generate_id(),
                        description=description,
                        inferred_from="explicit",
                        priority=priority
                    )
                    data = self._goal_tracker._load_active()
                    data["goals"].append(goal.to_dict())
                    self._goal_tracker._save_active(data)
                    return goal.to_dict()
                goal = await loop.run_in_executor(_executor, _add)
                return {"goal": goal, "success": True}

            elif action == "get":
                goal_id = kwargs.get("goal_id", "")
                def _get():
                    data = self._goal_tracker._load_active()
                    for g in data.get("goals", []):
                        if g.get("goal_id") == goal_id:
                            return g
                    return None
                goal = await loop.run_in_executor(_executor, _get)
                if goal:
                    return {"goal": goal, "success": True}
                return {"error": "Goal not found", "success": False}

            elif action == "update":
                goal_id = kwargs.get("goal_id", "")
                def _update():
                    data = self._goal_tracker._load_active()
                    for i, g in enumerate(data.get("goals", [])):
                        if g.get("goal_id") == goal_id:
                            # Apply updates
                            if "status" in kwargs:
                                g["status"] = kwargs["status"]
                            if "priority" in kwargs:
                                g["priority"] = kwargs["priority"]
                            if "add_blocker" in kwargs:
                                g.setdefault("known_blockers", [])
                                g["known_blockers"].append(kwargs["add_blocker"])
                            if "add_subgoal" in kwargs:
                                g.setdefault("subgoals", [])
                                g["subgoals"].append(kwargs["add_subgoal"])
                            if "progress_update" in kwargs:
                                g.setdefault("progress_history", [])
                                g["progress_history"].append({
                                    "update": kwargs["progress_update"],
                                    "timestamp": datetime.now().isoformat()
                                })
                            g["updated_at"] = datetime.now().isoformat()
                            data["goals"][i] = g
                            self._goal_tracker._save_active(data)
                            return g
                    return None
                goal = await loop.run_in_executor(_executor, _update)
                if goal:
                    return {"goal": goal, "success": True}
                return {"error": "Goal not found", "success": False}

            elif action == "complete":
                goal_id = kwargs.get("goal_id", "")
                return await self.goals("update", goal_id=goal_id, status="completed")

            elif action == "find_relevant":
                context = kwargs.get("context", "")
                def _find():
                    data = self._goal_tracker._load_active()
                    relevant = []
                    context_lower = context.lower()
                    for g in data.get("goals", []):
                        if g.get("status") != "active":
                            continue
                        desc = g.get("description", "").lower()
                        # Simple keyword matching
                        if any(word in context_lower for word in desc.split()[:5]):
                            relevant.append(g)
                    return relevant
                goals = await loop.run_in_executor(_executor, _find)
                return {"goals": goals, "count": len(goals), "success": True}

            elif action == "proactive_context":
                # Get context for proactive suggestions
                text = kwargs.get("text", "")
                def _context():
                    data = self._goal_tracker._load_active()
                    active_goals = [g for g in data.get("goals", []) if g.get("status") == "active"]
                    return {
                        "active_goals": active_goals[:5],
                        "total_active": len(active_goals),
                        "has_blockers": any(g.get("known_blockers") for g in active_goals)
                    }
                ctx = await loop.run_in_executor(_executor, _context)
                return {"context": ctx, "success": True}

            else:
                return {"error": f"Unknown action: {action}", "success": False}

        except Exception as e:
            print(f"[MCP Bridge] Goals error: {e}")
            return {"error": str(e), "success": False}

    # =========================================================================
    # CAUSAL API
    # =========================================================================

    async def causal(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Causal model API.

        Actions:
            - find_causes: What causes this effect
            - find_effects: What this cause produces
            - get_interventions: How to prevent/achieve an effect
            - add_link: Add a causal link
            - search: Search causal model
        """
        await self._ensure_initialized()

        if not self._causal_manager:
            return {"error": "Causal manager not available", "success": False}

        loop = asyncio.get_event_loop()

        try:
            if action == "find_causes":
                effect = kwargs.get("effect", "")
                threshold = kwargs.get("threshold", 0.5)
                def _find():
                    return self._causal_manager.find_causes(effect, threshold)
                causes = await loop.run_in_executor(_executor, _find)
                return {"causes": causes, "count": len(causes), "success": True}

            elif action == "find_effects":
                cause = kwargs.get("cause", "")
                threshold = kwargs.get("threshold", 0.5)
                def _find():
                    return self._causal_manager.find_effects(cause, threshold)
                effects = await loop.run_in_executor(_executor, _find)
                return {"effects": effects, "count": len(effects), "success": True}

            elif action == "get_interventions":
                effect = kwargs.get("effect", "")
                def _get():
                    return self._causal_manager.get_interventions(effect)
                interventions = await loop.run_in_executor(_executor, _get)
                return {"interventions": interventions, "success": True}

            elif action == "add_link":
                cause = kwargs.get("cause", "")
                effect = kwargs.get("effect", "")
                mechanism = kwargs.get("mechanism", "")
                interventions = kwargs.get("interventions", [])
                def _add():
                    return self._causal_manager.add_link(
                        cause=cause,
                        effect=effect,
                        mechanism=mechanism,
                        interventions=interventions
                    )
                link_id = await loop.run_in_executor(_executor, _add)
                return {"link_id": link_id, "success": True}

            elif action == "search":
                query = kwargs.get("query", "")
                def _search():
                    return self._causal_manager.search(query, limit=10)
                results = await loop.run_in_executor(_executor, _search)
                return {"results": results, "success": True}

            else:
                return {"error": f"Unknown action: {action}", "success": False}

        except Exception as e:
            print(f"[MCP Bridge] Causal error: {e}")
            return {"error": str(e), "success": False}

    # =========================================================================
    # PREDICT API
    # =========================================================================

    async def predict(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Prediction API.

        Actions:
            - anticipate_failures: Get warnings for a context
            - from_causal: Predict outcome using causal model
            - check_pattern: Check for specific failure patterns
            - preventive_actions: Get suggestions to prevent failures
            - verify: Record actual outcome
        """
        await self._ensure_initialized()

        if not self._predictor:
            return {"error": "Predictor not available", "success": False}

        loop = asyncio.get_event_loop()

        try:
            if action == "anticipate_failures":
                context = kwargs.get("context", "")
                def _anticipate():
                    # Check for dangerous patterns
                    warnings = []

                    # Built-in danger patterns
                    danger_patterns = [
                        ("rm -rf", "Recursive force delete can remove entire directories", 0.95),
                        ("delete all", "Mass deletion is irreversible", 0.9),
                        ("drop table", "Dropping database tables is irreversible", 0.95),
                        ("format", "Formatting will erase all data", 0.9),
                        ("--force", "Force flags bypass safety checks", 0.7),
                        ("sudo rm", "Root-level deletion is dangerous", 0.85),
                        (":wq!", "Force write can overwrite important files", 0.6),
                        ("git push -f", "Force push can overwrite remote history", 0.8),
                        ("reset --hard", "Hard reset discards uncommitted changes", 0.85),
                    ]

                    context_lower = context.lower()
                    for pattern, warning, confidence in danger_patterns:
                        if pattern.lower() in context_lower:
                            warnings.append({
                                "pattern": pattern,
                                "warning": warning,
                                "confidence": confidence,
                                "severity": "high" if confidence > 0.8 else "medium"
                            })

                    # Also check causal model for known failure patterns
                    try:
                        effects = self._causal_manager.find_effects(context, threshold=0.6)
                        for eff in effects[:3]:
                            if any(word in eff.get("effect", "").lower()
                                   for word in ["fail", "error", "break", "crash", "lose", "corrupt"]):
                                warnings.append({
                                    "pattern": "causal",
                                    "warning": f"May cause: {eff.get('effect', 'unknown issue')}",
                                    "confidence": eff.get("confidence", 0.5),
                                    "severity": "medium",
                                    "mechanism": eff.get("mechanism", "")
                                })
                    except:
                        pass

                    return warnings

                warnings = await loop.run_in_executor(_executor, _anticipate)
                return {
                    "warnings": warnings,
                    "has_warnings": len(warnings) > 0,
                    "count": len(warnings),
                    "success": True
                }

            elif action == "from_causal":
                action_text = kwargs.get("action_text", "")
                context = kwargs.get("context", "")
                def _predict():
                    pred = self._predictor.predict_from_causal(action_text, context)
                    return pred.to_dict() if pred else None
                prediction = await loop.run_in_executor(_executor, _predict)
                return {"prediction": prediction, "success": True}

            elif action == "check_pattern":
                pattern_type = kwargs.get("pattern_type", "")
                context = kwargs.get("context", "")
                def _check():
                    # Check specific failure pattern types
                    patterns = {
                        "timeout": ["timeout", "timed out", "deadline", "slow", "hang"],
                        "encoding": ["encoding", "utf-8", "unicode", "ascii", "charset"],
                        "path": ["path", "file not found", "directory", "\\", "/"],
                        "network": ["connection", "refused", "timeout", "dns", "socket"],
                        "permission": ["permission", "denied", "access", "forbidden", "unauthorized"]
                    }

                    check_patterns = patterns.get(pattern_type, [])
                    context_lower = context.lower()
                    matches = [p for p in check_patterns if p in context_lower]
                    return {
                        "matches": matches,
                        "is_match": len(matches) > 0,
                        "pattern_type": pattern_type
                    }
                result = await loop.run_in_executor(_executor, _check)
                return {"result": result, "success": True}

            elif action == "preventive_actions":
                context = kwargs.get("context", "")
                def _preventive():
                    # Get interventions from causal model
                    interventions = []
                    try:
                        effects = self._causal_manager.find_effects(context, threshold=0.5)
                        for eff in effects[:5]:
                            int_list = self._causal_manager.get_interventions(eff.get("effect", ""))
                            interventions.extend(int_list[:2])
                    except:
                        pass

                    # Add generic safety suggestions based on context
                    context_lower = context.lower()
                    if "delete" in context_lower or "rm" in context_lower:
                        interventions.append("Consider making a backup first")
                        interventions.append("Use --dry-run flag if available")
                    if "git" in context_lower:
                        interventions.append("Create a backup branch before destructive operations")
                    if "database" in context_lower or "sql" in context_lower:
                        interventions.append("Test query on a backup/staging database first")

                    return list(set(interventions))[:5]  # Dedupe and limit

                actions = await loop.run_in_executor(_executor, _preventive)
                return {"actions": actions, "count": len(actions), "success": True}

            elif action == "verify":
                prediction_id = kwargs.get("prediction_id", "")
                outcome = kwargs.get("outcome", "")  # correct, incorrect, partial
                def _verify():
                    pred = self._predictor.get_prediction(prediction_id)
                    if pred:
                        pred.outcome = outcome
                        pred.verified_at = datetime.now().isoformat()
                        self._predictor.save_prediction(pred)
                        return pred.to_dict()
                    return None
                result = await loop.run_in_executor(_executor, _verify)
                if result:
                    return {"prediction": result, "success": True}
                return {"error": "Prediction not found", "success": False}

            else:
                return {"error": f"Unknown action: {action}", "success": False}

        except Exception as e:
            print(f"[MCP Bridge] Predict error: {e}")
            return {"error": str(e), "success": False}

    # =========================================================================
    # LEARNING API
    # =========================================================================

    async def learning(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Learning API.

        Actions:
            - find: Find relevant learnings for a task
            - record: Record a new learning
            - get_antipatterns: Get antipatterns for a context
        """
        await self._ensure_initialized()

        if not self._learning_extractor:
            return {"error": "Learning extractor not available", "success": False}

        loop = asyncio.get_event_loop()

        try:
            if action == "find":
                problem = kwargs.get("problem", "")
                limit = kwargs.get("limit", 5)
                def _find():
                    learnings_path = self.base_path / "learnings"
                    results = []
                    if learnings_path.exists():
                        for f in sorted(learnings_path.glob("*.json"),
                                       key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
                            try:
                                with open(f, 'r', encoding='utf-8') as file:
                                    data = json.load(file)
                                # Simple keyword match
                                if problem.lower() in json.dumps(data).lower():
                                    results.append(data)
                                    if len(results) >= limit:
                                        break
                            except:
                                pass
                    return results
                learnings = await loop.run_in_executor(_executor, _find)
                return {"learnings": learnings, "count": len(learnings), "success": True}

            elif action == "record":
                learning_type = kwargs.get("type", "solution")  # solution, failure, antipattern
                problem = kwargs.get("problem", "")
                solution = kwargs.get("solution", "")
                tags = kwargs.get("tags", [])
                def _record():
                    learnings_path = self.base_path / "learnings"
                    learnings_path.mkdir(parents=True, exist_ok=True)

                    learning_id = f"learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    learning = {
                        "id": learning_id,
                        "type": learning_type,
                        "problem": problem,
                        "solution": solution,
                        "tags": tags,
                        "created_at": datetime.now().isoformat()
                    }

                    with open(learnings_path / f"{learning_id}.json", 'w', encoding='utf-8') as f:
                        json.dump(learning, f, indent=2)

                    return learning
                learning = await loop.run_in_executor(_executor, _record)
                return {"learning": learning, "success": True}

            elif action == "get_antipatterns":
                context = kwargs.get("context", "")
                def _antipatterns():
                    learnings_path = self.base_path / "learnings"
                    antipatterns = []
                    if learnings_path.exists():
                        for f in learnings_path.glob("*.json"):
                            try:
                                with open(f, 'r', encoding='utf-8') as file:
                                    data = json.load(file)
                                if data.get("type") == "antipattern":
                                    if context.lower() in json.dumps(data).lower():
                                        antipatterns.append(data)
                            except:
                                pass
                    return antipatterns[:5]
                patterns = await loop.run_in_executor(_executor, _antipatterns)
                return {"antipatterns": patterns, "count": len(patterns), "success": True}

            else:
                return {"error": f"Unknown action: {action}", "success": False}

        except Exception as e:
            print(f"[MCP Bridge] Learning error: {e}")
            return {"error": str(e), "success": False}

    # =========================================================================
    # PROACTIVE CONTEXT
    # =========================================================================

    async def get_proactive_context(self, user_message: str) -> Dict[str, Any]:
        """
        Get comprehensive proactive context before responding to a user message.

        Returns warnings, relevant goals, applicable learnings, and suggestions.
        """
        await self._ensure_initialized()

        results = {
            "warnings": [],
            "relevant_goals": [],
            "applicable_learnings": [],
            "suggestions": [],
            "success": True
        }

        try:
            # Check for warnings (parallel execution)
            warnings_task = self.predict("anticipate_failures", context=user_message)
            goals_task = self.goals("find_relevant", context=user_message)
            learnings_task = self.learning("find", problem=user_message, limit=3)

            warnings_result, goals_result, learnings_result = await asyncio.gather(
                warnings_task, goals_task, learnings_task,
                return_exceptions=True
            )

            # Process warnings
            if isinstance(warnings_result, dict) and warnings_result.get("success"):
                results["warnings"] = warnings_result.get("warnings", [])

            # Process goals
            if isinstance(goals_result, dict) and goals_result.get("success"):
                results["relevant_goals"] = goals_result.get("goals", [])

            # Process learnings
            if isinstance(learnings_result, dict) and learnings_result.get("success"):
                results["applicable_learnings"] = learnings_result.get("learnings", [])

            # Generate suggestions based on context
            if results["warnings"]:
                results["suggestions"].append({
                    "type": "warning",
                    "message": f"{len(results['warnings'])} potential issue(s) detected"
                })

            if results["relevant_goals"]:
                results["suggestions"].append({
                    "type": "goal",
                    "message": f"Related to {len(results['relevant_goals'])} active goal(s)"
                })

        except Exception as e:
            print(f"[MCP Bridge] Proactive context error: {e}")
            results["error"] = str(e)

        return results

    # =========================================================================
    # CORRECTIONS API - For cognitive loop tool access
    # =========================================================================

    async def get_corrections(self, topic: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Get known corrections from AI Memory.

        Used by cognitive loop to avoid repeating past mistakes.
        """
        await self._ensure_initialized()

        loop = asyncio.get_event_loop()

        try:
            def _get_corrections():
                corrections = []

                # Read from quick_facts for fast access
                qf_path = self.base_path / "quick_facts.json"
                if qf_path.exists():
                    try:
                        with open(qf_path, 'r', encoding='utf-8') as f:
                            qf = json.load(f)
                            raw = qf.get("top_corrections", [])
                            # top_corrections may be a dict with "most_common" key
                            if isinstance(raw, dict):
                                corrections = raw.get("most_common", [])
                            elif isinstance(raw, list):
                                corrections = raw
                            else:
                                corrections = []
                    except:
                        pass

                # Also search learnings for antipatterns
                learnings_path = self.base_path / "learnings"
                if learnings_path.exists():
                    for f in sorted(learnings_path.glob("*.json"),
                                   key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
                        try:
                            with open(f, 'r', encoding='utf-8') as file:
                                data = json.load(file)
                                if data.get("type") in ["antipattern", "failure", "correction"]:
                                    # Filter by topic if specified
                                    if topic:
                                        if topic.lower() in json.dumps(data).lower():
                                            corrections.append({
                                                "type": data.get("type"),
                                                "problem": data.get("problem", ""),
                                                "what_not_to_do": data.get("what_not_to_do", ""),
                                                "why_it_failed": data.get("why_it_failed", ""),
                                                "source": "learnings"
                                            })
                                    else:
                                        corrections.append({
                                            "type": data.get("type"),
                                            "problem": data.get("problem", ""),
                                            "what_not_to_do": data.get("what_not_to_do", ""),
                                            "source": "learnings"
                                        })
                        except:
                            pass

                return corrections[:limit]

            corrections = await loop.run_in_executor(_executor, _get_corrections)
            return {"corrections": corrections, "count": len(corrections), "success": True}

        except Exception as e:
            print(f"[MCP Bridge] Corrections error: {e}")
            return {"corrections": [], "error": str(e), "success": False}

    # =========================================================================
    # USER PROFILE API - For cognitive loop tool access
    # =========================================================================

    async def get_user_profile(self, category: str = "all") -> Dict[str, Any]:
        """
        Get user profile from AI Memory.

        Used by cognitive loop to understand user preferences and context.
        """
        await self._ensure_initialized()

        loop = asyncio.get_event_loop()

        try:
            def _get_profile():
                profile = {
                    "identity": {
                        "name": os.environ.get("CEREBRO_USER_NAME", ""),
                        "github": os.environ.get("CEREBRO_GITHUB_USER", ""),
                        "docker": os.environ.get("CEREBRO_DOCKER_USER", "")
                    },
                    "preferences": {},
                    "goals": [],
                    "technical_environment": {}
                }

                # Read from quick_facts
                qf_path = self.base_path / "quick_facts.json"
                if qf_path.exists():
                    try:
                        with open(qf_path, 'r', encoding='utf-8') as f:
                            qf = json.load(f)
                            profile["preferences"] = qf.get("preferences", {})
                            profile["goals"] = qf.get("active_goals", [])
                            profile["technical_environment"] = {
                                "os": platform.system(),
                                "python": platform.python_version(),
                                "nas": os.environ.get("NAS_HOSTNAME", ""),
                                "gpu_server": os.environ.get("DGX_HOST", ""),
                                "ai_memory_path": str(self.base_path)
                            }
                    except:
                        pass

                # Filter by category if requested
                if category != "all" and category in profile:
                    return {category: profile[category]}

                return profile

            profile = await loop.run_in_executor(_executor, _get_profile)
            return {"profile": profile, "category": category, "success": True}

        except Exception as e:
            print(f"[MCP Bridge] User profile error: {e}")
            return {"profile": {}, "error": str(e), "success": False}

    # =========================================================================
    # MEMORY SAVE API
    # =========================================================================

    async def save_conversation(self, messages: list, session_id: str = None,
                                metadata: dict = None, incremental: bool = False) -> Dict[str, Any]:
        """Save a conversation to AI Memory."""
        await self._ensure_initialized()
        if not self._memory_service:
            return {"error": "Memory service not available", "success": False}

        loop = asyncio.get_event_loop()
        try:
            def _save():
                return self._memory_service.save_conversation(
                    messages=messages, session_id=session_id,
                    metadata=metadata, incremental=incremental
                )
            conv_id = await loop.run_in_executor(_executor, _save)
            return {"conversation_id": conv_id, "success": True}
        except Exception as e:
            print(f"[MCP Bridge] Save conversation error: {e}")
            return {"error": str(e), "success": False}

    async def analyze_and_save_learnings(self, messages: list,
                                          conversation_id: str = None) -> Dict[str, Any]:
        """Run LearningExtractor on messages and save structured learnings.

        This creates files in learnings/ that /api/learnings can search.
        The save_conversation pipeline writes to facts.jsonl, but /api/learnings
        only reads from learnings/ — this bridges that gap.
        """
        await self._ensure_initialized()
        if not self._learning_extractor:
            return {"error": "Learning extractor not available", "saved": False}

        loop = asyncio.get_event_loop()
        try:
            def _analyze():
                conversation = {
                    "id": conversation_id or "unknown",
                    "messages": messages,
                }
                extracted = self._learning_extractor.analyze_conversation(conversation)

                # Only save if we found something meaningful
                has_content = (
                    extracted.get("learnings")
                    or extracted.get("problems_found")
                    or extracted.get("solutions_found")
                )
                if has_content:
                    filepath = self._learning_extractor.save_learnings(extracted)
                    return {
                        "saved": True,
                        "filepath": filepath,
                        "count": len(extracted.get("learnings", [])),
                        "problems": len(extracted.get("problems_found", [])),
                        "solutions": len(extracted.get("solutions_found", [])),
                    }
                return {"saved": False, "count": 0, "reason": "No learnings detected"}

            result = await loop.run_in_executor(_executor, _analyze)
            return result
        except Exception as e:
            print(f"[MCP Bridge] Analyze learnings error: {e}")
            return {"error": str(e), "saved": False}

    # =========================================================================
    # MAINTENANCE API - Decay & Search Index
    # =========================================================================

    async def run_decay(self, force=False):
        """Run memory decay pipeline in thread pool."""
        loop = asyncio.get_event_loop()
        def _run():
            from decay_pipeline import run_decay as _run_decay
            return _run_decay(force=force)
        return await loop.run_in_executor(_executor, _run)

    async def rebuild_search_index(self):
        """Rebuild FTS5 keyword search index from memory data."""
        loop = asyncio.get_event_loop()
        def _rebuild():
            from keyword_index import get_keyword_index
            idx = get_keyword_index()
            count = idx.build_index_from_memory(self.base_path)
            return {"indexed": count, "total": idx.get_indexed_count()}
        return await loop.run_in_executor(_executor, _rebuild)

    async def get_decay_stats(self):
        """Get decay pipeline statistics."""
        loop = asyncio.get_event_loop()
        def _stats():
            from decay_pipeline import get_decay_stats as _get_stats
            return _get_stats()
        return await loop.run_in_executor(_executor, _stats)


# Singleton instance
_bridge_instance = None

def get_mcp_bridge() -> MCPBridge:
    """Get or create the MCP bridge singleton."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = MCPBridge()
    return _bridge_instance
