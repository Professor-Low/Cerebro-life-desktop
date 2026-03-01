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
_executor = ThreadPoolExecutor(max_workers=8)


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
        self._solution_tracker = None
        self._embeddings_engine = None
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
                from solution_tracker import SolutionTracker

                try:
                    from ai_embeddings_engine import EmbeddingsEngine
                except ImportError:
                    EmbeddingsEngine = None

                bp = str(self.base_path)
                return {
                    'goal_tracker': GoalTracker(bp),
                    'causal_manager': CausalModelManager(bp),
                    'predictor': Predictor(bp),
                    'learning_extractor': LearningExtractor(bp),
                    'memory_service': UltimateMemoryService(bp),
                    'solution_tracker': SolutionTracker(bp),
                    'embeddings_engine': EmbeddingsEngine(bp) if EmbeddingsEngine else None
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
            self._solution_tracker = result.get('solution_tracker')
            self._embeddings_engine = result.get('embeddings_engine')
            self._initialized = True
            print("[MCP Bridge] Initialized successfully")

            # Build FAISS index in background if embeddings engine is ready
            if self._embeddings_engine:
                asyncio.get_event_loop().run_in_executor(
                    _executor, self._build_faiss_index_from_memory
                )
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
                    results = []
                    seen_ids = set()

                    # Tier 0: Semantic search from EmbeddingsEngine
                    if self._embeddings_engine:
                        try:
                            engine_results = self._embeddings_engine.semantic_search(problem, top_k=limit * 2)
                            for r in engine_results:
                                chunk_id = r.get("chunk_id", r.get("conversation_id", ""))
                                if chunk_id in seen_ids:
                                    continue
                                seen_ids.add(chunk_id)
                                entry = {
                                    "id": chunk_id,
                                    "problem": r.get("content", "")[:200],
                                    "solution": r.get("content", ""),
                                    "type": r.get("metadata", {}).get("type", "learning") if isinstance(r.get("metadata"), dict) else "learning",
                                    "score": r.get("similarity", r.get("score", 0)),
                                    "source": "semantic"
                                }
                                # Boost learning-type results
                                if r.get("chunk_type") == "learning":
                                    entry["score"] = min(1.0, entry["score"] * 1.5)
                                results.append(entry)
                        except Exception as e:
                            print(f"[MCP Bridge] Semantic search failed in learning find: {e}")

                    # Tier 1: FTS5 keyword search
                    try:
                        from keyword_index import get_keyword_index
                        idx = get_keyword_index()
                        fts_results = idx.search(problem, top_k=limit * 3)
                        for r in fts_results:
                            chunk_id = r.get("chunk_id", "")
                            if chunk_id in seen_ids:
                                continue
                            seen_ids.add(chunk_id)
                            # Boost learning-type results
                            entry = {
                                "id": chunk_id,
                                "problem": r.get("content", "")[:200],
                                "solution": r.get("content", ""),
                                "type": r.get("metadata", {}).get("type", "learning"),
                                "score": r.get("similarity", 0),
                                "source": "fts5"
                            }
                            if r.get("chunk_type") == "learning":
                                entry["score"] = min(1.0, entry["score"] * 1.5)
                            results.append(entry)
                    except Exception as e:
                        print(f"[MCP Bridge] FTS5 search failed: {e}")

                    # Tier 2: SolutionTracker versioned solutions
                    if self._solution_tracker:
                        try:
                            solutions = self._solution_tracker.find_solution(problem)
                            for s in solutions:
                                sid = s.get("id", "")
                                if sid in seen_ids:
                                    continue
                                seen_ids.add(sid)
                                confidence = s.get("success_confirmations", 0) - s.get("failure_count", 0)
                                results.append({
                                    "id": sid,
                                    "problem": s.get("problem", ""),
                                    "solution": s.get("solution", ""),
                                    "type": "solution",
                                    "status": s.get("status", "active"),
                                    "version": s.get("current_version", 1),
                                    "confidence": confidence,
                                    "score": min(1.0, 0.5 + confidence * 0.1),
                                    "source": "solution_tracker"
                                })
                        except Exception as e:
                            print(f"[MCP Bridge] SolutionTracker search failed: {e}")

                    # Tier 3: Fallback - old substring scan (only if tiers 1+2 empty)
                    if not results:
                        learnings_path = self.base_path / "learnings"
                        if learnings_path.exists():
                            for f in sorted(learnings_path.glob("*.json"),
                                           key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
                                try:
                                    with open(f, 'r', encoding='utf-8') as file:
                                        data = json.load(file)
                                    if problem.lower() in json.dumps(data).lower():
                                        data["source"] = "file_scan"
                                        results.append(data)
                                except:
                                    pass

                    # Sort by score descending, return up to limit
                    results.sort(key=lambda x: x.get("score", 0), reverse=True)
                    return results[:limit]

                learnings = await loop.run_in_executor(_executor, _find)
                return {"learnings": learnings, "count": len(learnings), "success": True}

            elif action == "record":
                learning_type = kwargs.get("type", "solution")  # solution, failure, antipattern
                problem = kwargs.get("problem", "")
                solution = kwargs.get("solution", "")
                tags = kwargs.get("tags", [])
                what_not_to_do = kwargs.get("what_not_to_do", "")
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
                    if what_not_to_do:
                        learning["what_not_to_do"] = what_not_to_do

                    learning_file = learnings_path / f"{learning_id}.json"
                    with open(learning_file, 'w', encoding='utf-8') as f:
                        json.dump(learning, f, indent=2)

                    # Also record in SolutionTracker for versioning + confidence
                    tracker_result = None
                    if self._solution_tracker:
                        try:
                            if learning_type == "solution" and solution:
                                tracker_result = self._solution_tracker.record_solution(
                                    problem=problem, solution=solution, tags=tags
                                )
                            elif learning_type == "antipattern" and what_not_to_do:
                                tracker_result = self._solution_tracker.record_antipattern(
                                    what_not_to_do=what_not_to_do,
                                    why_it_failed=solution or problem,
                                    original_problem=problem,
                                    tags=tags
                                )
                        except Exception as e:
                            print(f"[MCP Bridge] SolutionTracker record failed: {e}")

                    # Incremental index for immediate searchability
                    try:
                        from keyword_index import get_keyword_index
                        idx = get_keyword_index()
                        idx.index_single_learning(learning_file)
                    except Exception as e:
                        print(f"[MCP Bridge] Incremental index failed: {e}")

                    if tracker_result:
                        learning["tracker_id"] = tracker_result.get("id")
                    return learning

                learning = await loop.run_in_executor(_executor, _record)
                return {"learning": learning, "success": True}

            elif action == "get_antipatterns":
                context = kwargs.get("context", "")
                tags = kwargs.get("tags", [])
                def _antipatterns():
                    antipatterns = []

                    # Primary: SolutionTracker word-similarity matching
                    if self._solution_tracker:
                        try:
                            antipatterns = self._solution_tracker.find_antipatterns(
                                problem=context, tags=tags
                            )
                        except Exception as e:
                            print(f"[MCP Bridge] SolutionTracker antipattern search failed: {e}")

                    # Fallback: old file scan (only if SolutionTracker returned nothing)
                    if not antipatterns:
                        learnings_path = self.base_path / "learnings"
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

            elif action == "record_failure":
                if not self._solution_tracker:
                    return {"error": "SolutionTracker not available", "success": False}
                solution_id = kwargs.get("solution_id", "")
                description = kwargs.get("description", "")
                error_message = kwargs.get("error_message", "")
                def _record_failure():
                    return self._solution_tracker.record_failure(
                        solution_id=solution_id,
                        failure_description=description,
                        error_message=error_message
                    )
                result = await loop.run_in_executor(_executor, _record_failure)
                return {**result, "success": True}

            elif action == "confirm_solution":
                if not self._solution_tracker:
                    return {"error": "SolutionTracker not available", "success": False}
                solution_id = kwargs.get("solution_id", "")
                def _confirm():
                    return self._solution_tracker.confirm_solution_works(solution_id)
                result = await loop.run_in_executor(_executor, _confirm)
                if result.get("error"):
                    return {"error": result["error"], "success": False}
                return {**result, "success": True}

            elif action == "solution_chain":
                if not self._solution_tracker:
                    return {"error": "SolutionTracker not available", "success": False}
                solution_id = kwargs.get("solution_id", "")
                def _chain():
                    return self._solution_tracker.get_solution_chain(solution_id)
                result = await loop.run_in_executor(_executor, _chain)
                return {**result, "success": True}

            elif action == "summary":
                if not self._solution_tracker:
                    return {"error": "SolutionTracker not available", "success": False}
                def _summary():
                    return self._solution_tracker.get_learnings_summary()
                result = await loop.run_in_executor(_executor, _summary)
                return {**result, "success": True}

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

    # =========================================================================
    # EMBEDDINGS INDEX BUILDER
    # =========================================================================

    def _build_faiss_index_from_memory(self):
        """Build FAISS index from existing memory data (learnings, facts, conversations).

        Runs in background thread on startup. Reads all text content,
        embeds it via the engine (DGX or local model), and builds the FAISS index.
        """
        import time as _time
        start = _time.monotonic()
        engine = self._embeddings_engine
        if not engine:
            return

        # Check if index already exists and has vectors
        try:
            engine.build_faiss_index(rebuild=False)
            if engine.index is not None and engine.index.ntotal > 0:
                print(f"[Startup] FAISS index already loaded: {engine.index.ntotal} vectors")
                return
        except Exception:
            pass

        # Check if embedding model is available (local or DGX)
        can_embed = False
        try:
            from dgx_embedding_client import is_dgx_embedding_available_sync
            if is_dgx_embedding_available_sync():
                can_embed = True
        except (ImportError, Exception):
            pass
        if not can_embed and engine.model is not None:
            can_embed = True
        if not can_embed:
            print("[Startup] No embedding model available — FAISS index build skipped")
            return

        print("[Startup] Building FAISS index from memory data...")
        bp = self.base_path
        chunks = []
        chunk_id_counter = 0

        def make_chunk(content, chunk_type, conv_id, metadata=None):
            nonlocal chunk_id_counter
            chunk_id_counter += 1
            return {
                "content": content[:2000],  # Limit text length
                "chunk_type": chunk_type,
                "chunk_id": f"auto_{chunk_id_counter:06d}",
                "conversation_id": conv_id,
                "metadata": metadata or {},
            }

        # 1. Learnings
        learnings_path = bp / "learnings"
        if learnings_path.exists():
            for f in sorted(learnings_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:100]:
                try:
                    with open(f, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                    text_parts = []
                    if data.get("problem"):
                        text_parts.append(f"Problem: {data['problem']}")
                    if data.get("solution"):
                        text_parts.append(f"Solution: {data['solution']}")
                    if data.get("what_not_to_do"):
                        text_parts.append(f"Antipattern: {data['what_not_to_do']}")
                    for learn in data.get("learnings", [])[:3]:
                        if isinstance(learn, dict):
                            text_parts.append(learn.get("content", learn.get("problem", "")))
                    if text_parts:
                        chunks.append(make_chunk(
                            " | ".join(text_parts), "learning", f.stem,
                            {"type": data.get("type", "learning"), "tags": data.get("tags", [])}
                        ))
                except Exception:
                    pass

        # 2. Facts from JSONL
        for facts_path in [bp / "embeddings" / "chunks" / "facts.jsonl", bp / "facts" / "facts.jsonl"]:
            if facts_path.exists():
                try:
                    with open(facts_path, 'r', encoding='utf-8') as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                                text = entry.get("content", entry.get("fact", entry.get("learning", "")))
                                if text and len(text) > 20:
                                    chunks.append(make_chunk(
                                        text, entry.get("chunk_type", "fact"),
                                        entry.get("conversation_id", "facts"),
                                        entry.get("metadata", {})
                                    ))
                            except json.JSONDecodeError:
                                pass
                except Exception:
                    pass
                break  # Only use first facts file found

        # 3. Quick facts
        qf_path = bp / "quick_facts.json"
        if qf_path.exists():
            try:
                with open(qf_path, 'r', encoding='utf-8') as fh:
                    qf = json.load(fh)
                for key, val in qf.items():
                    if isinstance(val, str) and len(val) > 20:
                        chunks.append(make_chunk(f"{key}: {val}", "quick_fact", "quick_facts"))
                    elif isinstance(val, list):
                        for item in val[:10]:
                            text = str(item) if not isinstance(item, dict) else json.dumps(item)
                            if len(text) > 20:
                                chunks.append(make_chunk(f"{key}: {text}", "quick_fact", "quick_facts"))
            except Exception:
                pass

        if not chunks:
            print("[Startup] No memory content found to index")
            return

        print(f"[Startup] Embedding {len(chunks)} chunks...")

        try:
            import numpy as np
            import faiss

            # Embed all chunks
            texts = [c["content"] for c in chunks]
            embeddings = engine.embed_batch(texts, batch_size=32)

            if embeddings is None or len(embeddings) == 0:
                print("[Startup] Embedding failed — no vectors produced")
                return

            # Check for zero vectors (means embedding failed)
            nonzero_mask = np.any(embeddings != 0, axis=1)
            if not np.any(nonzero_mask):
                print("[Startup] All embeddings are zero — model not producing output")
                return

            # Build FAISS index
            embeddings = embeddings.astype(np.float32)
            if not embeddings.flags['C_CONTIGUOUS']:
                embeddings = np.ascontiguousarray(embeddings)
            faiss.normalize_L2(embeddings)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

            # Build id_mapping
            id_mapping = []
            for c in chunks:
                id_mapping.append({
                    "conversation_id": c["conversation_id"],
                    "chunk_id": c["chunk_id"],
                    "chunk_type": c["chunk_type"],
                    "metadata": c["metadata"],
                    "content": c["content"],  # Cache content for search results
                })

            # Set on engine
            engine.index = index
            engine.id_mapping = id_mapping

            # Save to disk for persistence
            engine._ensure_directories()
            index_file = engine.index_path / "faiss_index.bin"
            mapping_file = engine.index_path / "id_mapping.json"
            faiss.write_index(index, str(index_file))
            with open(mapping_file, 'w', encoding='utf-8') as fh:
                json.dump(id_mapping, fh)

            elapsed = _time.monotonic() - start
            print(f"[Startup] FAISS index built: {index.ntotal} vectors, {dim}d, {elapsed:.1f}s")
        except Exception as e:
            print(f"[Startup] FAISS index build failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # SEARCH API - Hybrid semantic + keyword search
    # =========================================================================

    async def search_memory(self, query: str, limit: int = 10, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Search memory using hybrid semantic + keyword approach.

        Modes:
            - hybrid: Semantic + FTS5 merged with deduplication (default)
            - semantic: Semantic search only (requires EmbeddingsEngine)
            - keyword: FTS5 keyword search only
        """
        await self._ensure_initialized()
        loop = asyncio.get_event_loop()
        results = []
        seen_ids = set()

        # Semantic component (from EmbeddingsEngine)
        if mode in ("hybrid", "semantic") and self._embeddings_engine:
            try:
                def _semantic():
                    return self._embeddings_engine.semantic_search(query, top_k=limit * 2)
                sem_results = await asyncio.wait_for(
                    loop.run_in_executor(_executor, _semantic), timeout=10
                )
                for r in sem_results:
                    cid = r.get("chunk_id", r.get("conversation_id", ""))
                    if cid and cid in seen_ids:
                        continue
                    if cid:
                        seen_ids.add(cid)
                    results.append({
                        "type": r.get("chunk_type", "fact"),
                        "id": cid,
                        "title": r.get("content", "")[:100],
                        "snippet": r.get("content", "")[:300],
                        "score": r.get("similarity", r.get("score", 0)),
                        "source": "semantic",
                    })
            except asyncio.TimeoutError:
                print("[MCP Bridge] Semantic search timed out (>10s)")
            except Exception as e:
                print(f"[MCP Bridge] Semantic search failed: {e}")

        # Keyword component (from FTS5 keyword_index)
        if mode in ("hybrid", "keyword"):
            try:
                def _fts5():
                    from keyword_index import get_keyword_index
                    idx = get_keyword_index()
                    if idx.get_indexed_count() > 0:
                        return idx.search(query, top_k=limit * 2)
                    return []
                fts_results = await loop.run_in_executor(_executor, _fts5)
                for r in fts_results:
                    cid = r.get("chunk_id", "")
                    if cid and cid in seen_ids:
                        continue
                    if cid:
                        seen_ids.add(cid)
                    results.append({
                        "type": r.get("chunk_type", "fact"),
                        "id": cid,
                        "title": r.get("content", "")[:100],
                        "snippet": r.get("content", "")[:300],
                        "score": r.get("similarity", 0),
                        "source": "fts5",
                    })
            except Exception as e:
                print(f"[MCP Bridge] FTS5 search failed: {e}")

        # Sort by score descending, deduplicated
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        trimmed = results[:limit]
        source = "hybrid" if any(r["source"] == "semantic" for r in trimmed) and any(r["source"] == "fts5" for r in trimmed) else (
            "semantic" if any(r["source"] == "semantic" for r in trimmed) else (
                "fts5" if trimmed else "none"
            )
        )
        return {"query": query, "results": trimmed, "count": len(trimmed), "source": source}

    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine status for diagnostics."""
        await self._ensure_initialized()
        loop = asyncio.get_event_loop()

        stats = {
            "engine_available": self._embeddings_engine is not None,
            "fts5_count": 0,
            "faiss_index_loaded": False,
            "faiss_vectors": 0,
            "model_loaded": False,
            "dgx_available": False,
            "search_mode": "keyword_only",
        }

        # FTS5 stats
        try:
            def _fts5_stats():
                from keyword_index import get_keyword_index
                idx = get_keyword_index()
                return idx.get_indexed_count()
            stats["fts5_count"] = await loop.run_in_executor(_executor, _fts5_stats)
        except Exception:
            pass

        # Embeddings engine stats
        if self._embeddings_engine:
            try:
                def _engine_stats():
                    engine = self._embeddings_engine
                    info = {}
                    # Check FAISS index
                    if hasattr(engine, 'index') and engine.index is not None:
                        info["faiss_index_loaded"] = True
                        info["faiss_vectors"] = engine.index.ntotal if hasattr(engine.index, 'ntotal') else 0
                    # Check model (lazy-loaded via @property, check _model directly)
                    if hasattr(engine, '_model') and engine._model is not None:
                        info["model_loaded"] = True
                    elif hasattr(engine, 'model') and engine.model is not None:
                        info["model_loaded"] = True
                    # Check DGX client
                    try:
                        from dgx_embedding_client import is_dgx_embedding_available_sync
                        info["dgx_available"] = is_dgx_embedding_available_sync()
                    except (ImportError, Exception):
                        pass
                    return info
                engine_info = await loop.run_in_executor(_executor, _engine_stats)
                stats.update(engine_info)

                if stats["model_loaded"] or stats["dgx_available"]:
                    stats["search_mode"] = "hybrid" if stats["fts5_count"] > 0 else "semantic_only"
                elif stats["faiss_index_loaded"]:
                    stats["search_mode"] = "hybrid" if stats["fts5_count"] > 0 else "semantic_only"
            except Exception:
                pass

        if not stats["engine_available"] and stats["fts5_count"] > 0:
            stats["search_mode"] = "keyword_only"

        return stats


# Singleton instance
_bridge_instance = None

def get_mcp_bridge() -> MCPBridge:
    """Get or create the MCP bridge singleton."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = MCPBridge()
    return _bridge_instance
