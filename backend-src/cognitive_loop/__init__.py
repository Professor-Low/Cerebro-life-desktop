"""
Cerebro Cognitive Loop - True Autonomous Intelligence

This module implements continuous autonomous thinking using local LLM (Qwen3-32B on DGX Spark).
Architecture: OODA + ReAct + Reflexion + Goal Pursuit + Adaptive Browser Learning

OBSERVE → ORIENT → DECIDE → ACT → REFLECT → LEARN
   ↑                                            │
   └────────────────────────────────────────────┘

Goal Pursuit System:
GOAL → MILESTONES → SUBTASKS → AGENTS → PROGRESS → LEARN

Adaptive Browser Learning:
EXPLORE → UNDERSTAND → ACT → RECORD → SKILL → SELF-HEAL
"""

from .ollama_client import OllamaClient
from .thought_journal import ThoughtJournal, Thought
from .ooda_engine import OODAEngine
from .safety_layer import SafetyLayer, RiskLevel
from .loop_manager import CognitiveLoopManager
from .reflexion_engine import ReflexionEngine, GoalReflexionEngine, get_goal_reflexion_engine
from .cognitive_tools import CognitiveTools, get_cognitive_tools
from .skill_loader import SkillLoader, AgentSkill, get_skill_loader

# Goal Pursuit System
from .goal_pursuit import (
    Goal, Milestone, Subtask, GoalProgress,
    GoalPursuitEngine, get_goal_pursuit_engine,
    GoalType, GoalStatus, MilestoneStatus, SubtaskStatus
)
from .progress_tracker import ProgressTracker, get_progress_tracker
from .goal_decomposer import GoalDecomposer, get_goal_decomposer

# Adaptive Browser Learning System
from .element_fingerprint import ElementFingerprint, SelfHealingLocator, FingerprintGenerator
from .page_understanding import PageUnderstanding, PageState, InteractableElement
from .adaptive_explorer import AdaptiveExplorer, ExplorationSession, ExplorationManager, get_exploration_manager
from .action_recorder import ActionRecorder, RecordingSession
from .skill_verifier import SkillVerifier, VerificationResult
from .recovery_recipes import RecoveryRecipes, BlockerType, RecoveryResult
from .browser_manager import BrowserManager, get_browser_manager

# Narration Engine
from .narration_engine import NarrationEngine, NarrationEvent, NarrationEventType

# Heartbeat Engine (replaces IdleThinker)
from .idle_thinker import HeartbeatEngine, get_heartbeat_engine, get_idle_thinker

__all__ = [
    # Core Components
    'OllamaClient',
    'ThoughtJournal',
    'Thought',
    'OODAEngine',
    'SafetyLayer',
    'RiskLevel',
    'CognitiveLoopManager',
    'ReflexionEngine',
    'GoalReflexionEngine',
    'get_goal_reflexion_engine',
    'CognitiveTools',
    'get_cognitive_tools',
    'SkillLoader',
    'AgentSkill',
    'get_skill_loader',

    # Goal Pursuit System
    'Goal',
    'Milestone',
    'Subtask',
    'GoalProgress',
    'GoalPursuitEngine',
    'get_goal_pursuit_engine',
    'GoalType',
    'GoalStatus',
    'MilestoneStatus',
    'SubtaskStatus',
    'ProgressTracker',
    'get_progress_tracker',
    'GoalDecomposer',
    'get_goal_decomposer',

    # Adaptive Browser Learning
    'ElementFingerprint',
    'SelfHealingLocator',
    'FingerprintGenerator',
    'PageUnderstanding',
    'PageState',
    'InteractableElement',
    'AdaptiveExplorer',
    'ExplorationSession',
    'ExplorationManager',
    'get_exploration_manager',
    'ActionRecorder',
    'RecordingSession',
    'SkillVerifier',
    'VerificationResult',
    'RecoveryRecipes',
    'BlockerType',
    'RecoveryResult',

    # Persistent Browser
    'BrowserManager',
    'get_browser_manager',

    # Narration Engine
    'NarrationEngine',
    'NarrationEvent',
    'NarrationEventType',

    # Idle Thinker
    'HeartbeatEngine',
    'get_heartbeat_engine',
    'get_idle_thinker',
]

__version__ = '1.3.0'  # Idle Thinking + Rich Chat + Question System
