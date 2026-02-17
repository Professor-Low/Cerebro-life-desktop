# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Cerebro backend server."""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Paths
SPEC_DIR = os.path.dirname(os.path.abspath(SPECPATH))
PROJECT_DIR = os.path.abspath(os.path.join(SPEC_DIR, '..'))
BACKEND_DIR = os.path.join(PROJECT_DIR, 'backend-src')

# Entry point (copied into backend dir before build)
entry_point = os.path.join(BACKEND_DIR, 'cerebro-entry.py')

# ----- Data files: include all .py modules from backend/ and cognitive_loop/ -----
backend_py_files = [
    'alpaca_client.py',
    'audit_logger.py',
    'autonomy_config.py',
    'git_manager.py',
    'health_monitor.py',
    'improvement_engine.py',
    'learning_injector.py',
    'main.py',
    'mcp_bridge.py',
    'predictive_interrupt.py',
    'proactive_agent.py',
    'rollback_engine.py',
    'safe_restart.py',
    'self_modification.py',
    'sim_engine_client.py',
    'staging_manager.py',
    'tools.py',
]

cognitive_loop_files = [
    'cognitive_loop/__init__.py',
    'cognitive_loop/action_recorder.py',
    'cognitive_loop/adaptive_explorer.py',
    'cognitive_loop/browser_manager.py',
    'cognitive_loop/cognitive_tools.py',
    'cognitive_loop/element_fingerprint.py',
    'cognitive_loop/goal_decomposer.py',
    'cognitive_loop/goal_pursuit.py',
    'cognitive_loop/idle_thinker.py',
    'cognitive_loop/loop_manager.py',
    'cognitive_loop/narration_engine.py',
    'cognitive_loop/ollama_client.py',
    'cognitive_loop/ooda_engine.py',
    'cognitive_loop/page_understanding.py',
    'cognitive_loop/progress_tracker.py',
    'cognitive_loop/recovery_recipes.py',
    'cognitive_loop/reflexion_engine.py',
    'cognitive_loop/safety_layer.py',
    'cognitive_loop/skill_executor.py',
    'cognitive_loop/skill_generator.py',
    'cognitive_loop/skill_loader.py',
    'cognitive_loop/skill_verifier.py',
    'cognitive_loop/thought_journal.py',
]

datas = []
for f in backend_py_files:
    datas.append((os.path.join(BACKEND_DIR, f), '.'))
for f in cognitive_loop_files:
    datas.append((os.path.join(BACKEND_DIR, f), 'cognitive_loop'))

# ----- Hidden imports -----
hiddenimports = [
    # Socket.IO / Engine.IO
    'socketio',
    'engineio',
    'engineio.async_drivers.aiohttp',
    # Redis
    'redis',
    'redis.asyncio',
    'redis.asyncio.client',
    'redis.asyncio.connection',
    # Auth
    'jwt',
    # HTTP
    'httpx',
    'aiohttp',
    # AI
    'anthropic',
    # Web framework
    'fastapi',
    'fastapi.middleware',
    'fastapi.middleware.cors',
    'starlette',
    'starlette.middleware',
    'starlette.routing',
    'starlette.responses',
    'uvicorn',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    # Validation / utilities
    'pydantic',
    'pydantic_settings',
    'python_multipart',
    'multipart',
    'multipart.multipart',
    # Env
    'dotenv',
    # Task queue
    'arq',
    # Collect all submodules for key packages
    *collect_submodules('socketio'),
    *collect_submodules('engineio'),
    *collect_submodules('pydantic'),
]

# De-duplicate
hiddenimports = list(set(hiddenimports))

a = Analysis(
    [entry_point],
    pathex=[BACKEND_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'PIL',
        'IPython',
        'notebook',
        'pytest',
    ],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='cerebro-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=(sys.platform != 'win32'),  # No console on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='cerebro-server',
)
