# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Cerebro backend server."""

import os
import sys
import glob
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Paths
# SPECPATH is set by PyInstaller to the directory containing the spec file (scripts/)
SCRIPTS_DIR = os.path.abspath(SPECPATH)
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
BACKEND_DIR = os.path.join(PROJECT_DIR, 'backend-src')

# Entry point (copied into backend dir before build)
entry_point = os.path.join(BACKEND_DIR, 'cerebro-entry.py')

# ----- Data files: auto-discover all .py modules from backend/ -----
# Instead of a hardcoded list, glob all .py files so new modules are always included.

datas = []

# 1. All top-level .py files in backend-src/
for py_file in glob.glob(os.path.join(BACKEND_DIR, '*.py')):
    datas.append((py_file, '.'))

# 2. cognitive_loop/ package
cognitive_dir = os.path.join(BACKEND_DIR, 'cognitive_loop')
if os.path.isdir(cognitive_dir):
    for py_file in glob.glob(os.path.join(cognitive_dir, '*.py')):
        datas.append((py_file, 'cognitive_loop'))

# 3. routers/ package (Phase 1 modular refactor)
routers_dir = os.path.join(BACKEND_DIR, 'routers')
if os.path.isdir(routers_dir):
    for py_file in glob.glob(os.path.join(routers_dir, '*.py')):
        datas.append((py_file, 'routers'))

# 4. core/ package (shared config, auth, state)
core_dir = os.path.join(BACKEND_DIR, 'core')
if os.path.isdir(core_dir):
    for py_file in glob.glob(os.path.join(core_dir, '*.py')):
        datas.append((py_file, 'core'))

# 5. mcp_modules/ package (memory/AI modules)
mcp_dir = os.path.join(BACKEND_DIR, 'mcp_modules')
if os.path.isdir(mcp_dir):
    for py_file in glob.glob(os.path.join(mcp_dir, '*.py')):
        datas.append((py_file, 'mcp_modules'))

print(f"[cerebro-spec] Collected {len(datas)} data files from backend-src/")

# ----- Hidden imports -----
hiddenimports = [
    # System monitoring
    'psutil',
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
    # Bcrypt (used by auth)
    'bcrypt',
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
