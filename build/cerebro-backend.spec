# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Cerebro v6 native backend bundle.

Builds a single-file executable that bundles:
  - FastAPI + uvicorn + socketio
  - The full backend-src/ tree (main.py + routers + cognitive_loop + mcp_modules + mcp_bridge)
  - Anthropic SDK + claude-agent-sdk
  - faiss-cpu + sentence-transformers (memory system)
  - playwright (CDP browser automation)
  - aioredis client (optional Redis)

Output: dist/cerebro-backend (Linux) or dist/cerebro-backend.exe (Windows)

Build command:
    pyinstaller build/cerebro-backend.spec --clean --noconfirm

Resulting binary is placed in dist/ — Electron bundles it via electron-builder
extraResources into resources/backend/.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(SPECPATH)).parent
BACKEND_SRC = PROJECT_ROOT / 'backend-src'

# --- Source data: include all of backend-src tree ---
datas = [
    (str(BACKEND_SRC / 'cognitive_loop'), 'cognitive_loop'),
    (str(BACKEND_SRC / 'mcp_modules'), 'mcp_modules'),
]

# Include routers if present (v6 port adds these)
routers_dir = BACKEND_SRC / 'routers'
if routers_dir.exists():
    datas.append((str(routers_dir), 'routers'))

# Include core if present
core_dir = BACKEND_SRC / 'core'
if core_dir.exists():
    datas.append((str(core_dir), 'core'))

# Include data templates if present
data_dir = BACKEND_SRC / 'data'
if data_dir.exists():
    datas.append((str(data_dir), 'data'))

# --- Hidden imports: PyInstaller can miss these from dynamic imports ---
hiddenimports = [
    # FastAPI / uvicorn ecosystem
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.loops.uvloop',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.http.httptools_impl',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.protocols.websockets.websockets_impl',
    'uvicorn.protocols.websockets.wsproto_impl',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'h11',
    'httptools',
    'websockets',
    'wsproto',

    # Socket.IO
    'socketio',
    'engineio',
    'engineio.async_drivers',
    'engineio.async_drivers.asgi',
    'engineio.async_drivers.aiohttp',

    # Pydantic v2
    'pydantic',
    'pydantic.deprecated',
    'pydantic.deprecated.decorator',
    'pydantic_core',
    'pydantic_settings',

    # Auth
    'jwt',
    'bcrypt',

    # AI clients
    'anthropic',
    'claude_agent_sdk',
    'httpx',
    'requests',

    # Async helpers
    'aiofiles',
    'arq',

    # Browser automation
    'playwright',
    'playwright.async_api',
    'playwright._impl',

    # Memory system
    'numpy',
    'faiss',
    'sentence_transformers',
    'torch',
    'transformers',

    # Redis (optional)
    'redis',
    'redis.asyncio',

    # Standard lib edge cases
    'multipart',
    'multipart.multipart',
    'email.mime.multipart',
    'email.mime.text',
    'email.mime.base',
]

# --- Auto-discover all .py files in backend-src/ for hidden imports ---
def _discover_modules(root: Path, prefix: str = ''):
    found = []
    for entry in root.iterdir():
        if entry.name.startswith('__') or entry.name.startswith('.'):
            continue
        if entry.is_dir() and (entry / '__init__.py').exists():
            mod_prefix = f'{prefix}{entry.name}'
            found.append(mod_prefix)
            found.extend(_discover_modules(entry, f'{mod_prefix}.'))
        elif entry.is_file() and entry.suffix == '.py' and entry.name != '__init__.py':
            found.append(f'{prefix}{entry.stem}')
    return found

# Auto-discover from backend-src
for mod in _discover_modules(BACKEND_SRC):
    if mod not in hiddenimports:
        hiddenimports.append(mod)

# --- Excludes: keep binary lean ---
excludes = [
    # Dev tooling we don't ship
    'pytest',
    'pytest_asyncio',
    'IPython',
    'jupyter',
    'notebook',
    # Windows-only packages on Linux build (and vice versa)
    'win32api' if sys.platform != 'win32' else 'fcntl',
    # Heavy ML deps we don't use
    'tensorflow',
    'tensorflow_hub',
    'jax',
    # GUI toolkits — backend is headless
    'tkinter',
    'PyQt5',
    'PyQt6',
    'PySide2',
    'PySide6',
    'matplotlib',
]

block_cipher = None

a = Analysis(
    [str(BACKEND_SRC / 'main.py')],
    pathex=[str(BACKEND_SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='cerebro-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX often triggers AV false positives on Windows — leave off
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / 'assets' / ('icon.ico' if sys.platform == 'win32' else 'icon.png')),
)
