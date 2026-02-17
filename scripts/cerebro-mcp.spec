# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Cerebro MCP Memory Server.

Builds a standalone binary for the Full Stack desktop edition.
Excludes large ML dependencies (torch, transformers, faiss, sentence_transformers)
to keep the bundle size manageable (~50-80 MB instead of ~2+ GB).
The bundled server operates in keyword-only search mode.
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Paths
# SPECPATH is set by PyInstaller to the directory containing the spec file (scripts/)
SCRIPTS_DIR = os.path.abspath(SPECPATH)
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
MEMORY_SRC_DIR = os.path.join(PROJECT_DIR, 'memory-src', 'src')

# Entry point
entry_point = os.path.join(SCRIPTS_DIR, 'cerebro-mcp-entry.py')

# ----- Data files: include all .py modules from memory-src/src/ -----
datas = []

# Walk the memory source tree and include all Python files
for root, dirs, files in os.walk(MEMORY_SRC_DIR):
    for f in files:
        if f.endswith('.py') or f.endswith('.json'):
            src_path = os.path.join(root, f)
            rel_dir = os.path.relpath(root, MEMORY_SRC_DIR)
            if rel_dir == '.':
                dest_dir = '.'
            else:
                dest_dir = rel_dir
            datas.append((src_path, dest_dir))

# Include config files
config_dir = os.path.join(MEMORY_SRC_DIR, 'config')
if os.path.exists(config_dir):
    for f in os.listdir(config_dir):
        datas.append((os.path.join(config_dir, f), 'config'))

# ----- Hidden imports -----
hiddenimports = [
    # MCP protocol
    'mcp',
    'mcp.server',
    'mcp.server.stdio',
    'mcp.types',
    *collect_submodules('mcp'),
    # Async
    'anyio',
    'anyio._backends',
    'anyio._backends._asyncio',
    *collect_submodules('anyio'),
    # Data / validation
    'numpy',
    'pydantic',
    'pydantic.deprecated',
    *collect_submodules('pydantic'),
    # Date handling
    'dateutil',
    'dateutil.parser',
    'dateutil.tz',
    *collect_submodules('dateutil'),
    # Stdlib used by memory modules
    'json',
    'uuid',
    'hashlib',
    'pathlib',
    'asyncio',
    'concurrent.futures',
    # HTTP (for optional remote embedding client)
    'httpx',
    # CLI (mcp[cli] dependency)
    'typer',
    'click',
    # Starlette/SSE deps that mcp might pull in
    'starlette',
    'sse_starlette',
    'httpx_sse',
]

# De-duplicate
hiddenimports = list(set(hiddenimports))

a = Analysis(
    [entry_point],
    pathex=[MEMORY_SRC_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Large ML packages — excluded to keep bundle small
        'torch',
        'torchvision',
        'torchaudio',
        'transformers',
        'sentence_transformers',
        'faiss',
        'faiss_cpu',
        'faiss_gpu',
        # Unused
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        'IPython',
        'notebook',
        'pytest',
        'ruff',
        'mypy',
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
    name='cerebro-mcp-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=(sys.platform != 'win32'),
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
    name='cerebro-mcp-server',
)
