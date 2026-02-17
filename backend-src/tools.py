"""
Cerebro Tool Executor

Implements the same tools as Claude Code:
- Read files
- Write files
- Edit files
- Run bash/PowerShell commands
- Search files (glob/grep)
- Search AI Memory
"""

import os
import asyncio
import json
import glob as glob_module
import re
import shutil
from pathlib import Path
from typing import Optional, Any, AsyncGenerator
from dataclasses import dataclass

# Configuration
AI_MEMORY_PATH = os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))

# Safety: Paths that are always blocked
BLOCKED_PATHS = [
    r"C:\Windows\System32",
    r"C:\Windows\SysWOW64",
    "/etc/passwd",
    "/etc/shadow",
]

# Safety: Commands that are blocked
BLOCKED_COMMANDS = [
    "rm -rf /",
    "format c:",
    "del /f /s /q c:\\",
    ":(){:|:&};:",  # Fork bomb
]

@dataclass
class ToolResult:
    success: bool
    output: Any
    error: Optional[str] = None

# ============================================================================
# File Tools
# ============================================================================

async def read_file(path: str, offset: int = 0, limit: int = 2000) -> ToolResult:
    """Read a file's contents."""
    try:
        # Security check
        if any(blocked in path for blocked in BLOCKED_PATHS):
            return ToolResult(False, None, "Access to this path is blocked")

        path = Path(path)
        if not path.exists():
            return ToolResult(False, None, f"File not found: {path}")

        if not path.is_file():
            return ToolResult(False, None, f"Not a file: {path}")

        # Check file size
        size = path.stat().st_size
        if size > 10 * 1024 * 1024:  # 10MB limit
            return ToolResult(False, None, "File too large (>10MB)")

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Apply offset and limit
        lines = lines[offset:offset + limit]

        # Add line numbers
        numbered_lines = []
        for i, line in enumerate(lines, start=offset + 1):
            numbered_lines.append(f"{i:6d}\t{line.rstrip()}")

        return ToolResult(True, "\n".join(numbered_lines))

    except Exception as e:
        return ToolResult(False, None, str(e))

async def write_file(path: str, content: str) -> ToolResult:
    """Write content to a file."""
    try:
        if any(blocked in path for blocked in BLOCKED_PATHS):
            return ToolResult(False, None, "Access to this path is blocked")

        path = Path(path)

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        return ToolResult(True, f"Wrote {len(content)} bytes to {path}")

    except Exception as e:
        return ToolResult(False, None, str(e))

async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> ToolResult:
    """Edit a file by replacing text."""
    try:
        if any(blocked in path for blocked in BLOCKED_PATHS):
            return ToolResult(False, None, "Access to this path is blocked")

        path = Path(path)
        if not path.exists():
            return ToolResult(False, None, f"File not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        if old_string not in content:
            return ToolResult(False, None, "String not found in file")

        if replace_all:
            new_content = content.replace(old_string, new_string)
            count = content.count(old_string)
        else:
            new_content = content.replace(old_string, new_string, 1)
            count = 1

        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return ToolResult(True, f"Replaced {count} occurrence(s)")

    except Exception as e:
        return ToolResult(False, None, str(e))

# ============================================================================
# Command Execution
# ============================================================================

async def run_command(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 120
) -> ToolResult:
    """Run a shell command."""
    try:
        # Security check
        command_lower = command.lower()
        for blocked in BLOCKED_COMMANDS:
            if blocked.lower() in command_lower:
                return ToolResult(False, None, f"Command blocked for safety: {blocked}")

        # Determine shell based on OS
        if os.name == 'nt':
            # Use PowerShell for better compatibility
            full_command = f'powershell -Command "{command}"'
        else:
            full_command = command

        # Run command
        process = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            return ToolResult(False, None, f"Command timed out after {timeout}s")

        output = stdout.decode('utf-8', errors='replace')
        errors = stderr.decode('utf-8', errors='replace')

        if process.returncode != 0:
            return ToolResult(
                False,
                output,
                f"Exit code {process.returncode}: {errors}"
            )

        return ToolResult(True, output + errors if errors else output)

    except Exception as e:
        return ToolResult(False, None, str(e))

# ============================================================================
# Search Tools
# ============================================================================

async def glob_search(pattern: str, path: Optional[str] = None) -> ToolResult:
    """Find files matching a glob pattern."""
    try:
        base_path = Path(path) if path else Path.cwd()

        if not base_path.exists():
            return ToolResult(False, None, f"Path not found: {base_path}")

        # Use glob to find matches
        full_pattern = str(base_path / pattern)
        matches = glob_module.glob(full_pattern, recursive=True)

        # Limit results
        if len(matches) > 100:
            matches = matches[:100]
            truncated = True
        else:
            truncated = False

        result = {
            "matches": matches,
            "count": len(matches),
            "truncated": truncated
        }

        return ToolResult(True, result)

    except Exception as e:
        return ToolResult(False, None, str(e))

async def grep_search(
    pattern: str,
    path: Optional[str] = None,
    file_pattern: Optional[str] = None
) -> ToolResult:
    """Search file contents using regex."""
    try:
        base_path = Path(path) if path else Path.cwd()

        if not base_path.exists():
            return ToolResult(False, None, f"Path not found: {base_path}")

        regex = re.compile(pattern, re.IGNORECASE)
        matches = []

        # Find files to search
        if file_pattern:
            files = list(base_path.glob(file_pattern))
        elif base_path.is_file():
            files = [base_path]
        else:
            files = list(base_path.rglob("*"))

        # Search files
        for file in files[:1000]:  # Limit files searched
            if not file.is_file():
                continue

            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append({
                                "file": str(file),
                                "line": i,
                                "content": line.strip()[:200]
                            })

                            if len(matches) >= 100:
                                break

            except:
                continue

            if len(matches) >= 100:
                break

        return ToolResult(True, {
            "matches": matches,
            "count": len(matches),
            "truncated": len(matches) >= 100
        })

    except Exception as e:
        return ToolResult(False, None, str(e))

# ============================================================================
# AI Memory Tools
# ============================================================================

async def search_memory(query: str, limit: int = 10) -> ToolResult:
    """Search AI Memory for past conversations and learnings.

    Returns actual message content, not just file names, so the LLM can
    understand context from previous sessions.
    """
    try:
        memory_path = Path(AI_MEMORY_PATH)

        if not memory_path.exists():
            return ToolResult(False, None, "AI Memory path not found")

        results = []
        query_lower = query.lower()

        # Split query into keywords for better matching
        keywords = [w.strip() for w in query_lower.split() if len(w.strip()) > 2]

        # Search recent conversations (most recent first)
        conv_path = memory_path / "conversations"
        if conv_path.exists():
            conv_files = sorted(conv_path.glob("*.json"), reverse=True)[:30]  # Last 30 convos
            for conv_file in conv_files:
                try:
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        conv = json.load(f)

                    # Search through messages for matching content
                    messages = conv.get("messages", [])
                    matching_messages = []

                    for msg in messages:
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            content_lower = content.lower()
                            # Check if any keyword matches
                            if any(kw in content_lower for kw in keywords) or query_lower in content_lower:
                                # Found a match - extract relevant snippet
                                snippet = content[:500] + "..." if len(content) > 500 else content
                                matching_messages.append({
                                    "role": msg.get("role", "unknown"),
                                    "snippet": snippet
                                })

                    if matching_messages:
                        results.append({
                            "type": "conversation",
                            "date": conv_file.stem[:10] if len(conv_file.stem) > 10 else conv_file.stem,
                            "matching_messages": matching_messages[:5],  # Top 5 matches per convo
                            "total_matches": len(matching_messages)
                        })

                except Exception:
                    continue

        # Search quick facts (always include active_work if present)
        quick_facts_path = memory_path / "quick_facts.json"
        if quick_facts_path.exists():
            try:
                with open(quick_facts_path, 'r', encoding='utf-8') as f:
                    facts = json.load(f)

                # Always include active_work context
                if "active_work" in facts:
                    results.insert(0, {
                        "type": "active_work",
                        "content": facts["active_work"]
                    })

                # Also include last_session info
                if "last_session" in facts:
                    results.insert(0, {
                        "type": "last_session",
                        "content": facts["last_session"]
                    })

                # Check if query matches other facts
                facts_text = json.dumps(facts).lower()
                if query_lower in facts_text:
                    # Find which sections match
                    for key, value in facts.items():
                        if key.startswith("_") or key in ["active_work", "last_session"]:
                            continue
                        section_text = json.dumps(value).lower()
                        if query_lower in section_text:
                            results.append({
                                "type": f"quick_facts.{key}",
                                "content": value
                            })
            except:
                pass

        # Search learnings
        learnings_path = memory_path / "learnings"
        if learnings_path.exists():
            for learning_file in sorted(learnings_path.glob("*.json"), reverse=True)[:20]:
                try:
                    with open(learning_file, 'r', encoding='utf-8') as f:
                        learning = json.load(f)

                    learning_text = json.dumps(learning).lower()
                    if any(kw in learning_text for kw in keywords) or query_lower in learning_text:
                        results.append({
                            "type": "learning",
                            "content": learning
                        })

                except:
                    continue

        # Format output for the LLM
        if not results:
            return ToolResult(True, {
                "query": query,
                "results": [],
                "message": f"No matches found for '{query}'. The user might be referring to something in the current conversation context."
            })

        return ToolResult(True, {
            "query": query,
            "results": results[:limit],
            "count": len(results),
            "message": f"Found {len(results)} results. Use this context to answer the user's question."
        })

    except Exception as e:
        return ToolResult(False, None, str(e))

async def save_learning(
    problem: str,
    solution: str,
    tags: list[str] = None
) -> ToolResult:
    """Save a learning to AI Memory."""
    try:
        memory_path = Path(AI_MEMORY_PATH)
        learnings_path = memory_path / "learnings"
        learnings_path.mkdir(parents=True, exist_ok=True)

        learning = {
            "problem": problem,
            "solution": solution,
            "tags": tags or [],
            "created_at": datetime.now().isoformat()
        }

        # Generate filename from date and hash
        import hashlib
        hash_str = hashlib.md5(f"{problem}{solution}".encode()).hexdigest()[:8]
        filename = f"{datetime.now().strftime('%Y-%m-%d')}_{hash_str}.json"

        filepath = learnings_path / filename
        with open(filepath, 'w') as f:
            json.dump(learning, f, indent=2)

        return ToolResult(True, f"Saved learning to {filepath}")

    except Exception as e:
        return ToolResult(False, None, str(e))

# ============================================================================
# Claude Code Delegation
# ============================================================================

async def delegate_to_claude_stream(task: str, context: str = "") -> AsyncGenerator[str, None]:
    """Delegate a task to Claude Code CLI with real-time streaming."""
    try:
        # Find Claude CLI
        claude_path = shutil.which("claude")
        if not claude_path:
            claude_path = os.environ.get("CLAUDE_PATH", "claude")

        # Build the prompt
        full_prompt = task
        if context:
            full_prompt = f"Context: {context}\n\nTask: {task}"

        # Call Claude Code with stream-json output for real-time updates
        process = await asyncio.create_subprocess_exec(
            claude_path,
            "-p", full_prompt,
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--max-turns", "10",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(AI_MEMORY_PATH))
        )

        # Stream output line by line
        while True:
            try:
                line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=60  # 60s timeout per line
                )
            except asyncio.TimeoutError:
                yield "\n[Waiting for response...]"
                continue

            if not line:
                break

            try:
                data = json.loads(line.decode().strip())

                # Handle different message types
                if data.get("type") == "assistant":
                    # Claude is speaking - yield the text
                    content = data.get("message", {}).get("content", [])
                    for block in content:
                        if block.get("type") == "text":
                            yield block.get("text", "")

                elif data.get("type") == "tool_use":
                    # Claude is using a tool - show what it's doing
                    tool_name = data.get("name", "unknown")
                    yield f"\n[Using tool: {tool_name}...]\n"

                elif data.get("type") == "tool_result":
                    # Tool completed
                    yield "[Tool completed]\n"

                elif data.get("type") == "result":
                    # Final result
                    result_text = data.get("result", "")
                    if result_text:
                        yield result_text

            except json.JSONDecodeError:
                # Non-JSON line, yield as-is if not empty
                decoded = line.decode().strip()
                if decoded:
                    yield decoded

        await process.wait()

        if process.returncode != 0:
            stderr = await process.stderr.read()
            error_text = stderr.decode().strip()
            if error_text:
                yield f"\n[Error: {error_text}]"

    except FileNotFoundError:
        yield "[Error: Claude CLI not found. Make sure 'claude' is installed and in PATH]"
    except asyncio.TimeoutError:
        yield "\n[Task timed out]"
    except Exception as e:
        yield f"\n[Delegation failed: {str(e)}]"


async def delegate_to_claude(task: str, context: str = "") -> ToolResult:
    """Delegate a task to Claude Code CLI (non-streaming version)."""
    chunks = []
    async for chunk in delegate_to_claude_stream(task, context):
        chunks.append(chunk)

    full_output = "".join(chunks)

    if "[Error:" in full_output or "[Delegation failed:" in full_output:
        return ToolResult(success=False, output=None, error=full_output)

    return ToolResult(success=True, output=full_output, error=None)


# ============================================================================
# Tool Router
# ============================================================================

from datetime import datetime

TOOL_MAP = {
    # Primary tools
    "delegate_to_claude": delegate_to_claude,
    "search_memory": search_memory,  # Direct AI Memory search - critical for context continuity!

    # Legacy tools kept for potential fallback (uncomment if needed)
    # "read_file": read_file,
    # "write_file": write_file,
    # "edit_file": edit_file,
    # "run_command": run_command,
    # "glob_search": glob_search,
    # "grep_search": grep_search,
    # "save_learning": save_learning,
}

async def execute_tool(name: str, arguments: dict) -> ToolResult:
    """Execute a tool by name with given arguments."""
    if name not in TOOL_MAP:
        return ToolResult(False, None, f"Unknown tool: {name}")

    tool_func = TOOL_MAP[name]

    try:
        result = await tool_func(**arguments)
        return result
    except Exception as e:
        return ToolResult(False, None, str(e))
