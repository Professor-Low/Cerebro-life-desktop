"""
Heartbeat Engine - Periodic monitor system for idle mode.

Replaces the old IdleThinker with cheap, structured monitors that run
on a configurable interval. No LLM calls. Detects real changes via
hash-based deduplication and pushes findings to the Stored tab.

Monitors:
  - git_repos: Check watched repos for uncommitted changes
  - file_changes: Scan AI_MEMORY for recently modified files
  - memory_health: Validate quick_facts, conversation count, FAISS index
  - system_health: NAS mount, Cerebro port, DGX Spark reachability
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

AI_MEMORY_PATH = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data")))
HEARTBEAT_CONFIG_FILE = AI_MEMORY_PATH / "cerebro" / "heartbeat_config.json"
STORED_ITEMS_FILE = AI_MEMORY_PATH / "cerebro" / "stored_items.json"
HEARTBEAT_MD_FILE = AI_MEMORY_PATH / "cerebro" / "heartbeat.md"

DEFAULT_HEARTBEAT_MD = """# Cerebro Heartbeat Configuration

## Settings

- **Interval:** 15 minutes
- **Monitors:** git_repos, file_changes, memory_health, system_health

## Focus Areas

Topics Cerebro should pay attention to when idle. Injected as context into idle-spawned agents.

-

## Idle Tasks

Background tasks for dormant mode. Use `- [ ]` for pending, `- [x]` for done.
One task processed per heartbeat cycle (only when Awake).

- [ ]

## Custom Monitors

Additional checks alongside built-in monitors. Format: `**name**: \\`command\\`` -- description

## Dormant Instructions

Freeform instructions for idle behavior. Appended to agent context during dormant mode.

""".lstrip()

DEFAULT_HEARTBEAT_CONFIG = {
    "interval_minutes": 15,
    "monitors": {
        "git_repos": {"enabled": True},
        "file_changes": {"enabled": True},
        "memory_health": {"enabled": True},
        "system_health": {"enabled": True},
        "screen_monitor": {"enabled": False},
    },
}

MONITOR_DISPLAY_NAMES = {
    "git_repos": "Git Repos",
    "file_changes": "File Changes",
    "memory_health": "Memory Health",
    "system_health": "System Health",
    "screen_monitor": "Screen Monitor",
}


# ---------------------------------------------------------------------------
# Config persistence (shared with main.py via import)
# ---------------------------------------------------------------------------

def load_heartbeat_config() -> dict:
    """Load heartbeat config from disk, returning defaults if missing."""
    if HEARTBEAT_CONFIG_FILE.exists():
        try:
            data = json.loads(HEARTBEAT_CONFIG_FILE.read_text(encoding="utf-8"))
            # Merge with defaults so new monitors are always present
            merged = {**DEFAULT_HEARTBEAT_CONFIG, **data}
            merged["monitors"] = {**DEFAULT_HEARTBEAT_CONFIG["monitors"], **data.get("monitors", {})}
            return merged
        except Exception:
            pass
    return dict(DEFAULT_HEARTBEAT_CONFIG)


def save_heartbeat_config(config: dict):
    """Write heartbeat config to disk."""
    HEARTBEAT_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Heartbeat Markdown persistence
# ---------------------------------------------------------------------------

def load_heartbeat_md() -> str:
    """Read heartbeat.md from disk, creating default if missing."""
    HEARTBEAT_MD_FILE.parent.mkdir(parents=True, exist_ok=True)
    if HEARTBEAT_MD_FILE.exists():
        try:
            return HEARTBEAT_MD_FILE.read_text(encoding="utf-8")
        except Exception:
            pass
    # Create default
    HEARTBEAT_MD_FILE.write_text(DEFAULT_HEARTBEAT_MD, encoding="utf-8")
    return DEFAULT_HEARTBEAT_MD


def save_heartbeat_md(content: str):
    """Write heartbeat.md to disk."""
    HEARTBEAT_MD_FILE.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_MD_FILE.write_text(content, encoding="utf-8")


def parse_heartbeat_md(content: str) -> dict:
    """Parse heartbeat.md sections into a structured dict."""
    result = {
        "interval_minutes": 15,
        "monitors_enabled": [],
        "focus_areas": [],
        "idle_tasks": [],
        "custom_monitors": [],
        "dormant_instructions": "",
    }

    # Split into sections by ## headings
    sections: Dict[str, str] = {}
    current_section = ""
    current_lines: List[str] = []
    for line in content.splitlines():
        if line.startswith("## "):
            if current_section:
                sections[current_section] = "\n".join(current_lines)
            current_section = line[3:].strip().lower()
            current_lines = []
        else:
            current_lines.append(line)
    if current_section:
        sections[current_section] = "\n".join(current_lines)

    # --- Settings ---
    settings_text = sections.get("settings", "")
    interval_match = re.search(r"\*\*Interval:\*\*\s*(\d+)", settings_text)
    if interval_match:
        result["interval_minutes"] = max(5, min(60, int(interval_match.group(1))))
    monitors_match = re.search(r"\*\*Monitors:\*\*\s*(.+)", settings_text)
    if monitors_match:
        result["monitors_enabled"] = [m.strip() for m in monitors_match.group(1).split(",") if m.strip()]

    # --- Focus Areas ---
    for line in sections.get("focus areas", "").splitlines():
        line = line.strip()
        if line.startswith("- ") and len(line) > 2:
            item = line[2:].strip()
            if item:
                result["focus_areas"].append(item)

    # --- Idle Tasks ---
    for line in sections.get("idle tasks", "").splitlines():
        line = line.strip()
        if line.startswith("- [x] ") or line.startswith("- [X] "):
            task_text = line[6:].strip()
            if task_text:
                result["idle_tasks"].append({"task": task_text, "done": True})
        elif line.startswith("- [ ] "):
            task_text = line[6:].strip()
            if task_text:
                result["idle_tasks"].append({"task": task_text, "done": False})

    # --- Custom Monitors ---
    for line in sections.get("custom monitors", "").splitlines():
        line = line.strip()
        # Format: **name**: `command` -- description
        cm_match = re.match(r"\*\*(.+?)\*\*:\s*`(.+?)`\s*(?:--\s*(.*))?", line)
        if cm_match:
            result["custom_monitors"].append({
                "name": cm_match.group(1).strip(),
                "command": cm_match.group(2).strip(),
                "description": (cm_match.group(3) or "").strip(),
            })

    # --- Dormant Instructions ---
    dormant_text = sections.get("dormant instructions", "").strip()
    result["dormant_instructions"] = dormant_text

    return result


# ---------------------------------------------------------------------------
# Stored-items helpers (tiny duplication to avoid circular imports)
# ---------------------------------------------------------------------------

def _load_stored_items() -> list:
    if STORED_ITEMS_FILE.exists():
        try:
            data = json.loads(STORED_ITEMS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            pass
    return []


def _save_stored_items(items: list):
    STORED_ITEMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if len(items) > 100:
        items = items[:100]
    STORED_ITEMS_FILE.write_text(json.dumps(items, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Screen capture helper
# ---------------------------------------------------------------------------

SCREENSHOTS_DIR = AI_MEMORY_PATH / "cerebro" / "screenshots"


def capture_screen() -> Optional[Dict]:
    """Capture the screen using mss + xdotool. Returns dict with path info or None."""
    try:
        from PIL import Image
        import mss
    except ImportError:
        logger.debug("[capture_screen] mss or Pillow not installed, skipping")
        return None

    try:
        # Get active window title via xdotool
        window_title = "Unknown"
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=3,
            )
            window_title = result.stdout.strip() or "Unknown"
        except Exception:
            pass

        # Grab primary monitor
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        # Resize if wider than 1920px
        max_w = 1920
        if img.width > max_w:
            ratio = max_w / img.width
            new_h = int(img.height * ratio)
            img = img.resize((max_w, new_h), Image.LANCZOS)

        # Save as JPEG
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = SCREENSHOTS_DIR / f"screen_monitor_{timestamp}.jpg"
        img.save(str(filepath), "JPEG", quality=85)

        return {
            "path": str(filepath),
            "window_title": window_title,
            "width": img.width,
            "height": img.height,
        }
    except Exception as exc:
        logger.debug("[capture_screen] Failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MonitorResult:
    monitor: str          # "git_repos", "file_changes", etc.
    changed: bool         # True if different from last check
    summary: str          # Human-readable 1-line summary
    details: str          # Full details for stored item
    severity: str         # "info" | "warning" | "alert"


@dataclass
class HeartbeatResult:
    any_changes: bool
    findings: List[MonitorResult]
    monitors_run: int
    monitors_skipped: int
    duration_ms: int


# ---------------------------------------------------------------------------
# HeartbeatEngine
# ---------------------------------------------------------------------------

class HeartbeatEngine:
    """Runs periodic monitors and returns only changed findings."""

    def __init__(self):
        self._last_hashes: Dict[str, str] = {}
        self._last_run: Optional[datetime] = None
        self._last_result: Optional[HeartbeatResult] = None

    # -- public API --

    async def run_heartbeat(self, config: dict, parsed_md: Optional[dict] = None) -> HeartbeatResult:
        """Run all enabled monitors, compare to last hashes, return only changes."""
        t0 = time.monotonic()
        findings: List[MonitorResult] = []
        monitors_run = 0
        monitors_skipped = 0

        for monitor_name, monitor_cfg in config.get("monitors", {}).items():
            if not monitor_cfg.get("enabled", True):
                monitors_skipped += 1
                continue

            monitors_run += 1
            try:
                result = await self._run_monitor(monitor_name)
            except Exception as exc:
                logger.warning("[Heartbeat] Monitor %s failed: %s", monitor_name, exc)
                result = MonitorResult(
                    monitor=monitor_name,
                    changed=False,
                    summary=f"Monitor error: {exc}",
                    details=str(exc),
                    severity="alert",
                )

            result_hash = hashlib.md5(result.details.encode()).hexdigest()
            if result_hash != self._last_hashes.get(monitor_name):
                self._last_hashes[monitor_name] = result_hash
                if result.summary:
                    result.changed = True
                    findings.append(result)

        # --- Custom monitors from heartbeat.md ---
        if parsed_md:
            for cm in parsed_md.get("custom_monitors", []):
                monitors_run += 1
                try:
                    result = await self._run_custom_monitor(cm)
                except Exception as exc:
                    logger.warning("[Heartbeat] Custom monitor '%s' failed: %s", cm.get("name", "?"), exc)
                    result = MonitorResult(
                        monitor=f"custom:{cm.get('name', '?')}",
                        changed=False,
                        summary=f"Custom monitor error: {exc}",
                        details=str(exc),
                        severity="alert",
                    )
                result_hash = hashlib.md5(result.details.encode()).hexdigest()
                cm_key = f"custom:{cm.get('name', '?')}"
                if result_hash != self._last_hashes.get(cm_key):
                    self._last_hashes[cm_key] = result_hash
                    if result.summary:
                        result.changed = True
                        findings.append(result)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        self._last_run = datetime.now(timezone.utc)
        self._last_result = HeartbeatResult(
            any_changes=len(findings) > 0,
            findings=findings,
            monitors_run=monitors_run,
            monitors_skipped=monitors_skipped,
            duration_ms=elapsed_ms,
        )
        logger.info(
            "[Heartbeat] Complete: %d monitors, %d findings, %dms",
            monitors_run, len(findings), elapsed_ms,
        )
        return self._last_result

    def get_status(self) -> dict:
        """Return status dict for the /api/heartbeat/status endpoint."""
        return {
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_findings": [asdict(f) for f in self._last_result.findings] if self._last_result else [],
            "last_monitors_run": self._last_result.monitors_run if self._last_result else 0,
            "last_duration_ms": self._last_result.duration_ms if self._last_result else 0,
        }

    # -- dispatcher --

    async def _run_monitor(self, name: str) -> MonitorResult:
        handler = {
            "git_repos": self._monitor_git_repos,
            "file_changes": self._monitor_file_changes,
            "memory_health": self._monitor_memory_health,
            "system_health": self._monitor_system_health,
            "network_devices": self._monitor_network_devices,
            "screen_monitor": self._monitor_screen,
        }.get(name)
        if handler is None:
            return MonitorResult(monitor=name, changed=False, summary="", details="unknown monitor", severity="info")
        return await handler()

    async def _run_custom_monitor(self, cm: dict) -> MonitorResult:
        """Run a custom monitor defined in heartbeat.md."""
        name = cm.get("name", "custom")
        command = cm.get("command", "")
        description = cm.get("description", "")
        monitor_key = f"custom:{name}"

        if not command:
            return MonitorResult(monitor=monitor_key, changed=False, summary="", details="no command", severity="info")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            output = stdout.decode("utf-8", errors="replace").strip()
            err_output = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                details = f"Exit code {proc.returncode}\n{err_output or output}"
                return MonitorResult(
                    monitor=monitor_key, changed=False,
                    summary=f"{name}: exit code {proc.returncode}",
                    details=details, severity="warning",
                )
            summary = f"{name}: {output[:80]}" if output else ""
            details = f"{description}\n{output}" if description else output
            return MonitorResult(monitor=monitor_key, changed=False, summary=summary, details=details or "OK", severity="info")
        except asyncio.TimeoutError:
            return MonitorResult(
                monitor=monitor_key, changed=False,
                summary=f"{name}: timed out (15s)",
                details=f"Command timed out: {command}", severity="warning",
            )
        except Exception as exc:
            return MonitorResult(
                monitor=monitor_key, changed=False,
                summary=f"{name}: error",
                details=str(exc), severity="alert",
            )

    # ===================================================================
    # MONITORS
    # ===================================================================

    async def _monitor_git_repos(self) -> MonitorResult:
        repos = [
            Path(os.environ.get("CEREBRO_MCP_SRC", os.path.expanduser("~/NAS-cerebral-interface"))),
            AI_MEMORY_PATH / "projects" / "digital_companion" / "cerebro",
        ]
        lines: List[str] = []
        total_changes = 0
        severity = "info"

        for repo in repos:
            git_dir = repo / ".git"
            if not git_dir.exists():
                continue
            try:
                status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(repo), capture_output=True, text=True, timeout=10,
                )
                branch = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=str(repo), capture_output=True, text=True, timeout=5,
                )
                log = subprocess.run(
                    ["git", "log", "--oneline", "-3"],
                    cwd=str(repo), capture_output=True, text=True, timeout=5,
                )
                branch_name = branch.stdout.strip() or "unknown"
                changed_files = [l for l in status.stdout.strip().splitlines() if l.strip()]
                n = len(changed_files)
                total_changes += n
                state = "clean" if n == 0 else f"{n} uncommitted"
                lines.append(f"{repo.name} ({branch_name}): {state}")
                if changed_files:
                    for cf in changed_files[:5]:
                        lines.append(f"  {cf.strip()}")
                    if n > 5:
                        lines.append(f"  ...and {n - 5} more")
                recent = log.stdout.strip()
                if recent:
                    lines.append(f"  Recent: {recent.splitlines()[0]}")
            except Exception as exc:
                lines.append(f"{repo.name}: error ({exc})")

        if total_changes > 0:
            severity = "warning"

        summary = f"{total_changes} uncommitted change{'s' if total_changes != 1 else ''}" if total_changes else "All repos clean"
        details = "\n".join(lines) if lines else "No git repos found"
        return MonitorResult(monitor="git_repos", changed=False, summary=summary, details=details, severity=severity)

    async def _monitor_file_changes(self) -> MonitorResult:
        cutoff = time.time() - (15 * 60)  # Default: last 15 min
        recent: List[str] = []

        for scan_dir in [AI_MEMORY_PATH, AI_MEMORY_PATH / "cerebro"]:
            if not scan_dir.exists():
                continue
            try:
                for entry in os.scandir(scan_dir):
                    if entry.is_file() and entry.stat().st_mtime > cutoff:
                        rel = entry.name if scan_dir == AI_MEMORY_PATH else f"cerebro/{entry.name}"
                        recent.append(rel)
            except Exception:
                pass

        severity = "info" if len(recent) <= 5 else "warning"
        summary = f"{len(recent)} file{'s' if len(recent) != 1 else ''} changed recently" if recent else ""
        details = "\n".join(recent[:20]) if recent else "No recent file changes"
        return MonitorResult(monitor="file_changes", changed=False, summary=summary, details=details, severity=severity)

    async def _monitor_memory_health(self) -> MonitorResult:
        checks: List[str] = []
        severity = "info"

        # quick_facts.json
        qf = AI_MEMORY_PATH / "quick_facts.json"
        if qf.exists():
            try:
                json.loads(qf.read_text(encoding="utf-8"))
                checks.append("quick_facts: OK")
            except Exception as exc:
                checks.append(f"quick_facts: CORRUPT ({exc})")
                severity = "alert"
        else:
            checks.append("quick_facts: MISSING")
            severity = "alert"

        # Conversation count
        conv_dir = AI_MEMORY_PATH / "conversations"
        if conv_dir.exists():
            try:
                count = sum(1 for _ in conv_dir.iterdir())
                checks.append(f"Conversations: {count}")
            except Exception:
                checks.append("Conversations: error reading")
        else:
            checks.append("Conversations: dir missing")

        # Stored items count
        try:
            items = _load_stored_items()
            checks.append(f"Stored items: {len(items)}")
            if len(items) >= 95:
                severity = max(severity, "warning")
                checks[-1] += " (near limit!)"
        except Exception:
            checks.append("Stored items: error")

        # FAISS index (skip in standalone — no FAISS, use Docker volume)
        _standalone = os.environ.get("CEREBRO_STANDALONE", "") == "1"
        if _standalone:
            try:
                file_count = sum(1 for f in AI_MEMORY_PATH.rglob("*") if f.is_file())
                checks.append(f"Local memory: {file_count} files")
            except Exception:
                checks.append("Local memory: error checking")
        else:
            faiss_path = AI_MEMORY_PATH / "faiss_index"
            if faiss_path.exists():
                try:
                    age_hours = (time.time() - faiss_path.stat().st_mtime) / 3600
                    if age_hours > 48:
                        checks.append(f"FAISS index: stale ({int(age_hours)}h old)")
                        if severity == "info":
                            severity = "warning"
                    else:
                        checks.append(f"FAISS index: OK ({int(age_hours)}h old)")
                except Exception:
                    checks.append("FAISS index: error checking")
            else:
                checks.append("FAISS index: MISSING")
                severity = "alert"

        summary_parts = [c for c in checks if "MISSING" in c or "CORRUPT" in c or "stale" in c or "near limit" in c]
        summary = "; ".join(summary_parts) if summary_parts else "Memory healthy"
        details = "\n".join(checks)
        return MonitorResult(monitor="memory_health", changed=False, summary=summary, details=details, severity=severity)

    async def _monitor_system_health(self) -> MonitorResult:
        checks: List[str] = []
        severity = "info"
        _standalone = os.environ.get("CEREBRO_STANDALONE", "") == "1"

        if _standalone:
            # Docker volume check
            if AI_MEMORY_PATH.exists():
                try:
                    test_file = AI_MEMORY_PATH / ".heartbeat_test"
                    test_file.write_text("ok", encoding="utf-8")
                    test_file.unlink()
                    checks.append("Memory volume: writable")
                except Exception:
                    checks.append("Memory volume: NOT writable")
                    severity = "alert"
            else:
                checks.append("Memory volume: NOT mounted")
                severity = "alert"
        else:
            # NAS mount
            nas_path = AI_MEMORY_PATH
            if nas_path.exists():
                try:
                    test_file = AI_MEMORY_PATH / ".heartbeat_test"
                    test_file.write_text("ok", encoding="utf-8")
                    test_file.unlink()
                    checks.append("NAS: mounted & writable")
                except Exception:
                    checks.append("NAS: mounted but NOT writable")
                    severity = "alert"
            else:
                checks.append("NAS: NOT mounted")
                severity = "alert"

            # DGX Spark reachability (quick TCP check on port 11434 - Ollama)
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(os.environ.get("DGX_HOST", ""), 11434), timeout=3.0
                )
                writer.close()
                await writer.wait_closed()
                checks.append("DGX Spark: reachable")
            except Exception:
                checks.append("DGX Spark: unreachable")
                if severity == "info":
                    severity = "warning"

        alert_parts = [c for c in checks if "NOT" in c or "unreachable" in c]
        summary = "; ".join(alert_parts) if alert_parts else "All systems OK"
        details = "\n".join(checks)
        return MonitorResult(monitor="system_health", changed=False, summary=summary, details=details, severity=severity)


    async def _monitor_network_devices(self) -> MonitorResult:
        """Check Darkhorse reachability and unresolved network alerts."""
        checks: List[str] = []
        severity = "info"

        # Darkhorse Pi reachability (TCP port 22)
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("192.168.0.81", 22), timeout=3.0
            )
            writer.close()
            await writer.wait_closed()
            checks.append("Darkhorse Pi: reachable (scanner active)")
        except Exception:
            checks.append("Darkhorse Pi: UNREACHABLE (scanner may be down)")
            severity = "warning"

        # Check for unresolved alerts
        alerts_file = AI_MEMORY_PATH / "cerebro" / "network_alerts.json"
        unresolved = 0
        if alerts_file.exists():
            try:
                alerts = json.loads(alerts_file.read_text())
                unresolved = sum(1 for a in alerts if a.get("status") == "unresolved")
            except Exception:
                checks.append("Network alerts file: unreadable")
                severity = "warning"

        if unresolved > 0:
            checks.append(f"Unresolved alerts: {unresolved}")
            severity = "alert"
        else:
            checks.append("Network alerts: all clear")

        alert_parts = [c for c in checks if "UNREACHABLE" in c or "Unresolved" in c]
        summary = "; ".join(alert_parts) if alert_parts else "Network monitoring OK"
        details = "\n".join(checks)
        return MonitorResult(monitor="network_devices", changed=False, summary=summary, details=details, severity=severity)

    async def _monitor_screen(self) -> MonitorResult:
        """Capture a screenshot and return it as a monitor finding."""
        result = capture_screen()
        if result is None:
            return MonitorResult(
                monitor="screen_monitor",
                changed=False,
                summary="Screen capture unavailable",
                details="mss/Pillow not installed or no display available",
                severity="info",
            )
        details = (
            f"Screenshot: {result['path']}\n"
            f"Window: {result['window_title']}\n"
            f"Resolution: {result['width']}x{result['height']}"
        )
        return MonitorResult(
            monitor="screen_monitor",
            changed=False,
            summary=f"Screen captured ({result['window_title'][:40]})",
            details=details,
            severity="observation",
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_heartbeat_engine: Optional[HeartbeatEngine] = None


def get_heartbeat_engine() -> HeartbeatEngine:
    """Get or create the singleton HeartbeatEngine instance."""
    global _heartbeat_engine
    if _heartbeat_engine is None:
        _heartbeat_engine = HeartbeatEngine()
    return _heartbeat_engine


# Backward-compat alias (ooda_engine used to import this)
get_idle_thinker = get_heartbeat_engine
