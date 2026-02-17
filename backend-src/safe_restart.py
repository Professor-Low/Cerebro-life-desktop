"""
Safe Restart - Graceful Shutdown and Verified Restart

Provides safe server restart capabilities:
- Graceful shutdown
- Verified restart
- Health confirmation after restart
"""

import asyncio
import subprocess
import sys
import os
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class RestartResult:
    """Result of a restart operation."""
    success: bool
    message: str
    old_pid: Optional[int] = None
    new_pid: Optional[int] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    health_verified: bool = False
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "message": self.message,
            "old_pid": self.old_pid,
            "new_pid": self.new_pid,
            "started_at": self.started_at,
            "health_verified": self.health_verified,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error
        }


class SafeRestart:
    """
    Provides safe server restart with health verification.

    Handles graceful shutdown and verified restart of the Cerebro server,
    with automatic rollback on failure.
    """

    def __init__(
        self,
        server_path: Path = None,
        port: int = 59000,
        health_monitor=None,
        rollback_engine=None
    ):
        """
        Initialize the safe restart manager.

        Args:
            server_path: Path to the server script
            port: Server port
            health_monitor: HealthMonitor instance
            rollback_engine: RollbackEngine instance
        """
        self.server_path = server_path
        self.port = port
        self.health_monitor = health_monitor
        self.rollback_engine = rollback_engine
        self._current_process: Optional[asyncio.subprocess.Process] = None

    def _get_server_pid(self) -> Optional[int]:
        """
        Get the PID of the running server process.

        Returns:
            Server PID or None
        """
        try:
            # On Windows, use netstat to find process on port
            if sys.platform == 'win32':
                result = subprocess.run(
                    ['netstat', '-ano'],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.split('\n'):
                    if f':{self.port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if parts:
                            return int(parts[-1])
            else:
                # On Unix, use lsof
                result = subprocess.run(
                    ['lsof', '-t', f'-i:{self.port}'],
                    capture_output=True,
                    text=True
                )
                if result.stdout.strip():
                    return int(result.stdout.strip().split('\n')[0])

        except Exception as e:
            print(f"Error getting server PID: {e}")

        return None

    async def graceful_shutdown(self, timeout_seconds: int = 30) -> bool:
        """
        Gracefully shutdown the current server.

        Args:
            timeout_seconds: Maximum wait time for shutdown

        Returns:
            True if shutdown successful
        """
        pid = self._get_server_pid()
        if not pid:
            return True  # No server running

        try:
            # Send SIGTERM for graceful shutdown
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/PID', str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)

            # Wait for process to exit
            start_time = asyncio.get_event_loop().time()
            while True:
                await asyncio.sleep(0.5)
                if not self._get_server_pid():
                    return True
                if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                    break

            # Force kill if still running
            print("Graceful shutdown timed out, force killing...")
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGKILL)

            await asyncio.sleep(1)
            return not self._get_server_pid()

        except ProcessLookupError:
            return True  # Process already exited
        except Exception as e:
            print(f"Error during shutdown: {e}")
            return False

    async def start_server(self) -> Optional[int]:
        """
        Start the server process.

        Returns:
            New process PID or None on failure
        """
        if not self.server_path:
            print("No server path configured")
            return None

        try:
            # Determine how to start the server
            python_exe = sys.executable
            server_script = str(self.server_path)

            # Check if it's a uvicorn-based server
            if 'main.py' in server_script:
                # Start with uvicorn
                self._current_process = await asyncio.create_subprocess_exec(
                    python_exe, '-m', 'uvicorn', 'main:app',
                    '--host', '0.0.0.0',
                    '--port', str(self.port),
                    cwd=str(self.server_path.parent),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            else:
                # Start directly
                self._current_process = await asyncio.create_subprocess_exec(
                    python_exe, server_script,
                    '--port', str(self.port),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

            # Wait briefly for process to start
            await asyncio.sleep(2)

            if self._current_process.returncode is not None:
                stderr = await self._current_process.stderr.read()
                print(f"Server failed to start: {stderr.decode()}")
                return None

            return self._current_process.pid

        except Exception as e:
            print(f"Error starting server: {e}")
            return None

    async def wait_for_healthy(
        self,
        timeout_seconds: int = 60,
        poll_interval: float = 2.0
    ) -> bool:
        """
        Wait for the server to become healthy.

        Args:
            timeout_seconds: Maximum wait time
            poll_interval: Seconds between health checks

        Returns:
            True if server is healthy
        """
        if not self.health_monitor:
            # No health monitor - just wait and assume healthy
            await asyncio.sleep(5)
            return self._get_server_pid() is not None

        health_report = await self.health_monitor.wait_for_healthy(
            port=self.port,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval
        )

        return health_report.healthy

    async def restart_with_verification(
        self,
        max_wait_seconds: int = 60,
        rollback_on_failure: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a full restart with health verification.

        Args:
            max_wait_seconds: Maximum wait for healthy status
            rollback_on_failure: Whether to rollback on failure

        Returns:
            RestartResult as dict
        """
        start_time = asyncio.get_event_loop().time()
        old_pid = self._get_server_pid()

        result = RestartResult(
            success=False,
            message="Starting restart",
            old_pid=old_pid
        )

        try:
            # Step 1: Graceful shutdown
            print(f"[SafeRestart] Shutting down server (PID: {old_pid})")
            shutdown_ok = await self.graceful_shutdown(timeout_seconds=30)

            if not shutdown_ok:
                result.error = "Failed to shutdown existing server"
                if rollback_on_failure and self.rollback_engine:
                    await self.rollback_engine.emergency_rollback()
                return result.to_dict()

            # Step 2: Start new server
            print("[SafeRestart] Starting new server...")
            new_pid = await self.start_server()

            if not new_pid:
                result.error = "Failed to start new server"
                if rollback_on_failure and self.rollback_engine:
                    await self.rollback_engine.emergency_rollback()
                return result.to_dict()

            result.new_pid = new_pid
            print(f"[SafeRestart] New server started (PID: {new_pid})")

            # Step 3: Wait for healthy
            print("[SafeRestart] Waiting for healthy status...")
            is_healthy = await self.wait_for_healthy(
                timeout_seconds=max_wait_seconds
            )

            if not is_healthy:
                result.error = "Server not healthy after restart"
                result.message = "Server started but health check failed"

                if rollback_on_failure and self.rollback_engine:
                    print("[SafeRestart] Health check failed, triggering rollback...")
                    await self.rollback_engine.rollback_on_failure(
                        health_failed=True,
                        reason="Server unhealthy after restart"
                    )

                return result.to_dict()

            # Success!
            duration = (asyncio.get_event_loop().time() - start_time) * 1000
            result.success = True
            result.health_verified = True
            result.duration_ms = duration
            result.message = f"Restart complete in {duration:.0f}ms"

            print(f"[SafeRestart] {result.message}")

            return result.to_dict()

        except Exception as e:
            result.error = str(e)
            result.message = f"Restart failed: {e}"

            if rollback_on_failure and self.rollback_engine:
                await self.rollback_engine.emergency_rollback()

            return result.to_dict()

    async def is_server_running(self) -> bool:
        """Check if the server is currently running."""
        return self._get_server_pid() is not None

    async def get_server_status(self) -> Dict[str, Any]:
        """
        Get current server status.

        Returns:
            Dict with server status information
        """
        pid = self._get_server_pid()

        status = {
            "running": pid is not None,
            "pid": pid,
            "port": self.port,
            "checked_at": datetime.now().isoformat()
        }

        if self.health_monitor and pid:
            try:
                health = await self.health_monitor.check_health(self.port)
                status["healthy"] = health.healthy
                status["latency_ms"] = health.avg_latency_ms
            except:
                status["healthy"] = None

        return status


# Singleton instance
_restart_instance: Optional[SafeRestart] = None


def get_safe_restart(
    server_path: Path = None,
    port: int = 59000,
    health_monitor=None,
    rollback_engine=None
) -> Optional[SafeRestart]:
    """Get or create the safe restart singleton."""
    global _restart_instance

    if _restart_instance is None:
        _restart_instance = SafeRestart(
            server_path=server_path,
            port=port,
            health_monitor=health_monitor,
            rollback_engine=rollback_engine
        )

    return _restart_instance
