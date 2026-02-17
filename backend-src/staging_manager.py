"""
Staging Manager - Test Changes Before Production

Provides isolated staging environment for testing changes:
- Create staging branch per proposal
- Apply changes to staging branch
- Start staging server on port 59001
- Track staging session state
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from dataclasses import dataclass, field
from enum import Enum


class StagingStatus(str, Enum):
    """Status of a staging session."""
    INITIALIZING = "initializing"
    READY = "ready"
    TESTING = "testing"
    PASSED = "passed"
    FAILED = "failed"
    PROMOTING = "promoting"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    ERROR = "error"


@dataclass
class StagingSession:
    """Represents a staging session for testing changes."""
    session_id: str
    proposal_id: str
    branch_name: str
    status: StagingStatus = StagingStatus.INITIALIZING
    staging_port: int = 59001
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    server_pid: Optional[int] = None
    test_results: Optional[Dict] = None
    health_check_results: Optional[Dict] = None
    error: Optional[str] = None
    original_branch: str = "main"
    original_commit: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "proposal_id": self.proposal_id,
            "branch_name": self.branch_name,
            "status": self.status.value,
            "staging_port": self.staging_port,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "server_pid": self.server_pid,
            "test_results": self.test_results,
            "health_check_results": self.health_check_results,
            "error": self.error,
            "original_branch": self.original_branch,
            "original_commit": self.original_commit
        }


class StagingManager:
    """
    Manages staging environments for testing Cerebro changes.

    Creates isolated staging branches and servers for safe testing
    before promoting changes to production.
    """

    STAGING_PORT = 59001  # Staging server runs on this port
    PRODUCTION_PORT = 59000  # Production server port

    def __init__(self, repo_path: Path, git_manager=None):
        """
        Initialize the staging manager.

        Args:
            repo_path: Path to the Cerebro repository
            git_manager: GitManager instance for version control
        """
        self.repo_path = Path(repo_path)
        self.git_manager = git_manager
        self.active_sessions: Dict[str, StagingSession] = {}
        self.current_session: Optional[StagingSession] = None
        self._server_process: Optional[asyncio.subprocess.Process] = None

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import hashlib
        ts = datetime.now().isoformat()
        return f"stg_{hashlib.sha256(ts.encode()).hexdigest()[:10]}"

    async def create_staging_session(self, proposal_id: str) -> StagingSession:
        """
        Create a new staging session for a proposal.

        Args:
            proposal_id: ID of the proposal being staged

        Returns:
            StagingSession object
        """
        # Clean up any existing session
        if self.current_session:
            await self.cleanup_session(self.current_session.session_id)

        session_id = self._generate_session_id()
        branch_name = f"staging/{proposal_id}"

        # Get current state before creating staging branch
        original_branch = "main"
        original_commit = None
        if self.git_manager:
            original_branch = self.git_manager.get_current_branch()
            original_commit = self.git_manager.get_current_commit()

        session = StagingSession(
            session_id=session_id,
            proposal_id=proposal_id,
            branch_name=branch_name,
            original_branch=original_branch,
            original_commit=original_commit
        )

        self.active_sessions[session_id] = session
        self.current_session = session

        return session

    async def setup_staging_branch(self, session: StagingSession) -> bool:
        """
        Create and checkout staging branch.

        Args:
            session: Staging session

        Returns:
            True if successful
        """
        if not self.git_manager:
            session.error = "Git manager not available"
            session.status = StagingStatus.ERROR
            return False

        # Create staging branch
        result = self.git_manager.create_staging_branch(session.proposal_id)
        if not result.success:
            session.error = result.message
            session.status = StagingStatus.ERROR
            return False

        session.branch_name = result.message  # Branch name is in message
        return True

    async def apply_changes(self, session: StagingSession, changes: Dict[str, str]) -> bool:
        """
        Apply proposed changes to the staging branch.

        Args:
            session: Staging session
            changes: Dict mapping file paths to new contents

        Returns:
            True if successful
        """
        try:
            for file_path, new_content in changes.items():
                full_path = self.repo_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(new_content, encoding='utf-8')

            # Commit changes if git manager available
            if self.git_manager:
                result = self.git_manager.commit_changes(
                    message=f"Staging changes for proposal {session.proposal_id}",
                    author="Cerebro Staging <cerebro-staging@professors-nas.local>"
                )
                if not result.success:
                    session.error = f"Failed to commit changes: {result.message}"
                    return False

            return True

        except Exception as e:
            session.error = f"Failed to apply changes: {str(e)}"
            session.status = StagingStatus.ERROR
            return False

    async def start_staging_server(self, session: StagingSession) -> bool:
        """
        Start a staging server on the staging port.

        Args:
            session: Staging session

        Returns:
            True if server started successfully
        """
        try:
            # Find Python executable
            python_exe = sys.executable

            # Start server process
            env = os.environ.copy()
            env["CEREBRO_PORT"] = str(self.STAGING_PORT)
            env["CEREBRO_STAGING"] = "1"

            main_py = self.repo_path / "backend" / "main.py"
            if not main_py.exists():
                main_py = self.repo_path / "main.py"

            self._server_process = await asyncio.create_subprocess_exec(
                python_exe, "-m", "uvicorn", "main:app",
                "--host", "0.0.0.0",
                "--port", str(self.STAGING_PORT),
                cwd=str(main_py.parent),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            session.server_pid = self._server_process.pid
            session.started_at = datetime.now().isoformat()
            session.status = StagingStatus.READY

            # Wait briefly for server to start
            await asyncio.sleep(3)

            # Check if process is still running
            if self._server_process.returncode is not None:
                stderr = await self._server_process.stderr.read()
                session.error = f"Server exited: {stderr.decode()}"
                session.status = StagingStatus.ERROR
                return False

            return True

        except Exception as e:
            session.error = f"Failed to start staging server: {str(e)}"
            session.status = StagingStatus.ERROR
            return False

    async def stop_staging_server(self, session: StagingSession) -> bool:
        """
        Stop the staging server.

        Args:
            session: Staging session

        Returns:
            True if stopped successfully
        """
        if self._server_process:
            try:
                self._server_process.terminate()
                try:
                    await asyncio.wait_for(
                        self._server_process.wait(),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    self._server_process.kill()

                self._server_process = None
                session.server_pid = None
                return True

            except Exception as e:
                print(f"Error stopping staging server: {e}")
                return False

        return True

    async def promote_to_production(self, session: StagingSession) -> bool:
        """
        Merge staging branch into main/production.

        Args:
            session: Staging session with passed tests

        Returns:
            True if promotion successful
        """
        if session.status != StagingStatus.PASSED:
            session.error = "Cannot promote - tests did not pass"
            return False

        session.status = StagingStatus.PROMOTING

        if not self.git_manager:
            session.error = "Git manager not available"
            session.status = StagingStatus.ERROR
            return False

        try:
            # Checkout main branch
            result = self.git_manager.checkout_branch(session.original_branch)
            if not result.success:
                session.error = f"Failed to checkout main: {result.message}"
                session.status = StagingStatus.ERROR
                return False

            # Merge staging branch
            result = self.git_manager.merge_branch(session.branch_name)
            if not result.success:
                session.error = f"Merge failed: {result.message}"
                session.status = StagingStatus.ERROR
                return False

            # Tag the successful deployment
            tag_name = f"deploy-{session.proposal_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.git_manager.create_tag(
                tag_name,
                message=f"Successful deployment of proposal {session.proposal_id}"
            )

            # Delete staging branch
            self.git_manager.delete_branch(session.branch_name, force=True)

            session.status = StagingStatus.PROMOTED
            session.completed_at = datetime.now().isoformat()

            return True

        except Exception as e:
            session.error = f"Promotion failed: {str(e)}"
            session.status = StagingStatus.ERROR
            return False

    async def rollback_staging(self, session: StagingSession) -> bool:
        """
        Rollback staging changes and return to original state.

        Args:
            session: Staging session to rollback

        Returns:
            True if rollback successful
        """
        try:
            # Stop staging server
            await self.stop_staging_server(session)

            if self.git_manager:
                # Checkout original branch
                self.git_manager.checkout_branch(session.original_branch)

                # Delete staging branch
                self.git_manager.delete_branch(session.branch_name, force=True)

                # Reset to original commit if we have it
                if session.original_commit:
                    self.git_manager.rollback_to_commit(session.original_commit)

            session.status = StagingStatus.ROLLED_BACK
            session.completed_at = datetime.now().isoformat()

            return True

        except Exception as e:
            session.error = f"Rollback failed: {str(e)}"
            return False

    async def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up a staging session.

        Args:
            session_id: ID of session to clean up

        Returns:
            True if cleanup successful
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return True

        # Stop server if running
        if session == self.current_session:
            await self.stop_staging_server(session)

        # Rollback if not already completed
        if session.status not in (StagingStatus.PROMOTED, StagingStatus.ROLLED_BACK):
            await self.rollback_staging(session)

        # Remove from tracking
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        if self.current_session and self.current_session.session_id == session_id:
            self.current_session = None

        return True

    def get_session(self, session_id: str) -> Optional[StagingSession]:
        """Get a staging session by ID."""
        return self.active_sessions.get(session_id)

    def get_current_session(self) -> Optional[StagingSession]:
        """Get the current active staging session."""
        return self.current_session

    def get_all_sessions(self) -> Dict[str, StagingSession]:
        """Get all staging sessions."""
        return self.active_sessions.copy()

    async def run_full_staging_pipeline(
        self,
        proposal_id: str,
        changes: Dict[str, str],
        test_runner=None,
        health_monitor=None
    ) -> StagingSession:
        """
        Run the full staging pipeline:
        1. Create staging branch
        2. Apply changes
        3. Start staging server
        4. Run tests
        5. Check health
        6. Return results (don't auto-promote)

        Args:
            proposal_id: Proposal being tested
            changes: File changes to apply
            test_runner: TestRunner instance for running tests
            health_monitor: HealthMonitor instance for health checks

        Returns:
            StagingSession with results
        """
        # Create session
        session = await self.create_staging_session(proposal_id)

        try:
            # Setup branch
            if not await self.setup_staging_branch(session):
                return session

            # Apply changes
            if not await self.apply_changes(session, changes):
                await self.rollback_staging(session)
                return session

            # Start staging server
            if not await self.start_staging_server(session):
                await self.rollback_staging(session)
                return session

            # Wait for server to be ready
            await asyncio.sleep(5)

            # Run tests if test runner provided
            if test_runner:
                session.status = StagingStatus.TESTING
                test_results = await test_runner.run_all_tests(port=self.STAGING_PORT)
                session.test_results = test_results.to_dict() if hasattr(test_results, 'to_dict') else test_results

                if not test_runner.critical_tests_passed(test_results):
                    session.status = StagingStatus.FAILED
                    session.error = "Critical tests failed"
                    await self.stop_staging_server(session)
                    return session

            # Run health checks if monitor provided
            if health_monitor:
                health_results = await health_monitor.check_health(port=self.STAGING_PORT)
                session.health_check_results = health_results

                if not health_results.get("healthy", False):
                    session.status = StagingStatus.FAILED
                    session.error = "Health check failed"
                    await self.stop_staging_server(session)
                    return session

            # All checks passed
            session.status = StagingStatus.PASSED
            await self.stop_staging_server(session)

            return session

        except Exception as e:
            session.error = str(e)
            session.status = StagingStatus.ERROR
            await self.stop_staging_server(session)
            return session


# Singleton instance
_manager_instance: Optional[StagingManager] = None


def get_staging_manager(repo_path: Path = None, git_manager=None) -> Optional[StagingManager]:
    """Get or create the staging manager singleton."""
    global _manager_instance

    if _manager_instance is None and repo_path:
        _manager_instance = StagingManager(repo_path, git_manager)

    return _manager_instance
