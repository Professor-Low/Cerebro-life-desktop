"""
Git Manager - Version Control for Cerebro Self-Improvement

Provides git operations for the self-improvement system:
- Auto-initialize git repo if not exists
- Create .gitignore for security
- Commit/branch/tag/rollback operations
- Initial commit of entire codebase
"""

import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class GitResult:
    """Result of a git operation."""
    success: bool
    message: str
    output: str = ""
    commit_hash: Optional[str] = None


class GitManager:
    """
    Manages git operations for Cerebro's self-improvement system.

    Provides safe git operations with automatic initialization,
    branch management, and rollback capabilities.
    """

    # Files/patterns to ignore in .gitignore
    GITIGNORE_PATTERNS = [
        # Secrets and credentials
        "*.key",
        "*.pem",
        "*.crt",
        "*.env",
        ".env*",
        "*.secrets",
        "*password*",
        "*secret*",
        "*credential*",
        "*token*",

        # SSH
        ".ssh/",
        "id_rsa*",
        "authorized_keys",
        "known_hosts",

        # Python
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".Python",
        "venv/",
        ".venv/",
        "env/",
        "*.egg-info/",
        "dist/",
        "build/",

        # Node
        "node_modules/",
        "npm-debug.log",

        # Logs and temp
        "*.log",
        "logs/",
        "tmp/",
        "temp/",
        "*.tmp",

        # IDE
        ".idea/",
        ".vscode/",
        "*.swp",
        "*.swo",

        # OS
        ".DS_Store",
        "Thumbs.db",

        # Cerebro specific
        "audit_logs/",  # Keep audit logs outside git
        "backups/",     # Backups managed separately
        "*.bak",
    ]

    def __init__(self, repo_path: Path):
        """
        Initialize the git manager.

        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = Path(repo_path)
        self.git_dir = self.repo_path / ".git"

    def _run_git(self, args: List[str], cwd: Path = None) -> Tuple[int, str, str]:
        """
        Run a git command and return exit code, stdout, stderr.

        Args:
            args: Git command arguments (without 'git' prefix)
            cwd: Working directory (defaults to repo_path)

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        cmd = ["git"] + args
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd or self.repo_path),
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def is_initialized(self) -> bool:
        """Check if git repository is initialized."""
        return self.git_dir.exists() and self.git_dir.is_dir()

    def initialize(self, create_initial_commit: bool = True) -> GitResult:
        """
        Initialize git repository if not already initialized.

        Args:
            create_initial_commit: Whether to create initial commit

        Returns:
            GitResult with operation status
        """
        if self.is_initialized():
            return GitResult(
                success=True,
                message="Repository already initialized"
            )

        # Initialize repo
        code, out, err = self._run_git(["init"])
        if code != 0:
            return GitResult(
                success=False,
                message=f"Failed to initialize: {err}",
                output=err
            )

        # Create .gitignore
        gitignore_result = self._create_gitignore()
        if not gitignore_result.success:
            return gitignore_result

        # Configure git
        self._run_git(["config", "user.name", "Cerebro"])
        self._run_git(["config", "user.email", "cerebro@professors-nas.local"])

        # Create initial commit
        if create_initial_commit:
            commit_result = self.create_initial_commit()
            if not commit_result.success:
                return commit_result
            return GitResult(
                success=True,
                message="Repository initialized with initial commit",
                commit_hash=commit_result.commit_hash
            )

        return GitResult(
            success=True,
            message="Repository initialized"
        )

    def _create_gitignore(self) -> GitResult:
        """Create or update .gitignore file."""
        gitignore_path = self.repo_path / ".gitignore"

        # Read existing patterns if file exists
        existing_patterns = set()
        if gitignore_path.exists():
            existing_patterns = set(
                line.strip()
                for line in gitignore_path.read_text().splitlines()
                if line.strip() and not line.startswith('#')
            )

        # Merge with required patterns
        all_patterns = existing_patterns | set(self.GITIGNORE_PATTERNS)

        # Write .gitignore
        content = "# Cerebro Auto-Generated .gitignore\n"
        content += "# Security-critical files are blocked\n\n"
        content += "\n".join(sorted(all_patterns))
        content += "\n"

        try:
            gitignore_path.write_text(content)
            return GitResult(success=True, message=".gitignore created/updated")
        except Exception as e:
            return GitResult(success=False, message=f"Failed to create .gitignore: {e}")

    def create_initial_commit(self) -> GitResult:
        """Create the initial commit with all files."""
        # Stage all files (respecting .gitignore)
        code, out, err = self._run_git(["add", "-A"])
        if code != 0:
            return GitResult(
                success=False,
                message=f"Failed to stage files: {err}",
                output=err
            )

        # Create commit
        timestamp = datetime.now().isoformat()
        message = f"Initial commit - Cerebro codebase\n\nInitialized on: {timestamp}"

        code, out, err = self._run_git(["commit", "-m", message])
        if code != 0:
            if "nothing to commit" in err or "nothing to commit" in out:
                return GitResult(
                    success=True,
                    message="No changes to commit"
                )
            return GitResult(
                success=False,
                message=f"Failed to create commit: {err}",
                output=err
            )

        # Get commit hash
        code, commit_hash, _ = self._run_git(["rev-parse", "HEAD"])

        return GitResult(
            success=True,
            message="Initial commit created",
            commit_hash=commit_hash if code == 0 else None
        )

    def get_current_branch(self) -> str:
        """Get the name of the current branch."""
        code, out, _ = self._run_git(["branch", "--show-current"])
        return out if code == 0 else "main"

    def get_current_commit(self) -> Optional[str]:
        """Get the current HEAD commit hash."""
        code, out, _ = self._run_git(["rev-parse", "HEAD"])
        return out if code == 0 else None

    def create_staging_branch(self, proposal_id: str) -> GitResult:
        """
        Create a new staging branch for a proposal.

        Args:
            proposal_id: Unique proposal identifier

        Returns:
            GitResult with branch name in message
        """
        branch_name = f"staging/{proposal_id}"

        # Check if branch already exists
        code, branches, _ = self._run_git(["branch", "--list", branch_name])
        if branches:
            return GitResult(
                success=False,
                message=f"Branch {branch_name} already exists"
            )

        # Create and checkout branch
        code, out, err = self._run_git(["checkout", "-b", branch_name])
        if code != 0:
            return GitResult(
                success=False,
                message=f"Failed to create branch: {err}",
                output=err
            )

        return GitResult(
            success=True,
            message=branch_name,
            output=f"Created and checked out branch: {branch_name}"
        )

    def checkout_branch(self, branch_name: str) -> GitResult:
        """
        Checkout an existing branch.

        Args:
            branch_name: Name of branch to checkout

        Returns:
            GitResult with operation status
        """
        code, out, err = self._run_git(["checkout", branch_name])
        if code != 0:
            return GitResult(
                success=False,
                message=f"Failed to checkout {branch_name}: {err}",
                output=err
            )

        return GitResult(
            success=True,
            message=f"Checked out {branch_name}"
        )

    def commit_changes(
        self,
        message: str,
        files: List[str] = None,
        author: str = "Cerebro <cerebro@professors-nas.local>"
    ) -> GitResult:
        """
        Commit changes to the repository.

        Args:
            message: Commit message
            files: Specific files to commit (None = all changes)
            author: Commit author string

        Returns:
            GitResult with commit hash
        """
        # Stage files
        if files:
            for f in files:
                code, _, err = self._run_git(["add", f])
                if code != 0:
                    return GitResult(
                        success=False,
                        message=f"Failed to stage {f}: {err}"
                    )
        else:
            code, _, err = self._run_git(["add", "-A"])
            if code != 0:
                return GitResult(
                    success=False,
                    message=f"Failed to stage changes: {err}"
                )

        # Check if there are changes to commit
        code, status, _ = self._run_git(["status", "--porcelain"])
        if code == 0 and not status:
            return GitResult(
                success=True,
                message="No changes to commit"
            )

        # Commit
        code, out, err = self._run_git([
            "commit",
            "-m", message,
            "--author", author
        ])

        if code != 0:
            return GitResult(
                success=False,
                message=f"Failed to commit: {err}",
                output=err
            )

        # Get commit hash
        code, commit_hash, _ = self._run_git(["rev-parse", "HEAD"])

        return GitResult(
            success=True,
            message="Changes committed",
            commit_hash=commit_hash if code == 0 else None
        )

    def merge_branch(self, source_branch: str, target_branch: str = None) -> GitResult:
        """
        Merge source branch into target branch.

        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into (None = current branch)

        Returns:
            GitResult with operation status
        """
        # Checkout target if specified
        if target_branch:
            checkout_result = self.checkout_branch(target_branch)
            if not checkout_result.success:
                return checkout_result

        # Merge
        code, out, err = self._run_git(["merge", source_branch, "--no-edit"])
        if code != 0:
            return GitResult(
                success=False,
                message=f"Merge failed: {err}",
                output=err
            )

        return GitResult(
            success=True,
            message=f"Merged {source_branch}",
            output=out
        )

    def delete_branch(self, branch_name: str, force: bool = False) -> GitResult:
        """
        Delete a branch.

        Args:
            branch_name: Branch to delete
            force: Force delete even if not merged

        Returns:
            GitResult with operation status
        """
        flag = "-D" if force else "-d"
        code, out, err = self._run_git(["branch", flag, branch_name])

        if code != 0:
            return GitResult(
                success=False,
                message=f"Failed to delete branch: {err}",
                output=err
            )

        return GitResult(
            success=True,
            message=f"Deleted branch {branch_name}"
        )

    def create_tag(self, tag_name: str, message: str = None) -> GitResult:
        """
        Create a tag at current HEAD.

        Args:
            tag_name: Name for the tag
            message: Optional tag message (creates annotated tag)

        Returns:
            GitResult with operation status
        """
        if message:
            code, out, err = self._run_git(["tag", "-a", tag_name, "-m", message])
        else:
            code, out, err = self._run_git(["tag", tag_name])

        if code != 0:
            return GitResult(
                success=False,
                message=f"Failed to create tag: {err}",
                output=err
            )

        return GitResult(
            success=True,
            message=f"Created tag {tag_name}"
        )

    def rollback_to_commit(self, commit_hash: str) -> GitResult:
        """
        Reset to a previous commit (hard reset).

        WARNING: This discards all changes after the commit.

        Args:
            commit_hash: Commit to reset to

        Returns:
            GitResult with operation status
        """
        code, out, err = self._run_git(["reset", "--hard", commit_hash])

        if code != 0:
            return GitResult(
                success=False,
                message=f"Rollback failed: {err}",
                output=err
            )

        return GitResult(
            success=True,
            message=f"Rolled back to {commit_hash[:8]}",
            commit_hash=commit_hash
        )

    def get_commit_history(self, limit: int = 20) -> List[Dict]:
        """
        Get recent commit history.

        Args:
            limit: Maximum number of commits to return

        Returns:
            List of commit info dicts
        """
        code, out, _ = self._run_git([
            "log",
            f"-{limit}",
            "--format=%H|%s|%an|%ai"
        ])

        if code != 0 or not out:
            return []

        commits = []
        for line in out.splitlines():
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append({
                    "hash": parts[0],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3]
                })

        return commits

    def get_diff(self, commit_a: str = None, commit_b: str = None) -> str:
        """
        Get diff between commits or working directory.

        Args:
            commit_a: First commit (None = working directory)
            commit_b: Second commit (None = HEAD)

        Returns:
            Diff as string
        """
        args = ["diff"]
        if commit_a:
            args.append(commit_a)
        if commit_b:
            args.append(commit_b)

        code, out, _ = self._run_git(args)
        return out if code == 0 else ""

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        code, out, _ = self._run_git(["status", "--porcelain"])
        return code == 0 and bool(out.strip())

    def stash_changes(self, message: str = None) -> GitResult:
        """
        Stash current changes.

        Args:
            message: Optional stash message

        Returns:
            GitResult with operation status
        """
        args = ["stash", "push"]
        if message:
            args.extend(["-m", message])

        code, out, err = self._run_git(args)

        if code != 0:
            return GitResult(
                success=False,
                message=f"Stash failed: {err}",
                output=err
            )

        return GitResult(
            success=True,
            message="Changes stashed",
            output=out
        )

    def pop_stash(self) -> GitResult:
        """
        Pop the most recent stash.

        Returns:
            GitResult with operation status
        """
        code, out, err = self._run_git(["stash", "pop"])

        if code != 0:
            return GitResult(
                success=False,
                message=f"Stash pop failed: {err}",
                output=err
            )

        return GitResult(
            success=True,
            message="Stash applied",
            output=out
        )

    def get_file_at_commit(self, file_path: str, commit_hash: str) -> Optional[str]:
        """
        Get contents of a file at a specific commit.

        Args:
            file_path: Path to file relative to repo root
            commit_hash: Commit to read from

        Returns:
            File contents or None if not found
        """
        code, out, _ = self._run_git(["show", f"{commit_hash}:{file_path}"])
        return out if code == 0 else None


# Singleton instance
_manager_instance: Optional[GitManager] = None


def get_git_manager(repo_path: Path = None) -> Optional[GitManager]:
    """Get or create the git manager singleton."""
    global _manager_instance

    if _manager_instance is None and repo_path:
        _manager_instance = GitManager(repo_path)

    return _manager_instance
