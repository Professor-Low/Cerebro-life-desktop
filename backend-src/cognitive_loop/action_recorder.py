"""
Action Recorder for Browser Skill Generation.

This module provides capabilities to record browser actions via Playwright
event listeners, enabling skill creation from user demonstrations.

Part of the Adaptive Browser Learning System for Cerebro.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from pathlib import Path
import json
import uuid
import asyncio

from .element_fingerprint import SelfHealingLocator, FingerprintGenerator


@dataclass
class RecordedAction:
    """A single recorded browser action."""
    timestamp: str
    action_type: str              # click, fill, navigate, select, keypress
    url: str                      # Page URL when action occurred
    selector: Optional[str] = None
    value: Optional[str] = None   # Filled text, selected option, pressed key
    element_text: Optional[str] = None  # Visible text of element
    element_tag: Optional[str] = None
    fingerprint: Optional[Dict] = None
    screenshot: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action_type": self.action_type,
            "url": self.url,
            "selector": self.selector,
            "value": self.value,
            "element_text": self.element_text,
            "element_tag": self.element_tag,
            "fingerprint": self.fingerprint,
            # Note: screenshot is excluded from dict (binary)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RecordedAction":
        return cls(
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            action_type=data.get("action_type", "unknown"),
            url=data.get("url", ""),
            selector=data.get("selector"),
            value=data.get("value"),
            element_text=data.get("element_text"),
            element_tag=data.get("element_tag"),
            fingerprint=data.get("fingerprint"),
        )


@dataclass
class RecordingSession:
    """A complete recording session."""
    recording_id: str
    name: str
    description: str
    start_url: str
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: Optional[str] = None
    actions: List[RecordedAction] = field(default_factory=list)
    status: str = "recording"     # recording, completed, cancelled
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "name": self.name,
            "description": self.description,
            "start_url": self.start_url,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "actions": [a.to_dict() for a in self.actions],
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RecordingSession":
        return cls(
            recording_id=data["recording_id"],
            name=data.get("name", "Unnamed Recording"),
            description=data.get("description", ""),
            start_url=data.get("start_url", ""),
            started_at=data.get("started_at", datetime.utcnow().isoformat()),
            ended_at=data.get("ended_at"),
            actions=[RecordedAction.from_dict(a) for a in data.get("actions", [])],
            status=data.get("status", "recording"),
            metadata=data.get("metadata", {}),
        )


class ActionRecorder:
    """
    Records browser actions for skill generation.

    Uses Playwright event listeners to capture clicks, fills, and navigation.
    Can convert recordings into reusable skills.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        capture_screenshots: bool = False,
        capture_fingerprints: bool = True,
    ):
        """
        Initialize the action recorder.

        Args:
            storage_path: Path to store recording sessions
            capture_screenshots: Capture screenshot after each action
            capture_fingerprints: Capture element fingerprints for self-healing
        """
        self.storage_path = storage_path or Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "recordings"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.capture_screenshots = capture_screenshots
        self.capture_fingerprints = capture_fingerprints

        self._current_session: Optional[RecordingSession] = None
        self._page = None
        self._listeners_attached = False

        # Self-healing support
        self._locator = SelfHealingLocator() if capture_fingerprints else None
        self._fingerprint_gen = FingerprintGenerator(self._locator) if self._locator else None

        # Callbacks for external notification
        self._on_action: Optional[Callable] = None

    async def start_recording(
        self,
        page,
        name: str,
        description: str = "",
        on_action: Optional[Callable] = None,
    ) -> RecordingSession:
        """
        Start recording actions on a page.

        Args:
            page: Playwright page object
            name: Name for the recording
            description: Description of what's being recorded
            on_action: Optional callback called after each recorded action

        Returns:
            RecordingSession that will be populated as actions occur
        """
        if self._current_session and self._current_session.status == "recording":
            raise RuntimeError("Recording already in progress. Stop it first.")

        self._page = page
        self._on_action = on_action

        # Create new session
        self._current_session = RecordingSession(
            recording_id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            start_url=page.url,
        )

        # Attach event listeners
        await self._attach_listeners()

        return self._current_session

    async def stop_recording(self) -> RecordingSession:
        """
        Stop recording and return the completed session.

        Returns:
            Completed RecordingSession
        """
        if not self._current_session:
            raise RuntimeError("No recording in progress")

        # Detach listeners
        await self._detach_listeners()

        # Finalize session
        self._current_session.status = "completed"
        self._current_session.ended_at = datetime.utcnow().isoformat()

        # Save session
        self._save_session(self._current_session)

        session = self._current_session
        self._current_session = None
        self._page = None

        return session

    async def cancel_recording(self):
        """Cancel the current recording without saving."""
        if self._current_session:
            await self._detach_listeners()
            self._current_session.status = "cancelled"
            self._current_session = None
            self._page = None

    async def _attach_listeners(self):
        """Attach event listeners to capture browser actions."""
        if self._listeners_attached or not self._page:
            return

        # Inject JavaScript to capture DOM events
        await self._page.evaluate("""
            () => {
                window.__recordedActions = [];

                // Capture clicks
                document.addEventListener('click', (e) => {
                    const target = e.target;
                    window.__lastClickedElement = {
                        tag: target.tagName.toLowerCase(),
                        text: target.textContent?.trim().slice(0, 100) || '',
                        selector: window.__getSelector(target),
                        timestamp: new Date().toISOString()
                    };
                }, true);

                // Capture input changes
                document.addEventListener('change', (e) => {
                    const target = e.target;
                    if (target.tagName.toLowerCase() === 'input' ||
                        target.tagName.toLowerCase() === 'textarea' ||
                        target.tagName.toLowerCase() === 'select') {
                        window.__recordedActions.push({
                            type: target.tagName.toLowerCase() === 'select' ? 'select' : 'fill',
                            selector: window.__getSelector(target),
                            value: target.value,
                            tag: target.tagName.toLowerCase(),
                            timestamp: new Date().toISOString()
                        });
                    }
                }, true);

                // Helper to generate selector
                window.__getSelector = (el) => {
                    if (el.id) return `#${el.id}`;
                    if (el.getAttribute('data-testid')) {
                        return `[data-testid="${el.getAttribute('data-testid')}"]`;
                    }
                    if (el.name) return `[name="${el.name}"]`;

                    let path = [];
                    while (el && el.nodeType === Node.ELEMENT_NODE) {
                        let selector = el.tagName.toLowerCase();
                        if (el.className) {
                            const classes = el.className.split(/\\s+/)
                                .filter(c => c && c.length < 30 && !c.match(/^(hover|active|focus)/))
                                .slice(0, 2);
                            if (classes.length) selector += '.' + classes.join('.');
                        }
                        path.unshift(selector);
                        el = el.parentElement;
                        if (path.length > 3) break;
                    }
                    return path.join(' > ');
                };

                // Get and clear recorded actions
                window.__getRecordedActions = () => {
                    const actions = window.__recordedActions;
                    window.__recordedActions = [];
                    return actions;
                };
            }
        """)

        # Listen for navigation
        self._page.on("framenavigated", self._on_navigation)

        # Start polling for recorded actions
        self._listeners_attached = True
        asyncio.create_task(self._poll_actions())

    async def _detach_listeners(self):
        """Remove event listeners."""
        if not self._listeners_attached or not self._page:
            return

        try:
            self._page.remove_listener("framenavigated", self._on_navigation)
        except Exception:
            pass

        self._listeners_attached = False

    async def _poll_actions(self):
        """Poll for recorded actions from the page."""
        while self._listeners_attached and self._page and self._current_session:
            try:
                # Get actions recorded by JavaScript
                actions = await self._page.evaluate("window.__getRecordedActions ? window.__getRecordedActions() : []")

                for action_data in actions:
                    await self._record_action(
                        action_type=action_data.get("type", "unknown"),
                        selector=action_data.get("selector"),
                        value=action_data.get("value"),
                        element_tag=action_data.get("tag"),
                    )

                await asyncio.sleep(0.1)  # Poll every 100ms

            except Exception as e:
                if "Target closed" in str(e) or "has been closed" in str(e):
                    break
                # Continue polling on other errors

    def _on_navigation(self, frame):
        """Handle navigation events."""
        if frame == self._page.main_frame and self._current_session:
            asyncio.create_task(self._record_action(
                action_type="navigate",
                value=frame.url,
            ))

    async def _record_action(
        self,
        action_type: str,
        selector: Optional[str] = None,
        value: Optional[str] = None,
        element_text: Optional[str] = None,
        element_tag: Optional[str] = None,
    ):
        """Record a single action."""
        if not self._current_session or not self._page:
            return

        action = RecordedAction(
            timestamp=datetime.utcnow().isoformat(),
            action_type=action_type,
            url=self._page.url,
            selector=selector,
            value=value,
            element_text=element_text,
            element_tag=element_tag,
        )

        # Capture fingerprint if enabled
        if self.capture_fingerprints and selector and self._locator:
            try:
                fp = await self._locator.capture_fingerprint(self._page, selector)
                if fp:
                    action.fingerprint = fp.to_dict()
            except Exception:
                pass

        # Capture screenshot if enabled
        if self.capture_screenshots:
            try:
                action.screenshot = await self._page.screenshot()
            except Exception:
                pass

        self._current_session.actions.append(action)

        # Notify callback
        if self._on_action:
            try:
                if asyncio.iscoroutinefunction(self._on_action):
                    await self._on_action(action)
                else:
                    self._on_action(action)
            except Exception:
                pass

    async def generate_skill_from_recording(
        self,
        recording_id: str,
        skill_name: str,
        skill_description: str,
        parameterize: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a skill from a recording session.

        Args:
            recording_id: ID of the recording session
            skill_name: Name for the generated skill
            skill_description: Description of the skill
            parameterize: List of values to turn into parameters

        Returns:
            Skill dictionary compatible with skill_generator.py
        """
        # Load recording
        session = self.load_recording(recording_id)
        if not session:
            raise ValueError(f"Recording not found: {recording_id}")

        if session.status != "completed":
            raise ValueError(f"Recording not completed: {session.status}")

        parameterize = parameterize or []
        param_map = {}  # value -> param name

        # Convert actions to skill steps
        skill_steps = []
        prev_url = None

        for action in session.actions:
            # Skip duplicate navigations
            if action.action_type == "navigate":
                if action.value == prev_url:
                    continue
                prev_url = action.value

            step = {
                "action": action.action_type,
                "selector": action.selector,
                "description": self._generate_step_description(action),
            }

            # Handle value parameterization
            if action.value:
                parameterized_value = action.value
                for param_value in parameterize:
                    if param_value in action.value:
                        if param_value not in param_map:
                            param_map[param_value] = self._generate_param_name(
                                param_value, param_map
                            )
                        parameterized_value = parameterized_value.replace(
                            param_value, f"{{{param_map[param_value]}}}"
                        )
                step["value"] = parameterized_value

            # Include fingerprint
            if action.fingerprint:
                step["fingerprint"] = action.fingerprint

            skill_steps.append(step)

        # Build skill
        skill = {
            "skill_id": f"skill_rec_{recording_id}_{skill_name.lower().replace(' ', '_')[:20]}",
            "name": skill_name,
            "description": skill_description,
            "start_url": session.start_url,
            "steps": skill_steps,
            "parameters": list(param_map.values()),
            "parameter_descriptions": {v: f"Value for {v}" for v in param_map.values()},
            "created_from_recording": recording_id,
            "created_at": datetime.utcnow().isoformat(),
            "status": "unverified",
            "version": 1,
        }

        # Save skill
        skill_path = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "skills" / f"{skill['skill_id']}.json"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        with open(skill_path, "w") as f:
            json.dump(skill, f, indent=2)

        return skill

    def _generate_step_description(self, action: RecordedAction) -> str:
        """Generate a human-readable description for an action."""
        if action.action_type == "navigate":
            return f"Navigate to {action.value}"
        elif action.action_type == "click":
            if action.element_text:
                return f"Click on '{action.element_text[:30]}'"
            return "Click element"
        elif action.action_type == "fill":
            return f"Enter text in {action.element_tag or 'field'}"
        elif action.action_type == "select":
            return f"Select '{action.value}'"
        else:
            return f"{action.action_type} action"

    def _generate_param_name(self, value: str, existing: Dict[str, str]) -> str:
        """Generate a parameter name from a value."""
        value_lower = value.lower()

        if "@" in value:
            name = "email"
        elif any(p in value_lower for p in ["password", "pass"]):
            name = "password"
        elif any(p in value_lower for p in ["user", "name"]):
            name = "username"
        elif value.isdigit():
            name = "number"
        else:
            name = "input"

        base_name = name
        counter = 1
        while name in existing.values():
            name = f"{base_name}_{counter}"
            counter += 1

        return name

    def _save_session(self, session: RecordingSession):
        """Save recording session to storage."""
        filepath = self.storage_path / f"{session.recording_id}.json"
        with open(filepath, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def load_recording(self, recording_id: str) -> Optional[RecordingSession]:
        """Load a recording session from storage."""
        filepath = self.storage_path / f"{recording_id}.json"
        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            data = json.load(f)
        return RecordingSession.from_dict(data)

    def list_recordings(self) -> List[Dict[str, Any]]:
        """List all recording sessions."""
        recordings = []
        for filepath in self.storage_path.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                recordings.append({
                    "recording_id": data.get("recording_id"),
                    "name": data.get("name"),
                    "description": data.get("description"),
                    "status": data.get("status"),
                    "started_at": data.get("started_at"),
                    "action_count": len(data.get("actions", [])),
                })
            except Exception:
                continue

        return sorted(recordings, key=lambda r: r.get("started_at", ""), reverse=True)

    def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording session."""
        filepath = self.storage_path / f"{recording_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False
