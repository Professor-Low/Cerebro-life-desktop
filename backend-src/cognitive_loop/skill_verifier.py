"""
Skill Verifier for Quality Assurance.

This module provides capabilities to verify skills work correctly,
auto-heal broken selectors, and track skill reliability.

Part of the Adaptive Browser Learning System for Cerebro.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
import json
import asyncio

from .element_fingerprint import SelfHealingLocator, ElementFingerprint


@dataclass
class StepVerificationResult:
    """Result of verifying a single step."""
    step_index: int
    action: str
    success: bool
    selector_found: bool = True
    selector_healed: bool = False
    original_selector: Optional[str] = None
    healed_selector: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class VerificationResult:
    """Result of verifying a complete skill."""
    skill_id: str
    skill_name: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    success: bool = False
    steps_total: int = 0
    steps_passed: int = 0
    failure_step: Optional[int] = None
    step_results: List[StepVerificationResult] = field(default_factory=list)
    healed_selectors: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0
    screenshots: List[bytes] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "timestamp": self.timestamp,
            "success": self.success,
            "steps_total": self.steps_total,
            "steps_passed": self.steps_passed,
            "failure_step": self.failure_step,
            "step_results": [
                {
                    "step_index": r.step_index,
                    "action": r.action,
                    "success": r.success,
                    "selector_found": r.selector_found,
                    "selector_healed": r.selector_healed,
                    "original_selector": r.original_selector,
                    "healed_selector": r.healed_selector,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                }
                for r in self.step_results
            ],
            "healed_selectors": self.healed_selectors,
            "error": self.error,
            "duration_ms": self.duration_ms,
            # Note: screenshots excluded (binary)
        }


class SkillVerifier:
    """
    Verifies skills and auto-heals broken selectors.

    Capabilities:
    - Execute skill in verification mode (dry run or actual)
    - Detect broken selectors and attempt healing
    - Update skill status based on verification results
    - Track verification history
    """

    def __init__(
        self,
        skill_generator=None,
        self_healing_locator: Optional[SelfHealingLocator] = None,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize the skill verifier.

        Args:
            skill_generator: SkillGenerator instance for skill access and updates
            self_healing_locator: SelfHealingLocator for healing broken selectors
            storage_path: Path to store verification results
        """
        self.skill_generator = skill_generator
        self.locator = self_healing_locator or SelfHealingLocator()
        self.storage_path = storage_path or Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "verifications"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def verify_skill(
        self,
        skill_id: str,
        page=None,
        test_params: Optional[Dict[str, str]] = None,
        dry_run: bool = False,
        take_screenshots: bool = False,
        auto_heal: bool = True,
    ) -> VerificationResult:
        """
        Verify a skill works correctly.

        Args:
            skill_id: ID of the skill to verify
            page: Playwright page (optional, will use skill_generator's if not provided)
            test_params: Test parameter values
            dry_run: Only check selectors exist, don't execute actions
            take_screenshots: Capture screenshot after each step
            auto_heal: Attempt to heal broken selectors

        Returns:
            VerificationResult with detailed step-by-step results
        """
        if not self.skill_generator:
            raise RuntimeError("SkillGenerator required for verification")

        skill = self.skill_generator.get_skill(skill_id)
        if not skill:
            return VerificationResult(
                skill_id=skill_id,
                skill_name="Unknown",
                success=False,
                error=f"Skill not found: {skill_id}",
            )

        result = VerificationResult(
            skill_id=skill_id,
            skill_name=skill.name,
            steps_total=len(skill.steps),
        )

        start_time = datetime.utcnow()
        test_params = test_params or {}

        try:
            # Get page
            if page is None:
                page = await self.skill_generator._ensure_browser()

            # Verify each step
            for i, step in enumerate(skill.steps):
                step_start = datetime.utcnow()
                step_result = StepVerificationResult(
                    step_index=i,
                    action=step.action,
                    success=False,
                )

                try:
                    # Substitute parameters
                    selector = self.skill_generator._substitute_params(
                        step.selector, test_params
                    ) if step.selector else None
                    value = self.skill_generator._substitute_params(
                        step.value, test_params
                    ) if step.value else None

                    # Actions that need selectors
                    if step.action in ("click", "fill", "type", "select", "hover") and selector:
                        # Check if selector exists
                        selector_works = await self._check_selector(page, selector)

                        if not selector_works:
                            step_result.selector_found = False
                            step_result.original_selector = selector

                            # Try to heal
                            if auto_heal and step.fingerprint:
                                healed = await self._try_heal(page, step, i, skill_id)
                                if healed:
                                    selector = healed
                                    step_result.selector_healed = True
                                    step_result.healed_selector = healed
                                    result.healed_selectors.append({
                                        "step_index": i,
                                        "original": step_result.original_selector,
                                        "healed": healed,
                                    })
                                else:
                                    raise Exception(f"Selector not found and healing failed: {selector}")

                    # Execute action (unless dry run)
                    if not dry_run:
                        await self._execute_verification_step(page, step, selector, value)

                    step_result.success = True
                    result.steps_passed += 1

                except Exception as e:
                    step_result.success = False
                    step_result.error = str(e)
                    result.failure_step = i
                    result.error = f"Step {i} failed: {str(e)}"

                step_result.duration_ms = (datetime.utcnow() - step_start).total_seconds() * 1000
                result.step_results.append(step_result)

                # Screenshot
                if take_screenshots:
                    try:
                        result.screenshots.append(await page.screenshot())
                    except Exception:
                        pass

                # Stop on failure
                if not step_result.success:
                    break

            # Overall result
            result.success = result.steps_passed == result.steps_total
            result.duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Update skill status
            if result.success:
                await self._update_skill_status(skill_id, "verified", result)
            else:
                await self._update_skill_status(skill_id, "failed", result)

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            print(f"[SkillVerifier] Verification failed for skill '{skill_id}': {e}")

        # Save verification result
        self._save_result(result)

        return result

    async def _check_selector(self, page, selector: str, timeout: int = 2000) -> bool:
        """Check if a selector exists on the page."""
        try:
            await page.wait_for_selector(selector, timeout=timeout, state="attached")
            return True
        except Exception:
            return False

    async def _try_heal(
        self,
        page,
        step,
        step_index: int,
        skill_id: str
    ) -> Optional[str]:
        """Try to heal a broken selector using fingerprint."""
        if not step.fingerprint:
            return None

        try:
            fingerprint = ElementFingerprint.from_dict(step.fingerprint)
            locator = await self.locator.locate(page, fingerprint)

            if locator and await locator.count() > 0:
                # Get new selector
                new_fp = await self.locator.heal_selector(page, fingerprint)
                if new_fp:
                    new_selector = new_fp.css_selector

                    # Update skill
                    skill = self.skill_generator.get_skill(skill_id)
                    if skill and step_index < len(skill.steps):
                        skill.steps[step_index].selector = new_selector
                        skill.steps[step_index].fingerprint = new_fp.to_dict()
                        skill.version += 1
                        skill.updated_at = datetime.now()
                        self.skill_generator._save_skill(skill)

                    return new_selector

        except Exception as e:
            print(f"[SkillVerifier] Healing failed: {e}")

        return None

    async def _execute_verification_step(
        self,
        page,
        step,
        selector: Optional[str],
        value: Optional[str]
    ):
        """Execute a step during verification."""
        action = step.action.lower()

        if action == "navigate":
            await page.goto(value, wait_until="domcontentloaded", timeout=step.timeout_ms)

        elif action == "click":
            await page.click(selector, timeout=step.timeout_ms)

        elif action in ("fill", "type"):
            await page.fill(selector, value or "", timeout=step.timeout_ms)

        elif action == "select":
            await page.select_option(selector, value, timeout=step.timeout_ms)

        elif action == "hover":
            await page.hover(selector, timeout=step.timeout_ms)

        elif action == "wait":
            if step.wait_for:
                await page.wait_for_selector(step.wait_for, timeout=step.timeout_ms)
            else:
                await asyncio.sleep(step.timeout_ms / 1000)

        elif action == "scroll":
            if selector:
                await page.locator(selector).scroll_into_view_if_needed()
            else:
                await page.evaluate(f"window.scrollBy(0, {value or 500})")

        # Wait for any post-action condition
        if step.wait_for and action not in ["wait"]:
            try:
                await page.wait_for_selector(step.wait_for, timeout=2000)
            except Exception:
                pass

    async def _update_skill_status(
        self,
        skill_id: str,
        status: str,
        result: VerificationResult
    ):
        """Update skill status based on verification result."""
        from .skill_generator import SkillStatus

        skill = self.skill_generator.get_skill(skill_id)
        if not skill:
            return

        if status == "verified":
            skill.status = SkillStatus.VERIFIED
            skill.success_count += 1
        elif status == "failed":
            skill.status = SkillStatus.FAILED
            skill.fail_count += 1

        skill.updated_at = datetime.now()
        self.skill_generator._save_skill(skill)

    async def batch_verify(
        self,
        skill_ids: List[str],
        test_params_map: Optional[Dict[str, Dict[str, str]]] = None,
        dry_run: bool = False,
    ) -> List[VerificationResult]:
        """
        Verify multiple skills.

        Args:
            skill_ids: List of skill IDs to verify
            test_params_map: Map of skill_id -> test parameters
            dry_run: Only check selectors, don't execute

        Returns:
            List of VerificationResult for each skill
        """
        results = []
        test_params_map = test_params_map or {}

        for skill_id in skill_ids:
            result = await self.verify_skill(
                skill_id,
                test_params=test_params_map.get(skill_id),
                dry_run=dry_run,
            )
            results.append(result)

        return results

    async def verify_all_unverified(
        self,
        dry_run: bool = True
    ) -> List[VerificationResult]:
        """
        Verify all skills with status DRAFT or FAILED.

        Args:
            dry_run: Only check selectors exist

        Returns:
            List of verification results
        """
        from .skill_generator import SkillStatus

        skills = self.skill_generator.list_skills()
        unverified = [
            s for s in skills
            if s.status in (SkillStatus.DRAFT, SkillStatus.FAILED)
        ]

        return await self.batch_verify(
            [s.id for s in unverified],
            dry_run=dry_run,
        )

    def _save_result(self, result: VerificationResult):
        """Save verification result to storage."""
        filename = f"{result.skill_id}_{result.timestamp.replace(':', '-')}.json"
        filepath = self.storage_path / filename
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def get_verification_history(
        self,
        skill_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get verification history for a skill."""
        results = []

        for filepath in self.storage_path.glob(f"{skill_id}_*.json"):
            try:
                with open(filepath, "r") as f:
                    results.append(json.load(f))
            except Exception:
                continue

        # Sort by timestamp descending
        results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return results[:limit]

    def get_all_failed_skills(self) -> List[str]:
        """Get IDs of all skills that failed verification."""
        from .skill_generator import SkillStatus

        if not self.skill_generator:
            return []

        skills = self.skill_generator.list_skills(status=SkillStatus.FAILED)
        return [s.id for s in skills]

    def get_skill_reliability(self, skill_id: str) -> Dict[str, Any]:
        """Get reliability metrics for a skill."""
        history = self.get_verification_history(skill_id, limit=100)

        if not history:
            return {
                "skill_id": skill_id,
                "verifications": 0,
                "success_rate": 0.0,
                "heals_count": 0,
            }

        successes = sum(1 for r in history if r.get("success"))
        heals = sum(len(r.get("healed_selectors", [])) for r in history)

        return {
            "skill_id": skill_id,
            "verifications": len(history),
            "success_rate": successes / len(history) if history else 0.0,
            "last_success": next(
                (r["timestamp"] for r in history if r.get("success")),
                None
            ),
            "last_failure": next(
                (r["timestamp"] for r in history if not r.get("success")),
                None
            ),
            "heals_count": heals,
        }
