"""
Recovery Recipes for Common Browser Blockers.

This module provides built-in recovery strategies for common situations
that block browser automation: cookie banners, login walls, CAPTCHAs, etc.

Part of the Adaptive Browser Learning System for Cerebro.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from enum import Enum
import asyncio
import re


class BlockerType(Enum):
    """Types of blockers that can interrupt automation."""
    COOKIE_CONSENT = "cookie_consent"
    LOGIN_WALL = "login_wall"
    CAPTCHA = "captcha"
    AGE_VERIFICATION = "age_verification"
    POPUP_MODAL = "popup_modal"
    NEWSLETTER_POPUP = "newsletter_popup"
    RATE_LIMIT = "rate_limit"
    STALE_ELEMENT = "stale_element"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"


@dataclass
class BlockerDetection:
    """Result of detecting a blocker on page."""
    detected: bool
    blocker_type: Optional[BlockerType] = None
    confidence: float = 0.0
    selector: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of applying a recovery recipe."""
    success: bool
    blocker_type: BlockerType
    action_taken: str
    error: Optional[str] = None
    requires_human: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class RecoveryRecipes:
    """
    Built-in recovery strategies for common browser blockers.

    Provides detection and automated resolution for:
    - Cookie consent banners
    - Login walls (with credential support)
    - CAPTCHAs (escalation to human)
    - Popup modals
    - Rate limiting
    - Stale elements
    - Network errors
    """

    # Common patterns for blocker detection
    COOKIE_PATTERNS = {
        "text": [
            "cookie", "cookies", "consent", "gdpr", "privacy",
            "accept all", "accept cookies", "i agree", "got it",
            "allow all", "accept & continue"
        ],
        "selectors": [
            "[class*='cookie']", "[id*='cookie']",
            "[class*='consent']", "[id*='consent']",
            "[class*='gdpr']", "[id*='gdpr']",
            "[aria-label*='cookie']",
        ],
        "accept_buttons": [
            "button:has-text('Accept')",
            "button:has-text('Accept All')",
            "button:has-text('Accept Cookies')",
            "button:has-text('I Agree')",
            "button:has-text('Allow All')",
            "button:has-text('Got It')",
            "button:has-text('OK')",
            "[class*='accept']",
            "[id*='accept']",
            "[data-testid*='accept']",
        ]
    }

    CAPTCHA_PATTERNS = {
        "text": [
            "captcha", "recaptcha", "hcaptcha",
            "verify you're human", "i'm not a robot",
            "security check", "prove you're human"
        ],
        "selectors": [
            "[class*='captcha']", "[id*='captcha']",
            "[class*='recaptcha']", "[id*='recaptcha']",
            "[class*='hcaptcha']", "[id*='hcaptcha']",
            "iframe[src*='captcha']",
            "iframe[src*='recaptcha']",
        ]
    }

    LOGIN_PATTERNS = {
        "text": [
            "sign in", "log in", "login required",
            "please log in", "create an account"
        ],
        "selectors": [
            "[class*='login-wall']", "[id*='login-wall']",
            "[class*='signin-prompt']", "[class*='auth-required']",
        ]
    }

    MODAL_PATTERNS = {
        "selectors": [
            "[role='dialog']", ".modal", ".popup",
            "[aria-modal='true']", ".overlay",
            "[class*='modal']", "[class*='popup']",
        ],
        "close_buttons": [
            "button:has-text('Close')",
            "button:has-text('×')",
            "button:has-text('X')",
            "button[aria-label='Close']",
            "[class*='close']",
            "[class*='dismiss']",
        ]
    }

    NEWSLETTER_PATTERNS = {
        "text": [
            "newsletter", "subscribe", "stay updated",
            "get updates", "sign up for emails"
        ],
        "selectors": [
            "[class*='newsletter']", "[id*='newsletter']",
            "[class*='subscribe']", "[class*='email-capture']",
        ]
    }

    def __init__(
        self,
        skill_storage_path: Optional[str] = None,
        credential_callback: Optional[Callable] = None,
        human_callback: Optional[Callable] = None,
    ):
        """
        Initialize recovery recipes.

        Args:
            skill_storage_path: Path to look for domain-specific login skills
            credential_callback: Callback to get credentials for login walls
            human_callback: Callback to request human intervention (CAPTCHA)
        """
        self.skill_storage_path = skill_storage_path
        self.credential_callback = credential_callback
        self.human_callback = human_callback

    async def detect_blocker(self, page) -> BlockerDetection:
        """
        Detect if page has any blocker.

        Args:
            page: Playwright page

        Returns:
            BlockerDetection with type and confidence
        """
        detections = await asyncio.gather(
            self._detect_cookie_banner(page),
            self._detect_captcha(page),
            self._detect_login_wall(page),
            self._detect_modal(page),
            self._detect_newsletter(page),
            return_exceptions=True
        )

        # Return highest confidence detection
        best = BlockerDetection(detected=False)
        for detection in detections:
            if isinstance(detection, BlockerDetection) and detection.detected:
                if detection.confidence > best.confidence:
                    best = detection

        return best

    async def _detect_cookie_banner(self, page) -> BlockerDetection:
        """Detect cookie consent banner."""
        try:
            page_text = (await page.content()).lower()

            # Check text patterns
            text_matches = sum(
                1 for p in self.COOKIE_PATTERNS["text"]
                if p in page_text
            )

            # Check selectors
            selector_found = False
            found_selector = None
            for selector in self.COOKIE_PATTERNS["selectors"]:
                try:
                    element = await page.query_selector(selector)
                    if element and await element.is_visible():
                        selector_found = True
                        found_selector = selector
                        break
                except Exception:
                    continue

            if text_matches >= 2 or selector_found:
                return BlockerDetection(
                    detected=True,
                    blocker_type=BlockerType.COOKIE_CONSENT,
                    confidence=min(0.3 * text_matches + (0.4 if selector_found else 0), 1.0),
                    selector=found_selector,
                )

        except Exception:
            pass

        return BlockerDetection(detected=False)

    async def _detect_captcha(self, page) -> BlockerDetection:
        """Detect CAPTCHA."""
        try:
            page_text = (await page.content()).lower()

            # Check text patterns
            text_matches = sum(
                1 for p in self.CAPTCHA_PATTERNS["text"]
                if p in page_text
            )

            # Check selectors
            selector_found = False
            for selector in self.CAPTCHA_PATTERNS["selectors"]:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        selector_found = True
                        break
                except Exception:
                    continue

            if text_matches >= 1 or selector_found:
                return BlockerDetection(
                    detected=True,
                    blocker_type=BlockerType.CAPTCHA,
                    confidence=min(0.4 * text_matches + (0.5 if selector_found else 0), 1.0),
                )

        except Exception:
            pass

        return BlockerDetection(detected=False)

    async def _detect_login_wall(self, page) -> BlockerDetection:
        """Detect login wall."""
        try:
            page_text = (await page.content()).lower()

            text_matches = sum(
                1 for p in self.LOGIN_PATTERNS["text"]
                if p in page_text
            )

            selector_found = False
            for selector in self.LOGIN_PATTERNS["selectors"]:
                try:
                    element = await page.query_selector(selector)
                    if element and await element.is_visible():
                        selector_found = True
                        break
                except Exception:
                    continue

            if text_matches >= 2 or selector_found:
                return BlockerDetection(
                    detected=True,
                    blocker_type=BlockerType.LOGIN_WALL,
                    confidence=min(0.3 * text_matches + (0.4 if selector_found else 0), 1.0),
                )

        except Exception:
            pass

        return BlockerDetection(detected=False)

    async def _detect_modal(self, page) -> BlockerDetection:
        """Detect popup modal."""
        try:
            for selector in self.MODAL_PATTERNS["selectors"]:
                try:
                    element = await page.query_selector(selector)
                    if element and await element.is_visible():
                        return BlockerDetection(
                            detected=True,
                            blocker_type=BlockerType.POPUP_MODAL,
                            confidence=0.8,
                            selector=selector,
                        )
                except Exception:
                    continue

        except Exception:
            pass

        return BlockerDetection(detected=False)

    async def _detect_newsletter(self, page) -> BlockerDetection:
        """Detect newsletter popup."""
        try:
            page_text = (await page.content()).lower()

            text_matches = sum(
                1 for p in self.NEWSLETTER_PATTERNS["text"]
                if p in page_text
            )

            # Check if it's in a visible modal/popup
            is_in_modal = False
            for modal_sel in self.MODAL_PATTERNS["selectors"]:
                try:
                    modal = await page.query_selector(modal_sel)
                    if modal and await modal.is_visible():
                        modal_text = (await modal.text_content() or "").lower()
                        if any(p in modal_text for p in self.NEWSLETTER_PATTERNS["text"]):
                            is_in_modal = True
                            break
                except Exception:
                    continue

            if text_matches >= 2 and is_in_modal:
                return BlockerDetection(
                    detected=True,
                    blocker_type=BlockerType.NEWSLETTER_POPUP,
                    confidence=0.7,
                )

        except Exception:
            pass

        return BlockerDetection(detected=False)

    async def apply_recovery(
        self,
        page,
        blocker: BlockerDetection
    ) -> RecoveryResult:
        """
        Apply appropriate recovery strategy for detected blocker.

        Args:
            page: Playwright page
            blocker: Detected blocker

        Returns:
            RecoveryResult indicating success/failure
        """
        if not blocker.detected or not blocker.blocker_type:
            return RecoveryResult(
                success=False,
                blocker_type=BlockerType.POPUP_MODAL,
                action_taken="none",
                error="No blocker to recover from",
            )

        handlers = {
            BlockerType.COOKIE_CONSENT: self._recover_cookie,
            BlockerType.CAPTCHA: self._recover_captcha,
            BlockerType.LOGIN_WALL: self._recover_login,
            BlockerType.POPUP_MODAL: self._recover_modal,
            BlockerType.NEWSLETTER_POPUP: self._recover_newsletter,
            BlockerType.RATE_LIMIT: self._recover_rate_limit,
            BlockerType.STALE_ELEMENT: self._recover_stale,
            BlockerType.NETWORK_ERROR: self._recover_network,
            BlockerType.TIMEOUT: self._recover_timeout,
        }

        handler = handlers.get(blocker.blocker_type)
        if handler:
            return await handler(page, blocker)

        return RecoveryResult(
            success=False,
            blocker_type=blocker.blocker_type,
            action_taken="none",
            error=f"No handler for blocker type: {blocker.blocker_type}",
        )

    async def _recover_cookie(
        self,
        page,
        blocker: BlockerDetection
    ) -> RecoveryResult:
        """Dismiss cookie consent banner."""
        for selector in self.COOKIE_PATTERNS["accept_buttons"]:
            try:
                button = await page.query_selector(selector)
                if button and await button.is_visible():
                    await button.click()
                    await asyncio.sleep(0.5)

                    return RecoveryResult(
                        success=True,
                        blocker_type=BlockerType.COOKIE_CONSENT,
                        action_taken=f"Clicked: {selector}",
                    )
            except Exception:
                continue

        # Try keyboard escape
        try:
            await page.keyboard.press("Escape")
            await asyncio.sleep(0.3)
        except Exception:
            pass

        return RecoveryResult(
            success=False,
            blocker_type=BlockerType.COOKIE_CONSENT,
            action_taken="Tried all accept buttons",
            error="Could not find accept button",
        )

    async def _recover_captcha(
        self,
        page,
        blocker: BlockerDetection
    ) -> RecoveryResult:
        """Handle CAPTCHA - requires human intervention."""
        # CAPTCHAs cannot be automated - escalate to human
        if self.human_callback:
            try:
                await self.human_callback(
                    "captcha",
                    {
                        "url": page.url,
                        "message": "CAPTCHA detected. Please solve it manually.",
                    }
                )
                return RecoveryResult(
                    success=True,
                    blocker_type=BlockerType.CAPTCHA,
                    action_taken="Escalated to human",
                    requires_human=True,
                )
            except Exception:
                pass

        return RecoveryResult(
            success=False,
            blocker_type=BlockerType.CAPTCHA,
            action_taken="Cannot automate CAPTCHA",
            error="Human intervention required",
            requires_human=True,
        )

    async def _recover_login(
        self,
        page,
        blocker: BlockerDetection
    ) -> RecoveryResult:
        """Handle login wall."""
        # Try to get credentials
        if self.credential_callback:
            try:
                domain = re.search(r'https?://([^/]+)', page.url)
                domain = domain.group(1) if domain else page.url

                credentials = await self.credential_callback(domain)
                if credentials:
                    # Look for login skill for this domain
                    # TODO: Integrate with skill system

                    return RecoveryResult(
                        success=False,
                        blocker_type=BlockerType.LOGIN_WALL,
                        action_taken="Got credentials, need login skill",
                        details={"domain": domain},
                    )
            except Exception:
                pass

        return RecoveryResult(
            success=False,
            blocker_type=BlockerType.LOGIN_WALL,
            action_taken="Cannot bypass login",
            error="Login credentials or skill required",
            requires_human=True,
        )

    async def _recover_modal(
        self,
        page,
        blocker: BlockerDetection
    ) -> RecoveryResult:
        """Close popup modal."""
        # Try close buttons
        for selector in self.MODAL_PATTERNS["close_buttons"]:
            try:
                button = await page.query_selector(selector)
                if button and await button.is_visible():
                    await button.click()
                    await asyncio.sleep(0.3)

                    # Check if modal is gone
                    if blocker.selector:
                        element = await page.query_selector(blocker.selector)
                        if not element or not await element.is_visible():
                            return RecoveryResult(
                                success=True,
                                blocker_type=BlockerType.POPUP_MODAL,
                                action_taken=f"Clicked close: {selector}",
                            )
            except Exception:
                continue

        # Try escape key
        try:
            await page.keyboard.press("Escape")
            await asyncio.sleep(0.3)

            if blocker.selector:
                element = await page.query_selector(blocker.selector)
                if not element or not await element.is_visible():
                    return RecoveryResult(
                        success=True,
                        blocker_type=BlockerType.POPUP_MODAL,
                        action_taken="Pressed Escape",
                    )
        except Exception:
            pass

        # Try clicking outside modal
        try:
            await page.mouse.click(10, 10)
            await asyncio.sleep(0.3)
        except Exception:
            pass

        return RecoveryResult(
            success=False,
            blocker_type=BlockerType.POPUP_MODAL,
            action_taken="Tried all close methods",
            error="Could not close modal",
        )

    async def _recover_newsletter(
        self,
        page,
        blocker: BlockerDetection
    ) -> RecoveryResult:
        """Close newsletter popup."""
        # Same as modal but with newsletter-specific selectors
        close_selectors = [
            "button:has-text('No Thanks')",
            "button:has-text('Maybe Later')",
            "button:has-text('Not Now')",
            "[class*='newsletter'] [class*='close']",
            "[class*='subscribe'] [class*='close']",
        ] + self.MODAL_PATTERNS["close_buttons"]

        for selector in close_selectors:
            try:
                button = await page.query_selector(selector)
                if button and await button.is_visible():
                    await button.click()
                    await asyncio.sleep(0.3)

                    return RecoveryResult(
                        success=True,
                        blocker_type=BlockerType.NEWSLETTER_POPUP,
                        action_taken=f"Clicked: {selector}",
                    )
            except Exception:
                continue

        # Fallback to escape
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass

        return RecoveryResult(
            success=False,
            blocker_type=BlockerType.NEWSLETTER_POPUP,
            action_taken="Tried all close methods",
            error="Could not close newsletter popup",
        )

    async def _recover_rate_limit(
        self,
        page,
        blocker: BlockerDetection
    ) -> RecoveryResult:
        """Handle rate limiting."""
        # Wait and retry
        wait_time = blocker.details.get("wait_seconds", 30)

        await asyncio.sleep(wait_time)

        return RecoveryResult(
            success=True,
            blocker_type=BlockerType.RATE_LIMIT,
            action_taken=f"Waited {wait_time} seconds",
            details={"waited": wait_time},
        )

    async def _recover_stale(
        self,
        page,
        blocker: BlockerDetection
    ) -> RecoveryResult:
        """Handle stale element."""
        try:
            # Wait for page to settle
            await asyncio.sleep(1)

            # Optionally refresh
            if blocker.details.get("should_refresh", False):
                await page.reload(wait_until="domcontentloaded")

            return RecoveryResult(
                success=True,
                blocker_type=BlockerType.STALE_ELEMENT,
                action_taken="Waited for page to settle",
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                blocker_type=BlockerType.STALE_ELEMENT,
                action_taken="Wait failed",
                error=str(e),
            )

    async def _recover_network(
        self,
        page,
        blocker: BlockerDetection
    ) -> RecoveryResult:
        """Handle network error."""
        try:
            # Wait and reload
            await asyncio.sleep(2)
            await page.reload(wait_until="domcontentloaded", timeout=30000)

            return RecoveryResult(
                success=True,
                blocker_type=BlockerType.NETWORK_ERROR,
                action_taken="Reloaded page",
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                blocker_type=BlockerType.NETWORK_ERROR,
                action_taken="Reload failed",
                error=str(e),
            )

    async def _recover_timeout(
        self,
        page,
        blocker: BlockerDetection
    ) -> RecoveryResult:
        """Handle timeout."""
        try:
            # Extend timeout and wait
            await asyncio.sleep(5)

            return RecoveryResult(
                success=True,
                blocker_type=BlockerType.TIMEOUT,
                action_taken="Extended wait",
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                blocker_type=BlockerType.TIMEOUT,
                action_taken="Extended wait failed",
                error=str(e),
            )

    async def auto_recover(self, page) -> Optional[RecoveryResult]:
        """
        Automatically detect and recover from any blocker.

        Args:
            page: Playwright page

        Returns:
            RecoveryResult if blocker found and recovery attempted, None otherwise
        """
        blocker = await self.detect_blocker(page)

        if blocker.detected:
            return await self.apply_recovery(page, blocker)

        return None
