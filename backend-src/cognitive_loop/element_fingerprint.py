"""
Element Fingerprinting System for Self-Healing Selectors.

This module provides multi-signal element fingerprinting and self-healing
locator capabilities for robust browser automation that can recover from
selector breakage.

Part of the Adaptive Browser Learning System for Cerebro.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import json
import hashlib
from datetime import datetime
from pathlib import Path


@dataclass
class ElementFingerprint:
    """
    Multi-signal fingerprint for self-healing selectors.

    Captures multiple identifying signals from an element to enable
    fallback strategies when primary selectors break.
    """
    # Priority 1: Semantic (most stable)
    role: Optional[str] = None          # ARIA role
    label: Optional[str] = None         # aria-label or aria-labelledby text
    text_content: str = ""              # Inner text (trimmed)

    # Priority 2: IDs (stable but can change)
    test_id: Optional[str] = None       # data-testid
    id: Optional[str] = None            # element id attribute
    name: Optional[str] = None          # form element name

    # Priority 3: Structure (less stable)
    tag: str = ""                       # HTML tag name
    css_selector: str = ""              # Generated CSS selector
    xpath: str = ""                     # XPath selector
    parent_text: Optional[str] = None   # Parent element text context

    # Position (for visual matching fallback)
    position: Dict[str, float] = field(default_factory=dict)  # x, y, width, height

    # Metadata
    confidence: float = 1.0
    captured_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    page_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert fingerprint to dictionary for storage."""
        return {
            "role": self.role,
            "label": self.label,
            "text_content": self.text_content,
            "test_id": self.test_id,
            "id": self.id,
            "name": self.name,
            "tag": self.tag,
            "css_selector": self.css_selector,
            "xpath": self.xpath,
            "parent_text": self.parent_text,
            "position": self.position,
            "confidence": self.confidence,
            "captured_at": self.captured_at,
            "page_url": self.page_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElementFingerprint":
        """Create fingerprint from dictionary."""
        return cls(
            role=data.get("role"),
            label=data.get("label"),
            text_content=data.get("text_content", ""),
            test_id=data.get("test_id"),
            id=data.get("id"),
            name=data.get("name"),
            tag=data.get("tag", ""),
            css_selector=data.get("css_selector", ""),
            xpath=data.get("xpath", ""),
            parent_text=data.get("parent_text"),
            position=data.get("position", {}),
            confidence=data.get("confidence", 1.0),
            captured_at=data.get("captured_at", datetime.utcnow().isoformat()),
            page_url=data.get("page_url", ""),
        )

    def fingerprint_hash(self) -> str:
        """Generate unique hash for this fingerprint."""
        key_data = f"{self.role}:{self.label}:{self.text_content}:{self.test_id}:{self.id}:{self.tag}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]


class SelfHealingLocator:
    """
    Locates elements with fallback strategies for self-healing.

    Uses prioritized selector strategies to find elements even when
    primary selectors have changed.
    """

    # Locator strategies in priority order (most reliable first)
    PRIORITY = ["role", "label", "text", "testid", "id", "name", "css", "xpath"]

    # Minimum match score to consider element found
    MATCH_THRESHOLD = 0.6

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize self-healing locator.

        Args:
            storage_path: Path to store fingerprint data for healing history
        """
        self.storage_path = storage_path or Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "skills" / "fingerprints"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.healing_history: List[Dict] = []

    async def capture_fingerprint(self, page, selector: str) -> Optional[ElementFingerprint]:
        """
        Capture full fingerprint of an element.

        Args:
            page: Playwright page object
            selector: Initial selector to find the element

        Returns:
            ElementFingerprint or None if element not found
        """
        try:
            element = page.locator(selector).first
            if not await element.count():
                return None

            # Extract all signals via JavaScript
            fingerprint_data = await page.evaluate("""
                (selector) => {
                    const el = document.querySelector(selector);
                    if (!el) return null;

                    // Get computed accessible name
                    const getAccessibleName = (element) => {
                        return element.getAttribute('aria-label') ||
                               element.getAttribute('aria-labelledby') &&
                               document.getElementById(element.getAttribute('aria-labelledby'))?.textContent ||
                               element.getAttribute('title') ||
                               '';
                    };

                    // Get parent text context
                    const getParentText = (element) => {
                        const parent = element.parentElement;
                        if (!parent) return null;
                        const clone = parent.cloneNode(true);
                        // Remove the target element from clone
                        const target = clone.querySelector(selector);
                        if (target) target.remove();
                        return clone.textContent?.trim().slice(0, 100) || null;
                    };

                    // Generate robust CSS selector
                    const getCssSelector = (element) => {
                        if (element.id) return `#${element.id}`;

                        let path = [];
                        while (element && element.nodeType === Node.ELEMENT_NODE) {
                            let selector = element.tagName.toLowerCase();
                            if (element.className) {
                                const classes = element.className.split(/\s+/)
                                    .filter(c => c && !c.match(/^(active|hover|focus|selected)/))
                                    .slice(0, 2);
                                if (classes.length) selector += '.' + classes.join('.');
                            }
                            path.unshift(selector);
                            element = element.parentElement;
                            if (path.length > 4) break;
                        }
                        return path.join(' > ');
                    };

                    // Generate XPath
                    const getXPath = (element) => {
                        if (element.id) return `//*[@id="${element.id}"]`;

                        let path = [];
                        while (element && element.nodeType === Node.ELEMENT_NODE) {
                            let index = 1;
                            let sibling = element.previousElementSibling;
                            while (sibling) {
                                if (sibling.tagName === element.tagName) index++;
                                sibling = sibling.previousElementSibling;
                            }
                            path.unshift(`${element.tagName.toLowerCase()}[${index}]`);
                            element = element.parentElement;
                            if (path.length > 5) break;
                        }
                        return '//' + path.join('/');
                    };

                    const rect = el.getBoundingClientRect();

                    return {
                        role: el.getAttribute('role') || el.tagName.toLowerCase(),
                        label: getAccessibleName(el),
                        text_content: el.textContent?.trim().slice(0, 200) || '',
                        test_id: el.getAttribute('data-testid') || el.getAttribute('data-test-id'),
                        id: el.id || null,
                        name: el.getAttribute('name'),
                        tag: el.tagName.toLowerCase(),
                        css_selector: getCssSelector(el),
                        xpath: getXPath(el),
                        parent_text: getParentText(el),
                        position: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        }
                    };
                }
            """, selector)

            if not fingerprint_data:
                return None

            fingerprint_data["page_url"] = page.url
            return ElementFingerprint.from_dict(fingerprint_data)

        except Exception as e:
            print(f"[SelfHealingLocator] Error capturing fingerprint: {e}")
            return None

    async def locate(self, page, fingerprint: ElementFingerprint) -> Optional[Any]:
        """
        Locate element using fingerprint with fallback strategies.

        Args:
            page: Playwright page object
            fingerprint: Element fingerprint to match

        Returns:
            Playwright Locator or None if not found
        """
        strategies = self._build_strategies(fingerprint)

        for strategy_name, locator_fn in strategies:
            try:
                locator = locator_fn(page)
                if await locator.count() > 0:
                    # Verify this is likely the same element
                    if await self._verify_match(page, locator, fingerprint):
                        return locator
            except Exception:
                continue

        return None

    def _build_strategies(self, fp: ElementFingerprint) -> List[Tuple[str, callable]]:
        """Build ordered list of locator strategies from fingerprint."""
        strategies = []

        # Priority 1: Role + Label (most semantic)
        if fp.role and fp.label:
            strategies.append((
                "role_label",
                lambda p: p.get_by_role(fp.role, name=fp.label)
            ))

        # Priority 2: Label alone
        if fp.label:
            strategies.append((
                "label",
                lambda p: p.get_by_label(fp.label)
            ))

        # Priority 3: Text content
        if fp.text_content:
            text_short = fp.text_content[:50]
            strategies.append((
                "text",
                lambda p: p.get_by_text(text_short, exact=False)
            ))

        # Priority 4: Test ID
        if fp.test_id:
            strategies.append((
                "testid",
                lambda p: p.get_by_test_id(fp.test_id)
            ))

        # Priority 5: Element ID
        if fp.id:
            strategies.append((
                "id",
                lambda p: p.locator(f"#{fp.id}")
            ))

        # Priority 6: Name attribute
        if fp.name:
            strategies.append((
                "name",
                lambda p: p.locator(f"[name='{fp.name}']")
            ))

        # Priority 7: CSS selector
        if fp.css_selector:
            strategies.append((
                "css",
                lambda p: p.locator(fp.css_selector)
            ))

        # Priority 8: XPath
        if fp.xpath:
            strategies.append((
                "xpath",
                lambda p: p.locator(f"xpath={fp.xpath}")
            ))

        return strategies

    async def _verify_match(self, page, locator, fingerprint: ElementFingerprint) -> bool:
        """Verify located element matches fingerprint."""
        try:
            # Get current element properties
            current = await page.evaluate("""
                (selector) => {
                    const el = document.querySelector(selector) ||
                               document.evaluate(selector, document, null,
                                   XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                    if (!el) return null;
                    return {
                        tag: el.tagName.toLowerCase(),
                        text: el.textContent?.trim().slice(0, 200) || '',
                        role: el.getAttribute('role') || el.tagName.toLowerCase()
                    };
                }
            """, await locator.evaluate("el => el.tagName"))

            if not current:
                return False

            score = self.match_score(fingerprint, current)
            return score >= self.MATCH_THRESHOLD

        except Exception:
            # If verification fails, accept the match optimistically
            return True

    def match_score(self, fp: ElementFingerprint, current: Dict) -> float:
        """
        Calculate match score between fingerprint and current element.

        Args:
            fp: Original fingerprint
            current: Current element properties

        Returns:
            Score from 0.0 to 1.0
        """
        scores = []

        # Tag match (weight: 0.2)
        if fp.tag and current.get("tag"):
            scores.append(1.0 if fp.tag == current["tag"] else 0.0)

        # Text similarity (weight: 0.4)
        if fp.text_content and current.get("text"):
            fp_text = fp.text_content.lower()
            curr_text = current["text"].lower()
            # Simple overlap ratio
            overlap = len(set(fp_text.split()) & set(curr_text.split()))
            max_words = max(len(fp_text.split()), len(curr_text.split()), 1)
            scores.append(overlap / max_words)

        # Role match (weight: 0.2)
        if fp.role and current.get("role"):
            scores.append(1.0 if fp.role == current["role"] else 0.0)

        if not scores:
            return 0.5  # Neutral score if no comparison possible

        return sum(scores) / len(scores)

    async def heal_selector(
        self,
        page,
        old_fingerprint: ElementFingerprint
    ) -> Optional[ElementFingerprint]:
        """
        Attempt to heal a broken selector by finding the element and updating fingerprint.

        Args:
            page: Playwright page object
            old_fingerprint: The original fingerprint that no longer works

        Returns:
            New fingerprint if healed, None if element cannot be found
        """
        # Try to locate using fallback strategies
        locator = await self.locate(page, old_fingerprint)

        if not locator:
            return None

        # Generate new CSS selector for the found element
        new_selector = await page.evaluate("""
            (el) => {
                if (el.id) return `#${el.id}`;
                let path = [];
                while (el && el.nodeType === Node.ELEMENT_NODE) {
                    let selector = el.tagName.toLowerCase();
                    if (el.className) {
                        const classes = el.className.split(/\\s+/)
                            .filter(c => c && !c.match(/^(active|hover|focus|selected)/))
                            .slice(0, 2);
                        if (classes.length) selector += '.' + classes.join('.');
                    }
                    path.unshift(selector);
                    el = el.parentElement;
                    if (path.length > 4) break;
                }
                return path.join(' > ');
            }
        """, await locator.element_handle())

        # Capture fresh fingerprint
        new_fingerprint = await self.capture_fingerprint(page, new_selector)

        if new_fingerprint:
            # Record healing event
            self.healing_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "old_selector": old_fingerprint.css_selector,
                "new_selector": new_fingerprint.css_selector,
                "page_url": page.url,
                "match_method": "fallback_strategies"
            })

        return new_fingerprint

    def save_fingerprint(self, skill_id: str, step_index: int, fingerprint: ElementFingerprint):
        """Save fingerprint to storage."""
        filepath = self.storage_path / f"{skill_id}_fps.json"

        # Load existing
        data = {}
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)

        # Update
        if "fingerprints" not in data:
            data["fingerprints"] = {}
        data["fingerprints"][str(step_index)] = fingerprint.to_dict()
        data["updated_at"] = datetime.utcnow().isoformat()

        # Save
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_fingerprint(self, skill_id: str, step_index: int) -> Optional[ElementFingerprint]:
        """Load fingerprint from storage."""
        filepath = self.storage_path / f"{skill_id}_fps.json"

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            data = json.load(f)

        fp_data = data.get("fingerprints", {}).get(str(step_index))
        if fp_data:
            return ElementFingerprint.from_dict(fp_data)
        return None


class FingerprintGenerator:
    """
    Utility class for generating fingerprints during skill creation.
    """

    def __init__(self, locator: Optional[SelfHealingLocator] = None):
        self.locator = locator or SelfHealingLocator()

    async def generate_for_skill_step(
        self,
        page,
        selector: str,
        skill_id: str,
        step_index: int
    ) -> Optional[ElementFingerprint]:
        """
        Generate and store fingerprint for a skill step.

        Args:
            page: Playwright page
            selector: The selector used in the skill step
            skill_id: ID of the skill
            step_index: Index of the step in the skill

        Returns:
            Generated fingerprint or None
        """
        fingerprint = await self.locator.capture_fingerprint(page, selector)

        if fingerprint:
            self.locator.save_fingerprint(skill_id, step_index, fingerprint)

        return fingerprint

    async def batch_generate(
        self,
        page,
        steps: List[Dict],
        skill_id: str
    ) -> Dict[int, ElementFingerprint]:
        """
        Generate fingerprints for all steps that have selectors.

        Args:
            page: Playwright page
            steps: List of skill step dictionaries
            skill_id: ID of the skill

        Returns:
            Dictionary mapping step index to fingerprint
        """
        fingerprints = {}

        for i, step in enumerate(steps):
            selector = step.get("selector")
            if selector:
                fp = await self.generate_for_skill_step(page, selector, skill_id, i)
                if fp:
                    fingerprints[i] = fp

        return fingerprints
