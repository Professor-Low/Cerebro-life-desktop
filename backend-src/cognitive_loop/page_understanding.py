"""
Page Understanding System for Adaptive Browser Exploration.

This module provides capabilities to analyze web pages, extract structured
information about interactable elements, and classify page types to inform
LLM-guided exploration decisions.

Part of the Adaptive Browser Learning System for Cerebro.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime


@dataclass
class InteractableElement:
    """Represents an interactable element on the page."""
    index: int                          # Reference index for LLM
    tag: str                            # HTML tag
    type: str                           # button, input, link, select, etc.
    text: str                           # Visible text
    selector: str                       # CSS selector for interaction
    attributes: Dict[str, str] = field(default_factory=dict)
    is_visible: bool = True
    is_enabled: bool = True


@dataclass
class FormInfo:
    """Represents a form on the page."""
    form_id: Optional[str]
    action: str
    method: str
    fields: List[Dict[str, Any]]        # List of form fields
    submit_button: Optional[str]        # Selector for submit button


@dataclass
class PageState:
    """
    Comprehensive state of a web page for LLM decision making.
    """
    # Basic info
    url: str
    title: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Interactable elements (indexed for LLM reference)
    interactable_elements: List[InteractableElement] = field(default_factory=list)

    # Forms
    forms: List[FormInfo] = field(default_factory=list)

    # Navigation
    navigation_links: List[Dict[str, str]] = field(default_factory=list)

    # Content
    main_content: str = ""              # Truncated main text content
    headings: List[str] = field(default_factory=list)

    # Classification
    page_type: str = "unknown"          # login, search, list, detail, form, error, dashboard
    detected_actions: List[str] = field(default_factory=list)

    # Metadata
    has_modal: bool = False
    has_cookie_banner: bool = False
    has_captcha: bool = False
    error_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "url": self.url,
            "title": self.title,
            "timestamp": self.timestamp,
            "interactable_elements": [
                {
                    "index": e.index,
                    "tag": e.tag,
                    "type": e.type,
                    "text": e.text,
                    "selector": e.selector,
                    "attributes": e.attributes,
                    "is_visible": e.is_visible,
                    "is_enabled": e.is_enabled,
                }
                for e in self.interactable_elements
            ],
            "forms": [
                {
                    "form_id": f.form_id,
                    "action": f.action,
                    "method": f.method,
                    "fields": f.fields,
                    "submit_button": f.submit_button,
                }
                for f in self.forms
            ],
            "navigation_links": self.navigation_links,
            "main_content": self.main_content,
            "headings": self.headings,
            "page_type": self.page_type,
            "detected_actions": self.detected_actions,
            "has_modal": self.has_modal,
            "has_cookie_banner": self.has_cookie_banner,
            "has_captcha": self.has_captcha,
            "error_messages": self.error_messages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageState":
        """Create PageState from dictionary."""
        elements = [
            InteractableElement(
                index=e["index"],
                tag=e["tag"],
                type=e["type"],
                text=e["text"],
                selector=e["selector"],
                attributes=e.get("attributes", {}),
                is_visible=e.get("is_visible", True),
                is_enabled=e.get("is_enabled", True),
            )
            for e in data.get("interactable_elements", [])
        ]

        forms = [
            FormInfo(
                form_id=f.get("form_id"),
                action=f.get("action", ""),
                method=f.get("method", "GET"),
                fields=f.get("fields", []),
                submit_button=f.get("submit_button"),
            )
            for f in data.get("forms", [])
        ]

        return cls(
            url=data.get("url", ""),
            title=data.get("title", ""),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            interactable_elements=elements,
            forms=forms,
            navigation_links=data.get("navigation_links", []),
            main_content=data.get("main_content", ""),
            headings=data.get("headings", []),
            page_type=data.get("page_type", "unknown"),
            detected_actions=data.get("detected_actions", []),
            has_modal=data.get("has_modal", False),
            has_cookie_banner=data.get("has_cookie_banner", False),
            has_captcha=data.get("has_captcha", False),
            error_messages=data.get("error_messages", []),
        )


class PageUnderstanding:
    """
    Analyzes web pages to extract structured information for LLM decision making.
    """

    # Page type classification patterns
    PAGE_TYPE_PATTERNS = {
        "login": ["login", "sign in", "log in", "signin", "authenticate"],
        "signup": ["sign up", "register", "create account", "join"],
        "search": ["search", "find", "lookup", "query"],
        "list": ["results", "items", "products", "articles", "posts"],
        "detail": ["details", "view", "profile", "item"],
        "form": ["form", "submit", "apply", "request"],
        "dashboard": ["dashboard", "overview", "home", "my account"],
        "error": ["error", "404", "not found", "oops", "something went wrong"],
        "checkout": ["checkout", "payment", "cart", "order"],
    }

    # Cookie banner patterns
    COOKIE_PATTERNS = [
        "cookie", "cookies", "consent", "gdpr", "privacy",
        "accept all", "accept cookies", "i agree"
    ]

    # CAPTCHA patterns
    CAPTCHA_PATTERNS = [
        "captcha", "recaptcha", "hcaptcha", "verify you're human",
        "i'm not a robot", "security check"
    ]

    def __init__(self, ollama_client=None):
        """
        Initialize page understanding.

        Args:
            ollama_client: Optional Ollama client for LLM classification
        """
        self.ollama = ollama_client

    async def analyze_page(self, page, include_visual: bool = False) -> PageState:
        """
        Perform comprehensive page analysis.

        Args:
            page: Playwright page object
            include_visual: Whether to include screenshot analysis (future)

        Returns:
            PageState with all extracted information
        """
        # Extract DOM state via JavaScript
        dom_state = await self._extract_dom_state(page)

        # Build PageState
        state = PageState(
            url=page.url,
            title=await page.title(),
        )

        # Process interactable elements
        state.interactable_elements = self._process_interactables(
            dom_state.get("interactables", [])
        )

        # Process forms
        state.forms = self._process_forms(dom_state.get("forms", []))

        # Process navigation
        state.navigation_links = dom_state.get("navigation", [])

        # Extract content
        state.main_content = dom_state.get("main_content", "")[:2000]
        state.headings = dom_state.get("headings", [])[:10]

        # Detect page features
        state.has_modal = dom_state.get("has_modal", False)
        state.has_cookie_banner = self._detect_cookie_banner(dom_state)
        state.has_captcha = self._detect_captcha(dom_state)
        state.error_messages = dom_state.get("error_messages", [])

        # Classify page type
        state.page_type = await self._classify_page(state)

        # Detect available actions
        state.detected_actions = self._detect_actions(state)

        return state

    async def _extract_dom_state(self, page) -> Dict[str, Any]:
        """Extract structured DOM information via JavaScript."""
        return await page.evaluate("""
            () => {
                const result = {
                    interactables: [],
                    forms: [],
                    navigation: [],
                    main_content: '',
                    headings: [],
                    has_modal: false,
                    error_messages: [],
                    all_text: ''
                };

                // Helper: generate selector
                const getSelector = (el) => {
                    if (el.id) return `#${el.id}`;
                    if (el.getAttribute('data-testid')) return `[data-testid="${el.getAttribute('data-testid')}"]`;
                    if (el.name) return `[name="${el.name}"]`;

                    let path = [];
                    while (el && el.nodeType === Node.ELEMENT_NODE) {
                        let selector = el.tagName.toLowerCase();
                        if (el.className) {
                            const classes = el.className.split(/\\s+/)
                                .filter(c => c && c.length < 30)
                                .slice(0, 2);
                            if (classes.length) selector += '.' + classes.join('.');
                        }
                        const siblings = el.parentElement?.querySelectorAll(`:scope > ${el.tagName}`) || [];
                        if (siblings.length > 1) {
                            const index = Array.from(siblings).indexOf(el) + 1;
                            selector += `:nth-of-type(${index})`;
                        }
                        path.unshift(selector);
                        el = el.parentElement;
                        if (path.length > 3) break;
                    }
                    return path.join(' > ');
                };

                // Helper: check visibility
                const isVisible = (el) => {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style.display !== 'none' &&
                           style.visibility !== 'hidden' &&
                           style.opacity !== '0' &&
                           rect.width > 0 &&
                           rect.height > 0;
                };

                // Extract interactable elements
                const interactables = document.querySelectorAll(
                    'button, a, input, select, textarea, [role="button"], [onclick], [tabindex="0"]'
                );

                let index = 0;
                interactables.forEach(el => {
                    if (!isVisible(el)) return;
                    if (index >= 50) return; // Limit to 50 elements

                    const tag = el.tagName.toLowerCase();
                    let type = tag;
                    if (tag === 'input') type = el.type || 'text';
                    if (el.getAttribute('role') === 'button') type = 'button';

                    result.interactables.push({
                        index: index++,
                        tag: tag,
                        type: type,
                        text: (el.textContent || el.value || el.placeholder || el.getAttribute('aria-label') || '').trim().slice(0, 100),
                        selector: getSelector(el),
                        attributes: {
                            id: el.id || null,
                            name: el.name || null,
                            placeholder: el.placeholder || null,
                            'aria-label': el.getAttribute('aria-label'),
                            href: el.href || null,
                            type: el.type || null,
                        },
                        is_visible: true,
                        is_enabled: !el.disabled
                    });
                });

                // Extract forms
                document.querySelectorAll('form').forEach(form => {
                    const fields = [];
                    form.querySelectorAll('input, select, textarea').forEach(field => {
                        fields.push({
                            tag: field.tagName.toLowerCase(),
                            type: field.type || 'text',
                            name: field.name,
                            placeholder: field.placeholder,
                            required: field.required,
                            selector: getSelector(field)
                        });
                    });

                    const submitBtn = form.querySelector('[type="submit"], button:not([type="button"])');

                    result.forms.push({
                        form_id: form.id || null,
                        action: form.action,
                        method: form.method,
                        fields: fields,
                        submit_button: submitBtn ? getSelector(submitBtn) : null
                    });
                });

                // Extract navigation links
                const navElements = document.querySelectorAll('nav a, header a, [role="navigation"] a');
                navElements.forEach(a => {
                    if (!isVisible(a)) return;
                    if (result.navigation.length >= 20) return;

                    result.navigation.push({
                        text: a.textContent.trim().slice(0, 50),
                        href: a.href,
                        selector: getSelector(a)
                    });
                });

                // Extract main content
                const main = document.querySelector('main, article, [role="main"], .content, #content');
                if (main) {
                    result.main_content = main.textContent.replace(/\\s+/g, ' ').trim().slice(0, 2000);
                } else {
                    result.main_content = document.body.textContent.replace(/\\s+/g, ' ').trim().slice(0, 2000);
                }

                // Extract headings
                document.querySelectorAll('h1, h2, h3').forEach(h => {
                    const text = h.textContent.trim();
                    if (text && result.headings.length < 10) {
                        result.headings.push(text.slice(0, 100));
                    }
                });

                // Detect modal/dialog
                const modal = document.querySelector('[role="dialog"], .modal, .popup, [aria-modal="true"]');
                result.has_modal = modal && isVisible(modal);

                // Detect error messages
                const errors = document.querySelectorAll('.error, .alert-danger, [role="alert"], .error-message');
                errors.forEach(err => {
                    const text = err.textContent.trim();
                    if (text && result.error_messages.length < 5) {
                        result.error_messages.push(text.slice(0, 200));
                    }
                });

                // All text for pattern matching
                result.all_text = document.body.textContent.toLowerCase();

                return result;
            }
        """)

    def _process_interactables(
        self,
        raw_elements: List[Dict]
    ) -> List[InteractableElement]:
        """Process raw interactable data into typed objects."""
        return [
            InteractableElement(
                index=e["index"],
                tag=e["tag"],
                type=e["type"],
                text=e["text"],
                selector=e["selector"],
                attributes={k: v for k, v in e.get("attributes", {}).items() if v},
                is_visible=e.get("is_visible", True),
                is_enabled=e.get("is_enabled", True),
            )
            for e in raw_elements
        ]

    def _process_forms(self, raw_forms: List[Dict]) -> List[FormInfo]:
        """Process raw form data into typed objects."""
        return [
            FormInfo(
                form_id=f.get("form_id"),
                action=f.get("action", ""),
                method=f.get("method", "GET").upper(),
                fields=f.get("fields", []),
                submit_button=f.get("submit_button"),
            )
            for f in raw_forms
        ]

    def _detect_cookie_banner(self, dom_state: Dict) -> bool:
        """Detect if page has a cookie consent banner."""
        all_text = dom_state.get("all_text", "").lower()
        return any(pattern in all_text for pattern in self.COOKIE_PATTERNS)

    def _detect_captcha(self, dom_state: Dict) -> bool:
        """Detect if page has a CAPTCHA."""
        all_text = dom_state.get("all_text", "").lower()
        return any(pattern in all_text for pattern in self.CAPTCHA_PATTERNS)

    async def _classify_page(self, state: PageState) -> str:
        """
        Classify page type based on content and structure.

        Uses heuristics first, falls back to LLM for ambiguous cases.
        """
        # Combine signals for pattern matching
        signals = " ".join([
            state.url.lower(),
            state.title.lower(),
            " ".join(state.headings).lower(),
            state.main_content[:500].lower(),
        ])

        # Score each page type
        scores = {}
        for page_type, patterns in self.PAGE_TYPE_PATTERNS.items():
            score = sum(1 for p in patterns if p in signals)
            if score > 0:
                scores[page_type] = score

        if scores:
            return max(scores, key=scores.get)

        # Check for specific structural patterns
        if state.forms and any(f.fields for f in state.forms):
            # Has form with fields - likely form or login page
            field_types = set()
            for form in state.forms:
                for field in form.fields:
                    field_types.add(field.get("type", "text"))

            if "password" in field_types:
                return "login" if "email" in field_types or "text" in field_types else "form"
            return "form"

        if len(state.interactable_elements) > 20:
            return "list"

        return "unknown"

    def _detect_actions(self, state: PageState) -> List[str]:
        """Detect available actions on the page."""
        actions = []

        # Form-related actions
        if state.forms:
            for form in state.forms:
                if any(f.get("type") == "password" for f in form.fields):
                    actions.append("login")
                elif any(f.get("type") == "search" for f in form.fields):
                    actions.append("search")
                else:
                    actions.append("submit_form")

        # Button actions
        for el in state.interactable_elements:
            text_lower = el.text.lower()
            if el.type in ("button", "submit"):
                if "add to cart" in text_lower or "buy" in text_lower:
                    actions.append("add_to_cart")
                elif "download" in text_lower:
                    actions.append("download")
                elif "subscribe" in text_lower:
                    actions.append("subscribe")

        # Link actions
        for nav in state.navigation_links:
            text_lower = nav.get("text", "").lower()
            if "sign out" in text_lower or "logout" in text_lower:
                actions.append("logout")
            elif "profile" in text_lower or "account" in text_lower:
                actions.append("view_profile")

        # Cookie banner
        if state.has_cookie_banner:
            actions.append("dismiss_cookie_banner")

        # Modal
        if state.has_modal:
            actions.append("close_modal")

        return list(set(actions))

    def compress_for_llm(self, state: PageState, max_tokens: int = 2000) -> str:
        """
        Compress page state into a token-efficient format for LLM consumption.

        Args:
            state: PageState to compress
            max_tokens: Approximate max tokens for output

        Returns:
            Compressed string representation
        """
        lines = []

        # Header
        lines.append(f"URL: {state.url}")
        lines.append(f"Title: {state.title}")
        lines.append(f"Type: {state.page_type}")

        # Alerts/blockers
        if state.has_cookie_banner:
            lines.append("⚠️ Cookie banner detected")
        if state.has_captcha:
            lines.append("⚠️ CAPTCHA detected")
        if state.has_modal:
            lines.append("⚠️ Modal/dialog open")
        if state.error_messages:
            lines.append(f"❌ Errors: {'; '.join(state.error_messages[:2])}")

        lines.append("")

        # Available actions
        if state.detected_actions:
            lines.append(f"Available actions: {', '.join(state.detected_actions)}")
            lines.append("")

        # Interactable elements (numbered for easy reference)
        lines.append("INTERACTABLE ELEMENTS:")
        for el in state.interactable_elements[:30]:  # Limit elements
            el_desc = f"[{el.index}] {el.type}"
            if el.text:
                el_desc += f": \"{el.text[:40]}\""
            if el.attributes.get("placeholder"):
                el_desc += f" (placeholder: {el.attributes['placeholder'][:30]})"
            if not el.is_enabled:
                el_desc += " [disabled]"
            lines.append(el_desc)

        lines.append("")

        # Page headings
        if state.headings:
            lines.append("PAGE HEADINGS:")
            for h in state.headings[:8]:
                lines.append(f"  - {h}")
            lines.append("")

        # Main visible text (critical for reading search results, articles, etc.)
        if state.main_content:
            content_preview = state.main_content[:1500]
            lines.append("VISIBLE PAGE CONTENT:")
            lines.append(content_preview)
            lines.append("")

        # Forms summary
        if state.forms:
            lines.append("FORMS:")
            for i, form in enumerate(state.forms):
                field_summary = ", ".join(
                    f.get("name") or f.get("type", "field")
                    for f in form.fields[:5]
                )
                lines.append(f"  Form {i}: fields=[{field_summary}]")

        # Truncate if needed
        result = "\n".join(lines)
        if len(result) > max_tokens * 4:  # Rough char to token ratio
            result = result[:max_tokens * 4] + "\n...[truncated]"

        return result

    async def suggest_actions(
        self,
        state: PageState,
        goal: str
    ) -> List[Dict[str, Any]]:
        """
        Suggest next actions based on page state and goal.

        Uses heuristics and optionally LLM for complex decisions.

        Args:
            state: Current page state
            goal: The exploration goal

        Returns:
            List of suggested actions with reasoning
        """
        suggestions = []
        goal_lower = goal.lower()

        # Handle blockers first
        if state.has_cookie_banner:
            # Find accept button
            for el in state.interactable_elements:
                if any(p in el.text.lower() for p in ["accept", "agree", "ok", "got it"]):
                    suggestions.append({
                        "action": "click",
                        "element_index": el.index,
                        "selector": el.selector,
                        "reasoning": "Dismiss cookie banner to proceed",
                        "priority": 1
                    })
                    break

        if state.has_modal:
            # Find close button
            for el in state.interactable_elements:
                if any(p in el.text.lower() for p in ["close", "×", "x", "dismiss", "cancel"]):
                    suggestions.append({
                        "action": "click",
                        "element_index": el.index,
                        "selector": el.selector,
                        "reasoning": "Close modal to access main content",
                        "priority": 1
                    })
                    break

        # Goal-specific suggestions
        if "login" in goal_lower and state.page_type == "login":
            # Find login form fields
            for form in state.forms:
                username_field = None
                password_field = None

                for field in form.fields:
                    if field.get("type") == "password":
                        password_field = field
                    elif field.get("type") in ("email", "text"):
                        if not username_field:
                            username_field = field

                if username_field and password_field:
                    suggestions.append({
                        "action": "fill_form",
                        "form_fields": [
                            {"selector": username_field["selector"], "param": "username"},
                            {"selector": password_field["selector"], "param": "password"},
                        ],
                        "submit_selector": form.submit_button,
                        "reasoning": "Fill and submit login form",
                        "priority": 2
                    })

        if "search" in goal_lower:
            for el in state.interactable_elements:
                if el.type == "search" or "search" in el.attributes.get("placeholder", "").lower():
                    suggestions.append({
                        "action": "fill",
                        "element_index": el.index,
                        "selector": el.selector,
                        "reasoning": "Enter search query",
                        "priority": 2
                    })

        return sorted(suggestions, key=lambda x: x.get("priority", 99))
