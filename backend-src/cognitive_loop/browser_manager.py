"""
Browser Manager - Real Chrome with CDP for Cerebro

Launches the user's real Chrome browser via subprocess and connects to it
through Chrome DevTools Protocol (CDP) using Playwright. Supports multi-tab
awareness, pause/resume for human-in-the-loop, and state persistence.

Drop-in replacement for the original Playwright-only browser manager.
"""

import asyncio
import json
import base64
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from playwright.async_api import (
        async_playwright,
        Browser,
        BrowserContext,
        Page,
        Playwright,
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = None
    BrowserContext = None
    Page = None
    Playwright = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# ============================================================================
# Constants
# ============================================================================

STATE_FILE = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "browser_state.json"
CDP_PORT = 9222
CDP_HOST = os.environ.get("CDP_HOST", "localhost")
CDP_URL = f"http://{CDP_HOST}:{CDP_PORT}"
# Chrome rejects CDP requests whose Host header isn't localhost or an IP.
# When connecting from Docker via host.docker.internal, we must override it.
# Include the port so Chrome generates correct webSocketDebuggerUrl values.
_CDP_HEADERS = {"Host": f"localhost:{CDP_PORT}"} if CDP_HOST not in ("localhost", "127.0.0.1") else {}
CHROME_PROFILE_DIR = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "chrome_profile"

# JavaScript injected into every page to show Cerebro's control border
CEREBRO_BORDER_JS = """
(function() {
    if (document.getElementById('cerebro-ctrl-border')) return;
    var el = document.createElement('div');
    el.id = 'cerebro-ctrl-border';
    el.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:2147483647;outline:3px solid rgba(139,92,246,0.55);outline-offset:-3px;contain:strict;';
    document.documentElement.appendChild(el);
    var badge = document.createElement('div');
    badge.id = 'cerebro-ctrl-badge';
    badge.style.cssText = 'position:fixed;top:6px;left:50%;transform:translateX(-50%);background:rgba(139,92,246,0.9);color:white;font:600 9px system-ui,sans-serif;padding:2px 10px;border-radius:0 0 6px 6px;z-index:2147483647;pointer-events:none;letter-spacing:1.5px;backdrop-filter:blur(8px);box-shadow:0 2px 8px rgba(139,92,246,0.3);';
    badge.textContent = 'CEREBRO';
    document.documentElement.appendChild(badge);
})();
"""

# Common Chrome install locations on Windows
CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]

# Common Chromium/Chrome install locations on Linux
CHROME_PATHS_LINUX = [
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
]


# ============================================================================
# BrowserManager
# ============================================================================

class BrowserManager:
    """
    Manages a real Chrome browser launched via subprocess and connected
    through CDP with Playwright. Supports multi-tab tracking, pause/resume
    for human-in-the-loop, and state persistence.
    """

    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._active_page: Optional[Page] = None
        self._chrome_process: Optional[asyncio.subprocess.Process] = None
        self._chrome_pid: Optional[int] = None
        self._lock = asyncio.Lock()
        self._started_at: Optional[str] = None
        self._restart_count: int = 0
        self._last_url: Optional[str] = None

        # Multi-tab tracking
        self._pages: Dict[str, Page] = {}  # tab_id -> Page
        self._tab_counter: int = 0

        # Pause/Resume for HITL
        self._paused: bool = False
        self._pause_reason: str = ""
        self._resume_event: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------
    # Chrome discovery
    # ------------------------------------------------------------------

    @staticmethod
    def find_chrome_path() -> Optional[str]:
        """Locate Chrome/Chromium on the system. Returns path or None."""
        if sys.platform.startswith("linux"):
            for p in CHROME_PATHS_LINUX:
                if os.path.isfile(p):
                    return p
        for p in CHROME_PATHS:
            if os.path.isfile(p):
                return p
        return None

    # ------------------------------------------------------------------
    # CDP availability check
    # ------------------------------------------------------------------

    async def _check_cdp_available(self) -> bool:
        """Check whether a CDP endpoint is already responding on the port."""
        if HTTPX_AVAILABLE:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(
                        f"{CDP_URL}/json/version", headers=_CDP_HEADERS
                    )
                    return resp.status_code == 200
            except Exception:
                return False
        else:
            # Fallback: try a raw TCP connection
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(CDP_HOST, CDP_PORT), timeout=2.0
                )
                writer.close()
                await writer.wait_closed()
                return True
            except Exception:
                return False

    # ------------------------------------------------------------------
    # Launch / connect
    # ------------------------------------------------------------------

    async def _launch_chrome(self) -> None:
        """Launch real Chrome with remote debugging enabled.

        If CDP_HOST points to a remote host (not localhost), skip the local
        launch and poll for the remote CDP endpoint to become available.
        """
        # Remote CDP mode: Chrome runs on the host, container just connects
        if CDP_HOST not in ("localhost", "127.0.0.1"):
            print(f"[BrowserManager] Remote CDP mode — waiting for Chrome on {CDP_HOST}:{CDP_PORT}")
            for _ in range(30):
                if await self._check_cdp_available():
                    print(f"[BrowserManager] Remote CDP available at {CDP_URL}")
                    return
                await asyncio.sleep(0.5)
            raise RuntimeError(
                f"Remote Chrome CDP did not respond at {CDP_URL} within 15 seconds. "
                "Ensure Chrome is running on the host with --remote-debugging-port=9222"
            )

        chrome_path = self.find_chrome_path()
        if chrome_path is None:
            searched = CHROME_PATHS_LINUX if sys.platform.startswith("linux") else CHROME_PATHS
            raise RuntimeError(
                "Chrome not found. Searched:\n  "
                + "\n  ".join(searched)
            )

        CHROME_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

        args = [
            chrome_path,
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={CHROME_PROFILE_DIR}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-infobars",
            "--window-size=1280,900",
        ]

        print(f"[BrowserManager] Launching Chrome: {chrome_path}")
        # Use subprocess.Popen instead of asyncio.create_subprocess_exec
        # because Windows SelectorEventLoop (required for Playwright) doesn't
        # support asyncio subprocess creation.
        self._chrome_process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._chrome_pid = self._chrome_process.pid
        print(f"[BrowserManager] Chrome started (PID {self._chrome_pid})")

        # Wait for CDP to become available (up to 15 seconds)
        for _ in range(30):
            if await self._check_cdp_available():
                return
            await asyncio.sleep(0.5)

        raise RuntimeError(
            f"Chrome launched but CDP did not respond on port {CDP_PORT} within 15 seconds"
        )

    async def _get_cdp_ws_url(self) -> str:
        """Fetch the WebSocket debugger URL from Chrome, rewriting the host
        so it's reachable from inside the Docker container."""
        if HTTPX_AVAILABLE:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{CDP_URL}/json/version", headers=_CDP_HEADERS
                )
                data = resp.json()
        else:
            raise RuntimeError("httpx is required for remote CDP connections")

        ws_url = data.get("webSocketDebuggerUrl", "")
        if not ws_url:
            raise RuntimeError("Chrome did not return a webSocketDebuggerUrl")

        # Chrome returns ws://localhost:9222/... but inside Docker we need
        # ws://host.docker.internal:9222/... to reach the host.
        if CDP_HOST not in ("localhost", "127.0.0.1"):
            ws_url = ws_url.replace("ws://localhost", f"ws://{CDP_HOST}")
            ws_url = ws_url.replace("ws://127.0.0.1", f"ws://{CDP_HOST}")
        return ws_url

    async def _connect_cdp(self) -> None:
        """Connect Playwright to Chrome via CDP."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            )

        self._playwright = await async_playwright().start()

        # For remote CDP (Docker→host), fetch WS URL and rewrite the host
        # so Playwright connects to the correct address.
        if CDP_HOST not in ("localhost", "127.0.0.1"):
            ws_url = await self._get_cdp_ws_url()
            self._browser = await self._playwright.chromium.connect_over_cdp(ws_url)
        else:
            self._browser = await self._playwright.chromium.connect_over_cdp(CDP_URL)

        # Grab the default context (Chrome's real profile context)
        contexts = self._browser.contexts
        if contexts:
            self._context = contexts[0]
        else:
            self._context = await self._browser.new_context()

        # Index existing pages/tabs
        self._pages.clear()
        self._tab_counter = 0
        for page in self._context.pages:
            tab_id = self._next_tab_id()
            self._pages[tab_id] = page
            self._active_page = page
            page.on("close", lambda p=page: self._on_page_closed(p))

        # If no pages exist, open one
        if not self._pages:
            page = await self._context.new_page()
            tab_id = self._next_tab_id()
            self._pages[tab_id] = page
            self._active_page = page
            page.on("close", lambda p=page: self._on_page_closed(p))

        # Listen for new pages opened in this context
        self._context.on("page", self._on_new_page)

        # Inject Cerebro control border into all existing pages
        for page in self._pages.values():
            try:
                if not page.is_closed():
                    await page.evaluate(CEREBRO_BORDER_JS)
            except Exception:
                pass  # page may not be ready yet

        self._started_at = datetime.now(timezone.utc).isoformat()
        self._save_state()
        print(
            f"[BrowserManager] Connected via CDP — "
            f"{len(self._pages)} tab(s) tracked"
        )

    def _on_new_page(self, page: Page) -> None:
        """Handler when a new page/tab is opened in the context."""
        tab_id = self._next_tab_id()
        self._pages[tab_id] = page
        self._active_page = page
        page.on("close", lambda p=page: self._on_page_closed(p))
        # Inject control border when page finishes loading
        async def _inject_border():
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
                await page.evaluate(CEREBRO_BORDER_JS)
            except Exception:
                pass
        try:
            asyncio.get_event_loop().create_task(_inject_border())
        except Exception:
            pass
        print(f"[BrowserManager] New tab detected: {tab_id}")

    def _on_page_closed(self, page: Page) -> None:
        """Handler when a page/tab is closed."""
        tab_id = self._tab_id_for(page)
        if tab_id and tab_id in self._pages:
            del self._pages[tab_id]
            print(f"[BrowserManager] Tab closed: {tab_id}")

        if self._active_page is page:
            self._active_page = self._pick_fallback_page()

    def _pick_fallback_page(self) -> Optional[Page]:
        """Pick a fallback page from remaining open tabs."""
        if self._pages:
            return list(self._pages.values())[-1]
        return None

    def _next_tab_id(self) -> str:
        self._tab_counter += 1
        return f"tab_{self._tab_counter}"

    def _tab_id_for(self, page: Page) -> Optional[str]:
        """Find the tab_id for a given Page object."""
        for tid, p in self._pages.items():
            if p is page:
                return tid
        return None

    async def _launch(self) -> None:
        """Full launch sequence: start Chrome if needed, then connect."""
        cdp_ready = await self._check_cdp_available()

        if cdp_ready:
            print("[BrowserManager] CDP already available — attaching to existing Chrome")
        else:
            await self._launch_chrome()

        await self._connect_cdp()

    # ------------------------------------------------------------------
    # Public API — existing interface (drop-in compatible)
    # ------------------------------------------------------------------

    async def ensure_running(self) -> None:
        """Ensure the browser is running. Launch or restart if needed."""
        async with self._lock:
            if self.is_alive():
                return

            if self._browser is not None:
                self._restart_count += 1
                print(
                    f"[BrowserManager] Browser died, restarting "
                    f"(attempt #{self._restart_count})..."
                )
                await self._cleanup()

            await self._launch()

    def is_alive(self) -> bool:
        """Check if the browser connection is still active."""
        if self._browser is None:
            return False
        try:
            return self._browser.is_connected()
        except Exception:
            return False

    async def get_browser(self) -> Browser:
        """Get the browser instance, ensuring it's running."""
        await self.ensure_running()
        return self._browser

    async def get_page(self) -> Page:
        """Get the active page (most recently used tab)."""
        await self.ensure_running()
        if self._active_page is None or self._active_page.is_closed():
            self._active_page = self._pick_fallback_page()
        if self._active_page is None:
            self._active_page = await self._context.new_page()
            tab_id = self._next_tab_id()
            self._pages[tab_id] = self._active_page
            self._active_page.on(
                "close", lambda p=self._active_page: self._on_page_closed(p)
            )
        return self._active_page

    async def navigate(
        self, url: str, wait_until: str = "domcontentloaded"
    ) -> Dict[str, Any]:
        """Navigate the active page to a URL and return page info."""
        await self.ensure_running()
        page = await self.get_page()
        print(f"[BrowserManager] Navigating to: {url}")

        try:
            response = await page.goto(url, wait_until=wait_until, timeout=30000)
            await page.wait_for_timeout(500)

            # Re-inject Cerebro control border after navigation
            try:
                await page.evaluate(CEREBRO_BORDER_JS)
            except Exception:
                pass

            title = await page.title()
            current_url = page.url
            self._last_url = current_url
            self._active_page = page
            self._save_state()

            return {
                "success": True,
                "url": current_url,
                "title": title,
                "status": response.status if response else None,
            }
        except Exception as e:
            print(f"[BrowserManager] Navigation error: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
            }

    async def screenshot(self, full_page: bool = False) -> Optional[str]:
        """Take a screenshot of the active page. Returns base64 PNG."""
        await self.ensure_running()
        page = await self.get_page()
        try:
            png_bytes = await page.screenshot(full_page=full_page, type="png", timeout=5000)
            return base64.b64encode(png_bytes).decode("utf-8")
        except Exception as e:
            print(f"[BrowserManager] Screenshot error: {e}")
            return None

    async def get_status(self) -> Dict[str, Any]:
        """Get current browser status including tabs and pause state."""
        alive = self.is_alive()

        status = {
            "running": alive,
            "started_at": self._started_at,
            "restart_count": self._restart_count,
            "cdp_port": CDP_PORT,
            "cdp_url": CDP_URL if alive else None,
            "chrome_pid": self._chrome_pid,
            "paused": self._paused,
            "pause_reason": self._pause_reason if self._paused else None,
            "tab_count": len(self._pages),
            "tabs": [],
        }

        if alive:
            for tab_id, page in list(self._pages.items()):
                try:
                    tab_info = {
                        "tab_id": tab_id,
                        "url": page.url,
                        "title": await page.title(),
                        "is_active": page is self._active_page,
                    }
                except Exception:
                    tab_info = {
                        "tab_id": tab_id,
                        "url": "unknown",
                        "title": None,
                        "is_active": page is self._active_page,
                    }
                status["tabs"].append(tab_info)

            if self._active_page:
                try:
                    status["current_url"] = self._active_page.url
                    status["title"] = await self._active_page.title()
                except Exception:
                    status["current_url"] = self._last_url
                    status["title"] = None

        return status

    async def new_page(self) -> Page:
        """Open a new tab in the browser (existing interface)."""
        await self.ensure_running()
        page = await self._context.new_page()
        # _on_new_page handler will register it
        return page

    async def close_page(self, page: Page) -> None:
        """Close a specific page/tab (existing interface)."""
        try:
            await page.close()
            # _on_page_closed handler handles cleanup
        except Exception:
            pass

    async def shutdown(self) -> None:
        """Gracefully shut down: disconnect Playwright, kill Chrome."""
        print("[BrowserManager] Shutting down browser...")
        async with self._lock:
            await self._cleanup()
            self._save_state(running=False)
        print("[BrowserManager] Browser shut down")

    # ------------------------------------------------------------------
    # New multi-tab methods
    # ------------------------------------------------------------------

    async def get_all_pages(self) -> List[Dict[str, Any]]:
        """
        Return info about all open tabs.
        Returns list of dicts: [{url, title, tab_id}, ...]
        """
        await self.ensure_running()
        result = []
        for tab_id, page in list(self._pages.items()):
            try:
                if page.is_closed():
                    continue
                result.append({
                    "tab_id": tab_id,
                    "url": page.url,
                    "title": await page.title(),
                })
            except Exception:
                result.append({
                    "tab_id": tab_id,
                    "url": "unknown",
                    "title": None,
                })
        return result

    async def get_active_page(self) -> Optional[Page]:
        """Return the most recently used page."""
        await self.ensure_running()
        if self._active_page and not self._active_page.is_closed():
            return self._active_page
        self._active_page = self._pick_fallback_page()
        return self._active_page

    def get_page_by_url(self, pattern: str) -> Optional[Page]:
        """Find a tab whose URL contains the given substring."""
        pattern_lower = pattern.lower()
        for page in self._pages.values():
            try:
                if not page.is_closed() and pattern_lower in page.url.lower():
                    self._active_page = page
                    return page
            except Exception:
                continue
        return None

    async def open_new_tab(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Open a new tab, optionally navigating to a URL."""
        await self.ensure_running()
        page = await self._context.new_page()
        # _on_new_page handler registers it and sets it active
        tab_id = self._tab_id_for(page)

        result = {"tab_id": tab_id, "url": "about:blank", "title": ""}

        if url:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(500)
                # Re-inject Cerebro control border
                try:
                    await page.evaluate(CEREBRO_BORDER_JS)
                except Exception:
                    pass
                result["url"] = page.url
                result["title"] = await page.title()
            except Exception as e:
                result["error"] = str(e)

        self._save_state()
        return result

    async def close_tab(self, tab_id: str) -> bool:
        """Close a tab by its tab_id. Returns True if closed."""
        page = self._pages.get(tab_id)
        if page is None:
            return False
        try:
            await page.close()
            # _on_page_closed handler handles cleanup
            return True
        except Exception:
            return False

    async def get_page_content(self, page: Optional[Page] = None) -> Optional[str]:
        """Return the text content of a page (default: active page)."""
        await self.ensure_running()
        if page is None:
            page = await self.get_page()
        try:
            return await page.inner_text("body")
        except Exception as e:
            print(f"[BrowserManager] get_page_content error: {e}")
            return None

    # ------------------------------------------------------------------
    # Pause / Resume (HITL)
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self, reason: str = "") -> None:
        """Pause the browser manager. The OODA loop should check is_paused."""
        self._paused = True
        self._pause_reason = reason
        print(f"[BrowserManager] PAUSED — {reason or 'no reason given'}")

    def resume(self) -> None:
        """Resume from paused state."""
        self._paused = False
        self._pause_reason = ""
        if self._resume_event is not None:
            self._resume_event.set()
        print("[BrowserManager] RESUMED")

    async def wait_for_user_action(
        self, description: str = "", timeout: float = 300
    ) -> bool:
        """
        Pause and wait for the user to call resume().

        Args:
            description: What the user should do.
            timeout: Max seconds to wait (default 5 minutes).

        Returns:
            True if resumed, False if timed out.
        """
        self.pause(reason=description)
        self._resume_event = asyncio.Event()

        try:
            await asyncio.wait_for(self._resume_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            print(f"[BrowserManager] wait_for_user_action timed out after {timeout}s")
            self._paused = False
            self._pause_reason = ""
            return False
        finally:
            self._resume_event = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _cleanup(self) -> None:
        """Clean up Playwright connection and kill Chrome process."""
        # Disconnect Playwright (does NOT close Chrome by itself)
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass

        # Kill Chrome subprocess if we launched it
        if self._chrome_process is not None:
            try:
                self._chrome_process.terminate()
                try:
                    await asyncio.wait_for(self._chrome_process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._chrome_process.kill()
            except Exception:
                pass

            # Fallback: kill by PID on Windows using taskkill
            if self._chrome_pid is not None:
                try:
                    import subprocess
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(self._chrome_pid)],
                        capture_output=True, timeout=5
                    )
                except Exception:
                    pass

        self._browser = None
        self._context = None
        self._active_page = None
        self._playwright = None
        self._chrome_process = None
        self._chrome_pid = None
        self._pages.clear()
        self._paused = False
        self._pause_reason = ""

    def _save_state(self, running: bool = True) -> None:
        """Persist browser state to disk."""
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            tabs = []
            for tab_id, page in list(self._pages.items()):
                try:
                    tabs.append({
                        "tab_id": tab_id,
                        "url": page.url if not page.is_closed() else "closed",
                    })
                except Exception:
                    tabs.append({"tab_id": tab_id, "url": "unknown"})

            state = {
                "running": running,
                "started_at": self._started_at,
                "restart_count": self._restart_count,
                "cdp_port": CDP_PORT,
                "chrome_pid": self._chrome_pid,
                "last_url": self._last_url,
                "tab_count": len(self._pages),
                "tabs": tabs,
                "paused": self._paused,
                "pause_reason": self._pause_reason,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception as e:
            print(f"[BrowserManager] Failed to save state: {e}")


# ============================================================================
# Singleton
# ============================================================================

_browser_manager: Optional[BrowserManager] = None


def get_browser_manager() -> BrowserManager:
    """Get or create the singleton BrowserManager instance."""
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = BrowserManager()
    return _browser_manager
