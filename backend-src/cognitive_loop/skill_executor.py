"""
Skill Executor - Runs Playwright in a separate process to avoid uvicorn conflicts.

This solves the Windows asyncio + Playwright NotImplementedError issue by
completely isolating Playwright from uvicorn's event loop.

Supports human-in-the-loop for authentication/CAPTCHA.
"""

import json
import sys
import os
import time
from pathlib import Path

# Signal file paths for human-in-the-loop
SIGNAL_DIR = Path(os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data"))) / "cerebro" / "signals"
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)


def detect_auth_needed(page) -> dict:
    """
    Detect if the page requires human authentication.

    Returns dict with:
        - needs_auth: bool
        - reason: str (login, captcha, blocked, etc.)
        - url: current URL
    """
    url = page.url.lower()
    title = page.title().lower() if page.title() else ""

    # URL-based detection
    auth_url_patterns = [
        "login", "signin", "sign-in", "sign_in",
        "auth", "authenticate", "oauth",
        "captcha", "challenge",
        "/prefs/apps",  # Reddit app creation
        "accounts.google",
        "login.microsoftonline"
    ]

    for pattern in auth_url_patterns:
        if pattern in url:
            return {
                "needs_auth": True,
                "reason": f"URL contains '{pattern}'",
                "url": page.url
            }

    # Title-based detection
    auth_title_patterns = [
        "log in", "login", "sign in", "signin",
        "captcha", "verify", "robot",
        "access denied", "blocked"
    ]

    for pattern in auth_title_patterns:
        if pattern in title:
            return {
                "needs_auth": True,
                "reason": f"Page title contains '{pattern}'",
                "url": page.url
            }

    # Element-based detection
    try:
        # Check for password fields (strong indicator of login page)
        password_fields = page.query_selector_all('input[type="password"]')
        if password_fields:
            return {
                "needs_auth": True,
                "reason": "Page has password input field",
                "url": page.url
            }

        # Check for CAPTCHA
        captcha_selectors = [
            'iframe[src*="recaptcha"]',
            'iframe[src*="captcha"]',
            '.g-recaptcha',
            '#captcha',
            '[data-sitekey]',
            'iframe[title*="reCAPTCHA"]'
        ]
        for selector in captcha_selectors:
            if page.query_selector(selector):
                return {
                    "needs_auth": True,
                    "reason": "CAPTCHA detected",
                    "url": page.url
                }
    except:
        pass

    return {"needs_auth": False, "reason": None, "url": page.url}


def wait_for_human_auth(session_id: str, page, reason: str, timeout_seconds: int = 300) -> bool:
    """
    Wait for human to complete authentication.

    Creates a signal file and waits for user to complete auth.
    Returns True if auth completed, False if timeout.
    """
    signal_file = SIGNAL_DIR / f"auth_needed_{session_id}.json"
    continue_file = SIGNAL_DIR / f"auth_continue_{session_id}.json"

    # Write signal that we need auth
    signal_data = {
        "session_id": session_id,
        "status": "waiting",
        "reason": reason,
        "url": page.url,
        "timestamp": time.time(),
        "message": "Please complete authentication in the browser window, then click 'Done' in Cerebro."
    }

    with open(signal_file, "w") as f:
        json.dump(signal_data, f)

    print(f"[AUTH] Waiting for human authentication: {reason}", file=sys.stderr)
    print(f"[AUTH] Signal file: {signal_file}", file=sys.stderr)

    # Also output to stdout for the parent process to detect
    print(json.dumps({
        "status": "auth_needed",
        "session_id": session_id,
        "reason": reason,
        "url": page.url
    }), flush=True)

    # Wait for continue signal
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if continue_file.exists():
            print("[AUTH] Continue signal received!", file=sys.stderr)
            try:
                continue_file.unlink()
            except:
                pass
            try:
                signal_file.unlink()
            except:
                pass
            return True

        # Also check if page navigated away from auth (user completed it)
        new_auth_check = detect_auth_needed(page)
        if not new_auth_check["needs_auth"]:
            print("[AUTH] Auth page no longer detected, assuming user completed auth", file=sys.stderr)
            try:
                signal_file.unlink()
            except:
                pass
            return True

        time.sleep(1)

    print("[AUTH] Timeout waiting for authentication", file=sys.stderr)
    try:
        signal_file.unlink()
    except:
        pass
    return False


def execute_skill_in_process(skill_json: str, parameters: dict, session_id: str = None) -> dict:
    """
    Execute a skill using Playwright in an isolated process.

    Args:
        skill_json: JSON string of the skill
        parameters: Parameter dict to substitute
        session_id: Unique session ID for human-in-the-loop signals

    Returns:
        Dict with success, output, error, steps_completed, total_steps, duration_ms
    """
    import re

    if not session_id:
        session_id = f"skill_{int(time.time() * 1000)}"

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {
            "success": False,
            "error": "Playwright not installed",
            "steps_completed": 0,
            "total_steps": 0,
            "duration_ms": 0
        }

    skill = json.loads(skill_json)
    steps = skill.get("steps", [])
    start_time = time.time()
    steps_completed = 0
    output = None
    extractions = {}  # Accumulate multiple extract results

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            for i, step in enumerate(steps):
                action = step.get("action", "").lower()
                selector = step.get("selector")
                value = step.get("value")
                timeout = step.get("timeout_ms", 10000)
                wait_for = step.get("wait_for")

                # Substitute parameters
                if selector and parameters:
                    for k, v in parameters.items():
                        selector = selector.replace(f"{{{{{k}}}}}", v)
                if value and parameters:
                    for k, v in parameters.items():
                        value = value.replace(f"{{{{{k}}}}}", v)

                try:
                    if action == "navigate":
                        page.goto(value, wait_until="domcontentloaded", timeout=timeout)

                        # Check for auth after navigation
                        time.sleep(1)  # Brief pause to let page settle
                        auth_check = detect_auth_needed(page)
                        if auth_check["needs_auth"]:
                            print(f"[AUTH] Authentication needed: {auth_check['reason']}", file=sys.stderr)
                            if not wait_for_human_auth(session_id, page, auth_check["reason"]):
                                duration = (time.time() - start_time) * 1000
                                browser.close()
                                return {
                                    "success": False,
                                    "error": f"Authentication timeout: {auth_check['reason']}",
                                    "auth_needed": True,
                                    "auth_reason": auth_check["reason"],
                                    "steps_completed": steps_completed,
                                    "total_steps": len(steps),
                                    "duration_ms": duration
                                }

                    elif action == "click":
                        page.click(selector, timeout=timeout)

                    elif action in ("fill", "type"):
                        page.fill(selector, value, timeout=timeout)

                    elif action == "press":
                        page.press(selector or "body", value, timeout=timeout)

                    elif action == "wait":
                        if wait_for:
                            page.wait_for_selector(wait_for, timeout=timeout)
                        else:
                            time.sleep(timeout / 1000)

                    elif action == "extract":
                        # Use step description or selector as key for named extractions
                        extract_key = step.get("description", f"extract_{i}")
                        print(f"[EXTRACT] Selector: {selector}", file=sys.stderr)

                        if selector:
                            elements = page.query_selector_all(selector)
                            print(f"[EXTRACT] Found {len(elements)} elements", file=sys.stderr)
                            if elements:
                                extracted = []
                                for el in elements[:20]:
                                    text = el.text_content()
                                    if text:
                                        # Clean up whitespace
                                        clean_text = re.sub(r'\s+', ' ', text).strip()
                                        if clean_text:
                                            extracted.append(clean_text)
                                print(f"[EXTRACT] Extracted {len(extracted)} items: {extracted[:3]}...", file=sys.stderr)
                                extractions[extract_key] = extracted
                                output = extracted
                            else:
                                print(f"[EXTRACT] No elements found for selector: {selector}", file=sys.stderr)
                        else:
                            html = page.content()
                            extractions[extract_key] = html[:500]
                            output = html[:500]

                    elif action == "screenshot":
                        output = page.screenshot()

                    elif action == "select":
                        page.select_option(selector, value, timeout=timeout)

                    elif action == "scroll":
                        if selector:
                            page.locator(selector).scroll_into_view_if_needed()
                        else:
                            page.evaluate(f"window.scrollBy(0, {value or 500})")

                    elif action == "hover":
                        page.hover(selector, timeout=timeout)

                    elif action in ("eval", "execute_js", "extract_js"):
                        extract_key = step.get("description", f"eval_{i}")
                        if value:
                            result = page.evaluate(value)
                            if result:
                                extractions[extract_key] = result
                                output = result
                                print(f"[EVAL] Result: {str(result)[:200]}", file=sys.stderr)

                    elif action == "wait_for_auth":
                        # Explicit action to wait for human authentication
                        if not wait_for_human_auth(session_id, page, step.get("description", "Manual authentication required")):
                            duration = (time.time() - start_time) * 1000
                            browser.close()
                            return {
                                "success": False,
                                "error": "Authentication timeout",
                                "auth_needed": True,
                                "steps_completed": steps_completed,
                                "total_steps": len(steps),
                                "duration_ms": duration
                            }

                    steps_completed += 1

                    # Optional wait after action
                    if wait_for and action != "wait":
                        try:
                            page.wait_for_selector(wait_for, timeout=2000)
                        except:
                            pass

                except Exception as e:
                    # Check if error is due to navigation to auth page
                    auth_check = detect_auth_needed(page)
                    if auth_check["needs_auth"]:
                        print(f"[AUTH] Error likely due to auth page: {auth_check['reason']}", file=sys.stderr)
                        if wait_for_human_auth(session_id, page, auth_check["reason"]):
                            # Retry the step after auth
                            continue

                    duration = (time.time() - start_time) * 1000
                    browser.close()
                    return {
                        "success": False,
                        "error": f"Step {i+1} failed: {step.get('description', action)} - {str(e)}",
                        "steps_completed": steps_completed,
                        "total_steps": len(steps),
                        "duration_ms": duration
                    }

            # Success - keep browser open briefly to see result
            time.sleep(1)
            browser.close()

            duration = (time.time() - start_time) * 1000

            # Combine all extractions into output
            final_output = output
            print(f"[RESULT] Extractions: {len(extractions)}, output type: {type(output)}", file=sys.stderr)
            if extractions:
                if len(extractions) == 1:
                    final_output = list(extractions.values())[0]
                else:
                    final_output = extractions
            print(f"[RESULT] Final output type: {type(final_output)}, is None: {final_output is None}", file=sys.stderr)

            result = {
                "success": True,
                "output": final_output,
                "error": None,
                "steps_completed": steps_completed,
                "total_steps": len(steps),
                "duration_ms": duration
            }
            print(f"[RESULT] Returning: success={result['success']}, has_output={result['output'] is not None}", file=sys.stderr)
            return result

    except Exception as e:
        duration = (time.time() - start_time) * 1000
        return {
            "success": False,
            "error": str(e),
            "steps_completed": steps_completed,
            "total_steps": len(steps),
            "duration_ms": duration
        }


if __name__ == "__main__":
    # Called as subprocess: python skill_executor.py <skill_json> <params_json> [session_id]
    if len(sys.argv) >= 2:
        skill_json = sys.argv[1]
        params_json = sys.argv[2] if len(sys.argv) > 2 else "{}"
        session_id = sys.argv[3] if len(sys.argv) > 3 else None
        parameters = json.loads(params_json)

        result = execute_skill_in_process(skill_json, parameters, session_id)
        print(json.dumps(result))
    else:
        print(json.dumps({"success": False, "error": "Missing skill argument"}))
