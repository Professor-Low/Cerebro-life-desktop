"""
Cerebro Usage Tracker — tracks Claude API token usage from Claude Code CLI.

Stores daily aggregates per model in a JSON file.
Thread-safe with cross-platform file locking.
"""

import json
import logging
import platform
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# Cross-platform file locking
if platform.system() == "Windows":
    import msvcrt

    def _lock_shared(f):
        """Acquire a shared (read) lock on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def _lock_exclusive(f):
        """Acquire an exclusive (write) lock on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock(f):
        """Release a file lock on Windows."""
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _lock_shared(f):
        """Acquire a shared (read) lock on Unix."""
        fcntl.flock(f, fcntl.LOCK_SH)

    def _lock_exclusive(f):
        """Acquire an exclusive (write) lock on Unix."""
        fcntl.flock(f, fcntl.LOCK_EX)

    def _unlock(f):
        """Release a file lock on Unix."""
        fcntl.flock(f, fcntl.LOCK_UN)

logger = logging.getLogger("usage_tracker")


class UsageTracker:
    """Tracks Claude API usage from stream-json result events."""

    # Pricing per million tokens (USD) — updated March 2026
    MODEL_PRICING = {
        "claude-opus-4-6":               {"input": 15.00, "output": 75.00, "cache_read": 1.50, "cache_write": 18.75},
        "claude-sonnet-4-6":             {"input": 3.00,  "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
        "claude-haiku-4-5-20251001":     {"input": 0.80,  "output": 4.00,  "cache_read": 0.08, "cache_write": 1.00},
        # Fallback for unknown models
        "_default":                      {"input": 3.00,  "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    }

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir) / "cerebro" / "usage"
        self.usage_file = self.data_dir / "daily_usage.json"
        self._ensure_storage()

    def _ensure_storage(self) -> bool:
        """Ensure the storage directory and file exist. Returns True if ready."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if not self.usage_file.exists():
                self._write_data({"days": {}, "version": 1})
            return True
        except Exception as e:
            logger.error(f"[UsageTracker] Failed to initialize storage: {e}")
            return False

    def _read_data(self) -> dict:
        """Read usage data. Returns None on transient errors to prevent data loss."""
        try:
            with open(self.usage_file, "r") as f:
                _lock_shared(f)
                data = json.load(f)
                _unlock(f)
                return data
        except FileNotFoundError:
            logger.warning("[UsageTracker] File not found, returning empty dataset")
            return {"days": {}, "version": 1}
        except json.JSONDecodeError:
            logger.warning("[UsageTracker] Corrupted JSON, returning empty dataset")
            return {"days": {}, "version": 1}
        except OSError as e:
            # Transient I/O error (NAS hiccup, etc.) — return None to prevent
            # record() from overwriting good data with an empty dataset
            logger.error(f"[UsageTracker] Transient read error, skipping write: {e}")
            return None

    def _write_data(self, data: dict):
        tmp = self.usage_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            _lock_exclusive(f)
            json.dump(data, f, indent=2)
            _unlock(f)
        tmp.rename(self.usage_file)

    def _get_pricing(self, model: str) -> dict:
        """Get pricing for a model, with fuzzy matching."""
        if model in self.MODEL_PRICING:
            return self.MODEL_PRICING[model]
        # Fuzzy match: check if model contains known keys
        for key in self.MODEL_PRICING:
            if key != "_default" and key in model:
                return self.MODEL_PRICING[key]
        # Match by family
        if "opus" in model.lower():
            return self.MODEL_PRICING["claude-opus-4-6"]
        if "haiku" in model.lower():
            return self.MODEL_PRICING["claude-haiku-4-5-20251001"]
        if "sonnet" in model.lower():
            return self.MODEL_PRICING["claude-sonnet-4-6"]
        return self.MODEL_PRICING["_default"]

    def record(self, model: str, input_tokens: int = 0, output_tokens: int = 0,
               cache_read_tokens: int = 0, cache_write_tokens: int = 0,
               duration_ms: int = 0, source: str = "chat",
               cost_usd: Optional[float] = None):
        """Record a single API call's usage. Resilient — never raises."""
        try:
            # Ensure storage exists (handles NAS remount, etc.)
            if not self.usage_file.exists():
                if not self._ensure_storage():
                    return

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            data = self._read_data()
            if data is None:
                logger.warning("[UsageTracker] Skipping record due to transient read error")
                return

            if today not in data["days"]:
                data["days"][today] = {"models": {}, "totals": {
                    "input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cache_write_tokens": 0,
                    "cost_usd": 0.0, "requests": 0, "duration_ms": 0
                }}

            day = data["days"][today]

            # Calculate cost if not provided
            if cost_usd is None:
                pricing = self._get_pricing(model)
                cost_usd = (
                    (input_tokens / 1_000_000) * pricing["input"] +
                    (output_tokens / 1_000_000) * pricing["output"] +
                    (cache_read_tokens / 1_000_000) * pricing["cache_read"] +
                    (cache_write_tokens / 1_000_000) * pricing["cache_write"]
                )

            # Model-level stats
            if model not in day["models"]:
                day["models"][model] = {
                    "input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cache_write_tokens": 0,
                    "cost_usd": 0.0, "requests": 0, "duration_ms": 0,
                    "by_source": {}
                }
            m = day["models"][model]
            m["input_tokens"] += input_tokens
            m["output_tokens"] += output_tokens
            m["cache_read_tokens"] += cache_read_tokens
            m["cache_write_tokens"] += cache_write_tokens
            m["cost_usd"] += cost_usd
            m["requests"] += 1
            m["duration_ms"] += duration_ms
            # Track by source (chat vs agent)
            if source not in m["by_source"]:
                m["by_source"][source] = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            m["by_source"][source]["requests"] += 1
            m["by_source"][source]["input_tokens"] += input_tokens
            m["by_source"][source]["output_tokens"] += output_tokens
            m["by_source"][source]["cost_usd"] += cost_usd

            # Day-level totals
            day["totals"]["input_tokens"] += input_tokens
            day["totals"]["output_tokens"] += output_tokens
            day["totals"]["cache_read_tokens"] += cache_read_tokens
            day["totals"]["cache_write_tokens"] += cache_write_tokens
            day["totals"]["cost_usd"] += cost_usd
            day["totals"]["requests"] += 1
            day["totals"]["duration_ms"] += duration_ms

            self._write_data(data)
            logger.info(
                f"[UsageTracker] Recorded: {model} | {source} | "
                f"in={input_tokens} out={output_tokens} | ${cost_usd:.4f}"
            )
        except Exception as e:
            logger.error(f"[UsageTracker] Failed to record usage: {e}")

    def get_usage(self, period: str = "week") -> dict:
        """Get usage stats for a period: 'today', 'week', 'month', 'all'."""
        data = self._read_data()
        if data is None:
            data = {"days": {}, "version": 1}
        now = datetime.now(timezone.utc)

        if period == "today":
            days_back = 0
        elif period == "week":
            days_back = 6
        elif period == "month":
            days_back = 29
        else:
            days_back = 9999

        start_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

        result = {
            "period": period,
            "start_date": start_date,
            "end_date": now.strftime("%Y-%m-%d"),
            "models": {},
            "totals": {
                "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "cost_usd": 0.0, "requests": 0, "duration_ms": 0
            },
            "daily": [],
            "by_source": {}
        }

        for day_key in sorted(data.get("days", {}).keys()):
            if day_key < start_date:
                continue
            day = data["days"][day_key]

            # Aggregate totals
            dt = day.get("totals", {})
            result["totals"]["input_tokens"] += dt.get("input_tokens", 0)
            result["totals"]["output_tokens"] += dt.get("output_tokens", 0)
            result["totals"]["cache_read_tokens"] += dt.get("cache_read_tokens", 0)
            result["totals"]["cache_write_tokens"] += dt.get("cache_write_tokens", 0)
            result["totals"]["cost_usd"] += dt.get("cost_usd", 0.0)
            result["totals"]["requests"] += dt.get("requests", 0)
            result["totals"]["duration_ms"] += dt.get("duration_ms", 0)

            # Daily breakdown
            result["daily"].append({
                "date": day_key,
                "requests": dt.get("requests", 0),
                "input_tokens": dt.get("input_tokens", 0),
                "output_tokens": dt.get("output_tokens", 0),
                "cost_usd": dt.get("cost_usd", 0.0)
            })

            # Model aggregates
            for model_id, model_data in day.get("models", {}).items():
                if model_id not in result["models"]:
                    result["models"][model_id] = {
                        "input_tokens": 0, "output_tokens": 0,
                        "cache_read_tokens": 0, "cache_write_tokens": 0,
                        "cost_usd": 0.0, "requests": 0
                    }
                rm = result["models"][model_id]
                rm["input_tokens"] += model_data.get("input_tokens", 0)
                rm["output_tokens"] += model_data.get("output_tokens", 0)
                rm["cache_read_tokens"] += model_data.get("cache_read_tokens", 0)
                rm["cache_write_tokens"] += model_data.get("cache_write_tokens", 0)
                rm["cost_usd"] += model_data.get("cost_usd", 0.0)
                rm["requests"] += model_data.get("requests", 0)

                # Source aggregates
                for src, src_data in model_data.get("by_source", {}).items():
                    if src not in result["by_source"]:
                        result["by_source"][src] = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
                    result["by_source"][src]["requests"] += src_data.get("requests", 0)
                    result["by_source"][src]["input_tokens"] += src_data.get("input_tokens", 0)
                    result["by_source"][src]["output_tokens"] += src_data.get("output_tokens", 0)
                    result["by_source"][src]["cost_usd"] += src_data.get("cost_usd", 0.0)

        # Round costs
        result["totals"]["cost_usd"] = round(result["totals"]["cost_usd"], 4)
        for m in result["models"].values():
            m["cost_usd"] = round(m["cost_usd"], 4)
        for d in result["daily"]:
            d["cost_usd"] = round(d["cost_usd"], 4)
        for s in result["by_source"].values():
            s["cost_usd"] = round(s["cost_usd"], 4)

        return result
