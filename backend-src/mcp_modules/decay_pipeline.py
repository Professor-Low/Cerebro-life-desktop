#!/usr/bin/env python3
"""
Decay Pipeline - Compresses and eventually removes old memories.
Decay is a feature, not a bug. Forgetting the irrelevant is as important
as remembering the important.

Ported from memory-src/src/decay_pipeline.py for Cerebro Docker backend.

Summary Decay Rules:
- 0-7 days:   Full summary (200 tokens max)
- 7-30 days:  Compressed (50 tokens max)
- 30-90 days: Minimal (10 tokens) or merged
- 90+ days:   Archived or deleted

Conversation Decay Rules:
- 0-90 days:   Full storage (all fields)
- 90-180 days: Compressed (keep extracted_data, drop raw messages)
- 180-365 days: Minimal (keep facts + metadata only)
- 365+ days:   Cold storage

Fact Decay Rules (JSONL-based for Docker):
- Low confidence (<0.5) + never accessed + >90 days: Archive
- Superseded + >365 days: Delete
- Corrections are NEVER decayed

Run: Daily via scheduler or on-demand via /api/memory/maintenance
"""

import hashlib
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import AI_MEMORY_BASE, ARCHIVE_DIR, CONVERSATIONS_DIR, DECAY_STATE_FILE, LOG_DIR, SUMMARIES_DIR

# Debug log path (from config)
DEBUG_LOG = LOG_DIR / "decay_pipeline_debug.log"

# Directories
FACTS_DIR = AI_MEMORY_BASE / "facts"
COLD_STORAGE_DIR = AI_MEMORY_BASE / "cold_storage"
DECAY_LOG_FILE = AI_MEMORY_BASE / "logs" / "decay_actions.jsonl"

# Summary decay rules
SUMMARY_DECAY_RULES = {
    7: {"max_words": 50, "level": "full"},
    30: {"max_words": 15, "level": "brief"},
    90: {"max_words": 5, "level": "minimal"},
    365: {"max_words": 0, "level": "archive"},
}

# Conversation decay rules
CONVERSATION_DECAY_RULES = {
    90: "full",
    180: "compressed",
    365: "minimal",
    730: "cold_storage",
}

# Fact decay thresholds
FACT_DECAY_CONFIG = {
    "low_confidence_threshold": 0.50,
    "min_age_days": 90,
    "max_access_count": 1,
    "never_accessed_days": 60,
    "superseded_delete_days": 365,
}

# Backward compat alias
DECAY_RULES = SUMMARY_DECAY_RULES


def log_debug(msg: str):
    """Write debug info to log file."""
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except Exception:
        pass


def load_decay_state() -> Dict[str, Any]:
    """Load decay pipeline state (last run time, stats)."""
    if DECAY_STATE_FILE.exists():
        try:
            with open(DECAY_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_run": None, "runs": 0, "total_compressed": 0, "total_archived": 0}


def save_decay_state(state: Dict[str, Any]):
    """Save decay pipeline state."""
    try:
        DECAY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DECAY_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log_debug(f"Error saving decay state: {e}")


def should_run() -> bool:
    """Check if decay pipeline should run (once per day max)."""
    state = load_decay_state()
    last_run = state.get("last_run")

    if not last_run:
        return True

    try:
        last_run_dt = datetime.fromisoformat(last_run)
        hours_since = (datetime.now() - last_run_dt).total_seconds() / 3600
        return hours_since >= 24
    except Exception:
        return True


def get_age_days(timestamp_str: str) -> int:
    """Get age in days from ISO timestamp string."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return (datetime.now() - dt.replace(tzinfo=None)).days
    except Exception:
        return 0


def compress_summary(summary: str, max_words: int) -> str:
    """Compress summary to max_words."""
    if max_words <= 0:
        return ""
    words = summary.split()
    if len(words) <= max_words:
        return summary
    return " ".join(words[:max_words]) + "..."


# =============================================================================
# Decay Action Logging
# =============================================================================

@dataclass
class DecayAction:
    """Record of a decay action for audit trail."""
    timestamp: str
    action_type: str        # compressed, archived, cold_storage, deleted
    item_type: str          # summary, conversation, fact
    item_id: str
    reason: str
    details: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def log_decay_action(action: DecayAction):
    """Log a decay action to the audit log."""
    try:
        DECAY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DECAY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(action.to_dict()) + "\n")
    except Exception as e:
        log_debug(f"Error logging decay action: {e}")


# =============================================================================
# Summary Decay
# =============================================================================

def process_summary_file(file_path: Path, stats: Dict[str, int]) -> bool:
    """Process a single summary file, applying decay rules. Returns True if modified."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)
    except Exception as e:
        log_debug(f"Error reading {file_path}: {e}")
        return False

    modified = False

    for summary in summaries:
        timestamp = summary.get("timestamp", "")
        age_days = get_age_days(timestamp)
        current_words = summary.get("word_count", len(summary.get("summary", "").split()))
        decay_level = summary.get("decay_level", "full")

        target_words = 50
        target_level = "full"

        for threshold_days, rule in sorted(DECAY_RULES.items()):
            if age_days >= threshold_days:
                target_words = rule["max_words"]
                target_level = rule["level"]

        if current_words <= target_words and decay_level == target_level:
            continue

        if target_level == "archive":
            summary["archived"] = True
            summary["archived_at"] = datetime.now().isoformat()
            stats["archived"] += 1
            modified = True
        elif target_words > 0 and current_words > target_words:
            summary["summary"] = compress_summary(summary.get("summary", ""), target_words)
            summary["word_count"] = target_words
            summary["decay_level"] = target_level
            summary["decayed_at"] = datetime.now().isoformat()
            stats["compressed"] += 1
            modified = True

    if modified:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_debug(f"Error writing {file_path}: {e}")
            return False

    return modified


def archive_old_summaries(stats: Dict[str, int]):
    """Move archived summaries to archive directory."""
    if not SUMMARIES_DIR.exists():
        return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in SUMMARIES_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                summaries = json.load(f)

            active = []
            archived = []

            for s in summaries:
                if s.get("archived"):
                    archived.append(s)
                else:
                    active.append(s)

            if archived:
                archive_file = ARCHIVE_DIR / file_path.name
                existing_archive = []
                if archive_file.exists():
                    with open(archive_file, "r", encoding="utf-8") as f:
                        existing_archive = json.load(f)

                existing_archive.extend(archived)

                with open(archive_file, "w", encoding="utf-8") as f:
                    json.dump(existing_archive, f, indent=2, ensure_ascii=False)

                if active:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(active, f, indent=2, ensure_ascii=False)
                else:
                    file_path.unlink()

                stats["files_archived"] += 1

        except Exception as e:
            log_debug(f"Error archiving {file_path}: {e}")


def delete_very_old_archives(max_age_days: int = 365) -> int:
    """Delete archives older than max_age_days. Returns count deleted."""
    if not ARCHIVE_DIR.exists():
        return 0

    deleted = 0
    cutoff = datetime.now() - timedelta(days=max_age_days)

    for file_path in ARCHIVE_DIR.glob("*.json"):
        try:
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_mtime < cutoff:
                file_path.unlink()
                deleted += 1
                log_debug(f"Deleted old archive: {file_path}")
        except Exception as e:
            log_debug(f"Error deleting {file_path}: {e}")

    return deleted


# =============================================================================
# Conversation Decay
# =============================================================================

def get_conversation_decay_level(age_days: int) -> str:
    """Determine decay level for a conversation based on age."""
    for threshold, level in sorted(CONVERSATION_DECAY_RULES.items()):
        if age_days < threshold:
            return level
    return "cold_storage"


def compress_conversation(conv: Dict, level: str) -> Dict:
    """Compress a conversation to the specified decay level."""
    if level == "full":
        return conv

    result = {
        "id": conv.get("id"),
        "timestamp": conv.get("timestamp"),
        "type": conv.get("type", "conversation"),
        "decay_level": level,
        "decayed_at": datetime.now().isoformat(),
    }

    if level == "compressed":
        result["extracted_data"] = conv.get("extracted_data", {})
        result["metadata"] = conv.get("metadata", {})
        result["search_index"] = conv.get("search_index", {})
        if "search_index" in conv and "summary" in conv["search_index"]:
            result["summary"] = conv["search_index"]["summary"]

    elif level == "minimal":
        result["metadata"] = {
            "topics": conv.get("metadata", {}).get("topics", []),
            "tags": conv.get("metadata", {}).get("tags", []),
            "message_count": conv.get("metadata", {}).get("message_count", 0),
        }
        extracted = conv.get("extracted_data", {})
        result["facts"] = extracted.get("facts", [])
        result["entities"] = extracted.get("entities", {})
        if "search_index" in conv and "summary" in conv["search_index"]:
            result["summary"] = conv["search_index"]["summary"][:100]

    result["original_size"] = len(json.dumps(conv))
    result["compressed_size"] = len(json.dumps(result))

    return result


def process_conversation_decay(stats: Dict[str, int]) -> None:
    """Process all conversations for decay."""
    if not CONVERSATIONS_DIR.exists():
        return

    cold_conv_dir = COLD_STORAGE_DIR / "conversations"
    cold_conv_dir.mkdir(parents=True, exist_ok=True)

    for file_path in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                conv = json.load(f)

            timestamp = conv.get("timestamp", "")
            age_days = get_age_days(timestamp)
            current_level = conv.get("decay_level", "full")
            target_level = get_conversation_decay_level(age_days)

            if current_level == target_level:
                continue

            level_order = ["full", "compressed", "minimal", "cold_storage"]
            if level_order.index(current_level) >= level_order.index(target_level):
                continue

            conv_id = conv.get("id", file_path.stem)

            if target_level == "cold_storage":
                dest = cold_conv_dir / file_path.name
                shutil.move(str(file_path), str(dest))
                stats["conversations_cold_storage"] += 1

                log_decay_action(DecayAction(
                    timestamp=datetime.now().isoformat(),
                    action_type="cold_storage",
                    item_type="conversation",
                    item_id=conv_id,
                    reason=f"Age {age_days} days exceeds threshold",
                    details={"age_days": age_days}
                ))
                log_debug(f"Moved conversation to cold storage: {conv_id}")

            else:
                compressed = compress_conversation(conv, target_level)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(compressed, f, indent=2, ensure_ascii=False)
                stats["conversations_compressed"] += 1

                log_decay_action(DecayAction(
                    timestamp=datetime.now().isoformat(),
                    action_type="compressed",
                    item_type="conversation",
                    item_id=conv_id,
                    reason=f"Age {age_days} days - compressed to {target_level}",
                    details={
                        "age_days": age_days,
                        "from_level": current_level,
                        "to_level": target_level,
                        "size_reduction": compressed.get("original_size", 0) - compressed.get("compressed_size", 0)
                    }
                ))
                log_debug(f"Compressed conversation {conv_id} to {target_level}")

        except Exception as e:
            log_debug(f"Error processing conversation {file_path}: {e}")
            stats["conversation_errors"] += 1


# =============================================================================
# Fact Decay (JSONL-based for Docker)
# =============================================================================

def get_fact_confidence(fact: Dict) -> float:
    """Get confidence score for a fact.

    In Docker, we read from the fact's own confidence field
    (no external confidence_tracker available).
    """
    return float(fact.get("confidence", 0.5))


def is_fact_superseded(fact: Dict) -> bool:
    """Check if a fact has been superseded."""
    return fact.get("superseded", False) or fact.get("metadata", {}).get("superseded", False)


def should_decay_fact(fact: Dict) -> Tuple[bool, str]:
    """Determine if a fact should be decayed.

    Returns: (should_decay, reason)
    """
    timestamp = fact.get("timestamp", "")
    age_days = get_age_days(timestamp)

    # Never decay corrections
    fact_type = fact.get("type", "")
    if fact_type in ("correction", "antipattern"):
        return False, "protected_type"

    # Rule 1: Superseded facts older than threshold -> delete
    if is_fact_superseded(fact):
        if age_days >= FACT_DECAY_CONFIG["superseded_delete_days"]:
            return True, "superseded_old"
        return False, "superseded_recent"

    # Rule 2: Never accessed + older than threshold -> archive
    never_accessed_days = FACT_DECAY_CONFIG.get("never_accessed_days", 60)
    if age_days >= never_accessed_days:
        access_count = fact.get("access_count", 0)
        if access_count == 0:
            return True, f"never_accessed_age_{age_days}"

    # Rule 3: Low confidence + old + rarely accessed -> archive
    min_age = FACT_DECAY_CONFIG["min_age_days"]
    if age_days < min_age:
        return False, "too_young"

    confidence = get_fact_confidence(fact)
    if confidence < FACT_DECAY_CONFIG["low_confidence_threshold"]:
        access_count = fact.get("access_count", 0)
        if access_count <= FACT_DECAY_CONFIG["max_access_count"]:
            return True, f"low_confidence_{confidence:.2f}_low_access_{access_count}"

    return False, "retained"


def process_fact_decay_jsonl(stats: Dict[str, int]) -> None:
    """Process facts.jsonl for decay.

    Docker stores facts as lines in facts/facts.jsonl rather than
    individual .json files. We read, evaluate each line, and write
    back only retained facts. Archived facts go to cold storage.
    """
    facts_file = FACTS_DIR / "facts.jsonl"
    if not facts_file.exists():
        return

    cold_facts_dir = COLD_STORAGE_DIR / "facts"
    cold_facts_dir.mkdir(parents=True, exist_ok=True)
    cold_facts_file = cold_facts_dir / "archived_facts.jsonl"

    retained = []
    archived = []

    try:
        with open(facts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    fact = json.loads(line)
                except json.JSONDecodeError:
                    retained.append(line)  # Keep unparseable lines as-is
                    continue

                should_decay, reason = should_decay_fact(fact)

                if not should_decay:
                    retained.append(json.dumps(fact, ensure_ascii=False))
                    continue

                fact_id = fact.get("fact_id", fact.get("id", "unknown"))

                if reason == "superseded_old":
                    stats["facts_deleted"] += 1
                    log_decay_action(DecayAction(
                        timestamp=datetime.now().isoformat(),
                        action_type="deleted",
                        item_type="fact",
                        item_id=fact_id,
                        reason=reason,
                        details={"age_days": get_age_days(fact.get("timestamp", ""))}
                    ))
                    log_debug(f"Deleted superseded fact: {fact_id}")
                else:
                    # Archive to cold storage
                    archived.append(json.dumps(fact, ensure_ascii=False))
                    stats["facts_archived"] += 1
                    log_decay_action(DecayAction(
                        timestamp=datetime.now().isoformat(),
                        action_type="archived",
                        item_type="fact",
                        item_id=fact_id,
                        reason=reason,
                        details={
                            "age_days": get_age_days(fact.get("timestamp", "")),
                            "content_preview": fact.get("content", "")[:50]
                        }
                    ))
                    log_debug(f"Archived fact to cold storage: {fact_id}")

    except Exception as e:
        log_debug(f"Error reading facts.jsonl: {e}")
        stats["fact_errors"] += 1
        return

    # Write back retained facts
    if stats["facts_deleted"] > 0 or stats["facts_archived"] > 0:
        try:
            with open(facts_file, "w", encoding="utf-8") as f:
                for line in retained:
                    f.write(line + "\n")
        except Exception as e:
            log_debug(f"Error writing facts.jsonl: {e}")
            stats["fact_errors"] += 1

        # Append archived facts to cold storage
        if archived:
            try:
                with open(cold_facts_file, "a", encoding="utf-8") as f:
                    for line in archived:
                        f.write(line + "\n")
            except Exception as e:
                log_debug(f"Error writing cold storage facts: {e}")

    # Also process individual .json fact files if any exist (backward compat)
    for file_path in FACTS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                fact = json.load(f)

            should_decay, reason = should_decay_fact(fact)
            if not should_decay:
                continue

            fact_id = fact.get("id", file_path.stem)

            if reason == "superseded_old":
                file_path.unlink()
                stats["facts_deleted"] += 1
            else:
                dest = cold_facts_dir / file_path.name
                shutil.move(str(file_path), str(dest))
                stats["facts_archived"] += 1

        except Exception as e:
            log_debug(f"Error processing fact {file_path}: {e}")
            stats["fact_errors"] += 1


# =============================================================================
# Fact Deduplication
# =============================================================================

def deduplicate_facts_jsonl() -> Dict[str, int]:
    """Deduplicate facts.jsonl by content hash. Keeps newest duplicate.

    Returns: {"before": N, "after": N, "removed": N}
    """
    facts_file = FACTS_DIR / "facts.jsonl"
    if not facts_file.exists():
        return {"before": 0, "after": 0, "removed": 0}

    seen_hashes = {}  # content_hash -> (line_index, line_text)
    lines = []

    try:
        with open(facts_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                lines.append(line)

                try:
                    entry = json.loads(line)
                    # Hash on the content/fact text to detect duplicates
                    content = entry.get("content",
                              entry.get("learning",
                              entry.get("fact", "")))
                    if content:
                        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
                    else:
                        content_hash = hashlib.md5(line.encode("utf-8")).hexdigest()

                    # Keep the latest occurrence (overwrite earlier ones)
                    seen_hashes[content_hash] = i
                except json.JSONDecodeError:
                    # Keep unparseable lines
                    seen_hashes[f"raw_{i}"] = i

        before_count = len(lines)
        keep_indices = set(seen_hashes.values())
        deduped = [lines[i] for i in sorted(keep_indices)]
        after_count = len(deduped)
        removed = before_count - after_count

        if removed > 0:
            with open(facts_file, "w", encoding="utf-8") as f:
                for line in deduped:
                    f.write(line + "\n")
            log_debug(f"Deduplicated facts.jsonl: {before_count} -> {after_count} ({removed} removed)")

        return {"before": before_count, "after": after_count, "removed": removed}

    except Exception as e:
        log_debug(f"Error deduplicating facts: {e}")
        return {"before": 0, "after": 0, "removed": 0, "error": str(e)}


# =============================================================================
# Main Decay Pipeline
# =============================================================================

def run_decay(
    force: bool = False,
    include_summaries: bool = True,
    include_conversations: bool = True,
    include_facts: bool = True
) -> Dict[str, Any]:
    """Run the decay pipeline.

    Args:
        force: Run even if already ran today
        include_summaries: Process summary decay
        include_conversations: Process conversation tier decay
        include_facts: Process fact decay + deduplication

    Returns:
        Stats dict with counts of actions taken
    """
    log_debug("Decay pipeline started")

    if not force and not should_run():
        log_debug("Skipping - already ran within 24 hours")
        return {"skipped": True, "reason": "already_ran_today"}

    stats = {
        # Summary decay stats
        "summaries_compressed": 0,
        "summaries_archived": 0,
        "files_processed": 0,
        "files_archived": 0,
        "files_deleted": 0,
        # Conversation decay stats
        "conversations_compressed": 0,
        "conversations_cold_storage": 0,
        "conversation_errors": 0,
        # Fact decay stats
        "facts_archived": 0,
        "facts_deleted": 0,
        "fact_errors": 0,
        # Deduplication stats
        "dedup_removed": 0,
    }

    # Backward compat aliases
    stats["compressed"] = 0
    stats["archived"] = 0

    # 1. Process summary files
    if include_summaries and SUMMARIES_DIR.exists():
        for file_path in SUMMARIES_DIR.glob("*.json"):
            if process_summary_file(file_path, stats):
                stats["files_processed"] += 1
        stats["compressed"] = stats.get("compressed", 0) + stats["summaries_compressed"]
        stats["archived"] = stats.get("archived", 0) + stats["summaries_archived"]

        archive_old_summaries(stats)
        stats["files_deleted"] = delete_very_old_archives(max_age_days=365)

    # 2. Process conversation decay
    if include_conversations:
        log_debug("Processing conversation decay...")
        process_conversation_decay(stats)

    # 3. Process fact decay (JSONL-based)
    if include_facts:
        log_debug("Processing fact decay...")
        process_fact_decay_jsonl(stats)

        # 4. Deduplicate facts.jsonl
        log_debug("Deduplicating facts...")
        dedup_result = deduplicate_facts_jsonl()
        stats["dedup_removed"] = dedup_result.get("removed", 0)

    # Update state
    state = load_decay_state()
    state["last_run"] = datetime.now().isoformat()
    state["runs"] = state.get("runs", 0) + 1
    state["total_compressed"] = state.get("total_compressed", 0) + stats["compressed"]
    state["total_archived"] = state.get("total_archived", 0) + stats["archived"]
    state["total_conversations_compressed"] = state.get("total_conversations_compressed", 0) + stats["conversations_compressed"]
    state["total_conversations_cold_storage"] = state.get("total_conversations_cold_storage", 0) + stats["conversations_cold_storage"]
    state["total_facts_archived"] = state.get("total_facts_archived", 0) + stats["facts_archived"]
    state["total_facts_deleted"] = state.get("total_facts_deleted", 0) + stats["facts_deleted"]
    state["total_dedup_removed"] = state.get("total_dedup_removed", 0) + stats["dedup_removed"]

    save_decay_state(state)

    log_debug(f"Decay complete: {stats}")
    return stats


def get_decay_stats() -> Dict[str, Any]:
    """Get decay pipeline statistics."""
    state = load_decay_state()

    # Count summaries by decay level
    summary_levels = {"full": 0, "brief": 0, "minimal": 0, "archived": 0}

    if SUMMARIES_DIR.exists():
        for file_path in SUMMARIES_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    summaries = json.load(f)
                for s in summaries:
                    level = s.get("decay_level", "full")
                    summary_levels[level] = summary_levels.get(level, 0) + 1
            except Exception:
                pass

    if ARCHIVE_DIR.exists():
        for file_path in ARCHIVE_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    summaries = json.load(f)
                summary_levels["archived"] += len(summaries)
            except Exception:
                pass

    # Count conversations by decay level
    conv_levels = {"full": 0, "compressed": 0, "minimal": 0, "cold_storage": 0}

    if CONVERSATIONS_DIR.exists():
        for file_path in CONVERSATIONS_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    conv = json.load(f)
                level = conv.get("decay_level", "full")
                conv_levels[level] = conv_levels.get(level, 0) + 1
            except Exception:
                pass

    cold_conv_dir = COLD_STORAGE_DIR / "conversations"
    if cold_conv_dir.exists():
        conv_levels["cold_storage"] = len(list(cold_conv_dir.glob("*.json")))

    # Count facts (JSONL)
    fact_counts = {"active": 0, "cold_storage": 0}
    facts_file = FACTS_DIR / "facts.jsonl"
    if facts_file.exists():
        try:
            with open(facts_file, "r", encoding="utf-8") as f:
                fact_counts["active"] = sum(1 for line in f if line.strip())
        except Exception:
            pass
    # Also count individual .json facts
    if FACTS_DIR.exists():
        fact_counts["active"] += len(list(FACTS_DIR.glob("*.json")))

    cold_facts_dir = COLD_STORAGE_DIR / "facts"
    if cold_facts_dir.exists():
        # Count JSONL lines in cold storage
        cold_file = cold_facts_dir / "archived_facts.jsonl"
        if cold_file.exists():
            try:
                with open(cold_file, "r", encoding="utf-8") as f:
                    fact_counts["cold_storage"] = sum(1 for line in f if line.strip())
            except Exception:
                pass
        fact_counts["cold_storage"] += len(list(cold_facts_dir.glob("*.json")))

    return {
        "last_run": state.get("last_run"),
        "total_runs": state.get("runs", 0),
        "summaries": {
            "total_compressed": state.get("total_compressed", 0),
            "total_archived": state.get("total_archived", 0),
            "by_level": summary_levels
        },
        "conversations": {
            "total_compressed": state.get("total_conversations_compressed", 0),
            "total_cold_storage": state.get("total_conversations_cold_storage", 0),
            "by_level": conv_levels
        },
        "facts": {
            "total_archived": state.get("total_facts_archived", 0),
            "total_deleted": state.get("total_facts_deleted", 0),
            "total_dedup_removed": state.get("total_dedup_removed", 0),
            "counts": fact_counts
        },
        # Legacy format
        "total_compressed": state.get("total_compressed", 0),
        "total_archived": state.get("total_archived", 0),
        "summaries_by_level": summary_levels
    }


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1:
        cmd = _sys.argv[1]

        if cmd == "--force":
            print("Running decay pipeline (forced)...")
            result = run_decay(force=True)
            print(json.dumps(result, indent=2))

        elif cmd == "--stats":
            print("Decay pipeline stats:")
            stats = get_decay_stats()
            print(json.dumps(stats, indent=2))

        elif cmd == "--dedup":
            print("Deduplicating facts.jsonl...")
            result = deduplicate_facts_jsonl()
            print(json.dumps(result, indent=2))

        else:
            print(f"Usage: python decay_pipeline.py [--force|--stats|--dedup]")
    else:
        print("Running decay pipeline...")
        result = run_decay()
        print(json.dumps(result, indent=2))
