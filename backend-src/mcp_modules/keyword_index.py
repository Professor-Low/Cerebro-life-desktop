"""
Fast Keyword Search using SQLite FTS5

Indexes conversations, facts, and learnings from Docker memory volume.
Searches in milliseconds using BM25 scoring.

Ported from memory-src/src/keyword_index.py for Cerebro Docker backend.
"""
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List


class KeywordIndex:
    """SQLite-based keyword search for fast queries"""

    def __init__(self, index_path: Path = None):
        # Default: store in the Docker memory volume
        if index_path is None:
            base = Path(os.environ.get("CEREBRO_DATA_DIR",
                                       os.environ.get("AI_MEMORY_PATH",
                                                       os.path.expanduser("~/.cerebro/data"))))
            index_path = base / "keyword_index.db"
        self.index_path = Path(index_path)
        self._ensure_dir()
        self.conn = None
        self._connect()

    def _ensure_dir(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(str(self.index_path), timeout=10, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self):
        """Create FTS5 table if not exists"""
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                conversation_id,
                chunk_id,
                chunk_type,
                content,
                metadata,
                tokenize='porter unicode61'
            )
        """)
        # Track indexed files to avoid re-indexing
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_files (
                filename TEXT PRIMARY KEY,
                mtime REAL,
                chunk_count INTEGER
            )
        """)
        self.conn.commit()

    def get_indexed_count(self) -> int:
        """Get number of indexed chunks"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunks_fts")
        return cursor.fetchone()[0]

    def needs_rebuild(self) -> bool:
        """Check if index needs rebuilding"""
        if self.get_indexed_count() == 0:
            return True
        return False

    def build_index_from_memory(self, memory_path) -> int:
        """Build keyword index from Docker memory data sources.

        Indexes three sources:
        1. conversations/*.json - extracted data, summaries, messages
        2. facts/facts.jsonl - JSONL fact entries
        3. learnings/*.json - structured problem/solution pairs

        Args:
            memory_path: Base path to memory data (e.g. /data/memory)

        Returns:
            Number of new entries indexed
        """
        memory_path = Path(memory_path)
        indexed = 0

        # 1. Index conversations
        conv_dir = memory_path / "conversations"
        if conv_dir.exists():
            for conv_file in conv_dir.glob("*.json"):
                try:
                    mtime = conv_file.stat().st_mtime
                    cursor = self.conn.execute(
                        "SELECT mtime FROM indexed_files WHERE filename = ?",
                        (f"conv:{conv_file.name}",)
                    )
                    row = cursor.fetchone()
                    if row and row[0] == mtime:
                        continue  # Already indexed

                    conv_id = conv_file.stem

                    # Remove old entries for this conversation
                    self.conn.execute(
                        "DELETE FROM chunks_fts WHERE conversation_id = ?",
                        (conv_id,)
                    )

                    with open(conv_file, "r", encoding="utf-8") as f:
                        conv = json.load(f)

                    chunk_count = 0

                    # Index extracted_data facts
                    extracted = conv.get("extracted_data", {})
                    for fact in extracted.get("facts", []):
                        content = fact if isinstance(fact, str) else fact.get("content", str(fact))
                        if content:
                            self.conn.execute("""
                                INSERT INTO chunks_fts (conversation_id, chunk_id, chunk_type, content, metadata)
                                VALUES (?, ?, ?, ?, ?)
                            """, (conv_id, f"{conv_id}_fact_{chunk_count}", "fact", content, "{}"))
                            chunk_count += 1

                    # Index search_index summary
                    summary = conv.get("search_index", {}).get("summary", "")
                    if not summary:
                        summary = conv.get("summary", "")
                    if summary:
                        self.conn.execute("""
                            INSERT INTO chunks_fts (conversation_id, chunk_id, chunk_type, content, metadata)
                            VALUES (?, ?, ?, ?, ?)
                        """, (conv_id, f"{conv_id}_summary", "summary", summary, "{}"))
                        chunk_count += 1

                    # Index search_index keywords
                    keywords = conv.get("search_index", {}).get("keywords", [])
                    if keywords:
                        kw_text = " ".join(keywords)
                        self.conn.execute("""
                            INSERT INTO chunks_fts (conversation_id, chunk_id, chunk_type, content, metadata)
                            VALUES (?, ?, ?, ?, ?)
                        """, (conv_id, f"{conv_id}_keywords", "keywords", kw_text, "{}"))
                        chunk_count += 1

                    # Index entities
                    entities = extracted.get("entities", {})
                    if entities:
                        entity_parts = []
                        for etype, elist in entities.items():
                            if isinstance(elist, list):
                                entity_parts.extend([str(e) for e in elist])
                            elif isinstance(elist, dict):
                                entity_parts.extend([str(v) for v in elist.values()])
                        if entity_parts:
                            self.conn.execute("""
                                INSERT INTO chunks_fts (conversation_id, chunk_id, chunk_type, content, metadata)
                                VALUES (?, ?, ?, ?, ?)
                            """, (conv_id, f"{conv_id}_entities", "entity", " ".join(entity_parts), "{}"))
                            chunk_count += 1

                    # Record indexed file
                    self.conn.execute("""
                        INSERT OR REPLACE INTO indexed_files (filename, mtime, chunk_count)
                        VALUES (?, ?, ?)
                    """, (f"conv:{conv_file.name}", mtime, chunk_count))

                    indexed += chunk_count

                except Exception as e:
                    print(f"Warning: Failed to index conversation {conv_file}: {e}")
                    continue

        # 2. Index facts.jsonl
        facts_file = memory_path / "facts" / "facts.jsonl"
        if facts_file.exists():
            try:
                mtime = facts_file.stat().st_mtime
                cursor = self.conn.execute(
                    "SELECT mtime FROM indexed_files WHERE filename = ?",
                    ("facts:facts.jsonl",)
                )
                row = cursor.fetchone()
                if not (row and row[0] == mtime):
                    # Remove old fact entries
                    self.conn.execute(
                        "DELETE FROM chunks_fts WHERE chunk_type IN ('fact_jsonl', 'learning_jsonl')"
                    )

                    chunk_count = 0
                    with open(facts_file, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                                content = entry.get("content",
                                          entry.get("learning",
                                          entry.get("fact", "")))
                                if not content:
                                    content = json.dumps(entry)

                                fact_type = entry.get("type", "fact")
                                conv_id = entry.get("conversation_id", "facts_jsonl")

                                self.conn.execute("""
                                    INSERT INTO chunks_fts (conversation_id, chunk_id, chunk_type, content, metadata)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (
                                    conv_id,
                                    entry.get("fact_id", f"fact_{i}"),
                                    "fact_jsonl",
                                    content,
                                    json.dumps({"type": fact_type})
                                ))
                                chunk_count += 1
                            except json.JSONDecodeError:
                                continue

                    self.conn.execute("""
                        INSERT OR REPLACE INTO indexed_files (filename, mtime, chunk_count)
                        VALUES (?, ?, ?)
                    """, ("facts:facts.jsonl", mtime, chunk_count))

                    indexed += chunk_count

            except Exception as e:
                print(f"Warning: Failed to index facts.jsonl: {e}")

        # 3. Index learnings
        learn_dir = memory_path / "learnings"
        if learn_dir.exists():
            for learn_file in learn_dir.glob("*.json"):
                try:
                    mtime = learn_file.stat().st_mtime
                    cursor = self.conn.execute(
                        "SELECT mtime FROM indexed_files WHERE filename = ?",
                        (f"learn:{learn_file.name}",)
                    )
                    row = cursor.fetchone()
                    if row and row[0] == mtime:
                        continue

                    learn_id = learn_file.stem

                    # Remove old entries
                    self.conn.execute(
                        "DELETE FROM chunks_fts WHERE chunk_id = ?",
                        (learn_id,)
                    )

                    with open(learn_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Build searchable text from structured fields
                    parts = []
                    if isinstance(data, dict):
                        for field in ["problem", "solution", "what_not_to_do",
                                      "why_it_failed", "context"]:
                            if data.get(field):
                                parts.append(str(data[field]))
                        for learn in data.get("learnings", []):
                            if isinstance(learn, dict):
                                parts.append(learn.get("content", learn.get("problem", "")))
                            elif isinstance(learn, str):
                                parts.append(learn)
                        for tag in data.get("tags", []):
                            parts.append(str(tag))

                    content = " ".join(parts) if parts else json.dumps(data)

                    self.conn.execute("""
                        INSERT INTO chunks_fts (conversation_id, chunk_id, chunk_type, content, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        data.get("conversation_id", "learnings"),
                        learn_id,
                        "learning",
                        content,
                        json.dumps({"type": data.get("type", "learning")})
                    ))

                    self.conn.execute("""
                        INSERT OR REPLACE INTO indexed_files (filename, mtime, chunk_count)
                        VALUES (?, ?, ?)
                    """, (f"learn:{learn_file.name}", mtime, 1))

                    indexed += 1

                except Exception as e:
                    print(f"Warning: Failed to index learning {learn_file}: {e}")
                    continue

        self.conn.commit()
        return indexed

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Fast keyword search using FTS5"""
        try:
            # FTS5 query - escape special characters
            safe_query = query.replace('"', '""')

            cursor = self.conn.execute("""
                SELECT
                    conversation_id,
                    chunk_id,
                    chunk_type,
                    content,
                    metadata,
                    bm25(chunks_fts) as score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
            """, (safe_query, top_k))

            raw_results = []
            for row in cursor:
                try:
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                except Exception:
                    metadata = {}

                raw_results.append({
                    "conversation_id": row["conversation_id"],
                    "chunk_id": row["chunk_id"],
                    "chunk_type": row["chunk_type"],
                    "content": row["content"],
                    "metadata": metadata,
                    "score": -row["score"],  # BM25 returns negative scores
                })

            # Normalize scores (0.0-1.0 range)
            if raw_results:
                max_raw = max(r["score"] for r in raw_results) if raw_results else 1.0
                for r in raw_results:
                    r["similarity"] = min(1.0, r["score"] / max_raw) if max_raw > 0 else 0.0

            return raw_results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def close(self):
        if self.conn:
            self.conn.close()


# Singleton instance
_keyword_index = None

def get_keyword_index(index_path: Path = None) -> KeywordIndex:
    """Get or create the keyword index singleton"""
    global _keyword_index
    if _keyword_index is None:
        _keyword_index = KeywordIndex(index_path)
    return _keyword_index
