#!/bin/sh
# Create required directory tree in shared volume and fix permissions
# This runs before the main process starts

# Create all needed directories
mkdir -p /data/memory/agents /data/memory/cerebro/cognitive_loop \
  /data/memory/cerebro/skills /data/memory/cerebro/chrome_profile \
  /data/memory/cerebro/recordings /data/memory/embeddings/chunks \
  /data/memory/learnings /data/memory/projects /data/memory/agent_contexts \
  /data/memory/mood /data/memory/conversations /data/memory/schedules

# Fix ownership if running as root (first boot with fresh volume)
if [ "$(id -u)" = "0" ]; then
  chown -R cerebro:cerebro /data/memory
  # Fix bind-mounted Claude config permissions (host may have different ownership)
  if [ -d /home/cerebro/.claude ]; then
    chown -R cerebro:cerebro /home/cerebro/.claude 2>/dev/null || true
  fi
  # Inject standalone CLAUDE.md into Claude config if not already there
  if [ -f /app/standalone-claude.md ] && [ -d /home/cerebro/.claude ]; then
    cp -f /app/standalone-claude.md /home/cerebro/.claude/CLAUDE.md 2>/dev/null || true
    chown cerebro:cerebro /home/cerebro/.claude/CLAUDE.md 2>/dev/null || true
  fi
  exec gosu cerebro "$@"
else
  exec "$@"
fi
