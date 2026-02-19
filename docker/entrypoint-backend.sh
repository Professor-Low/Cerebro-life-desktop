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
  exec gosu cerebro "$@"
else
  exec "$@"
fi
