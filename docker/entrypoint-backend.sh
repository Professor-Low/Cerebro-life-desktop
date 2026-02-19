#!/bin/sh
# Create required directory tree in shared volume
mkdir -p /data/memory/agents /data/memory/cerebro/cognitive_loop \
  /data/memory/cerebro/skills /data/memory/cerebro/chrome_profile \
  /data/memory/cerebro/recordings /data/memory/embeddings/chunks \
  /data/memory/learnings /data/memory/projects /data/memory/agent_contexts \
  /data/memory/mood /data/memory/conversations /data/memory/schedules
exec "$@"
