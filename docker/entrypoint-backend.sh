#!/bin/sh
# Create required directory tree in shared volume and fix permissions
# This runs before the main process starts

# Create all needed directories
mkdir -p /data/memory/agents /data/memory/cerebro/cognitive_loop \
  /data/memory/cerebro/skills /data/memory/cerebro/chrome_profile \
  /data/memory/cerebro/recordings /data/memory/embeddings/chunks \
  /data/memory/learnings /data/memory/projects /data/memory/agent_contexts \
  /data/memory/mood /data/memory/conversations /data/memory/schedules

# Ensure CEREBRO_SECRET is set (generate and persist if not provided)
SECRET_FILE="/data/memory/.cerebro_secret"
if [ -z "$CEREBRO_SECRET" ]; then
  if [ -f "$SECRET_FILE" ]; then
    export CEREBRO_SECRET=$(cat "$SECRET_FILE")
  else
    export CEREBRO_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    echo "$CEREBRO_SECRET" > "$SECRET_FILE"
    chmod 600 "$SECRET_FILE"
  fi
elif [ ! -f "$SECRET_FILE" ]; then
  # Secret provided via env but not persisted yet — save it
  echo "$CEREBRO_SECRET" > "$SECRET_FILE"
  chmod 600 "$SECRET_FILE"
fi

# Generate a long-lived JWT for Claude Code sessions (1 year)
CEREBRO_TOKEN=$(python3 -c "
import jwt, time, os
secret = os.environ['CEREBRO_SECRET']
payload = {'sub': 'claude-code', 'iat': time.time(), 'exp': time.time() + 86400 * 365}
print(jwt.encode(payload, secret, algorithm='HS256'))
")

# Fix ownership if running as root (first boot with fresh volume)
if [ "$(id -u)" = "0" ]; then
  chown -R cerebro:cerebro /data/memory
  # Fix bind-mounted Claude config permissions (host may have different ownership)
  if [ -d /home/cerebro/.claude ]; then
    chown -R cerebro:cerebro /home/cerebro/.claude 2>/dev/null || true
  fi
  # Inject standalone CLAUDE.md into Claude config with token
  if [ -f /app/standalone-claude.md ] && [ -d /home/cerebro/.claude ]; then
    sed "s|__CEREBRO_TOKEN__|${CEREBRO_TOKEN}|g" /app/standalone-claude.md \
      > /home/cerebro/.claude/CLAUDE.md 2>/dev/null || true
    chown cerebro:cerebro /home/cerebro/.claude/CLAUDE.md 2>/dev/null || true
  fi
  # Create hooks.json for auto-syncing memory after agent conversations
  HOOKS_DIR="/home/cerebro/.claude"
  if [ -d "$HOOKS_DIR" ]; then
    cat > "${HOOKS_DIR}/hooks.json" <<HOOKEOF
{
  "hooks": {
    "Stop": [
      {
        "type": "command",
        "command": "curl -sf -X POST http://localhost:59000/api/memory/sync -H 'Authorization: Bearer ${CEREBRO_TOKEN}' -H 'Content-Type: application/json' -d '{\"source\":\"claude-code-stop\"}' > /dev/null 2>&1 || true"
      }
    ]
  }
}
HOOKEOF
    chown cerebro:cerebro "${HOOKS_DIR}/hooks.json" 2>/dev/null || true
  fi
  # Persist the secret file permissions
  chown cerebro:cerebro "$SECRET_FILE" 2>/dev/null || true
  exec gosu cerebro "$@"
else
  exec "$@"
fi
