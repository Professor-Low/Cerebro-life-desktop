#!/usr/bin/env bash
set -euo pipefail

echo "[public-build] Building Cerebro Desktop — Public Edition"
cd "$(dirname "$0")/.."

# Safety check: verify we're in the public repo
REMOTE=$(git remote get-url origin 2>/dev/null || echo "unknown")
if [[ "$REMOTE" != *"Cerebro-life-desktop"* ]]; then
  echo "[ERROR] Git remote is $REMOTE — expected Cerebro-life-desktop"
  echo "[ERROR] Are you running this from the wrong repo?"
  exit 1
fi

# Verify package name
PACKAGE_NAME=$(node -e "console.log(require('./package.json').name)")
echo "[public-build] Package: $PACKAGE_NAME"

# Build for Windows (primary public platform)
npx electron-builder --win --config build/electron-builder-public.yml

echo "[public-build] Done. Output in dist/"
echo "[public-build] buildType=public is baked into the asar."
