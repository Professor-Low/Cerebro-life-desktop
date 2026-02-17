#!/usr/bin/env bash
#
# Generate all Cerebro app icons from the source SVG.
# Requirements: rsvg-convert (librsvg2-bin), imagemagick (for .ico)
#
# Usage: ./generate-icons.sh [path/to/icon.svg]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_DIR="$(dirname "$SCRIPT_DIR")"
ASSETS_DIR="$DESKTOP_DIR/assets"
FRONTEND_DIR="$DESKTOP_DIR/frontend"
SVG="${1:-$ASSETS_DIR/icon.svg}"

if [ ! -f "$SVG" ]; then
    echo "ERROR: SVG source not found: $SVG"
    exit 1
fi

# Check dependencies
if ! command -v rsvg-convert &>/dev/null; then
    echo "ERROR: rsvg-convert not found. Install with: sudo apt-get install librsvg2-bin"
    exit 1
fi

echo "Generating icons from: $SVG"
echo "Output: $ASSETS_DIR"

mkdir -p "$ASSETS_DIR" "$FRONTEND_DIR"

# Main app icons
rsvg-convert -w 512 -h 512 "$SVG" -o "$ASSETS_DIR/icon.png"
echo "  icon.png (512x512)"

rsvg-convert -w 256 -h 256 "$SVG" -o "$ASSETS_DIR/icon-256.png"
echo "  icon-256.png"

rsvg-convert -w 192 -h 192 "$SVG" -o "$ASSETS_DIR/icon-192.png"
echo "  icon-192.png"

# Tray icons
rsvg-convert -w 32 -h 32 "$SVG" -o "$ASSETS_DIR/tray-icon.png"
echo "  tray-icon.png (32x32)"

rsvg-convert -w 64 -h 64 "$SVG" -o "$ASSETS_DIR/tray-icon@2x.png"
echo "  tray-icon@2x.png (64x64)"

# Windows ICO (multi-resolution)
if command -v convert &>/dev/null; then
    TMPDIR=$(mktemp -d)
    for size in 16 32 48 64 128 256; do
        rsvg-convert -w "$size" -h "$size" "$SVG" -o "$TMPDIR/icon-${size}.png"
    done
    convert "$TMPDIR"/icon-*.png "$ASSETS_DIR/icon.ico"
    rm -rf "$TMPDIR"
    echo "  icon.ico (16,32,48,64,128,256)"
else
    echo "  SKIP icon.ico (ImageMagick not installed)"
fi

# Copy PWA icons to frontend
cp "$ASSETS_DIR/icon-192.png" "$FRONTEND_DIR/icon-192.png"
cp "$ASSETS_DIR/icon.png" "$FRONTEND_DIR/icon-512.png"
echo "  Copied PWA icons to frontend/"

echo ""
echo "Done. Generated files:"
ls -lh "$ASSETS_DIR"/icon* "$ASSETS_DIR"/tray-icon*
