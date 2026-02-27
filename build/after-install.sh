#!/bin/bash
# Post-install script for Cerebro .deb package
# Creates /usr/bin/cerebro symlink so users can launch from terminal

# The electron-builder deb installs the binary at /opt/cerebro-desktop/cerebro-desktop
BINARY_PATH="/opt/cerebro-desktop/cerebro-desktop"

if [ -f "$BINARY_PATH" ]; then
    ln -sf "$BINARY_PATH" /usr/bin/cerebro
fi
