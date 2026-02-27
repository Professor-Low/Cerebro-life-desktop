#!/bin/bash
# Post-remove script for Cerebro .deb package
# Removes the /usr/bin/cerebro symlink

if [ -L /usr/bin/cerebro ]; then
    rm -f /usr/bin/cerebro
fi
