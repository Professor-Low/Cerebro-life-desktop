#!/bin/sh
# Initialize AI Memory directory tree (same volume as backend)
cerebro init 2>/dev/null || true
exec "$@"
