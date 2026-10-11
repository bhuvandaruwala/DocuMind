#!/bin/bash
# Double-click this file in Finder (or run `bash start_mac.command` in Terminal)
# to set up (first run) and start the app on macOS.
# Requires Python 3.10-3.13 (https://www.python.org/downloads/ or `brew install python@3.12`).
cd "$(dirname "$0")" || exit 1

PY=""
for cand in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] else 1)' 2>/dev/null; then
      PY="$cand"; break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo ""
  echo "Python 3.10 or newer was not found."
  echo "Install Python 3.12 from https://www.python.org/downloads/ and run this file again."
  echo ""
  read -r -p "Press Enter to close..."
  exit 1
fi

"$PY" start.py "$@"
