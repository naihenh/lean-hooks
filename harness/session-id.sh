#!/usr/bin/env bash
# Session ID tracker — SessionStart hook
# Writes a persistent session ID to data/session_id.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SESSION_FILE="$HARNESS_ROOT/data/session_id.txt"
mkdir -p "$(dirname "$SESSION_FILE")"

# Generate new session ID each session start
SESSION_ID="session_$(date +%s)_$$"
echo "$SESSION_ID" > "$SESSION_FILE"

echo '{"continue":true,"suppressOutput":true}'
