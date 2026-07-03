#!/usr/bin/env bash
# Trigger Logger — UserPromptSubmit hook
# Records skill/multiagent/toolcall triggers to trigger_log SQLite table.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# Disable-self check
if [ -n "${DISABLED_HOOKS:-}" ] && echo "$DISABLED_HOOKS" | grep -q "trigger-logger"; then
    echo '{"continue":true,"suppressOutput":true}'
    exit 0
fi

# Read stdin
INPUT=$(cat)

# Extract prompt
PROMPT=$("$PY" -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    print(data.get('prompt', data.get('user_prompt', '')))
except Exception:
    print('')
" <<< "$INPUT" 2>/dev/null || echo "")

if [ -z "$PROMPT" ] || [ ${#PROMPT} -lt 3 ]; then
    echo '{"continue":true,"suppressOutput":true}'
    exit 0
fi

# Ensure session_id file exists (depends on session-id hook having run)
SESSION_ID=""
SID_FILE="$HARNESS_ROOT/data/session_id.txt"
if [ -f "$SID_FILE" ]; then
    SESSION_ID=$(cat "$SID_FILE" 2>/dev/null || echo "")
fi

# Run the Python logger — detects and records triggers to trigger_log
"$PY" "$SCRIPT_DIR/trigger-logger.py" --prompt "$PROMPT" --session "${SESSION_ID:-unknown}" 2>/dev/null || true

echo '{"continue":true,"suppressOutput":true}'
