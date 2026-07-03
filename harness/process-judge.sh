#!/usr/bin/env bash
# Process Judge — Stop hook
# Audits session trigger decisions and writes judgment signals back to trigger_log.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

if [ -n "${DISABLED_HOOKS:-}" ] && echo "$DISABLED_HOOKS" | grep -q "process-judge"; then
    echo '{"continue":true,"suppressOutput":true}'
    exit 0
fi

# Run Process Judge
"$PY" "$SCRIPT_DIR/process-judge.py" 2>/dev/null || true

echo '{"continue":true,"suppressOutput":true}'
