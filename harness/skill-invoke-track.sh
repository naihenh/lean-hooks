#!/usr/bin/env bash
# Skill Invocation Tracker wrapper — calls skill-invoke-track.py
# Usage: echo '{"skill":"name","signal":"correct|miss|fp","prompt":"text"}' | bash skill-invoke-track.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

INPUT=$(cat)

"$PY" "$SCRIPT_DIR/skill-invoke-track.py" <<< "$INPUT" 2>/dev/null || true
