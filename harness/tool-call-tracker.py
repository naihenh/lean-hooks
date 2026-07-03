#!/usr/bin/env python3
"""Tool Call Tracker — records tool invocations to trigger_log.

Called by AI at tool-call time (Phase 1: lightweight helper).
Usage:
    echo '{"tool":"Read","params":"file.py"}' | python tool-call-tracker.py

Phase 2 (deferred): auto-track via PreToolUse hook event.
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

HARNESS_ROOT = Path(os.environ.get("HARNESS_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
DB_PATH = HARNESS_ROOT / "data" / "claude-mem" / "claude-mem.db"


def main():
    if not DB_PATH.exists():
        return

    raw = sys.stdin.read().strip()
    if not raw:
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return

    tool = data.get("tool", "")
    params = str(data.get("params", {}))
    session_id = data.get("session_id", "")

    if not session_id:
        sid_file = HARNESS_ROOT / "data" / "session_id.txt"
        if sid_file.exists():
            session_id = sid_file.read_text(encoding="utf-8").strip()

    if not tool:
        return

    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            "INSERT INTO trigger_log (session_id, timestamp, trigger_type, target, user_prompt, result_signal, confidence) "
            "VALUES (?, ?, 'toolcall', ?, ?, 'tp', 0.6)",
            (session_id, int(time.time()), tool, params[:500]),
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
