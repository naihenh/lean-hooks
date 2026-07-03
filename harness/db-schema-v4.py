#!/usr/bin/env python3
"""Schema migration for SkillOpt v4 — adds trigger_log table to claude-mem.db."""
import os, sqlite3, sys
from pathlib import Path

HARNESS_ROOT = Path(os.environ.get("HARNESS_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
DB_PATH = HARNESS_ROOT / "data" / "claude-mem" / "claude-mem.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trigger_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    trigger_type TEXT NOT NULL,
    target TEXT NOT NULL,
    user_prompt TEXT NOT NULL,
    cosine_sim REAL,
    attention_weight REAL,
    result_signal TEXT DEFAULT 'tp',
    path_efficiency INTEGER,
    flipped_from TEXT,
    judge_reason TEXT,
    judged_at_epoch INTEGER,
    confidence REAL DEFAULT 0.6,
    task_outcome TEXT
);
CREATE INDEX IF NOT EXISTS idx_trigger_log_type ON trigger_log(trigger_type, target);
CREATE INDEX IF NOT EXISTS idx_trigger_log_session ON trigger_log(session_id);
"""

def main():
    if not DB_PATH.exists():
        print(f"[db-schema-v4] claude-mem.db not found at {DB_PATH}, skipping")
        return
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
    print(f"[db-schema-v4] trigger_log table created/verified in {DB_PATH}")

if __name__ == "__main__":
    main()
