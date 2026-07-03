#!/usr/bin/env python3
"""Records trigger events to trigger_log SQLite table.

Called by trigger-logger.sh hook on UserPromptSubmit.
Detects which skill/agent was triggered and writes the trigger to the log.

Phase 1 implementation:
- Reuses skill-attention.py query if available (cost_matrix + topk A/B)
- Falls back to direct SQL probe

Usage:
trigger-logger.py --prompt "user message" --session "session-xyz"
"""
import argparse
import importlib.util
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

HARNESS_ROOT = Path(os.environ.get("HARNESS_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
DB_PATH = HARNESS_ROOT / "data" / "claude-mem" / "claude-mem.db"


def get_db_path():
    if not DB_PATH.exists():
        return None
    return str(DB_PATH)


def get_session_id(session_arg: str) -> str:
    if session_arg and session_arg != "unknown":
        return session_arg
    sid_file = HARNESS_ROOT / "data" / "session_id.txt"
    if sid_file.exists():
        return sid_file.read_text(encoding="utf-8").strip()
    return f"session_{int(time.time())}"


def ensure_trigger_log_table(conn):
    """Ensure trigger_log table exists with v4 + A/B columns."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(trigger_log)").fetchall()] \
        if _table_exists(conn, "trigger_log") else []
    if not cols:
        conn.executescript("""
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
    task_outcome TEXT,
    ranking_strategy TEXT DEFAULT 'legacy'
);
""")
    elif "ranking_strategy" not in cols:
        # Idempotent migration for pre-existing tables created before ranking_strategy.
        conn.execute(
            "ALTER TABLE trigger_log ADD COLUMN ranking_strategy TEXT DEFAULT 'legacy'"
        )
        conn.executescript("""
CREATE INDEX IF NOT EXISTS idx_trigger_log_type ON trigger_log(trigger_type, target);
CREATE INDEX IF NOT EXISTS idx_trigger_log_session ON trigger_log(session_id);
CREATE INDEX IF NOT EXISTS idx_trigger_log_strategy ON trigger_log(ranking_strategy);
""")


def _table_exists(conn, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def _load_skill_attention_module():
    """Import skill-attention.py dynamically (avoids circular imports)."""
    try:
        sys.path.insert(0, str(HARNESS_ROOT / "config" / "harness"))
        spec = importlib.util.spec_from_file_location(
            "skill_attention",
            str(HARNESS_ROOT / "config" / "harness" / "skill-attention.py"),
        )
        sa = importlib.util.module_from_spec(spec)
        if "skill_attention" not in sys.modules:
            sys.modules["skill_attention"] = sa
        spec.loader.exec_module(sa)
        return sa
    except Exception:
        return None


def query_skill_attention(prompt: str, db_path: str, strategy: str = "cost_matrix") -> list[dict]:
    """Query skill-attention using the ranking strategy, fall back to SQL keyword match.

    strategy: "cost_matrix" (default) uses embedding + Hungarian assignment;
              "topk" uses top-K greedy similarity.
    Falls back to SQL keyword match when embeddings are unavailable.
    """
    sa = _load_skill_attention_module()
    if sa:
        try:
            results = sa.query_skills(prompt, top_k=3, strategy=strategy)
            if results:
                return results
        except Exception:
            pass

    # Fallback: SQL keyword match
    try:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='skill_attention'")
            if not cur.fetchone():
                return []
        except Exception:
            return []
        rows = conn.execute(
            "SELECT skill_name, utterance FROM skill_attention LIMIT 100"
        ).fetchall()
        conn.close()
        matches = []
        prompt_lower = prompt.lower()
        for skill_name, utterance in rows:
            if utterance and any(tok in prompt_lower for tok in utterance.lower().split() if len(tok) > 3):
                matches.append({"skill": skill_name, "weighted_sim": 0.5, "attention_weight": 1.0})
        return matches[:3]
    except Exception:
        return []


def write_trigger(db_path: str, session_id: str, trigger_type: str, target: str,
    prompt: str, cosine_sim=None, attention_weight=None,
    confidence=0.6, signal='tp', ranking_strategy=None):
    conn = sqlite3.connect(db_path)
    try:
        ensure_trigger_log_table(conn)
        conn.execute(
            """INSERT INTO trigger_log
            (session_id, timestamp, trigger_type, target, user_prompt,
            cosine_sim, attention_weight, result_signal, confidence, ranking_strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, int(time.time()), trigger_type, target, prompt[:1000],
             cosine_sim, attention_weight, signal, confidence, ranking_strategy),
        )
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--session", default="unknown")
    args = parser.parse_args()

    db_path = get_db_path()
    if not db_path:
        return

    session_id = get_session_id(args.session)

    # A/B: record skill triggers from both ranking strategies
    strategies = ["cost_matrix", "topk"]
    for strat in strategies:
        try:
            skills = query_skill_attention(args.prompt, db_path, strategy=strat)
            for s in skills[:3]:  # cap at top-3
                write_trigger(
                    db_path, session_id, "skill", s["skill"],
                    args.prompt, s.get("weighted_sim"), s.get("attention_weight"),
                    ranking_strategy=s.get("ranking_strategy", strat),
                )
        except Exception as e:
            print(f"[trigger-logger] skill phase ({strat}): {e}", file=sys.stderr)

    # Multiagent triggers via state file inspection
    state_file = HARNESS_ROOT / "data" / "multiagent_session_state.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if state.get("task_active"):
                write_trigger(
                    db_path, session_id, "multiagent",
                    state.get("last_mode", "unknown"),
                    args.prompt, confidence=0.7,
                )
        except Exception:
            pass


if __name__ == "__main__":
    main()