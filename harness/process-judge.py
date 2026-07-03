#!/usr/bin/env python3
"""Process Judge — audits session trigger decisions and writes to trigger_log.

Stop hook: reads trigger_log entries for the current session, computes cost
metrics, runs lightweight rule-based judgment, and writes back TP/FP/FN signals.

Phase 1 implementation (this file):
- Cost anomaly heuristics: marks triggers as lower-confidence when the session
  made excessive tool calls for a skill.
- Low-cosine-similarity detection: flips skill triggers with very weak
  semantic match to FP.

Phase 2 (deferred to later work): replace rule-based judgments with an AI
prompt audit using a lightweight LLM.

Usage:
    process-judge.py
"""
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

HARNESS_ROOT = Path(os.environ.get("HARNESS_ROOT", str(Path(__file__).resolve().parent.parent.parent)))


def get_db_path():
    db = HARNESS_ROOT / "data" / "claude-mem" / "claude-mem.db"
    return str(db) if db.exists() else None


def get_session_id():
    sid_file = HARNESS_ROOT / "data" / "session_id.txt"
    if sid_file.exists():
        return sid_file.read_text(encoding="utf-8").strip()
    return None


def get_session_metrics(conn, session_id):
    """Compute cost metrics for this session."""
    skills = conn.execute(
        "SELECT target, COUNT(*) as n FROM trigger_log WHERE session_id=? AND trigger_type='skill' GROUP BY target",
        (session_id,)
    ).fetchall()

    tool_count = conn.execute(
        "SELECT COUNT(*) FROM trigger_log WHERE session_id=? AND trigger_type='toolcall'",
        (session_id,)
    ).fetchone()[0]

    return {
        "skills": {r[0]: r[1] for r in skills},
        "tool_calls": tool_count,
        "total_triggers": sum(r[1] for r in skills) + tool_count,
    }


def get_historical_avg(conn, trigger_type):
    """Historical tool calls per skill for triggers this type."""
    try:
        rows = conn.execute(
            """SELECT target, AVG(tool_calls) as avg_calls, COUNT(*) as n
               FROM (
                   SELECT tl.target, COUNT(*) as tool_calls
                   FROM trigger_log tl
                   WHERE tl.trigger_type = ?
                     AND tl.judged_at_epoch IS NOT NULL
                   GROUP BY tl.session_id, tl.target
               ) GROUP BY target""",
            (trigger_type,),
        ).fetchall()
    except Exception:
        return {}
    return {r[0]: {"avg_calls": r[1] or 5, "n": r[2]} for r in rows}


def detect_anomalies(session_metrics, historical_avg):
    signals = []
    tc = session_metrics["tool_calls"]
    for skill, count in session_metrics["skills"].items():
        hist = historical_avg.get(skill, {"avg_calls": 5, "n": 0})
        if hist["n"] >= 3 and tc > 2 * hist["avg_calls"]:
            signals.append(f"excessive_tool_calls({skill})")
    return signals


def judge_triggers(conn, session_id, anomalies):
    """Rule-based judgment for triggers in the current session."""
    triggers = conn.execute(
        "SELECT id, trigger_type, target, cosine_sim FROM trigger_log "
        "WHERE session_id=? AND judged_at_epoch IS NULL",
        (session_id,)
    ).fetchall()

    for tid, ttype, target, cosine_sim in triggers:
        signal = "tp"
        flipped = None
        reason = None
        confidence = 0.6
        path_efficiency = 3

        # Low similarity → FP for skill triggers
        if ttype == "skill" and cosine_sim is not None and cosine_sim < 0.3:
            signal = "fp"
            flipped = "tp"
            reason = "low semantic similarity to skill utterances"
            confidence = 0.5
            path_efficiency = 2

        # Anomalies lower all skill confidences
        if anomalies and ttype == "skill" and signal == "tp":
            confidence = 0.4

        conn.execute(
            "UPDATE trigger_log SET result_signal=?, flipped_from=?, judge_reason=?, "
            "judged_at_epoch=?, confidence=?, path_efficiency=? WHERE id=?",
            (signal, flipped, reason, int(time.time()), confidence,
             path_efficiency, tid)
        )

    conn.commit()


def main():
    db_path = get_db_path()
    session_id = get_session_id()
    if not db_path or not session_id:
        return

    try:
        conn = sqlite3.connect(db_path)
    except Exception:
        return
    try:
        # Skip if no unjudged triggers
        n_unjudged = conn.execute(
            "SELECT COUNT(*) FROM trigger_log WHERE session_id=? AND judged_at_epoch IS NULL",
            (session_id,)
        ).fetchone()[0]
        if n_unjudged == 0:
            return

        metrics = get_session_metrics(conn, session_id)
        if metrics["total_triggers"] == 0:
            return

        historical = get_historical_avg(conn, "skill")
        anomalies = detect_anomalies(metrics, historical)
        judge_triggers(conn, session_id, anomalies)

        n_judged = conn.execute(
            "SELECT COUNT(*) FROM trigger_log WHERE session_id=? AND judged_at_epoch IS NOT NULL",
            (session_id,)
        ).fetchone()[0]

        print(f"[ProcessJudge] Judged {n_judged} triggers, anomalies: {len(anomalies)}", file=sys.stderr)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
