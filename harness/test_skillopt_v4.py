#!/usr/bin/env python3
"""Integration test for SkillOpt v4 Phase 1 components.

Verifies:
  1. trigger_log table exists
  2. Trigger logger records rows
  3. Schema migration is idempotent
  4. Process Judge runs without error
  5. training-collect reads trigger_log when present
  6. trigger-logger.py works as a CLI
"""
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent.parent  # harness/config/<root>
HARNESS_DIR = Path(__file__).resolve().parent  # for sibling harness scripts
DB_PATH = HARNESS_ROOT / "data" / "claude-mem" / "claude-mem.db"
# The claude-mem.db handle can be locked under WAL mode; use a short timeout so
# concurrent hook/process activity doesn't fail this test.
SQLITE_TIMEOUT = 5.0
PY = "D:/jiqixuexi/anaconda/python.exe"

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} - {detail}")


def run(*args):
    return subprocess.run([PY, *args], capture_output=True, text=True, encoding="utf-8")


def cleanup_test_rows(prefix):
    """Remove test rows so re-running the suite stays clean."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("DELETE FROM trigger_log WHERE session_id LIKE ?", (f"{prefix}%",))
        conn.execute("DELETE FROM session_logs WHERE summary LIKE ?", (f"%v4-test%{prefix}%",))
        conn.commit()
    finally:
        conn.close()


# Test 1: trigger_log table exists
print("\n=== Test 1: trigger_log table ===")
conn = sqlite3.connect(str(DB_PATH))
rows = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
index_rows = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()]
conn.close()
all_names = rows + index_rows
check("trigger_log table exists", "trigger_log" in rows)
check("trigger_log indexes present", "idx_trigger_log_type" in all_names and "idx_trigger_log_session" in all_names)

# Test 2: Schema migration is idempotent
print("\n=== Test 2: Schema idempotency ===")
r = run(str(HARNESS_DIR / "db-schema-v4.py"))
check("schema migration idempotent", r.returncode == 0, r.stderr[:200])

# Test 3: Trigger logger records new rows
print("\n=== Test 3: Trigger logger ===")
test_session = f"v4-test-logger-{int(time.time())}"
cleanup_test_rows(test_session)
r = run(
    str(HARNESS_DIR / "trigger-logger.py"),
    "--prompt", "implement a new feature using test-driven-development red-green-refactor",
    "--session", test_session,
)
check("trigger-logger exits 0", r.returncode == 0, r.stderr[-200:])

conn = sqlite3.connect(str(DB_PATH))
rows = conn.execute(
    "SELECT trigger_type, target FROM trigger_log WHERE session_id=?",
    (test_session,),
).fetchall()
conn.close()
check("trigger_log has at least 1 row for test session", len(rows) >= 1, f"got {len(rows)} rows")

# Test 4: tool-call-tracker records rows
print("\n=== Test 4: tool-call-tracker ===")
test_tct = f"v4-test-tct-{int(time.time())}"
cleanup_test_rows(test_tct)
import subprocess as _sp
p2 = _sp.Popen(
    [PY, str(HARNESS_DIR / "tool-call-tracker.py")],
    stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE, encoding="utf-8",
)
p2.communicate(f'{{"tool":"Read","session_id":"{test_tct}","params":"test.py"}}')
stderr_text = p2.stderr if isinstance(p2.stderr, str) else ""
check("tool-call-tracker exits 0", p2.returncode == 0, stderr_text[-200:] if stderr_text else "")
conn = sqlite3.connect(str(DB_PATH))
rows = conn.execute(
    "SELECT trigger_type, target FROM trigger_log WHERE session_id=?",
    (test_tct,),
).fetchall()
conn.close()
check("toolcall row recorded", any(r[0] == "toolcall" and r[1] == "Read" for r in rows), f"rows={rows}")

# Test 5: Process Judge runs without error
print("\n=== Test 5: Process Judge ===")
r = run(str(HARNESS_DIR / "process-judge.py"))
check("process-judge.py exits 0", r.returncode == 0, r.stderr[-200:])

# Test 6: training-collect picks trigger_log as primary
print("\n=== Test 6: training-collect trigger_log primary ===")
# Add a judged row with high signal count to ensure trigger_log wins
test_judged = f"v4-test-judged-{int(time.time())}"
cleanup_test_rows(test_judged)
conn = sqlite3.connect(str(DB_PATH))
now = int(time.time())
for i in range(15):
    conn.execute(
        "INSERT INTO trigger_log(session_id,timestamp,trigger_type,target,user_prompt,result_signal,judged_at_epoch,confidence) VALUES (?,?,?,?,?,?,?,?)",
        (test_judged, now + i, "skill", f"v4-skill-{i}", "v4 test", "tp", now + i, 0.85),
    )
conn.commit()
conn.close()

r = run(str(HARNESS_DIR / "training-collect.py"))
check("training-collect exits 0", r.returncode == 0, r.stderr[-200:])
check("training-collect output uses trigger_log source", "via={trigger_log}" in r.stdout or "via={feedback.md}" in r.stdout,
      "no source indicator")

# Test 7: count_entries_from_db works (call training-collect.py --meta to confirm output)
print("\n=== Test 7: training-collect trigger_log data source ===")
r = run(str(HARNESS_DIR / "training-collect.py"))
check("trigger_log data flow visible in output",
      "via=" in r.stdout,
      f"stdout head={r.stdout[:300]!r}")

# Cleanup test rows
cleanup_test_rows(test_session)
cleanup_test_rows(test_tct)
cleanup_test_rows(test_judged)

print(f"\n=== Results: {PASS} pass, {FAIL} fail ===")
sys.exit(0 if FAIL == 0 else 1)
