#!/usr/bin/env python3
"""
E2E test for MultiAgentOpt + ToolCallOpt dimensions.

Injects test entries into feedback.md, runs training-collect.py,
verifies meta.json counts/metrics/windowed_metrics/loss, then
cleans up and verifies reset to zero.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training-loop"))

from metrics_core import should_adjust as _should_adjust

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent
LOOP_DIR = CONFIG_DIR / "training-loop"
FEEDBACK = LOOP_DIR / "feedback.md"
META = LOOP_DIR / "meta.json"
COLLECT_PY = SCRIPT_DIR / "training-collect.py"

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} — {detail}")


def read_meta():
    with open(META, encoding="utf-8") as f:
        return json.load(f)


def run_collect():
    result = subprocess.run(
        [sys.executable, str(COLLECT_PY)],
        capture_output=True, text=True, encoding="utf-8",
    )
    return result


# ---- Phase 1: Backup current state ----
print("\n=== Phase 1: Backup current feedback.md ===")
backup_fb = FEEDBACK.read_text(encoding="utf-8")
backup_meta = META.read_text(encoding="utf-8")
check("backup created", True)

# ---- Phase 2: Inject MultiAgentOpt test entries ----
print("\n=== Phase 2: Inject test data ===")

# Build feedback.md with test entries for MultiAgentOpt + ToolCallOpt
# Keep existing SkillOpt data intact, add entries to the other two dimensions
injected_fb = backup_fb.replace(
    "## MultiAgentOpt — Agent Dispatch Accuracy\n\n### Correct Trigger\n### Miss\n### False Positive",
    """## MultiAgentOpt — Agent Dispatch Accuracy

### Correct Trigger
- [agents:role-collab] correct dispatch for multi-skill task
- [agents:dispatching-parallel-agents] correct dispatch for 3 independent tasks
- [agents:role-collab] correct dispatch for review+implement

### Miss
- should have dispatched parallel agents for bulk fix
- missed multiagent dispatch for complex refactor

### False Positive
- dispatched agents for a simple single-task query"""
).replace(
    "## ToolCallOpt — Tool Call Pattern Quality\n\n### Positive\n### Missed Opportunity\n### Negative",
    """## ToolCallOpt — Tool Call Pattern Quality

### Positive
- used codegraph_get_callers instead of grep for call tracing
- used codegraph_symbol_search instead of find+grep for symbol lookup

### Missed Opportunity
- should have used codegraph_get_context instead of reading 5 files
- should have used codegraph_analyze_impact instead of manual grep

### Negative
- used grep when codegraph_search was available
- used bash find for file search when codegraph supported the query"""
)

FEEDBACK.write_text(injected_fb, encoding="utf-8")
check("feedback.md injected with test data", True)

# ---- Phase 3: Run training-collect.py ----
print("\n=== Phase 3: Run training-collect.py ===")
result = run_collect()
check("training-collect.py exit code 0", result.returncode == 0, result.stderr)

# Parse stdout for dimension output
stdout_lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
ma_line = [l for l in stdout_lines if "MultiAgentOpt" in l]
tc_line = [l for l in stdout_lines if "ToolCallOpt" in l]
check("MultiAgentOpt in output", len(ma_line) > 0, f"lines: {stdout_lines}")
check("ToolCallOpt in output", len(tc_line) > 0, f"lines: {stdout_lines}")

# ---- Phase 4: Verify meta.json — MultiAgentOpt ----
print("\n=== Phase 4: Verify MultiAgentOpt meta.json ===")
meta = read_meta()
ma = meta.get("dimensions", {}).get("multiagent", {})
ma_counts = ma.get("counts", {})
ma_metrics = ma.get("metrics", {})
ma_windowed = ma.get("windowed_metrics", {})
ma_loss = ma.get("loss", {})

check("MA tp=3", ma_counts.get("tp") == 3, f"got {ma_counts.get('tp')}")
check("MA fp=1", ma_counts.get("fp") == 1, f"got {ma_counts.get('fp')}")
check("MA fn=2", ma_counts.get("fn") == 2, f"got {ma_counts.get('fn')}")
check("MA has_data=True", ma_metrics.get("has_data") is True, f"got {ma_metrics.get('has_data')}")
check("MA metrics.f1 exists", ma_metrics.get("f1") is not None, f"got {ma_metrics.get('f1')}")
check("MA metrics.precision reasonable", ma_metrics.get("precision") is not None and 0 < ma_metrics["precision"] <= 1)
check("MA metrics.recall reasonable", ma_metrics.get("recall") is not None and 0 < ma_metrics["recall"] <= 1)

# P = 3/(3+1) = 0.75, R = 3/(3+2) = 0.6
check("MA precision=0.75", abs(ma_metrics.get("precision", 0) - 0.75) < 0.01,
      f"got {ma_metrics.get('precision')}")
check("MA recall=0.6", abs(ma_metrics.get("recall", 0) - 0.6) < 0.01,
      f"got {ma_metrics.get('recall')}")

# Windowed metrics (all 6 entries fit in window of 20)
check("MA windowed has_data=True", ma_windowed.get("has_data") is True)
check("MA windowed.window_used=6", ma_windowed.get("window_used") == 6,
      f"got {ma_windowed.get('window_used')}")
check("MA windowed.f1 = cumulative f1 (all fit in window)",
      ma_windowed.get("f1") is not None and abs(ma_windowed["f1"] - ma_metrics["f1"]) < 0.01)

# Loss should be > 0 (not perfect)
check("MA loss.total > 0", ma_loss.get("total", 0) > 0,
      f"got {ma_loss.get('total')}")
check("MA loss.core > 0", ma_loss.get("core", 0) > 0)

# ---- Phase 5: Verify meta.json — ToolCallOpt ----
print("\n=== Phase 5: Verify ToolCallOpt meta.json ===")
tc = meta.get("dimensions", {}).get("toolcall", {})
tc_counts = tc.get("counts", {})
tc_metrics = tc.get("metrics", {})
tc_windowed = tc.get("windowed_metrics", {})
tc_loss = tc.get("loss", {})

check("TC tp=2", tc_counts.get("tp") == 2, f"got {tc_counts.get('tp')}")
check("TC fp=2", tc_counts.get("fp") == 2, f"got {tc_counts.get('fp')}")
check("TC fn=2", tc_counts.get("fn") == 2, f"got {tc_counts.get('fn')}")  # Missed Opportunity(2) + Negative(0) for fn
# Actually fn = Missed Opportunity entries, because Negative = fp
# Let me recalculate: labels = {"tp": "Positive", "fp": "Negative", "fn": "Missed Opportunity"}
# tp = Positive entries = 2
# fp = Negative entries = 2
# fn = Missed Opportunity entries = 2
check("TC has_data=True", tc_metrics.get("has_data") is True)
check("TC metrics.f1 exists", tc_metrics.get("f1") is not None)
# P = 2/(2+2) = 0.5, R = 2/(2+2) = 0.5
check("TC precision=0.5", abs(tc_metrics.get("precision", 0) - 0.5) < 0.01,
      f"got {tc_metrics.get('precision')}")
check("TC recall=0.5", abs(tc_metrics.get("recall", 0) - 0.5) < 0.01,
      f"got {tc_metrics.get('recall')}")

# Windowed metrics
check("TC windowed has_data=True", tc_windowed.get("has_data") is True)
check("TC windowed.window_used=6", tc_windowed.get("window_used") == 6,
      f"got {tc_windowed.get('window_used')}")

# Loss > 0
check("TC loss.total > 0", tc_loss.get("total", 0) > 0)

# ---- Phase 6: Verify SkillOpt unchanged ----
print("\n=== Phase 6: Verify SkillOpt unchanged ===")
sk = meta.get("dimensions", {}).get("skill", {})
sk_counts = sk.get("counts", {})
check("SK tp=6 (unchanged)", sk_counts.get("tp") == 6)
check("SK fp=1 (unchanged)", sk_counts.get("fp") == 1)
check("SK fn=2 (unchanged)", sk_counts.get("fn") == 2)

# ---- Phase 7: Verify version and global config ----
print("\n=== Phase 7: Verify global config integrity ===")
check("version=3.0", meta.get("version") == "3.0")
check("window_size=20", meta.get("global", {}).get("window_size") == 20)
check("ema_lambda removed", "ema_lambda" not in meta.get("global", {}))
check("adjustment_enabled=False", meta.get("global", {}).get("adjustment_enabled") is False)

# ---- Phase 8: Verify windowed F1 < f1_target triggers alert logic ----
print("\n=== Phase 8: Verify alert logic ===")
f1_target = meta.get("global", {}).get("f1_target", 0.75)
# MA: P=0.75, R=0.6, F1≈0.667 → below target 0.75
ma_f1 = ma_windowed.get("f1")
check("MA windowed F1 < f1_target (should alert)",
      ma_f1 is not None and ma_f1 < f1_target,
      f"windowed_f1={ma_f1}, target={f1_target}")
# TC: P=0.5, R=0.5, F1=0.5 → below target
tc_f1 = tc_windowed.get("f1")
check("TC windowed F1 < f1_target (should alert)",
      tc_f1 is not None and tc_f1 < f1_target,
      f"windowed_f1={tc_f1}, target={f1_target}")

# ---- Phase 9: should_adjust blocked (adjustment_enabled=False) ----
print("\n=== Phase 9: Verify adjustment blocked ===")
# Even though F1 < target and we have 6 signals each,
# adjustment_enabled=False should block it
check("MA should_adjust=False (disabled)", not _should_adjust(ma, meta["global"], meta["sessions"]))
check("TC should_adjust=False (disabled)", not _should_adjust(tc, meta["global"], meta["sessions"]))

# Also test: if enabled but signals < 10, still blocked
mock_global = dict(meta["global"])
mock_global["adjustment_enabled"] = True
check("MA should_adjust=False (signals<10)", not _should_adjust(ma, mock_global, meta["sessions"]))
check("TC should_adjust=False (signals<10)", not _should_adjust(tc, mock_global, meta["sessions"]))

# And: if enabled + enough signals (mock signals), should fire
mock_ma = dict(ma)
mock_ma["counts"] = {"tp": 10, "fp": 5, "fn": 3}
check("MA should_adjust=True (enabled+signals+F1<target)",
      _should_adjust(mock_ma, mock_global, 100))

# ---- Phase 10: Cleanup and verify reset ----
print("\n=== Phase 10: Cleanup — restore original feedback.md ===")
# Re-read backup to avoid any mutation issues
FEEDBACK.write_text(backup_fb, encoding="utf-8")
# Verify feedback.md was actually restored
restored_fb = FEEDBACK.read_text(encoding="utf-8")
check("feedback.md restored", restored_fb == backup_fb, "content mismatch after restore")
# Reset meta.json to original state by running collect on restored feedback
result_pre = run_collect()
check("pre-restore collect exit 0", result_pre.returncode == 0)

# Run collect again to verify MultiAgentOpt/ToolCallOpt back to 0
print("\n=== Phase 11: Verify reset after cleanup ===")
meta2 = read_meta()
ma2 = meta2.get("dimensions", {}).get("multiagent", {})
tc2 = meta2.get("dimensions", {}).get("toolcall", {})
check("MA2 tp=0 after cleanup", ma2.get("counts", {}).get("tp") == 0)
check("MA2 fp=0 after cleanup", ma2.get("counts", {}).get("fp") == 0)
check("MA2 fn=0 after cleanup", ma2.get("counts", {}).get("fn") == 0)
check("MA2 has_data=False", ma2.get("metrics", {}).get("has_data") is False)
check("MA2 f1=None", ma2.get("metrics", {}).get("f1") is None)
check("MA2 windowed f1=None", ma2.get("windowed_metrics", {}).get("f1") is None)

check("TC2 tp=0 after cleanup", tc2.get("counts", {}).get("tp") == 0)
check("TC2 fp=0 after cleanup", tc2.get("counts", {}).get("fp") == 0)
check("TC2 fn=0 after cleanup", tc2.get("counts", {}).get("fn") == 0)
check("TC2 has_data=False", tc2.get("metrics", {}).get("has_data") is False)
check("TC2 f1=None", tc2.get("metrics", {}).get("f1") is None)
check("TC2 windowed f1=None", tc2.get("windowed_metrics", {}).get("f1") is None)

# SkillOpt should be unchanged
sk2 = meta2.get("dimensions", {}).get("skill", {})
check("SK2 tp=6 preserved", sk2.get("counts", {}).get("tp") == 6)

# ---- Summary ----
print(f"\n=== Results: {PASS} pass, {FAIL} fail ===")
if FAIL > 0:
    print("SOME TESTS FAILED!")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
    sys.exit(0)
