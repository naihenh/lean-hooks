#!/usr/bin/env python3
"""
20-round stress test for TrainingLoop v3.0 (rolling window, no EMA).

Tests:
  1. feedback.md → meta.json counts correctness (including FP)
  2. Windowed metrics use last N entries only
  3. No-data dimensions: has_data=false, f1=null
  4. Windowed F1 replaces EMA for alerts and adjustment
  5. should_adjust respects gates with windowed F1
  6. parse_entries_from_text correctly extracts structured entries
  7. EMA fields no longer appear in new output
"""
import copy
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "training-loop"))
sys.path.insert(0, str(SCRIPT_DIR))

from metrics_core import (
    compute_metrics,
    compute_windowed_metrics,
    parse_entries_from_text,
    compute_loss,
    should_adjust,
    adjust_direction,
    adjust_magnitude,
    total_signal_count,
    DEFAULT_WINDOW_SIZE,
)

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


# ---- Round 1-5: Windowed metrics ----
print("\n=== Rounds 1-5: Sliding window metrics ===")

# Round 1: Window smaller than total
entries = [
    {"signal": "tp"}, {"signal": "tp"}, {"signal": "tp"},
    {"signal": "fp"}, {"signal": "fn"},
    {"signal": "tp"}, {"signal": "tp"}, {"signal": "tp"},
    {"signal": "tp"}, {"signal": "tp"},
]  # 10 entries: 8 tp, 1 fp, 1 fn
w1 = compute_windowed_metrics(entries, window_size=5)
check("R1: window_used=5", w1["window_used"] == 5)
check("R1: window_size=5", w1["window_size"] == 5)
# Last 5: tp,tp,tp,tp,tp → P=1, R=1, F1=1
check("R1: perfect F1=1.0", w1["f1"] > 0.99)

# Round 2: Window captures mixed quality
entries2 = [
    {"signal": "tp"}, {"signal": "fp"}, {"signal": "fp"},
    {"signal": "fn"}, {"signal": "fn"},
]  # 1 tp, 2 fp, 2 fn
w2 = compute_windowed_metrics(entries2, window_size=20)
check("R2: window_used=5 (less than window)", w2["window_used"] == 5)
check("R2: has_data=True", w2["has_data"] is True)
# P = 1/3, R = 1/3
check("R2: low precision", w2["precision"] < 0.5)
check("R2: low recall", w2["recall"] < 0.5)

# Round 3: Empty entries
w3 = compute_windowed_metrics([], window_size=20)
check("R3: has_data=False when empty", w3["has_data"] is False)
check("R3: f1=None when empty", w3["f1"] is None)

# Round 4: Window slides over old bad data
entries4 = [{"signal": "fp"}, {"signal": "fn"}] * 5 + [{"signal": "tp"}] * 10
# Last 10: all tp → perfect windowed metrics
w4 = compute_windowed_metrics(entries4, window_size=10)
check("R4: window shows recovery (F1=1.0)", w4["f1"] > 0.99)
# But cumulative would be 10 tp, 5 fp, 5 fn → much lower
cum4 = compute_metrics({"tp": 10, "fp": 5, "fn": 5})
check("R4: windowed > cumulative (key advantage)", w4["f1"] > cum4["f1"])

# Round 5: Window truncation
entries5 = [{"signal": "tp"}] * 30
w5 = compute_windowed_metrics(entries5, window_size=20)
check("R5: window_used capped at 20", w5["window_used"] == 20)


# ---- Round 6-10: parse_entries_from_text ----
print("\n=== Rounds 6-10: Entry parsing ===")

feedback_test = """## SkillOpt — Skill Trigger Accuracy

### Correct Trigger
- [skill:tdd] [prompt:"implement feature A"] correct
- [skill:tdd] [prompt:"implement feature B"] correct
- [skill:tdd] [prompt:"implement feature C"] correct

### Miss
- [skill:debug] [prompt:"program crashed"] miss

### False Positive
- [skill:ppt-master] [prompt:"make slides"] fp

## MultiAgentOpt — Agent Dispatch Accuracy

### Correct Trigger
- dispatch was correct for task

### Miss
- should have dispatched agents

### False Positive
- dispatched unnecessarily

## ToolCallOpt — Tool Call Pattern Quality

### Positive
- used codegraph instead of grep

### Missed Opportunity
- should have used codegraph

### Negative
- used grep when codegraph was available
"""

skill_labels = {"tp": "Correct Trigger", "fp": "False Positive", "fn": "Miss"}
ma_labels = {"tp": "Correct Trigger", "fp": "False Positive", "fn": "Miss"}
tc_labels = {"tp": "Positive", "fp": "Negative", "fn": "Missed Opportunity"}

# Round 6: SkillOpt entries
skill_entries = parse_entries_from_text(feedback_test, "SkillOpt", skill_labels)
check("R6: parsed 5 skill entries", len(skill_entries) == 5)
check("R6: 3 tp entries", sum(1 for e in skill_entries if e["signal"] == "tp") == 3)
check("R6: 1 fn entry", sum(1 for e in skill_entries if e["signal"] == "fn") == 1)
check("R6: 1 fp entry", sum(1 for e in skill_entries if e["signal"] == "fp") == 1)

# Round 7: MultiAgentOpt entries
ma_entries = parse_entries_from_text(feedback_test, "MultiAgentOpt", ma_labels)
check("R7: parsed 3 multiagent entries", len(ma_entries) == 3)

# Round 8: ToolCallOpt entries
tc_entries = parse_entries_from_text(feedback_test, "ToolCallOpt", tc_labels)
check("R8: parsed 3 toolcall entries", len(tc_entries) == 3)

# Round 9: Empty section
empty_entries = parse_entries_from_text("## Empty\n### Stuff\n", "Empty", {"tp": "Stuff"})
check("R9: empty section yields 0 entries", len(empty_entries) == 0)

# Round 10: Windowed metrics from parsed entries
w10 = compute_windowed_metrics(skill_entries, window_size=20)
check("R10: windowed metrics from parsed entries has_data=True", w10["has_data"] is True)
check("R10: P=0.75 (3/4)", abs(w10["precision"] - 0.75) < 0.05)
check("R10: R=0.75 (3/4)", abs(w10["recall"] - 0.75) < 0.05)


# ---- Round 11-15: should_adjust with windowed F1 ----
print("\n=== Rounds 11-15: Adjustment with windowed F1 ===")

# Round 11: adjustment blocked (disabled)
dim11 = {"counts": {"tp": 5, "fp": 1, "fn": 0}, "windowed_metrics": {"f1": 0.5}, "last_adjusted_session": 0}
cfg11 = {"adjustment_enabled": False, "min_signals_for_adjustment": 10, "f1_target": 0.75, "min_adjust_interval": 3}
check("R11: blocked when disabled", not should_adjust(dim11, cfg11, 10))

# Round 12: insufficient signals
dim12 = {"counts": {"tp": 2, "fp": 1, "fn": 0}, "windowed_metrics": {"f1": 0.5}, "last_adjusted_session": 0}
cfg12 = {"adjustment_enabled": True, "min_signals_for_adjustment": 10, "f1_target": 0.75, "min_adjust_interval": 3}
check("R12: blocked when signals < min", not should_adjust(dim12, cfg12, 10))

# Round 13: windowed F1 above target
dim13 = {"counts": {"tp": 10, "fp": 0, "fn": 0}, "windowed_metrics": {"f1": 0.9}, "last_adjusted_session": 0}
cfg13 = {"adjustment_enabled": True, "min_signals_for_adjustment": 5, "f1_target": 0.75, "min_adjust_interval": 3}
check("R13: blocked when WindowF1 >= target", not should_adjust(dim13, cfg13, 10))

# Round 14: windowed F1 is None
dim14 = {"counts": {"tp": 10, "fp": 0, "fn": 0}, "windowed_metrics": {"f1": None}, "last_adjusted_session": 0}
cfg14 = {"adjustment_enabled": True, "min_signals_for_adjustment": 5, "f1_target": 0.75, "min_adjust_interval": 3}
check("R14: blocked when WindowF1 is None", not should_adjust(dim14, cfg14, 10))

# Round 15: windowed F1 below target, allowed
dim15 = {"counts": {"tp": 10, "fp": 5, "fn": 3}, "windowed_metrics": {"f1": 0.5}, "last_adjusted_session": 0}
cfg15 = {"adjustment_enabled": True, "min_signals_for_adjustment": 10, "f1_target": 0.75, "min_adjust_interval": 3}
check("R15: allowed when all gates pass (WindowF1)", should_adjust(dim15, cfg15, 5))


# ---- Round 16-20: Integration — window vs cumulative divergence ----
print("\n=== Rounds 16-20: Window vs cumulative divergence ===")

# Simulate a system that started bad but improved
bad_then_good = [
    {"signal": "fp"}, {"signal": "fp"}, {"signal": "fp"},
    {"signal": "fn"}, {"signal": "fn"},
    {"signal": "tp"}, {"signal": "tp"}, {"signal": "tp"},
    {"signal": "tp"}, {"signal": "tp"},
]

# Round 16: Cumulative metric: 5 tp, 3 fp, 2 fn → mediocre
cum16 = compute_metrics({"tp": 5, "fp": 3, "fn": 2})
check("R16: cumulative F1 < 0.8", cum16["f1"] < 0.8)

# Round 17: Windowed metric (last 5): 5 tp → perfect
w17 = compute_windowed_metrics(bad_then_good, window_size=5)
check("R17: windowed F1 = 1.0 (recovered)", w17["f1"] > 0.99)

# Round 18: This divergence is the KEY advantage
check("R18: windowed >> cumulative (correctly shows current state)", w17["f1"] > cum16["f1"])

# Round 19: Loss uses windowed metrics, not cumulative
loss19 = compute_loss(w17, {"current": 0, "target": 2.0}, 0.15)
check("R19: loss=0 when windowed perfect", loss19["total"] < 0.01)

# Round 20: Verify no EMA-related functions are called
# (This is a structural test — just check that the function doesn't exist in v3)
check("R20: update_ema NOT in metrics_core v3", not hasattr(sys.modules.get("metrics_core", type("M")()), "update_ema"))

# ---- Summary ----
print(f"\n=== Results: {PASS} pass, {FAIL} fail ===")
if FAIL > 0:
    print("SOME TESTS FAILED!")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
    sys.exit(0)
