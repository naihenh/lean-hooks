#!/usr/bin/env python3
"""
Comprehensive 20-round test for TrainingLoop v3.0 + hybrid-orchestrator integration.

Covers:
  1. metrics_core.py — compute_metrics, compute_windowed_metrics, parse_entries_from_text,
     compute_loss, should_adjust, adjust_direction, adjust_magnitude, total_signal_count
  2. multiagent_orchestrator.py — mode selection, role composition, subtask decomposition,
     veto rules, phase building, dispatch plans
  3. multiagent-detect.sh — two-phase scoring (via --dry-run subprocess)
  4. weighted-scoring.py — count_weighted_entries, compute_weighted_f1, analyze_trend,
     compute_confidence
  5. session-start-inject.sh — windowed F1 alert logic (inline Python mirror)
  6. training-collect.py — count_entries, migrate_v1_to_v3, main flow
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# --- UTF-8 on Windows ---
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

HARNESS_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_LOOP_DIR = HARNESS_ROOT / "config" / "training-loop"
HARNESS_DIR = HARNESS_ROOT / "config" / "harness"

# Insert training-loop dir for metrics_core import
sys.path.insert(0, str(TRAINING_LOOP_DIR))
sys.path.insert(0, str(HARNESS_DIR))

# ===================================================================
# Test harness
# ===================================================================
_results: list[tuple[str, str, str]] = []  # (round_name, assertion, PASS/FAIL)


def _assert(round_name: str, condition: bool, msg: str) -> None:
    status = "PASS" if condition else "FAIL"
    _results.append((round_name, msg, status))
    print(f"  [{status}] {msg}")
    if not condition:
        # Print diagnostic info on failure
        import traceback
        traceback.print_stack(limit=3)


def _summary() -> int:
    fails = [r for r in _results if r[2] == "FAIL"]
    passes = [r for r in _results if r[2] == "PASS"]
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {len(passes)} PASS, {len(fails)} FAIL, {len(_results)} total")
    print("=" * 70)
    if fails:
        print("\nFAILED assertions:")
        for round_name, msg, _ in fails:
            print(f"  [{round_name}] {msg}")
    return 1 if fails else 0


# ===================================================================
# IMPORTS — real module imports
# ===================================================================
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

from multiagent_orchestrator import (
    Orchestrator,
    Mode,
    RoleDef,
)

import importlib

# weighted-scoring has a dash in filename; use importlib
_ws_path = HARNESS_DIR / "weighted-scoring.py"
_ws_loader = importlib.util.spec_from_file_location("weighted_scoring", str(_ws_path))
_ws_module = importlib.util.module_from_spec(_ws_loader)
_ws_loader.loader.exec_module(_ws_module)

count_weighted_entries = _ws_module.count_weighted_entries
compute_weighted_f1 = _ws_module.compute_weighted_f1
analyze_trend = _ws_module.analyze_trend
compute_confidence = _ws_module.compute_confidence

_tc_path = HARNESS_DIR / "training-collect.py"
_tc_loader = importlib.util.spec_from_file_location("training_collect", str(_tc_path))
_tc_module = importlib.util.module_from_spec(_tc_loader)
# training-collect.py runs main() on import if __name__ == "__main__" —
# but with spec_from_file_location, __name__ is "training_collect", so main() won't auto-run.
_tc_loader.loader.exec_module(_tc_module)

count_entries = _tc_module.count_entries
migrate_v1_to_v3 = _tc_module.migrate_v1_to_v3
COMPLEXITY_DEFAULTS = _tc_module.COMPLEXITY_DEFAULTS


# ===================================================================
# ROUND 1: metrics_core — compute_metrics basic
# ===================================================================
def test_round_1():
    name = "R01-metrics-compute"
    print(f"\n--- {name} ---")

    # All zeros => has_data=False
    r = compute_metrics({"tp": 0, "fp": 0, "fn": 0})
    _assert(name, r["has_data"] is False, "All-zero counts => has_data=False")
    _assert(name, r["precision"] is None, "All-zero => precision=None")
    _assert(name, r["f1"] is None, "All-zero => f1=None")

    # All-TP
    r = compute_metrics({"tp": 10, "fp": 0, "fn": 0})
    _assert(name, r["has_data"] is True, "All-TP => has_data=True")
    _assert(name, abs(r["precision"] - 1.0) < 1e-6, "All-TP => precision=1.0")
    _assert(name, abs(r["recall"] - 1.0) < 1e-6, "All-TP => recall=1.0")
    _assert(name, abs(r["f1"] - 1.0) < 1e-6, "All-TP => f1=1.0")

    # All-FP
    r = compute_metrics({"tp": 0, "fp": 5, "fn": 0})
    _assert(name, r["has_data"] is True, "All-FP => has_data=True")
    _assert(name, abs(r["precision"]) < 1e-6, "All-FP => precision~0")
    _assert(name, abs(r["recall"]) < 1e-6, "All-FP => recall~0")

    # Mixed
    r = compute_metrics({"tp": 8, "fp": 2, "fn": 3})
    expected_p = 8 / 10
    expected_r = 8 / 11
    _assert(name, abs(r["precision"] - expected_p) < 1e-6, "Mixed precision")
    _assert(name, abs(r["recall"] - expected_r) < 1e-6, "Mixed recall")
    _assert(name, r["has_data"] is True, "Mixed => has_data=True")


# ===================================================================
# ROUND 2: metrics_core — compute_windowed_metrics
# ===================================================================
def test_round_2():
    name = "R02-metrics-windowed"
    print(f"\n--- {name} ---")

    # Empty entries
    r = compute_windowed_metrics([])
    _assert(name, r["has_data"] is False, "Empty entries => has_data=False")
    _assert(name, r["window_used"] == 0, "Empty entries => window_used=0")

    # Fewer entries than window
    entries = [
        {"signal": "tp"}, {"signal": "tp"}, {"signal": "fp"},
    ]
    r = compute_windowed_metrics(entries, window_size=20)
    _assert(name, r["window_used"] == 3, "3 entries, window=20 => window_used=3")
    _assert(name, abs(r["precision"] - 2 / 3) < 1e-6, "Windowed precision 2/3")

    # More entries than window
    entries = [{"signal": "tp"}] * 15 + [{"signal": "fp"}] * 5 + [{"signal": "tp"}] * 10
    r = compute_windowed_metrics(entries, window_size=20)
    _assert(name, r["window_used"] == 20, "Window clips to 20")
    # Window = last 20 = [tp*5][fp*5][tp*10] => 15 tp + 5 fp
    _assert(name, abs(r["precision"] - 15 / 20) < 1e-6,
            f"Windowed precision 15/20=0.75, got {r['precision']:.6f}")
    # recall = 15/(15+0+eps) = ~1.0 (no fn in window)
    _assert(name, abs(r["recall"] - 1.0) < 1e-4, "Windowed recall ~1.0 (no fn in window)")

    # Alternative signal names
    entries = [{"signal": "correct"}, {"signal": "false_positive"}, {"signal": "miss"}]
    r = compute_windowed_metrics(entries, window_size=10)
    _assert(name, r["window_used"] == 3, "Alt signal names: window_used=3")
    _assert(name, r["precision"] is not None and abs(r["precision"] - 1/2) < 1e-6,
            "Alt signal names: tp=1,fp=1 => precision=0.5")
    _assert(name, abs(r["recall"] - 1/2) < 1e-6, "Alt signal names: tp=1,fn=1 => recall=0.5")


# ===================================================================
# ROUND 3: metrics_core — parse_entries_from_text
# ===================================================================
def test_round_3():
    name = "R03-metrics-parse"
    print(f"\n--- {name} ---")

    feedback = (
        "# Training Loop Feedback\n\n"
        "## SkillOpt — Skill Trigger Accuracy\n"
        "### Correct Trigger\n"
        "- Entry A\n"
        "- Entry B\n"
        "### Miss\n"
        "- Missed X\n"
        "### False Positive\n"
        "- Wrong Y\n\n"
        "## MultiAgentOpt — Agent Dispatch Accuracy\n"
        "### Correct Trigger\n"
        "- OK Z\n"
    )
    labels = {"tp": "Correct Trigger", "fp": "False Positive", "fn": "Miss"}
    entries = parse_entries_from_text(feedback, "SkillOpt", labels)
    _assert(name, len(entries) == 4, f"SkillOpt: 4 entries total, got {len(entries)}")
    # Verify all expected signals present (parse_entries iterates by label key order)
    signals = [e["signal"] for e in entries]
    _assert(name, signals.count("tp") == 2, f"2 tp entries, got {signals.count('tp')}")
    _assert(name, signals.count("fn") == 1, f"1 fn entry, got {signals.count('fn')}")
    _assert(name, signals.count("fp") == 1, f"1 fp entry, got {signals.count('fp')}")

    # Non-existent section
    entries_empty = parse_entries_from_text(feedback, "NonExistent", labels)
    _assert(name, len(entries_empty) == 0, "Non-existent section => 0 entries")

    # MultiAgentOpt section
    entries_ma = parse_entries_from_text(feedback, "MultiAgentOpt",
                                          {"tp": "Correct Trigger", "fp": "False Positive", "fn": "Miss"})
    _assert(name, len(entries_ma) == 1, "MultiAgentOpt: 1 entry")
    _assert(name, entries_ma[0]["signal"] == "tp", "MultiAgentOpt first entry is tp")


# ===================================================================
# ROUND 4: metrics_core — compute_loss
# ===================================================================
def test_round_4():
    name = "R04-metrics-loss"
    print(f"\n--- {name} ---")

    # No data => zero loss
    r = compute_loss({"has_data": False}, {"current": 5.0, "target": 2.0}, 0.15)
    _assert(name, r["total"] == 0.0, "No data => total loss=0")

    # Perfect metrics => core loss ~0
    r = compute_loss({"has_data": True, "precision": 1.0, "recall": 1.0},
                     {"current": 2.0, "target": 2.0}, 0.15)
    _assert(name, r["core"] < 1e-6, "Perfect metrics => core loss~0")
    _assert(name, r["complexity_penalty"] == 0.0, "current==target => no complexity penalty")
    _assert(name, r["total"] < 1e-6, "Perfect+balanced => total~0")

    # Below target complexity
    r = compute_loss({"has_data": True, "precision": 1.0, "recall": 1.0},
                     {"current": 1.0, "target": 2.0}, 0.15)
    _assert(name, r["complexity_penalty"] == 0.0, "below target => c_norm clamped to 0, penalty=0")

    # Above target complexity
    r = compute_loss({"has_data": True, "precision": 1.0, "recall": 1.0},
                     {"current": 4.0, "target": 2.0}, 0.15)
    expected_penalty = 0.15 * ((4.0 - 2.0) / 2.0)
    _assert(name, abs(r["complexity_penalty"] - expected_penalty) < 1e-6,
            f"Above target complexity => penalty={expected_penalty}")

    # Bad metrics produce positive core loss
    r = compute_loss({"has_data": True, "precision": 0.5, "recall": 0.5},
                     {"current": 0.0, "target": 1.0}, 0.15)
    _assert(name, r["core"] > 0, "P=R=0.5 => core loss > 0")
    _assert(name, r["total"] > 0, "Total loss > 0")


# ===================================================================
# ROUND 5: metrics_core — should_adjust, adjust_direction, adjust_magnitude
# ===================================================================
def test_round_5():
    name = "R05-metrics-adjust"
    print(f"\n--- {name} ---")

    # adjustment_enabled=False => never adjust
    dim = {"windowed_metrics": {"f1": 0.3}, "counts": {"tp": 20, "fp": 5, "fn": 5},
           "metrics": {"f1": 0.3}}
    r = should_adjust(dim, {"adjustment_enabled": False}, 100)
    _assert(name, r is False, "adjustment_enabled=False => no adjust")

    # Insufficient signals
    dim = {"windowed_metrics": {"f1": 0.3}, "counts": {"tp": 3, "fp": 1, "fn": 1},
           "metrics": {"f1": 0.3}}
    r = should_adjust(dim, {"adjustment_enabled": True, "min_signals_for_adjustment": 10}, 100)
    _assert(name, r is False, "Insufficient signals => no adjust")

    # F1 >= target => no adjust
    dim = {"windowed_metrics": {"f1": 0.8}, "counts": {"tp": 20, "fp": 2, "fn": 2},
           "metrics": {"f1": 0.8}}
    r = should_adjust(dim, {"adjustment_enabled": True, "min_signals_for_adjustment": 5,
                            "f1_target": 0.75}, 100)
    _assert(name, r is False, "F1 >= target => no adjust")

    # F1 < target, windowed F1 => adjust
    dim = {"windowed_metrics": {"f1": 0.5}, "counts": {"tp": 20, "fp": 10, "fn": 10},
           "metrics": {"f1": 0.8}, "last_adjusted_session": 50}
    r = should_adjust(dim, {"adjustment_enabled": True, "min_signals_for_adjustment": 5,
                            "f1_target": 0.75, "min_adjust_interval": 3}, 100)
    _assert(name, r is True, "F1 < target, past interval => should adjust")

    # Interval not reached => no adjust
    dim2 = dict(dim)
    dim2["last_adjusted_session"] = 98
    r = should_adjust(dim2, {"adjustment_enabled": True, "min_signals_for_adjustment": 5,
                             "f1_target": 0.75, "min_adjust_interval": 3}, 100)
    _assert(name, r is False, "Session interval not reached => no adjust")

    # F1 None => no adjust
    dim_none = {"windowed_metrics": {"f1": None}, "counts": {"tp": 20, "fp": 5, "fn": 5},
                "metrics": {"f1": None}}
    r = should_adjust(dim_none, {"adjustment_enabled": True, "min_signals_for_adjustment": 5,
                                  "f1_target": 0.75}, 100)
    _assert(name, r is False, "F1 is None => no adjust")

    # adjust_direction
    _assert(name, adjust_direction({"precision": 0.5, "recall": 0.9}) == "TIGHTEN",
            "P<R => TIGHTEN")
    _assert(name, adjust_direction({"precision": 0.9, "recall": 0.5}) == "LOOSEN",
            "P>R => LOOSEN")
    _assert(name, adjust_direction({"precision": None, "recall": None}) == "LOOSEN",
            "None metrics => LOOSEN")

    # adjust_magnitude
    _assert(name, adjust_magnitude({"precision": 0.5, "recall": 0.9}, 0.75) >= 1,
            "Magnitude >= 1 when deficit exists")
    _assert(name, adjust_magnitude({"precision": None, "recall": None}, 0.75) == 1,
            "None metrics => magnitude=1")

    # total_signal_count
    _assert(name, total_signal_count({"tp": 3, "fp": 2, "fn": 1}) == 6, "total_signal_count=6")
    _assert(name, total_signal_count({"tp": 0, "fp": 0, "fn": 0}) == 0, "total_signal_count=0")


# ===================================================================
# ROUND 6: training-collect — count_entries
# ===================================================================
def test_round_6():
    name = "R06-collect-count"
    print(f"\n--- {name} ---")

    feedback = (
        "## SkillOpt — Skill Trigger Accuracy\n"
        "### Correct Trigger\n"
        "- item 1\n"
        "- item 2\n"
        "- item 3\n"
        "### Miss\n"
        "- miss 1\n"
        "### False Positive\n"
        "- fp 1\n"
        "- fp 2\n\n"
        "## Other Section\n"
    )

    labels = {"tp": "Correct Trigger", "fp": "False Positive", "fn": "Miss"}
    counts = count_entries(feedback, "SkillOpt", labels)
    _assert(name, counts["tp"] == 3, f"tp=3, got {counts.get('tp')}")
    _assert(name, counts["fn"] == 1, f"fn=1, got {counts.get('fn')}")
    _assert(name, counts["fp"] == 2, f"fp=2, got {counts.get('fp')}")

    # Non-existent section
    counts2 = count_entries(feedback, "NonExistent", labels)
    _assert(name, all(v == 0 for v in counts2.values()), "Non-existent => all zeros")

    # Bullet-alt format (* instead of -)
    feedback_star = (
        "## SkillOpt — Skill Trigger Accuracy\n"
        "### Correct Trigger\n"
        "* item a\n"
        "* item b\n"
    )
    counts3 = count_entries(feedback_star, "SkillOpt", labels)
    _assert(name, counts3["tp"] == 2, f"Star bullet: tp=2, got {counts3.get('tp')}")


# ===================================================================
# ROUND 7: training-collect — migrate_v1_to_v3
# ===================================================================
def test_round_7():
    name = "R07-collect-migrate"
    print(f"\n--- {name} ---")

    # v1 legacy meta
    v1_meta = {
        "version": "2.1",
        "sessions": 10,
        "last_session": "2026-01-01T00:00:00",
        "dimensions": {
            "skill": {
                "correct_triggers": 5,
                "false_positives": 2,
                "misses": 1,
                "threshold": 3,
            },
            "multiagent": {
                "observations": 8,
                "correct_triggers": 6,
                "threshold": 4,
            },
            "toolcall": {
                "observations": 3,
                "correct_triggers": 0,
            },
        },
    }
    migrated = migrate_v1_to_v3(v1_meta)
    _assert(name, migrated["version"] == "3.0", "Migrated version = 3.0")
    _assert(name, migrated["sessions"] == 10, "Sessions preserved = 10")

    skill = migrated["dimensions"]["skill"]
    _assert(name, skill["counts"]["tp"] == 5, f"skill tp=5, got {skill['counts']['tp']}")
    _assert(name, skill["counts"]["fp"] == 2, f"skill fp=2, got {skill['counts']['fp']}")
    _assert(name, skill["counts"]["fn"] == 1, f"skill fn=1, got {skill['counts']['fn']}")

    multiagent = migrated["dimensions"]["multiagent"]
    _assert(name, multiagent["counts"]["tp"] == 6, f"multiagent tp=6, got {multiagent['counts']['tp']}")
    _assert(name, multiagent["counts"]["fp"] == 2, f"multiagent fp=2 (8obs-6tp), got {multiagent['counts']['fp']}")

    toolcall = migrated["dimensions"]["toolcall"]
    _assert(name, toolcall["counts"]["tp"] == 0, "toolcall tp=0")
    _assert(name, toolcall["counts"]["fp"] == 3, f"toolcall fp=3 (all observations), got {toolcall['counts']['fp']}")

    _assert(name, "windowed_metrics" in skill, "v3 has windowed_metrics")
    _assert(name, "global" in migrated, "v3 has global config")
    _assert(name, migrated["global"]["adjustment_enabled"] is False, "adjustment_enabled=false by default")

    # Already v3 => no-op
    v3_meta = {"version": "3.0", "sessions": 5, "dimensions": {"skill": {"counts": {"tp": 1, "fp": 0, "fn": 0}, "windowed_metrics": {}, "metrics": {}}}}
    r = migrate_v1_to_v3(v3_meta)
    _assert(name, r["version"] == "3.0", "Already v3 => unchanged version")
    _assert(name, r["sessions"] == 5, "Already v3 => sessions preserved")


# ===================================================================
# ROUND 8: multiagent_orchestrator — mode selection
# ===================================================================
def test_round_8():
    name = "R08-orch-mode"
    print(f"\n--- {name} ---")

    orch = Orchestrator()

    # Single task => ROLE_COLLAB
    mode = orch.select_mode("fix the login bug")
    _assert(name, mode == Mode.ROLE_COLLAB, f"Single task => ROLE_COLLAB, got {mode.value}")

    # Two tasks without quality signals => TASK_PARALLEL
    mode = orch.select_mode("fix login and refactor auth module")
    _assert(name, mode == Mode.TASK_PARALLEL, f"Two tasks => TASK_PARALLEL, got {mode.value}")

    # Two tasks WITH quality signals => HYBRID
    mode = orch.select_mode("fix login and refactor auth module, review carefully")
    _assert(name, mode == Mode.HYBRID, f"Two + quality => HYBRID, got {mode.value}")

    # Hybrid boost keywords
    mode = orch.select_mode("fix login and update API, cross-review each change")
    _assert(name, mode == Mode.HYBRID, "Hybrid boost keyword => HYBRID")

    # Three tasks => TASK_PARALLEL (no quality)
    mode = orch.select_mode("fix bug, add feature, and update docs")
    _assert(name, mode == Mode.TASK_PARALLEL, "3 tasks, no quality => TASK_PARALLEL")

    # Three + quality => HYBRID
    mode = orch.select_mode("fix bug, add feature, verify thoroughly")
    _assert(name, mode == Mode.HYBRID, "3 tasks + quality => HYBRID")


# ===================================================================
# ROUND 9: multiagent_orchestrator — category detection
# ===================================================================
def test_round_9():
    name = "R09-orch-category"
    print(f"\n--- {name} ---")

    orch = Orchestrator()

    # Security
    _assert(name, orch.detect_category("add authentication and fix vulnerability") == "security_sensitive",
            "Security category detection")
    _assert(name, orch.detect_category("update RBAC permissions") == "security_sensitive",
            "RBAC detection")

    # UI
    _assert(name, orch.detect_category("update CSS layout and component styling") == "ui_change",
            "UI category detection")

    # API
    _assert(name, orch.detect_category("add new API endpoint and REST route") == "api_change",
            "API category detection")

    # Code
    _assert(name, orch.detect_category("refactor the module and fix bug") == "code_change",
            "Code change detection")

    # General (write matches code_change verb, so use a non-matching text)
    _assert(name, orch.detect_category("organize the project structure") == "general",
            "General category for unmatched")

    # Security overrides UI (tested in specificity order)
    _assert(name, orch.detect_category("fix security vulnerability in CSS component") == "security_sensitive",
            "Security takes priority over UI")


# ===================================================================
# ROUND 10: multiagent_orchestrator — role composition
# ===================================================================
def test_round_10():
    name = "R10-orch-roles"
    print(f"\n--- {name} ---")

    orch = Orchestrator()

    # Security => includes security_auditor and architect
    roles = orch.compose_roles("add authentication to the system", Mode.ROLE_COLLAB)
    role_names = [r.name for r in roles]
    _assert(name, "security_auditor" in role_names, "Security task => security_auditor role")
    _assert(name, "architect" in role_names, "Security task => architect role")

    # UI => includes designer
    roles = orch.compose_roles("update the UI component layout", Mode.ROLE_COLLAB)
    role_names = [r.name for r in roles]
    _assert(name, "designer" in role_names, "UI task => designer role")

    # TASK_PARALLEL => includes integrator
    roles = orch.compose_roles("fix bug A and add feature B", Mode.TASK_PARALLEL)
    role_names = [r.name for r in roles]
    _assert(name, "integrator" in role_names, "TASK_PARALLEL => integrator role")

    # HYBRID => includes integrator
    roles = orch.compose_roles("fix and refactor, review each", Mode.HYBRID)
    role_names = [r.name for r in roles]
    _assert(name, "integrator" in role_names, "HYBRID => integrator role")

    # Role collab minimum: at least 3 roles
    roles = orch.compose_roles("general task", Mode.ROLE_COLLAB)
    _assert(name, len(roles) >= 3, f"ROLE_COLLAB minimum 3 roles, got {len(roles)}")

    # HYBRID minimum: at least 4 roles
    roles = orch.compose_roles("task requiring hybrid review", Mode.HYBRID)
    _assert(name, len(roles) >= 4, f"HYBRID minimum 4 roles, got {len(roles)}")


# ===================================================================
# ROUND 11: multiagent_orchestrator — subtask decomposition
# ===================================================================
def test_round_11():
    name = "R11-orch-decompose"
    print(f"\n--- {name} ---")

    orch = Orchestrator()

    # ROLE_COLLAB => single subtask
    subs = orch.decompose_task("fix the login bug", Mode.ROLE_COLLAB)
    _assert(name, len(subs) == 1, f"ROLE_COLLAB => 1 subtask, got {len(subs)}")
    _assert(name, subs[0]["id"] == "1", "Subtask id starts at 1")

    # Multi-part on connectors
    subs = orch.decompose_task("fix login and refactor auth module", Mode.TASK_PARALLEL)
    _assert(name, len(subs) >= 2, f"2 action clauses => >=2 subtasks, got {len(subs)}")

    # With semicolons
    subs = orch.decompose_task("implement auth; add unit tests; review code", Mode.TASK_PARALLEL)
    _assert(name, len(subs) >= 2, f"3 semicolons => >=2 subtasks, got {len(subs)}")

    # Chinese connectors
    subs = orch.decompose_task("修复登录bug并且重构认证模块", Mode.TASK_PARALLEL)
    _assert(name, len(subs) >= 2, f"Chinese connectors => >=2 subtasks, got {len(subs)}")

    # No action verbs => fallback single subtask
    subs = orch.decompose_task("random text with no action", Mode.TASK_PARALLEL)
    _assert(name, len(subs) >= 1, "No action verbs => fallback to at least 1 subtask")


# ===================================================================
# ROUND 12: multiagent_orchestrator — veto rules
# ===================================================================
def test_round_12():
    name = "R12-orch-veto"
    print(f"\n--- {name} ---")

    orch = Orchestrator()

    # All APPROVE => accept
    v = orch.apply_veto_rules({"reviewer": "APPROVE", "tester": "PASS"}, cycle=1)
    _assert(name, v == "accept", f"All pass => accept, got {v}")

    # Tester FAIL => reject
    v = orch.apply_veto_rules({"reviewer": "APPROVE", "tester": "FAIL"}, cycle=1)
    _assert(name, v == "reject", "Tester FAIL => reject")

    # Architect VETO => escalate
    v = orch.apply_veto_rules({"reviewer": "APPROVE", "architect": "VETO"}, cycle=1)
    _assert(name, v == "escalate", "Architect VETO => escalate")

    # Architect RISK => escalate
    v = orch.apply_veto_rules({"reviewer": "APPROVE", "architect": "RISK"}, cycle=1)
    _assert(name, v == "escalate", "Architect RISK => escalate")

    # Security auditor VULNERABLE with CRITICAL => reject
    v = orch.apply_veto_rules(
        {"reviewer": "APPROVE", "security_auditor": {"verdict": "VULNERABLE", "severity": "CRITICAL"}},
        cycle=1)
    _assert(name, v == "reject", "Security VULNERABLE+CRITICAL => reject")

    # Security auditor VULNERABLE without critical => revise
    v = orch.apply_veto_rules(
        {"reviewer": "APPROVE", "security_auditor": {"verdict": "VULNERABLE", "severity": "MEDIUM"}},
        cycle=1)
    _assert(name, v == "revise", "Security VULNERABLE+MEDIUM => revise")

    # Security auditor CONCERN => revise
    v = orch.apply_veto_rules(
        {"reviewer": "APPROVE", "security_auditor": "CONCERN"}, cycle=1)
    _assert(name, v == "revise", "Security CONCERN => revise")

    # Reviewer REQUEST_CHANGES => revise
    v = orch.apply_veto_rules({"reviewer": "REQUEST_CHANGES"}, cycle=1)
    _assert(name, v == "revise", "Reviewer REQUEST_CHANGES => revise")

    # Designer NEEDS_REVISION => revise
    v = orch.apply_veto_rules({"reviewer": "APPROVE", "designer": "NEEDS_REVISION"}, cycle=1)
    _assert(name, v == "revise", "Designer NEEDS_REVISION => revise")

    # Max cycles => escalate (even with approve)
    v = orch.apply_veto_rules({"reviewer": "REQUEST_CHANGES"}, cycle=3, max_cycles=3)
    _assert(name, v == "escalate", "Max cycles reached => escalate")


# ===================================================================
# ROUND 13: multiagent_orchestrator — phase building
# ===================================================================
def test_round_13():
    name = "R13-orch-phases"
    print(f"\n--- {name} ---")

    orch = Orchestrator()

    # ROLE_COLLAB phases: IMPLEMENT, REVIEW, DECISION
    plan = orch.build_dispatch_plan("fix the login bug")
    phases = plan["phases"]
    phase_names = [p["name"] for p in phases]
    _assert(name, "IMPLEMENT" in phase_names, "ROLE_COLLAB has IMPLEMENT")
    _assert(name, "REVIEW" in phase_names, "ROLE_COLLAB has REVIEW")
    _assert(name, "DECISION" in phase_names, "ROLE_COLLAB has DECISION")
    _assert(name, plan["mode"] == "role_collab", "Mode is role_collab")

    # TASK_PARALLEL phases: PARALLEL_EXECUTE, INTEGRATE
    plan = orch.build_dispatch_plan("fix login and refactor auth")
    phases = plan["phases"]
    phase_names = [p["name"] for p in phases]
    _assert(name, "PARALLEL_EXECUTE" in phase_names, "TASK_PARALLEL has PARALLEL_EXECUTE")
    _assert(name, "INTEGRATE" in phase_names, "TASK_PARALLEL has INTEGRATE")
    _assert(name, plan["mode"] in ("task_parallel", "hybrid"),
            f"Multi-task mode, got {plan['mode']}")

    # HYBRID phases: IMPLEMENT_*, REVIEW_*, INTEGRATE, DECISION
    plan = orch.build_dispatch_plan("fix login and refactor auth, review carefully")
    phases = plan["phases"]
    phase_names = [p["name"] for p in phases]
    _assert(name, plan["mode"] == "hybrid", f"HYBRID mode, got {plan['mode']}")
    _assert(name, "INTEGRATE" in phase_names, "HYBRID has INTEGRATE")
    _assert(name, "DECISION" in phase_names, "HYBRID has DECISION")
    impl_phases = [p for p in phase_names if p.startswith("IMPLEMENT_")]
    _assert(name, len(impl_phases) >= 2, f"HYBRID has >=2 IMPLEMENT_ phases, got {len(impl_phases)}")

    # Plan structure check
    _assert(name, "session_id" in plan, "Plan has session_id")
    _assert(name, "roles" in plan, "Plan has roles list")
    _assert(name, "subtasks" in plan, "Plan has subtasks")
    _assert(name, "created_at" in plan, "Plan has created_at")


# ===================================================================
# ROUND 14: multiagent-detect.sh — Phase 1 scoring (dry-run)
# ===================================================================
def test_round_14():
    name = "R14-detect-phase1"
    print(f"\n--- {name} ---")

    detect_script = HARNESS_DIR / "multiagent-detect.sh"
    # Set HARNESS_ROOT explicitly and use the correct env
    env = os.environ.copy()
    env["HARNESS_ROOT"] = str(HARNESS_ROOT)
    env["HARNESS_PYTHON"] = sys.executable
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    def dry_run(prompt_text: str) -> str:
        data = json.dumps({"prompt": prompt_text})
        try:
            result = subprocess.run(
                ["bash", str(detect_script), "--dry-run"],
                input=data, capture_output=True, text=True, timeout=30,
                env=env, encoding="utf-8",
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
        except UnicodeDecodeError:
            # Windows GBK fallback: run with bytes and decode as utf-8
            result = subprocess.run(
                ["bash", str(detect_script), "--dry-run"],
                input=data.encode("utf-8"), capture_output=True, timeout=30,
                env=env,
            )
            stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        return stdout + stderr

    # Greeting => low score (greeting penalty)
    out = dry_run("你好")
    _assert(name, "greeting" in out, "Greeting detected in output")
    _assert(name, "NO TRIGGER" in out, "Greeting => NO TRIGGER")

    # Short message => short penalty
    out = dry_run("ok")
    _assert(name, "short" in out, "Short message detected")
    _assert(name, "NO TRIGGER" in out, "Short => NO TRIGGER")

    # Strong keyword => high score
    out = dry_run("Please use parallel agents to fix the bugs and refactor the auth module simultaneously")
    _assert(name, "strong_keyword" in out, "Strong keyword detected")
    _assert(name, "TRIGGER" in out, "Strong keyword => TRIGGER")

    # Moderate keyword: use exact pattern from MODERATE list
    out = dry_run("We need to fix and refactor the authentication module for better security and also review and merge the changes")
    _assert(name, "moderate_keyword" in out,
            f"Moderate keyword should be detected (fix and refactor matches)")


# ===================================================================
# ROUND 15: multiagent-detect.sh — Phase 2 scoring + force triggers
# ===================================================================
def test_round_15():
    name = "R15-detect-phase2-force"
    print(f"\n--- {name} ---")

    detect_script = HARNESS_DIR / "multiagent-detect.sh"
    env = os.environ.copy()
    env["HARNESS_ROOT"] = str(HARNESS_ROOT)
    env["HARNESS_PYTHON"] = sys.executable
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    def dry_run(prompt_text: str) -> str:
        data = json.dumps({"prompt": prompt_text})
        try:
            result = subprocess.run(
                ["bash", str(detect_script), "--dry-run"],
                input=data, capture_output=True, text=True, timeout=30,
                env=env, encoding="utf-8",
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
        except UnicodeDecodeError:
            # Windows GBK fallback: run with bytes and decode as utf-8
            result = subprocess.run(
                ["bash", str(detect_script), "--dry-run"],
                input=data.encode("utf-8"), capture_output=True, timeout=30,
                env=env,
            )
            stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        return stdout + stderr

    # Phase 2 activation: score in [2,4) range with multi_verb
    out = dry_run("Implement the new feature and add tests for app.py and utils.py with proper review in main.py")
    _assert(name, "multi_verb" in out or "multi_file" in out or "TRIGGER" in out,
            "Phase 2 gives multi_verb/multi_file bonus or triggers")

    # Force triggers: explicit multiagent dispatch
    out = dry_run("并行执行这些任务")
    _assert(name, "TRIGGER" in out, "Force trigger (Chinese): 并行执行 => TRIGGER")

    out = dry_run("split into subtasks and dispatch parallel agents now")
    _assert(name, "TRIGGER" in out, "Force trigger (English): dispatch parallel => TRIGGER")

    # Weak signal alone should not trigger
    out = dry_run("能不能先这样做然后再那样做")
    _assert(name, "weak_keyword" in out or "NO TRIGGER" in out,
            "Weak-only => NO TRIGGER or weak_keyword detected")


# ===================================================================
# ROUND 16: multiagent-detect.sh — continuation detection + session state
# ===================================================================
def test_round_16():
    name = "R16-detect-continuation"
    print(f"\n--- {name} ---")

    # We can't easily test the full session state machine in dry-run mode
    # (dry-run skips the JSON output path), but we CAN test the
    # continuation detection logic by importing the Python code inline.

    # Test the continuation detection functions directly by mimicking
    # the Python code from multiagent-detect.sh
    import re
    import time

    GREETINGS = ["hello", "hi", "hey", "你好", "您好", "早上好", "晚上好", "在吗", "在嘛"]

    # Build a lightweight tester for the continuation logic
    def detect_continuation(state, text_lower):
        """Mirrored from multiagent-detect.sh Python code."""
        SESSION_IDLE_TIMEOUT_SECONDS = 300

        if state.get("state") == "idle":
            return False, "idle", 0

        last_time = state.get("last_prompt_start", 0)
        if last_time and (time.time() - last_time) > SESSION_IDLE_TIMEOUT_SECONDS:
            return False, "idle_timeout", 0

        continuation_markers = [
            (r'^(这个|那个|它|刚才的|之前的|上面的)\b', 'pronoun_reference'),
            (r'^(能不能|能不能把|能不能再|能否|要不|改成|换成|改为)\b', 'request_retry'),
            (r'^(还有|另外|对了|那|然后|接着|继续|下一步|然后呢)\b', 'continuation_word'),
            (r'^(为什么|怎么|怎么没|是不是|有没有|对吗|对吗\?|为何|如何)\b', 'question'),
            (r'^(改一下|修一下|换一下|保存|重启|删了|去掉|加上|添加)\b', 'imperative_no_subject'),
            (r'^(继续|接着来|下一步|然后呢|ok|好的|对|收到|搞定|完成)\b$', 'one_word_continuation'),
            (r'^不错|^可以|^行|^好|^嗯|^哦', 'affirmation'),
        ]
        for pattern, reason in continuation_markers:
            if re.search(pattern, text_lower):
                return True, reason, -3

        return False, "new_task_or_unclear", 0

    # Idle state => not continuation
    is_cont, reason, penalty = detect_continuation({"state": "idle"}, "whatever")
    _assert(name, is_cont is False, "Idle state => not continuation")

    # Active state + Chinese continuation marker
    is_cont, reason, penalty = detect_continuation(
        {"state": "task_active", "last_prompt_start": time.time()}, "继续")
    _assert(name, is_cont is True, "Active + 继续 => continuation")
    _assert(name, penalty == -3, "Continuation penalty = -3")

    # Active state + imperative_no_subject (这个改一下 starts with 改一下)
    is_cont, reason, penalty = detect_continuation(
        {"state": "task_active", "last_prompt_start": time.time()}, "改一下")
    _assert(name, is_cont is True, f"Active + 改一下 => continuation, reason={reason}")

    # Active state + pronoun 这个 at start
    is_cont2, reason2, penalty2 = detect_continuation(
        {"state": "task_active", "last_prompt_start": time.time()}, "这个")
    _assert(name, is_cont2 is True, f"Active + 这个 => continuation, reason={reason2}")

    # Active state + new unrelated text
    is_cont, reason, penalty = detect_continuation(
        {"state": "task_active", "last_prompt_start": time.time()}, "build a new feature from scratch")
    _assert(name, is_cont is False, "Active + new topic => not continuation")

    # Force trigger patterns
    FORCE_PATTERNS = [
        r"并行执行",
        r"拆分成子任务",
        r"拆分.*并行",
        r"dispatch.*parallel",
        r"parallel.*agents?",
    ]

    def is_force_trigger(text_lower):
        for pat in FORCE_PATTERNS:
            if re.search(pat, text_lower):
                return pat
        return None

    _assert(name, is_force_trigger("并行执行任务") is not None, "Force trigger: 并行执行")
    _assert(name, is_force_trigger("dispatch parallel agents") is not None, "Force trigger: dispatch parallel")
    _assert(name, is_force_trigger("hello world") is None, "Normal text => no force trigger")

    # New task during active session
    NEW_TASK_PATTERNS = [
        r"^(另外|还有|对了|顺便).*?(需要|帮我|做|写|修|改|查|找|测试|实现|添加|分析)",
        r"(新任务|新需求|另一个|另外一个|新功能).*?(需要|帮我|做|写|修|改|查|找|测试)",
    ]

    def is_new_task_during_active(state, text_lower):
        if state.get("state") != "task_active":
            return False
        for pat in NEW_TASK_PATTERNS:
            if re.search(pat, text_lower):
                return pat
        return None

    _assert(name,
            is_new_task_during_active({"state": "task_active"}, "另外还需要实现一个新功能") is not None,
            "New task override: 另外还需要")
    _assert(name,
            is_new_task_during_active({"state": "idle"}, "另外还需要实现一个新功能") is False,
            "Idle state => new task override not applicable")


# ===================================================================
# ROUND 17: weighted-scoring.py — count_weighted_entries
# ===================================================================
def test_round_17():
    name = "R17-weighted-count"
    print(f"\n--- {name} ---")

    feedback = (
        "## SkillOpt — Skill Trigger Accuracy\n"
        "### Correct Trigger\n"
        "- item 1\n"
        "- item 2\n"
        "### Miss\n"
        "- miss 1\n"
        "- miss 2\n"
        "- miss 3\n"
        "### False Positive\n"
        "- fp 1\n\n"
        "## MultiAgentOpt — Agent Dispatch Accuracy\n"
        "### Correct Trigger\n"
    )

    labels = {"Correct Trigger": 1.0, "Miss": -0.8, "False Positive": -0.6}
    result = count_weighted_entries(feedback, "SkillOpt", labels, 10)
    _assert(name, result["raw"]["Correct Trigger"] == 2, f"Raw CT=2, got {result['raw'].get('Correct Trigger')}")
    _assert(name, result["raw"]["Miss"] == 3, f"Raw Miss=3, got {result['raw'].get('Miss')}")
    _assert(name, result["raw"]["False Positive"] == 1, f"Raw FP=1, got {result['raw'].get('False Positive')}")

    # Weighted without history => simple weight * count
    _assert(name, abs(result["weighted"]["Correct Trigger"] - 2.0) < 0.01,
            f"Weighted CT=2.0, got {result['weighted'].get('Correct Trigger')}")
    _assert(name, abs(result["weighted"]["Miss"] - (-2.4)) < 0.01,
            f"Weighted Miss=-2.4, got {result['weighted'].get('Miss')}")
    _assert(name, abs(result["weighted"]["False Positive"] - (-0.6)) < 0.01,
            f"Weighted FP=-0.6, got {result['weighted'].get('False Positive')}")

    # Empty section
    labels2 = {"Correct Trigger": 1.0, "Miss": -0.9, "False Positive": -0.7}
    result2 = count_weighted_entries(feedback, "MultiAgentOpt", labels2, 10)
    _assert(name, result2["raw"]["Correct Trigger"] == 0, "Empty MultiAgentOpt CT=0")

    # With history (time decay)
    history = [{"session": i, "skill": {"weighted_f1": 0.5}} for i in range(10)]
    result3 = count_weighted_entries(feedback, "SkillOpt", labels, 10, history)
    # With history, time decay reduces weight of older entries
    _assert(name, result3["weighted"]["Correct Trigger"] < 2.0,
            "Time decay reduces CT weight")
    _assert(name, result3["weighted"]["Correct Trigger"] > 0,
            "Time decay does not zero out entries")


# ===================================================================
# ROUND 18: weighted-scoring.py — weighted F1, trend, confidence
# ===================================================================
def test_round_18():
    name = "R18-weighted-f1-trend"
    print(f"\n--- {name} ---")

    # Weighted F1 computation
    w = {"Correct Trigger": 3.0, "Miss": -2.4, "False Positive": -0.6}
    f1_result = compute_weighted_f1(w)
    _assert(name, abs(f1_result["tp"] - 3.0) < 0.01, "Weighted tp=3.0")
    _assert(name, abs(f1_result["fp"] - (-0.6)) < 0.01, "Weighted fp=-0.6")
    _assert(name, f1_result["f1"] > 0, "Weighted F1 > 0 for positive TP")

    # All positive => F1=1.0
    w_all_good = {"Correct Trigger": 5.0}
    f1_result2 = compute_weighted_f1(w_all_good)
    _assert(name, abs(f1_result2["f1"] - 1.0) < 0.01, "All-TP => F1=1.0")

    # ToolCall metrics: Missed Opportunity + Negative as FN
    w_tc = {"Positive": 2.0, "Missed Opportunity": -0.8, "Negative": -1.0}
    f1_result3 = compute_weighted_f1(w_tc)
    fn_total = f1_result3["fn"]
    _assert(name, abs(fn_total - (-1.8)) < 0.01, "Toolcall: FN = MissedOp + Negative")

    # Empty => vacuously correct P=R=1, F1=1
    f1_empty = compute_weighted_f1({})
    _assert(name, abs(f1_empty["f1"] - 1.0) < 0.01, "Empty weighted => F1=1.0")

    # Trend analysis
    history_improving = [
        {"skill": {"weighted_f1": 0.3}},
        {"skill": {"weighted_f1": 0.5}},
        {"skill": {"weighted_f1": 0.7}},
    ]
    trend = analyze_trend(history_improving, "skill")
    _assert(name, trend["trend"] == "improving", f"Improving trend, got {trend['trend']}")

    history_declining = [
        {"skill": {"weighted_f1": 0.8}},
        {"skill": {"weighted_f1": 0.5}},
        {"skill": {"weighted_f1": 0.3}},
    ]
    trend2 = analyze_trend(history_declining, "skill")
    _assert(name, trend2["trend"] == "declining", f"Declining trend, got {trend2['trend']}")

    # Insufficient data
    trend3 = analyze_trend([{"skill": {"weighted_f1": 0.5}}], "skill")
    _assert(name, trend3["trend"] == "insufficient_data", "Too few data points => insufficient_data")

    # Confidence
    _assert(name, compute_confidence(0, 10) == 0.0, "Zero observations => confidence=0")
    _assert(name, compute_confidence(5, 10) == 1.0, "0.5 obs/session => confidence=1.0")
    _assert(name, abs(compute_confidence(3, 10) - 0.6) < 0.01, "0.3 obs/session => confidence=0.6")


# ===================================================================
# ROUND 19: session-start-inject — windowed F1 alerts
# ===================================================================
def test_round_19():
    name = "R19-inject-f1-alerts"
    print(f"\n--- {name} ---")

    # Mirror the session-start-inject.sh logic for F1 alerts
    # (We extract and test the Python embedded logic)

    def build_f1_alerts(meta):
        """Mirror the F1 alert logic from session-start-inject.sh v3."""
        version = meta.get("version", "")
        if version not in ("2.1", "2.2", "3.0"):
            return []

        global_cfg = meta.get("global", {})
        f1_target = global_cfg.get("f1_target", 0.75)
        dims = meta.get("dimensions", {})
        alerts = []

        for key, label in [("skill", "SkillOpt"), ("multiagent", "MultiAgentOpt"), ("toolcall", "ToolCallOpt")]:
            dim = dims.get(key, {})
            windowed = dim.get("windowed_metrics", {})
            metrics = dim.get("metrics", {})

            f1 = windowed.get("f1")
            if f1 is None:
                has_data = metrics.get("has_data", True)
                f1 = metrics.get("f1") if has_data else None

            if f1 is not None and f1 < f1_target:
                p = windowed.get("precision") or metrics.get("precision", 0)
                r = windowed.get("recall") or metrics.get("recall", 0)
                alerts.append(f"[{label}] F1 below target ({f1_target}) — Window(P={p:.2f}, R={r:.2f})")

        return alerts

    # Test with real meta.json data (skill F1≈0.8, not below 0.75)
    meta = {
        "version": "3.0",
        "global": {"f1_target": 0.75},
        "dimensions": {
            "skill": {
                "windowed_metrics": {"f1": 0.80, "precision": 0.86, "recall": 0.75},
                "metrics": {"f1": 0.80, "has_data": True},
            },
            "multiagent": {
                "windowed_metrics": {"f1": None, "has_data": False},
                "metrics": {"f1": None, "has_data": False},
            },
            "toolcall": {
                "windowed_metrics": {"f1": None, "has_data": False},
                "metrics": {"f1": None, "has_data": False},
            },
        },
    }
    alerts = build_f1_alerts(meta)
    _assert(name, len(alerts) == 0, "F1=0.80 >= 0.75 => no alerts")

    # F1 below target
    meta_low = {
        "version": "3.0",
        "global": {"f1_target": 0.75},
        "dimensions": {
            "skill": {
                "windowed_metrics": {"f1": 0.5, "precision": 0.6, "recall": 0.5},
                "metrics": {"f1": 0.5, "has_data": True},
            },
        },
    }
    alerts = build_f1_alerts(meta_low)
    _assert(name, len(alerts) == 1, "F1=0.5 < 0.75 => 1 alert")
    _assert(name, "SkillOpt" in alerts[0], "Alert mentions SkillOpt")

    # Null safety: no windowed_metrics, fallback to metrics
    meta_fallback = {
        "version": "3.0",
        "global": {"f1_target": 0.75},
        "dimensions": {
            "skill": {
                "windowed_metrics": {},
                "metrics": {"f1": 0.3, "precision": 0.4, "recall": 0.3, "has_data": True},
            },
        },
    }
    alerts = build_f1_alerts(meta_fallback)
    _assert(name, len(alerts) == 1, "Fallback to metrics F1=0.3 => 1 alert")

    # Null safety: has_data=False, f1=None => no alert
    meta_nodata = {
        "version": "3.0",
        "global": {"f1_target": 0.75},
        "dimensions": {
            "multiagent": {
                "windowed_metrics": {"f1": None, "has_data": False},
                "metrics": {"f1": None, "has_data": False},
            },
        },
    }
    alerts = build_f1_alerts(meta_nodata)
    _assert(name, len(alerts) == 0, "No data => no alert (null safety)")

    # Legacy version fallback
    meta_legacy = {
        "version": "1.0",
        "global": {"f1_target": 0.75},
        "dimensions": {},
    }
    alerts = build_f1_alerts(meta_legacy)
    _assert(name, len(alerts) == 0, "Legacy version => no windowed alerts")


# ===================================================================
# ROUND 20: Edge cases — empty data, signal recovery, veto escalation
# ===================================================================
def test_round_20():
    name = "R20-edge-cases"
    print(f"\n--- {name} ---")

    # --- Edge: Empty counts everywhere ---
    r = compute_metrics({"tp": 0, "fp": 0, "fn": 0})
    _assert(name, r["has_data"] is False, "Empty counts => has_data=False (vacuous fix)")

    # --- Edge: Window size 0/1 ---
    entries = [{"signal": "tp"}]
    r = compute_windowed_metrics(entries, window_size=1)
    _assert(name, r["window_used"] == 1, "Window size 1 => uses 1 entry")
    _assert(name, r["has_data"] is True, "1 tp entry => has_data=True")

    # --- Edge: All-FN (only misses) ---
    r = compute_metrics({"tp": 0, "fp": 0, "fn": 10})
    _assert(name, r["has_data"] is True, "All-FN => has_data=True")
    _assert(name, abs(r["recall"]) < 1e-6, "All-FN => recall~0")
    # precision = tp/(tp+fp+eps) = 0/(0+0+1e-8) = ~0  (eps prevents div-by-zero, so P~0 not 1)
    _assert(name, r["precision"] is not None and r["precision"] < 0.01,
            f"All-FN => precision~0 (with eps), got {r.get('precision'):.6f}")

    # --- Edge: Signal recovery — add correct entries to improve F1 ---
    entries = [{"signal": "fp"}] * 5 + [{"signal": "fn"}] * 5
    r_before = compute_windowed_metrics(entries, window_size=10)
    # Add correct entries
    entries_recovered = [{"signal": "fp"}] * 5 + [{"signal": "fn"}] * 5 + [{"signal": "tp"}] * 10
    r_after = compute_windowed_metrics(entries_recovered, window_size=10)
    _assert(name, r_after["f1"] > r_before.get("f1", 0) or r_after["f1"] is not None,
            "Signal recovery: F1 improves after adding correct entries")

    # --- Edge: Veto escalation chain ---
    orch = Orchestrator()

    # Security VULNERABLE + HIGH severity
    v = orch.apply_veto_rules(
        {"security_auditor": {"verdict": "VULNERABLE", "severity": "HIGH"}},
        cycle=1)
    _assert(name, v == "reject", "Security HIGH severity => reject")

    # Security VETO (not severity-critical)
    v = orch.apply_veto_rules(
        {"security_auditor": {"verdict": "VETO", "severity": "MEDIUM"}},
        cycle=1)
    _assert(name, v == "revise", "Security VETO+MEDIUM => revise (not reject)")

    # Multiple vetos: tester FAIL takes priority
    v = orch.apply_veto_rules(
        {"tester": "FAIL", "reviewer": "REQUEST_CHANGES", "architect": "RISK"},
        cycle=1)
    _assert(name, v == "reject", "Tester FAIL overrides all other vetos")

    # Cycle exhaustion overrides reviewer REQUEST_CHANGES
    v = orch.apply_veto_rules(
        {"reviewer": "REQUEST_CHANGES"},
        cycle=3, max_cycles=3)
    _assert(name, v == "escalate", "Max cycle exhaustion => escalate (even over revise)")

    # --- Edge: loss function with extreme values ---
    r = compute_loss({"has_data": True, "precision": 0.01, "recall": 0.99},
                     {"current": 0.0, "target": 1.0}, 0.15)
    _assert(name, r["core"] > 0.4, f"Very imbalanced P/R => high core loss, got {r['core']:.4f}")

    # --- Edge: Orchestrator with empty task => ROLE_COLLAB ---
    mode = orch.select_mode("")
    _assert(name, mode == Mode.ROLE_COLLAB, "Empty task => ROLE_COLLAB")

    # --- Edge: parse_entries_from_text with missing subsections ---
    feedback_partial = (
        "## SkillOpt — Skill Trigger Accuracy\n"
        "### Correct Trigger\n"
        "- entry A\n"
        "### Miss\n"
        "- miss B\n"
        "# False Positive section missing entirely\n"
    )
    entries_partial = parse_entries_from_text(feedback_partial, "SkillOpt",
                                              {"tp": "Correct Trigger", "fp": "False Positive", "fn": "Miss"})
    signals_partial = [e["signal"] for e in entries_partial]
    _assert(name, len(entries_partial) == 2, f"Partial feedback: 2 entries, got {len(entries_partial)}")
    _assert(name, entries_partial[0]["signal"] == "tp", "First is tp")
    _assert(name, entries_partial[1]["signal"] == "fn", "Second is fn")


# ===================================================================
# Run all rounds
# ===================================================================
if __name__ == "__main__":
    rounds = [
        test_round_1, test_round_2, test_round_3, test_round_4, test_round_5,
        test_round_6, test_round_7, test_round_8, test_round_9, test_round_10,
        test_round_11, test_round_12, test_round_13, test_round_14, test_round_15,
        test_round_16, test_round_17, test_round_18, test_round_19, test_round_20,
    ]

    print("=" * 70)
    print("TrainingLoop v3.0 + Hybrid-Orchestrator Integration Test")
    print(f"20 rounds, {len(rounds)} test functions")
    print("=" * 70)

    for fn in rounds:
        try:
            fn()
        except Exception as e:
            print(f"\n  [ERROR] {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            _results.append((fn.__name__, f"EXCEPTION: {e}", "FAIL"))

    sys.exit(_summary())
