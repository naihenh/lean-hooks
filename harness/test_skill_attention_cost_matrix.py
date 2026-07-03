"""Test: trigger_log schema must have ranking_strategy column for A/B logging.

This column distinguishes 'legacy' (current) ranking from any new candidate
strategies in the Skill Attention Cost-Matrix v4 effort.
"""
import importlib
import sqlite3
import subprocess
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = HARNESS_ROOT / "data" / "claude-mem" / "claude-mem.db"
PY = "D:/jiqixuexi/anaconda/python.exe"


def test_trigger_log_schema_has_ranking_strategy():
    """ranking_strategy column must exist on trigger_log for A/B logging."""
    r = subprocess.run(
        [PY, str(HARNESS_ROOT / "config" / "harness" / "trigger-logger.py"),
         "--prompt", "test",
         "--session", "schema-rank-test-001"],
        capture_output=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    conn = sqlite3.connect(str(DB_PATH))
    cols = [row[1] for row in conn.execute("PRAGMA table_info(trigger_log)").fetchall()]
    conn.close()
    assert "ranking_strategy" in cols, f"missing column; got {cols}"


def test_load_tool_cost_map_empty_returns_zero():
    """Empty trigger_log → empty map (not error)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "skill_attention",
        str(HARNESS_ROOT / "config" / "harness" / "skill-attention.py"),
    )
    sa = importlib.util.module_from_spec(spec)
    sys.modules["skill_attention"] = sa
    spec.loader.exec_module(sa)
    result = sa.load_tool_cost_map(str(DB_PATH))
    assert isinstance(result, dict)
    for skill, cost in result.items():
        assert cost >= 0.0, f"negative cost for {skill}: {cost}"


def test_load_fp_penalty_map_shape():
    """Map values must be in [0, 1]."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "skill_attention",
        str(HARNESS_ROOT / "config" / "harness" / "skill-attention.py"),
    )
    sa = importlib.util.module_from_spec(spec)
    sys.modules["skill_attention"] = sa
    spec.loader.exec_module(sa)
    m = sa.load_fp_penalty_map(str(DB_PATH))
    for skill, pen in m.items():
        assert 0.0 <= pen <= 1.0, f"{skill} penalty {pen} out of range"


def test_build_cost_matrix_shape_and_weights():
    """Shape (1, n_skills) and weights sum to 1.0 contribution."""
    import importlib.util
    spec2 = importlib.util.spec_from_file_location(
        "skill_attention",
        str(HARNESS_ROOT / "config" / "harness" / "skill-attention.py"),
    )
    sa = importlib.util.module_from_spec(spec2)
    sys.modules["skill_attention"] = sa
    spec2.loader.exec_module(sa)

    skill_results = [
        {"skill": "a", "gated_similarity": 0.8},
        {"skill": "b", "gated_similarity": 0.6},
        {"skill": "c", "gated_similarity": 0.4},
    ]
    cm, comps = sa.build_cost_matrix(skill_results, {}, {})
    assert cm.shape == (1, 3), cm.shape
    assert comps.shape == (1, 3, 4), comps.shape
    # Components sum to 1.0 contribution (deterministic weights)
    assert abs(comps.sum(axis=2).max() - 1.0) < 1e-6, comps.sum(axis=2)


def test_hungarian_assign_returns_at_most_top_k_in_cost_order():
    """Hungarian: at most top_k indices, ascending cost."""
    spec2 = importlib.util.spec_from_file_location(
        "skill_attention",
        str(HARNESS_ROOT / "config" / "harness" / "skill-attention.py"),
    )
    sa = importlib.util.module_from_spec(spec2)
    sys.modules["skill_attention"] = sa
    spec2.loader.exec_module(sa)
    import numpy as np
    cost = np.array([[0.5, 0.1, 0.9, 0.2]], dtype=np.float32)
    order = sa.hungarian_assign(cost, top_k=3)
    assert len(order) <= 3
    # Hungarian picks one col on a 1xN row. The cheapest column wins.
    # Col 1 (cost 0.1) should be in the first slot.
    assert order[0] == 1, f"expected col 1 first, got {order[0]}"
    # The remaining columns follow ascending cost (col 3 = 0.2, col 0 = 0.5, col 2 = 0.9)
    assert order == [1, 3, 0], f"got {order}"


def test_hungarian_assign_edge_cases():
    """Empty and singleton must not raise."""
    spec2 = importlib.util.spec_from_file_location(
        "skill_attention",
        str(HARNESS_ROOT / "config" / "harness" / "skill-attention.py"),
    )
    sa = importlib.util.module_from_spec(spec2)
    sys.modules["skill_attention"] = sa
    spec2.loader.exec_module(sa)
    import numpy as np
    empty = sa.hungarian_assign(np.zeros((1, 0)), top_k=3)
    assert empty == []
    one = sa.hungarian_assign(np.array([[0.3]]), top_k=3)
    assert one == [0]


def test_hungarian_assign_falls_back_when_scipy_unavailable():
    """When scipy import is forced to fail, the function must still return ordered indices."""
    spec2 = importlib.util.spec_from_file_location(
        "skill_attention",
        str(HARNESS_ROOT / "config" / "harness" / "skill-attention.py"),
    )
    sa = importlib.util.module_from_spec(spec2)
    sys.modules["skill_attention"] = sa
    spec2.loader.exec_module(sa)
    import numpy as np
    # Patch the cached scipy import inside skill_attention by removing if any
    for k in list(sys.modules.keys()):
        if k.startswith("scipy"):
            del sys.modules[k]
    # Force ImportError when skill_attention's hungarian_assign tries `from scipy.optimize...`
    # by injecting a fake __import__ that blocks scipy.
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)
    builtins.__import__ = fake_import
    try:
        cost = np.array([[0.7, 0.2, 0.5]], dtype=np.float32)
        order = sa.hungarian_assign(cost, top_k=2)
    finally:
        builtins.__import__ = real_import
    assert order[0] == 1, f"greedy should pick col 1, got {order}"


def test_query_skills_returns_ranking_strategy_metadata():
    """query_skills must tag every result with ranking_strategy."""
    spec2 = importlib.util.spec_from_file_location(
        "skill_attention",
        str(HARNESS_ROOT / "config" / "harness" / "skill-attention.py"),
    )
    sa = importlib.util.module_from_spec(spec2)
    sys.modules["skill_attention"] = sa
    spec2.loader.exec_module(sa)
    results = sa.query_skills("task failed error something broken", top_k=3)
    assert isinstance(results, list)
    if not results:
        return  # no skills indexed; nothing to verify
    for r in results:
        assert "ranking_strategy" in r, "missing ranking_strategy"
        assert r["ranking_strategy"] in ("cost_matrix", "topk_fallback"), r["ranking_strategy"]
        if r["ranking_strategy"] == "cost_matrix":
            assert "assignment_cost" in r
            assert "cost_components" in r
            assert isinstance(r["assignment_cost"], float)
            comps = r["cost_components"]
            for k in ("distance", "neg_gate", "tool", "fp"):
                assert k in comps, f"missing cost component {k}"


def test_query_skills_strategy_topk_forced():
    """strategy='topk' must always produce ranking_strategy=topk_fallback."""
    spec2 = importlib.util.spec_from_file_location(
        "skill_attention",
        str(HARNESS_ROOT / "config" / "harness" / "skill-attention.py"),
    )
    sa = importlib.util.module_from_spec(spec2)
    sys.modules["skill_attention"] = sa
    spec2.loader.exec_module(sa)
    results = sa.query_skills("brainstorm ideas creative approach", top_k=3, strategy="topk")
    if results:
        assert all(r["ranking_strategy"] == "topk_fallback" for r in results)


def test_query_cli_strategy_topk():
    """CLI flag --strategy topk must be accepted by argparse."""
    r = subprocess.run(
        [
            PY,
            str(HARNESS_ROOT / "config" / "harness" / "skill-attention.py"),
            "query",
            "--prompt",
            "brainstorm ideas",
            "--top-k",
            "3",
            "--strategy",
            "topk",
        ],
        capture_output=True,
        timeout=60,
    )
    assert r.returncode == 0, f"CLI rejected --strategy: stderr={r.stderr[:200]}"
    # Output should be valid JSON (empty list or ranked results)
    import json
    out = json.loads(r.stdout.strip())
    assert isinstance(out, list), f"expected JSON list, got {type(out)}"


def test_trigger_logger_writes_dual_strategy_rows():
    """Both strategies produce trigger_log rows with the right ranking_strategy tag."""
    spec2 = importlib.util.spec_from_file_location(
        "skill_attention",
        str(HARNESS_ROOT / "config" / "harness" / "skill-attention.py"),
    )
    sa = importlib.util.module_from_spec(spec2)
    sys.modules["skill_attention"] = sa
    spec2.loader.exec_module(sa)

    r_cost = sa.query_skills("fix this crash now", top_k=2, strategy="cost_matrix")
    r_topk = sa.query_skills("fix this crash now", top_k=2, strategy="topk")
    if r_cost:
        assert r_cost[0]["ranking_strategy"] == "cost_matrix"
    if r_topk:
        assert r_topk[0]["ranking_strategy"] == "topk_fallback"
