"""Tests for the Multi-Agent Orchestrator V2 role registry and orchestrator."""
import os

import pytest

try:
    import yaml
except ImportError:
    pytest.skip("PyYAML not installed", allow_module_level=True)

REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "role-registry.yaml"
)


def _load_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- Original registry tests ----

def test_role_registry_loads():
    """Verify the registry file exists and has the expected top-level structure."""
    assert os.path.isfile(REGISTRY_PATH), f"Registry file not found: {REGISTRY_PATH}"

    data = _load_registry()
    assert "roles" in data, "Registry missing 'roles' key"
    assert "implementer" in data["roles"], "Registry missing 'implementer' role"
    assert "reviewer" in data["roles"], "Registry missing 'reviewer' role"


def test_role_registry_dynamic_roles():
    """Verify every role has the required dynamic-role fields."""
    data = _load_registry()
    roles = data["roles"]

    for role_name, role_def in roles.items():
        assert "name" in role_def, f"Role '{role_name}' missing 'name'"
        assert "prompt_template" in role_def, f"Role '{role_name}' missing 'prompt_template'"
        assert "skill_hint" in role_def, f"Role '{role_name}' missing 'skill_hint'"

        # Verify prompt_template contains required placeholders
        tmpl = role_def["prompt_template"]
        for placeholder in ["{role_name}", "{mode}", "{task_description}", "{context_block}"]:
            assert placeholder in tmpl, (
                f"Role '{role_name}' prompt_template missing {placeholder}"
            )


def test_role_registry_compatibility():
    """Verify at least 2 roles declare parallel_with lists (compatibility graph)."""
    data = _load_registry()
    roles = data["roles"]

    parallel_roles = [
        name for name, defn in roles.items()
        if "parallel_with" in defn
    ]
    assert len(parallel_roles) >= 2, (
        f"Expected >= 2 roles with 'parallel_with', got {len(parallel_roles)}: {parallel_roles}"
    )

    # Integrator must NOT run in parallel with anyone
    integrator = roles.get("integrator", {})
    assert integrator.get("parallel_with") == [], (
        "Integrator must have empty parallel_with list"
    )


# ---- Orchestrator tests ----

import sys
from pathlib import Path

# Ensure the harness directory is on the import path for module resolution
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multiagent_orchestrator import Mode, Orchestrator


@pytest.fixture
def orch():
    return Orchestrator()


def test_mode_selection_single_task(orch):
    """A single focused task should select ROLE_COLLAB."""
    mode = orch.select_mode("Fix the login bug in auth.py")
    assert mode == Mode.ROLE_COLLAB


def test_mode_selection_multi_task(orch):
    """Multiple independent tasks without quality signals → TASK_PARALLEL."""
    mode = orch.select_mode(
        "Fix the login bug AND refactor the auth module AND add tests for the API"
    )
    assert mode == Mode.TASK_PARALLEL


def test_mode_selection_multi_task_with_roles(orch):
    """Multiple tasks with quality review signals → HYBRID."""
    mode = orch.select_mode(
        "Fix the login bug, refactor auth, and add tests "
        "— review each carefully before merging"
    )
    assert mode == Mode.HYBRID


def test_role_composition_code_change(orch):
    """A code_change task in ROLE_COLLAB should include implementer+reviewer+tester."""
    roles = orch.compose_roles("Refactor the database module", Mode.ROLE_COLLAB)
    names = [r.name for r in roles]
    assert "implementer" in names
    assert "reviewer" in names
    assert "tester" in names


def test_role_composition_security_sensitive(orch):
    """A security-sensitive task must include security_auditor."""
    roles = orch.compose_roles(
        "Add authentication to the API endpoint", Mode.ROLE_COLLAB
    )
    names = [r.name for r in roles]
    assert "security_auditor" in names


def test_role_composition_ui_change(orch):
    """A UI change task must include designer."""
    roles = orch.compose_roles(
        "Update the login page UI styling", Mode.ROLE_COLLAB
    )
    names = [r.name for r in roles]
    assert "designer" in names


# ---- Hook integration tests ----

def test_multiagent_dispatch_hook_exists():
    """The multiagent-dispatch.sh hook file must exist."""
    hook_path = Path(__file__).parent / "multiagent-dispatch.sh"
    assert hook_path.is_file(), f"Hook file not found: {hook_path}"


def test_multiagent_dispatch_hook_passes_simple_input():
    """The hook should accept simple input and return valid JSON with continue=True."""
    import subprocess
    import json as _json

    hook_path = Path(__file__).parent / "multiagent-dispatch.sh"
    input_json = '{"prompt":"hello"}'
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        ["bash", str(hook_path)],
        input=input_json,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, f"Hook exited with code {result.returncode}: {result.stderr}"
    output = result.stdout.strip()
    assert output, "Hook produced no output"
    data = _json.loads(output)
    assert data.get("continue") is True, f"Expected continue=True, got: {data}"


# ---- End-to-end integration tests ----


def test_e2e_role_collab_workflow():
    """Full role_collab workflow: plan -> phases -> veto rules."""
    orch = Orchestrator()
    plan = orch.build_dispatch_plan("Fix the authentication bug in login.py")

    assert plan["mode"] == "role_collab"
    assert len(plan["subtasks"]) == 1
    assert len(plan["phases"]) == 3  # IMPLEMENT, REVIEW, DECISION

    impl_phase = plan["phases"][0]
    assert impl_phase["name"] == "IMPLEMENT"
    assert "implementer" in impl_phase["roles"]

    review_phase = plan["phases"][1]
    review_roles = set(review_phase["roles"])
    assert "reviewer" in review_roles
    assert "tester" in review_roles

    verdicts = {
        "reviewer": {"verdict": "APPROVE", "issues": [], "summary": "LGTM"},
        "tester": {"verdict": "PASS", "test_results": "3/3", "failures": []},
        "architect": {"verdict": "SAFE", "concerns": [], "impact_scope": "local"},
    }
    decision = orch.apply_veto_rules(verdicts, cycle=1)
    assert decision == "accept"


def test_e2e_task_parallel_workflow():
    """Full task_parallel workflow: auto-split tasks -> parallel dispatch plan."""
    orch = Orchestrator()
    plan = orch.build_dispatch_plan(
        "Fix the login bug and refactor the auth module and add tests for the API"
    )

    assert plan["mode"] in ("task_parallel", "hybrid")
    assert len(plan["subtasks"]) >= 2
    phase_names = [p["name"] for p in plan["phases"]]
    assert any("EXECUTE" in n or "INTEGRATE" in n for n in phase_names)


def test_e2e_hybrid_workflow():
    """Hybrid workflow: multiple tasks, each with review phase."""
    orch = Orchestrator()
    plan = orch.build_dispatch_plan(
        "Fix the login bug, refactor auth, and add tests — review each carefully"
    )

    assert plan["mode"] == "hybrid"
    assert len(plan["subtasks"]) >= 2
    phase_names = [p["name"] for p in plan["phases"]]
    impl_phases = [n for n in phase_names if n.startswith("IMPLEMENT_")]
    review_phases = [n for n in phase_names if n.startswith("REVIEW_")]
    assert len(impl_phases) >= 2
    assert len(review_phases) >= 2
    assert "INTEGRATE" in phase_names


def test_e2e_veto_reject_on_tester_fail():
    """Tester FAIL -> reject (mandatory fix)."""
    orch = Orchestrator()
    verdicts = {
        "reviewer": {"verdict": "APPROVE"},
        "tester": {"verdict": "FAIL"},
        "architect": {"verdict": "SAFE"},
    }
    assert orch.apply_veto_rules(verdicts, cycle=1) == "reject"


def test_e2e_veto_escalate_on_architect_risk():
    """Architect RISK -> escalate to human."""
    orch = Orchestrator()
    verdicts = {
        "reviewer": {"verdict": "APPROVE"},
        "tester": {"verdict": "PASS"},
        "architect": {"verdict": "RISK"},
    }
    assert orch.apply_veto_rules(verdicts, cycle=1) == "escalate"


def test_e2e_veto_security_critical():
    """Security auditor VULNERABLE with critical severity -> reject."""
    orch = Orchestrator()
    verdicts = {
        "reviewer": {"verdict": "APPROVE"},
        "tester": {"verdict": "PASS"},
        "security_auditor": {"verdict": "VULNERABLE", "severity": "critical"},
    }
    assert orch.apply_veto_rules(verdicts, cycle=1) == "reject"


def test_e2e_veto_cycle_exhaustion():
    """Max cycles reached -> escalate."""
    orch = Orchestrator()
    verdicts = {
        "reviewer": {"verdict": "REQUEST_CHANGES", "issues": ["fix style"]},
        "tester": {"verdict": "PASS"},
    }
    assert orch.apply_veto_rules(verdicts, cycle=3, max_cycles=3) == "escalate"
