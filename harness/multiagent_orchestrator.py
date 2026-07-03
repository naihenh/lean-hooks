"""Multi-Agent Orchestrator V2 — Core orchestration module.

Selects dispatch mode, composes roles, decomposes tasks, and applies veto rules
for multi-agent workflows. Reads role definitions from role-registry.yaml.
"""
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import yaml

# --- UTF-8 for Windows ---
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
ROLE_REGISTRY_PATH = SCRIPT_DIR / "role-registry.yaml"
STATE_PATH = SCRIPT_DIR.parent / "loop-engineering" / "states" / "multiagent-state.json"


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------

class Mode(Enum):
    ROLE_COLLAB = "role_collab"
    TASK_PARALLEL = "task_parallel"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# RoleDef
# ---------------------------------------------------------------------------

class RoleDef:
    """Lightweight wrapper around a role definition from the registry."""

    def __init__(self, data: dict):
        self.name: str = data["name"]
        self.description: str = data.get("description", "")
        self.prompt_template: str = data.get("prompt_template", "")
        self.skill_hint: str = data.get("skill_hint", "")
        self.verdict_format: str = data.get("verdict_format", "")
        self.parallel_with: list[str] = data.get("parallel_with", [])
        self.veto_power: bool = data.get("veto_power", False)
        self.can_escalate: bool = data.get("can_escalate", False)

    def format_prompt(self, mode: Mode, task_description: str,
                      context_block: str = "") -> str:
        """Render the prompt template with concrete values."""
        return self.prompt_template.format(
            role_name=self.name,
            mode=mode.value,
            task_description=task_description,
            context_block=context_block,
        )


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------

def _load_role_registry() -> dict:
    """Load and return the full role-registry.yaml structure."""
    with open(ROLE_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Task category detection
# ---------------------------------------------------------------------------

_TASK_CATEGORIES: dict[str, list[str]] = {
    "security_sensitive": [
        r"\bauth(entication|or)?\b",
        r"\bsecurity\b",
        r"\bpermission",
        r"\bvulnerability",
        r"\bencrypt",
        r"\btoken",
        r"\b credential",
        r"\bCSRF",
        r"\bXSS",
        r"\bSQL\s*inject",
        r"\bRBAC",
        r"\baccess\s*control",
        r"认证",
        r"授权",
        r"安全",
        r"权限",
        r"加密",
        r"令牌",
        r"漏洞",
    ],
    "ui_change": [
        r"\bUI\b",
        r"\bstyling",
        r"\bCSS\b",
        r"\bdesign\b",
        r"\blayout\b",
        r"\bfrontend\b",
        r"\bcomponent\b",
        r"\baccessibility",
        r"\bresponsive\b",
        r"\bpage\b",
        r"\binterface\b",
        r"界面",
        r"样式",
        r"布局",
        r"前端",
        r"组件",
    ],
    "api_change": [
        r"\bAPI\b",
        r"\bendpoint",
        r"\broute",
        r"\bREST",
        r"\bGraphQL",
        r"\bwebhook",
        r"\bschema\b",
        r"\bversioning",
        r"\bbackwards?\s*compat",
        r"\b接口",
        r"端点",
        r"路由",
    ],
    "code_change": [
        r"\brefactor\b",
        r"\bimplement\b",
        r"\bfix\b",
        r"\bbug\b",
        r"\badd\b",
        r"\bupdate\b",
        r"\bchange\b",
        r"\bmodify\b",
        r"\bcreate\b",
        r"\bwrite\b",
        r"\bcode\b",
        r"\bfunction\b",
        r"\bmodule\b",
        r"\bclass\b",
        r"重构",
        r"修复",
        r"实现",
        r"添加",
        r"更新",
        r"修改",
        r"编写",
        r"代码",
    ],
}

# Quality signals that push multi-task toward HYBRID instead of TASK_PARALLEL
_QUALITY_KEYWORDS: list[str] = [
    "review", "carefully", "thorough", "verify", "quality",
    "审查", "仔细", "彻底", "验证", "质量",
]

_HYBRID_BOOST_KEYWORDS: list[str] = [
    "review each", "before merging", "review carefully",
    "cross-review", "quality gate",
    "逐一审查", "合并前", "交叉审查", "质量门",
]


# ---------------------------------------------------------------------------
# Connector / action-verb patterns for task decomposition
# ---------------------------------------------------------------------------

_CONNECTORS: list[str] = [
    r"\bAND\b",
    r"\band\b",
    r"，",
    r",",
    r"并且",
    r"同时",
    r";",
    r"；",
]

_ACTION_VERBS: list[str] = [
    r"\bfix\b", r"\badd\b", r"\bimplement\b", r"\brefactor\b",
    r"\bupdate\b", r"\bcreate\b", r"\bwrite\b", r"\bremove\b",
    r"\btest\b", r"\bbuild\b", r"\bmodify\b", r"\bchange\b",
    r"\boptimize\b", r"\bdeploy\b", r"\breview\b", r"\bdesign\b",
    r"修复", r"添加", r"实现", r"重构", r"更新", r"创建",
    r"编写", r"删除", r"测试", r"构建", r"修改", r"优化",
    r"部署", r"审查", r"设计",
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Core multi-agent orchestrator."""

    def __init__(self, registry: dict | None = None):
        self._registry = registry or _load_role_registry()
        self._roles: dict[str, RoleDef] = {
            name: RoleDef(defn)
            for name, defn in self._registry["roles"].items()
        }
        self._composition_rules = self._registry.get("composition_rules", {})

    # ---- public API -------------------------------------------------------

    def select_mode(self, task_description: str) -> Mode:
        """Choose dispatch mode based on task description."""
        n = self._count_independent_tasks(task_description)
        if n >= 2:
            if self._has_quality_signals(task_description):
                return Mode.HYBRID
            return Mode.TASK_PARALLEL
        return Mode.ROLE_COLLAB

    def compose_roles(self, task_description: str, mode: Mode) -> list[RoleDef]:
        """Pick the right role set for the task category and mode."""
        category = self.detect_category(task_description)
        default_sets = self._composition_rules.get("default_sets", {})
        role_names: list[str] = list(default_sets.get(category, default_sets.get("general", [])))

        # Ensure mode minimums
        minimums = self._composition_rules.get("mode_minimums", {})
        minimum = minimums.get(mode.value, 0)
        if len(role_names) < minimum:
            # Pad with reviewer then architect until we hit the minimum
            for filler in ("reviewer", "architect", "tester"):
                if len(role_names) >= minimum:
                    break
                if filler not in role_names:
                    role_names.append(filler)

        # TASK_PARALLEL and HYBRID always get an integrator
        if mode in (Mode.TASK_PARALLEL, Mode.HYBRID):
            if "integrator" not in role_names:
                role_names.append("integrator")

        # Build RoleDef list (skip unknown names gracefully)
        result: list[RoleDef] = []
        for name in role_names:
            if name in self._roles:
                result.append(self._roles[name])
        return result

    def detect_category(self, text: str) -> str:
        """Return the most specific matching task category."""
        # Check in specificity order: security > ui > api > code
        for cat in ("security_sensitive", "ui_change", "api_change", "code_change"):
            patterns = _TASK_CATEGORIES.get(cat, [])
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    return cat
        return "general"

    def decompose_task(self, task_description: str, mode: Mode) -> list[dict]:
        """Break a task description into subtasks."""
        if mode == Mode.ROLE_COLLAB:
            return [{"id": "1", "description": task_description.strip()}]

        # Split on connectors, filter to action clauses
        parts = self._split_on_connectors(task_description)
        subtasks: list[dict] = []
        for i, part in enumerate(parts, 1):
            part = part.strip()
            if not part:
                continue
            if self._has_action_verb(part):
                subtasks.append({"id": str(i), "description": part})
        if not subtasks:
            # Fallback: single subtask with the whole description
            subtasks.append({"id": "1", "description": task_description.strip()})
        return subtasks

    def apply_veto_rules(self, verdicts: dict[str, dict | str],
                         cycle: int, max_cycles: int = 3) -> str:
        """Apply veto rules and return a decision string.

        Returns one of: 'accept', 'reject', 'revise', 'escalate'.

        Verdict values may be plain strings like "APPROVE" or dicts like
        {"verdict": "APPROVE", "issues": []}. Dicts are normalised to their
        ``verdict`` key; ``severity`` is extracted when present.
        """
        def _extract(field: dict | str) -> str:
            """Return the verdict string from a plain string or a dict."""
            if isinstance(field, dict):
                return field.get("verdict", "")
            return field

        # Tester FAIL → reject
        if "tester" in verdicts:
            tv = _extract(verdicts["tester"]).upper()
            if "FAIL" in tv:
                return "reject"

        # Architect RISK / VETO → escalate
        if "architect" in verdicts:
            av = _extract(verdicts["architect"]).upper()
            if "VETO" in av or "RISK" in av:
                return "escalate"

        # Security auditor VULNERABLE/VETO/CONCERN
        if "security_auditor" in verdicts:
            sa = verdicts["security_auditor"]
            sv = _extract(sa).upper()
            severity = ""
            if isinstance(sa, dict):
                severity = sa.get("severity", "").upper()
            if "VULNERABLE" in sv or "VETO" in sv:
                if "CRITICAL" in severity or "HIGH" in severity:
                    return "reject"
                return "revise"
            if "CONCERN" in sv:
                return "revise"

        # Max cycles reached → escalate (even if revision requested)
        if cycle >= max_cycles:
            return "escalate"

        # Reviewer REQUEST_CHANGES → revise
        if "reviewer" in verdicts:
            rv = _extract(verdicts["reviewer"]).upper()
            if "REQUEST_CHANGES" in rv or "VETO" in rv:
                return "revise"

        # Designer NEEDS_REVISION → revise
        if "designer" in verdicts:
            dv = _extract(verdicts["designer"]).upper()
            if "REVISION" in dv or "NEEDS_REVISION" in dv:
                return "revise"

        # All pass → accept
        return "accept"

    def build_dispatch_plan(self, task_description: str) -> dict:
        """Build a full dispatch plan for the given task."""
        mode = self.select_mode(task_description)
        roles = self.compose_roles(task_description, mode)
        subtasks = self.decompose_task(task_description, mode)
        phases = self._build_phases(mode, subtasks)
        return {
            "session_id": uuid.uuid4().hex[:12],
            "mode": mode.value,
            "task_description": task_description,
            "roles": [r.name for r in roles],
            "subtasks": subtasks,
            "phases": phases,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    # ---- internal helpers --------------------------------------------------

    def _count_independent_tasks(self, text: str) -> int:
        """Split text on connectors and count action-verb clauses."""
        parts = self._split_on_connectors(text)
        count = 0
        for part in parts:
            part = part.strip()
            if part and self._has_action_verb(part):
                count += 1
        return max(count, 1)

    def _split_on_connectors(self, text: str) -> list[str]:
        """Split task text on connector patterns."""
        # Build a single regex from all connectors
        connector_re = "|".join(f"(?:{c})" for c in _CONNECTORS)
        parts = re.split(connector_re, text)
        return [p for p in parts if p.strip()]

    @staticmethod
    def _has_action_verb(text: str) -> bool:
        for pat in _ACTION_VERBS:
            if re.search(pat, text, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _has_quality_signals(text: str) -> bool:
        lower = text.lower()
        for kw in _QUALITY_KEYWORDS:
            if kw in lower:
                return True
        for kw in _HYBRID_BOOST_KEYWORDS:
            if kw.lower() in lower:
                return True
        return False

    def _build_phases(self, mode: Mode, subtasks: list[dict]) -> list[dict]:
        """Build phase structures for the given mode and subtasks."""
        if mode == Mode.ROLE_COLLAB:
            return [
                {
                    "name": "IMPLEMENT",
                    "roles": ["implementer"],
                    "agents": [{"role": "implementer"}],
                    "subtasks": ["1"],
                    "parallel": False,
                },
                {
                    "name": "REVIEW",
                    "roles": ["reviewer", "tester"],
                    "agents": [{"role": "reviewer"}, {"role": "tester"}],
                    "subtasks": ["1"],
                    "parallel": True,
                },
                {
                    "name": "DECISION",
                    "roles": ["architect"],
                    "agents": [{"role": "architect"}],
                    "subtasks": ["1"],
                    "parallel": False,
                },
            ]

        if mode == Mode.TASK_PARALLEL:
            subtask_ids = [s["id"] for s in subtasks]
            agents = [{"role": "implementer", "subtask_id": sid} for sid in subtask_ids]
            return [
                {
                    "name": "PARALLEL_EXECUTE",
                    "roles": ["implementer"],
                    "agents": agents,
                    "subtasks": subtask_ids,
                    "parallel": True,
                },
                {
                    "name": "INTEGRATE",
                    "roles": ["integrator"],
                    "agents": [{"role": "integrator"}],
                    "subtasks": subtask_ids,
                    "parallel": False,
                },
            ]

        # HYBRID — per-subtask implement + review, then integrate + decide
        phases: list[dict] = []
        for s in subtasks:
            phases.append({
                "name": f"IMPLEMENT_{s['id']}",
                "roles": ["implementer"],
                "agents": [{"role": "implementer", "subtask_id": s["id"]}],
                "subtasks": [s["id"]],
                "parallel": False,
            })
        for s in subtasks:
            phases.append({
                "name": f"REVIEW_{s['id']}",
                "roles": ["reviewer", "tester"],
                "agents": [{"role": "reviewer"}, {"role": "tester"}],
                "subtasks": [s["id"]],
                "parallel": True,
            })
        all_ids = [s["id"] for s in subtasks]
        phases.append({
            "name": "INTEGRATE",
            "roles": ["integrator"],
            "agents": [{"role": "integrator"}],
            "subtasks": all_ids,
            "parallel": False,
        })
        phases.append({
            "name": "DECISION",
            "roles": ["architect"],
            "agents": [{"role": "architect"}],
            "subtasks": all_ids,
            "parallel": False,
        })
        return phases

    # ---- state management --------------------------------------------------

    @staticmethod
    def _load_state() -> dict:
        if STATE_PATH.exists():
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save_state(state: dict) -> None:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Simple CLI for manual invocation."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Orchestrator V2")
    sub = parser.add_subparsers(dest="command")

    plan_cmd = sub.add_parser("plan", help="Build a dispatch plan")
    plan_cmd.add_argument("task", help="Task description")

    sub.add_parser("roles", help="List available roles")

    sub.add_parser("status", help="Show orchestrator state")

    veto_cmd = sub.add_parser("veto", help="Apply veto rules")
    veto_cmd.add_argument("--verdicts", required=True,
                          help="JSON dict of role→verdict")
    veto_cmd.add_argument("--cycle", type=int, default=1)
    veto_cmd.add_argument("--max-cycles", type=int, default=3)

    args = parser.parse_args()
    orch = Orchestrator()

    if args.command == "plan":
        plan = orch.build_dispatch_plan(args.task)
        print(json.dumps(plan, indent=2, ensure_ascii=False))

    elif args.command == "roles":
        for name, role in orch._roles.items():
            print(f"  {name}: {role.description}")

    elif args.command == "status":
        state = Orchestrator._load_state()
        if state:
            print(json.dumps(state, indent=2, ensure_ascii=False))
        else:
            print("No active state.")

    elif args.command == "veto":
        verdicts = json.loads(args.verdicts)
        result = orch.apply_veto_rules(verdicts, args.cycle, args.max_cycles)
        print(result)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
