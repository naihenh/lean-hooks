#!/usr/bin/env bash
# Multi-Agent Orchestrator dispatch — UserPromptSubmit hook
# Calls multiagent_orchestrator.py plan to get dispatch mode,
# then injects structured context for the AI to act on.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# Read JSON input from stdin
INPUT=$(cat)

# Extract prompt text via Python
PROMPT=$(echo "$INPUT" | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d.get('prompt',''))" 2>/dev/null || echo "")

# Skip empty/short prompts (< 5 chars)
if [ ${#PROMPT} -lt 5 ]; then
    echo '{"continue":true,"suppressOutput":true}'
    exit 0
fi

# Check DISABLED_HOOKS
if [ -n "${DISABLED_HOOKS:-}" ]; then
    IFS=',' read -ra _disabled <<< "$DISABLED_HOOKS"
    for _hook in "${_disabled[@]}"; do
        if [ "$_hook" = "multiagent-dispatch" ]; then
            echo '{"continue":true,"suppressOutput":true}'
            exit 0
        fi
    done
fi

# Call orchestrator to get dispatch plan
ORCHESTRATOR_SCRIPT="$SCRIPT_DIR/multiagent_orchestrator.py"
PLAN_JSON=$("$PY" "$ORCHESTRATOR_SCRIPT" plan "$PROMPT" 2>/dev/null || echo '{}')

# Extract mode from plan JSON
MODE=$(echo "$PLAN_JSON" | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d.get('mode','role_collab'))" 2>/dev/null || echo "role_collab")

# Build mode-specific injection header
case "$MODE" in
    role_collab)
        HEADER="[MultiAgent Orchestrator] 检测到单任务需要多角色协作。  模式: 多角色协作 (role_collab)"
        ;;
    task_parallel)
        HEADER="[MultiAgent Orchestrator] 检测到多个独立任务可并行执行。  模式: 多任务并行 (task_parallel)"
        ;;
    hybrid)
        HEADER="[MultiAgent Orchestrator] 检测到多任务+多角色混合模式。  模式: 混合 (hybrid)"
        ;;
    *)
        HEADER="[MultiAgent Orchestrator] 模式: $MODE"
        ;;
esac

# Extract subtask and phase summaries via Python
SUBTASK_SUMMARY=$(echo "$PLAN_JSON" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
subtasks = d.get('subtasks', [])
lines = []
for s in subtasks:
    lines.append(f\"  [{s.get('id','?')}] {s.get('description','')}\")
print(chr(10).join(lines) if lines else '  (single task)')
" 2>/dev/null || echo "  (parse error)")

PHASE_SUMMARY=$(echo "$PLAN_JSON" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
phases = d.get('phases', [])
lines = []
for p in phases:
    roles = ','.join(p.get('roles', []))
    sids = ','.join(p.get('subtasks', []))
    lines.append(f\"  {p.get('name','?')}: roles=[{roles}] subtasks=[{sids}]\")
print(chr(10).join(lines) if lines else '  (no phases)')
" 2>/dev/null || echo "  (parse error)")

# Extract roles list
ROLES_SUMMARY=$(echo "$PLAN_JSON" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
roles = d.get('roles', [])
print(', '.join(roles) if roles else '(none)')
" 2>/dev/null || echo "(none)")

# Assemble full injection message
ADDITIONAL_CONTEXT="${HEADER}
角色: ${ROLES_SUMMARY}

子任务:
${SUBTASK_SUMMARY}

阶段:
${PHASE_SUMMARY}

执行方式: AI 读取上述计划，使用 Agent 工具按阶段 dispatch 对应角色的 subagent。  若判断有误，说 'multiagent mode wrong' 帮助改进。"

# Output hook JSON
"$PY" -c "
import json, sys
ctx = sys.argv[1]
out = {
    'continue': True,
    'suppressOutput': True,
    'hookSpecificOutput': {
        'hookEventName': 'UserPromptSubmit',
        'additionalContext': ctx
    }
}
print(json.dumps(out, ensure_ascii=False))
" "$ADDITIONAL_CONTEXT"
