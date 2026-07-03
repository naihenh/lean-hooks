#!/usr/bin/env bash
# Test harness for multiagent-detect.sh
# Usage: bash test-multiagent-detect.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SCRIPT="$HARNESS_DIR/multiagent-detect.sh"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

run_test() {
    local name="$1"
    local prompt="$2"
    local expected="$3"  # "trigger" or "silent"

    result=$(echo "{\"prompt\":\"$prompt\"}" | bash "$SCRIPT" 2>/dev/null || true)

    if [ -z "$result" ]; then
        echo -e "${RED}FAIL${NC} $name — no output"
        FAIL=$((FAIL + 1))
        return
    fi

    has_context=$(echo "$result" | "$PY" -c "import sys,json; d=json.load(sys.stdin); print('trigger' if 'additionalContext' in d.get('hookSpecificOutput',{}) else 'silent')" 2>/dev/null || echo "parse_error")

    if [ "$has_context" = "$expected" ]; then
        echo -e "${GREEN}PASS${NC} $name — $expected"
        PASS=$((PASS + 1))
    elif [ "$has_context" = "parse_error" ]; then
        echo -e "${RED}FAIL${NC} $name — JSON parse error"
        FAIL=$((FAIL + 1))
    else
        echo -e "${YELLOW}MISMATCH${NC} $name — expected $expected, got $has_context"
        echo "  Output snippet: $(echo "$result" | "$PY" -c 'import sys,json; d=json.load(sys.stdin); print(str(d)[:200])')"
        WARN=$((WARN + 1))
    fi
}

echo "=== MultiAgent Detector Test Suite ==="
echo ""

# --- SHOULD TRIGGER ---

echo "--- Should TRIGGER ---"
run_test "Explicit parallel agents" "帮我并行审查这3个文件" "trigger"
run_test "Cleanup + modify same time" "清理playwright，同时修改multiagent逻辑" "trigger"
run_test "Investigate + find + analyze" "你找找前面的记忆，另外找一个叫browseract的skill，分析它的功能，还有我的playwright应该是已经删除了，你怎么调用的？看看怎么回事" "trigger"
run_test "Lower threshold + new patterns + URLs" "降低阈值，同时看看有没有新的识别模式能改善触发，另外看看这个https://example.com，并对比https://github.com/browser-act/，修改完multiagent之后自动测试" "trigger"
run_test "Multiple tasks with commas" "找文件，修bug，写测试，同时部署" "trigger"
run_test "Split into parts" "把这个需求分成三个部分，分别处理" "trigger"
run_test "Compare two things then modify" "对比一下A和B的区别，另外再优化一下C" "trigger"
run_test "Many action verbs" "找找记忆，查查代码，看看日志，分析一下原因，修修bug" "trigger"

# --- SHOULD NOT TRIGGER ---

echo ""
echo "--- Should NOT trigger ---"
run_test "Greeting" "你好" "silent"
run_test "Simple question" "今天天气怎么样" "silent"
run_test "Single task" "帮我写一个hello world" "silent"
run_test "Short thanks" "谢谢" "silent"
run_test "Simple how-to" "怎么用git commit？" "silent"
run_test "Single file request" "帮我看看这个文件" "silent"

echo ""
echo "=== Results ==="
echo -e "${GREEN}PASS: $PASS${NC}"
echo -e "${YELLOW}WARN/MISMATCH: $WARN${NC}"
echo -e "${RED}FAIL: $FAIL${NC}"

if [ $FAIL -gt 0 ] || [ $WARN -gt 0 ]; then
    echo ""
    echo "Some tests failed. Review the detector logic."
    exit 1
else
    echo ""
    echo "All tests passed."
fi
