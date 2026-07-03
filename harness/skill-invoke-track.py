#!/usr/bin/env python3
"""
Skill Invocation Tracker — records skill tool invocations to training-loop/feedback.md
Called by AI or hooks when a skill invocation is observed.

Usage:
    echo '{"skill":"ppt-master","signal":"correct","prompt":"做PPT"}' | python skill-invoke-track.py
    echo '{"skill":"systematic-debugging","signal":"miss","prompt":"fix this error"}' | python skill-invoke-track.py
    echo '{"skill":"security-review","signal":"fp","prompt":"add auth feature"}' | python skill-invoke-track.py
"""
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Resolve paths
HARNESS_ROOT = Path(os.environ.get("HARNESS_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
loop_dir = HARNESS_ROOT / "config" / "training-loop" if (HARNESS_ROOT / "config" / "training-loop").exists() else HARNESS_ROOT / "training-loop"
feedback_path = loop_dir / "feedback.md"

try:
    raw = sys.stdin.read().strip()
    data = json.loads(raw)
except Exception:
    print("Usage: echo '{\"skill\":\"name\",\"signal\":\"correct|miss|fp\",\"prompt\":\"text\"}' | python skill-invoke-track.py", file=sys.stderr)
    sys.exit(1)

skill = data.get("skill", "").strip()
signal = data.get("signal", "").strip().lower()
prompt = data.get("prompt", "").strip()

if not skill or signal not in ("correct", "miss", "fp"):
    print("Invalid input. Required: skill (str), signal (correct|miss|fp), prompt (str)", file=sys.stderr)
    sys.exit(1)

if not feedback_path.exists():
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    feedback_path.write_text(
        "# Training Loop Feedback\n\n"
        "## SkillOpt — Skill Trigger Accuracy\n### Correct Trigger\n### Miss\n### False Positive\n\n"
        "## MultiAgentOpt — Agent Dispatch Accuracy\n### Correct Trigger\n### Miss\n### False Positive\n\n"
        "## ToolCallOpt — Tool Call Pattern Quality\n### Positive\n### Missed Opportunity\n### Negative\n",
        encoding="utf-8",
    )

text = feedback_path.read_text(encoding="utf-8", errors="replace")

signal_map = {"correct": "Correct Trigger", "miss": "Miss", "fp": "False Positive"}
label = signal_map[signal]
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
entry_line = f'- [skill:{skill}] [prompt:"{prompt}"] {signal} signal ({timestamp})'

# Find the SkillOpt block
block_re = r"^##\s+SkillOpt.*?(?=^##[^#]|\Z)"
block_m = re.search(block_re, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)

if block_m:
    block_text = block_m.group(0)
    subsec_re = rf"^###\s+{re.escape(label)}.*$"
    subsec_m = re.search(subsec_re, block_text, re.MULTILINE)

    if subsec_m:
        # Insert after the ### header line, within the full text
        abs_pos = block_m.start() + subsec_m.end()
        text = text[:abs_pos] + "\n" + entry_line + text[abs_pos:]
    else:
        # Insert new subsection at end of block
        abs_pos = block_m.end()
        new_subsec = f"\n### {label}\n{entry_line}\n"
        text = text[:abs_pos] + new_subsec + text[abs_pos:]
else:
    new_section = f"\n## SkillOpt — Skill Trigger Accuracy\n\n### {label}\n{entry_line}\n"
    text += new_section

feedback_path.write_text(text, encoding="utf-8", errors="replace")
print(f"[SkillInvokeTrack] Recorded {signal} for skill={skill}")
