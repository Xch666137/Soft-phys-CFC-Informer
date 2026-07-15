#!/usr/bin/env bash
# Codex peer-review driver
# Usage: codex_review.sh <target_file_relative_to_repo> <output_json_path> [<raw_log_path>]
#
# 约束：
# - 强制 --sandbox read-only（Codex 不能动文件）
# - 若目标文件有未提交改动，先 git stash 再 pop（保证 Codex 看到干净 HEAD）
# - 通过 ===JSON_START===/===JSON_END=== 包围提取 JSON，避免 Codex 的 stdout 噪音

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <target_file> <output_json> [<raw_log>]" >&2
    exit 1
fi

TARGET="$1"
OUTPUT="$2"
RAW_LOG="${3:-${OUTPUT%.json}_raw.log}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROMPT_TEMPLATE="$SCRIPT_DIR/prompt_template.txt"

if [ ! -f "$TARGET" ]; then
    echo "[peer-review] ERROR: target file not found: $TARGET" >&2
    exit 1
fi

if [ ! -f "$PROMPT_TEMPLATE" ]; then
    echo "[peer-review] ERROR: prompt template not found: $PROMPT_TEMPLATE" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "$(dirname "$RAW_LOG")"

# 构造 prompt（替换 {{TARGET}}）
PROMPT=$(sed "s|{{TARGET}}|$TARGET|g" "$PROMPT_TEMPLATE")

# Stash 未提交改动（仅针对目标文件）
STASH_TAG="pre-codex-peer-$(date +%s)"
STASHED=0
if git status --porcelain -- "$TARGET" 2>/dev/null | grep -q .; then
    echo "[peer-review] Stashing uncommitted changes on $TARGET (tag: $STASH_TAG)" >&2
    if git stash push -m "$STASH_TAG" -- "$TARGET"; then
        STASHED=1
    else
        echo "[peer-review] WARN: stash failed, proceeding without checkpoint" >&2
    fi
fi

echo "[peer-review] Calling codex exec (read-only) on $TARGET ..." >&2
CODEX_EXIT=0
codex exec --skip-git-repo-check --sandbox read-only "$PROMPT" > "$RAW_LOG" 2>&1 || CODEX_EXIT=$?

# 恢复 stash
if [ "$STASHED" -eq 1 ]; then
    echo "[peer-review] Restoring stash $STASH_TAG" >&2
    git stash pop || echo "[peer-review] WARN: stash pop had conflicts; inspect 'git stash list'" >&2
fi

if [ "$CODEX_EXIT" -ne 0 ]; then
    echo "[peer-review] WARN: codex exec exited with code $CODEX_EXIT (continuing to parse)" >&2
fi

# 提取 ===JSON_START=== 与 ===JSON_END=== 之间的 JSON
# 在 Windows 下优先使用 python；fallback 到 python3
PY_BIN="$(command -v python || command -v python3)"
if [ -z "$PY_BIN" ]; then
    echo "[peer-review] ERROR: no python interpreter found" >&2
    exit 4
fi

"$PY_BIN" - "$RAW_LOG" "$OUTPUT" <<'PYEOF'
import json, re, sys
raw_path, out_path = sys.argv[1], sys.argv[2]
with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

# Codex exec 会把 prompt 文本也 echo 到 stdout；prompt 里描述了 ===JSON_START===/===END===
# 字面标记，会产生多对假匹配。策略：取最后一个 END 之前最近的 START，这一对才是真实输出。
starts = [m.start() for m in re.finditer(r"===JSON_START===", text)]
ends = [m.end() for m in re.finditer(r"===JSON_END===", text)]
if not starts or not ends:
    print(f"[peer-review] ERROR: markers missing (start:{len(starts)}, end:{len(ends)})", file=sys.stderr)
    sys.exit(2)

last_end = ends[-1]
candidate_starts = [s for s in starts if s < last_end]
if not candidate_starts:
    print("[peer-review] ERROR: no START before last END", file=sys.stderr)
    sys.exit(2)
last_start = candidate_starts[-1]

body = text[last_start + len("===JSON_START==="):last_end - len("===JSON_END===")].strip()
body = re.sub(r"^```(?:json)?\s*", "", body)
body = re.sub(r"\s*```$", "", body)

try:
    data = json.loads(body)
except json.JSONDecodeError as e:
    print(f"[peer-review] ERROR: JSON parse failed: {e}", file=sys.stderr)
    print(f"[peer-review] first 200 chars: {body[:200]!r}", file=sys.stderr)
    sys.exit(3)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

n_issues = len(data.get("issues", []))
print(f"[peer-review] OK: {n_issues} issues -> {out_path}", file=sys.stderr)
PYEOF
