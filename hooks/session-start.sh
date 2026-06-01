#!/usr/bin/env bash
# SessionStart hook — injects ARA context into every new Claude Code session.
#
# Reads the ARA artifact and produces a compact context injection containing:
#   1. Current project phase and status
#   2. Known dead ends (to prevent repetition)
#   3. Open research threads
#
# Output: JSON with hookSpecificOutput.additionalContext for Claude Code.
set -euo pipefail

ARA_DIR="./ara"
PAPER="$ARA_DIR/PAPER.md"
TREE="$ARA_DIR/trace/exploration_tree.yaml"
INDEX="$ARA_DIR/trace/sessions/session_index.yaml"

# ── Extract dead ends from exploration tree ──────────────────────
extract_dead_ends() {
    if [ ! -f "$TREE" ]; then
        echo "  (ARA exploration tree not found)"
        return
    fi
    # Extract lines containing "type: dead_end" and the following "title:" line
    awk '
        /type: dead_end/ { found=1; next }
        found && /title:/ {
            gsub(/^[ \t]+/, "");
            gsub(/^title: /, "");
            gsub(/"/, "");
            print "  - " $0
            found=0
        }
    ' "$TREE"
}

# ── Extract open threads from session index ──────────────────────
extract_open_threads() {
    if [ ! -f "$INDEX" ]; then
        echo "  (no session index)"
        return
    fi
    # Get open_threads count from the latest session summary
    local count
    count=$(grep "open_threads:" "$INDEX" | tail -1 | grep -o '[0-9]\+' || echo "0")
    if [ "$count" = "0" ]; then
        echo "  (no open threads)"
    else
        echo "  ${count} open thread(s) — check ara/trace/sessions/ for details"
    fi
}

# ── Extract phase from PAPER.md ──────────────────────────────────
extract_phase() {
    if [ ! -f "$PAPER" ]; then
        echo "  (ARA not initialized — run /compiler to create)"
        return
    fi
    local phase
    phase=$(grep "phase:" "$PAPER" | head -1 | sed 's/.*phase:[ "]*//;s/".*//' || echo "unknown")
    local status
    status=$(grep "status:" "$PAPER" | head -1 | sed 's/.*status:[ "]*//;s/".*//' || echo "unknown")
    echo "  Phase: ${phase}, Status: ${status}"
}

# ── Build context injection ──────────────────────────────────────
build_context() {
    local phase dead_ends threads
    phase=$(extract_phase)
    dead_ends=$(extract_dead_ends)
    threads=$(extract_open_threads)

    cat << 'CTX'
<ARA-SESSION-BOOTSTRAP>
## Project State (from ara/PAPER.md)
CTX
    echo "$phase"
    cat << 'CTX'

## Known Dead Ends (DO NOT REPEAT)
CTX
    echo "$dead_ends"
    cat << 'CTX'

## Open Threads
CTX
    echo "$threads"
    cat << 'CTX'

## Quick Instructions
1. Before proposing ANY change: check if it conflicts with a dead end above
2. If user proposes something matching a dead end → cite the node ID and lesson
3. For new experiments: define success criteria BEFORE writing code (CLAUDE.md §8)
4. After significant findings: run /research-manager
</ARA-SESSION-BOOTSTRAP>
CTX
}

# ── Escape for JSON ──────────────────────────────────────────────
escape_for_json() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}

# ── Main ─────────────────────────────────────────────────────────
if [ ! -d "$ARA_DIR" ]; then
    # ARA not initialized — inject a minimal bootstrap message
    printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"<ARA-SESSION-BOOTSTRAP>\\nARA not yet initialized for this project.\\nRun /compiler or create ara/ directory to enable research context injection.\\n</ARA-SESSION-BOOTSTRAP>\\n"}}\n'
    exit 0
fi

CONTEXT=$(build_context)
CONTEXT_ESCAPED=$(escape_for_json "$CONTEXT")

printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"%s"}}\n' "$CONTEXT_ESCAPED"
exit 0
