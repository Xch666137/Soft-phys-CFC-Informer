#!/usr/bin/env python3
"""聚合两个 peer reviewer 的 JSON 输出，产出共识分类 Markdown 报告。

Usage:
  aggregate.py <reviewer_a.json> <reviewer_b.json> <output.md> [--tolerance N] [--strict]

匹配规则 (v2)：
  两条 issue 视为同一问题当且仅当：
    1. 行区间重叠（允许 ±tolerance 松弛；单点 issue 退化为 line_start±tol）。
    2. category 兼容：严格相等 或 属于同一等价组（symbol_undefined ↔ formula_consistency）。
  --strict 关闭等价组，恢复 v1 行为（仅允许 category 严格相等）。
"""

import argparse
import json
import sys
from pathlib import Path

# category 等价组：同一问题可能被不同 reviewer 归入不同但语义重叠的 category
CATEGORY_EQUIV_GROUPS = [
    {"symbol_undefined", "formula_consistency"},
]


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _range(issue: dict) -> tuple[int, int]:
    lo = int(issue["line_start"])
    hi = int(issue.get("line_end", lo))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _categories_compatible(ca: str, cb: str, strict: bool) -> bool:
    if ca == cb:
        return True
    if strict:
        return False
    for group in CATEGORY_EQUIV_GROUPS:
        if ca in group and cb in group:
            return True
    return False


def match(a: dict, b: dict, tol: int, strict: bool) -> bool:
    a_lo, a_hi = _range(a)
    b_lo, b_hi = _range(b)
    # 两个区间的 "间距"：若相交则为 0；若分离则为最小端点距离
    gap = max(0, max(a_lo, b_lo) - min(a_hi, b_hi))
    if gap > tol:
        return False
    return _categories_compatible(a["category"], b["category"], strict)


def aggregate(rev_a: dict, rev_b: dict, tol: int, strict: bool):
    a_issues = rev_a.get("issues", [])
    b_issues = rev_b.get("issues", [])
    matched_b: set[int] = set()
    consensus = []
    only_a = []
    for a in a_issues:
        found = None
        for j, b in enumerate(b_issues):
            if j in matched_b:
                continue
            if match(a, b, tol, strict):
                found = (j, b)
                break
        if found is not None:
            matched_b.add(found[0])
            consensus.append({"a": a, "b": found[1]})
        else:
            only_a.append(a)
    only_b = [b for j, b in enumerate(b_issues) if j not in matched_b]
    return consensus, only_a, only_b


def fmt_issue(iss: dict) -> str:
    loc = iss.get("location", f"L{iss['line_start']}")
    sev = iss.get("severity", "?")
    cat = iss.get("category", "?")
    desc = iss.get("description", "")
    fix = iss.get("suggested_fix", "-")
    why = iss.get("rationale", "-")
    snip = iss.get("original_snippet", "")
    parts = [f"- **{loc}** `[{sev}/{cat}]` {desc}"]
    if snip:
        parts.append(f"  - Original: `{snip}`")
    parts.append(f"  - Fix: `{fix}`")
    parts.append(f"  - Why: {why}")
    return "\n".join(parts)


def render(rev_a: dict, rev_b: dict, consensus, only_a, only_b, strict: bool, tol: int) -> str:
    name_a = rev_a.get("reviewer", "A")
    name_b = rev_b.get("reviewer", "B")
    file = rev_a.get("file", "?")
    rule_mode = "strict (v1)" if strict else "loose (v2: category equivalents + range overlap)"
    lines = []
    lines.append(f"# Peer Review Consensus Report")
    lines.append("")
    lines.append(f"- File: `{file}`")
    lines.append(f"- Reviewers: **{name_a}** vs **{name_b}**")
    lines.append(f"- Matching rule: {rule_mode}, tolerance={tol}")
    lines.append("")
    lines.append(f"## Summary")
    lines.append(f"| Bucket | Count |")
    lines.append(f"|---|---:|")
    lines.append(f"| Consensus (both flagged) | {len(consensus)} |")
    lines.append(f"| Only {name_a} | {len(only_a)} |")
    lines.append(f"| Only {name_b} | {len(only_b)} |")
    lines.append(f"| **Total unique** | **{len(consensus) + len(only_a) + len(only_b)}** |")
    lines.append("")
    lines.append(f"`{name_a}` reported {len(rev_a.get('issues', []))} raw issues; "
                 f"`{name_b}` reported {len(rev_b.get('issues', []))} raw issues.")
    lines.append("")

    lines.append(f"## 1. Consensus Issues ({len(consensus)})")
    lines.append("_Both reviewers independently flagged the same location + (compatible) category._")
    lines.append("")
    if consensus:
        for pair in sorted(consensus, key=lambda p: p["a"]["line_start"]):
            a, b = pair["a"], pair["b"]
            cat_hint = a["category"] if a["category"] == b["category"] else f"{a['category']}↔{b['category']}"
            lines.append(f"### L{a['line_start']} — `{cat_hint}` (severity: {a.get('severity','?')}/{b.get('severity','?')})")
            lines.append(f"- **{name_a}**: {a.get('description','')}")
            lines.append(f"  - Fix: `{a.get('suggested_fix','-')}`")
            lines.append(f"- **{name_b}**: {b.get('description','')}")
            lines.append(f"  - Fix: `{b.get('suggested_fix','-')}`")
            lines.append("")
    else:
        lines.append("_(none)_")
        lines.append("")

    lines.append(f"## 2. Only from {name_a} ({len(only_a)})")
    lines.append("")
    if only_a:
        for iss in sorted(only_a, key=lambda x: x["line_start"]):
            lines.append(fmt_issue(iss))
            lines.append("")
    else:
        lines.append("_(none)_")
        lines.append("")

    lines.append(f"## 3. Only from {name_b} ({len(only_b)})")
    lines.append("")
    if only_b:
        for iss in sorted(only_b, key=lambda x: x["line_start"]):
            lines.append(fmt_issue(iss))
            lines.append("")
    else:
        lines.append("_(none)_")
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reviewer_a")
    ap.add_argument("reviewer_b")
    ap.add_argument("output_md")
    ap.add_argument("--tolerance", type=int, default=2)
    ap.add_argument("--strict", action="store_true",
                    help="Disable category equivalence (v1 behaviour).")
    args = ap.parse_args()

    rev_a = load(args.reviewer_a)
    rev_b = load(args.reviewer_b)
    consensus, only_a, only_b = aggregate(rev_a, rev_b, args.tolerance, args.strict)

    md = render(rev_a, rev_b, consensus, only_a, only_b, args.strict, args.tolerance)
    Path(args.output_md).write_text(md, encoding="utf-8")

    name_a = rev_a.get("reviewer", "A")
    name_b = rev_b.get("reviewer", "B")
    print(f"[aggregate] OK -> {args.output_md}")
    print(f"  Consensus: {len(consensus)} | Only-{name_a}: {len(only_a)} | Only-{name_b}: {len(only_b)}")


if __name__ == "__main__":
    main()
