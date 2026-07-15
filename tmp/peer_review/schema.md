# Peer Review JSON Schema (v1.0)

两个 reviewer（Claude / Codex）均按此 schema 产出 issue 列表。
聚合脚本按 `(line_start ±2, category)` 判定 consensus。

```json
{
  "reviewer": "claude" | "codex" | "anonymous",
  "file":     "<path relative to repo root>",
  "issues": [
    {
      "location":         "<filename:line>",
      "line_start":       <int>,
      "line_end":         <int, optional, defaults to line_start>,
      "severity":         "high" | "medium" | "low",
      "category":         "grammar" | "style" | "latex_command"
                        | "formula_consistency" | "symbol_undefined" | "format",
      "description":      "<中文简短描述>",
      "original_snippet": "<原文片段，保留 LaTeX 源码>",
      "suggested_fix":    "<建议改法>",
      "rationale":        "<简短理由，中文>"
    }
  ]
}
```

## Category 定义

| Category | 含义 |
|---|---|
| `grammar` | 拼写、时态、冠词、主谓一致、明显语法错误 |
| `style` | 学术英文风格（Chinglish、被动滥用、冗余、口语化） |
| `latex_command` | LaTeX 命令选择（`\text` vs `\operatorname`、浮动参数等） |
| `formula_consistency` | 公式与正文或其他公式之间不自洽 |
| `symbol_undefined` | 符号首次出现未定义、同一实体多种写法 |
| `format` | 标点、大小写、caption 格式、单位表达 |

## Severity 定义

| Level | 判据 |
|---|---|
| `high` | 读者理解会出错（公式矛盾、符号未定义、关键数字错） |
| `medium` | 可读性或规范性明显受损 |
| `low` | 风格或小瑕疵 |

## 聚合规则（aggregate.py）

- **Consensus**：双方 issue 满足 `|Δline_start| ≤ 2` 且 `category` 相同
- **Only-X**：仅 X 一方提出

注：两个 reviewer 的情况下，"Majority" 等价于 "Consensus"。三个以上 reviewer 时才引入 Majority 层级。
