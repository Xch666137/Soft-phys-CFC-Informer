from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SVG_HEADER = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>"""


@dataclass(frozen=True)
class Node:
    id: str
    x: int
    y: int
    w: int
    h: int
    title: str
    body: tuple[str, ...]
    fill: str
    stroke: str = "#1f2937"


@dataclass(frozen=True)
class Edge:
    start: str
    end: str
    label: str | None = None
    start_side: str = "right"
    end_side: str = "left"
    start_offset: int = 0
    end_offset: int = 0


NODES = [
    Node(
        id="raw_household",
        x=40,
        y=110,
        w=280,
        h=130,
        title="Real Household Signals",
        body=("load", "pv", "battery power", "battery state"),
        fill="#DBEAFE",
    ),
    Node(
        id="weather",
        x=40,
        y=295,
        w=280,
        h=100,
        title="Shared Weather",
        body=("temperature", "irradiance", "wind speed"),
        fill="#DCFCE7",
    ),
    Node(
        id="wind_template",
        x=40,
        y=455,
        w=280,
        h=100,
        title="Wind Template",
        body=("Rye template", "synthetic wind generation"),
        fill="#FCE7F3",
    ),
    Node(
        id="align",
        x=390,
        y=205,
        w=300,
        h=120,
        title="15-min Alignment",
        body=("time align signals", "merge weather", "add synthetic wind"),
        fill="#FEF3C7",
    ),
    Node(
        id="household",
        x=770,
        y=205,
        w=300,
        h=120,
        title="Household Time Series",
        body=("unified household timeline", "semi-synthetic internal components"),
        fill="#EDE9FE",
    ),
    Node(
        id="audit_split",
        x=1150,
        y=205,
        w=320,
        h=120,
        title="Eligibility + Split",
        body=("audit household quality", "split households into train / val / test"),
        fill="#FEE2E2",
    ),
    Node(
        id="portfolio",
        x=1550,
        y=195,
        w=340,
        h=145,
        title="Portfolio-level Series",
        body=(
            "aggregate households into portfolios",
            "target: p_vpp",
            "components: load / pv / wind / battery / soc",
        ),
        fill="#DBEAFE",
    ),
    Node(
        id="window",
        x=1970,
        y=205,
        w=300,
        h=125,
        title="Window Builder",
        body=(
            "history seq_len",
            "future pred_len",
            "fit scaler on train split only",
        ),
        fill="#DCFCE7",
    ),
    Node(
        id="model_input",
        x=2350,
        y=180,
        w=330,
        h=170,
        title="Model Input",
        body=(
            "history p_vpp",
            "history weather",
            "history battery state",
            "future weather",
        ),
        fill="#FEF3C7",
    ),
    Node(
        id="stage_a",
        x=2760,
        y=95,
        w=280,
        h=120,
        title="Stage A",
        body=("net-first training", "predict future p_vpp"),
        fill="#DBEAFE",
    ),
    Node(
        id="stage_b",
        x=2760,
        y=295,
        w=280,
        h=120,
        title="Stage B",
        body=("operational fit", "predict component/state interface"),
        fill="#FCE7F3",
    ),
    Node(
        id="eval",
        x=3120,
        y=95,
        w=300,
        h=120,
        title="Benchmark Metrics",
        body=("MSE", "MAE", "RMSE", "ramp violation"),
        fill="#DCFCE7",
    ),
    Node(
        id="export",
        x=3120,
        y=295,
        w=300,
        h=120,
        title="Export / Validation",
        body=("forecast export", "optional power-flow validation"),
        fill="#EDE9FE",
    ),
]


EDGES = [
    Edge("raw_household", "align"),
    Edge("weather", "align", start_offset=-8, end_offset=0),
    Edge("wind_template", "align", start_offset=8, end_offset=0),
    Edge("align", "household"),
    Edge("household", "audit_split"),
    Edge("audit_split", "portfolio"),
    Edge("portfolio", "window"),
    Edge("window", "model_input"),
    Edge("model_input", "stage_a", "main target", start_offset=-42, end_side="top", end_offset=-70),
    Edge("model_input", "stage_b", "optional interface", start_offset=42, end_side="bottom", end_offset=-70),
    Edge("stage_a", "eval"),
    Edge("stage_a", "export"),
    Edge("stage_b", "export"),
]


def node_lookup(nodes: Iterable[Node]) -> dict[str, Node]:
    return {node.id: node for node in nodes}


def svg_text(x: int, y: int, text: str, font_size: int = 18, weight: str = "normal", anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="{font_size}" font-weight="{weight}" fill="#111827" text-anchor="{anchor}">{text}</text>'
    )


def svg_box(node: Node) -> str:
    padding_left = 18
    parts = [
        f'<rect x="{node.x}" y="{node.y}" width="{node.w}" height="{node.h}" rx="16" ry="16" '
        f'fill="{node.fill}" stroke="{node.stroke}" stroke-width="2"/>',
        svg_text(node.x + node.w // 2, node.y + 30, node.title, font_size=20, weight="600"),
    ]
    line_y = node.y + 64
    for line in node.body:
        parts.append(svg_text(node.x + padding_left, line_y, line, font_size=15, anchor="start"))
        line_y += 24
    return "\n".join(parts)


def edge_anchor(node: Node, side: str, offset: int = 0) -> tuple[int, int]:
    if side == "right":
        return node.x + node.w, node.y + node.h // 2 + offset
    if side == "left":
        return node.x, node.y + node.h // 2 + offset
    if side == "top":
        return node.x + node.w // 2 + offset, node.y
    if side == "bottom":
        return node.x + node.w // 2 + offset, node.y + node.h
    raise ValueError(f"Unsupported side: {side}")


def svg_edge(edge: Edge, lookup: dict[str, Node]) -> str:
    start = lookup[edge.start]
    end = lookup[edge.end]
    x1, y1 = edge_anchor(start, edge.start_side, edge.start_offset)
    x2, y2 = edge_anchor(end, edge.end_side, edge.end_offset)
    mid_x = x1 + max(28, (x2 - x1) // 2)
    points = [(x1, y1), (mid_x, y1), (mid_x, y2), (x2, y2)]
    point_str = " ".join(f"{x},{y}" for x, y in points)
    label_svg = ""
    if edge.label:
        label_x = mid_x
        label_y = min(y1, y2) - 10 if y1 != y2 else y1 - 10
        label_svg = svg_text(label_x, label_y, edge.label, font_size=14)
    return (
        f'<polyline points="{point_str}" fill="none" stroke="#374151" stroke-width="3" '
        f'stroke-linejoin="round" stroke-linecap="round" marker-end="url(#arrow)"/>\n{label_svg}'
    )


def build_svg(nodes: list[Node], edges: list[Edge]) -> str:
    width = 3500
    height = 700
    lookup = node_lookup(nodes)
    boxes = "\n".join(svg_box(node) for node in nodes)
    arrows = "\n".join(svg_edge(edge, lookup) for edge in edges)
    title = svg_text(1520, 50, "Current Training Data Flow", font_size=30, weight="700")
    subtitle = svg_text(
        1520,
        82,
        "Current pipeline uses the semi-synthetic multi-portfolio benchmark, not the newly downloaded external datasets",
        font_size=16,
    )
    return "\n".join(
        [
            SVG_HEADER,
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<defs>",
            '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
            '<path d="M0,0 L0,6 L9,3 z" fill="#374151"/>',
            "</marker>",
            "</defs>",
            '<rect width="100%" height="100%" fill="#FFFFFF"/>',
            title,
            subtitle,
            arrows,
            boxes,
            "</svg>",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the current training data-flow diagram as SVG.")
    parser.add_argument(
        "--output",
        default="analysis/training_pipeline_dataflow.svg",
        help="Output SVG path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_svg(NODES, EDGES), encoding="utf-8")
    print(f"Saved diagram to: {output_path}")


if __name__ == "__main__":
    main()
