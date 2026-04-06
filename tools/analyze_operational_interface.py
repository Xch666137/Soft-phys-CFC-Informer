import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def classify(summary, metrics):
    consistency = float(summary.get("component_net_consistency_residual", metrics.get("component_net_consistency_residual", 0.0)))
    battery_soc_mae = float(summary.get("battery_soc_mae", metrics.get("battery_soc_mae", 0.0)))
    battery_power_mae = float(summary.get("battery_power_mae", metrics.get("battery_power_mae", 0.0)))
    confidence_mean = summary.get("confidence_mean", {})
    attribution_mean = summary.get("attribution_mean", {})

    avg_conf = 0.0
    if confidence_mean:
        avg_conf = sum(float(v) for v in confidence_mean.values()) / len(confidence_mean)

    if consistency <= 0.05 and battery_soc_mae <= 0.10:
        status = "ready"
        recommendation = "Operational interface is coherent enough for diagnostic export and qualitative attribution review."
    elif consistency <= 0.15 and battery_soc_mae <= 0.20:
        status = "usable_with_caution"
        recommendation = "Operational interface is usable, but component consistency still needs review before treating it as an operationally trusted state interface."
    else:
        status = "needs_work"
        recommendation = "Operational interface is not yet self-consistent enough. Revisit Stage B supervision weights or initialization before downstream use."

    return {
        "status": status,
        "recommendation": recommendation,
        "component_net_consistency_residual": consistency,
        "battery_power_mae": battery_power_mae,
        "battery_soc_mae": battery_soc_mae,
        "average_confidence": avg_conf,
        "confidence_mean": confidence_mean,
        "attribution_mean": attribution_mean,
    }


def write_markdown(result, summary, output_path: Path):
    lines = [
        "# Operational Interface Summary",
        "",
        f"- Status: `{result['status']}`",
        f"- Recommendation: {result['recommendation']}",
        "",
        "## Core Diagnostics",
        "",
        f"- Component-to-net consistency residual: `{result['component_net_consistency_residual']:.6f}`",
        f"- Battery power MAE: `{result['battery_power_mae']:.6f}`",
        f"- Battery SOC MAE: `{result['battery_soc_mae']:.6f}`",
        f"- Average confidence: `{result['average_confidence']:.6f}`",
        "",
        "## Component Confidence Mean",
        "",
    ]
    for name, value in result["confidence_mean"].items():
        lines.append(f"- `{name}`: `{float(value):.6f}`")

    lines += [
        "",
        "## Attribution Mean",
        "",
    ]
    for name, value in result["attribution_mean"].items():
        lines.append(f"- `{name}`: `{float(value):.6f}`")

    lines += [
        "",
        "## Raw Diagnostic Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize PhysFormer operational-fit diagnostics.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing extras/diagnostic_summary.json.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary_path = run_dir / "extras" / "diagnostic_summary.json"
    metrics_path = run_dir / "metrics.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"diagnostic_summary.json not found: {summary_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")

    summary = load_json(summary_path)
    metrics = load_json(metrics_path)
    result = classify(summary, metrics)

    output_json = Path(args.output_json) if args.output_json else run_dir / "reports" / "operational_interface_summary.json"
    output_md = Path(args.output_md) if args.output_md else run_dir / "reports" / "operational_interface_summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(result, summary, output_md)
    print(json.dumps({"summary_json": str(output_json), "summary_md": str(output_md), **result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
