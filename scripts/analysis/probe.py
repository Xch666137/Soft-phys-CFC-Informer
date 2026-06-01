import argparse
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+)\s*\|\s*Cost:\s*(?P<cost>[0-9.]+)s\s*\|\s*Steps:\s*(?P<steps>\d+)\s*\|\s*"
    r"Samples/s:\s*(?P<samples_per_second>[0-9.]+)\s*\|\s*Max GPU Mem:\s*(?P<max_gpu_mem_gb>[0-9.]+)\s*GB\s*\|\s*"
    r"LR:\s*(?P<lr>[0-9.eE+-]+)\s*\|\s*Train Loss:\s*(?P<train_loss>[0-9.eE+-]+)\s*\|\s*"
    r"Vali Loss:\s*(?P<vali_loss>[0-9.eE+-]+)\s*\|\s*Vali Net MSE:\s*(?P<vali_net_mse>[0-9.eE+-]+)"
)


def load_epochs(log_path: Path):
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EPOCH_RE.search(line)
        if not match:
            continue
        row = {key: float(value) for key, value in match.groupdict().items() if key not in {"epoch", "steps"}}
        row["epoch"] = int(match.group("epoch"))
        row["steps"] = int(match.group("steps"))
        rows.append(row)
    return rows


def classify_probe(rows, warmup_epochs: int):
    if not rows:
        return {
            "status": "no_data",
            "message": "No epoch lines were found in train.log.",
            "recommendation": "Check whether training started and whether the run name is correct.",
        }

    best_row = min(rows, key=lambda row: row["vali_net_mse"])
    last_row = rows[-1]

    post_warmup = [row for row in rows if row["epoch"] >= max(warmup_epochs, 1)]
    if not post_warmup:
        post_warmup = rows

    best_post = min(post_warmup, key=lambda row: row["vali_net_mse"])
    last_three = post_warmup[-3:] if len(post_warmup) >= 3 else post_warmup
    last_three_mean = sum(row["vali_net_mse"] for row in last_three) / len(last_three)
    min_train_post = min(row["train_loss"] for row in post_warmup)
    max_val_post = max(row["vali_net_mse"] for row in post_warmup)

    ratio_last_to_best = last_three_mean / max(best_post["vali_net_mse"], 1e-8)
    ratio_max_to_best = max_val_post / max(best_post["vali_net_mse"], 1e-8)
    ratio_train_rebound = last_row["train_loss"] / max(min_train_post, 1e-8)

    if ratio_max_to_best >= 1.8 or ratio_train_rebound >= 1.8:
        status = "unstable"
        recommendation = "Peak LR still looks too aggressive. Reduce learning_rate or lengthen warmup before running the full benchmark."
    elif ratio_last_to_best <= 1.2 and last_row["epoch"] >= warmup_epochs:
        status = "reasonable"
        recommendation = "This 10-epoch probe looks stable enough to continue into longer training with the current hyperparameters."
    else:
        status = "caution"
        recommendation = "The probe is usable but not fully settled. Watch epochs 10-20 before committing to full multi-seed runs."

    message = (
        f"Best vali_net_mse={best_row['vali_net_mse']:.6f} at epoch {best_row['epoch']}; "
        f"last vali_net_mse={last_row['vali_net_mse']:.6f}; "
        f"post-warmup max/best={ratio_max_to_best:.2f}; "
        f"last-train/min-train={ratio_train_rebound:.2f}."
    )

    return {
        "status": status,
        "message": message,
        "recommendation": recommendation,
        "best_epoch": int(best_row["epoch"]),
        "best_vali_net_mse": float(best_row["vali_net_mse"]),
        "last_epoch": int(last_row["epoch"]),
        "last_vali_net_mse": float(last_row["vali_net_mse"]),
        "last_train_loss": float(last_row["train_loss"]),
        "warmup_epochs": int(warmup_epochs),
        "post_warmup_max_over_best": float(ratio_max_to_best),
        "post_warmup_last3_over_best": float(ratio_last_to_best),
        "train_rebound_ratio": float(ratio_train_rebound),
    }


def write_markdown(summary, rows, output_path: Path):
    lines = [
        "# PhysFormer 10-Epoch Probe",
        "",
        "- Probe mode: `train_epochs=10` with preserved long-horizon curriculum pacing.",
        "- Interpretation rule: use this probe to judge early optimization stability, not final convergence.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Best epoch: `{summary.get('best_epoch', 'n/a')}`",
        f"- Best validation net MSE: `{summary.get('best_vali_net_mse', 'n/a')}`",
        f"- Last epoch: `{summary.get('last_epoch', 'n/a')}`",
        f"- Last validation net MSE: `{summary.get('last_vali_net_mse', 'n/a')}`",
        "",
        f"Summary: {summary['message']}",
        "",
        f"Recommendation: {summary['recommendation']}",
        "",
        "## Epoch Trace",
        "",
        "| epoch | lr | train_loss | vali_net_mse | samples_per_second | max_gpu_mem_gb |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['epoch']} | {row['lr']:.6g} | {row['train_loss']:.6f} | {row['vali_net_mse']:.6f} | "
            f"{row['samples_per_second']:.2f} | {row['max_gpu_mem_gb']:.2f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze PhysFormer 10-epoch hyperparameter probe logs.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing train.log.")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = run_dir / "train.log"
    if not log_path.exists():
        raise FileNotFoundError(f"train.log not found: {log_path}")

    rows = load_epochs(log_path)
    summary = classify_probe(rows, args.warmup_epochs)

    output_json = Path(args.output_json) if args.output_json else run_dir / "reports" / "hparam_probe_summary.json"
    output_md = Path(args.output_md) if args.output_md else run_dir / "reports" / "hparam_probe_summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, rows, output_md)

    print(json.dumps({"summary_json": str(output_json), "summary_md": str(output_md), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
