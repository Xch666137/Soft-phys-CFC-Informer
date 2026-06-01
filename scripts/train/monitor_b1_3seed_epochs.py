#!/usr/bin/env python
"""Monitor three B1 seed logs and print once all seeds finish each epoch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


DEFAULT_RUNS = [
    "physformer_igt_b1_r1_reg_finetune_s2025",
    "physformer_igt_b1_r1_reg_finetune_s2026",
    "physformer_igt_b1_r1_reg_finetune_s2027",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default="/root/autodl-tmp/physformer")
    parser.add_argument("--session-tag", default="")
    parser.add_argument("--run-name", action="append", dest="runs", default=None)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def read_epochs(log_path: Path) -> dict[int, dict[str, str]]:
    epochs: dict[int, dict[str, str]] = {}
    if not log_path.exists():
        return epochs
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split("|")]
        if not parts or not parts[0].startswith("Epoch:"):
            continue
        row = {
            "lr": "---",
            "train": "---",
            "val_loss": "---",
            "val_mse": "---",
            "val_mse_real": "---",
        }
        try:
            epoch = int(parts[0].split(":", 1)[1].strip())
        except (IndexError, ValueError):
            continue
        for part in parts[1:]:
            if ":" not in part:
                continue
            key, value = [item.strip() for item in part.split(":", 1)]
            key_lower = key.lower()
            if key_lower == "lr":
                row["lr"] = value
            elif key_lower == "train":
                row["train"] = value
            elif key_lower == "val loss":
                row["val_loss"] = value
            elif key_lower == "val mse":
                row["val_mse"] = value
            elif key_lower.startswith("val mse("):
                row["val_mse_real"] = value
        epochs[epoch] = row
    return epochs


def is_done(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return True
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return "Early stopping" in text or "Test (" in text


def read_metric(run_dir: Path, key: str) -> str:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return "---"
    try:
        value = json.loads(metrics_path.read_text(encoding="utf-8")).get(key)
    except Exception:
        return "---"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_epoch(epoch: int, runs: list[str], per_run: dict[str, dict[int, dict[str, str]]]) -> None:
    print(f"\n=== common epoch {epoch} complete ===", flush=True)
    print("run\ttrain\tval_mse\tval_mse_mw2\tlr", flush=True)
    for run in runs:
        row = per_run[run][epoch]
        print(
            f"{run}\t{row['train']}\t{row['val_mse']}\t{row['val_mse_real']}\t{row['lr']}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir)
    runs = args.runs or DEFAULT_RUNS
    printed: set[int] = set()

    if args.session_tag:
        print(f"monitor_session={args.session_tag}", flush=True)
    print(f"project_dir={project_dir}", flush=True)
    print("runs=" + ",".join(runs), flush=True)

    while True:
        per_run = {run: read_epochs(project_dir / "runs" / run / "train.log") for run in runs}
        common = set.intersection(*(set(v) for v in per_run.values())) if per_run else set()
        for epoch in sorted(common - printed):
            print_epoch(epoch, runs, per_run)
            printed.add(epoch)

        done = all(is_done(project_dir / "runs" / run) for run in runs)
        if done:
            print("\n=== all runs finished ===", flush=True)
            print("run\tmae\tmse\trmse", flush=True)
            for run in runs:
                run_dir = project_dir / "runs" / run
                print(
                    f"{run}\t{read_metric(run_dir, 'mae')}\t{read_metric(run_dir, 'mse')}\t{read_metric(run_dir, 'rmse')}",
                    flush=True,
                )
            return

        if args.once:
            missing = {run: max(per_run[run], default=0) for run in runs}
            print("\nlatest_epoch_by_run=" + json.dumps(missing, sort_keys=True), flush=True)
            return
        time.sleep(max(args.poll_seconds, 5))


if __name__ == "__main__":
    main()
