"""
CPU-only resume verification for local Windows environments.

This script reuses the synthetic smoke dataset and verifies that:
1. A fresh 1-epoch run creates checkpoint and training_state artifacts
2. A second run with --resume continues from the saved state

Run it inside the `Soft-phys-CFC-Informer` conda environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from verify_local_smoke import REPO_ROOT, build_smoke_dataset


def run_cmd(args: list[str], expect_resume: bool = False) -> str:
    print(f"\n=== Running: {' '.join(args)} ===")
    result = subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

    combined = (result.stdout or "") + (result.stderr or "")
    if expect_resume and "Successfully resumed" not in combined:
        raise AssertionError("Resume run did not report successful state restoration.")

    return combined


def verify_training_state(checkpoint_dir: Path, min_epoch: int) -> None:
    checkpoint_path = checkpoint_dir / "checkpoint.pth"
    state_path = checkpoint_dir / "training_state.pth"
    metrics_path = checkpoint_dir / "metrics.npy"

    for artifact in [checkpoint_path, state_path, metrics_path]:
        if not artifact.exists():
            raise FileNotFoundError(f"Expected artifact was not created: {artifact}")

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    epoch = state.get("epoch", -1)
    global_step = state.get("global_step", -1)
    if epoch < min_epoch:
        raise AssertionError(f"Expected resumed epoch >= {min_epoch}, got {epoch}")
    if global_step <= 0:
        raise AssertionError(f"Expected positive global_step in {state_path}, got {global_step}")


def main() -> int:
    build_smoke_dataset()
    smoke_data_arg = "data.data_path=data/vpp_dataset_smoke.csv"

    physformer_checkpoint = "resume_physformer_cpu"
    baseline_checkpoint = "resume_informer_cpu"

    physformer_base = [
        "run.py",
        "--config",
        "configs/physformer_default.yaml",
        "--checkpoint_name",
        physformer_checkpoint,
        "--override",
        smoke_data_arg,
        "--override",
        "hardware.use_gpu=false",
        "--override",
        "hardware.num_workers=0",
        "--override",
        "training.use_amp=false",
        "--override",
        "training.batch_size=8",
        "--override",
        "data.val_batch_size=8",
        "--override",
        "data.test_batch_size=8",
        "--override",
        "data.seq_len=48",
        "--override",
        "data.pred_len=12",
        "--override",
        "model.d_model=32",
        "--override",
        "model.n_heads=4",
        "--override",
        "model.e_layers=1",
        "--override",
        "model.d_ff=64",
        "--override",
        "model.factor=1",
        "--override",
        "training.plot_val_every=1000",
    ]

    baseline_base = [
        "run.py",
        "--config",
        "configs/baselines/informer.yaml",
        "--checkpoint_name",
        baseline_checkpoint,
        "--override",
        smoke_data_arg,
        "--override",
        "hardware.use_gpu=false",
        "--override",
        "hardware.num_workers=0",
        "--override",
        "training.use_amp=false",
        "--override",
        "training.batch_size=8",
        "--override",
        "data.val_batch_size=8",
        "--override",
        "data.test_batch_size=8",
        "--override",
        "data.seq_len=48",
        "--override",
        "data.label_len=24",
        "--override",
        "data.pred_len=12",
        "--override",
        "model.d_model=32",
        "--override",
        "model.n_heads=4",
        "--override",
        "model.e_layers=1",
        "--override",
        "model.d_layers=1",
        "--override",
        "model.d_ff=64",
        "--override",
        "model.factor=1",
    ]

    run_cmd([*physformer_base, "--override", "training.train_epochs=1"])
    verify_training_state(
        REPO_ROOT / "exp_results" / "PhysFormer" / "checkpoints" / physformer_checkpoint,
        min_epoch=0,
    )
    run_cmd([*physformer_base, "--resume", "--override", "training.train_epochs=2"], expect_resume=True)
    verify_training_state(
        REPO_ROOT / "exp_results" / "PhysFormer" / "checkpoints" / physformer_checkpoint,
        min_epoch=1,
    )

    run_cmd([*baseline_base, "--override", "training.train_epochs=1"])
    verify_training_state(
        REPO_ROOT / "exp_results" / "Baselines" / "checkpoints" / baseline_checkpoint,
        min_epoch=0,
    )
    run_cmd([*baseline_base, "--resume", "--override", "training.train_epochs=2"], expect_resume=True)
    verify_training_state(
        REPO_ROOT / "exp_results" / "Baselines" / "checkpoints" / baseline_checkpoint,
        min_epoch=1,
    )

    print("\n=== Local CPU resume verification passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
