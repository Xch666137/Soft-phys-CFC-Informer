"""
CPU-only baseline matrix smoke verification.

This script reuses the synthetic smoke dataset and runs all baseline configs
through a minimal 1-epoch train+test cycle to catch unsupported model/config
combinations on local Windows environments.
"""

from __future__ import annotations

import subprocess
import sys

from verify_local_smoke import REPO_ROOT, build_smoke_dataset


BASELINE_CONFIGS = {
    "LSTM": "configs/baselines/lstm.yaml",
    "GRU": "configs/baselines/gru.yaml",
    "PINN": "configs/baselines/pinn.yaml",
    "Informer": "configs/baselines/informer.yaml",
    "Autoformer": "configs/baselines/autoformer.yaml",
    "DLinear": "configs/baselines/dlinear.yaml",
    "PatchTST": "configs/baselines/patchtst.yaml",
    "iTransformer": "configs/baselines/itransformer.yaml",
}


def run_cmd(args: list[str]) -> None:
    print(f"\n=== Running: {' '.join(args)} ===")
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=True)


def main() -> int:
    build_smoke_dataset()
    smoke_data_arg = "data.data_path=data/vpp_dataset_smoke.csv"

    for model_name, config_path in BASELINE_CONFIGS.items():
        checkpoint_name = f"smoke_{model_name.lower()}_matrix"
        command = [
            "run.py",
            "--config",
            config_path,
            "--checkpoint_name",
            checkpoint_name,
            "--override",
            smoke_data_arg,
            "--override",
            "hardware.use_gpu=false",
            "--override",
            "hardware.num_workers=0",
            "--override",
            "training.use_amp=false",
            "--override",
            "training.train_epochs=1",
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
            "model.d_ff=64",
            "--override",
            "model.factor=1",
        ]

        if model_name in {"Informer", "Autoformer", "iTransformer"}:
            command.extend(["--override", "model.d_layers=1"])

        run_cmd(command)
        metrics_path = REPO_ROOT / "exp_results" / "Baselines" / "checkpoints" / checkpoint_name / "metrics.npy"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Expected artifact was not created: {metrics_path}")

    print("\n=== Local baseline matrix verification passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
