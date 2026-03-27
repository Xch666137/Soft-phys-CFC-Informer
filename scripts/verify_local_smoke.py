"""
CPU-only smoke verification for local Windows environments.

This script generates a small synthetic VPP dataset and runs:
1. A minimal PhysFormer train+test cycle
2. A minimal baseline Informer train+test cycle

Run it inside the `Soft-phys-CFC-Informer` conda environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DATA_PATH = REPO_ROOT / "data" / "vpp_dataset_smoke.csv"


def build_smoke_dataset(rows: int = 512) -> None:
    SMOKE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2024)
    index = np.arange(rows)
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="15min")

    day_phase = 2 * np.pi * (index % 96) / 96.0
    week_phase = 2 * np.pi * index / (96.0 * 7.0)

    temp = 18.0 + 7.0 * np.sin(day_phase - 0.4) + 2.0 * np.sin(week_phase)
    irradiance = np.clip(np.sin(day_phase - np.pi / 2), 0.0, None) ** 1.5 * 850.0
    wind_speed = np.clip(
        5.0 + 1.8 * np.sin(day_phase + 0.6) + 0.9 * np.sin(week_phase + 1.1) + rng.normal(0.0, 0.25, rows),
        0.3,
        None,
    )

    pv = np.clip(irradiance * 0.055 * (1.0 - 0.002 * (temp - 25.0)), 0.0, None)
    wind = np.clip(0.10 * np.power(wind_speed, 3) + rng.normal(0.0, 0.6, rows), 0.0, None)
    load = np.clip(
        65.0
        + 10.0 * np.sin(day_phase - 1.2)
        + 5.0 * np.cos(week_phase)
        + 0.12 * (22.0 - temp) ** 2
        + rng.normal(0.0, 0.8, rows),
        5.0,
        None,
    )

    df = pd.DataFrame(
        {
            "date": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "Load": load.round(4),
            "PV": pv.round(4),
            "Wind": wind.round(4),
            "Temp": temp.round(4),
            "Irradiance": irradiance.round(4),
            "WindSpeed": wind_speed.round(4),
        }
    )
    df.to_csv(SMOKE_DATA_PATH, index=False)


def run_cmd(args: list[str]) -> None:
    print(f"\n=== Running: {' '.join(args)} ===")
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=True)


def main() -> int:
    build_smoke_dataset()
    smoke_data_arg = f"data.data_path={SMOKE_DATA_PATH.relative_to(REPO_ROOT).as_posix()}"

    physformer_cmd = [
        "run.py",
        "--config",
        "configs/physformer_default.yaml",
        "--checkpoint_name",
        "smoke_physformer_cpu",
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

    baseline_cmd = [
        "run.py",
        "--config",
        "configs/baselines/informer.yaml",
        "--checkpoint_name",
        "smoke_informer_cpu",
        "--override",
        smoke_data_arg,
        "--override",
        "hardware.use_gpu=false",
        "--override",
        "hardware.num_workers=0",
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
        "model.d_layers=1",
        "--override",
        "model.d_ff=64",
        "--override",
        "model.factor=1",
    ]

    run_cmd(physformer_cmd)
    run_cmd(baseline_cmd)

    physformer_metrics = REPO_ROOT / "exp_results" / "PhysFormer" / "checkpoints" / "smoke_physformer_cpu" / "metrics.npy"
    baseline_metrics = REPO_ROOT / "exp_results" / "Baselines" / "checkpoints" / "smoke_informer_cpu" / "metrics.npy"

    for artifact in [SMOKE_DATA_PATH, physformer_metrics, baseline_metrics]:
        if not artifact.exists():
            raise FileNotFoundError(f"Expected artifact was not created: {artifact}")

    print("\n=== Local CPU smoke verification passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
