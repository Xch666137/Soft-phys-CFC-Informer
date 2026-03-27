"""
Local static verification for Windows/CPU environments.

This script intentionally avoids launching training. It only checks:
1. Python syntax for the refactored entrypoints/modules
2. Config resolution through `run.py --print_config`
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PY_COMPILE_TARGETS = [
    "run.py",
    "physformer/cli.py",
    "physformer/data/data_factory.py",
    "physformer/exp/exp_physformer.py",
    "physformer/exp/exp_baseline.py",
    "scripts/run.py",
    "scripts/run_PhysFormer.py",
    "scripts/run_ensemble.py",
    "scripts/run_benchmark.py",
    "scripts/verify_local_all.py",
    "scripts/verify_local_static.py",
    "scripts/verify_local_smoke.py",
    "scripts/verify_local_resume.py",
    "scripts/verify_local_baseline_matrix.py",
    "verify_imports.py",
]

CONFIG_CHECKS = [
    ["run.py", "--config", "configs/physformer_default.yaml", "--print_config"],
    ["run.py", "--config", "configs/physformer_ablation_v1.yaml", "--print_config"],
    ["run.py", "--config", "configs/baselines/informer.yaml", "--print_config"],
    ["run.py", "--config", "configs/baselines/autoformer.yaml", "--print_config"],
    ["run.py", "--config", "configs/baselines/lstm.yaml", "--print_config"],
    ["run.py", "--config", "configs/baselines/gru.yaml", "--print_config"],
    ["run.py", "--config", "configs/baselines/pinn.yaml", "--print_config"],
    ["run.py", "--config", "configs/baselines/dlinear.yaml", "--print_config"],
    ["run.py", "--config", "configs/baselines/patchtst.yaml", "--print_config"],
    ["run.py", "--config", "configs/baselines/itransformer.yaml", "--print_config"],
]


def run_cmd(args: list[str], skippable_modules: tuple[str, ...] = ()) -> bool:
    print(f"\n=== Running: {' '.join(args)} ===")
    result = subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        return True

    if skippable_modules:
        stderr = result.stderr or ""
        missing = [name for name in skippable_modules if f"No module named '{name}'" in stderr]
        if missing:
            print(
                f"--- Skipped: missing optional local dependency {', '.join(missing)}. "
                f"Install it to enable config validation."
            )
            return False

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def main() -> int:
    run_cmd(["-m", "py_compile", *PY_COMPILE_TARGETS])

    for command in CONFIG_CHECKS:
        run_cmd(command, skippable_modules=("yaml",))

    print("\n=== Local static verification passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
