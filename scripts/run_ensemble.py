import argparse
import subprocess
import sys
from pathlib import Path


def build_command(root_dir, config_path, checkpoint_name, seed, test_only):
    command = [
        sys.executable,
        str(root_dir / "run.py"),
        "--config",
        config_path,
        "--checkpoint_name",
        checkpoint_name,
        "--override",
        f"runtime.seed={seed}",
    ]
    if test_only:
        command.append("--test_only")
    return command


def main():
    parser = argparse.ArgumentParser(description='PhysFormer Ensemble Runner')
    parser.add_argument('--config', type=str, default='configs/physformer_default.yaml')
    parser.add_argument('--checkpoint_prefix', type=str, default='PhysFormer_ensemble')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026])
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]

    for seed in args.seeds:
        checkpoint_name = f"{args.checkpoint_prefix}_seed{seed}"
        command = build_command(root_dir, args.config, checkpoint_name, seed, args.test_only)
        print(f"Running ensemble member seed={seed}: {' '.join(command)}")
        subprocess.run(command, check=True, cwd=root_dir)


if __name__ == "__main__":
    main()
