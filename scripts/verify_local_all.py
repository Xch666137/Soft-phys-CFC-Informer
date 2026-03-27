"""
One-shot local verification for the refactored training pipeline.

This runs the full non-GPU validation suite in the active Python environment:
1. Static checks and config resolution
2. Import validation
3. CPU smoke training
4. CPU resume verification
5. CPU baseline matrix verification
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

VERIFY_SCRIPTS = [
    "scripts/verify_local_static.py",
    "verify_imports.py",
    "scripts/verify_local_smoke.py",
    "scripts/verify_local_resume.py",
    "scripts/verify_local_baseline_matrix.py",
]


def run_cmd(script_path: str) -> None:
    print(f"\n=== Running: {script_path} ===")
    subprocess.run([sys.executable, script_path], cwd=REPO_ROOT, check=True)


def main() -> int:
    for script_path in VERIFY_SCRIPTS:
        run_cmd(script_path)

    print("\n=== Local end-to-end verification passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
