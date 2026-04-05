#!/bin/bash
# Remote server verification - run on GPU server after refactoring
set -euo pipefail

echo "=== 1. Import verification ==="
python verify_imports.py

echo "=== 2. Config loading test (PhysFormer v2) ==="
python run.py train --config configs/physformer_default.yaml --print-config

echo "=== 3. Config loading test (TiDE baseline) ==="
python run.py train --config configs/baselines/tide_net_injection.yaml --print-config

echo "=== 4. PhysFormer single epoch training smoke ==="
python run.py train --config configs/physformer_default.yaml --epochs 1 --batch-size 32 --run-name physformer_verify_smoke

echo "=== 5. TiDE single epoch training smoke ==="
python run.py train --config configs/baselines/tide_net_injection.yaml --epochs 1 --batch-size 64 --run-name tide_verify_smoke

echo "=== 6. PhysFormer test smoke ==="
python run.py test --config configs/physformer_default.yaml --run-name physformer_verify_smoke

echo "=== 7. TimeXer config loading smoke ==="
python run.py train --config configs/baselines/timexer_net_injection.yaml --print-config

echo "=== ALL PASSED ==="
