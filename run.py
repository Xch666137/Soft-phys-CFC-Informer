"""
PhysFormer experiment entrypoint.

Examples:
    python run.py train --config configs/physformer_v5_5.yaml --print-config
    python run.py test --config configs/physformer_v5_5.yaml
    python run.py build-dataset --output-dir data_processed/multi_portfolio
"""

from physformer.cli import main

if __name__ == "__main__":
    main()
