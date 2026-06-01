"""Wrapper: disable MIOpen before importing physformer, then run training."""
import os
os.environ["MIOPEN_DISABLE"] = "1"
os.environ["MIOPEN_DEBUG_DISABLE"] = "1"
os.environ["PYTORCH_MIOPEN_DISABLE"] = "1"

import torch
torch.backends.cudnn.enabled = False  # disables MIOpen on ROCm

from physformer.cli import main

if __name__ == "__main__":
    main()
