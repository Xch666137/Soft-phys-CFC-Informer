import torch
print(f"PyTorch: {torch.__version__}")
print(f"ROCm: {torch.version.hip}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
print(f"GPU compute: {torch.cuda.get_device_capability(0)}")
import os
for d in ['data', 'data_processed', 'data_raw']:
    if os.path.exists(d):
        files = os.listdir(d)[:5]
        print(f"{d}/: {files}")
    else:
        print(f"{d}/: MISSING")
