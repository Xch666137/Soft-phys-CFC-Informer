#!/usr/bin/env python3
"""Check checkpoint key prefixes."""
import torch
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "runs/physformer_igt_b1_pretrain_lam10/pretrained_checkpoint.pth"
state = torch.load(path, map_location='cpu')
keys = list(state.keys())
print(f'Total keys: {len(keys)}')
has_orig = any('_orig_mod.' in k for k in keys)
print(f'Has _orig_mod. prefix: {has_orig}')
if has_orig:
    print('Sample keys with prefix:')
    for k in keys[:5]:
        if '_orig_mod.' in k:
            print(f'  {k}')
else:
    print('Sample keys:')
    for k in keys[:5]:
        print(f'  {k}')