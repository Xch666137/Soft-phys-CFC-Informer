import sys
import numpy as np

p = "e:/Py_program/Soft-phys-CFC-Informer/exp_results/PhysFormer/checkpoints/PhysFormer_full_seed2024/"

pv = np.load(p + "vis_gate_pv.npy")
w = np.load(p + "vis_gate_wind.npy")
irr = np.load(p + "vis_irr.npy")
spd = np.load(p + "vis_speed.npy")
hist = np.load(p + "gate_history.npy", allow_pickle=True)

print(f"vis_gate_pv shape: {pv.shape}")
print(f"vis_irr shape: {irr.shape}")
print(f"hist preview: {type(hist.item()) if hist.ndim==0 else hist.shape}")
print(f"gate_history contains: {hist.item().keys() if hist.ndim==0 else 'array'}")
