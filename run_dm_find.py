import numpy as np
import os
import glob

base_dir = "exp_results/"

phys_dirs = glob.glob(os.path.join(base_dir, "PhysFormer/checkpoints/*"))
for pd in phys_dirs:
    met_file = os.path.join(pd, "metrics.npy")
    if os.path.exists(met_file):
        metrics = np.load(met_file)
        mse = metrics[0] if metrics.size > 0 else 0
        print(f"{pd}: MSE = {mse:.6f}")
