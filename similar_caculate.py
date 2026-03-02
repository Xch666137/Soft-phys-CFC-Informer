import numpy as np

gate_pv = np.load('exp_results/PhysFormer/checkpoints/PhysFormer_ensemble_seed2024/vis_gate_pv.npy', allow_pickle=True)
irr = np.load('exp_results/PhysFormer/checkpoints/PhysFormer_ensemble_seed2024/vis_irr.npy', allow_pickle=True)

gate_pv = np.array(gate_pv)  # [5, 672]
irr = np.array(irr)           # [5, 672]

# 计算每个样本的相关系数
for i in range(len(gate_pv)):
    corr = np.corrcoef(gate_pv[i], irr[i])[0, 1]
    print(f"Sample {i}: Pearson r = {corr:.4f}")

# 全局相关
corr_all = np.corrcoef(gate_pv.flatten(), irr.flatten())[0, 1]
print(f"\nOverall Pearson r(gate_pv, irradiance) = {corr_all:.4f}")