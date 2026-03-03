"""
极端天气/高波动场景评估脚本 (Extreme Weather / OOD Evaluator)
用于提取测试集中波动最剧烈的前 10% 样本，验证物理模型的分布外泛化能力。
"""

import os
import numpy as np
import pandas as pd

# ==========================================
# 1. 严格使用生成论文结果的绝对/相对路径映射
# ==========================================
base_dir = './exp_results'

model_paths = {
    'LSTM':       f'{base_dir}/LSTM_vpp_dataset_3years_sl672_pl96_vpp',
    'GRU':        f'{base_dir}/GRU_vpp_dataset_3years_sl672_pl96_vpp',
    'PINN':       f'{base_dir}/PINN_vpp_dataset_3years_sl672_pl96_vpp',
    'Informer':   f'{base_dir}/Informer_vpp_dataset_3years_sl672_pl96_vpp',
    'Autoformer': f'{base_dir}/Autoformer_vpp_dataset_3years_sl672_pl96_vpp',
    'DLinear':    f'{base_dir}/DLinear_vpp_dataset_3years_sl672_pl96_vpp',
    'PatchTST':   f'{base_dir}/PatchTST_vpp_dataset_3years_sl672_pl96_vpp',
    'PhysFormer': f'{base_dir}/PhysFormer/checkpoints/PhysFormer_full_seed2024',
}

# ==========================================
# 2. 筛选极端天气样本 (Top 10% 波动率)
# ==========================================
print("正在定位极端天气样本 (Top 10% 波动率)...")

# 以 PhysFormer 的 true.npy 为基准（所有模型的 true 都是严格对齐的）
ref_true_path = os.path.join(model_paths['PhysFormer'], 'true.npy')
if not os.path.exists(ref_true_path):
    raise FileNotFoundError("找不到参考真实数据！请检查 PhysFormer 路径。")

ref_true = np.load(ref_true_path)  # [N, Pred_Len, Channels]

# 计算新能源 (PV+Wind) 的总波动率 (Total Variation)
# 波动率 = 每一步之间变化的绝对值之和
pv_true = ref_true[:, :, 1]
wind_true = ref_true[:, :, 2]

pv_volatility = np.sum(np.abs(np.diff(pv_true, axis=1)), axis=1)
wind_volatility = np.sum(np.abs(np.diff(wind_true, axis=1)), axis=1)
total_volatility = pv_volatility + wind_volatility

# 提取 Top 10% 剧烈波动的样本索引
top_k = int(len(total_volatility) * 0.10)
extreme_indices = np.argsort(total_volatility)[-top_k:]

print(f"-> 测试集总样本数: {len(total_volatility)}")
print(f"-> 已提取极端波动样本数: {top_k}\n")

# ==========================================
# 3. 核心计算循环 (仅在极端样本上)
# ==========================================
results = {}

print("开始计算极端天气下的模型表现...")

for model, folder_path in model_paths.items():
    pred_path = os.path.join(folder_path, 'pred.npy')
    true_path = os.path.join(folder_path, 'true.npy')

    if os.path.exists(pred_path) and os.path.exists(true_path):
        # 加载数据并切片，只保留极端样本
        pred = np.load(pred_path)[extreme_indices]
        true = np.load(true_path)[extreme_indices]

        # 1. 传统 MSE (全通道)
        mse = np.mean((pred - true) ** 2)

        # 2. BVR (%) - PV(1) 和 Wind(2) 通道
        pred_pv_wind = pred[:, :, 1:3]
        violations = pred_pv_wind[pred_pv_wind < 0]
        bvr = (len(violations) / pred_pv_wind.size) * 100

        # 3. MVS (MW): 仅在违规点上的平均绝对违规幅度
        mvs = float(np.mean(np.abs(violations))) if len(violations) > 0 else 0.0

        # 4. KCL 物理联合残差 (NET MAE)
        pred_net = pred[:, :, 0] - pred[:, :, 1] - pred[:, :, 2]
        true_net = true[:, :, 0] - true[:, :, 1] - true[:, :, 2]
        net_mae = np.mean(np.abs(pred_net - true_net))

        # 5. RVM (边界违规幅度): 全部预测点上 relu(-pred) 均值，衡量整体违规严重程度
        rvm = float(np.mean(np.maximum(-pred_pv_wind, 0)))

        # 6. DSA（差分误差对齐）: mean(|Δpred - Δtrue|)，衡量预测趋势变化与真实趋势的小心度
        if pred.shape[1] > 1:
            dsa = float(np.mean(np.abs(np.diff(pred, axis=1) - np.diff(true, axis=1))))
        else:
            dsa = 0.0

        results[model] = {
            'MSE ↓':        mse,
            'BVR (%) ↓':   bvr,
            'MVS (MW) ↓':  mvs,
            'NET MAE ↓':   net_mae,
            'RVM ↓':       rvm,
            'DSA ↓':       dsa,
        }
        print(f"[{model}] 处理完成.")
    else:
        print(f"[警告] 找不到 {model} 的数据文件")

# ==========================================
# 4. 生成极端天气专属表格
# ==========================================
if results:
    df = pd.DataFrame(results).T
    df = df.reindex(model_paths.keys())

    # 按指定列顺序输出
    col_order = ['MSE ↓', 'BVR (%) ↓', 'MVS (MW) ↓', 'NET MAE ↓', 'RVM ↓', 'DSA ↓']
    df = df[[c for c in col_order if c in df.columns]]

    print("\n" + "=" * 88)
    print("   IEEE TABLE II - EXTREME WEATHER ROBUSTNESS (TOP 10% VOLATILITY)")
    print("=" * 88)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 88 + "\n")

    df.to_csv("IEEE_Extreme_Weather_Table.csv", float_format="%.4f")
    print("已生成极端天气评估表格: IEEE_Extreme_Weather_Table.csv")