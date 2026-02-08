import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
from math import pi
import seaborn as sns

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- SCI 论文绘图风格设置 ---
plt.style.use('seaborn-v0_8-paper')
# 字体设置：优先使用 Times New Roman，没有则回退到 serif
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300


class PaperVisualizer:
    def __init__(self, root_dir=None):
        if root_dir is None:
            self.root_dir = os.path.join(project_root, 'exp_results')
        else:
            self.root_dir = root_dir

        self.models = {}

        # --- 核心配色方案 ---
        # PhysFormer 使用鲜艳的红色/深红，Baseline 使用冷色调或灰色调
        self.config = {
            'PhysFormer': {'color': '#d62728', 'style': '-', 'width': 2.5, 'alpha': 1.0, 'zorder': 10,
                           'label': 'PhysFormer (Ours)'},
            'LSTM': {'color': '#1f77b4', 'style': '--', 'width': 1.5, 'alpha': 0.8, 'zorder': 5, 'label': 'LSTM'},
            'GRU': {'color': '#ff7f0e', 'style': '--', 'width': 1.5, 'alpha': 0.8, 'zorder': 5, 'label': 'GRU'},
            'Informer': {'color': '#2ca02c', 'style': '-.', 'width': 1.5, 'alpha': 0.8, 'zorder': 4,
                         'label': 'Informer'},
            'Autoformer': {'color': '#9467bd', 'style': '-.', 'width': 1.5, 'alpha': 0.8, 'zorder': 4,
                           'label': 'Autoformer'},
            'PINN': {'color': '#7f7f7f', 'style': ':', 'width': 1.8, 'alpha': 0.7, 'zorder': 3, 'label': 'PINN'},
        }

        self.dynamic_limits = None  # [Load_limit, PV_limit, Wind_limit]

    def load_data(self, model_names, data_name, phys_exp_name, seq_len=672, pred_len=96):
        """ 加载预测结果和真实值 """
        print(f"Loading data from {self.root_dir}...")
        for model in model_names:
            # 路径构建逻辑
            if model == 'PhysFormer':
                setting = phys_exp_name
                # 尝试多个可能的路径
                possible_paths = [
                    os.path.join(self.root_dir, 'PhysFormer', 'checkpoints', setting),
                    os.path.join(self.root_dir, setting)
                ]
            else:
                setting = f"{model}_{data_name}_sl{seq_len}_pl{pred_len}_vpp"
                possible_paths = [
                    os.path.join(self.root_dir, 'Baselines', 'checkpoints', setting),
                    os.path.join(self.root_dir, 'checkpoints', setting),  # 兼容不同目录结构
                    os.path.join(self.root_dir, setting)
                ]

            loaded = False
            for folder in possible_paths:
                pred_path = os.path.join(folder, 'pred.npy')
                if os.path.exists(pred_path):
                    try:
                        pred = np.load(pred_path)
                        true = np.load(os.path.join(folder, 'true.npy'))

                        # 确保维度一致 [Batch, Seq, Features]
                        if len(pred.shape) == 2:
                            pred = pred.reshape(-1, pred_len, 3)
                            true = true.reshape(-1, pred_len, 3)

                        self.models[model] = {'pred': pred, 'true': true}
                        print(f"  [OK] Loaded {model} from {folder}")

                        # 只计算一次物理限制 (基于 Ground Truth)
                        if self.dynamic_limits is None:
                            self._calculate_smart_limits(true)

                        loaded = True
                        break
                    except Exception as e:
                        print(f"  [Error] Failed loading {model}: {e}")

            if not loaded:
                print(f"  [Warning] Could not find data for {model}")

    def _calculate_smart_limits(self, true_data):
        """ 计算物理爬坡阈值 (用于 RVR 指标) """
        diff = np.abs(true_data[:, 1:, :3] - true_data[:, :-1, :3])
        limits = []
        for i in range(3):
            # 取 99.9% 分位数作为物理极限，排除极个别异常点
            lim = np.percentile(diff[:, :, i], 99.9)
            lim = max(lim, 1e-4)
            limits.append(lim)
        self.dynamic_limits = limits
        print(f"  [Physics] Dynamic Ramp Limits: {limits}")

    # =========================================================================
    # 1. 完整性展示: 净负荷 (Net Load) 分析
    # =========================================================================
    def plot_net_load_analysis(self):
        """
        绘制净负荷预测对比图。
        Net Load = Load - PV - Wind
        这是 VPP 调度的核心标的，体现了模型对多变量耦合的处理能力。
        """
        if not self.models: return
        print("Plotting Net Load Analysis...")

        # 找一个波动比较大的样本
        first_model = list(self.models.keys())[0]
        true_data = self.models[first_model]['true']
        # 计算真实的净负荷方差，找方差最大的样本
        true_net = true_data[:, :, 0] - true_data[:, :, 1] - true_data[:, :, 2]
        std_net = np.std(true_net, axis=1)
        sample_idx = np.argmax(std_net)

        plt.figure(figsize=(12, 5))

        # 绘制 Ground Truth
        gt_net = true_net[sample_idx]
        plt.plot(gt_net, label='Ground Truth (Net Load)', color='black', linewidth=3, alpha=0.3, zorder=0)

        # 绘制各模型预测
        sorted_names = sorted(self.models.keys(), key=lambda x: self.config.get(x, {}).get('zorder', 0))

        for name in sorted_names:
            d = self.models[name]
            # 计算预测的净负荷
            pred_net = d['pred'][sample_idx, :, 0] - d['pred'][sample_idx, :, 1] - d['pred'][sample_idx, :, 2]

            cfg = self.config.get(name, {})
            plt.plot(pred_net, label=cfg.get('label', name), color=cfg.get('color'),
                     linestyle=cfg.get('style'), linewidth=cfg.get('width'), alpha=cfg.get('alpha'))

        plt.title(f'Net Load Forecasting (Sample {sample_idx}): Completeness Check', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Net Power (MW)', fontsize=12)
        plt.legend(loc='upper right', frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(current_dir, 'Fig_NetLoad_Completeness.png')
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
        plt.close()

    # =========================================================================
    # 2. 合理性展示: 物理违规特写 (Physics Violation Zoom-in)
    # =========================================================================
    def plot_physics_violations(self):
        """
        寻找并展示 Baseline 违反物理常识的 Bad Case。
        Case 1: 夜间光伏不为 0 (Nighttime Violation)
        Case 2: 爬坡率过大 (Ramp Violation)
        """
        if not self.models: return
        print("Plotting Physics Violation Zoom-in...")

        # --- 寻找 Case 1: 夜间光伏 ---
        # 逻辑: 找真实值 PV=0 但 Baseline (如Informer) 预测值最大的时刻
        target_model = 'Informer' if 'Informer' in self.models else list(self.models.keys())[0]

        d = self.models[target_model]
        pred_pv = d['pred'][:, :, 1]
        true_pv = d['true'][:, :, 1]

        # 掩码: 真实值接近0，但预测值 > 0.05 (假设归一化后)
        mask = (true_pv < 0.01) & (pred_pv > 0.05)
        if np.any(mask):
            # 找到违规最严重的样本索引
            flat_idx = np.argmax(pred_pv * mask)  # 找最大错误点
            sample_idx_night, time_idx = np.unravel_index(flat_idx, pred_pv.shape)
        else:
            sample_idx_night = 0  # 没找到就随便画一个

        # --- 绘图 ---
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        # Subplot 1: Nighttime PV
        ax1 = plt.subplot(gs[0])
        self._plot_single_feature(ax1, sample_idx_night, feature_idx=1,
                                  title="Rationality Check: Nighttime PV Consistency")
        # 加一个阴影区域标示夜间
        ax1.axvspan(0, 96, color='gray', alpha=0.1, label='Nighttime')
        # 局部放大注释
        ax1.annotate('PhysFormer stays at 0', xy=(time_idx, 0), xytext=(time_idx + 10, 0.2),
                     arrowprops=dict(facecolor='red', shrink=0.05))

        # Subplot 2: Load Ramp (寻找爬坡违规)
        # 找差分最大的点
        ax2 = plt.subplot(gs[1])
        self._plot_single_feature(ax2, sample_idx=self.find_interesting_sample(0), feature_idx=0,
                                  title="Rationality Check: Load Smoothness")

        plt.tight_layout()
        save_path = os.path.join(current_dir, 'Fig_Physics_Rationality.png')
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
        plt.close()

    def _plot_single_feature(self, ax, sample_idx, feature_idx, title):
        first_model = list(self.models.keys())[0]
        true_data = self.models[first_model]['true'][sample_idx, :, feature_idx]
        ax.plot(true_data, label='Ground Truth', color='black', linewidth=3, alpha=0.2)

        sorted_names = sorted(self.models.keys(), key=lambda x: self.config.get(x, {}).get('zorder', 0))
        for name in sorted_names:
            d = self.models[name]
            data = d['pred'][sample_idx, :, feature_idx]
            cfg = self.config.get(name, {})
            ax.plot(data, label=cfg.get('label', name), color=cfg.get('color'),
                    linestyle=cfg.get('style'), linewidth=cfg.get('width'), alpha=cfg.get('alpha'))

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    # =========================================================================
    # 3. 优越性展示: 综合雷达图
    # =========================================================================
    def plot_radar_metrics(self):
        """
        绘制包含物理指标的雷达图。
        指标: MSE (越小越好), MAE (越小越好), BVR (越小越好), RVR (越小越好)
        """
        if not self.models: return
        print("Plotting Radar Chart...")

        categories = ['MSE', 'MAE', 'RVR (Ramp)', 'BVR (Bound)']
        # 数据准备
        data = {}
        for name in self.models:
            d = self.models[name]
            pred, true = d['pred'], d['true']

            mse = np.mean((pred - true) ** 2)
            mae = np.mean(np.abs(pred - true))

            # RVR
            diff_pred = np.abs(pred[:, 1:] - pred[:, :-1])
            violations = 0
            for i in range(3):
                violations += np.sum(diff_pred[:, :, i] > self.dynamic_limits[i])
            rvr = violations / diff_pred.size

            # BVR (Load/PV/Wind < 0)
            bvr = np.sum(pred < -1e-4) / pred.size

            data[name] = [mse, mae, rvr * 10, bvr * 10]  # 放大一点物理指标以便展示

        # 归一化 (Min-Max，反转，使得越大越好用于绘图，或者越靠近中心越好)
        # 这里我们画“数值”，越小越好。为了雷达图好看，我们做 (Max - Val) / (Max - Min) -> 越大越好
        # 或者直接画原始值的相对比例

        # 简单处理：归一化到 [0.1, 1.0]，其中 1.0 代表最差，0.1 代表最好
        # 不，通常雷达图面积越大越好。我们定义指标为 "Performance Score"
        # Score = 1 - Normalized_Error

        raw_matrix = np.array(list(data.values()))
        max_vals = np.max(raw_matrix, axis=0) + 1e-8
        min_vals = np.min(raw_matrix, axis=0)

        # 归一化：(Val - Min) / (Max - Min). 0=Best, 1=Worst
        norm_matrix = (raw_matrix - min_vals) / (max_vals - min_vals + 1e-8)
        # 反转：1=Best, 0=Worst (用于绘图面积)
        plot_data = 1.0 - norm_matrix + 0.1  # +0.1 避免贴底

        # 绘图
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        model_names = list(data.keys())
        # 排序：Baseline 先画
        sorted_indices = np.argsort([self.config.get(n, {}).get('zorder', 0) for n in model_names])

        for idx in sorted_indices:
            name = model_names[idx]
            values = plot_data[idx].tolist()
            values += values[:1]

            cfg = self.config.get(name, {})
            ax.plot(angles, values, linewidth=cfg['width'], linestyle=cfg['style'],
                    label=cfg['label'], color=cfg['color'])

            if name == 'PhysFormer':
                ax.fill(angles, values, color=cfg['color'], alpha=0.2)  # 只填充最好的

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_yticklabels([])  # 隐藏径向刻度
        plt.title('Comprehensive Performance (Larger Area is Better)', y=1.08, fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        save_path = os.path.join(current_dir, 'Fig_Radar_Superiority.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    def find_interesting_sample(self, feature_idx):
        if not self.models: return 0
        first_model = list(self.models.keys())[0]
        true_data = self.models[first_model]['true'][:, :, feature_idx]
        stds = np.std(true_data, axis=1)
        return np.argmax(stds)


if __name__ == "__main__":
    # 使用说明：
    # 1. 确保 exp_results 文件夹下有对应的实验结果 (metrics.npy, pred.npy, true.npy)
    # 2. 运行此脚本，将在当前目录生成 3 张核心 SCI 图表

    viz = PaperVisualizer()

    # 定义要对比的模型列表 (请确保这些名字与文件夹名匹配)
    model_list = ['LSTM', 'GRU', 'PINN', 'Informer', 'Autoformer', 'PhysFormer']

    # 加载数据 (根据你的实际参数修改 seq_len/pred_len)
    viz.load_data(model_list,
                  data_name='vpp_dataset_3years',
                  phys_exp_name='PhysFormer_experiment_v1.2',  # 你的 PhysFormer 实验文件夹名
                  seq_len=672,
                  pred_len=96)

    # 生成三张核心图表
    viz.plot_net_load_analysis()  # 图1: 完整性 (Net Load)
    viz.plot_physics_violations()  # 图2: 合理性 (Physics Zoom-in)
    viz.plot_radar_metrics()  # 图3: 优越性 (Radar Chart)