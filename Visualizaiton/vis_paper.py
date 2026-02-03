import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from math import pi
import seaborn as sns

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 设置 SCI 论文绘图风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


class PaperVisualizer:
    def __init__(self, root_dir=None):
        if root_dir is None:
            self.root_dir = os.path.join(project_root, 'exp_results')
        else:
            self.root_dir = root_dir

        self.models = {}
        # 优化配色与线型，确保 Baseline 清晰可见
        self.config = {
            'PhysFormer': {'color': '#d62728', 'style': '-', 'width': 2.5, 'alpha': 1.0, 'zorder': 10,
                           'label': 'PhysFormer (Ours)'},  # 红
            'LSTM': {'color': '#1f77b4', 'style': '--', 'width': 1.8, 'alpha': 0.9, 'zorder': 5, 'label': 'LSTM'},  # 蓝
            'GRU': {'color': '#ff7f0e', 'style': '--', 'width': 1.8, 'alpha': 0.9, 'zorder': 5, 'label': 'GRU'},  # 橙
            'Informer': {'color': '#2ca02c', 'style': '-.', 'width': 1.8, 'alpha': 0.9, 'zorder': 4,
                         'label': 'Informer'},  # 绿
            'Autoformer': {'color': '#9467bd', 'style': '-.', 'width': 1.8, 'alpha': 0.9, 'zorder': 4,
                           'label': 'Autoformer'},  # 紫
            'PINN': {'color': '#7f7f7f', 'style': ':', 'width': 2.0, 'alpha': 0.8, 'zorder': 3, 'label': 'PINN'},  # 灰
        }

        # 动态计算的爬坡阈值
        self.dynamic_limits = None

    def load_data(self, model_names, data_name, phys_exp_name, seq_len=672, pred_len=96):
        print(f"Loading data from {self.root_dir}...")
        for model in model_names:
            if model == 'PhysFormer':
                setting = phys_exp_name
                possible_paths = [
                    os.path.join(self.root_dir, 'PhysFormer', 'checkpoints', setting),
                    os.path.join(self.root_dir, setting)
                ]
            else:
                setting = f"{model}_{data_name}_sl{seq_len}_pl{pred_len}_vpp"
                possible_paths = [
                    os.path.join(self.root_dir, 'Baselines', 'checkpoints', setting),
                    os.path.join(self.root_dir, setting)
                ]

            loaded = False
            for folder in possible_paths:
                if os.path.exists(os.path.join(folder, 'pred.npy')):
                    try:
                        pred = np.load(os.path.join(folder, 'pred.npy'))
                        true = np.load(os.path.join(folder, 'true.npy'))

                        # 加载 Metrics (如果不存在或长度不对，后面会重算)
                        metric_file = os.path.join(folder, 'metrics.npy')
                        if os.path.exists(metric_file):
                            metrics = np.load(metric_file)
                        else:
                            metrics = np.array([])  # 空数组标记

                        self.models[model] = {'pred': pred, 'true': true, 'metrics': metrics}
                        print(f"  [OK] Loaded {model}")

                        # 利用 Ground Truth 计算合理的物理阈值 (只算一次)
                        if self.dynamic_limits is None:
                            self._calculate_smart_limits(true)

                        loaded = True
                        break
                    except Exception as e:
                        print(f"  [Error] {model}: {e}")

            if not loaded:
                print(f"  [Fail] Could not find {model}")

    def _calculate_smart_limits(self, true_data):
        """
        [关键修正] 智能计算 RVR 阈值
        逻辑：取 Ground Truth 爬坡率的 99.5% 分位数作为物理极限。
        """
        print("  [Info] Calculating dynamic physical limits from Ground Truth...")
        # true_data: [Samples, Seq, Dims]
        diff = np.abs(true_data[:, 1:, :3] - true_data[:, :-1, :3])
        # 计算每个维度的 99.5% 分位数
        limits = []
        for i in range(3):
            lim = np.percentile(diff[:, :, i], 99.5)
            # 防止全是0的情况 (如光伏夜间)，给一个极小值兜底
            lim = max(lim, 1e-4)
            limits.append(lim)

        self.dynamic_limits = limits
        print(f"  [Info] Dynamic Limits (Load, PV, Wind): {limits}")

    def _recalculate_metrics(self, name, pred, true):
        """ 使用智能阈值重新计算指标 """
        # MSE, MAE
        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))

        # BVR (Load/PV/Wind 均不能小于0)
        # 注意：这里稍微放宽一点点容忍度 -1e-3，避免浮点误差
        bvr = (np.sum(pred < -1e-3) / pred.size) * 100

        # RVR (使用 dynamic_limits)
        diff = np.abs(pred[:, 1:, :3] - pred[:, :-1, :3])
        violations = 0
        total_points = diff.size
        for i in range(3):
            violations += np.sum(diff[:, :, i] > self.dynamic_limits[i])

        rvr = (violations / total_points) * 100

        return [mae, mse, 0, 0, 0, bvr, rvr]

    def plot_radar_chart(self):
        """ [修正版] 雷达图：清晰展示线条，正确计算RVR """
        if not self.models: return

        categories = ['MSE', 'MAE', 'BVR', 'RVR']
        indices = [1, 0, 5, 6]

        # 1. 重新汇总数据 (确保使用统一的 dynamic_limits)
        model_names = list(self.models.keys())
        raw_data = []

        for name in model_names:
            d = self.models[name]
            # 强制重算，确保大家用的是同一套标准
            new_metrics = self._recalculate_metrics(name, d['pred'], d['true'])
            raw_data.append([new_metrics[i] for i in indices])

        raw_data = np.array(raw_data)

        # 2. 归一化 (Min-Max Normalization to [0.1, 1.0])
        # 避免 0 值导致图缩成一点，加一个 epsilon
        max_vals = np.max(raw_data, axis=0) + 1e-8
        # 让最好的模型不完全贴在中心，保留一点距离 (0.1)
        norm_data = 0.1 + 0.9 * (raw_data / max_vals)

        # 3. 绘图
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # 调整绘制顺序：Baseline 先画，PhysFormer 最后画
        sorted_indices = np.argsort([self.config.get(n, {}).get('zorder', 0) for n in model_names])

        for i in sorted_indices:
            name = model_names[i]
            values = norm_data[i].tolist()
            values += values[:1]

            cfg = self.config.get(name, {'color': 'gray'})

            # [修正] 填充透明度设低 (0.05)，线条不透明 (1.0)，确保看不混
            ax.plot(angles, values, linewidth=cfg['width'], linestyle=cfg['style'],
                    label=cfg['label'], color=cfg['color'], zorder=cfg.get('zorder', 1))

            fill_alpha = 0.05 if name != 'PhysFormer' else 0.15
            ax.fill(angles, values, color=cfg['color'], alpha=fill_alpha)

        plt.xticks(angles[:-1], categories, fontsize=12, fontweight='bold')
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [], color="grey", size=7)
        plt.ylim(0, 1.05)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title('Normalized Metrics (Relative Performance)', y=1.08, fontsize=14, fontweight='bold')

        save_path = os.path.join(current_dir, 'vis_radar.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Radar Chart: {save_path}")
        plt.close()

    def plot_cumulative_energy_error(self):
        """
        [新增图表 1] 累计能量偏差图
        展示 Baseline 预测的电量随时间产生积分漂移，而 PhysFormer 保持守恒。
        """
        if not self.models: return

        plt.figure(figsize=(10, 5))

        # 随机选一个 Batch 或 Test set 前 1000 个点
        limit_steps = 288  # 3天数据 (96*3)

        for name in self.models:
            d = self.models[name]
            # 取出前 limit_steps 的数据，Flatten
            # 假设我们只关心 Load + PV + Wind 的总能量偏差，或者单独看 PV
            # 这里计算所有通道的总偏差: sum(Pred - True)
            pred = d['pred'].reshape(-1, 3)[:limit_steps]
            true = d['true'].reshape(-1, 3)[:limit_steps]

            # 计算每个时间步的误差 (Pred - True)
            error = np.sum(pred - true, axis=1)  # [T]
            # 累计误差
            cum_error = np.cumsum(error)

            cfg = self.config.get(name, {'color': 'gray'})
            plt.plot(cum_error, label=cfg['label'], color=cfg['color'],
                     linewidth=cfg['width'], linestyle=cfg['style'])

        plt.axhline(0, color='black', linestyle='-', linewidth=1)
        plt.title('Cumulative Energy Deviation (3 Days)', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps (15min)', fontsize=12)
        plt.ylabel('Cumulative Energy Error (MWh)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(current_dir, 'vis_energy_cum.png')
        plt.savefig(save_path, dpi=300)
        print(f"Saved Energy Plot: {save_path}")
        plt.close()

    def plot_error_boxplot(self):
        """
        [新增图表 2] 误差分布箱线图
        展示 PhysFormer 的误差更集中，离群点更少。
        """
        if not self.models: return

        data_to_plot = []
        labels = []
        colors = []

        # 排序
        sorted_names = [n for n in self.models.keys() if n != 'PhysFormer'] + ['PhysFormer']

        for name in sorted_names:
            if name not in self.models: continue
            d = self.models[name]
            # 计算绝对误差
            abs_err = np.abs(d['pred'] - d['true']).flatten()
            # 为了画图清晰，去掉极端的 outliers (前99%)
            # abs_err = abs_err[abs_err < np.percentile(abs_err, 99)]

            data_to_plot.append(abs_err)
            labels.append(name)
            colors.append(self.config.get(name, {}).get('color', 'gray'))

        plt.figure(figsize=(10, 6))

        # 使用 Seaborn 绘制 Violin Plot (比 Boxplot 更漂亮，能看到密度)
        ax = sns.violinplot(data=data_to_plot, palette=colors, inner="quartile")

        # 替换 X 轴标签
        ax.set_xticklabels(labels, fontsize=11)
        plt.title('Absolute Error Distribution (Violin Plot)', fontsize=14, fontweight='bold')
        plt.ylabel('Absolute Error (MW)', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(current_dir, 'vis_error_dist.png')
        plt.savefig(save_path, dpi=300)
        print(f"Saved Error Distribution: {save_path}")
        plt.close()

    # 保留之前的时序图函数
    def find_interesting_sample(self, feature_idx):
        if not self.models: return 0
        first_model = list(self.models.keys())[0]
        true_data = self.models[first_model]['true'][:, :, feature_idx]
        stds = np.std(true_data, axis=1)
        best_idx = np.argmax(stds)
        return best_idx

    def plot_time_series_zoom(self, feature_idx=1, feature_name='PV Power'):
        if not self.models: return
        sample_idx = self.find_interesting_sample(feature_idx)
        plt.figure(figsize=(10, 4))

        first_key = list(self.models.keys())[0]
        true_data = self.models[first_key]['true'][sample_idx, :, feature_idx]
        plt.plot(true_data, label='Ground Truth', color='black', linewidth=2.5, alpha=0.9, zorder=0)

        sorted_names = [m for m in self.models.keys() if m != 'PhysFormer']
        # 按 Config 中的 zorder 排序
        sorted_names.sort(key=lambda x: self.config.get(x, {}).get('zorder', 0))
        if 'PhysFormer' in self.models: sorted_names.append('PhysFormer')

        for name in sorted_names:
            if name not in self.models: continue
            pred_data = self.models[name]['pred'][sample_idx, :, feature_idx]
            cfg = self.config.get(name, {'color': 'gray', 'style': '--', 'width': 1})
            plt.plot(pred_data, label=cfg['label'], color=cfg['color'], linestyle=cfg['style'],
                     linewidth=cfg['width'], alpha=cfg['alpha'], zorder=cfg.get('zorder', 1))

        plt.title(f'{feature_name} Forecasting (Sample {sample_idx})', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
        plt.tight_layout()
        save_path = os.path.join(current_dir, f'vis_series_{feature_name.split()[0]}.png')
        plt.savefig(save_path, dpi=300)
        print(f"Saved Time Series: {save_path}")
        plt.close()


if __name__ == "__main__":
    viz = PaperVisualizer()
    # 确保列表里的模型都跑过
    model_list = ['LSTM', 'GRU', 'PINN', 'Informer', 'Autoformer', 'PhysFormer']

    viz.load_data(model_list,
                  data_name='vpp_dataset_3years',
                  phys_exp_name='PhysFormer_experiment_v6',
                  seq_len=672,
                  pred_len=96)

    # 1. 基础时序图
    viz.plot_time_series_zoom(feature_idx=0, feature_name='Load')
    viz.plot_time_series_zoom(feature_idx=1, feature_name='PV Power')
    viz.plot_time_series_zoom(feature_idx=2, feature_name='Wind Power')

    # 2. 修正后的雷达图 (能看到线，能看到RVR)
    viz.plot_radar_chart()

    # 3. [新增] 累计能量偏差图
    viz.plot_cumulative_energy_error()

    # 4. [新增] 误差分布提琴图
    viz.plot_error_boxplot()