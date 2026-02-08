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

    def load_gate_data(self, phys_exp_name):
        """ 加载 PhysFormer 的门控细节数据 (用于解释性分析) """
        # 尝试从可能的路径加载 gate_details.npy
        possible_paths = [
            os.path.join(self.root_dir, 'PhysFormer', 'checkpoints', phys_exp_name),
            os.path.join(self.root_dir, phys_exp_name)
        ]

        self.gate_data = None
        for folder in possible_paths:
            path = os.path.join(folder, 'gate_details.npy')
            if os.path.exists(path):
                try:
                    # allow_pickle=True 因为保存的是字典
                    self.gate_data = np.load(path, allow_pickle=True).item()
                    print(f"  [Mechanism] Loaded Gate Details from {path}")
                    break
                except Exception as e:
                    print(f"  [Error] Failed loading gates: {e}")

        if self.gate_data is None:
            print("  [Warning] No gate_details.npy found. Skipping mechanism plot.")

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

    def plot_gate_mechanism(self):
        """
        绘制物理门控 (Gate) 的动态变化，解释模型是如何工作的。
        """
        if self.gate_data is None:
            return
        print("Plotting Gate Mechanism Analysis...")

        try:
            # 数据处理逻辑保持不变
            g_load = np.array(self.gate_data['load']).flatten()
            g_pv = np.array(self.gate_data['pv']).flatten()
            g_wind = np.array(self.gate_data['wind']).flatten()
            pv_true = np.array(self.gate_data['pv_true']).flatten()

            # 截取一段典型的时间窗口
            start_idx = 96
            end_idx = 96 * 4
            if len(g_load) < end_idx:
                start_idx = 0
                end_idx = len(g_load)

            g_load = g_load[start_idx:end_idx]
            g_pv = g_pv[start_idx:end_idx]
            g_wind = g_wind[start_idx:end_idx]
            pv_true = pv_true[start_idx:end_idx]

            pv_norm = (pv_true - pv_true.min()) / (pv_true.max() - pv_true.min() + 1e-6)

        except Exception as e:
            print(f"  [Error] Processing gate data failed: {e}")
            return

        # --- 绘图 ---
        fig = plt.figure(figsize=(12, 8))
        # 增加 hspace 以避免子图标题重叠
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)

        x_axis = np.arange(len(g_load))

        # 1. Load Gate Dynamics
        ax1 = plt.subplot(gs[0])
        # 【修复1】添加 r 前缀
        ax1.plot(x_axis, g_load, color='#1f77b4', linewidth=2, label=r'Load Gate $\mathcal{G}_{load}$')
        ax1.set_ylabel('Gate Value (0-1)', fontsize=11)
        ax1.set_title('(a) Load Gate Dynamics (Coupling Strength)', loc='left', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_ylim(0, 1.1)
        ax1.set_xlim(0, len(x_axis))

        # 2. PV Gate vs Solar Irradiance
        ax2 = plt.subplot(gs[1])
        # 【修复1】添加 r 前缀
        l1, = ax2.plot(x_axis, g_pv, color='#d62728', linewidth=2.5, label=r'PV Gate $\mathcal{G}_{pv}$')
        ax2.set_ylabel('Gate Value', fontsize=11, color='#d62728')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        ax2.set_ylim(-0.05, 1.1)
        ax2.set_xlim(0, len(x_axis))

        ax2r = ax2.twinx()
        l2, = ax2r.plot(x_axis, pv_norm, color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.6,
                        label='Normalized PV Power')
        ax2r.set_ylabel('PV Power (Norm.)', fontsize=11, color='#ff7f0e')
        ax2r.tick_params(axis='y', labelcolor='#ff7f0e')

        is_night = pv_norm < 0.05
        ax2.fill_between(x_axis, 0, 1.1, where=is_night, color='gray', alpha=0.15, transform=ax2.get_xaxis_transform(),
                         label='Nighttime')

        ax2.set_title('(b) PV Gate vs. Solar Cycle (Adaptive Attention)', loc='left', fontsize=12, fontweight='bold')

        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper center', ncol=2)
        ax2.grid(True, linestyle='--', alpha=0.3)

        # 3. Wind Gate Dynamics
        ax3 = plt.subplot(gs[2])
        # 【修复1】添加 r 前缀
        ax3.plot(x_axis, g_wind, color='#2ca02c', linewidth=2, label=r'Wind Gate $\mathcal{G}_{wind}$')
        ax3.set_ylabel('Gate Value', fontsize=11)
        ax3.set_xlabel('Time Steps (15 min)', fontsize=12)
        ax3.set_title('(c) Wind Gate Dynamics (Stochastic Adaptation)', loc='left', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.set_ylim(0, 1.1)
        ax3.set_xlim(0, len(x_axis))

        # 【修复2】移除 plt.tight_layout()，改用 savefig 的 bbox_inches='tight' 自动裁剪
        # plt.tight_layout()

        save_path = os.path.join(current_dir, 'Fig_Mechanism_Gates.png')
        # bbox_inches='tight' 会自动计算边界，解决重叠问题
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
    viz = PaperVisualizer()

    # 定义要对比的模型列表
    model_list = ['LSTM', 'GRU', 'PINN', 'Informer', 'Autoformer', 'PhysFormer']

    # 你的实验名称 (确保和 exp_PhysFormer.py 中的一致)
    EXP_NAME = 'PhysFormer_experiment_v1.0'

    # 1. 加载预测数据
    viz.load_data(model_list,
                  data_name='vpp_dataset_3years',
                  phys_exp_name=EXP_NAME,
                  seq_len=672,
                  pred_len=96)

    # 2. 加载 Gate 数据 (新增)
    viz.load_gate_data(phys_exp_name=EXP_NAME)

    # 3. 生成图表
    viz.plot_net_load_analysis()  # 图1
    viz.plot_physics_violations()  # 图2
    viz.plot_radar_metrics()  # 图3
    viz.plot_gate_mechanism()  # 图4 (新增)