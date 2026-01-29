import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm
from sklearn.metrics import r2_score, mean_absolute_error


# --- IEEE Transactions 绘图风格设置 ---
def set_ieee_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    # 颜色设置：蓝色(预测), 红色(真实), 灰色(辅助)
    return ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']


colors = set_ieee_style()


class ResultVisualizer:
    def __init__(self, exp_name='PhysFormer_experiment_v1',
                 root_dir='exp_results/PhysFormer/checkpoints/'):

        self.path = os.path.join(root_dir, exp_name)
        self.pred_path = os.path.join(self.path, 'real_prediction.npy')
        self.true_path = os.path.join(self.path, 'true.npy')

        # 物理参数 (对应 losses.py 中的定义)
        self.targets = ['Load', 'PV', 'Wind']
        self.ramp_limits = [1.0, 0.25, 0.35]  # MW/15min

        self.load_data()

    def load_data(self):
        if not os.path.exists(self.pred_path):
            raise FileNotFoundError(f"Result files not found at {self.path}. Run test() first.")

        print(f"Loading results from: {self.path}")
        self.preds = np.load(self.pred_path)  # [N, Pred_Len, 3]
        self.trues = np.load(self.true_path)  # [N, Pred_Len, 3]

        print(f"Data Shape: {self.preds.shape}")

    def plot_time_series(self, sample_idx=0):
        """
        图1: 时序预测对比 (IEEE Trans标准双栏宽图)
        展示一个完整的预测窗口，证明模型对趋势的捕捉。
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)

        # 时间轴 (假设是15min间隔)
        time_steps = np.arange(self.preds.shape[1]) * 0.25

        for i, target in enumerate(self.targets):
            ax = axes[i]
            # 绘制真实值
            ax.plot(time_steps, self.trues[sample_idx, :, i],
                    color='k', linestyle='-', linewidth=1.5, label='Ground Truth', alpha=0.6)
            # 绘制预测值
            ax.plot(time_steps, self.preds[sample_idx, :, i],
                    color=colors[0], linestyle='--', linewidth=1.5, label='CFC-Informer')

            # 计算局部 MAE
            mae = mean_absolute_error(self.trues[sample_idx, :, i], self.preds[sample_idx, :, i])

            ax.set_title(f"({chr(97 + i)}) {target} Forecasting (MAE={mae:.3f})")
            ax.set_xlabel("Time (Hours)")
            if i == 0:
                ax.set_ylabel("Power (MW)")
            ax.grid(True, linestyle=':', alpha=0.6)

            # 仅在第一个图显示图例
            if i == 0:
                ax.legend(loc='upper right', frameon=True, edgecolor='black')

        plt.tight_layout()
        save_path = os.path.join(self.path, 'Fig1_TimeSeries.pdf')
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        plt.show()

    def plot_physical_compliance(self):
        """
        图2: 物理流遵循率分析 (核心创新点证明)
        通过 Ramp Rate 的概率密度分布(KDE)，证明模型没有出现非物理的剧烈抖动。
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for i, target in enumerate(self.targets):
            ax = axes[i]
            limit = self.ramp_limits[i]

            # 计算差分 (Ramp Rate)
            # Flatten 整个测试集的所有时间步
            pred_flat = self.preds[:, :, i].flatten()
            true_flat = self.trues[:, :, i].flatten()

            diff_pred = np.abs(np.diff(pred_flat))
            diff_true = np.abs(np.diff(true_flat))

            # 绘制 KDE 分布
            sns.kdeplot(diff_true, ax=ax, color='k', fill=True, alpha=0.1, label='Truth Distribution')
            sns.kdeplot(diff_pred, ax=ax, color=colors[1], linestyle='--', linewidth=2, label='Model Distribution')

            # 绘制物理限制线
            ax.axvline(limit, color='r', linestyle=':', linewidth=2, label=f'Phy Limit ({limit})')

            # 计算越限比例
            violation_rate = np.mean(diff_pred > limit) * 100

            ax.set_title(f"({chr(97 + i)}) {target} Dynamics\nViolation Rate: {violation_rate:.2f}%")
            ax.set_xlabel("| Ramp Rate | (MW/15min)")
            if i == 0:
                ax.set_ylabel("Density")
            else:
                ax.set_ylabel("")

            ax.set_xlim(0, limit * 2.5)  # 聚焦在限制线附近
            ax.grid(True, linestyle=':', alpha=0.5)

            if i == 2:  # 图例放在最后
                ax.legend(loc='upper right')

        plt.tight_layout()
        save_path = os.path.join(self.path, 'Fig2_PhysicalCompliance.pdf')
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        plt.show()

    def plot_scatter_accuracy(self):
        """
        图3: 全局精度回归分析
        展示预测值与真实值的拟合程度 (R2 Score)
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 为了避免点太多，随机采样 5000 个点进行绘图
        sample_indices = np.random.choice(self.preds.shape[0] * self.preds.shape[1], 5000, replace=False)

        for i, target in enumerate(self.targets):
            ax = axes[i]

            y_pred = self.preds[:, :, i].flatten()[sample_indices]
            y_true = self.trues[:, :, i].flatten()[sample_indices]

            # 计算 R2
            r2 = r2_score(y_true, y_pred)

            # 散点图
            ax.scatter(y_true, y_pred, c=colors[0], s=10, alpha=0.3, edgecolors='none')

            # 对角线
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

            ax.set_title(f"({chr(97 + i)}) {target} Regression ($R^2$={r2:.3f})")
            ax.set_xlabel("Ground Truth (MW)")
            if i == 0:
                ax.set_ylabel("Prediction (MW)")

            ax.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(self.path, 'Fig3_AccuracyRegression.pdf')
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        plt.show()


if __name__ == "__main__":
    # 确保这里的 exp_name 与你 run_cfc.py 中的 checkpoint_name 一致
    viz = ResultVisualizer(exp_name='PhysFormer_experiment_v1')

    print(">>> Drawing Time Series Comparison...")
    viz.plot_time_series(sample_idx=10)  # 选择第10个样本展示

    print(">>> Drawing Physical Compliance Report...")
    viz.plot_physical_compliance()

    print(">>> Drawing Accuracy Scatter Plots...")
    viz.plot_scatter_accuracy()

    print("All figures generated successfully.")