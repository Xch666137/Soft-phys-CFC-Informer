import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def NRMSE_Channel_Avg(pred, true):
    """
    计算各通道独立 NRMSE 后的平均值，消除量纲影响。
    专用于 Early Stopping，防止模型坍塌向数值绝对量级大的通道（如Load）。
    """
    nrmse_list = []
    for i in range(pred.shape[-1]):
        rmse = np.sqrt(np.mean((pred[..., i] - true[..., i]) ** 2))
        range_val = np.max(true[..., i]) - np.min(true[..., i]) + 1e-8
        nrmse_list.append(rmse / range_val)
    return np.mean(nrmse_list)


# --- 论文核心物理指标 ---

def BVR(pred):
    """
    Boundary Violation Rate (边界违规率)
    计算预测值中违反物理边界（如 < 0）的比例。
    对于 Load/PV/Wind，通常下界为 0。
    """
    # 统计小于 0 的元素个数
    violation_count = np.sum(pred < 0)
    total_count = pred.size
    return (violation_count / total_count) * 100  # 返回百分比


def RVR(pred, limits=None):
    """
    Ramp Violation Rate (爬坡违规率)
    计算相邻时间步的变化率超过物理极限的比例。

    Args:
        pred: [Batch, Seq_Len, Variables] 或 [Total_Samples, Variables]
        limits: list, 对应每个变量的爬坡阈值 (MW/step)。
                如果为 None，则使用默认经验值或跳过。
    """
    if limits is None:
        raise ValueError("[Metrics Error] Ramp limits must be explicitly provided. "
                         "Do not use arbitrary hardcoded values.")

    # 计算一阶差分 (绝对值)
    # axis=-2 表示在时间维度 (Seq_Len) 上做差分
    if pred.ndim == 3:
        diff = np.abs(pred[:, 1:, :] - pred[:, :-1, :])
    elif pred.ndim == 2:
        diff = np.abs(pred[1:, :] - pred[:-1, :])
    else:
        return 0.0

    total_violations = 0
    total_points = diff.shape[0] * diff.shape[1]  # Batch * (Seq-1)

    # 针对每个变量分别统计
    num_vars = min(len(limits), pred.shape[-1])

    for i in range(num_vars):
        limit = limits[i]
        # 该变量的所有样本、所有时间步的违规次数
        v_count = np.sum(diff[..., i] > limit)
        total_violations += v_count

    # 计算总违规率 (所有变量平均)
    total_points_all_vars = total_points * num_vars
    if total_points_all_vars == 0: return 0.0

    return (total_violations / total_points_all_vars) * 100


def metric(pred, true, ramp_limits):
    """
    综合计算所有指标
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    # 物理指标 (仅针对预测值 pred 计算)
    bvr = BVR(pred)
    rvr = RVR(pred, limits=ramp_limits)

    return mae, mse, rmse, mape, mspe, bvr, rvr


# ============================================================================
# 物理约束合规性检查类 (Physics Compliance Metrics)
# ============================================================================

class PhysicsComplianceMetrics:
    """
    物理约束合规性检查类

    评估模型输出是否符合VPP硬性物理约束，包括：
    1. 功率边界约束 (非负性)
    2. 基尔霍夫电流定律 (净负荷守恒)
    3. 能量守恒定律
    4. 爬坡率约束
    """

    def __init__(self, means=None, stds=None, ramp_limits=None):
        """
        Args:
            means: 各变量的均值，用于反归一化 [3]
            stds: 各变量的标准差，用于反归一化 [3]
            ramp_limits: 爬坡极限 [Load, PV, Wind] (MW/step)
        """
        self.means = np.array(means) if means is not None else np.zeros(3)
        self.stds = np.array(stds) if stds is not None else np.ones(3)
        self.ramp_limits = np.array(ramp_limits)

    def denormalize(self, data):
        """反归一化数据到真实物理单位"""
        if len(data.shape) == 2:
            # [N, 3]
            return data * self.stds + self.means
        elif len(data.shape) == 3:
            # [B, T, 3]
            return data * self.stds + self.means
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

    def violation_rate(self, pred, threshold=0.0):
        """
        计算功率越限比例 (Violation Rate)

        Args:
            pred: 预测值 [..., 3] (归一化或已反归一化)
            threshold: 下界阈值 (默认0.0，表示非负性约束)

        Returns:
            violation_rate: 违规比例 (%)
            violation_count: 违规点数量
            total_points: 总点数
        """
        # 确保使用真实物理单位
        if np.all(self.stds == 1.0) and np.all(self.means == 0.0):
            pred_real = pred  # 假设已经是真实单位
        else:
            pred_real = self.denormalize(pred)

        # 统计小于阈值的违规点
        violation_mask = pred_real < threshold
        violation_count = np.sum(violation_mask)
        total_points = pred_real.size

        violation_rate = (violation_count / total_points) * 100 if total_points > 0 else 0.0

        return {
            'violation_rate': violation_rate,
            'violation_count': int(violation_count),
            'total_points': total_points,
            'per_channel_rate': [np.sum(violation_mask[..., i]) / pred_real[..., i].size * 100
                                for i in range(pred_real.shape[-1])]
        }

    def kcl_residual(self, pred, true=None):
        """
        计算基尔霍夫电流定律残差 (KCL Residual)

        净负荷守恒：Load - PV - Wind = Net_Load
        理想情况下，预测的净负荷应与真实净负荷一致（如果true提供），
        或者净负荷的统计特性应符合物理规律。

        Args:
            pred: 预测值 [..., 3] (归一化或已反归一化)
            true: 真实值 [..., 3] (可选，用于计算残差)

        Returns:
            kcl_metrics: 包含各种KCL指标的字典
        """
        # 反归一化到真实物理单位
        pred_real = self.denormalize(pred)

        # 计算预测净负荷
        net_pred = pred_real[..., 0] - pred_real[..., 1] - pred_real[..., 2]  # Load - PV - Wind

        metrics = {
            'net_pred_mean': np.mean(net_pred),
            'net_pred_std': np.std(net_pred),
            'net_pred_min': np.min(net_pred),
            'net_pred_max': np.max(net_pred),
        }

        if true is not None:
            true_real = self.denormalize(true)
            net_true = true_real[..., 0] - true_real[..., 1] - true_real[..., 2]

            # 计算残差
            residual = net_pred - net_true
            metrics.update({
                'residual_mean': np.mean(residual),
                'residual_std': np.std(residual),
                'residual_mae': np.mean(np.abs(residual)),
                'residual_rmse': np.sqrt(np.mean(residual ** 2)),
                'correlation': np.corrcoef(net_pred.flatten(), net_true.flatten())[0, 1]
                              if net_pred.size > 1 else 0.0,
            })

        return metrics

    def energy_conservation_error(self, pred, true=None, time_interval=0.25):
        """
        计算能量守恒误差 (Energy Conservation Error)

        总发电量（PV + Wind）与负荷的能量差应在合理范围内。
        这里计算每日/每小时的能量平衡误差。

        Args:
            pred: 预测值 [B, T, 3] 或 [T, 3]
            true: 真实值 [B, T, 3] 或 [T, 3] (可选)
            time_interval: 时间间隔 (小时)，默认0.25小时 (15分钟)

        Returns:
            energy_metrics: 能量守恒相关指标
        """
        pred_real = self.denormalize(pred)

        # 计算能量 (功率 × 时间)
        # 假设数据是15分钟间隔 (0.25小时)
        energy_load = pred_real[..., 0] * time_interval  # [B, T] 或 [T]
        energy_pv = pred_real[..., 1] * time_interval
        energy_wind = pred_real[..., 2] * time_interval

        # 净能量：负荷 - 发电
        net_energy = energy_load - energy_pv - energy_wind

        metrics = {
            'total_energy_load': np.sum(energy_load),
            'total_energy_pv': np.sum(energy_pv),
            'total_energy_wind': np.sum(energy_wind),
            'net_energy': np.sum(net_energy),
            'energy_imbalance_ratio': np.abs(np.sum(net_energy)) / (np.sum(energy_load) + 1e-8) * 100,
        }

        if true is not None:
            true_real = self.denormalize(true)
            energy_load_true = true_real[..., 0] * time_interval
            energy_pv_true = true_real[..., 1] * time_interval
            energy_wind_true = true_real[..., 2] * time_interval
            net_energy_true = energy_load_true - energy_pv_true - energy_wind_true

            # 能量预测误差
            energy_error = net_energy - net_energy_true
            metrics.update({
                'energy_error_mean': np.mean(energy_error),
                'energy_error_std': np.std(energy_error),
                'energy_error_mae': np.mean(np.abs(energy_error)),
                'energy_error_rmse': np.sqrt(np.mean(energy_error ** 2)),
            })

        return metrics

    def ramp_violation_analysis(self, pred):
        """
        爬坡违规详细分析

        Args:
            pred: 预测值 [..., 3]

        Returns:
            ramp_metrics: 爬坡违规详细统计
        """
        pred_real = self.denormalize(pred)

        if len(pred_real.shape) == 3:
            # [B, T, 3]
            diff = np.abs(pred_real[:, 1:, :] - pred_real[:, :-1, :])  # [B, T-1, 3]
        elif len(pred_real.shape) == 2:
            # [T, 3]
            diff = np.abs(pred_real[1:, :] - pred_real[:-1, :])  # [T-1, 3]
        else:
            return {'error': 'Unsupported shape'}

        metrics = {}
        for i, var_name in enumerate(['Load', 'PV', 'Wind']):
            if self.ramp_limits is None or i >= len(self.ramp_limits):
                raise ValueError(f"[Metrics Error] Missing ramp limit for {var_name}")

            limit = self.ramp_limits[i]
            var_diff = diff[..., i]

            violations = var_diff > limit
            violation_count = np.sum(violations)
            total_points = var_diff.size

            metrics.update({
                f'{var_name}_ramp_violation_rate': violation_count / total_points * 100 if total_points > 0 else 0.0,
                f'{var_name}_ramp_violation_count': int(violation_count),
                f'{var_name}_max_ramp': np.max(var_diff),
                f'{var_name}_mean_ramp': np.mean(var_diff),
                f'{var_name}_std_ramp': np.std(var_diff),
            })

        return metrics

    def compute_all_metrics(self, pred, true=None, time_interval=0.25):
        """
        计算所有物理合规性指标

        Args:
            pred: 预测值 [..., 3]
            true: 真实值 [..., 3] (可选)
            time_interval: 时间间隔 (小时)

        Returns:
            all_metrics: 所有指标的字典
        """
        all_metrics = {}

        # 1. 边界违规率
        violation = self.violation_rate(pred)
        all_metrics.update({f'violation_{k}': v for k, v in violation.items()})

        # 2. KCL残差
        kcl = self.kcl_residual(pred, true)
        all_metrics.update({f'kcl_{k}': v for k, v in kcl.items()})

        # 3. 能量守恒
        energy = self.energy_conservation_error(pred, true, time_interval)
        all_metrics.update({f'energy_{k}': v for k, v in energy.items()})

        # 4. 爬坡分析
        ramp = self.ramp_violation_analysis(pred)
        all_metrics.update({f'ramp_{k}': v for k, v in ramp.items()})

        return all_metrics


def physics_compliance_metrics(pred, true=None, means=None, stds=None, ramp_limits=None, time_interval=0.25):
    """
    便捷函数：计算物理合规性指标

    Args:
        pred: 预测值 [..., 3]
        true: 真实值 [..., 3] (可选)
        means: 各变量的均值 [3]
        stds: 各变量的标准差 [3]
        ramp_limits: 爬坡极限 [Load, PV, Wind]
        time_interval: 时间间隔 (小时)

    Returns:
        metrics_dict: 指标字典
    """
    compliance = PhysicsComplianceMetrics(means=means, stds=stds, ramp_limits=ramp_limits)
    return compliance.compute_all_metrics(pred, true=true, time_interval=time_interval)