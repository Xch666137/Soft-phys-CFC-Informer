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


# --- 新增：论文核心物理指标 ---

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
        # 默认阈值，需根据 vpp_dataset_3years.csv 的实际波动范围调整
        # 假设顺序: [Load, PV, Wind]
        # 这里的数值建议设为训练集最大爬坡率的 1.1 倍，作为"物理极限"
        limits = [1.5, 0.8, 1.0]

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


def metric(pred, true, ramp_limits=None):
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