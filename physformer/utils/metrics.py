import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def NRMSE_Channel_Avg(pred, true):
    nrmse_list = []
    for i in range(pred.shape[-1]):
        rmse = np.sqrt(np.mean((pred[..., i] - true[..., i]) ** 2))
        range_val = np.max(true[..., i]) - np.min(true[..., i]) + 1e-8
        nrmse_list.append(rmse / range_val)
    return float(np.mean(nrmse_list))


def ramp_violation_rate(pred, ramp_limits=None, last_hist=None):
    if ramp_limits is None:
        return 0.0

    pred = np.asarray(pred)
    if pred.ndim < 2:
        return 0.0

    if pred.ndim == 3:
        diffs = [np.abs(pred[:, 1:, :] - pred[:, :-1, :])]
        if last_hist is not None:
            last_hist = np.asarray(last_hist)
            if last_hist.ndim == 1:
                last_hist = last_hist[:, None]
            elif last_hist.ndim == 3:
                last_hist = last_hist[:, -1:, :].squeeze(1)
            diffs.insert(0, np.abs(pred[:, :1, :] - last_hist[:, None, :]))
        diff = np.concatenate(diffs, axis=1)
    elif pred.ndim == 2:
        diff = np.abs(pred[1:, :] - pred[:-1, :])
        diff = diff[:, None, :] if diff.ndim == 2 else diff
    else:
        raise ValueError(f"Unsupported prediction shape for ramp violation: {pred.shape}")

    limits = np.asarray(ramp_limits, dtype=float).reshape(-1)
    if limits.size == 1:
        limits = np.repeat(limits, diff.shape[-1])
    limits = limits[: diff.shape[-1]]
    violations = diff > limits.reshape((1,) * (diff.ndim - 1) + (-1,))
    return float(violations.mean() * 100.0)


def metric(pred, true, ramp_limits=None, last_hist=None):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    ramp = ramp_violation_rate(pred, ramp_limits=ramp_limits, last_hist=last_hist)
    return mae, mse, rmse, ramp


def compute_forecast_metrics(pred, true, ramp_limits=None, last_hist=None):
    mae, mse, rmse, ramp = metric(pred, true, ramp_limits=ramp_limits, last_hist=last_hist)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "net_ramp_violation": float(ramp),
    }


def per_channel_mae(pred, true, names):
    metrics = {}
    for idx, name in enumerate(names):
        metrics[name] = float(np.mean(np.abs(pred[..., idx] - true[..., idx])))
    return metrics


def soc_consistency_error(pred_battery, pred_soc, dt_hours=0.25):
    pred_battery = np.asarray(pred_battery)
    pred_soc = np.asarray(pred_soc)
    if pred_battery.shape != pred_soc.shape:
        raise ValueError("Battery power and SOC shapes must match for SOC consistency evaluation.")
    if pred_battery.shape[1] < 2:
        return 0.0
    soc_delta = pred_soc[:, 1:, :] - pred_soc[:, :-1, :]
    implied_delta = pred_battery[:, :-1, :] * dt_hours
    return float(np.mean(np.abs(soc_delta - implied_delta)))


def composite_val_metric(net_mse, soc_bounds_loss=0.0, ramp_violation=0.0, weights=None):
    """Physics-aware composite metric for model selection.

    Combines forecasting accuracy with physical feasibility penalties so that
    early-stopping and checkpoint selection cannot trade physical plausibility
    for marginal net-MSE improvements.
    """
    if weights is None:
        weights = dict(
            net_mse=1.0,
            soc_bounds=0.1,
            ramp_violation=0.05,
        )
    return (
        weights["net_mse"] * float(net_mse)
        + weights["soc_bounds"] * float(soc_bounds_loss)
        + weights["ramp_violation"] * float(ramp_violation)
    )
