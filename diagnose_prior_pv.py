"""
diagnose_prior_pv.py
====================
独立诊断脚本：使用已有权重和已保存的 vis_irr.npy，
直接绘制 prior_pv 曲线，无需修改任何模型代码、无需重新训练。

使用方式：
    python diagnose_prior_pv.py

输出：
    1. 终端打印：learned irr_threshold, slope, 辐照度归一化统计
    2. 图片：IEEE_PhysFormer_PriorDiagnosis.pdf / .png
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models.src.models import PhysFormer # 按你实际的 import 路径修改

# ============================================================
# ★ 用户配置区（按实际路径修改）
# ============================================================
CHECKPOINT_DIR = 'exp_results/PhysFormer/checkpoints/PhysFormer_ensemble_seed2024'
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'checkpoint.pth')

# 与训练时保持一致的模型超参数
MODEL_ARGS = dict(
    enc_in     = 6,
    seq_len    = 672,
    pred_len   = 96,
    d_model    = 256,
    n_heads    = 8,
    e_layers   = 3,
    d_ff       = 1024,
    dropout    = 0.10,
    attn       = 'full',
    embed      = 'custom',
    freq       = 'h',
    activation = 'gelu',
    use_rope   = True,
    rope_base  = 10000,
    distil     = False,
    factor     = 5,
)

SAVE_PATH = CHECKPOINT_DIR
# ============================================================

def set_ieee_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def load_model(checkpoint_path, args_dict, device):
    """加载 PhysFormer 模型并恢复权重"""
    # 将项目根目录加入路径（根据你的目录结构调整）
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    model = PhysFormer(**args_dict).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    # 兼容 DataParallel 包装的权重
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[✓] Model loaded from: {checkpoint_path}")
    return model


def compute_prior_pv(model, irr_data_normalized, device):
    """
    直接调用 causal_coupling.get_hard_prior() 提取 prior_pv。

    Args:
        irr_data_normalized: numpy array, shape [N, seq_len]
            来自已保存的 vis_irr.npy（归一化后的辐照度）
        device: torch.device

    Returns:
        prior_pv_np: numpy array, shape [N, seq_len]
        irr_threshold: float (学到的归一化阈值)
        irr_slope: float (学到的斜率)
    """
    N, S = irr_data_normalized.shape

    # 构造伪天气张量 [N, S, 3]，只有辐照度（索引1）有实际意义
    # 温度（索引0）和风速（索引2）填 0，不影响 prior_pv 的计算
    x_weather = np.zeros((N, S, 3), dtype=np.float32)
    x_weather[:, :, 1] = irr_data_normalized  # 辐照度填入索引1

    x_weather_tensor = torch.tensor(x_weather, dtype=torch.float32).to(device)

    with torch.no_grad():
        # 直接调用 get_hard_prior，不需要完整的 forward pass
        _, prior_pv, _ = model.causal_coupling.get_hard_prior(x_weather_tensor)
        # prior_pv shape: [N, S, 1]

        # 打印诊断信息
        irr_t, wind_t, irr_s, wind_s = model.causal_coupling.get_current_thresholds()
        print("\n" + "="*55)
        print("  [Diagnosis] Learned Physical Thresholds")
        print("="*55)
        print(f"  irr_threshold  (normalized) = {irr_t.item():.4f}")
        print(f"  irr_slope                   = {irr_s.item():.2f}")
        print(f"  wind_threshold (normalized) = {wind_t.item():.4f}")
        print(f"  wind_slope                  = {wind_s.item():.2f}")
        print(f"\n  Irradiance data stats:")
        print(f"    min  = {irr_data_normalized.min():.4f}")
        print(f"    max  = {irr_data_normalized.max():.4f}")
        print(f"    mean = {irr_data_normalized.mean():.4f}")

        # 关键判断：夜间辐照度 vs 阈值
        night_irr = irr_data_normalized[irr_data_normalized < 0.0]
        day_irr   = irr_data_normalized[irr_data_normalized > 0.5]
        print(f"\n    Nighttime values (irr < 0): count={len(night_irr)}, "
              f"mean={night_irr.mean():.4f}" if len(night_irr) > 0 else "    No nighttime values found")
        print(f"    Daytime values  (irr > 0.5): count={len(day_irr)}, "
              f"mean={day_irr.mean():.4f}" if len(day_irr) > 0 else "    No daytime values found")

        # 计算 prior_pv 的日夜对比度
        prior_np = prior_pv.squeeze(-1).cpu().numpy()  # [N, S]
        print(f"\n  prior_pv stats:")
        print(f"    min  = {prior_np.min():.4f}")
        print(f"    max  = {prior_np.max():.4f}")
        print(f"    mean = {prior_np.mean():.4f}")
        print(f"    std  = {prior_np.std():.4f}  "
              f"{'<< 接近0，prior已退化为常数！' if prior_np.std() < 0.05 else '>> 有波动，prior正常'}")
        print("="*55)

        speed_data = np.load(os.path.join(CHECKPOINT_DIR, 'vis_speed.npy'))
        print(f"Wind Speed stats:")
        print(f"  min  = {speed_data.min():.4f}")
        print(f"  max  = {speed_data.max():.4f}")
        print(f"  mean = {speed_data.mean():.4f}")
        low_wind = speed_data[speed_data < -1.0]
        print(f"  Values below -1.0: count={len(low_wind)}, "
              f"ratio={len(low_wind) / speed_data.size * 100:.1f}%")


        return prior_np, irr_t.item(), irr_s.item()


def plot_prior_diagnosis(irr_data, gate_pv_data, prior_pv_data,
                         sample_idx=0, save_path='./'):
    """
    绘制三行对比图：
    (a) 辐照度原始曲线
    (b) prior_pv（硬物理开关）
    (c) gate_pv（最终门控）vs prior_pv 叠加
    """
    irr      = irr_data[sample_idx]       # [S]
    gate_pv  = gate_pv_data[sample_idx]   # [S]
    prior_pv = prior_pv_data[sample_idx]  # [S]

    S = len(irr)
    t = np.arange(S)

    COLORS = {
        'irr':      '#FDAE61',
        'gate':     '#1A9641',
        'prior':    '#D7191C',
        'prior_bg': '#fee0d2',
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # --- Row 1: Irradiance ---
    ax = axes[0]
    ax.fill_between(t, 0, irr, color=COLORS['irr'], alpha=0.5)
    ax.plot(t, irr, color='#E69500', linewidth=1.2)
    ax.set_ylabel('Irradiance\n(Normalized)')
    ax.set_title('(a) Normalized Irradiance Input', loc='left', fontweight='bold')
    ax.set_ylim(min(irr) - 0.2, max(irr) * 1.15)

    # --- Row 2: prior_pv ---
    ax = axes[1]
    ax.fill_between(t, 0, prior_pv, color=COLORS['prior_bg'], alpha=0.6)
    ax.plot(t, prior_pv, color=COLORS['prior'], linewidth=2.0,
            label='$P_{pv}$ (Hard Prior)')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.0, alpha=0.7, label='y=0.5 baseline')
    ax.set_ylabel('Prior Value ∈ [0, 1]')
    ax.set_title('(b) Physics Hard Prior $P_{pv}$ (should show day/night cycle)',
                 loc='left', fontweight='bold')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='upper right', frameon=False)

    # --- Row 3: gate_pv vs prior_pv ---
    ax = axes[2]
    ax.plot(t, prior_pv, color=COLORS['prior'], linewidth=1.5, linestyle='--',
            alpha=0.7, label='$P_{pv}$ (Hard Prior)')
    ax.plot(t, gate_pv, color=COLORS['gate'], linewidth=2.0,
            label='$g_{pv}$ (Final Gate)')
    ax.set_ylabel('Gate/Prior ∈ [0, 1]')
    ax.set_xlabel('Time Steps (15 min)')
    ax.set_title('(c) Comparison: Final Gate vs Hard Prior', loc='left', fontweight='bold')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='upper right', frameon=False)

    os.makedirs(save_path, exist_ok=True)
    out_pdf = os.path.join(save_path, 'IEEE_PhysFormer_PriorDiagnosis.pdf')
    out_png = os.path.join(save_path, 'IEEE_PhysFormer_PriorDiagnosis.png')
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f"\n[✓] Diagnosis plot saved:\n    {out_pdf}\n    {out_png}")


def main():
    set_ieee_style()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # 1. 加载已保存的 vis 数据
    irr_path     = os.path.join(CHECKPOINT_DIR, 'vis_irr.npy')
    gate_pv_path = os.path.join(CHECKPOINT_DIR, 'vis_gate_pv.npy')

    if not os.path.exists(irr_path):
        print(f"[✗] vis_irr.npy not found at: {irr_path}")
        print("    请先运行一次 test（is_training=0），确保 vis_irr.npy 已保存。")
        return

    irr_data    = np.load(irr_path)      # [5, seq_len]
    gate_pv_data = np.load(gate_pv_path) # [5, seq_len]
    print(f"[✓] Loaded vis_irr.npy:     shape={irr_data.shape}")
    print(f"[✓] Loaded vis_gate_pv.npy: shape={gate_pv_data.shape}")

    # 2. 加载模型
    model = load_model(CHECKPOINT_FILE, MODEL_ARGS, device)

    # 3. 计算 prior_pv
    prior_pv_data, irr_t, irr_s = compute_prior_pv(model, irr_data, device)

    # 4. 绘图（默认取 sample 0）
    plot_prior_diagnosis(
        irr_data     = irr_data,
        gate_pv_data = gate_pv_data,
        prior_pv_data= prior_pv_data,
        sample_idx   = 0,
        save_path    = SAVE_PATH
    )

    # 5. 额外保存 prior_pv 供后续使用
    out_npy = os.path.join(CHECKPOINT_DIR, 'vis_prior_pv.npy')
    np.save(out_npy, prior_pv_data)
    print(f"[✓] prior_pv array saved to: {out_npy}")


if __name__ == '__main__':
    main()
