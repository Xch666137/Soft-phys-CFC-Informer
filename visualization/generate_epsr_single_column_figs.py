from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "paper" / "en" / "Visualization_output"
DATA_ROOT = Path(r"E:\Py_program\Soft-phys-CFC-Informer\exp_results")
PHYS_DIR = DATA_ROOT / "PhysFormer" / "checkpoints" / "PhysFormer_full_seed2024"


COLORS = {
    "phys": "#c73e3a",
    "phys_light": "#f6d6d5",
    "blue": "#3b6ea8",
    "blue_light": "#dce8f6",
    "green": "#2c7c59",
    "green_light": "#d8eee5",
    "gold": "#c58f17",
    "gold_light": "#f7ebca",
    "gray": "#5d6770",
    "gray_light": "#ebeff2",
    "dark": "#1e2832",
    "informer": "#d06b36",
    "dlinear": "#6a9f58",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.2,
            "lines.linewidth": 2.0,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d8dee6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    plt.close(fig)


NOTE_BBOX = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#d0d7df", alpha=0.95)


def add_box(ax, xy, wh, text, fc, ec="#334", fontsize=10.5, weight="bold"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)
    return patch


def add_step(ax, x, y, label, fc):
    circ = Circle((x, y), 0.026, facecolor=fc, edgecolor=COLORS["dark"], linewidth=1.0)
    ax.add_patch(circ)
    ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold", color=COLORS["dark"])


def add_arrow(ax, start, end, color=COLORS["dark"], style="-|>", lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def generate_architecture() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Inputs
    add_box(ax, (0.05, 0.07), (0.30, 0.10), "Historical operational\nrecords", COLORS["blue_light"], fontsize=9.9)
    add_box(ax, (0.40, 0.07), (0.22, 0.10), "Historical\nweather", COLORS["green_light"], fontsize=10.0)
    add_box(ax, (0.67, 0.07), (0.23, 0.10), "Future\nweather", COLORS["gold_light"], fontsize=10.0)

    # Dual streams
    add_box(ax, (0.09, 0.27), (0.31, 0.13), "Statistical stream", COLORS["blue_light"], fontsize=11.0)
    add_box(ax, (0.55, 0.27), (0.32, 0.13), "Explicit physical\nmapping", COLORS["green_light"], fontsize=10.7)

    # Main pathway
    add_box(ax, (0.23, 0.47), (0.54, 0.11), "PGCC fusion", "#f7ebca", fontsize=11.4)
    add_box(ax, (0.17, 0.63), (0.66, 0.12), "Horizon projection\n+ future-weather GLU", COLORS["gray_light"], fontsize=10.8)
    add_box(ax, (0.15, 0.785), (0.70, 0.125), "BPAR + activity gating", COLORS["phys_light"], fontsize=11.3)
    add_box(ax, (0.29, 0.928), (0.42, 0.058), "Load / PV / wind forecasts", "#f7f7f7", fontsize=9.8, weight="normal")

    # Input connections
    add_arrow(ax, (0.20, 0.17), (0.245, 0.27), color=COLORS["blue"], lw=1.8)
    add_arrow(ax, (0.51, 0.17), (0.245, 0.315), color=COLORS["blue"], lw=1.5)
    add_arrow(ax, (0.51, 0.17), (0.71, 0.27), color=COLORS["green"], lw=1.7)
    add_arrow(ax, (0.785, 0.17), (0.745, 0.63), color=COLORS["gold"], lw=1.8)

    # Stream to output connections
    add_arrow(ax, (0.39, 0.34), (0.41, 0.47), color=COLORS["dark"], lw=1.7)
    add_arrow(ax, (0.56, 0.34), (0.59, 0.47), color=COLORS["dark"], lw=1.7)
    add_arrow(ax, (0.50, 0.58), (0.50, 0.63), color=COLORS["dark"], lw=1.9)
    add_arrow(ax, (0.50, 0.75), (0.50, 0.79), color=COLORS["phys"], lw=1.9)
    add_arrow(ax, (0.50, 0.91), (0.50, 0.928), color=COLORS["phys"], lw=1.9)

    fig.subplots_adjust(left=0.04, right=0.96, top=0.98, bottom=0.04)

    save(fig, "EPSR_Fig1_Architecture")


def generate_mapping() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.95))

    temps = np.linspace(-10, 45, 220)
    beta = 0.0041
    pv_handles = []
    pv_labels = []
    for G, label, color in [(0.25, "Low irradiance", "#4c78a8"), (0.60, "Medium irradiance", "#f28e2b"), (1.00, "High irradiance", "#59a14f")]:
        pv = G * np.clip(1 - beta * (temps - 25), 0, 1.5)
        line, = axes[0].plot(temps, pv, color=color, label=label)
        pv_handles.append(line)
        pv_labels.append(label)
    axes[0].axvline(25, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    axes[0].annotate(
        r"$\beta_T = 0.0041\ /\!^\circ\mathrm{C}$",
        xy=(25, 1.00),
        xytext=(31.5, 1.07),
        fontsize=8.8,
        color=COLORS["gray"],
        bbox=NOTE_BBOX,
        arrowprops=dict(arrowstyle="-", color=COLORS["gray"], linewidth=0.9),
    )
    axes[0].set_ylabel("Normalized PV output")
    axes[0].set_xlabel("Cell temperature ($^\circ$C)")
    axes[0].set_title("(a) PV output decreases smoothly as temperature rises", pad=12)
    axes[0].grid(True)

    speeds = np.linspace(0, 30, 400)
    v_ci, v_r, v_co = 3.458, 11.970, 24.968
    wind = np.zeros_like(speeds)
    mask_ramp = (speeds >= v_ci) & (speeds < v_r)
    mask_flat = (speeds >= v_r) & (speeds <= v_co)
    wind[mask_ramp] = 1 / (1 + np.exp(-5 * (((speeds[mask_ramp] - v_ci) / (v_r - v_ci)) - 0.5)))
    wind[mask_flat] = 1.0
    axes[1].plot(speeds, wind, color=COLORS["green"], linewidth=2.3)
    axes[1].fill_between(speeds, 0, wind, where=((speeds >= v_ci) & (speeds <= v_co)), color=COLORS["green_light"], alpha=0.72)
    for x, lbl in [(v_ci, r"$v_{ci}$"), (v_r, r"$v_r$"), (v_co, r"$v_{co}$")]:
        axes[1].axvline(x, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    axes[1].annotate(
        r"$v_{ci}$",
        xy=(v_ci, 0.12),
        xytext=(v_ci - 0.15, 1.05),
        ha="center",
        fontsize=8.8,
        color=COLORS["gray"],
        bbox=NOTE_BBOX,
        arrowprops=dict(arrowstyle="-", color=COLORS["gray"], linewidth=0.9),
    )
    axes[1].annotate(
        r"$v_r$",
        xy=(v_r, 0.90),
        xytext=(v_r + 0.1, 0.14),
        ha="center",
        fontsize=8.8,
        color=COLORS["gray"],
        bbox=NOTE_BBOX,
        arrowprops=dict(arrowstyle="-", color=COLORS["gray"], linewidth=0.9),
    )
    axes[1].annotate(
        r"$v_{co}$",
        xy=(v_co, 1.0),
        xytext=(v_co - 0.05, 1.05),
        ha="center",
        fontsize=8.8,
        color=COLORS["gray"],
        bbox=NOTE_BBOX,
        arrowprops=dict(arrowstyle="-", color=COLORS["gray"], linewidth=0.9),
    )
    axes[1].set_ylim(-0.03, 1.12)
    axes[1].set_ylabel("Wind power coefficient")
    axes[1].set_xlabel("Wind speed (m/s)")
    axes[1].set_title("(b) Learned wind thresholds preserve the operating region", pad=10)
    axes[1].grid(True)

    fig.legend(
        pv_handles,
        pv_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.tight_layout(h_pad=1.9, rect=(0.0, 0.0, 1.0, 0.965))
    save(fig, "EPSR_Fig2_PhysicalMapping")


def generate_bpar_mechanism() -> None:
    fig = plt.figure(figsize=(6.0, 7.25))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.16], hspace=0.54)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    legend_ax = fig.add_subplot(gs[2])
    legend_ax.axis("off")

    x = np.linspace(-2.5, 3.0, 400)
    axes[0].plot(x, x, linestyle="--", color=COLORS["gray"], label="Unconstrained residual head")
    axes[0].plot(x, np.maximum(x, 0), color=COLORS["informer"], label="Hard post-clipping")
    axes[0].plot(x, np.log1p(np.exp(x)), color=COLORS["phys"], label="BPAR softplus manifold")
    axes[0].axhline(0, color=COLORS["dark"], linewidth=1.0)
    axes[0].axvline(0, color=COLORS["dark"], linewidth=1.0, linestyle=":")
    axes[0].annotate(
        "Smooth positive floor\nwith non-zero gradient",
        xy=(0.2, np.log1p(np.exp(0.2))),
        xytext=(1.15, 1.45),
        arrowprops=dict(arrowstyle="->", color=COLORS["phys"], linewidth=1.2),
        color=COLORS["phys"],
        fontsize=8.8,
        bbox=NOTE_BBOX,
    )
    axes[0].set_xlabel("Pre-activation relative to physical zero")
    axes[0].set_ylabel(r"Output above $P_{zero}$")
    axes[0].set_title("(a) BPAR keeps the output non-negative without a dead zone", pad=10)
    axes[0].grid(True)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.98))

    t = np.arange(96)
    raw = 0.72 * np.exp(-((t - 46) / 25) ** 2) + 0.05
    gate = 1 / (1 + np.exp((t - 60) / 3.0))
    final = gate * raw
    ax2 = axes[1]
    ax2.plot(t, raw, color=COLORS["gray"], linestyle="--", label=r"$\hat P_{raw}$ from BPAR")
    ax2.plot(t, final, color=COLORS["phys"], label=r"Final output after $a_x$ gating")
    ax2.axhline(0, color=COLORS["dark"], linewidth=1.0)
    ax2.fill_between(t, 0, final, color=COLORS["phys_light"], alpha=0.85)
    ax2.set_ylabel("Illustrative output")
    ax2.set_xlabel("Transition interval")
    ax2.set_title("(b) Activity gating drives the bounded output back to zero smoothly", pad=12)
    ax2.grid(True)
    ax2b = ax2.twinx()
    ax2b.plot(t, gate, color=COLORS["blue"], linewidth=1.6, label=r"Activity gate $a_x$")
    ax2b.set_ylim(-0.05, 1.05)
    ax2b.set_ylabel(r"$a_x$", color=COLORS["blue"])
    ax2b.tick_params(axis="y", labelcolor=COLORS["blue"])
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    legend_ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        loc="center",
        ncol=2,
        columnspacing=1.1,
        handlelength=1.8,
    )

    save(fig, "EPSR_Fig3_BPARMechanism")


def generate_gate_alignment() -> None:
    gate = np.load(PHYS_DIR / "vis_gate_pv.npy")
    irr = np.load(PHYS_DIR / "vis_irr.npy")

    sample_idx = 0
    start, stop = 96, 320
    gate_seq = gate[sample_idx, start:stop]
    irr_seq = irr[sample_idx, start:stop]

    flat_gate = gate.reshape(-1).astype(np.float64)
    flat_irr = irr.reshape(-1).astype(np.float64)
    subset_r, _ = pearsonr(flat_gate, flat_irr)

    rng = np.random.default_rng(2026)
    pick = rng.choice(len(flat_gate), size=min(1800, len(flat_gate)), replace=False)
    irr_sub = flat_irr[pick]
    gate_sub = flat_gate[pick]
    fit = np.polyfit(flat_irr, flat_gate, 1)
    x_line = np.linspace(flat_irr.min(), flat_irr.max(), 150)

    fig = plt.figure(figsize=(6.0, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    t = np.arange(len(gate_seq))
    ax1.fill_between(t, 0, irr_seq, color=COLORS["gold"], alpha=0.35, label="Normalized irradiance")
    ax1.set_ylabel("Irradiance")
    ax1.grid(True)
    ax1.set_title("(a) The learned PV gate follows irradiance on representative test windows", pad=18)
    ax1b = ax1.twinx()
    ax1b.plot(t, gate_seq, color=COLORS["phys"], linewidth=2.2, label="PV gate")
    ax1b.set_ylabel("Gate activation", color=COLORS["phys"])
    ax1b.tick_params(axis="y", labelcolor=COLORS["phys"])
    ax1.set_xlabel("Historical time steps (15 min)")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1b.get_legend_handles_labels()
    ax1.legend(
        l1 + l2,
        lb1 + lb2,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        columnspacing=1.0,
        handlelength=1.8,
    )

    ax2.scatter(irr_sub, gate_sub, s=13, alpha=0.22, color=COLORS["blue"], edgecolors="none")
    ax2.plot(x_line, fit[0] * x_line + fit[1], color=COLORS["phys"], linewidth=2.0, linestyle="--")
    ax2.text(
        0.05,
        0.83,
        f"Visualization subset: N={len(flat_gate):,}, r={subset_r:.3f}\nReported full-test Pearson correlation: r=0.8306",
        transform=ax2.transAxes,
        va="top",
        fontsize=8.6,
        bbox=NOTE_BBOX,
    )
    ax2.set_xlabel("Normalized irradiance")
    ax2.set_ylabel("PV gate activation")
    ax2.set_title("(b) The alignment persists over the visualization subset")
    ax2.grid(True)

    fig.subplots_adjust(top=0.95)
    save(fig, "EPSR_Fig4_GateAlignment")


def generate_pareto() -> None:
    data = [
        ("LSTM", 0.0168, 11.25, 0.0093),
        ("GRU", 0.0163, 11.15, 0.0103),
        ("PINN", 0.0157, 10.94, 0.0097),
        ("Informer", 0.0121, 19.53, 0.0036),
        ("Informer-Post", 0.0121, 0.00, 0.0000),
        ("iTransformer", 0.0122, 16.26, 0.0054),
        ("PatchTST", 0.0230, 12.63, 0.0207),
        ("PhysFormer", 0.0128, 0.00, 0.0000),
    ]

    fig, ax = plt.subplots(figsize=(5.95, 4.55))
    bubble_scale = 42000
    offsets = {
        "Informer": (-8, -12),
        "iTransformer": (6, -10),
        "PINN": (6, -10),
        "GRU": (-18, 8),
        "LSTM": (10, 10),
        "PatchTST": (5, 10),
    }
    anchored_labels = {
        "Informer-Post": dict(x=0.01192, y=1.15, ha="left"),
        "PhysFormer": dict(x=0.01338, y=1.15, ha="left"),
    }

    for name, mse, bvr, mvs in data:
        if name == "PhysFormer":
            color = COLORS["phys"]
            edge = COLORS["dark"]
            alpha = 0.95
        elif name == "Informer-Post":
            color = "#fff5eb"
            edge = COLORS["informer"]
            alpha = 1.0
        elif name in {"Informer", "iTransformer", "PatchTST"}:
            color = COLORS["blue"]
            edge = "white"
            alpha = 0.78
        elif name == "PINN":
            color = COLORS["gold"]
            edge = "white"
            alpha = 0.85
        else:
            color = "#9fbfd3"
            edge = "white"
            alpha = 0.85

        size = max(mvs * bubble_scale, 85)
        ax.scatter(mse, bvr, s=size, color=color, edgecolors=edge, linewidth=1.2, alpha=alpha, zorder=4 if name == "PhysFormer" else 3)
        if name == "PhysFormer":
            ax.scatter(mse, bvr, s=size * 3.0, color=COLORS["phys"], alpha=0.10, zorder=2)
        if name in anchored_labels:
            spec = anchored_labels[name]
            ax.annotate(
                name,
                xy=(mse, bvr),
                xytext=(spec["x"], spec["y"]),
                textcoords="data",
                fontsize=8.8,
                fontweight="bold" if name == "PhysFormer" else "normal",
                bbox=NOTE_BBOX,
                arrowprops=dict(arrowstyle="-", color="#b0b7bf", linewidth=0.8),
                ha=spec["ha"],
                va="center",
            )
        else:
            dx, dy = offsets[name]
            ax.annotate(
                name,
                xy=(mse, bvr),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.9,
                fontweight="bold" if name == "PhysFormer" else "normal",
                ha="left",
                va="center",
            )

    ax.plot([0.0121, 0.0128], [19.53, 0.0], linestyle="--", color="#b0b7bf", linewidth=1.4)
    ax.set_xlim(0.0116, 0.0238)
    ax.set_ylim(-1.0, 20.8)
    ax.set_xlabel("Mean squared error (MSE)")
    ax.set_ylabel("Boundary violation rate (BVR %)")
    ax.grid(True)
    ax.text(
        0.60,
        0.08,
        "Bubble size is proportional to MVS.\nOutliers with MSE > 0.03 are omitted.",
        transform=ax.transAxes,
        fontsize=7.9,
        color=COLORS["gray"],
        va="bottom",
        ha="left",
        bbox=NOTE_BBOX,
    )

    save(fig, "EPSR_Fig5_Pareto")


def _transition_windows(true_seq: np.ndarray) -> tuple[int, int]:
    active = np.where(true_seq > 1e-4)[0]
    return int(active[0]), int(active[-1])


def generate_case_study() -> None:
    phys = np.load(PHYS_DIR / "pred.npy")[:, :, 1]
    true = np.load(PHYS_DIR / "true.npy")[:, :, 1]
    informer = np.load(DATA_ROOT / "Informer_vpp_dataset_3years_sl672_pl96_vpp" / "pred.npy")[:, :, 1]
    dlinear = np.load(DATA_ROOT / "DLinear_vpp_dataset_3years_sl672_pl96_vpp" / "pred.npy")[:, :, 1]

    sample_idx = 241
    true_seq = true[sample_idx]
    phys_seq = phys[sample_idx]
    inf_seq = informer[sample_idx]
    dlinear_seq = dlinear[sample_idx]
    dawn_idx, dusk_idx = _transition_windows(true_seq)

    t = np.arange(len(true_seq))
    x1 = max(dusk_idx - 6, 0)
    x2 = min(dusk_idx + 10, 95)
    y_upper = max(true_seq.max(), dlinear_seq.max()) * 1.12
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.0, 5.55),
        sharex=False,
        gridspec_kw={"height_ratios": [1.68, 1.0], "hspace": 0.24},
    )
    ax_top, ax_zoom = axes

    for ax in axes:
        ax.axhspan(-0.18, 0, color="#f1f1f1", alpha=1.0, zorder=0)
        ax.axhline(0, color=COLORS["dark"], linewidth=1.0, linestyle="--", zorder=1)
        ax.plot(t, true_seq, color="black", linewidth=2.2, label="Ground truth", zorder=4)
        ax.plot(t, phys_seq, color=COLORS["phys"], linewidth=2.1, label="PhysFormer", zorder=5)
        ax.plot(t, inf_seq, color=COLORS["informer"], linewidth=1.7, linestyle="--", label="Informer", zorder=3)
        ax.plot(t, dlinear_seq, color=COLORS["dlinear"], linewidth=1.7, linestyle="-.", label="DLinear", zorder=2)
        ax.grid(True)

    ax_top.axvspan(max(dawn_idx - 3, 0), min(dawn_idx + 4, 95), color=COLORS["gold_light"], alpha=0.75)
    ax_top.axvspan(max(dusk_idx - 3, 0), min(dusk_idx + 4, 95), color=COLORS["phys_light"], alpha=0.75)
    ax_top.set_xlim(0, 95)
    ax_top.set_ylim(-0.16, y_upper)
    ax_top.set_ylabel("PV power (MW)")
    ax_top.set_title("(a) Full PV forecast horizon with boundary-sensitive transition windows", pad=10)
    ax_top.annotate(
        "Dawn window",
        xy=(dawn_idx, y_upper * 0.82),
        xytext=(dawn_idx - 11, y_upper * 0.93),
        fontsize=8.3,
        color=COLORS["gold"],
        bbox=NOTE_BBOX,
        arrowprops=dict(arrowstyle="-", color=COLORS["gold"], linewidth=0.9),
    )
    ax_top.annotate(
        "Dusk window",
        xy=(dusk_idx, y_upper * 0.80),
        xytext=(dusk_idx + 5, y_upper * 0.93),
        fontsize=8.3,
        color=COLORS["phys"],
        bbox=NOTE_BBOX,
        arrowprops=dict(arrowstyle="-", color=COLORS["phys"], linewidth=0.9),
    )

    ax_zoom.axvspan(x1, x2, color=COLORS["phys_light"], alpha=0.42)
    ax_zoom.set_xlim(x1, x2)
    y_local = np.concatenate([true_seq[x1:x2], phys_seq[x1:x2], inf_seq[x1:x2], dlinear_seq[x1:x2]])
    ax_zoom.set_ylim(min(-0.06, y_local.min() - 0.02), max(0.17, y_local.max() + 0.035))
    ax_zoom.set_xlabel("Prediction horizon (15 min)")
    ax_zoom.set_ylabel("PV power (MW)")
    ax_zoom.set_title("(b) Zoomed dusk transition", pad=8)
    ax_zoom.set_xticks([x1, dusk_idx, x2])
    ax_zoom.annotate(
        "PhysFormer remains above the zero boundary\nthrough the shutdown transition",
        xy=(dusk_idx + 1, max(phys_seq[x1:x2]) * 0.55),
        xytext=(x1 + 0.7, ax_zoom.get_ylim()[1] * 0.78),
        fontsize=8.0,
        color=COLORS["phys"],
        bbox=NOTE_BBOX,
        arrowprops=dict(arrowstyle="->", color=COLORS["phys"], linewidth=0.9),
    )

    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.997),
        ncol=4,
        columnspacing=1.1,
        handlelength=1.9,
    )
    fig.subplots_adjust(top=0.91)

    save(fig, "EPSR_Fig6_PVCaseStudy")


def generate_extreme_weather() -> None:
    models = ["LSTM", "GRU", "PINN", "Informer", "iTransformer", "PatchTST", "PhysFormer"]
    mse = np.array([0.0192, 0.0182, 0.0177, 0.0146, 0.0143, 0.0261, 0.0150])
    bvr = np.array([13.43, 13.79, 13.29, 21.38, 19.79, 12.22, 0.00])
    colors = [COLORS["phys"] if m == "PhysFormer" else COLORS["blue"] for m in models]

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 5.15), sharey=True)
    y = np.arange(len(models))

    axes[0].barh(y, mse, color=colors, alpha=0.88)
    axes[0].set_xlabel("Extreme-weather MSE")
    axes[0].invert_yaxis()
    axes[0].grid(True, axis="x")

    axes[1].barh(y, bvr, color=colors, alpha=0.88)
    axes[1].set_xlabel("Extreme-weather BVR (%)")
    axes[1].grid(True, axis="x")

    for ax in axes:
        ax.set_yticks(y, models)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].text(0.0, 1.04, "(a) Extreme-weather MSE", transform=axes[0].transAxes, fontsize=9.5, fontweight="bold")
    axes[1].text(0.0, 1.04, "(b) Extreme-weather BVR", transform=axes[1].transAxes, fontsize=9.5, fontweight="bold")
    for yi, val in enumerate(mse):
        axes[0].text(val + 0.00028, yi, f"{val:.4f}", va="center", fontsize=8.0)
    for yi, val in enumerate(bvr):
        axes[1].text(val + 0.28, yi, f"{val:.2f}", va="center", fontsize=8.0, color=COLORS["phys"] if models[yi] == "PhysFormer" else COLORS["dark"])

    fig.tight_layout(h_pad=0.9)
    save(fig, "EPSR_Fig7_ExtremeWeather")


def main() -> None:
    set_style()
    generate_architecture()
    generate_mapping()
    generate_bpar_mechanism()
    generate_gate_alignment()
    generate_pareto()
    generate_case_study()
    generate_extreme_weather()
    print("Generated EPSR single-column figure set in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
