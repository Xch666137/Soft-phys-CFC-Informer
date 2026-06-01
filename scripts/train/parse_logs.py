"""Extract structured training diagnostics from PhysFormer train.log files.

Usage: python scripts/parse_train_logs.py runs/physformer_v5_4/train.log
Outputs JSON with per-epoch metrics, phase transitions, and gradient diagnostics.
"""
import json
import re
import sys
from pathlib import Path
from datetime import datetime


def parse_log(log_path):
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Filter out tqdm progress bars (lines with carriage return or progress bar pattern)
    clean_lines = []
    for line in lines:
        # Skip tqdm progress lines
        if "\r" in line or "it/s]" in line or "s/it]" in line:
            continue
        # Skip pure progress bar chars
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Epoch") and "/" in stripped and ("|" in stripped):
            continue
        clean_lines.append(stripped)

    epochs = []
    gradient_diag = []
    phase_changes = []
    early_stop_events = []
    optimizer_info = {}

    epoch_start_re = re.compile(r"Epoch (\d+)/(\d+) \| Val Loss: ([\d.]+) \| Val Net MSE: ([\d.e+\-]+)")
    grad_cos_re = re.compile(r"cos_sim[\"']?\s*:\s*([\d.\-e+]+).*?angle_deg[\"']?\s*:\s*([\d.\-e+]+).*?norm_net[\"']?\s*:\s*([\d.\-e+]+).*?norm_theory[\"']?\s*:\s*([\d.\-e+]+)")
    grad_log_re = re.compile(r"gradient.*angle|cos_sim|angle_deg")
    phase_re = re.compile(r"Phase (\d).*?:.*?cw=([\d.\-e+]+).*?tw=([\d.\-e+]+).*?rr=([\d.\-e+]+)")
    early_re = re.compile(r"Early stopping|counter (\d+).*?best.*?([\d.\-e+]+)")

    for line in clean_lines:
        m = epoch_start_re.search(line)
        if m:
            epoch = int(m.group(1))
            total = int(m.group(2))
            val_loss = float(m.group(3))
            val_net_mse = float(m.group(4))
            epochs.append({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_net_mse": val_net_mse,
            })
            continue

        # Catch phase transitions
        m = phase_re.search(line)
        if m:
            phase_changes.append({
                "phase": int(m.group(1)),
                "cw": float(m.group(2)),
                "tw": float(m.group(3)),
                "rr": float(m.group(4)),
                "raw": line,
            })
            continue

        # Catch gradient diagnostics (relaxed patterns)
        if "cos_sim" in line.lower() or "angle_deg" in line.lower() or "gradient angle" in line.lower():
            # Try extracting numeric values
            cos_match = re.search(r"cos_sim[\"':\s]+([\d.\-e+]+)", line)
            angle_match = re.search(r"angle_deg[\"':\s]+([\d.\-e+]+)", line)
            norm_net_match = re.search(r"norm_net[\"':\s]+([\d.\-e+]+)", line)
            norm_theory_match = re.search(r"norm_theory[\"':\s]+([\d.\-e+]+)", line)
            grad_info = {"raw": line}
            if cos_match:
                grad_info["cos_sim"] = float(cos_match.group(1))
            if angle_match:
                grad_info["angle_deg"] = float(angle_match.group(1))
            if norm_net_match:
                grad_info["norm_net"] = float(norm_net_match.group(1))
            if norm_theory_match:
                grad_info["norm_theory"] = float(norm_theory_match.group(1))
            gradient_diag.append(grad_info)
            continue

        # Catch early stopping
        if "EarlyStopping" in line or "Early stopping" in line:
            counter_match = re.search(r"counter (\d+).*?best.*?([\d.\-e+]+)", line)
            if counter_match:
                early_stop_events.append({
                    "counter": int(counter_match.group(1)),
                    "best_net_mse": float(counter_match.group(2)),
                    "raw": line,
                })
            else:
                early_stop_events.append({"raw": line})
            continue

        # Catch optimizer / LR info
        if "Optimizer" in line or "lr=" in line or "warmup" in line:
            optimizer_info["raw"] = line

    # Also try to find per-batch debug output (loss term decomposition)
    debug_terms = []
    for line in clean_lines:
        if "net_mse" in line.lower() and ("theory_mse" in line.lower() or "res_reg" in line.lower()):
            terms = {}
            for key in ["net_mse", "theory_mse", "res_reg", "soc_loss", "component_loss",
                        "battery_power_mae", "net_mae"]:
                m = re.search(rf"{key}[\"':\s]+([\d.\-e+]+)", line)
                if m:
                    terms[key] = float(m.group(1))
            if terms:
                debug_terms.append(terms)

    # Find best validation metrics (from early stopping or explicit output)
    best_net_mse = None
    for e in early_stop_events:
        if "best_net_mse" in e:
            best_net_mse = e["best_net_mse"]

    # From epochs data, find actual best epoch
    best_epoch = None
    best_val = float("inf")
    for ep in epochs:
        if ep["val_net_mse"] < best_val:
            best_val = ep["val_net_mse"]
            best_epoch = ep["epoch"]

    return {
        "log_path": str(log_path),
        "total_epochs_logged": len(epochs),
        "epochs": epochs,
        "best_val_net_mse": best_val if best_epoch else None,
        "best_epoch": best_epoch,
        "best_from_early_stop": best_net_mse,
        "phase_changes": phase_changes,
        "gradient_diagnostics": gradient_diag,
        "early_stop_events": early_stop_events,
        "debug_terms": debug_terms,
        "optimizer_info": optimizer_info,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_train_logs.py <train.log> [<train.log> ...]")
        sys.exit(1)

    for log_path in sys.argv[1:]:
        result = parse_log(log_path)
        # Print JSON to stdout
        print(f"=== {Path(log_path).parent.name} ===")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        print()


if __name__ == "__main__":
    main()
