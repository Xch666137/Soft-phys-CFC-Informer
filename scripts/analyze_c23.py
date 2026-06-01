"""
C23 analysis: C-2 (soft transition) + C-3 (3-stage cw) + e3+detach effects.
Compares against old p1a baselines (hard reset + 2-stage cw).
"""
import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

results_dir = r"C:\Users\Xch\.codex\worktrees\7c57\Physformer\results"

experiments = {
    # Old baselines (pulled in V-3)
    "physformer_p1a_baseline_s2025": "Hard + 2-stage, e2, no detach",
    "physformer_p1a_detach_s2025": "Hard + 2-stage, e2, detach",
    # New C23
    "physformer_c23_baseline": "Soft + 3-stage, e2, no detach",
    "physformer_c23_detach": "Soft + 3-stage, e2, detach",
    "physformer_c23_e3": "Soft + 3-stage, e3, no detach",
}

def get_metrics(name):
    path = os.path.join(results_dir, name, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def get_val_mse(name):
    log_path = os.path.join(results_dir, name, "train.log")
    if not os.path.exists(log_path):
        return None
    best = None
    with open(log_path, errors='ignore') as f:
        for line in f:
            if "Validation Net MSE" in line and "best:" in line:
                parts = line.split("best:")
                if len(parts) > 1:
                    best = float(parts[1].strip().rstrip(")"))
    return best

print("=" * 90)
print("C23 Analysis: C-2 (soft transition) + C-3 (3-stage cw) + e3+detach")
print("=" * 90)

print(f"\n{'Experiment':<30} {'Description':<35} {'MAE':>10} {'MSE':>12} {'Theory':>10} {'Ramp%':>8} {'ValMSE':>10}")
print("-" * 115)

rows = []
for name, desc in experiments.items():
    m = get_metrics(name)
    v = get_val_mse(name)
    if m is None:
        print(f"{name:<30} {'MISSING':<35}")
        continue
    rows.append({
        "name": name, "desc": desc,
        "mae": m["mae"], "mse": m["mse"], "theory": m.get("theory_mae", 0),
        "ramp": m.get("ramp_violation_pct", 0) * 100 if m.get("ramp_violation_pct", 0) < 1 else m.get("ramp_violation_pct", 0),
        "val": v,
    })

for r in rows:
    val_str = f"{r['val']:.4f}" if r['val'] is not None else "N/A"
    print(f"{r['name']:<30} {r['desc']:<35} {r['mae']:.6f} {r['mse']:>10.3e} {r['theory']:>10.6f} {r['ramp']:>8.4f} {val_str:>10}")

# --- Effect isolation ---
print("\n" + "=" * 90)
print("EFFECT ISOLATION")
print("=" * 90)

old_b = next((r for r in rows if r["name"] == "physformer_p1a_baseline_s2025"), None)
new_b = next((r for r in rows if r["name"] == "physformer_c23_baseline"), None)
old_d = next((r for r in rows if r["name"] == "physformer_p1a_detach_s2025"), None)
new_d = next((r for r in rows if r["name"] == "physformer_c23_detach"), None)
new_e3 = next((r for r in rows if r["name"] == "physformer_c23_e3"), None)

if old_b and new_b:
    print("\n--- C-2+C-3 effect (baseline, no detach) ---")
    d_mae = (new_b["mae"] - old_b["mae"]) / old_b["mae"] * 100
    d_theory = (new_b["theory"] - old_b["theory"]) / old_b["theory"] * 100
    d_ramp = new_b["ramp"] - old_b["ramp"]
    print(f"  MAE:     {old_b['mae']:.6f} -> {new_b['mae']:.6f}  ({d_mae:+.1f}%)")
    print(f"  Theory:  {old_b['theory']:.6f} -> {new_b['theory']:.6f}  ({d_theory:+.1f}%)")
    print(f"  Ramp:    {old_b['ramp']:.4f} -> {new_b['ramp']:.4f}    ({d_ramp:+.4f}pp)")
    print(f"  Val MSE: {old_b['val']} -> {new_b['val']:.4f}")

if old_d and new_d:
    print("\n--- C-2+C-3 effect (detach) ---")
    d_mae = (new_d["mae"] - old_d["mae"]) / old_d["mae"] * 100
    d_theory = (new_d["theory"] - old_d["theory"]) / old_d["theory"] * 100
    print(f"  MAE:     {old_d['mae']:.6f} -> {new_d['mae']:.6f}  ({d_mae:+.1f}%)")
    print(f"  Theory:  {old_d['theory']:.6f} -> {new_d['theory']:.6f}  ({d_theory:+.1f}%)")
    print(f"  Ramp:    {old_d['ramp']:.4f} -> {new_d['ramp']:.4f}")
    print(f"  Val MSE: {old_d['val']:.4f} -> {new_d['val']:.4f}")

if new_b and new_d:
    print("\n--- detach effect (under C-2+C-3) ---")
    d_mae = (new_d["mae"] - new_b["mae"]) / new_b["mae"] * 100
    d_theory = (new_d["theory"] - new_b["theory"]) / new_b["theory"] * 100
    print(f"  MAE:     {new_b['mae']:.6f} -> {new_d['mae']:.6f}  ({d_mae:+.1f}%)")
    print(f"  Theory:  {new_b['theory']:.6f} -> {new_d['theory']:.6f}  ({d_theory:+.1f}%)")
    print(f"  Ramp:    {new_b['ramp']:.4f} -> {new_d['ramp']:.4f}")

if new_b and new_e3:
    print("\n--- e3 effect (under C-2+C-3, no detach) ---")
    d_mae = (new_e3["mae"] - new_b["mae"]) / new_b["mae"] * 100
    d_theory = (new_e3["theory"] - new_b["theory"]) / new_b["theory"] * 100
    print(f"  MAE:     {new_b['mae']:.6f} -> {new_e3['mae']:.6f}  ({d_mae:+.1f}%)")
    print(f"  Theory:  {new_b['theory']:.6f} -> {new_e3['theory']:.6f}  ({d_theory:+.1f}%)")
    print(f"  Ramp:    {new_b['ramp']:.4f} -> {new_e3['ramp']:.4f}")

# --- Component metrics ---
print("\n--- Component MAE comparison ---")
comp_names = ["component_load_mae", "component_pv_mae", "component_wind_mae",
              "component_battery_power_mae", "component_battery_soc_mae"]
comp_short = ["Load", "PV", "Wind", "BattP", "BattSOC"]

for r in rows:
    m = get_metrics(r["name"])
    comps = {cn: m.get(cn, 0) for cn in comp_names}
    vals = " ".join([f"{comps[cn]:.5f}" for cn in comp_names])
    print(f"  {r['name']:<35s} {vals}")

print("\nDone.")
