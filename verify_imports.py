"""Import verification script for the refactored physformer package (CPU-only, no GPU needed)."""
import importlib
import sys

modules = [
    "physformer",
    "physformer.models",
    "physformer.models.physformer",
    "physformer.models.physical_layer",
    "physformer.models.causal_coupling",
    "physformer.models.cfc",
    "physformer.models.flatten_head",
    "physformer.models.informer",
    "physformer.models.autoformer",
    "physformer.models.lstm",
    "physformer.models.gru",
    "physformer.models.pinn",
    "physformer.models.dlinear",
    "physformer.models.patchtst",
    "physformer.models.itransformer",
    "physformer.layers",
    "physformer.layers.attention",
    "physformer.layers.embedding",
    "physformer.layers.encoder",
    "physformer.layers.decoder",
    "physformer.layers.positional",
    "physformer.layers.distillation",
    "physformer.layers.mask",
    "physformer.layers.revin",
    "physformer.data.data_factory",
    "physformer.exp.exp_physformer",
    "physformer.exp.exp_baseline",
    "physformer.utils.losses",
    "physformer.utils.metrics",
    "physformer.utils.tools",
]

failed = []
for mod in modules:
    try:
        importlib.import_module(mod)
        print(f"  OK  {mod}")
    except Exception as e:
        print(f"  FAIL {mod}: {e}")
        failed.append(mod)

print(f"\n{'=' * 50}")
if failed:
    print(f"FAILED: {len(failed)} modules failed to import")
    for m in failed:
        print(f"  - {m}")
    sys.exit(1)
else:
    print(f"ALL {len(modules)} modules imported successfully")
    sys.exit(0)
