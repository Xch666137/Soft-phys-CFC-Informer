# Unified Thesis Runner

The thesis workflow now uses a single Python entrypoint:

```bash
python run.py <subcommand> ...
```

Supported subcommands:

- `build-dataset`
- `train`
- `test`
- `benchmark`
- `ablation`
- `export-forecast`
- `validate-powerflow`
- `pipeline`

Run artifacts are written to:

```text
runs/<run_name>/
```

Typical Linux remote usage:

```bash
bash scripts/train.sh --config configs/baselines/informer_net_injection.yaml --run-name informer_net_injection
bash scripts/pipeline.sh --config configs/baselines/informer_net_injection.yaml --mapping-csv templates/network_mapping.csv
```

Legacy entrypoints under `scripts/`, obsolete `tools/`, and old `analysis/` files are no longer part of the supported workflow.
