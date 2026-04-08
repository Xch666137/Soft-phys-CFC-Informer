# AutoDL Automation

This branch uses an SSH + `tmux` AutoDL flow. It does not depend on AutoDL enterprise APIs.

Primary scripts:

- local submit:
  [autodl_submit.ps1](/C:/Users/Xch/.codex/worktrees/7c57/Soft-phys-CFC-Informer/scripts/autodl_submit.ps1)
- remote runner:
  [autodl_remote_run.sh](/C:/Users/Xch/.codex/worktrees/7c57/Soft-phys-CFC-Informer/scripts/autodl_remote_run.sh)
- local fetch:
  [autodl_fetch_results.ps1](/C:/Users/Xch/.codex/worktrees/7c57/Soft-phys-CFC-Informer/scripts/autodl_fetch_results.ps1)
- local watch:
  [autodl_watch.ps1](/C:/Users/Xch/.codex/worktrees/7c57/Soft-phys-CFC-Informer/scripts/autodl_watch.ps1)

## Defaults

- remote user:
  `root`
- remote project dir:
  `/root/autodl-tmp/Soft-phys-CFC-Informer`
- remote conda env:
  `Soft-phys-CFC-Informer`
- default source sync:
  `upload`
- default stages:
  `verify,build_dataset,benchmark_main,benchmark_time`

## Submit

Default benchmark submission:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autodl_submit.ps1 `
  -RemoteHost "<AUTODL_HOST>" `
  -Port <AUTODL_PORT>
```

Restart without re-uploading source or data:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autodl_submit.ps1 `
  -RemoteHost "<AUTODL_HOST>" `
  -Port <AUTODL_PORT> `
  -SkipSourceUpload `
  -SkipDataUpload
```

Run the full Stage A + Stage B preset:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autodl_submit.ps1 `
  -RemoteHost "<AUTODL_HOST>" `
  -Port <AUTODL_PORT> `
  -Preset plan-a `
  -SkipDataUpload
```

This preset runs:

- `verify`
- `build_dataset`
- `stage_a_single`
- `operational_fit`
- `export_operational`

Default run names:

- Stage A:
  `physformer_net_injection__s2024`
- Stage B:
  `physformer_operational_fit_s2024`

## Stages

- `verify`
  - import and config parsing only
- `build_dataset`
  - builds `data_processed/multi_portfolio`
- `benchmark_main`
  - runs `configs/drivers/benchmark_net_injection.yaml`
- `benchmark_time`
  - runs `configs/drivers/benchmark_net_injection_time_generalization.yaml`
- `stage_a_single`
  - runs one Stage A PhysFormer train + test
- `operational_fit`
  - runs Stage B from a Stage A checkpoint
- `export_operational`
  - exports `portfolio_forecasts_operational.csv`
- `ablation`
  - runs `configs/drivers/physformer_ablation.yaml`
- `appendix`
  - runs legacy appendix drivers under `configs/legacy/drivers/`

## Monitor

Watch the current master log:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autodl_watch.ps1 `
  -RemoteHost "<AUTODL_HOST>" `
  -Port <AUTODL_PORT> `
  -Mode Master
```

Watch one run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autodl_watch.ps1 `
  -RemoteHost "<AUTODL_HOST>" `
  -Port <AUTODL_PORT> `
  -Mode Run `
  -RunName "physformer_net_injection__s2024"
```

## Fetch

Fetch summaries and selected runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autodl_fetch_results.ps1 `
  -RemoteHost "<AUTODL_HOST>" `
  -Port <AUTODL_PORT>
```

Fetch additional runs explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autodl_fetch_results.ps1 `
  -RemoteHost "<AUTODL_HOST>" `
  -Port <AUTODL_PORT> `
  -AdditionalRunNames "physformer_net_injection__s2024,physformer_operational_fit_s2024"
```

Fetched artifacts include, when present:

- `metrics.json`
- `config_merged.yaml`
- `train.log`
- `training_state.pth`
- `portfolio_forecasts.csv`
- `portfolio_forecasts_operational.csv`
- `diagnostic_summary.json`
- `component_confidence.npz`
- `component_attribution.npz`
- `battery_state_preds.npz`

Outputs are written under:

```text
downloads/autodl/<host>_<timestamp>/
```
