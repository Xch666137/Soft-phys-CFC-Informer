# AutoDL Automation

This thesis branch includes a first-pass AutoDL automation path built around:

- SSH key login
- `tmux` for long-running jobs
- remote `git clone` over HTTPS
- local tar packaging + `scp` upload for `data_raw/`

It does not depend on AutoDL enterprise APIs. This follows the official
AutoDL guidance for SSH-based instance use and long-running terminal jobs:

- SSH access:
  [https://www.autodl.com/docs/ssh/](https://www.autodl.com/docs/ssh/)
- Daemon / long-running shell sessions:
  [https://api.autodl.com/docs/daemon/](https://api.autodl.com/docs/daemon/)
- Git usage:
  [https://www.autodl.com/docs/git/](https://www.autodl.com/docs/git/)
- File transfer reference:
  [https://api.autodl.com/docs/scp/](https://api.autodl.com/docs/scp/)

## Defaults

The scripts assume:

- remote user: `root`
- remote project dir:
  `/root/autodl-tmp/Soft-phys-CFC-Informer`
- remote conda env:
  `Soft-phys-CFC-Informer`
- default branch:
  `codex/thesis-mainline`
- default stages:
  `verify,build_dataset,benchmark_main,benchmark_time`

## Scripts

Local submit:

- [autodl_submit.ps1](/C:/Users/Xch/.codex/worktrees/7c57/Soft-phys-CFC-Informer/scripts/autodl_submit.ps1)

Remote execution:

- [autodl_remote_run.sh](/C:/Users/Xch/.codex/worktrees/7c57/Soft-phys-CFC-Informer/scripts/autodl_remote_run.sh)

Local fetch:

- [autodl_fetch_results.ps1](/C:/Users/Xch/.codex/worktrees/7c57/Soft-phys-CFC-Informer/scripts/autodl_fetch_results.ps1)

## First run

1. Make sure local SSH key login to the AutoDL instance already works.
2. Make sure the GitHub HTTPS repo URL is correct.
3. Make sure local raw data is present under `data_raw/`.
4. Start the remote chain with a dry-run first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autodl_submit.ps1 `
  -RemoteHost "<AUTODL_HOST>" `
  -Port <AUTODL_PORT> `
  -DryRun
```

5. Submit the real remote job:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autodl_submit.ps1 `
  -RemoteHost "<AUTODL_HOST>" `
  -Port <AUTODL_PORT>
```

6. Attach to the remote session if needed:

```bash
ssh -p <AUTODL_PORT> root@<AUTODL_HOST> -t "tmux attach -t autodl-thesis"
```

## Stage meanings

- `verify`
  - imports and config parsing only
- `build_dataset`
  - builds `data_processed/multi_portfolio`
- `benchmark_main`
  - runs `configs/drivers/benchmark_net_injection.yaml`
- `benchmark_time`
  - runs `configs/drivers/benchmark_net_injection_time_generalization.yaml`
- `ablation`
  - runs `configs/drivers/physformer_ablation.yaml`
- `appendix`
  - runs appendix benchmark drivers
- `validate_powerflow`
  - requires explicit validation config, run name, and mapping CSV

## Result collection

Fetch the benchmark summaries and the default best runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autodl_fetch_results.ps1 `
  -RemoteHost "<AUTODL_HOST>" `
  -Port <AUTODL_PORT>
```

The fetch script downloads:

- main benchmark raw/grouped summaries
- time benchmark raw/grouped summaries
- the median-seed PhysFormer run for each benchmark
- the median-seed strongest baseline for each benchmark

Downloaded files are written under:

```text
downloads/autodl/<host>_<timestamp>/
```

## Summary file naming

Benchmark summaries are now written per driver, so main and time benchmarks do
not overwrite each other:

- `runs/reports/benchmark_net_injection_summary_raw.csv`
- `runs/reports/benchmark_net_injection_summary_grouped.csv`
- `runs/reports/benchmark_net_injection_time_generalization_summary_raw.csv`
- `runs/reports/benchmark_net_injection_time_generalization_summary_grouped.csv`

## Failure handling

- If the local branch is dirty or not pushed to `origin/<branch>`, the submit
  script aborts by default. Use `-SkipGitSyncCheck` only if you know the remote
  clone can safely diverge.
- If `data_raw/` has already been uploaded once, the submit script skips
  re-upload unless you pass `-ForceDataUpload`.
- The submit script uses a local temporary tar archive plus `scp` instead of a
  raw `tar | ssh | tar` pipeline because Windows PowerShell native binary
  piping is less reliable for large archives.
- If `tmux` is missing on the remote host, submission fails with a clear error
  before training starts.
- If `conda` is missing on the remote host, the remote script stops before any
  build or training stage.
