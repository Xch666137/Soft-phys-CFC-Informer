# AutoDL Automation

This thesis branch includes a first-pass AutoDL automation path built around:

- SSH key login
- `tmux` for long-running jobs
- local source archive upload by default
- local tar packaging + `scp` upload for `data_raw/`

It does not depend on AutoDL enterprise APIs. This follows the official
AutoDL guidance for SSH-based instance use and long-running terminal jobs:

- SSH access:
  [https://www.autodl.com/docs/ssh/](https://www.autodl.com/docs/ssh/)
- Daemon / long-running shell sessions:
  [https://api.autodl.com/docs/daemon/](https://api.autodl.com/docs/daemon/)
- Git usage reference:
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
- default source sync mode:
  `upload`
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
3. Make sure local source tree and local raw data are present.
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

If a previous submission already uploaded the source tree and `data_raw/`, and
you only need to restart the remote runner after fixing something like a
missing `tmux`, you can skip both uploads:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autodl_submit.ps1 `
  -RemoteHost "<AUTODL_HOST>" `
  -Port <AUTODL_PORT> `
  -SkipSourceUpload `
  -SkipDataUpload
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
  - runs `configs/drivers/benchmark_net_injection_5090.yaml`
- `benchmark_time`
  - runs `configs/drivers/benchmark_net_injection_time_generalization_5090.yaml`
- `audit_batch`
  - runs 2-epoch batch-size audit sweeps for PhysFormer / DLinear / TiDE / TimeXer / TFT
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

The AutoDL benchmark stages use 5090-specific driver configs with larger batch
sizes and loader overrides. Base thesis configs remain unchanged.

Downloaded files are written under:

```text
downloads/autodl/<host>_<timestamp>/
```

## Summary file naming

Benchmark summaries are written per driver, so main and time benchmarks do not
overwrite each other.

For AutoDL 5090 runs, the benchmark stages write:

- `runs/reports/benchmark_net_injection_5090_summary_raw.csv`
- `runs/reports/benchmark_net_injection_5090_summary_grouped.csv`
- `runs/reports/benchmark_net_injection_time_generalization_5090_summary_raw.csv`
- `runs/reports/benchmark_net_injection_time_generalization_5090_summary_grouped.csv`

For non-5090 or legacy local runs, the default drivers still write:

- `runs/reports/benchmark_net_injection_summary_raw.csv`
- `runs/reports/benchmark_net_injection_summary_grouped.csv`
- `runs/reports/benchmark_net_injection_time_generalization_summary_raw.csv`
- `runs/reports/benchmark_net_injection_time_generalization_summary_grouped.csv`

`autodl_fetch_results.ps1` now prefers the 5090 summary names and falls back to
the legacy names automatically.

## Failure handling

- Default source sync mode is local upload, so the remote host does not need to
  `git clone` the thesis branch.
- If you switch to `-SourceSyncMode clone`, the submit script aborts when the
  local branch is dirty or not pushed to `origin/<branch>`, unless you pass
  `-SkipGitSyncCheck`.
- If `data_raw/` has already been uploaded once, the submit script skips
  re-upload unless you pass `-ForceDataUpload`.
- If the remote project directory is already present and valid, you can skip
  source re-upload with `-SkipSourceUpload`.
- The submit script uses a local temporary tar archive plus `scp` instead of a
  raw `tar | ssh | tar` pipeline because Windows PowerShell native binary
  piping is less reliable for large archives.
- You can still opt into remote clone with:

```powershell
-SourceSyncMode clone -RepoUrl "https://github.com/<owner>/<repo>.git"
```
- If `tmux` is missing on the remote host, submission fails with a clear error
  before training starts.
- If `conda` is missing on the remote host, the remote script stops before any
  build or training stage.
- On AutoDL, `conda` may exist under `/root/miniconda3/` but not appear in
  `PATH` for non-interactive shells. The remote runner now probes common AutoDL
  locations and initializes conda explicitly before creating or activating the
  project environment.
