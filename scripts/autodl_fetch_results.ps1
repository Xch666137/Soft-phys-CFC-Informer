param(
    [Parameter(Mandatory = $true)]
    [Alias("Host")]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [int]$Port,

    [string]$RemoteUser = "root",
    [string]$RemoteProjectDir = "/root/autodl-tmp/Soft-phys-CFC-Informer",
    [string]$LocalOutputDir = "downloads/autodl",
    [switch]$IncludeAppendix,
    [switch]$DryRun,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdditionalRunNames = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host "[autodl-fetch] $Message"
}

function Ensure-Dir {
    param([string]$PathValue)
    if (-not (Test-Path $PathValue)) {
        New-Item -ItemType Directory -Path $PathValue -Force | Out-Null
    }
}

function Copy-RemoteFile {
    param(
        [string]$RemotePath,
        [string]$LocalPath
    )

    if ($DryRun) {
        Write-Step "DRY-RUN copy $RemotePath -> $LocalPath"
        return $false
    }

    $remoteTarget = "${RemoteUser}@${RemoteHost}:$RemotePath"
    Ensure-Dir (Split-Path -Parent $LocalPath)
    & scp -P $Port $remoteTarget $LocalPath
    if ($LASTEXITCODE -ne 0) {
        Write-Step "Remote file not copied: $RemotePath"
        return $false
    }
    return $true
}

function Resolve-RemoteSummarySpec {
    param(
        [string]$PreferredPath,
        [string]$FallbackPath
    )

    if ($DryRun) {
        return @{
            Remote = $PreferredPath
            Label = [System.IO.Path]::GetFileName($PreferredPath)
        }
    }

    $checkPreferred = "test -f '$PreferredPath'"
    & ssh -p $Port "${RemoteUser}@${RemoteHost}" $checkPreferred | Out-Null
    if ($LASTEXITCODE -eq 0) {
        return @{
            Remote = $PreferredPath
            Label = [System.IO.Path]::GetFileName($PreferredPath)
        }
    }

    $checkFallback = "test -f '$FallbackPath'"
    & ssh -p $Port "${RemoteUser}@${RemoteHost}" $checkFallback | Out-Null
    if ($LASTEXITCODE -eq 0) {
        return @{
            Remote = $FallbackPath
            Label = [System.IO.Path]::GetFileName($FallbackPath)
        }
    }

    return $null
}

function Get-MedianSeedRun {
    param(
        [array]$Rows,
        [string]$ExperimentName
    )

    $filtered = @($Rows | Where-Object { $_.experiment_name -eq $ExperimentName -and $_.mse -ne "" })
    if (-not $filtered -or $filtered.Count -eq 0) {
        return $null
    }

    $ordered = $filtered | Sort-Object @{ Expression = { [double]$_.mse } }, @{ Expression = { [int]$_.seed } }
    return $ordered[[Math]::Floor($ordered.Count / 2)]
}

function Get-BestBaselineExperiment {
    param(
        [array]$Rows,
        [string]$PhysformerName
    )

    $filtered = @($Rows | Where-Object { $_.experiment_name -ne $PhysformerName -and $_.mse_mean -ne "" })
    if (-not $filtered -or $filtered.Count -eq 0) {
        return $null
    }

    return ($filtered | Sort-Object @{ Expression = { [double]$_.mse_mean } } | Select-Object -First 1).experiment_name
}

$sshTarget = "$RemoteUser@$RemoteHost"
$sanitizedHost = ($RemoteHost -replace "[:/\\]", "_")
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$localRoot = Join-Path $LocalOutputDir "${sanitizedHost}_$timestamp"
$localSummaryDir = Join-Path $localRoot "reports"
$localRunsDir = Join-Path $localRoot "runs"

Ensure-Dir $localSummaryDir
Ensure-Dir $localRunsDir

$summarySpecs = @(
    @{
        Name = "benchmark_net_injection"
        RawPreferred = "$RemoteProjectDir/runs/reports/benchmark_net_injection_5090_summary_raw.csv"
        RawFallback = "$RemoteProjectDir/runs/reports/benchmark_net_injection_summary_raw.csv"
        GroupedPreferred = "$RemoteProjectDir/runs/reports/benchmark_net_injection_5090_summary_grouped.csv"
        GroupedFallback = "$RemoteProjectDir/runs/reports/benchmark_net_injection_summary_grouped.csv"
        PhysFormer = "physformer_net_injection"
    },
    @{
        Name = "benchmark_net_injection_time_generalization"
        RawPreferred = "$RemoteProjectDir/runs/reports/benchmark_net_injection_time_generalization_5090_summary_raw.csv"
        RawFallback = "$RemoteProjectDir/runs/reports/benchmark_net_injection_time_generalization_summary_raw.csv"
        GroupedPreferred = "$RemoteProjectDir/runs/reports/benchmark_net_injection_time_generalization_5090_summary_grouped.csv"
        GroupedFallback = "$RemoteProjectDir/runs/reports/benchmark_net_injection_time_generalization_summary_grouped.csv"
        PhysFormer = "physformer_net_injection_time_generalization"
    }
)

if ($IncludeAppendix) {
    $summarySpecs += @(
        @{
            Name = "benchmark_net_injection_appendix"
            Raw = "$RemoteProjectDir/runs/reports/benchmark_net_injection_appendix_summary_raw.csv"
            Grouped = "$RemoteProjectDir/runs/reports/benchmark_net_injection_appendix_summary_grouped.csv"
            PhysFormer = ""
        },
        @{
            Name = "benchmark_net_injection_appendix_time_generalization"
            Raw = "$RemoteProjectDir/runs/reports/benchmark_net_injection_appendix_time_generalization_summary_raw.csv"
            Grouped = "$RemoteProjectDir/runs/reports/benchmark_net_injection_appendix_time_generalization_summary_grouped.csv"
            PhysFormer = ""
        }
    )
}

$manifest = [ordered]@{
    host = $RemoteHost
    port = $Port
    remote_project_dir = $RemoteProjectDir
    local_output_dir = $localRoot
    fetched_reports = @()
    selected_runs = @()
}

$normalizedAdditionalRunNames = @()
foreach ($rawName in $AdditionalRunNames) {
    if ([string]::IsNullOrWhiteSpace($rawName)) {
        continue
    }
    foreach ($piece in ($rawName -split ",")) {
        $name = $piece.Trim()
        if (-not [string]::IsNullOrWhiteSpace($name)) {
            $normalizedAdditionalRunNames += $name
        }
    }
}

foreach ($spec in $summarySpecs) {
    $rawResolved = Resolve-RemoteSummarySpec -PreferredPath $spec.RawPreferred -FallbackPath $spec.RawFallback
    $groupedResolved = Resolve-RemoteSummarySpec -PreferredPath $spec.GroupedPreferred -FallbackPath $spec.GroupedFallback
    if ($null -eq $rawResolved -or $null -eq $groupedResolved) {
        Write-Step "Skipping $($spec.Name): summary files not found on remote host."
        continue
    }

    $rawLocal = Join-Path $localSummaryDir $rawResolved.Label
    $groupedLocal = Join-Path $localSummaryDir $groupedResolved.Label

    $rawOk = Copy-RemoteFile -RemotePath $rawResolved.Remote -LocalPath $rawLocal
    $groupedOk = Copy-RemoteFile -RemotePath $groupedResolved.Remote -LocalPath $groupedLocal

    if (-not ($rawOk -and $groupedOk)) {
        continue
    }

    $manifest.fetched_reports += @{
        name = $spec.Name
        raw = $rawLocal
        grouped = $groupedLocal
    }

    if ($DryRun -or [string]::IsNullOrWhiteSpace($spec.PhysFormer)) {
        continue
    }

    $rawRows = @(Import-Csv $rawLocal)
    $groupedRows = @(Import-Csv $groupedLocal)
    if ($rawRows.Count -eq 0 -or $groupedRows.Count -eq 0) {
        continue
    }

    $physformerMedian = Get-MedianSeedRun -Rows $rawRows -ExperimentName $spec.PhysFormer
    if ($null -ne $physformerMedian) {
        $manifest.selected_runs += @{
            benchmark = $spec.Name
            role = "physformer_median_seed"
            experiment_name = $physformerMedian.experiment_name
            run_name = $physformerMedian.run_name
        }
    }

    $bestBaseline = Get-BestBaselineExperiment -Rows $groupedRows -PhysformerName $spec.PhysFormer
    if ($null -ne $bestBaseline) {
        $baselineMedian = Get-MedianSeedRun -Rows $rawRows -ExperimentName $bestBaseline
        if ($null -ne $baselineMedian) {
            $manifest.selected_runs += @{
                benchmark = $spec.Name
                role = "best_baseline_median_seed"
                experiment_name = $baselineMedian.experiment_name
                run_name = $baselineMedian.run_name
            }
        }
    }
}

foreach ($selection in $manifest.selected_runs) {
    $runName = $selection.run_name
    $remoteRunDir = "$RemoteProjectDir/runs/$runName"
    $localRunDir = Join-Path $localRunsDir $runName
    Ensure-Dir $localRunDir

    Copy-RemoteFile -RemotePath "$remoteRunDir/metrics.json" -LocalPath (Join-Path $localRunDir "metrics.json") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/config_merged.yaml" -LocalPath (Join-Path $localRunDir "config_merged.yaml") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/train.log" -LocalPath (Join-Path $localRunDir "train.log") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/training_state.pth" -LocalPath (Join-Path $localRunDir "training_state.pth") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/exports/portfolio_forecasts.csv" -LocalPath (Join-Path $localRunDir "portfolio_forecasts.csv") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/exports/portfolio_forecasts_operational.csv" -LocalPath (Join-Path $localRunDir "portfolio_forecasts_operational.csv") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/powerflow/powerflow_summary.json" -LocalPath (Join-Path $localRunDir "powerflow_summary.json") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/extras/diagnostic_summary.json" -LocalPath (Join-Path $localRunDir "diagnostic_summary.json") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/extras/component_confidence.npz" -LocalPath (Join-Path $localRunDir "component_confidence.npz") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/extras/component_attribution.npz" -LocalPath (Join-Path $localRunDir "component_attribution.npz") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/extras/battery_state_preds.npz" -LocalPath (Join-Path $localRunDir "battery_state_preds.npz") | Out-Null
}

foreach ($runName in $normalizedAdditionalRunNames) {
    if ([string]::IsNullOrWhiteSpace($runName)) {
        continue
    }
    $remoteRunDir = "$RemoteProjectDir/runs/$runName"
    $localRunDir = Join-Path $localRunsDir $runName
    Ensure-Dir $localRunDir

    $manifest.selected_runs += @{
        benchmark = "manual"
        role = "additional_run"
        experiment_name = $runName
        run_name = $runName
    }

    Copy-RemoteFile -RemotePath "$remoteRunDir/metrics.json" -LocalPath (Join-Path $localRunDir "metrics.json") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/config_merged.yaml" -LocalPath (Join-Path $localRunDir "config_merged.yaml") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/train.log" -LocalPath (Join-Path $localRunDir "train.log") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/training_state.pth" -LocalPath (Join-Path $localRunDir "training_state.pth") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/exports/portfolio_forecasts.csv" -LocalPath (Join-Path $localRunDir "portfolio_forecasts.csv") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/exports/portfolio_forecasts_operational.csv" -LocalPath (Join-Path $localRunDir "portfolio_forecasts_operational.csv") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/extras/diagnostic_summary.json" -LocalPath (Join-Path $localRunDir "diagnostic_summary.json") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/extras/component_confidence.npz" -LocalPath (Join-Path $localRunDir "component_confidence.npz") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/extras/component_attribution.npz" -LocalPath (Join-Path $localRunDir "component_attribution.npz") | Out-Null
    Copy-RemoteFile -RemotePath "$remoteRunDir/extras/battery_state_preds.npz" -LocalPath (Join-Path $localRunDir "battery_state_preds.npz") | Out-Null
}

$manifestPath = Join-Path $localRoot "fetch_manifest.json"
if ($DryRun) {
    Write-Step "DRY-RUN manifest would be written to $manifestPath"
} else {
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestPath -Encoding UTF8
    Write-Step "Saved fetch manifest to $manifestPath"
}

Write-Step "Result fetch complete."
