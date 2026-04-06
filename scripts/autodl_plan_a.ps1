param(
    [Parameter(Mandatory = $true)]
    [Alias("Host")]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [int]$Port,

    [string]$StageARunName = "physformer_net_injection__s2024",
    [string]$StageAConfig = "configs/physformer_default.yaml",
    [string]$StageBRunName = "physformer_operational_fit_s2024",
    [string]$StageBConfig = "configs/physformer_operational_fit.yaml",
    [string]$SessionName = "autodl-thesis",
    [switch]$SkipSourceUpload,
    [switch]$SkipDataUpload,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$submitScript = Join-Path $scriptDir "autodl_submit.ps1"

$remoteArgs = @(
    "--stage-a-config $StageAConfig"
    "--stage-a-run-name $StageARunName"
    "--operational-config $StageBConfig"
    "--operational-run-name $StageBRunName"
) -join " "

$params = @{
    RemoteHost = $RemoteHost
    Port = $Port
    Stages = "verify,build_dataset,stage_a_single,operational_fit,export_operational"
    RemoteArgs = $remoteArgs
    SessionName = $SessionName
}

if ($SkipSourceUpload) {
    $params["SkipSourceUpload"] = $true
}
if ($SkipDataUpload) {
    $params["SkipDataUpload"] = $true
}
if ($DryRun) {
    $params["DryRun"] = $true
}

& $submitScript @params
