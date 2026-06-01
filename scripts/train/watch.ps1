param(
    [Parameter(Mandatory = $true)]
    [Alias("Host")]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [int]$Port,

    [string]$RemoteUser = "root",
    [string]$RemoteProjectDir = "/root/autodl-tmp/Soft-phys-CFC-Informer",

    [ValidateSet("Auto", "Master", "Run")]
    [string]$Mode = "Auto",

    [string]$RunName = "",
    [int]$Lines = 120
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-RemoteTail {
    param([string]$RemoteCommand)
    & ssh "-p" "$Port" "${RemoteUser}@${RemoteHost}" $RemoteCommand
}

switch ($Mode) {
    "Master" {
        $remoteCommand = "latest=`$(ls -dt $RemoteProjectDir/logs/autodl/* 2>/dev/null | head -n 1); if [ -z ""`$latest"" ]; then echo 'No AutoDL master log found.'; exit 1; fi; echo LATEST=`$latest; tail -n $Lines -f ""`$latest/master.log"""
        Invoke-RemoteTail -RemoteCommand $remoteCommand
        break
    }
    "Run" {
        if ([string]::IsNullOrWhiteSpace($RunName)) {
            throw "Mode=Run requires -RunName."
        }
        $remoteCommand = "tail -n $Lines -f $RemoteProjectDir/runs/$RunName/train.log"
        Invoke-RemoteTail -RemoteCommand $remoteCommand
        break
    }
    default {
        if (-not [string]::IsNullOrWhiteSpace($RunName)) {
            $remoteCommand = "tail -n $Lines -f $RemoteProjectDir/runs/$RunName/train.log"
        } else {
            $remoteCommand = "latest=`$(ls -dt $RemoteProjectDir/logs/autodl/* 2>/dev/null | head -n 1); if [ -z ""`$latest"" ]; then echo 'No AutoDL master log found.'; exit 1; fi; echo LATEST=`$latest; tail -n $Lines -f ""`$latest/master.log"""
        }
        Invoke-RemoteTail -RemoteCommand $remoteCommand
        break
    }
}
