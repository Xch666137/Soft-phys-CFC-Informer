param(
    [Parameter(Mandatory = $true)]
    [Alias("Host")]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [int]$Port,

    [string]$RemoteUser = "root",
    [string]$RepoUrl = "https://github.com/Xch666137/Soft-phys-CFC-Informer.git",
    [string]$Branch = "codex/thesis-mainline",
    [ValidateSet("upload", "clone")]
    [string]$SourceSyncMode = "upload",
    [string]$RemoteProjectDir = "/root/autodl-tmp/Soft-phys-CFC-Informer",
    [string]$RemoteEnvName = "Soft-phys-CFC-Informer",
    [string]$LocalDataRoot = "data_raw",
    [string]$Stages = "verify,build_dataset,benchmark_main,benchmark_time",
    [string]$SessionName = "autodl-thesis",
    [string]$PythonVersion = "3.10",
    [switch]$SkipGitSyncCheck,
    [switch]$SkipSourceUpload,
    [switch]$SkipDataUpload,
    [switch]$ForceDataUpload,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host "[autodl-submit] $Message"
}

function Quote-Bash {
    param([string]$Value)
    $joiner = "'" + '"' + "'" + '"' + "'"
    return "'" + (($Value -split "'") -join $joiner) + "'"
}

function Get-RemoteParent {
    param([string]$UnixPath)
    $lastSlash = $UnixPath.LastIndexOf("/")
    if ($lastSlash -le 0) {
        return "/"
    }
    return $UnixPath.Substring(0, $lastSlash)
}

function Invoke-OrPrint {
    param(
        [string]$Label,
        [scriptblock]$Action,
        [string]$Display
    )

    if ($DryRun) {
        Write-Step "DRY-RUN $Label"
        Write-Host $Display
        return
    }

    Write-Step $Label
    & $Action
}

function Assert-Tool {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required tool not found in PATH: $Name"
    }
}

function New-RemoteTempScriptPath {
    param([string]$Prefix)
    $stamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    return "/tmp/{0}_{1}.sh" -f $Prefix, $stamp
}

function Invoke-RemoteScript {
    param(
        [string]$Label,
        [string]$ScriptContent,
        [string]$Display
    )

    if ($DryRun) {
        Write-Step "DRY-RUN $Label"
        Write-Host $Display
        return
    }

    Write-Step $Label
    $localTempScript = Join-Path $env:TEMP ("codex_autodl_" + [guid]::NewGuid().ToString("N") + ".sh")
    $remoteTempScript = New-RemoteTempScriptPath -Prefix "codex_autodl"

    try {
        [System.IO.File]::WriteAllText($localTempScript, $ScriptContent, [System.Text.UTF8Encoding]::new($false))
        & scp -P $Port $localTempScript "${sshTarget}:$remoteTempScript"
        if ($LASTEXITCODE -ne 0) {
            throw "Remote temp script upload failed."
        }
        & ssh -p $Port $sshTarget "bash $remoteTempScript"
        if ($LASTEXITCODE -ne 0) {
            throw "Remote script execution failed."
        }
    } finally {
        Remove-Item $localTempScript -Force -ErrorAction SilentlyContinue
        & ssh -p $Port $sshTarget "rm -f $remoteTempScript" | Out-Null
    }
}

function Assert-GitSync {
    param(
        [string]$RepoRoot,
        [string]$BranchName
    )

    $status = git -C $RepoRoot status --porcelain
    if ($LASTEXITCODE -ne 0) {
        throw "git status failed in $RepoRoot"
    }
    if ($status) {
        throw "Working tree is dirty. Commit or stash changes before AutoDL clone-based submission."
    }

    $localHead = (git -C $RepoRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to resolve local HEAD."
    }

    $remoteHead = (git ls-remote origin ("refs/heads/" + $BranchName) | ForEach-Object { ($_ -split "\s+")[0] } | Select-Object -First 1)
    if (-not $remoteHead) {
        throw "Failed to resolve remote branch head for origin/$BranchName."
    }

    if ($localHead -ne $remoteHead.Trim()) {
        throw "Local HEAD does not match origin/$BranchName. Push the branch first or rerun with -SkipGitSyncCheck."
    }
}

function Assert-RemotePathExists {
    param(
        [string]$RemotePath,
        [string]$Description
    )

    & ssh -p $Port $sshTarget "test -e $RemotePath"
    if ($LASTEXITCODE -ne 0) {
        throw "$Description not found on remote host: $RemotePath"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
Set-Location $repoRoot

Assert-Tool "ssh"
Assert-Tool "git"
Assert-Tool "tar"
Assert-Tool "scp"

$sshTarget = "$RemoteUser@$RemoteHost"
$remoteParent = Get-RemoteParent -UnixPath $RemoteProjectDir
$localDataRootPath = (Resolve-Path $LocalDataRoot).Path
$remoteDataRoot = "$RemoteProjectDir/data_raw"
$remoteScriptPath = "$RemoteProjectDir/scripts/autodl_remote_run.sh"
$remoteUploadStamp = "$remoteDataRoot/.autodl_upload_complete"
$remoteUploadTar = "/tmp/soft_phys_cfc_informer_data_raw.tar"
$remoteSourceTar = "/tmp/soft_phys_cfc_informer_source.tar"

if ($SourceSyncMode -eq "clone" -and -not $SkipGitSyncCheck -and -not $DryRun) {
    Write-Step "Checking local git sync against origin/$Branch"
    Assert-GitSync -RepoRoot $repoRoot -BranchName $Branch
}

if (-not (Test-Path $localDataRootPath)) {
    throw "Local data root not found: $LocalDataRoot"
}

$cloneCommand = @"
set -euo pipefail
mkdir -p $remoteParent
if [ -d $RemoteProjectDir ] && ! git -C $RemoteProjectDir rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  rm -rf $RemoteProjectDir
fi
if [ ! -d $RemoteProjectDir/.git ]; then
  rm -rf $RemoteProjectDir
  for attempt in 1 2 3; do
    git clone --depth 1 --branch $(Quote-Bash $Branch) --single-branch $(Quote-Bash $RepoUrl) $RemoteProjectDir && break
    if [ "`$attempt" -eq 3 ]; then
      echo "git clone failed after 3 attempts" >&2
      exit 1
    fi
    rm -rf $RemoteProjectDir
    sleep 3
  done
else
  git -C $RemoteProjectDir fetch origin
  git -C $RemoteProjectDir checkout $(Quote-Bash $Branch)
  git -C $RemoteProjectDir pull --ff-only origin $(Quote-Bash $Branch)
fi
chmod +x $remoteScriptPath
"@

if ($SourceSyncMode -eq "clone") {
    $cloneDisplay = "scp -P $Port <temp_clone_script> ${sshTarget}:/tmp/<temp>.sh`nssh -p $Port $sshTarget bash /tmp/<temp>.sh"
    Invoke-RemoteScript -Label "Clone or update remote repository" -Display $cloneDisplay -ScriptContent $cloneCommand
} else {
    if ($SkipSourceUpload) {
        if (-not $DryRun) {
            Assert-RemotePathExists -RemotePath $RemoteProjectDir -Description "Remote project directory"
            Assert-RemotePathExists -RemotePath $remoteScriptPath -Description "Remote AutoDL runner script"
        }
        Write-Step "Skipping source upload by request."
    } else {
        $tempSourceArchive = Join-Path $env:TEMP "soft_phys_cfc_informer_source.tar"
        $sourceUploadDisplay = @"
tar -cf "$tempSourceArchive" --exclude=.git --exclude=data_raw --exclude=runs --exclude=downloads --exclude=.pytest_cache --exclude=__pycache__ -C "$repoRoot" .
scp -P $Port "$tempSourceArchive" ${sshTarget}:$remoteSourceTar
scp -P $Port <temp_extract_script> ${sshTarget}:/tmp/<temp>.sh
ssh -p $Port $sshTarget bash /tmp/<temp>.sh
"@
        Invoke-OrPrint -Label "Upload source tree to remote host" -Display $sourceUploadDisplay -Action {
            if (Test-Path $tempSourceArchive) {
                Remove-Item $tempSourceArchive -Force
            }
            tar -cf $tempSourceArchive --exclude=.git --exclude=data_raw --exclude=runs --exclude=downloads --exclude=.pytest_cache --exclude=__pycache__ -C $repoRoot .
            if ($LASTEXITCODE -ne 0) {
                throw "Local source archive creation failed."
            }
            & scp -P $Port $tempSourceArchive "${sshTarget}:$remoteSourceTar"
            if ($LASTEXITCODE -ne 0) {
                throw "Remote source archive upload failed."
            }
            $extractSourceScript = @"
set -euo pipefail
mkdir -p $remoteParent
rm -rf $RemoteProjectDir
mkdir -p $RemoteProjectDir
tar -xf $remoteSourceTar -C $RemoteProjectDir
rm -f $remoteSourceTar
chmod +x $remoteScriptPath
"@
            Invoke-RemoteScript -Label "Extract uploaded source tree on remote host" -Display "scp -P $Port <temp_extract_script> ${sshTarget}:/tmp/<temp>.sh`nssh -p $Port $sshTarget bash /tmp/<temp>.sh" -ScriptContent $extractSourceScript
            Remove-Item $tempSourceArchive -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not $SkipDataUpload) {
    $needsUpload = $true
    if (-not $ForceDataUpload -and -not $DryRun) {
        & ssh -p $Port $sshTarget "bash -lc $(Quote-Bash "test -f $(Quote-Bash $remoteUploadStamp)")"
        if ($LASTEXITCODE -eq 0) {
            $needsUpload = $false
        }
    }

    if ($ForceDataUpload -or $needsUpload) {
        $tempArchive = Join-Path $env:TEMP "soft_phys_cfc_informer_data_raw.tar"
        $uploadDisplay = @"
tar -cf "$tempArchive" -C "$localDataRootPath" .
scp -P $Port "$tempArchive" ${sshTarget}:$remoteUploadTar
scp -P $Port <temp_extract_script> ${sshTarget}:/tmp/<temp>.sh
ssh -p $Port $sshTarget bash /tmp/<temp>.sh
"@
        Invoke-OrPrint -Label "Upload data_raw via tar stream" -Display $uploadDisplay -Action {
            if (Test-Path $tempArchive) {
                Remove-Item $tempArchive -Force
            }
            tar -cf $tempArchive -C $localDataRootPath .
            if ($LASTEXITCODE -ne 0) {
                throw "Local tar archive creation failed."
            }
            & scp -P $Port $tempArchive "${sshTarget}:$remoteUploadTar"
            if ($LASTEXITCODE -ne 0) {
                throw "Remote archive upload failed."
            }
            $extractDataScript = @"
set -euo pipefail
mkdir -p $remoteDataRoot
tar -xf $remoteUploadTar -C $remoteDataRoot
touch $remoteUploadStamp
rm -f $remoteUploadTar
"@
            Invoke-RemoteScript -Label "Extract uploaded data_raw on remote host" -Display "scp -P $Port <temp_extract_script> ${sshTarget}:/tmp/<temp>.sh`nssh -p $Port $sshTarget bash /tmp/<temp>.sh" -ScriptContent $extractDataScript
            Remove-Item $tempArchive -Force -ErrorAction SilentlyContinue
        }
    } else {
        Write-Step "Remote data upload skipped because upload stamp already exists."
    }
} else {
    Write-Step "Skipping data upload by request."
}

$remoteRunCommand = "cd $RemoteProjectDir && bash scripts/autodl_remote_run.sh --project-dir $(Quote-Bash $RemoteProjectDir) --env-name $(Quote-Bash $RemoteEnvName) --python-version $(Quote-Bash $PythonVersion) --stages $(Quote-Bash $Stages); status=`$?; echo; echo '[autodl-submit] remote runner exited with status' `$status; echo '[autodl-submit] session kept open for inspection. Type exit to close.'; exec bash"
$tmuxCommand = @"
set -euo pipefail
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found on remote host." >&2
  exit 1
fi
if tmux has-session -t $(Quote-Bash $SessionName) 2>/dev/null; then
  echo "tmux session already exists: $SessionName" >&2
  exit 1
fi
tmux new-session -d -s $(Quote-Bash $SessionName) "bash -lc $(Quote-Bash $remoteRunCommand)"
tmux ls
"@

$tmuxDisplay = "scp -P $Port <temp_tmux_script> ${sshTarget}:/tmp/<temp>.sh`nssh -p $Port $sshTarget bash /tmp/<temp>.sh"
Invoke-RemoteScript -Label "Launch remote tmux session" -Display $tmuxDisplay -ScriptContent $tmuxCommand

Write-Step "Submission complete."
Write-Step "Attach with: ssh -p $Port $sshTarget -t `"tmux attach -t $SessionName`""
