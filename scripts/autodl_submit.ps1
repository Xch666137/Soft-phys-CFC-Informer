param(
    [Parameter(Mandatory = $true)]
    [Alias("Host")]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [int]$Port,

    [string]$RemoteUser = "root",
    [string]$RepoUrl = "https://github.com/Xch666137/Soft-phys-CFC-Informer.git",
    [string]$Branch = "codex/thesis-mainline",
    [string]$RemoteProjectDir = "/root/autodl-tmp/Soft-phys-CFC-Informer",
    [string]$RemoteEnvName = "Soft-phys-CFC-Informer",
    [string]$LocalDataRoot = "data_raw",
    [string]$Stages = "verify,build_dataset,benchmark_main,benchmark_time",
    [string]$SessionName = "autodl-thesis",
    [string]$PythonVersion = "3.10",
    [switch]$SkipGitSyncCheck,
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

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
Set-Location $repoRoot

Assert-Tool "ssh"
Assert-Tool "git"
Assert-Tool "tar"

$sshTarget = "$RemoteUser@$RemoteHost"
$remoteParent = Get-RemoteParent -UnixPath $RemoteProjectDir
$localDataRootPath = (Resolve-Path $LocalDataRoot).Path
$remoteDataRoot = "$RemoteProjectDir/data_raw"
$remoteScriptPath = "$RemoteProjectDir/scripts/autodl_remote_run.sh"
$remoteUploadStamp = "$remoteDataRoot/.autodl_upload_complete"

if (-not $SkipGitSyncCheck -and -not $DryRun) {
    Write-Step "Checking local git sync against origin/$Branch"
    Assert-GitSync -RepoRoot $repoRoot -BranchName $Branch
}

if (-not (Test-Path $localDataRootPath)) {
    throw "Local data root not found: $LocalDataRoot"
}

$cloneCommand = @"
mkdir -p $(Quote-Bash $remoteParent)
if [ ! -d $(Quote-Bash "$RemoteProjectDir/.git") ]; then
  git clone --branch $(Quote-Bash $Branch) --single-branch $(Quote-Bash $RepoUrl) $(Quote-Bash $RemoteProjectDir)
else
  git -C $(Quote-Bash $RemoteProjectDir) fetch origin
  git -C $(Quote-Bash $RemoteProjectDir) checkout $(Quote-Bash $Branch)
  git -C $(Quote-Bash $RemoteProjectDir) pull --ff-only origin $(Quote-Bash $Branch)
fi
chmod +x $(Quote-Bash $remoteScriptPath)
"@

$cloneDisplay = "ssh -p $Port $sshTarget bash -lc " + (Quote-Bash $cloneCommand)
Invoke-OrPrint -Label "Clone or update remote repository" -Display $cloneDisplay -Action {
    & ssh -p $Port $sshTarget "bash -lc $(Quote-Bash $cloneCommand)"
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
        $uploadDisplay = "tar -cf - -C `"$localDataRootPath`" . | ssh -p $Port $sshTarget ""mkdir -p $(Quote-Bash $remoteDataRoot) && tar -xf - -C $(Quote-Bash $remoteDataRoot) && touch $(Quote-Bash $remoteUploadStamp)"""
        Invoke-OrPrint -Label "Upload data_raw via tar stream" -Display $uploadDisplay -Action {
            tar -cf - -C $localDataRootPath . | ssh -p $Port $sshTarget "mkdir -p $(Quote-Bash $remoteDataRoot) && tar -xf - -C $(Quote-Bash $remoteDataRoot) && touch $(Quote-Bash $remoteUploadStamp)"
        }
    } else {
        Write-Step "Remote data upload skipped because upload stamp already exists."
    }
} else {
    Write-Step "Skipping data upload by request."
}

$remoteRunCommand = "cd $(Quote-Bash $RemoteProjectDir) && bash $(Quote-Bash "scripts/autodl_remote_run.sh") --project-dir $(Quote-Bash $RemoteProjectDir) --env-name $(Quote-Bash $RemoteEnvName) --python-version $(Quote-Bash $PythonVersion) --stages $(Quote-Bash $Stages)"
$tmuxCommand = @"
if ! command -v tmux >/dev/null 2>&1; then
  echo 'tmux not found on remote host.' >&2
  exit 1
fi
if tmux has-session -t $(Quote-Bash $SessionName) 2>/dev/null; then
  echo 'tmux session already exists: $SessionName' >&2
  exit 1
fi
tmux new-session -d -s $(Quote-Bash $SessionName) $(Quote-Bash $remoteRunCommand)
tmux ls
"@

$tmuxDisplay = "ssh -p $Port $sshTarget bash -lc " + (Quote-Bash $tmuxCommand)
Invoke-OrPrint -Label "Launch remote tmux session" -Display $tmuxDisplay -Action {
    & ssh -p $Port $sshTarget "bash -lc $(Quote-Bash $tmuxCommand)"
}

Write-Step "Submission complete."
Write-Step "Attach with: ssh -p $Port $sshTarget -t `"tmux attach -t $SessionName`""
