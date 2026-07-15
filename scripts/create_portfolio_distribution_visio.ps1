param(
    [string]$OutputPath = "docs/figures/physformer_portfolio_electrical_distribution.vsdx",
    [string]$PreviewDir = "docs/figures/visio_preview"
)

$ErrorActionPreference = "Stop"

$Script:FontDisplay = "Segoe UI Semibold"
$Script:FontBody = "Segoe UI"
$Script:FontMono = "Bahnschrift SemiBold"

function Invoke-WithComRetry {
    param(
        [scriptblock]$Action,
        [int]$Retries = 35,
        [int]$DelayMilliseconds = 450
    )

    $lastError = $null
    for ($i = 0; $i -lt $Retries; $i++) {
        try {
            return & $Action
        } catch {
            $lastError = $_
            if ($_.Exception.Message -notmatch "RPC_E_CALL_REJECTED|Call was rejected by callee|0x80010001") {
                throw
            }
            Start-Sleep -Milliseconds $DelayMilliseconds
        }
    }
    throw $lastError
}

function ColorFormula {
    param([string]$Hex)
    $h = $Hex.TrimStart("#")
    $r = [Convert]::ToInt32($h.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($h.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($h.Substring(4, 2), 16)
    return "RGB($r,$g,$b)"
}

function Set-CellSafe {
    param($Shape, [string]$Cell, [string]$Formula)
    try {
        $Shape.CellsU($Cell).FormulaU = $Formula
    } catch {
    }
}

function Send-ToBackSafe {
    param($Shape)
    try {
        $Shape.SendToBack() | Out-Null
    } catch {
    }
}

function Style-Shape {
    param(
        $Shape,
        [string]$Fill = "#FFFFFF",
        [string]$Line = "#C9D0D5",
        [string]$Text = "#15181B",
        [double]$FontSize = 7.0,
        [bool]$Bold = $false,
        [double]$LineWeight = 0.65,
        [string]$HAlign = "1",
        [string]$VAlign = "1",
        [double]$Margin = 0.035,
        [bool]$NoFill = $false,
        [bool]$NoLine = $false,
        [bool]$Dashed = $false,
        [string]$Font = $Script:FontBody,
        [double]$Rounding = 0.045
    )

    if ($NoFill) {
        Set-CellSafe $Shape "FillPattern" "0"
    } else {
        Set-CellSafe $Shape "FillForegnd" (ColorFormula $Fill)
        Set-CellSafe $Shape "FillPattern" "1"
    }

    if ($NoLine) {
        Set-CellSafe $Shape "LinePattern" "0"
    } else {
        Set-CellSafe $Shape "LineColor" (ColorFormula $Line)
        Set-CellSafe $Shape "LineWeight" "$LineWeight pt"
        Set-CellSafe $Shape "LinePattern" ($(if ($Dashed) { "2" } else { "1" }))
    }

    Set-CellSafe $Shape "Char.Color" (ColorFormula $Text)
    Set-CellSafe $Shape "Char.Size" "$FontSize pt"
    Set-CellSafe $Shape "Char.Font" "FONT(`"$Font`")"
    Set-CellSafe $Shape "Char.Style" ($(if ($Bold) { "1" } else { "0" }))
    Set-CellSafe $Shape "Para.HorzAlign" $HAlign
    Set-CellSafe $Shape "VerticalAlign" $VAlign
    Set-CellSafe $Shape "LeftMargin" "$Margin in"
    Set-CellSafe $Shape "RightMargin" "$Margin in"
    Set-CellSafe $Shape "TopMargin" "$Margin in"
    Set-CellSafe $Shape "BottomMargin" "$Margin in"
    Set-CellSafe $Shape "Rounding" "$Rounding in"
}

function Add-Text {
    param(
        $Page,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [string]$Text,
        [string]$TextColor = "#15181B",
        [double]$FontSize = 7.0,
        [bool]$Bold = $false,
        [string]$HAlign = "0",
        [string]$VAlign = "1",
        [string]$Font = $Script:FontBody
    )
    $shape = $Page.DrawRectangle($X, $Y, $X + $W, $Y + $H)
    $shape.Text = $Text
    Style-Shape $shape "#FFFFFF" "#FFFFFF" $TextColor $FontSize $Bold 0 $HAlign $VAlign 0.012 $true $true $false $Font 0
    return $shape
}

function Add-Box {
    param(
        $Page,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [string]$Text,
        [string]$Fill = "#FFFFFF",
        [string]$Line = "#C9D0D5",
        [string]$TextColor = "#15181B",
        [double]$FontSize = 6.8,
        [bool]$Bold = $false,
        [double]$LineWeight = 0.65,
        [string]$HAlign = "1",
        [bool]$Dashed = $false,
        [string]$Font = $Script:FontBody,
        [double]$Rounding = 0.045
    )
    $shape = $Page.DrawRectangle($X, $Y, $X + $W, $Y + $H)
    $shape.Text = $Text
    Style-Shape $shape $Fill $Line $TextColor $FontSize $Bold $LineWeight $HAlign "1" 0.04 $false $false $Dashed $Font $Rounding
    return $shape
}

function Add-Dot {
    param(
        $Page,
        [double]$X,
        [double]$Y,
        [double]$D,
        [string]$Text = "",
        [string]$Fill = "#EAF6F5",
        [string]$Line = "#249089",
        [string]$TextColor = "#15181B",
        [double]$FontSize = 6.2,
        [bool]$Bold = $true,
        [string]$Font = $Script:FontMono
    )
    $shape = $Page.DrawOval($X, $Y, $X + $D, $Y + $D)
    $shape.Text = $Text
    Style-Shape $shape $Fill $Line $TextColor $FontSize $Bold 0.8 "1" "1" 0.006 $false $false $false $Font 0
    return $shape
}

function Add-Line {
    param(
        $Page,
        [double]$X1,
        [double]$Y1,
        [double]$X2,
        [double]$Y2,
        [string]$Color = "#7A838A",
        [double]$Weight = 0.8,
        [bool]$Arrow = $false,
        [bool]$Dashed = $false,
        [bool]$StartArrow = $false
    )
    $line = $Page.DrawLine($X1, $Y1, $X2, $Y2)
    Set-CellSafe $line "LineColor" (ColorFormula $Color)
    Set-CellSafe $line "LineWeight" "$Weight pt"
    Set-CellSafe $line "LinePattern" ($(if ($Dashed) { "2" } else { "1" }))
    if ($Arrow) { Set-CellSafe $line "EndArrow" "13" }
    if ($StartArrow) { Set-CellSafe $line "BeginArrow" "13" }
    return $line
}

function Add-Rule {
    param($Page, [double]$X1, [double]$Y, [double]$X2, [string]$Color = "#D6DADD", [double]$Weight = 0.6)
    Add-Line $Page $X1 $Y $X2 $Y $Color $Weight $false $false | Out-Null
}

function Add-PanelLabel {
    param($Page, [string]$Letter, [double]$X, [double]$Y, [string]$Accent = "#15181B")
    Add-Text $Page $X $Y 0.18 0.18 $Letter $Accent 9.2 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Line $Page ($X + 0.23) ($Y + 0.08) ($X + 0.34) ($Y + 0.08) $Accent 1.2 $false | Out-Null
}

function Add-Backplane {
    param($Page)
    $bg = $Page.DrawRectangle(0, 0, 11.4, 7.2)
    Style-Shape $bg "#FBFAF7" "#FBFAF7" "#15181B" 1 $false 0 "0" "0" 0 $false $true
    Send-ToBackSafe $bg

    foreach ($x in @(0.42, 3.98, 7.62, 10.98)) {
        Add-Line $Page $x 0.64 $x 6.55 "#ECE6DC" 0.28 $false | Out-Null
    }
}

function Add-Header {
    param($Page)
    $ink = "#15181B"
    $muted = "#667078"
    $teal = "#249089"
    Add-Text $Page 0.42 6.82 0.86 0.22 "FIG P" $ink 8.4 $true "0" "1" $Script:FontMono | Out-Null
    Add-Text $Page 1.20 6.79 8.60 0.27 "Portfolio distribution as an electrical single-line schematic" $ink 12.0 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Text $Page 1.22 6.52 9.65 0.19 "VPP assets, household-disjoint portfolios, and A1/B1 training combinations are separated so evidence status stays explicit." $muted 6.9 $false "0" "1" $Script:FontBody | Out-Null
    Add-Line $Page 0.42 6.44 10.98 6.44 $ink 1.0 $false | Out-Null
    Add-Line $Page 0.42 6.40 2.35 6.40 $teal 2.1 $false | Out-Null
}

function Configure-Page {
    param($Page)
    $Page.Name = "Portfolio single-line"
    $Page.PageSheet.CellsU("PageWidth").FormulaU = "11.4 in"
    $Page.PageSheet.CellsU("PageHeight").FormulaU = "7.2 in"
    Add-Backplane $Page
}

function Add-Breaker {
    param($Page, [double]$X, [double]$Y, [string]$Color = "#15181B")
    Add-Line $Page ($X - 0.13) $Y ($X + 0.13) $Y $Color 1.0 $false | Out-Null
    Add-Line $Page ($X - 0.04) ($Y - 0.08) ($X + 0.08) ($Y + 0.08) $Color 1.0 $false | Out-Null
}

function Add-Feeder {
    param(
        $Page,
        [double]$BusX,
        [double]$BusY,
        [double]$BoxX,
        [double]$BoxY,
        [string]$BoxText,
        [string]$Tag,
        [string]$Color,
        [string]$Fill,
        [string]$Note
    )
    Add-Line $Page $BusX $BusY $BusX ($BoxY + 0.66) "#15181B" 1.15 $false | Out-Null
    Add-Breaker $Page $BusX ($BoxY + 0.86) "#15181B"
    Add-Box $Page $BoxX $BoxY 1.04 0.50 $BoxText $Fill $Color "#15181B" 6.0 $true 0.85 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Dot $Page ($BusX - 0.13) ($BoxY + 0.55) 0.26 $Tag "#FFFFFF" $Color $Color 5.5 $true $Script:FontMono | Out-Null
    Add-Text $Page ($BoxX - 0.04) ($BoxY - 0.28) 1.14 0.18 $Note "#667078" 5.4 $false "1" "1" $Script:FontBody | Out-Null
}

function Build-PortfolioDistributionPage {
    param($Page)
    Configure-Page $Page

    $ink = "#15181B"
    $muted = "#667078"
    $grid = "#D6DADD"
    $teal = "#249089"
    $tealSoft = "#E8F4F2"
    $blue = "#276FB7"
    $blueSoft = "#EAF1FA"
    $amber = "#B77A22"
    $amberSoft = "#F6E9D0"
    $red = "#C44E52"
    $redSoft = "#F8E7E3"
    $violet = "#596AA6"
    $violetSoft = "#ECEEFA"

    Add-Header $Page

    Add-PanelLabel $Page "a" 0.48 6.06 $ink
    Add-Text $Page 0.84 6.04 2.86 0.20 "Electrical asset layer" $ink 7.5 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Text $Page 0.84 5.81 5.45 0.16 "The physical aggregation is drawn as a bus equation, not as a model architecture." $muted 5.8 $false "0" "1" | Out-Null

    Add-Box $Page 0.76 4.98 0.72 0.42 "Grid /`nmarket" "#FFFFFF" "#15181B" $ink 5.8 $true 0.8 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Dot $Page 1.78 5.04 0.28 "PCC" "#FFFFFF" "#15181B" $ink 4.6 $true $Script:FontMono | Out-Null
    Add-Line $Page 1.48 5.19 1.78 5.19 $ink 1.15 $true | Out-Null
    Add-Line $Page 2.06 5.19 6.22 5.19 $ink 2.0 $false | Out-Null
    Add-Text $Page 2.38 5.35 1.96 0.16 "VPP aggregation bus" $ink 6.2 $true "0" "1" $Script:FontMono | Out-Null
    Add-Box $Page 5.82 4.94 0.92 0.42 "net meter`nPnet" "#FFFFFF" "#15181B" $ink 5.7 $true 0.8 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Line $Page 6.22 5.19 5.82 5.15 $ink 1.0 $true | Out-Null

    Add-Feeder $Page 2.20 5.19 1.68 3.78 "Load`nfeeder" "+L" $blue $blueSoft "demand input"
    Add-Feeder $Page 3.16 5.19 2.64 3.78 "PV`ninverter" "-PV" $teal $tealSoft "solar generation"
    Add-Feeder $Page 4.12 5.19 3.60 3.78 "Wind`ninverter" "-W" $teal $tealSoft "wind generation"
    Add-Feeder $Page 5.08 5.19 4.56 3.78 "Battery`nPCS + SOC" "+B" $amber $amberSoft "bi-directional"

    Add-Box $Page 1.00 3.10 5.30 0.32 "Power-balance contract:  Pnet = Load - PV - Wind + Battery" "#FFFFFF" "#15181B" $ink 6.7 $true 0.75 "1" $false $Script:FontMono 0.035 | Out-Null
    Add-Text $Page 1.00 2.78 5.28 0.18 "This is the electrical meaning behind both the model decoder and aggregate evaluation." $muted 5.7 $false "1" "1" | Out-Null

    Add-Box $Page 6.02 4.42 0.90 0.36 "weather`ncontext" "#FFFFFF" "#9DA7AE" $muted 5.5 $true 0.6 "1" $true $Script:FontBody 0.035 | Out-Null
    Add-Line $Page 6.02 4.60 3.16 4.72 "#9DA7AE" 0.7 $false $true | Out-Null
    Add-Line $Page 6.02 4.60 4.12 4.72 "#9DA7AE" 0.7 $false $true | Out-Null
    Add-Line $Page 6.02 4.60 2.20 4.72 "#9DA7AE" 0.7 $false $true | Out-Null

    Add-PanelLabel $Page "b" 7.08 6.06 $violet
    Add-Text $Page 7.44 6.04 2.60 0.20 "Portfolio split layer" $ink 7.5 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Text $Page 7.44 5.81 3.28 0.16 "Portfolio membership is the switchgear that prevents household leakage." $muted 5.8 $false "0" "1" | Out-Null

    Add-Box $Page 7.42 5.18 2.60 0.34 "portfolio_manifest" $violetSoft $violet $ink 6.2 $true 0.85 "1" $false $Script:FontMono 0.04 | Out-Null
    Add-Line $Page 8.72 5.18 8.72 4.82 $violet 1.0 $false | Out-Null
    Add-Line $Page 7.62 4.82 10.72 4.82 $violet 1.0 $false | Out-Null

    $splitY = 4.34
    $splitW = 0.92
    Add-Line $Page 7.72 4.82 7.72 4.54 $violet 0.9 $false | Out-Null
    Add-Line $Page 8.56 4.82 8.56 4.54 $violet 0.9 $false | Out-Null
    Add-Line $Page 9.40 4.82 9.40 4.54 $violet 0.9 $false | Out-Null
    Add-Line $Page 10.24 4.82 10.24 4.54 $violet 0.9 $false | Out-Null
    Add-Box $Page 7.26 $splitY $splitW 0.36 "Train`nports" "#FFFFFF" $violet $ink 5.7 $true 0.7 "1" $false $Script:FontBody 0.035 | Out-Null
    Add-Box $Page 8.10 $splitY $splitW 0.36 "Val`nports" "#FFFFFF" $violet $ink 5.7 $true 0.7 "1" $false $Script:FontBody 0.035 | Out-Null
    Add-Box $Page 8.94 $splitY $splitW 0.36 "Held-out`ntest" "#FFFFFF" $violet $ink 5.6 $true 0.7 "1" $false $Script:FontBody 0.035 | Out-Null
    Add-Box $Page 9.78 $splitY $splitW 0.36 "Target`nfew-shot" $amberSoft $amber $ink 5.6 $true 0.75 "1" $false $Script:FontBody 0.035 | Out-Null

    Add-Box $Page 7.24 3.58 3.50 0.46 "Leakage rule: household_id split before portfolio use; test portfolios remain unseen during training." "#FFFFFF" "#9DA7AE" $ink 5.8 $true 0.55 "1" $false $Script:FontBody 0.035 | Out-Null
    Add-Text $Page 7.32 3.22 3.34 0.22 "Scope: held-out composition under ACT/Canberra weather; no cross-climate claim." $muted 5.7 $false "1" "1" | Out-Null

    Add-Rule $Page 0.42 2.58 10.98 "#DAD5CB" 0.75

    Add-PanelLabel $Page "c" 0.48 2.24 $teal
    Add-Text $Page 0.84 2.21 2.82 0.20 "Training-combination layer" $ink 7.5 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Text $Page 0.84 1.98 5.95 0.16 "The diagram keeps supported baseline evidence separate from open Phase-B rescue/adaptation gates." $muted 5.8 $false "0" "1" | Out-Null

    $laneY = 1.34
    Add-Box $Page 0.92 $laneY 1.12 0.42 "multi-portfolio`ndata" "#FFFFFF" "#15181B" $ink 5.8 $true 0.75 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Box $Page 2.54 $laneY 1.18 0.42 "A1 scratch`n8-token iGT" $tealSoft $teal $ink 5.8 $true 0.85 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Box $Page 4.22 $laneY 1.08 0.42 "test`nMAE" "#FFFFFF" "#15181B" $ink 5.8 $true 0.75 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Box $Page 5.80 $laneY 1.46 0.42 "supported gate`nA1 = 1.811e-3" $tealSoft $teal $teal 5.7 $true 0.85 "1" $false $Script:FontMono 0.04 | Out-Null
    Add-Line $Page 2.04 ($laneY + 0.21) 2.54 ($laneY + 0.21) $ink 0.95 $true | Out-Null
    Add-Line $Page 3.72 ($laneY + 0.21) 4.22 ($laneY + 0.21) $ink 0.95 $true | Out-Null
    Add-Line $Page 5.30 ($laneY + 0.21) 5.80 ($laneY + 0.21) $teal 0.95 $true | Out-Null

    Add-Box $Page 2.54 0.58 1.18 0.42 "B1 MCP`npretrain" $amberSoft $amber $ink 5.7 $true 0.8 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Line $Page 1.48 $laneY 2.54 1.00 $amber 0.9 $true $true | Out-Null
    Add-Box $Page 4.20 0.58 0.82 0.42 "R0`ndirect" "#FFFFFF" $amber $amber 5.7 $true 0.75 "1" $true $Script:FontMono 0.04 | Out-Null
    Add-Box $Page 5.24 0.58 0.82 0.42 "R1`nlow-LR" "#FFFFFF" $amber $amber 5.7 $true 0.75 "1" $true $Script:FontMono 0.04 | Out-Null
    Add-Box $Page 6.28 0.58 0.82 0.42 "R2`ntarget" "#FFFFFF" $amber $amber 5.7 $true 0.75 "1" $true $Script:FontMono 0.04 | Out-Null
    Add-Line $Page 3.72 0.79 4.20 0.79 $amber 0.9 $true $true | Out-Null
    Add-Line $Page 5.02 0.79 5.24 0.79 $amber 0.75 $true $true | Out-Null
    Add-Line $Page 6.06 0.79 6.28 0.79 $amber 0.75 $true $true | Out-Null

    Add-PanelLabel $Page "d" 7.68 2.24 $red
    Add-Text $Page 8.04 2.21 1.90 0.20 "Evidence status" $ink 7.5 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Box $Page 8.04 1.48 2.74 0.34 "C11/C12: A1 is the current architecture baseline" $tealSoft $teal $ink 5.9 $true 0.65 "1" $false $Script:FontBody 0.035 | Out-Null
    Add-Box $Page 8.04 1.06 2.74 0.34 "N113: old high-LR static finetune is a dead end" $redSoft $red $ink 5.9 $true 0.65 "1" $false $Script:FontBody 0.035 | Out-Null
    Add-Box $Page 8.04 0.64 2.74 0.34 "N114: clean R0/R1/R2 protocol remains open" $amberSoft $amber $ink 5.9 $true 0.65 "1" $false $Script:FontBody 0.035 | Out-Null

    Add-Rule $Page 0.42 0.42 10.98 "#DAD5CB" 0.65
    Add-Text $Page 0.42 0.18 10.9 0.16 "Solid lines denote supported data/power paths; amber dashed lines denote unresolved Phase-B branches. The figure is a communication schematic, not a protection-grade electrical drawing." $muted 5.6 $false "0" "1" | Out-Null
}

$resolvedOutput = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPath)
$resolvedPreview = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PreviewDir)
$outputDir = Split-Path -Parent $resolvedOutput
if (-not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}
if (-not (Test-Path -LiteralPath $resolvedPreview)) {
    New-Item -ItemType Directory -Path $resolvedPreview | Out-Null
}

$script:visio = Invoke-WithComRetry { New-Object -ComObject Visio.Application }
Invoke-WithComRetry { $script:visio.Visible = $true } | Out-Null
Start-Sleep -Seconds 1
Invoke-WithComRetry { $script:visio.AlertResponse = 7 } | Out-Null

$script:doc = Invoke-WithComRetry { $script:visio.Documents.Add("") }
$page = Invoke-WithComRetry { $script:visio.ActivePage }
Build-PortfolioDistributionPage $page

if (Test-Path -LiteralPath $resolvedOutput) {
    Remove-Item -LiteralPath $resolvedOutput -Force
}

Invoke-WithComRetry { $script:doc.SaveAs($script:resolvedOutput) } | Out-Null

$previewPdf = Join-Path $resolvedPreview "physformer_portfolio_distribution.pdf"
$previewPng = Join-Path $resolvedPreview "physformer_portfolio_distribution-1.png"
foreach ($previewFile in @($previewPdf, $previewPng)) {
    if (Test-Path -LiteralPath $previewFile) {
        Remove-Item -LiteralPath $previewFile -Force
    }
}

Invoke-WithComRetry { $script:doc.ExportAsFixedFormat(1, $script:previewPdf, 1, 0) } | Out-Null

$pdftoppm = Get-Command pdftoppm -ErrorAction SilentlyContinue
if ($pdftoppm) {
    $pngPrefix = Join-Path $resolvedPreview "physformer_portfolio_distribution"
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $pdftoppm.Source -png -r 220 $previewPdf $pngPrefix 2>$null | Out-Null
    $pdftoppmExitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    if ($pdftoppmExitCode -ne 0) {
        Write-Output "PNG_EXPORT_WARNING=pdftoppm exited with code $pdftoppmExitCode"
    }
} else {
    Write-Output "PNG_EXPORT_SKIPPED=pdftoppm not found"
}

try {
    $visio.ActiveWindow.Page = $page
} catch {
}

Invoke-WithComRetry { $script:doc.Close() } | Out-Null
Invoke-WithComRetry { $script:visio.Quit() } | Out-Null

Write-Output "SAVED_VSDX=$resolvedOutput"
Write-Output "PREVIEW_PDF=$previewPdf"
Write-Output "PREVIEW_PNG=$previewPng"
Write-Output "VISIO_CLOSED=True"
