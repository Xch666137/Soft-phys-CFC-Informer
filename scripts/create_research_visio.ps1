param(
    [string]$OutputPath = "docs/figures/physformer_nature_style_research_architecture.vsdx",
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
        [int]$DelayMilliseconds = 500
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
        [double]$Rounding = 0.055
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
        [bool]$Arrow = $true,
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

function Add-PanelLabel {
    param($Page, [string]$Letter, [double]$X, [double]$Y, [string]$Accent = "#15181B")
    Add-Text $Page $X $Y 0.18 0.18 $Letter $Accent 9.2 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Line $Page ($X + 0.23) ($Y + 0.08) ($X + 0.34) ($Y + 0.08) $Accent 1.2 $false | Out-Null
}

function Add-Rule {
    param($Page, [double]$X1, [double]$Y, [double]$X2, [string]$Color = "#D6DADD", [double]$Weight = 0.6)
    Add-Line $Page $X1 $Y $X2 $Y $Color $Weight $false $false | Out-Null
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
    param($Page, [string]$Figure, [string]$Title, [string]$Subtitle, [string]$Accent = "#249089")
    Add-Text $Page 0.42 6.82 0.82 0.22 $Figure "#15181B" 8.4 $true "0" "1" $Script:FontMono | Out-Null
    Add-Text $Page 1.20 6.79 7.70 0.27 $Title "#15181B" 12.0 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Text $Page 1.22 6.52 9.65 0.19 $Subtitle "#667078" 6.9 $false "0" "1" $Script:FontBody | Out-Null
    Add-Line $Page 0.42 6.44 10.98 6.44 "#15181B" 1.0 $false | Out-Null
    Add-Line $Page 0.42 6.40 2.35 6.40 $Accent 2.1 $false | Out-Null
}

function Add-Step {
    param(
        $Page,
        [string]$Number,
        [double]$X,
        [double]$Y,
        [string]$Label,
        [string]$Accent = "#249089"
    )
    Add-Dot $Page $X $Y 0.22 $Number "#15181B" "#15181B" "#FFFFFF" 5.8 $true $Script:FontMono | Out-Null
    Add-Text $Page ($X + 0.28) ($Y - 0.01) 1.30 0.18 $Label $Accent 5.7 $true "0" "1" $Script:FontMono | Out-Null
}

function Map-Value {
    param([double]$Value, [double]$Min, [double]$Max, [double]$Start, [double]$Length)
    return $Start + (($Value - $Min) / ($Max - $Min)) * $Length
}

function Configure-Page {
    param($Page, [string]$Name)
    $Page.Name = $Name
    $Page.PageSheet.CellsU("PageWidth").FormulaU = "11.4 in"
    $Page.PageSheet.CellsU("PageHeight").FormulaU = "7.2 in"
    Add-Backplane $Page
}

function Build-ResearchLogicPage {
    param($Page)
    Configure-Page $Page "Fig1 Research logic"

    $ink = "#15181B"
    $muted = "#667078"
    $grid = "#D6DADD"
    $soft = "#F3F0EA"
    $teal = "#249089"
    $tealSoft = "#E8F4F2"
    $blue = "#276FB7"
    $blueSoft = "#EAF1FA"
    $red = "#C44E52"
    $redSoft = "#F8E7E3"
    $amber = "#B77A22"
    $amberSoft = "#F6E9D0"
    $violet = "#596AA6"
    $violetSoft = "#ECEEFA"

    Add-Header $Page "FIG 1" "Research path as visual argument" "Supported claim: component-token separation beats fixed physics-prior escalation for aggregate VPP forecasting." $teal

    Add-PanelLabel $Page "a" 0.48 6.10 $ink
    Add-Text $Page 0.84 6.08 3.6 0.20 "Mechanism: replace the shared cancellation channel" $ink 7.6 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Text $Page 0.84 5.84 5.2 0.16 "The figure separates the rejected shared-encoder route from the supported A1/iGT route." $muted 5.9 $false "0" "1" | Out-Null

    Add-Text $Page 0.90 5.47 1.72 0.16 "REJECTED FAMILY" $red 5.8 $true "0" "1" $Script:FontMono | Out-Null
    Add-Box $Page 0.90 4.88 1.38 0.46 "c23 PhysFormer`nFiLM + residual" "#FFFFFF" "#9DA7AE" $ink 6.1 $false 0.65 "1" $false $Script:FontBody 0.045 | Out-Null
    Add-Box $Page 2.76 4.88 1.52 0.46 "one encoder space`nfor all branches" $redSoft $red $ink 6.1 $true 0.85 "1" $false $Script:FontBody 0.045 | Out-Null
    Add-Line $Page 2.28 5.11 2.76 5.11 "#9DA7AE" 0.85 $true | Out-Null
    Add-Line $Page 3.52 4.88 3.52 4.38 $red 0.95 $true $true | Out-Null
    Add-Text $Page 2.80 4.13 1.55 0.27 "signed errors can`ncancel in aggregate" $red 6.0 $true "1" "1" | Out-Null
    Add-Text $Page 0.96 4.26 1.48 0.22 "Pnet = L - PV - W + B" $ink 6.7 $true "0" "1" $Script:FontMono | Out-Null

    Add-Line $Page 4.60 5.08 5.16 5.08 "#15181B" 1.05 $true | Out-Null
    Add-Text $Page 4.37 5.25 0.96 0.18 "replace" $ink 5.6 $true "1" "1" $Script:FontMono | Out-Null

    Add-Text $Page 5.26 5.47 1.92 0.16 "SUPPORTED FAMILY" $teal 5.8 $true "0" "1" $Script:FontMono | Out-Null
    $tokenX = 5.28
    $tokenY = 4.99
    $tokenD = 0.30
    $tokenGap = 0.37
    Add-Dot $Page $tokenX $tokenY $tokenD "L" $tealSoft $teal $ink 5.8 $true | Out-Null
    Add-Dot $Page ($tokenX + $tokenGap) $tokenY $tokenD "PV" $tealSoft $teal $ink 5.1 $true | Out-Null
    Add-Dot $Page ($tokenX + 2*$tokenGap) $tokenY $tokenD "W" $tealSoft $teal $ink 5.8 $true | Out-Null
    Add-Dot $Page ($tokenX + 3*$tokenGap) $tokenY $tokenD "B" $tealSoft $teal $ink 5.8 $true | Out-Null
    Add-Dot $Page ($tokenX + 4*$tokenGap) $tokenY $tokenD "S" $tealSoft $teal $ink 5.8 $true | Out-Null
    Add-Dot $Page ($tokenX + 5.35*$tokenGap) $tokenY $tokenD "T" $blueSoft $blue $ink 5.8 $true | Out-Null
    Add-Dot $Page ($tokenX + 6.35*$tokenGap) $tokenY $tokenD "G" $blueSoft $blue $ink 5.8 $true | Out-Null
    Add-Dot $Page ($tokenX + 7.35*$tokenGap) $tokenY $tokenD "V" $blueSoft $blue $ink 5.8 $true | Out-Null
    Add-Box $Page 8.42 4.76 1.30 0.56 "inverted`nattention" $violetSoft $violet $ink 6.1 $true 0.85 "1" $false $Script:FontBody 0.045 | Out-Null
    Add-Box $Page 10.20 4.76 0.76 0.56 "real-unit`nbalance" $amberSoft $amber $ink 5.7 $true 0.85 "1" $false $Script:FontBody 0.045 | Out-Null
    Add-Line $Page 8.30 5.04 8.42 5.04 "#9DA7AE" 0.85 $true | Out-Null
    Add-Line $Page 9.72 5.04 10.20 5.04 "#9DA7AE" 0.85 $true | Out-Null
    Add-Text $Page 5.30 4.33 5.58 0.21 "A1: 8 variable tokens, no fixed physics priors, net-MSE supervision only." $muted 6.2 $false "0" "1" | Out-Null

    Add-Rule $Page 0.66 3.86 10.74 "#DAD5CB" 0.75

    Add-PanelLabel $Page "b" 0.48 3.50 $teal
    Add-Text $Page 0.84 3.47 2.46 0.20 "Primary outcome" $ink 7.2 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Text $Page 0.84 3.22 2.75 0.16 "Test MAE, x10^-3 MW. Lower is better." $muted 5.8 $false "0" "1" | Out-Null
    Add-Text $Page 0.84 2.78 1.06 0.34 "12.5%" $teal 16.0 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Text $Page 1.88 2.84 1.50 0.16 "lower mean MAE" $teal 6.4 $true "0" "1" $Script:FontMono | Out-Null
    $axisX = 0.96
    $axisY = 2.04
    $axisW = 2.32
    Add-Line $Page $axisX $axisY ($axisX + $axisW) $axisY $grid 0.65 $false | Out-Null
    foreach ($tick in @(1.80, 1.90, 2.00, 2.10)) {
        $tx = Map-Value $tick 1.75 2.10 $axisX $axisW
        Add-Line $Page $tx ($axisY - 0.04) $tx ($axisY + 0.04) $grid 0.45 $false | Out-Null
        Add-Text $Page ($tx - 0.18) 1.82 0.36 0.12 ("{0:N2}" -f $tick) $muted 5.0 $false "1" "1" $Script:FontMono | Out-Null
    }
    $xC23 = Map-Value 2.069 1.75 2.10 $axisX $axisW
    $xA1 = Map-Value 1.811 1.75 2.10 $axisX $axisW
    Add-Line $Page $xA1 2.33 $xC23 2.33 "#9DA7AE" 1.1 $false | Out-Null
    Add-Dot $Page ($xC23 - 0.07) 2.26 0.14 "" $violet $violet | Out-Null
    Add-Dot $Page ($xA1 - 0.08) 2.25 0.16 "" $teal $teal | Out-Null
    Add-Text $Page ($xC23 - 0.50) 2.46 1.10 0.14 "c23 2.069 +/- 0.134" $ink 5.5 $false "1" "1" $Script:FontMono | Out-Null
    Add-Text $Page ($xA1 - 0.46) 2.04 1.16 0.14 "A1 1.811 +/- 0.006" $teal 5.7 $true "1" "1" $Script:FontMono | Out-Null
    Add-Text $Page 0.84 1.36 2.70 0.18 "~20x lower seed s.d. across three seeds." $teal 6.2 $true "0" "1" | Out-Null

    Add-PanelLabel $Page "c" 4.12 3.50 $red
    Add-Text $Page 4.48 3.47 2.72 0.20 "Negative control chain" $ink 7.2 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Text $Page 4.48 3.22 2.92 0.16 "Fixed priors improve Val but worsen Test." $muted 5.8 $false "0" "1" | Out-Null
    $plotX = 4.58
    $plotY = 1.86
    $plotW = 2.36
    $plotH = 0.84
    Add-Line $Page $plotX $plotY ($plotX + $plotW) $plotY $grid 0.65 $false | Out-Null
    Add-Line $Page $plotX $plotY $plotX ($plotY + $plotH) $grid 0.65 $false | Out-Null
    $names = @("A1", "A2", "A3", "A4", "A5")
    $vals = @(1.811, 1.819, 1.843, 1.863, 1.947)
    $prevX = $null
    $prevY = $null
    for ($i = 0; $i -lt $names.Count; $i++) {
        $px = $plotX + 0.20 + $i * 0.50
        $py = Map-Value $vals[$i] 1.80 1.96 $plotY $plotH
        if ($prevX -ne $null) {
            Add-Line $Page $prevX $prevY $px $py $red 0.85 $false | Out-Null
        }
        $dotFill = $(if ($i -eq 0) { $teal } else { $redSoft })
        $dotLine = $(if ($i -eq 0) { $teal } else { $red })
        Add-Dot $Page ($px - 0.05) ($py - 0.05) 0.10 "" $dotFill $dotLine | Out-Null
        Add-Text $Page ($px - 0.15) 1.62 0.30 0.12 $names[$i] $muted 5.2 $false "1" "1" $Script:FontMono | Out-Null
        $prevX = $px
        $prevY = $py
    }
    Add-Text $Page 4.52 1.17 2.80 0.18 "C12: complexity is an overfitting amplifier here." $red 6.2 $true "0" "1" | Out-Null
    Add-Text $Page 6.48 2.70 0.56 0.14 "1.947" $red 5.2 $true "1" "1" $Script:FontMono | Out-Null
    Add-Text $Page 4.55 2.02 0.56 0.14 "1.811" $teal 5.2 $true "1" "1" $Script:FontMono | Out-Null

    Add-PanelLabel $Page "d" 7.76 3.50 $amber
    Add-Text $Page 8.12 3.47 2.62 0.20 "Open gate, not a result" $ink 7.2 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Text $Page 8.12 3.22 2.82 0.16 "Phase B remains deliberately amber." $muted 5.8 $false "0" "1" | Out-Null
    Add-Box $Page 8.12 2.58 0.96 0.40 "MCP`npretrain" $amberSoft $amber $ink 5.8 $true 0.8 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Box $Page 9.46 2.58 0.96 0.40 "target`nadapt" $amberSoft $amber $ink 5.8 $true 0.8 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Box $Page 10.20 1.98 0.72 0.38 "PASS`n< A1" "#FFFFFF" $amber $amber 5.7 $true 0.75 "1" $true $Script:FontMono 0.04 | Out-Null
    Add-Line $Page 9.08 2.78 9.46 2.78 $amber 0.9 $true | Out-Null
    Add-Line $Page 10.42 2.58 10.56 2.36 $amber 0.9 $true | Out-Null
    Add-Text $Page 8.12 1.68 2.92 0.18 "Gate threshold: A1 mean Test MAE = 1.811e-3." $muted 5.8 $false "0" "1" | Out-Null
    Add-Text $Page 8.12 1.30 2.92 0.28 "Do not cite Phase B as evidence until repaired protocol and target-portfolio tests pass." $amber 5.9 $true "0" "1" | Out-Null

    Add-Rule $Page 0.42 0.94 10.98 "#DAD5CB" 0.75
    Add-Text $Page 0.42 0.61 10.9 0.18 "Claim binding: C08-C12 are supported; Phase B is shown as a gated research branch, not a reported result." $muted 5.8 $false "0" "1" | Out-Null
}

function Build-ArchitecturePage {
    param($Page)
    Configure-Page $Page "Fig2 Architecture"

    $ink = "#15181B"
    $muted = "#667078"
    $grid = "#D6DADD"
    $soft = "#F3F0EA"
    $teal = "#249089"
    $tealSoft = "#E8F4F2"
    $blue = "#276FB7"
    $blueSoft = "#EAF1FA"
    $red = "#C44E52"
    $redSoft = "#F8E7E3"
    $amber = "#B77A22"
    $amberSoft = "#F6E9D0"
    $violet = "#596AA6"
    $violetSoft = "#ECEEFA"

    Add-Header $Page "FIG 2" "Editable A1 architecture and training branch" "The main model stays intentionally simple; rejected additions and Phase-B training are subordinate annotations." $violet

    Add-PanelLabel $Page "a" 0.48 6.10 $ink
    Add-Text $Page 0.84 6.08 2.82 0.20 "A1/iGT main path" $ink 7.6 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Text $Page 0.84 5.84 5.7 0.16 "Five component histories and three weather summaries become variable tokens before inverted attention." $muted 5.9 $false "0" "1" | Out-Null

    Add-Step $Page "1" 0.92 5.43 "inputs" $blue
    Add-Box $Page 0.92 4.84 1.28 0.44 "component`nhistory" $blueSoft $blue $ink 6.0 $true 0.85 "1" $false $Script:FontBody 0.045 | Out-Null
    Add-Box $Page 0.92 4.14 1.28 0.44 "future`nweather" $blueSoft $blue $ink 6.0 $true 0.85 "1" $false $Script:FontBody 0.045 | Out-Null

    Add-Step $Page "2" 2.70 5.43 "batched encoders" $teal
    Add-Box $Page 2.70 4.84 1.26 0.44 "GRU`n5 channels" $tealSoft $teal $ink 6.0 $true 0.85 "1" $false $Script:FontBody 0.045 | Out-Null
    Add-Box $Page 2.70 4.14 1.26 0.44 "MLP`n3 summaries" $tealSoft $teal $ink 6.0 $true 0.85 "1" $false $Script:FontBody 0.045 | Out-Null
    Add-Line $Page 2.20 5.06 2.70 5.06 "#9DA7AE" 0.85 $true | Out-Null
    Add-Line $Page 2.20 4.36 2.70 4.36 "#9DA7AE" 0.85 $true | Out-Null

    Add-Step $Page "3" 4.42 5.43 "8 variable tokens" $teal
    $tokenX = 4.42
    $tokenY1 = 4.96
    $tokenY2 = 4.26
    $d = 0.30
    $g = 0.36
    Add-Dot $Page $tokenX $tokenY1 $d "L" $tealSoft $teal $ink 5.7 $true | Out-Null
    Add-Dot $Page ($tokenX + $g) $tokenY1 $d "PV" $tealSoft $teal $ink 5.1 $true | Out-Null
    Add-Dot $Page ($tokenX + 2*$g) $tokenY1 $d "W" $tealSoft $teal $ink 5.7 $true | Out-Null
    Add-Dot $Page ($tokenX + 3*$g) $tokenY1 $d "B" $tealSoft $teal $ink 5.7 $true | Out-Null
    Add-Dot $Page ($tokenX + 4*$g) $tokenY1 $d "S" $tealSoft $teal $ink 5.7 $true | Out-Null
    Add-Dot $Page ($tokenX + 0.5*$g) $tokenY2 $d "T" $blueSoft $blue $ink 5.7 $true | Out-Null
    Add-Dot $Page ($tokenX + 1.95*$g) $tokenY2 $d "G" $blueSoft $blue $ink 5.7 $true | Out-Null
    Add-Dot $Page ($tokenX + 3.40*$g) $tokenY2 $d "V" $blueSoft $blue $ink 5.7 $true | Out-Null
    Add-Line $Page 3.96 5.06 4.42 5.10 "#9DA7AE" 0.85 $true | Out-Null
    Add-Line $Page 3.96 4.36 4.60 4.41 "#9DA7AE" 0.85 $true | Out-Null

    Add-Step $Page "4" 6.80 5.43 "attention" $violet
    Add-Box $Page 6.80 4.52 1.44 0.66 "inverted`nself-attention`n2 layers, 8 heads" $violetSoft $violet $ink 5.9 $true 0.9 "1" $false $Script:FontBody 0.045 | Out-Null
    Add-Line $Page 5.94 4.92 6.80 4.85 "#9DA7AE" 0.85 $true | Out-Null

    Add-Step $Page "5" 8.78 5.43 "forecast" $amber
    Add-Box $Page 8.78 4.94 1.02 0.38 "shared FFN`n96 steps" "#FFFFFF" "#9DA7AE" $ink 5.8 $true 0.7 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Box $Page 8.78 4.26 1.02 0.38 "real-unit`nbalance" $amberSoft $amber $ink 5.8 $true 0.85 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Box $Page 10.20 4.42 0.86 0.56 "net`nforecast" "#FFFFFF" "#15181B" $ink 6.1 $true 0.85 "1" $false $Script:FontBody 0.05 | Out-Null
    Add-Line $Page 8.24 4.85 8.78 5.10 "#9DA7AE" 0.85 $true | Out-Null
    Add-Line $Page 9.29 4.94 9.29 4.64 "#9DA7AE" 0.75 $true | Out-Null
    Add-Line $Page 9.80 4.45 10.20 4.70 "#9DA7AE" 0.85 $true | Out-Null
    Add-Text $Page 8.58 3.94 1.80 0.16 "Pnet = L - PV - W + B" $ink 6.3 $true "1" "1" $Script:FontMono | Out-Null

    Add-Rule $Page 0.66 3.64 10.74 "#DAD5CB" 0.75

    Add-PanelLabel $Page "b" 0.48 3.26 $teal
    Add-Text $Page 0.84 3.23 2.30 0.20 "Model contract" $ink 7.2 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Box $Page 0.84 2.70 2.18 0.30 "zero fixed physics priors" "#FFFFFF" "#9DA7AE" $ink 5.9 $true 0.55 "1" $false $Script:FontBody 0.035 | Out-Null
    Add-Box $Page 0.84 2.30 2.18 0.30 "zero component loss" "#FFFFFF" "#9DA7AE" $ink 5.9 $true 0.55 "1" $false $Script:FontBody 0.035 | Out-Null
    Add-Box $Page 0.84 1.90 2.18 0.30 "net-MSE only" "#FFFFFF" "#9DA7AE" $ink 5.9 $true 0.55 "1" $false $Script:FontBody 0.035 | Out-Null
    Add-Box $Page 0.84 1.50 2.18 0.30 "1.3M params; 700+ samples/s" "#FFFFFF" "#9DA7AE" $ink 5.9 $true 0.55 "1" $false $Script:FontBody 0.035 | Out-Null
    Add-Text $Page 0.84 1.08 2.34 0.18 "This contract is the key thesis constraint, not a placeholder." $muted 5.6 $false "0" "1" | Out-Null

    Add-PanelLabel $Page "c" 4.12 3.26 $red
    Add-Text $Page 4.48 3.23 2.52 0.20 "Rejected additions" $ink 7.2 $true "0" "1" $Script:FontDisplay | Out-Null
    $rX = 4.48
    $rY = 2.70
    Add-Box $Page $rX $rY 0.54 0.30 "A2" $redSoft $red $red 5.8 $true 0.65 "1" $false $Script:FontMono 0.035 | Out-Null
    Add-Box $Page ($rX + 0.70) $rY 0.54 0.30 "A3" $redSoft $red $red 5.8 $true 0.65 "1" $false $Script:FontMono 0.035 | Out-Null
    Add-Box $Page ($rX + 1.40) $rY 0.54 0.30 "A4" $redSoft $red $red 5.8 $true 0.65 "1" $false $Script:FontMono 0.035 | Out-Null
    Add-Box $Page ($rX + 2.10) $rY 0.54 0.30 "A5" $redSoft $red $red 5.8 $true 0.65 "1" $false $Script:FontMono 0.035 | Out-Null
    Add-Text $Page 4.48 2.28 3.00 0.25 "+ physics token -> + twin/constraint tokens -> + graph bias -> + horizon decoder" $muted 5.7 $false "0" "1" | Out-Null
    Add-Text $Page 4.48 1.86 3.00 0.18 "Test MAE worsens monotonically: 1.819, 1.843, 1.863, 1.947." $red 5.9 $true "0" "1" | Out-Null
    Add-Text $Page 4.48 1.42 3.00 0.24 "Interpretation: stronger Val gains predict worse Test transfer." $muted 5.8 $false "0" "1" | Out-Null

    Add-PanelLabel $Page "d" 7.76 3.26 $amber
    Add-Text $Page 8.12 3.23 2.82 0.20 "Phase-B training branch" $ink 7.2 $true "0" "1" $Script:FontDisplay | Out-Null
    Add-Box $Page 8.12 2.62 0.98 0.36 "mask 1-2`ncomponents" $amberSoft $amber $ink 5.7 $true 0.75 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Box $Page 9.46 2.62 0.98 0.36 "reconstruct`nmasked parts" $amberSoft $amber $ink 5.7 $true 0.75 "1" $false $Script:FontBody 0.04 | Out-Null
    Add-Box $Page 8.82 1.86 1.06 0.36 "net-loss`nanchor" "#FFFFFF" $amber $amber 5.7 $true 0.75 "1" $true $Script:FontMono 0.04 | Out-Null
    Add-Line $Page 9.10 2.80 9.46 2.80 $amber 0.85 $true | Out-Null
    Add-Line $Page 8.58 2.62 9.04 2.22 $amber 0.85 $true $true | Out-Null
    Add-Text $Page 8.12 1.32 2.92 0.30 "Shown as a branch because target-portfolio adaptation remains unresolved." $amber 5.9 $true "0" "1" | Out-Null

    Add-Rule $Page 0.42 0.86 10.98 "#DAD5CB" 0.75
    Add-Text $Page 0.42 0.55 10.9 0.18 "Abbreviations: L, load; PV, photovoltaic; W, wind; B, battery power; S, battery SOC; T, temperature; G, irradiance; V, wind speed." $muted 5.7 $false "0" "1" | Out-Null
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

$visio = Invoke-WithComRetry { New-Object -ComObject Visio.Application }
Invoke-WithComRetry { $script:visio.Visible = $true } | Out-Null
Start-Sleep -Seconds 1
Invoke-WithComRetry { $script:visio.AlertResponse = 7 } | Out-Null

$doc = Invoke-WithComRetry { $script:visio.Documents.Add("") }
$page1 = Invoke-WithComRetry { $script:visio.ActivePage }
Build-ResearchLogicPage $page1

$page2 = Invoke-WithComRetry { $script:doc.Pages.Add() }
Build-ArchitecturePage $page2

if (Test-Path -LiteralPath $resolvedOutput) {
    Remove-Item -LiteralPath $resolvedOutput -Force
}

Invoke-WithComRetry { $script:doc.SaveAs($script:resolvedOutput) } | Out-Null

$previewPdf = Join-Path $resolvedPreview "physformer_nature_style.pdf"
$previewPng1 = Join-Path $resolvedPreview "physformer_nature_style-1.png"
$previewPng2 = Join-Path $resolvedPreview "physformer_nature_style-2.png"
foreach ($previewFile in @($previewPdf, $previewPng1, $previewPng2)) {
    if (Test-Path -LiteralPath $previewFile) {
        Remove-Item -LiteralPath $previewFile -Force
    }
}

Invoke-WithComRetry { $script:doc.ExportAsFixedFormat(1, $script:previewPdf, 1, 0) } | Out-Null

$pdftoppm = Get-Command pdftoppm -ErrorAction SilentlyContinue
if ($pdftoppm) {
    $pngPrefix = Join-Path $resolvedPreview "physformer_nature_style"
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
    $visio.ActiveWindow.Page = $page1
} catch {
}

Invoke-WithComRetry { $script:doc.Close() } | Out-Null
Invoke-WithComRetry { $script:visio.Quit() } | Out-Null

Write-Output "SAVED_VSDX=$resolvedOutput"
Write-Output "PREVIEW_PDF=$previewPdf"
Write-Output "PREVIEW_PNG_1=$previewPng1"
Write-Output "PREVIEW_PNG_2=$previewPng2"
Write-Output "VISIO_CLOSED=True"
