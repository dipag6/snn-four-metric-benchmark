# MR604 four-metric SNN benchmark -- full run on a CUDA GPU machine (Windows).
#
#   .\run_gpu.ps1              # resume: skips anything already in results\
#   .\run_gpu.ps1 -Force       # retrain everything from scratch
#
# Runs are ordered cheapest-first by RESULT ROWS PER GPU-HOUR, so an interrupted
# session loses the expensive tail rather than the productive head. Every run
# checkpoints per epoch, so re-running this script always resumes.

param(
    [switch]$Force,
    [int]$Epochs   = 64,
    [string]$Dataset = "mnist",
    [int[]]$Seeds  = @(42, 123, 789, 456, 1337),
    [string]$Results,
    [string]$Ckpts,
    [string]$Tables,
    [string]$Figures,
    [string]$Data
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

if (-not $Results) { $Results = Join-Path $PSScriptRoot "results\$Dataset" }
if (-not $Ckpts)   { $Ckpts   = Join-Path $PSScriptRoot "checkpoints\$Dataset" }
if (-not $Tables)  { $Tables  = Join-Path $PSScriptRoot "tables\$Dataset" }
if (-not $Figures) { $Figures = Join-Path $PSScriptRoot "figures\$Dataset" }
if (-not $Data)    { $Data    = Join-Path $PSScriptRoot "data" }

$ForceFlag = if ($Force) { @("--force") } else { @() }

foreach ($d in @($Results, $Ckpts, $Tables, $Figures, $Data)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

$Common = @("--epochs", $Epochs, "--dataset", $Dataset, "--out-dir", $Results,
            "--ckpt-dir", $Ckpts, "--data-root", $Data)

function Write-Rule { Write-Host ("=" * 78) }

# ---------------------------------------------------------------- preflight --
Write-Rule; Write-Host "MR604 benchmark -- preflight"; Write-Rule

python -c @"
import sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: no CUDA GPU visible. This benchmark is not viable on CPU.')
print('GPU  :', torch.cuda.get_device_name(0))
print('VRAM :', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
print('torch:', torch.__version__)
"@
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "results  -> $Results"
Write-Host "ckpts    -> $Ckpts"
Write-Host "epochs   -> $Epochs"
Write-Host ""

# Operation counts underpin every energy number. Refuse to train if they disagree.
python run.py verify
if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: operation count verification failed."; exit 1 }

# MNIST: ~11 MB, fetched once, verified.
python run.py data --dataset $Dataset --data-root $Data
if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: dataset check failed."; exit 1 }

python tools\check_backends.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: backend consistency check failed -- consider adding" `
               "'--backend native' to the Invoke-Run calls below."
}

# --------------------------------------------------------------------- runs --
# run.py 'one' trains a single (pathway, T, seed). It skips automatically when the
# result json already exists, so completed work is never repeated.
function Invoke-Run {
    param([string]$Pathway, [int]$T, [int]$Seed)
    Write-Host ""; Write-Rule
    Write-Host "### $Pathway T=$T seed=$Seed"
    Write-Rule
    python run.py one --pathway $Pathway --T $T --seed $Seed @Common @ForceFlag
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!!! ${Pathway}_T${T}_s${Seed} FAILED -- continuing"
    }
}

# 1. ANN baseline -- 3 runs, ~4 min each, 3 result rows.
foreach ($s in $Seeds) { Invoke-Run -Pathway ann -T 0 -Seed $s }

# 2. Path B / QCFS -- 9 runs, ~5 min each, TWO rows each (rate + direct) = 18 rows.
#    Highest yield per GPU-hour in the whole grid. Always do these before STBP.
foreach ($T in @(2, 4, 8)) {
    foreach ($s in $Seeds) { Invoke-Run -Pathway qcfs -T $T -Seed $s }
}

# 3. Path A / STBP -- 9 runs, increasingly expensive, 1 row each.
foreach ($s in $Seeds) { Invoke-Run -Pathway stbp -T 4  -Seed $s }
foreach ($s in $Seeds) { Invoke-Run -Pathway stbp -T 8  -Seed $s }
foreach ($s in $Seeds) { Invoke-Run -Pathway stbp -T 16 -Seed $s }   # ~70 min/seed

# ----------------------------------------------------------------- analysis --
Write-Host ""; Write-Rule; Write-Host "Validity gates"; Write-Rule
python run.py gates --out-dir $Results

Write-Host ""; Write-Rule; Write-Host "Tables and figures"; Write-Rule
python analyse.py --out-dir $Results --tables-dir $Tables --figures-dir $Figures

$n = (Get-ChildItem $Results -Filter *.json |
      Where-Object { $_.Name -notlike "_*" }).Count
Write-Host ""
Write-Rule
Write-Host "Done. $n result rows in $Results (expect 30 for a complete grid: 21 runs,"
Write-Host "with each QCFS run contributing both a rate and a direct encoding row)."
Write-Host "Chapter 5 tables: $Tables\thesis_tables.md"
Write-Rule
