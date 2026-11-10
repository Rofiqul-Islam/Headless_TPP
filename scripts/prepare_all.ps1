# Windows PowerShell equivalent of scripts/prepare_all.sh
# Run from the project root with: powershell -ExecutionPolicy Bypass -File .\scripts\prepare_all.ps1
# Internet access is required for EasyTPP and Amazon Reviews 2023 downloads.
param(
    [string]$AmazonCategory = $(if ($env:AMAZON_CATEGORY) { $env:AMAZON_CATEGORY } else { 'Cell_Phones_and_Accessories' })
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

# Use Python from the active Conda environment, not an unrelated system Python.
$Python = (Get-Command python -ErrorAction Stop).Source
Write-Host "Project: $ProjectRoot"
Write-Host "Python: $Python"

function Invoke-Preparation {
    param([string[]]$Arguments)
    Write-Host "`n> python $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Python 'scripts/prepare_data.py' @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset preparation failed (exit code $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

# RQ1 / E1a: synthetic Hawkes with known likelihood
Invoke-Preparation @('hawkes', '--out', 'data/hawkes', '--K', '3', '--T', '60')

# RQ1 / E1c: short observation windows
Invoke-Preparation @('hawkes', '--out', 'data/hawkes_short', '--K', '3', '--T', '6', '--n_train', '6000', '--n_val', '800', '--n_test', '1500', '--seed', '1')

# RQ1 / E1b: EasyTPP benchmark datasets
foreach ($Dataset in @('retweet', 'stackoverflow', 'taobao', 'amazon')) {
    Invoke-Preparation @('easytpp', '--name', $Dataset, '--out', "data/$Dataset")
}

# RQ1 / E1c: randomly censored Taobao dataset
Invoke-Preparation @('easytpp', '--name', 'taobao', '--out', 'data/taobao_cut', '--censor', 'random_cut', '--split_long', '32')

# RQ3: Amazon Reviews 2023 with text
Invoke-Preparation @('amazon-download', '--category', $AmazonCategory, '--out', 'raw/amazon')

# Match either .jsonl or .jsonl.gz, as the Linux script does.
$ReviewFiles = @(Get-ChildItem -LiteralPath 'raw/amazon' -File | Where-Object {
    $_.Name -eq "$AmazonCategory.jsonl" -or $_.Name -eq "$AmazonCategory.jsonl.gz"
} | Sort-Object Name)
$MetaFiles = @(Get-ChildItem -LiteralPath 'raw/amazon' -File | Where-Object {
    $_.Name -eq "meta_$AmazonCategory.jsonl" -or $_.Name -eq "meta_$AmazonCategory.jsonl.gz"
} | Sort-Object Name)
if ($ReviewFiles.Count -eq 0) { throw "Amazon reviews file not found for $AmazonCategory in raw/amazon" }
if ($MetaFiles.Count -eq 0) { throw "Amazon metadata file not found for $AmazonCategory in raw/amazon" }

Invoke-Preparation @('amazon', '--reviews', $ReviewFiles[0].FullName, '--meta', $MetaFiles[0].FullName, '--out', 'data/amazon_phones', '--cat_level', '2', '--K', '24', '--min_events', '5', '--max_users', '20000')

Write-Host "`nAll datasets ready in data/" -ForegroundColor Green
