param(
    [ValidateSet('demo','test','study','app')][string]$Mode = 'demo'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.cache/matplotlib'
$env:HF_HOME = Join-Path $PSScriptRoot '.cache/huggingface'
$env:TORCH_HOME = Join-Path $PSScriptRoot '.cache/torch'
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE. See the output above." }
}
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) {
    Invoke-Checked 'py' @('-3.13','-m','venv','.venv')
}
Invoke-Checked $python @('-c','import sys; assert sys.version_info[:2] == (3, 13), "Python 3.13 is required"')
$lockHash = (Get-FileHash -LiteralPath 'requirements-lock.txt' -Algorithm SHA256).Hash
$stamp = Join-Path $PSScriptRoot '.venv/visualguard-lock.sha256'
if (!(Test-Path -LiteralPath $stamp) -or (Get-Content -LiteralPath $stamp -Raw).Trim() -ne $lockHash) {
    Invoke-Checked $python @('-m','pip','install','--upgrade','pip==26.3')
    Invoke-Checked $python @('-m','pip','install','-r','requirements-lock.txt')
    Invoke-Checked $python @('-m','pip','install','-e','.','--no-deps')
    Invoke-Checked $python @('-m','pip','check')
    Set-Content -LiteralPath $stamp -Value $lockHash
}
if ($Mode -eq 'test') {
    Invoke-Checked $python @('-m','ruff','check','.')
    Invoke-Checked $python @('-m','ruff','format','--check','.')
    Invoke-Checked $python @('-m','pytest','-q')
    exit
}
if ($Mode -eq 'study') {
    Invoke-Checked $python @('scripts/download_mvtec.py')
    Invoke-Checked $python @('-c',"import timm; timm.create_model('resnet18.tv_in1k', pretrained=True)")
    $run = 'runs/study-' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff')
    Invoke-Checked $python @('scripts/run_study.py','--output-root',$run)
    Invoke-Checked $python @('scripts/export_report.py','--root',"$run/reports")
    $env:VISUALGUARD_MODEL_DIR = Join-Path $PSScriptRoot "$run/artifacts"
}
if ($Mode -eq 'demo') {
    if (!(Test-Path -LiteralPath 'local_data/synthetic')) {
        Invoke-Checked $python @('-m','visualguard.cli','synthetic','--output','local_data/synthetic')
    }
    if (!(Test-Path -LiteralPath 'runs/demo.csv')) {
        Invoke-Checked $python @('-m','visualguard.cli','prepare','--root','local_data/synthetic','--category','synthetic_part','--manifest','runs/demo.csv')
    }
    if (!(Test-Path -LiteralPath 'artifacts/demo')) {
        Invoke-Checked $python @('-m','visualguard.cli','train','--root','local_data/synthetic','--manifest','runs/demo.csv','--config','configs/smoke.yaml','--output','artifacts/demo')
    }
    $env:VISUALGUARD_MODEL_DIR = Join-Path $PSScriptRoot 'artifacts/demo'
    Write-Host 'Software demonstration with synthetic images. Upload local_data/synthetic/synthetic_part/test/scratch/000.png.'
}
Invoke-Checked $python @('-m','streamlit','run','app/streamlit_app.py')
