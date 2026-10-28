param([switch]$Installer, [switch]$Full, [string]$Python)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $project
if (-not $Python) { $Python = if ($env:DESKPILOT_BUILD_PYTHON) { $env:DESKPILOT_BUILD_PYTHON } else { (Get-Command python).Source } }
if ($Full) { $env:DESKPILOT_BUNDLE_LOCAL_MODELS = '1' } else { $env:DESKPILOT_BUNDLE_LOCAL_MODELS = '0' }
& $Python -m PyInstaller --noconfirm --clean (Join-Path $PSScriptRoot 'deskpilot.spec')
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed' }
$exe = Join-Path $project 'dist\DeskPilot\DeskPilot.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing release executable: $exe" }
if ($Installer) {
    $iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    $compiler = if ($iscc) { $iscc.Source } else { Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe' }
    if (-not (Test-Path -LiteralPath $compiler)) { throw 'Inno Setup ISCC.exe is not installed' }
    & $compiler /Q (Join-Path $PSScriptRoot 'DeskPilot.iss')
    if ($LASTEXITCODE -ne 0) { throw 'Inno Setup build failed' }
}
