param(
  [string]$TexFile = "report.tex"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $root
Set-Location $repo

if (!(Test-Path $TexFile)) {
  throw "Input TeX file not found: $TexFile"
}

$tectonicDir = Join-Path $repo "tectonic_bin"
$tectonicExe = Join-Path $tectonicDir "tectonic.exe"

if (!(Test-Path $tectonicExe)) {
  New-Item -ItemType Directory -Force -Path $tectonicDir | Out-Null

  # Known-good Windows MSVC build (continuous). Update if needed.
  $zipUrl = "https://github.com/tectonic-typesetting/tectonic/releases/download/continuous/tectonic-0.15.0%2B20251006-x86_64-pc-windows-msvc.zip"
  $zipPath = Join-Path $repo "tectonic.zip"

  Write-Host "Downloading Tectonic..."
  Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath

  Write-Host "Extracting Tectonic..."
  Expand-Archive -Path $zipPath -DestinationPath $tectonicDir -Force
  Remove-Item -Force $zipPath
}

Write-Host "Compiling $TexFile ..."
& $tectonicExe -p --keep-logs --keep-intermediates $TexFile
