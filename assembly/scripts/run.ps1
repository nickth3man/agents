#Requires -Version 5.1
<#
    run.ps1 - start the native Assembly chat agent DETACHED.
    The terminal returns immediately; the process keeps running after the
    terminal closes. Logs go to out\logs\.
#>
param(
    [string]$Exe = (Join-Path $PSScriptRoot '..\out\current\chat-agent.exe')
)
$ErrorActionPreference = 'Stop'
$root   = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root 'out\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path $Exe)) { throw "server exe not found: $Exe (run build.ps1 first)" }
$envFile = Join-Path (Split-Path -Parent $root) '.env'
if (-not (Test-Path $envFile)) { throw "shared .env not found: $envFile" }
foreach ($line in Get-Content $envFile) {
    $line = $line.Trim()
    if (-not $line -or $line.StartsWith('#') -or -not $line.Contains('=')) { continue }
    $name, $value = $line.Split('=', 2)
    $name = $name.Trim(); $value = $value.Trim().Trim('"').Trim("'")
    if ($name -in @('OPENROUTER_MODEL','OPENROUTER_API_KEY')) {
        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}
if (-not $env:OPENROUTER_MODEL -or -not $env:OPENROUTER_API_KEY) {
    throw 'OPENROUTER_MODEL and OPENROUTER_API_KEY must be set in the shared .env'
}

# Start the assembly server detached, hidden window.
$sp = Start-Process -FilePath $Exe -WindowStyle Hidden -PassThru `
     -WorkingDirectory $root `
     -RedirectStandardOutput (Join-Path $logDir 'server.out.log') `
     -RedirectStandardError  (Join-Path $logDir 'server.err.log')
Start-Sleep -Milliseconds 700

$v  = & curl.exe -s --max-time 3 http://127.0.0.1:8080/version
$health = & curl.exe -s --max-time 3 http://127.0.0.1:8080/health

Write-Host ""
Write-Host "asm-chat running (detached)"
Write-Host "  browser   : http://127.0.0.1:8080/"
Write-Host "  health    : $health"
Write-Host "  model     : $env:OPENROUTER_MODEL"
Write-Host "  build id  : $v"
Write-Host "  pid       : $($sp.Id)"
Write-Host "  logs      : $logDir\server.{out,err}.log"
Write-Host ""
Write-Host "stop with:  powershell -NoProfile -File .\scripts\stop.ps1"
Write-Host ""
