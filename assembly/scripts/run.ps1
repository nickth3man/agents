#Requires -Version 5.1
<#
    run.ps1 - start the asm-chat relay + server DETACHED.
    The terminal returns immediately; both processes keep running after the
    terminal closes. Logs go to out\logs\.
#>
param(
    [string]$Exe = (Join-Path $PSScriptRoot '..\out\current\chat-agent.exe')
)
$ErrorActionPreference = 'Stop'
$root   = Split-Path -Parent $PSScriptRoot
$relay  = Join-Path $PSScriptRoot 'relay.py'
$logDir = Join-Path $root 'out\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path $Exe)) { throw "server exe not found: $Exe (run build.ps1 first)" }
if (-not (Test-Path $relay)) { throw "relay not found: $relay" }

# Start relay (Python) detached, hidden window. Redirect BOTH stdout and stderr
# so the children do not inherit the caller's console pipes (which would keep a
# wrapping terminal's read open after run.ps1 returns).
$rp = Start-Process -FilePath 'python' -ArgumentList @($relay) `
     -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput (Join-Path $logDir 'relay.out.log') `
     -RedirectStandardError  (Join-Path $logDir 'relay.err.log')
Start-Sleep -Milliseconds 800

# Start the assembly server detached, hidden window.
$sp = Start-Process -FilePath $Exe -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput (Join-Path $logDir 'server.out.log') `
     -RedirectStandardError  (Join-Path $logDir 'server.err.log')
Start-Sleep -Milliseconds 700

$v  = & curl.exe -s --max-time 3 http://127.0.0.1:8080/version
$rh = & curl.exe -s --max-time 3 http://127.0.0.1:8081/

Write-Host ""
Write-Host "asm-chat running (detached)"
Write-Host "  browser   : http://127.0.0.1:8080/"
Write-Host "  relay     : $rh"
Write-Host "  build id  : $v"
Write-Host "  pids      : relay=$($rp.Id)  server=$($sp.Id)"
Write-Host "  logs      : $logDir\{relay,server}.log"
Write-Host ""
Write-Host "stop with:  powershell -NoProfile -File .\scripts\stop.ps1"
Write-Host ""
