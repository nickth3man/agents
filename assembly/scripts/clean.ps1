#Requires -Version 5.1
<#
    clean.ps1 - remove build outputs (out\, dist\, generated\).
    All three are gitignored build artifacts per the repo conventions.
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$StopServer
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

if ($StopServer) {
    & "$PSScriptRoot\stop.ps1"
    # Also kill any remaining chat-agent.exe instances (stop.ps1 kills by port;
    # this catches orphaned processes not bound to 8080/8081).
    Get-Process -Name 'chat-agent' -ErrorAction SilentlyContinue | ForEach-Object {
        try { Stop-Process -Id $_.Id -Force -ErrorAction Stop; Write-Host "[clean] killed stray pid $($_.Id)" } catch {}
    }
    # Wait for all chat-agent processes to exit (OS file handles may persist briefly).
    for ($i = 0; $i -lt 10; $i++) {
        $remaining = Get-Process -Name 'chat-agent' -ErrorAction SilentlyContinue
        if (-not $remaining) { break }
        Start-Sleep -Milliseconds 200
    }
}

$dirs    = @('out','dist','generated')
$removed = 0

foreach ($d in $dirs) {
    $path = Join-Path $repoRoot $d
    if (-not (Test-Path $path)) {
        if ($DryRun) { Write-Host "[clean] would skip $path (not found)" }
        continue
    }
    if ($DryRun) {
        Write-Host "[clean] would remove $path"
        $removed++
        continue
    }
    try {
        Write-Host "[clean] removing $path"
        $ok = $false
        for ($retry = 0; $retry -lt 3; $retry++) {
            try {
                Remove-Item -Path $path -Recurse -Force -ErrorAction Stop
                $ok = $true
                break
            } catch {
                if ($retry -lt 2) { Start-Sleep -Milliseconds 500 }
            }
        }
        if (-not $ok) { throw "failed after 3 attempts" }
        $removed++
    } catch {
        Write-Host "[clean] ERROR: could not remove $path - $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "[clean]   (a running server may hold out\current\chat-agent.exe; try -StopServer)" -ForegroundColor Yellow
        exit 1
    }
}

# Recreate empty generated\ scaffold (build.ps1 writes version.inc / .buildcount
# into it but does not create the directory itself).
if (-not $DryRun -and (Test-Path (Join-Path $repoRoot 'generated')) -eq $false) {
    New-Item -ItemType Directory -Path (Join-Path $repoRoot 'generated') -Force | Out-Null
}

if ($DryRun) {
    Write-Host "[clean] dry-run complete (would remove $removed dir(s))"
} else {
    Write-Host "[clean] done (removed $removed dir(s))"
}
exit 0
