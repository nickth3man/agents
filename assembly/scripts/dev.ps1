#Requires -Version 5.1
<#
    dev.ps1 - watch -> rebuild -> restart -> test -> crash-recover (PLAN §5.2).
    Watches src/, include/, web/ for changes; debounces; serializes builds;
    keeps the old server running during a failed build; restarts the new unique
    exe on success; runs test.ps1; marks last-known-good; restarts LKG on
    unexpected exit with crash-loop suppression. Ctrl-C to exit.
#>
param(
    [string[]]$Watch = @('src','include','web'),
    [int]$DebounceMs = 250,
    [int]$CrashWindowSec = 10,
    [int]$CrashMax = 3,
    [int]$BindRetries = 15,
    [switch]$NoTest
)
$ErrorActionPreference = 'Stop'
$root   = Split-Path -Parent $PSScriptRoot
Set-Location $root
$logDir = Join-Path $root 'out\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$buildPs1 = Join-Path $PSScriptRoot 'build.ps1'
$testPs1  = Join-Path $PSScriptRoot 'test.ps1'

# Cross-runspace shared state
$state = [hashtable]::Synchronized(@{
    Pending=$false; LastChange=[DateTime]::UtcNow; Building=$false; Crashed=$false
    IgnoreRe='(\\|/)(generated|out|dist|\.git|\.vscode|node_modules)(\\|/)|\.tmp$|~$|\.bak$|\.log$|\.pyc$'
    WatchDirs=$Watch; Root=$root
})
$global:ServerProc = $null
$global:LkgExe     = $null
$global:LkgBuild   = $null
$global:Stopping   = $false
$global:CrashLog   = @()

# ---- filesystem watcher ----
$watcher = New-Object System.IO.FileSystemWatcher $root
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = 'FileName,LastWrite,Size,CreationTime'
$watcher.EnableRaisingEvents = $true
$action = {
    $path = $Event.SourceEventArgs.FullPath
    if (-not $path) { return }
    if ($path -match $state.IgnoreRe) { return }
    $rel = $path.Substring($state.Root.Length).TrimStart('\','/')
    $ok = $false
    foreach ($d in $state.WatchDirs) { if ($rel -like "$d/*" -or $rel -like "$d\*") { $ok=$true; break } }
    if (-not $ok) { return }
    $state.Pending = $true
    $state.LastChange = [DateTime]::UtcNow
}
Register-ObjectEvent -InputObject $watcher -EventName Changed -Action $action -SourceIdentifier devChanged | Out-Null
Register-ObjectEvent -InputObject $watcher -EventName Created -Action $action -SourceIdentifier devCreated | Out-Null
Register-ObjectEvent -InputObject $watcher -EventName Renamed -Action $action -SourceIdentifier devRenamed | Out-Null

# ---- helpers ----
function Wait-Version {
    param([string]$Expected, [int]$TimeoutSec = 12)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try { $v = & curl.exe -s --max-time 2 'http://127.0.0.1:8080/version'; if ($v -eq $Expected) { return $true } } catch {}
        Start-Sleep -Milliseconds 200
    }
    return $false
}
function Stop-Current {
    if ($global:ServerProc -and -not $global:ServerProc.HasExited) {
        $global:Stopping = $true
        try { Stop-Process -Id $global:ServerProc.Id -Force -ErrorAction Stop } catch {}
        if (-not $global:ServerProc.WaitForExit(3000)) { Write-Host "[dev] old server did not exit in 3s" -ForegroundColor Yellow }
        $global:ServerProc = $null
    }
}
function Register-Exit {
    param($proc)
    # Crash detection is done by polling $global:ServerProc.HasExited in the
    # main loop (robust against Exited-event races from the killed old server).
    # Kept as a no-op for call-site compatibility.
}
function Start-New {
    param([string]$ExePath, [string]$ExpectedBuild)
    $log = Join-Path $root 'out\logs\server.log'
    for ($i=1; $i -le $BindRetries; $i++) {
        $global:Stopping = $false
        $global:ServerProc = Start-Process -FilePath $ExePath -WindowStyle Hidden -PassThru -RedirectStandardError $log
        Register-Exit $global:ServerProc
        if (Wait-Version $ExpectedBuild 8) { return $true }
        Write-Host "[dev]   server not ready (try $i/$BindRetries)" -ForegroundColor Yellow
        Stop-Current
        Start-Sleep -Milliseconds 250
    }
    return $false
}
function Invoke-BuildSwap {
    param([string]$Reason)
    Write-Host "[dev] rebuild: $Reason" -ForegroundColor Cyan
    $state.Building = $true
    # Native stderr (nasm errors) becomes ErrorRecord; under Stop it throws and
    # kills the loop. Run the build child under Continue so failures are caught
    # by exit code instead of exceptions.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $out = & powershell.exe -NoProfile -File $buildPs1 -Config Debug 2>&1 | Out-String
    $ErrorActionPreference = $prevEAP
    $code = $LASTEXITCODE
    $jsonLine = ($out -split "`n" | Where-Object { $_ -match '^\{.*buildId' } | Select-Object -Last 1)
    if ($code -ne 0 -or -not $jsonLine) {
        Write-Host "[dev] BUILD FAILED (exit $code); keeping current server" -ForegroundColor Red
        ($out -split "`n" | Where-Object { $_ -match 'error|Error|FAILED' } | Select-Object -Last 4) | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
        $state.Building = $false
        return
    }
    $info = $jsonLine | ConvertFrom-Json
    Write-Host "[dev] build ok: $($info.buildId)" -ForegroundColor Green
    Stop-Current
    if (-not (Start-New $info.exe $info.buildId)) {
        Write-Host "[dev] new server failed to bind; restoring LKG" -ForegroundColor Red
        if ($global:LkgExe) { Start-New $global:LkgExe $global:LkgBuild | Out-Null }
        $state.Building = $false
        return
    }
    if (-not $NoTest) {
        & powershell.exe -NoProfile -File $testPs1 -ExpectedBuild $info.buildId 2>&1 | ForEach-Object { Write-Host "    $_" }
    }
    if ($NoTest -or $LASTEXITCODE -eq 0) {
        $global:LkgExe = $info.exe; $global:LkgBuild = $info.buildId
        Write-Host "[dev] last-known-good: $($info.buildId)" -ForegroundColor Green
    } else {
        Write-Host "[dev] tests failed; LKG unchanged ($global:LkgBuild)" -ForegroundColor Yellow
    }
    $state.Building = $false
}

# ---- initial build + start ----
Write-Host "[dev] asm-chat dev loop (watch: $($Watch -join ', '))  Ctrl-C to exit" -ForegroundColor Cyan
Invoke-BuildSwap "initial"
if (-not $global:LkgExe) { Write-Host "[dev] initial build failed; aborting" -ForegroundColor Red; exit 1 }
Write-Host "[dev] watching... open http://127.0.0.1:8080/  (browser polls /version and reloads on change)" -ForegroundColor Cyan

try {
    while ($true) {
        Start-Sleep -Milliseconds 150
        # crash detection: poll the CURRENT server (not an Exited event, which
        # can race with the killed old server during a swap).
        if ($global:ServerProc -and -not $global:Stopping -and $global:ServerProc.HasExited) {
            $now = [DateTime]::UtcNow
            $global:CrashLog += $now
            $global:CrashLog = @($global:CrashLog | Where-Object { ($now - $_).TotalSeconds -lt $CrashWindowSec })
            if ($global:CrashLog.Count -ge $CrashMax) {
                Write-Host "[dev] crash-loop suppressed ($($global:CrashLog.Count) in ${CrashWindowSec}s). Stopping." -ForegroundColor Red
                break
            }
            $code = $global:ServerProc.ExitCode
            Write-Host "[dev] unexpected server exit (code $code); restarting LKG ($global:LkgBuild)" -ForegroundColor Yellow
            $global:ServerProc = $null
            if ($global:LkgExe) { Start-New $global:LkgExe $global:LkgBuild | Out-Null }
            continue
        }
        if ($state.Pending -and -not $state.Building) {
            $elapsed = ([DateTime]::UtcNow - $state.LastChange).TotalMilliseconds
            if ($elapsed -ge $DebounceMs) {
                $state.Pending = $false
                Invoke-BuildSwap "file change"
            }
        }
    }
} finally {
    Stop-Current
    'devChanged','devCreated','devRenamed' | ForEach-Object {
        Get-EventSubscriber -SourceIdentifier $_ -ErrorAction SilentlyContinue | Unregister-Event
    }
    Write-Host "[dev] stopped" -ForegroundColor Cyan
}
