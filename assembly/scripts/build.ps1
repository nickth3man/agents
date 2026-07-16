#Requires -Version 5.1
<#
    build.ps1 - assemble + link the asm-chat agent (NASM + GoLink).
    See ../PLAN.md sections 5.1 and the milestone road map.

    Usage:
      powershell.exe -NoProfile -File .\scripts\build.ps1 -Preflight
      powershell.exe -NoProfile -File .\scripts\build.ps1
      powershell.exe -NoProfile -File .\scripts\build.ps1 -Config Debug -BuildId 20260715T120000.000Z-0001

    Tool locations: set env vars NASM_PATH / GOLINK_PATH, or pass -NasmPath /
    -GoLinkPath, or put them on PATH. dumpbin (optional, for import audit) is
    auto-detected from any installed Visual Studio Build Tools.
#>
[CmdletBinding()]
param(
    [switch]$Preflight,
    [string]$NasmPath   = $env:NASM_PATH,
    [string]$GoLinkPath = $env:GOLINK_PATH,
    [string]$LinkPath   = $env:LINK_PATH,
    [string]$BuildId,
    [ValidateSet('Debug','Release')] [string]$Config = 'Release',
    [ValidateSet('golink','msvc')] [string]$Linker = 'golink',
    # Modules to link (grown per milestone). Assembled set is always src/*.asm.
    [string[]]$LinkModules = @('state','decimal','text','log','http_read','http_parse','http_write','router','assets','engine_gateway','net_init','net_io','start'),
    # DLLs to hand to GoLink for import resolution (grown per milestone).
    [string[]]$LinkDlls    = @('kernel32.dll','ws2_32.dll','winhttp.dll'),
    [string]$OutName = 'chat-agent',     # M1 was abi-hello; server artifact is chat-agent
    [string]$Entry   = 'start'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

# ---------------------------------------------------------------------------
# Tool resolution
# ---------------------------------------------------------------------------
function Find-Tool {
    param([string]$Name, [string]$Hint)
    if ($Hint -and (Test-Path $Hint)) { return (Resolve-Path $Hint).Path }
    $c = Get-Command $Name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    # GoLink is intentionally vendored so a clean checkout needs only NASM.
    if ($Name -ieq 'GoLink.exe') {
        $bundled = Join-Path $repoRoot 'tools\golink\GoLink.exe'
        if (Test-Path $bundled) { return (Resolve-Path $bundled).Path }
    }
    return $null
}

function Find-Dumpbin {
    $c = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    # Search any installed VS BuildTools / Community for Hostx64/x64 dumpbin.
    $roots = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio",
        "$env:ProgramFiles\Microsoft Visual Studio"
    )
    foreach ($r in $roots) {
        if (-not (Test-Path $r)) { continue }
        $f = Get-ChildItem $r -Recurse -Filter dumpbin.exe -ErrorAction SilentlyContinue |
             Where-Object { $_.FullName -match 'Hostx64\\x64' } |
             Select-Object -First 1 -ExpandProperty FullName
        if ($f) { return $f }
    }
    return $null
}

function Find-WindowsSdkLibPath {
    $roots = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\Lib",
        "$env:ProgramFiles\Windows Kits\10\Lib"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        foreach ($version in (Get-ChildItem $root -Directory | Sort-Object Name -Descending)) {
            $candidate = Join-Path $version.FullName 'um\x64'
            if (Test-Path (Join-Path $candidate 'kernel32.lib')) { return $candidate }
        }
    }
    return $null
}

# ---------------------------------------------------------------------------
# Preflight (PLAN Milestone 0 deliverable + acceptance test)
# ---------------------------------------------------------------------------
function Invoke-Preflight {
    $tools = [ordered]@{ 'nasm'=''; 'curl'='' }
    $tools['nasm']   = Find-Tool 'nasm.exe'   $NasmPath
    $tools['curl']   = Find-Tool 'curl.exe'   $null
    if ($Linker -eq 'golink') { $tools['GoLink'] = Find-Tool 'GoLink.exe' $GoLinkPath }
    else                      { $tools['link']   = Find-Tool 'link.exe'   $LinkPath }

    Write-Host '== build.ps1 preflight =='
    $missing = 0
    foreach ($k in $tools.Keys) {
        if ($tools[$k]) {
            $ver = switch ($k) {
                'nasm'   { (& $tools[$k] -v 2>$null | Select-Object -First 1) }
                'GoLink' { (& $tools[$k] /? 2>$null | Select-Object -First 1) }
                'link'   { (& $tools[$k] /? 2>$null | Select-Object -First 1) }
                'curl'   { (& $tools[$k] --version 2>$null | Select-Object -First 1) }
            }
            if (-not $ver) { $ver = '(present; version query returned nothing)' }
            Write-Host ("  [OK]   {0,-8} {1}" -f $k, $tools[$k])
        } else {
            Write-Host ("  [MISS] {0,-8} not found (set -{0}Path or env var {0}_PATH)" -f $k)
            $missing++
        }
    }
    if ($missing -gt 0) {
        Write-Host ""
        Write-Host "PREFLIGHT FAILED: $missing required tool(s) missing."
        Write-Host "  NASM   : https://www.nasm.us/   (winget install nasm)"
        if ($Linker -eq 'golink') { Write-Host "  GoLink : https://www.godevtool.com/" }
        else { Write-Host "  link   : install Visual Studio Build Tools and run from a Developer PowerShell" }
        exit 1
    }
    Write-Host "PREFLIGHT OK"
    if ($Preflight) { exit 0 }
}

# ---------------------------------------------------------------------------
# Build-ID generation (PLAN §2.9)
# ---------------------------------------------------------------------------
function New-BuildId {
    param([string]$Forced)
    if ($Forced) { return $Forced }
    $now = [DateTime]::UtcNow
    $stamp = $now.ToString("yyyyMMddTHHmmss.fff") + "Z"
    $countFile = Join-Path $repoRoot 'generated\.buildcount'
    $count = 1
    if (Test-Path $countFile) {
        $v = (Get-Content $countFile -Raw).Trim()
        if ([int]::TryParse($v,[ref]$null)) { $count = [int]$v + 1 }
    }
    Set-Content -Path $countFile -Value $count -NoNewline
    $countStr = "{0:D4}" -f $count
    return "$stamp-$countStr"
}

function Write-VersionInc {
    param([string]$BuildId)
    $inc = Join-Path $repoRoot 'generated\version.inc'
    $body = @(
        "; ============================================================================"
        "; generated/version.inc - GENERATED BY build.ps1, DO NOT EDIT (PLAN §2.9)"
        "; build id: $BuildId"
        "; ============================================================================"
        "%ifndef VERSION_INC"
        "%define VERSION_INC"
        ("%define BUILD_ID     `"$BuildId`"")
        "%strlen BUILD_ID_LEN BUILD_ID"
        "%endif"
    ) -join "`r`n"
    Set-Content -Path $inc -Value $body -NoNewline
}

# ---------------------------------------------------------------------------
# chat.html build-time validation (PLAN §2.8)
# ---------------------------------------------------------------------------
function Assert-ChatHtml {
    $html = Join-Path $repoRoot 'web\chat.html'
    if (-not (Test-Path $html)) { throw "web/chat.html missing" }
    $lines = Get-Content $html
    if ($lines.Count -ge 60) { throw ("chat.html is {0} lines (must be < 60)" -f $lines.Count) }
    $text = $lines -join "`n"
    if ($text -notmatch 'asm-chat') { throw "chat.html missing stable marker 'asm-chat'" }
    $bad = [regex]::Matches($text,'(src\s*=|href\s*=|@import|url\(|https?://|cdn|fonts\.)')
    if ($bad.Count -gt 0) { throw "chat.html references external asset(s): $($bad.Count) match(es)" }
    Write-Host "  [OK] web/chat.html ($($lines.Count) lines, marker present, no external assets)"
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
function Invoke-Build {
    $nasm   = Find-Tool 'nasm.exe'   $NasmPath
    $linkTool = if ($Linker -eq 'golink') { Find-Tool 'GoLink.exe' $GoLinkPath } else { Find-Tool 'link.exe' $LinkPath }
    if (-not $nasm -or -not $linkTool) {
        Write-Error "Run -Preflight first: missing tool (nasm=$nasm linker=$linkTool)."; exit 1
    }

    $buildId = New-BuildId $BuildId
    Write-VersionInc $buildId
    Write-Host "build id: $buildId  (config=$Config)"

    Assert-ChatHtml

    $buildDir = Join-Path $repoRoot "out\builds\$buildId"
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    $dbg = [bool]($Config -eq 'Debug')
    $nasmFlags = @('-f','win64','-I',(Join-Path $repoRoot 'include'),
                                 '-I',$repoRoot,
                                 '-I',(Join-Path $repoRoot 'generated'),
                   ("-DDEBUG=" + (& { if ($dbg) {'1'} else {'0'} })))

    # --- assemble every src/*.asm (fail fast on first error) ---------------
    $allObj = @()
    foreach ($src in Get-ChildItem (Join-Path $repoRoot 'src') -Filter '*.asm' | Sort-Object Name) {
        $obj = Join-Path $buildDir ($src.BaseName + '.obj')
        $lst = Join-Path $buildDir ($src.BaseName + '.lst')
        $args = @($nasmFlags) + @('-o',$obj)
        if ($dbg) { $args += @('-l',$lst) }
        $args += ,$src.FullName
        Write-Host "  nasm $($src.Name)"
        & $nasm @args 2>&1 | ForEach-Object { Write-Host "    $_" }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ASSEMBLE FAILED: $($src.Name) (nasm exit $LASTEXITCODE)"
            Write-Host "Keeping previous good executable (if any). Build dir: $buildDir"
            exit 1
        }
        $allObj += $obj
    }

    # --- link only the modules in $LinkModules -----------------------------
    $linkObjs = foreach ($m in $LinkModules) {
        $o = Join-Path $buildDir "$m.obj"
        if (-not (Test-Path $o)) { throw "link module '$m.obj' not produced by assembly" }
        ,$o
    }
    $exe = Join-Path $buildDir "$OutName.exe"
    if ($Linker -eq 'golink') {
        $linkArgs = @('/entry',$Entry,'/console','/nxcompat','/dynamicbase',
                      '/largeaddressaware','/fo',$exe) + $linkObjs + $LinkDlls
        Write-Host "  GoLink -> $OutName.exe"
    } else {
        $libs = $LinkDlls | ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_) + '.lib' }
        $sdkLib = Find-WindowsSdkLibPath
        if (-not $sdkLib) { throw 'Windows SDK x64 import libraries not found' }
        $linkArgs = @('/MACHINE:X64','/SUBSYSTEM:CONSOLE',"/ENTRY:$Entry",'/NODEFAULTLIB',
                      '/DYNAMICBASE','/NXCOMPAT',"/LIBPATH:$sdkLib", "/OUT:$exe") + $linkObjs + $libs
        Write-Host "  link.exe -> $OutName.exe"
    }
    & $linkTool $linkArgs 2>&1 | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $exe)) {
        Write-Host "LINK FAILED ($Linker exit $LASTEXITCODE). Build dir: $buildDir"
        exit 1
    }

    # --- publish stable artifact to out\current ----------------------------
    $curDir = Join-Path $repoRoot 'out\current'
    New-Item -ItemType Directory -Force -Path $curDir | Out-Null
    Copy-Item $exe (Join-Path $curDir "$OutName.exe") -Force
    if ($Config -eq 'Release') {
        $distDir = Join-Path $repoRoot 'dist'
        New-Item -ItemType Directory -Force -Path $distDir | Out-Null
        Copy-Item $exe (Join-Path $distDir "$OutName.exe") -Force
    }
    $buildId | Set-Content (Join-Path $buildDir 'BUILD_ID') -NoNewline

    Write-Host ""
    Write-Host "BUILD OK: $exe"
    Write-Host "         $(Join-Path $curDir "$OutName.exe")"
    # Return build id + paths on stdout (last line is consumed by dev.ps1 later).
    @{ buildId = $buildId; buildDir = $buildDir; exe = $exe; current = (Join-Path $curDir "$OutName.exe") } |
        ConvertTo-Json -Compress
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
Invoke-Preflight
Invoke-Build
exit 0
