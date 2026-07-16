#Requires -Version 5.1
<#
    test.ps1 - smoke + protocol tests for asm-chat (PLAN §5.3).
    Always invokes curl.exe (never the PowerShell alias). Uses a TcpClient
    helper only for malformed/fragmented requests curl cannot express.
#>
param(
    [string]$Base = 'http://127.0.0.1:8080',
    [string]$ExpectedBuild,
    [int]$StartupWait = 10,
    [switch]$Smoke, [switch]$Routes, [switch]$OversizedBody,
    [switch]$ContentLength, [switch]$FragmentedRequest, [switch]$Malformed,
    [switch]$Gateway, [switch]$Repeat, [switch]$All
)
$ErrorActionPreference = 'Continue'
if (-not ($Smoke -or $Routes -or $OversizedBody -or $ContentLength -or $FragmentedRequest -or $Malformed -or $Gateway -or $Repeat -or $All)) { $Smoke = $true }
if ($All) { $Smoke=$Routes=$OversizedBody=$ContentLength=$FragmentedRequest=$Malformed=$Repeat=$true }

$curl = 'curl.exe'
$results = New-Object System.Collections.Generic.List[object]
function Check($name, $cond, $detail='') {
    if ($cond) { Write-Host "  [PASS] $name" -ForegroundColor Green; $results.Add(@{name=$name;pass=$true}) }
    else       { Write-Host "  [FAIL] $name  $detail" -ForegroundColor Red; $results.Add(@{name=$name;pass=$false}) }
}
function Wait-Server {
    $deadline = (Get-Date).AddSeconds($StartupWait)
    while ((Get-Date) -lt $deadline) {
        try { $null = & $curl -s --max-time 2 "$Base/version"; if ($LASTEXITCODE -eq 0) { return $true } } catch {}
        Start-Sleep -Milliseconds 200
    }
    return $false
}
# Send a request as explicit byte fragments; return the full response string.
# Reads as data arrives (DataAvailable loop) so responses are captured even
# when the server closes with unread request body (Windows sends RST).
function Send-Raw {
    param([string[]]$Fragments, [int]$ReadMs = 2500, [switch]$CloseWrite)
    $cli = New-Object System.Net.Sockets.TcpClient
    $cli.Connect('127.0.0.1', 8080)
    $cli.SendTimeout = 2000; $cli.ReceiveTimeout = $ReadMs
    $s = $cli.GetStream()
    foreach ($f in $Fragments) {
        $b = [Text.Encoding]::ASCII.GetBytes($f)
        $s.Write($b, 0, $b.Length); $s.Flush()
        Start-Sleep -Milliseconds 8
    }
    if ($CloseWrite) { $cli.Client.Shutdown([Net.Sockets.SocketShutdown]::Send) }
    $buf = New-Object byte[] 16384
    $sb  = New-Object Text.StringBuilder
    $deadline = (Get-Date).AddMilliseconds($ReadMs)
    while ((Get-Date) -lt $deadline) {
        if ($s.DataAvailable) {
            $n = 0; try { $n = $s.Read($buf, 0, $buf.Length) } catch { break }
            if ($n -le 0) { break }
            [void]$sb.Append([Text.Encoding]::ASCII.GetString($buf, 0, $n))
            if ($sb.ToString() -match "\r\n\r\n") { break }
        } else { Start-Sleep -Milliseconds 5 }
    }
    $cli.Close()
    return $sb.ToString()
}

if ($Smoke) {
    Write-Host "[Smoke]"
    Check "server reachable" (Wait-Server)
    $v = & $curl -s --max-time 3 "$Base/version"
    Check "version nonempty" ($v -ne '') "got='$v'"
    if ($ExpectedBuild) { Check "version matches build" ($v -eq $ExpectedBuild) "got='$v' want='$ExpectedBuild'" }
    $root = & $curl -s --max-time 3 "$Base/"
    Check "GET / 200 + asm-chat" ($root -match 'asm-chat')
    $miss = & $curl -s --max-time 3 -o NUL -w "%{http_code}" "$Base/missing"
    Check "GET /missing 404" ($miss -eq '404') "got=$miss"
    $again = & $curl -s --max-time 3 -o NUL -w "%{http_code}" "$Base/version"
    Check "second request 200" ($again -eq '200')
}
if ($Routes) {
    Write-Host "[Routes]"
    $ver = & $curl -s --max-time 3 -o NUL -w "%{http_code}" "$Base/version"
    Check "GET /version 200" ($ver -eq '200') "got=$ver"
    $hbody = & $curl -s --max-time 3 "$Base/health"
    $hct   = & $curl -s --max-time 3 -o NUL -w "%{content_type}" "$Base/health"
    Check "GET /health 200 + status:ok" ($hbody -match '"status":"ok"') "body='$hbody'"
    Check "GET /health content-type json" ($hct -eq 'application/json') "got='$hct'"
    if ($ExpectedBuild) { Check "GET /health has build" ($hbody -match $ExpectedBuild) }
    $put = & $curl -s --max-time 3 -o NUL -w "%{http_code}" -X PUT "$Base/"
    Check "PUT / 405" ($put -eq '405') "got=$put"
    $gc  = & $curl -s --max-time 3 -o NUL -w "%{http_code}" "$Base/chat"
    Check "GET /chat 405" ($gc -eq '405') "got=$gc"
    $raw = Send-Raw @('POST /chat HTTP/1.1', "`r", "`n", "`r", "`n")
    Check "POST /chat no-CL 411" ($raw -match '411') "got='$($raw.Substring(0,[Math]::Min(30,$raw.Length)))'"
}
if ($OversizedBody) {
    Write-Host "[OversizedBody]"
    $body = ('x' * 4096)
    $code = $body | & $curl -s --max-time 5 -o NUL -w "%{http_code}" -H 'Content-Type: text/plain' --data-binary '@-' "$Base/chat"
    Check "4KB body 413" ($code -eq '413') "got=$code"
}
if ($ContentLength) {
    Write-Host "[ContentLength]"
    $bad = Send-Raw @("POST /chat HTTP/1.1`r`nContent-Length: abc`r`n`r`n")
    Check "invalid CL 400" ($bad -match '400')
    $dup = Send-Raw @("POST /chat HTTP/1.1`r`nContent-Length: 2`r`nContent-Length: 2`r`n`r`nhi")
    Check "duplicate CL 400" ($dup -match '400')
    $te  = Send-Raw @("POST /chat HTTP/1.1`r`nContent-Length: 2`r`nTransfer-Encoding: chunked`r`n`r`nhi")
    Check "CL+TE 400" ($te -match '400')
    $overflow = Send-Raw @("POST /chat HTTP/1.1`r`nContent-Length: 18446744073709551616`r`n`r`n")
    Check "decimal-overflow CL 400" ($overflow -match '400')
    $teOnly = Send-Raw @("POST /chat HTTP/1.1`r`nTransfer-Encoding: chunked`r`n`r`n")
    Check "unsupported TE 501" ($teOnly -match '501')
}
if ($FragmentedRequest) {
    Write-Host "[FragmentedRequest]"
    $req = "GET / HTTP/1.1`r`nHost: x`r`n`r`n"
    $frags = @()
    for ($i=0; $i -lt $req.Length; $i+=7) { $frags += $req.Substring($i, [Math]::Min(7, $req.Length-$i)) }
    $resp = Send-Raw -Fragments $frags -ReadMs 2500
    Check "fragmented GET / 200+asm-chat" ($resp -match '200 OK' -and $resp -match 'asm-chat')
}
if ($Malformed) {
    Write-Host "[Malformed]"
    $noLine = Send-Raw -Fragments @('GET / HTTP/1.1') -CloseWrite
    Check "missing request-line terminator 400" ($noLine -match '400')
    $spacing = Send-Raw @("GET  / HTTP/1.1`r`n`r`n")
    Check "invalid method spacing 400" ($spacing -match '400')
    $noPath = Send-Raw @("GET HTTP/1.1`r`n`r`n")
    Check "missing path 400" ($noPath -match '400')
    $shortBody = Send-Raw -Fragments @("POST /chat HTTP/1.1`r`nContent-Length: 4`r`n`r`nxy") -CloseWrite
    Check "premature body close 400" ($shortBody -match '400')
    $largeHeader = "GET / HTTP/1.1`r`nX-Fill: " + ('x' * 6200)
    $overHeader = Send-Raw -Fragments @($largeHeader) -ReadMs 4000
    Check "oversized headers 431" ($overHeader -match '431')
}
if ($Gateway) {
    Write-Host "[Gateway]"
    $tmp = New-TemporaryFile
    $code = & $curl -s --max-time 70 -o $tmp -w "%{http_code}" -H 'Content-Type: text/plain' --data-binary 'Reply with one short sentence about assembly language.' "$Base/chat"
    $reply = Get-Content $tmp -Raw
    Remove-Item $tmp -Force
    Check "gateway chat 200" ($code -eq '200') "got=$code"
    Check "gateway reply nonempty" (-not [string]::IsNullOrWhiteSpace($reply))
    $again = & $curl -s --max-time 3 -o NUL -w "%{http_code}" "$Base/version"
    Check "server survives gateway request" ($again -eq '200') "got=$again"
}
if ($Repeat) {
    Write-Host "[Repeat]"
    $ok = $true; $failCode=''
    for ($i=1; $i -le 100; $i++) {
        $c = & $curl -s --max-time 3 -o NUL -w "%{http_code}" "$Base/version"
        if ($c -ne '200') { $ok=$false; $failCode=$c; break }
    }
    Check "100x /version 200" $ok "failed with $failCode"
}

$failed = @($results | Where-Object { -not $_.pass }).Count
Write-Host ""
if ($failed -eq 0) { Write-Host "ALL PASS ($($results.Count))" -ForegroundColor Green; exit 0 }
else { Write-Host "$failed FAILED of $($results.Count)" -ForegroundColor Red; exit 1 }
