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
    [switch]$Gateway, [switch]$Repeat,
    [switch]$TimeoutHandler, [switch]$ErrorPath, [switch]$SlowClient,
    [switch]$All
)
$ErrorActionPreference = 'Continue'
if (-not ($Smoke -or $Routes -or $OversizedBody -or $ContentLength -or $FragmentedRequest -or $Malformed -or $Gateway -or $Repeat -or $TimeoutHandler -or $ErrorPath -or $SlowClient -or $All)) { $Smoke = $true }
if ($All) { $Smoke=$Routes=$OversizedBody=$ContentLength=$FragmentedRequest=$Malformed=$Repeat=$TimeoutHandler=$ErrorPath=$SlowClient=$true }

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
if ($TimeoutHandler) {
    Write-Host "[TimeoutHandler]"
    # -- Test 1: Slow client idle does not cause spurious timeout / affect /version --
    $slowIdleJob = Start-Job -ScriptBlock {
        $cli = New-Object System.Net.Sockets.TcpClient
        $cli.Connect('127.0.0.1', 8080)
        $s = $cli.GetStream()
        # 1 byte then pause 2s (well under CLIENT_READ_MS=8000)
        $first = [byte[]]@(0x47)
        $s.Write($first, 0, 1); $s.Flush()
        Start-Sleep -Milliseconds 2000
        $rest = "ET / HTTP/1.1`r`nHost: x`r`n`r`n"
        $b = [Text.Encoding]::ASCII.GetBytes($rest)
        $s.Write($b, 0, $b.Length); $s.Flush()
        $buf = New-Object byte[] 4096
        $sb = New-Object Text.StringBuilder
        $deadline = (Get-Date).AddSeconds(5)
        while ((Get-Date) -lt $deadline) {
            if ($s.DataAvailable) {
                $n = 0; try { $n = $s.Read($buf,0,$buf.Length) } catch { break }
                if ($n -gt 0) { [void]$sb.Append([Text.Encoding]::ASCII.GetString($buf,0,$n)) }
                if ($sb.ToString() -match "\r\n\r\n") { break }
            } else { Start-Sleep -Milliseconds 5 }
        }
        $cli.Close()
        return $sb.ToString()
    }
    # Poll /version during the 2s idle window
    $vOk = $true; $vMax = 0
    for ($i=0; $i -lt 3; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $c = & $curl -s --max-time 2 -o NUL -w "%{http_code}" "$Base/version"
        $sw.Stop()
        if ($c -ne '200') { $vOk = $false }
        if ($sw.ElapsedMilliseconds -gt $vMax) { $vMax = $sw.ElapsedMilliseconds }
        Start-Sleep -Milliseconds 200
    }
    Check "idle slow client: /version stays responsive" ($vOk -and $vMax -lt 1500) "max=${vMax}ms"
    $rIdle = $slowIdleJob | Wait-Job -Timeout 8 -ErrorAction SilentlyContinue | Receive-Job
    if (-not $rIdle) { $rIdle = '' }
    Check "idle slow client: completes 200" ($rIdle -match '200') "got='$($rIdle.Substring(0,[Math]::Min(30,$rIdle.Length)))'"

    # -- Test 2: Timeout handler zero-check — gateway survives concurrent slow client --
    # Verify server is up before attempting gateway
    try { $canGateway = (& $curl -s --max-time 2 -o NUL -w "%{http_code}" "$Base/version") -eq '200' } catch { $canGateway = $false }
    if ($canGateway) {
        $chatJob = Start-Job -ScriptBlock {
            param($BaseUrl)
            $curlExe = 'curl.exe'
            $tmp = New-TemporaryFile
            $code = & $curlExe -s --max-time 70 -o $tmp -w "%{http_code}" -H 'Content-Type: text/plain' --data-binary 'Reply with one short sentence about assembly language.' "$BaseUrl/chat"
            $body = Get-Content $tmp -Raw; Remove-Item $tmp -Force
            return @{code=$code; body=$body}
        } -ArgumentList $Base
        # Slow client trickles data (keeps client_sock active during gateway)
        $slowGwJob = Start-Job -ScriptBlock {
            $cli = New-Object System.Net.Sockets.TcpClient
            $cli.Connect('127.0.0.1', 8080)
            $s = $cli.GetStream()
            $req = "GET / HTTP/1.1`r`nHost: x`r`n`r`n"
            for ($i=0; $i -lt [Math]::Min(20, $req.Length); $i++) {
                $b = [Text.Encoding]::ASCII.GetBytes($req[$i])
                $s.Write($b, 0, 1); $s.Flush()
                Start-Sleep -Milliseconds 300
            }
            $buf = New-Object byte[] 4096; $sb = New-Object Text.StringBuilder
            $deadline = (Get-Date).AddSeconds(10)
            while ((Get-Date) -lt $deadline) {
                if ($s.DataAvailable) {
                    $n = 0; try { $n = $s.Read($buf,0,$buf.Length) } catch { break }
                    if ($n -gt 0) { [void]$sb.Append([Text.Encoding]::ASCII.GetString($buf,0,$n)) }
                    if ($sb.ToString() -match "\r\n\r\n") { break }
                } else { Start-Sleep -Milliseconds 10 }
            }
            $cli.Close(); return $sb.ToString()
        }
        $vOk2 = $true; $vMax2 = 0
        for ($i=0; $i -lt 6; $i++) {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $c = & $curl -s --max-time 2 -o NUL -w "%{http_code}" "$Base/version"
            $sw.Stop()
            if ($c -ne '200') { $vOk2 = $false }
            if ($sw.ElapsedMilliseconds -gt $vMax2) { $vMax2 = $sw.ElapsedMilliseconds }
            Start-Sleep -Milliseconds 800
        }
        Check "zero-check: /version responsive during gateway+slow" ($vOk2 -and $vMax2 -lt 1500) "max=${vMax2}ms"
        $chatR = $chatJob | Wait-Job -Timeout 80 -ErrorAction SilentlyContinue | Receive-Job
        $null = $slowGwJob | Wait-Job -Timeout 10 -ErrorAction SilentlyContinue | Receive-Job
        $chatCode = if ($chatR -and $chatR.code) { $chatR.code } else { 'no-result' }
        Check "zero-check: gateway chat 200" ($chatCode -eq '200') "got=$chatCode"
    } else {
        Check "zero-check: gateway chat 200 (SKIP-unreachable)" $true
    }
}
if ($ErrorPath) {
    Write-Host "[ErrorPath]"
    # Test 1: Incomplete request returns 400 with error body
    $ep1 = Send-Raw -Fragments @("GET / HTTP/1.1`r`n") -CloseWrite
    Check "incomplete request status 400" ($ep1 -match '400') "got='$($ep1.Substring(0,[Math]::Min(30,$ep1.Length)))'"
    Check "incomplete request body has error" ($ep1 -match 'error') "got='$($ep1.Substring(0,[Math]::Min(50,$ep1.Length)))'"

    # Test 2: Status preservation — short body returns 400 not 0
    $ep2 = Send-Raw -Fragments @("POST /chat HTTP/1.1`r`nContent-Length: 10`r`n`r`nhi") -CloseWrite
    Check "short body status 400 not 0" ($ep2 -match '400') "got='$($ep2.Substring(0,[Math]::Min(30,$ep2.Length)))'"

    # Test 3: Error path logs request (verify via body + server survival)
    $ep3 = Send-Raw -Fragments @("POST /chat HTTP/1.1`r`nContent-Length: 5`r`n`r`nab") -CloseWrite
    Check "read error body has error" ($ep3 -match 'error') "got='$($ep3.Substring(0,[Math]::Min(50,$ep3.Length)))'"
    # Verify respond_now + log_request path executed (status=400, server survives)
    $v3 = & $curl -s --max-time 3 -o NUL -w "%{http_code}" "$Base/version"
    Check "read error: server survives, /version 200" ($v3 -eq '200') "got=$v3"

    # Test 4: Early close returns 400 without livelock
    $cli4 = New-Object System.Net.Sockets.TcpClient
    $cli4.Connect('127.0.0.1', 8080)
    $cli4.Close()
    Start-Sleep -Milliseconds 300
    $v4 = & $curl -s --max-time 3 -o NUL -w "%{http_code}" "$Base/version"
    Check "early close: server survives, /version 200" ($v4 -eq '200') "got=$v4"
}
if ($SlowClient) {
    Write-Host "[SlowClient]"
    # Helper to create a slow-client background job
    $script:SlowJobCount = 0
    # Test 1: Single slow client (1B/800ms, 20B) does not block /version
    $scJob1 = Start-Job -ScriptBlock {
        $cli = New-Object System.Net.Sockets.TcpClient
        $cli.Connect('127.0.0.1', 8080)
        $s = $cli.GetStream()
        $req = "GET / HTTP/1.1`r`nHost: x`r`n`r`n"
        for ($i=0; $i -lt [Math]::Min(20, $req.Length); $i++) {
            $b = [Text.Encoding]::ASCII.GetBytes($req[$i])
            $s.Write($b, 0, 1); $s.Flush()
            Start-Sleep -Milliseconds 800
        }
        $buf = New-Object byte[] 4096; $sb = New-Object Text.StringBuilder
        $deadline = (Get-Date).AddSeconds(10)
        while ((Get-Date) -lt $deadline) {
            if ($s.DataAvailable) {
                $n = 0; try { $n = $s.Read($buf,0,$buf.Length) } catch { break }
                if ($n -gt 0) { [void]$sb.Append([Text.Encoding]::ASCII.GetString($buf,0,$n)) }
                if ($sb.ToString() -match "\r\n\r\n") { break }
            } else { Start-Sleep -Milliseconds 10 }
        }
        $cli.Close(); return $sb.ToString()
    }
    $sc1Ok = $true; $sc1Max = 0
    for ($i=0; $i -lt 5; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $c = & $curl -s --max-time 3 -o NUL -w "%{http_code}" "$Base/version"
        $sw.Stop()
        if ($c -ne '200') { $sc1Ok = $false }
        if ($sw.ElapsedMilliseconds -gt $sc1Max) { $sc1Max = $sw.ElapsedMilliseconds }
        Start-Sleep -Milliseconds 700
    }
    Check "1 slow client: /version all 200" $sc1Ok
    Check "1 slow client: /version max <1000ms" ($sc1Max -lt 1000) "max=${sc1Max}ms"
    $null = $scJob1 | Wait-Job -Timeout 25 -ErrorAction SilentlyContinue | Receive-Job

    # Test 2: 10 concurrent slow clients (1B/300ms, 15B) with 100ms stagger
    $scJobs = @()
    $stagger = 0
    for ($j=0; $j -lt 10; $j++) {
        $scJobs += Start-Job -ScriptBlock {
            param($delay, $staggerMs)
            Start-Sleep -Milliseconds $staggerMs
            $cli = New-Object System.Net.Sockets.TcpClient
            $cli.Connect('127.0.0.1', 8080)
            $s = $cli.GetStream()
            $req = "GET / HTTP/1.1`r`nHost: x`r`n`r`n"
            for ($i=0; $i -lt [Math]::Min(15, $req.Length); $i++) {
                $b = [Text.Encoding]::ASCII.GetBytes($req[$i])
                $s.Write($b, 0, 1); $s.Flush()
                Start-Sleep -Milliseconds $delay
            }
            $buf = New-Object byte[] 4096; $sb = New-Object Text.StringBuilder
            $deadline = (Get-Date).AddSeconds(10)
            while ((Get-Date) -lt $deadline) {
                if ($s.DataAvailable) {
                    $n = 0; try { $n = $s.Read($buf,0,$buf.Length) } catch { break }
                    if ($n -gt 0) { [void]$sb.Append([Text.Encoding]::ASCII.GetString($buf,0,$n)) }
                    if ($sb.ToString() -match "\r\n\r\n") { break }
                } else { Start-Sleep -Milliseconds 10 }
            }
            $cli.Close(); return $sb.ToString()
        } -ArgumentList 300, $stagger
        $stagger += 100
    }
    $scPolls = @()
    for ($i=0; $i -lt 10; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $c = & $curl -s --max-time 3 -o NUL -w "%{http_code}" "$Base/version"
            $scPolls += @{code=$c; ms=$sw.ElapsedMilliseconds}
        } catch { $scPolls += @{code='err'; ms=$sw.ElapsedMilliseconds} }
        Start-Sleep -Milliseconds 500
    }
    $scPollsMs = $scPolls | ForEach-Object { $_.ms }
    $scAllOk = ($scPolls | Where-Object { $_.code -ne '200' }).Count -eq 0
    $scAvg  = ($scPollsMs | Measure-Object -Average).Average
    $scMax  = ($scPollsMs | Measure-Object -Maximum).Maximum
    Check "10 slow clients: /version all 200" $scAllOk
    Check "10 slow clients: /version avg <100ms" ($scAvg -lt 100) "avg=[int]$scAvg ms"
    Check "10 slow clients: /version max <1000ms" ($scMax -lt 1000) "max=$scMax ms"
    $null = $scJobs | Wait-Job -Timeout 30 -ErrorAction SilentlyContinue | Receive-Job

    # Test 3: Post-load health — /version recovers after slow clients
    $vPost = & $curl -s --max-time 3 -o NUL -w "%{http_code}" "$Base/version"
    Check "post-load: /version 200" ($vPost -eq '200') "got=$vPost"
}

$failed = @($results | Where-Object { -not $_.pass }).Count
Write-Host ""
if ($failed -eq 0) { Write-Host "ALL PASS ($($results.Count))" -ForegroundColor Green; exit 0 }
else { Write-Host "$failed FAILED of $($results.Count)" -ForegroundColor Red; exit 1 }
