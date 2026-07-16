#Requires -Version 5.1
<# stop.ps1 - stop any asm-chat relay (8081) and server (8080). #>
$killed = 0
foreach ($port in 8080, 8081) {
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction Stop
            $killed++
            Write-Host "stopped pid $($_.OwningProcess) on port $port"
        } catch {}
    }
}
if ($killed -eq 0) { Write-Host "no asm-chat processes found on 8080/8081" }
