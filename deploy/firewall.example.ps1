# The GPU PC's firewall for the internet server (design.md 6.17): its port 8001 from the proxy PC only. Nothing in this
# repository runs it. Edit $proxy, then run it in an administrator's PowerShell on the GPU PC.
#   Undo: Remove-NetFirewallRule -DisplayName "sclmd 8001 from the proxy PC", "sclmd 8001 from anyone else"
$proxy = "192.168.0.10"   # EDIT: the proxy PC's LAN address (the same as trusted-proxies in serve.toml)
$port = 8001

# 1. An allow rule cannot narrow another allow rule. The first time python.exe listens on 0.0.0.0 Windows asks, and
#    "Allow" makes a rule for python.exe on every port from anywhere. These are the inbound allow rules for a python.exe:
Get-NetFirewallApplicationFilter | Where-Object { $_.Program -like "*python*.exe" } | Get-NetFirewallRule |
    Where-Object { $_.Direction -eq "Inbound" -and $_.Action -eq "Allow" } | Format-Table DisplayName, Enabled, Profile
#    Disable them (Disable-NetFirewallRule -DisplayName "<the name>"); the block rule of step 3 covers port 8001 anyway.

# 2. Port 8001 from the proxy PC.
New-NetFirewallRule -DisplayName "sclmd 8001 from the proxy PC" -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port `
    -RemoteAddress $proxy -Profile Any

# 3. Port 8001 from every other address: a block rule wins over every allow rule (a python.exe one too). Loopback is not
#    filtered by Windows, so this PC still reaches its own server (/metrics, /health, the supervisor).
function Step([string]$ip, [int]$by) {
    $b = ([System.Net.IPAddress]::Parse($ip)).GetAddressBytes()
    [Array]::Reverse($b)
    $n = [BitConverter]::ToUInt32($b, 0) + $by
    $b = [BitConverter]::GetBytes([uint32]$n)
    [Array]::Reverse($b)
    return ([System.Net.IPAddress]::new($b)).ToString()
}
$others = @("0.0.0.0-$(Step $proxy -1)", "$(Step $proxy 1)-255.255.255.255", "::-ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff")
New-NetFirewallRule -DisplayName "sclmd 8001 from anyone else" -Direction Inbound -Action Block -Protocol TCP -LocalPort $port `
    -RemoteAddress $others -Profile Any

# 4. What is there now:
Get-NetFirewallRule -DisplayName "sclmd 8001*" | Format-Table DisplayName, Enabled, Action, Direction
Get-NetFirewallRule -DisplayName "sclmd 8001*" | Get-NetFirewallAddressFilter | Format-Table RemoteAddress
