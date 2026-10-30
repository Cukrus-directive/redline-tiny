# Start redline-tiny: runs the local server (models on the NPU) and opens the UI in the browser. Started by
# "Start redline-tiny.cmd" or the desktop shortcut. Closing the window stops redline-tiny.
#
# Blocks marked "Just in case:" are for machines that aren't a clean, current Windows 11 (see install.ps1).
param([int]$Port = 11290)

$ErrorActionPreference = "Continue"   # never "Stop" around python: PowerShell 5.1 can turn its stderr into a fatal error
$Root = Split-Path -Parent $PSScriptRoot
$RT = Join-Path $Root "runtime"
$Py = Join-Path $RT "venv\Scripts\python.exe"
$Url = "http://127.0.0.1:$Port/"
try { $Host.UI.RawUI.WindowTitle = "redline-tiny (keep this window open; close it to stop)" } catch {}

function Quit([string]$m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
function Under([string]$path, [string]$dir) {   # $path is $dir or inside it (case-insensitive)
    if (-not $path -or -not $dir) { return $false }
    return ($path.TrimEnd("\") + "\").StartsWith($dir.TrimEnd("\") + "\", [StringComparison]::OrdinalIgnoreCase)
}
function Status {   # the app's /api/status, or $null. No proxy: a system proxy must not see 127.0.0.1.
    try {
        $req = [Net.WebRequest]::Create("$($Url)api/status")
        $req.Proxy = $null
        $req.Timeout = 3000
        $resp = $req.GetResponse()
        try { return (New-Object IO.StreamReader $resp.GetResponseStream()).ReadToEnd() | ConvertFrom-Json } finally { $resp.Close() }
    } catch { return $null }
}

if (-not [IO.File]::Exists($Py) -or -not [IO.File]::Exists((Join-Path $Root "data\weights\.done-prepare"))) {
    Quit "redline-tiny is not installed, or setup did not finish. Run Install.cmd."
}
# Just in case: Install.cmd is still running (the shortcut exists before its self-test ends).
try { ([IO.File]::Open((Join-Path $RT "install.lock"), "OpenOrCreate", "ReadWrite", "None")).Close() }
catch { Quit "Install.cmd is running. Start redline-tiny after it finishes." }

$st = Status
if ($st -and $st.state) {
    Write-Host "redline-tiny is already running: opening $Url"
    Start-Process $Url
    exit 0
}
# Just in case: a second double-click while the first window is still starting. That window holds this lock and has
# written its port next to it.
try { $StartLock = [IO.File]::Open((Join-Path $RT "start.lock"), "OpenOrCreate", "ReadWrite", "None") }
catch {
    try { $Port = [int][IO.File]::ReadAllText((Join-Path $RT "start.port")).Trim(); $Url = "http://127.0.0.1:$Port/" } catch {}
    Write-Host "redline-tiny is running or starting in another window: opening $Url when it answers."
    for ($i = 0; $i -lt 60 -and -not (Status); $i++) { Start-Sleep -Seconds 1 }
    Start-Process $Url
    exit 0
}
[IO.File]::WriteAllText((Join-Path $RT "start.port"), "$Port")

# Just in case: the folder was moved or copied after setup; the Python environment keeps absolute paths.
$cfg = Join-Path $RT "venv\pyvenv.cfg"
if ([IO.File]::Exists($cfg)) {
    $line = [IO.File]::ReadAllLines($cfg) | Where-Object { $_ -match "^\s*home\s*=" } | Select-Object -First 1
    $pyHome = if ($line) { ($line -replace "^\s*home\s*=\s*", "").Trim() } else { "" }
    if ($pyHome -and -not (Under $pyHome $RT)) { Quit "This folder was moved or copied after setup. Run Install.cmd in $Root to repair it." }
}
# Just in case: a model or NPU program folder that setup was pointed at is gone (drive unplugged, folder moved).
$sf = Join-Path $Root "settings.json"
if ([IO.File]::Exists($sf)) {
    try { $s = [IO.File]::ReadAllText($sf) | ConvertFrom-Json } catch { Quit "settings.json is damaged. Run Install.cmd to recreate it." }
    foreach ($k in @(@("model", "Model"), @("npu", "NPU program"))) {
        $d = $s.($k[0])
        if ($d -and -not [IO.Directory]::Exists($d)) {
            Quit "$($k[1]) folder not found: $d (drive disconnected, or folder moved). Reconnect it, or run Install.cmd to choose another."
        }
    }
}
# Just in case: another program already listens on the port.
$owner = $null
try { $owner = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1 -ExpandProperty OwningProcess } catch {}
if ($owner) {
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$owner" -ErrorAction SilentlyContinue
    if (-not (Under "$($p.ExecutablePath)" $RT)) {
        Quit "Port $Port is in use by $($p.Name) (PID $owner). Close that program, or use another port: `"Start redline-tiny.cmd`" -Port 11291"
    }
}

# A previous run that crashed can leave its helper processes behind, holding RAM and NPU memory. This window holds
# start.lock, so any python running from runtime\ now is such a leftover.
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object { Under "$($_.ExecutablePath)" $RT } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# Just in case: leftovers from other Python or Hugging Face setups change what python does. Cleared for this window.
foreach ($v in "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONEXECUTABLE", "PYTHONSAFEPATH",
               "VIRTUAL_ENV", "CONDA_PREFIX", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE") { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
Get-ChildItem Env: | Where-Object { $_.Name -like "REDLINE_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:PYTHONNOUSERSITE = "1"
$env:XRT_CONTEXT_CACHE_SIZE = "14"    # Strix has 16 hardware contexts; the DiT keeps 2 outside the runtime's cache
$env:PYTHONUTF8 = "1"
$env:REDLINE_LOG = Join-Path $Root "logs\server.log"
[void][IO.Directory]::CreateDirectory((Join-Path $Root "logs"))

# Just in case: QuickEdit mode. A click in this window would freeze the server until a key is pressed.
try {
    Add-Type -Namespace RedlineTiny -Name Con -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern IntPtr GetStdHandle(int which);
[DllImport("kernel32.dll")] public static extern bool GetConsoleMode(IntPtr h, out uint mode);
[DllImport("kernel32.dll")] public static extern bool SetConsoleMode(IntPtr h, uint mode);
'@
    $h = [RedlineTiny.Con]::GetStdHandle(-10)
    $mode = [uint32]0
    if ([RedlineTiny.Con]::GetConsoleMode($h, [ref]$mode)) {
        $new = $mode; if ($new -band 0x40) { $new -= 0x40 }
        [void][RedlineTiny.Con]::SetConsoleMode($h, [uint32]($new -bor 0x80))
    }
} catch {}

Write-Host ""
Write-Host "  redline" -ForegroundColor White -NoNewline; Write-Host "-tiny" -ForegroundColor Red
Write-Host "  Qwen-Image on the Ryzen AI NPU. Browser: $Url"
Write-Host "  Keep this window open while in use. Close it to stop redline-tiny and free the memory."
Write-Host ""
& $Py (Join-Path $Root "redline\server.py") --port $Port --open
$rc = $LASTEXITCODE
if ($rc -ne 0) {
    Write-Host "`nredline-tiny stopped with an error. Log: $env:REDLINE_LOG" -ForegroundColor Red
    exit $(if ($null -eq $rc) { 1 } else { $rc })
}
