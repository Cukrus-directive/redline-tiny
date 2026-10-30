# redline-tiny installer. Start it with Install.cmd. Runs in Windows PowerShell 5.1 and PowerShell 7.
# Re-runnable: completed steps are skipped, interrupted steps are redone, downloads resume. Nothing is downloaded
# without confirmation unless -Yes is given.
#   -ModelPath DIR   existing Qwen-Image-2.1-Turbo folder (skips the 31 GB download)
#   -NpuPath DIR     existing NPU program folder: manifest.json + merged\ (skips the 310 MB download)
#   -Yes             answer yes to every prompt; download whatever is not supplied by path (unattended install)
#   -SkipSelfTest    skip the test image at the end
#   -CheckOnly       system check only; writes and downloads nothing
#
# Blocks marked "Just in case:" are for machines that aren't a clean, current Windows 11: leftovers from other Python
# setups, company proxies and policies, security software, odd places to unzip to. On a normal PC they do nothing.
param([string]$ModelPath, [string]$NpuPath, [switch]$Yes, [switch]$SkipSelfTest, [switch]$CheckOnly)

# Just in case: company-managed PC with PowerShell locked down. Checked first, everything below needs .NET.
if ($ExecutionContext.SessionState.LanguageMode -ne "FullLanguage") {
    Write-Host "ERROR: PowerShell is restricted on this PC ($($ExecutionContext.SessionState.LanguageMode), set by device policy). redline-tiny cannot be installed on this PC." -ForegroundColor Red
    exit 1
}
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Log = $null

# ---------------------------------------------------------------- output, prompts, exit
# Just in case: started some other way than Install.cmd (right-click "Run with PowerShell", another script, a job).
$cmdLine = [Environment]::GetCommandLineArgs() -join " "
$FromCmd = $env:REDLINE_LAUNCHER -eq "cmd"
$OwnWindow = (-not $FromCmd) -and $PSCommandPath -and $cmdLine.Contains([IO.Path]::GetFileName("$PSCommandPath"))
$NonInteractive = (-not [Environment]::UserInteractive) -or ($cmdLine -match "(?i)(^|\s)-noni")

function Quit([int]$code) {   # a window opened just for this script ("Run with PowerShell") closes on exit: hold it
    if ($OwnWindow -and -not $NonInteractive) { try { $null = Read-Host "Press Enter to close" } catch {} }
    exit $code
}
function Log([string]$m) { if ($Log) { try { Add-Content -LiteralPath $Log -Value "$(Get-Date -Format s) $m" -Encoding UTF8 } catch {} } }
function Say([string]$m, [string]$c = "Gray") { Write-Host $m -ForegroundColor $c; Log $m }
function Step([int]$n, [string]$m) { Say "`n[$n/8] $m" "White" }
function Fail([string]$m) {
    Say "`nERROR: $m" "Red"
    if ($Log) { Say "Installation stopped. Fix the cause and run Install.cmd again; completed steps are skipped. Log: $Log" "Yellow" }
    Quit 1
}
function Abort { Say "`nInstallation cancelled. Nothing further was downloaded or changed." "Yellow"; Quit 2 }
# Just in case: anything not handled below. One readable error line instead of a red stack trace.
trap {
    $m = $_.Exception.Message
    $hint = ""
    if ($m -match "(?i)virus|potentially unwanted|application control|blocked by") {
        $hint = " Security software blocked a file: see Windows Security > Protection history."
    } elseif ($m -match "(?i)access to the path .* is denied|access is denied|being used by another process") {
        $hint = " A file is locked or not writable: close redline-tiny and any window showing files in this folder, then run Install.cmd again."
    } elseif ($m -match "(?i)cancell?ed by the user") {
        $hint = " A Windows permission prompt was declined."
    }
    Fail "$m (install.ps1 line $($_.InvocationInfo.ScriptLineNumber)).$hint"
}
function ReadLine([string]$q) {
    if ($NonInteractive) { Fail "Input required ($q) but this session is non-interactive. Run Install.cmd, or pass -Yes with -ModelPath and -NpuPath." }
    $a = Read-Host $q
    if ($null -eq $a) { Fail "No input (input stream closed or prompt cancelled)." }
    Log "$q : $a"
    return "$a".Trim()
}
function Ask([string]$q) {
    if ($Yes) { Say "$q [y/n] y (-Yes)"; return $true }
    for ($i = 0; $i -lt 10; $i++) {
        $a = (ReadLine "$q [y/n]").ToLower()
        if ($a -in "y", "yes") { return $true }
        if ($a -in "n", "no") { return $false }
        Say "Type y or n, then press Enter." "Yellow"
    }
    Abort
}
function Under([string]$path, [string]$dir) {   # $path is $dir or inside it (case-insensitive)
    if (-not $path -or -not $dir) { return $false }
    return ($path.TrimEnd("\") + "\").StartsWith($dir.TrimEnd("\") + "\", [StringComparison]::OrdinalIgnoreCase)
}

# ---------------------------------------------------------------- 0. host and install folder (nothing written)
if ($env:OS -ne "Windows_NT") { Fail "Windows only." }
if ($PSVersionTable.PSVersion -lt [version]"5.1") { Fail "PowerShell 5.1 or newer required (found $($PSVersionTable.PSVersion))." }
# Just in case: Snapdragon laptops and 32-bit PowerShell.
if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64" -or $env:PROCESSOR_ARCHITEW6432 -eq "ARM64") {
    Fail "ARM64 Windows. Requires an x64 PC with an AMD Ryzen AI 300-series processor."
}
if (-not [Environment]::Is64BitProcess) { Fail "32-bit PowerShell. Start the installer with Install.cmd." }
if (-not $PSScriptRoot) { Fail "Not started from a script file. Start the installer with Install.cmd." }
$Root = Split-Path -Parent $PSScriptRoot

$required = "Install.cmd", "requirements.txt", "scripts\release.json", "scripts\start.ps1", "redline\prepare.py",
            "redline\selftest.py", "redline\server.py", "redline\paths.py"
$absent = @($required | Where-Object { -not [IO.File]::Exists((Join-Path $Root $_)) })
if ($absent) { Fail "Installation files missing from ${Root}: $($absent -join ', '). Extract the complete ZIP again." }
$tmpDirs = @($env:TEMP, $env:TMP, [IO.Path]::GetTempPath()) | Where-Object { $_ }
if ($tmpDirs | Where-Object { Under $Root $_ }) {
    Fail "Running from a temporary folder ($Root), usually from inside a ZIP archive. Extract the ZIP to a folder such as C:\redline-tiny and run Install.cmd there."
}
if ($Root.StartsWith("\\")) { Fail "Install folder is on a network path ($Root). Use a folder on a local drive, such as C:\redline-tiny." }
$PathRoot = [IO.Path]::GetPathRoot($Root)
if ($Root.TrimEnd("\") -eq $PathRoot.TrimEnd("\")) {
    Fail "redline-tiny's files are at the root of drive $PathRoot. Extract them into their own folder, such as C:\redline-tiny."
}
$shared = @("Desktop", "MyDocuments", "UserProfile", "MyPictures", "MyVideos", "MyMusic", "ProgramFiles", "ProgramFilesX86", "Windows") |
    ForEach-Object { [Environment]::GetFolderPath($_) }
$shared += (Join-Path $env:USERPROFILE "Downloads"), $env:OneDrive
if ($shared | Where-Object { $_ -and $_.TrimEnd("\") -eq $Root.TrimEnd("\") }) {
    Fail "redline-tiny's files are directly in $Root. Extract them into their own folder, such as C:\redline-tiny."
}
if ($Root -match "[^\x20-\x7E]") { Fail "Install folder path contains non-ASCII characters: $Root. Use a folder such as C:\redline-tiny." }
$Drive = $null
try { $Drive = New-Object IO.DriveInfo $PathRoot } catch {}
if ($Drive -and $Drive.DriveType -eq [IO.DriveType]::Network) { Fail "Install folder is on a network drive ($PathRoot). Use a local drive." }
if ($Drive -and $Drive.IsReady -and $Drive.DriveFormat -match "^FAT") {
    Fail "Drive $PathRoot is formatted $($Drive.DriveFormat), which cannot hold files over 4 GB. Use an NTFS drive."
}
# Just in case: deeply nested folders. Some packages have very long file paths, and Windows stops at 260 characters
# unless long paths are enabled.
$longPaths = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -ErrorAction SilentlyContinue).LongPathsEnabled -eq 1
if (-not $longPaths -and $Root.Length -gt 80) {
    Fail "Install folder path is $($Root.Length) characters; maximum 80 while Windows long paths are disabled. Use a short folder such as C:\redline-tiny."
}
$probe = Join-Path $Root ".write-test"
try { [IO.File]::WriteAllText($probe, "x"); [IO.File]::Delete($probe) }
catch { Fail "Install folder is not writable: $Root. Use a folder such as C:\redline-tiny (not Program Files, not read-only media)." }
Set-Location -LiteralPath $Root

$Rel = $null
try { $Rel = [IO.File]::ReadAllText((Join-Path $Root "scripts\release.json")) | ConvertFrom-Json } catch {}
if (-not $Rel -or -not $Rel.version) { Fail "scripts\release.json is missing or damaged. Extract the complete ZIP again." }
$RT = Join-Path $Root "runtime"
$Venv = Join-Path $RT "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Uv = Join-Path $RT "uv\uv.exe"
$Site = Join-Path $Venv "Lib\site-packages"
$Eudsl = "eudsl-python-extras==0.1.0.20260914.759+4853bb0"
$Stamp = Join-Path $RT "packages.sha256"   # requirements fingerprint of the last successful package install
$DefModel = Join-Path $Root "models\Qwen-Image-2.1-Turbo"
$DefNpu = Join-Path $Root "data\npu"
$Prepared = Join-Path $Root "data\weights\.done-prepare"   # written by prepare.py when every preparation step finished
$Curl = Join-Path $env:SystemRoot "System32\curl.exe"
if (-not [IO.File]::Exists($Curl)) { $Curl = $null }   # Download() falls back to Invoke-WebRequest
$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $CheckOnly) {
    [void][IO.Directory]::CreateDirectory((Join-Path $Root "logs"))
    [void][IO.Directory]::CreateDirectory($RT)
    $Log = Join-Path $Root "logs\install.log"
    try { $LockFile = [IO.File]::Open((Join-Path $RT "install.lock"), "OpenOrCreate", "ReadWrite", "None") }
    catch { Fail "Another installer is already running for this folder. Wait for it to finish, or close its window." }
    [IO.File]::WriteAllText((Join-Path $RT ".redline-tiny"), "redline-tiny install marker. Uninstall.cmd only deletes files of a folder that has it.`r`n")
}
Say "redline-tiny $($Rel.version) installer" "White"
Say "Install folder: $Root"
Log "PowerShell $($PSVersionTable.PSVersion) $($PSVersionTable.PSEdition), host '$($Host.Name)', admin $IsAdmin, command line: $cmdLine"
foreach ($d in "scripts", "redline", "tools") {
    Get-ChildItem -LiteralPath (Join-Path $Root $d) -Recurse -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
}
Get-ChildItem -LiteralPath $Root -File -Filter *.cmd -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue

# Just in case: leftovers from other Python, uv or Hugging Face setups (conda, PYTHONPATH, HF_HUB_OFFLINE, ...)
# change what the child processes do. Cleared for this window only.
$drop = "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONEXECUTABLE", "PYTHONSAFEPATH", "VIRTUAL_ENV",
        "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_ENABLE_HF_TRANSFER", "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN", "HF_HOME", "HF_HUB_CACHE"
$drop += Get-ChildItem Env: | Where-Object { $_.Name -like "UV_*" -or ($_.Name -like "REDLINE_*" -and $_.Name -notin "REDLINE_LAUNCHER", "REDLINE_DIR") } |
    ForEach-Object { $_.Name }
$ignored = @($drop | Where-Object { Test-Path "Env:$_" })
foreach ($v in $ignored) { Remove-Item "Env:$v" }
if ($ignored) { Log "Environment variables ignored: $($ignored -join ', ')" }
if ($ignored -like "REDLINE_*") { Say "WARNING: REDLINE_* environment variables are set ($(@($ignored -like 'REDLINE_*') -join ', ')); setup ignores them." "Yellow" }
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $RT "python"
$env:UV_CACHE_DIR = Join-Path $RT "uv-cache"
$env:UV_LINK_MODE = "copy"
$env:UV_NATIVE_TLS = "1"   # Windows certificate store: works behind TLS-inspecting company proxies
$env:HF_HOME = Join-Path $RT "hf"   # keep the Hugging Face cache on this drive, inside this folder
$env:HF_HUB_DISABLE_TELEMETRY = "1"
if ($env:HF_ENDPOINT) { Say "Hugging Face endpoint (HF_ENDPOINT): $env:HF_ENDPOINT" }
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}
# Just in case: company proxy. curl, uv and hf ignore the Windows proxy setting, so hand it to them.
if (-not $env:HTTPS_PROXY) {
    try {
        $u = [Uri]"https://pypi.org/"
        $p = [Net.WebRequest]::GetSystemWebProxy().GetProxy($u)
        if ($p -and $p.Host -ne $u.Host) {
            $env:HTTPS_PROXY = $p.AbsoluteUri.TrimEnd("/"); $env:HTTP_PROXY = $env:HTTPS_PROXY
            Say "Proxy (Windows setting): $env:HTTPS_PROXY"
        }
    } catch {}
}

# Just in case: the laptop going to sleep mid-download, or a click in this window pausing everything (QuickEdit mode
# freezes a console program until a key is pressed).
try {
    Add-Type -Namespace RedlineTiny -Name Win -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
[DllImport("kernel32.dll")] public static extern IntPtr GetStdHandle(int which);
[DllImport("kernel32.dll")] public static extern bool GetConsoleMode(IntPtr h, out uint mode);
[DllImport("kernel32.dll")] public static extern bool SetConsoleMode(IntPtr h, uint mode);
'@
    [void][RedlineTiny.Win]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    $h = [RedlineTiny.Win]::GetStdHandle(-10)
    $mode = [uint32]0
    if ([RedlineTiny.Win]::GetConsoleMode($h, [ref]$mode)) {
        $new = $mode; if ($new -band 0x40) { $new -= 0x40 }                      # ENABLE_QUICK_EDIT_MODE off
        [void][RedlineTiny.Win]::SetConsoleMode($h, [uint32]($new -bor 0x80))   # ENABLE_EXTENDED_FLAGS
    }
} catch {}

# ---------------------------------------------------------------- helpers
function Run([string]$exe) {   # native program; its stderr output is not a PowerShell error (ISE, VS Code)
    $ErrorActionPreference = "Continue"
    & $exe @args
}
function Native([string]$what) { if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit code $LASTEXITCODE)." } }
function PyRun([string[]]$pyArgs) {   # Python with stdout and stderr to console and log; returns the exit code
    $ErrorActionPreference = "Continue"
    & $Py @pyArgs 2>&1 | ForEach-Object { $l = "$_"; Write-Host $l; Log $l }
    return $LASTEXITCODE
}
function PyOk {   # the environment runs and has the packages the app imports
    if (-not [IO.File]::Exists($Py)) { return $false }
    $ErrorActionPreference = "Continue"
    try { & $Py -c "import pyxrt, aie.iron, numpy, ml_dtypes, onnxruntime, tokenizers" *> $null; return $LASTEXITCODE -eq 0 } catch { return $false }
}
function UvOk {
    if (-not [IO.File]::Exists($Uv)) { return $false }
    $ErrorActionPreference = "Continue"
    try { & $Uv --version *> $null; return $LASTEXITCODE -eq 0 } catch { return $false }
}
function RemoveTree([string]$d) {   # rmdir removes junctions and symlinks without deleting what they point to
    if (-not $d -or -not (Under $d $Root) -or $d.TrimEnd("\").Length -le $Root.TrimEnd("\").Length) {
        Fail "Refusing to delete '$d': not inside $Root."
    }
    $ErrorActionPreference = "Continue"
    # Just in case: antivirus or the search indexer holding a file for a moment makes rmdir report "not empty".
    for ($i = 0; $i -lt 5 -and [IO.Directory]::Exists($d); $i++) {
        if ($i) { Start-Sleep -Seconds 2 }
        & cmd.exe /d /c "rmdir /s /q `"$d`"" 2>$null
    }
    if ([IO.Directory]::Exists($d)) { Fail "Could not delete $d (files in use: close redline-tiny and any window showing this folder, or restart Windows)." }
}
function Download([string]$url, [string]$dest) {
    $part = "$dest.part"
    if ($Curl) {
        Run $Curl -L --fail --retry 5 --retry-delay 3 --connect-timeout 30 --speed-limit 1024 --speed-time 120 -C - -o $part $url
        if ($LASTEXITCODE -ne 0) {
            [IO.File]::Delete($part)   # never resume a bad partial (proxy page, server error) on the next run
            Fail "Download failed (curl exit code $LASTEXITCODE): $url. Check the internet connection, proxy or firewall."
        }
    } else {
        try { Invoke-WebRequest -Uri $url -OutFile $part -UseBasicParsing }
        catch { [IO.File]::Delete($part); Fail "Download failed: $url ($($_.Exception.Message))." }
    }
    Move-Item -LiteralPath $part -Destination $dest -Force
}
function Unzip([string]$zip, [string]$dest, [string]$what) {   # extract beside $dest, then swap in: never half-extracted
    $tmp = "$dest.extract"
    RemoveTree $tmp
    try { Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force }
    catch {
        [IO.File]::Delete($zip); RemoveTree $tmp
        Fail "$what download is damaged (incomplete, or replaced by a proxy or login page). It was deleted; run Install.cmd again to download it again."
    }
    [IO.File]::Delete($zip)
    RemoveTree $dest
    Move-Item -LiteralPath $tmp -Destination $dest
}
# Just in case: no internet, a hotel wifi login page, a firewall. Better to find out before downloading than halfway.
function Reach([string]$url) {   # $null if the host answers; any HTTP status counts
    try {
        $r = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 20
        $final = if ($r.BaseResponse.ResponseUri) { $r.BaseResponse.ResponseUri } else { $r.BaseResponse.RequestMessage.RequestUri }
        $want = ([Uri]$url).Host.Split(".")[-2]
        if ($final -and $final.Host -notmatch [regex]::Escape($want)) { return "redirected to $($final.Host) (captive portal or web filter)" }
        return $null
    } catch {
        if ($_.Exception.Response) { return $null }
        return $_.Exception.Message
    }
}
function RepoCheck([string]$kind, [string]$repo, [string]$rev) {
    $base = if ($env:HF_ENDPOINT) { $env:HF_ENDPOINT.TrimEnd("/") } else { "https://huggingface.co" }
    try { Invoke-WebRequest -Uri "$base/api/$kind/$repo/revision/$rev" -UseBasicParsing -TimeoutSec 20 | Out-Null; return $null }
    catch {
        if ($_.Exception.Response) { return "$repo (revision $rev) is not available: HTTP $([int]$_.Exception.Response.StatusCode)" }
        return $_.Exception.Message
    }
}
function OurProcs {   # python processes running from this folder's runtime (the app, its helpers, a self-test)
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
        Where-Object { Under "$($_.ExecutablePath)" $RT })
}
function CleanPath([string]$p) {   # user input: quotes, %VARS%, ~, relative paths, a file inside the folder
    $p = "$p".Trim().Trim('"', "'").Trim()
    if (-not $p) { return "" }
    $p = [Environment]::ExpandEnvironmentVariables($p)
    if ($p -eq "~" -or $p.StartsWith("~\") -or $p.StartsWith("~/")) { $p = $env:USERPROFILE + $p.Substring(1) }
    if ($p -match "^[A-Za-z]:$") { $p += "\" }
    if (-not [IO.Path]::IsPathRooted($p)) { $p = Join-Path $Root $p }
    try { $p = [IO.Path]::GetFullPath($p) } catch { return $p }
    if ([IO.File]::Exists($p)) { $p = [IO.Path]::GetDirectoryName($p) }
    if ($p.Length -gt 3) { $p = $p.TrimEnd("\") }
    return $p
}
function NotFound([string]$d) {   # just in case: "Run as administrator" hides mapped network drives
    $msg = "Not found: $d"
    $dr = [IO.Path]::GetPathRoot($d)
    if ($IsAdmin -and $dr -and -not [IO.Directory]::Exists($dr)) { $msg += " (mapped network drives are not visible to an installer run as administrator)" }
    return $msg
}
function ModelMissing([string]$d) {   # required files absent from a model folder; empty = complete
    $req = "model_index.json", "processor\tokenizer.json", "processor\processor_config.json",
           "text_encoder\model.safetensors", "vae\config.json", "vae\diffusion_pytorch_model.safetensors"
    $miss = @($req | Where-Object { -not [IO.File]::Exists((Join-Path $d $_)) })
    $idx = Join-Path $d "transformer\diffusion_pytorch_model.safetensors.index.json"
    if (-not [IO.File]::Exists($idx)) { return $miss + "transformer\diffusion_pytorch_model.safetensors.index.json" }
    try { $ix = [IO.File]::ReadAllText($idx) | ConvertFrom-Json } catch { return $miss + "transformer index (unreadable)" }
    $shards = @($ix.weight_map.PSObject.Properties | ForEach-Object { $_.Value } | Sort-Object -Unique)
    $miss += @($shards | Where-Object { -not [IO.File]::Exists((Join-Path $d "transformer\$_")) } | ForEach-Object { "transformer\$_" })
    if (-not $miss -and $ix.metadata.total_size) {   # shards present: their sizes must add up to the tensor total
        $bytes = ($shards | ForEach-Object { (New-Object IO.FileInfo (Join-Path $d "transformer\$_")).Length } | Measure-Object -Sum).Sum
        if ($bytes -lt [int64]$ix.metadata.total_size) { $miss += "transformer shards incomplete ($bytes of $($ix.metadata.total_size) bytes)" }
    }
    return $miss
}
# Folder checks: $null if usable, otherwise the reason
$ModelCheck = {
    param($d)
    if (-not [IO.Directory]::Exists($d)) { return NotFound $d }
    if ([IO.File]::Exists((Join-Path $d "manifest.json"))) { return "$d is an NPU program or packed-weights folder, not the model folder." }
    $miss = ModelMissing $d
    if (-not $miss) { return $null }
    if ([IO.File]::Exists((Join-Path $d "model_index.json"))) { return "Incomplete model folder. Missing: $($miss -join ', ')" }
    $sub = Get-ChildItem -LiteralPath $d -Directory -ErrorAction SilentlyContinue | Where-Object { -not (ModelMissing $_.FullName) } | Select-Object -First 1
    if ($sub) { return "$d is not the model folder. Model found in subfolder: $($sub.FullName)" }
    return "$d is not a Qwen-Image-2.1-Turbo folder (no model_index.json)."
}
function IsNpuDir([string]$d) { [IO.File]::Exists((Join-Path $d "manifest.json")) -and [IO.Directory]::Exists((Join-Path $d "merged")) }
$NpuCheck = {
    param($d)
    if (-not [IO.Directory]::Exists($d)) { return NotFound $d }
    if ($d -match "[^\x20-\x7E]") { return "NPU program folder path contains non-ASCII characters: $d. Move it to a path such as D:\redline-tiny-npu." }
    if (IsNpuDir $d) { return $null }
    if ([IO.File]::Exists((Join-Path $d "model_index.json"))) { return "$d is the model folder, not the NPU program folder." }
    if ([IO.File]::Exists((Join-Path $d "manifest.json"))) { return "$d has manifest.json but no merged\ folder: not an NPU program folder (packed weights?)." }
    $sub = Get-ChildItem -LiteralPath $d -Directory -ErrorAction SilentlyContinue | Where-Object { IsNpuDir $_.FullName } | Select-Object -First 1
    if ($sub) { return "$d is not the NPU program folder. Found in subfolder: $($sub.FullName)" }
    return "$d is not an NPU program folder (no manifest.json)."
}
function AskPath([string]$what, [scriptblock]$check) {   # a valid folder, or $null for "download"
    for ($i = 0; $i -lt 10; $i++) {
        $p = CleanPath (ReadLine "$what path (empty = download)")
        if (-not $p) { return $null }
        $err = & $check $p
        if (-not $err) { return $p }
        Say $err "Yellow"
    }
    Abort
}
function SaveSettings($s) { [IO.File]::WriteAllText($sf, ($s | ConvertTo-Json), (New-Object Text.UTF8Encoding $false)) }

# ---------------------------------------------------------------- 1. system
Step 1 "System check"
$problems = New-Object Collections.Generic.List[string]
function Problem([string]$m) { $problems.Add($m); Say "  FAIL: $m" "Red" }
$driverHelp = @"
NPU driver: $($Rel.min_npu_driver) or newer (XRT 2.21) required.
  Sources: AMD Software: Adrenalin Edition, https://www.amd.com/en/support/download/drivers.html
           or the laptop manufacturer's support page.
  Do not use NPU_RAI_*_WHQL.zip from the Ryzen AI Software page (32.0.203.x branch, older).
  Restart Windows after updating.
"@
$build = [Environment]::OSVersion.Version.Build
Say "Windows: build $build$(if ($IsAdmin) { ', running as administrator' })"
if ($build -lt 22000) { Problem "Windows 11 required." }
Say "Querying hardware..."
try {
    $npuDrv = @(Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceName LIKE '%NPU Compute Accelerator%' OR DeviceName LIKE '%AMD IPU%'")
    $npuDev = @(Get-CimInstance Win32_PnPEntity -Filter "Name LIKE '%NPU Compute Accelerator%' OR Name LIKE '%AMD IPU%'")
    $cpu = "$((Get-CimInstance Win32_Processor | Select-Object -First 1).Name)".Trim()
    $ramUsable = [double](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    $ramInstalled = [double]((Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum)
    $gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { "$($_.Name)" })
} catch { Fail "Hardware query failed (Windows Management Instrumentation): $($_.Exception.Message)" }
$drv = $null; $npuName = ""
foreach ($d in $npuDrv) { try { $v = [version]$d.DriverVersion; if (-not $drv -or $v -gt $drv) { $drv = $v; $npuName = $d.DeviceName } } catch {} }
$driverProblem = $true
if (-not $npuDrv -and -not $npuDev) {
    Problem "No AMD NPU found (Device Manager > Neural processors). Requires a Ryzen AI 300-series (XDNA2) processor, the NPU enabled in the BIOS/UEFI setup, and the AMD NPU driver."
} elseif (-not $drv) {
    Problem "AMD NPU present but its driver is missing or unreadable."
} else {
    Say "NPU:     $npuName, driver $drv"
    if ($drv -lt [version]$Rel.min_npu_driver) {
        $branch = if ("$drv" -match "^32\.0\.20[0-9]\.") { " (older 32.0.203.x branch)" } else { "" }
        Problem "NPU driver $drv$branch is below the minimum $($Rel.min_npu_driver)."
    } else { $driverProblem = $false }
}
$bad = $npuDev | Where-Object { $_.ConfigManagerErrorCode -ne 0 } | Select-Object -First 1
if ($bad) { Problem "NPU device reports an error in Device Manager (code $($bad.ConfigManagerErrorCode)). Reinstall the NPU driver and restart Windows."; $driverProblem = $true }
if ($driverProblem) { Say $driverHelp "Yellow" }
Say "CPU:     $cpu"
if ($cpu -notmatch "Ryzen AI (9|7|5) (HX )?3[0-9]{2}|Ryzen AI Max") { Say "  WARNING: untested processor. Tested: Ryzen AI 9 HX 375, Ryzen AI 7 350." "Yellow" }
$instGb = [math]::Round($ramInstalled / 1GB); $useGb = [math]::Round($ramUsable / 1GB, 1)
Say "RAM:     $(if ($ramInstalled) { "$instGb GB installed, " })$useGb GB usable"
if ($ramInstalled -and $instGb -lt 31) { Problem "32 GB RAM required ($instGb GB installed). The NPU's buffers are pinned in system RAM." }
elseif (-not $ramInstalled -and $useGb -lt $Rel.min_ram_gb) { Problem "32 GB RAM required ($useGb GB usable)." }
elseif ($useGb -lt 26) {
    Say "  WARNING: only $useGb GB usable; the iGPU reserves the rest (BIOS/UEFI 'UMA frame buffer size'). Generation can run out of memory. Set it to Auto or 4 GB or less." "Yellow"
}
Say "GPU:     $($gpus -join ', ')"
if (-not ($gpus -match "Radeon|AMD")) {
    Problem "AMD Radeon graphics driver not installed. Install AMD Software: Adrenalin Edition (the VAE encoder runs on the iGPU)."
}
$vcMissing = @("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll" | Where-Object { -not [IO.File]::Exists((Join-Path $env:SystemRoot "System32\$_")) })
if ($vcMissing) { Say "  Microsoft Visual C++ Redistributable x64: not installed ($($vcMissing -join ', '))" "Yellow" }
$freeGb = if ($Drive -and $Drive.IsReady) { [math]::Floor($Drive.AvailableFreeSpace / 1GB) } else { -1 }
Say "Disk:    $PathRoot $freeGb GB free$(if ($Drive -and $Drive.IsReady) { ", $($Drive.DriveFormat)" })"
# Just in case: USB drives, OneDrive folders, Smart App Control, Controlled folder access. Warnings only; the OneDrive
# one is asked again before anything is written.
if ($Drive -and $Drive.DriveType -eq [IO.DriveType]::Removable) { Say "  WARNING: removable drive. redline-tiny only runs while it is connected." "Yellow" }
$oneDrive = ($Root -match "(?i)\\OneDrive") -or (Under $Root $env:OneDrive)
if ($oneDrive) { Say "  WARNING: install folder is inside OneDrive; about 50 GB of files would be synced." "Yellow" }
$sac = $null
try { $sac = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" -ErrorAction Stop).VerifiedAndReputablePolicyState } catch {}
if ($sac -eq 1) { Say "  WARNING: Smart App Control is on. It can block the Python runtime setup installs; a blocked file is listed in Windows Security > Protection history." "Yellow" }
$cfa = 0
try { $cfa = (Get-MpPreference -ErrorAction Stop).EnableControlledFolderAccess } catch {}
if ($cfa -eq 1 -and (Under $Root $env:USERPROFILE)) {
    Say "  WARNING: Controlled folder access is on and the install folder is in your user profile; Windows Security can block writes. Recommended: C:\redline-tiny." "Yellow"
}

if ($CheckOnly) {
    if ($vcMissing) { Problem "Microsoft Visual C++ Redistributable x64 missing (setup offers to install it)." }
    $need = [int]$Rel.min_free_gb
    if ($freeGb -ge 0 -and $freeGb -lt $need - 31) { Problem "Not enough disk space: $freeGb GB free, $need GB required ($($need - 31) GB with an existing model folder)." }
    elseif ($freeGb -ge 0 -and $freeGb -lt $need) { Say "  NOTE: $freeGb GB free: enough only with an existing model folder ($($need - 31) GB); downloading the model needs $need GB." "Yellow" }
    if ($problems.Count) { Say "`nResult: FAIL ($($problems.Count) problem$(if ($problems.Count -gt 1) { 's' }))" "Red"; Quit 1 }
    Say "`nResult: PASS" "Green"
    Quit 0
}
if ($problems.Count) { Fail ("System check failed:`n  - " + ($problems -join "`n  - ")) }
if ($vcMissing) {   # just in case: a fresh Windows without the Visual C++ runtime
    Say "The Microsoft Visual C++ Redistributable x64 is required (PyTorch, ONNX Runtime)."
    if (-not (Ask "Download it from Microsoft (aka.ms/vs/17/release/vc_redist.x64.exe, ~25 MB) and run its installer?")) {
        Fail "Microsoft Visual C++ Redistributable x64 required: https://aka.ms/vs/17/release/vc_redist.x64.exe"
    }
    $vc = Join-Path $RT "vc_redist.x64.exe"
    Download "https://aka.ms/vs/17/release/vc_redist.x64.exe" $vc
    try { $p = Start-Process -FilePath $vc -ArgumentList "/install", "/passive", "/norestart" -Wait -PassThru }
    catch { Fail "Visual C++ Redistributable installation did not run: $($_.Exception.Message)" }
    [IO.File]::Delete($vc)
    if ($p.ExitCode -notin 0, 1638, 3010) { Fail "Visual C++ Redistributable installer failed (exit code $($p.ExitCode))." }
    Say "Visual C++ Redistributable: installed$(if ($p.ExitCode -eq 3010) { ' (restart Windows after setup)' })"
}
if ($oneDrive -and -not (Ask "Install inside OneDrive anyway?")) { Abort }

# ---------------------------------------------------------------- 2. sources and plan
Step 2 "Sources"
$settings = @{}
$sf = Join-Path $Root "settings.json"
if ([IO.File]::Exists($sf)) {
    try { ([IO.File]::ReadAllText($sf) | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $settings[$_.Name] = [string]$_.Value } }
    catch { Say "WARNING: settings.json is damaged; ignoring it (rewritten below)." "Yellow"; $settings = @{} }
}

Say "Model: Qwen-Image-2.1-Turbo. License: Qwen Research License, non-commercial use only (docs\QWEN-RESEARCH-LICENSE.txt)."
$getModel = $false
$staleModel = $false
if ($settings.ContainsKey("model")) {
    $err = & $ModelCheck $settings["model"]
    if ($err) { Say "WARNING: saved model folder is no longer usable: $err" "Yellow"; $settings.Remove("model"); $staleModel = $true }
}
if ($ModelPath) {
    $p = CleanPath $ModelPath
    $err = & $ModelCheck $p
    if ($err) { Fail "-ModelPath: $err" }
    $settings["model"] = $p
} elseif ($settings.ContainsKey("model")) {
} elseif (-not (ModelMissing $DefModel)) {
} elseif ($Yes) {
    if ($staleModel) { Fail "The saved model folder is no longer usable; -Yes would download 31 GB instead. Pass -ModelPath, or run without -Yes." }
    $getModel = $true
} else {
    Say "Model folder: the Hugging Face download of $($Rel.model_repo)."
    Say "  Contains model_index.json, transformer\, text_encoder\, vae\, processor\. Example: D:\models\Qwen-Image-2.1-Turbo"
    if ([IO.Directory]::Exists($DefModel)) { Say "  Partial download present in $DefModel; an empty answer resumes it." }
    $p = AskPath "Model folder" $ModelCheck
    if ($p) { $settings["model"] = $p }
    else {
        Say "WARNING: downloads 31 GB from huggingface.co/$($Rel.model_repo) into $DefModel." "Yellow"
        if (-not (Ask "Download the model?")) { Abort }
        $getModel = $true
    }
}
$modelDir = if ($settings.ContainsKey("model")) { $settings["model"] } else { $DefModel }
Say "  Model source:        $(if ($getModel) { "download -> $DefModel" } else { $modelDir })"

$getNpu = $false
$staleNpu = $false
if ($settings.ContainsKey("npu")) {
    $err = & $NpuCheck $settings["npu"]
    if ($err) { Say "WARNING: saved NPU program folder is no longer usable: $err" "Yellow"; $settings.Remove("npu"); $staleNpu = $true }
}
if ($NpuPath) {
    $p = CleanPath $NpuPath
    $err = & $NpuCheck $p
    if ($err) { Fail "-NpuPath: $err" }
    $settings["npu"] = $p
} elseif ($settings.ContainsKey("npu")) {
} elseif ($Yes) {
    if ($staleNpu) { Fail "The saved NPU program folder is no longer usable; -Yes would download instead. Pass -NpuPath, or run without -Yes." }
    $getNpu = $true
} else {
    Say "NPU program folder: the compiled NPU programs ($($Rel.npu_repo)). Not the model folder."
    Say "  Contains manifest.json, merged\ and 126 program folders. Example: D:\redline-tiny-npu"
    if ([IO.Directory]::Exists($DefNpu)) { Say "  Previous download present in $DefNpu; an empty answer resumes and verifies it." }
    $p = AskPath "NPU program folder" $NpuCheck
    if ($p) { $settings["npu"] = $p }
    else {
        Say "WARNING: downloads 310 MB from huggingface.co/$($Rel.npu_repo) into $DefNpu." "Yellow"
        if (-not (Ask "Download the NPU programs?")) { Abort }
        $getNpu = $true
    }
}
Say "  NPU program source:  $(if ($getNpu) { "download -> $DefNpu" } else { $settings['npu'] })"

$prepared = [IO.File]::Exists($Prepared)
$needGb = [int]$Rel.min_free_gb
if (-not $getModel) { $needGb -= 31 }
if (-not $prepared -and $freeGb -ge 0 -and $freeGb -lt $needGb) { Fail "Not enough disk space on ${PathRoot}: $freeGb GB free, $needGb GB required." }

# Python environment state. Absolute paths inside the environment break when the folder is moved or copied.
$venvMoved = $false
$cfg = Join-Path $Venv "pyvenv.cfg"
if ([IO.File]::Exists($cfg)) {
    $line = [IO.File]::ReadAllLines($cfg) | Where-Object { $_ -match "^\s*home\s*=" } | Select-Object -First 1
    $pyHome = if ($line) { ($line -replace "^\s*home\s*=\s*", "").Trim() } else { "" }
    if ($pyHome -and -not (Under $pyHome $RT)) { $venvMoved = $true; Say "Python environment was created in another folder ($pyHome): it is rebuilt." }
}
$reqHash = (Get-FileHash -LiteralPath (Join-Path $Root "requirements.txt") -Algorithm SHA256).Hash + " $Eudsl"
$hasPyxrt = [IO.File]::Exists((Join-Path $Site "pyxrt.pyd"))
$stampOk = [IO.File]::Exists($Stamp) -and ([IO.File]::ReadAllText($Stamp).Trim() -eq $reqHash)
$pkgsOk = (-not $venvMoved) -and $stampOk -and $hasPyxrt -and (PyOk)
if ((-not $venvMoved) -and $stampOk -and -not $pkgsOk) { Say "Python environment is damaged: it is rebuilt." }
$rebuild = $venvMoved -or ($stampOk -and -not $pkgsOk)
$uvOk = UvOk

$dl = @()
if (-not $pkgsOk) {
    if (-not $uvOk) { $dl += "  uv (Python installer)              github.com/astral-sh/uv" }
    if (-not [IO.Directory]::Exists((Join-Path $RT "python"))) { $dl += "  Python $($Rel.python)                        python-build-standalone (github.com)" }
    $dl += "  Python packages, up to 1.5 GB      pypi.org, download.pytorch.org, github.com, llvm.github.io"
    if (-not $hasPyxrt -or $rebuild) { $dl += "  XRT SDK (pyxrt)                    github.com/Xilinx/XRT" }
}
if ($getModel) { $dl += "  Qwen-Image-2.1-Turbo, 31 GB        huggingface.co/$($Rel.model_repo)" }
if ($getNpu) { $dl += "  NPU programs, 310 MB               huggingface.co/$($Rel.npu_repo)" }
if ($dl) {
    Say "`nDownloads:"; $dl | ForEach-Object { Say $_ }
    Say "Checking the download sources..."
    $urls = @()
    if (-not $pkgsOk) { $urls += "https://github.com/", "https://pypi.org/simple/", "https://files.pythonhosted.org/", "https://download.pytorch.org/whl/cpu/", "https://llvm.github.io/eudsl/" }
    if (($getModel -or $getNpu) -and -not $env:HF_ENDPOINT) { $urls += "https://huggingface.co/" }
    $netErr = @()
    foreach ($u in $urls) { $e = Reach $u; if ($e) { $netErr += "$u : $e" } }
    if ($getModel) { $e = RepoCheck "models" $Rel.model_repo $Rel.model_revision; if ($e) { $netErr += $e } }
    if ($getNpu) { $e = RepoCheck "models" $Rel.npu_repo $Rel.npu_revision; if ($e) { $netErr += $e } }
    if ($netErr) { Fail ("Download sources not reachable:`n  - " + ($netErr -join "`n  - ") + "`nCheck the internet connection, proxy or firewall. Already downloaded files are kept.") }
    Say "  OK"
} else {
    Say "`nDownloads: none"
}
Say "Writes (inside $Root):"
if (-not $pkgsOk) { Say "  runtime\       Python environment, ~4 GB" }
if ($getModel) { Say "  models\        Qwen-Image-2.1-Turbo, 31 GB" }
if (-not $prepared) { Say "  data\weights\  packed NPU weights, ~17 GB"; Say "  data\onnx\     VAE encoder for the iGPU" }
Say "  logs\          install log"

$procs = OurProcs
if ($procs) {
    Say "redline-tiny is running from this folder (python PID $(($procs | ForEach-Object { $_.ProcessId }) -join ', '))." "Yellow"
    if (-not (Ask "Stop it and continue?")) { Abort }
    $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
if (-not (Ask "Proceed?")) { Abort }
if ($settings.Count -or [IO.File]::Exists($sf)) { SaveSettings $settings }

# ---------------------------------------------------------------- 3. Python environment
Step 3 "Python environment"
if ($pkgsOk) {
    Say "Up to date: $Venv"
} else {
    if ($rebuild) { RemoveTree $Venv; [IO.File]::Delete($Stamp) }
    if (-not $uvOk) {
        $zip = Join-Path $RT "uv.zip"
        Download $Rel.uv_url $zip
        Unzip $zip (Join-Path $RT "uv") "uv"
        if (-not (UvOk)) { Fail "uv.exe does not run after extraction (blocked or removed by security software? See Windows Security > Protection history)." }
    }
    if (-not [IO.File]::Exists($Py)) {
        Run $Uv venv $Venv --python $Rel.python --managed-python
        Native "Python environment creation"
    }
    Run $Uv pip install --python $Py -r (Join-Path $Root "requirements.txt") --index-strategy unsafe-best-match
    Native "Package installation (network, proxy or security software; see the output above)"
    Run $Uv pip install --python $Py --find-links https://llvm.github.io/eudsl $Eudsl -C EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX=aie
    Native "mlir-aie Python extras installation"
    [IO.File]::WriteAllText((Join-Path $Site "aie.pth"), "mlir_aie/python`n")
    if (-not [IO.File]::Exists((Join-Path $Site "pyxrt.pyd"))) {
        $zip = Join-Path $RT "xrt_windows_sdk.zip"
        $tmp = Join-Path $RT "xrt-sdk"
        Download $Rel.xrt_sdk_url $zip
        Unzip $zip $tmp "XRT SDK"
        $pyd = Get-ChildItem -LiteralPath $tmp -Recurse -Filter pyxrt.pyd | Select-Object -First 1
        if (-not $pyd) { Fail "pyxrt.pyd not found in the XRT SDK archive." }
        Copy-Item -LiteralPath $pyd.FullName -Destination $Site
        RemoveTree $tmp
    }
}
$rc = PyRun @("-c", "import pyxrt, aie.iron, numpy, ml_dtypes, onnxruntime, tokenizers; print('Packages: OK')")
if ($rc -ne 0) {
    [IO.File]::Delete($Stamp)
    $xrt = if ($env:XILINX_XRT) { " XILINX_XRT is set to $env:XILINX_XRT: another XRT installation (Ryzen AI Software) can conflict." } else { "" }
    Fail ("Package import check failed (output above). 'DLL load failed' with pyxrt: NPU driver missing or outdated.$xrt " +
          "'DLL load failed' with other packages: install the Microsoft Visual C++ Redistributable x64 (https://aka.ms/vs/17/release/vc_redist.x64.exe).")
}
if (-not $pkgsOk) { [IO.File]::WriteAllText($Stamp, $reqHash) }
$hf = Join-Path $Venv "Scripts\hf.exe"

# ---------------------------------------------------------------- 4. model
Step 4 "Model"
if ($getModel) {
    Run $hf download $Rel.model_repo --revision $Rel.model_revision --local-dir $DefModel --exclude "assets/*"
    Native "Model download from huggingface.co/$($Rel.model_repo) (run Install.cmd again to resume)"
    $miss = ModelMissing $DefModel
    if ($miss) { Fail "Model download incomplete. Missing: $($miss -join ', '). Run Install.cmd again to resume." }
}
Say "Model: $modelDir"

# ---------------------------------------------------------------- 5. NPU programs
Step 5 "NPU programs"
if ($getNpu) {
    Run $hf download $Rel.npu_repo --revision $Rel.npu_revision --local-dir $DefNpu
    Native "NPU program download from huggingface.co/$($Rel.npu_repo) (run Install.cmd again to resume)"
}
Say "NPU programs: $(if ($settings.ContainsKey('npu')) { $settings['npu'] } else { $DefNpu })"

# ---------------------------------------------------------------- 6. one-time preparation
Step 6 "Weight preparation"
if (-not [IO.File]::Exists($Prepared)) {
    $freeNow = if ($Drive -and $Drive.IsReady) { [math]::Floor((New-Object IO.DriveInfo $PathRoot).AvailableFreeSpace / 1GB) } else { -1 }
    if ($freeNow -ge 0 -and $freeNow -lt 20) { Fail "Not enough disk space on ${PathRoot} for the packed weights: $freeNow GB free, 20 GB required." }
}
$rc = PyRun @((Join-Path $Root "redline\prepare.py"))
if ($rc -ne 0) { Fail "Weight preparation failed (exit code $rc; output above)." }

# ---------------------------------------------------------------- 7. shortcuts
Step 7 "Shortcuts"
$launcher = Join-Path $Root "Start redline-tiny.cmd"
if (Ask "Create 'redline-tiny' shortcuts on the desktop and in the Start menu?") {
    $ws = $null
    try { $ws = New-Object -ComObject WScript.Shell } catch { Say "WARNING: shortcuts not created (Windows Script Host unavailable)." "Yellow" }
    $icon = Join-Path $Root "redline\assets\redline.ico"
    foreach ($dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
        if (-not $ws -or -not $dir -or -not [IO.Directory]::Exists($dir)) { continue }
        $path = Join-Path $dir "redline-tiny.lnk"
        try {
            $lnk = $ws.CreateShortcut($path)
            $lnk.TargetPath = $launcher
            $lnk.WorkingDirectory = $Root
            $lnk.Description = "Qwen-Image editing and text-to-image on the Ryzen AI NPU"
            if ([IO.File]::Exists($icon)) { $lnk.IconLocation = $icon }
            $lnk.Save()
            Say "Created: $path"
        } catch { Say "WARNING: shortcut not created in ${dir}: $($_.Exception.Message)" "Yellow" }
    }
} else {
    Say "Skipped."
}

# ---------------------------------------------------------------- 8. self-test
Step 8 "Self-test"
if ($SkipSelfTest) {
    Say "Skipped (-SkipSelfTest)."
} elseif (Ask "Run the self-test (loads all models on the NPU, generates one 1024x1024 image)?") {
    $procs = OurProcs
    if ($procs) {
        Say "redline-tiny is running from this folder; the self-test needs the NPU and the memory it holds." "Yellow"
        if (-not (Ask "Stop it and run the self-test?")) { Abort }
        $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
    $env:XRT_CONTEXT_CACHE_SIZE = "14"
    $rc = PyRun @((Join-Path $Root "redline\selftest.py"))
    OurProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }   # helper processes
    if ($rc -ne 0) {
        Fail "Self-test failed (exit code $rc; output above). Close other programs that use the NPU (camera effects, Windows Studio Effects), restart Windows, then run Install.cmd again."
    }
} else {
    Say "Skipped."
}

Say "`nInstallation complete." "Green"
Say "Start: $launcher$(if ([IO.File]::Exists((Join-Path ([Environment]::GetFolderPath('Desktop')) 'redline-tiny.lnk'))) { ' (desktop shortcut: redline-tiny)' })"
Say "Do not move this folder; if it is moved, run Install.cmd again in the new location."
Quit 0
