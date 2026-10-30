# Uninstall redline-tiny: removes its shortcuts and the application files that came in the ZIP. Everything setup
# downloaded or generated stays (models\, runtime\, data\, outputs\, logs\, settings.json) and is listed at the end,
# to be deleted by hand. Nothing outside this folder is touched.
#
# Blocks marked "Just in case:" are for machines that aren't a clean, current Windows 11 (see install.ps1).
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$RT = Join-Path $Root "runtime"
$App = "build", "docs", "hf-npu", "kernels", "redline", "scripts", "space", "tools", "Install.cmd", "Start redline-tiny.cmd",
       "Uninstall.cmd", "README.md", "LICENSE", "NOTICE", "requirements.txt", ".gitignore", ".gitattributes", ".github"
$Kept = "models", "runtime", "data", "outputs", "logs", "settings.json"

function Under([string]$path, [string]$dir) {   # $path is $dir or inside it (case-insensitive)
    if (-not $path -or -not $dir) { return $false }
    return ($path.TrimEnd("\") + "\").StartsWith($dir.TrimEnd("\") + "\", [StringComparison]::OrdinalIgnoreCase)
}
function Ask([string]$q) {
    while ($true) {
        $a = Read-Host "$q [y/n]"
        if ($null -eq $a) { return $false }
        $a = "$a".Trim().ToLower()
        if ($a -in "y", "yes") { return $true }
        if ($a -in "n", "no") { return $false }
        Write-Host "Type y or n, then press Enter." -ForegroundColor Yellow
    }
}

Write-Host "redline-tiny uninstall" -ForegroundColor White
Write-Host "Folder: $Root"

# Just in case: the ZIP was unzipped loose into a drive root or a shared folder (Downloads, Desktop, Documents). Names
# like docs\, tools\ or README.md there can be the user's own files, so only the shortcuts are removed.
$shared = @("Desktop", "MyDocuments", "UserProfile", "MyPictures", "MyVideos", "MyMusic") | ForEach-Object { [Environment]::GetFolderPath($_) }
$shared += (Join-Path $env:USERPROFILE "Downloads"), $env:OneDrive, [IO.Path]::GetPathRoot($Root)
$loose = [bool]($shared | Where-Object { $_ -and $_.TrimEnd("\") -eq $Root.TrimEnd("\") })

# Just in case: Install.cmd is still running in another window.
if ([IO.File]::Exists((Join-Path $RT "install.lock"))) {
    try { ([IO.File]::Open((Join-Path $RT "install.lock"), "Open", "ReadWrite", "None")).Close() }
    catch { Write-Host "ERROR: Install.cmd is running. Close it, then run Uninstall.cmd again." -ForegroundColor Red; exit 1 }
}

$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object { Under "$($_.ExecutablePath)" $RT })
if ($procs) {
    Write-Host "Stopping redline-tiny (python PID $(($procs | ForEach-Object { $_.ProcessId }) -join ', '))."
    $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

# Shortcuts: only the ones that start this folder's copy (another redline-tiny folder may own them).
$links = @()
$ws = $null
try { $ws = New-Object -ComObject WScript.Shell } catch {}
foreach ($dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
    if (-not $dir) { continue }
    $l = Join-Path $dir "redline-tiny.lnk"
    if (-not [IO.File]::Exists($l)) { continue }
    $target = ""
    if ($ws) { try { $target = $ws.CreateShortcut($l).TargetPath } catch {} }
    if (Under $target $Root) { $links += $l } else { Write-Host "Kept: $l (starts another folder: $target)" }
}
$files = @()
if (-not $loose) { $files = @($App | ForEach-Object { Join-Path $Root $_ } | Where-Object { Test-Path -LiteralPath $_ }) }

if (-not $links -and -not $files) { Write-Host "Nothing to remove."; exit 0 }
Write-Host "`nTo be removed:"
$links | ForEach-Object { Write-Host "  $_" }
$files | ForEach-Object { Write-Host "  $_$(if ([IO.Directory]::Exists($_)) { '\' })" }
if ($loose) { Write-Host "Application files are not removed: $Root is not a folder of its own." -ForegroundColor Yellow }
if (-not (Ask "Remove these?")) { Write-Host "Cancelled. Nothing was removed."; exit 0 }

$failed = @()
foreach ($l in $links) { try { [IO.File]::Delete($l) } catch { $failed += $l } }
foreach ($f in $files) {
    if (-not (Under $f $Root) -or $f.TrimEnd("\").Length -le $Root.TrimEnd("\").Length) { continue }
    if ([IO.Directory]::Exists($f)) {
        # rmdir removes junctions without entering them. Just in case: antivirus or the search indexer holding a file
        # for a moment makes it report "not empty", so retry.
        for ($i = 0; $i -lt 5 -and [IO.Directory]::Exists($f); $i++) {
            if ($i) { Start-Sleep -Seconds 2 }
            & cmd.exe /d /c "rmdir /s /q `"$f`"" 2>$null
        }
        if ([IO.Directory]::Exists($f)) { $failed += $f }
    } else {
        try { [IO.File]::Delete($f) } catch { $failed += $f }
    }
}
if ($failed) {
    Write-Host "`nNot removed (in use; close windows showing these files and run Uninstall.cmd again):" -ForegroundColor Yellow
    $failed | ForEach-Object { Write-Host "  $_" }
}

$left = @($Kept | ForEach-Object { Join-Path $Root $_ } | Where-Object { Test-Path -LiteralPath $_ })
if ($left) {
    Write-Host "`nKept: downloaded or generated by setup, including your pictures (outputs\). Delete by hand if no longer needed:"
    $left | ForEach-Object { Write-Host "  $_" }
}
Write-Host "`nDone."
