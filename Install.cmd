@echo off
setlocal EnableExtensions DisableDelayedExpansion
title redline-tiny setup
rem Just in case: opened from inside the ZIP. Explorer then runs this file alone from a temp folder.
if not exist "%~dp0scripts\install.ps1" (
    echo ERROR: scripts\install.ps1 not found next to Install.cmd.
    echo Extract the complete ZIP to a folder such as C:\redline-tiny, then run Install.cmd from that folder.
    echo.
    pause
    exit /b 1
)
rem Just in case: a broken PATH, or a 32-bit cmd.exe.
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "PS=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" (
    echo ERROR: Windows PowerShell not found: %PS%
    echo.
    pause
    exit /b 1
)
set "REDLINE_LAUNCHER=cmd"
set "REDLINE_DIR=%~dp0"
rem Just in case: a downloaded-file mark on the scripts, or scripts blocked by Group Policy on a managed PC.
rem -Command is not subject to the execution policy, so this runs either way.
"%PS%" -NoProfile -NonInteractive -Command "Get-ChildItem -LiteralPath (Join-Path $env:REDLINE_DIR 'scripts') -Filter *.ps1 | Unblock-File -ErrorAction SilentlyContinue; if (@(Get-ExecutionPolicy -Scope MachinePolicy; Get-ExecutionPolicy -Scope UserPolicy) -match '^(Restricted|AllSigned)$') { exit 3 }"
if errorlevel 3 (
    echo ERROR: PowerShell scripts are blocked by Group Policy on this PC ^(managed device^).
    echo redline-tiny cannot be installed here without an exception from the administrator.
    echo.
    pause
    exit /b 1
)
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
