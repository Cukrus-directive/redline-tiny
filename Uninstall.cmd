@echo off
setlocal EnableExtensions DisableDelayedExpansion
title redline-tiny uninstall
rem Just in case: a broken PATH, or a 32-bit cmd.exe.
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "PS=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%~dp0scripts\uninstall.ps1" (
    echo ERROR: scripts\uninstall.ps1 not found next to Uninstall.cmd.
    echo.
    pause
    exit /b 1
)
rem One line on purpose: uninstall deletes this file, and cmd.exe reads a batch file line by line while running it.
rem (goto) ends the batch without going back to the file; exit /b would look for it.
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\uninstall.ps1" & echo. & pause & (goto) 2>nul
