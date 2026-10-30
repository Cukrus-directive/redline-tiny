@echo off
setlocal EnableExtensions DisableDelayedExpansion
title redline-tiny
rem Just in case: a broken PATH, or a 32-bit cmd.exe.
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "PS=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%~dp0scripts\start.ps1" (
    echo ERROR: scripts\start.ps1 not found next to this file. Start redline-tiny from its own folder.
    echo.
    pause
    exit /b 1
)
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
if errorlevel 1 pause
