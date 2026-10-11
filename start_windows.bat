@echo off
REM Double-click this file on Windows to set up (first run) and start the app.
REM Requires Python 3.10-3.13 from https://www.python.org/downloads/ ("Add python.exe to PATH" ticked).
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set PYTHONUTF8=1

where py >nul 2>&1
if %errorlevel%==0 (
    py -3 start.py %*
    goto :end
)
where python >nul 2>&1
if %errorlevel%==0 (
    python start.py %*
    goto :end
)
echo.
echo Python was not found.
echo Install Python 3.12 from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during installation, then run this file again.
echo.

:end
echo.
pause
